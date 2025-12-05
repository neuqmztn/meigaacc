`timescale 1ns / 1ps

module weight_update_engine #(
    //======================================================================
    // 基本参数
    //======================================================================
    parameter NUM_LAYERS        = 5,
    parameter BACKBONE_DIM      = 32,
    parameter SIDENET_DIM       = 8,
    parameter D_FF              = 32,
    parameter DATA_WIDTH        = 16,        // BFP 尾数位宽
    parameter EXP_WIDTH         = 8,         // BFP 指数位宽
    parameter Q412_WIDTH        = 16,        // Q4.12 位宽
    parameter DRAM_DATA_WIDTH   = 256,       // Burst 宽度
    parameter NUM_TOKENS        = 641,       // token 数量（与 gradient_buffer 一致）
    
    //======================================================================
    // 权重矩阵最大尺寸（行×列）
    //======================================================================
    parameter MAX_ROWS          = 32,
    parameter MAX_COLS          = 32,
    
    //======================================================================
    // 梯度累加宽度（列级别）
    //======================================================================
    parameter GRAD_ACCUM_WIDTH  = 32,
    
    //======================================================================
    // 默认学习率 Q4.12（0.001）
    //======================================================================
    parameter [Q412_WIDTH-1:0]  DEFAULT_LR_Q412 = 16'h0001,
    
    //======================================================================
    // 超时阈值
    //======================================================================
    parameter TIMEOUT_THRESHOLD = 12'd4095
)(
    //======================================================================
    // 时钟和复位
    //======================================================================
    input  wire clk,
    input  wire rst_n,
    
    //======================================================================
    // 控制接口
    //======================================================================
    input  wire        update_start,         // 启动更新（脉冲）
    output reg         update_done,          // 更新完成
    output reg         update_busy,          // 更新进行中
    
    //======================================================================
    // 配置接口
    //======================================================================
    input  wire [2:0]  cfg_layer_id,         // 要更新的层 ID (0-4)
    input  wire [3:0]  cfg_weight_type,      // 权重类型
    input  wire [15:0] cfg_learning_rate,    // 学习率(Q4.12 格式)
    
    //======================================================================
    // 旧权重读取接口（从权重存储读取，BFP 格式）
    //======================================================================
    output reg         weight_rd_req,        // 读请求
    output reg  [2:0]  weight_rd_layer_id,   // 层 ID
    output reg  [3:0]  weight_rd_type,       // 权重类型
    output reg  [4:0]  weight_rd_col_id,     // 列 ID
    output reg  [4:0]  weight_rd_row_id,     // 行 ID
    input  wire        weight_rd_valid,      // 读数据有效
    input  wire [EXP_WIDTH-1:0]  weight_rd_exp,   // BFP 指数
    input  wire [DATA_WIDTH-1:0] weight_rd_mant,  // BFP 尾数
    
    //======================================================================
    // 梯度读取接口（从 Gradient Buffer 读取，Q4.12 格式）
    //======================================================================
    output reg         grad_rd_req,          // 读请求
    output reg  [2:0]  grad_rd_layer_id,     // 层 ID
    output reg  [9:0]  grad_rd_token_id,     // Token ID
    output reg  [4:0]  grad_rd_dim_id,       // 维度 ID
    input  wire        grad_rd_valid,        // 读数据有效
    input  wire [Q412_WIDTH-1:0] grad_rd_data,    // 梯度数据(Q4.12)
    
    //======================================================================
    // 新权重写入接口（写入权重存储，BFP 格式）
    //======================================================================
    output reg         weight_wr_req,        // 写请求
    output reg  [2:0]  weight_wr_layer_id,   // 层 ID
    output reg  [3:0]  weight_wr_type,       // 权重类型
    output reg  [5:0]  weight_wr_burst_idx,  // Burst 索引
    output reg  [EXP_WIDTH*MAX_COLS-1:0] weight_wr_exp_array,  // 指数数组
    output reg  [DRAM_DATA_WIDTH-1:0]     weight_wr_data_burst,// 数据 burst
    input  wire        weight_wr_ready,      // 写就绪
    
    //======================================================================
    // 调试接口
    //======================================================================
    output wire [3:0]  dbg_state,            // 当前状态
    output reg  [31:0] dbg_update_count,     // 更新计数
    output reg  [31:0] dbg_col_count,        // 列计数
    output reg  [31:0] dbg_row_count,        // 行计数
    output reg  [31:0] dbg_timeout_count,    // 超时计数
    output reg  [31:0] dbg_overflow_count    // 溢出计数
);

//======================================================================
// 层 ID 和权重类型编码（需要与权重存储保持一致）
//======================================================================
localparam WEIGHT_COMPRESS = 4'd0;  // 32×8
localparam WEIGHT_EXPAND   = 4'd1;  // 8×32
localparam WEIGHT_FFN_W1   = 4'd2;  // 8×32
localparam WEIGHT_FFN_W2   = 4'd3;  // 32×8
localparam WEIGHT_SIDE_ATT_Q = 4'd4;  // SIDENET_DIM × SIDENET_DIM
localparam WEIGHT_SIDE_ATT_K = 4'd5;  // SIDENET_DIM × SIDENET_DIM
localparam WEIGHT_SIDE_ATT_V = 4'd6;  // SIDENET_DIM × SIDENET_DIM
//======================================================================
// 状态机定义
//======================================================================
localparam STATE_IDLE           = 4'd0;
localparam STATE_CONFIG         = 4'd1;
localparam STATE_COL_START      = 4'd2;
localparam STATE_TK_READ_GRAD   = 4'd3;   // 按 token 读梯度
localparam STATE_TK_WAIT_GRAD   = 4'd4;   // 等梯度并累加
localparam STATE_ROW_START      = 4'd5;
localparam STATE_READ_WEIGHT    = 4'd6;
localparam STATE_WAIT_WEIGHT    = 4'd7;
localparam STATE_CONV_WEIGHT    = 4'd8;
localparam STATE_COMPUTE_UPDATE = 4'd9;
localparam STATE_STORE_TEMP     = 4'd10;
localparam STATE_ROW_NEXT       = 4'd11;
localparam STATE_COL_NORMALIZE  = 4'd12;
localparam STATE_CONV_TO_BFP    = 4'd13;
localparam STATE_WRITE_BACK     = 4'd14;
localparam STATE_COL_NEXT       = 4'd15;

reg [3:0] state_reg, state_next;
assign dbg_state = state_reg;

//======================================================================
// 寄存器定义
//======================================================================
reg [2:0]  layer_id_reg;       // 当前层 ID
reg [3:0]  weight_type_reg;    // 当前权重类型
reg [15:0] learning_rate_reg;  // 当前学习率(Q4.12)

reg [5:0] total_rows;
reg [5:0] total_cols;
reg [4:0] current_col;
reg [4:0] current_row;
reg [5:0] conv_row_idx;        // CONV_TO_BFP 逐行索引

integer i;

reg signed [Q412_WIDTH-1:0] temp_weight;
reg signed [7:0]            shift_amount;

// 权重缓存（Q4.12 格式） - 每列最多 32 行
reg signed [Q412_WIDTH-1:0] weight_col_buffer [0:MAX_ROWS-1];

// BFP 量化后的尾数缓存：按 [列][行] 存储
reg signed [DATA_WIDTH-1:0] bfp_mant_buffer [0:MAX_COLS-1][0:MAX_ROWS-1];

// 梯度累加（列级别）
reg signed [GRAD_ACCUM_WIDTH-1:0] grad_accum;   // 梯度累加器（列级别）
reg [9:0]  token_count;          // 已累加的 token 数
reg [9:0]  grad_token_idx;       // 当前累加的 token 索引

// 供 Q4.12 乘法器使用的截断版本（低 Q412_WIDTH 位）
wire signed [Q412_WIDTH-1:0] grad_accum_q412;
assign grad_accum_q412 = grad_accum[Q412_WIDTH-1:0];

// 旧权重 / 新权重（Q4.12）
reg signed [Q412_WIDTH-1:0] old_weight_q412;
reg signed [Q412_WIDTH-1:0] new_weight_q412;
reg        weight_overflow;
reg [EXP_WIDTH-1:0] old_weight_exp;

// BFP->Q4.12 输入打一拍
reg [DATA_WIDTH-1:0] weight_mant_pipe;
reg [EXP_WIDTH-1:0]  weight_exp_pipe;

// 列归一化（找最大值，计算共享指数）
reg signed [Q412_WIDTH-1:0] col_max_abs;
reg [EXP_WIDTH-1:0]  col_shared_exp;
reg [4:0]            col_leading_zeros;

// 超时计数器
reg [11:0] timeout_counter;

//======================================================================
// 列归一化组合结果
//======================================================================
reg [4:0]            col_leading_zeros_comb;
reg [EXP_WIDTH-1:0]  col_shared_exp_comb;
reg signed [7:0]     shift_amount_comb;

// CONV_TO_BFP 临时变量
reg signed [Q412_WIDTH-1:0] temp_shifted;
reg signed [DATA_WIDTH-1:0]  mant_sat;

//======================================================================
// BFP→Q4.12 转换器
//======================================================================
wire        conv_bfp2q_valid_in;
wire [DATA_WIDTH-1:0] conv_bfp2q_mant_in;
wire [EXP_WIDTH-1:0]  conv_bfp2q_exp_in;
wire        conv_bfp2q_valid_out;
wire signed [Q412_WIDTH-1:0] conv_bfp2q_data_out;
wire        conv_bfp2q_overflow;

assign conv_bfp2q_valid_in = (state_reg == STATE_CONV_WEIGHT);
assign conv_bfp2q_mant_in  = weight_mant_pipe;
assign conv_bfp2q_exp_in   = weight_exp_pipe;

bfp_to_q412_converter #(
    .BFP_MANT_WIDTH (DATA_WIDTH),
    .BFP_EXP_WIDTH  (EXP_WIDTH),
    .Q412_WIDTH     (Q412_WIDTH)
) u_bfp_to_q412 (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (conv_bfp2q_valid_in),
    .bfp_mant   (conv_bfp2q_mant_in),
    .bfp_exp    (conv_bfp2q_exp_in),
    .valid_out  (conv_bfp2q_valid_out),
    .q412_data  (conv_bfp2q_data_out),
    .overflow   (conv_bfp2q_overflow),
    .underflow  (),  // 未使用
    .pipeline_stage ()  // 未使用
);

//======================================================================
// Q4.12 学习率 × 梯度累加 乘法器
//======================================================================
wire        mult_valid_in;
wire signed [Q412_WIDTH-1:0] mult_a;
wire signed [Q412_WIDTH-1:0] mult_b;
wire        mult_valid_out;
wire signed [Q412_WIDTH-1:0] mult_result;
wire        mult_overflow;

assign mult_valid_in = (state_reg == STATE_COMPUTE_UPDATE);
assign mult_a        = learning_rate_reg;   // 学习率
assign mult_b        = grad_accum_q412;     // 梯度累加（截断到 16 位）

q412_multiplier #(
    .DATA_WIDTH (Q412_WIDTH),
    .FRAC_BITS  (12),
    .PIPELINE   (1)
) u_q412_mult (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in  (mult_valid_in),
    .a         (mult_a),
    .b         (mult_b),
    .valid_out (mult_valid_out),
    .result    (mult_result),
    .overflow  (mult_overflow),
    .saturated ()  // 未使用，但需要连接
);

//======================================================================
// 状态机时序寄存器
//======================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        state_reg <= STATE_IDLE;
    else
        state_reg <= state_next;
end

//======================================================================
// 状态机组合逻辑（state_next）
//======================================================================
always @(*) begin
    state_next = state_reg;
    
    case (state_reg)
        //------------------------------------------------------------------
        // IDLE
        //------------------------------------------------------------------
        STATE_IDLE: begin
            if (update_start)
                state_next = STATE_CONFIG;
        end
        
        //------------------------------------------------------------------
        // CONFIG：根据 cfg_layer_id / cfg_weight_type 设置矩阵尺寸
        //------------------------------------------------------------------
        STATE_CONFIG: begin
            state_next = STATE_COL_START;
        end
        
        //------------------------------------------------------------------
        // COL_START：开始新列 → 先累加该列的梯度
        //------------------------------------------------------------------
        STATE_COL_START: begin
            if (current_col < total_cols)
                state_next = STATE_TK_READ_GRAD;
            else
                state_next = STATE_IDLE;
        end
        
        //------------------------------------------------------------------
        // TK_READ_GRAD：发起当前列的梯度读取（逐 token）
        //------------------------------------------------------------------
        STATE_TK_READ_GRAD: begin
            if (grad_token_idx < NUM_TOKENS)
                state_next = STATE_TK_WAIT_GRAD;
            else
                state_next = STATE_ROW_START;  // 所有 token 累加完，进入按行更新
        end
        
        //------------------------------------------------------------------
        // TK_WAIT_GRAD：等待梯度返回 + 累加
        //------------------------------------------------------------------
        STATE_TK_WAIT_GRAD: begin
            if (grad_rd_valid)
                state_next = STATE_TK_READ_GRAD;
            else if (timeout_counter >= TIMEOUT_THRESHOLD)
                state_next = STATE_IDLE;
        end
        
        //------------------------------------------------------------------
        // ROW_START：开始按行读取旧权重并更新
        //------------------------------------------------------------------
        STATE_ROW_START: begin
            if (current_row < total_rows)
                state_next = STATE_READ_WEIGHT;
            else
                state_next = STATE_COL_NORMALIZE;
        end
        
        //------------------------------------------------------------------
        // READ_WEIGHT
        //------------------------------------------------------------------
        STATE_READ_WEIGHT: begin
            state_next = STATE_WAIT_WEIGHT;
        end
        
        //------------------------------------------------------------------
        // WAIT_WEIGHT
        //------------------------------------------------------------------
        STATE_WAIT_WEIGHT: begin
            if (weight_rd_valid)
                state_next = STATE_CONV_WEIGHT;
            else if (timeout_counter >= TIMEOUT_THRESHOLD)
                state_next = STATE_IDLE;
        end
        
        //------------------------------------------------------------------
        // CONV_WEIGHT：BFP→Q4.12 完成后，直接去计算更新
        //------------------------------------------------------------------
        STATE_CONV_WEIGHT: begin
            if (conv_bfp2q_valid_out)
                state_next = STATE_COMPUTE_UPDATE;
        end
        
        //------------------------------------------------------------------
        // COMPUTE_UPDATE：权重更新
        //------------------------------------------------------------------
        STATE_COMPUTE_UPDATE: begin
            if (mult_valid_out)
                state_next = STATE_STORE_TEMP;
        end
        
        //------------------------------------------------------------------
        // STORE_TEMP：缓存新权重，并遍历所有行
        //------------------------------------------------------------------
        STATE_STORE_TEMP: begin
            state_next = STATE_ROW_NEXT;
        end
        
        //------------------------------------------------------------------
        // ROW_NEXT：行自增
        //------------------------------------------------------------------
        STATE_ROW_NEXT: begin
            if (current_row + 5'd1 < total_rows)
                state_next = STATE_ROW_START;
            else
                state_next = STATE_COL_NORMALIZE;
        end
        
        //------------------------------------------------------------------
        // COL_NORMALIZE：根据整列的最大值计算共享指数
        //------------------------------------------------------------------
        STATE_COL_NORMALIZE: begin
            state_next = STATE_CONV_TO_BFP;
        end
        
        //------------------------------------------------------------------
        // CONV_TO_BFP：按共享指数量化整列（逐行流水）
        //------------------------------------------------------------------
        STATE_CONV_TO_BFP: begin
            if (conv_row_idx >= total_rows)
                state_next = STATE_WRITE_BACK;
            else
                state_next = STATE_CONV_TO_BFP;
        end
        
        //------------------------------------------------------------------
        // WRITE_BACK：写回权重存储
        //------------------------------------------------------------------
        STATE_WRITE_BACK: begin
            if (weight_wr_ready)
                state_next = STATE_COL_NEXT;
        end
        
        //------------------------------------------------------------------
        // COL_NEXT：列自增
        //------------------------------------------------------------------
        STATE_COL_NEXT: begin
            state_next = STATE_COL_START;
        end
        
        default: begin
            state_next = STATE_IDLE;
        end
    endcase
end

//======================================================================
// 根据 col_max_abs 计算列共享指数和移位量（组合逻辑）
//======================================================================
always @(*) begin:h
    integer k;
    integer mag_bits;
    reg [Q412_WIDTH-1:0] abs_max;
    reg found;

    col_leading_zeros_comb = 5'd16;
    shift_amount_comb      = 8'sd0;
    col_shared_exp_comb    = {EXP_WIDTH{1'b0}};

    abs_max = col_max_abs[Q412_WIDTH-1] ? -col_max_abs : col_max_abs;

    if (abs_max != 0) begin
        found = 1'b0;
        for (k = Q412_WIDTH-1; k >= 0; k = k - 1) begin
            if (!found && abs_max[k]) begin
                col_leading_zeros_comb = Q412_WIDTH-1 - k;
                found = 1'b1;
            end
        end

        mag_bits = Q412_WIDTH - col_leading_zeros_comb;

        if (mag_bits > (DATA_WIDTH-1))
            shift_amount_comb = mag_bits - (DATA_WIDTH-1);
        else
            shift_amount_comb = 8'sd0;

        if (shift_amount_comb < 0)
            shift_amount_comb = 8'sd0;

        // ✅ 修复：exponent = shift_amount - 12
        col_shared_exp_comb = shift_amount_comb - 8'sd12;
    end else begin
        // ✅ 全零列的情况
        col_shared_exp_comb = -8'sd12;
    end
end
//======================================================================
// 主数据通路时序逻辑
//======================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 复位
        layer_id_reg      <= 3'd0;
        weight_type_reg   <= 4'd0;
        learning_rate_reg <= DEFAULT_LR_Q412;
        total_rows        <= 6'd0;
        total_cols        <= 6'd0;
        current_col       <= 5'd0;
        current_row       <= 5'd0;
        conv_row_idx      <= 6'd0;
        update_done       <= 1'b0;
        update_busy       <= 1'b0;
        timeout_counter   <= 12'd0;
        weight_rd_req     <= 1'b0;
        grad_rd_req       <= 1'b0;
        weight_wr_req     <= 1'b0;
        dbg_update_count  <= 32'd0;
        dbg_col_count     <= 32'd0;
        dbg_row_count     <= 32'd0;
        dbg_timeout_count <= 32'd0;
        dbg_overflow_count<= 32'd0;
        grad_accum        <= {GRAD_ACCUM_WIDTH{1'b0}};
        token_count       <= 10'd0;
        grad_token_idx    <= 10'd0;
        col_max_abs       <= {Q412_WIDTH{1'b0}};
        col_shared_exp    <= {EXP_WIDTH{1'b0}};
        col_leading_zeros <= 5'd0;
        old_weight_q412   <= {Q412_WIDTH{1'b0}};
        old_weight_exp    <= {EXP_WIDTH{1'b0}};
        new_weight_q412   <= {Q412_WIDTH{1'b0}};
        weight_overflow   <= 1'b0;
        weight_wr_exp_array  <= {EXP_WIDTH*MAX_COLS{1'b0}};
        weight_wr_data_burst <= {DRAM_DATA_WIDTH{1'b0}};
        weight_mant_pipe  <= {DATA_WIDTH{1'b0}};
        weight_exp_pipe   <= {EXP_WIDTH{1'b0}};
    end else begin
        // 默认值
        update_done    <= 1'b0;
        weight_rd_req  <= 1'b0;
        grad_rd_req    <= 1'b0;
        weight_wr_req  <= 1'b0;

        // 超时计数只在等待状态自增
        if (state_reg != STATE_WAIT_WEIGHT && state_reg != STATE_TK_WAIT_GRAD)
            timeout_counter <= 12'd0;
        
        case (state_reg)
            //------------------------------------------------------------------
            // IDLE
            //------------------------------------------------------------------
            STATE_IDLE: begin
                if (update_start) begin
                    update_busy      <= 1'b1;
                    dbg_update_count <= dbg_update_count + 32'd1;
                    layer_id_reg     <= cfg_layer_id;
                    weight_type_reg  <= cfg_weight_type;
                    learning_rate_reg<= cfg_learning_rate;
                    current_col      <= 5'd0;
                    current_row      <= 5'd0;
                    grad_accum       <= {GRAD_ACCUM_WIDTH{1'b0}};
                    token_count      <= 10'd0;
                    grad_token_idx   <= 10'd0;
                    col_max_abs      <= {Q412_WIDTH{1'b0}};
                    col_shared_exp   <= {EXP_WIDTH{1'b0}};
                    col_leading_zeros<= 5'd0;
                end else begin
                    update_busy      <= 1'b0;
                end
            end
            
            //------------------------------------------------------------------
            // CONFIG：根据权重类型配置行列数
            //------------------------------------------------------------------
            STATE_CONFIG: begin
                case (cfg_weight_type)
                    // Sidenet 压缩 / 展开
                    WEIGHT_COMPRESS: begin
                        total_rows <= 6'd32;
                        total_cols <= 6'd8;
                    end
                    WEIGHT_EXPAND: begin
                        total_rows <= 6'd8;
                        total_cols <= 6'd32;
                    end
            
                    // Backbone FFN
                    WEIGHT_FFN_W1: begin
                        total_rows <= 6'd8;
                        total_cols <= 6'd32;
                    end
                    WEIGHT_FFN_W2: begin
                        total_rows <= 6'd32;
                        total_cols <= 6'd8;
                    end
            
                    WEIGHT_SIDE_ATT_Q,
                    WEIGHT_SIDE_ATT_K,
                    WEIGHT_SIDE_ATT_V: begin
            
                        total_rows <= SIDENET_DIM[5:0];  // 行数
                        total_cols <= SIDENET_DIM[5:0];  // 列数
                    end
            
                    default: begin
                        total_rows <= 6'd32;
                        total_cols <= 6'd8;
                    end
                endcase
            end 
            
            //------------------------------------------------------------------
            // COL_START：清列相关累加器
            //------------------------------------------------------------------
            STATE_COL_START: begin
                if (current_col >= total_cols) begin
                    update_done <= 1'b1;
                    update_busy <= 1'b0;
                end else begin
                    current_row    <= 5'd0;
                    conv_row_idx   <= 6'd0;
                    col_max_abs    <= {Q412_WIDTH{1'b0}};
                    grad_accum     <= {GRAD_ACCUM_WIDTH{1'b0}};
                    token_count    <= 10'd0;
                    grad_token_idx <= 10'd0;
                    dbg_col_count  <= dbg_col_count + 32'd1;
                end
            end
            
            //------------------------------------------------------------------
            // TK_READ_GRAD：发起梯度读取
            //------------------------------------------------------------------
            STATE_TK_READ_GRAD: begin
                if (grad_token_idx < NUM_TOKENS) begin
                    grad_rd_req      <= 1'b1;
                    grad_rd_layer_id <= layer_id_reg;
                    grad_rd_token_id <= grad_token_idx;
                    grad_rd_dim_id   <= current_col;
                end
            end
            
            //------------------------------------------------------------------
            // TK_WAIT_GRAD：累加梯度
            //------------------------------------------------------------------
            STATE_TK_WAIT_GRAD: begin
                timeout_counter <= timeout_counter + 12'd1;
                if (grad_rd_valid) begin
                    grad_accum <= grad_accum
                                  + {{(GRAD_ACCUM_WIDTH-Q412_WIDTH){grad_rd_data[Q412_WIDTH-1]}},
                                     grad_rd_data};
                    token_count     <= token_count + 10'd1;
                    grad_token_idx  <= grad_token_idx + 10'd1;
                    timeout_counter <= 12'd0;
                end
                if (timeout_counter == TIMEOUT_THRESHOLD)
                    dbg_timeout_count <= dbg_timeout_count + 32'd1;
            end
            
            //------------------------------------------------------------------
            // ROW_START：读旧权重
            //------------------------------------------------------------------
            STATE_ROW_START: begin
                if (current_row < total_rows)
                    dbg_row_count <= dbg_row_count + 32'd1;
            end
            
            //------------------------------------------------------------------
            // READ_WEIGHT：发起旧权重读取
            //------------------------------------------------------------------
            STATE_READ_WEIGHT: begin
                weight_rd_req      <= 1'b1;
                weight_rd_layer_id <= layer_id_reg;
                weight_rd_type     <= weight_type_reg;
                weight_rd_col_id   <= current_col;
                weight_rd_row_id   <= current_row;
            end
            
            //------------------------------------------------------------------
            // WAIT_WEIGHT：等待旧权重
            //------------------------------------------------------------------
            STATE_WAIT_WEIGHT: begin
                timeout_counter <= timeout_counter + 12'd1;
                if (weight_rd_valid) begin
                    weight_mant_pipe <= weight_rd_mant;
                    weight_exp_pipe  <= weight_rd_exp;
                    timeout_counter  <= 12'd0;
                end
                if (timeout_counter == TIMEOUT_THRESHOLD)
                    dbg_timeout_count <= dbg_timeout_count + 32'd1;
            end
            
            //------------------------------------------------------------------
            // CONV_WEIGHT：BFP→Q4.12
            //------------------------------------------------------------------
            STATE_CONV_WEIGHT: begin
                if (conv_bfp2q_valid_out) begin
                    old_weight_q412 <= conv_bfp2q_data_out;
                    old_weight_exp  <= weight_exp_pipe;
                    weight_overflow <= conv_bfp2q_overflow;
                    if (conv_bfp2q_overflow)
                        dbg_overflow_count <= dbg_overflow_count + 32'd1;
                end
            end
            
            //------------------------------------------------------------------
            // COMPUTE_UPDATE：W_new = W_old - lr * grad_accum
            //------------------------------------------------------------------
            STATE_COMPUTE_UPDATE: begin
                if (mult_valid_out) begin
                    new_weight_q412 <= old_weight_q412 - mult_result;
                    weight_overflow <= mult_overflow;
                    if (mult_overflow)
                        dbg_overflow_count <= dbg_overflow_count + 32'd1;
                end
            end
            
            //------------------------------------------------------------------
            // STORE_TEMP：缓存新权重并更新列最大绝对值
            //------------------------------------------------------------------
            STATE_STORE_TEMP: begin
                weight_col_buffer[current_row] <= new_weight_q412;
                if (new_weight_q412[Q412_WIDTH-1]) begin
                    if (-new_weight_q412 > col_max_abs)
                        col_max_abs <= -new_weight_q412;
                end else begin
                    if (new_weight_q412 > col_max_abs)
                        col_max_abs <= new_weight_q412;
                end
            end
            
            //------------------------------------------------------------------
            // ROW_NEXT：行自增
            //------------------------------------------------------------------
            STATE_ROW_NEXT: begin
                if (current_row + 5'd1 < total_rows)
                    current_row <= current_row + 5'd1;
                else
                    current_row <= 5'd0;
            end
            
            //------------------------------------------------------------------
            // COL_NORMALIZE：根据 col_max_abs 计算共享指数 / 移位量
            //------------------------------------------------------------------
            STATE_COL_NORMALIZE: begin
                col_shared_exp    <= col_shared_exp_comb;
                col_leading_zeros <= col_leading_zeros_comb;
                shift_amount      <= shift_amount_comb;
                conv_row_idx      <= 6'd0;
            end
            
            //------------------------------------------------------------------
            // CONV_TO_BFP：按共享指数量化整列（逐行流水）
            //------------------------------------------------------------------
            STATE_CONV_TO_BFP: begin
                if (conv_row_idx < total_rows) begin
                    temp_weight = weight_col_buffer[conv_row_idx];

                    // 1) 按 shift_amount 算术右移对齐
                    if (shift_amount >= Q412_WIDTH)
                        temp_shifted = {Q412_WIDTH{1'b0}};
                    else
                        temp_shifted = temp_weight >>> shift_amount;

                    // 2) 饱和到 DATA_WIDTH 位尾数范围
                    if (temp_shifted > $signed({1'b0, {(DATA_WIDTH-1){1'b1}}})) begin
                        mant_sat = $signed({1'b0, {(DATA_WIDTH-1){1'b1}}});
                    end else if (temp_shifted < $signed({1'b1, {(DATA_WIDTH-1){1'b0}}})) begin
                        mant_sat = $signed({1'b1, {(DATA_WIDTH-1){1'b0}}});
                    end else begin
                        mant_sat = temp_shifted[DATA_WIDTH-1:0];
                    end

                    bfp_mant_buffer[current_col][conv_row_idx] <= mant_sat;

                    conv_row_idx <= conv_row_idx + 6'd1;
                end
            end

            //------------------------------------------------------------------
            // WRITE_BACK：打包 burst 写回
            //------------------------------------------------------------------
            STATE_WRITE_BACK: begin
                weight_wr_req       <= 1'b1;
                weight_wr_layer_id  <= layer_id_reg;
                weight_wr_type      <= weight_type_reg;
                weight_wr_burst_idx <= current_col;   // 行数>16 时当前只写前 16 行
                
                // 指数数组：当前列的指数放在对应位置
                for (i = 0; i < MAX_COLS; i = i + 1) begin
                    if (i == current_col)
                        weight_wr_exp_array[i*EXP_WIDTH +: EXP_WIDTH] <= col_shared_exp;
                end

                // 清零 burst
                weight_wr_data_burst <= {DRAM_DATA_WIDTH{1'b0}};
                
                // 数据 burst 打包
                if (total_rows <= 16) begin
                    // 一个 burst 包含一列所有行
                    for (i = 0; i < total_rows; i = i + 1) begin
                        weight_wr_data_burst[i*DATA_WIDTH +: DATA_WIDTH] 
                            <= bfp_mant_buffer[current_col][i];
                    end
                end else begin
                    // 行数 >16：先写前 16 行（后 16 行可在后续扩展第二个 burst）
                    for (i = 0; i < 16; i = i + 1) begin
                        weight_wr_data_burst[i*DATA_WIDTH +: DATA_WIDTH]
                            <= bfp_mant_buffer[current_col][i];
                    end
                end
            end
            
            //------------------------------------------------------------------
            // COL_NEXT
            //------------------------------------------------------------------
            STATE_COL_NEXT: begin
                current_col <= current_col + 5'd1;
            end
            
            default: ;
        endcase
    end
end

endmodule