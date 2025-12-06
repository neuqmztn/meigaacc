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
    parameter NUM_TOKENS        = 641,       // token 数量
    
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
// 层 ID 和权重类型编码
//======================================================================
localparam WEIGHT_COMPRESS = 4'd0;  // 32×8
localparam WEIGHT_EXPAND   = 4'd1;  // 8×32
localparam WEIGHT_FFN_W1   = 4'd2;  // 8×32
localparam WEIGHT_FFN_W2   = 4'd3;  // 32×8
localparam WEIGHT_SIDE_ATT_Q = 4'd4; // SIDENET_DIM × SIDENET_DIM
localparam WEIGHT_SIDE_ATT_K = 4'd5;
localparam WEIGHT_SIDE_ATT_V = 4'd6;

//======================================================================
// 状态机定义
//======================================================================
localparam STATE_IDLE           = 4'd0;
localparam STATE_CONFIG         = 4'd1;
localparam STATE_COL_START      = 4'd2;
localparam STATE_TK_READ_GRAD   = 4'd3;
localparam STATE_TK_WAIT_GRAD   = 4'd4;
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

localparam integer ROWS_PER_BURST = DRAM_DATA_WIDTH / DATA_WIDTH; // 16

reg [3:0] state_reg, state_next;
assign dbg_state = state_reg;

//======================================================================
// 寄存器定义
//======================================================================
reg [2:0]  layer_id_reg;
reg [3:0]  weight_type_reg;
reg [15:0] learning_rate_reg;

reg [5:0] total_rows;
reg [5:0] total_cols;
reg [4:0] current_col;
reg [4:0] current_row;
reg [5:0] conv_row_idx;

// [新增] Burst 计数器，支持分两次写回 (>16行的情况)
reg       burst_cnt; 

integer i;

// 临时变量 (已移出复位逻辑)
reg signed [Q412_WIDTH-1:0] temp_weight;
reg signed [Q412_WIDTH-1:0] temp_shifted;
reg signed [DATA_WIDTH-1:0] mant_sat;
reg signed [7:0]            shift_amount; 

// 权重缓存 - 不在复位逻辑中清除
reg signed [Q412_WIDTH-1:0] weight_col_buffer [0:MAX_ROWS-1];

// BFP 量化后的尾数缓存 - 不在复位逻辑中清除
reg signed [DATA_WIDTH-1:0] bfp_mant_buffer [0:MAX_COLS-1][0:MAX_ROWS-1];

// 梯度累加
reg signed [GRAD_ACCUM_WIDTH-1:0] grad_accum;
reg [9:0]  token_count;
reg [9:0]  grad_token_idx;

// 供 Q4.12 乘法器使用的截断版本
wire signed [Q412_WIDTH-1:0] grad_accum_q412;
assign grad_accum_q412 = grad_accum[Q412_WIDTH-1:0];

// 旧权重 / 新权重
reg signed [Q412_WIDTH-1:0] old_weight_q412;
reg signed [Q412_WIDTH-1:0] new_weight_q412;
reg        weight_overflow;
reg [EXP_WIDTH-1:0] old_weight_exp;

// BFP->Q4.12 输入打一拍
reg [DATA_WIDTH-1:0] weight_mant_pipe;
reg [EXP_WIDTH-1:0]  weight_exp_pipe;

// 列归一化
reg signed [Q412_WIDTH-1:0] col_max_abs;
reg [EXP_WIDTH-1:0]  col_shared_exp;
reg [4:0]            col_leading_zeros;

// 超时计数器
reg [11:0] timeout_counter;

// 组合逻辑变量
reg [4:0]            col_leading_zeros_comb;
reg [EXP_WIDTH-1:0]  col_shared_exp_comb;
reg signed [7:0]     shift_amount_comb;

//======================================================================
// BFP→Q4.12 转换器实例
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
    .underflow  (),
    .pipeline_stage ()
);

//======================================================================
// Q4.12 乘法器实例
//======================================================================
wire        mult_valid_in;
wire signed [Q412_WIDTH-1:0] mult_a;
wire signed [Q412_WIDTH-1:0] mult_b;
wire        mult_valid_out;
wire signed [Q412_WIDTH-1:0] mult_result;
wire        mult_overflow;

assign mult_valid_in = (state_reg == STATE_COMPUTE_UPDATE);
assign mult_a        = learning_rate_reg;
assign mult_b        = grad_accum_q412;

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
    .saturated ()
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
// 状态机组合逻辑
//======================================================================
always @(*) begin
    state_next = state_reg;
    case (state_reg)
        STATE_IDLE: begin
            if (update_start)
                state_next = STATE_CONFIG;
        end
        STATE_CONFIG:       state_next = STATE_COL_START;
        STATE_COL_START: begin
            if (current_col < total_cols)
                state_next = STATE_TK_READ_GRAD;
            else
                state_next = STATE_IDLE;
        end
        STATE_TK_READ_GRAD: begin
            if (grad_token_idx < NUM_TOKENS)
                state_next = STATE_TK_WAIT_GRAD;
            else
                state_next = STATE_ROW_START;
        end
        STATE_TK_WAIT_GRAD: begin
            if (grad_rd_valid)
                state_next = STATE_TK_READ_GRAD;
            else if (timeout_counter >= TIMEOUT_THRESHOLD)
                state_next = STATE_IDLE;
        end
        STATE_ROW_START: begin
            if (current_row < total_rows)
                state_next = STATE_READ_WEIGHT;
            else
                state_next = STATE_COL_NORMALIZE;
        end
        STATE_READ_WEIGHT:  state_next = STATE_WAIT_WEIGHT;
        STATE_WAIT_WEIGHT: begin
            if (weight_rd_valid)
                state_next = STATE_CONV_WEIGHT;
            else if (timeout_counter >= TIMEOUT_THRESHOLD)
                state_next = STATE_IDLE;
        end
        STATE_CONV_WEIGHT: begin
            if (conv_bfp2q_valid_out)
                state_next = STATE_COMPUTE_UPDATE;
        end
        STATE_COMPUTE_UPDATE: begin
            if (mult_valid_out)
                state_next = STATE_STORE_TEMP;
        end
        STATE_STORE_TEMP:   state_next = STATE_ROW_NEXT;
        STATE_ROW_NEXT: begin
            if (current_row + 5'd1 < total_rows)
                state_next = STATE_ROW_START;
            else
                state_next = STATE_COL_NORMALIZE;
        end
        STATE_COL_NORMALIZE: state_next = STATE_CONV_TO_BFP;
        
        STATE_CONV_TO_BFP: begin
            if (conv_row_idx >= total_rows)
                state_next = STATE_WRITE_BACK;
            else
                state_next = STATE_CONV_TO_BFP;
        end
        
        // 支持分次Burst
        STATE_WRITE_BACK: begin
            if (weight_wr_ready) begin
                if (total_rows > ROWS_PER_BURST && burst_cnt == 1'b0)
                    state_next = STATE_WRITE_BACK;
                else
                    state_next = STATE_COL_NEXT;
            end else begin
                state_next = STATE_WRITE_BACK;
            end
        end
        
        STATE_COL_NEXT:     state_next = STATE_COL_START;
        default:            state_next = STATE_IDLE;
    endcase
end

//======================================================================
// 根据 col_max_abs 计算列共享指数和移位量
//======================================================================
always @(*) begin: calc_shared_exp
    integer k;
    integer mag_bits;
    reg [Q412_WIDTH-1:0] abs_max;
    reg found;

    col_leading_zeros_comb = 5'd16;
    shift_amount_comb      = 8'sd0;
    col_shared_exp_comb    = {EXP_WIDTH{1'b0}};
    
    // 取绝对值
    abs_max = col_max_abs[Q412_WIDTH-1] ? -col_max_abs : col_max_abs;

    if (abs_max != 0) begin
        found = 1'b0;
        // 查找前导零
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
            
        col_shared_exp_comb = shift_amount_comb - 8'sd12;
    end else begin
        col_shared_exp_comb = -8'sd12;
    end
end

//======================================================================
// 主数据通路时序逻辑
//======================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        //--------------------------------------------------------------
        // [关键] 复位逻辑：只复位控制信号、计数器和接口
        // 确保没有 bfp_mant_buffer 和 weight_col_buffer
        //--------------------------------------------------------------
        layer_id_reg        <= 3'd0;
        weight_type_reg     <= 4'd0;
        learning_rate_reg   <= DEFAULT_LR_Q412;
        total_rows          <= 6'd0;
        total_cols          <= 6'd0;
        current_col         <= 5'd0;
        current_row         <= 5'd0;
        conv_row_idx        <= 6'd0;
        burst_cnt           <= 1'b0;
        
        update_done         <= 1'b0;
        update_busy         <= 1'b0;
        timeout_counter     <= 12'd0;
        
        weight_rd_req       <= 1'b0;
        weight_rd_layer_id  <= 3'd0;
        weight_rd_type      <= 4'd0;
        weight_rd_col_id    <= 5'd0;
        weight_rd_row_id    <= 5'd0;
        
        grad_rd_req         <= 1'b0;
        grad_rd_layer_id    <= 3'd0;
        grad_rd_token_id    <= 10'd0;
        grad_rd_dim_id      <= 5'd0;
        
        weight_wr_req       <= 1'b0;
        weight_wr_layer_id  <= 3'd0;
        weight_wr_type      <= 4'd0;
        weight_wr_burst_idx <= 6'd0;
        
        dbg_update_count    <= 32'd0;
        dbg_col_count       <= 32'd0;
        dbg_row_count       <= 32'd0;
        dbg_timeout_count   <= 32'd0;
        dbg_overflow_count  <= 32'd0;
        
        grad_accum          <= {GRAD_ACCUM_WIDTH{1'b0}};
        token_count         <= 10'd0;
        grad_token_idx      <= 10'd0;
        col_max_abs         <= {Q412_WIDTH{1'b0}};
        col_shared_exp      <= {EXP_WIDTH{1'b0}};
        col_leading_zeros   <= 5'd0;
        
        old_weight_q412     <= {Q412_WIDTH{1'b0}};
        old_weight_exp      <= {EXP_WIDTH{1'b0}};
        new_weight_q412     <= {Q412_WIDTH{1'b0}};
        weight_overflow     <= 1'b0;
        
        weight_wr_exp_array  <= {EXP_WIDTH*MAX_COLS{1'b0}};
        weight_wr_data_burst <= {DRAM_DATA_WIDTH{1'b0}};
        
        weight_mant_pipe     <= {DATA_WIDTH{1'b0}};
        weight_exp_pipe      <= {EXP_WIDTH{1'b0}};
        
    end else begin
        //--------------------------------------------------------------
        // 默认信号
        //--------------------------------------------------------------
        update_done    <= 1'b0;
        weight_rd_req  <= 1'b0;
        grad_rd_req    <= 1'b0;
        weight_wr_req  <= 1'b0;
        
        if (state_reg != STATE_WAIT_WEIGHT && state_reg != STATE_TK_WAIT_GRAD)
            timeout_counter <= 12'd0;
            
        case (state_reg)
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
            
            STATE_CONFIG: begin
                case (cfg_weight_type)
                    WEIGHT_COMPRESS: begin
                        total_rows <= 6'd32;
                        total_cols <= 6'd8;
                    end
                    WEIGHT_EXPAND: begin
                        total_rows <= 6'd8;
                        total_cols <= 6'd32;
                    end
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
                        total_rows <= SIDENET_DIM[5:0];
                        total_cols <= SIDENET_DIM[5:0];
                    end
                    default: begin
                        total_rows <= 6'd32;
                        total_cols <= 6'd8;
                    end
                endcase
            end 
            
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
            
            STATE_TK_READ_GRAD: begin
                if (grad_token_idx < NUM_TOKENS) begin
                    grad_rd_req      <= 1'b1;
                    grad_rd_layer_id <= layer_id_reg;
                    grad_rd_token_id <= grad_token_idx;
                    grad_rd_dim_id   <= current_col;
                end
            end
            
            STATE_TK_WAIT_GRAD: begin
                timeout_counter <= timeout_counter + 12'd1;
                if (grad_rd_valid) begin
                    grad_accum <= grad_accum + 
                                  {{(GRAD_ACCUM_WIDTH-Q412_WIDTH){grad_rd_data[Q412_WIDTH-1]}}, 
                                  grad_rd_data};
                    token_count     <= token_count + 10'd1;
                    grad_token_idx  <= grad_token_idx + 10'd1;
                    timeout_counter <= 12'd0;
                end
                if (timeout_counter == TIMEOUT_THRESHOLD)
                    dbg_timeout_count <= dbg_timeout_count + 32'd1;
            end
            
            STATE_ROW_START: begin
                if (current_row < total_rows)
                    dbg_row_count <= dbg_row_count + 32'd1;
            end
            
            STATE_READ_WEIGHT: begin
                weight_rd_req      <= 1'b1;
                weight_rd_layer_id <= layer_id_reg;
                weight_rd_type     <= weight_type_reg;
                weight_rd_col_id   <= current_col;
                weight_rd_row_id   <= current_row;
            end
            
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
            
            STATE_CONV_WEIGHT: begin
                if (conv_bfp2q_valid_out) begin
                    old_weight_q412 <= conv_bfp2q_data_out;
                    old_weight_exp  <= weight_exp_pipe;
                    weight_overflow <= conv_bfp2q_overflow;
                    if (conv_bfp2q_overflow)
                        dbg_overflow_count <= dbg_overflow_count + 32'd1;
                end
            end
            
            STATE_COMPUTE_UPDATE: begin
                if (mult_valid_out) begin
                    new_weight_q412 <= old_weight_q412 - mult_result;
                    weight_overflow <= mult_overflow;
                    if (mult_overflow)
                        dbg_overflow_count <= dbg_overflow_count + 32'd1;
                end
            end
            
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
            
            STATE_ROW_NEXT: begin
                if (current_row + 5'd1 < total_rows)
                    current_row <= current_row + 5'd1;
                else
                    current_row <= 5'd0;
            end
            
            STATE_COL_NORMALIZE: begin
                col_shared_exp    <= col_shared_exp_comb;
                col_leading_zeros <= col_leading_zeros_comb;
                shift_amount      <= shift_amount_comb;
                conv_row_idx      <= 6'd0;
            end
            
            STATE_CONV_TO_BFP: begin
                if (conv_row_idx < total_rows) begin
                    temp_weight = weight_col_buffer[conv_row_idx];
                    
                    if (shift_amount >= Q412_WIDTH)
                        temp_shifted = {Q412_WIDTH{1'b0}};
                    else
                        temp_shifted = temp_weight >>> shift_amount;
                        
                    if (temp_shifted > $signed({1'b0, {(DATA_WIDTH-1){1'b1}}})) begin
                        mant_sat = $signed({1'b0, {(DATA_WIDTH-1){1'b1}}});
                    end else if (temp_shifted < $signed({1'b1, {(DATA_WIDTH-1){1'b0}}})) begin
                        mant_sat = $signed({1'b1, {(DATA_WIDTH-1){1'b0}}});
                    end else begin
                        mant_sat = temp_shifted[DATA_WIDTH-1:0];
                    end

                    bfp_mant_buffer[current_col][conv_row_idx] <= mant_sat;
                    
                    conv_row_idx <= conv_row_idx + 6'd1;
                end else begin
                    burst_cnt <= 1'b0;
                end
            end

            STATE_WRITE_BACK: begin
                weight_wr_req       <= 1'b1;
                weight_wr_layer_id  <= layer_id_reg;
                weight_wr_type      <= weight_type_reg;
                
                weight_wr_burst_idx <= (current_col << 1) + {5'd0, burst_cnt};
                
                for (i = 0; i < MAX_COLS; i = i + 1) begin
                    if (i == current_col)
                        weight_wr_exp_array[i*EXP_WIDTH +: EXP_WIDTH] <= col_shared_exp;
                end

                weight_wr_data_burst <= {DRAM_DATA_WIDTH{1'b0}};
                
                for (i = 0; i < ROWS_PER_BURST; i = i + 1) begin:t
                    integer row_ptr;
                    row_ptr = i + (burst_cnt ? 16 : 0);
                    
                    if (row_ptr < total_rows) begin
                        weight_wr_data_burst[i*DATA_WIDTH +: DATA_WIDTH]
                            <= bfp_mant_buffer[current_col][row_ptr];
                    end else begin
                        weight_wr_data_burst[i*DATA_WIDTH +: DATA_WIDTH]
                            <= {DATA_WIDTH{1'b0}};
                    end
                end
                
                if (weight_wr_ready) begin
                    if (total_rows > ROWS_PER_BURST && burst_cnt == 1'b0) begin
                        burst_cnt <= 1'b1;
                        weight_wr_req <= 1'b1;
                    end else begin
                        burst_cnt <= 1'b0;
                        weight_wr_req <= 1'b0;
                    end
                end
            end
            
            STATE_COL_NEXT: begin
                current_col <= current_col + 5'd1;
            end
            
            default: ;
        endcase
    end
end

endmodule