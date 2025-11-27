`timescale 1ns / 1ps

//================================================================================
// 实现DFA训练的权重更新，包含完整的格式转换和列共享指数优化
// 1. 读取旧权重列（BFP格式）
// 2. 转换为Q4.12格式
// 3. 读取对应梯度（Q4.12格式）
// 4. 计算新权重：W_new = W_old - learning_rate × gradient
// 5. 列共享指数优化：找列内最大值，计算统一指数
// 6. 转换回BFP格式
// 7. 写回权重存储
//================================================================================

module weight_update_engine #(
    //==========================================================================
    // 基本参数
    //==========================================================================
    parameter NUM_LAYERS        = 5,
    parameter BACKBONE_DIM      = 32,
    parameter SIDENET_DIM       = 8,
    parameter D_FF              = 32,
    parameter DATA_WIDTH        = 16,        // BFP尾数位宽
    parameter EXP_WIDTH         = 8,         // BFP指数位宽
    parameter Q412_WIDTH        = 16,        // Q4.12位宽
    parameter DRAM_DATA_WIDTH   = 256,       // Burst宽度
    parameter NUM_TOKENS        = 641,       // token数量（与gradient_buffer一致）
    
    //==========================================================================
    // 学习率参数
    //==========================================================================
    parameter LR_Q412_WIDTH     = 16,        // 学习率Q4.12位宽
    parameter DEFAULT_LR_Q412   = 16'd41,    // 默认学习率=41/4096≈0.01
    
    //==========================================================================
    // 行列上限（用于BFP缓存）
    //==========================================================================
    parameter MAX_ROWS          = 32,
    parameter MAX_COLS          = 32,
    
    //==========================================================================
    // 超时阈值
    //==========================================================================
    parameter TIMEOUT_THRESHOLD = 12'd4095
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire        update_start,         // 启动更新（脉冲）
    output reg         update_done,          // 更新完成
    output reg         update_busy,          // 更新进行中
    
    //==========================================================================
    // 配置接口
    //==========================================================================
    input  wire [2:0]  cfg_layer_id,         // 要更新的层ID (0-4)
    input  wire [3:0]  cfg_weight_type,      // 权重类型
    input  wire [15:0] cfg_learning_rate,    // 学习率(Q4.12格式)
    
    //==========================================================================
    // 旧权重读取接口（从权重存储读取，BFP格式）
    //==========================================================================
    output reg         weight_rd_req,        // 读请求
    output reg  [2:0]  weight_rd_layer_id,   // 层ID
    output reg  [3:0]  weight_rd_type,       // 权重类型
    output reg  [4:0]  weight_rd_col_id,     // 列ID
    output reg  [4:0]  weight_rd_row_id,     // 行ID（单行读取）
    input  wire        weight_rd_valid,      // 读数据有效
    input  wire [EXP_WIDTH-1:0]  weight_rd_exp,   // BFP指数
    input  wire [DATA_WIDTH-1:0] weight_rd_mant,  // BFP尾数
    
    //==========================================================================
    // 梯度读取接口（从Gradient Buffer读取，Q4.12格式）
    //==========================================================================
    output reg         grad_rd_req,          // 读请求
    output reg  [2:0]  grad_rd_layer_id,     // 层ID
    output reg  [9:0]  grad_rd_token_id,     // Token ID
    output reg  [4:0]  grad_rd_dim_id,       // 维度ID
    input  wire        grad_rd_valid,        // 读数据有效
    input  wire [Q412_WIDTH-1:0] grad_rd_data,    // 梯度数据(Q4.12)
    
    //==========================================================================
    // 新权重写入接口（写入权重存储，BFP格式）
    //==========================================================================
    output reg         weight_wr_req,        // 写请求
    output reg  [2:0]  weight_wr_layer_id,   // 层ID
    output reg  [3:0]  weight_wr_type,       // 权重类型
    output reg  [5:0]  weight_wr_burst_idx,  // Burst索引
    output reg  [EXP_WIDTH*MAX_COLS-1:0] weight_wr_exp_array,  // 指数数组
    output reg  [DRAM_DATA_WIDTH-1:0] weight_wr_data_burst,    // 数据burst
    input  wire        weight_wr_ready,      // 写就绪
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [3:0]  dbg_state,            // 当前状态
    output reg  [31:0] dbg_update_count,     // 更新计数
    output reg  [31:0] dbg_row_count,        // 行处理计数
    output reg  [31:0] dbg_col_count,        // 列处理计数
    output reg  [31:0] dbg_overflow_count,   // 溢出计数
    output wire [15:0] dbg_current_lr        // 当前学习率
);

//================================================================================
// 权重类型定义
//================================================================================
localparam WEIGHT_COMPRESS = 4'd0;   // 压缩层：32×8
localparam WEIGHT_FFN_W1   = 4'd1;   // FFN第一层：8×32
localparam WEIGHT_FFN_W2   = 4'd2;   // FFN第二层：32×8
localparam WEIGHT_EXPAND   = 4'd3;   // 扩展层：8×32
// 其他类型可按需扩展

//================================================================================
// 状态定义
//================================================================================
localparam STATE_IDLE           = 4'd0;
localparam STATE_CONFIG         = 4'd1;
localparam STATE_COL_START      = 4'd2;
localparam STATE_ROW_START      = 4'd3;
localparam STATE_READ_WEIGHT    = 4'd4;
localparam STATE_WAIT_WEIGHT    = 4'd5;
localparam STATE_CONV_WEIGHT    = 4'd6;
localparam STATE_READ_GRAD      = 4'd7;
localparam STATE_WAIT_GRAD      = 4'd8;
localparam STATE_COMPUTE_UPDATE = 4'd9;
localparam STATE_STORE_TEMP     = 4'd10;
localparam STATE_ROW_NEXT       = 4'd11;
localparam STATE_COL_NORMALIZE  = 4'd12;
localparam STATE_CONV_TO_BFP    = 4'd13;
localparam STATE_WRITE_BACK     = 4'd14;
localparam STATE_COL_NEXT       = 4'd15;

reg [3:0] state_reg, state_next;
assign dbg_state = state_reg;
//================================================================================
// 内部寄存器与缓存
//================================================================================
reg [2:0] layer_id_reg;
reg [3:0] weight_type_reg;
reg [LR_Q412_WIDTH-1:0] learning_rate_reg;

reg [5:0] total_rows;
reg [5:0] total_cols;
reg [4:0] current_col;
reg [4:0] current_row;

integer i;
integer j;
reg signed [Q412_WIDTH-1:0] temp_weight;
reg signed [7:0] shift_amount;

// 权重缓存（Q4.12格式） - 每列最多32行
reg signed [Q412_WIDTH-1:0] weight_col_buffer [0:MAX_ROWS-1];

// 梯度缓存（Q4.12格式）
reg signed [Q412_WIDTH-1:0] grad_buffer [0:MAX_ROWS-1];

// BFP转换中间结果
reg [EXP_WIDTH-1:0]  old_weight_exp;
reg [DATA_WIDTH-1:0] old_weight_mant;
reg signed [Q412_WIDTH-1:0] old_weight_q412;

// 梯度累加器
localparam GRAD_ACCUM_WIDTH = 32;

reg signed [GRAD_ACCUM_WIDTH-1:0] grad_accum;   // 梯度累加器（列级别）
reg [9:0]  token_count;          // 已累加的token数
reg [9:0]  grad_token_idx;       // 当前累加的token索引

// 供 Q4.12 乘法器使用的截断版本（低 Q412_WIDTH 位）
wire signed [Q412_WIDTH-1:0] grad_accum_q412;
assign grad_accum_q412 = grad_accum[Q412_WIDTH-1:0];
// 新权重计算
reg signed [Q412_WIDTH-1:0] new_weight_q412;
reg        weight_overflow;

// 列归一化（找最大值，计算共享指数）
reg signed [Q412_WIDTH-1:0] col_max_abs;
reg [EXP_WIDTH-1:0]  col_shared_exp;
reg [4:0]  col_leading_zeros;

// BFP输出缓存
reg [EXP_WIDTH-1:0]  bfp_exp_buffer [0:MAX_COLS-1];
reg [DATA_WIDTH-1:0] bfp_mant_buffer [0:MAX_COLS-1][0:MAX_ROWS-1];

// 超时计数器
reg [11:0] timeout_counter;

assign dbg_current_lr = learning_rate_reg;

//================================================================================
// BFP→Q4.12 转换器
//================================================================================
wire        conv_bfp2q_valid_in;
wire [DATA_WIDTH-1:0] conv_bfp2q_mant_in;
wire [EXP_WIDTH-1:0]  conv_bfp2q_exp_in;
wire        conv_bfp2q_valid_out;
wire signed [Q412_WIDTH-1:0] conv_bfp2q_data_out;
wire        conv_bfp2q_overflow;

assign conv_bfp2q_valid_in = (state_reg == STATE_CONV_WEIGHT);
assign conv_bfp2q_mant_in  = weight_rd_mant;
assign conv_bfp2q_exp_in   = weight_rd_exp;

bfp_to_q412_converter #(
    .DATA_WIDTH (DATA_WIDTH),
    .EXP_WIDTH  (EXP_WIDTH),
    .Q_WIDTH    (Q412_WIDTH)
) u_bfp_to_q412 (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in  (conv_bfp2q_valid_in),
    .bfp_mant   (conv_bfp2q_mant_in),
    .bfp_exp    (conv_bfp2q_exp_in),
    .valid_out (conv_bfp2q_valid_out),
    .q412_data  (conv_bfp2q_data_out),
    .overflow  (conv_bfp2q_overflow)
);

//================================================================================
// Q4.12 乘法器（学习率 × 梯度）
//================================================================================
wire        mult_valid_in;
wire signed [Q412_WIDTH-1:0] mult_a;
wire signed [Q412_WIDTH-1:0] mult_b;
wire signed [Q412_WIDTH-1:0] mult_result;
wire        mult_valid_out;
wire        mult_overflow;

assign mult_valid_in = (state_reg == STATE_COMPUTE_UPDATE);
assign mult_a        = learning_rate_reg;
assign mult_b        = grad_accum_q412;  // 每列累加后的梯度

q412_multiplier #(
    .DATA_WIDTH (Q412_WIDTH),
    .FRAC_BITS  (12)
) u_q412_mult (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in  (mult_valid_in),
    .a         (mult_a),
    .b         (mult_b),
    .result    (mult_result),
    .valid_out (mult_valid_out),
    .overflow  (mult_overflow)
);

//================================================================================
// 状态机时序（state_reg）
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg <= STATE_IDLE;
    end else begin
        state_reg <= state_next;
    end
end

//================================================================================
// 状态机组合逻辑（state_next）
//================================================================================
always @(*) begin
    state_next = state_reg;
    
    case (state_reg)
        //======================================================================
        // IDLE
        //======================================================================
        STATE_IDLE: begin
            if (update_start) begin
                state_next = STATE_CONFIG;
            end
        end
        
        //======================================================================
        // CONFIG
        //======================================================================
        STATE_CONFIG: begin
            state_next = STATE_COL_START;
        end
        
        //======================================================================
        // COL_START：开始新列 → 先累加该列的梯度
        //======================================================================
        STATE_COL_START: begin
            if (current_col < total_cols) begin
                state_next = STATE_READ_GRAD;
            end else begin
                state_next = STATE_IDLE;
            end
        end
        
        //======================================================================
        // ROW_START：所有token的梯度已累加，开始按行更新权重
        //======================================================================
        STATE_ROW_START: begin
            if (current_row < total_rows) begin
                state_next = STATE_READ_WEIGHT;
            end else begin
                state_next = STATE_COL_NORMALIZE;
            end
        end
        
        //======================================================================
        // READ_WEIGHT
        //======================================================================
        STATE_READ_WEIGHT: begin
            state_next = STATE_WAIT_WEIGHT;
        end
        
        //======================================================================
        // WAIT_WEIGHT
        //======================================================================
        STATE_WAIT_WEIGHT: begin
            if (weight_rd_valid) begin
                state_next = STATE_CONV_WEIGHT;
            end else if (timeout_counter >= TIMEOUT_THRESHOLD) begin
                state_next = STATE_IDLE;
            end
        end
        
        //======================================================================
        // CONV_WEIGHT：BFP→Q4.12 完成后，直接去计算更新
        //======================================================================
        STATE_CONV_WEIGHT: begin
            if (conv_bfp2q_valid_out) begin
                state_next = STATE_COMPUTE_UPDATE;
            end
        end
        
        //======================================================================
        // READ_GRAD：发起当前列的梯度读取（逐token）
        //======================================================================
        STATE_READ_GRAD: begin
            state_next = STATE_WAIT_GRAD;
        end
        
        //======================================================================
        // WAIT_GRAD：等待梯度数据，并跨token累加
        //======================================================================
        STATE_WAIT_GRAD: begin
            if (grad_rd_valid) begin
                if (grad_token_idx == NUM_TOKENS-1) begin
                    // 所有token的梯度都已累加，进入行处理
                    state_next = STATE_ROW_START;
                end else begin
                    // 继续读下一个token
                    state_next = STATE_READ_GRAD;
                end
            end else if (timeout_counter >= TIMEOUT_THRESHOLD) begin
                state_next = STATE_IDLE;
            end
        end
        
        //======================================================================
        // COMPUTE_UPDATE
        //======================================================================
        STATE_COMPUTE_UPDATE: begin
            if (mult_valid_out) begin
                state_next = STATE_STORE_TEMP;
            end
        end
        
        //======================================================================
        // STORE_TEMP
        //======================================================================
        STATE_STORE_TEMP: begin
            state_next = STATE_ROW_NEXT;
        end
        
        //======================================================================
        // ROW_NEXT
        //======================================================================
        STATE_ROW_NEXT: begin
            state_next = STATE_ROW_START;
        end
        
        //======================================================================
        // COL_NORMALIZE
        //======================================================================
        STATE_COL_NORMALIZE: begin
            state_next = STATE_CONV_TO_BFP;
        end
        
        //======================================================================
        // CONV_TO_BFP
        //======================================================================
        STATE_CONV_TO_BFP: begin
            state_next = STATE_WRITE_BACK;
        end
        
        //======================================================================
        // WRITE_BACK
        //======================================================================
        STATE_WRITE_BACK: begin
            if (weight_wr_ready) begin
                state_next = STATE_COL_NEXT;
            end else if (timeout_counter >= TIMEOUT_THRESHOLD) begin
                state_next = STATE_IDLE;
            end
        end
        
        //======================================================================
        // COL_NEXT
        //======================================================================
        STATE_COL_NEXT: begin
            state_next = STATE_COL_START;
        end
        
        default: begin
            state_next = STATE_IDLE;
        end
    endcase
end

//================================================================================
// 主数据通路时序逻辑
//================================================================================
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
        update_done       <= 1'b0;
        update_busy       <= 1'b0;
        timeout_counter   <= 12'd0;
        weight_rd_req     <= 1'b0;
        grad_rd_req       <= 1'b0;
        weight_wr_req     <= 1'b0;
        dbg_update_count  <= 32'd0;
        dbg_row_count     <= 32'd0;
        dbg_col_count     <= 32'd0;
        dbg_overflow_count<= 32'd0;
        grad_accum        <= {GRAD_ACCUM_WIDTH{1'b0}};
        token_count       <= 10'd0;
        grad_token_idx    <= 10'd0;
        col_max_abs       <= {Q412_WIDTH{1'b0}};
        col_shared_exp    <= {EXP_WIDTH{1'b0}};
        col_leading_zeros <= 5'd0;
    end else begin
        // 默认：清除单周期脉冲
        update_done   <= 1'b0;
        weight_rd_req <= 1'b0;
        grad_rd_req   <= 1'b0;
        weight_wr_req <= 1'b0;
        
        case (state_reg)
            //------------------------------------------------------------------
            // IDLE
            //------------------------------------------------------------------
            STATE_IDLE: begin
                timeout_counter <= 12'd0;
                if (update_start) begin
                    update_busy      <= 1'b1;
                    dbg_update_count <= dbg_update_count + 32'd1;
                end
            end
            
            //------------------------------------------------------------------
            // CONFIG：根据类型设置行列数
            //------------------------------------------------------------------
            STATE_CONFIG: begin
                layer_id_reg      <= cfg_layer_id;
                weight_type_reg   <= cfg_weight_type;
                learning_rate_reg <= cfg_learning_rate;
                
                case (cfg_weight_type)
                    WEIGHT_COMPRESS: begin
                        total_rows <= 6'd32;
                        total_cols <= 6'd8;
                    end
                    WEIGHT_FFN_W1: begin
                        total_rows <= 6'd8;
                        total_cols <= 6'd32;
                    end
                    WEIGHT_FFN_W2: begin
                        total_rows <= 6'd32;
                        total_cols <= 6'd8;
                    end
                    WEIGHT_EXPAND: begin
                        total_rows <= 6'd8;
                        total_cols <= 6'd32;
                    end
                    default: begin
                        total_rows <= 6'd8;
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
                    col_max_abs    <= {Q412_WIDTH{1'b0}};
                    grad_accum     <= {GRAD_ACCUM_WIDTH{1'b0}};
                    token_count    <= 10'd0;
                    grad_token_idx <= 10'd0;
                    dbg_col_count  <= dbg_col_count + 32'd1;
                end
            end
            
            //------------------------------------------------------------------
            // READ_WEIGHT
            //------------------------------------------------------------------
            STATE_READ_WEIGHT: begin
                weight_rd_req      <= 1'b1;
                weight_rd_layer_id <= layer_id_reg;
                weight_rd_type     <= weight_type_reg;
                weight_rd_col_id   <= current_col;
                weight_rd_row_id   <= current_row;
            end
            
            //------------------------------------------------------------------
            // WAIT_WEIGHT
            //------------------------------------------------------------------
            STATE_WAIT_GRAD: begin
                timeout_counter <= timeout_counter + 12'd1;
            
                if (grad_rd_valid) begin
                    // 对 grad_rd_data 做显式符号扩展到 32bit，然后累加
                    grad_accum <= grad_accum
                                  + {{(GRAD_ACCUM_WIDTH-Q412_WIDTH){grad_rd_data[Q412_WIDTH-1]}},
                                     grad_rd_data};
            
                    token_count     <= token_count + 10'd1;
                    grad_token_idx  <= grad_token_idx + 10'd1;
                    timeout_counter <= 12'd0;
                end
            end
                        
            //------------------------------------------------------------------
            // CONV_WEIGHT：等待BFP→Q4.12完成
            //------------------------------------------------------------------
            STATE_CONV_WEIGHT: begin
                if (conv_bfp2q_valid_out) begin
                    old_weight_q412 <= conv_bfp2q_data_out;
                    if (conv_bfp2q_overflow) begin
                        dbg_overflow_count <= dbg_overflow_count + 32'd1;
                    end
                end
            end
            
            //------------------------------------------------------------------
            // READ_GRAD：跨token读取当前列梯度
            //------------------------------------------------------------------
            STATE_READ_GRAD: begin
                grad_rd_req      <= 1'b1;
                grad_rd_layer_id <= layer_id_reg;
                grad_rd_token_id <= grad_token_idx; // 0..NUM_TOKENS-1
                grad_rd_dim_id   <= current_col;    // 列 = 维度
            end
            
            //------------------------------------------------------------------
            // WAIT_GRAD：累加梯度
            //------------------------------------------------------------------
            STATE_WAIT_GRAD: begin
                timeout_counter <= timeout_counter + 12'd1;
                if (grad_rd_valid) begin
                    grad_accum      <= grad_accum + grad_rd_data;
                    token_count     <= token_count + 10'd1;
                    grad_token_idx  <= grad_token_idx + 10'd1;
                    timeout_counter <= 12'd0;
                end
            end
            
            //------------------------------------------------------------------
            // COMPUTE_UPDATE：W_new = W_old - lr * grad_accum
            //------------------------------------------------------------------
            STATE_COMPUTE_UPDATE: begin
                if (mult_valid_out) begin
                    new_weight_q412 <= old_weight_q412 - mult_result;
                    weight_overflow <= mult_overflow;
                    if (mult_overflow) begin
                        dbg_overflow_count <= dbg_overflow_count + 32'd1;
                    end
                end
            end
            
            //------------------------------------------------------------------
            // STORE_TEMP：缓存新权重并更新列最大绝对值
            //------------------------------------------------------------------
            STATE_STORE_TEMP: begin
                weight_col_buffer[current_row] <= new_weight_q412;
                if (new_weight_q412[Q412_WIDTH-1]) begin
                    if (-new_weight_q412 > col_max_abs) begin
                        col_max_abs <= -new_weight_q412;
                    end
                end else begin
                    if (new_weight_q412 > col_max_abs) begin
                        col_max_abs <= new_weight_q412;
                    end
                end
                dbg_row_count <= dbg_row_count + 32'd1;
            end
            
            //------------------------------------------------------------------
            // ROW_NEXT
            //------------------------------------------------------------------
            STATE_ROW_NEXT: begin
                current_row <= current_row + 5'd1;
            end
            
            //------------------------------------------------------------------
            // COL_NORMALIZE：根据col_max_abs计算共享指数
            //------------------------------------------------------------------
            STATE_COL_NORMALIZE: begin
             
                col_shared_exp    <= old_weight_exp;
                col_leading_zeros <= 5'd0;
            end
            
            //------------------------------------------------------------------
            // CONV_TO_BFP：按共享指数量化整列
            //------------------------------------------------------------------
            STATE_CONV_TO_BFP: begin
                for (i = 0; i < total_rows; i = i + 1) begin
                    temp_weight = weight_col_buffer[i];
                    if (temp_weight[Q412_WIDTH-1]) begin
                        if (-temp_weight > col_max_abs)
                            col_max_abs <= -temp_weight;
                    end else begin
                        if (temp_weight > col_max_abs)
                            col_max_abs <= temp_weight;
                    end
                    shift_amount = 0;
                    if (temp_weight != 0) begin
                        bfp_mant_buffer[current_col][i] <= temp_weight[DATA_WIDTH-1:0];
                    end else begin
                        bfp_mant_buffer[current_col][i] <= {DATA_WIDTH{1'b0}};
                    end
                end
            end
            
            //------------------------------------------------------------------
            // WRITE_BACK：打包burst写回
            //------------------------------------------------------------------
            STATE_WRITE_BACK: begin
                weight_wr_req       <= 1'b1;
                weight_wr_layer_id  <= layer_id_reg;
                weight_wr_type      <= weight_type_reg;
                weight_wr_burst_idx <= current_col;
                
                // 指数数组：当前列的指数放在对应位置，其它可保持不变/零
                for (i = 0; i < MAX_COLS; i = i + 1) begin
                    if (i == current_col)
                        weight_wr_exp_array[i*EXP_WIDTH +: EXP_WIDTH] <= col_shared_exp;
                end
                
                // 数据burst打包（按已有布局）
                if (total_rows <= 16) begin
                    // 一个burst包含一列所有行
                    for (i = 0; i < total_rows; i = i + 1) begin
                        weight_wr_data_burst[i*DATA_WIDTH +: DATA_WIDTH] 
                            <= bfp_mant_buffer[current_col][i];
                    end
                end else begin
                    // 行数>16：按两burst写
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
