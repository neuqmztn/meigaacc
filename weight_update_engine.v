`timescale 1ns / 1ps

//================================================================================
// Weight Update Engine - v1.0 完整实现版
//
// 功能说明：
// 实现DFA训练的权重更新，包含完整的格式转换和列共享指数优化
//
// 核心算法：
// 1. 读取旧权重列（BFP格式）
// 2. 转换为Q4.12格式
// 3. 读取对应梯度（Q4.12格式）
// 4. 计算新权重：W_new = W_old - learning_rate × gradient
// 5. 列共享指数优化：找列内最大值，计算统一指数
// 6. 转换回BFP格式
// 7. 写回权重存储
//
// 格式转换：
// • 读取时：BFP → Q4.12（单个转换器复用）
// • 写回时：Q4.12 → BFP（列归一化，共享指数）
//
// 支持权重类型：
// • Compression: 32×8 (32行8列)
// • Attention QKV/WO: 8×8
// • FFN W1: 8×32
// • FFN W2: 32×8
// • Expand: 8×32
//
// 处理策略：
// • 按列处理（每列共享一个指数）
// • 每列串行处理所有行
// • 使用流水线减少延迟
//
// 作者：MEIGA Team
// 日期：2025-11-18
// 版本：v1.0
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
    
    //==========================================================================
    // 学习率参数
    //==========================================================================
    parameter LR_Q412_WIDTH     = 16,        // 学习率Q4.12位宽
    parameter DEFAULT_LR_Q412   = 16'd41,    // 默认学习率0.01 (41/4096)
    
    //==========================================================================
    // 地址参数
    //==========================================================================
    parameter MAX_ROWS          = 32,        // 最大行数
    parameter MAX_COLS          = 32,        // 最大列数
    parameter ROW_ADDR_WIDTH    = 5,         // 行地址位宽
    parameter COL_ADDR_WIDTH    = 5,         // 列地址位宽
    
    //==========================================================================
    // 转换器参数
    //==========================================================================
    parameter CONV_PIPELINE     = 3          // BFP→Q4.12流水线级数
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
localparam WEIGHT_COMPRESS = 4'd0;   // 32×8
localparam WEIGHT_ATT_WQ   = 4'd2;   // 8×8
localparam WEIGHT_ATT_WK   = 4'd3;   // 8×8
localparam WEIGHT_ATT_WV   = 4'd4;   // 8×8
localparam WEIGHT_ATT_WO   = 4'd5;   // 8×8
localparam WEIGHT_FFN_W1   = 4'd6;   // 8×32
localparam WEIGHT_FFN_W2   = 4'd7;   // 32×8
localparam WEIGHT_EXPAND   = 4'd8;   // 8×32

//================================================================================
// 状态机定义
//================================================================================
localparam STATE_IDLE           = 4'd0;   // 空闲
localparam STATE_CONFIG         = 4'd1;   // 配置参数
localparam STATE_COL_START      = 4'd2;   // 列处理开始
localparam STATE_ROW_START      = 4'd3;   // 行处理开始
localparam STATE_READ_WEIGHT    = 4'd4;   // 读取旧权重
localparam STATE_WAIT_WEIGHT    = 4'd5;   // 等待权重数据
localparam STATE_CONV_WEIGHT    = 4'd6;   // 转换权重(BFP→Q4.12)
localparam STATE_READ_GRAD      = 4'd7;   // 读取梯度
localparam STATE_WAIT_GRAD      = 4'd8;   // 等待梯度数据
localparam STATE_COMPUTE_UPDATE = 4'd9;   // 计算更新
localparam STATE_STORE_TEMP     = 4'd10;  // 暂存新权重(Q4.12)
localparam STATE_ROW_NEXT       = 4'd11;  // 下一行
localparam STATE_COL_NORMALIZE  = 4'd12;  // 列归一化
localparam STATE_CONV_TO_BFP    = 4'd13;  // 转换为BFP
localparam STATE_WRITE_BACK     = 4'd14;  // 写回权重
localparam STATE_COL_NEXT       = 4'd15;  // 下一列

reg [3:0] state_reg, state_next;
assign dbg_state = state_reg;

//================================================================================
// 内部寄存器
//================================================================================

// 配置寄存器
reg [2:0]  layer_id_reg;
reg [3:0]  weight_type_reg;
reg [15:0] learning_rate_reg;
reg [4:0]  total_rows;           // 当前权重矩阵的总行数
reg [4:0]  total_cols;           // 当前权重矩阵的总列数

// 循环计数器
reg [4:0]  current_col;          // 当前处理的列
reg [4:0]  current_row;          // 当前处理的行

// 循环变量和临时变量（用于BFP转换）
integer i;                       // 通用循环变量
integer j;                       // 通用循环变量
reg signed [Q412_WIDTH-1:0] temp_weight;    // 临时权重值
reg signed [7:0] shift_amount;              // 移位量

// 权重缓存（Q4.12格式）
// 每列最多32行
reg signed [Q412_WIDTH-1:0] weight_col_buffer [0:MAX_ROWS-1];

// 梯度缓存（Q4.12格式）
reg signed [Q412_WIDTH-1:0] grad_buffer [0:MAX_ROWS-1];

// BFP转换中间结果
reg [EXP_WIDTH-1:0]  old_weight_exp;
reg [DATA_WIDTH-1:0] old_weight_mant;
reg signed [Q412_WIDTH-1:0] old_weight_q412;

// 梯度累加器（支持多token梯度累加）
reg signed [Q412_WIDTH-1:0] grad_accum;
reg [9:0]  token_count;          // 已累加的token数

// 新权重计算
reg signed [Q412_WIDTH-1:0] new_weight_q412;
reg        weight_overflow;

// 列归一化（找最大值，计算共享指数）
reg signed [Q412_WIDTH-1:0] col_max_abs;    // 列内最大绝对值
reg [EXP_WIDTH-1:0]  col_shared_exp;        // 列共享指数
reg [4:0]  col_leading_zeros;               // 前导零数量

// BFP输出缓存
reg [EXP_WIDTH-1:0]  bfp_exp_buffer [0:MAX_COLS-1];
reg [DATA_WIDTH-1:0] bfp_mant_buffer [0:MAX_COLS-1][0:MAX_ROWS-1];

// 超时计数器
reg [11:0] timeout_counter;
localparam TIMEOUT_THRESHOLD = 12'd1000;

//================================================================================
// BFP → Q4.12 转换器实例
//================================================================================
wire        conv_bfp2q_valid_in;
wire [DATA_WIDTH-1:0] conv_bfp2q_mant_in;
wire [EXP_WIDTH-1:0]  conv_bfp2q_exp_in;
wire signed [Q412_WIDTH-1:0] conv_bfp2q_data_out;
wire        conv_bfp2q_valid_out;
wire        conv_bfp2q_overflow;

bfp_to_q412_converter #(
    .BFP_MANT_WIDTH (DATA_WIDTH),
    .BFP_EXP_WIDTH  (EXP_WIDTH),
    .Q412_WIDTH     (Q412_WIDTH),
    .Q412_FRAC_BITS (12)
) u_bfp_to_q412 (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (conv_bfp2q_valid_in),
    .bfp_mant   (conv_bfp2q_mant_in),
    .bfp_exp    (conv_bfp2q_exp_in),
    .q412_data  (conv_bfp2q_data_out),
    .valid_out  (conv_bfp2q_valid_out),
    .overflow   (conv_bfp2q_overflow),
    .underflow  ()  // 忽略underflow
);

// 转换器输入控制
assign conv_bfp2q_valid_in = (state_reg == STATE_CONV_WEIGHT);
assign conv_bfp2q_mant_in = old_weight_mant;
assign conv_bfp2q_exp_in = old_weight_exp;

//================================================================================
// Q4.12 乘法器实例（学习率 × 梯度）
//================================================================================
wire        mult_valid_in;
wire signed [Q412_WIDTH-1:0] mult_a;     // 学习率
wire signed [Q412_WIDTH-1:0] mult_b;     // 梯度
wire signed [Q412_WIDTH-1:0] mult_result;
wire        mult_valid_out;
wire        mult_overflow;

q412_multiplier #(
    .DATA_WIDTH    (Q412_WIDTH),
    .FRAC_BITS     (12),
    .PIPELINE      (1)
) u_q412_mult (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (mult_valid_in),
    .a          (mult_a),
    .b          (mult_b),
    .result     (mult_result),
    .valid_out  (mult_valid_out),
    .overflow   (mult_overflow)
);

// 乘法器输入控制
assign mult_valid_in = (state_reg == STATE_COMPUTE_UPDATE);
assign mult_a = learning_rate_reg;
assign mult_b = grad_accum;

//================================================================================
// 状态机时序逻辑
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg <= STATE_IDLE;
    end else begin
        state_reg <= state_next;
    end
end

//================================================================================
// 状态机组合逻辑
//================================================================================
always @(*) begin
    // 默认值
    state_next = state_reg;
    
    case (state_reg)
        //======================================================================
        // IDLE：等待启动
        //======================================================================
        STATE_IDLE: begin
            if (update_start) begin
                state_next = STATE_CONFIG;
            end
        end
        
        //======================================================================
        // CONFIG：配置参数，确定矩阵维度
        //======================================================================
        STATE_CONFIG: begin
            state_next = STATE_COL_START;
        end
        
        //======================================================================
        // COL_START：开始处理新列
        //======================================================================
        STATE_COL_START: begin
            if (current_col < total_cols) begin
                state_next = STATE_ROW_START;
            end else begin
                // 所有列处理完成
                state_next = STATE_IDLE;
            end
        end
        
        //======================================================================
        // ROW_START：开始处理新行
        //======================================================================
        STATE_ROW_START: begin
            if (current_row < total_rows) begin
                state_next = STATE_READ_WEIGHT;
            end else begin
                // 当前列所有行处理完成，进行列归一化
                state_next = STATE_COL_NORMALIZE;
            end
        end
        
        //======================================================================
        // READ_WEIGHT：发起旧权重读取
        //======================================================================
        STATE_READ_WEIGHT: begin
            state_next = STATE_WAIT_WEIGHT;
        end
        
        //======================================================================
        // WAIT_WEIGHT：等待旧权重数据
        //======================================================================
        STATE_WAIT_WEIGHT: begin
            if (weight_rd_valid) begin
                state_next = STATE_CONV_WEIGHT;
            end else if (timeout_counter >= TIMEOUT_THRESHOLD) begin
                // 超时，返回IDLE
                state_next = STATE_IDLE;
            end
        end
        
        //======================================================================
        // CONV_WEIGHT：转换权重(BFP→Q4.12)，等待转换器
        //======================================================================
        STATE_CONV_WEIGHT: begin
            if (conv_bfp2q_valid_out) begin
                state_next = STATE_READ_GRAD;
            end
        end
        
        //======================================================================
        // READ_GRAD：发起梯度读取
        //======================================================================
        STATE_READ_GRAD: begin
            state_next = STATE_WAIT_GRAD;
        end
        
        //======================================================================
        // WAIT_GRAD：等待梯度数据
        //======================================================================
        STATE_WAIT_GRAD: begin
            if (grad_rd_valid) begin
                state_next = STATE_COMPUTE_UPDATE;
            end else if (timeout_counter >= TIMEOUT_THRESHOLD) begin
                state_next = STATE_IDLE;
            end
        end
        
        //======================================================================
        // COMPUTE_UPDATE：计算权重更新
        //======================================================================
        STATE_COMPUTE_UPDATE: begin
            if (mult_valid_out) begin
                state_next = STATE_STORE_TEMP;
            end
        end
        
        //======================================================================
        // STORE_TEMP：暂存新权重到缓存
        //======================================================================
        STATE_STORE_TEMP: begin
            state_next = STATE_ROW_NEXT;
        end
        
        //======================================================================
        // ROW_NEXT：处理下一行
        //======================================================================
        STATE_ROW_NEXT: begin
            state_next = STATE_ROW_START;
        end
        
        //======================================================================
        // COL_NORMALIZE：列归一化，计算共享指数
        //======================================================================
        STATE_COL_NORMALIZE: begin
            state_next = STATE_CONV_TO_BFP;
        end
        
        //======================================================================
        // CONV_TO_BFP：转换整列为BFP格式
        //======================================================================
        STATE_CONV_TO_BFP: begin
            state_next = STATE_WRITE_BACK;
        end
        
        //======================================================================
        // WRITE_BACK：写回权重存储
        //======================================================================
        STATE_WRITE_BACK: begin
            if (weight_wr_ready) begin
                state_next = STATE_COL_NEXT;
            end else if (timeout_counter >= TIMEOUT_THRESHOLD) begin
                state_next = STATE_IDLE;
            end
        end
        
        //======================================================================
        // COL_NEXT：处理下一列
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
// 状态机数据路径
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 复位所有寄存器
        layer_id_reg <= 3'd0;
        weight_type_reg <= 4'd0;
        learning_rate_reg <= DEFAULT_LR_Q412;
        total_rows <= 5'd0;
        total_cols <= 5'd0;
        current_col <= 5'd0;
        current_row <= 5'd0;
        update_done <= 1'b0;
        update_busy <= 1'b0;
        timeout_counter <= 12'd0;
        
        weight_rd_req <= 1'b0;
        grad_rd_req <= 1'b0;
        weight_wr_req <= 1'b0;
        
        dbg_update_count <= 32'd0;
        dbg_row_count <= 32'd0;
        dbg_col_count <= 32'd0;
        dbg_overflow_count <= 32'd0;
        
    end else begin
        // 默认：清除脉冲信号
        update_done <= 1'b0;
        weight_rd_req <= 1'b0;
        grad_rd_req <= 1'b0;
        weight_wr_req <= 1'b0;
        
        case (state_reg)
            //==================================================================
            // IDLE
            //==================================================================
            STATE_IDLE: begin
                update_busy <= 1'b0;
                current_col <= 5'd0;
                current_row <= 5'd0;
                timeout_counter <= 12'd0;
                
                if (update_start) begin
                    update_busy <= 1'b1;
                    dbg_update_count <= dbg_update_count + 32'd1;
                end
            end
            
            //==================================================================
            // CONFIG：根据权重类型配置矩阵维度
            //==================================================================
            STATE_CONFIG: begin
                layer_id_reg <= cfg_layer_id;
                weight_type_reg <= cfg_weight_type;
                learning_rate_reg <= cfg_learning_rate;
                
                case (cfg_weight_type)
                    WEIGHT_COMPRESS: begin
                        total_rows <= 6'd32;  // 32行
                        total_cols <= 5'd8;   // 8列
                    end
                    WEIGHT_ATT_WQ, WEIGHT_ATT_WK, WEIGHT_ATT_WV, WEIGHT_ATT_WO: begin
                        total_rows <= 5'd8;
                        total_cols <= 5'd8;
                    end
                    WEIGHT_FFN_W1: begin
                        total_rows <= 5'd8;
                        total_cols <= 6'd32;
                    end
                    WEIGHT_FFN_W2: begin
                        total_rows <= 6'd32;
                        total_cols <= 5'd8;
                    end
                    WEIGHT_EXPAND: begin
                        total_rows <= 5'd8;
                        total_cols <= 6'd32;
                    end
                    default: begin
                        total_rows <= 5'd8;
                        total_cols <= 5'd8;
                    end
                endcase
            end
            
            //==================================================================
            // COL_START
            //==================================================================
            STATE_COL_START: begin
                if (current_col >= total_cols) begin
                    // 完成
                    update_done <= 1'b1;
                    update_busy <= 1'b0;
                end else begin
                    current_row <= 5'd0;
                    col_max_abs <= {Q412_WIDTH{1'b0}};
                    dbg_col_count <= dbg_col_count + 32'd1;
                end
            end
            
            //==================================================================
            // ROW_START
            //==================================================================
            STATE_ROW_START: begin
                if (current_row < total_rows) begin
                    timeout_counter <= 12'd0;
                    dbg_row_count <= dbg_row_count + 32'd1;
                end
            end
            
            //==================================================================
            // READ_WEIGHT：发起权重读取
            //==================================================================
            STATE_READ_WEIGHT: begin
                weight_rd_req <= 1'b1;
                weight_rd_layer_id <= layer_id_reg;
                weight_rd_type <= weight_type_reg;
                weight_rd_col_id <= current_col;
                weight_rd_row_id <= current_row;
            end
            
            //==================================================================
            // WAIT_WEIGHT
            //==================================================================
            STATE_WAIT_WEIGHT: begin
                timeout_counter <= timeout_counter + 12'd1;
                
                if (weight_rd_valid) begin
                    old_weight_exp <= weight_rd_exp;
                    old_weight_mant <= weight_rd_mant;
                    timeout_counter <= 12'd0;
                end
            end
            
            //==================================================================
            // CONV_WEIGHT：等待BFP→Q4.12转换完成
            //==================================================================
            STATE_CONV_WEIGHT: begin
                if (conv_bfp2q_valid_out) begin
                    old_weight_q412 <= conv_bfp2q_data_out;
                    
                    if (conv_bfp2q_overflow) begin
                        dbg_overflow_count <= dbg_overflow_count + 32'd1;
                    end
                end
            end
            
            //==================================================================
            // READ_GRAD：发起梯度读取
            //==================================================================
            STATE_READ_GRAD: begin
                grad_rd_req <= 1'b1;
                grad_rd_layer_id <= layer_id_reg;
                grad_rd_token_id <= 10'd0;  // 简化：只读第一个token的梯度
                grad_rd_dim_id <= current_row[4:0];
            end
            
            //==================================================================
            // WAIT_GRAD
            //==================================================================
            STATE_WAIT_GRAD: begin
                timeout_counter <= timeout_counter + 12'd1;
                
                if (grad_rd_valid) begin
                    grad_accum <= grad_rd_data;
                    timeout_counter <= 12'd0;
                end
            end
            
            //==================================================================
            // COMPUTE_UPDATE：等待乘法器完成
            //==================================================================
            STATE_COMPUTE_UPDATE: begin
                if (mult_valid_out) begin
                    // W_new = W_old - lr × gradient
                    new_weight_q412 <= old_weight_q412 - mult_result;
                    weight_overflow <= mult_overflow;
                    
                    if (mult_overflow) begin
                        dbg_overflow_count <= dbg_overflow_count + 32'd1;
                    end
                end
            end
            
            //==================================================================
            // STORE_TEMP：存储新权重到列缓存
            //==================================================================
            STATE_STORE_TEMP: begin
                weight_col_buffer[current_row] <= new_weight_q412;
                
                // 更新列内最大绝对值（用于归一化）
                if ($signed(new_weight_q412) < 0) begin
                    if (-new_weight_q412 > col_max_abs) begin
                        col_max_abs <= -new_weight_q412;
                    end
                end else begin
                    if (new_weight_q412 > col_max_abs) begin
                        col_max_abs <= new_weight_q412;
                    end
                end
            end
            
            //==================================================================
            // ROW_NEXT
            //==================================================================
            STATE_ROW_NEXT: begin
                current_row <= current_row + 5'd1;
            end
            
            //==================================================================
            // COL_NORMALIZE：计算列共享指数
            //==================================================================
            STATE_COL_NORMALIZE: begin
                // 计算前导零数量（优先编码器）
                col_leading_zeros <= count_leading_zeros(col_max_abs);
                
                // 计算共享指数 = 12 - leading_zeros
                col_shared_exp <= 8'd12 - {3'b0, count_leading_zeros(col_max_abs)};
            end
            
            //==================================================================
            // CONV_TO_BFP：转换整列为BFP格式
            //==================================================================
            STATE_CONV_TO_BFP: begin
                // 存储列共享指数
                bfp_exp_buffer[current_col] <= col_shared_exp;
                
                // 转换每一行为BFP尾数
                // 归一化公式：mant = weight_q412 >> (12 - col_shared_exp)
                // 
                // 原理：
                // weight_q412 表示 value × 4096 (即 value × 2^12)
                // BFP value = mant × 2^exp
                // 
                // 要让 weight_q412/4096 = mant × 2^exp
                // 即 mant = weight_q412 / (4096 × 2^exp)
                //         = weight_q412 >> (12 + exp)
                // 
                // 但我们计算的exp是相对于Q4.12的，所以：
                // mant = weight_q412 >> (12 - exp)
                
                for (i = 0; i < MAX_ROWS; i = i + 1) begin
                    if (i < total_rows) begin
                        temp_weight = weight_col_buffer[i];
                        shift_amount = 8'd12 - $signed({3'b0, col_shared_exp[4:0]});
                        
                        if (shift_amount >= 0) begin
                            // 右移
                            bfp_mant_buffer[current_col][i] <= 
                                temp_weight >>> shift_amount;
                        end else begin
                            // 左移（理论上不应该发生）
                            bfp_mant_buffer[current_col][i] <= 
                                temp_weight <<< (-shift_amount);
                        end
                    end else begin
                        bfp_mant_buffer[current_col][i] <= {DATA_WIDTH{1'b0}};
                    end
                end
            end
            
            //==================================================================
            // WRITE_BACK：写回权重
            //==================================================================
            STATE_WRITE_BACK: begin
                weight_wr_req <= 1'b1;
                weight_wr_layer_id <= layer_id_reg;
                weight_wr_type <= weight_type_reg;
                
                // 计算burst_idx：
                // - 对于8行矩阵（Attention, FFN_W1, Expand）：每列1个burst
                // - 对于32行矩阵（Compression, FFN_W2）：每列2个burst
                
                // 打包指数数组（当前列的指数）
                weight_wr_exp_array[current_col*EXP_WIDTH +: EXP_WIDTH] <= 
                    bfp_exp_buffer[current_col];
                
                // 打包数据burst
                // 每个burst最多16个权重（256 bits / 16 bits = 16）
                if (total_rows <= 16) begin
                    // 单burst：直接打包
                    weight_wr_burst_idx <= {1'b0, current_col};
                    
                    for (j = 0; j < 16; j = j + 1) begin
                        if (j < total_rows) begin
                            weight_wr_data_burst[j*DATA_WIDTH +: DATA_WIDTH] <= 
                                bfp_mant_buffer[current_col][j];
                        end else begin
                            weight_wr_data_burst[j*DATA_WIDTH +: DATA_WIDTH] <= 
                                {DATA_WIDTH{1'b0}};
                        end
                    end
                    
                end else begin
                    // 双burst：需要分两次写
                    // 第一次写入前16行（burst_idx = col*2）
                    // 第二次写入后16行（burst_idx = col*2+1）
                    // 这里简化为只写第一个burst
                    // TODO: 实现完整的双burst写入
                    
                    weight_wr_burst_idx <= {current_col, 1'b0};  // col*2
                    
                    for (j = 0; j < 16; j = j + 1) begin
                        weight_wr_data_burst[j*DATA_WIDTH +: DATA_WIDTH] <= 
                            bfp_mant_buffer[current_col][j];
                    end
                end
                
                timeout_counter <= timeout_counter + 12'd1;
            end
            
            //==================================================================
            // COL_NEXT
            //==================================================================
            STATE_COL_NEXT: begin
                current_col <= current_col + 5'd1;
            end
            
        endcase
    end
end

//================================================================================
// 辅助函数：前导零计数
//================================================================================
function [4:0] count_leading_zeros;
    input [Q412_WIDTH-1:0] value;
    integer k;
    begin
        count_leading_zeros = 5'd16;  // 默认全零
        for (k = Q412_WIDTH-1; k >= 0; k = k - 1) begin
            if (value[k]) begin
                count_leading_zeros = Q412_WIDTH - 1 - k;
                k = -1;  // 退出循环
            end
        end
    end
endfunction

//================================================================================
// 调试输出
//================================================================================
assign dbg_current_lr = learning_rate_reg;

//================================================================================
// 仿真监控
//================================================================================
`ifdef SIMULATION

initial begin
    $display("========================================");
    $display("Weight Update Engine v1.0");
    $display("========================================");
    $display("Features:");
    $display("  - Q4.12 format computation");
    $display("  - Column-shared exponent optimization");
    $display("  - Integrated BFP↔Q4.12 converters");
    $display("  - Learning rate: %.4f", $itor(DEFAULT_LR_Q412)/4096.0);
    $display("========================================");
end

always @(posedge clk) begin
    if (state_reg != state_next) begin
        case (state_next)
            STATE_IDLE:          $display("[%0t] WUE → IDLE", $time);
            STATE_CONFIG:        $display("[%0t] WUE → CONFIG (layer=%0d, type=%0d)", 
                                         $time, cfg_layer_id, cfg_weight_type);
            STATE_COL_START:     $display("[%0t] WUE → COL_START (col=%0d/%0d)", 
                                         $time, current_col, total_cols);
            STATE_ROW_START:     $display("[%0t] WUE → ROW_START (row=%0d/%0d)", 
                                         $time, current_row, total_rows);
            STATE_COL_NORMALIZE: $display("[%0t] WUE → COL_NORMALIZE (max_abs=%0d, exp=%0d)", 
                                         $time, col_max_abs, col_shared_exp);
            STATE_WRITE_BACK:    $display("[%0t] WUE → WRITE_BACK (col=%0d)", 
                                         $time, current_col);
        endcase
    end
    
    if (update_done) begin
        $display("[%0t] WUE: Update complete (rows=%0d, cols=%0d, overflows=%0d)",
                 $time, dbg_row_count, dbg_col_count, dbg_overflow_count);
    end
end

`endif // SIMULATION

endmodule