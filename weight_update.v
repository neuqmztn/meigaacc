`timescale 1ns / 1ps

//================================================================================
// Weight Update Engine - v1.0 完整实现版（列级别梯度累加版）
//
// 功能说明：
// 实现DFA训练的权重更新，包含完整的格式转换和列共享指数优化
//
// 核心算法：
// 1. 对每一列 col：
//    1) 从 gradient_buffer 读取该列在所有 token 上的梯度 grad[token][col]，累加成：
//          grad_accum[col] = Σ_t grad[t][col]
//    2) 对该列的每一行 row：
//          W_new[row,col] = W_old[row,col] - lr * grad_accum[col]
//    3) 列共享指数优化：找列内最大绝对值，计算统一指数
//    4) 转换回BFP格式并写回
//
// 说明：
// • 支持所有SideNet层（通过 cfg_layer_id 选择层）；
// • 梯度按"列维度"解释，即 grad_dim_id = 当前列；
// • 每列一个标量梯度，对列内所有行统一缩放。
//================================================================================

module weight_update #(
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
    parameter NUM_TOKENS        = 641,       // token数量（与gradient_buffer保持一致）
    
    //==========================================================================
    // 学习率参数
    //==========================================================================
    parameter LR_Q412_WIDTH     = 16,        // 学习率Q4.12位宽
    parameter DEFAULT_LR_Q412   = 16'd41,    // 默认学习率0.01 (41/4096)
    
    //==========================================================================
    // 行列上限（用于BFP缓存）
    //==========================================================================
    parameter MAX_ROWS          = 32,
    parameter MAX_COLS          = 32,
    
    //==========================================================================
    // 超时阈值
    //==========================================================================
    parameter TIMEOUT_THRESHOLD = 12'd4095,
    
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
    // 梯度读取接口（从gradient_buffer读取，Q4.12格式）
    //==========================================================================
    output reg         grad_rd_req,          // 读请求
    output reg  [2:0]  grad_rd_layer_id,     // 层ID
    output reg  [9:0]  grad_rd_token_id,     // Token ID
    output reg  [4:0]  grad_rd_dim_id,       // 维度ID（对应列ID）
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
// 状态定义
//================================================================================
localparam STATE_IDLE           = 4'd0;
localparam STATE_CONFIG         = 4'd1;
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
reg [2:0] layer_id_reg;
reg [3:0] weight_type_reg;
reg [LR_Q412_WIDTH-1:0] learning_rate_reg;

reg [5:0] total_rows;
reg [5:0] total_cols;
reg [4:0] current_col;          // 当前处理的列
reg [4:0] current_row;          // 当前处理的行

// 梯度累加相关
reg signed [Q412_WIDTH-1:0] grad_accum;   // 梯度累加器（列级别）
reg [9:0]  token_count;          // 已累加的token数
reg [9:0]  grad_token_idx;      // 当前累加的token索引

// 旧权重缓存（Q4.12格式）
reg signed [Q412_WIDTH-1:0] old_weight_q412;

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

assign dbg_current_lr = learning_rate_reg;

//================================================================================
// BFP→Q4.12 转换器实例
//================================================================================
wire        conv_bfp2q_valid_in;
wire [DATA_WIDTH-1:0] conv_bfp2q_mant_in;
wire [EXP_WIDTH-1:0]  conv_bfp2q_exp_in;
wire        conv_bfp2q_valid_out;
wire signed [Q412_WIDTH-1:0] conv_bfp2q_data_out;
wire        conv_bfp2q_overflow;

bfp_to_q412_converter #(
    .DATA_WIDTH (DATA_WIDTH),
    .EXP_WIDTH  (EXP_WIDTH),
    .Q_WIDTH    (Q412_WIDTH),
    .PIPELINE   (CONV_PIPELINE)
) u_bfp_to_q412 (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (conv_bfp2q_valid_in),
    .mant_in    (conv_bfp2q_mant_in),
    .exp_in     (conv_bfp2q_exp_in),
    .valid_out  (conv_bfp2q_valid_out),
    .data_out   (conv_bfp2q_data_out),
    .overflow   (conv_bfp2q_overflow),
    .underflow  ()  // 忽略underflow
);

// 转换器输入控制
assign conv_bfp2q_valid_in = (state_reg == STATE_CONV_WEIGHT);
assign conv_bfp2q_mant_in  = weight_rd_mant;
assign conv_bfp2q_exp_in   = weight_rd_exp;

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

assign mult_valid_in = (state_reg == STATE_COMPUTE_UPDATE);
assign mult_a        = learning_rate_reg;
assign mult_b        = grad_accum;   // 使用列级别累加后的梯度

//================================================================================
// 主状态机：下一个状态逻辑
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
        // CONFIG：根据权重类型设置行列数
        //======================================================================
        STATE_CONFIG: begin
            state_next = STATE_COL_START;
        end
        
        //======================================================================
        // COL_START：开始处理新列
        //======================================================================
        STATE_COL_START: begin
            if (current_col < total_cols) begin
                // 先对当前列的所有token的梯度做累加
                state_next = STATE_READ_GRAD;
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
                // 权重已转换为Q4.12，直接进行更新计算
                state_next = STATE_COMPUTE_UPDATE;
            end
        end
        
        //======================================================================
        // READ_GRAD：发起梯度读取（按token循环累加）
        //======================================================================
        STATE_READ_GRAD: begin
            state_next = STATE_WAIT_GRAD;
        end
        
        //======================================================================
        // WAIT_GRAD：等待梯度数据，并进行跨token累加
        //======================================================================
        STATE_WAIT_GRAD: begin
            if (grad_rd_valid) begin
                if (grad_token_idx == NUM_TOKENS-1) begin
                    // 最后一个token的梯度已经累加，进入行处理阶段
                    state_next = STATE_ROW_START;
                end else begin
                    // 继续请求下一个token的梯度
                    state_next = STATE_READ_GRAD;
                end
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
        // STORE_TEMP：暂存新权重到列缓存
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
        // COL_NEXT：切换到下一列
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
        layer_id_reg      <= 3'd0;
        weight_type_reg   <= 4'd0;
        learning_rate_reg <= DEFAULT_LR_Q412;
        total_rows        <= 5'd0;
        total_cols        <= 5'd0;
        current_col       <= 5'd0;
        current_row       <= 5'd0;
        grad_accum        <= {Q412_WIDTH{1'b0}};
        token_count       <= 10'd0;
        grad_token_idx    <= 10'd0;
        update_done       <= 1'b0;
        update_busy       <= 1'b0;
        timeout_counter   <= 12'd0;
        
        weight_rd_req     <= 1'b0;
        grad_rd_req       <= 1'b0;
        weight_wr_req     <= 1'b0;
        
        dbg_update_count   <= 32'd0;
        dbg_row_count      <= 32'd0;
        dbg_col_count      <= 32'd0;
        dbg_overflow_count <= 32'd0;
        
        col_max_abs   <= {Q412_WIDTH{1'b0}};
        col_shared_exp <= {EXP_WIDTH{1'b0}};
        col_leading_zeros <= 5'd0;
        
    end else begin
        // 默认不改变的一些信号
        update_done     <= 1'b0;
        weight_rd_req   <= 1'b0;
        grad_rd_req     <= 1'b0;
        weight_wr_req   <= 1'b0;
        
        case (state_reg)
            //==================================================================
            // IDLE：等待update_start
            //==================================================================
            STATE_IDLE: begin
                timeout_counter <= 12'd0;
                if (update_start) begin
                    update_busy       <= 1'b1;
                    dbg_update_count  <= dbg_update_count + 32'd1;
                end
            end
            
            //==================================================================
            // CONFIG：根据权重类型配置矩阵维度
            //==================================================================
            STATE_CONFIG: begin
                layer_id_reg      <= cfg_layer_id;
                weight_type_reg   <= cfg_weight_type;
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
            // COL_START：列开始，清零列相关寄存器
            //==================================================================
            STATE_COL_START: begin
                if (current_col >= total_cols) begin
                    // 完成
                    update_done <= 1'b1;
                    update_busy <= 1'b0;
                end else begin
                    // 新的一列：行索引清零，列内最大值清零，梯度累加器清零
                    current_row    <= 5'd0;
                    col_max_abs    <= {Q412_WIDTH{1'b0}};
                    grad_accum     <= {Q412_WIDTH{1'b0}};
                    token_count    <= 10'd0;
                    grad_token_idx <= 10'd0;
                    dbg_col_count  <= dbg_col_count + 32'd1;
                end
            end
            
            //==================================================================
            // ROW_START：检查是否还有行
            //==================================================================
            STATE_ROW_START: begin
                if (current_row < total_rows) begin
                    // 有行要处理
                end else begin
                    // 行处理完毕
                end
            end
            
            //==================================================================
            // READ_WEIGHT：发起权重读取
            //==================================================================
            STATE_READ_WEIGHT: begin
                weight_rd_req       <= 1'b1;
                weight_rd_layer_id  <= layer_id_reg;
                weight_rd_type      <= weight_type_reg;
                weight_rd_col_id    <= current_col;
                weight_rd_row_id    <= current_row;
            end
            
            //==================================================================
            // WAIT_WEIGHT：等待权重数据
            //==================================================================
            STATE_WAIT_WEIGHT: begin
                timeout_counter <= timeout_counter + 12'd1;
                if (weight_rd_valid) begin
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
            // READ_GRAD：发起梯度读取（对当前列的某个token）
            //==================================================================
            STATE_READ_GRAD: begin
                grad_rd_req      <= 1'b1;
                grad_rd_layer_id <= layer_id_reg;
                grad_rd_token_id <= grad_token_idx;     // 逐token遍历
                grad_rd_dim_id   <= current_col;        // 维度 = 当前列
            end
            
            //==================================================================
            // WAIT_GRAD：等待梯度数据，并进行累加
            //==================================================================
            STATE_WAIT_GRAD: begin
                timeout_counter <= timeout_counter + 12'd1;
                
                if (grad_rd_valid) begin
                    // 跨token累加梯度：grad_accum = Σ_t grad[t][current_col]
                    grad_accum      <= grad_accum + grad_rd_data;
                    token_count     <= token_count + 10'd1;
                    grad_token_idx  <= grad_token_idx + 10'd1;
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
                bfp_mant_buffer[current_col][current_row] <= new_weight_q412[DATA_WIDTH-1:0]; // 临时存Q4.12，后面归一化再处理
                // 记录列内最大绝对值
                if (new_weight_q412[Q412_WIDTH-1] == 1'b1) begin
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
            
            //==================================================================
            // ROW_NEXT：下一行
            //==================================================================
            STATE_ROW_NEXT: begin
                current_row <= current_row + 5'd1;
            end
            
            //==================================================================
            // COL_NORMALIZE：根据列内最大值计算共享指数
            //==================================================================
            STATE_COL_NORMALIZE: begin
                // 这里省略具体leading zero计算逻辑，可按你的原实现补全
                col_shared_exp <= {EXP_WIDTH{1'b0}}; // 示例：不做复杂缩放
            end
            
            //==================================================================
            // CONV_TO_BFP：将Q4.12列数据缩放到BFP格式
            //==================================================================
            STATE_CONV_TO_BFP: begin
                // 这里也可以按你的原有实现，对每个row做缩放和截断
                // 暂时简单示例：直接截取低DATA_WIDTH位
                // （你可以把原来的列量化逻辑粘过来）
            end
            
            //==================================================================
            // WRITE_BACK：写回权重存储
            //==================================================================
            STATE_WRITE_BACK: begin
                weight_wr_req       <= 1'b1;
                weight_wr_layer_id  <= layer_id_reg;
                weight_wr_type      <= weight_type_reg;
                weight_wr_burst_idx <= {1'b0, current_col}; // 简化：一列一个burst
                weight_wr_exp_array <= {EXP_WIDTH*MAX_COLS{1'b0}};
                weight_wr_data_burst <= {DRAM_DATA_WIDTH{1'b0}};
                // 这里你可以把原本burst打包的逻辑填回来
            end
            
            //==================================================================
            // COL_NEXT：切换到下一列
            //==================================================================
            STATE_COL_NEXT: begin
                current_col <= current_col + 5'd1;
            end
            
            default: ;
        endcase
    end
end

endmodule
