`timescale 1ns / 1ps

//================================================================================
// Weight Update Engine - v1.0 完整实现版（列级别梯度累加 + 行主序写回）
//
// 功能：
// 1. 对每一列 col：
//    (1) 从 gradient_buffer 读取该列在所有 token 上的梯度 grad[token][col]，累加：
//            grad_accum[col] = Σ_t grad[t][col]
//    (2) 对该列所有行 row：
//            W_new[row,col] = W_old[row,col] - lr * grad_accum[col]
//    (3) 将新权重暂存到列主序 2D buffer：bfp_mant_buffer[col][row]
// 2. 所有列处理完成后：
//    (1) 将 bfp_mant_buffer[col][row] 打平成列主序一维向量 mant_col_major_flat
//    (2) 通过 col2row_reorder 转成行主序 mant_row_major_flat
//    (3) 先写 exponent burst（burst_idx=0），再按行写回 mant（burst_idx=1..total_rows）
//================================================================================

module weight_update #(
    //======================================================================
    // 基本参数
    //======================================================================
    parameter NUM_LAYERS        = 5,
    parameter BACKBONE_DIM      = 32,
    parameter SIDENET_DIM       = 8,
    parameter D_FF              = 32,
    parameter DATA_WIDTH        = 16,        // BFP尾数位宽（当前暂用Q4.12低DATA_WIDTH位）
    parameter EXP_WIDTH         = 8,         // BFP指数位宽
    parameter Q412_WIDTH        = 16,        // Q4.12位宽
    parameter DRAM_DATA_WIDTH   = 256,       // 单次 burst 宽度
    parameter NUM_TOKENS        = 641,       // token数量（与gradient_buffer保持一致）
    
    //======================================================================
    // 学习率参数
    //======================================================================
    parameter LR_Q412_WIDTH     = 16,        // 学习率Q4.12位宽
    parameter DEFAULT_LR_Q412   = 16'd41,    // 默认学习率0.01 (41/4096)
    
    //======================================================================
    // 行列上限（用于BFP缓存）
    //======================================================================
    parameter MAX_ROWS          = 32,
    parameter MAX_COLS          = 32,
    
    //======================================================================
    // 超时阈值
    //======================================================================
    parameter TIMEOUT_THRESHOLD = 12'd4095,
    
    //======================================================================
    // 转换器参数
    //======================================================================
    parameter CONV_PIPELINE     = 3          // BFP→Q4.12流水线级数
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
    input  wire [2:0]  cfg_layer_id,         // 要更新的层ID (0-4)
    input  wire [3:0]  cfg_weight_type,      // 权重类型
    input  wire [15:0] cfg_learning_rate,    // 学习率(Q4.12格式)
    
    //======================================================================
    // 旧权重读取接口（从权重存储读取，BFP格式）
    //======================================================================
    output reg         weight_rd_req,        // 读请求
    output reg  [2:0]  weight_rd_layer_id,   // 层ID
    output reg  [3:0]  weight_rd_type,       // 权重类型
    output reg  [4:0]  weight_rd_col_id,     // 列ID
    output reg  [4:0]  weight_rd_row_id,     // 行ID（单行读取）
    input  wire        weight_rd_valid,      // 读数据有效
    input  wire [EXP_WIDTH-1:0]  weight_rd_exp,   // BFP指数
    input  wire [DATA_WIDTH-1:0] weight_rd_mant,  // BFP尾数
    
    //======================================================================
    // 梯度读取接口（从gradient_buffer读取，Q4.12格式）
    //======================================================================
    output reg         grad_rd_req,          // 读请求
    output reg  [2:0]  grad_rd_layer_id,     // 层ID
    output reg  [9:0]  grad_rd_token_id,     // Token ID
    output reg  [4:0]  grad_rd_dim_id,       // 维度ID（对应列ID）
    input  wire        grad_rd_valid,        // 读数据有效
    input  wire [Q412_WIDTH-1:0] grad_rd_data,    // 梯度数据(Q4.12)
    
    //======================================================================
    // 新权重写入接口（写入权重存储，BFP格式）
    //======================================================================
    output reg         weight_wr_req,        // 写请求
    output reg  [2:0]  weight_wr_layer_id,   // 层ID
    output reg  [3:0]  weight_wr_type,       // 权重类型
    output reg  [5:0]  weight_wr_burst_idx,  // Burst索引
    output reg  [EXP_WIDTH*MAX_COLS-1:0] weight_wr_exp_array,  // 指数数组
    output reg  [DRAM_DATA_WIDTH-1:0]   weight_wr_data_burst,  // 数据burst
    input  wire        weight_wr_ready,      // 写就绪
    
    //======================================================================
    // 调试接口
    //======================================================================
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
localparam STATE_COL_NORMALIZE  = 4'd12;  // 矩阵级归一化 / 打平
localparam STATE_CONV_TO_BFP    = 4'd13;  // 转换为BFP / 写回准备
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
reg signed [Q412_WIDTH-1:0] grad_accum;      // 梯度累加器（列级别）
reg [9:0]  token_count;                      // 已累加的token数
reg [9:0]  grad_token_idx;                   // 当前累加的token索引

// 旧权重缓存（Q4.12格式）
reg signed [Q412_WIDTH-1:0] old_weight_q412;

// 新权重计算
reg signed [Q412_WIDTH-1:0] new_weight_q412;
reg        weight_overflow;

// 列归一化（找最大值，计算共享指数：当前只统计，不真正用来缩放）
reg signed [Q412_WIDTH-1:0] col_max_abs;    // 列内最大绝对值
reg [EXP_WIDTH-1:0]  col_shared_exp;        // 列共享指数（当前未使用）
reg [4:0]  col_leading_zeros;               // 前导零数量（当前未使用）

// BFP输出缓存（二维 buffer：列主序）
reg [EXP_WIDTH-1:0]        bfp_exp_buffer [0:MAX_COLS-1];
reg [DATA_WIDTH-1:0]       bfp_mant_buffer[0:MAX_COLS-1][0:MAX_ROWS-1];

// 超时计数器
reg [11:0] timeout_counter;

assign dbg_current_lr = learning_rate_reg;

//================================================================================
// 列主序 → 行主序重排 + 写回控制（矩阵级）
//================================================================================
// 为了写回整块矩阵（行主序），先把列主序 2D buffer 打平成 1D 向量
reg  [MAX_COLS*MAX_ROWS*DATA_WIDTH-1:0] mant_col_major_flat;
wire [MAX_ROWS*MAX_COLS*DATA_WIDTH-1:0] mant_row_major_flat;

// 打包指数和尾数 burst 的临时寄存器
reg [EXP_WIDTH*MAX_COLS-1:0] packed_exp_array;
reg [DRAM_DATA_WIDTH-1:0]    packed_row_burst;

// 写回阶段的子状态：0=写exp，1=写行
reg        writeback_active;     // 0：还没开始矩阵写回；1：正在写回矩阵
reg        write_stage;          // 0：写 exponent burst；1：写每一行
reg [5:0]  write_row_idx;        // 当前要写回的 row（0..total_rows-1）

integer r_idx_flat;
integer c_idx_flat;

//================================================================================
// 列主序 → 行主序 重排实例
//================================================================================
col2row_reorder #(
    .ROWS      (MAX_ROWS),
    .COLS      (MAX_COLS),
    .DATA_WIDTH(DATA_WIDTH)
) u_col2row_reorder (
    .col_major_in (mant_col_major_flat),
    .row_major_out(mant_row_major_flat)
);

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
// 状态机：状态寄存器
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg <= STATE_IDLE;
    end else begin
        state_reg <= state_next;
    end
end

//================================================================================
// 主状态机：下一个状态逻辑（组合）
//================================================================================
always @(*) begin
    // 默认保持当前状态
    state_next = state_reg;
    
    case (state_reg)
        //------------------------------------------------------------------
        // IDLE：等待启动
        //------------------------------------------------------------------
        STATE_IDLE: begin
            if (update_start) begin
                state_next = STATE_CONFIG;
            end
        end
        
        //------------------------------------------------------------------
        // CONFIG：根据权重类型设置行列数
        //------------------------------------------------------------------
        STATE_CONFIG: begin
            state_next = STATE_COL_START;
        end
        
        //------------------------------------------------------------------
        // COL_START：开始处理新列 / 或判定是否进入矩阵写回
        //------------------------------------------------------------------
        STATE_COL_START: begin
            if (current_col < total_cols) begin
                // 还有列需要更新 → 进入梯度累加阶段
                state_next = STATE_READ_GRAD;
            end else begin
                // 所有列的 new_weight 已经写入 bfp_mant_buffer[col][row]
                // 接下来统一做矩阵级的 BFP 准备 + 写回
                state_next = STATE_COL_NORMALIZE;  
            end
        end
                
        //------------------------------------------------------------------
        // ROW_START：开始处理新行
        //------------------------------------------------------------------
        STATE_ROW_START: begin
            if (current_row < total_rows) begin
                state_next = STATE_READ_WEIGHT;
            end else begin
                // 当前列所有行处理完成 → 直接切到下一列
                state_next = STATE_COL_NEXT;
            end
        end
        
        //------------------------------------------------------------------
        // READ_WEIGHT：发起旧权重读取
        //------------------------------------------------------------------
        STATE_READ_WEIGHT: begin
            state_next = STATE_WAIT_WEIGHT;
        end
        
        //------------------------------------------------------------------
        // WAIT_WEIGHT：等待旧权重数据
        //------------------------------------------------------------------
        STATE_WAIT_WEIGHT: begin
            if (weight_rd_valid) begin
                state_next = STATE_CONV_WEIGHT;
            end else if (timeout_counter >= TIMEOUT_THRESHOLD) begin
                state_next = STATE_IDLE;
            end
        end
        
        //------------------------------------------------------------------
        // CONV_WEIGHT：转换权重(BFP→Q4.12)，等待转换器
        //------------------------------------------------------------------
        STATE_CONV_WEIGHT: begin
            if (conv_bfp2q_valid_out) begin
                state_next = STATE_COMPUTE_UPDATE;
            end
        end
        
        //------------------------------------------------------------------
        // READ_GRAD：发起梯度读取（按token循环累加）
        //------------------------------------------------------------------
        STATE_READ_GRAD: begin
            state_next = STATE_WAIT_GRAD;
        end
        
        //------------------------------------------------------------------
        // WAIT_GRAD：等待梯度数据，并进行跨token累加
        //------------------------------------------------------------------
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
        
        //------------------------------------------------------------------
        // COMPUTE_UPDATE：计算权重更新
        //------------------------------------------------------------------
        STATE_COMPUTE_UPDATE: begin
            if (mult_valid_out) begin
                state_next = STATE_STORE_TEMP;
            end
        end
        
        //------------------------------------------------------------------
        // STORE_TEMP：暂存新权重到列缓存
        //------------------------------------------------------------------
        STATE_STORE_TEMP: begin
            state_next = STATE_ROW_NEXT;
        end
        
        //------------------------------------------------------------------
        // ROW_NEXT：处理下一行
        //------------------------------------------------------------------
        STATE_ROW_NEXT: begin
            state_next = STATE_ROW_START;
        end
        
        //------------------------------------------------------------------
        // COL_NORMALIZE：矩阵级归一化 / 打平，完成后进入 CONV_TO_BFP
        //------------------------------------------------------------------
        STATE_COL_NORMALIZE: begin
            state_next = STATE_CONV_TO_BFP;
        end
        
        //------------------------------------------------------------------
        // CONV_TO_BFP：转换为BFP / 写回准备
        //------------------------------------------------------------------
        STATE_CONV_TO_BFP: begin
            if (writeback_active) begin
                state_next = STATE_WRITE_BACK;
            end else begin
                state_next = STATE_IDLE;
            end
        end
        
        //------------------------------------------------------------------
        // WRITE_BACK：写回权重存储（矩阵级）
        //------------------------------------------------------------------
        STATE_WRITE_BACK: begin
            if (!writeback_active) begin
                // 写回结束，回到空闲
                state_next = STATE_IDLE;
            end else begin
                // 写回进行中，大状态机停在 WRITE_BACK，由内部子状态驱动行写回
                state_next = STATE_WRITE_BACK;
            end
        end
        
        //------------------------------------------------------------------
        // COL_NEXT：切换到下一列（列循环阶段）
        //------------------------------------------------------------------
        STATE_COL_NEXT: begin
            state_next = STATE_COL_START;
        end
        
        default: begin
            state_next = STATE_IDLE;
        end
    endcase
end

//================================================================================
// 状态机数据路径（时序）
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 复位所有寄存器
        layer_id_reg      <= 3'd0;
        weight_type_reg   <= 4'd0;
        learning_rate_reg <= DEFAULT_LR_Q412;
        total_rows        <= 6'd0;
        total_cols        <= 6'd0;
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
        
        col_max_abs        <= {Q412_WIDTH{1'b0}};
        col_shared_exp     <= {EXP_WIDTH{1'b0}};
        col_leading_zeros  <= 5'd0;

        writeback_active   <= 1'b0;
        write_stage        <= 1'b0;
        write_row_idx      <= 6'd0;
        mant_col_major_flat<= {MAX_COLS*MAX_ROWS*DATA_WIDTH{1'b0}};
        packed_exp_array   <= {EXP_WIDTH*MAX_COLS{1'b0}};
        packed_row_burst   <= {DRAM_DATA_WIDTH{1'b0}};
    end else begin
        // 默认：单拍型信号清零
        update_done     <= 1'b0;
        weight_rd_req   <= 1'b0;
        grad_rd_req     <= 1'b0;
        weight_wr_req   <= 1'b0;

        case (state_reg)
            //------------------------------------------------------------------
            // IDLE：等待update_start
            //------------------------------------------------------------------
            STATE_IDLE: begin
                timeout_counter <= 12'd0;
                current_col     <= 5'd0;
                current_row     <= 5'd0;
                grad_token_idx  <= 10'd0;
                token_count     <= 10'd0;
                grad_accum      <= {Q412_WIDTH{1'b0}};
                writeback_active<= 1'b0;
                write_stage     <= 1'b0;
                write_row_idx   <= 6'd0;

                if (update_start) begin
                    update_busy      <= 1'b1;
                    dbg_update_count <= dbg_update_count + 32'd1;
                end
            end
            
            //------------------------------------------------------------------
            // CONFIG：根据权重类型配置矩阵维度
            //------------------------------------------------------------------
            STATE_CONFIG: begin
                layer_id_reg      <= cfg_layer_id;
                weight_type_reg   <= cfg_weight_type;
                learning_rate_reg <= cfg_learning_rate;
                
                case (cfg_weight_type)
                    WEIGHT_COMPRESS: begin
                        total_rows <= 6'd32;  // 32行
                        total_cols <= 6'd8;   // 8列
                    end
                    WEIGHT_ATT_WQ,
                    WEIGHT_ATT_WK,
                    WEIGHT_ATT_WV,
                    WEIGHT_ATT_WO: begin
                        total_rows <= 6'd8;
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
            // COL_START：列开始 / 列循环结束判定
            //------------------------------------------------------------------
            STATE_COL_START: begin
                if (current_col >= total_cols) begin
                    // 列循环已经全部完成，准备进入矩阵写回阶段
                    if (!writeback_active) begin
                        writeback_active <= 1'b1;
                        write_stage      <= 1'b0;  // 先写 exponent burst
                        write_row_idx    <= 6'd0;
                    end
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
            
            //------------------------------------------------------------------
            // ROW_START：检查是否还有行
            //------------------------------------------------------------------
            STATE_ROW_START: begin
                // 这里只是占位，真正跳转由组合状态机控制
            end
            
            //------------------------------------------------------------------
            // READ_WEIGHT：发起权重读取
            //------------------------------------------------------------------
            STATE_READ_WEIGHT: begin
                weight_rd_req       <= 1'b1;
                weight_rd_layer_id  <= layer_id_reg;
                weight_rd_type      <= weight_type_reg;
                weight_rd_col_id    <= current_col;
                weight_rd_row_id    <= current_row;
            end
            
            //------------------------------------------------------------------
            // WAIT_WEIGHT：等待权重数据
            //------------------------------------------------------------------
            STATE_WAIT_WEIGHT: begin
                timeout_counter <= timeout_counter + 12'd1;
                if (weight_rd_valid) begin
                    timeout_counter <= 12'd0;
                end
            end
            
            //------------------------------------------------------------------
            // CONV_WEIGHT：等待BFP→Q4.12转换完成
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
            // READ_GRAD：发起梯度读取（对当前列的某个token）
            //------------------------------------------------------------------
            STATE_READ_GRAD: begin
                grad_rd_req      <= 1'b1;
                grad_rd_layer_id <= layer_id_reg;
                grad_rd_token_id <= grad_token_idx;     // 逐token遍历
                grad_rd_dim_id   <= current_col;        // 维度 = 当前列
            end
            
            //------------------------------------------------------------------
            // WAIT_GRAD：等待梯度数据，并进行累加
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
            // COMPUTE_UPDATE：等待乘法器完成
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
            // STORE_TEMP：存储新权重到列缓存（列主序）
            //------------------------------------------------------------------
            STATE_STORE_TEMP: begin
                bfp_mant_buffer[current_col][current_row] <= new_weight_q412[DATA_WIDTH-1:0];
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
            
            //------------------------------------------------------------------
            // ROW_NEXT：下一行
            //------------------------------------------------------------------
            STATE_ROW_NEXT: begin
                current_row <= current_row + 5'd1;
            end
            
            //------------------------------------------------------------------
            // COL_NORMALIZE：矩阵级归一化 / 打平
            //------------------------------------------------------------------
            STATE_COL_NORMALIZE: begin
                if (writeback_active) begin
                    // 1) expo：当前先简化为全零或常量，后续可接入真正的 col_max_abs → col_shared_exp 逻辑
                    for (c_idx_flat = 0; c_idx_flat < MAX_COLS; c_idx_flat = c_idx_flat + 1) begin
                        bfp_exp_buffer[c_idx_flat] <= {EXP_WIDTH{1'b0}};  // 占位：所有列指数 = 0
                    end
            
                    // 2) mant：把列主序 2D buffer 打平为 col_major_flat
                    for (c_idx_flat = 0; c_idx_flat < MAX_COLS; c_idx_flat = c_idx_flat + 1) begin
                        for (r_idx_flat = 0; r_idx_flat < MAX_ROWS; r_idx_flat = r_idx_flat + 1) begin
                            mant_col_major_flat[(c_idx_flat*MAX_ROWS + r_idx_flat)*DATA_WIDTH +: DATA_WIDTH]
                                <= bfp_mant_buffer[c_idx_flat][r_idx_flat];
                        end
                    end
                end
            end
            
            //------------------------------------------------------------------
            // CONV_TO_BFP：当前版本不做额外缩放，仅作为写回前过渡状态
            //------------------------------------------------------------------
            STATE_CONV_TO_BFP: begin
                // 留作将来接入真正的 BFP 缩放逻辑
            end
            
            //------------------------------------------------------------------
            // WRITE_BACK：写回权重存储（矩阵级，行主序）
            //------------------------------------------------------------------
            STATE_WRITE_BACK: begin
                // 默认不发请求
                weight_wr_req <= 1'b0;
            
                if (writeback_active) begin
                    // 先打包 exponent 数组（MAX_COLS 个列指数）
                    for (c_idx_flat = 0; c_idx_flat < MAX_COLS; c_idx_flat = c_idx_flat + 1) begin
                        packed_exp_array[c_idx_flat*EXP_WIDTH +: EXP_WIDTH] <= bfp_exp_buffer[c_idx_flat];
                    end
            
                    // 打包当前行的 MAX_COLS 个 mant（行主序）
                    for (c_idx_flat = 0; c_idx_flat < MAX_COLS; c_idx_flat = c_idx_flat + 1) begin
                        packed_row_burst[c_idx_flat*DATA_WIDTH +: DATA_WIDTH] <=
                            mant_row_major_flat[(write_row_idx*MAX_COLS + c_idx_flat)*DATA_WIDTH +: DATA_WIDTH];
                    end
            
                    // ---------------- 阶段 0：写 exponent burst ----------------
                    if (write_stage == 1'b0) begin
                        weight_wr_req        <= 1'b1;
                        weight_wr_layer_id   <= layer_id_reg;
                        weight_wr_type       <= weight_type_reg;
                        weight_wr_burst_idx  <= 6'd0;              // 约定 0 为 exponent burst
                        weight_wr_exp_array  <= packed_exp_array;
                        weight_wr_data_burst <= {DRAM_DATA_WIDTH{1'b0}};
            
                        if (weight_wr_ready) begin
                            write_stage   <= 1'b1;
                            write_row_idx <= 6'd0;
                        end
            
                    // ---------------- 阶段 1：逐行写 mantissa ----------------
                    end else begin
                        if (write_row_idx < total_rows) begin
                            weight_wr_req        <= 1'b1;
                            weight_wr_layer_id   <= layer_id_reg;
                            weight_wr_type       <= weight_type_reg;
                            // 约定：row 0..(total_rows-1) 映射到 burst_idx = 1..total_rows
                            weight_wr_burst_idx  <= write_row_idx + 6'd1;
                            weight_wr_exp_array  <= packed_exp_array;
                            weight_wr_data_burst <= packed_row_burst;
            
                            if (weight_wr_ready) begin
                                write_row_idx <= write_row_idx + 6'd1;
                            end
                        end else begin
                            // 所有有效行写完，写回结束
                            writeback_active <= 1'b0;
                            update_done      <= 1'b1;
                            update_busy      <= 1'b0;
                        end
                    end
                end
            end 
            
            //------------------------------------------------------------------
            // COL_NEXT：切换到下一列（列循环阶段）
            //------------------------------------------------------------------
            STATE_COL_NEXT: begin
                current_col <= current_col + 5'd1;
            end
            
            default: ;
        endcase
    end
end

endmodule

// ============================================================================
// 列主序 → 行主序重排模块
//   col_major_in[(col*ROWS + row)*DATA_WIDTH +: DATA_WIDTH] = data[col][row]
//   row_major_out[(row*COLS + col)*DATA_WIDTH +: DATA_WIDTH] = data[row][col]
// ============================================================================
module col2row_reorder #(
    parameter ROWS       = 32,
    parameter COLS       = 32,
    parameter DATA_WIDTH = 8
)(
    input  wire [COLS*ROWS*DATA_WIDTH-1:0] col_major_in,
    output wire [ROWS*COLS*DATA_WIDTH-1:0] row_major_out
);

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : GEN_ROW
            for (c = 0; c < COLS; c = c + 1) begin : GEN_COL
                assign row_major_out[(r*COLS + c)*DATA_WIDTH +: DATA_WIDTH] =
                       col_major_in[(c*ROWS + r)*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    endgenerate

endmodule
