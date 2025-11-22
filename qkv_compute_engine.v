`timescale 1ns / 1ps

//================================================================================
// QKV计算引擎 v2.3
// 
// 功能：
//   计算Q/K/V矩阵，支持三种工作模式：
//   - Q模式(00): 计算Query，结果写入storage
//   - K模式(01): 计算Key，结果写入KV Cache
//   - V模式(10): 计算Value，结果写入KV Cache
//
// 数据流：
//   Token(BFP) × Weight(BFP) → CE(定点) → BFP Converter → Output(BFP)
//================================================================================

module qkv_compute_engine #(
    // Token配置
    parameter TOKEN_NUM      = 640,    // prefill阶段总token数
    parameter TOKEN_BATCH    = 32,     // 每个batch的token数
    parameter DIM            = 32,     // 输入维度
    
    // Attention配置
    parameter NUM_HEADS      = 4,      // 注意力头数
    parameter HEAD_DIM       = 8,      // 每个头的维度
    
    // 数据格式
    parameter DATA_WIDTH     = 8,      // BFP尾数位宽
    parameter EXP_WIDTH      = 8,      // BFP指数位宽
    parameter MANT_WIDTH     = 8,      // 输出BFP尾数位宽
    
    // CE矩阵配置
    parameter G_OUT          = 4,      // CE输出列数
    parameter T_OUT          = 8,      // CE每列输出行数(对应头维度)
    parameter TOTAL_ELEM     = 32,     // 输入向量元素数
    parameter TOTAL_WIDTH    = 256,    // 输入向量总位宽
    
    // CE输出格式
    parameter CE_OUTPUT_WIDTH = 32,    // CE定点输出位宽
    parameter CE_BASE_EXP_WIDTH = 9    // CE输出基础指数位宽
)(
    input  wire clk,
    input  wire rst_n,
    
    //------------ 控制接口 ------------
    input  wire start,                 // 启动信号(脉冲)
    input  wire [1:0] compute_mode,    // 00=Q, 01=K, 10=V
    input  wire [5:0] batch_id,        // batch编号(Q模式使用)
    input  wire [4:0] tokens_in_batch, // 本batch的token数(Q模式使用)
    input  wire enable_shared_exp,     // 启用共享指数转换(暂未使用)
    output reg  done,                  // 完成标志(脉冲)
    output reg  busy,                  // 忙碌标志(电平)
    
    //------------ Token读取接口 ------------
    output reg  token_rd_en,
    output reg  [9:0] token_rd_addr,
    input  wire [EXP_WIDTH-1:0] token_rd_exp,
    input  wire [DIM*DATA_WIDTH-1:0] token_rd_mant,
    
    //------------ 权重读取接口 ------------
    output reg  weight_req,
    output reg  [1:0] weight_type,     // 请求的权重类型
    input  wire weight_ack,            // 权重控制器应答
    input  wire weight_valid,          // 权重数据有效
    input  wire [G_OUT*T_OUT*EXP_WIDTH-1:0] weight_exp_array,
    input  wire [G_OUT*T_OUT*TOTAL_WIDTH-1:0] weight_mant_blocks,
    
    //------------ Q矩阵存储接口 ------------
    // Q模式时使用，将结果写入storage以供后续QK^T计算
    output reg  storage_wr_en,
    output reg  [1:0] storage_matrix_type,
    output reg  [1:0] storage_head_id,
    output reg  [4:0] storage_token_id,
    output reg  [EXP_WIDTH-1:0] storage_shared_exp,
    output reg  [HEAD_DIM*DATA_WIDTH-1:0] storage_mant_packed,
    output reg  storage_overflow_flag,
    
    //------------ KV Cache接口 ------------
    // K/V模式时使用，将结果直接写入片上cache
    output reg  kv_wr_en,
    output reg  kv_wr_type,            // 0=K, 1=V
    output reg  [1:0] kv_wr_head,
    output reg  [9:0] kv_wr_token,
    output reg  [EXP_WIDTH-1:0] kv_wr_exp,
    output reg  [HEAD_DIM*DATA_WIDTH-1:0] kv_wr_mant,
    
    //------------ 状态输出 ------------
    output reg  all_kv_written         // 所有KV已写入完成标志
);

//================================================================================
// 状态机定义
//================================================================================
localparam STATE_IDLE           = 4'h0;  // 空闲，等待start
localparam STATE_REQ_WEIGHT     = 4'h1;  // 请求权重
localparam STATE_WAIT_WEIGHT    = 4'h2;  // 等待权重返回
localparam STATE_LOAD_TOKEN     = 4'h3;  // 读取token(3周期)
localparam STATE_SEND_CE        = 4'h4;  // 发送数据到CE
localparam STATE_WAIT_CE        = 4'h5;  // 等待CE计算完成
localparam STATE_WAIT_CONVERTER = 4'h6;  // 等待BFP转换完成
localparam STATE_SAVE_RESULT    = 4'h7;  // 保存结果(逐头写入)
localparam STATE_CLEAR_WR       = 4'h8;  // 清除写使能
localparam STATE_NEXT_TOKEN     = 4'h9;  // 准备下一个token
localparam STATE_DONE           = 4'hA;  // 完成

//================================================================================
// 内部信号
//================================================================================

// 状态机相关
reg [3:0] state;
reg [9:0] token_counter;        // token读取计数器
reg [9:0] global_token_idx;     // 全局token索引
reg [4:0] batch_token_idx;      // batch内token索引(Q模式)
reg [1:0] head_counter;         // 头索引(保存结果时使用)
reg [9:0] total_tokens;         // 本次需要处理的token总数

// 模式标志
reg is_mode_q, is_mode_k, is_mode_v;

// 权重缓存
// 每种权重(Q/K/V)只加载一次，复用于所有token
reg [2:0] weight_loaded_flags;  // bit0=Q, bit1=K, bit2=V
reg [G_OUT*T_OUT*EXP_WIDTH-1:0] weight_exp_cached;     // 缓存的权重指数数组
reg [G_OUT*T_OUT*TOTAL_WIDTH-1:0] weight_mant_cached;  // 缓存的权重尾数数组

// Token缓存
reg [EXP_WIDTH-1:0] token_exp_cached;
reg [TOTAL_WIDTH-1:0] token_mant_cached;

// 统计
reg [9:0] kv_written_count;     // K/V模式下已写入的token数

//================================================================================
// CE接口信号
//================================================================================

// CE输入控制
reg ce_input_valid;
reg [EXP_WIDTH-1:0] ce_exp_X;
reg [TOTAL_WIDTH-1:0] ce_mant_X;
// 注意：CE的权重输入直接连接到weight_xxx_cached，无需额外信号
// ✅ 新增：CE握手信号
wire ce_input_ready;    // CE是否准备好接收
reg  ce_result_ready;   // 告诉CE我们是否准备好接收结果

// CE输出(定点格式)
wire [G_OUT*T_OUT-1:0] ce_result_valids;
wire signed [G_OUT*T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_array;
wire [G_OUT*T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_array;
wire [G_OUT*T_OUT-1:0] ce_result_zero_array;

//================================================================================
// CE输出缓存
//================================================================================
reg signed [G_OUT*T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_buf;
reg [G_OUT*T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_buf;
reg [G_OUT*T_OUT-1:0] ce_result_zero_buf;
reg ce_result_cached;  // 缓存有效标志

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ce_result_fixed_buf    <= {(G_OUT*T_OUT*CE_OUTPUT_WIDTH){1'b0}};
        ce_result_base_exp_buf <= {(G_OUT*T_OUT*CE_BASE_EXP_WIDTH){1'b0}};
        ce_result_zero_buf     <= {(G_OUT*T_OUT){1'b0}};
        ce_result_cached       <= 1'b0;
    end else begin
        case (state)
            STATE_WAIT_CE: begin
                // CE输出有效时，采样到缓存寄存器
                if (ce_result_valids[0] && !ce_result_cached) begin
                    ce_result_fixed_buf    <= ce_result_fixed_array;
                    ce_result_base_exp_buf <= ce_result_base_exp_array;
                    ce_result_zero_buf     <= ce_result_zero_array;
                    ce_result_cached       <= 1'b1;
                end
            end
            
            STATE_WAIT_CONVERTER: begin
                // 保持缓存，供BFP转换器稳定使用
            end
            
            STATE_CLEAR_WR: begin
                // 结果已保存，清除缓存标志，准备下一次
                ce_result_cached <= 1'b0;
            end
        endcase
    end
end

//================================================================================
// 多头BFP转换器输出接口
//================================================================================
wire [G_OUT*T_OUT-1:0] converter_output_valids;
wire signed [G_OUT*T_OUT*MANT_WIDTH-1:0] converter_output_mants;
wire [G_OUT*EXP_WIDTH-1:0] converter_output_shared_exps;  // 每个头一个共享指数
wire [G_OUT-1:0] converter_alignment_overflow;            // 每个头一个溢出标志
//================================================================================
// CE配置参数（使用localparam，不需要外部修改）
//================================================================================
// PE配置
localparam CE_NUM_PE      = 2;     // 每个PU有2个PE
localparam CE_PE_TYPE_0   = 0;     // PE0类型：A型
localparam CE_PE_TYPE_1   = 2;     // PE1类型：C型

// 向量维度配置
localparam CE_ELEM_PE0    = 16;    // PE0处理16个元素
localparam CE_ELEM_PE1    = 16;    // PE1处理16个元素

// 位宽优化配置
localparam CE_INTERNAL_WIDTH  = 39;  // PU内部39位计算
localparam CE_GUARD_BITS      = 7;   // 截断7位（39-32）
localparam CE_ENABLE_ROUNDING = 1;   // 启用舍入

// 流控配置
localparam CE_FIFO_DEPTH        = 16;   // FIFO深度
localparam CE_HANDSHAKE_TIMEOUT = 100;  // 握手超时周期

//================================================================================
// Compute Engine实例化
//
// CE负责执行 Y = X × W 的矩阵乘法
// 输入：X(token, BFP格式), W(权重, BFP格式)
// 输出：Y(定点格式, 32位)
//================================================================================
compute_engine #(
    // ========== 矩阵维度 ==========
    .G_OUT(G_OUT),
    .T_OUT(T_OUT),
    
    // ========== PE配置 ========== 
    .NUM_PE(CE_NUM_PE),
    .PE_TYPE_0(0),
    .PE_TYPE_1(0),
    
    // ========== 数据位宽 ==========
    .EXP_WIDTH(EXP_WIDTH),
    .INPUT_MANT_WIDTH(DATA_WIDTH),    // 
    .MANT_WIDTH(MANT_WIDTH),          // 保留兼容性
    
    // ========== 向量维度 ========== 
    .ELEM_PE0(CE_ELEM_PE0),
    .ELEM_PE1(CE_ELEM_PE1),
    .TOTAL_ELEM(TOTAL_ELEM),
    
    // ========== 位宽优化配置 ========== 
    .INTERNAL_WIDTH(CE_INTERNAL_WIDTH),
    .OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
    .GUARD_BITS(CE_GUARD_BITS),
    .ENABLE_ROUNDING(CE_ENABLE_ROUNDING),
    
    // ========== FIFO配置 ========== 
    .FIFO_DEPTH(CE_FIFO_DEPTH),
    .HANDSHAKE_TIMEOUT(CE_HANDSHAKE_TIMEOUT)
) u_compute_engine (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    .input_valid(ce_input_valid),
    .input_ready(ce_input_ready), 
    // 输入X：当前token
    .exp_X(ce_exp_X),
    .mant_X_block(ce_mant_X),
    
    // 输入W：权重矩阵(直接使用缓存)
    .exp_W_array(weight_exp_cached),
    .mant_W_blocks(weight_mant_cached),
    
    // 输出：定点格式结果
    .result_valids(ce_result_valids),
    .result_ready(ce_result_ready),
    .result_fixed_array(ce_result_fixed_array),
    .result_base_exp_array(ce_result_base_exp_array),
    .result_zero_array(ce_result_zero_array)
);

//================================================================================
// 多头BFP转换器实例化
//
// 将CE的定点输出转换回BFP格式
// 特点：每个attention head独立计算共享指数
//================================================================================
multi_head_bfp_converter #(
    .NUM_HEADS(G_OUT),                  // 4个头
    .RESULTS_PER_HEAD(T_OUT),           // 每头8个元素
    .FIXED_WIDTH(CE_OUTPUT_WIDTH),      // 32位定点输入
    .BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH), // 9位基础指数
    .OUTPUT_MANT_WIDTH(MANT_WIDTH),     // 8位BFP尾数
    .OUTPUT_EXP_WIDTH(EXP_WIDTH)        // 8位BFP指数
) u_multi_head_bfp_converter (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    
    // 输入：使用缓存的CE结果(确保数据稳定)
    .input_valids({(G_OUT*T_OUT){ce_result_cached}}),
    .input_fixed_array(ce_result_fixed_buf),
    .input_base_exp_array(ce_result_base_exp_buf),
    .input_zero_array(ce_result_zero_buf),
    
    // 输出：每个头的共享指数BFP格式
    .output_valids(converter_output_valids),
    .output_mant_array(converter_output_mants),
    .output_shared_exps(converter_output_shared_exps),
    .output_overflow(converter_alignment_overflow)
);

//================================================================================
// 主状态机
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 状态机复位
        state              <= STATE_IDLE;
        done               <= 1'b0;
        busy               <= 1'b0;
        token_counter      <= 10'd0;
        global_token_idx   <= 10'd0;
        batch_token_idx    <= 5'd0;
        head_counter       <= 2'd0;
        total_tokens       <= 10'd0;
        
        is_mode_q          <= 1'b0;
        is_mode_k          <= 1'b0;
        is_mode_v          <= 1'b0;
        
        // 权重缓存复位
        weight_loaded_flags <= 3'b000;
        weight_exp_cached   <= {G_OUT*T_OUT*EXP_WIDTH{1'b0}};
        weight_mant_cached  <= {G_OUT*T_OUT*TOTAL_WIDTH{1'b0}};
        
        // Token缓存复位
        token_exp_cached    <= {EXP_WIDTH{1'b0}};
        token_mant_cached   <= {TOTAL_WIDTH{1'b0}};
        
        kv_written_count    <= 10'd0;
        all_kv_written      <= 1'b0;
        
        // CE输入控制复位
        ce_input_valid      <= 1'b0;
        ce_exp_X            <= {EXP_WIDTH{1'b0}};
        ce_mant_X           <= {TOTAL_WIDTH{1'b0}};
        ce_result_ready     <= 1'b0;
        
        // 输出接口复位
        token_rd_en         <= 1'b0;
        token_rd_addr       <= 10'd0;
        
        weight_req          <= 1'b0;
        weight_type         <= 2'd0;
        
        storage_wr_en       <= 1'b0;
        storage_matrix_type <= 2'd0;
        storage_head_id     <= 2'd0;
        storage_token_id    <= 5'd0;
        storage_shared_exp  <= {EXP_WIDTH{1'b0}};
        storage_mant_packed <= {HEAD_DIM*DATA_WIDTH{1'b0}};
        storage_overflow_flag <= 1'b0;
        
        kv_wr_en            <= 1'b0;
        kv_wr_type          <= 1'b0;
        kv_wr_head          <= 2'd0;
        kv_wr_token         <= 10'd0;
        kv_wr_exp           <= {EXP_WIDTH{1'b0}};
        kv_wr_mant          <= {HEAD_DIM*DATA_WIDTH{1'b0}};
        
    end else begin
        case (state)
            
            //================================================================
            // IDLE: 等待启动信号
            //================================================================
            STATE_IDLE: begin
                done <= 1'b0;
                busy <= 1'b0;
                
                if (start) begin
                    busy  <= 1'b1;
                    done  <= 1'b0;
                    
                    // 解码工作模式
                    is_mode_q <= (compute_mode == 2'b00);
                    is_mode_k <= (compute_mode == 2'b01);
                    is_mode_v <= (compute_mode == 2'b10);
                    
                    token_counter   <= 10'd0;
                    batch_token_idx <= 5'd0;
                    
                    // Q模式：只处理当前batch的token
                    // K/V模式：处理所有token(prefill阶段)
                    if (compute_mode == 2'b00) begin
                        global_token_idx <= batch_id * TOKEN_BATCH;
                        total_tokens     <= tokens_in_batch;
                    end else begin
                        global_token_idx <= 10'd0;
                        total_tokens     <= TOKEN_NUM;
                    end
                    
                    // 检查权重是否已加载
                    if (weight_loaded_flags[compute_mode]) begin
                        state <= STATE_LOAD_TOKEN;
                    end else begin
                        state <= STATE_REQ_WEIGHT;
                    end
                end
            end
            
            //================================================================
            // REQ_WEIGHT: 请求权重
            //================================================================
            STATE_REQ_WEIGHT: begin
                if (!weight_req) begin
                    weight_req  <= 1'b1;
                    weight_type <= compute_mode;
                end
                
                if (weight_ack) begin
                    weight_req <= 1'b0;
                    state      <= STATE_WAIT_WEIGHT;
                end
            end
            
            //================================================================
            // WAIT_WEIGHT: 等待权重返回并缓存
            //================================================================
            STATE_WAIT_WEIGHT: begin
                if (weight_valid) begin
                    weight_exp_cached  <= weight_exp_array;
                    weight_mant_cached <= weight_mant_blocks;
                    weight_loaded_flags[compute_mode] <= 1'b1;
                    state              <= STATE_LOAD_TOKEN;
                end
            end
            
            //================================================================
            // LOAD_TOKEN: 读取token数据(3周期流程)
            //
            // Cycle 0: 发起读请求
            // Cycle 1: 等待BRAM
            // Cycle 2: 采样数据
            //================================================================
            STATE_LOAD_TOKEN: begin
                case (token_counter)
                    10'd0: begin
                        token_rd_en   <= 1'b1;
                        token_rd_addr <= global_token_idx;
                        token_counter <= token_counter + 1;
                    end
                    
                    10'd1: begin
                        token_rd_en   <= 1'b0;
                        token_counter <= token_counter + 1;
                    end
                    
                    10'd2: begin
                        token_exp_cached  <= token_rd_exp;
                        token_mant_cached <= token_rd_mant;
                        token_counter     <= 10'd0;
                        state             <= STATE_SEND_CE;
                    end
                endcase
            end
            
            //================================================================
            // SEND_CE: 发送数据到CE
            //================================================================
            STATE_SEND_CE: begin
                ce_input_valid <= 1'b1;
                ce_exp_X       <= token_exp_cached;
                ce_mant_X      <= token_mant_cached;
                if (ce_input_ready) begin
                    state <= STATE_WAIT_CE;
                end 
                // 否则保持在当前状态，继续等待
            end
            
            //================================================================
            // WAIT_CE: 等待CE计算完成
            //
            // CE输出有效后会被自动采样到ce_result_xxx_buf
            // 这里只需等待ce_result_cached标志
            //================================================================
            STATE_WAIT_CE: begin
                ce_input_valid  <= 1'b0;
                ce_result_ready <= 1'b1;  // 告诉CE我们准备好接收结果
                
                if (ce_result_cached) begin
                    ce_result_ready <= 1'b0;  // 采样完成，不再ready
                    state <= STATE_WAIT_CONVERTER;
                end
            end
            
            //================================================================
            // WAIT_CONVERTER: 等待BFP转换完成
            //================================================================
            STATE_WAIT_CONVERTER: begin
                ce_result_ready <= 1'b0;
                if (|converter_output_valids) begin
                    state        <= STATE_SAVE_RESULT;
                    head_counter <= 2'd0;
                end
            end
            
            //================================================================
            // SAVE_RESULT: 保存结果
            //
            // 逐个head写入，每个head一个周期
            // Q模式写storage，K/V模式写KV cache
            //================================================================
            STATE_SAVE_RESULT: begin
                if (is_mode_q) begin
                    // Q模式：写入storage
                    storage_wr_en         <= 1'b1;
                    storage_matrix_type   <= compute_mode;
                    storage_head_id       <= head_counter;
                    storage_token_id      <= batch_token_idx;
                    
                    // 从转换器输出中提取当前head的数据
                    storage_shared_exp    <= converter_output_shared_exps[head_counter*EXP_WIDTH +: EXP_WIDTH];
                    storage_mant_packed   <= converter_output_mants[head_counter*T_OUT*MANT_WIDTH +: T_OUT*MANT_WIDTH];
                    storage_overflow_flag <= converter_alignment_overflow[head_counter];
                    
                    if (head_counter < NUM_HEADS - 1) begin
                        head_counter <= head_counter + 1'b1;
                    end else begin
                        state <= STATE_CLEAR_WR;
                    end
                    
                end else begin
                    // K/V模式：写入KV Cache
                    kv_wr_en    <= 1'b1;
                    kv_wr_type  <= compute_mode[0];  // K=0, V=1
                    kv_wr_head  <= head_counter;
                    kv_wr_token <= global_token_idx;
                    
                    kv_wr_exp   <= converter_output_shared_exps[head_counter*EXP_WIDTH +: EXP_WIDTH];
                    kv_wr_mant  <= converter_output_mants[head_counter*T_OUT*MANT_WIDTH +: T_OUT*MANT_WIDTH];
                    
                    if (head_counter < NUM_HEADS - 1) begin
                        head_counter <= head_counter + 1'b1;
                    end else begin
                        kv_written_count <= kv_written_count + 1'b1;
                        state <= STATE_CLEAR_WR;
                    end
                end
            end
            
            //================================================================
            // CLEAR_WR: 清除写使能，准备下一个token
            //================================================================
            STATE_CLEAR_WR: begin
                storage_wr_en <= 1'b0;
                kv_wr_en      <= 1'b0;
                state         <= STATE_NEXT_TOKEN;
            end
            
            //================================================================
            // NEXT_TOKEN: 更新token索引
            //================================================================
            STATE_NEXT_TOKEN: begin
                global_token_idx <= global_token_idx + 1'b1;
                
                if (is_mode_q) begin
                    batch_token_idx <= batch_token_idx + 1'b1;
                end
                
                // 检查是否处理完所有token
                if (global_token_idx >= total_tokens - 1) begin
                    state <= STATE_DONE;
                end else begin
                    state <= STATE_LOAD_TOKEN;
                end
            end
            
            //================================================================
            // DONE: 完成一次计算
            //================================================================
            STATE_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= STATE_IDLE;
                
                // K/V模式：检查是否所有token的KV都已写入
                if (!is_mode_q && kv_written_count >= TOKEN_NUM) begin
                    all_kv_written <= 1'b1;
                end
            end
            
            default: state <= STATE_IDLE;
        endcase
    end
end

//================================================================================
// 仿真信息
//================================================================================
initial begin
    $display("========================================");
    $display("QKV Compute Engine v2.3");
    $display("========================================");
    $display("Configuration:");
    $display("  Attention Heads: %0d", NUM_HEADS);
    $display("  Head Dimension: %0d", HEAD_DIM);
    $display("  Total PUs: %0d (G_OUT=%0d × T_OUT=%0d)", G_OUT*T_OUT, G_OUT, T_OUT);
    $display("  CE Output: %0d-bit fixed-point", CE_OUTPUT_WIDTH);
    $display("  BFP Output: %0d-bit mantissa", MANT_WIDTH);
    $display("========================================");
end

endmodule