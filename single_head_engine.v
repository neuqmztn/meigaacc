`timescale 1ns / 1ps

//===================================================================================
// 单Head注意力引擎 v2.0
//
// 功能：
//   处理单个attention head的21个chunks循环
//   每个chunk执行：QK计算 → Softmax → Apply V → 累加
//
// 改进点（v2.0）：
//   - 内部集成BFP Accumulator，完全自包含
//   - 无需外部累加器和仲裁逻辑
//   - 每个head独立管理自己的输出累加
//
// 参数化设计：
//   HEAD_ID: 0/1/2/3，决定读取哪个bank的数据
//
// 数据流：
//   Q Storage Bank[HEAD_ID] → Q[32×8]
//   KV Cache Bank[HEAD_ID]  → K[32×8], V[32×8]
//   QK → Scores[32×32] → Softmax → Weights[32×32] → Apply V → Output[32×8]
//   Output → 内部Accumulator[32×8] → 最终输出
//
// 循环结构：
//   for chunk_id in 21:
//       1. 读Q batch, K chunk
//       2. QK批量计算
//       3. Softmax批量处理
//       4. 读V chunk
//       5. Apply V批量计算
//       6. 累加到内部accumulator
//
// 4个Head并行：
//   每个Head有独立的本模块实例，完全异步运行
//
// 版本：v2.0
// 日期：2024-11-15
//===================================================================================
module single_head_engine #(
    parameter HEAD_ID       = 0,       // Head编号：0/1/2/3
    parameter NUM_CHUNKS    = 21,      // Chunk总数
    parameter TOKEN_BATCH   = 32,      // 每个batch的token数
    parameter CHUNK_SIZE    = 32,      // 每个chunk的token数
    parameter HEAD_DIM      = 8,       // Head维度
    parameter DATA_WIDTH    = 8,       // BFP尾数位宽
    parameter EXP_WIDTH     = 8,       // BFP指数位宽
    parameter SCORE_WIDTH   = 8,      // Score位宽
    parameter ACCUM_WIDTH   = 24,      // 累加器位宽
    
    // CE配置
    parameter NUM_PE = 1,
    parameter PE_TYPE_0 = 2,
    parameter PE_TYPE_1 = 2,
    parameter ELEM_PE0 = 8,
    parameter ELEM_PE1 = 8,
    parameter CE_OUTPUT_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 控制接口
    //===========================================================================
    input  wire start,                 // 从顶层启动
    output reg  done,                  // 21个chunk全部完成
    output reg  busy,
    
    //===========================================================================
    // Q读取接口（连接到Q Storage的Bank HEAD_ID）
    //===========================================================================
    output wire q_rd_en,
    input  wire [(TOKEN_BATCH*EXP_WIDTH)-1:0] q_batch_exp,
    input  wire [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] q_batch_mant,
    
    //===========================================================================
    // ✅ K读取接口（添加valid输入）
    //===========================================================================
    output reg  k_rd_en,
    output reg  [4:0] k_rd_chunk_id,
    input  wire k_rd_valid,               // ✅ 新增：K数据有效信号
    input  wire [(CHUNK_SIZE*EXP_WIDTH)-1:0] k_chunk_exp,
    input  wire [(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)-1:0] k_chunk_mant,
    
    //===========================================================================
    // ✅ V读取接口（添加valid输入）
    //===========================================================================
    output reg  v_rd_en,
    output reg  [4:0] v_rd_chunk_id,
    input  wire v_rd_valid,               // ✅ 新增：V数据有效信号
    input  wire [(CHUNK_SIZE*EXP_WIDTH)-1:0] v_chunk_exp,
    input  wire [(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)-1:0] v_chunk_mant,
    
    //===========================================================================
    // 最终输出读取接口（可选，供output projection使用）
    //===========================================================================
    input  wire final_rd_en,
    input  wire [4:0] final_rd_row,
    input  wire [2:0] final_rd_dim,
    output wire signed [ACCUM_WIDTH-1:0] final_rd_mant,
    output wire [EXP_WIDTH-1:0] final_rd_exp,
    output wire final_rd_valid,
    
    //===========================================================================
    // 调试接口
    //===========================================================================
    output wire [4:0] current_chunk_id,
    output wire [31:0] dbg_accum_wr_count
);

//===================================================================================
// ✅ 状态机定义（添加WAIT_K_VALID和WAIT_V_VALID状态）
//===================================================================================
localparam IDLE          = 4'd0;
localparam READ_Q        = 4'd1;
localparam READ_K        = 4'd2;
localparam WAIT_K_VALID  = 4'd3;   // ✅ 新增：等待K数据有效
localparam QK_COMPUTE    = 4'd4;
localparam WAIT_QK       = 4'd5;
localparam SOFTMAX       = 4'd6;
localparam WAIT_SM       = 4'd7;
localparam READ_V        = 4'd8;
localparam WAIT_V_VALID  = 4'd9;   // ✅ 新增：等待V数据有效
localparam APPLY_V       = 4'd10;
localparam WAIT_AV       = 4'd11;
localparam NEXT_CHUNK    = 4'd12;
localparam DONE_ST       = 4'd13;

reg [3:0] state;

//===================================================================================
// Chunk控制信号
//===================================================================================
reg [4:0] chunk_counter;
reg chunk_first;
reg chunk_last;

//===================================================================================
// Q数据有效标志
//===================================================================================
reg q_data_valid;
assign q_rd_en = !q_data_valid && (state == READ_Q);

//===================================================================================
// 子模块控制信号
//===================================================================================
reg qk_start;
wire qk_done;
wire qk_busy;
wire qk_valid;

reg softmax_start;
wire softmax_done;
wire softmax_busy;
wire weights_valid;

reg apply_v_start;
wire apply_v_done;
wire apply_v_busy;
wire apply_v_valid;

//===================================================================================
// 数据通路
//===================================================================================
wire [TOKEN_BATCH*CHUNK_SIZE*EXP_WIDTH-1:0] scores_exp;
wire [TOKEN_BATCH*CHUNK_SIZE*SCORE_WIDTH-1:0] scores_mants;

wire [TOKEN_BATCH*CHUNK_SIZE*SCORE_WIDTH-1:0] weights;

// ✅ Apply V逐个输出信号（修复版本）

wire [4:0] apply_v_query;
wire [2:0] apply_v_dim;
wire signed [ACCUM_WIDTH-1:0] apply_v_mant;
wire [EXP_WIDTH-1:0] apply_v_exp;
wire apply_v_first_chunk;

//===================================================================================
// Stats Buffer接口
//===================================================================================
wire max_rd_en, max_wr_en;
wire [1:0] max_rd_head, max_wr_head;
wire [4:0] max_rd_row, max_wr_row;
wire [SCORE_WIDTH-1:0] max_rd_value, max_wr_value;

wire sum_rd_en, sum_wr_en;
wire [1:0] sum_rd_head, sum_wr_head;
wire [4:0] sum_rd_row, sum_wr_row;
wire signed [ACCUM_WIDTH-1:0] sum_rd_value, sum_wr_value;

//===================================================================================
// ✅ Accumulator写入接口（简化为直接连接）
//===================================================================================
wire accum_wr_en;
wire [4:0] accum_wr_row;
wire [2:0] accum_wr_dim_wire;
wire signed [ACCUM_WIDTH-1:0] accum_wr_mant;
wire [EXP_WIDTH-1:0] accum_wr_exp;
wire accum_wr_first_chunk;

//===================================================================================
// 实例化：QK批量计算模块
//===================================================================================
qk_compute_batch #(
    .NUM_QUERIES(TOKEN_BATCH),
    .CHUNK_SIZE(CHUNK_SIZE),
    .HEAD_DIM(HEAD_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .SCORE_WIDTH(SCORE_WIDTH),
    .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
    .NUM_PE(NUM_PE),
    .PE_TYPE_0(PE_TYPE_0),
    .PE_TYPE_1(PE_TYPE_1),
    .ELEM_PE0(ELEM_PE0),
    .ELEM_PE1(ELEM_PE1)
) u_qk_batch (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(qk_start),
    .done(qk_done),
    .busy(qk_busy),
    
    // Q输入
    .q_batch_exp(q_batch_exp),
    .q_batch_mant(q_batch_mant),
    
    // K输入
    .k_chunk_exp(k_chunk_exp),
    .k_chunk_mant(k_chunk_mant),
    
    // Scores输出（直接连到Softmax）
    .scores_valid(qk_valid),
    .scores_exp_batch(scores_exp),
    .scores_batch(scores_mants)
);

//===================================================================================
// 实例化：Softmax批量处理模块
//===================================================================================
online_softmax_batch #(
    .NUM_QUERIES(TOKEN_BATCH),
    .CHUNK_SIZE(CHUNK_SIZE),
    .SCORE_WIDTH(SCORE_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ACCUM_WIDTH(ACCUM_WIDTH)
) u_softmax_batch (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(softmax_start),
    .done(softmax_done),
    .busy(softmax_busy),
    
    // Chunk标志
    .chunk_first(chunk_first),
    .chunk_last(chunk_last),
    
    // Scores输入
    .scores_exp_batch(scores_exp),
    .scores_batch(scores_mants),
    
    // Weights输出（连到Apply V）
    .weights_valid(weights_valid),
    .weights_batch(weights),
    
    // Stats Buffer接口
    .max_rd_en(max_rd_en),
    .max_rd_head(max_rd_head),
    .max_rd_row(max_rd_row),
    .max_rd_value(max_rd_value),
    
    .max_wr_en(max_wr_en),
    .max_wr_head(max_wr_head),
    .max_wr_row(max_wr_row),
    .max_wr_value(max_wr_value),
    
    .sum_rd_en(sum_rd_en),
    .sum_rd_head(sum_rd_head),
    .sum_rd_row(sum_rd_row),
    .sum_rd_value(sum_rd_value),
    
    .sum_wr_en(sum_wr_en),
    .sum_wr_head(sum_wr_head),
    .sum_wr_row(sum_wr_row),
    .sum_wr_value(sum_wr_value)
);

//===================================================================================
// 实例化：Apply V批量计算模块
//===================================================================================
apply_v_batch #(
    .NUM_QUERIES(TOKEN_BATCH),
    .CHUNK_SIZE(CHUNK_SIZE),
    .HEAD_DIM(HEAD_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .SCORE_WIDTH(SCORE_WIDTH),
    .ACCUM_WIDTH(ACCUM_WIDTH)
) u_apply_v_batch (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(apply_v_start),
    .done(apply_v_done),
    .busy(apply_v_busy),
    
    .chunk_first(chunk_first),
    
    // Weights输入
    .weights_batch(weights),
    
    // V输入
    .v_chunk_exp(v_chunk_exp),
    .v_chunk_mant(v_chunk_mant),
    
    // ✅ Output逐个输出（修复版本）
    .output_valid(apply_v_valid),
    .output_query(apply_v_query),
    .output_dim(apply_v_dim),
    .output_mant(apply_v_mant),
    .output_exp(apply_v_exp),
    .output_first_chunk(apply_v_first_chunk)
);

//===================================================================================
// 实例化：Stats Buffer（本head私有）
//===================================================================================
softmax_stats_buffer #(
         // 每个head engine只管理自己的stats
    .TOKEN_BATCH(TOKEN_BATCH),
    .SCORE_WIDTH(SCORE_WIDTH),
    .ACCUM_WIDTH(ACCUM_WIDTH)
) u_stats_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    .max_rd_en(max_rd_en),
    .max_rd_head(2'b0),      // 固定为head 0
    .max_rd_row(max_rd_row),
    .max_rd_value(max_rd_value),
    
    .max_wr_en(max_wr_en),
    .max_wr_head(2'b0),
    .max_wr_row(max_wr_row),
    .max_wr_value(max_wr_value),
    
    .sum_rd_en(sum_rd_en),
    .sum_rd_head(2'b0),
    .sum_rd_row(sum_rd_row),
    .sum_rd_value(sum_rd_value),
    
    .sum_wr_en(sum_wr_en),
    .sum_wr_head(2'b0),
    .sum_wr_row(sum_wr_row),
    .sum_wr_value(sum_wr_value)
);

//===================================================================================
// 实例化：BFP Accumulator（本head私有）
//===================================================================================
attention_bfp_accumulator #(
    .TOKEN_BATCH(TOKEN_BATCH),
    .HEAD_DIM(HEAD_DIM),
    .EXP_WIDTH(EXP_WIDTH),
    .ACCUM_WIDTH(ACCUM_WIDTH)
) u_accumulator (
    .clk(clk),
    .rst_n(rst_n),
    
    // 写入接口
    .wr_en(accum_wr_en),
    .wr_row(accum_wr_row),
    .wr_dim(accum_wr_dim_wire),
    .wr_mant(accum_wr_mant),
    .wr_exp(accum_wr_exp),
    .first_chunk(accum_wr_first_chunk),
    
    // 读取接口
    .rd_en(final_rd_en),
    .rd_row(final_rd_row),
    .rd_dim(final_rd_dim),
    .rd_mant(final_rd_mant),
    .rd_exp(final_rd_exp),
    .rd_valid(final_rd_valid)
);

//===================================================================================
// Accumulator写入逻辑（触发apply_v_valid后开始写入）
//===================================================================================
// ✅ Accumulator写入控制（简化版本 - 直接连接）
//===================================================================================
// 调试计数器
reg [31:0] total_accum_writes;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        total_accum_writes <= 32'd0;
    end else if (apply_v_valid) begin
        total_accum_writes <= total_accum_writes + 32'd1;
    end
end

// 直接连接Apply V的输出到Accumulator
assign accum_wr_en = apply_v_valid;
assign accum_wr_row = apply_v_query;
assign accum_wr_dim_wire = apply_v_dim;
assign accum_wr_mant = apply_v_mant;
assign accum_wr_exp = apply_v_exp;
assign accum_wr_first_chunk = apply_v_first_chunk;

//===================================================================================
// ✅ 主状态机：Chunk循环控制（添加Valid等待状态）
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        chunk_counter <= 5'd0;
        chunk_first <= 1'b0;
        chunk_last <= 1'b0;
        done <= 1'b0;
        busy <= 1'b0;
        q_data_valid <= 1'b0;
        
        k_rd_en <= 1'b0;
        k_rd_chunk_id <= 5'd0;
        v_rd_en <= 1'b0;
        v_rd_chunk_id <= 5'd0;
        
        qk_start <= 1'b0;
        softmax_start <= 1'b0;
        apply_v_start <= 1'b0;
        
    end else begin
        // 默认：清除单周期脉冲
        k_rd_en <= 1'b0;
        v_rd_en <= 1'b0;
        qk_start <= 1'b0;
        softmax_start <= 1'b0;
        apply_v_start <= 1'b0;
        
        case (state)
            //===================================================================
            // IDLE: 等待启动
            //===================================================================
            IDLE: begin
                done <= 1'b0;
                
                if (start) begin
                    busy <= 1'b1;
                    chunk_counter <= 5'd0;
                    chunk_first <= 1'b1;
                    chunk_last <= 1'b0;
                    q_data_valid <= 1'b0;
                    state <= READ_Q;
                    
                    $display("[%0t] Head %0d: Started, processing %0d chunks",
                             $time, HEAD_ID, NUM_CHUNKS);
                end else begin
                    busy <= 1'b0;
                end
            end
            
            //===================================================================
            // READ_Q: 读取Q batch（只在第一个chunk读取一次）
            //===================================================================
            READ_Q: begin
                if (!q_data_valid) begin
                    q_data_valid <= 1'b1;
                    
                    $display("[%0t] Head %0d: Reading Q batch",
                             $time, HEAD_ID);
                end
                state <= READ_K;
            end
            
            //===================================================================
            // ✅ READ_K: 发起K chunk读取请求
            //===================================================================
            READ_K: begin
                k_rd_en <= 1'b1;
                k_rd_chunk_id <= chunk_counter;
                state <= WAIT_K_VALID;  // ✅ 修改：进入等待valid状态
                
                $display("[%0t] Head %0d: Chunk %0d - Issuing K read request",
                         $time, HEAD_ID, chunk_counter);
            end
            
            //===================================================================
            // ✅ WAIT_K_VALID: 等待K数据有效（新增状态）
            //===================================================================
            WAIT_K_VALID: begin
                if (k_rd_valid) begin  // ✅ 检查valid信号
                    state <= QK_COMPUTE;
                    
                    $display("[%0t] Head %0d: Chunk %0d - K data valid, proceeding to compute",
                             $time, HEAD_ID, chunk_counter);
                end else begin
                    // 继续等待
                    $display("[%0t] Head %0d: Chunk %0d - Waiting for K valid...",
                             $time, HEAD_ID, chunk_counter);
                end
            end
            
            //===================================================================
            // QK_COMPUTE: 启动QK批量计算
            //===================================================================
            QK_COMPUTE: begin
                qk_start <= 1'b1;
                state <= WAIT_QK;
                
                $display("[%0t] Head %0d: Chunk %0d - Starting QK compute",
                         $time, HEAD_ID, chunk_counter);
            end
            
            //===================================================================
            // WAIT_QK: 等待QK计算完成
            //===================================================================
            WAIT_QK: begin
                if (qk_done) begin
                    state <= SOFTMAX;
                    
                    $display("[%0t] Head %0d: Chunk %0d - QK done",
                             $time, HEAD_ID, chunk_counter);
                end
            end
            
            //===================================================================
            // SOFTMAX: 启动Softmax批量处理
            //===================================================================
            SOFTMAX: begin
                softmax_start <= 1'b1;
                state <= WAIT_SM;
                
                $display("[%0t] Head %0d: Chunk %0d - Starting Softmax (first=%0b, last=%0b)",
                         $time, HEAD_ID, chunk_counter, chunk_first, chunk_last);
            end
            
            //===================================================================
            // WAIT_SM: 等待Softmax完成
            //===================================================================
            WAIT_SM: begin
                if (softmax_done) begin
                    state <= READ_V;
                    
                    $display("[%0t] Head %0d: Chunk %0d - Softmax done",
                             $time, HEAD_ID, chunk_counter);
                end
            end
            
            //===================================================================
            // ✅ READ_V: 发起V chunk读取请求
            //===================================================================
            READ_V: begin
                v_rd_en <= 1'b1;
                v_rd_chunk_id <= chunk_counter;
                state <= WAIT_V_VALID;  // ✅ 修改：进入等待valid状态
                
                $display("[%0t] Head %0d: Chunk %0d - Issuing V read request",
                         $time, HEAD_ID, chunk_counter);
            end
            
            //===================================================================
            // ✅ WAIT_V_VALID: 等待V数据有效（新增状态）
            //===================================================================
            WAIT_V_VALID: begin
                if (v_rd_valid) begin  // ✅ 检查valid信号
                    state <= APPLY_V;
                    
                    $display("[%0t] Head %0d: Chunk %0d - V data valid, proceeding to apply",
                             $time, HEAD_ID, chunk_counter);
                end else begin
                    // 继续等待
                    $display("[%0t] Head %0d: Chunk %0d - Waiting for V valid...",
                             $time, HEAD_ID, chunk_counter);
                end
            end
            
            //===================================================================
            // APPLY_V: 启动Apply V批量计算
            //===================================================================
            APPLY_V: begin
                apply_v_start <= 1'b1;
                state <= WAIT_AV;
                
                $display("[%0t] Head %0d: Chunk %0d - Starting Apply V",
                         $time, HEAD_ID, chunk_counter);
            end
            
            //===================================================================
            // ✅ WAIT_AV: 等待Apply V完成（简化版本）
            //===================================================================
            WAIT_AV: begin
                if (apply_v_done) begin
                    state <= NEXT_CHUNK;
                    
                    $display("[%0t] Head %0d: Chunk %0d - Apply V completed",
                             $time, HEAD_ID, chunk_counter);
                end
            end
            
            //===================================================================
            // NEXT_CHUNK: 移到下一个chunk
            //===================================================================
            NEXT_CHUNK: begin
                if (chunk_counter < NUM_CHUNKS - 1) begin
                    chunk_counter <= chunk_counter + 5'd1;
                    chunk_first <= 1'b0;
                    chunk_last <= (chunk_counter == NUM_CHUNKS - 2);
                    state <= READ_K;
                    
                    $display("[%0t] Head %0d: Moving to chunk %0d",
                             $time, HEAD_ID, chunk_counter + 1);
                end else begin
                    state <= DONE_ST;
                end
            end
            
            //===================================================================
            // DONE_ST: 全部完成
            //===================================================================
            DONE_ST: begin
                done <= 1'b1;
                busy <= 1'b0;
                state <= IDLE;
                
                $display("[%0t] Head %0d: All chunks completed!",
                         $time, HEAD_ID);
            end
            
            default: state <= IDLE;
        endcase
    end
end

//===================================================================================
// 调试输出
//===================================================================================
assign current_chunk_id = chunk_counter;
assign dbg_accum_wr_count = total_accum_writes;

//===================================================================================
// 初始化信息
//===================================================================================
initial begin
    $display("========================================");
    $display("Single Head Engine v2.1 - Head %0d", HEAD_ID);
    $display("========================================");
    $display("Improvements:");
    $display("  ✅ K read with valid signal handshake");
    $display("  ✅ V read with valid signal handshake");
    $display("  ✅ Robust variable-latency support");
    $display("========================================");
end

endmodule