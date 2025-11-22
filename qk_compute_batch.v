`timescale 1ns / 1ps

//===================================================================================
// QK批量计算模块 
// 功能：
//   计算32个Q向量与32个K向量的点积，生成32×32的scores矩阵
//   Q[32×8] × K[32×8] → Scores[32×32]
//
// 算法：
//   for q_idx in 32:
//       调用CE计算 Q[q_idx] × K[0:31] → Scores[q_idx][0:31]
//   
// 数据格式：
//   - 输入Q：32个向量，每个向量有独立BFP指数
//   - 输入K：32个向量，每个向量有独立BFP指数
//   - 输出Scores：32×32矩阵，每行共享一个BFP指数（32个指数）
//
// CE配置：
//   G_OUT = 1   （1列，因为每次处理1个Q）
//   T_OUT = 32  （32行，对应32个K）
//   TOTAL_ELEM = 8  （向量维度）
//
// 时序：
//   32次循环 × CE延迟 ≈ 32 × 5 cycles = 160 cycles
//===================================================================================

module qk_compute_batch #(
    parameter NUM_QUERIES  = 32,       // Q向量数量
    parameter CHUNK_SIZE   = 32,       // K向量数量
    parameter HEAD_DIM     = 8,        // 向量维度
    parameter DATA_WIDTH   = 8,        // BFP尾数位宽
    parameter EXP_WIDTH    = 8,        // BFP指数位宽
    parameter SCORE_WIDTH  = 8,       // 输出score位宽
    parameter CE_OUTPUT_WIDTH = 32,    // CE输出位宽
    
    // CE的PE配置
    parameter NUM_PE = 2,
    parameter PE_TYPE_0 = 0,
    parameter PE_TYPE_1 = 2,
    parameter ELEM_PE0 = 16,
    parameter ELEM_PE1 = 8
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 控制接口
    //===========================================================================
    input  wire start,
    output reg  done,
    output reg  busy,
    
    //===========================================================================
    // Q输入：32个向量，每个有独立指数
    //===========================================================================
    input  wire [(NUM_QUERIES*EXP_WIDTH)-1:0] q_batch_exp,
    input  wire [(NUM_QUERIES*HEAD_DIM*DATA_WIDTH)-1:0] q_batch_mant,
    
    //===========================================================================
    // K输入：32个向量，每个有独立指数
    //===========================================================================
    input  wire [(CHUNK_SIZE*EXP_WIDTH)-1:0] k_chunk_exp,
    input  wire [(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)-1:0] k_chunk_mant,
    
    //===========================================================================
    // Scores输出：32×32矩阵，每行共享指数
    //===========================================================================
    output reg  scores_valid,
    output reg  [(NUM_QUERIES*EXP_WIDTH)-1:0] scores_exp_batch,
    output reg  [(NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH)-1:0] scores_batch
);

//===================================================================================
// 本地参数
//===================================================================================

localparam TOTAL_ELEM  = HEAD_DIM;
localparam TOTAL_WIDTH = HEAD_DIM * DATA_WIDTH;
localparam CE_BASE_EXP_WIDTH = EXP_WIDTH + 1;

// CE配置
localparam G_OUT = 1;      // 1列输出
localparam T_OUT = CHUNK_SIZE;  // 32行输出

//===================================================================================
// 状态机
//===================================================================================

localparam IDLE    = 2'd0;
localparam COMPUTE = 2'd1;
localparam WAIT_CE = 2'd2;
localparam DONE_ST = 2'd3;

reg [1:0] state;
reg [5:0] query_idx;  // 0-31

//===================================================================================
// CE接口信号
//===================================================================================

reg ce_input_valid;
wire ce_input_ready;
wire [T_OUT-1:0] ce_result_valids;
reg ce_result_ready;

// 当前query的Q向量
reg [EXP_WIDTH-1:0] current_q_exp;
reg [(HEAD_DIM*DATA_WIDTH)-1:0] current_q_mant;

// CE输出
wire signed [T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_array;
wire [T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_array;
wire [T_OUT-1:0] ce_result_zero_array;

// CE输出有效标志（所有PU都输出）
wire ce_output_valid;
assign ce_output_valid = &ce_result_valids;

//===================================================================================
// 提取当前query的Q向量
//===================================================================================

always @(*) begin
    // 从q_batch中提取第query_idx个Q向量
    current_q_exp = q_batch_exp[query_idx*EXP_WIDTH +: EXP_WIDTH];
    current_q_mant = q_batch_mant[query_idx*HEAD_DIM*DATA_WIDTH +: HEAD_DIM*DATA_WIDTH];
end

//===================================================================================
// 实例化Compute Engine
//===================================================================================

compute_engine #(
    .G_OUT(G_OUT),              // 1列
    .T_OUT(T_OUT),              // 32行
    .NUM_PE(NUM_PE),
    .PE_TYPE_0(PE_TYPE_0),
    .PE_TYPE_1(PE_TYPE_1),
    .EXP_WIDTH(EXP_WIDTH),
    .INPUT_MANT_WIDTH(DATA_WIDTH),
    .ELEM_PE0(ELEM_PE0),
    .ELEM_PE1(ELEM_PE1),
    .TOTAL_ELEM(TOTAL_ELEM),
    .OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
    .INTERNAL_WIDTH(39),
    .GUARD_BITS(7),
    .ENABLE_ROUNDING(1)
) u_ce (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    
    .input_valid(ce_input_valid),
    .input_ready(ce_input_ready),
    
    // 输入X：当前Q向量
    .exp_X(current_q_exp),
    .mant_X_block(current_q_mant),
    
    // 输入W：所有32个K向量
    // 注意：k_chunk_exp和k_chunk_mant的格式已经和CE期望的完全一致
    // k_chunk_exp = {k[31]_exp, k[30]_exp, ..., k[1]_exp, k[0]_exp}
    // k_chunk_mant = {k[31]_mant, k[30]_mant, ..., k[1]_mant, k[0]_mant}
    .exp_W_array(k_chunk_exp),
    .mant_W_blocks(k_chunk_mant),
    
    .result_valids(ce_result_valids),
    .result_ready(ce_result_ready),
    
    .result_fixed_array(ce_result_fixed_array),
    .result_base_exp_array(ce_result_base_exp_array),
    .result_zero_array(ce_result_zero_array)
);

//===================================================================================
// BFP转换：将CE的定点输出转为共享指数BFP格式
//
// 关键修改：使用正确的 bfp_converter 模块
//===================================================================================

// BFP转换后的结果（当前行）
wire [EXP_WIDTH-1:0] bfp_shared_exp;
wire signed [(CHUNK_SIZE*SCORE_WIDTH)-1:0] bfp_mants_packed;
wire [CHUNK_SIZE-1:0] bfp_output_valids;
wire bfp_overflow;

// 调用 bfp_converter 模块（共享指数版本）
bfp_converter #(
    .TOTAL_RESULTS(CHUNK_SIZE),           // 32个结果
    .FIXED_WIDTH(CE_OUTPUT_WIDTH),        // 32位定点输入
    .BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH),   // 9位基础指数
    .OUTPUT_MANT_WIDTH(SCORE_WIDTH),      // 8位输出尾数
    .OUTPUT_EXP_WIDTH(EXP_WIDTH)          // 8位输出指数
) u_bfp_converter (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    
    // 输入：CE的32个定点输出
    .input_valids(ce_result_valids),
    .input_fixed_array(ce_result_fixed_array),
    .input_base_exp_array(ce_result_base_exp_array),
    .input_zero_array(ce_result_zero_array),
    
    // 输出：BFP格式（1个共享指数 + 32个尾数）
    .output_valids(bfp_output_valids),
    .output_mant_array(bfp_mants_packed),
    .output_shared_exp(bfp_shared_exp),
    .output_overflow(bfp_overflow)
);

//===================================================================================
// 主状态机
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        query_idx <= 6'd0;
        done <= 1'b0;
        busy <= 1'b0;
        ce_input_valid <= 1'b0;
        ce_result_ready <= 1'b0;
        scores_valid <= 1'b0;
        scores_exp_batch <= {(NUM_QUERIES*EXP_WIDTH){1'b0}};
        scores_batch <= {(NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH){1'b0}};
        
    end else begin
        case (state)
            //===================================================================
            // IDLE: 等待启动
            //===================================================================
            IDLE: begin
                done <= 1'b0;
                scores_valid <= 1'b0;
                
                if (start) begin
                    busy <= 1'b1;
                    query_idx <= 6'd0;
                    state <= COMPUTE;
                    
                    $display("[%0t] QK_Batch: Started", $time);
                end else begin
                    busy <= 1'b0;
                end
            end
            
            //===================================================================
            // COMPUTE: 启动CE计算当前query
            //===================================================================
            COMPUTE: begin
                // 启动CE
                ce_input_valid <= 1'b1;
                ce_result_ready <= 1'b1;
                
                // 等待CE握手
                if (ce_input_valid && ce_input_ready) begin
                    ce_input_valid <= 1'b0;
                    state <= WAIT_CE;
                    
                    $display("[%0t] QK_Batch: Computing Q[%0d] × K[0:31]", 
                             $time, query_idx);
                end
            end
            
            //===================================================================
            // WAIT_CE: 等待BFP转换器输出
            // 注意：bfp_converter有1周期寄存延迟
            //===================================================================
            WAIT_CE: begin
                // 检查BFP转换器输出有效
                if (|bfp_output_valids) begin
                    // 保存BFP转换后的结果
                    scores_exp_batch[query_idx*EXP_WIDTH +: EXP_WIDTH] <= bfp_shared_exp;
                    scores_batch[query_idx*CHUNK_SIZE*SCORE_WIDTH +: CHUNK_SIZE*SCORE_WIDTH] <= bfp_mants_packed;
                    
                    ce_result_ready <= 1'b0;
                    
                    $display("[%0t] QK_Batch: Q[%0d] done, exp=%0d", 
                             $time, query_idx, bfp_shared_exp);
                    
                    // 检查是否所有query都完成
                    if (query_idx < NUM_QUERIES - 1) begin
                        query_idx <= query_idx + 6'd1;
                        state <= COMPUTE;
                    end else begin
                        state <= DONE_ST;
                    end
                end
            end
            
            //===================================================================
            // DONE: 全部完成
            //===================================================================
            DONE_ST: begin
                done <= 1'b1;
                busy <= 1'b0;
                scores_valid <= 1'b1;
                state <= IDLE;
                
                $display("[%0t] QK_Batch: All 32 queries completed", $time);
            end
            
            default: state <= IDLE;
        endcase
    end
end

//===================================================================================
// 参数合法性检查
//===================================================================================

initial begin
    if (NUM_QUERIES != 32) begin
        $display("WARNING: NUM_QUERIES=%0d, expected 32", NUM_QUERIES);
    end
    if (CHUNK_SIZE != 32) begin
        $display("WARNING: CHUNK_SIZE=%0d, expected 32", CHUNK_SIZE);
    end
    if (HEAD_DIM != 8) begin
        $display("WARNING: HEAD_DIM=%0d, expected 8", HEAD_DIM);
    end
    
    $display("========================================");
    $display("QK Compute Batch Module (Fixed)");
    $display("========================================");
    $display("Configuration:");
    $display("  Q vectors: %0d", NUM_QUERIES);
    $display("  K vectors: %0d", CHUNK_SIZE);
    $display("  Vector dim: %0d", HEAD_DIM);
    $display("  Output matrix: %0d × %0d", NUM_QUERIES, CHUNK_SIZE);
    $display("  BFP Converter: bfp_converter (shared exp)");
    $display("========================================");
end

endmodule