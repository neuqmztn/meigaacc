`timescale 1ns / 1ps

//===================================================================================
// Backbone Attention Parallel Top - 完整的4-Head并行Attention系统
//
// 架构特点：
//   - 三层控制结构：
//     * L1: Batch Controller（管理21个batch循环 + Output Projection）
//     * L2: 4× Single Head Engine（各自管理21个chunk循环 + 内部accumulator）
//     * L3: Output Projection（读取4个head输出并投影）
//   - 4-Head完全并行，无竞争，无仲裁
//   - Bank化存储：Q Storage、KV Cache都按head分bank
//
// 完整数据流：
//   Token Buffer → QKV Compute → Q Storage / KV Cache
//   Q Storage + KV Cache → 4× Single Head Engine (并行，内含Accumulator)
//   4× Head Engine Accumulator → Output Projection → Result Buffer
//
// 性能：
//   - Batch处理：32个query并行
//   - Head并行：4个head独立
//   - 总加速比：~128× vs. 串行单query处理
//


//===================================================================================

module backbone_attention_parallel_top #(
    // ========== Token配置 ==========
    parameter TOKEN_NUM      = 640,
    parameter TOKEN_BATCH    = 32,
    parameter NUM_BATCHES    = 20,
    parameter DIM            = 32,
    
    // ========== Attention配置 ==========
    parameter NUM_HEADS      = 4,
    parameter HEAD_DIM       = 8,
    parameter NUM_CHUNKS     = 20,
    parameter CHUNK_SIZE     = 32,
    
    // ========== 数据格式 ==========
    parameter DATA_WIDTH     = 8,
    parameter EXP_WIDTH      = 8,
    parameter MANT_WIDTH     = 8,
    parameter SCORE_WIDTH    = 16,
    parameter ACCUM_WIDTH    = 24,
    
    // ========== CE配置 ==========
    parameter NUM_PE         = 2,
    parameter PE_TYPE_0      = 0,
    parameter PE_TYPE_1      = 2,
    parameter ELEM_PE0       = 16,
    parameter ELEM_PE1       = 8,
    parameter CE_OUTPUT_WIDTH = 32,
    parameter CE_INTERNAL_WIDTH = 39,
    parameter CE_GUARD_BITS  = 7,
    parameter CE_ENABLE_ROUNDING = 1,
    
    // ========== QKV Compute配置 ==========
    parameter G_OUT          = 4,
    parameter T_OUT          = 8,
    parameter TOTAL_ELEM     = 32,
    parameter TOTAL_WIDTH    = 256,
    parameter CE_BASE_EXP_WIDTH = 9
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 顶层控制接口
    //===========================================================================
    input  wire start,
    output wire done,
    output wire busy,
    
    //===========================================================================
    // Token Buffer接口
    //===========================================================================
    output wire token_rd_en,
    output wire [9:0] token_rd_addr,
    input  wire [EXP_WIDTH-1:0] token_rd_exp,
    input  wire [DIM*DATA_WIDTH-1:0] token_rd_mant,
    
    //===========================================================================
    // Weight Storage接口
    //===========================================================================
    // QKV权重
    output wire qkv_weight_req,
    output wire [1:0] qkv_weight_type,
    input  wire qkv_weight_ack,
    input  wire qkv_weight_valid,
    input  wire [G_OUT*T_OUT*EXP_WIDTH-1:0] qkv_weight_exp_array,
    input  wire [G_OUT*T_OUT*TOTAL_WIDTH-1:0] qkv_weight_mant_blocks,
    
    // W_O权重（Output Projection）
    output wire wo_weight_req,
    input  wire wo_weight_ready,
    input  wire [DIM*EXP_WIDTH-1:0] wo_weight_exp_array,
    input  wire [DIM*DIM*DATA_WIDTH-1:0] wo_weight_mant,
    
    //===========================================================================
    // Result Buffer接口
    //===========================================================================
    output wire result_wr_en,
    output wire [9:0] result_wr_addr,
    output wire [EXP_WIDTH-1:0] result_exp,
    output wire [DIM*DATA_WIDTH-1:0] result_mant,
    
    //===========================================================================
    // 调试接口
    //===========================================================================
    output wire [4:0] dbg_current_batch,
    output wire [3:0] dbg_batch_ctrl_state,
    output wire [3:0] dbg_heads_done,
    output wire [3:0] dbg_heads_busy,
    output wire dbg_first_batch,
    output wire [31:0] dbg_cycle_count,
    output wire [31:0] dbg_batch_cycle_count,
    output wire [31:0] dbg_accum_wr_count_h0,
    output wire [31:0] dbg_accum_wr_count_h1,
    output wire [31:0] dbg_accum_wr_count_h2,
    output wire [31:0] dbg_accum_wr_count_h3,
    output wire [3:0] dbg_output_proj_state,
    output wire dbg_output_proj_busy,
    output wire dbg_output_proj_done,
    
    // KV Cache valid信号调试接口
    output wire [3:0] dbg_k_rd_valid,
    output wire [3:0] dbg_v_rd_valid,
    output wire dbg_kv_cache_ready,
    output wire [1:0] dbg_kv_cache_status
);

//===================================================================================
// 内部信号：Batch Controller ↔ QKV Compute
//===================================================================================
wire qkv_start;
wire [1:0] qkv_compute_mode;
wire [5:0] qkv_batch_id;
wire [4:0] qkv_tokens_in_batch;
wire qkv_done;
wire qkv_busy;

//===================================================================================
// 内部信号：Batch Controller ↔ Head Engines
//===================================================================================
wire [NUM_HEADS-1:0] heads_start;
wire [NUM_HEADS-1:0] heads_done;
wire [NUM_HEADS-1:0] heads_busy;

//===================================================================================
// 内部信号：Batch Controller ↔ Output Projection
//===================================================================================
wire output_proj_start;
wire output_proj_done;
wire output_proj_busy;

//===================================================================================
// 内部信号：QKV Compute → Q Storage
//===================================================================================
wire q_storage_wr_en;
wire [1:0] q_storage_wr_head;
wire [4:0] q_storage_wr_token;
wire [EXP_WIDTH-1:0] q_storage_wr_exp;
wire [HEAD_DIM*DATA_WIDTH-1:0] q_storage_wr_mant;

//===================================================================================
// 内部信号：QKV Compute → KV Cache
//===================================================================================
wire kv_cache_wr_en;
wire kv_cache_wr_type;
wire [1:0] kv_cache_wr_head;
wire [9:0] kv_cache_wr_token;
wire [EXP_WIDTH-1:0] kv_cache_wr_exp;
wire [HEAD_DIM*DATA_WIDTH-1:0] kv_cache_wr_mant;

wire kv_cache_ready;
wire [1:0] kv_cache_status;

//===================================================================================
// 内部信号：Q Storage → Head Engines
//===================================================================================
wire [NUM_HEADS-1:0] q_rd_en;
wire [(NUM_HEADS*TOKEN_BATCH*EXP_WIDTH)-1:0] q_batch_exp_flat;
wire [(NUM_HEADS*TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] q_batch_mant_flat;

//===================================================================================
// 内部信号：KV Cache → Head Engines（包含valid信号）
//===================================================================================
// K读取信号
wire [NUM_HEADS-1:0] k_rd_en;
wire [(NUM_HEADS*5)-1:0] k_rd_chunk_flat;
wire [NUM_HEADS-1:0] k_rd_valid; 
wire [(NUM_HEADS*CHUNK_SIZE*EXP_WIDTH)-1:0] k_chunk_exp_flat;
wire [(NUM_HEADS*CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)-1:0] k_chunk_mant_flat;

// V读取信号
wire [NUM_HEADS-1:0] v_rd_en;
wire [(NUM_HEADS*5)-1:0] v_rd_chunk_flat;
wire [NUM_HEADS-1:0] v_rd_valid;  // V读取valid信号
wire [(NUM_HEADS*CHUNK_SIZE*EXP_WIDTH)-1:0] v_chunk_exp_flat;
wire [(NUM_HEADS*CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)-1:0] v_chunk_mant_flat;

//===================================================================================
// 内部信号：Output Projection → Head Engines Accumulators
//===================================================================================
wire [NUM_HEADS-1:0] accum_rd_en;
wire [(NUM_HEADS*5)-1:0] accum_rd_row_flat;
wire [(NUM_HEADS*3)-1:0] accum_rd_dim_flat;
wire signed [(NUM_HEADS*ACCUM_WIDTH)-1:0] accum_rd_mant_flat;
wire [(NUM_HEADS*EXP_WIDTH)-1:0] accum_rd_exp_flat;
wire [NUM_HEADS-1:0] accum_rd_valid;

//===================================================================================
// 内部信号：调试
//===================================================================================
wire [4:0] current_batch;
wire first_batch_flag;
wire [3:0] batch_ctrl_state;
wire [31:0] batch_cycle_count_internal;
// 改为packed array避免generate块综合错误
wire [(NUM_HEADS*32)-1:0] accum_wr_count_flat;

assign dbg_current_batch = current_batch;
assign dbg_batch_ctrl_state = batch_ctrl_state;
assign dbg_heads_done = heads_done;
assign dbg_heads_busy = heads_busy;
assign dbg_first_batch = first_batch_flag;
assign dbg_batch_cycle_count = batch_cycle_count_internal;
// 使用packed array的位选择
assign dbg_accum_wr_count_h0 = accum_wr_count_flat[0*32 +: 32];
assign dbg_accum_wr_count_h1 = accum_wr_count_flat[1*32 +: 32];
assign dbg_accum_wr_count_h2 = accum_wr_count_flat[2*32 +: 32];
assign dbg_accum_wr_count_h3 = accum_wr_count_flat[3*32 +: 32];
assign dbg_output_proj_busy = output_proj_busy;
assign dbg_output_proj_done = output_proj_done;

// KV Cache相关调试信号
assign dbg_k_rd_valid = k_rd_valid;
assign dbg_v_rd_valid = v_rd_valid;
assign dbg_kv_cache_ready = kv_cache_ready;
assign dbg_kv_cache_status = kv_cache_status;

//===================================================================================
// 模块实例化 1: Batch Controller（顶层控制器）
//===================================================================================
attention_batch_controller #(
    .NUM_BATCHES(NUM_BATCHES),
    .NUM_HEADS(NUM_HEADS),
    .TOKEN_BATCH(TOKEN_BATCH)
) u_batch_controller (
    .clk(clk),
    .rst_n(rst_n),
    
    // 顶层控制
    .start(start),
    .done(done),
    .busy(busy),
    
    // QKV引擎控制
    .qkv_start(qkv_start),
    .qkv_compute_mode(qkv_compute_mode),
    .qkv_batch_id(qkv_batch_id),
    .qkv_tokens_in_batch(qkv_tokens_in_batch),
    .qkv_done(qkv_done),
    .qkv_busy(qkv_busy),
    
    // Head引擎控制
    .heads_start(heads_start),
    .heads_done(heads_done),
    .heads_busy(heads_busy),
    
    // Output Projection控制
    .output_proj_start(output_proj_start),
    .output_proj_done(output_proj_done),
    .output_proj_busy(output_proj_busy),
    
    // 状态输出
    .current_batch(current_batch),
    .first_batch_flag(first_batch_flag),
    
    // 调试
    .current_state(batch_ctrl_state),
    .cycle_count(dbg_cycle_count),
    .batch_cycle_count(batch_cycle_count_internal)
);

//===================================================================================
// 模块实例化 2: QKV Compute Engine
//===================================================================================
qkv_compute_engine #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .DIM(DIM),
    .NUM_HEADS(NUM_HEADS),
    .HEAD_DIM(HEAD_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .MANT_WIDTH(MANT_WIDTH),
    .G_OUT(G_OUT),
    .T_OUT(T_OUT),
    .TOTAL_ELEM(TOTAL_ELEM),
    .TOTAL_WIDTH(TOTAL_WIDTH),
    .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
    .CE_BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH)
) u_qkv_compute (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制接口
    .start(qkv_start),
    .compute_mode(qkv_compute_mode),
    .batch_id(qkv_batch_id),
    .tokens_in_batch(qkv_tokens_in_batch),
    .enable_shared_exp(1'b1),
    .done(qkv_done),
    .busy(qkv_busy),
    
    // Token读取接口
    .token_rd_en(token_rd_en),
    .token_rd_addr(token_rd_addr),
    .token_rd_exp(token_rd_exp),
    .token_rd_mant(token_rd_mant),
    
    // 权重读取接口
    .weight_req(qkv_weight_req),
    .weight_type(qkv_weight_type),
    .weight_ack(qkv_weight_ack),
    .weight_valid(qkv_weight_valid),
    .weight_exp_array(qkv_weight_exp_array),
    .weight_mant_blocks(qkv_weight_mant_blocks),
    
    // Q矩阵存储接口
    .storage_wr_en(q_storage_wr_en),
    .storage_matrix_type(),
    .storage_head_id(q_storage_wr_head),
    .storage_token_id(q_storage_wr_token),
    .storage_shared_exp(q_storage_wr_exp),
    .storage_mant_packed(q_storage_wr_mant),
    .storage_overflow_flag(),
    
    // KV Cache接口
    .kv_wr_en(kv_cache_wr_en),
    .kv_wr_type(kv_cache_wr_type),
    .kv_wr_head(kv_cache_wr_head),
    .kv_wr_token(kv_cache_wr_token),
    .kv_wr_exp(kv_cache_wr_exp),
    .kv_wr_mant(kv_cache_wr_mant),
    
    // 状态输出
    .all_kv_written(),
    .error()
);

//===================================================================================
// 模块实例化 3: Q Matrix Storage（4-Bank并行）
//===================================================================================
q_matrix_storage #(
    .NUM_HEADS(NUM_HEADS),
    .TOKEN_BATCH(TOKEN_BATCH),
    .HEAD_DIM(HEAD_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH)
) u_q_storage (
    .clk(clk),
    .rst_n(rst_n),
    
    // 写入接口
    .wr_en(q_storage_wr_en),
    .wr_head(q_storage_wr_head),
    .wr_token(q_storage_wr_token),
    .wr_exp(q_storage_wr_exp),
    .wr_mant_packed(q_storage_wr_mant),
    
    // ✅ 4个并行读接口 - 使用packed array
    .rd_en_h0(q_rd_en[0]),
    .rd_batch_exp_h0(q_batch_exp_flat[0*(TOKEN_BATCH*EXP_WIDTH) +: (TOKEN_BATCH*EXP_WIDTH)]),
    .rd_batch_mant_h0(q_batch_mant_flat[0*(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH) +: (TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)]),
    
    .rd_en_h1(q_rd_en[1]),
    .rd_batch_exp_h1(q_batch_exp_flat[1*(TOKEN_BATCH*EXP_WIDTH) +: (TOKEN_BATCH*EXP_WIDTH)]),
    .rd_batch_mant_h1(q_batch_mant_flat[1*(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH) +: (TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)]),
    
    .rd_en_h2(q_rd_en[2]),
    .rd_batch_exp_h2(q_batch_exp_flat[2*(TOKEN_BATCH*EXP_WIDTH) +: (TOKEN_BATCH*EXP_WIDTH)]),
    .rd_batch_mant_h2(q_batch_mant_flat[2*(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH) +: (TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)]),
    
    .rd_en_h3(q_rd_en[3]),
    .rd_batch_exp_h3(q_batch_exp_flat[3*(TOKEN_BATCH*EXP_WIDTH) +: (TOKEN_BATCH*EXP_WIDTH)]),
    .rd_batch_mant_h3(q_batch_mant_flat[3*(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH) +: (TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)])
);

//===================================================================================
// ✅ 模块实例化 4: KV Cache Manager（连接所有valid信号）
//===================================================================================
kv_cache_manager #(
    .NUM_HEADS(NUM_HEADS),
    .TOTAL_TOKENS(TOKEN_NUM),
    .HEAD_DIM(HEAD_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .CHUNK_SIZE(CHUNK_SIZE),
    .NUM_CHUNKS(NUM_CHUNKS)
) u_kv_cache (
    .clk(clk),
    .rst_n(rst_n),
    
    // 写入接口
    .wr_en(kv_cache_wr_en),
    .wr_type(kv_cache_wr_type),
    .wr_head(kv_cache_wr_head),
    .wr_token(kv_cache_wr_token),
    .wr_exp(kv_cache_wr_exp),
    .wr_mant_packed(kv_cache_wr_mant),
    
    // ✅ Head 0 K/V读取 - 使用packed array
    .k_rd_en_h0(k_rd_en[0]),
    .k_rd_chunk_h0(k_rd_chunk_flat[0*5 +: 5]),
    .k_rd_valid_h0(k_rd_valid[0]),
    .k_chunk_exp_h0(k_chunk_exp_flat[0*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
    .k_chunk_mant_h0(k_chunk_mant_flat[0*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
    
    .v_rd_en_h0(v_rd_en[0]),
    .v_rd_chunk_h0(v_rd_chunk_flat[0*5 +: 5]),
    .v_rd_valid_h0(v_rd_valid[0]),
    .v_chunk_exp_h0(v_chunk_exp_flat[0*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
    .v_chunk_mant_h0(v_chunk_mant_flat[0*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
    
    // ✅ Head 1 K/V读取 - 使用packed array
    .k_rd_en_h1(k_rd_en[1]),
    .k_rd_chunk_h1(k_rd_chunk_flat[1*5 +: 5]),
    .k_rd_valid_h1(k_rd_valid[1]),
    .k_chunk_exp_h1(k_chunk_exp_flat[1*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
    .k_chunk_mant_h1(k_chunk_mant_flat[1*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
    
    .v_rd_en_h1(v_rd_en[1]),
    .v_rd_chunk_h1(v_rd_chunk_flat[1*5 +: 5]),
    .v_rd_valid_h1(v_rd_valid[1]),
    .v_chunk_exp_h1(v_chunk_exp_flat[1*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
    .v_chunk_mant_h1(v_chunk_mant_flat[1*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
    
    // ✅ Head 2 K/V读取 - 使用packed array
    .k_rd_en_h2(k_rd_en[2]),
    .k_rd_chunk_h2(k_rd_chunk_flat[2*5 +: 5]),
    .k_rd_valid_h2(k_rd_valid[2]),
    .k_chunk_exp_h2(k_chunk_exp_flat[2*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
    .k_chunk_mant_h2(k_chunk_mant_flat[2*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
    
    .v_rd_en_h2(v_rd_en[2]),
    .v_rd_chunk_h2(v_rd_chunk_flat[2*5 +: 5]),
    .v_rd_valid_h2(v_rd_valid[2]),
    .v_chunk_exp_h2(v_chunk_exp_flat[2*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
    .v_chunk_mant_h2(v_chunk_mant_flat[2*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
    
    // ✅ Head 3 K/V读取 - 使用packed array
    .k_rd_en_h3(k_rd_en[3]),
    .k_rd_chunk_h3(k_rd_chunk_flat[3*5 +: 5]),
    .k_rd_valid_h3(k_rd_valid[3]),
    .k_chunk_exp_h3(k_chunk_exp_flat[3*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
    .k_chunk_mant_h3(k_chunk_mant_flat[3*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
    
    .v_rd_en_h3(v_rd_en[3]),
    .v_rd_chunk_h3(v_rd_chunk_flat[3*5 +: 5]),
    .v_rd_valid_h3(v_rd_valid[3]),
    .v_chunk_exp_h3(v_chunk_exp_flat[3*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
    .v_chunk_mant_h3(v_chunk_mant_flat[3*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
    
    // 状态
    .kv_ready(kv_cache_ready),
    .kv_status(kv_cache_status)
);

//===================================================================================
// ✅ 模块实例化 5: 4个并行的Single Head Engine（使用valid信号）
//===================================================================================
genvar h;
generate
    for (h = 0; h < 4; h = h + 1) begin: gen_head_engines
        
        single_head_engine #(
            .HEAD_ID(h),
            .NUM_CHUNKS(NUM_CHUNKS),
            .TOKEN_BATCH(TOKEN_BATCH),
            .CHUNK_SIZE(CHUNK_SIZE),
            .HEAD_DIM(HEAD_DIM),
            .DATA_WIDTH(DATA_WIDTH),
            .EXP_WIDTH(EXP_WIDTH),
            .SCORE_WIDTH(SCORE_WIDTH),
            .ACCUM_WIDTH(ACCUM_WIDTH),
            .NUM_PE(1),
            .PE_TYPE_0(2),
            .PE_TYPE_1(2),
            .ELEM_PE0(8),
            .ELEM_PE1(8),
            .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH)
        ) u_head_engine (
            .clk(clk),
            .rst_n(rst_n),
            
            // 控制接口
            .start(heads_start[h]),
            .done(heads_done[h]),
            .busy(heads_busy[h]),
            
            // ✅ Q读取接口 - 使用packed array的位选择
            .q_rd_en(q_rd_en[h]),
            .q_batch_exp(q_batch_exp_flat[h*(TOKEN_BATCH*EXP_WIDTH) +: (TOKEN_BATCH*EXP_WIDTH)]),
            .q_batch_mant(q_batch_mant_flat[h*(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH) +: (TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)]),
            
            // ✅ K读取接口 - 使用packed array的位选择
            .k_rd_en(k_rd_en[h]),
            .k_rd_chunk_id(k_rd_chunk_flat[h*5 +: 5]),
            .k_rd_valid(k_rd_valid[h]),
            .k_chunk_exp(k_chunk_exp_flat[h*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
            .k_chunk_mant(k_chunk_mant_flat[h*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
            
            // ✅ V读取接口 - 使用packed array的位选择
            .v_rd_en(v_rd_en[h]),
            .v_rd_chunk_id(v_rd_chunk_flat[h*5 +: 5]),
            .v_rd_valid(v_rd_valid[h]),
            .v_chunk_exp(v_chunk_exp_flat[h*(CHUNK_SIZE*EXP_WIDTH) +: (CHUNK_SIZE*EXP_WIDTH)]),
            .v_chunk_mant(v_chunk_mant_flat[h*(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH) +: (CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)]),
            
            // ✅ 内部Accumulator读取接口 - 使用packed array的位选择
            .final_rd_en(accum_rd_en[h]),
            .final_rd_row(accum_rd_row_flat[h*5 +: 5]),
            .final_rd_dim(accum_rd_dim_flat[h*3 +: 3]),
            .final_rd_mant(accum_rd_mant_flat[h*ACCUM_WIDTH +: ACCUM_WIDTH]),
            .final_rd_exp(accum_rd_exp_flat[h*EXP_WIDTH +: EXP_WIDTH]),
            .final_rd_valid(accum_rd_valid[h]),
            
            // 调试
            .current_chunk_id(),
            .dbg_accum_wr_count(accum_wr_count_flat[h*32 +: 32])
        );
        
    end
endgenerate

//===================================================================================
// 模块实例化 6: Output Projection
//===================================================================================
output_projection #(
    .NUM_HEADS(NUM_HEADS),
    .TOKEN_BATCH(TOKEN_BATCH),
    .HEAD_DIM(HEAD_DIM),
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ACCUM_WIDTH(ACCUM_WIDTH),
    .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
    .CE_INTERNAL_WIDTH(CE_INTERNAL_WIDTH),
    .CE_GUARD_BITS(CE_GUARD_BITS),
    .CE_ENABLE_ROUNDING(CE_ENABLE_ROUNDING)
) u_output_projection (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制接口
    .start(output_proj_start),
    .batch_id(current_batch),
    .tokens_in_batch(qkv_tokens_in_batch),
    .done(output_proj_done),
    .busy(output_proj_busy),
    
    // ✅ Head 0 Accumulator读取 - 使用packed array的位选择
    .accum_rd_en_h0(accum_rd_en[0]),
    .accum_rd_row_h0(accum_rd_row_flat[0*5 +: 5]),
    .accum_rd_dim_h0(accum_rd_dim_flat[0*3 +: 3]),
    .accum_rd_mant_h0(accum_rd_mant_flat[0*ACCUM_WIDTH +: ACCUM_WIDTH]),
    .accum_rd_exp_h0(accum_rd_exp_flat[0*EXP_WIDTH +: EXP_WIDTH]),
    .accum_rd_valid_h0(accum_rd_valid[0]),
    
    // ✅ Head 1 Accumulator读取 - 使用packed array的位选择
    .accum_rd_en_h1(accum_rd_en[1]),
    .accum_rd_row_h1(accum_rd_row_flat[1*5 +: 5]),
    .accum_rd_dim_h1(accum_rd_dim_flat[1*3 +: 3]),
    .accum_rd_mant_h1(accum_rd_mant_flat[1*ACCUM_WIDTH +: ACCUM_WIDTH]),
    .accum_rd_exp_h1(accum_rd_exp_flat[1*EXP_WIDTH +: EXP_WIDTH]),
    .accum_rd_valid_h1(accum_rd_valid[1]),
    
    // ✅ Head 2 Accumulator读取 - 使用packed array的位选择
    .accum_rd_en_h2(accum_rd_en[2]),
    .accum_rd_row_h2(accum_rd_row_flat[2*5 +: 5]),
    .accum_rd_dim_h2(accum_rd_dim_flat[2*3 +: 3]),
    .accum_rd_mant_h2(accum_rd_mant_flat[2*ACCUM_WIDTH +: ACCUM_WIDTH]),
    .accum_rd_exp_h2(accum_rd_exp_flat[2*EXP_WIDTH +: EXP_WIDTH]),
    .accum_rd_valid_h2(accum_rd_valid[2]),
    
    // ✅ Head 3 Accumulator读取 - 使用packed array的位选择
    .accum_rd_en_h3(accum_rd_en[3]),
    .accum_rd_row_h3(accum_rd_row_flat[3*5 +: 5]),
    .accum_rd_dim_h3(accum_rd_dim_flat[3*3 +: 3]),
    .accum_rd_mant_h3(accum_rd_mant_flat[3*ACCUM_WIDTH +: ACCUM_WIDTH]),
    .accum_rd_exp_h3(accum_rd_exp_flat[3*EXP_WIDTH +: EXP_WIDTH]),
    .accum_rd_valid_h3(accum_rd_valid[3]),
    
    // W_O权重接口
    .weight_req(wo_weight_req),
    .weight_ready(wo_weight_ready),
    .weight_exp_array(wo_weight_exp_array),
    .weight_mant(wo_weight_mant),
    
    // 结果输出接口
    .result_wr_en(result_wr_en),
    .result_wr_addr(result_wr_addr),
    .result_exp(result_exp),
    .result_mant(result_mant),
    
    // 调试
    .dbg_state(dbg_output_proj_state)
);

//===================================================================================
// 初始化信息
//===================================================================================
initial begin
    $display("========================================");
    $display("Backbone Attention Parallel Top v5.4");
    $display("========================================");
    $display("Key improvements:");
    $display("  ✅ Fixed synthesis errors by converting unpacked to packed arrays");
    $display("  ✅ All array indices resolved at compile time");
    $display("  ✅ KV Cache valid signals connected");
    $display("  ✅ Head Engines use valid for handshake");
    $display("  ✅ Robust variable-latency support");
    $display("  ✅ Industry-standard flow control");
    $display("========================================");
end

endmodule