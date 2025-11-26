`timescale 1ns / 1ps

//================================================================================
// SideNet Adaptation Transformer v2 - 适配器变换层
//
// 主要变更 (v2):
// - 移除FFN DRAM接口
// - 新增FFN权重接口（W1和W2）
// - FFN权重通过weight_controller统一管理
//
// 功能：
// 处理SideNet标准层（Layer 1-3）的Adaptation Transformer计算
// 
// 架构：
// 1. 层内存储：result_buffer_single_port (复用)
// 2. 计算模块：Attention, FFN, Residual x2, LayerNorm x2
// 3. 控制器：transformer_block_ctrl_fsm（复用）
//
// 数据流：
// Gated Buffer(外部) → Attention → Result Buffer
//                    → Residual 1 → Result Buffer
//                    → LayerNorm 1 → Result Buffer + LN1暂存(外部)
//                    → FFN → Result Buffer
//                    → Residual 2 → Result Buffer
//                    → LayerNorm 2 → Adapted Buffer(外部)
//
// 与Backbone Transformer的差异：
// - DIM: 32 → 8
// - NUM_HEADS: 4 → 4（保持）
// - HEAD_DIM: 8 → 2
// - D_FF: 128 → 32
// - Token数量：641（不变）
//
// 注意：
// - Adapted Buffer在外部（层间存储）
// - 本模块只负责单层计算，使用外部提供的读写接口
// - 支持时分复用（Layer 1-3共用一个实例）
//
// 作者: MEIGA Team
// 日期: 2025-11-16
// 版本: v2
//================================================================================

module sidenet_adaptation_transformer#(
    // ========== 网络参数 ==========
    parameter TOKEN_NUM       = 640,
    parameter TOKEN_BATCH     = 32,
    parameter BATCH_NUM       = 20,
    parameter DIM             = 8,
    parameter NUM_HEADS       = 4,
    parameter HEAD_DIM        = 8,
    parameter D_FF            = 128,
    parameter FEATURE_CHUNK   = 32,
    
    // ========== 数据参数 ==========
    parameter DATA_WIDTH      = 16,
    parameter EXP_WIDTH       = 8,
    parameter ADDR_WIDTH      = 10,
    
    // ========== Attention参数 ==========
    parameter K_CHUNK_SIZE    = 32,
    parameter K_CHUNK_NUM     = 20,
    parameter SCORE_WIDTH     = 16,
    parameter ACCUM_WIDTH     = 24,
    
    // ========== Compute Engine参数 ==========
    parameter NUM_PE          = 2,
    parameter PE_TYPE_0       = 2,
    parameter PE_TYPE_1       = 2,
    parameter ELEM_PE0        = 4,
    parameter ELEM_PE1        = 4,
    parameter CE_OUTPUT_WIDTH = 32,
    parameter CE_INTERNAL_WIDTH = 39,
    parameter CE_GUARD_BITS   = 7,
    parameter CE_ENABLE_ROUNDING = 1,
    
    // ========== QKV Compute参数 ==========
    parameter G_OUT           = 4,
    parameter T_OUT           = 2,
    parameter TOTAL_ELEM      = 8,
    parameter TOTAL_WIDTH     = 256,
    parameter CE_BASE_EXP_WIDTH = 9,
    
    // ========== LN1 Buffer参数 ==========
    parameter LN1_TOKEN_NUM   = 32,        // LN1 buffer容量：1个batch
    parameter LN1_ADDR_WIDTH  = 5          // log2(32) = 5
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire start,                     // 开始处理
    input  wire [1:0] layer_id,            // 层ID
    output wire done,                      // 完成
    output wire busy,
    
    //================================================================================
    // Layer Token Buffer接口（读，来自外部层间存储）
    //================================================================================
    output wire layer_token_rd_en,
    output wire [ADDR_WIDTH-1:0] layer_token_rd_addr,
    input  wire [EXP_WIDTH-1:0] layer_token_rd_exp,
    input  wire [DIM*DATA_WIDTH-1:0] layer_token_rd_mant,
    input  wire layer_token_rd_valid,
    
    //================================================================================
    // Layer Token Buffer接口（写，到外部层间存储）
    //================================================================================
    output wire layer_token_wr_en,
    output wire [ADDR_WIDTH-1:0] layer_token_wr_addr,
    output wire [EXP_WIDTH-1:0] layer_token_wr_exp,
    output wire [DIM*DATA_WIDTH-1:0] layer_token_wr_mant,
    
    
    //================================================================================
    // QKV权重接口
    //================================================================================
    output wire qkv_weight_req,
    output wire [1:0] qkv_weight_type,     // 00=Q, 01=K, 10=V
    input  wire qkv_weight_ack,            // 握手ACK
    input  wire qkv_weight_valid,          // 数据有效
    input  wire [DIM*EXP_WIDTH-1:0] qkv_weight_exp_array,       // 32个指数
    input  wire [DIM*DIM*DATA_WIDTH-1:0] qkv_weight_mant_blocks, // 32×32尾数
    
    //================================================================================
    // WO权重接口（Output Projection - 新）
    //================================================================================
    output wire wo_weight_req,
    input  wire wo_weight_ready,
    input  wire [DIM*EXP_WIDTH-1:0] wo_weight_exp_array,
    input  wire [DIM*DIM*DATA_WIDTH-1:0] wo_weight_mant,
    
    //================================================================================
    // FFN权重接口
    //================================================================================
    output wire ffn_weight_req,
    output wire [1:0] ffn_weight_type,     // 00=W1, 01=W2
    output wire [1:0] ffn_weight_chunk_id, // 0-3
    input  wire ffn_weight_ready,
    input  wire [DIM*EXP_WIDTH-1:0] ffn_weight_exp_array,       // 32个指数
    input  wire [DIM*FEATURE_CHUNK*DATA_WIDTH-1:0] ffn_weight_mant,
    
    //================================================================================
    // LayerNorm参数接口（修改）
    //================================================================================
    output wire ln_param_req,
    output wire [1:0] ln_param_type,       // 00=LN1_gamma, 01=LN1_beta, 10=LN2_gamma, 11=LN2_beta
    input  wire [EXP_WIDTH-1:0] ln_param_exp,
    input  wire [DIM*DATA_WIDTH-1:0] ln_param_mant,
    input  wire ln_param_valid,
    
    //================================================================================
    // 调试接口
    //================================================================================
    output wire [3:0] dbg_fsm_state,
    output wire [3:0] dbg_att_state,
    output wire [4:0] dbg_att_batch,
    output wire [3:0] dbg_ffn_state,
    output wire [31:0] dbg_result_rd_count,
    output wire [31:0] dbg_result_wr_count,
    output wire [31:0] dbg_ln1_rd_count,
    output wire [31:0] dbg_ln1_wr_count
);

//================================================================================
// 内部信号
//================================================================================

// FSM控制
wire attention_start, attention_done, attention_busy;
wire ffn_start, ffn_done, ffn_busy;
wire residual1_start, residual1_done;
wire layernorm1_start, layernorm1_done;
wire residual2_start, residual2_done;
wire layernorm2_start, layernorm2_done;

// Result Buffer
wire result_rd_en, result_rd_valid;
wire [ADDR_WIDTH-1:0] result_rd_addr;
wire [EXP_WIDTH-1:0] result_rd_exp;
wire [DIM*DATA_WIDTH-1:0] result_rd_mant;

wire result_wr_en;
wire [ADDR_WIDTH-1:0] result_wr_addr;
wire [EXP_WIDTH-1:0] result_wr_exp;
wire [DIM*DATA_WIDTH-1:0] result_wr_mant;

// LN1 Result Buffer
wire ln1_buffer_rd_en, ln1_buffer_rd_valid;
wire [LN1_ADDR_WIDTH-1:0] ln1_buffer_rd_addr;
wire [EXP_WIDTH-1:0] ln1_buffer_rd_exp;
wire [DIM*DATA_WIDTH-1:0] ln1_buffer_rd_mant;

wire ln1_buffer_wr_en;
wire [LN1_ADDR_WIDTH-1:0] ln1_buffer_wr_addr;
wire [EXP_WIDTH-1:0] ln1_buffer_wr_exp;
wire [DIM*DATA_WIDTH-1:0] ln1_buffer_wr_mant;

// Attention
wire att_token_rd_en;
wire [ADDR_WIDTH-1:0] att_token_rd_addr;
wire att_result_wr_en;
wire [ADDR_WIDTH-1:0] att_result_wr_addr;
wire [EXP_WIDTH-1:0] att_result_exp;
wire [DIM*DATA_WIDTH-1:0] att_result_mant;

// Residual 1
wire res1_input_rd_en;
wire [ADDR_WIDTH-1:0] res1_input_rd_addr;
wire res1_result_rd_en;
wire [ADDR_WIDTH-1:0] res1_result_rd_addr;
wire res1_output_wr_en;
wire [ADDR_WIDTH-1:0] res1_output_wr_addr;
wire [EXP_WIDTH-1:0] res1_output_exp;
wire [DIM*DATA_WIDTH-1:0] res1_output_mant;
wire res1_output_valid;

// LayerNorm 1
wire ln1_input_rd_en;
wire [ADDR_WIDTH-1:0] ln1_input_rd_addr;
wire ln1_rb_output_wr_en;         // 写Result Buffer的使能
wire [ADDR_WIDTH-1:0] ln1_rb_output_wr_addr;
wire [EXP_WIDTH-1:0] ln1_rb_output_exp;
wire [DIM*DATA_WIDTH-1:0] ln1_rb_output_mant;
wire ln1_output_valid;
wire ln1_param_req_internal;
wire ln1_param_gamma_internal;
// Residual 1 调试信号
wire res1_busy;
wire res1_error;
wire [3:0] res1_state;
wire [9:0] res1_processed_count;
wire [31:0] res1_cycle_count;
wire res1_overflow_detected;
wire res1_underflow_detected;
// Residual 2 调试信号
wire res2_busy;
wire res2_error;
wire [3:0] res2_state;
wire [9:0] res2_processed_count;
wire [31:0] res2_cycle_count;
wire res2_overflow_detected;
wire res2_underflow_detected;
// FFN (Updated Interface)
wire ffn_rb_rd_en;
wire [ADDR_WIDTH-1:0] ffn_rb_rd_addr;
wire ffn_rb_wr_en;
wire [ADDR_WIDTH-1:0] ffn_rb_wr_addr;
wire [EXP_WIDTH-1:0] ffn_rb_wr_exp;
wire [DIM*DATA_WIDTH-1:0] ffn_rb_wr_mant;

// Residual 2
wire res2_result_rd_en;
wire [ADDR_WIDTH-1:0] res2_result_rd_addr;
wire res2_output_wr_en;
wire [ADDR_WIDTH-1:0] res2_output_wr_addr;
wire [EXP_WIDTH-1:0] res2_output_exp;
wire [DIM*DATA_WIDTH-1:0] res2_output_mant;
wire res2_output_valid;

// LayerNorm 2
wire ln2_input_rd_en;
wire [ADDR_WIDTH-1:0] ln2_input_rd_addr;
wire ln2_param_req_internal;
wire ln2_param_gamma_internal;

//================================================================================
// 模块例化
//================================================================================
//========== 1. 控制FSM ==========
transformer_block_ctrl_fsm #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .DIM(DIM),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_ctrl_fsm (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制接口
    .start(start),
    .done(done),
    .busy(busy),
    
    // Attention控制
    .attention_start(attention_start),
    .attention_done(attention_done),
    .attention_busy(attention_busy),
    
    // Residual1控制
    .residual1_start(residual1_start),
    .residual1_done(residual1_done),
    
    // LayerNorm1控制
    .layernorm1_start(layernorm1_start),
    .layernorm1_done(layernorm1_done),
    
    // FFN控制
    .ffn_start(ffn_start),
    .ffn_done(ffn_done),
    .ffn_busy(ffn_busy),
    
    // Residual2控制
    .residual2_start(residual2_start),
    .residual2_done(residual2_done),
    
    // LayerNorm2控制
    .layernorm2_start(layernorm2_start),
    .layernorm2_done(layernorm2_done),
    
    // Buffer控制信号（FSM定义但未使用，保留未连接）
    .bank_swap_req(),          // 在DONE_STATE输出，但backbone_transformer内部不需要
    .porta_rd_req(),           // 未使用
    .porta_rd_addr(),          // 未使用
    .ln1_save_req(),           // 未使用
    .ln1_save_addr(),          // 未使用
    .ln1_load_req(),           // 未使用
    .ln1_load_addr(),          // 未使用
    .layer_out_wr_req(),       // 未使用
    .layer_out_wr_addr(),      // 未使用
    .result_rd_req(),          // 未使用
    .result_rd_addr(),         // 未使用
    .result_wr_req(),          // 未使用
    .result_wr_addr(),         // 未使用
    
    // 调试接口
    .dbg_state(dbg_fsm_state),
    .dbg_token_idx()           // 可选调试信号，未连接
);
//========== 2.1. Result Buffer ==========
result_buffer_single_port #(
    .TOKEN_NUM(TOKEN_NUM),
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_result_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    .rd_en(result_rd_en),
    .rd_addr(result_rd_addr),
    .rd_exp(result_rd_exp),
    .rd_mant(result_rd_mant),
    .rd_valid(result_rd_valid),
    
    .wr_en(result_wr_en),
    .wr_addr(result_wr_addr),
    .wr_exp(result_wr_exp),
    .wr_mant(result_wr_mant),
    
    .dbg_rd_count(dbg_result_rd_count),
    .dbg_wr_count(dbg_result_wr_count)
);

//========== 2.2. LN1 Result Buffer ==========
ln1_result_buffer #(
    .TOKEN_NUM(LN1_TOKEN_NUM),
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(LN1_ADDR_WIDTH)
) u_ln1_result_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    // 读端口：供Residual2使用
    .rd_en(ln1_buffer_rd_en),
    .rd_addr(ln1_buffer_rd_addr),
    .rd_exp(ln1_buffer_rd_exp),
    .rd_mant(ln1_buffer_rd_mant),
    .rd_valid(ln1_buffer_rd_valid),
    
    // 写端口：来自LayerNorm1
    .wr_en(ln1_buffer_wr_en),
    .wr_addr(ln1_buffer_wr_addr),
    .wr_exp(ln1_buffer_wr_exp),
    .wr_mant(ln1_buffer_wr_mant),
    
    .dbg_rd_count(dbg_ln1_rd_count),
    .dbg_wr_count(dbg_ln1_wr_count)
);

//========== 3. Attention模块 ==========
backbone_attention_parallel_top #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .NUM_BATCHES(BATCH_NUM),
    .DIM(DIM),
    .NUM_HEADS(NUM_HEADS),
    .HEAD_DIM(HEAD_DIM),
    .CHUNK_SIZE(K_CHUNK_SIZE),
    .NUM_CHUNKS(K_CHUNK_NUM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .MANT_WIDTH(DATA_WIDTH),
    .SCORE_WIDTH(SCORE_WIDTH),
    .ACCUM_WIDTH(ACCUM_WIDTH),
    .NUM_PE(NUM_PE),
    .PE_TYPE_0(PE_TYPE_0),
    .PE_TYPE_1(PE_TYPE_1),
    .ELEM_PE0(ELEM_PE0),
    .ELEM_PE1(ELEM_PE1),
    .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
    .CE_INTERNAL_WIDTH(CE_INTERNAL_WIDTH),
    .CE_GUARD_BITS(CE_GUARD_BITS),
    .CE_ENABLE_ROUNDING(CE_ENABLE_ROUNDING),
    .G_OUT(G_OUT),
    .T_OUT(T_OUT),
    .TOTAL_ELEM(TOTAL_ELEM),
    .TOTAL_WIDTH(TOTAL_WIDTH),
    .CE_BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH)
) u_attention (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制接口
    .start(attention_start),
    .done(attention_done),
    .busy(attention_busy),
    
    // Token输入接口（从Layer Token Buffer读）
    .token_rd_en(att_token_rd_en),
    .token_rd_addr(att_token_rd_addr),
    .token_rd_exp(layer_token_rd_exp),
    .token_rd_mant(layer_token_rd_mant),
    
    // QKV权重接口
    .qkv_weight_req(qkv_weight_req),
    .qkv_weight_type(qkv_weight_type),
    .qkv_weight_ack(qkv_weight_ack),
    .qkv_weight_valid(qkv_weight_valid),
    .qkv_weight_exp_array(qkv_weight_exp_array),
    .qkv_weight_mant_blocks(qkv_weight_mant_blocks),
    
    // WO权重接口
    .wo_weight_req(wo_weight_req),
    .wo_weight_ready(wo_weight_ready),
    .wo_weight_exp_array(wo_weight_exp_array),
    .wo_weight_mant(wo_weight_mant),
    
    // 结果输出接口（写到Result Buffer）
    .result_wr_en(att_result_wr_en),
    .result_wr_addr(att_result_wr_addr),
    .result_exp(att_result_exp),
    .result_mant(att_result_mant),
    
    // 调试接口
    .dbg_current_batch(dbg_att_batch),
    .dbg_batch_ctrl_state(dbg_att_state),
    .dbg_heads_done(),
    .dbg_heads_busy(),
    .dbg_first_batch(),
    .dbg_cycle_count(),
    .dbg_batch_cycle_count(),
    .dbg_accum_wr_count_h0(),
    .dbg_accum_wr_count_h1(),
    .dbg_accum_wr_count_h2(),
    .dbg_accum_wr_count_h3(),
    .dbg_output_proj_state(),
    .dbg_output_proj_busy(),
    .dbg_output_proj_done(),
    .dbg_k_rd_valid(),
    .dbg_v_rd_valid(),
    .dbg_kv_cache_ready(),
    .dbg_kv_cache_status()
);

//========== 4. Residual Add 1 ==========
residual_add_bfp #(
    // ========== 参数配置 ==========
    .TOKEN_NUM(TOKEN_NUM),       // 641: 总token数量
    .DIM(DIM),                   // 32: 特征维度
    .DATA_WIDTH(DATA_WIDTH),     // 8: 尾数位宽
    .EXP_WIDTH(EXP_WIDTH),       // 8: 指数位宽
    .ADDR_WIDTH(ADDR_WIDTH),     // 10: 地址位宽
    .GUARD_BITS(3)               // 3: 保护位（提高精度）
) u_residual1 (
    // ========== 基础信号 ==========
    .clk(clk),
    .rst_n(rst_n),
    
    // ========== 控制接口 ==========
    .start(residual1_start),     // 从FSM来的启动信号
    .done(residual1_done),       // 完成信号（返回FSM）
    .busy(res1_busy),            // 忙标志（调试用）
    .error(res1_error),          // 错误标志（调试用）
    
    // ========== 输入1: Layer Token Buffer (原始输入，跳跃连接) ==========
    .input_rd_en(res1_input_rd_en),
    .input_rd_addr(res1_input_rd_addr),
    .input_exp(layer_token_rd_exp),        // 共享指数
    .input_mant(layer_token_rd_mant),      // 32个8位尾数 (256位)
    .input_valid(layer_token_rd_valid),    // 数据有效标志
    
    // ========== 输入2: Result Buffer (Attention输出) ==========
    .result_rd_en(res1_result_rd_en),
    .result_rd_addr(res1_result_rd_addr),
    .result_exp(result_rd_exp),            // 共享指数
    .result_mant(result_rd_mant),          // 32个8位尾数 (256位)
    .result_valid(result_rd_valid),        // 数据有效标志
    
    // ========== 输出: Result Buffer ==========
    .output_wr_en(res1_output_wr_en),
    .output_wr_addr(res1_output_wr_addr),
    .output_exp(res1_output_exp),
    .output_mant(res1_output_mant),
    .output_valid(res1_output_valid),
    .output_ready(1'b1),                   // 常高，Result Buffer总是准备好
    
    // ========== 调试接口 ==========
    .state(res1_state),                    // 状态机当前状态
    .processed_count(res1_processed_count),// 已处理token计数
    .cycle_count(res1_cycle_count),        // 周期计数
    .overflow_detected(res1_overflow_detected),   // 溢出检测
    .underflow_detected(res1_underflow_detected)  // 下溢检测
);

//========== 5. LayerNorm 1 ==========
layer_norm_bfp #(
    .TOKEN_NUM(TOKEN_NUM),
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_layernorm1 (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(layernorm1_start),
    .mode(1'b0),  // LN1
    .done(layernorm1_done),
    .busy(),
    .error(),
    
    // 输入: Result Buffer
    .input_rd_en(ln1_input_rd_en),
    .input_rd_addr(ln1_input_rd_addr),
    .input_exp(result_rd_exp),
    .input_mant(result_rd_mant),
    .input_valid(result_rd_valid),
    
    // 输出: Result Buffer（供FFN使用）
    .output_wr_en(ln1_rb_output_wr_en),
    .output_wr_addr(ln1_rb_output_wr_addr),
    .output_exp(ln1_rb_output_exp),
    .output_mant(ln1_rb_output_mant),
    .output_valid(ln1_output_valid),
    .output_ready(1'b1),
    
    // 参数接口
    .param_rd_en(ln1_param_req_internal),
    .param_rd_gamma(ln1_param_gamma_internal),
    .param_exp(ln_param_exp),
    .param_mant(ln_param_mant),
    .param_valid(ln_param_valid)
);

// LN1输出需要同时写入两个buffer:
// 1. LN1 buffer (供Residual2使用)
// 2. Result buffer (供FFN使用)
assign ln1_buffer_wr_en = ln1_rb_output_wr_en;
assign ln1_buffer_wr_addr = ln1_rb_output_wr_addr[LN1_ADDR_WIDTH-1:0];  // 截取地址低位
assign ln1_buffer_wr_exp = ln1_rb_output_exp;
assign ln1_buffer_wr_mant = ln1_rb_output_mant;

//========== 6. FFN模块 ==========
ffn_backbone_top #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_CHUNK(TOKEN_BATCH),
    .BATCH_NUM(BATCH_NUM),
    .D_MODEL(DIM),
    .D_FF(D_FF),
    .FEATURE_CHUNK(FEATURE_CHUNK),
    .BFP_EXP_W(EXP_WIDTH),
    .BFP_MANT_W(DATA_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_ffn (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(ffn_start),
    .done(ffn_done),
    .busy(ffn_busy),
    
    // Token输入 (From Result Buffer)
    .rb_rd_en(ffn_rb_rd_en),
    .rb_rd_addr(ffn_rb_rd_addr),
    .rb_rd_exp(result_rd_exp),
    .rb_rd_mant(result_rd_mant),
    .rb_rd_valid(result_rd_valid),
    
    // 结果输出 (To Result Buffer)
    .rb_wr_en(ffn_rb_wr_en),
    .rb_wr_addr(ffn_rb_wr_addr),
    .rb_wr_exp(ffn_rb_wr_exp),
    .rb_wr_mant(ffn_rb_wr_mant),
    
    // 权重接口
    .weight_req(ffn_weight_req),
    .weight_type(ffn_weight_type),
    .weight_chunk_id(ffn_weight_chunk_id),
    .weight_ready(ffn_weight_ready),
    .weight_exp_array(ffn_weight_exp_array),
    .weight_mant(ffn_weight_mant),
    
    .dbg_token_batch(),
    .dbg_feature_chunk(),
    .dbg_cycle_count(),
    .dbg_state(dbg_ffn_state)
);

//========== 7. Residual Add 2 ==========
residual_add_bfp #(
    // ========== 参数配置 ==========
    .TOKEN_NUM(TOKEN_NUM),
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .GUARD_BITS(3)
) u_residual2 (
    // ========== 基础信号 ==========
    .clk(clk),
    .rst_n(rst_n),
    
    // ========== 控制接口 ==========
    .start(residual2_start),
    .done(residual2_done),
    .busy(res2_busy),
    .error(res2_error),
    
    // ========== 输入1: LN1 Result Buffer (LayerNorm1输出，跳跃连接) ==========
    /*
     * 注意：LN1 Buffer只有32个token容量
     * - 地址范围: 0-31 (5位地址)
     * - 需要循环读取: addr[4:0] = token_idx % 32
     * - 由Residual2内部控制循环读取逻辑
     */
    .input_rd_en(ln1_buffer_rd_en),
    .input_rd_addr(ln1_buffer_rd_addr),    // 5位地址 -> 自动截取到正确位宽
    .input_exp(ln1_buffer_rd_exp),
    .input_mant(ln1_buffer_rd_mant),
    .input_valid(ln1_buffer_rd_valid),
    
    // ========== 输入2: Result Buffer (FFN输出) ==========
    .result_rd_en(res2_result_rd_en),
    .result_rd_addr(res2_result_rd_addr),
    .result_exp(result_rd_exp),
    .result_mant(result_rd_mant),
    .result_valid(result_rd_valid),
    
    // ========== 输出: Result Buffer ==========
    .output_wr_en(res2_output_wr_en),
    .output_wr_addr(res2_output_wr_addr),
    .output_exp(res2_output_exp),
    .output_mant(res2_output_mant),
    .output_valid(res2_output_valid),
    .output_ready(1'b1),
    
    // ========== 调试接口 ==========
    .state(res2_state),
    .processed_count(res2_processed_count),
    .cycle_count(res2_cycle_count),
    .overflow_detected(res2_overflow_detected),
    .underflow_detected(res2_underflow_detected)
);


//========== 8. LayerNorm 2 ==========
layer_norm_bfp #(
    .TOKEN_NUM(TOKEN_NUM),
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_layernorm2 (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(layernorm2_start),
    .mode(1'b1),  // LN2
    .done(layernorm2_done),
    .busy(),
    .error(),
    
    // 输入: Result Buffer
    .input_rd_en(ln2_input_rd_en),
    .input_rd_addr(ln2_input_rd_addr),
    .input_exp(result_rd_exp),
    .input_mant(result_rd_mant),
    .input_valid(result_rd_valid),
    
    // 输出: Layer Token Buffer写BANK
    .output_wr_en(layer_token_wr_en),
    .output_wr_addr(layer_token_wr_addr),
    .output_exp(layer_token_wr_exp),
    .output_mant(layer_token_wr_mant),
    .output_valid(),
    .output_ready(1'b1),
    
    // 参数接口
    .param_rd_en(ln2_param_req_internal),
    .param_rd_gamma(ln2_param_gamma_internal),
    .param_exp(ln_param_exp),
    .param_mant(ln_param_mant),
    .param_valid(ln_param_valid)
);

//================================================================================
// 读写仲裁逻辑
//================================================================================

// Layer Token Buffer读仲裁（Port A）
assign layer_token_rd_en = att_token_rd_en | res1_input_rd_en;
assign layer_token_rd_addr = att_token_rd_en ? att_token_rd_addr : 
                              res1_input_rd_en ? res1_input_rd_addr :
                              {ADDR_WIDTH{1'b0}};

// Result Buffer读仲裁
assign result_rd_en = res1_result_rd_en | ln1_input_rd_en | 
                      ffn_rb_rd_en | res2_result_rd_en | ln2_input_rd_en;

assign result_rd_addr = res1_result_rd_en ? res1_result_rd_addr :
                        ln1_input_rd_en ? ln1_input_rd_addr :
                        ffn_rb_rd_en    ? ffn_rb_rd_addr :
                        res2_result_rd_en ? res2_result_rd_addr :
                        ln2_input_rd_en ? ln2_input_rd_addr :
                        {ADDR_WIDTH{1'b0}};

// Result Buffer写仲裁
assign result_wr_en = att_result_wr_en | res1_output_wr_en | 
                      ln1_rb_output_wr_en | ffn_rb_wr_en | res2_output_wr_en;

assign result_wr_addr = att_result_wr_en ? att_result_wr_addr :
                        res1_output_wr_en ? res1_output_wr_addr :
                        ln1_rb_output_wr_en ? ln1_rb_output_wr_addr :
                        ffn_rb_wr_en     ? ffn_rb_wr_addr :
                        res2_output_wr_en ? res2_output_wr_addr :
                        {ADDR_WIDTH{1'b0}};

assign result_wr_exp = att_result_wr_en ? att_result_exp :
                       res1_output_wr_en ? res1_output_exp :
                       ln1_rb_output_wr_en ? ln1_rb_output_exp :
                       ffn_rb_wr_en     ? ffn_rb_wr_exp :
                       res2_output_wr_en ? res2_output_exp :
                       {EXP_WIDTH{1'b0}};

assign result_wr_mant = att_result_wr_en ? att_result_mant :
                        res1_output_wr_en ? res1_output_mant :
                        ln1_rb_output_wr_en ? ln1_rb_output_mant :
                        ffn_rb_wr_en     ? ffn_rb_wr_mant :
                        res2_output_wr_en ? res2_output_mant :
                        {DIM*DATA_WIDTH{1'b0}};

//================================================================================
// LayerNorm参数请求仲裁
//================================================================================

assign ln_param_req = ln1_param_req_internal | ln2_param_req_internal;

// LayerNorm参数类型编码
// 标准编码: 00=LN1_gamma, 01=LN1_beta, 10=LN2_gamma, 11=LN2_beta
assign ln_param_type = ln1_param_req_internal ? 
                       (ln1_param_gamma_internal ? 2'b00 : 2'b01) : 
                       (ln2_param_gamma_internal ? 2'b10 : 2'b11);

//================================================================================
// FSM状态输出
//================================================================================
assign dbg_fsm_state = {residual2_start, layernorm1_start, ffn_start, attention_start};

//================================================================================
// 仿真信息
//================================================================================

`ifdef SIMULATION
initial begin
    $display("========================================");
    $display("Backbone Transformer v2.4 (Updated)");
    $display("========================================");
    $display("Updates:");
    $display("  ✓ Integrated new backbone_attention_parallel_top");
    $display("  ✓ Fixed attention port connections");
    $display("  ✓ Updated FFN interface (RB read/write)");
    $display("  ✓ Connected valid signals for flow control");
    $display("  ✓ Unified weight_controller interface");
    $display("  ✓ LayerNorm param interface: ln_param_valid");
    $display("  ✓ Parallel Attention (4-Head)");
    $display("========================================");
end
`endif

endmodule