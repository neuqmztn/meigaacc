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
    parameter TOKEN_NUM       = 641,
    parameter TOKEN_BATCH     = 32,
    parameter BATCH_NUM       = 21,
    parameter DIM             = 8,
    parameter NUM_HEADS       = 4,
    parameter HEAD_DIM        = 8,
    parameter D_FF            = 128,
    parameter FEATURE_CHUNK   = 32,
    
    // ========== 数据参数 ==========
    parameter DATA_WIDTH      = 8,
    parameter EXP_WIDTH       = 8,
    parameter ADDR_WIDTH      = 10,
    
    // ========== Attention参数 ==========
    parameter K_CHUNK_SIZE    = 32,
    parameter K_CHUNK_NUM     = 21,
    parameter SCORE_WIDTH     = 16,
    parameter ACCUM_WIDTH     = 24,
    
    // ========== Compute Engine参数 ==========
    parameter NUM_PE          = 2,
    parameter PE_TYPE_0       = 0,
    parameter PE_TYPE_1       = 2,
    parameter ELEM_PE0        = 16,
    parameter ELEM_PE1        = 8,
    parameter CE_OUTPUT_WIDTH = 32,
    parameter CE_INTERNAL_WIDTH = 39,
    parameter CE_GUARD_BITS   = 7,
    parameter CE_ENABLE_ROUNDING = 1,
    
    // ========== QKV Compute参数 ==========
    parameter G_OUT           = 4,
    parameter T_OUT           = 8,
    parameter TOTAL_ELEM      = 32,
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
    
    // [移除] LayerNorm1暂存接口 - 现在使用内部ln1_result_buffer
    
    //================================================================================
    // QKV权重接口（数组格式 - 新）
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
    // FFN权重接口（新增 - 替代DRAM）
    //================================================================================
    // Linear1权重接口
    output wire ffn_w1_weight_req,
    input  wire ffn_w1_weight_ready,
    input  wire [D_FF*EXP_WIDTH-1:0] ffn_w1_weight_exp_array,
    input  wire [DIM*D_FF*DATA_WIDTH-1:0] ffn_w1_weight_mant,
    
    // Linear2权重接口
    output wire ffn_w2_weight_req,
    input  wire ffn_w2_weight_ready,
    input  wire [DIM*EXP_WIDTH-1:0] ffn_w2_weight_exp_array,
    input  wire [D_FF*DIM*DATA_WIDTH-1:0] ffn_w2_weight_mant,
    
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

// LN1 Result Buffer (新增)
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
wire ln1_output_wr_en;
wire [ADDR_WIDTH-1:0] ln1_output_wr_addr;
wire [EXP_WIDTH-1:0] ln1_output_exp;
wire [DIM*DATA_WIDTH-1:0] ln1_output_mant;
wire ln1_output_valid;
wire ln1_param_req_internal;
wire ln1_param_gamma_internal;

// FFN
wire ffn_token_rd_en;
wire [ADDR_WIDTH-1:0] ffn_token_rd_addr;
wire ffn_result_wr_en;
wire [ADDR_WIDTH-1:0] ffn_result_wr_addr;
wire [EXP_WIDTH-1:0] ffn_result_exp;
wire [DIM*DATA_WIDTH-1:0] ffn_result_mant;

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
transformer_block_ctrl_fsm u_ctrl_fsm (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(start),
    .done(done),
    .busy(busy),
    
    .attention_start(attention_start),
    .attention_done(attention_done),
    .attention_busy(attention_busy),
    
    .residual1_start(residual1_start),
    .residual1_done(residual1_done),
    
    .layernorm1_start(layernorm1_start),
    .layernorm1_done(layernorm1_done),
    
    .ffn_start(ffn_start),
    .ffn_done(ffn_done),
    .ffn_busy(ffn_busy),
    
    .residual2_start(residual2_start),
    .residual2_done(residual2_done),
    
    .layernorm2_start(layernorm2_start),
    .layernorm2_done(layernorm2_done),
    
    .state(dbg_fsm_state)
);

//========== 2. Result Buffer ==========
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

//========== 2.5. LN1 Result Buffer（新增）==========
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
    .SEQ_LEN(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .BATCH_NUM(BATCH_NUM),
    .D_MODEL(DIM),
    .NUM_HEADS(NUM_HEADS),
    .HEAD_DIM(HEAD_DIM),
    .K_CHUNK_SIZE(K_CHUNK_SIZE),
    .K_CHUNK_NUM(K_CHUNK_NUM),
    .BFP_EXP_W(EXP_WIDTH),
    .BFP_MANT_W(DATA_WIDTH),
    .SCORE_WIDTH(SCORE_WIDTH),
    .ACCUM_WIDTH(ACCUM_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
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
    
    .start(attention_start),
    .done(attention_done),
    .busy(attention_busy),
    
    // Token输入: Layer Token Buffer
    .token_rd_en(att_token_rd_en),
    .token_rd_addr(att_token_rd_addr),
    .token_exp(layer_token_rd_exp),
    .token_mant(layer_token_rd_mant),
    
    // 结果输出: Result Buffer
    .result_wr_en(att_result_wr_en),
    .result_wr_addr(att_result_wr_addr),
    .result_exp(att_result_exp),
    .result_mant(att_result_mant),
    
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
    
    .state(dbg_att_state),
    .current_batch(dbg_att_batch)
);

//========== 4. Residual Add 1 ==========
residual_add_bfp #(
    .TOKEN_NUM(TOKEN_NUM),
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_residual1 (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(residual1_start),
    .mode(2'b00),  // Residual 1
    .done(residual1_done),
    .busy(),
    .error(),
    
    // 输入1: Layer Token Buffer读BANK (原始输入)
    .input_rd_en(res1_input_rd_en),
    .input_rd_addr(res1_input_rd_addr),
    .input_exp(layer_token_rd_exp),
    .input_mant(layer_token_rd_mant),
    .input_valid(layer_token_rd_valid),
    
    // 输入2: Result Buffer (Attention输出)
    .result_rd_en(res1_result_rd_en),
    .result_rd_addr(res1_result_rd_addr),
    .result_exp(result_rd_exp),
    .result_mant(result_rd_mant),
    .result_valid(result_rd_valid),
    
    // 输出: Result Buffer
    .output_wr_en(res1_output_wr_en),
    .output_wr_addr(res1_output_wr_addr),
    .output_exp(res1_output_exp),
    .output_mant(res1_output_mant),
    .output_valid(res1_output_valid),
    
    .state(),
    .processed_count(),
    .cycle_count(),
    .overflow_detected(),
    .underflow_detected()
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
    
    // 输入: Result Buffer (Residual1输出)
    .input_rd_en(ln1_input_rd_en),
    .input_rd_addr(ln1_input_rd_addr),
    .input_exp(result_rd_exp),
    .input_mant(result_rd_mant),
    .input_valid(result_rd_valid),
    
    // 输出: Result Buffer（保持原有逻辑）
    .output_wr_en(ln1_output_wr_en),
    .output_wr_addr(ln1_output_wr_addr),
    .output_exp(ln1_output_exp),
    .output_mant(ln1_output_mant),
    .output_valid(ln1_output_valid),
    
    // 参数接口
    .param_rd_en(ln1_param_req_internal),
    .param_rd_gamma(ln1_param_gamma_internal),
    .param_exp(ln_param_exp),
    .param_mant(ln_param_mant),
    .param_valid(ln_param_valid),
    
    .state(),
    .processed_tokens(),
    .cycle_count(),
    .overflow_flag(),
    .underflow_flag()
);

// LayerNorm1输出同时写入LN1 Result Buffer
assign ln1_buffer_wr_en = ln1_output_wr_en;
assign ln1_buffer_wr_addr = ln1_output_wr_addr[LN1_ADDR_WIDTH-1:0];  // 取低位地址
assign ln1_buffer_wr_exp = ln1_output_exp;
assign ln1_buffer_wr_mant = ln1_output_mant;

//========== 6. FFN模块 ==========
ffn_backbone_top #(
    .TOKEN_NUM(TOKEN_NUM),
    .D_MODEL(DIM),
    .D_FF(D_FF),
    .TOKEN_CHUNK(TOKEN_BATCH),
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
    
    // Token输入: Result Buffer (LN1输出)
    .token_rd_en(ffn_token_rd_en),
    .token_rd_addr(ffn_token_rd_addr),
    .token_exp(result_rd_exp),
    .token_mant(result_rd_mant),
    
    // 结果输出: Result Buffer
    .result_wr_en(ffn_result_wr_en),
    .result_wr_addr(ffn_result_wr_addr),
    .result_exp(ffn_result_exp),
    .result_mant(ffn_result_mant),
    
    // FFN权重接口（新增 - 替代DRAM）
    .w1_weight_req(ffn_w1_weight_req),
    .w1_weight_ready(ffn_w1_weight_ready),
    .w1_weight_exp_array(ffn_w1_weight_exp_array),
    .w1_weight_mant(ffn_w1_weight_mant),
    
    .w2_weight_req(ffn_w2_weight_req),
    .w2_weight_ready(ffn_w2_weight_ready),
    .w2_weight_exp_array(ffn_w2_weight_exp_array),
    .w2_weight_mant(ffn_w2_weight_mant),
    
    .cycle_count(),
    .current_token_chunk(),
    .current_feature_chunk(),
    .fsm_state(dbg_ffn_state)
);

//========== 7. Residual Add 2 ==========
residual_add_bfp #(
    .TOKEN_NUM(TOKEN_NUM),
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_residual2 (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(residual2_start),
    .mode(2'b01),  // Residual 2
    .done(residual2_done),
    .busy(),
    .error(),
    
    // 输入1: LN1 Result Buffer（新：从内部buffer读）
    .input_rd_en(ln1_buffer_rd_en),
    .input_rd_addr(ln1_buffer_rd_addr),
    .input_exp(ln1_buffer_rd_exp),
    .input_mant(ln1_buffer_rd_mant),
    .input_valid(ln1_buffer_rd_valid),
    
    // 输入2: Result Buffer (FFN输出)
    .result_rd_en(res2_result_rd_en),
    .result_rd_addr(res2_result_rd_addr),
    .result_exp(result_rd_exp),
    .result_mant(result_rd_mant),
    .result_valid(result_rd_valid),
    
    // 输出: Result Buffer
    .output_wr_en(res2_output_wr_en),
    .output_wr_addr(res2_output_wr_addr),
    .output_exp(res2_output_exp),
    .output_mant(res2_output_mant),
    .output_valid(res2_output_valid),
    
    .state(),
    .processed_count(),
    .cycle_count(),
    .overflow_detected(),
    .underflow_detected()
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
    
    // 参数接口
    .param_rd_en(ln2_param_req_internal),
    .param_rd_gamma(ln2_param_gamma_internal),
    .param_exp(ln_param_exp),
    .param_mant(ln_param_mant),
    .param_valid(ln_param_valid),
    
    .state(),
    .processed_tokens(),
    .cycle_count(),
    .overflow_flag(),
    .underflow_flag()
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
                      ffn_token_rd_en | res2_result_rd_en | ln2_input_rd_en;

assign result_rd_addr = res1_result_rd_en ? res1_result_rd_addr :
                        ln1_input_rd_en ? ln1_input_rd_addr :
                        ffn_token_rd_en ? ffn_token_rd_addr :
                        res2_result_rd_en ? res2_result_rd_addr :
                        ln2_input_rd_en ? ln2_input_rd_addr :
                        {ADDR_WIDTH{1'b0}};

// Result Buffer写仲裁
assign result_wr_en = att_result_wr_en | res1_output_wr_en | 
                      ln1_output_wr_en | ffn_result_wr_en | res2_output_wr_en;

assign result_wr_addr = att_result_wr_en ? att_result_wr_addr :
                        res1_output_wr_en ? res1_output_wr_addr :
                        ln1_output_wr_en ? ln1_output_wr_addr :
                        ffn_result_wr_en ? ffn_result_wr_addr :
                        res2_output_wr_en ? res2_output_wr_addr :
                        {ADDR_WIDTH{1'b0}};

assign result_wr_exp = att_result_wr_en ? att_result_exp :
                       res1_output_wr_en ? res1_output_exp :
                       ln1_output_wr_en ? ln1_output_exp :
                       ffn_result_wr_en ? ffn_result_exp :
                       res2_output_wr_en ? res2_output_exp :
                       {EXP_WIDTH{1'b0}};

assign result_wr_mant = att_result_wr_en ? att_result_mant :
                        res1_output_wr_en ? res1_output_mant :
                        ln1_output_wr_en ? ln1_output_mant :
                        ffn_result_wr_en ? ffn_result_mant :
                        res2_output_wr_en ? res2_output_mant :
                        {DIM*DATA_WIDTH{1'b0}};

//================================================================================
// LayerNorm参数请求仲裁（修改 - 新）
//================================================================================

assign ln_param_req = ln1_param_req_internal | ln2_param_req_internal;

// 将gamma信号映射到type编码
// LN1: gamma=0→type=00 (gamma), gamma=1→type=01 (beta)
// LN2: gamma=0→type=10 (gamma), gamma=1→type=11 (beta)
assign ln_param_type = ln1_param_req_internal ? 
                       {1'b0, ln1_param_gamma_internal} :   // LN1: 00 or 01
                       {1'b1, ln2_param_gamma_internal};    // LN2: 10 or 11

//================================================================================
// 仿真信息
//================================================================================

`ifdef SIMULATION
initial begin
    $display("========================================");
    $display("Backbone Transformer v2.1");
    $display("========================================");
    $display("Updates:");
    $display("  ✓ Parallel Attention (4-Head)");
    $display("  ✓ QKV array format (32 exponents)");
    $display("  ✓ WO weight interface added");
    $display("  ✓ LayerNorm param type encoding");
    $display("  ✓ Independent LN1 Result Buffer (1KB)");
    $display("  ✓ Removed external LN1 save/load ports");
    $display("========================================");
end
`endif

endmodule