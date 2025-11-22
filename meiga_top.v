`timescale 1ns / 1ps

//================================================================================
// MEIGA Top Module - 系统顶层集成 (Updated Version)
//
// 更新日期: 2025-11-16
// 版本: v2.0
//
// 主要更新:
// 1. ✅ 新增 FFN权重接口（Backbone和SideNet）
// 2. ❌ 删除 LN1暂存接口（现在使用内部buffer）
// 3. ⚠️  更新 LayerNorm接口为ln_param_valid
// 4. ✅ 所有权重改为数组格式（每列独立指数）
//
// 功能：
// 集成Backbone Transformer和SideNet，实现完整的MEIGA架构
// - 4层Backbone Transformer（分时复用单个模块）
// - 5层SideNet（Layer 0专用，Layer 1-3复用，Layer 4专用）
// - 层间Token Buffer（双Bank乒乓切换）
// - SideNet输出Buffer（4-Bank独立存储）
// - 权重管理（Backbone和SideNet独立）
//
// 架构：
// 1. meiga_control: 顶层FSM控制器
// 2. backbone_transformer: 单层Transformer，复用4次
// 3. layer_token_buffer_dual_bank: 层间存储（42KB双Bank）
// 4. sidenet_layer0: Layer 0专用（仅Compression）
// 5. sidenet_layer_standard: Layer 1-3复用
// 6. sidenet_layer4: Layer 4专用（含Expand）
// 7. sidenet_layer_output_buffer: SideNet输出4-Bank存储
// 8. weight_controller: Backbone权重管理
// 9. sidenet_weight_storage: SideNet权重存储
// 10. sidenet_weight_controller: SideNet权重控制
//
// 数据流：
// Input → Layer Token Buffer BANK A
// Layer i: Backbone读BANK A/B → 写BANK B/A
//          Sidenet读BANK A/B → 写Layer Output Buffer
// Bank Swap after each layer
// Layer 4: Sidenet Only → Final Output
//
// Author: MEIGA Design Team
// Date: 2025-11-16
// Version: 2.0
// Standard: Verilog 2001
//================================================================================

module meiga_top #(
    // ========== 网络参数 ==========
    parameter NUM_LAYERS       = 4,         // Backbone层数
    parameter NUM_SIDENET_LAYERS = 5,       // Sidenet层数 (0-4)
    parameter TOKEN_NUM        = 641,
    parameter TOKEN_BATCH      = 32,
    parameter BATCH_NUM        = 21,
    
    // ========== 维度参数 ==========
    parameter BACKBONE_DIM     = 32,
    parameter SIDENET_DIM      = 8,
    parameter NUM_HEADS_BB     = 4,
    parameter HEAD_DIM_BB      = 8,
    parameter NUM_HEADS_SN     = 4,
    parameter HEAD_DIM_SN      = 2,
    parameter D_FF_BB          = 128,
    parameter D_FF_SN          = 32,
    parameter FEATURE_CHUNK    = 32,
    
    // ========== 数据格式 ==========
    parameter DATA_WIDTH_BB    = 8,         // Backbone: 8-bit BFP
    parameter DATA_WIDTH_SN    = 16,        // Sidenet: 16-bit BFP
    parameter EXP_WIDTH        = 8,
    parameter ADDR_WIDTH       = 10,
    
    // ========== Attention参数 ==========
    parameter K_CHUNK_SIZE     = 32,
    parameter K_CHUNK_NUM      = 21,
    
    // ========== DRAM参数 ==========
    parameter DRAM_ADDR_WIDTH  = 32,
    parameter DRAM_DATA_WIDTH  = 256,
    
    // ========== 超时参数 ==========
    parameter TIMEOUT_CYCLES   = 100000
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 系统控制接口
    //================================================================================
    input  wire start,              // 启动信号
    output wire done,               // 完成信号
    output wire busy,               // 忙状态
    output wire error,              // 错误标志
    
    //================================================================================
    // DRAM接口 - 输入加载
    //================================================================================
    output wire dram_input_rd_en,
    output wire [DRAM_ADDR_WIDTH-1:0] dram_input_rd_addr,
    input  wire [DRAM_DATA_WIDTH-1:0] dram_input_rd_data,
    input  wire dram_input_rd_valid,
    
    //================================================================================
    // DRAM接口 - 输出存储
    //================================================================================
    output wire dram_output_wr_en,
    output wire [DRAM_ADDR_WIDTH-1:0] dram_output_wr_addr,
    output wire [DRAM_DATA_WIDTH-1:0] dram_output_wr_data,
    input  wire dram_output_wr_ready,
    
    //================================================================================
    // DRAM接口 - Backbone权重
    //================================================================================
    output wire dram_bb_weight_req,
    output wire [3:0] dram_bb_weight_type,
    output wire [1:0] dram_bb_layer_id,
    input  wire dram_bb_weight_ready,
    input  wire [DRAM_DATA_WIDTH-1:0] dram_bb_weight_data,
    input  wire dram_bb_weight_valid,
    
    //================================================================================
    // DRAM接口 - Sidenet权重
    //================================================================================
    output wire dram_sn_weight_req,
    output wire [3:0] dram_sn_weight_type,
    output wire [2:0] dram_sn_layer_id,
    input  wire dram_sn_weight_ready,
    input  wire [DRAM_DATA_WIDTH-1:0] dram_sn_weight_data,
    input  wire dram_sn_weight_valid,
    
    //================================================================================
    // 调试接口
    //================================================================================
    output wire [3:0] dbg_ctrl_state,
    output wire [1:0] dbg_current_bb_layer,
    output wire [2:0] dbg_current_sn_layer,
    output wire dbg_backbone_done_latch,
    output wire dbg_sidenet_done_latch,
    output wire dbg_bank_select
);

//================================================================================
// 内部信号定义
//================================================================================

//------------------------------------------------------------------------
// 控制信号
//------------------------------------------------------------------------
wire [1:0] current_backbone_layer;
wire [2:0] current_sidenet_layer;
wire backbone_start;
wire sidenet_start;
wire backbone_done;
wire backbone_busy;
wire sidenet_done;
wire sidenet_busy;
wire bank_swap_pulse;
wire load_input_start;
wire load_input_done;
wire store_output_start;
wire store_output_done;

//------------------------------------------------------------------------
// Layer Token Buffer信号
//------------------------------------------------------------------------
// Backbone读接口
wire bb_layer_token_rd_en;
wire [ADDR_WIDTH-1:0] bb_layer_token_rd_addr;
wire [EXP_WIDTH-1:0] bb_layer_token_rd_exp;
wire [BACKBONE_DIM*DATA_WIDTH_BB-1:0] bb_layer_token_rd_mant;
wire bb_layer_token_rd_valid;

// Backbone写接口
wire bb_layer_token_wr_en;
wire [ADDR_WIDTH-1:0] bb_layer_token_wr_addr;
wire [EXP_WIDTH-1:0] bb_layer_token_wr_exp;
wire [BACKBONE_DIM*DATA_WIDTH_BB-1:0] bb_layer_token_wr_mant;

// Sidenet读接口（Port B）
wire sn_layer_token_rd_en;
wire [ADDR_WIDTH-1:0] sn_layer_token_rd_addr;
wire [EXP_WIDTH-1:0] sn_layer_token_rd_exp;
wire [BACKBONE_DIM*DATA_WIDTH_BB-1:0] sn_layer_token_rd_mant;
wire sn_layer_token_rd_valid;

// Layer Token Buffer输出信号
wire current_read_bank;
wire current_write_bank;
wire layer_wr_ready;

// ❌ 删除：LN1暂存接口（现在backbone_transformer内部处理）

//------------------------------------------------------------------------
// Backbone权重信号 - 更新为数组格式
//------------------------------------------------------------------------
// QKV权重接口（数组格式）
wire bb_qkv_weight_req;
wire [1:0] bb_qkv_weight_type;
wire bb_qkv_weight_ack;
wire bb_qkv_weight_valid;
wire [BACKBONE_DIM*EXP_WIDTH-1:0] bb_qkv_weight_exp_array;
wire [BACKBONE_DIM*BACKBONE_DIM*DATA_WIDTH_BB-1:0] bb_qkv_weight_mant_blocks;

// WO权重接口（数组格式）
wire bb_wo_weight_req;
wire bb_wo_weight_ready;
wire [BACKBONE_DIM*EXP_WIDTH-1:0] bb_wo_weight_exp_array;
wire [BACKBONE_DIM*BACKBONE_DIM*DATA_WIDTH_BB-1:0] bb_wo_weight_mant;

// ✅ 新增：FFN权重接口（数组格式）
wire bb_ffn_weight_req;
wire [1:0] bb_ffn_weight_type;
wire [1:0] bb_ffn_weight_chunk_id;
wire bb_ffn_weight_ready;
wire [BACKBONE_DIM*EXP_WIDTH-1:0] bb_ffn_weight_exp_array;
wire [BACKBONE_DIM*FEATURE_CHUNK*DATA_WIDTH_BB-1:0] bb_ffn_weight_mant;

// LayerNorm参数接口
wire bb_ln_param_req;
wire [1:0] bb_ln_param_type;
wire bb_ln_param_valid;  // ✅ 更新：使用valid而非ready
wire [EXP_WIDTH-1:0] bb_ln_param_exp;
wire [BACKBONE_DIM*DATA_WIDTH_BB-1:0] bb_ln_param_mant;

//------------------------------------------------------------------------
// Sidenet Layer 0信号
//------------------------------------------------------------------------
wire layer0_start;
wire layer0_done;
wire layer0_busy;
wire layer0_error;

wire layer0_backbone_rd_en;
wire [ADDR_WIDTH-1:0] layer0_backbone_rd_addr;

wire layer0_wr_en;
wire [ADDR_WIDTH-1:0] layer0_token_id;
wire [EXP_WIDTH-1:0] layer0_wr_exp;
wire [SIDENET_DIM*DATA_WIDTH_SN-1:0] layer0_wr_mant;

// Layer 0 Compression权重（数组格式）
wire layer0_compress_weight_req;
wire layer0_compress_weight_ready;
wire [SIDENET_DIM*EXP_WIDTH-1:0] layer0_compress_weight_exp;
wire [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH_SN-1:0] layer0_compress_weight_mant;

//------------------------------------------------------------------------
// Sidenet Layer Standard (1-3)信号
//------------------------------------------------------------------------
wire layer_std_start;
wire layer_std_done;
wire layer_std_busy;
wire [1:0] layer_std_id;

wire layer_std_backbone_rd_en;
wire [ADDR_WIDTH-1:0] layer_std_backbone_rd_addr;

wire layer_std_prev_rd_en;
wire [1:0] layer_std_prev_layer_id;
wire [ADDR_WIDTH-1:0] layer_std_prev_token_id;
wire [EXP_WIDTH-1:0] layer_std_prev_exp;
wire [SIDENET_DIM*DATA_WIDTH_SN-1:0] layer_std_prev_mant;
wire layer_std_prev_valid;

wire layer_std_curr_wr_en;
wire [1:0] layer_std_curr_layer_id;
wire [ADDR_WIDTH-1:0] layer_std_curr_token_id;
wire [EXP_WIDTH-1:0] layer_std_curr_exp;
wire [SIDENET_DIM*DATA_WIDTH_SN-1:0] layer_std_curr_mant;

// Layer Standard Compression权重（数组格式）
wire layer_std_compress_weight_req;
wire layer_std_compress_weight_ready;
wire [SIDENET_DIM*EXP_WIDTH-1:0] layer_std_compress_weight_exp;
wire [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH_SN-1:0] layer_std_compress_weight_mant;

// Layer Standard QKV权重（数组格式）
wire layer_std_qkv_weight_req;
wire [1:0] layer_std_qkv_weight_type;
wire layer_std_qkv_weight_ack;
wire layer_std_qkv_weight_valid;
wire layer_std_qkv_weight_ready;  // 添加：用于连接controller
wire [SIDENET_DIM*EXP_WIDTH-1:0] layer_std_qkv_weight_exp_array;
wire [SIDENET_DIM*SIDENET_DIM*DATA_WIDTH_SN-1:0] layer_std_qkv_weight_mant_blocks;

// Layer Standard WO权重（数组格式）
wire layer_std_wo_weight_req;
wire layer_std_wo_weight_ready;
wire [SIDENET_DIM*EXP_WIDTH-1:0] layer_std_wo_weight_exp_array;
wire [SIDENET_DIM*SIDENET_DIM*DATA_WIDTH_SN-1:0] layer_std_wo_weight_mant;

// ✅ 新增：Layer Standard FFN权重（数组格式）
wire layer_std_ffn_w1_weight_req;
wire layer_std_ffn_w1_weight_ready;
wire [D_FF_SN*EXP_WIDTH-1:0] layer_std_ffn_w1_weight_exp_array;
wire [SIDENET_DIM*D_FF_SN*DATA_WIDTH_SN-1:0] layer_std_ffn_w1_weight_mant;

wire layer_std_ffn_w2_weight_req;
wire layer_std_ffn_w2_weight_ready;
wire [SIDENET_DIM*EXP_WIDTH-1:0] layer_std_ffn_w2_weight_exp_array;
wire [D_FF_SN*SIDENET_DIM*DATA_WIDTH_SN-1:0] layer_std_ffn_w2_weight_mant;

// Layer Standard LayerNorm参数
wire layer_std_ln_param_req;
wire [1:0] layer_std_ln_param_type;
wire layer_std_ln_param_valid;
wire [EXP_WIDTH-1:0] layer_std_ln_param_exp;
wire [SIDENET_DIM*DATA_WIDTH_SN-1:0] layer_std_ln_param_mant;

//------------------------------------------------------------------------
// Sidenet Layer 4信号
//------------------------------------------------------------------------
wire layer4_start;
wire layer4_done;
wire layer4_busy;
wire layer4_error;

wire layer4_backbone_rd_en;
wire [ADDR_WIDTH-1:0] layer4_backbone_rd_addr;

wire layer4_layer3_rd_en;
wire [ADDR_WIDTH-1:0] layer4_layer3_token_id;
wire [EXP_WIDTH-1:0] layer4_layer3_exp;
wire [SIDENET_DIM*DATA_WIDTH_SN-1:0] layer4_layer3_mant;
wire layer4_layer3_valid;

wire layer4_final_wr_en;
wire [ADDR_WIDTH-1:0] layer4_final_wr_addr;
wire [EXP_WIDTH-1:0] layer4_final_wr_exp;
wire [BACKBONE_DIM*DATA_WIDTH_SN-1:0] layer4_final_wr_mant;

// Layer 4 Compression权重（数组格式）
wire layer4_compress_weight_req;
wire layer4_compress_weight_ready;
wire [SIDENET_DIM*EXP_WIDTH-1:0] layer4_compress_weight_exp;
wire [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH_SN-1:0] layer4_compress_weight_mant;

// Layer 4 Expand权重（数组格式）
wire layer4_expand_weight_req;
wire layer4_expand_weight_ready;
wire [BACKBONE_DIM*EXP_WIDTH-1:0] layer4_expand_weight_exp;
wire [SIDENET_DIM*BACKBONE_DIM*DATA_WIDTH_SN-1:0] layer4_expand_weight_mant;

//------------------------------------------------------------------------
// Sidenet Layer Output Buffer信号
//------------------------------------------------------------------------
wire sn_output_wr_en;
wire [1:0] sn_output_wr_layer_id;
wire [ADDR_WIDTH-1:0] sn_output_wr_token_id;
wire [EXP_WIDTH-1:0] sn_output_wr_exp;
wire [SIDENET_DIM*DATA_WIDTH_SN-1:0] sn_output_wr_mant;
wire sn_output_wr_ready;
wire sn_output_wr_error;

wire sn_output_rd_en;
wire [1:0] sn_output_rd_layer_id;
wire [ADDR_WIDTH-1:0] sn_output_rd_token_id;
wire [EXP_WIDTH-1:0] sn_output_rd_exp;
wire [SIDENET_DIM*DATA_WIDTH_SN-1:0] sn_output_rd_mant;
wire sn_output_rd_valid;
wire sn_output_rd_error;

//------------------------------------------------------------------------
// Sidenet Weight Controller信号
//------------------------------------------------------------------------
wire sn_weight_storage_rd_en;
wire [2:0] sn_weight_storage_rd_layer_id;
wire [3:0] sn_weight_storage_rd_weight_type;
wire [5:0] sn_weight_storage_rd_burst_idx;
wire sn_weight_storage_rd_valid;
wire [BACKBONE_DIM*EXP_WIDTH-1:0] sn_weight_storage_rd_exp_array;  // 最大位宽
wire [DRAM_DATA_WIDTH-1:0] sn_weight_storage_rd_data_burst;

wire sn_weight_storage_wr_en;
wire [2:0] sn_weight_storage_wr_layer_id;
wire [3:0] sn_weight_storage_wr_weight_type;
wire [5:0] sn_weight_storage_wr_burst_idx;
wire [BACKBONE_DIM*EXP_WIDTH-1:0] sn_weight_storage_wr_exp_array;
wire [DRAM_DATA_WIDTH-1:0] sn_weight_storage_wr_data_burst;
wire sn_weight_storage_wr_ready;

//================================================================================
// Sidenet层启动信号Mux
//================================================================================
assign layer0_start = (current_sidenet_layer == 3'd0) ? sidenet_start : 1'b0;
assign layer_std_start = ((current_sidenet_layer >= 3'd1) && 
                          (current_sidenet_layer <= 3'd3)) ? sidenet_start : 1'b0;
assign layer4_start = (current_sidenet_layer == 3'd4) ? sidenet_start : 1'b0;

assign layer_std_id = current_sidenet_layer[1:0] - 2'd1;  // 1->0, 2->1, 3->2

//================================================================================
// Sidenet完成信号Mux
//================================================================================
assign sidenet_done = (current_sidenet_layer == 3'd0) ? layer0_done :
                      (current_sidenet_layer == 3'd4) ? layer4_done :
                      layer_std_done;

assign sidenet_busy = (current_sidenet_layer == 3'd0) ? layer0_busy :
                      (current_sidenet_layer == 3'd4) ? layer4_busy :
                      layer_std_busy;

//================================================================================
// Sidenet读Layer Token Buffer的Mux
//================================================================================
assign sn_layer_token_rd_en = (current_sidenet_layer == 3'd0) ? layer0_backbone_rd_en :
                              (current_sidenet_layer == 3'd4) ? layer4_backbone_rd_en :
                              layer_std_backbone_rd_en;

assign sn_layer_token_rd_addr = (current_sidenet_layer == 3'd0) ? layer0_backbone_rd_addr :
                                (current_sidenet_layer == 3'd4) ? layer4_backbone_rd_addr :
                                layer_std_backbone_rd_addr;

//================================================================================
// Sidenet Layer Output Buffer写接口Mux
//================================================================================
assign sn_output_wr_en = (current_sidenet_layer == 3'd0) ? layer0_wr_en :
                         (current_sidenet_layer == 3'd4) ? 1'b0 :  // Layer 4不写output buffer
                         layer_std_curr_wr_en;

assign sn_output_wr_layer_id = (current_sidenet_layer == 3'd0) ? 2'd0 :
                                layer_std_curr_layer_id;

assign sn_output_wr_token_id = (current_sidenet_layer == 3'd0) ? layer0_token_id :
                                layer_std_curr_token_id;

assign sn_output_wr_exp = (current_sidenet_layer == 3'd0) ? layer0_wr_exp :
                          layer_std_curr_exp;

assign sn_output_wr_mant = (current_sidenet_layer == 3'd0) ? layer0_wr_mant :
                           layer_std_curr_mant;

//================================================================================
// Sidenet Layer Output Buffer读接口Mux
//================================================================================
assign sn_output_rd_en = (current_sidenet_layer == 3'd4) ? layer4_layer3_rd_en :
                         layer_std_prev_rd_en;

assign sn_output_rd_layer_id = (current_sidenet_layer == 3'd4) ? 2'd3 :  // Layer 4读Layer 3
                                layer_std_prev_layer_id;

assign sn_output_rd_token_id = (current_sidenet_layer == 3'd4) ? layer4_layer3_token_id :
                                layer_std_prev_token_id;

// Output buffer读出数据分配
assign layer_std_prev_exp = sn_output_rd_exp;
assign layer_std_prev_mant = sn_output_rd_mant;
assign layer_std_prev_valid = sn_output_rd_valid;

assign layer4_layer3_exp = sn_output_rd_exp;
assign layer4_layer3_mant = sn_output_rd_mant;
assign layer4_layer3_valid = sn_output_rd_valid;

//================================================================================
// 调试信号
//================================================================================
assign dbg_bank_select = current_read_bank;
assign dbg_current_sn_layer = current_sidenet_layer;

//================================================================================
// 模块实例化
//================================================================================

//------------------------------------------------------------------------
// 1. 控制FSM
//------------------------------------------------------------------------
meiga_control #(
    .NUM_LAYERS(NUM_LAYERS),
    .NUM_SIDENET_LAYERS(NUM_SIDENET_LAYERS),
    .ENABLE_LAYER_0(1),
    .ENABLE_LAYER_1(1),
    .ENABLE_LAYER_2(1),
    .ENABLE_LAYER_3(1),
    .ENABLE_LAYER_4(1),
    .TIMEOUT_CYCLES(TIMEOUT_CYCLES)
) u_meiga_control (
    .clk(clk),
    .rst_n(rst_n),
    
    // 系统控制
    .start(start),
    .done(done),
    .busy(busy),
    .error(error),
    
    // 层级控制
    .current_backbone_layer(current_backbone_layer),
    .current_sidenet_layer(current_sidenet_layer),
    
    // 启动信号
    .backbone_start(backbone_start),
    .sidenet_start(sidenet_start),
    
    // 完成信号
    .backbone_done(backbone_done),
    .backbone_busy(backbone_busy),
    .sidenet_done(sidenet_done),
    .sidenet_busy(sidenet_busy),
    
    // Bank控制
    .bank_swap_pulse(bank_swap_pulse),
    
    // 加载/存储控制
    .load_input_start(load_input_start),
    .load_input_done(load_input_done),
    .store_output_start(store_output_start),
    .store_output_done(store_output_done),
    
    // 调试接口
    .dbg_state(dbg_ctrl_state),
    .dbg_current_layer(dbg_current_bb_layer),
    .dbg_backbone_done_latch(dbg_backbone_done_latch),
    .dbg_sidenet_done_latch(dbg_sidenet_done_latch)
);

//------------------------------------------------------------------------
// 2. Backbone Transformer（复用4层）
// ✅ 更新：新增FFN权重接口，删除LN1暂存接口
//------------------------------------------------------------------------
backbone_transformer #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .BATCH_NUM(BATCH_NUM),
    .DIM(BACKBONE_DIM),
    .NUM_HEADS(NUM_HEADS_BB),
    .HEAD_DIM(HEAD_DIM_BB),
    .D_FF(D_FF_BB),
    .FEATURE_CHUNK(FEATURE_CHUNK),
    .DATA_WIDTH(DATA_WIDTH_BB),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .K_CHUNK_SIZE(K_CHUNK_SIZE),
    .K_CHUNK_NUM(K_CHUNK_NUM)
) u_backbone_transformer (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制接口
    .start(backbone_start),
    .layer_id(current_backbone_layer),
    .done(backbone_done),
    .busy(backbone_busy),
    
    // Layer Token Buffer读接口
    .layer_token_rd_en(bb_layer_token_rd_en),
    .layer_token_rd_addr(bb_layer_token_rd_addr),
    .layer_token_rd_exp(bb_layer_token_rd_exp),
    .layer_token_rd_mant(bb_layer_token_rd_mant),
    .layer_token_rd_valid(bb_layer_token_rd_valid),
    
    // Layer Token Buffer写接口
    .layer_token_wr_en(bb_layer_token_wr_en),
    .layer_token_wr_addr(bb_layer_token_wr_addr),
    .layer_token_wr_exp(bb_layer_token_wr_exp),
    .layer_token_wr_mant(bb_layer_token_wr_mant),
    
    // ❌ 删除：LN1暂存接口（现在使用内部ln1_result_buffer）
    
    // QKV权重接口（数组格式）
    .qkv_weight_req(bb_qkv_weight_req),
    .qkv_weight_type(bb_qkv_weight_type),
    .qkv_weight_ack(bb_qkv_weight_ack),
    .qkv_weight_valid(bb_qkv_weight_valid),
    .qkv_weight_exp_array(bb_qkv_weight_exp_array),
    .qkv_weight_mant_blocks(bb_qkv_weight_mant_blocks),
    
    // WO权重接口（Output Projection）
    .wo_weight_req(bb_wo_weight_req),
    .wo_weight_ready(bb_wo_weight_ready),
    .wo_weight_exp_array(bb_wo_weight_exp_array),
    .wo_weight_mant(bb_wo_weight_mant),
    
    // ✅ 新增：FFN权重接口（连接到weight_controller）
    .ffn_weight_req(bb_ffn_weight_req),
    .ffn_weight_type(bb_ffn_weight_type),
    .ffn_weight_chunk_id(bb_ffn_weight_chunk_id),
    .ffn_weight_ready(bb_ffn_weight_ready),
    .ffn_weight_exp_array(bb_ffn_weight_exp_array),
    .ffn_weight_mant(bb_ffn_weight_mant),
    
    // LayerNorm参数接口
    .ln_param_req(bb_ln_param_req),
    .ln_param_type(bb_ln_param_type),
    .ln_param_exp(bb_ln_param_exp),
    .ln_param_mant(bb_ln_param_mant),
    .ln_param_valid(bb_ln_param_valid),  // ✅ 确认：使用valid
    
    // 调试接口
    .dbg_fsm_state(),
    .dbg_att_state(),
    .dbg_att_batch(),
    .dbg_ffn_state(),
    .dbg_result_rd_count(),
    .dbg_result_wr_count(),
    .dbg_ln1_rd_count(),
    .dbg_ln1_wr_count()
);

//------------------------------------------------------------------------
// 3. Layer Token Buffer（双Bank）
//------------------------------------------------------------------------
layer_token_buffer_dual_bank #(
    .TOKEN_NUM(TOKEN_NUM),
    .DIM(BACKBONE_DIM),
    .EXP_WIDTH(EXP_WIDTH),
    .MANT_WIDTH(DATA_WIDTH_BB),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_layer_token_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    // Bank控制
    .bank_swap(bank_swap_pulse),
    .current_read_bank(current_read_bank),
    .current_write_bank(current_write_bank),
    
    // Backbone读接口（Port A）
    .backbone_rd_en(bb_layer_token_rd_en),
    .backbone_rd_addr(bb_layer_token_rd_addr),
    .backbone_rd_exp(bb_layer_token_rd_exp),
    .backbone_rd_mant(bb_layer_token_rd_mant),
    .backbone_rd_valid(bb_layer_token_rd_valid),
    
    // Sidenet读接口（Port B）
    .sidenet_rd_en(sn_layer_token_rd_en),
    .sidenet_rd_addr(sn_layer_token_rd_addr),
    .sidenet_rd_exp(sn_layer_token_rd_exp),
    .sidenet_rd_mant(sn_layer_token_rd_mant),
    .sidenet_rd_valid(sn_layer_token_rd_valid),
    
    // 写接口
    .layer_wr_en(bb_layer_token_wr_en),
    .layer_wr_addr(bb_layer_token_wr_addr),
    .layer_wr_exp(bb_layer_token_wr_exp),
    .layer_wr_mant(bb_layer_token_wr_mant),
    .layer_wr_ready(layer_wr_ready),
    
    // ❌ 删除：LN1暂存接口
    
    // DRAM加载接口
    .dram_load_en(load_input_start),
    .dram_load_addr(),  // TODO: 需要添加地址生成逻辑
    .dram_load_exp(),   // TODO: 需要连接DRAM数据转换
    .dram_load_mant(),
    
    // DRAM存储接口
    .dram_store_en(store_output_start),
    .dram_store_addr(),  // TODO: 需要添加地址生成逻辑
    .dram_store_exp(),
    .dram_store_mant(),
    .dram_store_valid()
);

//------------------------------------------------------------------------
// 4. Sidenet Layer 0（仅Compression）
//------------------------------------------------------------------------
sidenet_layer0 #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .BATCH_NUM(BATCH_NUM),
    .BACKBONE_DIM(BACKBONE_DIM),
    .SIDENET_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH_SN),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_sidenet_layer0 (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制接口
    .start(layer0_start),
    .done(layer0_done),
    .busy(layer0_busy),
    .error(layer0_error),
    
    // Backbone输出读取接口
    .backbone_rd_en(layer0_backbone_rd_en),
    .backbone_rd_addr(layer0_backbone_rd_addr),
    .backbone_rd_exp(sn_layer_token_rd_exp),
    .backbone_rd_mant(sn_layer_token_rd_mant),
    .backbone_rd_valid(sn_layer_token_rd_valid),
    
    // Layer Output Buffer写接口
    .layer0_wr_en(layer0_wr_en),
    .layer0_token_id(layer0_token_id),
    .layer0_exp(layer0_wr_exp),
    .layer0_mant(layer0_wr_mant),
    
    // Compression权重接口
    .compress_weight_req(layer0_compress_weight_req),
    .compress_weight_ready(layer0_compress_weight_ready),
    .compress_weight_exp(layer0_compress_weight_exp),
    .compress_weight_mant(layer0_compress_weight_mant),
    
    // 调试接口
    .dbg_state(),
    .dbg_compress_tokens()
);

//------------------------------------------------------------------------
// 5. Sidenet Layer Standard（Layer 1-3复用）
// ✅ 更新：新增FFN权重接口
//------------------------------------------------------------------------
sidenet_layer_standard #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .BATCH_NUM(BATCH_NUM),
    .BACKBONE_DIM(BACKBONE_DIM),
    .SIDENET_DIM(SIDENET_DIM),
    .NUM_HEADS(NUM_HEADS_SN),
    .HEAD_DIM(HEAD_DIM_SN),
    .D_FF(D_FF_SN),
    .FEATURE_CHUNK(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH_SN),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .K_CHUNK_SIZE(K_CHUNK_SIZE),
    .K_CHUNK_NUM(K_CHUNK_NUM),
    .NUM_LAYERS(NUM_SIDENET_LAYERS)
) u_sidenet_layer_standard (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制接口
    .start(layer_std_start),
    .layer_id(layer_std_id),
    .done(layer_std_done),
    .busy(layer_std_busy),
    
    // Backbone输出读取接口
    .backbone_rd_en(layer_std_backbone_rd_en),
    .backbone_rd_addr(layer_std_backbone_rd_addr),
    .backbone_rd_exp(sn_layer_token_rd_exp),
    .backbone_rd_mant(sn_layer_token_rd_mant),
    .backbone_rd_valid(sn_layer_token_rd_valid),
    
    // Layer Output Buffer读接口（读前一层）
    .prev_layer_rd_en(layer_std_prev_rd_en),
    .prev_layer_id(layer_std_prev_layer_id),
    .prev_layer_token_id(layer_std_prev_token_id),
    .prev_layer_exp(layer_std_prev_exp),
    .prev_layer_mant(layer_std_prev_mant),
    .prev_layer_valid(layer_std_prev_valid),
    
    // Layer Output Buffer写接口（写当前层）
    .curr_layer_wr_en(layer_std_curr_wr_en),
    .curr_layer_id(layer_std_curr_layer_id),
    .curr_layer_token_id(layer_std_curr_token_id),
    .curr_layer_exp(layer_std_curr_exp),
    .curr_layer_mant(layer_std_curr_mant),
    
    // Compression权重接口
    .compress_weight_req(layer_std_compress_weight_req),
    .compress_weight_ready(layer_std_compress_weight_ready),
    .compress_weight_exp(layer_std_compress_weight_exp),
    .compress_weight_mant(layer_std_compress_weight_mant),
    
    // QKV权重接口
    .qkv_weight_req(layer_std_qkv_weight_req),
    .qkv_weight_type(layer_std_qkv_weight_type),
    .qkv_weight_ack(layer_std_qkv_weight_ack),
    .qkv_weight_valid(layer_std_qkv_weight_valid),
    .qkv_weight_exp_array(layer_std_qkv_weight_exp_array),
    .qkv_weight_mant_blocks(layer_std_qkv_weight_mant_blocks),
    
    // WO权重接口
    .wo_weight_req(layer_std_wo_weight_req),
    .wo_weight_ready(layer_std_wo_weight_ready),
    .wo_weight_exp_array(layer_std_wo_weight_exp_array),
    .wo_weight_mant(layer_std_wo_weight_mant),
    
    // ✅ 新增：FFN权重接口
    .ffn_w1_weight_req(layer_std_ffn_w1_weight_req),
    .ffn_w1_weight_ready(layer_std_ffn_w1_weight_ready),
    .ffn_w1_weight_exp_array(layer_std_ffn_w1_weight_exp_array),
    .ffn_w1_weight_mant(layer_std_ffn_w1_weight_mant),
    
    .ffn_w2_weight_req(layer_std_ffn_w2_weight_req),
    .ffn_w2_weight_ready(layer_std_ffn_w2_weight_ready),
    .ffn_w2_weight_exp_array(layer_std_ffn_w2_weight_exp_array),
    .ffn_w2_weight_mant(layer_std_ffn_w2_weight_mant),
    
    // LayerNorm参数接口
    .ln_param_req(layer_std_ln_param_req),
    .ln_param_type(layer_std_ln_param_type),
    .ln_param_exp(layer_std_ln_param_exp),
    .ln_param_mant(layer_std_ln_param_mant),
    .ln_param_valid(layer_std_ln_param_valid),
    
    // 调试接口
    .dbg_fsm_state(),
    .dbg_compress_state(),
    .dbg_gate_state(),
    .dbg_transformer_state(),
    .dbg_att_state(),
    .dbg_cycle_count()
);

//------------------------------------------------------------------------
// 6. Sidenet Layer 4（含Expand）
//------------------------------------------------------------------------
sidenet_layer4 #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .BATCH_NUM(BATCH_NUM),
    .BACKBONE_DIM(BACKBONE_DIM),
    .SIDENET_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH_SN),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_sidenet_layer4 (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制接口
    .start(layer4_start),
    .done(layer4_done),
    .busy(layer4_busy),
    .error(layer4_error),
    
    // Backbone输出读取接口
    .backbone_rd_en(layer4_backbone_rd_en),
    .backbone_rd_addr(layer4_backbone_rd_addr),
    .backbone_rd_exp(sn_layer_token_rd_exp),
    .backbone_rd_mant(sn_layer_token_rd_mant),
    .backbone_rd_valid(sn_layer_token_rd_valid),
    
    // Layer 3输出读取接口
    .layer3_rd_en(layer4_layer3_rd_en),
    .layer3_token_id(layer4_layer3_token_id),
    .layer3_exp(layer4_layer3_exp),
    .layer3_mant(layer4_layer3_mant),
    .layer3_valid(layer4_layer3_valid),
    
    // 最终输出接口
    .final_wr_en(layer4_final_wr_en),
    .final_wr_addr(layer4_final_wr_addr),
    .final_wr_exp(layer4_final_wr_exp),
    .final_wr_mant(layer4_final_wr_mant),
    
    // Compression权重接口
    .compress_weight_req(layer4_compress_weight_req),
    .compress_weight_ready(layer4_compress_weight_ready),
    .compress_weight_exp(layer4_compress_weight_exp),
    .compress_weight_mant(layer4_compress_weight_mant),
    
    // Expand权重接口
    .expand_weight_req(layer4_expand_weight_req),
    .expand_weight_ready(layer4_expand_weight_ready),
    .expand_weight_exp(layer4_expand_weight_exp),
    .expand_weight_mant(layer4_expand_weight_mant),
    
    // 调试接口
    .dbg_state(),
    .dbg_compress_tokens(),
    .dbg_gate_tokens(),
    .dbg_expand_tokens()
);

//------------------------------------------------------------------------
// 7. Sidenet Layer Output Buffer（4-Bank独立存储）
//------------------------------------------------------------------------
sidenet_layer_output_buffer #(
    .TOKEN_NUM(TOKEN_NUM),
    .DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH_SN),
    .EXP_WIDTH(EXP_WIDTH),
    .NUM_LAYERS(4),
    .ADDR_WIDTH(ADDR_WIDTH),
    .LAYER_WIDTH(2)
) u_sidenet_layer_output_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    // 写接口
    .wr_en(sn_output_wr_en),
    .wr_layer_id(sn_output_wr_layer_id),
    .wr_token_id(sn_output_wr_token_id),
    .wr_exp(sn_output_wr_exp),
    .wr_mant(sn_output_wr_mant),
    .wr_ready(sn_output_wr_ready),
    .wr_error(sn_output_wr_error),
    
    // 推理读接口
    .rd_infer_en(sn_output_rd_en),
    .rd_infer_layer_id(sn_output_rd_layer_id),
    .rd_infer_token_id(sn_output_rd_token_id),
    .rd_infer_exp(sn_output_rd_exp),
    .rd_infer_mant(sn_output_rd_mant),
    .rd_infer_valid(sn_output_rd_valid),
    .rd_infer_error(sn_output_rd_error),
    
    // DFA训练读接口
    .rd_dfa_en(1'b0),
    .rd_dfa_layer_id(2'b0),
    .rd_dfa_token_id(10'b0),
    .rd_dfa_exp(),
    .rd_dfa_mant(),
    .rd_dfa_valid(),
    .rd_dfa_error(),
    
    // 调试接口
    .dbg_wr_count_l0(),
    .dbg_wr_count_l1(),
    .dbg_wr_count_l2(),
    .dbg_wr_count_l3(),
    .dbg_rd_count_l0(),
    .dbg_rd_count_l1(),
    .dbg_rd_count_l2(),
    .dbg_rd_count_l3()
);

//------------------------------------------------------------------------
// 8. Weight Controller（Backbone权重管理）
// ✅ 更新：新增FFN权重接口，确认ln_param_valid
//------------------------------------------------------------------------
weight_controller #(
    .NUM_LAYERS(NUM_LAYERS),
    .DIM(BACKBONE_DIM),
    .DATA_WIDTH(DATA_WIDTH_BB),
    .EXP_WIDTH(EXP_WIDTH),
    .DRAM_ADDR_WIDTH(DRAM_ADDR_WIDTH),
    .DRAM_DATA_WIDTH(DRAM_DATA_WIDTH),
    .CACHE_ENTRIES(4)
) u_weight_controller (
    .clk(clk),
    .rst_n(rst_n),
    
    // 配置接口
    .current_layer_id(current_backbone_layer),
    .weight_base_addr(32'h0),  // TODO: 配置实际基地址
    
    // QKV权重接口（数组格式）
    .qkv_weight_req(bb_qkv_weight_req),
    .qkv_weight_type(bb_qkv_weight_type),
    .qkv_weight_ack(bb_qkv_weight_ack),
    .qkv_weight_valid(bb_qkv_weight_valid),
    .qkv_weight_exp_array(bb_qkv_weight_exp_array),
    .qkv_weight_mant_blocks(bb_qkv_weight_mant_blocks),
    
    // WO权重接口（数组格式）
    .wo_weight_req(bb_wo_weight_req),
    .wo_weight_ready(bb_wo_weight_ready),
    .wo_weight_exp_array(bb_wo_weight_exp_array),
    .wo_weight_mant(bb_wo_weight_mant),
    
    // ✅ 新增：FFN权重接口（数组格式）
    .ffn_weight_req(bb_ffn_weight_req),
    .ffn_weight_type(bb_ffn_weight_type),
    .ffn_weight_chunk_id(bb_ffn_weight_chunk_id),
    .ffn_weight_ready(bb_ffn_weight_ready),
    .ffn_weight_exp_array(bb_ffn_weight_exp_array),
    .ffn_weight_mant(bb_ffn_weight_mant),
    
    // LayerNorm参数接口（单指数格式）
    .ln_param_req(bb_ln_param_req),
    .ln_param_type(bb_ln_param_type),
    .ln_param_valid(bb_ln_param_valid),  // ✅ 确认：valid
    .ln_param_exp(bb_ln_param_exp),
    .ln_param_mant(bb_ln_param_mant),
    
    // DMA接口
    .dma_req_valid(),
    .dma_req_addr(),
    .dma_req_burst_len(),
    .dma_req_ready(1'b1),  // TODO: 连接到实际DMA
    
    .dma_rsp_valid(1'b0),  // TODO: 连接到实际DMA
    .dma_rsp_data(256'b0),
    .dma_rsp_last(1'b0),
    .dma_rsp_ready(),
    
    // 调试接口
    .dbg_state(),
    .dbg_cache_hit_count(),
    .dbg_cache_miss_count(),
    .dbg_dma_req_count()
);

//------------------------------------------------------------------------
// 9. Sidenet Weight Storage（Sidenet权重存储）
//------------------------------------------------------------------------
sidenet_weight_storage #(
    .NUM_LAYERS(NUM_SIDENET_LAYERS),
    .BACKBONE_DIM(BACKBONE_DIM),
    .SIDENET_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH_SN),
    .EXP_WIDTH(EXP_WIDTH),
    .DRAM_DATA_WIDTH(DRAM_DATA_WIDTH),
    .WEIGHTS_PER_BURST(DRAM_DATA_WIDTH/DATA_WIDTH_SN),
    .MAX_EXP_ARRAY_WIDTH(BACKBONE_DIM*EXP_WIDTH)
) u_sidenet_weight_storage (
    .clk(clk),
    .rst_n(rst_n),
    
    // BANK切换
    .bank_swap(1'b0),  // TODO: 训练时需要
    .current_read_bank(),
    .current_write_bank(),
    
    // 读接口
    .rd_en(sn_weight_storage_rd_en),
    .rd_layer_id(sn_weight_storage_rd_layer_id),
    .rd_weight_type(sn_weight_storage_rd_weight_type),
    .rd_burst_idx(sn_weight_storage_rd_burst_idx),
    .rd_valid(sn_weight_storage_rd_valid),
    .rd_exp_array(sn_weight_storage_rd_exp_array),
    .rd_data_burst(sn_weight_storage_rd_data_burst),
    
    // 写接口
    .wr_en(sn_weight_storage_wr_en),
    .wr_layer_id(sn_weight_storage_wr_layer_id),
    .wr_weight_type(sn_weight_storage_wr_weight_type),
    .wr_burst_idx(sn_weight_storage_wr_burst_idx),
    .wr_exp_array(sn_weight_storage_wr_exp_array),
    .wr_data_burst(sn_weight_storage_wr_data_burst),
    .wr_ready(sn_weight_storage_wr_ready),
    
    // DMA初始化
    .dma_init_en(1'b0),
    .dma_init_bank(1'b0),
    .dma_init_layer_id(3'b0),
    .dma_init_weight_type(4'b0),
    .dma_init_burst_idx(6'b0),
    .dma_init_exp_array({BACKBONE_DIM*EXP_WIDTH{1'b0}}),
    .dma_init_data(256'b0),
    .dma_init_ready(),
    
    // 调试接口
    .dbg_rd_count(),
    .dbg_wr_count(),
    .dbg_dma_count()
);

//------------------------------------------------------------------------
// 10. Sidenet Weight Controller（Sidenet权重管理）
// ✅ 更新：新增FFN权重接口
//------------------------------------------------------------------------
sidenet_weight_controller #(
    .NUM_LAYERS(NUM_SIDENET_LAYERS),
    .BACKBONE_DIM(BACKBONE_DIM),
    .SIDENET_DIM(SIDENET_DIM),
    .D_FF(D_FF_SN),
    .DATA_WIDTH(DATA_WIDTH_SN),
    .EXP_WIDTH(EXP_WIDTH),
    .DRAM_DATA_WIDTH(DRAM_DATA_WIDTH),
    .MAX_EXP_ARRAY_WIDTH(BACKBONE_DIM*EXP_WIDTH)
) u_sidenet_weight_controller (
    .clk(clk),
    .rst_n(rst_n),
    
    // 模式控制
    .training_mode(1'b0),
    .forward_phase(1'b1),
    .current_layer_id(current_sidenet_layer),
    
    // Layer 0 权重接口
    .layer0_compress_weight_req(layer0_compress_weight_req),
    .layer0_compress_weight_ready(layer0_compress_weight_ready),
    .layer0_compress_weight_exp_array(layer0_compress_weight_exp),
    .layer0_compress_weight_mant(layer0_compress_weight_mant),
    
    // Layer Standard (1-3) 权重接口
    .layer_std_compress_weight_req(layer_std_compress_weight_req),
    .layer_std_compress_weight_ready(layer_std_compress_weight_ready),
    .layer_std_compress_weight_exp_array(layer_std_compress_weight_exp),
    .layer_std_compress_weight_mant(layer_std_compress_weight_mant),
    
    // Layer Standard QKV权重接口
    .layer_std_att_weight_req(layer_std_qkv_weight_req),
    .layer_std_att_weight_type(layer_std_qkv_weight_type),
    .layer_std_att_weight_ready(layer_std_qkv_weight_ready),
    .layer_std_att_weight_exp_array(layer_std_qkv_weight_exp_array),
    .layer_std_att_weight_mant(layer_std_qkv_weight_mant_blocks),
    
    // Layer Standard WO权重接口
    .layer_std_wo_weight_req(layer_std_wo_weight_req),
    .layer_std_wo_weight_ready(layer_std_wo_weight_ready),
    .layer_std_wo_weight_exp_array(layer_std_wo_weight_exp_array),
    .layer_std_wo_weight_mant(layer_std_wo_weight_mant),
    
    // ✅ 新增：Layer Standard FFN权重接口
    .layer_std_ffn_w1_weight_req(layer_std_ffn_w1_weight_req),
    .layer_std_ffn_w1_weight_ready(layer_std_ffn_w1_weight_ready),
    .layer_std_ffn_w1_weight_exp_array(layer_std_ffn_w1_weight_exp_array),
    .layer_std_ffn_w1_weight_mant(layer_std_ffn_w1_weight_mant),
    
    .layer_std_ffn_w2_weight_req(layer_std_ffn_w2_weight_req),
    .layer_std_ffn_w2_weight_ready(layer_std_ffn_w2_weight_ready),
    .layer_std_ffn_w2_weight_exp_array(layer_std_ffn_w2_weight_exp_array),
    .layer_std_ffn_w2_weight_mant(layer_std_ffn_w2_weight_mant),
    
    // Layer 4 权重接口
    .layer4_compress_weight_req(layer4_compress_weight_req),
    .layer4_compress_weight_ready(layer4_compress_weight_ready),
    .layer4_compress_weight_exp_array(layer4_compress_weight_exp),
    .layer4_compress_weight_mant(layer4_compress_weight_mant),
    
    .layer4_expand_weight_req(layer4_expand_weight_req),
    .layer4_expand_weight_ready(layer4_expand_weight_ready),
    .layer4_expand_weight_exp_array(layer4_expand_weight_exp),
    .layer4_expand_weight_mant(layer4_expand_weight_mant),
    
    // DFA权重更新接口
    .dfa_weight_update_req(1'b0),
    .dfa_layer_id(3'b0),
    .dfa_weight_type(4'b0),
    .dfa_weight_exp_array({BACKBONE_DIM*EXP_WIDTH{1'b0}}),
    .dfa_weight_data(256'b0),
    .dfa_weight_burst_idx(6'b0),
    .dfa_weight_update_ready(),
    
    // Storage读接口
    .storage_rd_en(sn_weight_storage_rd_en),
    .storage_rd_layer_id(sn_weight_storage_rd_layer_id),
    .storage_rd_weight_type(sn_weight_storage_rd_weight_type),
    .storage_rd_burst_idx(sn_weight_storage_rd_burst_idx),
    .storage_rd_valid(sn_weight_storage_rd_valid),
    .storage_rd_exp_array(sn_weight_storage_rd_exp_array),
    .storage_rd_data_burst(sn_weight_storage_rd_data_burst),
    
    // Storage写接口
    .storage_wr_en(sn_weight_storage_wr_en),
    .storage_wr_layer_id(sn_weight_storage_wr_layer_id),
    .storage_wr_weight_type(sn_weight_storage_wr_weight_type),
    .storage_wr_burst_idx(sn_weight_storage_wr_burst_idx),
    .storage_wr_exp_array(sn_weight_storage_wr_exp_array),
    .storage_wr_data_burst(sn_weight_storage_wr_data_burst),
    .storage_wr_ready(sn_weight_storage_wr_ready),
    
    // 调试接口
    .dbg_state(),
    .dbg_read_count(),
    .dbg_write_count()
);

endmodule