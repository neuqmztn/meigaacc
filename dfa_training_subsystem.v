`timescale 1ns / 1ps

//================================================================================
// DFA Training Subsystem - 完整DFA训练子系统
//
// 功能说明：
// 完整的DFA训练子系统，包含：
// 1. DFA Master Controller (训练流程控制)
// 2. 5个 Gradient Compute Units (Layer 0-4)
// 3. DFA Matrix Bank (B矩阵存储)
// 4. Gradient Buffer (梯度缓存)
// 5. Weight Update Engine (权重更新)
//
// 不包含：
// • classification_subsystem (独立的分类系统)
//
// 设计理念：
// • 完整封装DFA训练流程
// • 内部控制器协调各模块
// • 对外提供简单接口
//
// 数据流：
// error (来自分类系统) → 5×GCU → Gradient Buffer → Weight Update Engine → Sidenet Weights
//          ↑                        ↑
//      B Matrix                Activation (LOB)
//
// 作者：MEIGA Team
// 日期：2025-11-19
// 版本：v2.0 (包含控制器)
//================================================================================

module dfa_training_subsystem #(
    //==========================================================================
    // Layer参数
    //==========================================================================
    parameter NUM_LAYERS        = 5,
    parameter DIM_SMALL         = 8,
    parameter DIM_LARGE         = 32,
    parameter NUM_TOKENS        = 641,
    parameter DATA_WIDTH        = 16,
    parameter EXP_WIDTH         = 8,
    
    //==========================================================================
    // 地址参数
    //==========================================================================
    parameter TOKEN_ADDR_WIDTH  = 10,
    parameter DIM_ADDR_WIDTH    = 5,
    
    //==========================================================================
    // DFA参数
    //==========================================================================
    parameter NUM_CLASSES       = 1,
    
    //==========================================================================
    // 训练参数
    //==========================================================================
    parameter MAX_BATCH_SIZE    = 32,
    parameter MAX_EPOCHS        = 100,
    parameter DEFAULT_LR        = 16'h0029,  // 0.01
    
    //==========================================================================
    // 权重更新参数
    //==========================================================================
    parameter DRAM_DATA_WIDTH   = 256,
    parameter MAX_EXP_ARRAY_WIDTH = 256
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 训练控制接口（顶层）
    //==========================================================================
    input  wire        train_start,
    input  wire [7:0]  cfg_batch_size,
    input  wire [7:0]  cfg_num_epochs,
    input  wire [15:0] cfg_learning_rate,
    output wire        train_done,
    output wire        train_active,
    
    //==========================================================================
    // 前向传播接口（连接Backbone）
    //==========================================================================
    output wire        forward_start,
    input  wire        forward_done,
    output wire        capture_enable,
    
    //==========================================================================
    // 误差输入接口（来自classification_subsystem）
    //==========================================================================
    output wire        error_start,          // 启动误差计算
    input  wire        error_done,           // 误差计算完成
    input  wire        error_valid,          // 误差有效
    input  wire [DATA_WIDTH-1:0] error_scalar,  // 误差标量
    
    //==========================================================================
    // LOB读取接口 - 连接到layer_token_buffer (Layer 0-4)
    //==========================================================================
    output wire        lob_rd_dfa0_en,
    output wire [TOKEN_ADDR_WIDTH-1:0] lob_rd_dfa0_addr,
    input  wire [DIM_SMALL*DATA_WIDTH-1:0] lob_dfa0_vector,
    input  wire        lob_dfa0_valid,
    
    output wire        lob_rd_dfa1_en,
    output wire [TOKEN_ADDR_WIDTH-1:0] lob_rd_dfa1_addr,
    input  wire [DIM_SMALL*DATA_WIDTH-1:0] lob_dfa1_vector,
    input  wire        lob_dfa1_valid,
    
    output wire        lob_rd_dfa2_en,
    output wire [TOKEN_ADDR_WIDTH-1:0] lob_rd_dfa2_addr,
    input  wire [DIM_SMALL*DATA_WIDTH-1:0] lob_dfa2_vector,
    input  wire        lob_dfa2_valid,
    
    output wire        lob_rd_dfa3_en,
    output wire [TOKEN_ADDR_WIDTH-1:0] lob_rd_dfa3_addr,
    input  wire [DIM_SMALL*DATA_WIDTH-1:0] lob_dfa3_vector,
    input  wire        lob_dfa3_valid,
    
    output wire        lob_rd_dfa4_en,
    output wire [TOKEN_ADDR_WIDTH-1:0] lob_rd_dfa4_addr,
    input  wire [DIM_LARGE*DATA_WIDTH-1:0] lob_dfa4_vector,
    input  wire        lob_dfa4_valid,
    
    //==========================================================================
    // 权重存储接口 - 连接到sidenet_weight_storage
    //==========================================================================
    output wire        weight_rd_req,
    output wire [2:0]  weight_rd_layer_id,
    output wire [3:0]  weight_rd_type,
    output wire [4:0]  weight_rd_col_id,
    output wire [4:0]  weight_rd_row_id,
    input  wire        weight_rd_valid,
    input  wire [EXP_WIDTH-1:0]  weight_rd_exp,
    input  wire [DATA_WIDTH-1:0] weight_rd_mant,
    
    output wire        weight_wr_req,
    output wire [2:0]  weight_wr_layer_id,
    output wire [3:0]  weight_wr_type,
    output wire [5:0]  weight_wr_burst_idx,
    output wire [MAX_EXP_ARRAY_WIDTH-1:0] weight_wr_exp_array,
    output wire [DRAM_DATA_WIDTH-1:0] weight_wr_data_burst,
    input  wire        weight_wr_ready,
    
    //==========================================================================
    // Bank切换接口
    //==========================================================================
    output wire        request_bank_switch,
    input  wire        bank_switch_ready,
    output wire        confirm_switch,
    input  wire        switch_complete,
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [3:0]  dbg_controller_state,
    output wire [31:0] dbg_iteration_count,
    output wire [7:0]  dbg_batch_count,
    output wire [7:0]  dbg_epoch_count,
    output wire        dbg_timeout_error,
    output wire [2:0]  dbg_gcu0_state,
    output wire [31:0] dbg_gcu0_cycles,
    output wire [3:0]  dbg_update_state,
    output wire [31:0] dbg_update_count
);

//================================================================================
// 内部信号：控制器 → B矩阵
//================================================================================
wire        init_b_start;
wire        init_b_done;
wire        b_matrices_exist;

//================================================================================
// 内部信号：控制器 → GCU
//================================================================================
wire [4:0]  gcu_start;
wire [4:0]  gcu_done;
wire [4:0]  gcu_busy;

//================================================================================
// 内部信号：控制器 → 权重更新
//================================================================================
wire        update_start;
wire        update_done;
wire        update_busy;

//================================================================================
// 内部信号：B矩阵读取接口
//================================================================================
wire        gcu0_b_rd_en, gcu1_b_rd_en, gcu2_b_rd_en, gcu3_b_rd_en, gcu4_b_rd_en;
wire [2:0]  gcu0_b_row_addr, gcu1_b_row_addr, gcu2_b_row_addr, gcu3_b_row_addr;
wire [4:0]  gcu4_b_row_addr;
wire [3:0]  gcu0_b_col_addr, gcu1_b_col_addr, gcu2_b_col_addr, gcu3_b_col_addr, gcu4_b_col_addr;
wire [DATA_WIDTH-1:0] gcu0_b_data, gcu1_b_data, gcu2_b_data, gcu3_b_data, gcu4_b_data;
wire        gcu0_b_valid, gcu1_b_valid, gcu2_b_valid, gcu3_b_valid, gcu4_b_valid;

//================================================================================
// 内部信号：梯度写入接口
//================================================================================
wire        gcu0_grad_wr_en, gcu1_grad_wr_en, gcu2_grad_wr_en, gcu3_grad_wr_en, gcu4_grad_wr_en;
wire [TOKEN_ADDR_WIDTH-1:0] gcu0_grad_token, gcu1_grad_token, gcu2_grad_token, gcu3_grad_token, gcu4_grad_token;
wire [DIM_ADDR_WIDTH-1:0] gcu0_grad_dim, gcu1_grad_dim, gcu2_grad_dim, gcu3_grad_dim, gcu4_grad_dim;
wire [DATA_WIDTH-1:0] gcu0_grad_data, gcu1_grad_data, gcu2_grad_data, gcu3_grad_data, gcu4_grad_data;

//================================================================================
// 内部信号：权重更新引擎 → 梯度缓存
//================================================================================
wire        grad_rd_req;
wire [2:0]  grad_rd_layer_id;
wire [9:0]  grad_rd_token_id;
wire [4:0]  grad_rd_dim_id;
wire        grad_rd_valid;
wire [DATA_WIDTH-1:0] grad_rd_data;

//================================================================================
// 模块实例化：DFA Master Controller
//================================================================================
dfa_master_controller #(
    .MAX_BATCH_SIZE(MAX_BATCH_SIZE),
    .MAX_EPOCHS(MAX_EPOCHS),
    .FORWARD_TIMEOUT(2000),
    .ERROR_TIMEOUT(500),
    .GRADIENT_TIMEOUT(25000),
    .UPDATE_TIMEOUT(15000),
    .BANK_SWITCH_DELAY(10),
    .GCU_PARALLEL_START(1)
) u_dfa_master_controller (
    .clk(clk),
    .rst_n(rst_n),
    
    // 训练控制
    .train_start(train_start),
    .cfg_batch_size(cfg_batch_size),
    .cfg_num_epochs(cfg_num_epochs),
    .train_done(train_done),
    .train_active(train_active),
    
    // B矩阵初始化
    .b_matrices_exist(b_matrices_exist),
    .init_b_matrices(init_b_start),
    .b_init_done(init_b_done),
    
    // 前向传播控制
    .forward_start(forward_start),
    .forward_done(forward_done),
    .capture_enable(capture_enable),
    
    // 误差计算接口
    .error_start(error_start),
    .error_done(error_done),
    .error_valid(error_valid),
    
    // 梯度计算控制
    .grad_start(gcu_start),
    .grad_done(gcu_done),
    
    // 权重更新控制
    .update_start(update_start),
    .update_done(update_done),
    
    // Bank切换
    .request_bank_switch(request_bank_switch),
    .bank_switch_ready(bank_switch_ready),
    .confirm_switch(confirm_switch),
    .switch_complete(switch_complete),
    
    // 调试
    .current_state(dbg_controller_state),
    .iteration_counter(dbg_iteration_count),
    .batch_counter(dbg_batch_count),
    .epoch_counter(dbg_epoch_count),
    .total_samples(),
    .timeout_error(dbg_timeout_error),
    .timeout_counter()
);

//================================================================================
// 模块实例化：5个Gradient Compute Units
//================================================================================

// GCU 0 - Layer 0 (DIM=8)
gradient_compute_unit #(
    .LAYER_ID(0),
    .DIM(DIM_SMALL),
    .NUM_CLASSES(NUM_CLASSES),
    .NUM_TOKENS(NUM_TOKENS),
    .DATA_WIDTH(DATA_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH)
) u_gcu_layer0 (
    .clk(clk),
    .rst_n(rst_n),
    .start(gcu_start[0]),
    .done(gcu_done[0]),
    .busy(gcu_busy[0]),
    .b_rd_en(gcu0_b_rd_en),
    .b_row_addr(gcu0_b_row_addr),
    .b_col_addr(gcu0_b_col_addr),
    .b_data(gcu0_b_data),
    .b_valid(gcu0_b_valid),
    .error_scalar(error_scalar),
    .act_rd_en(lob_rd_dfa0_en),
    .act_token_addr(lob_rd_dfa0_addr),
    .act_vector(lob_dfa0_vector),
    .act_valid(lob_dfa0_valid),
    .grad_wr_en(gcu0_grad_wr_en),
    .grad_token_addr(gcu0_grad_token),
    .grad_dim_addr(gcu0_grad_dim),
    .grad_data(gcu0_grad_data),
    .state(dbg_gcu0_state),
    .cycle_count(dbg_gcu0_cycles),
    .token_count()
);

// GCU 1 - Layer 1 (DIM=8)
gradient_compute_unit #(
    .LAYER_ID(1),
    .DIM(DIM_SMALL),
    .NUM_CLASSES(NUM_CLASSES),
    .NUM_TOKENS(NUM_TOKENS),
    .DATA_WIDTH(DATA_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH)
) u_gcu_layer1 (
    .clk(clk),
    .rst_n(rst_n),
    .start(gcu_start[1]),
    .done(gcu_done[1]),
    .busy(gcu_busy[1]),
    .b_rd_en(gcu1_b_rd_en),
    .b_row_addr(gcu1_b_row_addr),
    .b_col_addr(gcu1_b_col_addr),
    .b_data(gcu1_b_data),
    .b_valid(gcu1_b_valid),
    .error_scalar(error_scalar),
    .act_rd_en(lob_rd_dfa1_en),
    .act_token_addr(lob_rd_dfa1_addr),
    .act_vector(lob_dfa1_vector),
    .act_valid(lob_dfa1_valid),
    .grad_wr_en(gcu1_grad_wr_en),
    .grad_token_addr(gcu1_grad_token),
    .grad_dim_addr(gcu1_grad_dim),
    .grad_data(gcu1_grad_data),
    .state(),
    .cycle_count(),
    .token_count()
);

// GCU 2 - Layer 2 (DIM=8)
gradient_compute_unit #(
    .LAYER_ID(2),
    .DIM(DIM_SMALL),
    .NUM_CLASSES(NUM_CLASSES),
    .NUM_TOKENS(NUM_TOKENS),
    .DATA_WIDTH(DATA_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH)
) u_gcu_layer2 (
    .clk(clk),
    .rst_n(rst_n),
    .start(gcu_start[2]),
    .done(gcu_done[2]),
    .busy(gcu_busy[2]),
    .b_rd_en(gcu2_b_rd_en),
    .b_row_addr(gcu2_b_row_addr),
    .b_col_addr(gcu2_b_col_addr),
    .b_data(gcu2_b_data),
    .b_valid(gcu2_b_valid),
    .error_scalar(error_scalar),
    .act_rd_en(lob_rd_dfa2_en),
    .act_token_addr(lob_rd_dfa2_addr),
    .act_vector(lob_dfa2_vector),
    .act_valid(lob_dfa2_valid),
    .grad_wr_en(gcu2_grad_wr_en),
    .grad_token_addr(gcu2_grad_token),
    .grad_dim_addr(gcu2_grad_dim),
    .grad_data(gcu2_grad_data),
    .state(),
    .cycle_count(),
    .token_count()
);

// GCU 3 - Layer 3 (DIM=8)
gradient_compute_unit #(
    .LAYER_ID(3),
    .DIM(DIM_SMALL),
    .NUM_CLASSES(NUM_CLASSES),
    .NUM_TOKENS(NUM_TOKENS),
    .DATA_WIDTH(DATA_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH)
) u_gcu_layer3 (
    .clk(clk),
    .rst_n(rst_n),
    .start(gcu_start[3]),
    .done(gcu_done[3]),
    .busy(gcu_busy[3]),
    .b_rd_en(gcu3_b_rd_en),
    .b_row_addr(gcu3_b_row_addr),
    .b_col_addr(gcu3_b_col_addr),
    .b_data(gcu3_b_data),
    .b_valid(gcu3_b_valid),
    .error_scalar(error_scalar),
    .act_rd_en(lob_rd_dfa3_en),
    .act_token_addr(lob_rd_dfa3_addr),
    .act_vector(lob_dfa3_vector),
    .act_valid(lob_dfa3_valid),
    .grad_wr_en(gcu3_grad_wr_en),
    .grad_token_addr(gcu3_grad_token),
    .grad_dim_addr(gcu3_grad_dim),
    .grad_data(gcu3_grad_data),
    .state(),
    .cycle_count(),
    .token_count()
);

// GCU 4 - Layer 4 (DIM=32)
gradient_compute_unit #(
    .LAYER_ID(4),
    .DIM(DIM_LARGE),
    .NUM_CLASSES(NUM_CLASSES),
    .NUM_TOKENS(NUM_TOKENS),
    .DATA_WIDTH(DATA_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH)
) u_gcu_layer4 (
    .clk(clk),
    .rst_n(rst_n),
    .start(gcu_start[4]),
    .done(gcu_done[4]),
    .busy(gcu_busy[4]),
    .b_rd_en(gcu4_b_rd_en),
    .b_row_addr(gcu4_b_row_addr),
    .b_col_addr(gcu4_b_col_addr),
    .b_data(gcu4_b_data),
    .b_valid(gcu4_b_valid),
    .error_scalar(error_scalar),
    .act_rd_en(lob_rd_dfa4_en),
    .act_token_addr(lob_rd_dfa4_addr),
    .act_vector(lob_dfa4_vector),
    .act_valid(lob_dfa4_valid),
    .grad_wr_en(gcu4_grad_wr_en),
    .grad_token_addr(gcu4_grad_token),
    .grad_dim_addr(gcu4_grad_dim),
    .grad_data(gcu4_grad_data),
    .state(),
    .cycle_count(),
    .token_count()
);

//================================================================================
// 模块实例化：DFA Matrix Bank
//================================================================================
dfa_matrix_bank #(
    .NUM_CLASSES(NUM_CLASSES),
    .LAYER0_DIM(DIM_SMALL),
    .LAYER4_DIM(DIM_LARGE),
    .DATA_WIDTH(DATA_WIDTH)
) u_dfa_matrix_bank (
    .clk(clk),
    .rst_n(rst_n),
    .init_start(init_b_start),
    .init_done(init_b_done),
    .matrices_exist(b_matrices_exist),
    .gcu0_rd_en(gcu0_b_rd_en),
    .gcu0_row_addr(gcu0_b_row_addr),
    .gcu0_col_addr(gcu0_b_col_addr),
    .gcu0_data(gcu0_b_data),
    .gcu0_valid(gcu0_b_valid),
    .gcu1_rd_en(gcu1_b_rd_en),
    .gcu1_row_addr(gcu1_b_row_addr),
    .gcu1_col_addr(gcu1_b_col_addr),
    .gcu1_data(gcu1_b_data),
    .gcu1_valid(gcu1_b_valid),
    .gcu2_rd_en(gcu2_b_rd_en),
    .gcu2_row_addr(gcu2_b_row_addr),
    .gcu2_col_addr(gcu2_b_col_addr),
    .gcu2_data(gcu2_b_data),
    .gcu2_valid(gcu2_b_valid),
    .gcu3_rd_en(gcu3_b_rd_en),
    .gcu3_row_addr(gcu3_b_row_addr),
    .gcu3_col_addr(gcu3_b_col_addr),
    .gcu3_data(gcu3_b_data),
    .gcu3_valid(gcu3_b_valid),
    .gcu4_rd_en(gcu4_b_rd_en),
    .gcu4_row_addr(gcu4_b_row_addr),
    .gcu4_col_addr(gcu4_b_col_addr),
    .gcu4_data(gcu4_b_data),
    .gcu4_valid(gcu4_b_valid),
    .delta_rd_en(1'b0),
    .delta_layer_id(3'd0),
    .delta_row_addr(5'd0),
    .delta_col_addr(4'd0),
    .delta_data(),
    .delta_valid(),
    .init_state(),
    .init_counter()
);

//================================================================================
// 模块实例化：Gradient Buffer
//================================================================================
gradient_buffer #(
    .NUM_TOKENS(NUM_TOKENS),
    .DIM_SMALL(DIM_SMALL),
    .DIM_LARGE(DIM_LARGE),
    .DATA_WIDTH(DATA_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH)
) u_gradient_buffer (
    .clk(clk),
    .rst_n(rst_n),
    .gcu0_wr_en(gcu0_grad_wr_en),
    .gcu0_token_addr(gcu0_grad_token),
    .gcu0_dim_addr(gcu0_grad_dim),
    .gcu0_data(gcu0_grad_data),
    .gcu1_wr_en(gcu1_grad_wr_en),
    .gcu1_token_addr(gcu1_grad_token),
    .gcu1_dim_addr(gcu1_grad_dim),
    .gcu1_data(gcu1_grad_data),
    .gcu2_wr_en(gcu2_grad_wr_en),
    .gcu2_token_addr(gcu2_grad_token),
    .gcu2_dim_addr(gcu2_grad_dim),
    .gcu2_data(gcu2_grad_data),
    .gcu3_wr_en(gcu3_grad_wr_en),
    .gcu3_token_addr(gcu3_grad_token),
    .gcu3_dim_addr(gcu3_grad_dim),
    .gcu3_data(gcu3_grad_data),
    .gcu4_wr_en(gcu4_grad_wr_en),
    .gcu4_token_addr(gcu4_grad_token),
    .gcu4_dim_addr(gcu4_grad_dim),
    .gcu4_data(gcu4_grad_data),
    .rd_en(1'b0),
    .rd_layer_id(3'd0),
    .rd_addr(10'd0),
    .rd_data(),
    .rd_valid(),
    .rd_simple_en(grad_rd_req),
    .rd_simple_layer(grad_rd_layer_id),
    .rd_simple_token(grad_rd_token_id),
    .rd_simple_dim(grad_rd_dim_id),
    .rd_simple_data(grad_rd_data),
    .rd_simple_valid(grad_rd_valid)
);

//================================================================================
// 模块实例化：Weight Update Engine
//================================================================================
weight_update_engine #(
    .NUM_LAYERS(NUM_LAYERS),
    .BACKBONE_DIM(DIM_LARGE),
    .SIDENET_DIM(DIM_SMALL),
    .D_FF(DIM_LARGE),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .Q412_WIDTH(DATA_WIDTH),
    .DRAM_DATA_WIDTH(DRAM_DATA_WIDTH),
    .LR_Q412_WIDTH(DATA_WIDTH)
) u_weight_update_engine (
    .clk(clk),
    .rst_n(rst_n),
    .update_start(update_start),
    .update_done(update_done),
    .update_busy(update_busy),
    .cfg_layer_id(3'd0),  // 暂时固定，后续可扩展
    .cfg_weight_type(4'd0),
    .cfg_learning_rate(cfg_learning_rate),
    .weight_rd_req(weight_rd_req),
    .weight_rd_layer_id(weight_rd_layer_id),
    .weight_rd_type(weight_rd_type),
    .weight_rd_col_id(weight_rd_col_id),
    .weight_rd_row_id(weight_rd_row_id),
    .weight_rd_valid(weight_rd_valid),
    .weight_rd_exp(weight_rd_exp),
    .weight_rd_mant(weight_rd_mant),
    .grad_rd_req(grad_rd_req),
    .grad_rd_layer_id(grad_rd_layer_id),
    .grad_rd_token_id(grad_rd_token_id),
    .grad_rd_dim_id(grad_rd_dim_id),
    .grad_rd_valid(grad_rd_valid),
    .grad_rd_data(grad_rd_data),
    .weight_wr_req(weight_wr_req),
    .weight_wr_layer_id(weight_wr_layer_id),
    .weight_wr_type(weight_wr_type),
    .weight_wr_burst_idx(weight_wr_burst_idx),
    .weight_wr_exp_array(weight_wr_exp_array),
    .weight_wr_data_burst(weight_wr_data_burst),
    .weight_wr_ready(weight_wr_ready),
    .dbg_state(dbg_update_state),
    .dbg_update_count(dbg_update_count),
    .dbg_row_count(),
    .dbg_col_count(),
    .dbg_overflow_count(),
    .dbg_current_lr()
);

endmodule