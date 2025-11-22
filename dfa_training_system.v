`timescale 1ns / 1ps

//================================================================================
// DFA Training System - 顶层模块
//
// 功能说明：
// 连接两个主要子系统：
// 1. Classification Subsystem - 分类与误差计算
// 2. DFA Training Subsystem - DFA训练（包含控制器）
//
// 设计理念：
// • 顶层只负责连接，不含控制逻辑
// • 分类系统和DFA系统完全独立
// • 接口清晰，易于维护
//
// 数据流：
// Input → Backbone → LOB → Classification → Error
//                    ↓                       ↓
//                    └─────→ DFA Training ←──┘
//                            ↓
//                         Sidenet Weights
//
// 作者：MEIGA Team
// 日期：2025-11-19
// 版本：v3.0 (简化顶层)
//================================================================================

module dfa_training_system #(
    //==========================================================================
    // 基本参数
    //==========================================================================
    parameter DIM = 32,
    parameter DIM_SMALL = 8,
    parameter NUM_TOKENS = 641,
    parameter DATA_WIDTH = 16,
    parameter EXP_WIDTH = 8,
    parameter TOKEN_ADDR_WIDTH = 10,
    parameter DIM_ADDR_WIDTH = 5,
    
    //==========================================================================
    // 训练参数
    //==========================================================================
    parameter MAX_BATCH_SIZE = 32,
    parameter MAX_EPOCHS = 100,
    parameter DEFAULT_LR = 16'h0029,
    
    //==========================================================================
    // Burst参数
    //==========================================================================
    parameter DRAM_DATA_WIDTH = 256,
    parameter MAX_EXP_ARRAY_WIDTH = 256
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 训练控制接口
    //==========================================================================
    input  wire        train_start,
    input  wire        train_mode,
    input  wire        true_label,
    output wire        train_done,
    output wire        train_active,
    
    //==========================================================================
    // 训练配置
    //==========================================================================
    input  wire [7:0]  cfg_batch_size,
    input  wire [7:0]  cfg_num_epochs,
    input  wire [15:0] cfg_learning_rate,
    
    //==========================================================================
    // 前向传播接口
    //==========================================================================
    output wire        forward_start,
    input  wire        forward_done,
    output wire        capture_enable,
    
    //==========================================================================
    // LOB读取接口 - CLS token
    //==========================================================================
    output wire        lob_rd_cls_en,
    input  wire [DIM*DATA_WIDTH-1:0] lob_cls_data_q412,
    input  wire        lob_cls_valid,
    
    //==========================================================================
    // LOB读取接口 - Layer 0-4 (DFA)
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
    input  wire [DIM*DATA_WIDTH-1:0] lob_dfa4_vector,
    input  wire        lob_dfa4_valid,
    
    //==========================================================================
    // 分类头权重加载接口
    //==========================================================================
    input  wire        cls_weight_load_en,
    input  wire [4:0]  cls_weight_load_addr,
    input  wire [DATA_WIDTH-1:0] cls_weight_load_data,
    input  wire [DATA_WIDTH-1:0] cls_bias_load_data,
    
    //==========================================================================
    // Sidenet权重存储接口
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
    // 分类结果输出
    //==========================================================================
    output wire        result_valid,
    output wire [DATA_WIDTH-1:0] prob,
    output wire        predicted_class,
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [3:0]  dbg_controller_state,
    output wire [31:0] dbg_iteration_count,
    output wire [7:0]  dbg_batch_count,
    output wire [7:0]  dbg_epoch_count,
    output wire        dbg_timeout_error,
    output wire [3:0]  dbg_cls_state,
    output wire [DATA_WIDTH-1:0] dbg_error_value,
    output wire [2:0]  dbg_gcu0_state,
    output wire [31:0] dbg_gcu0_cycles
);

//================================================================================
// 内部信号：分类子系统 ↔ DFA训练子系统
//================================================================================
wire        cls_start;
wire        cls_done;
wire        cls_busy;
wire [DATA_WIDTH-1:0] error_scalar;
wire        error_valid;

//================================================================================
// 模块实例化：Classification Subsystem
//================================================================================
classification_subsystem #(
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH)
) u_classification_subsystem (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制
    .start(cls_start),
    .train_mode(train_mode),
    .done(cls_done),
    .busy(cls_busy),
    
    // 标签输入
    .true_label(true_label),
    
    // LOB读取接口
    .lob_rd_cls_en(lob_rd_cls_en),
    .lob_cls_data_q412(lob_cls_data_q412),
    .lob_cls_valid(lob_cls_valid),
    
    // 权重加载接口
    .weight_load_en(cls_weight_load_en),
    .weight_load_addr(cls_weight_load_addr),
    .weight_load_data(cls_weight_load_data),
    .bias_load_data(cls_bias_load_data),
    
    // 分类结果输出
    .result_valid(result_valid),
    .prob(prob),
    .predicted_class(predicted_class),
    
    // 误差输出
    .error(error_scalar),
    .error_valid(error_valid),
    
    // 调试
    .state(dbg_cls_state),
    .debug_logit(dbg_error_value)
);

//================================================================================
// 模块实例化：DFA Training Subsystem
//================================================================================
dfa_training_subsystem #(
    .NUM_LAYERS(5),
    .DIM_SMALL(DIM_SMALL),
    .DIM_LARGE(DIM),
    .NUM_TOKENS(NUM_TOKENS),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH),
    .NUM_CLASSES(1),
    .MAX_BATCH_SIZE(MAX_BATCH_SIZE),
    .MAX_EPOCHS(MAX_EPOCHS),
    .DEFAULT_LR(DEFAULT_LR),
    .DRAM_DATA_WIDTH(DRAM_DATA_WIDTH),
    .MAX_EXP_ARRAY_WIDTH(MAX_EXP_ARRAY_WIDTH)
) u_dfa_training_subsystem (
    .clk(clk),
    .rst_n(rst_n),
    
    // 训练控制
    .train_start(train_start),
    .cfg_batch_size(cfg_batch_size),
    .cfg_num_epochs(cfg_num_epochs),
    .cfg_learning_rate(cfg_learning_rate),
    .train_done(train_done),
    .train_active(train_active),
    
    // 前向传播接口
    .forward_start(forward_start),
    .forward_done(forward_done),
    .capture_enable(capture_enable),
    
    // 误差输入接口（来自分类子系统）
    .error_start(cls_start),
    .error_done(cls_done),
    .error_valid(error_valid),
    .error_scalar(error_scalar),
    
    // LOB读取接口 (Layer 0-4)
    .lob_rd_dfa0_en(lob_rd_dfa0_en),
    .lob_rd_dfa0_addr(lob_rd_dfa0_addr),
    .lob_dfa0_vector(lob_dfa0_vector),
    .lob_dfa0_valid(lob_dfa0_valid),
    
    .lob_rd_dfa1_en(lob_rd_dfa1_en),
    .lob_rd_dfa1_addr(lob_rd_dfa1_addr),
    .lob_dfa1_vector(lob_dfa1_vector),
    .lob_dfa1_valid(lob_dfa1_valid),
    
    .lob_rd_dfa2_en(lob_rd_dfa2_en),
    .lob_rd_dfa2_addr(lob_rd_dfa2_addr),
    .lob_dfa2_vector(lob_dfa2_vector),
    .lob_dfa2_valid(lob_dfa2_valid),
    
    .lob_rd_dfa3_en(lob_rd_dfa3_en),
    .lob_rd_dfa3_addr(lob_rd_dfa3_addr),
    .lob_dfa3_vector(lob_dfa3_vector),
    .lob_dfa3_valid(lob_dfa3_valid),
    
    .lob_rd_dfa4_en(lob_rd_dfa4_en),
    .lob_rd_dfa4_addr(lob_rd_dfa4_addr),
    .lob_dfa4_vector(lob_dfa4_vector),
    .lob_dfa4_valid(lob_dfa4_valid),
    
    // 权重存储接口
    .weight_rd_req(weight_rd_req),
    .weight_rd_layer_id(weight_rd_layer_id),
    .weight_rd_type(weight_rd_type),
    .weight_rd_col_id(weight_rd_col_id),
    .weight_rd_row_id(weight_rd_row_id),
    .weight_rd_valid(weight_rd_valid),
    .weight_rd_exp(weight_rd_exp),
    .weight_rd_mant(weight_rd_mant),
    
    .weight_wr_req(weight_wr_req),
    .weight_wr_layer_id(weight_wr_layer_id),
    .weight_wr_type(weight_wr_type),
    .weight_wr_burst_idx(weight_wr_burst_idx),
    .weight_wr_exp_array(weight_wr_exp_array),
    .weight_wr_data_burst(weight_wr_data_burst),
    .weight_wr_ready(weight_wr_ready),
    
    // Bank切换接口
    .request_bank_switch(request_bank_switch),
    .bank_switch_ready(bank_switch_ready),
    .confirm_switch(confirm_switch),
    .switch_complete(switch_complete),
    
    // 调试接口
    .dbg_controller_state(dbg_controller_state),
    .dbg_iteration_count(dbg_iteration_count),
    .dbg_batch_count(dbg_batch_count),
    .dbg_epoch_count(dbg_epoch_count),
    .dbg_timeout_error(dbg_timeout_error),
    .dbg_gcu0_state(dbg_gcu0_state),
    .dbg_gcu0_cycles(dbg_gcu0_cycles),
    .dbg_update_state(),
    .dbg_update_count()
);

endmodule