`timescale 1ns / 1ps


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
    // 误差计算接口（连接classification_subsystem）
    //==========================================================================
    output wire        error_start,
    input  wire        error_done,
    input  wire        error_valid,
    input  wire [DATA_WIDTH-1:0] error_scalar,
    
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
    // B矩阵初始化接口（可选，连接到DRAM或配置模块）
    //==========================================================================
    input  wire        b_matrices_exist,     // 1: B矩阵已存在，无需初始化
    
    //==========================================================================
    // 权重存储接口（连接到sidenet_weight_storage或统一权重控制模块）
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
    output wire [DRAM_DATA_WIDTH-1:0]     weight_wr_data_burst,
    input  wire        weight_wr_ready,
    
    //==========================================================================
    // Bank切换接口（可连接到Backbone/SideNet控制）
    //==========================================================================
    output wire        request_bank_switch,
    input  wire        bank_switch_ready,
    
    //==========================================================================
    // 调试输出
    //==========================================================================
    output wire [3:0]  dbg_ctrl_state,
    output wire [3:0]  dbg_gcu0_state,
    output wire [3:0]  dbg_gcu1_state,
    output wire [3:0]  dbg_gcu2_state,
    output wire [3:0]  dbg_gcu3_state,
    output wire [3:0]  dbg_gcu4_state,
    output wire [3:0]  dbg_update_state,
    output wire [31:0] dbg_update_count,
    output wire [31:0] dbg_gcu0_cycles,
    output wire [31:0] dbg_gcu1_cycles,
    output wire [31:0] dbg_gcu2_cycles,
    output wire [31:0] dbg_gcu3_cycles,
    output wire [31:0] dbg_gcu4_cycles
);

//================================================================================
// 内部信号：控制器 → B矩阵
//================================================================================
wire        init_b_start;
wire        init_b_done;
wire        b_matrices_exist_internal;
assign      b_matrices_exist_internal = b_matrices_exist;

//================================================================================
// 内部信号：控制器 → GCU
//================================================================================
wire [4:0]  gcu_start;
wire [4:0]  gcu_done;
wire [4:0]  gcu_busy;

//================================================================================
// 内部信号：控制器 → 权重更新（外层控制器视角）
//================================================================================
wire        ctrl_update_start;
reg         ctrl_update_done;
wire        update_busy;

//================================================================================
// 内部信号：多层权重更新调度
// - ctrl_update_start / ctrl_update_done : 与 DFA Master Controller 握手
// - wue_update_start / wue_update_done   : 与 Weight Update Engine 握手
// - update_layer_id                      : 当前正在更新的 SideNet 层 ID
//================================================================================
reg  [2:0] update_layer_id;
reg        update_seq_active;
reg        wue_update_start_reg;
wire       wue_update_start;
wire       wue_update_done;

assign wue_update_start = wue_update_start_reg;

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
wire [DIM_ADDR_WIDTH-1:0]   gcu0_grad_dim,   gcu1_grad_dim,   gcu2_grad_dim,   gcu3_grad_dim,   gcu4_grad_dim;
wire [DATA_WIDTH-1:0]       gcu0_grad_data,  gcu1_grad_data,  gcu2_grad_data,  gcu3_grad_data,  gcu4_grad_data;

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
// DFA Master Controller 实例
//================================================================================
dfa_master_controller #(
    .MAX_BATCH_SIZE   (MAX_BATCH_SIZE),
    .MAX_EPOCHS       (MAX_EPOCHS),
    .FORWARD_TIMEOUT  (16'd2000),
    .ERROR_TIMEOUT    (16'd500),
    .GRADIENT_TIMEOUT (32'd25000),
    .UPDATE_TIMEOUT   (32'd15000),
    .BANK_SWITCH_DELAY(16'd10),
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
    .b_matrices_exist(b_matrices_exist_internal),
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
    
    // 权重更新控制（外层只看到一次update，对应内部5层顺序更新）
    .update_start(ctrl_update_start),
    .update_done(ctrl_update_done),
    
    // Bank切换
    .request_bank_switch(request_bank_switch),
    .bank_switch_ready(bank_switch_ready),
    
    // 调试
    .dbg_state(dbg_ctrl_state)
);

//================================================================================
// DFA Matrix Bank - B矩阵存储
//================================================================================
dfa_matrix_bank #(
    .NUM_LAYERS(NUM_LAYERS),
    .DIM_SMALL(DIM_SMALL),
    .DIM_LARGE(DIM_LARGE),
    .DATA_WIDTH(DATA_WIDTH)
) u_dfa_matrix_bank (
    .clk(clk),
    .rst_n(rst_n),
    
    .init_start(init_b_start),
    .init_done(init_b_done),
    
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
    .gcu4_valid(gcu4_b_valid)
);

//================================================================================
// 模块实例化：5个 Gradient Compute Units
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
    .state(dbg_gcu1_state),
    .cycle_count(dbg_gcu1_cycles),
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
    .state(dbg_gcu2_state),
    .cycle_count(dbg_gcu2_cycles),
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
    .state(dbg_gcu3_state),
    .cycle_count(dbg_gcu3_cycles),
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
    .state(dbg_gcu4_state),
    .cycle_count(dbg_gcu4_cycles),
    .token_count()
);

//================================================================================
// Gradient Buffer 实例
//================================================================================
gradient_buffer #(
    .NUM_LAYERS(NUM_LAYERS),
    .DIM_SMALL(DIM_SMALL),
    .DIM_LARGE(DIM_LARGE),
    .NUM_TOKENS(NUM_TOKENS),
    .DATA_WIDTH(DATA_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH)
) u_gradient_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    // 写入接口（来自5个GCU）
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
    
    // 复杂读接口（暂未使用）
    .rd_en(1'b0),
    .rd_layer_id(3'd0),
    .rd_addr(10'd0),
    .rd_data(),
    .rd_valid(),
    
    // 简化读接口（提供给Weight Update Engine）
    .rd_simple_en(grad_rd_req),
    .rd_simple_layer(grad_rd_layer_id),
    .rd_simple_token(grad_rd_token_id),
    .rd_simple_dim(grad_rd_dim_id),
    .rd_simple_data(grad_rd_data),
    .rd_simple_valid(grad_rd_valid)
);

//================================================================================
// 多层权重更新调度器
//--------------------------------------------------------------------------------
// 行为：
// 1) 当 ctrl_update_start 拉高且当前没有更新任务时：
//      - update_layer_id <= 0
//      - update_seq_active <= 1
//      - 产生一个周期的 wue_update_start 脉冲，启动第0层更新
// 2) 每当当前层的 wue_update_done = 1：
//      - 如果还没到最后一层(NUM_LAYERS-1)，层号+1，并再次产生 wue_update_start 脉冲
//      - 如果已经是最后一层，则：
//          * update_seq_active <= 0
//          * 给控制器一个周期的 ctrl_update_done 脉冲
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        update_layer_id      <= 3'd0;
        update_seq_active    <= 1'b0;
        wue_update_start_reg <= 1'b0;
        ctrl_update_done     <= 1'b0;
    end else begin
        // 默认拉低单周期脉冲
        wue_update_start_reg <= 1'b0;
        ctrl_update_done     <= 1'b0;

        if (!update_seq_active) begin
            // 当前没有在做多层更新，等待控制器发起一次update_start
            if (ctrl_update_start) begin
                update_seq_active    <= 1'b1;
                update_layer_id      <= 3'd0;
                wue_update_start_reg <= 1'b1;  // 启动第0层更新
            end
        end else begin
            // 正在进行多层权重更新序列
            if (wue_update_done) begin
                if (update_layer_id == NUM_LAYERS-1) begin
                    // 所有层更新完成
                    update_seq_active <= 1'b0;
                    ctrl_update_done  <= 1'b1;  // 通知上层：一次完整的UPDATE阶段完成
                end else begin
                    // 切换到下一层，重新启动WUE
                    update_layer_id      <= update_layer_id + 3'd1;
                    wue_update_start_reg <= 1'b1;
                end
            end
        end
    end
end

//================================================================================
// Weight Update Engine 实例
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
    .update_start(wue_update_start),
    .update_done(wue_update_done),
    .update_busy(update_busy),
    .cfg_layer_id(update_layer_id),  // 由子系统内部多层调度器控制
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
