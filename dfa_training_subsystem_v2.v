`timescale 1ns / 1ps

module dfa_training_subsystem_v2 #(
    parameter NUM_TOKENS        = 640,
    parameter DIM_SMALL         = 8,
    parameter DATA_WIDTH        = 16,
    parameter EXP_WIDTH         = 8,
    parameter TOKEN_ADDR_WIDTH  = 10,
    parameter DIM_ADDR_WIDTH    = 5,
    parameter DRAM_DATA_WIDTH   = 256
)(
    input  wire clk,
    input  wire rst_n,

    // =========================================================================
    // 控制接口：开始一次完整 DFA (layer0~3)
    // =========================================================================
    input  wire train_start,         // 外部触发一次训练
    output reg  train_done,          // 整个训练 + 更新完成
    output wire train_busy,          // 只要 DFA 子系统正在运行就为 1

    // =========================================================================
    // 误差信号（来自分类器的 BCE delta）
    // =========================================================================
    input  wire [DATA_WIDTH-1:0] error_scalar,

    // =========================================================================
    // LOB (activation buffer) 接口
    // =========================================================================
    output wire                      lob_rd_en,
    output wire [TOKEN_ADDR_WIDTH-1:0] lob_rd_token_id,
    input  wire [DIM_SMALL*DATA_WIDTH-1:0] lob_vector,
    input  wire                      lob_valid,

    // =========================================================================
    // 权重存储端口（直接转发 WUE 的请求到 sidenet_weight_storage）
    // =========================================================================
    output wire                      weight_rd_req,
    output wire [2:0]                weight_rd_layer_id,
    output wire [3:0]                weight_rd_type,
    output wire [4:0]                weight_rd_col_id,
    output wire [4:0]                weight_rd_row_id,
    input  wire                      weight_rd_valid,
    input  wire [EXP_WIDTH-1:0]      weight_rd_exp,
    input  wire [DATA_WIDTH-1:0]     weight_rd_mant,

    output wire                      weight_wr_req,
    output wire [2:0]                weight_wr_layer_id,
    output wire [3:0]                weight_wr_type,
    output wire [5:0]                weight_wr_burst_idx,
    output wire [EXP_WIDTH*32-1:0]   weight_wr_exp_array,
    output wire [DRAM_DATA_WIDTH-1:0] weight_wr_data_burst,
    input  wire                      weight_wr_ready
);


// ============================================================================
// 内部信号
// ============================================================================

// -------- B matrix bank -----------
reg  b_init_start;
wire b_init_done;
wire b_exist;

wire mb_rd_en;
wire [1:0] mb_layer_id;
wire [DIM_ADDR_WIDTH-1:0] mb_row_addr;
wire [DATA_WIDTH-1:0] mb_rd_data;
wire mb_rd_valid;

// -------- GCU 训练 core ----------
reg  train_start_all;
wire train_done_all;
wire train_busy_all;

wire grad_wr_en;
wire [2:0] grad_wr_layer_id;
wire [TOKEN_ADDR_WIDTH-1:0] grad_wr_token_id;
wire [DIM_ADDR_WIDTH-1:0]   grad_wr_dim_id;
wire [DATA_WIDTH-1:0]       grad_wr_data;


// -------- Gradient buffer → WUE --------
wire                      grad_rd_req;
wire [2:0]                grad_rd_layer_id;
wire [TOKEN_ADDR_WIDTH-1:0] grad_rd_token_id;
wire [DIM_ADDR_WIDTH-1:0] grad_rd_dim_id;
wire                      grad_rd_valid;
wire [DATA_WIDTH-1:0]     grad_rd_data;


// -------- WUE ----------
reg  wue_update_start;
wire wue_update_done;
wire wue_update_busy;

reg  [2:0] wue_cfg_layer_id;
reg  [3:0] wue_cfg_weight_type;
reg  [15:0] wue_cfg_learning_rate;


// ============================================================================
// 1. B 矩阵初始化 FSM（简单处理：训练开始前自动初始化一次）
// ============================================================================
localparam BS_IDLE = 0;
localparam BS_INIT = 1;
localparam BS_WAIT_INIT = 2;
localparam BS_DONE = 3;

reg [1:0] bstate, bstate_next;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        bstate <= BS_IDLE;
    else
        bstate <= bstate_next;
end

always @(*) begin
    bstate_next = bstate;
    b_init_start = 1'b0;
    case (bstate)
        BS_IDLE: begin
            if (train_start) begin
                b_init_start = 1'b1;
                bstate_next  = BS_INIT;
            end
        end

        BS_INIT: begin
            // 拉起后等待 b_init_done
            if (b_init_done)
                bstate_next = BS_DONE;
            else
                bstate_next = BS_INIT;
        end

        BS_DONE: begin
            // 初始化只做一次，后续保持 exist = 1
        end
    endcase
end


// ============================================================================
// 2. DFA 训练子系统（仅 layer0~3）
// ============================================================================
dfa_gcu_training_core_v2 #(
    .NUM_TOKENS(NUM_TOKENS),
    .DIM_SMALL(DIM_SMALL),
    .DATA_WIDTH(DATA_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH)
) u_train_core (
    .clk(clk),
    .rst_n(rst_n),

    .start_all(train_start_all),
    .error_scalar(error_scalar),
    .done_all(train_done_all),
    .busy_all(train_busy_all),

    // B matrix
    .mb_rd_en(mb_rd_en),
    .mb_layer_id(mb_layer_id),
    .mb_row_addr(mb_row_addr),
    .mb_rd_data(mb_rd_data),
    .mb_rd_valid(mb_rd_valid),

    // LOB
    .act_rd_en(lob_rd_en),
    .act_token_addr(lob_rd_token_id),
    .act_vector(lob_vector),
    .act_valid(lob_valid),

    // Gradient 写口
    .gradbuf_wr_en(grad_wr_en),
    .gradbuf_wr_layer_id(grad_wr_layer_id),
    .gradbuf_wr_token_id(grad_wr_token_id),
    .gradbuf_wr_dim_id(grad_wr_dim_id),
    .gradbuf_wr_data(grad_wr_data)
);


// ============================================================================
// 3. B-Matrix Bank (单口)
// ============================================================================
dfa_matrix_bank_v2 #(
    .NUM_LAYERS(4),
    .LAYER_DIM(DIM_SMALL),
    .DATA_WIDTH(DATA_WIDTH)
) u_bmatrix (
    .clk(clk),
    .rst_n(rst_n),

    .init_start(b_init_start),
    .init_done(b_init_done),
    .matrices_exist(b_exist),

    .init_state(),
    .init_counter(),

    .rd_en(mb_rd_en),
    .layer_id(mb_layer_id),
    .row_addr(mb_row_addr),
    .rd_data(mb_rd_data),
    .rd_valid(mb_rd_valid)
);


// ============================================================================
// 4. Gradient Buffer v2 (BRAM)
// ============================================================================
gradient_buffer_v2 #(
    .NUM_TOKENS(NUM_TOKENS),
    .DIM_SMALL(DIM_SMALL),
    .DIM_LARGE(32),
    .DATA_WIDTH(DATA_WIDTH),
    .TOKEN_ADDR_WIDTH(TOKEN_ADDR_WIDTH),
    .DIM_ADDR_WIDTH(DIM_ADDR_WIDTH)
) u_gbuf (
    .clk(clk),
    .rst_n(rst_n),

    .wr_en(grad_wr_en),
    .wr_layer_id(grad_wr_layer_id),
    .wr_token_id(grad_wr_token_id),
    .wr_dim_id(grad_wr_dim_id),
    .wr_data(grad_wr_data),

    .grad_rd_req(grad_rd_req),
    .grad_rd_layer_id(grad_rd_layer_id),
    .grad_rd_token_id(grad_rd_token_id),
    .grad_rd_dim_id(grad_rd_dim_id),
    .grad_rd_valid(grad_rd_valid),
    .grad_rd_data(grad_rd_data)
);


// ============================================================================
// 5. Weight Update Engine
// ============================================================================
weight_update_engine #(
    .NUM_LAYERS(5),  // 包含 layer4，但 layer4 不会训练
    .BACKBONE_DIM(32),
    .SIDENET_DIM(DIM_SMALL),
    .D_FF(32),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .Q412_WIDTH(DATA_WIDTH),
    .DRAM_DATA_WIDTH(DRAM_DATA_WIDTH),
    .NUM_TOKENS(NUM_TOKENS),
    .MAX_ROWS(32),
    .MAX_COLS(32),
    .GRAD_ACCUM_WIDTH(32),
    .DEFAULT_LR_Q412(16'h0001),
    .TIMEOUT_THRESHOLD(12'd4095)
) u_wue (
    .clk(clk),
    .rst_n(rst_n),

    .update_start(wue_update_start),
    .update_done(wue_update_done),
    .update_busy(wue_update_busy),

    .cfg_layer_id(wue_cfg_layer_id),
    .cfg_weight_type(wue_cfg_weight_type),
    .cfg_learning_rate(wue_cfg_learning_rate),

    // grad read
    .grad_rd_req(grad_rd_req),
    .grad_rd_layer_id(grad_rd_layer_id),
    .grad_rd_token_id(grad_rd_token_id),
    .grad_rd_dim_id(grad_rd_dim_id),
    .grad_rd_valid(grad_rd_valid),
    .grad_rd_data(grad_rd_data),

    // weight read/write
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

    .dbg_state(),
    .dbg_update_count(),
    .dbg_col_count(),
    .dbg_row_count(),
    .dbg_timeout_count(),
    .dbg_overflow_count()
);


// ============================================================================
// 6. 顶层控制 FSM：控制 train_core 和 WUE 的先后顺序
// ============================================================================
localparam TS_IDLE   = 0;
localparam TS_TRAIN  = 1;
localparam TS_WUE    = 2;
localparam TS_DONE   = 3;

reg [1:0] tstate, tstate_next;

assign train_busy = (tstate != TS_IDLE && tstate != TS_DONE);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        tstate <= TS_IDLE;
    else
        tstate <= tstate_next;
end

always @(*) begin
    tstate_next = tstate;
    train_done  = 1'b0;

    train_start_all  = 1'b0;
    wue_update_start = 1'b0;
    wue_cfg_layer_id = 3'd0;
    wue_cfg_weight_type = 4'd0;
    wue_cfg_learning_rate = 16'h0001;

    case (tstate)
        // ----------------------------------------------------------
        TS_IDLE: begin
            if (train_start)
                tstate_next = TS_TRAIN;
        end

        // ----------------------------------------------------------
        TS_TRAIN: begin
            train_start_all = 1'b1;
            if (train_done_all)
                tstate_next = TS_WUE;
        end

        // ----------------------------------------------------------
        TS_WUE: begin
            wue_update_start = 1'b1;
            wue_cfg_layer_id = 3'd0;  // 只训练 0~3 层
            if (wue_update_done)
                tstate_next = TS_DONE;
        end

        // ----------------------------------------------------------
        TS_DONE: begin
            train_done = 1'b1;
            if (!train_start)
                tstate_next = TS_IDLE;
        end
    endcase
end

endmodule
