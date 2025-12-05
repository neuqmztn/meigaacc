`timescale 1ns / 1ps

//================================================================================
// dfa_gcu_training_core_v2
//
// 阶段 3 训练子系统（只包含 GCU + 梯度写口）：
//   * 用一个 gcu_core_v2 顺序计算 layer0~3 的梯度；
//   * 通过单一写口，把梯度写入 gradient_buffer_v2：
//         wr_en + wr_layer_id + wr_token_id + wr_dim_id + wr_data
//   * dfa_matrix_bank_v2 作为外部模块，通过单口 rd_* 接口提供 B 矩阵；
//   * LOB 作为外部模块，通过 act_* 接口提供激活值。
//================================================================================
module dfa_gcu_training_core_v2 #(
    parameter NUM_TOKENS        = 640,
    parameter DIM_SMALL         = 8,
    parameter DATA_WIDTH        = 16,
    parameter TOKEN_ADDR_WIDTH  = 10,
    parameter DIM_ADDR_WIDTH    = 5
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // 顶层控制
    input  wire                         start_all,     // 启动 layer0~3 的梯度计算
    input  wire [DATA_WIDTH-1:0]        error_scalar,  // 全局 BCE 误差标量
    output reg                          done_all,      // 4 个 layer 都完成
    output wire                         busy_all,      // 只要有一层在跑就为 1

    //==================== B 矩阵 / dfa_matrix_bank_v2 接口 =======================
    output wire                         mb_rd_en,
    output wire [1:0]                   mb_layer_id,
    output wire [DIM_ADDR_WIDTH-1:0]    mb_row_addr,
    output wire [3:0]                   mb_col_addr,
    input  wire [DATA_WIDTH-1:0]        mb_rd_data,
    input  wire                         mb_rd_valid,

    //======================= 激活值 / LOB 接口（单口） ==========================
    output wire                         act_rd_en,
    output wire [TOKEN_ADDR_WIDTH-1:0]  act_token_addr,
    input  wire [DIM_SMALL*DATA_WIDTH-1:0] act_vector,
    input  wire                         act_valid,

    //=================== 梯度写入 gradient_buffer_v2 的统一写口 ==================
    output wire                         gradbuf_wr_en,
    output wire [2:0]                   gradbuf_wr_layer_id,  // 0..3
    output wire [TOKEN_ADDR_WIDTH-1:0]  gradbuf_wr_token_id,
    output wire [DIM_ADDR_WIDTH-1:0]    gradbuf_wr_dim_id,
    output wire [DATA_WIDTH-1:0]        gradbuf_wr_data

);

    // 当前正在训练的 layer_id：0~3
    reg  [1:0] curr_layer_id_reg, curr_layer_id_next;

    // FSM：按层顺序调度 GCU
    localparam S_IDLE     = 2'd0;
    localparam S_RUN      = 2'd1;
    localparam S_WAIT     = 2'd2;
    localparam S_DONE_ALL = 2'd3;

    reg [1:0] state_reg, state_next;

    // GCU <-> 本模块内部信号
    reg        gcu_start_reg;
    wire       gcu_done;
    wire       gcu_busy;

    wire       gcu_mb_rd_en;
    wire [1:0] gcu_mb_layer_id;
    wire [DIM_ADDR_WIDTH-1:0] gcu_mb_row_addr;
    wire [3:0] gcu_mb_col_addr;

    wire       gcu_act_rd_en;
    wire [TOKEN_ADDR_WIDTH-1:0] gcu_act_token_addr;

    wire       gcu_grad_valid;
    wire [TOKEN_ADDR_WIDTH-1:0] gcu_grad_token_addr;
    wire [DIM_ADDR_WIDTH-1:0]   gcu_grad_dim_addr;
    wire [DATA_WIDTH-1:0]       gcu_grad_data;

    // 对外 busy_all = 只要没有回到 IDLE/DONE_ALL 就说明还在训练
    assign busy_all = (state_reg != S_IDLE) && (state_reg != S_DONE_ALL);

    //============================= 实例化单个 GCU ===============================
    gcu_core_v2 #(
        .NUM_TOKENS       (NUM_TOKENS),
        .DIM_SMALL        (DIM_SMALL),
        .DATA_WIDTH       (DATA_WIDTH),
        .TOKEN_ADDR_WIDTH (TOKEN_ADDR_WIDTH),
        .DIM_ADDR_WIDTH   (DIM_ADDR_WIDTH)
    ) u_gcu_core (
        .clk            (clk),
        .rst_n          (rst_n),

        .start          (gcu_start_reg),
        .layer_id       (curr_layer_id_reg),
        .error_scalar   (error_scalar),

        .done           (gcu_done),
        .busy           (gcu_busy),

        .mb_rd_en       (gcu_mb_rd_en),
        .mb_layer_id    (gcu_mb_layer_id),
        .mb_row_addr    (gcu_mb_row_addr),
        .mb_col_addr    (gcu_mb_col_addr),
        .mb_rd_data     (mb_rd_data),
        .mb_rd_valid    (mb_rd_valid),

        .act_rd_en      (gcu_act_rd_en),
        .act_token_addr (gcu_act_token_addr),
        .act_vector     (act_vector),
        .act_valid      (act_valid),

        .grad_valid     (gcu_grad_valid),
        .grad_token_addr(gcu_grad_token_addr),
        .grad_dim_addr  (gcu_grad_dim_addr),
        .grad_data      (gcu_grad_data)
    );

    // B bank 接口：直接转发 GCU 信号
    assign mb_rd_en    = gcu_mb_rd_en;
    assign mb_layer_id = gcu_mb_layer_id;
    assign mb_row_addr = gcu_mb_row_addr;
    assign mb_col_addr = gcu_mb_col_addr;

    // LOB 接口：直接转发 GCU 信号
    assign act_rd_en      = gcu_act_rd_en;
    assign act_token_addr = gcu_act_token_addr;

    //=================== 统一写口：写入 gradient_buffer_v2 ======================
    // 注意：gradbuf_wr_layer_id 是 3bit，这里 {1'b0, curr_layer_id_reg} → 0..3
    assign gradbuf_wr_en        = gcu_grad_valid;
    assign gradbuf_wr_layer_id  = {1'b0, curr_layer_id_reg};
    assign gradbuf_wr_token_id  = gcu_grad_token_addr;
    assign gradbuf_wr_dim_id    = gcu_grad_dim_addr;
    assign gradbuf_wr_data      = gcu_grad_data;

    //========================== 状态寄存器 & curr_layer =========================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg         <= S_IDLE;
            curr_layer_id_reg <= 2'd0;
        end else begin
            state_reg         <= state_next;
            curr_layer_id_reg <= curr_layer_id_next;
        end
    end

    //=========================== 顶层调度 FSM 组合逻辑 =========================
    always @(*) begin
        state_next         = state_reg;
        curr_layer_id_next = curr_layer_id_reg;
        gcu_start_reg      = 1'b0;
        done_all           = 1'b0;

        case (state_reg)
            //--------------------------------------------------------------
            // S_IDLE：等待一次完整训练启动
            //--------------------------------------------------------------
            S_IDLE: begin
                if (start_all) begin
                    curr_layer_id_next = 2'd0;
                    gcu_start_reg      = 1'b1;  // 拉高 1 个周期
                    state_next         = S_RUN;
                end
            end

            //--------------------------------------------------------------
            // S_RUN：当前 layer 正在运行
            //--------------------------------------------------------------
            S_RUN: begin
                if (gcu_done) begin
                    state_next = S_WAIT;
                end
            end

            //--------------------------------------------------------------
            // S_WAIT：当前 layer 已完成，准备切到下一层或结束
            //--------------------------------------------------------------
            S_WAIT: begin
                if (curr_layer_id_reg == 2'd3) begin
                    // layer0~3 全部完成
                    state_next = S_DONE_ALL;
                end else begin
                    // 切换到下一层
                    curr_layer_id_next = curr_layer_id_reg + 2'd1;
                    gcu_start_reg      = 1'b1;  // 启动下一层
                    state_next         = S_RUN;
                end
            end

            //--------------------------------------------------------------
            // S_DONE_ALL：所有层完成，拉 done_all，等待上层复位 start_all
            //--------------------------------------------------------------
            S_DONE_ALL: begin
                done_all = 1'b1;
                if (!start_all) begin
                    state_next = S_IDLE;
                end
            end

            default: begin
                state_next = S_IDLE;
            end
        endcase
    end

endmodule
