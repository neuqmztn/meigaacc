`timescale 1ns / 1ps

//==============================================================================
// DFA Master Controller v2
//
// 功能：协调一次完整 DFA 训练流程（多 batch × 多 epoch）：
//   IDLE → CHECK_B_INIT → (INIT_B) → FORWARD → WAIT_FORWARD
//     → CALC_ERROR → WAIT_ERROR → GRAD → WAIT_GRAD
//     → UPDATE → WAIT_UPDATE → CHECK_BATCH → CHECK_EPOCH
//     → (可选 BANK_SWITCH) → DONE
//
// 关键变化（相对旧版 v2.1）：
//   1) 梯度阶段只使用 grad_start[0] / grad_done[0] 这一条物理 GCU 链路；
//   2) 不再等待 5 路 GCU 全部 done，避免与 "单 GCU 串行 4 层" 冲突；
//   3) expand / layer4 的训练在数据通路里已经冻结，本控制器不再特殊处理；
//   4) 仍然保留 batch / epoch / timeout 计数逻辑，方便后续扩展。
//==============================================================================

module dfa_master_controller_v2 #(
    //==========================================================================
    // 训练参数
    //==========================================================================
    parameter MAX_BATCH_SIZE    = 32,
    parameter MAX_EPOCHS        = 100,

    //==========================================================================
    // 超时参数（单位：时钟周期）
    //==========================================================================
    parameter FORWARD_TIMEOUT   = 16'd2000,
    parameter ERROR_TIMEOUT     = 16'd500,
    parameter GRADIENT_TIMEOUT  = 32'd25000,
    parameter UPDATE_TIMEOUT    = 32'd15000,
    parameter BANK_SWITCH_DELAY = 16'd10
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,

    //==========================================================================
    // 训练控制接口
    //==========================================================================
    input  wire        train_start,      // 外部触发一次"完整训练过程"
    input  wire [7:0]  cfg_batch_size,   // 每个 epoch 内的 batch 数
    input  wire [7:0]  cfg_num_epochs,   // 总 epoch 数
    output reg         train_done,       // 所有 epoch 完成
    output reg         train_active,     // 训练期间为 1

    //==========================================================================
    // B 矩阵初始化接口
    //==========================================================================
    input  wire        b_matrices_exist, // B 是否已存在
    output reg         init_b_matrices,  // 启动 B 初始化
    input  wire        b_init_done,      // B 初始化完成

    //==========================================================================
    // 前向传播控制接口
    //==========================================================================
    output reg         forward_start,    // 启动 Backbone 前向
    input  wire        forward_done,     // 前向完成
    output reg         capture_enable,   // LOB 捕获开关

    //==========================================================================
    // 误差计算接口（分类子系统）
    //==========================================================================
    output reg         error_start,      // 启动误差计算
    input  wire        error_done,       // 误差计算完成
    input  wire        error_valid,      // 误差有效（例如 error_scalar 已就绪）

    //==========================================================================
    // 梯度计算控制接口（v2：只使用 bit[0]）
    //==========================================================================
    output reg  [4:0]  grad_start,       // 现在仅 grad_start[0] 有意义
    input  wire [4:0]  grad_done,        // 现在仅 grad_done[0] 会被等待

    //==========================================================================
    // 权重更新控制接口（对接 weight_update_engine）
    //==========================================================================
    output reg         update_start,
    input  wire        update_done,

    //==========================================================================
    // Bank 切换接口（例如 LOB ping-pong）
    //==========================================================================
    output reg         request_bank_switch,
    input  wire        bank_switch_ready,
    output reg         confirm_switch,
    input  wire        switch_complete,

    //==========================================================================
    // 状态监控和调试接口
    //==========================================================================
    output wire [3:0]  current_state,
    output reg  [31:0] iteration_counter,   // 总迭代次数（样本/批）
    output reg  [7:0]  batch_counter,       // 当前 epoch 内的 batch 计数
    output reg  [7:0]  epoch_counter,       // 当前 epoch 计数
    output reg  [31:0] total_samples,       // 累计样本数
    output reg         timeout_error,       // 任一阶段超时置 1（保持）
    output wire [11:0] timeout_counter      // 当前阶段的超时计数
);

    //==========================================================================
    // 状态编码（格雷码，与旧版本兼容，方便波形阅读）
    //==========================================================================
    localparam STATE_IDLE           = 4'b0000;
    localparam STATE_CHECK_B_INIT   = 4'b0001;
    localparam STATE_INIT_B_MATRIX  = 4'b0011;
    localparam STATE_FORWARD_TRIGGER= 4'b0010;
    localparam STATE_WAIT_FORWARD   = 4'b0110;
    localparam STATE_CALC_ERROR     = 4'b0101;
    localparam STATE_WAIT_ERROR     = 4'b0100;
    localparam STATE_GRAD_TRIGGER   = 4'b1100;
    localparam STATE_WAIT_GRAD      = 4'b1101;
    localparam STATE_UPDATE_TRIGGER = 4'b1111;
    localparam STATE_WAIT_UPDATE    = 4'b1110;
    localparam STATE_CHECK_BATCH    = 4'b1010;
    localparam STATE_CHECK_EPOCH    = 4'b1011;
    localparam STATE_BANK_SWITCH    = 4'b1001;
    localparam STATE_DONE           = 4'b1000;

    // 状态寄存器
    reg [3:0] state_reg, state_next;

    // 超时计数
    reg [11:0] timeout_cnt_reg, timeout_cnt_next;
    reg [11:0] timeout_threshold;

    // 内部计数器 next 值
    reg [31:0] iteration_next;
    reg [7:0]  batch_cnt_next;
    reg [7:0]  epoch_cnt_next;
    reg [31:0] total_samples_next;

    // Bank 切换延时计数
    reg [15:0] bank_delay_cnt_reg, bank_delay_cnt_next;

    //--------------------------------------------------------------------------
    // 状态 / 计数器寄存器
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg         <= STATE_IDLE;
            timeout_cnt_reg   <= 12'd0;
            iteration_counter <= 32'd0;
            batch_counter     <= 8'd0;
            epoch_counter     <= 8'd0;
            total_samples     <= 32'd0;
            bank_delay_cnt_reg<= 16'd0;
            timeout_error     <= 1'b0;
        end else begin
            state_reg         <= state_next;
            timeout_cnt_reg   <= timeout_cnt_next;
            iteration_counter <= iteration_next;
            batch_counter     <= batch_cnt_next;
            epoch_counter     <= epoch_cnt_next;
            total_samples     <= total_samples_next;
            bank_delay_cnt_reg<= bank_delay_cnt_next;
        end
    end

    //--------------------------------------------------------------------------
    // 主组合逻辑：状态转移 & 输出信号
    //--------------------------------------------------------------------------
    always @(*) begin
        // 默认保持
        state_next           = state_reg;
        timeout_cnt_next     = timeout_cnt_reg;
        iteration_next       = iteration_counter;
        batch_cnt_next       = batch_counter;
        epoch_cnt_next       = epoch_counter;
        total_samples_next   = total_samples;
        bank_delay_cnt_next  = bank_delay_cnt_reg;

        // 默认输出清零
        train_done           = 1'b0;
        train_active         = 1'b0;

        init_b_matrices      = 1'b0;
        forward_start        = 1'b0;
        capture_enable       = 1'b0;
        error_start          = 1'b0;

        grad_start           = 5'b00000;
        update_start         = 1'b0;

        request_bank_switch  = 1'b0;
        confirm_switch       = 1'b0;

        timeout_threshold    = 12'd0;

        case (state_reg)
            //==================================================================
            // IDLE：等待上层拉起 train_start
            //==================================================================
            STATE_IDLE: begin
                if (train_start) begin
                    // 清空计数（可视情况保留 epoch_counter）
                    iteration_next     = 32'd0;
                    batch_cnt_next     = 8'd0;
                    epoch_cnt_next     = 8'd0;
                    total_samples_next = 32'd0;
                    timeout_cnt_next   = 12'd0;
                    timeout_error      = 1'b0;
                    state_next         = STATE_CHECK_B_INIT;
                end
            end

            //==================================================================
            // CHECK_B_INIT：检查 B 是否已经初始化
            //==================================================================
            STATE_CHECK_B_INIT: begin
                train_active = 1'b1;
                if (b_matrices_exist) begin
                    state_next = STATE_FORWARD_TRIGGER;
                end else begin
                    state_next = STATE_INIT_B_MATRIX;
                end
            end

            //==================================================================
            // INIT_B_MATRIX：启动 B 初始化
            //==================================================================
            STATE_INIT_B_MATRIX: begin
                train_active    = 1'b1;
                init_b_matrices = 1'b1;
                if (b_init_done) begin
                    state_next     = STATE_FORWARD_TRIGGER;
                    timeout_cnt_next = 12'd0;
                end
            end

            //==================================================================
            // FORWARD_TRIGGER：触发前向传播
            //==================================================================
            STATE_FORWARD_TRIGGER: begin
                train_active   = 1'b1;
                forward_start  = 1'b1;
                capture_enable = 1'b1;  // 允许 LOB 捕获当前 batch 的输出
                timeout_cnt_next = 12'd0;
                state_next     = STATE_WAIT_FORWARD;
            end

            //==================================================================
            // WAIT_FORWARD：等待前向传播完成
            //==================================================================
            STATE_WAIT_FORWARD: begin
                train_active     = 1'b1;
                capture_enable   = 1'b1;
                timeout_threshold= FORWARD_TIMEOUT[11:0];

                if (forward_done) begin
                    timeout_cnt_next = 12'd0;
                    state_next       = STATE_CALC_ERROR;
                end else if (timeout_cnt_reg >= timeout_threshold) begin
                    // Forward 超时
                    timeout_error    = 1'b1;
                    state_next       = STATE_DONE;
                end else begin
                    timeout_cnt_next = timeout_cnt_reg + 12'd1;
                end
            end

            //==================================================================
            // CALC_ERROR：启动误差计算
            //==================================================================
            STATE_CALC_ERROR: begin
                train_active     = 1'b1;
                error_start      = 1'b1;
                timeout_cnt_next = 12'd0;
                state_next       = STATE_WAIT_ERROR;
            end

            //==================================================================
            // WAIT_ERROR：等待误差计算完成
            //==================================================================
            STATE_WAIT_ERROR: begin
                train_active      = 1'b1;
                timeout_threshold = ERROR_TIMEOUT[11:0];

                if (error_done && error_valid) begin
                    timeout_cnt_next = 12'd0;
                    state_next       = STATE_GRAD_TRIGGER;
                end else if (timeout_cnt_reg >= timeout_threshold) begin
                    timeout_error    = 1'b1;
                    state_next       = STATE_DONE;
                end else begin
                    timeout_cnt_next = timeout_cnt_reg + 12'd1;
                end
            end

            //==================================================================
            // GRAD_TRIGGER：启动梯度计算 (v2：单 GCU → grad_start[0])
            //==================================================================
            STATE_GRAD_TRIGGER: begin
                train_active     = 1'b1;
                grad_start       = 5'b00001; // 仅启动 bit0
                timeout_cnt_next = 12'd0;
                state_next       = STATE_WAIT_GRAD;
            end

            //==================================================================
            // WAIT_GRAD：等待梯度计算完成 (仅等待 grad_done[0])
            //==================================================================
            STATE_WAIT_GRAD: begin
                train_active      = 1'b1;
                timeout_threshold = GRADIENT_TIMEOUT[11:0];

                if (grad_done[0]) begin
                    timeout_cnt_next = 12'd0;
                    state_next       = STATE_UPDATE_TRIGGER;
                end else if (timeout_cnt_reg >= timeout_threshold) begin
                    timeout_error    = 1'b1;
                    state_next       = STATE_DONE;
                end else begin
                    timeout_cnt_next = timeout_cnt_reg + 12'd1;
                end
            end

            //==================================================================
            // UPDATE_TRIGGER：启动权重更新
            //==================================================================
            STATE_UPDATE_TRIGGER: begin
                train_active     = 1'b1;
                update_start     = 1'b1;
                timeout_cnt_next = 12'd0;
                state_next       = STATE_WAIT_UPDATE;
            end

            //==================================================================
            // WAIT_UPDATE：等待权重更新完成
            //==================================================================
            STATE_WAIT_UPDATE: begin
                train_active      = 1'b1;
                timeout_threshold = UPDATE_TIMEOUT[11:0];

                if (update_done) begin
                    // 完成一次"前向 + 误差 + 梯度 + 更新"
                    timeout_cnt_next   = 12'd0;
                    iteration_next     = iteration_counter + 32'd1;
                    batch_cnt_next     = batch_counter + 8'd1;
                    total_samples_next = total_samples + 32'd1;
                    state_next         = STATE_CHECK_BATCH;
                end else if (timeout_cnt_reg >= timeout_threshold) begin
                    timeout_error    = 1'b1;
                    state_next       = STATE_DONE;
                end else begin
                    timeout_cnt_next = timeout_cnt_reg + 12'd1;
                end
            end

            //==================================================================
            // CHECK_BATCH：本 epoch 是否还有 batch 未完成
            //==================================================================
            STATE_CHECK_BATCH: begin
                train_active = 1'b1;

                if (batch_counter >= cfg_batch_size - 1) begin
                    // 当前 epoch 的 batch 结束，准备检查 epoch
                    batch_cnt_next = 8'd0;
                    state_next     = STATE_CHECK_EPOCH;
                end else begin
                    // 继续下一 batch
                    state_next     = STATE_FORWARD_TRIGGER;
                end
            end

            //==================================================================
            // CHECK_EPOCH：检查是否还有 epoch 未完成
            //==================================================================
            STATE_CHECK_EPOCH: begin
                train_active = 1'b1;

                if (epoch_counter >= cfg_num_epochs - 1) begin
                    // 所有 epoch 完成
                    state_next = STATE_DONE;
                end else begin
                    // 进入下一个 epoch
                    epoch_cnt_next = epoch_counter + 8'd1;
                    // 如需严格的 Bank 切换，可在此进入 STATE_BANK_SWITCH
                    state_next     = STATE_FORWARD_TRIGGER;
                end
            end

            //==================================================================
            // BANK_SWITCH：如果你有 LOB / Result buffer ping-pong，可在这里切换
            //==================================================================
            STATE_BANK_SWITCH: begin
                train_active        = 1'b1;
                request_bank_switch = 1'b1;

                if (bank_switch_ready) begin
                    confirm_switch = 1'b1;
                    if (switch_complete) begin
                        bank_delay_cnt_next = 16'd0;
                        state_next          = STATE_FORWARD_TRIGGER;
                    end
                end
            end

            //==================================================================
            // DONE：整个训练过程完成，等待 train_start 释放后回到 IDLE
            //==================================================================
            STATE_DONE: begin
                train_done   = 1'b1;
                train_active = 1'b0;

                if (!train_start) begin
                    state_next = STATE_IDLE;
                end
            end

            default: begin
                state_next = STATE_IDLE;
            end
        endcase
    end

    // 输出状态 & timeout 计数
    assign current_state   = state_reg;
    assign timeout_counter = timeout_cnt_reg;

endmodule
