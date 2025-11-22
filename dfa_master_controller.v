`timescale 1ns / 1ps

//================================================================================
// DFA Master Controller - v2.1 (适配内部误差计算)
//
//
// 功能说明：
// DFA训练流程的主状态机，协调前向传播、误差计算、梯度计算、权重更新
//
// 状态机流程：
// IDLE → CHECK_B_INIT → (INIT_B) → FORWARD → WAIT_FORWARD 
//   → CAPTURE_VALID → CALC_ERROR → WAIT_ERROR → GRAD → WAIT_GRAD 
//   → UPDATE → WAIT_UPDATE → CHECK_BATCH → CHECK_EPOCH 
//   → BANK_SWITCH → DONE
//
// 作者：MEIGA Team
// 日期：2025-11-19
// 版本：v2.1
//================================================================================

module dfa_master_controller #(
    //==========================================================================
    // 训练参数
    //==========================================================================
    parameter MAX_BATCH_SIZE    = 32,
    parameter MAX_EPOCHS        = 100,
    
    //==========================================================================
    // 超时参数
    //==========================================================================
    parameter FORWARD_TIMEOUT   = 2000,
    parameter ERROR_TIMEOUT     = 500,
    parameter GRADIENT_TIMEOUT  = 25000,
    parameter UPDATE_TIMEOUT    = 15000,
    parameter BANK_SWITCH_DELAY = 10,
    
    //==========================================================================
    // GCU启动模式
    //==========================================================================
    parameter GCU_PARALLEL_START = 1
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
    input  wire [7:0]  cfg_batch_size,
    input  wire [7:0]  cfg_num_epochs,
    output reg         train_done,
    output reg         train_active,
    
    //==========================================================================
    // B矩阵初始化接口
    //==========================================================================
    input  wire        b_matrices_exist,
    output reg         init_b_matrices,
    input  wire        b_init_done,
    
    //==========================================================================
    // 前向传播控制接口
    //==========================================================================
    output reg         forward_start,
    input  wire        forward_done,
    output reg         capture_enable,
    
    //==========================================================================
    // 误差计算接口（内部计算）
    //==========================================================================
    output reg         error_start,          // 启动误差计算
    input  wire        error_done,           // 误差计算完成
    input  wire        error_valid,          // 误差有效
    
    //==========================================================================
    // 梯度计算控制接口
    //==========================================================================
    output reg  [4:0]  grad_start,
    input  wire [4:0]  grad_done,
    
    //==========================================================================
    // 权重更新控制接口
    //==========================================================================
    output reg         update_start,
    input  wire        update_done,
    
    //==========================================================================
    // Bank切换接口
    //==========================================================================
    output reg         request_bank_switch,
    input  wire        bank_switch_ready,
    output reg         confirm_switch,
    input  wire        switch_complete,
    
    //==========================================================================
    // 状态监控和调试接口
    //==========================================================================
    output wire [3:0]  current_state,
    output reg  [31:0] iteration_counter,
    output reg  [7:0]  batch_counter,
    output reg  [7:0]  epoch_counter,
    output reg  [31:0] total_samples,
    output reg         timeout_error,
    output wire [11:0] timeout_counter
);

//================================================================================
// 状态编码（格雷码）
//================================================================================
localparam STATE_IDLE           = 4'b0000;
localparam STATE_CHECK_B_INIT   = 4'b0001;
localparam STATE_INIT_B_MATRIX  = 4'b0011;
localparam STATE_FORWARD_TRIGGER= 4'b0010;
localparam STATE_WAIT_FORWARD   = 4'b0110;
localparam STATE_CAPTURE_VALID  = 4'b0111;
localparam STATE_CALC_ERROR     = 4'b0101;  // 启动误差计算
localparam STATE_WAIT_ERROR     = 4'b0100;  // 等待误差计算完成
localparam STATE_GRAD_TRIGGER   = 4'b1100;
localparam STATE_WAIT_GRAD      = 4'b1101;
localparam STATE_UPDATE_TRIGGER = 4'b1111;
localparam STATE_WAIT_UPDATE    = 4'b1110;
localparam STATE_CHECK_BATCH    = 4'b1010;
localparam STATE_CHECK_EPOCH    = 4'b1011;
localparam STATE_BANK_SWITCH    = 4'b1001;
localparam STATE_DONE           = 4'b1000;

//================================================================================
// 内部信号
//================================================================================
reg [3:0]  state_reg, state_next;
reg [11:0] timeout_cnt_reg, timeout_cnt_next;
reg [3:0]  switch_delay_cnt_reg, switch_delay_cnt_next;
reg [2:0]  gcu_start_phase_reg, gcu_start_phase_next;

reg [31:0] iteration_next;
reg [7:0]  batch_cnt_next;
reg [7:0]  epoch_cnt_next;
reg [31:0] total_samples_next;

reg [11:0] timeout_threshold;

//================================================================================
// 状态寄存器
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg             <= STATE_IDLE;
        timeout_cnt_reg       <= 12'd0;
        switch_delay_cnt_reg  <= 4'd0;
        gcu_start_phase_reg   <= 3'd0;
        iteration_counter     <= 32'd0;
        batch_counter         <= 8'd0;
        epoch_counter         <= 8'd0;
        total_samples         <= 32'd0;
        timeout_error         <= 1'b0;
    end else begin
        state_reg             <= state_next;
        timeout_cnt_reg       <= timeout_cnt_next;
        switch_delay_cnt_reg  <= switch_delay_cnt_next;
        gcu_start_phase_reg   <= gcu_start_phase_next;
        iteration_counter     <= iteration_next;
        batch_counter         <= batch_cnt_next;
        epoch_counter         <= epoch_cnt_next;
        total_samples         <= total_samples_next;
        
        // 超时错误标志
        if (timeout_cnt_reg >= timeout_threshold && 
            (state_reg == STATE_WAIT_FORWARD || 
             state_reg == STATE_WAIT_ERROR ||
             state_reg == STATE_WAIT_GRAD || 
             state_reg == STATE_WAIT_UPDATE)) begin
            timeout_error <= 1'b1;
        end
    end
end

//================================================================================
// 状态机组合逻辑
//================================================================================
always @(*) begin
    // 默认值
    state_next = state_reg;
    timeout_cnt_next = timeout_cnt_reg;
    switch_delay_cnt_next = switch_delay_cnt_reg;
    gcu_start_phase_next = gcu_start_phase_reg;
    
    iteration_next = iteration_counter;
    batch_cnt_next = batch_counter;
    epoch_cnt_next = epoch_counter;
    total_samples_next = total_samples;
    
    // 默认输出
    train_done = 1'b0;
    train_active = 1'b0;
    init_b_matrices = 1'b0;
    forward_start = 1'b0;
    capture_enable = 1'b0;
    error_start = 1'b0;
    grad_start = 5'b00000;
    update_start = 1'b0;
    request_bank_switch = 1'b0;
    confirm_switch = 1'b0;
    
    timeout_threshold = 12'd4095;
    
    case (state_reg)
        //======================================================================
        // IDLE：等待训练启动
        //======================================================================
        STATE_IDLE: begin
            if (train_start) begin
                state_next = STATE_CHECK_B_INIT;
                iteration_next = 32'd0;
                batch_cnt_next = 8'd0;
                epoch_cnt_next = 8'd1;
                total_samples_next = 32'd0;
            end
        end
        
        //======================================================================
        // CHECK_B_INIT：检查B矩阵
        //======================================================================
        STATE_CHECK_B_INIT: begin
            train_active = 1'b1;
            if (b_matrices_exist) begin
                state_next = STATE_FORWARD_TRIGGER;
            end else begin
                state_next = STATE_INIT_B_MATRIX;
            end
        end
        
        //======================================================================
        // INIT_B_MATRIX：初始化B矩阵
        //======================================================================
        STATE_INIT_B_MATRIX: begin
            train_active = 1'b1;
            init_b_matrices = 1'b1;
            if (b_init_done) begin
                state_next = STATE_FORWARD_TRIGGER;
            end
        end
        
        //======================================================================
        // FORWARD_TRIGGER：触发前向传播
        //======================================================================
        STATE_FORWARD_TRIGGER: begin
            train_active = 1'b1;
            forward_start = 1'b1;
            capture_enable = 1'b1;
            timeout_cnt_next = 12'd0;
            state_next = STATE_WAIT_FORWARD;
        end
        
        //======================================================================
        // WAIT_FORWARD：等待前向传播完成
        //======================================================================
        STATE_WAIT_FORWARD: begin
            train_active = 1'b1;
            capture_enable = 1'b1;
            timeout_threshold = FORWARD_TIMEOUT[11:0];
            
            if (forward_done) begin
                state_next = STATE_CAPTURE_VALID;
                timeout_cnt_next = 12'd0;
            end else if (timeout_cnt_reg >= timeout_threshold) begin
                state_next = STATE_IDLE;
            end else begin
                timeout_cnt_next = timeout_cnt_reg + 12'd1;
            end
        end
        
        //======================================================================
        // CAPTURE_VALID：确认激活值已捕获
        //======================================================================
        STATE_CAPTURE_VALID: begin
            train_active = 1'b1;
            // 等待1个周期确保数据稳定
            state_next = STATE_CALC_ERROR;
        end
        
        //======================================================================
        // CALC_ERROR：启动误差计算
        //======================================================================
        STATE_CALC_ERROR: begin
            train_active = 1'b1;
            error_start = 1'b1;
            timeout_cnt_next = 12'd0;
            state_next = STATE_WAIT_ERROR;
        end
        
        //======================================================================
        // WAIT_ERROR：等待误差计算完成
        //======================================================================
        STATE_WAIT_ERROR: begin
            train_active = 1'b1;
            timeout_threshold = ERROR_TIMEOUT[11:0];
            
            if (error_done && error_valid) begin
                state_next = STATE_GRAD_TRIGGER;
                timeout_cnt_next = 12'd0;
                gcu_start_phase_next = 3'd0;
            end else if (timeout_cnt_reg >= timeout_threshold) begin
                state_next = STATE_IDLE;
            end else begin
                timeout_cnt_next = timeout_cnt_reg + 12'd1;
            end
        end
        
        //======================================================================
        // GRAD_TRIGGER：触发梯度计算
        //======================================================================
        STATE_GRAD_TRIGGER: begin
            train_active = 1'b1;
            timeout_cnt_next = 12'd0;
            
            if (GCU_PARALLEL_START) begin
                grad_start = 5'b11111;
                state_next = STATE_WAIT_GRAD;
            end else begin
                // 串行启动
                case (gcu_start_phase_reg)
                    3'd0: begin
                        grad_start = 5'b10000;
                        gcu_start_phase_next = 3'd1;
                    end
                    3'd1: begin
                        grad_start = 5'b01000;
                        gcu_start_phase_next = 3'd2;
                    end
                    3'd2: begin
                        grad_start = 5'b00100;
                        gcu_start_phase_next = 3'd3;
                    end
                    3'd3: begin
                        grad_start = 5'b00010;
                        gcu_start_phase_next = 3'd4;
                    end
                    3'd4: begin
                        grad_start = 5'b00001;
                        gcu_start_phase_next = 3'd5;
                        state_next = STATE_WAIT_GRAD;
                    end
                    default: begin
                        state_next = STATE_WAIT_GRAD;
                    end
                endcase
            end
        end
        
        //======================================================================
        // WAIT_GRAD：等待梯度计算完成
        //======================================================================
        STATE_WAIT_GRAD: begin
            train_active = 1'b1;
            timeout_threshold = GRADIENT_TIMEOUT[11:0];
            
            if (&grad_done) begin
                state_next = STATE_UPDATE_TRIGGER;
                timeout_cnt_next = 12'd0;
            end else if (timeout_cnt_reg >= timeout_threshold) begin
                state_next = STATE_IDLE;
            end else begin
                timeout_cnt_next = timeout_cnt_reg + 12'd1;
            end
        end
        
        //======================================================================
        // UPDATE_TRIGGER：触发权重更新
        //======================================================================
        STATE_UPDATE_TRIGGER: begin
            train_active = 1'b1;
            update_start = 1'b1;
            timeout_cnt_next = 12'd0;
            state_next = STATE_WAIT_UPDATE;
        end
        
        //======================================================================
        // WAIT_UPDATE：等待权重更新完成
        //======================================================================
        STATE_WAIT_UPDATE: begin
            train_active = 1'b1;
            timeout_threshold = UPDATE_TIMEOUT[11:0];
            
            if (update_done) begin
                state_next = STATE_CHECK_BATCH;
                iteration_next = iteration_counter + 32'd1;
                batch_cnt_next = batch_counter + 8'd1;
                total_samples_next = total_samples + 32'd1;
            end else if (timeout_cnt_reg >= timeout_threshold) begin
                state_next = STATE_IDLE;
            end else begin
                timeout_cnt_next = timeout_cnt_reg + 12'd1;
            end
        end
        
        //======================================================================
        // CHECK_BATCH：检查批次
        //======================================================================
        STATE_CHECK_BATCH: begin
            train_active = 1'b1;
            if (batch_counter >= cfg_batch_size) begin
                batch_cnt_next = 8'd0;
                state_next = STATE_CHECK_EPOCH;
            end else begin
                state_next = STATE_FORWARD_TRIGGER;
            end
        end
        
        //======================================================================
        // CHECK_EPOCH：检查epoch
        //======================================================================
        STATE_CHECK_EPOCH: begin
            train_active = 1'b1;
            if (epoch_counter >= cfg_num_epochs) begin
                state_next = STATE_BANK_SWITCH;
            end else begin
                epoch_cnt_next = epoch_counter + 8'd1;
                state_next = STATE_FORWARD_TRIGGER;
            end
        end
        
        //======================================================================
        // BANK_SWITCH：切换Bank
        //======================================================================
        STATE_BANK_SWITCH: begin
            train_active = 1'b1;
            
            if (switch_delay_cnt_reg == 4'd0) begin
                request_bank_switch = 1'b1;
                switch_delay_cnt_next = 4'd1;
            end else if (switch_delay_cnt_reg < BANK_SWITCH_DELAY[3:0]) begin
                if (bank_switch_ready) begin
                    confirm_switch = 1'b1;
                    switch_delay_cnt_next = BANK_SWITCH_DELAY[3:0];
                end else begin
                    switch_delay_cnt_next = switch_delay_cnt_reg + 4'd1;
                end
            end else begin
                if (switch_complete) begin
                    switch_delay_cnt_next = 4'd0;
                    state_next = STATE_DONE;
                end
            end
        end
        
        //======================================================================
        // DONE：训练完成
        //======================================================================
        STATE_DONE: begin
            train_done = 1'b1;
            state_next = STATE_IDLE;
        end
        
        default: begin
            state_next = STATE_IDLE;
        end
    endcase
end

//================================================================================
// 输出赋值
//================================================================================
assign current_state = state_reg;
assign timeout_counter = timeout_cnt_reg;

endmodule