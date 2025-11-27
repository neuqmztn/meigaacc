`timescale 1ns / 1ps

module classification_error_module #(
    parameter DIM = 32,
    parameter DATA_WIDTH = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire train_mode,
    output reg  done,
    output reg  busy,
    input  wire true_label,
    output reg  lob_rd_cls_en,
    input  wire [DIM*DATA_WIDTH-1:0] lob_cls_data_q412,
    input  wire lob_cls_valid,
    output reg  [4:0]  weight_addr,
    input  wire [DATA_WIDTH-1:0] weight_data,
    input  wire [DATA_WIDTH-1:0] bias,
    output reg  [DATA_WIDTH-1:0] prob,
    output reg  predicted_class,
    output reg  result_valid,
    output reg  [DATA_WIDTH-1:0] error,
    output reg  error_valid,
    output wire [3:0]  state,
    output wire [DATA_WIDTH-1:0] debug_logit
);

//================================================================================
// 状态定义
//================================================================================
localparam IDLE         = 4'd0;
localparam READ_CLS     = 4'd1;
localparam WAIT_CLS     = 4'd2;
localparam COMPUTE_MAC  = 4'd3;
localparam ADD_BIAS     = 4'd4;
localparam SIGMOID      = 4'd5;
localparam OUTPUT_PROB  = 4'd6;
localparam CALC_ERROR   = 4'd7;
localparam DONE_STATE   = 4'd8;

reg [3:0] state_reg, state_next;

// 内部寄存器
reg [DATA_WIDTH-1:0] cls_token_q412 [0:DIM-1];
reg [5:0] mac_cnt_reg, mac_cnt_next;
reg signed [39:0] accumulator_reg, accumulator_next;
reg signed [DATA_WIDTH-1:0] logit_reg, logit_next;
reg [DATA_WIDTH-1:0] prob_reg, prob_next;
reg signed [DATA_WIDTH-1:0] error_reg, error_next;
reg predicted_class_reg, predicted_class_next;

//================================================================================
// CLS Token 锁存逻辑
//================================================================================
integer k;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (k = 0; k < DIM; k = k + 1) cls_token_q412[k] <= 16'd0;
    end else begin
        if (lob_cls_valid) begin
            for (k = 0; k < DIM; k = k + 1) begin
                cls_token_q412[k] <= lob_cls_data_q412[k*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
end

//================================================================================
// 主状态机
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg <= IDLE;
        mac_cnt_reg <= 6'd0;
        accumulator_reg <= 40'sd0;
        logit_reg <= 16'sd0;
        prob_reg <= 16'sd0;
        error_reg <= 16'sd0;
        predicted_class_reg <= 1'b0;
    end else begin
        state_reg <= state_next;
        mac_cnt_reg <= mac_cnt_next;
        accumulator_reg <= accumulator_next;
        logit_reg <= logit_next;
        prob_reg <= prob_next;
        error_reg <= error_next;
        predicted_class_reg <= predicted_class_next;
    end
end

//================================================================================
// 组合逻辑
//================================================================================
reg signed [39:0] mult_result;
reg signed [39:0] acc_shifted;
reg signed [39:0] logit_temp;

always @(*) begin
    state_next = state_reg;
    mac_cnt_next = mac_cnt_reg;
    accumulator_next = accumulator_reg;
    logit_next = logit_reg;
    prob_next = prob_reg;
    error_next = error_reg;
    predicted_class_next = predicted_class_reg;
    
    lob_rd_cls_en = 1'b0;
    done = 1'b0;
    busy = (state_reg != IDLE);
    result_valid = 1'b0;
    error_valid = 1'b0;
    weight_addr = 5'd0;
    
    mult_result = 40'sd0;
    acc_shifted = 40'sd0;
    logit_temp = 40'sd0;

    case (state_reg)
        IDLE: begin
            if (start) begin
                state_next = READ_CLS;
                mac_cnt_next = 6'd0;
                accumulator_next = 40'sd0;
            end
        end
        
        READ_CLS: begin
            lob_rd_cls_en = 1'b1;
            state_next = WAIT_CLS;
        end
        
        WAIT_CLS: begin
            if (lob_cls_valid) begin
                state_next = COMPUTE_MAC;
                mac_cnt_next = 6'd0;
            end
        end
        
        COMPUTE_MAC: begin
            weight_addr = mac_cnt_reg[4:0];
            mult_result = $signed(cls_token_q412[mac_cnt_reg]) * $signed(weight_data);
            accumulator_next = accumulator_reg + mult_result;

            if (mac_cnt_reg == DIM-1) state_next = ADD_BIAS;
            else mac_cnt_next = mac_cnt_reg + 1;
        end
        
        ADD_BIAS: begin
            acc_shifted = accumulator_reg >>> 12;
            logit_temp = acc_shifted + $signed(bias);
            
            if (logit_temp > $signed(40'd32767)) logit_next = 16'h7FFF;
            else if (logit_temp < $signed(-40'd32768)) logit_next = 16'h8000;
            else logit_next = logit_temp[15:0];
            
            state_next = SIGMOID;
        end
        
        SIGMOID: begin
            prob_next = sigmoid_pwla_16seg(logit_reg);
            predicted_class_next = (prob_next > 16'h0800) ? 1'b1 : 1'b0;
            state_next = OUTPUT_PROB;
        end
        
        OUTPUT_PROB: begin
            result_valid = 1'b1;
            if (train_mode) state_next = CALC_ERROR;
            else state_next = DONE_STATE;
        end
        
        CALC_ERROR: begin
            if (true_label) error_next = $signed(prob_reg) - $signed(16'h1000);
            else error_next = $signed(prob_reg);
            
            error_valid = 1'b1;
            state_next = DONE_STATE;
        end
        
        DONE_STATE: begin
            done = 1'b1;
            state_next = IDLE;
        end
        
        default: state_next = IDLE;
    endcase
end

//================================================================================
// 输出寄存器同步 [关键修复]
//================================================================================
// 修复点：使用 _next 信号更新输出，消除与 done 信号之间的时钟偏差
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        prob <= 16'd0;
        predicted_class <= 1'b0;
        error <= 16'd0;
    end else begin
        // 使用 _next 而不是 _reg
        prob <= prob_next;
        predicted_class <= predicted_class_next;
        error <= error_next;
    end
end

// Sigmoid 函数
function [DATA_WIDTH-1:0] sigmoid_pwla_16seg;
    input signed [DATA_WIDTH-1:0] x_in;
    reg signed [DATA_WIDTH-1:0] x_abs;
    reg is_negative;
    reg [DATA_WIDTH-1:0] y_temp;
    reg [DATA_WIDTH-1:0] delta;
    begin
        if (x_in < 0) begin
            x_abs = -x_in;
            is_negative = 1'b1;
        end else begin
            x_abs = x_in;
            is_negative = 1'b0;
        end
        
        if (x_abs >= 16'h4000) y_temp = 16'h1000; 
        else if (x_abs >= 16'h2000) begin
            delta = (x_abs - 16'h2000) >> 4;
            y_temp = 16'h0E00 + delta;
        end else if (x_abs >= 16'h1000) begin
            delta = (x_abs - 16'h1000) >> 3;
            y_temp = 16'h0C00 + delta;
        end else begin
            delta = x_abs >> 2;
            y_temp = 16'h0800 + delta;
        end
        
        if (is_negative) sigmoid_pwla_16seg = 16'h1000 - y_temp;
        else sigmoid_pwla_16seg = y_temp;
    end
endfunction

assign state = state_reg;
assign debug_logit = logit_reg;

endmodule