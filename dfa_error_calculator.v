`timescale 1ns / 1ps

//================================================================================
// DFA Error Calculator - 改进版（可配置标签位宽）
//
// 功能说明：
// 使用Sigmoid + Binary Cross Entropy计算分类误差向量
// 
// 改进点（v2.1）：
// ✅ 标签位宽可配置（支持不同的分类数）
// ✅ 真实标签范围：0 到 (NUM_CLASSES-1)
// ✅ 自动计算所需的标签位宽
//
// 公式：
// prob[i] = sigmoid(logits[i]) = 1 / (1 + exp(-logits[i]))
// error[i] = prob[i] - target[i]
//   其中 target[i] = 1 (i == true_label)
//                   = 0 (i != true_label)
//
// Sigmoid硬件近似：
// 使用分段线性近似（3段）：
//   x < -2: sigmoid ≈ 0
//   -2 ≤ x ≤ 2: sigmoid ≈ 0.5 + 0.25*x
//   x > 2: sigmoid ≈ 1
//
// 作者：MEIGA Team
// 日期：2025-11-18
// 版本：v2.1 (可配置标签位宽)
//================================================================================

module dfa_error_calculator #(
    parameter NUM_CLASSES = 10,          // 分类类别数（可配置：2, 10, 100, 1000等）
    parameter DATA_WIDTH = 16,           // Q4.12格式
    parameter EXP_WIDTH = 8,
    // 自动计算标签位宽（向上取整log2）
    parameter LABEL_WIDTH = (NUM_CLASSES <= 2)   ? 1 :
                            (NUM_CLASSES <= 4)   ? 2 :
                            (NUM_CLASSES <= 8)   ? 3 :
                            (NUM_CLASSES <= 16)  ? 4 :
                            (NUM_CLASSES <= 32)  ? 5 :
                            (NUM_CLASSES <= 64)  ? 6 :
                            (NUM_CLASSES <= 128) ? 7 : 8
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 输入接口
    //==========================================================================
    input  wire        compute_en,              // 计算使能
    input  wire [LABEL_WIDTH-1:0] true_label,   // 真实标签 (0到NUM_CLASSES-1)
    input  wire [EXP_WIDTH-1:0]    logits_exp,  // Logits共享指数（可选，当前未用）
    input  wire [NUM_CLASSES*DATA_WIDTH-1:0] logits_mant, // Logits尾数
    
    //==========================================================================
    // 输出接口
    //==========================================================================
    output reg         error_valid,             // 误差有效
    output reg  [NUM_CLASSES*DATA_WIDTH-1:0] error_vector, // 误差向量
    output reg  [DATA_WIDTH-1:0] loss_value,    // 损失值（用于监控）
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [2:0]  calc_state,              // 计算状态
    output wire [LABEL_WIDTH-1:0] predicted_class  // 预测类别
);

//================================================================================
// 参数检查
//================================================================================
initial begin
    if (NUM_CLASSES < 2) begin
        $display("ERROR: NUM_CLASSES must be >= 2");
        $finish;
    end
    if (NUM_CLASSES > 256) begin
        $display("ERROR: NUM_CLASSES must be <= 256");
        $finish;
    end
    $display("INFO: Error Calculator configured for %0d classes", NUM_CLASSES);
    $display("INFO: Label width = %0d bits (range 0-%0d)", LABEL_WIDTH, NUM_CLASSES-1);
end

//================================================================================
// 状态定义
//================================================================================
localparam IDLE            = 2'd0;
localparam COMPUTE_SIGMOID = 2'd1;
localparam COMPUTE_ERROR   = 2'd2;
localparam DONE            = 2'd3;

reg [1:0] state_reg, state_next;
reg [LABEL_WIDTH:0] class_cnt_reg, class_cnt_next;  // 需要+1位来表示NUM_CLASSES

//================================================================================
// 内部信号
//================================================================================
// Logits拆分
wire [DATA_WIDTH-1:0] logit [0:NUM_CLASSES-1];
genvar i;
generate
    for (i = 0; i < NUM_CLASSES; i = i + 1) begin : gen_logit_split
        assign logit[i] = logits_mant[i*DATA_WIDTH +: DATA_WIDTH];
    end
endgenerate

// Sigmoid计算结果缓存
reg [DATA_WIDTH-1:0] sigmoid_values [0:NUM_CLASSES-1];

// 预测类别（选择最大sigmoid值）
reg [DATA_WIDTH-1:0] max_sigmoid;
reg [LABEL_WIDTH-1:0] max_sigmoid_idx;

//================================================================================
// 状态机
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg <= IDLE;
        class_cnt_reg <= {(LABEL_WIDTH+1){1'b0}};
        error_valid <= 1'b0;
    end else begin
        state_reg <= state_next;
        class_cnt_reg <= class_cnt_next;
        
        if (state_reg == DONE) begin
            error_valid <= 1'b1;
        end else if (state_reg == IDLE) begin
            error_valid <= 1'b0;
        end
    end
end

//================================================================================
// 状态机组合逻辑
//================================================================================
always @(*) begin
    state_next = state_reg;
    class_cnt_next = class_cnt_reg;
    
    case (state_reg)
        IDLE: begin
            if (compute_en) begin
                state_next = COMPUTE_SIGMOID;
                class_cnt_next = {(LABEL_WIDTH+1){1'b0}};
            end
        end
        
        COMPUTE_SIGMOID: begin
            // 对每个类别计算sigmoid
            if (class_cnt_reg < NUM_CLASSES) begin
                class_cnt_next = class_cnt_reg + 1'b1;
            end else begin
                state_next = COMPUTE_ERROR;
                class_cnt_next = {(LABEL_WIDTH+1){1'b0}};
            end
        end
        
        COMPUTE_ERROR: begin
            // 计算误差：sigmoid - target
            if (class_cnt_reg < NUM_CLASSES) begin
                class_cnt_next = class_cnt_reg + 1'b1;
            end else begin
                state_next = DONE;
            end
        end
        
        DONE: begin
            state_next = IDLE;
        end
        
        default: state_next = IDLE;
    endcase
end

//================================================================================
// Sigmoid计算 - 分段线性近似
//================================================================================
// 函数：计算sigmoid(x)的近似值
// 输入：x (Q4.12格式，有符号16位)
// 输出：sigmoid(x) (Q4.12格式，范围[0,1])
//
// 分段线性近似：
//   x <= -2.0 (0xF800):  sigmoid ≈ 0.0    (0x0000)
//   -2.0 < x < 2.0:      sigmoid ≈ 0.5 + 0.25*x
//   x >= 2.0 (0x0800):   sigmoid ≈ 1.0    (0x0FFF)
//================================================================================
function [DATA_WIDTH-1:0] sigmoid_approx;
    input [DATA_WIDTH-1:0] x;
    reg signed [DATA_WIDTH-1:0] x_signed;
    reg signed [DATA_WIDTH-1:0] result;
    reg signed [DATA_WIDTH-1:0] scaled;
    begin
        x_signed = x;
        
        // 检查范围
        if (x_signed <= -16'sd8192) begin  // x <= -2.0 in Q4.12
            result = 16'd0;  // sigmoid ≈ 0
        end
        else if (x_signed >= 16'sd8192) begin  // x >= 2.0 in Q4.12
            result = 16'h0FFF;  // sigmoid ≈ 1.0 (0.9998 in Q4.12)
        end
        else begin
            // sigmoid ≈ 0.5 + 0.25*x
            // 0.5 in Q4.12 = 2048 (0x0800)
            // 0.25*x = x >> 2
            scaled = x_signed >>> 2;  // 算术右移2位（除以4）
            result = 16'sd2048 + scaled;
            
            // 饱和处理（确保在[0,1]范围）
            if (result < 16'sd0)
                result = 16'd0;
            else if (result > 16'h0FFF)
                result = 16'h0FFF;
        end
        
        sigmoid_approx = result;
    end
endfunction

//================================================================================
// Sigmoid和误差计算逻辑
//================================================================================
integer j;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        max_sigmoid <= 16'd0;
        max_sigmoid_idx <= {LABEL_WIDTH{1'b0}};
        // 初始化sigmoid_values数组
        for (j = 0; j < NUM_CLASSES; j = j + 1) begin
            sigmoid_values[j] <= 16'd0;
        end
    end else begin
        case (state_reg)
            COMPUTE_SIGMOID: begin
                if (class_cnt_reg < NUM_CLASSES) begin
                    // 计算sigmoid值并存储
                    sigmoid_values[class_cnt_reg[LABEL_WIDTH-1:0]] <= 
                        sigmoid_approx(logit[class_cnt_reg[LABEL_WIDTH-1:0]]);
                    
                    // 同时跟踪最大值（用于预测）
                    if (class_cnt_reg == 0) begin
                        max_sigmoid <= sigmoid_approx(logit[0]);
                        max_sigmoid_idx <= {LABEL_WIDTH{1'b0}};
                    end else begin
                        if (sigmoid_approx(logit[class_cnt_reg[LABEL_WIDTH-1:0]]) > max_sigmoid) begin
                            max_sigmoid <= sigmoid_approx(logit[class_cnt_reg[LABEL_WIDTH-1:0]]);
                            max_sigmoid_idx <= class_cnt_reg[LABEL_WIDTH-1:0];
                        end
                    end
                end
            end
            
            COMPUTE_ERROR: begin
                if (class_cnt_reg < NUM_CLASSES) begin
                    // 比较当前类别是否为真实标签
                    if (class_cnt_reg[LABEL_WIDTH-1:0] == true_label) begin
                        // 正确类别：error = sigmoid - 1
                        error_vector[class_cnt_reg[LABEL_WIDTH-1:0]*DATA_WIDTH +: DATA_WIDTH] <= 
                            sigmoid_values[class_cnt_reg[LABEL_WIDTH-1:0]] - 16'h1000;  // -1.0 in Q4.12
                    end else begin
                        // 错误类别：error = sigmoid - 0 = sigmoid
                        error_vector[class_cnt_reg[LABEL_WIDTH-1:0]*DATA_WIDTH +: DATA_WIDTH] <= 
                            sigmoid_values[class_cnt_reg[LABEL_WIDTH-1:0]];
                    end
                end
            end
        endcase
    end
end

//================================================================================
// Loss计算 - Binary Cross Entropy
//================================================================================
// BCE Loss = -[y*log(p) + (1-y)*log(1-p)]
// 简化：对于正确类别，loss ≈ -log(sigmoid(logit))
//       使用近似：-log(p) ≈ 1 - p (当p接近1)
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        loss_value <= 16'd0;
    end else if (state_reg == DONE) begin
        // 简化loss：1.0 - sigmoid(correct_class)
        // 如果sigmoid接近1，loss接近0（好）
        // 如果sigmoid接近0，loss接近1（差）
        loss_value <= 16'h1000 - sigmoid_values[true_label];  // 1.0 - p
    end
end

//================================================================================
// 调试输出
//================================================================================
assign calc_state = {1'b0, state_reg};  // 扩展到3位
assign predicted_class = max_sigmoid_idx;

endmodule