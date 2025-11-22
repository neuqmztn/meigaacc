`timescale 1ns / 1ps

//================================================================================
// Q4.12 Fixed-Point Multiplier with Saturation
//
// 功能说明：
// 实现Q4.12格式的定点乘法，带溢出检测和饱和处理
//
// Q4.12格式：
// • 16-bit 有符号定点数
// • 4位整数部分 + 12位小数部分
// • 范围: [-8.0, 7.9998]
// • 精度: 1/4096 ≈ 0.000244
//
// 乘法运算：
// (A × B) / 4096 → Q4.12结果
//
// 算法：
// 1. 16位×16位有符号乘法 → 32位乘积 (Q8.24格式)
// 2. 右移12位提取Q4.12结果
// 3. 检测溢出并饱和
//
// 详细计算：
// A = a_int + a_frac/4096 (Q4.12)
// B = b_int + b_frac/4096 (Q4.12)
// 
// A_raw = A × 4096 (16-bit整数表示)
// B_raw = B × 4096 (16-bit整数表示)
//
// Product_raw = A_raw × B_raw (32-bit)
//             = (A × 4096) × (B × 4096)
//             = A × B × 4096²
//
// Product_Q4.12 = Product_raw / 4096
//               = A × B × 4096  (正确!)
//
// 所以：result = product_raw[27:12]  // 右移12位
//
// 溢出检测：
// 检查product_raw[31:27]是否为符号扩展
// • 正数：product_raw[31:27] 应全为0
// • 负数：product_raw[31:27] 应全为1
//
// 饱和：
// if (overflow):
//     result = (product_raw[31]) ? -32768 : 32767
//
// 流水线选项：
// • 1级流水线（推荐）: 乘法+饱和在1个周期
// • 2级流水线（高频）: 乘法1周期，饱和1周期
//
// 作者：MEIGA Team
// 日期：2025-11-18
// 版本：v1.0
//================================================================================

module q412_multiplier #(
    parameter DATA_WIDTH    = 16,       // Q4.12位宽
    parameter FRAC_BITS     = 12,       // 小数位数
    parameter PIPELINE      = 1         // 流水线级数 (1 or 2)
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire        valid_in,        // 输入有效
    output reg         valid_out,       // 输出有效
    
    //==========================================================================
    // 输入
    //==========================================================================
    input  wire signed [DATA_WIDTH-1:0] a,      // 操作数A (Q4.12)
    input  wire signed [DATA_WIDTH-1:0] b,      // 操作数B (Q4.12)
    
    //==========================================================================
    // 输出
    //==========================================================================
    output reg  signed [DATA_WIDTH-1:0] result, // 结果 (Q4.12)
    
    //==========================================================================
    // 状态标志
    //==========================================================================
    output reg         overflow,        // 溢出标志
    output reg         saturated        // 饱和标志
);

//================================================================================
// 内部信号
//================================================================================
wire signed [31:0] product_full;    // 完整32位乘积 (Q8.24)
wire signed [DATA_WIDTH-1:0] product_truncated;  // 截断到Q4.12

// 溢出检测信号
wire overflow_pos;  // 正溢出
wire overflow_neg;  // 负溢出
wire overflow_detected;

// 饱和值
localparam signed [DATA_WIDTH-1:0] SAT_MAX =  16'sd32767;  // +7.9998
localparam signed [DATA_WIDTH-1:0] SAT_MIN = -16'sd32768;  // -8.0

//================================================================================
// Stage 1: 乘法运算
//================================================================================

// 32位有符号乘法
assign product_full = a * b;  // 自动综合为DSP48

// 右移12位得到Q4.12结果
assign product_truncated = product_full[27:12];

// 溢出检测
// 正数溢出：高5位不全为0
// 负数溢出：高5位不全为1
assign overflow_pos = (!product_full[31]) && (|product_full[31:27]);
assign overflow_neg = (product_full[31])  && (!(&product_full[31:27]));
assign overflow_detected = overflow_pos || overflow_neg;

//================================================================================
// 饱和处理和输出
//================================================================================

generate
    if (PIPELINE == 1) begin : gen_pipeline_1
        // 1级流水线：乘法和饱和在同一周期
        
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                result     <= {DATA_WIDTH{1'b0}};
                overflow   <= 1'b0;
                saturated  <= 1'b0;
                valid_out  <= 1'b0;
            end else begin
                valid_out <= valid_in;
                
                if (valid_in) begin
                    overflow  <= overflow_detected;
                    saturated <= overflow_detected;
                    
                    if (overflow_detected) begin
                        // 饱和处理
                        if (overflow_pos) begin
                            result <= SAT_MAX;  // 正溢出
                        end else begin
                            result <= SAT_MIN;  // 负溢出
                        end
                    end else begin
                        // 正常结果
                        result <= product_truncated;
                    end
                end
            end
        end
        
    end else if (PIPELINE == 2) begin : gen_pipeline_2
        // 2级流水线：乘法1周期，饱和1周期（用于高频设计）
        
        reg signed [31:0] s1_product;
        reg               s1_valid;
        
        // Stage 1: 乘法
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                s1_product <= 32'sd0;
                s1_valid   <= 1'b0;
            end else begin
                s1_product <= product_full;
                s1_valid   <= valid_in;
            end
        end
        
        // Stage 2: 饱和
        wire signed [DATA_WIDTH-1:0] s1_truncated;
        wire s1_overflow_pos, s1_overflow_neg, s1_overflow;
        
        assign s1_truncated    = s1_product[27:12];
        assign s1_overflow_pos = (!s1_product[31]) && (|s1_product[31:27]);
        assign s1_overflow_neg = (s1_product[31])  && (!(&s1_product[31:27]));
        assign s1_overflow     = s1_overflow_pos || s1_overflow_neg;
        
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                result     <= {DATA_WIDTH{1'b0}};
                overflow   <= 1'b0;
                saturated  <= 1'b0;
                valid_out  <= 1'b0;
            end else begin
                valid_out <= s1_valid;
                
                if (s1_valid) begin
                    overflow  <= s1_overflow;
                    saturated <= s1_overflow;
                    
                    if (s1_overflow) begin
                        if (s1_overflow_pos) begin
                            result <= SAT_MAX;
                        end else begin
                            result <= SAT_MIN;
                        end
                    end else begin
                        result <= s1_truncated;
                    end
                end
            end
        end
        
    end
endgenerate

//================================================================================
// 仿真监控
//================================================================================
`ifdef SIMULATION

reg [31:0] mult_count;
reg [31:0] overflow_count;
real a_float, b_float, result_float, expected;

initial begin
    mult_count = 0;
    overflow_count = 0;
    
    $display("========================================");
    $display("Q4.12 Multiplier");
    $display("========================================");
    $display("Pipeline: %0d stage(s)", PIPELINE);
    $display("Latency: %0d cycle(s)", PIPELINE);
    $display("========================================");
end

always @(posedge clk) begin
    if (valid_out) begin
        mult_count = mult_count + 1;
        
        if (overflow) begin
            overflow_count = overflow_count + 1;
        end
        
        // 每1000次乘法显示统计
        if (mult_count % 1000 == 0) begin
            $display("[%0t] Q4.12 Mult Stats: Total=%0d, Overflow=%0d (%.2f%%)",
                     $time, mult_count, overflow_count, 
                     100.0 * overflow_count / mult_count);
        end
        
        // 前10次显示详细信息
        if (mult_count <= 10) begin
            // 转换为浮点验证（需要pipeline延迟对齐，这里简化）
            a_float = $itor($signed(a)) / 4096.0;
            b_float = $itor($signed(b)) / 4096.0;
            result_float = $itor($signed(result)) / 4096.0;
            expected = a_float * b_float;
            
            $display("[%0t] Mult #%0d: %f × %f = %f (expected %f, error=%.6f)",
                     $time, mult_count, a_float, b_float, result_float, 
                     expected, result_float - expected);
            
            if (saturated) begin
                $display("       → SATURATED!");
            end
        end
    end
end

`endif // SIMULATION

endmodule