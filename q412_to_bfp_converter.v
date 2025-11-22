`timescale 1ns / 1ps

//================================================================================
// Q4.12 to BFP Converter (修复版本)
//
// 功能: 将Q4.12定点格式转换为16-bit BFP格式
//
// Q4.12格式 (16-bit定点):
//   - 表示范围: [-8.0, 7.9998]
//   - 精度: 1/4096 ≈ 0.000244
//
// BFP格式 (Block Floating Point):
//   - mantissa: 16-bit有符号尾数 (归一化到最高位为1)
//   - exponent: 8-bit有符号指数
//   - 实际值 = mantissa × 2^exponent
//
// 转换步骤:
//   1. 取Q4.12值的绝对值
//   2. 计算前导零个数 (Leading Zeros)
//   3. 左移尾数使最高位为1 (归一化)
//   4. 计算指数 = 12 - leading_zeros
//   5. 恢复符号
//
// 修复内容:
//   1. 将Stage 3的计算逻辑分离为组合逻辑
//   2. 统一使用非阻塞赋值进行寄存
//
// 作者: DFA Training System
// 日期: 2025-11-17
// 版本: 1.1 (修复版)
// 标准: Verilog-2001
//================================================================================

module q412_to_bfp_converter (
    input  wire clk,
    input  wire rst_n,
    
    // 输入: Q4.12格式
    input  wire signed [15:0] q412_value,     // Q4.12定点数
    input  wire               valid_in,       // 输入有效
    
    // 输出: BFP格式
    output reg  signed [15:0] bfp_mantissa,   // 16-bit有符号尾数
    output reg  signed [7:0]  bfp_exponent,   // 8-bit有符号指数
    output reg                valid_out       // 输出有效
);

//================================================================================
// 内部信号
//================================================================================

// Stage 1: 输入寄存和符号处理
reg signed [15:0] q412_r;
reg               sign_r;
reg        [15:0] abs_value;
reg               valid_r1;

// Stage 2: 前导零计数
reg        [3:0]  leading_zeros;
reg        [15:0] abs_value_r2;
reg               sign_r2;
reg               valid_r2;
reg               is_zero;

// Stage 3: 归一化计算 (组合逻辑)
reg signed [15:0] normalized_mant_comb;
reg signed [7:0]  exponent_comb;
reg signed [15:0] bfp_mantissa_comb;
reg signed [7:0]  bfp_exponent_comb;

//================================================================================
// Stage 1: 符号处理和取绝对值
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        q412_r    <= 16'd0;
        sign_r    <= 1'b0;
        abs_value <= 16'd0;
        valid_r1  <= 1'b0;
    end else begin
        q412_r   <= q412_value;
        sign_r   <= q412_value[15];
        valid_r1 <= valid_in;
        
        // 取绝对值
        if (q412_value == 16'sh8000) begin
            // -32768的特殊情况 (无法简单取反+1)
            abs_value <= 16'h8000;
        end else if (q412_value[15]) begin
            // 负数: 取反+1
            abs_value <= (~q412_value) + 16'd1;
        end else begin
            // 正数或零
            abs_value <= q412_value;
        end
    end
end

//================================================================================
// Stage 2: 前导零计数 (优先编码器)
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        leading_zeros <= 4'd0;
        abs_value_r2  <= 16'd0;
        sign_r2       <= 1'b0;
        valid_r2      <= 1'b0;
        is_zero       <= 1'b0;
    end else begin
        abs_value_r2 <= abs_value;
        sign_r2      <= sign_r;
        valid_r2     <= valid_r1;
        
        // 检查是否为零
        is_zero <= (abs_value == 16'd0);
        
        // 优先编码器: 找到最高位1的位置
        if (valid_r1) begin
            casez (abs_value)
                16'b1???????????????: leading_zeros <= 4'd0;
                16'b01??????????????: leading_zeros <= 4'd1;
                16'b001?????????????: leading_zeros <= 4'd2;
                16'b0001????????????: leading_zeros <= 4'd3;
                16'b00001???????????: leading_zeros <= 4'd4;
                16'b000001??????????: leading_zeros <= 4'd5;
                16'b0000001?????????: leading_zeros <= 4'd6;
                16'b00000001????????: leading_zeros <= 4'd7;
                16'b000000001???????: leading_zeros <= 4'd8;
                16'b0000000001??????: leading_zeros <= 4'd9;
                16'b00000000001?????: leading_zeros <= 4'd10;
                16'b000000000001????: leading_zeros <= 4'd11;
                16'b0000000000001???: leading_zeros <= 4'd12;
                16'b00000000000001??: leading_zeros <= 4'd13;
                16'b000000000000001?: leading_zeros <= 4'd14;
                16'b0000000000000001: leading_zeros <= 4'd15;
                default:              leading_zeros <= 4'd15;  // 全0情况
            endcase
        end
    end
end

//================================================================================
// Stage 3A: 归一化和指数计算 (组合逻辑)
//================================================================================

always @(*) begin
    // 默认值
    normalized_mant_comb = 16'd0;
    exponent_comb = 8'd0;
    bfp_mantissa_comb = 16'd0;
    bfp_exponent_comb = 8'd0;
    
    if (is_zero) begin
        // 特殊情况: 输入为0
        bfp_mantissa_comb = 16'd0;
        bfp_exponent_comb = 8'sd0;
        
    end else begin
        // 归一化: 左移使最高位为1
        normalized_mant_comb = abs_value_r2 << leading_zeros;
        
        // 计算指数: exponent = -12 + leading_zeros
        // 因为Q4.12小数点在第12位，左移leading_zeros位相当于乘以2^leading_zeros
        // 原始值 = (abs_value / 2^12)
        // 归一化后 = (abs_value << leading_zeros) = abs_value * 2^leading_zeros
        // 所以 BFP值 = normalized_mant * 2^exp，其中 exp = -12 + leading_zeros
        exponent_comb = 8'sd0 - 8'sd12 + {4'd0, leading_zeros};
        
        // 恢复符号：直接对归一化后的值应用符号
        if (sign_r2) begin
            // 负数：将无符号归一化值转为有符号负数
            // 使用二进制补码：取反加1
            bfp_mantissa_comb = (~normalized_mant_comb) + 16'd1;
        end else begin
            // 正数：直接使用，但要确保符号位为0
            bfp_mantissa_comb = {1'b0, normalized_mant_comb[14:0]};
        end
        
        bfp_exponent_comb = exponent_comb;
    end
end

//================================================================================
// Stage 3B: 输出寄存 (时序逻辑) - 全部使用非阻塞赋值
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bfp_mantissa <= 16'd0;
        bfp_exponent <= 8'd0;
        valid_out    <= 1'b0;
    end else begin
        // valid信号直接传递
        valid_out <= valid_r2;
        
        // 当输入有效时更新输出，否则保持
        if (valid_r2) begin
            bfp_mantissa <= bfp_mantissa_comb;
            bfp_exponent <= bfp_exponent_comb;
        end
        // 当输入无效时，寄存器保持不变（隐式）
    end
end

//================================================================================
// 仿真调试输出 (综合时忽略)
//================================================================================

`ifdef SIMULATION
    always @(posedge clk) begin
        if (valid_out) begin
            $display("[Q4.12→BFP] Q4.12=%d (0x%h), BFP: mant=%d, exp=%d, leading_zeros=%d",
                     $signed(q412_r), q412_r, 
                     $signed(bfp_mantissa), $signed(bfp_exponent),
                     leading_zeros);
        end
    end
`endif

endmodule