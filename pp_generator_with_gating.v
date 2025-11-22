`timescale 1ns / 1ps

module pp_generator_with_gating #(
    parameter SHIFT = 0  // 权重移位量 (0, 2, 4, 6, 8)
)(
    //==============================================================
    // 端口定义
    //==============================================================
    // 为了支持 8-bit 无符号数 (0~255)，必须用 9 位有符号数来装载 (0xxxxxxxx)
    input  wire signed [8:0]  multiplicand,
    
    // Booth 编码输入 {sign, sel[1:0]}
    input  wire [2:0]         booth_code,
    
    // Skip 信号 (1=跳过/置零)
    input  wire               skip,
    
 
    // 最大可能值为 255(输入) * 2(Booth) * 256(Shift=8) ≈ 130,560
    // 17-bit 有符号数上限仅为 65,535，会导致溢出。19-bit 足够安全。
    output wire signed [18:0] pp_aligned
);

    //==============================================================
    // 1. 输入数据门控 (Input Gating)
    //==============================================================
    // 如果 skip 有效，直接强制输入为 0，降低后续逻辑翻转功耗
    wire signed [8:0] gated_multiplicand;
    assign gated_multiplicand = skip ? 9'sd0 : multiplicand;

    //==============================================================
    // 2. 预计算基础倍数 (Pre-computation)
    //==============================================================
    // 为了防止计算 -2x 时发生溢出，中间变量使用 11 位 (9位数据 + 1位左移 + 1位符号缓冲)
    
    wire signed [10:0] mult_1x;
    wire signed [10:0] mult_2x;

    // 符号扩展并赋值
    assign mult_1x = {{2{gated_multiplicand[8]}}, gated_multiplicand};      // 1x
    assign mult_2x = {{1{gated_multiplicand[8]}}, gated_multiplicand, 1'b0}; // 2x (左移1位)

    //==============================================================
    // 3. Booth 选择逻辑 (Booth Selector)
    //==============================================================
    // 根据 booth_encoder_3bit 输出的编码选择操作数
    // 编码定义:
    // 000, 011, 100, 111 -> 0
    // 001, 010           -> +1x
    // 011 (编码后变为010) -> +2x 
    // 100 (编码后变为110) -> -2x
    // 101, 110           -> -1x
    
    reg signed [10:0] pp_base;

    always @(*) begin
        case (booth_code)
            3'b000: pp_base = 11'sd0;     // 0
            3'b001: pp_base = mult_1x;    // +1x
            3'b010: pp_base = mult_2x;    // +2x 
            3'b011: pp_base = 11'sd0;     // 0 (skip)
            3'b100: pp_base = 11'sd0;     // 0 (reserved)
            3'b101: pp_base = -mult_1x;   // -1x
            3'b110: pp_base = -mult_2x;   // -2x 
            3'b111: pp_base = 11'sd0;     // 0 (skip)
            default: pp_base = 11'sd0;
        endcase
    end

    //==============================================================
    // 4. 移位对齐与符号扩展 (Shift & Alignment)
    //==============================================================
    
    // 先将 11 位的 pp_base 符号扩展到 19 位
    wire signed [18:0] pp_extended;
    assign pp_extended = {{8{pp_base[10]}}, pp_base}; 

    // 执行算术左移 (<<< 会自动处理低位补0)
    // 这一步实现了 Booth 算法中的权重偏移 (x1, x4, x16, x64, x256)
    assign pp_aligned = pp_extended <<< SHIFT;

endmodule