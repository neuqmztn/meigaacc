`timescale 1ns / 1ps

module pp_generator_with_gating #(
    parameter SHIFT = 0  
)(

    input  wire signed [8:0]  multiplicand,
    input  wire [2:0]         booth_code,
    input  wire               skip,
    output wire signed [18:0] pp_aligned
);

    //==============================================================
    // 1. 输入数据门控 (Input Gating)
    //==============================================================
    wire signed [8:0] gated_multiplicand;
    assign gated_multiplicand = skip ? 9'sd0 : multiplicand;

    //==============================================================
    // 2. 预计算基础倍数 (Pre-computation)
    //==============================================================
  
    wire signed [10:0] mult_1x;
    wire signed [10:0] mult_2x;

    // 符号扩展并赋值
    assign mult_1x = {{2{gated_multiplicand[8]}}, gated_multiplicand};      // 1x
    assign mult_2x = {{1{gated_multiplicand[8]}}, gated_multiplicand, 1'b0}; // 2x (左移1位)

    //==============================================================
    // 3. Booth 选择逻辑 (Booth Selector)
    //==============================================================
    
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
 
    wire signed [18:0] pp_extended;
    assign pp_extended = {{8{pp_base[10]}}, pp_base}; 
    assign pp_aligned = pp_extended <<< SHIFT;

endmodule