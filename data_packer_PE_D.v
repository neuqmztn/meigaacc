`timescale 1ns / 1ps
//==============================================================
// PE_D 数据打包器：2个 INT16 元素 → 单 PE，4MAC
//==============================================================

module data_packer_PE_D (
    // 向量输入：2 个 16bit 元素
    input  wire [2*16-1:0] mant_X_vec,  // a0 在 [15:0]，a1 在 [31:16]
    input  wire [2*16-1:0] mant_W_vec,  // b0 在 [15:0]，b1 在 [31:16]

    // 打包输出：4 MAC，32bit
    output wire [31:0] x_data_a_packed,
    output wire [31:0] x_data_b_packed,
    output wire [31:0] w_data_a_packed,
    output wire [31:0] w_data_b_packed
);

    wire [15:0] a0 = mant_X_vec[15:0];
    wire [15:0] a1 = mant_X_vec[31:16];

    wire [15:0] b0 = mant_W_vec[15:0];
    wire [15:0] b1 = mant_W_vec[31:16];


    assign x_data_a_packed = { a0[15:8], a0[15:8], a0[7:0], a0[7:0] };
    assign x_data_b_packed = { a1[15:8], a1[15:8], a1[7:0], a1[7:0] };

    assign w_data_a_packed = { b0[15:8], b0[7:0],  b0[15:8], b0[7:0] };
    assign w_data_b_packed = { b1[15:8], b1[7:0],  b1[15:8], b1[7:0] };

endmodule
