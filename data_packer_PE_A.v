`timescale 1ns / 1ps
//==============================================================
// PE_A 数据打包器（单 PE 版）：16 个 INT8 → 1 个 PE（8MAC）
//==============================================================

module data_packer_PE_A_single (
    // 16 个 8bit 元素：X[0] 在 [7:0]，X[15] 在 [127:120]
    input  wire [16*8-1:0] mant_X_vec,
    input  wire [16*8-1:0] mant_W_vec,

    // 对接 PE（NUM_MAC=8, DATA_WIDTH=8）的一套 64bit 端口
    output wire [8*8-1:0]  x_data_a_packed, // 8 个 A
    output wire [8*8-1:0]  x_data_b_packed, // 8 个 C
    output wire [8*8-1:0]  w_data_a_packed, // 8 个 B
    output wire [8*8-1:0]  w_data_b_packed  // 8 个 D
);

    // 拆成 16 个 8bit 元素
    wire [7:0] X [0:15];
    wire [7:0] W [0:15];

    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : UNPACK
            assign X[i] = mant_X_vec[8*i +: 8];
            assign W[i] = mant_W_vec[8*i +: 8];
        end
    endgenerate

    genvar m;
    generate
        for (m = 0; m < 8; m = m + 1) begin : PACK
            assign x_data_a_packed[8*m +: 8] = X[2*m];
            assign x_data_b_packed[8*m +: 8] = X[2*m+1];
            assign w_data_a_packed[8*m +: 8] = W[2*m];
            assign w_data_b_packed[8*m +: 8] = W[2*m+1];
        end
    endgenerate

endmodule

