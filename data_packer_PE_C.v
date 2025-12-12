`timescale 1ns / 1ps
//==============================================================
// PE_C 数据打包器：8个 INT8 元素 → 单 PE，4MAC
//==============================================================

module data_packer_PE_C (
    // 向量输入：8 个 8bit 元素
    input  wire [8*8-1:0] mant_X_vec,  // X[0] 在 [7:0]，X[7] 在 [63:56]
    input  wire [8*8-1:0] mant_W_vec,

    // 打包输出：4 MAC，32bit
    output wire [31:0] x_data_a_packed,
    output wire [31:0] x_data_b_packed,
    output wire [31:0] w_data_a_packed,
    output wire [31:0] w_data_b_packed
);

    wire [7:0] X [0:7];
    wire [7:0] W [0:7];

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : UNPACK
            assign X[i] = mant_X_vec[8*i +: 8];
            assign W[i] = mant_W_vec[8*i +: 8];
        end
    endgenerate

    // 布局规则同上：每个 MAC 处理两对 (Xi,Wi)
    assign x_data_a_packed = { X[6], X[4], X[2], X[0] };
    assign x_data_b_packed = { X[7], X[5], X[3], X[1] };

    assign w_data_a_packed = { W[6], W[4], W[2], W[0] };
    assign w_data_b_packed = { W[7], W[5], W[3], W[1] };

endmodule
