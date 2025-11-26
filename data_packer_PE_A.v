`timescale 1ns / 1ps
//==============================================================
// PE_A 数据打包器：16个 INT8 元素 → 双 PE，每个 PE 4MAC
//==============================================================

module data_packer_PE_A (
    // 向量输入：16 个 8bit 元素
    input  wire [16*8-1:0] mant_X_vec,  // X[0] 在 [7:0]，X[15] 在 [127:120]
    input  wire [16*8-1:0] mant_W_vec,

    // PE0 的打包输出（4 MAC，32bit）
    output wire [31:0] pe0_x_data_a_packed,
    output wire [31:0] pe0_x_data_b_packed,
    output wire [31:0] pe0_w_data_a_packed,
    output wire [31:0] pe0_w_data_b_packed,

    // PE1 的打包输出（4 MAC，32bit）
    output wire [31:0] pe1_x_data_a_packed,
    output wire [31:0] pe1_x_data_b_packed,
    output wire [31:0] pe1_w_data_a_packed,
    output wire [31:0] pe1_w_data_b_packed
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

    //==============================
    // PE0：处理 X/W[0..7]
    //==============================
    assign pe0_x_data_a_packed = {
        X[6], X[4], X[2], X[0]   // A3,A2,A1,A0
    };

    assign pe0_x_data_b_packed = {
        X[7], X[5], X[3], X[1]   // C3,C2,C1,C0
    };

    assign pe0_w_data_a_packed = {
        W[6], W[4], W[2], W[0]   // B3,B2,B1,B0
    };

    assign pe0_w_data_b_packed = {
        W[7], W[5], W[3], W[1]   // D3,D2,D1,D0
    };

    //==============================
    // PE1：处理 X/W[8..15]
    //==============================
    assign pe1_x_data_a_packed = {
        X[14], X[12], X[10], X[8]
    };

    assign pe1_x_data_b_packed = {
        X[15], X[13], X[11], X[9]
    };

    assign pe1_w_data_a_packed = {
        W[14], W[12], W[10], W[8]
    };

    assign pe1_w_data_b_packed = {
        W[15], W[13], W[11], W[9]
    };

endmodule
