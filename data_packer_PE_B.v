`timescale 1ns / 1ps
//==============================================================
// PE_B 数据打包器：4个 INT16 元素 → 双 PE，每个 PE 4MAC
//
// 设计：每个 PE_B 复用 PE_D 的拆分逻辑（2个 int16）
//       PE0 处理 (a0,a1,b0,b1)，PE1 处理 (a2,a3,b2,b3)
//==============================================================

module data_packer_PE_B (
    // 向量输入：4 个 16bit 元素
    input  wire [4*16-1:0] mant_X_vec,  // a0 在 [15:0]，a3 在 [63:48]
    input  wire [4*16-1:0] mant_W_vec,  // b0 在 [15:0]，b3 在 [63:48]

    // PE0 打包输出（2×INT16 → 4MAC）
    output wire [31:0] pe0_x_data_a_packed,
    output wire [31:0] pe0_x_data_b_packed,
    output wire [31:0] pe0_w_data_a_packed,
    output wire [31:0] pe0_w_data_b_packed,

    // PE1 打包输出（2×INT16 → 4MAC）
    output wire [31:0] pe1_x_data_a_packed,
    output wire [31:0] pe1_x_data_b_packed,
    output wire [31:0] pe1_w_data_a_packed,
    output wire [31:0] pe1_w_data_b_packed
);

    // 拆成 4 个 int16：a0..a3
    wire [15:0] a0 = mant_X_vec[15:0];
    wire [15:0] a1 = mant_X_vec[31:16];
    wire [15:0] a2 = mant_X_vec[47:32];
    wire [15:0] a3 = mant_X_vec[63:48];

    wire [15:0] b0 = mant_W_vec[15:0];
    wire [15:0] b1 = mant_W_vec[31:16];
    wire [15:0] b2 = mant_W_vec[47:32];
    wire [15:0] b3 = mant_W_vec[63:48];

    //==============================
    // PE0：等价于一个 PE_D(a0,a1,b0,b1)
    //==============================
    assign pe0_x_data_a_packed = { a0[15:8], a0[15:8], a0[7:0], a0[7:0] };
    assign pe0_x_data_b_packed = { a1[15:8], a1[15:8], a1[7:0], a1[7:0] };

    assign pe0_w_data_a_packed = { b0[15:8], b0[7:0],  b0[15:8], b0[7:0] };
    assign pe0_w_data_b_packed = { b1[15:8], b1[7:0],  b1[15:8], b1[7:0] };

    //==============================
    // PE1：等价于一个 PE_D(a2,a3,b2,b3)
    //==============================
    assign pe1_x_data_a_packed = { a2[15:8], a2[15:8], a2[7:0], a2[7:0] };
    assign pe1_x_data_b_packed = { a3[15:8], a3[15:8], a3[7:0], a3[7:0] };

    assign pe1_w_data_a_packed = { b2[15:8], b2[7:0],  b2[15:8], b2[7:0] };
    assign pe1_w_data_b_packed = { b3[15:8], b3[7:0],  b3[15:8], b3[7:0] };

endmodule
