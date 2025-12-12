`timescale 1ns / 1ps
//==============================================================
// PE_B 数据打包器（INT16，4 元素 / PE，8 MAC）
// 说明：
//   4 个 int16，打包顺序为：
//       X = {x3, x2, x1, x0}，每个 16bit
//       W = {w3, w2, w1, w0}
//   对于每个 PE：
//       pair0: x0,x1,w0,w1 → MAC0..3（等价于一个 PE_D 打包器）
//       pair1: x2,x3,w2,w3 → MAC4..7（再一个 PE_D 打包器）
//==============================================================

module data_packer_PE_B (
    // 4 个 INT16 元素
    input  wire [4*16-1:0] mant_X_vec,  // X[0] 在 [15:0]，X[3] 在 [63:48]
    input  wire [4*16-1:0] mant_W_vec,

    // ---------- PE0：64bit（8MAC×8bit） ----------
    output wire [8*8-1:0]  x_data_a_packed,
    output wire [8*8-1:0]  x_data_b_packed,
    output wire [8*8-1:0]  w_data_a_packed,
    output wire [8*8-1:0]  w_data_b_packed
);

    // 拆成 4 个 16bit
    wire [15:0] X [0:3];
    wire [15:0] W [0:3];

    assign X[0] = mant_X_vec[16*0 +: 16];
    assign X[1] = mant_X_vec[16*1 +: 16];
    assign X[2] = mant_X_vec[16*2 +: 16];
    assign X[3] = mant_X_vec[16*3 +: 16];

    assign W[0] = mant_W_vec[16*0 +: 16];
    assign W[1] = mant_W_vec[16*1 +: 16];
    assign W[2] = mant_W_vec[16*2 +: 16];
    assign W[3] = mant_W_vec[16*3 +: 16];

    // 高 8 位 / 低 8 位
    wire [7:0] x0_l = X[0][7:0];
    wire [7:0] x0_h = X[0][15:8];
    wire [7:0] x1_l = X[1][7:0];
    wire [7:0] x1_h = X[1][15:8];
    wire [7:0] x2_l = X[2][7:0];
    wire [7:0] x2_h = X[2][15:8];
    wire [7:0] x3_l = X[3][7:0];
    wire [7:0] x3_h = X[3][15:8];

    wire [7:0] w0_l = W[0][7:0];
    wire [7:0] w0_h = W[0][15:8];
    wire [7:0] w1_l = W[1][7:0];
    wire [7:0] w1_h = W[1][15:8];
    wire [7:0] w2_l = W[2][7:0];
    wire [7:0] w2_h = W[2][15:8];
    wire [7:0] w3_l = W[3][7:0];
    wire [7:0] w3_h = W[3][15:8];

    assign x_data_a_packed = {
        // MAC7..4 : pair1 (x2)
        x2_h, x2_h, x2_l, x2_l,
        // MAC3..0 : pair0 (x0)
        x0_h, x0_h, x0_l, x0_l
    };

    assign x_data_b_packed = {
        // MAC7..4 : pair1 (x3)
        x3_h, x3_h, x3_l, x3_l,
        // MAC3..0 : pair0 (x1)
        x1_h, x1_h, x1_l, x1_l
    };

    assign w_data_a_packed = {
        // MAC7..4 : pair1 (w2)
        w2_h, w2_l, w2_h, w2_l,
        // MAC3..0 : pair0 (w0)
        w0_h, w0_l, w0_h, w0_l
    };

    assign w_data_b_packed = {
        // MAC7..4 : pair1 (w3)
        w3_h, w3_l, w3_h, w3_l,
        // MAC3..0 : pair0 (w1)
        w1_h, w1_l, w1_h, w1_l
    };

endmodule