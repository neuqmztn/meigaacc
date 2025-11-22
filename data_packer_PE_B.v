`timescale 1ns / 1ps
//==============================================================
// PE_B 数据打包器：4个INT16元素
//==============================================================

module data_packer_PE_B (
    // 向量输入：4个16bit元素
    input  wire [63:0] mant_X_vec,  // 4×16 = 64bit
    input  wire [63:0] mant_W_vec,
    
    // 打包输出：4×64bit
    output wire [63:0] x_data_a_packed,
    output wire [63:0] x_data_b_packed,
    output wire [63:0] w_data_a_packed,
    output wire [63:0] w_data_b_packed
);

// 解包
wire [15:0] X [0:3];
wire [15:0] W [0:3];

genvar i;
generate
    for (i = 0; i < 4; i = i + 1) begin : unpack
        assign X[i] = mant_X_vec[(i+1)*16-1 : i*16];
        assign W[i] = mant_W_vec[(i+1)*16-1 : i*16];
    end
endgenerate

// 打包：拆分高低8bit并重复
// X布局：[X1_H, X1_H, X1_L, X1_L, X0_H, X0_H, X0_L, X0_L]
assign x_data_a_packed = {X[1][15:8], X[1][15:8], X[1][7:0], X[1][7:0],
                          X[0][15:8], X[0][15:8], X[0][7:0], X[0][7:0]};
assign x_data_b_packed = {X[3][15:8], X[3][15:8], X[3][7:0], X[3][7:0],
                          X[2][15:8], X[2][15:8], X[2][7:0], X[2][7:0]};

// W布局：[W1_H, W1_L, W1_H, W1_L, W0_H, W0_L, W0_H, W0_L]
assign w_data_a_packed = {W[1][15:8], W[1][7:0], W[1][15:8], W[1][7:0],
                          W[0][15:8],W[0][7:0], W[0][15:8], W[0][7:0]};
assign w_data_b_packed = {W[3][15:8], W[3][7:0], W[3][15:8], W[3][7:0],
                          W[2][15:8], W[2][7:0], W[2][15:8], W[2][7:0]};

endmodule