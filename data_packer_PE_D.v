`timescale 1ns / 1ps
//==============================================================
// PE_D 数据打包器：2个INT16元素
//==============================================================

module data_packer_PE_D (
    // 向量输入：2个16bit元素
    input  wire [31:0] mant_X_vec,  // 2×16 = 32bit
    input  wire [31:0] mant_W_vec,
    
    // 打包输出：4×64bit
    output wire [63:0] x_data_a_packed,
    output wire [63:0] x_data_b_packed,
    output wire [63:0] w_data_a_packed,
    output wire [63:0] w_data_b_packed
);

// 解包
wire [15:0] X0, X1;
wire [15:0] W0, W1;

assign X0 = mant_X_vec[15:0];
assign X1 = mant_X_vec[31:16];
assign W0 = mant_W_vec[15:0];
assign W1 = mant_W_vec[31:16];

// 打包：高32bit为0，低32bit为重复的高低8bit
assign x_data_a_packed = {32'h0, X0[15:8], X0[15:8], X0[7:0], X0[7:0]};
assign x_data_b_packed = {32'h0, X1[15:8], X1[15:8], X1[7:0], X1[7:0]};
assign w_data_a_packed = {32'h0, W0[15:8], W0[7:0], W0[15:8], W0[7:0]};
assign w_data_b_packed = {32'h0, W1[15:8], W1[7:0], W1[15:8], W1[7:0]};

endmodule