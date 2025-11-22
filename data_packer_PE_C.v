`timescale 1ns / 1ps
//==============================================================
// PE_C 数据打包器：8个INT8元素
//==============================================================

module data_packer_PE_C (
    // 向量输入：8个8bit元素
    input  wire [63:0] mant_X_vec,  // 8×8 = 64bit
    input  wire [63:0] mant_W_vec,
    
    // 打包输出：4×64bit
    output wire [63:0] x_data_a_packed,
    output wire [63:0] x_data_b_packed,
    output wire [63:0] w_data_a_packed,
    output wire [63:0] w_data_b_packed
);

// 解包
wire [7:0] X [0:7];
wire [7:0] W [0:7];

genvar i;
generate
    for (i = 0; i < 8; i = i + 1) begin : unpack
        assign X[i] = mant_X_vec[(i+1)*8-1 : i*8];
        assign W[i] = mant_W_vec[(i+1)*8-1 : i*8];
    end
endgenerate

// 打包：高32bit为0，低32bit为数据
assign x_data_a_packed = {32'h0, X[7], X[6], X[5], X[4]};
assign x_data_b_packed = {32'h0, X[3], X[2], X[1], X[0]};
assign w_data_a_packed = {32'h0, W[7], W[6], W[5], W[4]};
assign w_data_b_packed = {32'h0, W[3], W[2], W[1], W[0]};

endmodule