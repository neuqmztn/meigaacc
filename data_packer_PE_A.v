`timescale 1ns / 1ps
//==============================================================
// PE_A 数据打包器：16个INT8元素
//==============================================================

module data_packer_PE_A (
    // 向量输入：16个8bit元素
    input  wire [127:0] mant_X_vec,  // 16×8 = 128bit
    input  wire [127:0] mant_W_vec,
    
    // 打包输出：4×64bit
    output wire [63:0] x_data_a_packed,
    output wire [63:0] x_data_b_packed,
    output wire [63:0] w_data_a_packed,
    output wire [63:0] w_data_b_packed
);

// 解包
wire [7:0] X [0:15];
wire [7:0] W [0:15];

genvar i;
generate
    for (i = 0; i < 16; i = i + 1) begin : unpack
        assign X[i] = mant_X_vec[(i+1)*8-1 : i*8];
        assign W[i] = mant_W_vec[(i+1)*8-1 : i*8];
    end
endgenerate

// 打包：高8个和低8个分开
assign x_data_a_packed = {X[15], X[14], X[13], X[12], X[11], X[10], X[9], X[8]};
assign x_data_b_packed = {X[7],  X[6],  X[5],  X[4],  X[3],  X[2],  X[1], X[0]};
assign w_data_a_packed = {W[15], W[14], W[13], W[12], W[11], W[10], W[9], W[8]};
assign w_data_b_packed = {W[7],  W[6],  W[5],  W[4],  W[3],  W[2],  W[1], W[0]};

endmodule