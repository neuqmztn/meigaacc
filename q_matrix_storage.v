`timescale 1ns / 1ps

//================================================================================
// Q Matrix Storage - Q矩阵存储（4-Bank并行架构）
//
// 功能说明:
//   - 存储当前batch的Q矩阵: 4头 × 32 tokens × 8维
//   - 支持4个Head并行读取，每个Head读取自己的bank
//   - 每次读取输出整个batch（32个token）
//
//================================================================================

`timescale 1ns / 1ps

module q_matrix_storage #(
    parameter NUM_HEADS      = 4,
    parameter TOKEN_BATCH    = 32,
    parameter HEAD_DIM       = 8,
    parameter DATA_WIDTH     = 8,
    parameter EXP_WIDTH      = 8
)(
    input  wire clk,
    input  wire rst_n,
    
    // 写接口
    input  wire wr_en,
    input  wire [1:0] wr_head,
    input  wire [4:0] wr_token,
    input  wire [EXP_WIDTH-1:0] wr_exp,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0] wr_mant_packed,
    
    // Head 0 读取接口
    input  wire rd_en_h0,
    output reg  [(TOKEN_BATCH*EXP_WIDTH)-1:0] rd_batch_exp_h0,
    output reg  [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] rd_batch_mant_h0,
    
    // Head 1 读取接口
    input  wire rd_en_h1,
    output reg  [(TOKEN_BATCH*EXP_WIDTH)-1:0] rd_batch_exp_h1,
    output reg  [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] rd_batch_mant_h1,
    
    // Head 2 读取接口
    input  wire rd_en_h2,
    output reg  [(TOKEN_BATCH*EXP_WIDTH)-1:0] rd_batch_exp_h2,
    output reg  [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] rd_batch_mant_h2,
    
    // Head 3 读取接口
    input  wire rd_en_h3,
    output reg  [(TOKEN_BATCH*EXP_WIDTH)-1:0] rd_batch_exp_h3,
    output reg  [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] rd_batch_mant_h3
);

    //================================================================================
    // 存储阵列定义 (关键修改点)
    //================================================================================
    // 添加综合属性，强制使用 Distributed RAM (LUTRAM)
    // 注意：Vivado/Quartus 都能识别 (* ram_style *)
    
    (* ram_style = "distributed" *) 
    reg [EXP_WIDTH-1:0] Q_exp_mem [0:NUM_HEADS-1][0:TOKEN_BATCH-1];
    
    (* ram_style = "distributed" *) 
    reg [DATA_WIDTH-1:0] Q_mant_mem [0:NUM_HEADS-1][0:TOKEN_BATCH-1][0:HEAD_DIM-1];

    integer i;

    //================================================================================
    // 写入逻辑 (关键修改点：移除复位)
    //================================================================================
    // 只有移除 if(!rst_n) 对 mem 的清零，综合器才会推断为 RAM
    
    always @(posedge clk) begin
        if (wr_en && wr_token < TOKEN_BATCH) begin
            // 写入指数
            Q_exp_mem[wr_head][wr_token] <= wr_exp;
            
            // 写入尾数
            for (i = 0; i < HEAD_DIM; i = i + 1) begin
                Q_mant_mem[wr_head][wr_token][i] <= 
                    wr_mant_packed[i*DATA_WIDTH +: DATA_WIDTH];
            end
            
            // 调试打印 (仿真用)
            // $display("[%0t] Q_Storage WR: head=%0d token=%0d exp=%0d", $time, wr_head, wr_token, wr_exp);
        end
    end

    //================================================================================
    // 读取逻辑 - Head 0
    //================================================================================
    integer t0, d0;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 依然复位输出端口 (这是安全的，且必要的)
            rd_batch_exp_h0  <= {(TOKEN_BATCH*EXP_WIDTH){1'b0}};
            rd_batch_mant_h0 <= {(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH){1'b0}};
        end else if (rd_en_h0) begin
            for (t0 = 0; t0 < TOKEN_BATCH; t0 = t0 + 1) begin
                rd_batch_exp_h0[t0*EXP_WIDTH +: EXP_WIDTH] <= Q_exp_mem[0][t0];
                for (d0 = 0; d0 < HEAD_DIM; d0 = d0 + 1) begin
                    rd_batch_mant_h0[(t0*HEAD_DIM + d0)*DATA_WIDTH +: DATA_WIDTH] <= Q_mant_mem[0][t0][d0];
                end
            end
        end
    end

    //================================================================================
    // 读取逻辑 - Head 1
    //================================================================================
    integer t1, d1;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_batch_exp_h1  <= {(TOKEN_BATCH*EXP_WIDTH){1'b0}};
            rd_batch_mant_h1 <= {(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH){1'b0}};
        end else if (rd_en_h1) begin
            for (t1 = 0; t1 < TOKEN_BATCH; t1 = t1 + 1) begin
                rd_batch_exp_h1[t1*EXP_WIDTH +: EXP_WIDTH] <= Q_exp_mem[1][t1];
                for (d1 = 0; d1 < HEAD_DIM; d1 = d1 + 1) begin
                    rd_batch_mant_h1[(t1*HEAD_DIM + d1)*DATA_WIDTH +: DATA_WIDTH] <= Q_mant_mem[1][t1][d1];
                end
            end
        end
    end

    //================================================================================
    // 读取逻辑 - Head 2
    //================================================================================
    integer t2, d2;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_batch_exp_h2  <= {(TOKEN_BATCH*EXP_WIDTH){1'b0}};
            rd_batch_mant_h2 <= {(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH){1'b0}};
        end else if (rd_en_h2) begin
            for (t2 = 0; t2 < TOKEN_BATCH; t2 = t2 + 1) begin
                rd_batch_exp_h2[t2*EXP_WIDTH +: EXP_WIDTH] <= Q_exp_mem[2][t2];
                for (d2 = 0; d2 < HEAD_DIM; d2 = d2 + 1) begin
                    rd_batch_mant_h2[(t2*HEAD_DIM + d2)*DATA_WIDTH +: DATA_WIDTH] <= Q_mant_mem[2][t2][d2];
                end
            end
        end
    end

    //================================================================================
    // 读取逻辑 - Head 3
    //================================================================================
    integer t3, d3;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_batch_exp_h3  <= {(TOKEN_BATCH*EXP_WIDTH){1'b0}};
            rd_batch_mant_h3 <= {(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH){1'b0}};
        end else if (rd_en_h3) begin
            for (t3 = 0; t3 < TOKEN_BATCH; t3 = t3 + 1) begin
                rd_batch_exp_h3[t3*EXP_WIDTH +: EXP_WIDTH] <= Q_exp_mem[3][t3];
                for (d3 = 0; d3 < HEAD_DIM; d3 = d3 + 1) begin
                    rd_batch_mant_h3[(t3*HEAD_DIM + d3)*DATA_WIDTH +: DATA_WIDTH] <= Q_mant_mem[3][t3][d3];
                end
            end
        end
    end

endmodule