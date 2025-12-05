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
    
    //===========================================================================
    // 写接口
    //===========================================================================
    input  wire wr_en,
    input  wire [1:0] wr_head,
    input  wire [4:0] wr_token,
    input  wire [EXP_WIDTH-1:0] wr_exp,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0] wr_mant_packed,
    
    //===========================================================================
    // Head 0 读取接口（batch读取）
    //===========================================================================
    input  wire rd_en_h0,
    output reg  [(TOKEN_BATCH*EXP_WIDTH)-1:0] rd_batch_exp_h0,
    output reg  [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] rd_batch_mant_h0,
    
    //===========================================================================
    // Head 1 读取接口（batch读取）
    //===========================================================================
    input  wire rd_en_h1,
    output reg  [(TOKEN_BATCH*EXP_WIDTH)-1:0] rd_batch_exp_h1,
    output reg  [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] rd_batch_mant_h1,
    
    //===========================================================================
    // Head 2 读取接口（batch读取）
    //===========================================================================
    input  wire rd_en_h2,
    output reg  [(TOKEN_BATCH*EXP_WIDTH)-1:0] rd_batch_exp_h2,
    output reg  [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] rd_batch_mant_h2,
    
    //===========================================================================
    // Head 3 读取接口（batch读取）
    //===========================================================================
    input  wire rd_en_h3,
    output reg  [(TOKEN_BATCH*EXP_WIDTH)-1:0] rd_batch_exp_h3,
    output reg  [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] rd_batch_mant_h3
);

    //================================================================================
    // 存储阵列定义
    //================================================================================
    // 注意：此处移除了 (* ram_style *) 属性。
    // 因为下游逻辑需要同时读取所有地址，综合器会自动将其推断为 Flip-Flops (寄存器)。
    // 这种实现方式虽然消耗寄存器，但是唯一能满足单周期全Batch带宽的方法。
    
    reg [EXP_WIDTH-1:0] Q_exp_mem [0:NUM_HEADS-1][0:TOKEN_BATCH-1];
    reg [DATA_WIDTH-1:0] Q_mant_mem [0:NUM_HEADS-1][0:TOKEN_BATCH-1][0:HEAD_DIM-1];

    integer i;

    //================================================================================
    // 写入逻辑
    //================================================================================
    
    always @(posedge clk) begin
        if (wr_en && wr_token < TOKEN_BATCH) begin
            // 写入指数
            Q_exp_mem[wr_head][wr_token] <= wr_exp;
            
            // 写入尾数
            for (i = 0; i < HEAD_DIM; i = i + 1) begin
                Q_mant_mem[wr_head][wr_token][i] <= 
                    wr_mant_packed[i*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end

    //================================================================================
    // 读取逻辑 - Head 0
    //================================================================================
    integer t0, d0;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位输出端口
            rd_batch_exp_h0  <= {(TOKEN_BATCH*EXP_WIDTH){1'b0}};
            rd_batch_mant_h0 <= {(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH){1'b0}};
        end else if (rd_en_h0) begin
            // 并行读取所有 Token
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