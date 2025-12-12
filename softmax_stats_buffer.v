`timescale 1ns / 1ps

module softmax_stats_buffer #(
    parameter TOKEN_BATCH    = 32,
    parameter SCORE_WIDTH    = 16,
    parameter ACCUM_WIDTH    = 24
)(
    input  wire clk,
    input  wire rst_n,
    
    // ========== 最大值写入接口 ==========
    input  wire max_wr_en,
    input  wire [4:0] max_wr_row,
    input  wire signed [SCORE_WIDTH-1:0] max_wr_value,
    
    // ========== 最大值读取接口（零延迟）==========
    input  wire [4:0] max_rd_row,
    output wire signed [SCORE_WIDTH-1:0] max_rd_value,
    
    // ========== 累加和写入接口 ==========
    input  wire sum_wr_en,
    input  wire [4:0] sum_wr_row,
    input  wire signed [ACCUM_WIDTH-1:0] sum_wr_value,
    
    // ========== 累加和读取接口（零延迟）==========
    input  wire [4:0] sum_rd_row,
    output wire signed [ACCUM_WIDTH-1:0] sum_rd_value
);

    //================================================================================
    // 存储阵列 (Distributed RAM / LUTRAM)
    //================================================================================
    (* ram_style = "distributed" *) reg signed [SCORE_WIDTH-1:0] max_score_mem [0:TOKEN_BATCH-1];
    (* ram_style = "distributed" *) reg signed [ACCUM_WIDTH-1:0] sum_exp_mem [0:TOKEN_BATCH-1];

    //================================================================================
    // 写入逻辑 (同步写)
    //================================================================================
    // 注意：移除了 if (!rst_n) 的复位分支
    
    always @(posedge clk) begin
        if (max_wr_en) begin
            max_score_mem[max_wr_row] <= max_wr_value;
        end
    end

    always @(posedge clk) begin
        if (sum_wr_en) begin
            sum_exp_mem[sum_wr_row] <= sum_wr_value;
        end
    end

    //================================================================================
    // 读取逻辑 (组合/异步读) - 保持零延迟特性
    //================================================================================
    
    assign max_rd_value = max_score_mem[max_rd_row];
    assign sum_rd_value = sum_exp_mem[sum_rd_row];

endmodule
