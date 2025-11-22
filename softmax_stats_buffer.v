`timescale 1ns / 1ps

//================================================================================
// Softmax Stats Buffer - 单Head版本
//
// 功能：
//   存储单个head的32个query的统计量
//   每个single_head_engine实例化一个本模块
//
// 接口说明：
// - 写入：时序逻辑（1周期延迟）
// - 读取：组合逻辑（0周期延迟）
//
// 存储容量：
// - Max: 32行 × 16位 = 64 bytes
// - Sum: 32行 × 24位 = 96 bytes
// - 总计: 160 bytes (每个head)
//================================================================================

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
    
    // ========== 最大值读取接口（组合逻辑，零延迟）==========
    input  wire [4:0] max_rd_row,
    output wire signed [SCORE_WIDTH-1:0] max_rd_value,
    
    // ========== 累加和写入接口 ==========
    input  wire sum_wr_en,
    input  wire [4:0] sum_wr_row,
    input  wire signed [ACCUM_WIDTH-1:0] sum_wr_value,
    
    // ========== 累加和读取接口（组合逻辑，零延迟）==========
    input  wire [4:0] sum_rd_row,
    output wire signed [ACCUM_WIDTH-1:0] sum_rd_value
);

//================================================================================
// 存储阵列（只存储1个head的数据）
//================================================================================

reg signed [SCORE_WIDTH-1:0] max_score_mem [0:TOKEN_BATCH-1];
reg signed [ACCUM_WIDTH-1:0] sum_exp_mem [0:TOKEN_BATCH-1];

//================================================================================
// 循环变量
//================================================================================

integer i;

//================================================================================
// 最大值写入逻辑（时序）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 初始化为负无穷大
        for (i = 0; i < TOKEN_BATCH; i = i + 1) begin
            max_score_mem[i] <= -16'sh7FFF;  // 负无穷大
        end
    end else begin
        if (max_wr_en && max_wr_row < TOKEN_BATCH) begin
            max_score_mem[max_wr_row] <= max_wr_value;
            
            `ifdef DEBUG_STATS_BUFFER
            $display("[%0t] Stats Buffer: Write max[%0d] = %0d",
                     $time, max_wr_row, max_wr_value);
            `endif
        end
    end
end

//================================================================================
// 最大值读取逻辑（组合）- 零延迟
//================================================================================

assign max_rd_value = (max_rd_row < TOKEN_BATCH) ?
                      max_score_mem[max_rd_row] :
                      -16'sh7FFF;  // 越界返回负无穷

//================================================================================
// 累加和写入逻辑（时序）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 初始化为0
        for (i = 0; i < TOKEN_BATCH; i = i + 1) begin
            sum_exp_mem[i] <= {ACCUM_WIDTH{1'b0}};
        end
    end else begin
        if (sum_wr_en && sum_wr_row < TOKEN_BATCH) begin
            sum_exp_mem[sum_wr_row] <= sum_wr_value;
            
            `ifdef DEBUG_STATS_BUFFER
            $display("[%0t] Stats Buffer: Write sum[%0d] = %0d",
                     $time, sum_wr_row, sum_wr_value);
            `endif
        end
    end
end

//================================================================================
// 累加和读取逻辑（组合）- 零延迟
//================================================================================

assign sum_rd_value = (sum_rd_row < TOKEN_BATCH) ?
                      sum_exp_mem[sum_rd_row] :
                      {ACCUM_WIDTH{1'b0}};  // 越界返回0

//================================================================================
// 边界检查（仿真用）
//================================================================================

`ifdef SIMULATION
always @(posedge clk) begin
    if (max_wr_en && max_wr_row >= TOKEN_BATCH) begin
        $display("[ERROR] Stats Buffer: Max write out of bounds! row=%0d",
                 max_wr_row);
    end
    if (sum_wr_en && sum_wr_row >= TOKEN_BATCH) begin
        $display("[ERROR] Stats Buffer: Sum write out of bounds! row=%0d",
                 sum_wr_row);
    end
end
`endif

endmodule