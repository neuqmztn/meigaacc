`timescale 1ns / 1ps

//================================================================================
// Q Matrix Storage - Q矩阵存储（4-Bank并行架构）
//
// 功能说明:
//   - 存储当前batch的Q矩阵: 4头 × 32 tokens × 8维
//   - 支持4个Head并行读取，每个Head读取自己的bank
//   - 每次读取输出整个batch（32个token）
//
// 架构特点:
//   - Bank化设计：每个head独立bank，完全无冲突
//   - 批量读取：一次性输出32个token的完整数据
//   - 单周期延迟：组合逻辑读取或1-cycle寄存输出
//
// 存储容量:
//   - 指数: 4 heads × 32 tokens × 8 bits = 128 bytes
//   - 尾数: 4 heads × 32 tokens × 8 dims × 8 bits = 1024 bytes
//   - 总计: 1.15 KB
//
// 版本: v2.0 (并行架构)
// 日期: 2024-11-15
//================================================================================

module q_matrix_storage #(
    parameter NUM_HEADS      = 4,        // 注意力头数
    parameter TOKEN_BATCH    = 32,       // 批大小
    parameter HEAD_DIM       = 8,        // 每头维度
    parameter DATA_WIDTH     = 8,        // 尾数位宽
    parameter EXP_WIDTH      = 8         // 指数位宽
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 写接口（来自QKV计算引擎）
    //===========================================================================
    input  wire wr_en,
    input  wire [1:0] wr_head,           // 写入哪个head (0~3)
    input  wire [4:0] wr_token,          // 写入哪个token (0~31)
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
// 存储阵列（Bank化设计）
//================================================================================

// 指数存储: Q_exp_mem[head][token]
reg [EXP_WIDTH-1:0] Q_exp_mem [0:NUM_HEADS-1][0:TOKEN_BATCH-1];

// 尾数存储: Q_mant_mem[head][token][dim]
reg [DATA_WIDTH-1:0] Q_mant_mem [0:NUM_HEADS-1][0:TOKEN_BATCH-1][0:HEAD_DIM-1];

//================================================================================
// 循环变量
//================================================================================
integer i, j, k, t, d;

//================================================================================
// 写入逻辑（逐token写入）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        //------------------------------------------------------------------------
        // 复位：清空所有存储
        //------------------------------------------------------------------------
        for (i = 0; i < NUM_HEADS; i = i + 1) begin
            for (j = 0; j < TOKEN_BATCH; j = j + 1) begin
                Q_exp_mem[i][j] <= {EXP_WIDTH{1'b0}};
                for (k = 0; k < HEAD_DIM; k = k + 1) begin
                    Q_mant_mem[i][j][k] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
        
    end else if (wr_en && wr_token < TOKEN_BATCH) begin
        //------------------------------------------------------------------------
        // 写入逻辑：写入指定head的指定token
        //------------------------------------------------------------------------
        // 1. 写入指数
        Q_exp_mem[wr_head][wr_token] <= wr_exp;
        
        // 2. 拆分并写入8个维度的尾数
        for (i = 0; i < HEAD_DIM; i = i + 1) begin
            Q_mant_mem[wr_head][wr_token][i] <= 
                wr_mant_packed[i*DATA_WIDTH +: DATA_WIDTH];
        end
        
        $display("[%0t] Q_Storage WR: head=%0d token=%0d exp=%0d",
                 $time, wr_head, wr_token, wr_exp);
    end
end

//================================================================================
// 读取逻辑 - Head 0（批量读取32个token）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_batch_exp_h0 <= {(TOKEN_BATCH*EXP_WIDTH){1'b0}};
        rd_batch_mant_h0 <= {(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH){1'b0}};
        
    end else if (rd_en_h0) begin
        //------------------------------------------------------------------------
        // 读取Head 0的所有32个token
        //------------------------------------------------------------------------
        for (t = 0; t < TOKEN_BATCH; t = t + 1) begin
            // 打包指数
            rd_batch_exp_h0[t*EXP_WIDTH +: EXP_WIDTH] <= Q_exp_mem[0][t];
            
            // 打包尾数（8个维度）
            for (d = 0; d < HEAD_DIM; d = d + 1) begin
                rd_batch_mant_h0[(t*HEAD_DIM + d)*DATA_WIDTH +: DATA_WIDTH] <= 
                    Q_mant_mem[0][t][d];
            end
        end
        
        $display("[%0t] Q_Storage RD: head=0 reading batch (32 tokens)",
                 $time);
    end
end

//================================================================================
// 读取逻辑 - Head 1
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_batch_exp_h1 <= {(TOKEN_BATCH*EXP_WIDTH){1'b0}};
        rd_batch_mant_h1 <= {(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH){1'b0}};
        
    end else if (rd_en_h1) begin
        for (t = 0; t < TOKEN_BATCH; t = t + 1) begin
            rd_batch_exp_h1[t*EXP_WIDTH +: EXP_WIDTH] <= Q_exp_mem[1][t];
            
            for (d = 0; d < HEAD_DIM; d = d + 1) begin
                rd_batch_mant_h1[(t*HEAD_DIM + d)*DATA_WIDTH +: DATA_WIDTH] <= 
                    Q_mant_mem[1][t][d];
            end
        end
        
        $display("[%0t] Q_Storage RD: head=1 reading batch (32 tokens)",
                 $time);
    end
end

//================================================================================
// 读取逻辑 - Head 2
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_batch_exp_h2 <= {(TOKEN_BATCH*EXP_WIDTH){1'b0}};
        rd_batch_mant_h2 <= {(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH){1'b0}};
        
    end else if (rd_en_h2) begin
        for (t = 0; t < TOKEN_BATCH; t = t + 1) begin
            rd_batch_exp_h2[t*EXP_WIDTH +: EXP_WIDTH] <= Q_exp_mem[2][t];
            
            for (d = 0; d < HEAD_DIM; d = d + 1) begin
                rd_batch_mant_h2[(t*HEAD_DIM + d)*DATA_WIDTH +: DATA_WIDTH] <= 
                    Q_mant_mem[2][t][d];
            end
        end
        
        $display("[%0t] Q_Storage RD: head=2 reading batch (32 tokens)",
                 $time);
    end
end

//================================================================================
// 读取逻辑 - Head 3
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_batch_exp_h3 <= {(TOKEN_BATCH*EXP_WIDTH){1'b0}};
        rd_batch_mant_h3 <= {(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH){1'b0}};
        
    end else if (rd_en_h3) begin
        for (t = 0; t < TOKEN_BATCH; t = t + 1) begin
            rd_batch_exp_h3[t*EXP_WIDTH +: EXP_WIDTH] <= Q_exp_mem[3][t];
            
            for (d = 0; d < HEAD_DIM; d = d + 1) begin
                rd_batch_mant_h3[(t*HEAD_DIM + d)*DATA_WIDTH +: DATA_WIDTH] <= 
                    Q_mant_mem[3][t][d];
            end
        end
        
        $display("[%0t] Q_Storage RD: head=3 reading batch (32 tokens)",
                 $time);
    end
end

//================================================================================
// 初始化和调试信息
//================================================================================

initial begin
    $display("========================================");
    $display("Q Matrix Storage - 4-Bank Parallel");
    $display("========================================");
    $display("Configuration:");
    $display("  Heads: %0d", NUM_HEADS);
    $display("  Tokens per batch: %0d", TOKEN_BATCH);
    $display("  Head dimension: %0d", HEAD_DIM);
    $display("  Data width: %0d bits", DATA_WIDTH);
    $display("  Exp width: %0d bits", EXP_WIDTH);
    $display("----------------------------------------");
    $display("Bank organization:");
    $display("  Bank 0 (Head 0): 32 tokens × 8 dim");
    $display("  Bank 1 (Head 1): 32 tokens × 8 dim");
    $display("  Bank 2 (Head 2): 32 tokens × 8 dim");
    $display("  Bank 3 (Head 3): 32 tokens × 8 dim");
    $display("----------------------------------------");
    $display("Read interface:");
    $display("  4 parallel read ports");
    $display("  Each outputs 32 tokens in one cycle");
    $display("========================================");
end

endmodule