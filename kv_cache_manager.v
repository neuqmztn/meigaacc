`timescale 1ns / 1ps

//===================================================================================
// KV Cache Manager - 4-Bank并行架构
//
// 功能：
//   - 存储所有token的K和V矩阵
//   - K/V只在第一个batch时计算一次，后续复用
//   - 支持4个Head并行读取，每个Head独立bank
//
// 改进点（v2.0）：
//   - 4个Head的K读取接口（并行）
//   - 4个Head的V读取接口（并行）
//   - 每个Head可以读取不同的chunk_id
//   - 完全无冲突的并行架构
//
// 存储容量：
//   - K矩阵：4 heads × 641 tokens × 8 dim × 16 bits ≈ 40 KB
//   - V矩阵：4 heads × 641 tokens × 8 dim × 16 bits ≈ 40 KB
//   - 总计：80 KB
//
// 版本：v2.0 (并行架构)
// 日期：2024-11-15
//===================================================================================

module kv_cache_manager #(
    parameter NUM_HEADS      = 4,
    parameter TOTAL_TOKENS   = 641,
    parameter HEAD_DIM       = 8,
    parameter DATA_WIDTH     = 8,
    parameter EXP_WIDTH      = 8,
    parameter CHUNK_SIZE     = 32,
    parameter NUM_CHUNKS     = 21
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 写入接口
    //===========================================================================
    input  wire wr_en,
    input  wire wr_type,                    // 0=K, 1=V
    input  wire [1:0] wr_head,
    input  wire [9:0] wr_token,
    input  wire [EXP_WIDTH-1:0] wr_exp,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0] wr_mant_packed,
    
    //===========================================================================
    // Head 0 读取接口
    //===========================================================================
    // K读取
    input  wire k_rd_en_h0,
    input  wire [4:0] k_rd_chunk_h0,
    output reg  k_rd_valid_h0,
    output reg  [CHUNK_SIZE*EXP_WIDTH-1:0] k_chunk_exp_h0,
    output reg  [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] k_chunk_mant_h0,
    
    // V读取
    input  wire v_rd_en_h0,
    input  wire [4:0] v_rd_chunk_h0,
    output reg  v_rd_valid_h0,
    output reg  [CHUNK_SIZE*EXP_WIDTH-1:0] v_chunk_exp_h0,
    output reg  [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] v_chunk_mant_h0,
    
    //===========================================================================
    // Head 1 读取接口
    //===========================================================================
    input  wire k_rd_en_h1,
    input  wire [4:0] k_rd_chunk_h1,
    output reg  k_rd_valid_h1,
    output reg  [CHUNK_SIZE*EXP_WIDTH-1:0] k_chunk_exp_h1,
    output reg  [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] k_chunk_mant_h1,
    
    input  wire v_rd_en_h1,
    input  wire [4:0] v_rd_chunk_h1,
    output reg  v_rd_valid_h1,
    output reg  [CHUNK_SIZE*EXP_WIDTH-1:0] v_chunk_exp_h1,
    output reg  [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] v_chunk_mant_h1,
    
    //===========================================================================
    // Head 2 读取接口
    //===========================================================================
    input  wire k_rd_en_h2,
    input  wire [4:0] k_rd_chunk_h2,
    output reg  k_rd_valid_h2,
    output reg  [CHUNK_SIZE*EXP_WIDTH-1:0] k_chunk_exp_h2,
    output reg  [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] k_chunk_mant_h2,
    
    input  wire v_rd_en_h2,
    input  wire [4:0] v_rd_chunk_h2,
    output reg  v_rd_valid_h2,
    output reg  [CHUNK_SIZE*EXP_WIDTH-1:0] v_chunk_exp_h2,
    output reg  [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] v_chunk_mant_h2,
    
    //===========================================================================
    // Head 3 读取接口
    //===========================================================================
    input  wire k_rd_en_h3,
    input  wire [4:0] k_rd_chunk_h3,
    output reg  k_rd_valid_h3,
    output reg  [CHUNK_SIZE*EXP_WIDTH-1:0] k_chunk_exp_h3,
    output reg  [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] k_chunk_mant_h3,
    
    input  wire v_rd_en_h3,
    input  wire [4:0] v_rd_chunk_h3,
    output reg  v_rd_valid_h3,
    output reg  [CHUNK_SIZE*EXP_WIDTH-1:0] v_chunk_exp_h3,
    output reg  [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] v_chunk_mant_h3,
    
    //===========================================================================
    // 状态接口
    //===========================================================================
    output reg  kv_ready,
    output reg  [1:0] kv_status
);

//===================================================================================
// 存储阵列（Bank化设计）
// 添加综合属性以确保使用Block RAM而非寄存器
//===================================================================================

// K矩阵存储 - 使用Block RAM
(* ram_style = "block" *) reg [EXP_WIDTH-1:0] K_exp_mem [0:NUM_HEADS-1][0:TOTAL_TOKENS-1];
(* ram_style = "block" *) reg [DATA_WIDTH-1:0] K_mant_mem [0:NUM_HEADS-1][0:TOTAL_TOKENS-1][0:HEAD_DIM-1];

// V矩阵存储 - 使用Block RAM
(* ram_style = "block" *) reg [EXP_WIDTH-1:0] V_exp_mem [0:NUM_HEADS-1][0:TOTAL_TOKENS-1];
(* ram_style = "block" *) reg [DATA_WIDTH-1:0] V_mant_mem [0:NUM_HEADS-1][0:TOTAL_TOKENS-1][0:HEAD_DIM-1];

// 有效位标记
reg [TOTAL_TOKENS-1:0] K_valid [0:NUM_HEADS-1];
reg [TOTAL_TOKENS-1:0] V_valid [0:NUM_HEADS-1];

// 写入计数器
reg [10:0] K_write_count [0:NUM_HEADS-1];
reg [10:0] V_write_count [0:NUM_HEADS-1];

//===================================================================================
// 循环变量声明
//===================================================================================
integer i, j, k;

//===================================================================================
// ✅ FIX #2: Token索引计算 - 使用wire组合逻辑而非阻塞赋值
// 这样避免了在时序逻辑中使用阻塞赋值的问题
//===================================================================================
wire [31:0] token_idx_h0_k = k_rd_chunk_h0 * CHUNK_SIZE;
wire [31:0] token_idx_h0_v = v_rd_chunk_h0 * CHUNK_SIZE;
wire [31:0] token_idx_h1_k = k_rd_chunk_h1 * CHUNK_SIZE;
wire [31:0] token_idx_h1_v = v_rd_chunk_h1 * CHUNK_SIZE;
wire [31:0] token_idx_h2_k = k_rd_chunk_h2 * CHUNK_SIZE;
wire [31:0] token_idx_h2_v = v_rd_chunk_h2 * CHUNK_SIZE;
wire [31:0] token_idx_h3_k = k_rd_chunk_h3 * CHUNK_SIZE;
wire [31:0] token_idx_h3_v = v_rd_chunk_h3 * CHUNK_SIZE;

//===================================================================================
// 写入逻辑
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 复位所有存储器和计数器
        for (i = 0; i < NUM_HEADS; i = i + 1) begin
            K_valid[i] <= {TOTAL_TOKENS{1'b0}};
            V_valid[i] <= {TOTAL_TOKENS{1'b0}};
            K_write_count[i] <= 11'h0;
            V_write_count[i] <= 11'h0;
            
            for (j = 0; j < TOTAL_TOKENS; j = j + 1) begin
                K_exp_mem[i][j] <= {EXP_WIDTH{1'b0}};
                V_exp_mem[i][j] <= {EXP_WIDTH{1'b0}};
                for (k = 0; k < HEAD_DIM; k = k + 1) begin
                    K_mant_mem[i][j][k] <= {DATA_WIDTH{1'b0}};
                    V_mant_mem[i][j][k] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end else if (wr_en && wr_token < TOTAL_TOKENS) begin
        if (wr_type == 1'b0) begin
            // 写K矩阵
            K_exp_mem[wr_head][wr_token] <= wr_exp;
            for (i = 0; i < HEAD_DIM; i = i + 1) begin
                K_mant_mem[wr_head][wr_token][i] <= 
                    wr_mant_packed[i*DATA_WIDTH +: DATA_WIDTH];
            end
            if (!K_valid[wr_head][wr_token]) begin
                K_valid[wr_head][wr_token] <= 1'b1;
                K_write_count[wr_head] <= K_write_count[wr_head] + 1'b1;
            end
        end else begin
            // 写V矩阵
            V_exp_mem[wr_head][wr_token] <= wr_exp;
            for (i = 0; i < HEAD_DIM; i = i + 1) begin
                V_mant_mem[wr_head][wr_token][i] <= 
                    wr_mant_packed[i*DATA_WIDTH +: DATA_WIDTH];
            end
            if (!V_valid[wr_head][wr_token]) begin
                V_valid[wr_head][wr_token] <= 1'b1;
                V_write_count[wr_head] <= V_write_count[wr_head] + 1'b1;
            end
        end
    end
end

//===================================================================================
// ✅ FIX #1: Head 0 - K读取 (修复Valid信号时序)
// 移除else分支，让valid信号持续保持而非单周期脉冲
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        k_rd_valid_h0 <= 1'b0;
        k_chunk_exp_h0 <= {(CHUNK_SIZE*EXP_WIDTH){1'b0}};
        k_chunk_mant_h0 <= {(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH){1'b0}};
    end else if (k_rd_en_h0) begin
        k_rd_valid_h0 <= 1'b1;
        
        // 读取32个token的K指数和尾数
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin
            if (token_idx_h0_k + i < TOTAL_TOKENS) begin
                k_chunk_exp_h0[i*EXP_WIDTH +: EXP_WIDTH] <= K_exp_mem[0][token_idx_h0_k + i];
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    k_chunk_mant_h0[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= 
                        K_mant_mem[0][token_idx_h0_k + i][j];
                end
            end else begin
                // 边界外填充0
                k_chunk_exp_h0[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    k_chunk_mant_h0[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end
    // ✅ 关键修复：移除了else分支，valid保持为1直到下次读取或复位
end

//===================================================================================
// ✅ FIX #1: Head 0 - V读取 (修复Valid信号时序)
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        v_rd_valid_h0 <= 1'b0;
        v_chunk_exp_h0 <= {(CHUNK_SIZE*EXP_WIDTH){1'b0}};
        v_chunk_mant_h0 <= {(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH){1'b0}};
    end else if (v_rd_en_h0) begin
        v_rd_valid_h0 <= 1'b1;
        
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin
            if (token_idx_h0_v + i < TOTAL_TOKENS) begin
                v_chunk_exp_h0[i*EXP_WIDTH +: EXP_WIDTH] <= V_exp_mem[0][token_idx_h0_v + i];
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    v_chunk_mant_h0[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= 
                        V_mant_mem[0][token_idx_h0_v + i][j];
                end
            end else begin
                v_chunk_exp_h0[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    v_chunk_mant_h0[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end
    // ✅ 移除else分支
end

//===================================================================================
// ✅ Head 1 - K读取 (完整修复版)
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        k_rd_valid_h1 <= 1'b0;
        k_chunk_exp_h1 <= {(CHUNK_SIZE*EXP_WIDTH){1'b0}};
        k_chunk_mant_h1 <= {(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH){1'b0}};
    end else if (k_rd_en_h1) begin
        k_rd_valid_h1 <= 1'b1;
        
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin
            if (token_idx_h1_k + i < TOTAL_TOKENS) begin
                k_chunk_exp_h1[i*EXP_WIDTH +: EXP_WIDTH] <= K_exp_mem[1][token_idx_h1_k + i];
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    k_chunk_mant_h1[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= 
                        K_mant_mem[1][token_idx_h1_k + i][j];
                end
            end else begin
                k_chunk_exp_h1[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    k_chunk_mant_h1[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end
end

//===================================================================================
// ✅ Head 1 - V读取 (完整修复版)
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        v_rd_valid_h1 <= 1'b0;
        v_chunk_exp_h1 <= {(CHUNK_SIZE*EXP_WIDTH){1'b0}};
        v_chunk_mant_h1 <= {(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH){1'b0}};
    end else if (v_rd_en_h1) begin
        v_rd_valid_h1 <= 1'b1;
        
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin
            if (token_idx_h1_v + i < TOTAL_TOKENS) begin
                v_chunk_exp_h1[i*EXP_WIDTH +: EXP_WIDTH] <= V_exp_mem[1][token_idx_h1_v + i];
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    v_chunk_mant_h1[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= 
                        V_mant_mem[1][token_idx_h1_v + i][j];
                end
            end else begin
                v_chunk_exp_h1[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    v_chunk_mant_h1[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end
end

//===================================================================================
// ✅ Head 2 - K读取 (完整修复版)
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        k_rd_valid_h2 <= 1'b0;
        k_chunk_exp_h2 <= {(CHUNK_SIZE*EXP_WIDTH){1'b0}};
        k_chunk_mant_h2 <= {(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH){1'b0}};
    end else if (k_rd_en_h2) begin
        k_rd_valid_h2 <= 1'b1;
        
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin
            if (token_idx_h2_k + i < TOTAL_TOKENS) begin
                k_chunk_exp_h2[i*EXP_WIDTH +: EXP_WIDTH] <= K_exp_mem[2][token_idx_h2_k + i];
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    k_chunk_mant_h2[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= 
                        K_mant_mem[2][token_idx_h2_k + i][j];
                end
            end else begin
                k_chunk_exp_h2[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    k_chunk_mant_h2[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end
end

//===================================================================================
// ✅ Head 2 - V读取 (完整修复版)
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        v_rd_valid_h2 <= 1'b0;
        v_chunk_exp_h2 <= {(CHUNK_SIZE*EXP_WIDTH){1'b0}};
        v_chunk_mant_h2 <= {(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH){1'b0}};
    end else if (v_rd_en_h2) begin
        v_rd_valid_h2 <= 1'b1;
        
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin
            if (token_idx_h2_v + i < TOTAL_TOKENS) begin
                v_chunk_exp_h2[i*EXP_WIDTH +: EXP_WIDTH] <= V_exp_mem[2][token_idx_h2_v + i];
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    v_chunk_mant_h2[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= 
                        V_mant_mem[2][token_idx_h2_v + i][j];
                end
            end else begin
                v_chunk_exp_h2[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    v_chunk_mant_h2[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end
end

//===================================================================================
// ✅ Head 3 - K读取 (完整修复版)
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        k_rd_valid_h3 <= 1'b0;
        k_chunk_exp_h3 <= {(CHUNK_SIZE*EXP_WIDTH){1'b0}};
        k_chunk_mant_h3 <= {(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH){1'b0}};
    end else if (k_rd_en_h3) begin
        k_rd_valid_h3 <= 1'b1;
        
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin
            if (token_idx_h3_k + i < TOTAL_TOKENS) begin
                k_chunk_exp_h3[i*EXP_WIDTH +: EXP_WIDTH] <= K_exp_mem[3][token_idx_h3_k + i];
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    k_chunk_mant_h3[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= 
                        K_mant_mem[3][token_idx_h3_k + i][j];
                end
            end else begin
                k_chunk_exp_h3[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    k_chunk_mant_h3[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end
end

//===================================================================================
// ✅ Head 3 - V读取 (完整修复版)
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        v_rd_valid_h3 <= 1'b0;
        v_chunk_exp_h3 <= {(CHUNK_SIZE*EXP_WIDTH){1'b0}};
        v_chunk_mant_h3 <= {(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH){1'b0}};
    end else if (v_rd_en_h3) begin
        v_rd_valid_h3 <= 1'b1;
        
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin
            if (token_idx_h3_v + i < TOTAL_TOKENS) begin
                v_chunk_exp_h3[i*EXP_WIDTH +: EXP_WIDTH] <= V_exp_mem[3][token_idx_h3_v + i];
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    v_chunk_mant_h3[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= 
                        V_mant_mem[3][token_idx_h3_v + i][j];
                end
            end else begin
                v_chunk_exp_h3[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    v_chunk_mant_h3[(i*HEAD_DIM + j)*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end
end

//===================================================================================
// 状态管理
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        kv_ready <= 1'b0;
        kv_status <= 2'b00;
    end else begin: status_check
        reg all_k_ready, all_v_ready;
        
        all_k_ready = 1'b1;
        all_v_ready = 1'b1;
        
        for (i = 0; i < NUM_HEADS; i = i + 1) begin
            if (K_write_count[i] < TOTAL_TOKENS) all_k_ready = 1'b0;
            if (V_write_count[i] < TOTAL_TOKENS) all_v_ready = 1'b0;
        end
        
        kv_status <= {all_v_ready, all_k_ready};
        kv_ready <= all_k_ready && all_v_ready;
    end
end

//===================================================================================
// 调试信息
//===================================================================================

initial begin
    $display("========================================");
    $display("KV Cache Manager - Complete Fix v2.1");
    $display("========================================");
    $display("Configuration:");
    $display("  Heads: %0d", NUM_HEADS);
    $display("  Total tokens: %0d", TOTAL_TOKENS);
    $display("  Chunk size: %0d", CHUNK_SIZE);
    $display("  Chunks: %0d", NUM_CHUNKS);
    $display("----------------------------------------");
    $display("Fixes applied:");
    $display("  ✅ Valid signal timing fixed");
    $display("  ✅ token_idx blocking assignment fixed");
    $display("  ✅ Block RAM synthesis attributes added");
    $display("----------------------------------------");
    $display("Read interfaces:");
    $display("  4 parallel K read ports");
    $display("  4 parallel V read ports");
    $display("  Total: 8 independent read ports");
    $display("========================================");
end
endmodule