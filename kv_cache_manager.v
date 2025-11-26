`timescale 1ns / 1ps

//===================================================================================
// KV Cache Manager - 32-Bank Architecture (Fixed v3 - BRAM Inference Optimized)
//
// 修复说明：
//   1. [关键修复] 将存储阵列(RAM)的写入逻辑剥离到独立的 always @(posedge clk) 块中。
//      - 原代码将RAM写入放在异步复位块(negedge rst_n)中，违反了BRAM硬件特性，导致推断失败。
//   2. 保持了计数器的异步复位逻辑。
//   3. 保留了 wr_token[9:5] 的位宽修复。
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
    input  wire [9:0] wr_token,             // 10-bit token index (0~1023)
    input  wire [EXP_WIDTH-1:0] wr_exp,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0] wr_mant_packed,
    
    //===========================================================================
    // Head 0 读取接口
    //===========================================================================
    input  wire k_rd_en_h0,
    input  wire [4:0] k_rd_chunk_h0,
    output reg  k_rd_valid_h0,
    output reg  [CHUNK_SIZE*EXP_WIDTH-1:0] k_chunk_exp_h0,
    output reg  [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] k_chunk_mant_h0,
    
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

    //===========================================================================
    // 参数计算
    //===========================================================================
    localparam BANK_DEPTH = (TOTAL_TOKENS + CHUNK_SIZE - 1) / CHUNK_SIZE; // 21
    localparam MANT_WIDTH_PACKED = HEAD_DIM * DATA_WIDTH;

    //===========================================================================
    // 存储阵列定义 (Banking)
    //===========================================================================
    // 维度: [Head][Bank 0-31][Depth 0-20]
    // 使用 (* ram_style = "block" *) 强制 Block RAM
    
    (* ram_style = "block" *) reg [EXP_WIDTH-1:0]         K_exp_mem  [0:NUM_HEADS-1][0:CHUNK_SIZE-1][0:BANK_DEPTH-1];
    (* ram_style = "block" *) reg [MANT_WIDTH_PACKED-1:0] K_mant_mem [0:NUM_HEADS-1][0:CHUNK_SIZE-1][0:BANK_DEPTH-1];
    
    (* ram_style = "block" *) reg [EXP_WIDTH-1:0]         V_exp_mem  [0:NUM_HEADS-1][0:CHUNK_SIZE-1][0:BANK_DEPTH-1];
    (* ram_style = "block" *) reg [MANT_WIDTH_PACKED-1:0] V_mant_mem [0:NUM_HEADS-1][0:CHUNK_SIZE-1][0:BANK_DEPTH-1];

    // 状态寄存器
    reg [10:0] K_write_count [0:NUM_HEADS-1];
    reg [10:0] V_write_count [0:NUM_HEADS-1];
    
    integer h;

    //===========================================================================
    // 写入地址逻辑
    //===========================================================================
    wire [4:0] wr_bank_sel  = wr_token[4:0]; // Token % 32
    wire [4:0] wr_bank_addr = wr_token[9:5]; // Token / 32

    //===========================================================================
    // 1. RAM 写入逻辑 (纯同步，无复位) -> 关键修复点
    //===========================================================================
    // 必须剥离 rst_n，否则综合器无法推断 BRAM
    always @(posedge clk) begin
        if (wr_en && wr_token < TOTAL_TOKENS) begin
            if (wr_type == 1'b0) begin
                // 写 K 矩阵
                K_exp_mem[wr_head][wr_bank_sel][wr_bank_addr]   <= wr_exp;
                K_mant_mem[wr_head][wr_bank_sel][wr_bank_addr]  <= wr_mant_packed;
            end else begin
                // 写 V 矩阵
                V_exp_mem[wr_head][wr_bank_sel][wr_bank_addr]   <= wr_exp;
                V_mant_mem[wr_head][wr_bank_sel][wr_bank_addr]  <= wr_mant_packed;
            end
        end
    end

    //===========================================================================
    // 2. 计数器逻辑 (带复位)
    //===========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (h = 0; h < NUM_HEADS; h = h + 1) begin
                K_write_count[h] <= 11'h0;
                V_write_count[h] <= 11'h0;
            end
        end else if (wr_en && wr_token < TOTAL_TOKENS) begin
            if (wr_type == 1'b0) begin
                if (wr_token == K_write_count[wr_head])
                    K_write_count[wr_head] <= K_write_count[wr_head] + 1'b1;
            end else begin
                if (wr_token == V_write_count[wr_head])
                    V_write_count[wr_head] <= V_write_count[wr_head] + 1'b1;
            end
        end
    end

    //===========================================================================
    // 读取逻辑生成 (Generate Loops for 32 Banks)
    //===========================================================================
    
    genvar i;
    
    //---------------------------------------------------------------------------
    // Head 0 读取逻辑
    //---------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) k_rd_valid_h0 <= 1'b0;
        else if (k_rd_en_h0) k_rd_valid_h0 <= 1'b1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_rd_valid_h0 <= 1'b0;
        else if (v_rd_en_h0) v_rd_valid_h0 <= 1'b1;
    end

    generate
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin : h0_banks
            wire [31:0] k_global_idx = {22'b0, k_rd_chunk_h0, 5'b0} + i;
            wire [31:0] v_global_idx = {22'b0, v_rd_chunk_h0, 5'b0} + i;
            
            always @(posedge clk) begin
                if (k_rd_en_h0) begin
                    if (k_global_idx < TOTAL_TOKENS) begin
                        k_chunk_exp_h0[i*EXP_WIDTH +: EXP_WIDTH] <= 
                            K_exp_mem[0][i][k_rd_chunk_h0];
                        k_chunk_mant_h0[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= 
                            K_mant_mem[0][i][k_rd_chunk_h0];
                    end else begin
                        k_chunk_exp_h0[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                        k_chunk_mant_h0[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= {MANT_WIDTH_PACKED{1'b0}};
                    end
                end
                if (v_rd_en_h0) begin
                    if (v_global_idx < TOTAL_TOKENS) begin
                        v_chunk_exp_h0[i*EXP_WIDTH +: EXP_WIDTH] <= 
                            V_exp_mem[0][i][v_rd_chunk_h0];
                        v_chunk_mant_h0[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= 
                            V_mant_mem[0][i][v_rd_chunk_h0];
                    end else begin
                        v_chunk_exp_h0[i*EXP_WIDTH +: EXP_WIDTH] <= {EXP_WIDTH{1'b0}};
                        v_chunk_mant_h0[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= {MANT_WIDTH_PACKED{1'b0}};
                    end
                end
            end
        end
    endgenerate

    //---------------------------------------------------------------------------
    // Head 1 读取逻辑
    //---------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) k_rd_valid_h1 <= 1'b0;
        else if (k_rd_en_h1) k_rd_valid_h1 <= 1'b1;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_rd_valid_h1 <= 1'b0;
        else if (v_rd_en_h1) v_rd_valid_h1 <= 1'b1;
    end

    generate
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin : h1_banks
            wire [31:0] k_global_idx = {22'b0, k_rd_chunk_h1, 5'b0} + i;
            wire [31:0] v_global_idx = {22'b0, v_rd_chunk_h1, 5'b0} + i;
            
            always @(posedge clk) begin
                if (k_rd_en_h1) begin
                    if (k_global_idx < TOTAL_TOKENS) begin
                        k_chunk_exp_h1[i*EXP_WIDTH +: EXP_WIDTH] <= K_exp_mem[1][i][k_rd_chunk_h1];
                        k_chunk_mant_h1[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= K_mant_mem[1][i][k_rd_chunk_h1];
                    end else begin
                        k_chunk_exp_h1[i*EXP_WIDTH +: EXP_WIDTH] <= 0;
                        k_chunk_mant_h1[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= 0;
                    end
                end
                if (v_rd_en_h1) begin
                    if (v_global_idx < TOTAL_TOKENS) begin
                        v_chunk_exp_h1[i*EXP_WIDTH +: EXP_WIDTH] <= V_exp_mem[1][i][v_rd_chunk_h1];
                        v_chunk_mant_h1[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= V_mant_mem[1][i][v_rd_chunk_h1];
                    end else begin
                        v_chunk_exp_h1[i*EXP_WIDTH +: EXP_WIDTH] <= 0;
                        v_chunk_mant_h1[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= 0;
                    end
                end
            end
        end
    endgenerate

    //---------------------------------------------------------------------------
    // Head 2 读取逻辑
    //---------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) k_rd_valid_h2 <= 1'b0;
        else if (k_rd_en_h2) k_rd_valid_h2 <= 1'b1;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_rd_valid_h2 <= 1'b0;
        else if (v_rd_en_h2) v_rd_valid_h2 <= 1'b1;
    end

    generate
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin : h2_banks
            wire [31:0] k_global_idx = {22'b0, k_rd_chunk_h2, 5'b0} + i;
            wire [31:0] v_global_idx = {22'b0, v_rd_chunk_h2, 5'b0} + i;
            
            always @(posedge clk) begin
                if (k_rd_en_h2) begin
                    if (k_global_idx < TOTAL_TOKENS) begin
                        k_chunk_exp_h2[i*EXP_WIDTH +: EXP_WIDTH] <= K_exp_mem[2][i][k_rd_chunk_h2];
                        k_chunk_mant_h2[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= K_mant_mem[2][i][k_rd_chunk_h2];
                    end else begin
                        k_chunk_exp_h2[i*EXP_WIDTH +: EXP_WIDTH] <= 0;
                        k_chunk_mant_h2[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= 0;
                    end
                end
                if (v_rd_en_h2) begin
                    if (v_global_idx < TOTAL_TOKENS) begin
                        v_chunk_exp_h2[i*EXP_WIDTH +: EXP_WIDTH] <= V_exp_mem[2][i][v_rd_chunk_h2];
                        v_chunk_mant_h2[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= V_mant_mem[2][i][v_rd_chunk_h2];
                    end else begin
                        v_chunk_exp_h2[i*EXP_WIDTH +: EXP_WIDTH] <= 0;
                        v_chunk_mant_h2[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= 0;
                    end
                end
            end
        end
    endgenerate

    //---------------------------------------------------------------------------
    // Head 3 读取逻辑
    //---------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) k_rd_valid_h3 <= 1'b0;
        else if (k_rd_en_h3) k_rd_valid_h3 <= 1'b1;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_rd_valid_h3 <= 1'b0;
        else if (v_rd_en_h3) v_rd_valid_h3 <= 1'b1;
    end

    generate
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin : h3_banks
            wire [31:0] k_global_idx = {22'b0, k_rd_chunk_h3, 5'b0} + i;
            wire [31:0] v_global_idx = {22'b0, v_rd_chunk_h3, 5'b0} + i;
            
            always @(posedge clk) begin
                if (k_rd_en_h3) begin
                    if (k_global_idx < TOTAL_TOKENS) begin
                        k_chunk_exp_h3[i*EXP_WIDTH +: EXP_WIDTH] <= K_exp_mem[3][i][k_rd_chunk_h3];
                        k_chunk_mant_h3[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= K_mant_mem[3][i][k_rd_chunk_h3];
                    end else begin
                        k_chunk_exp_h3[i*EXP_WIDTH +: EXP_WIDTH] <= 0;
                        k_chunk_mant_h3[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= 0;
                    end
                end
                if (v_rd_en_h3) begin
                    if (v_global_idx < TOTAL_TOKENS) begin
                        v_chunk_exp_h3[i*EXP_WIDTH +: EXP_WIDTH] <= V_exp_mem[3][i][v_rd_chunk_h3];
                        v_chunk_mant_h3[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= V_mant_mem[3][i][v_rd_chunk_h3];
                    end else begin
                        v_chunk_exp_h3[i*EXP_WIDTH +: EXP_WIDTH] <= 0;
                        v_chunk_mant_h3[i*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] <= 0;
                    end
                end
            end
        end
    endgenerate

    //===========================================================================
    // 状态管理
    //===========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kv_ready <= 1'b0;
            kv_status <= 2'b00;
        end else begin:t
            reg all_k_ready, all_v_ready;
            all_k_ready = 1'b1;
            all_v_ready = 1'b1;
            
            for (h = 0; h < NUM_HEADS; h = h + 1) begin
                if (K_write_count[h] < TOTAL_TOKENS) all_k_ready = 1'b0;
                if (V_write_count[h] < TOTAL_TOKENS) all_v_ready = 1'b0;
            end
            
            kv_status <= {all_v_ready, all_k_ready};
            kv_ready <= all_k_ready && all_v_ready;
        end
    end

endmodule