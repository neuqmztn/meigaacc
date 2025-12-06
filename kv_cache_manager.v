`timescale 1ns / 1ps

module kv_cache_manager #(
    parameter NUM_HEADS      = 4,
    parameter TOTAL_TOKENS   = 640, // 实际 Token 数量
    parameter HEAD_DIM       = 8,
    parameter DATA_WIDTH     = 8,
    parameter EXP_WIDTH      = 8,
    parameter CHUNK_SIZE     = 32,  // 并行度
    parameter NUM_CHUNKS     = 20   // ceil(641/32)
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 写入接口 (单点随机写入)
    //===========================================================================
    input  wire                       wr_en,
    input  wire                       wr_type,        // 0=K, 1=V
    input  wire [1:0]                 wr_head,
    input  wire [9:0]                 wr_token,       // 0~1023
    input  wire [EXP_WIDTH-1:0]       wr_exp,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0] wr_mant_packed,
    
    //===========================================================================
    // Head 0 读取接口 (并行读取整个 Chunk)
    //===========================================================================
    input  wire                       k_rd_en_h0,
    input  wire [4:0]                 k_rd_chunk_h0,
    output wire                       k_rd_valid_h0,
    output wire [CHUNK_SIZE*EXP_WIDTH-1:0] k_chunk_exp_h0,
    output wire [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] k_chunk_mant_h0,
    
    input  wire                       v_rd_en_h0,
    input  wire [4:0]                 v_rd_chunk_h0,
    output wire                       v_rd_valid_h0,
    output wire [CHUNK_SIZE*EXP_WIDTH-1:0] v_chunk_exp_h0,
    output wire [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] v_chunk_mant_h0,
    
    //===========================================================================
    // Head 1 读取接口
    //===========================================================================
    input  wire                       k_rd_en_h1,
    input  wire [4:0]                 k_rd_chunk_h1,
    output wire                       k_rd_valid_h1,
    output wire [CHUNK_SIZE*EXP_WIDTH-1:0] k_chunk_exp_h1,
    output wire [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] k_chunk_mant_h1,
    
    input  wire                       v_rd_en_h1,
    input  wire [4:0]                 v_rd_chunk_h1,
    output wire                       v_rd_valid_h1,
    output wire [CHUNK_SIZE*EXP_WIDTH-1:0] v_chunk_exp_h1,
    output wire [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] v_chunk_mant_h1,
    
    //===========================================================================
    // Head 2 读取接口
    //===========================================================================
    input  wire                       k_rd_en_h2,
    input  wire [4:0]                 k_rd_chunk_h2,
    output wire                       k_rd_valid_h2,
    output wire [CHUNK_SIZE*EXP_WIDTH-1:0] k_chunk_exp_h2,
    output wire [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] k_chunk_mant_h2,
    
    input  wire                       v_rd_en_h2,
    input  wire [4:0]                 v_rd_chunk_h2,
    output wire                       v_rd_valid_h2,
    output wire [CHUNK_SIZE*EXP_WIDTH-1:0] v_chunk_exp_h2,
    output wire [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] v_chunk_mant_h2,
    
    //===========================================================================
    // Head 3 读取接口
    //===========================================================================
    input  wire                       k_rd_en_h3,
    input  wire [4:0]                 k_rd_chunk_h3,
    output wire                       k_rd_valid_h3,
    output wire [CHUNK_SIZE*EXP_WIDTH-1:0] k_chunk_exp_h3,
    output wire [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] k_chunk_mant_h3,
    
    input  wire                       v_rd_en_h3,
    input  wire [4:0]                 v_rd_chunk_h3,
    output wire                       v_rd_valid_h3,
    output wire [CHUNK_SIZE*EXP_WIDTH-1:0] v_chunk_exp_h3,
    output wire [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0] v_chunk_mant_h3,
    
    //===========================================================================
    // 状态接口
    //===========================================================================
    output reg                        kv_ready,
    output reg  [1:0]                 kv_status
);

    // 参数计算
    localparam BANK_DEPTH = (TOTAL_TOKENS + CHUNK_SIZE - 1) / CHUNK_SIZE; // ~21
    localparam MANT_WIDTH_PACKED = HEAD_DIM * DATA_WIDTH;

    //===========================================================================
    // 信号预处理：统一输入数组，方便 generate 循环
    //===========================================================================
    wire [4:0] k_rd_chunk [0:3];
    wire       k_rd_en    [0:3];
    wire [4:0] v_rd_chunk [0:3];
    wire       v_rd_en    [0:3];

    assign k_rd_chunk[0] = k_rd_chunk_h0; assign k_rd_en[0] = k_rd_en_h0;
    assign k_rd_chunk[1] = k_rd_chunk_h1; assign k_rd_en[1] = k_rd_en_h1;
    assign k_rd_chunk[2] = k_rd_chunk_h2; assign k_rd_en[2] = k_rd_en_h2;
    assign k_rd_chunk[3] = k_rd_chunk_h3; assign k_rd_en[3] = k_rd_en_h3;

    assign v_rd_chunk[0] = v_rd_chunk_h0; assign v_rd_en[0] = v_rd_en_h0;
    assign v_rd_chunk[1] = v_rd_chunk_h1; assign v_rd_en[1] = v_rd_en_h1;
    assign v_rd_chunk[2] = v_rd_chunk_h2; assign v_rd_en[2] = v_rd_en_h2;
    assign v_rd_chunk[3] = v_rd_chunk_h3; assign v_rd_en[3] = v_rd_en_h3;

    // 处理 Valid 信号 (对输入使能打一拍，匹配 RAM 读取延迟)
    reg [3:0] k_rd_valid_reg;
    reg [3:0] v_rd_valid_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k_rd_valid_reg <= 4'b0;
            v_rd_valid_reg <= 4'b0;
        end else begin
            k_rd_valid_reg <= {k_rd_en[3], k_rd_en[2], k_rd_en[1], k_rd_en[0]};
            v_rd_valid_reg <= {v_rd_en[3], v_rd_en[2], v_rd_en[1], v_rd_en[0]};
        end
    end

    assign k_rd_valid_h0 = k_rd_valid_reg[0]; assign v_rd_valid_h0 = v_rd_valid_reg[0];
    assign k_rd_valid_h1 = k_rd_valid_reg[1]; assign v_rd_valid_h1 = v_rd_valid_reg[1];
    assign k_rd_valid_h2 = k_rd_valid_reg[2]; assign v_rd_valid_h2 = v_rd_valid_reg[2];
    assign k_rd_valid_h3 = k_rd_valid_reg[3]; assign v_rd_valid_h3 = v_rd_valid_reg[3];

    //===========================================================================
    // 核心存储逻辑：Generate Loop
    //===========================================================================
    // 使用 wire 数组收集 generate 内部的输出
    wire [CHUNK_SIZE*EXP_WIDTH-1:0]        k_exp_out_wire  [0:3];
    wire [CHUNK_SIZE*MANT_WIDTH_PACKED-1:0] k_mant_out_wire [0:3];
    wire [CHUNK_SIZE*EXP_WIDTH-1:0]        v_exp_out_wire  [0:3];
    wire [CHUNK_SIZE*MANT_WIDTH_PACKED-1:0] v_mant_out_wire [0:3];

    genvar h, b;
    generate
        for (h = 0; h < NUM_HEADS; h = h + 1) begin : heads
            for (b = 0; b < CHUNK_SIZE; b = b + 1) begin : banks
                
                // --- 1. 定义存储器 ---
                // 使用 "distributed" 属性，因为深度浅且数量巨大 (256个)
                // xc7k70t 的 BRAM 数量不足以支撑 "block" 模式
                (* ram_style = "distributed" *) reg [EXP_WIDTH-1:0]         mem_k_exp  [0:BANK_DEPTH-1];
                (* ram_style = "distributed" *) reg [MANT_WIDTH_PACKED-1:0] mem_k_mant [0:BANK_DEPTH-1];
                (* ram_style = "distributed" *) reg [EXP_WIDTH-1:0]         mem_v_exp  [0:BANK_DEPTH-1];
                (* ram_style = "distributed" *) reg [MANT_WIDTH_PACKED-1:0] mem_v_mant [0:BANK_DEPTH-1];

                // --- 2. 写入解码 ---
                wire is_target_head = (wr_head == h[1:0]);
                wire is_target_bank = (wr_token[4:0] == b[4:0]); // Token % 32
                wire [4:0] wr_addr  = wr_token[9:5];             // Token / 32
                
                wire we_k = wr_en && (wr_type == 1'b0) && is_target_head && is_target_bank;
                wire we_v = wr_en && (wr_type == 1'b1) && is_target_head && is_target_bank;

                // 写入逻辑 (Distributed RAM 不需要复位)
                always @(posedge clk) begin
                    if (we_k) begin
                        mem_k_exp[wr_addr]  <= wr_exp;
                        mem_k_mant[wr_addr] <= wr_mant_packed;
                    end
                    if (we_v) begin
                        mem_v_exp[wr_addr]  <= wr_exp;
                        mem_v_mant[wr_addr] <= wr_mant_packed;
                    end
                end

                // --- 3. 读取逻辑 ---
                reg [EXP_WIDTH-1:0]         k_exp_raw;
                reg [MANT_WIDTH_PACKED-1:0] k_mant_raw;
                reg [EXP_WIDTH-1:0]         v_exp_raw;
                reg [MANT_WIDTH_PACKED-1:0] v_mant_raw;

                // 纯粹的 RAM 读取，不包含任何掩码逻辑，确保推断正确
                always @(posedge clk) begin
                    if (k_rd_en[h]) begin
                        k_exp_raw  <= mem_k_exp[k_rd_chunk[h]];
                        k_mant_raw <= mem_k_mant[k_rd_chunk[h]];
                    end
                    if (v_rd_en[h]) begin
                        v_exp_raw  <= mem_v_exp[v_rd_chunk[h]];
                        v_mant_raw <= mem_v_mant[v_rd_chunk[h]];
                    end
                end

                // --- 4. 越界 Masking 逻辑 ---
                // 计算当前 Bank 对应的全局 Token Index
                wire [31:0] k_global_idx = {22'b0, k_rd_chunk[h], 5'b0} + b;
                wire [31:0] v_global_idx = {22'b0, v_rd_chunk[h], 5'b0} + b;
                
                // 有效性判断打一拍，对齐 RAM 读取延迟
                reg k_valid_d1, v_valid_d1;
                always @(posedge clk) begin
                    k_valid_d1 <= (k_global_idx < TOTAL_TOKENS);
                    v_valid_d1 <= (v_global_idx < TOTAL_TOKENS);
                end

                // --- 5. 输出赋值 ---
                // 如果 Token Index 越界，强制输出 0
                assign k_exp_out_wire[h][b*EXP_WIDTH +: EXP_WIDTH] = 
                    k_valid_d1 ? k_exp_raw : {EXP_WIDTH{1'b0}};
                assign k_mant_out_wire[h][b*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] = 
                    k_valid_d1 ? k_mant_raw : {MANT_WIDTH_PACKED{1'b0}};
                
                assign v_exp_out_wire[h][b*EXP_WIDTH +: EXP_WIDTH] = 
                    v_valid_d1 ? v_exp_raw : {EXP_WIDTH{1'b0}};
                assign v_mant_out_wire[h][b*MANT_WIDTH_PACKED +: MANT_WIDTH_PACKED] = 
                    v_valid_d1 ? v_mant_raw : {MANT_WIDTH_PACKED{1'b0}};

            end // end banks loop
        end // end heads loop
    endgenerate

    //===========================================================================
    // 输出端口连线
    //===========================================================================
    assign k_chunk_exp_h0 = k_exp_out_wire[0]; assign k_chunk_mant_h0 = k_mant_out_wire[0];
    assign v_chunk_exp_h0 = v_exp_out_wire[0]; assign v_chunk_mant_h0 = v_mant_out_wire[0];
    
    assign k_chunk_exp_h1 = k_exp_out_wire[1]; assign k_chunk_mant_h1 = k_mant_out_wire[1];
    assign v_chunk_exp_h1 = v_exp_out_wire[1]; assign v_chunk_mant_h1 = v_mant_out_wire[1];
    
    assign k_chunk_exp_h2 = k_exp_out_wire[2]; assign k_chunk_mant_h2 = k_mant_out_wire[2];
    assign v_chunk_exp_h2 = v_exp_out_wire[2]; assign v_chunk_mant_h2 = v_mant_out_wire[2];
    
    assign k_chunk_exp_h3 = k_exp_out_wire[3]; assign k_chunk_mant_h3 = k_mant_out_wire[3];
    assign v_chunk_exp_h3 = v_exp_out_wire[3]; assign v_chunk_mant_h3 = v_mant_out_wire[3];

    //===========================================================================
    // 状态与计数器逻辑 (严谨版)
    //===========================================================================
    reg [10:0] K_write_count [0:NUM_HEADS-1];
    reg [10:0] V_write_count [0:NUM_HEADS-1];
    integer j;

    // 1. 写入计数
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < NUM_HEADS; j = j + 1) begin
                K_write_count[j] <= 11'h0;
                V_write_count[j] <= 11'h0;
            end
        end else begin
            if (wr_en && wr_token < TOTAL_TOKENS) begin
                if (wr_type == 1'b0) begin // Write K
                    if (wr_token == K_write_count[wr_head])
                        K_write_count[wr_head] <= K_write_count[wr_head] + 1'b1;
                end else begin // Write V
                    if (wr_token == V_write_count[wr_head])
                        V_write_count[wr_head] <= V_write_count[wr_head] + 1'b1;
                end
            end
        end
    end

    // 2. 完成检测 (组合逻辑遍历所有 Head)
    reg all_k_done, all_v_done;
    integer k;
    always @(*) begin
        all_k_done = 1'b1;
        all_v_done = 1'b1;
        for (k = 0; k < NUM_HEADS; k = k + 1) begin
            if (K_write_count[k] < TOTAL_TOKENS) all_k_done = 1'b0;
            if (V_write_count[k] < TOTAL_TOKENS) all_v_done = 1'b0;
        end
    end

    // 3. 状态输出
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kv_status <= 2'b00;
            kv_ready  <= 1'b0;
        end else begin
            kv_status <= {all_v_done, all_k_done};
            kv_ready  <= all_k_done && all_v_done;
        end
    end

endmodule