`timescale 1ns / 1ps

//================================================================================
// Sidenet Gated Buffer - 门控特征单BANK存储（优化版）
//
// 功能说明：
// 1. 存储Gate Engine输出的门控融合特征 a_i
// 2. 单BANK循环复用设计，节省50%资源（11.2KB → 5.6KB）
// 3. 数据格式：16-bit BFP (blockBFP: 1个共享指数 + 8个16-bit尾数)
// 4. 容量：641 tokens × (8 + 8×16) bits = 5.6 KB
//
// 门控公式：
//   Layer 0: a_0 = ẑ_0 (直接通过)
//   Layer i: a_i = (1-γ_i)·ẑ_i + γ_i·â_(i-1)
//            其中 â_(i-1) 从Layer Output Buffer读取
//
// 使用场景：
// Layer 0: Gate → 写Buffer → Adaptation读取
// Layer 1: Gate(融合â_0) → 写Buffer (覆盖Layer 0) → Adaptation读取
// Layer 2: Gate(融合â_1) → 写Buffer (覆盖Layer 1) → Adaptation读取
// Layer 3: Gate(融合â_2) → 写Buffer (覆盖Layer 2) → Adaptation读取
// Layer 4: Gate(融合â_3) → 写Buffer (覆盖Layer 3) → Expand读取
//
// 设计原理：
// - 这是Level 2临时存储，每层的门控数据用完即可丢弃
// - 层与层之间无需保留历史数据
// - 单BANK循环复用，大幅节省BRAM资源
//
// 接口说明：
// - 写接口：单端口，连接Gate Engine输出
// - 读接口：单端口，连接Sidenet Transformer (Adaptation)输入
// - 无需BANK切换控制（单BANK设计）
//
//================================================================================

module sidenet_gated_buffer #(
    parameter TOKEN_NUM      = 641,     // Token数量
    parameter COMPRESSED_DIM = 8,       // 压缩维度
    parameter DATA_WIDTH     = 16,      // 尾数位宽 (16-bit BFP)
    parameter EXP_WIDTH      = 8,       // 指数位宽
    parameter ADDR_WIDTH     = 10       // 地址位宽
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 写接口 - 连接Gate Engine输出 (a_i)
    //==========================================================================
    input  wire wr_en,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [EXP_WIDTH-1:0] wr_exp,                      // 共享指数
    input  wire [COMPRESSED_DIM*DATA_WIDTH-1:0] wr_mant,     // 8个16-bit尾数 = 128 bits
    output wire wr_ready,
    
    //==========================================================================
    // 读接口 - 连接Sidenet Transformer输入
    //==========================================================================
    input  wire rd_en,
    input  wire [ADDR_WIDTH-1:0] rd_addr,
    output wire [EXP_WIDTH-1:0] rd_exp,
    output wire [COMPRESSED_DIM*DATA_WIDTH-1:0] rd_mant,
    output wire rd_valid,
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [31:0] dbg_total_writes,
    output wire [31:0] dbg_total_reads,
    output wire [31:0] dbg_write_collisions    // 读写冲突次数
);

//================================================================================
// 内部信号
//================================================================================

// 单BANK存储 (5.6 KB)
reg [EXP_WIDTH-1:0]                  exp_mem  [0:TOKEN_NUM-1];
reg [COMPRESSED_DIM*DATA_WIDTH-1:0]  mant_mem [0:TOKEN_NUM-1];

// 读寄存器（流水线）
reg                                  rd_valid_reg;
reg [EXP_WIDTH-1:0]                  rd_exp_reg;
reg [COMPRESSED_DIM*DATA_WIDTH-1:0]  rd_mant_reg;

// 调试计数器
reg [31:0] write_count;
reg [31:0] read_count;
reg [31:0] collision_count;

// 内部地址有效信号
wire wr_addr_valid;
wire rd_addr_valid;

//================================================================================
// 地址有效性检查
//================================================================================

assign wr_addr_valid = (wr_addr < TOKEN_NUM);
assign rd_addr_valid = (rd_addr < TOKEN_NUM);

//================================================================================
// 初始化存储
//================================================================================

integer i;
initial begin
    // 初始化存储器
    for (i = 0; i < TOKEN_NUM; i = i + 1) begin
        exp_mem[i]  = {EXP_WIDTH{1'b0}};
        mant_mem[i] = {COMPRESSED_DIM*DATA_WIDTH{1'b0}};
    end
    
    // 初始化读寄存器
    rd_valid_reg = 1'b0;
    rd_exp_reg   = {EXP_WIDTH{1'b0}};
    rd_mant_reg  = {COMPRESSED_DIM*DATA_WIDTH{1'b0}};
    
    // 初始化调试计数器
    write_count     = 32'd0;
    read_count      = 32'd0;
    collision_count = 32'd0;
end

//================================================================================
// 写逻辑 - 单BANK写入
//================================================================================

always @(posedge clk) begin
    if (wr_en && wr_addr_valid) begin
        exp_mem[wr_addr]  <= wr_exp;
        mant_mem[wr_addr] <= wr_mant;
    end
end

assign wr_ready = 1'b1;  // 总是ready（单周期写入）

//================================================================================
// 读逻辑 - 单BANK读取（带流水线）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_valid_reg <= 1'b0;
        rd_exp_reg   <= {EXP_WIDTH{1'b0}};
        rd_mant_reg  <= {COMPRESSED_DIM*DATA_WIDTH{1'b0}};
    end else begin
        rd_valid_reg <= rd_en && rd_addr_valid;  // 流水线valid信号
        
        if (rd_en && rd_addr_valid) begin
            rd_exp_reg  <= exp_mem[rd_addr];
            rd_mant_reg <= mant_mem[rd_addr];
        end
    end
end

assign rd_exp   = rd_exp_reg;
assign rd_mant  = rd_mant_reg;
assign rd_valid = rd_valid_reg;

//================================================================================
// 调试计数器
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        write_count     <= 32'd0;
        read_count      <= 32'd0;
        collision_count <= 32'd0;
    end else begin
        // 写计数
        if (wr_en && wr_addr_valid) begin
            write_count <= write_count + 32'd1;
        end
        
        // 读计数
        if (rd_en && rd_addr_valid) begin
            read_count <= read_count + 32'd1;
        end
        
        // 冲突计数（同时读写同一地址）
        if (wr_en && rd_en && wr_addr_valid && rd_addr_valid && (wr_addr == rd_addr)) begin
            collision_count <= collision_count + 32'd1;
        end
    end
end

assign dbg_total_writes     = write_count;
assign dbg_total_reads      = read_count;
assign dbg_write_collisions = collision_count;


endmodule