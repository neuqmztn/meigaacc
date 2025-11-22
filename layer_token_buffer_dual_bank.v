`timescale 1ns / 1ps

//================================================================================
// 层间Token双BANK存储模块 - 乒乓操作
//
// 功能说明：
// 1. 双BANK设计（BANK A 和 BANK B），支持乒乓操作
// 2. 每个BANK双端口读：Port A供Backbone，Port B供Sidenet
// 3. 单端口写：每层处理完后写入另一个BANK
// 4. 支持4层Transformer网络的层间数据传递
//
// 数据格式：8-bit BFP
// - 每token共享指数：8 bits
// - 每维度尾数：8 bits × 32 = 256 bits
// - 每token总计：264 bits = 33 bytes
// - 每BANK容量：641 tokens × 33 bytes ≈ 21 KB
// - 双BANK总计：42 KB
//
// 使用场景：
// Layer 0: 从BANK A读 → 处理 → 写BANK B
// Layer 1: 从BANK B读 → 处理 → 写BANK A (乒)
// Layer 2: 从BANK A读 → 处理 → 写BANK B (乓)
// Layer 3: 从BANK B读 → 处理 → 写BANK A (乒)
//================================================================================

module layer_token_buffer_dual_bank #(
    parameter TOKEN_NUM   = 641,
    parameter DIM         = 32,
    parameter EXP_WIDTH   = 8,
    parameter MANT_WIDTH  = 8,
    parameter ADDR_WIDTH  = 10
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 控制信号
    //==========================================================================
    input  wire bank_swap,                    // 层切换时脉冲，翻转BANK选择
    output wire current_read_bank,            // 当前读哪个BANK (0=A, 1=B)
    output wire current_write_bank,           // 当前写哪个BANK (0=A, 1=B)
    
    //==========================================================================
    // Backbone读接口 (Port A) - 连接到Attention/FFN/Residual等模块
    //==========================================================================
    input  wire backbone_rd_en,
    input  wire [ADDR_WIDTH-1:0] backbone_rd_addr,
    output wire [EXP_WIDTH-1:0] backbone_rd_exp,
    output wire [DIM*MANT_WIDTH-1:0] backbone_rd_mant,
    output wire backbone_rd_valid,
    
    //==========================================================================
    // Sidenet读接口 (Port B) - 连接到Sidenet处理
    //==========================================================================
    input  wire sidenet_rd_en,
    input  wire [ADDR_WIDTH-1:0] sidenet_rd_addr,
    output wire [EXP_WIDTH-1:0] sidenet_rd_exp,
    output wire [DIM*MANT_WIDTH-1:0] sidenet_rd_mant,
    output wire sidenet_rd_valid,
    
    //==========================================================================
    // 写接口 - 连接到当前层的输出（经过完整Block处理）
    //==========================================================================
    input  wire layer_wr_en,
    input  wire [ADDR_WIDTH-1:0] layer_wr_addr,
    input  wire [EXP_WIDTH-1:0] layer_wr_exp,
    input  wire [DIM*MANT_WIDTH-1:0] layer_wr_mant,
    output wire layer_wr_ready,
    
    //==========================================================================
    // DRAM加载接口 - 初始输入加载到BANK A
    //==========================================================================
    input  wire dram_load_en,
    input  wire [ADDR_WIDTH-1:0] dram_load_addr,
    input  wire [EXP_WIDTH-1:0] dram_load_exp,
    input  wire [DIM*MANT_WIDTH-1:0] dram_load_mant,
    
    //==========================================================================
    // DRAM存储接口 - 最终输出从当前写BANK读出
    //==========================================================================
    input  wire dram_store_en,
    input  wire [ADDR_WIDTH-1:0] dram_store_addr,
    output wire [EXP_WIDTH-1:0] dram_store_exp,
    output wire [DIM*MANT_WIDTH-1:0] dram_store_mant,
    output wire dram_store_valid,
    
    //==========================================================================
    // LN1暂存接口 - LayerNorm1输出暂存，供Residual2使用
    //==========================================================================
    // LN1保存接口 - LayerNorm1输出写入写BANK
    input  wire ln1_save_en,
    input  wire [ADDR_WIDTH-1:0] ln1_save_addr,
    input  wire [EXP_WIDTH-1:0] ln1_save_exp,
    input  wire [DIM*MANT_WIDTH-1:0] ln1_save_mant,
    
    // LN1加载接口 - Residual2从写BANK读取LN1暂存
    input  wire ln1_load_en,
    input  wire [ADDR_WIDTH-1:0] ln1_load_addr,
    output wire [EXP_WIDTH-1:0] ln1_load_exp,
    output wire [DIM*MANT_WIDTH-1:0] ln1_load_mant,
    output wire ln1_load_valid
);

//================================================================================
// 内部信号定义
//================================================================================

// BANK选择寄存器
// 0: BANK A读，BANK B写
// 1: BANK B读，BANK A写
reg bank_select_reg;

// BANK A 存储器 (指数和尾数分开存储)
reg [EXP_WIDTH-1:0] bank_a_exp_mem [0:TOKEN_NUM-1];
reg [DIM*MANT_WIDTH-1:0] bank_a_mant_mem [0:TOKEN_NUM-1];

// BANK B 存储器
reg [EXP_WIDTH-1:0] bank_b_exp_mem [0:TOKEN_NUM-1];
reg [DIM*MANT_WIDTH-1:0] bank_b_mant_mem [0:TOKEN_NUM-1];

// 读数据寄存器（用于流水线）
reg [EXP_WIDTH-1:0] backbone_rd_exp_reg;
reg [DIM*MANT_WIDTH-1:0] backbone_rd_mant_reg;
reg backbone_rd_valid_reg;

reg [EXP_WIDTH-1:0] sidenet_rd_exp_reg;
reg [DIM*MANT_WIDTH-1:0] sidenet_rd_mant_reg;
reg sidenet_rd_valid_reg;

reg [EXP_WIDTH-1:0] dram_store_exp_reg;
reg [DIM*MANT_WIDTH-1:0] dram_store_mant_reg;
reg dram_store_valid_reg;

// LN1 load输出寄存器
reg [EXP_WIDTH-1:0] ln1_load_exp_reg;
reg [DIM*MANT_WIDTH-1:0] ln1_load_mant_reg;
reg ln1_load_valid_reg;

// 写使能信号（分解到各BANK）
wire bank_a_wr_en;
wire bank_b_wr_en;

// 写地址和数据（公共）
wire [ADDR_WIDTH-1:0] wr_addr;
wire [EXP_WIDTH-1:0] wr_exp;
wire [DIM*MANT_WIDTH-1:0] wr_mant;

//================================================================================
// BANK选择控制逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bank_select_reg <= 1'b0;  // 复位：读BANK A，写BANK B
    end else if (bank_swap) begin
        bank_select_reg <= ~bank_select_reg;  // 翻转BANK
    end
end

assign current_read_bank = bank_select_reg;
assign current_write_bank = ~bank_select_reg;

//================================================================================
// 写入逻辑
//================================================================================

// 写入源多路选择：层输出 或 DRAM加载 或 LN1暂存
assign wr_addr = dram_load_en ? dram_load_addr : 
                 ln1_save_en  ? ln1_save_addr  : 
                 layer_wr_addr;
                 
assign wr_exp  = dram_load_en ? dram_load_exp  : 
                 ln1_save_en  ? ln1_save_exp   : 
                 layer_wr_exp;
                 
assign wr_mant = dram_load_en ? dram_load_mant : 
                 ln1_save_en  ? ln1_save_mant  : 
                 layer_wr_mant;

// BANK写使能控制
// DRAM加载：始终写BANK A（初始状态）
// LN1暂存：写当前write_bank（写BANK，与layer_wr互斥）
// 层输出：写当前write_bank
assign bank_a_wr_en = dram_load_en || ((layer_wr_en || ln1_save_en) && ~bank_select_reg);
assign bank_b_wr_en = (layer_wr_en || ln1_save_en) && bank_select_reg;

// BANK A 写入
always @(posedge clk) begin
    if (bank_a_wr_en) begin
        bank_a_exp_mem[wr_addr]  <= wr_exp;
        bank_a_mant_mem[wr_addr] <= wr_mant;
    end
end

// BANK B 写入
always @(posedge clk) begin
    if (bank_b_wr_en) begin
        bank_b_exp_mem[wr_addr]  <= wr_exp;
        bank_b_mant_mem[wr_addr] <= wr_mant;
    end
end

assign layer_wr_ready = 1'b1;  // 简化：假设总是ready

//================================================================================
// Backbone读取逻辑 (Port A) - 从当前read_bank读
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        backbone_rd_valid_reg <= 1'b0;
        backbone_rd_exp_reg   <= {EXP_WIDTH{1'b0}};
        backbone_rd_mant_reg  <= {(DIM*MANT_WIDTH){1'b0}};
    end else begin
        backbone_rd_valid_reg <= backbone_rd_en;
        
        if (backbone_rd_en) begin
            if (bank_select_reg == 1'b0) begin
                // 读BANK A
                backbone_rd_exp_reg  <= bank_a_exp_mem[backbone_rd_addr];
                backbone_rd_mant_reg <= bank_a_mant_mem[backbone_rd_addr];
            end else begin
                // 读BANK B
                backbone_rd_exp_reg  <= bank_b_exp_mem[backbone_rd_addr];
                backbone_rd_mant_reg <= bank_b_mant_mem[backbone_rd_addr];
            end
        end
    end
end

assign backbone_rd_exp   = backbone_rd_exp_reg;
assign backbone_rd_mant  = backbone_rd_mant_reg;
assign backbone_rd_valid = backbone_rd_valid_reg;

//================================================================================
// Sidenet读取逻辑 (Port B) - 从当前read_bank读
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sidenet_rd_valid_reg <= 1'b0;
        sidenet_rd_exp_reg   <= {EXP_WIDTH{1'b0}};
        sidenet_rd_mant_reg  <= {(DIM*MANT_WIDTH){1'b0}};
    end else begin
        sidenet_rd_valid_reg <= sidenet_rd_en;
        
        if (sidenet_rd_en) begin
            if (bank_select_reg == 1'b0) begin
                // 读BANK A
                sidenet_rd_exp_reg  <= bank_a_exp_mem[sidenet_rd_addr];
                sidenet_rd_mant_reg <= bank_a_mant_mem[sidenet_rd_addr];
            end else begin
                // 读BANK B
                sidenet_rd_exp_reg  <= bank_b_exp_mem[sidenet_rd_addr];
                sidenet_rd_mant_reg <= bank_b_mant_mem[sidenet_rd_addr];
            end
        end
    end
end

assign sidenet_rd_exp   = sidenet_rd_exp_reg;
assign sidenet_rd_mant  = sidenet_rd_mant_reg;
assign sidenet_rd_valid = sidenet_rd_valid_reg;

//================================================================================
// DRAM存储读取逻辑 - 从当前write_bank读（最终输出在write_bank）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dram_store_valid_reg <= 1'b0;
        dram_store_exp_reg   <= {EXP_WIDTH{1'b0}};
        dram_store_mant_reg  <= {(DIM*MANT_WIDTH){1'b0}};
    end else begin
        dram_store_valid_reg <= dram_store_en;
        
        if (dram_store_en) begin
            if (bank_select_reg == 1'b0) begin
                // 最终输出在BANK B (当前write bank)
                dram_store_exp_reg  <= bank_b_exp_mem[dram_store_addr];
                dram_store_mant_reg <= bank_b_mant_mem[dram_store_addr];
            end else begin
                // 最终输出在BANK A
                dram_store_exp_reg  <= bank_a_exp_mem[dram_store_addr];
                dram_store_mant_reg <= bank_a_mant_mem[dram_store_addr];
            end
        end
    end
end

assign dram_store_exp   = dram_store_exp_reg;
assign dram_store_mant  = dram_store_mant_reg;
assign dram_store_valid = dram_store_valid_reg;

//================================================================================
// LN1加载读取逻辑 - 从当前write_bank读（LN1暂存在write_bank）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ln1_load_valid_reg <= 1'b0;
        ln1_load_exp_reg   <= {EXP_WIDTH{1'b0}};
        ln1_load_mant_reg  <= {(DIM*MANT_WIDTH){1'b0}};
    end else begin
        ln1_load_valid_reg <= ln1_load_en;
        
        if (ln1_load_en) begin
            if (bank_select_reg == 1'b0) begin
                // 写BANK是B，从BANK B读取LN1暂存
                ln1_load_exp_reg  <= bank_b_exp_mem[ln1_load_addr];
                ln1_load_mant_reg <= bank_b_mant_mem[ln1_load_addr];
            end else begin
                // 写BANK是A，从BANK A读取LN1暂存
                ln1_load_exp_reg  <= bank_a_exp_mem[ln1_load_addr];
                ln1_load_mant_reg <= bank_a_mant_mem[ln1_load_addr];
            end
        end
    end
end

assign ln1_load_exp   = ln1_load_exp_reg;
assign ln1_load_mant  = ln1_load_mant_reg;
assign ln1_load_valid = ln1_load_valid_reg;

//================================================================================
// 仿真初始化
//================================================================================

`ifdef SIMULATION
integer i;
initial begin
    $display("========================================");
    $display("Layer Token Buffer Dual-BANK");
    $display("========================================");
    $display("Configuration:");
    $display("  Tokens:        %0d", TOKEN_NUM);
    $display("  Dimension:     %0d", DIM);
    $display("  Exponent bits: %0d", EXP_WIDTH);
    $display("  Mantissa bits: %0d per dim", MANT_WIDTH);
    $display("  Total bits:    %0d per token", EXP_WIDTH + DIM*MANT_WIDTH);
    $display("  BANK size:     %0d KB", (TOKEN_NUM * (EXP_WIDTH + DIM*MANT_WIDTH)) / 8 / 1024);
    $display("  Total size:    %0d KB", 2 * (TOKEN_NUM * (EXP_WIDTH + DIM*MANT_WIDTH)) / 8 / 1024);
    $display("========================================");
    
    // 初始化存储器（避免X态）
    for (i = 0; i < TOKEN_NUM; i = i + 1) begin
        bank_a_exp_mem[i] = {EXP_WIDTH{1'b0}};
        bank_a_mant_mem[i] = {(DIM*MANT_WIDTH){1'b0}};
        bank_b_exp_mem[i] = {EXP_WIDTH{1'b0}};
        bank_b_mant_mem[i] = {(DIM*MANT_WIDTH){1'b0}};
    end
end
`endif

endmodule