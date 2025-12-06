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
`timescale 1ns / 1ps

module layer_token_buffer_dual_bank #(
    parameter TOKEN_NUM   = 641,
    parameter DIM         = 32,
    parameter EXP_WIDTH   = 8,
    parameter MANT_WIDTH  = 8,
    parameter ADDR_WIDTH  = 10
)(
    input  wire clk,
    input  wire rst_n,
    
    // Control
    input  wire bank_swap,
    output wire current_read_bank,
    output wire current_write_bank,
    
    // Port A (Backbone Read)
    input  wire backbone_rd_en,
    input  wire [ADDR_WIDTH-1:0] backbone_rd_addr,
    output wire [EXP_WIDTH-1:0] backbone_rd_exp,
    output wire [DIM*MANT_WIDTH-1:0] backbone_rd_mant,
    output wire backbone_rd_valid,
    
    // Port B (Sidenet Read)
    input  wire sidenet_rd_en,
    input  wire [ADDR_WIDTH-1:0] sidenet_rd_addr,
    output wire [EXP_WIDTH-1:0] sidenet_rd_exp,
    output wire [DIM*MANT_WIDTH-1:0] sidenet_rd_mant,
    output wire sidenet_rd_valid,
    
    // Write Port
    input  wire layer_wr_en,
    input  wire [ADDR_WIDTH-1:0] layer_wr_addr,
    input  wire [EXP_WIDTH-1:0] layer_wr_exp,
    input  wire [DIM*MANT_WIDTH-1:0] layer_wr_mant,
    output wire layer_wr_ready,
    
    // DRAM Load/Store & LN1
    input  wire dram_load_en,
    input  wire [ADDR_WIDTH-1:0] dram_load_addr,
    input  wire [EXP_WIDTH-1:0] dram_load_exp,
    input  wire [DIM*MANT_WIDTH-1:0] dram_load_mant,
    
    input  wire dram_store_en,
    input  wire [ADDR_WIDTH-1:0] dram_store_addr,
    output wire [EXP_WIDTH-1:0] dram_store_exp,
    output wire [DIM*MANT_WIDTH-1:0] dram_store_mant,
    output wire dram_store_valid,
    
    input  wire ln1_save_en,
    input  wire [ADDR_WIDTH-1:0] ln1_save_addr,
    input  wire [EXP_WIDTH-1:0] ln1_save_exp,
    input  wire [DIM*MANT_WIDTH-1:0] ln1_save_mant,
    
    input  wire ln1_load_en,
    input  wire [ADDR_WIDTH-1:0] ln1_load_addr,
    output wire [EXP_WIDTH-1:0] ln1_load_exp,
    output wire [DIM*MANT_WIDTH-1:0] ln1_load_mant,
    output wire ln1_load_valid
);

    //==========================================================================
    // 1. BANK Management
    //==========================================================================
    reg bank_select_reg; 
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) bank_select_reg <= 1'b0;
        else if (bank_swap) bank_select_reg <= ~bank_select_reg;
    end

    assign current_read_bank  = bank_select_reg;
    assign current_write_bank = ~bank_select_reg;

    //==========================================================================
    // 2. Physical Port Mapping
    //==========================================================================
    
    // --- Write Bus Aggregation ---
    wire common_wr_en;
    wire [ADDR_WIDTH-1:0] common_wr_addr;
    wire [EXP_WIDTH-1:0] common_wr_exp;
    wire [DIM*MANT_WIDTH-1:0] common_wr_mant;
    
    assign common_wr_en = dram_load_en || ln1_save_en || layer_wr_en;
    
    assign common_wr_addr = dram_load_en ? dram_load_addr : 
                            ln1_save_en  ? ln1_save_addr  : 
                            layer_wr_addr;
                            
    assign common_wr_exp  = dram_load_en ? dram_load_exp  : 
                            ln1_save_en  ? ln1_save_exp   : 
                            layer_wr_exp;

    assign common_wr_mant = dram_load_en ? dram_load_mant : 
                            ln1_save_en  ? ln1_save_mant  : 
                            layer_wr_mant;

    assign layer_wr_ready = 1'b1;

    // --- Port Address / Write Control Logic ---
    // Definition for Port A and Port B inputs for both banks
    
    reg        bank_a_we_a;
    reg [ADDR_WIDTH-1:0] bank_a_addr_a;
    reg [ADDR_WIDTH-1:0] bank_a_addr_b;

    reg        bank_b_we_a;
    reg [ADDR_WIDTH-1:0] bank_b_addr_a;
    reg [ADDR_WIDTH-1:0] bank_b_addr_b;

    // BANK A Control
    always @(*) begin
        // Port A: Handles Writes OR Backbone/Aux Read
        if (dram_load_en) begin
            bank_a_we_a   = 1'b1;
            bank_a_addr_a = dram_load_addr;
        end 
        else if (bank_select_reg == 1'b1) begin // Write Mode
            bank_a_we_a   = common_wr_en;
            bank_a_addr_a = common_wr_addr;
        end 
        else begin // Read Mode
            bank_a_we_a   = 1'b0;
            if (backbone_rd_en)      bank_a_addr_a = backbone_rd_addr;
            else if (dram_store_en)  bank_a_addr_a = dram_store_addr;
            else if (ln1_load_en)    bank_a_addr_a = ln1_load_addr;
            else                     bank_a_addr_a = backbone_rd_addr;
        end

        // Port B: Handles Sidenet Read only
        bank_a_addr_b = sidenet_rd_addr;
    end

    // BANK B Control
    always @(*) begin
        // Port A: Handles Writes OR Backbone/Aux Read
        if (bank_select_reg == 1'b0) begin // Write Mode
            bank_b_we_a   = common_wr_en;
            bank_b_addr_a = common_wr_addr;
        end 
        else begin // Read Mode
            bank_b_we_a   = 1'b0;
            if (backbone_rd_en)      bank_b_addr_a = backbone_rd_addr;
            else if (dram_store_en)  bank_b_addr_a = dram_store_addr;
            else if (ln1_load_en)    bank_b_addr_a = ln1_load_addr;
            else                     bank_b_addr_a = backbone_rd_addr;
        end

        // Port B: Handles Sidenet Read only
        bank_b_addr_b = sidenet_rd_addr;
    end

    //==========================================================================
    // 3. BRAM Instantiation (TRUE DUAL PORT, SYNCHRONOUS READ)
    //==========================================================================
    
    // ---------------- BANK A ----------------
    (* ram_style = "block" *) reg [EXP_WIDTH-1:0] bank_a_exp_mem [0:TOKEN_NUM-1];
    (* ram_style = "block" *) reg [DIM*MANT_WIDTH-1:0] bank_a_mant_mem [0:TOKEN_NUM-1];
    
    reg [EXP_WIDTH-1:0]      bank_a_dout_exp_a, bank_a_dout_exp_b;
    reg [DIM*MANT_WIDTH-1:0] bank_a_dout_mant_a, bank_a_dout_mant_b;

    // Port A (Read/Write)
    always @(posedge clk) begin
        if (bank_a_we_a) begin
            bank_a_exp_mem[bank_a_addr_a]  <= common_wr_exp;
            bank_a_mant_mem[bank_a_addr_a] <= common_wr_mant;
        end
        // SYNCHRONOUS READ: Always read, result available next cycle
        bank_a_dout_exp_a  <= bank_a_exp_mem[bank_a_addr_a];
        bank_a_dout_mant_a <= bank_a_mant_mem[bank_a_addr_a];
    end

    // Port B (Read Only)
    always @(posedge clk) begin
        bank_a_dout_exp_b  <= bank_a_exp_mem[bank_a_addr_b];
        bank_a_dout_mant_b <= bank_a_mant_mem[bank_a_addr_b];
    end

    // ---------------- BANK B ----------------
    (* ram_style = "block" *) reg [EXP_WIDTH-1:0] bank_b_exp_mem [0:TOKEN_NUM-1];
    (* ram_style = "block" *) reg [DIM*MANT_WIDTH-1:0] bank_b_mant_mem [0:TOKEN_NUM-1];

    reg [EXP_WIDTH-1:0]      bank_b_dout_exp_a, bank_b_dout_exp_b;
    reg [DIM*MANT_WIDTH-1:0] bank_b_dout_mant_a, bank_b_dout_mant_b;

    // Port A (Read/Write)
    always @(posedge clk) begin
        if (bank_b_we_a) begin
            bank_b_exp_mem[bank_b_addr_a]  <= common_wr_exp;
            bank_b_mant_mem[bank_b_addr_a] <= common_wr_mant;
        end
        // SYNCHRONOUS READ
        bank_b_dout_exp_a  <= bank_b_exp_mem[bank_b_addr_a];
        bank_b_dout_mant_a <= bank_b_mant_mem[bank_b_addr_a];
    end

    // Port B (Read Only)
    always @(posedge clk) begin
        bank_b_dout_exp_b  <= bank_b_exp_mem[bank_b_addr_b];
        bank_b_dout_mant_b <= bank_b_mant_mem[bank_b_addr_b];
    end

    //==========================================================================
    // 4. Output Muxing (Comb Logic after BRAM Regs)
    //==========================================================================
    // 由于 BRAM 输出已经是寄存器输出（delayed by 1 clk），
    // 这里的 MUX 只是组合逻辑选择，不需要再打一拍，
    // 从而保证从输入地址到 valid 只有 1 个周期的延迟。

    // BACKBONE
    assign backbone_rd_exp  = (bank_select_reg == 0) ? bank_a_dout_exp_a  : bank_b_dout_exp_a;
    assign backbone_rd_mant = (bank_select_reg == 0) ? bank_a_dout_mant_a : bank_b_dout_mant_a;
    
    // Generate Valid Signal (Delayed by 1 cycle to match BRAM latency)
    reg bb_valid_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) bb_valid_reg <= 0;
        else bb_valid_reg <= backbone_rd_en;
    end
    assign backbone_rd_valid = bb_valid_reg;

    // SIDENET
    assign sidenet_rd_exp  = (bank_select_reg == 0) ? bank_a_dout_exp_b  : bank_b_dout_exp_b;
    assign sidenet_rd_mant = (bank_select_reg == 0) ? bank_a_dout_mant_b : bank_b_dout_mant_b;
    
    reg sn_valid_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) sn_valid_reg <= 0;
        else sn_valid_reg <= sidenet_rd_en;
    end
    assign sidenet_rd_valid = sn_valid_reg;

    // DRAM STORE & LN1 LOAD (From Port A)
    assign dram_store_exp  = (bank_select_reg == 0) ? bank_a_dout_exp_a  : bank_b_dout_exp_a;
    assign dram_store_mant = (bank_select_reg == 0) ? bank_a_dout_mant_a : bank_b_dout_mant_a;
    assign ln1_load_exp    = dram_store_exp;
    assign ln1_load_mant   = dram_store_mant;

    reg ds_valid_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) ds_valid_reg <= 0;
        else ds_valid_reg <= dram_store_en;
    end
    assign dram_store_valid = ds_valid_reg;

    reg ln1_valid_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) ln1_valid_reg <= 0;
        else ln1_valid_reg <= ln1_load_en;
    end
    assign ln1_load_valid = ln1_valid_reg;

    // Simulation Init
`ifdef SIMULATION
    integer i;
    initial begin
        for (i = 0; i < TOKEN_NUM; i = i + 1) begin
            bank_a_exp_mem[i]  = 0; bank_a_mant_mem[i] = 0;
            bank_b_exp_mem[i]  = 0; bank_b_mant_mem[i] = 0;
        end
    end
`endif

endmodule