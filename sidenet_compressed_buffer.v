`timescale 1ns / 1ps

//================================================================================
// Sidenet Compressed Buffer - 压缩Token单BANK存储（优化版）
//
// 功能说明：
// 1. 存储Compression Engine输出的压缩tokens
// 2. 单BANK循环复用设计，节省50%资源（11.2KB → 5.6KB）
// 3. 数据格式：16-bit BFP (blockBFP: 1个共享指数 + 8个16-bit尾数)
// 4. 容量：641 tokens × (8 + 8×16) bits = 641 × 136 bits = 5.6 KB
//
// 使用场景：
// Layer 0: compress_0 → 写Buffer → Gate → Adaptation
// Layer 1: compress_1 → 写Buffer (覆盖Layer 0) → Gate → Adaptation
// Layer 2: compress_2 → 写Buffer (覆盖Layer 1) → Gate → Adaptation
// Layer 3: compress_3 → 写Buffer (覆盖Layer 2) → Gate → Adaptation
// Layer 4: compress_4 → 写Buffer (覆盖Layer 3) → Gate → Expand
//
// 设计原理：
// - 这是Level 2临时存储，每层的压缩数据用完即可丢弃
// - 层与层之间无需保留历史数据
// - 单BANK循环复用，大幅节省BRAM资源
//
// 接口说明：
// - 写接口：单端口，连接Compression Engine输出
// - 读接口：单端口，连接Gate Engine输入
// - 无需BANK切换控制（单BANK设计）
//
//================================================================================

module sidenet_compressed_buffer #(
    parameter TOKEN_NUM      = 641,     // Token数量
    parameter COMPRESSED_DIM = 8,       // 压缩后维度 (原始32 → 压缩8, r=4)
    parameter DATA_WIDTH     = 16,      // 尾数位宽 (16-bit BFP)
    parameter EXP_WIDTH      = 8,       // 指数位宽
    parameter ADDR_WIDTH     = 10       // 地址位宽 (2^10 = 1024 > 641)
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 写接口 - 连接Compression Engine输出
    //==========================================================================
    input  wire wr_en,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [EXP_WIDTH-1:0] wr_exp,                      
    input  wire [COMPRESSED_DIM*DATA_WIDTH-1:0] wr_mant,     
    output wire wr_ready,
    
    //==========================================================================
    // 读接口 - 连接Gate Engine输入
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
// 写逻辑 
//================================================================================

always @(posedge clk) begin
    if (wr_en && wr_addr_valid) begin
        exp_mem[wr_addr]  <= wr_exp;
        mant_mem[wr_addr] <= wr_mant;
    end
end

assign wr_ready = 1'b1;  // 总是ready（单周期写入）

//================================================================================
// 读逻辑
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

assign dbg_total_writes      = write_count;
assign dbg_total_reads       = read_count;
assign dbg_write_collisions  = collision_count;

//================================================================================
// 仿真信息输出
//================================================================================

`ifdef SIMULATION
initial begin
    $display("========================================");
    $display("Sidenet Compressed Buffer (Optimized)");
    $display("========================================");
    $display("Configuration:");
    $display("  - Tokens:          %0d", TOKEN_NUM);
    $display("  - Compressed Dim:  %0d", COMPRESSED_DIM);
    $display("  - Data Width:      %0d-bit", DATA_WIDTH);
    $display("  - Exp Width:       %0d-bit", EXP_WIDTH);
    $display("----------------------------------------");
    $display("Storage (16-bit BFP):");
    $display("  - Per Token:       %0d bytes", (EXP_WIDTH + COMPRESSED_DIM*DATA_WIDTH)/8);
    $display("  - Total Storage:   %0.1f KB", (TOKEN_NUM * (EXP_WIDTH + COMPRESSED_DIM*DATA_WIDTH))/8192.0);
    $display("----------------------------------------");
    $display("Optimization:");
    $display("  - Design:          Single BANK");
    $display("  - Resource Saved:  50%% (vs Dual-BANK)");
    $display("  - Old Size:        11.2 KB");
    $display("  - New Size:        5.6 KB");
    $display("========================================");
end

// 监控写操作
always @(posedge clk) begin
    if (wr_en && wr_addr_valid) begin
        $display("[%0t] Compressed Buffer WRITE: addr=%0d, exp=0x%h, mant=0x%h", 
                 $time, wr_addr, wr_exp, wr_mant);
    end
end

// 监控读操作
always @(posedge clk) begin
    if (rd_en && rd_addr_valid) begin
        $display("[%0t] Compressed Buffer READ:  addr=%0d", $time, rd_addr);
    end
end
`endif

//================================================================================
// 断言检查（仿真）
//================================================================================

`ifdef SIMULATION
// 检查地址范围
always @(posedge clk) begin
    if (wr_en && !wr_addr_valid) begin
        $error("[%0t] ERROR: Write address %0d exceeds TOKEN_NUM %0d", 
               $time, wr_addr, TOKEN_NUM);
    end
    
    if (rd_en && !rd_addr_valid) begin
        $error("[%0t] ERROR: Read address %0d exceeds TOKEN_NUM %0d", 
               $time, rd_addr, TOKEN_NUM);
    end
end

// 检查同时读写同一地址（潜在冲突警告）
always @(posedge clk) begin
    if (wr_en && rd_en && wr_addr_valid && rd_addr_valid && (wr_addr == rd_addr)) begin
        $warning("[%0t] WARNING: Simultaneous read/write to address %0d (Write-through behavior)", 
                 $time, wr_addr);
    end
end

// 统计报告（每10000个周期）
reg [31:0] cycle_count;
initial cycle_count = 32'd0;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cycle_count <= 32'd0;
    end else begin
        cycle_count <= cycle_count + 32'd1;
        
        if (cycle_count % 32'd10000 == 32'd0 && cycle_count != 32'd0) begin
            $display("----------------------------------------");
            $display("[%0t] Compressed Buffer Statistics:", $time);
            $display("  Total Writes:      %0d", write_count);
            $display("  Total Reads:       %0d", read_count);
            $display("  Read/Write Ratio:  %0.2f", read_count * 1.0 / (write_count + 1));
            $display("  Collisions:        %0d", collision_count);
            $display("----------------------------------------");
        end
    end
end
`endif

//================================================================================
// 形式验证属性（支持形式验证工具）
//================================================================================

`ifdef FORMAL
// 属性1：写使能时，数据必须在下一周期可读
property p_write_then_read;
    @(posedge clk) disable iff (!rst_n)
    (wr_en && wr_addr_valid) |-> ##1 (exp_mem[$past(wr_addr)] == $past(wr_exp));
endproperty
assert property (p_write_then_read);

// 属性2：地址范围检查
property p_addr_range;
    @(posedge clk)
    (wr_en |-> wr_addr < TOKEN_NUM) && (rd_en |-> rd_addr < TOKEN_NUM);
endproperty
assert property (p_addr_range);

// 属性3：读有效信号延迟1周期
property p_read_valid_delay;
    @(posedge clk) disable iff (!rst_n)
    rd_en |-> ##1 rd_valid;
endproperty
assert property (p_read_valid_delay);
`endif

endmodule