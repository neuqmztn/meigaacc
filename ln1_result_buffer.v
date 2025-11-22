`timescale 1ns / 1ps

//================================================================================
// LayerNorm1 Result Buffer - LN1输出专用存储
//
// 功能：
// 1. 存储LayerNorm1的输出结果（1KB）
//    - 32 tokens × 33 bytes = 1KB
//    - 按batch处理，一个batch = 32 tokens
//
// 2. 用途：
//    - LayerNorm1输出暂存
//    - 供Residual2读取，用于残差连接
//
// 3. 访问模式：
//    - 单端口读写
//    - 时分复用
//    - 支持写后读旁路(write-after-read bypass)
//
// 4. 数据流：
//    LayerNorm1 → 写LN1 Result Buffer
//    Residual2  ← 读LN1 Result Buffer
//
// 数据格式：8-bit BFP
//   - 每token共享指数：8 bits
//   - 32个维度尾数：32 × 8 bits = 256 bits
//   - 总计：264 bits = 33 bytes
//
// 日期：2024-11-16
//================================================================================

module ln1_result_buffer #(
    parameter TOKEN_NUM    = 32,   // 一个batch的token数量
    parameter DIM          = 32,
    parameter DATA_WIDTH   = 8,
    parameter EXP_WIDTH    = 8,
    parameter ADDR_WIDTH   = 5     // log2(32) = 5 bits
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 读端口
    //================================================================================
    input  wire                      rd_en,
    input  wire [ADDR_WIDTH-1:0]     rd_addr,
    output reg  [EXP_WIDTH-1:0]      rd_exp,
    output reg  [DIM*DATA_WIDTH-1:0] rd_mant,
    output reg                       rd_valid,
    
    //================================================================================
    // 写端口
    //================================================================================
    input  wire                      wr_en,
    input  wire [ADDR_WIDTH-1:0]     wr_addr,
    input  wire [EXP_WIDTH-1:0]      wr_exp,
    input  wire [DIM*DATA_WIDTH-1:0] wr_mant,
    
    //================================================================================
    // 状态和调试接口
    //================================================================================
    output reg  [31:0]               dbg_rd_count,
    output reg  [31:0]               dbg_wr_count
);

//================================================================================
// 参数检查（综合时会被优化掉）
//================================================================================
initial begin
    if (TOKEN_NUM > (1 << ADDR_WIDTH)) begin
        $display("ERROR: ADDR_WIDTH=%0d too small for TOKEN_NUM=%0d", 
                 ADDR_WIDTH, TOKEN_NUM);
        $finish;
    end
    if (TOKEN_NUM != 32) begin
        $display("WARNING: TOKEN_NUM=%0d, expected 32 for one batch", TOKEN_NUM);
    end
end

//================================================================================
// 存储器（推断为BRAM）
//================================================================================
(* ram_style = "block" *) reg [EXP_WIDTH-1:0]      exp_mem  [0:TOKEN_NUM-1];
(* ram_style = "block" *) reg [DIM*DATA_WIDTH-1:0] mant_mem [0:TOKEN_NUM-1];

//================================================================================
// 地址有效性检查
//================================================================================
wire rd_addr_valid = (rd_addr < TOKEN_NUM);
wire wr_addr_valid = (wr_addr < TOKEN_NUM);

//================================================================================
// 写端口逻辑（优先级最高）
//================================================================================
always @(posedge clk) begin
    if (wr_en && wr_addr_valid) begin
        exp_mem[wr_addr]  <= wr_exp;
        mant_mem[wr_addr] <= wr_mant;
    end
end

//================================================================================
// 读端口逻辑（带写后读旁路）
//================================================================================

// 检测同地址读写冲突
wire same_addr_access = rd_en && wr_en && (rd_addr == wr_addr) && 
                        rd_addr_valid && wr_addr_valid;

// 旁路寄存器（存储上一周期的写数据）
reg [EXP_WIDTH-1:0]      bypass_exp;
reg [DIM*DATA_WIDTH-1:0] bypass_mant;
reg                      use_bypass;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_exp     <= {EXP_WIDTH{1'b0}};
        rd_mant    <= {DIM*DATA_WIDTH{1'b0}};
        rd_valid   <= 1'b0;
        use_bypass <= 1'b0;
        
        bypass_exp  <= {EXP_WIDTH{1'b0}};
        bypass_mant <= {DIM*DATA_WIDTH{1'b0}};
    end else begin
        //------------------------------------------------------------------------
        // 旁路逻辑：捕获同地址写入的数据
        //------------------------------------------------------------------------
        if (same_addr_access) begin
            bypass_exp  <= wr_exp;
            bypass_mant <= wr_mant;
            use_bypass  <= 1'b1;
        end else begin
            use_bypass <= 1'b0;
        end
        
        //------------------------------------------------------------------------
        // 读数据选择
        //------------------------------------------------------------------------
        rd_valid <= rd_en && rd_addr_valid;
        
        if (rd_en && rd_addr_valid) begin
            if (same_addr_access) begin
                // 情况1：当前周期同地址写，使用写入数据（组合旁路）
                rd_exp  <= wr_exp;
                rd_mant <= wr_mant;
            end else if (use_bypass) begin
                // 情况2：上一周期同地址写，使用旁路寄存器（时序旁路）
                rd_exp  <= bypass_exp;
                rd_mant <= bypass_mant;
            end else begin
                // 情况3：正常从RAM读取
                rd_exp  <= exp_mem[rd_addr];
                rd_mant <= mant_mem[rd_addr];
            end
        end
    end
end

//================================================================================
// 调试计数器
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dbg_rd_count <= 32'h0;
        dbg_wr_count <= 32'h0;
    end else begin
        if (rd_en && rd_addr_valid) dbg_rd_count <= dbg_rd_count + 1'b1;
        if (wr_en && wr_addr_valid) dbg_wr_count <= dbg_wr_count + 1'b1;
    end
end

//================================================================================
// 断言（仿真时检查）
//================================================================================
`ifdef SIMULATION
always @(posedge clk) begin
    if (rd_en && !rd_addr_valid) begin
        $display("ERROR @%0t: [LN1_BUFFER] Read address out of bounds! rd_addr=%0d >= TOKEN_NUM=%0d", 
                 $time, rd_addr, TOKEN_NUM);
    end
    
    if (wr_en && !wr_addr_valid) begin
        $display("ERROR @%0t: [LN1_BUFFER] Write address out of bounds! wr_addr=%0d >= TOKEN_NUM=%0d", 
                 $time, wr_addr, TOKEN_NUM);
    end
    
    if (same_addr_access) begin
        $display("INFO @%0t: [LN1_BUFFER] Same address read-write conflict at addr=%0d (using bypass)", 
                 $time, rd_addr);
    end
end

// 初始化存储器（避免X态）
integer i;
initial begin
    $display("========================================");
    $display("LN1 Result Buffer");
    $display("========================================");
    $display("Configuration:");
    $display("  Tokens:        %0d (1 batch)", TOKEN_NUM);
    $display("  Dimension:     %0d", DIM);
    $display("  Exponent bits: %0d", EXP_WIDTH);
    $display("  Mantissa bits: %0d per dim", DATA_WIDTH);
    $display("  Total bits:    %0d per token", EXP_WIDTH + DIM*DATA_WIDTH);
    $display("  Buffer size:   %0d bytes", TOKEN_NUM * (EXP_WIDTH + DIM*DATA_WIDTH) / 8);
    $display("========================================");
    
    for (i = 0; i < TOKEN_NUM; i = i + 1) begin
        exp_mem[i] = {EXP_WIDTH{1'b0}};
        mant_mem[i] = {(DIM*DATA_WIDTH){1'b0}};
    end
end
`endif

endmodule