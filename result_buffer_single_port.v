`timescale 1ns / 1ps

//================================================================================
// Result Buffer - 层内中间结果存储 (单端口)
//
// 功能：
// 1. 存储层内处理的中间结果 (21KB)
//    - 641 tokens × 33 bytes = 21KB
//
// 2. 存储内容（按时间顺序）：
//    - Attention输出
//    - Residual Add 1结果
//    - LayerNorm 1输出
//    - FFN输出  
//    - Residual Add 2结果
//    - LayerNorm 2输出
//
// 3. 访问模式：
//    - 单端口读写
//    - 时分复用
//    - 支持就地更新(in-place update)
//
// 4. 数据流：
//    Attention → 写Result Buffer
//    Residual Add 1 → 读Result Buffer + 读Layer Token Buffer → 写Result Buffer
//    LayerNorm 1 → 读Result Buffer → 写Result Buffer + 写Layer Token Buffer(暂存)
//    FFN → 读Result Buffer → 写Result Buffer
//    Residual Add 2 → 读Result Buffer + 读Layer Token Buffer(暂存) → 写Result Buffer
//    LayerNorm 2 → 读Result Buffer → 写Layer Token Buffer(层输出)
//
// 数据格式：8-bit BFP
//   - 每token共享指数：8 bits
//   - 32个维度尾数：32 × 8 bits = 256 bits
//   - 总计：264 bits = 33 bytes
//
//================================================================================


module result_buffer_single_port #(
    parameter TOKEN_NUM    = 641,
    parameter DIM          = 32,
    parameter DATA_WIDTH   = 8,
    parameter EXP_WIDTH    = 8,
    parameter ADDR_WIDTH   = 10
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
end

//================================================================================
// 双端口RAM存储（推断为BRAM）
//================================================================================
(* ram_style = "block" *) reg [EXP_WIDTH-1:0]      exp_mem  [0:TOKEN_NUM-1];
(* ram_style = "block" *) reg [DIM*DATA_WIDTH-1:0] mant_mem [0:TOKEN_NUM-1];

//================================================================================
// 地址有效性检查
//================================================================================
wire rd_addr_valid = (rd_addr < TOKEN_NUM);
wire wr_addr_valid = (wr_addr < TOKEN_NUM);

//================================================================================
// 写端口逻辑（端口B，优先级最高）
//================================================================================
always @(posedge clk) begin
    if (wr_en && wr_addr_valid) begin
        exp_mem[wr_addr]  <= wr_exp;
        mant_mem[wr_addr] <= wr_mant;
    end
end

//================================================================================
// 读端口逻辑（端口A，带写后读旁路）
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
        $display("ERROR @%0t: Read address out of bounds! rd_addr=%0d >= TOKEN_NUM=%0d", 
                 $time, rd_addr, TOKEN_NUM);
    end
    
    if (wr_en && !wr_addr_valid) begin
        $display("ERROR @%0t: Write address out of bounds! wr_addr=%0d >= TOKEN_NUM=%0d", 
                 $time, wr_addr, TOKEN_NUM);
    end
    
    if (same_addr_access) begin
        $display("WARNING @%0t: Same address read-write conflict at addr=%0d (using bypass)", 
                 $time, rd_addr);
    end
end
`endif

endmodule