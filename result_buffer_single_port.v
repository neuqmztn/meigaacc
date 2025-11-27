module result_buffer_single_port #(
    parameter TOKEN_NUM    = 641,
    parameter DIM          = 32,
    parameter DATA_WIDTH   = 8,
    parameter EXP_WIDTH    = 8,
    // 建议使用 $clog2(TOKEN_NUM) 自动计算地址宽度，对于 641，结果是 10
    parameter ADDR_WIDTH   = 10
)(
    input  wire clk,
    input  wire rst_n,
    
    // 读端口
    input  wire                          rd_en,
    input  wire [ADDR_WIDTH-1:0]         rd_addr,
    output reg  [EXP_WIDTH-1:0]          rd_exp,
    output reg  [DIM*DATA_WIDTH-1:0]     rd_mant,
    output reg                           rd_valid,
    
    // 写端口
    input  wire                          wr_en,
    input  wire [ADDR_WIDTH-1:0]         wr_addr,
    input  wire [EXP_WIDTH-1:0]          wr_exp,
    input  wire [DIM*DATA_WIDTH-1:0]     wr_mant,
    
    // 调试接口
    output reg  [31:0]                   dbg_rd_count,
    output reg  [31:0]                   dbg_wr_count
);

    // 存储阵列 (Block RAM)
    (* ram_style = "block" *) reg [EXP_WIDTH-1:0]        exp_mem  [0:TOKEN_NUM-1];
    (* ram_style = "block" *) reg [DIM*DATA_WIDTH-1:0]   mant_mem [0:TOKEN_NUM-1];

    // 地址有效性
    wire rd_addr_valid = (rd_addr < TOKEN_NUM);
    wire wr_addr_valid = (wr_addr < TOKEN_NUM);

    // 定义有效的使能信号，简化后续逻辑
    wire effective_wr_en = wr_en && wr_addr_valid;
    wire effective_rd_en = rd_en && rd_addr_valid;

    // ============================================================
    // BRAM 访问逻辑 (SDP RAM, Write-First 模式推断模板)
    // ============================================================
    // 读写逻辑合并在一个 always 块中

    always @(posedge clk) begin
        // 1. 写逻辑
        if (effective_wr_en) begin
            exp_mem[wr_addr]  <= wr_exp;
            mant_mem[wr_addr] <= wr_mant;
        end

        // 2. 读逻辑 (包含转发逻辑描述)
        if (effective_rd_en) begin
            // 检查是否在同一周期对同一地址进行读写 (冲突检测)
            if (effective_wr_en && (wr_addr == rd_addr)) begin
                // 冲突：直接转发写入的数据 (实现 Write-First)
                // 综合工具会自动处理这里的旁路逻辑
                rd_exp  <= wr_exp;
                rd_mant <= wr_mant;
            end else begin
                // 无冲突：从内存阵列读取
                rd_exp  <= exp_mem[rd_addr];
                rd_mant <= mant_mem[rd_addr];
            end
        end
    end
    // ============================================================

    // rd_valid：单独控制
    // BRAM 有一个周期的读取延迟，rd_valid 自然也会延迟一个周期
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_valid <= 1'b0;
        else
            rd_valid <= effective_rd_en;
    end

    // 调试计数器 (保持不变)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_rd_count <= 32'h0;
            dbg_wr_count <= 32'h0;
        end else begin
            if (effective_rd_en)
                dbg_rd_count <= dbg_rd_count + 1'b1;
            if (effective_wr_en)
                dbg_wr_count <= dbg_wr_count + 1'b1;
        end
    end

endmodule