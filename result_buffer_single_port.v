module result_buffer_single_port #(
    parameter TOKEN_NUM    = 640,
    parameter DIM          = 32,
    parameter DATA_WIDTH   = 8,
    parameter EXP_WIDTH    = 8,
    parameter ADDR_WIDTH   = 10
)(
    input  wire clk,
    input  wire rst_n,
    
    // 读端口
    input  wire                       rd_en,
    input  wire [ADDR_WIDTH-1:0]      rd_addr,
    output wire [EXP_WIDTH-1:0]       rd_exp,  
    output wire [DIM*DATA_WIDTH-1:0]  rd_mant,  
    output reg                        rd_valid,
    
    // 写端口
    input  wire                       wr_en,
    input  wire [ADDR_WIDTH-1:0]      wr_addr,
    input  wire [EXP_WIDTH-1:0]       wr_exp,
    input  wire [DIM*DATA_WIDTH-1:0]  wr_mant,
    
    // 调试接口
    output reg  [31:0]                dbg_rd_count,
    output reg  [31:0]                dbg_wr_count
);

    // 存储阵列 (Block RAM)
    (* ram_style = "block" *) reg [EXP_WIDTH-1:0]       exp_mem  [0:TOKEN_NUM-1];
    (* ram_style = "block" *) reg [DIM*DATA_WIDTH-1:0]  mant_mem [0:TOKEN_NUM-1];

    wire effective_wr_en = wr_en && (wr_addr < TOKEN_NUM);
    wire effective_rd_en = rd_en && (rd_addr < TOKEN_NUM);

    // ============================================================
    // 1. RAM 推断逻辑
    // ============================================================
    reg [EXP_WIDTH-1:0]      mem_rd_exp;
    reg [DIM*DATA_WIDTH-1:0] mem_rd_mant;

    always @(posedge clk) begin
        if (effective_wr_en) begin
            exp_mem[wr_addr]  <= wr_exp;
            mant_mem[wr_addr] <= wr_mant;
        end
        if (effective_rd_en) begin
            mem_rd_exp  <= exp_mem[rd_addr];
            mem_rd_mant <= mant_mem[rd_addr];
        end
    end

    // ============================================================
    // 2. 外部冲突检测与转发 
    // ============================================================
    reg                      bypass_valid;
    reg [EXP_WIDTH-1:0]      bypass_exp;
    reg [DIM*DATA_WIDTH-1:0] bypass_mant;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bypass_valid <= 1'b0;
            bypass_exp   <= 'd0;
            bypass_mant  <= 'd0;
        end else begin
            // 检测读写冲突：有效写 && 有效读 && 地址相同
            if (effective_wr_en && effective_rd_en && (wr_addr == rd_addr)) begin
                bypass_valid <= 1'b1;
                bypass_exp   <= wr_exp;  // 锁存当前写入的数据
                bypass_mant  <= wr_mant;
            end else begin
                bypass_valid <= 1'b0;
            end
        end
    end

    // ============================================================
    // 3. 输出多路选择
    // ============================================================
    // 如果发生了冲突
    assign rd_exp  = bypass_valid ? bypass_exp  : mem_rd_exp;
    assign rd_mant = bypass_valid ? bypass_mant : mem_rd_mant;

    // ============================================================
    // 4. 控制信号与调试
    // ============================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_valid <= 1'b0;
        else        rd_valid <= effective_rd_en;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_rd_count <= 0;
            dbg_wr_count <= 0;
        end else begin
            if (effective_rd_en) dbg_rd_count <= dbg_rd_count + 1;
            if (effective_wr_en) dbg_wr_count <= dbg_wr_count + 1;
        end
    end

endmodule