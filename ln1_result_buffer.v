`timescale 1ns / 1ps

module ln1_result_buffer #(
    parameter TOKEN_NUM    = 32,   // 一个batch的token数量
    parameter DIM          = 32,
    parameter DATA_WIDTH   = 8,
    parameter EXP_WIDTH    = 8,
    parameter ADDR_WIDTH   = 5     // log2(32) = 5 bits
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

    // 计算总位宽 (256 bits)
    localparam MANT_WIDTH = DIM * DATA_WIDTH;

    //================================================================================
    // 1. 存储阵列定义
    //================================================================================
    (* ram_style = "distributed" *) reg [EXP_WIDTH-1:0]    exp_mem  [0:TOKEN_NUM-1];
    (* ram_style = "distributed" *) reg [MANT_WIDTH-1:0]   mant_mem [0:TOKEN_NUM-1];

    // 内部输出寄存器（保持读延迟为1周期，与原设计一致）
    reg [EXP_WIDTH-1:0]    mem_rd_exp;
    reg [MANT_WIDTH-1:0]   mem_rd_mant;

    //================================================================================
    // 2. 读写逻辑
    //================================================================================
    
    always @(posedge clk) begin
        // 写逻辑
        if (wr_en) begin
            exp_mem[wr_addr]  <= wr_exp;
            mant_mem[wr_addr] <= wr_mant;
        end
        
        // 读逻辑
        if (rd_en) begin
            mem_rd_exp  <= exp_mem[rd_addr];
            mem_rd_mant <= mant_mem[rd_addr];
        end
    end

    //================================================================================
    // 3. 外部冲突检测与旁路 (External Bypass Logic)
    //================================================================================

    reg                  bypass_valid;
    reg [EXP_WIDTH-1:0]  bypass_exp;
    reg [MANT_WIDTH-1:0] bypass_mant;

    // 检测冲突条件：读写同时有效，且地址相同
    wire collision = rd_en && wr_en && (rd_addr == wr_addr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bypass_valid <= 1'b0;
            bypass_exp   <= {EXP_WIDTH{1'b0}};
            bypass_mant  <= {MANT_WIDTH{1'b0}};
        end else begin
            if (collision) begin
                // 如果发生冲突，锁存当前写入的数据
                bypass_valid <= 1'b1;
                bypass_exp   <= wr_exp;
                bypass_mant  <= wr_mant;
            end else begin
                bypass_valid <= 1'b0;
            end
        end
    end

    //================================================================================
    // 4. 输出多路选择 
    //================================================================================
    
    assign rd_exp  = bypass_valid ? bypass_exp  : mem_rd_exp;
    assign rd_mant = bypass_valid ? bypass_mant : mem_rd_mant;

    //================================================================================
    // 5. 控制信号与调试
    //================================================================================
    
    // 生成 rd_valid 信号 (延后一拍，与数据对齐)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_valid <= 1'b0;
        else        rd_valid <= rd_en;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_rd_count <= 32'h0;
            dbg_wr_count <= 32'h0;
        end else begin
            if (rd_en) dbg_rd_count <= dbg_rd_count + 1'b1;
            if (wr_en) dbg_wr_count <= dbg_wr_count + 1'b1;
        end
    end

endmodule