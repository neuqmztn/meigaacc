module sidenet_weight_storage #(
    parameter DATA_WIDTH       = 16,
    parameter EXP_WIDTH        = 8,
    parameter DRAM_DATA_WIDTH  = 256,
    parameter MAX_EXP_ARRAY_WIDTH = 256,

    parameter NUM_LAYERS = 4,       // Sidenet 实际包含 4 层
    parameter LAYER_STRIDE = 88,    // 每层占用的行数
    parameter DEPTH = LAYER_STRIDE * NUM_LAYERS, // 总深度 = 352
    parameter ADDR_WIDTH = 9        // log2(352) -> 9 位地址线
)(
    input wire clk,

    //============================================================
    // 读取接口 (前向传播 / 推理阶段使用)
    //============================================================
    input  wire rd_en,
    input  wire [1:0] rd_layer_id,      // 层索引 0~3
    input  wire [3:0] rd_weight_type,   // 权重类型
    input  wire [5:0] rd_burst_idx,     // 突发传输索引 / 列索引

    output reg        rd_valid,
    output reg [DRAM_DATA_WIDTH-1:0]     rd_data_burst,
    output reg [MAX_EXP_ARRAY_WIDTH-1:0] rd_exp_array,

    //============================================================
    // 写入接口 (来自权重更新引擎)
    //============================================================
    input  wire weight_wr_req,
    input  wire [1:0] weight_wr_layer_id,
    input  wire [3:0] weight_wr_type,
    input  wire [5:0] weight_wr_burst_idx,
    input  wire [DRAM_DATA_WIDTH-1:0]     weight_wr_data_burst,
    input  wire [MAX_EXP_ARRAY_WIDTH-1:0] weight_wr_exp_array,

    //============================================================
    // Bank 选择信号 (实现乒乓操作):
    //   0 = 前向传播读取 Bank A，更新引擎写入 Bank B
    //   1 = 前向传播读取 Bank B，更新引擎写入 Bank A
    //============================================================
    input  wire bank_select
);

//======================================================================
// 权重类型编码
//======================================================================
localparam TYPE_COMPRESS   = 4'd0;   // 32×8 -> 占 8 列
localparam TYPE_EXPAND     = 4'd1;   // 8×32 -> 占 32 列
localparam TYPE_FFN_W1     = 4'd2;   // 8×32 -> 占 32 列
localparam TYPE_FFN_W2     = 4'd3;   // 32×8 -> 占 8 列
localparam TYPE_SIDE_ATT_Q = 4'd4;   // 8×8  -> 占 8 列
localparam TYPE_SIDE_ATT_K = 4'd5;   // 8×8  -> 占 8 列
localparam TYPE_SIDE_ATT_V = 4'd6;   // 8×8  -> 占 8 列

//======================================================================
// 类型 -> 层内基地址偏移 (与你的 weightsidetxt.txt 布局一致)
//======================================================================
function [6:0] type_base;
    input [3:0] t;
begin
    case (t)
        TYPE_COMPRESS:     type_base = 7'd0;   // 行号 0~7
        TYPE_EXPAND:       type_base = 7'd8;   // 行号 8~39
        TYPE_FFN_W1:       type_base = 7'd40;  // 行号 40~71
        TYPE_FFN_W2:       type_base = 7'd72;  // 行号 72~79
        TYPE_SIDE_ATT_Q:   type_base = 7'd80;  // 行号 80~87 (共享起始)
        TYPE_SIDE_ATT_K:   type_base = 7'd80;  // 注意：此处 Q/K/V 逻辑需确保突发索引不冲突，或物理共享
        TYPE_SIDE_ATT_V:   type_base = 7'd80;
        default:           type_base = 7'd0;
    endcase
end
endfunction

//======================================================================
// 计算全局物理地址 = 层偏移 + 类型偏移 + 突发索引
//======================================================================
function [ADDR_WIDTH-1:0] calc_addr;
    input [1:0] layer;
    input [3:0] t;
    input [5:0] burst;
    reg [8:0] base;
begin
    base = layer * LAYER_STRIDE + type_base(t);
    calc_addr = base + burst;
end
endfunction

//======================================================================
// 定义两个 BRAM Bank，分别存储尾数 (data) 和指数 (exp array)
//======================================================================
(* ram_style="block" *) reg [DRAM_DATA_WIDTH-1:0]     mem_bank_a [0:DEPTH-1];
(* ram_style="block" *) reg [DRAM_DATA_WIDTH-1:0]     mem_bank_b [0:DEPTH-1];

(* ram_style="block" *) reg [MAX_EXP_ARRAY_WIDTH-1:0] exp_bank_a [0:DEPTH-1];
(* ram_style="block" *) reg [MAX_EXP_ARRAY_WIDTH-1:0] exp_bank_b [0:DEPTH-1];

reg [DRAM_DATA_WIDTH-1:0]     mem_out_a, mem_out_b;
reg [MAX_EXP_ARRAY_WIDTH-1:0] exp_out_a, exp_out_b;

//======================================================================
// 写入逻辑: 真正的乒乓操作 -> 始终写入"未被选中"的 Bank
//======================================================================
wire [ADDR_WIDTH-1:0] wr_addr = calc_addr(
    weight_wr_layer_id,
    weight_wr_type,
    weight_wr_burst_idx
);

always @(posedge clk) begin
    if (weight_wr_req) begin

        if (bank_select == 1'b0) begin
            // 前向传播正在读 Bank A -> 更新引擎写入 Bank B
            mem_bank_b[wr_addr] <= weight_wr_data_burst;
            exp_bank_b[wr_addr] <= weight_wr_exp_array;
        end else begin
            // 前向传播正在读 Bank B -> 更新引擎写入 Bank A
            mem_bank_a[wr_addr] <= weight_wr_data_burst;
            exp_bank_a[wr_addr] <= weight_wr_exp_array;
        end

    end
end

//======================================================================
// 读取逻辑: 始终同时读取两个 Bank (BRAM 单周期读取延迟)
//======================================================================
wire [ADDR_WIDTH-1:0] rd_addr = calc_addr(
    rd_layer_id,
    rd_weight_type,
    rd_burst_idx
);

always @(posedge clk) begin
    if (rd_en) begin
        mem_out_a <= mem_bank_a[rd_addr];
        mem_out_b <= mem_bank_b[rd_addr];
        exp_out_a <= exp_bank_a[rd_addr];
        exp_out_b <= exp_bank_b[rd_addr];
        rd_valid  <= 1'b1;
    end else begin
        rd_valid  <= 1'b0;
    end
end

//======================================================================
// 输出多路复用器 (MUX): 根据 bank_select 选择最终输出的数据
//======================================================================
always @(*) begin
    if (bank_select == 1'b0) begin
        // 前向传播读取 Bank A
        rd_data_burst = mem_out_a;
        rd_exp_array  = exp_out_a;
    end else begin
        // 前向传播读取 Bank B
        rd_data_burst = mem_out_b;
        rd_exp_array  = exp_out_b;
    end
end

endmodule