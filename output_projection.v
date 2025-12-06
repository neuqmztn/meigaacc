`timescale 1ns / 1ps

module output_projection #(
    parameter NUM_HEADS       = 4,
    parameter TOKEN_BATCH     = 32,
    parameter HEAD_DIM        = 8,
    parameter DIM             = 32,
    parameter DATA_WIDTH      = 8,    // CE 输入 / 输出 尾数宽度（INT8）
    parameter EXP_WIDTH       = 8,
    parameter ACCUM_WIDTH     = 24,   // Head 内部累加器宽度（INT24）

    // CE配置
    parameter CE_OUTPUT_WIDTH   = 32,
    parameter CE_INTERNAL_WIDTH = 39,
    parameter CE_GUARD_BITS     = 7,
    parameter CE_ENABLE_ROUNDING= 1
)(
    input  wire clk,
    input  wire rst_n,

    // 控制接口
    input  wire start,
    input  wire [5:0] batch_id,
    input  wire [4:0] tokens_in_batch,
    output reg  done,
    output reg  busy,

    // 4个Head Engine的Accumulator读取接口
    // Head 0
    output reg  accum_rd_en_h0,
    output reg  [4:0] accum_rd_row_h0,
    output reg  [2:0] accum_rd_dim_h0,
    input  wire signed [ACCUM_WIDTH-1:0] accum_rd_mant_h0,
    input  wire [EXP_WIDTH-1:0]          accum_rd_exp_h0,
    input  wire                          accum_rd_valid_h0,

    // Head 1
    output reg  accum_rd_en_h1,
    output reg  [4:0] accum_rd_row_h1,
    output reg  [2:0] accum_rd_dim_h1,
    input  wire signed [ACCUM_WIDTH-1:0] accum_rd_mant_h1,
    input  wire [EXP_WIDTH-1:0]          accum_rd_exp_h1,
    input  wire                          accum_rd_valid_h1,

    // Head 2
    output reg  accum_rd_en_h2,
    output reg  [4:0] accum_rd_row_h2,
    output reg  [2:0] accum_rd_dim_h2,
    input  wire signed [ACCUM_WIDTH-1:0] accum_rd_mant_h2,
    input  wire [EXP_WIDTH-1:0]          accum_rd_exp_h2,
    input  wire                          accum_rd_valid_h2,

    // Head 3
    output reg  accum_rd_en_h3,
    output reg  [4:0] accum_rd_row_h3,
    output reg  [2:0] accum_rd_dim_h3,
    input  wire signed [ACCUM_WIDTH-1:0] accum_rd_mant_h3,
    input  wire [EXP_WIDTH-1:0]          accum_rd_exp_h3,
    input  wire                          accum_rd_valid_h3,

    // W_O 权重接口（INT8）
    output reg  weight_req,
    input  wire weight_ready,
    input  wire [DIM*EXP_WIDTH-1:0]      weight_exp_array,
    input  wire [DIM*DIM*DATA_WIDTH-1:0] weight_mant,

    // 结果输出接口（BFP格式到Result Buffer）
    output reg  result_wr_en,
    output reg  [9:0] result_wr_addr,
    output reg  [EXP_WIDTH-1:0] result_exp,
    output reg  [DIM*DATA_WIDTH-1:0] result_mant,

    // 调试
    output wire [3:0] dbg_state
);

//------------------------------------------------------------------------------
// 本地参数与状态
//------------------------------------------------------------------------------
localparam CE_BASE_EXP_WIDTH = EXP_WIDTH + 1;

localparam STATE_IDLE        = 4'd0;
localparam STATE_LOAD_WEIGHT = 4'd1;
localparam STATE_WAIT_WEIGHT = 4'd2;
localparam STATE_READ_HEADS  = 4'd3;
localparam STATE_WAIT_READ   = 4'd4;
localparam STATE_ALIGN_CONCAT= 4'd5;
localparam STATE_SEND_CE     = 4'd6;
localparam STATE_WAIT_CE     = 4'd7;
localparam STATE_BFP_CONVERT = 4'd8;
localparam STATE_WRITE_RESULT= 4'd9;
localparam STATE_NEXT_TOKEN  = 4'd10;
localparam STATE_DONE        = 4'd11;

reg [3:0] state;
assign dbg_state = state;

//------------------------------------------------------------------------------
// 计数器/寄存器
//------------------------------------------------------------------------------
reg [4:0] token_idx;
reg [9:0] global_token_addr;

reg [2:0] dim_idx;
reg [3:0] read_wait_counter;

// 权重缓存
reg [DIM*EXP_WIDTH-1:0]      weight_exp_array_buf;
reg [DIM*DIM*DATA_WIDTH-1:0] weight_mant_buf;
reg weight_loaded;

// Head 数据缓存（BFP 24bit）
reg signed [ACCUM_WIDTH-1:0] heads_mant [0:NUM_HEADS-1][0:HEAD_DIM-1];
reg [EXP_WIDTH-1:0]          heads_exp  [0:NUM_HEADS-1];

// 对齐与拼接
reg [EXP_WIDTH-1:0]          max_exp;
reg signed [ACCUM_WIDTH-1:0] aligned_mants [0:NUM_HEADS-1][0:HEAD_DIM-1];
reg signed [ACCUM_WIDTH-1:0] concat_mant [0:DIM-1];
reg [EXP_WIDTH-1:0]          concat_exp;

// 24→8 bit 下采样
localparam [EXP_WIDTH-1:0] DOWN_SHIFT = ACCUM_WIDTH - DATA_WIDTH;
wire [EXP_WIDTH-1:0]         concat_exp_q;
wire signed [DATA_WIDTH-1:0] concat_mant_q [0:DIM-1];
wire [DIM*DATA_WIDTH-1:0]    concat_mant_q_packed;

// Compute Engine 接口
reg  ce_input_valid;
wire ce_input_ready;

wire [DIM-1:0]                           ce_result_valids;
wire signed [DIM*CE_OUTPUT_WIDTH-1:0]    ce_result_fixed_array;
wire [DIM*CE_BASE_EXP_WIDTH-1:0]         ce_result_base_exp_array;
wire [DIM-1:0]                           ce_result_zero_array;

wire ce_compute_done;
assign ce_compute_done = &ce_result_valids;

// CE 输出锁存
reg [DIM-1:0]                        ce_out_valids_r;
reg signed [DIM*CE_OUTPUT_WIDTH-1:0] ce_out_fixed_r;
reg [DIM*CE_BASE_EXP_WIDTH-1:0]      ce_out_base_exp_r;
reg [DIM-1:0]                        ce_out_zero_r;

// BFP 转换器接口
wire [DIM-1:0]                       bfp_valids;
wire signed [DIM*DATA_WIDTH-1:0]     bfp_mants;
wire [EXP_WIDTH-1:0]                 bfp_shared_exp;
wire                                 bfp_overflow;
wire                                 bfp_convert_done;
assign bfp_convert_done = &bfp_valids;

//------------------------------------------------------------------------------
// 24→8 bit 下采样函数
//------------------------------------------------------------------------------
function [DATA_WIDTH-1:0] sat_trunc_mant;
    input signed [ACCUM_WIDTH-1:0] in_val;
    reg signed [ACCUM_WIDTH-1:0] shifted;
    reg signed [DATA_WIDTH-1:0] max_val;
    reg signed [DATA_WIDTH-1:0] min_val;
begin
    shifted = in_val >>> DOWN_SHIFT;
    max_val = {1'b0, {(DATA_WIDTH-1){1'b1}}}; // +127
    min_val = {1'b1, {(DATA_WIDTH-1){1'b0}}}; // -128
    if (shifted > max_val)
        sat_trunc_mant = max_val;
    else if (shifted < min_val)
        sat_trunc_mant = min_val;
    else
        sat_trunc_mant = shifted[DATA_WIDTH-1:0];
end
endfunction

assign concat_exp_q = concat_exp + DOWN_SHIFT;

genvar gi;
generate
    for (gi = 0; gi < DIM; gi = gi + 1) begin : GEN_DOWNCAST
        assign concat_mant_q[gi] = sat_trunc_mant(concat_mant[gi]);
        assign concat_mant_q_packed[gi*DATA_WIDTH +: DATA_WIDTH] = concat_mant_q[gi];
    end
endgenerate

//------------------------------------------------------------------------------
// Compute Engine
//------------------------------------------------------------------------------
compute_engine #(
    .G_OUT(4), .T_OUT(8), .NUM_PE(2), .PE_TYPE_0(0), .PE_TYPE_1(0),
    .EXP_WIDTH(EXP_WIDTH), .INPUT_MANT_WIDTH(DATA_WIDTH),
    .ELEM_PE0(16), .ELEM_PE1(16), .TOTAL_ELEM(DIM),
    .INTERNAL_WIDTH(CE_INTERNAL_WIDTH), .OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
    .GUARD_BITS(CE_GUARD_BITS), .ENABLE_ROUNDING(CE_ENABLE_ROUNDING),
    .HANDSHAKE_TIMEOUT(100)
) u_compute_engine (
    .clk(clk), .rst_n(rst_n), .flush(1'b0),
    .input_valid (ce_input_valid), .input_ready (ce_input_ready),
    .exp_X(concat_exp_q), .mant_X_block(concat_mant_q_packed),
    .exp_W_array (weight_exp_array_buf), .mant_W_blocks(weight_mant_buf),
    .result_valids(ce_result_valids), .result_ready(1'b1),
    .result_fixed_array(ce_result_fixed_array),
    .result_base_exp_array(ce_result_base_exp_array),
    .result_zero_array(ce_result_zero_array)
);

//------------------------------------------------------------------------------
// BFP 转换器
//------------------------------------------------------------------------------
bfp_converter #(
    .TOTAL_RESULTS(DIM), .FIXED_WIDTH(CE_OUTPUT_WIDTH),
    .BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH), .OUTPUT_MANT_WIDTH(DATA_WIDTH),
    .OUTPUT_EXP_WIDTH(EXP_WIDTH)
) u_bfp (
    .clk(clk), .rst_n(rst_n), .flush(1'b0),
    .input_valids(ce_out_valids_r), .input_fixed_array(ce_out_fixed_r),
    .input_base_exp_array(ce_out_base_exp_r), .input_zero_array(ce_out_zero_r),
    .output_valids(bfp_valids), .output_mant_array(bfp_mants),
    .output_shared_exp(bfp_shared_exp), .output_overflow(bfp_overflow)
);

//------------------------------------------------------------------------------
// 主状态机
// 修正说明：
// 1. 移除了 else 下方的 default assignment 块，这是导致 Synth 8-7137 的主要原因。
// 2. 将 accum_rd_row/dim 等数据寄存器加入复位列表，消除不确定性。
// 3. 显式地在状态跳转时拉低 enable 信号。
//------------------------------------------------------------------------------
reg [3:0] read_done_mask;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_IDLE;
        done  <= 1'b0;
        busy  <= 1'b0;

        token_idx         <= 5'd0;
        global_token_addr <= 10'd0;
        dim_idx           <= 3'd0;
        read_wait_counter <= 4'd0;
        read_done_mask    <= 4'b0000;

        weight_loaded        <= 1'b0;
        weight_exp_array_buf <= {DIM*EXP_WIDTH{1'b0}};
        weight_mant_buf      <= {DIM*DIM*DATA_WIDTH{1'b0}};

        // --- 控制信号复位 ---
        accum_rd_en_h0 <= 1'b0;
        accum_rd_en_h1 <= 1'b0;
        accum_rd_en_h2 <= 1'b0;
        accum_rd_en_h3 <= 1'b0;
        weight_req     <= 1'b0;
        result_wr_en   <= 1'b0;
        ce_input_valid <= 1'b0;
        
        // --- 数据寄存器复位（解决 row/dim 报错的关键）---
        accum_rd_row_h0 <= 5'd0; accum_rd_dim_h0 <= 3'd0;
        accum_rd_row_h1 <= 5'd0; accum_rd_dim_h1 <= 3'd0;
        accum_rd_row_h2 <= 5'd0; accum_rd_dim_h2 <= 3'd0;
        accum_rd_row_h3 <= 5'd0; accum_rd_dim_h3 <= 3'd0;

        // 其他寄存器清理
        ce_out_valids_r   <= {DIM{1'b0}};
        ce_out_fixed_r    <= {DIM*CE_OUTPUT_WIDTH{1'b0}};
        ce_out_base_exp_r <= {DIM*CE_BASE_EXP_WIDTH{1'b0}};
        ce_out_zero_r     <= {DIM{1'b0}};

        result_wr_addr <= 10'd0;
        result_exp     <= {EXP_WIDTH{1'b0}};
        // result_mant 太宽，一般不建议复位，消耗资源，如有必要也可复位
        
    end else begin
        // --- 核心修正：此处不再写 accum_rd_en_h0 <= 0 等默认值 ---
        // 信号的值将保持，直到在 Case 语句中显式改变
        
        case (state)
            STATE_IDLE: begin
                done <= 1'b0; // 确保拉低 done
                if (start) begin
                    busy              <= 1'b1;
                    token_idx         <= 5'd0;
                    global_token_addr <= batch_id * TOKEN_BATCH;
                    dim_idx           <= 3'd0;
                    read_done_mask    <= 4'b0000;

                    if (!weight_loaded)
                        state <= STATE_LOAD_WEIGHT;
                    else
                        state <= STATE_READ_HEADS;
                end else begin
                    busy <= 1'b0;
                end
            end

            STATE_LOAD_WEIGHT: begin
                weight_req <= 1'b1; // 拉高
                state      <= STATE_WAIT_WEIGHT;
            end

            STATE_WAIT_WEIGHT: begin
                weight_req <= 1'b0; // 【显式拉低】
                if (weight_ready) begin
                    weight_exp_array_buf <= weight_exp_array;
                    weight_mant_buf      <= weight_mant;
                    weight_loaded        <= 1'b1;
                    state                <= STATE_READ_HEADS;
                end
            end

            STATE_READ_HEADS: begin
                if (dim_idx < HEAD_DIM) begin
                    // 拉高读使能
                    accum_rd_en_h0 <= 1'b1; accum_rd_row_h0 <= token_idx; accum_rd_dim_h0 <= dim_idx;
                    accum_rd_en_h1 <= 1'b1; accum_rd_row_h1 <= token_idx; accum_rd_dim_h1 <= dim_idx;
                    accum_rd_en_h2 <= 1'b1; accum_rd_row_h2 <= token_idx; accum_rd_dim_h2 <= dim_idx;
                    accum_rd_en_h3 <= 1'b1; accum_rd_row_h3 <= token_idx; accum_rd_dim_h3 <= dim_idx;

                    read_done_mask    <= 4'b0000;
                    read_wait_counter <= 4'd0;
                    state             <= STATE_WAIT_READ;
                end else begin
                    dim_idx <= 3'd0;
                    state   <= STATE_ALIGN_CONCAT;
                end
            end

            STATE_WAIT_READ: begin
                // 【显式拉低】产生脉冲
                accum_rd_en_h0 <= 1'b0;
                accum_rd_en_h1 <= 1'b0;
                accum_rd_en_h2 <= 1'b0;
                accum_rd_en_h3 <= 1'b0;

                // 收集数据
                if (accum_rd_valid_h0 && !read_done_mask[0]) begin
                    heads_mant[0][dim_idx] <= accum_rd_mant_h0;
                    if (dim_idx == 3'd0) heads_exp[0] <= accum_rd_exp_h0;
                    read_done_mask[0] <= 1'b1;
                end
                if (accum_rd_valid_h1 && !read_done_mask[1]) begin
                    heads_mant[1][dim_idx] <= accum_rd_mant_h1;
                    if (dim_idx == 3'd0) heads_exp[1] <= accum_rd_exp_h1;
                    read_done_mask[1] <= 1'b1;
                end
                if (accum_rd_valid_h2 && !read_done_mask[2]) begin
                    heads_mant[2][dim_idx] <= accum_rd_mant_h2;
                    if (dim_idx == 3'd0) heads_exp[2] <= accum_rd_exp_h2;
                    read_done_mask[2] <= 1'b1;
                end
                if (accum_rd_valid_h3 && !read_done_mask[3]) begin
                    heads_mant[3][dim_idx] <= accum_rd_mant_h3;
                    if (dim_idx == 3'd0) heads_exp[3] <= accum_rd_exp_h3;
                    read_done_mask[3] <= 1'b1;
                end

                if (&read_done_mask) begin
                    dim_idx <= dim_idx + 3'd1;
                    state   <= STATE_READ_HEADS;
                end else begin
                    read_wait_counter <= read_wait_counter + 4'd1;
                    if (read_wait_counter > 4'd15) state <= STATE_IDLE; // 超时保护
                end
            end

            STATE_ALIGN_CONCAT: begin
                state <= STATE_SEND_CE;
            end

            STATE_SEND_CE: begin
                if (ce_input_ready && !ce_input_valid) begin
                    ce_input_valid <= 1'b1; // 拉高 Valid
                    state          <= STATE_WAIT_CE;
                end
            end

            STATE_WAIT_CE: begin
                ce_input_valid <= 1'b0; // 【显式拉低】

                if (ce_compute_done) begin
                    // 锁存结果
                    ce_out_valids_r   <= ce_result_valids;
                    ce_out_fixed_r    <= ce_result_fixed_array;
                    ce_out_base_exp_r <= ce_result_base_exp_array;
                    ce_out_zero_r     <= ce_result_zero_array;
                    state             <= STATE_BFP_CONVERT;
                end
            end

            STATE_BFP_CONVERT: begin
                if (bfp_convert_done) begin
                    state <= STATE_WRITE_RESULT;
                end
            end

            STATE_WRITE_RESULT: begin
                result_wr_en   <= 1'b1; // 拉高写使能
                result_wr_addr <= global_token_addr + token_idx;
                result_exp     <= bfp_shared_exp;
                result_mant    <= bfp_mants;
                state          <= STATE_NEXT_TOKEN;
            end

            STATE_NEXT_TOKEN: begin
                result_wr_en <= 1'b0; // 【显式拉低】
                
                if (token_idx < tokens_in_batch - 1) begin
                    token_idx <= token_idx + 5'd1;
                    dim_idx   <= 3'd0;
                    state     <= STATE_READ_HEADS;
                end else begin
                    state <= STATE_DONE;
                end
            end

            STATE_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= STATE_IDLE;
            end

            default: state <= STATE_IDLE;
        endcase
    end
end

//------------------------------------------------------------------------------
// 指数对齐 + Concat（组合逻辑）
//------------------------------------------------------------------------------
always @(*) begin:dui
    integer ih, id;
    // 1) 找最大指数
    max_exp = heads_exp[0];
    for (ih = 1; ih < NUM_HEADS; ih = ih + 1) begin
        if (heads_exp[ih] > max_exp) max_exp = heads_exp[ih];
    end

    // 2) 尾数右移对齐
    for (ih = 0; ih < NUM_HEADS; ih = ih + 1) begin
        for (id = 0; id < HEAD_DIM; id = id + 1) begin
            if (max_exp > heads_exp[ih])
                aligned_mants[ih][id] = heads_mant[ih][id] >>> (max_exp - heads_exp[ih]);
            else
                aligned_mants[ih][id] = heads_mant[ih][id];
        end
    end

    // 3) Concat
    for (ih = 0; ih < NUM_HEADS; ih = ih + 1) begin
        for (id = 0; id < HEAD_DIM; id = id + 1) begin
            concat_mant[ih*HEAD_DIM + id] = aligned_mants[ih][id];
        end
    end

    // 4) 共享指数
    concat_exp = max_exp;
end

endmodule