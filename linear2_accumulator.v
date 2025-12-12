`timescale 1ns / 1ps

//================================================================================
// Linear2 Accumulator - 多 Token 指数对齐版 (Verilog-2001)
//
// - 输入：每次一个 "chunk" 的 Linear2 部分结果
//      * partial_exp  : TOKEN_CHUNK 个 token，各自一个 BFP 指数
//      * partial_mant : TOKEN_CHUNK × OUTPUT_DIM 个元素，每个 INPUT_MANT_W 位
// - 内部：对每个 token 独立做 BFP 指数对齐，然后对所有元素累加
// - 输出：
//      * result_exp   : TOKEN_CHUNK 个 token 的最终 BFP 指数
//      * result_mant  : TOKEN_CHUNK × OUTPUT_DIM 个元素，每个 OUTPUT_MANT_W 位
//
// 接口与 ffn_backbone_top / linear_compute_engine 对齐：
//   partial_exp, result_exp : [TOKEN_CHUNK*BFP_EXP_W-1:0]
//================================================================================

module linear2_accumulator #(
    parameter TOKEN_CHUNK    = 32,
    parameter OUTPUT_DIM     = 32,
    parameter BFP_EXP_W      = 8,
    parameter INPUT_MANT_W   = 15,
    parameter OUTPUT_MANT_W  = 15,
    parameter NUM_CHUNKS     = 4
)(
    input  wire clk,
    input  wire rst_n,

    // 控制
    input  wire clear,     // 清空累加器（新 token 块）
    input  wire enable,    // 本次分块结果有效（做一次累加）

    // 当前分块结果（每个 token 一个指数）
    input  wire [TOKEN_CHUNK*BFP_EXP_W-1:0] partial_exp,
    input  wire [TOKEN_CHUNK*OUTPUT_DIM*INPUT_MANT_W-1:0] partial_mant,

    // 完整累加结果（每个 token 一个指数）
    output reg  [TOKEN_CHUNK*BFP_EXP_W-1:0] result_exp,
    output reg  [TOKEN_CHUNK*OUTPUT_DIM*OUTPUT_MANT_W-1:0] result_mant,
    output reg  result_valid,

    // 调试：已经累加的分块数
    output reg  [2:0] debug_accum_count
);

    // token × 输出维度
    localparam TOTAL_ELEMENTS = TOKEN_CHUNK * OUTPUT_DIM;

    //================================================================
    // 内部寄存器
    //================================================================

    // 累加缓冲：保存对齐后的尾数（每个元素）
    reg signed [OUTPUT_MANT_W-1:0] accum_buffer [0:TOTAL_ELEMENTS-1];

    // 各 token 的当前共享指数（向量形式）
    reg [TOKEN_CHUNK*BFP_EXP_W-1:0] accum_exp_vec;

    // 累加的分块计数
    reg [2:0] accum_count;

    // 输入 mantissa 拆包
    reg signed [INPUT_MANT_W-1:0] partial_mant_unpacked [0:TOTAL_ELEMENTS-1];

    // 每个 token 的指数对齐中间量
    reg [BFP_EXP_W-1:0] accum_exp_arr     [0:TOKEN_CHUNK-1];
    reg [BFP_EXP_W-1:0] partial_exp_arr   [0:TOKEN_CHUNK-1];
    reg [BFP_EXP_W:0]   diff_val_arr      [0:TOKEN_CHUNK-1];
    reg                 partial_ge_accum  [0:TOKEN_CHUNK-1];
    reg [5:0]           shift_amount_arr  [0:TOKEN_CHUNK-1];
    reg [BFP_EXP_W-1:0] new_shared_exp_arr[0:TOKEN_CHUNK-1];

    // 对齐后的尾数
    reg signed [OUTPUT_MANT_W-1:0] accum_aligned   [0:TOTAL_ELEMENTS-1];
    reg signed [OUTPUT_MANT_W-1:0] partial_aligned [0:TOTAL_ELEMENTS-1];

    // 临时扩展后的输入尾数
    reg signed [OUTPUT_MANT_W-1:0] pm_ext;

    integer i;
    integer t;
    integer j;
    integer idx;

    //================================================================
    // 1) 拆包：partial_mant -> partial_mant_unpacked
    //================================================================
    always @(*) begin
        for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
            partial_mant_unpacked[i] =
                partial_mant[i*INPUT_MANT_W +: INPUT_MANT_W];
        end
    end

    //================================================================
    // 2) 指数对齐（组合逻辑，逐 token）
    //================================================================
    always @(*) begin
        // --- 2.1 先对每个 token 计算指数差和右移量 ---
        for (t = 0; t < TOKEN_CHUNK; t = t + 1) begin
            accum_exp_arr[t]   = accum_exp_vec[t*BFP_EXP_W +: BFP_EXP_W];
            partial_exp_arr[t] = partial_exp   [t*BFP_EXP_W +: BFP_EXP_W];

            if (partial_exp_arr[t] >= accum_exp_arr[t]) begin
                partial_ge_accum[t]   = 1'b1;
                diff_val_arr[t]       = partial_exp_arr[t] - accum_exp_arr[t];
                new_shared_exp_arr[t] = partial_exp_arr[t];
            end else begin
                partial_ge_accum[t]   = 1'b0;
                diff_val_arr[t]       = accum_exp_arr[t] - partial_exp_arr[t];
                new_shared_exp_arr[t] = accum_exp_arr[t];
            end

            // 限制右移量
            if (diff_val_arr[t] >= OUTPUT_MANT_W)
                shift_amount_arr[t] = OUTPUT_MANT_W;    // 自动截成 6 位
            else
                shift_amount_arr[t] = diff_val_arr[t][5:0];
        end

        // --- 2.2 根据每个 token 的 shift_amount，对应 token 的元素做对齐 ---
        for (t = 0; t < TOKEN_CHUNK; t = t + 1) begin
            for (j = 0; j < OUTPUT_DIM; j = j + 1) begin
                idx = t * OUTPUT_DIM + j;

                // 扩展输入尾数到 OUTPUT_MANT_W 位
                if (OUTPUT_MANT_W > INPUT_MANT_W) begin
                    pm_ext = {{(OUTPUT_MANT_W-INPUT_MANT_W){partial_mant_unpacked[idx][INPUT_MANT_W-1]}},
                              partial_mant_unpacked[idx]};
                end else begin
                    pm_ext = partial_mant_unpacked[idx][INPUT_MANT_W-1 -: OUTPUT_MANT_W];
                end

                if (partial_ge_accum[t]) begin
                    // partial 的指数更大：对齐 accum_buffer
                    if (shift_amount_arr[t] >= OUTPUT_MANT_W)
                        accum_aligned[idx] = {OUTPUT_MANT_W{1'b0}};
                    else
                        accum_aligned[idx] = accum_buffer[idx] >>> shift_amount_arr[t];

                    partial_aligned[idx] = pm_ext;  // partial 不需要右移
                end else begin
                    // accum 的指数更大：对齐 partial
                    if (shift_amount_arr[t] >= OUTPUT_MANT_W)
                        partial_aligned[idx] = {OUTPUT_MANT_W{1'b0}};
                    else
                        partial_aligned[idx] = pm_ext >>> shift_amount_arr[t];

                    accum_aligned[idx] = accum_buffer[idx]; // accum 不动
                end
            end
        end
    end

    //================================================================
    // 3) 累加逻辑（时序）
    //================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
                accum_buffer[i] <= {OUTPUT_MANT_W{1'b0}};
            end
            accum_exp_vec      <= {TOKEN_CHUNK*BFP_EXP_W{1'b0}};
            accum_count        <= 3'd0;
            result_valid       <= 1'b0;
            debug_accum_count  <= 3'd0;

        end else begin
            if (clear) begin
                // 清空累加器
                for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
                    accum_buffer[i] <= {OUTPUT_MANT_W{1'b0}};
                end
                accum_exp_vec      <= {TOKEN_CHUNK*BFP_EXP_W{1'b0}};
                accum_count        <= 3'd0;
                result_valid       <= 1'b0;
                debug_accum_count  <= 3'd0;

            end else if (enable) begin
                // 第一次累加：直接把 partial 写入 buffer，并记录各 token 的指数
                if (accum_count == 3'd0) begin
                    for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
                        if (OUTPUT_MANT_W > INPUT_MANT_W) begin
                            accum_buffer[i] <=
                                {{(OUTPUT_MANT_W-INPUT_MANT_W){partial_mant_unpacked[i][INPUT_MANT_W-1]}},
                                  partial_mant_unpacked[i]};
                        end else begin
                            accum_buffer[i] <=
                                partial_mant_unpacked[i][INPUT_MANT_W-1 -: OUTPUT_MANT_W];
                        end
                    end

                    accum_exp_vec <= partial_exp;   // 每个 token 的指数
                    accum_count   <= 3'd1;
                    result_valid  <= 1'b0;
                    debug_accum_count <= 3'd1;

                end else if (accum_count < NUM_CHUNKS) begin
                    // 后续累加：用对齐后的值相加
                    for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
                        accum_buffer[i] <= accum_aligned[i] + partial_aligned[i];
                    end

                    // 更新每个 token 的共享指数
                    for (t = 0; t < TOKEN_CHUNK; t = t + 1) begin
                        accum_exp_vec[t*BFP_EXP_W +: BFP_EXP_W] <= new_shared_exp_arr[t];
                    end

                    accum_count       <= accum_count + 3'd1;
                    debug_accum_count <= accum_count + 3'd1;

                    if (accum_count + 3'd1 == NUM_CHUNKS)
                        result_valid <= 1'b1;
                    else
                        result_valid <= 1'b0;
                end
            end
        end
    end

    //================================================================
    // 4) 输出打包：result_exp / result_mant
    //================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_exp  <= {TOKEN_CHUNK*BFP_EXP_W{1'b0}};
            result_mant <= {TOKEN_CHUNK*OUTPUT_DIM*OUTPUT_MANT_W{1'b0}};
        end else if (result_valid) begin
            result_exp <= accum_exp_vec;  // 按 token 输出指数

            for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
                result_mant[i*OUTPUT_MANT_W +: OUTPUT_MANT_W] <= accum_buffer[i];
            end
        end
    end

endmodule
