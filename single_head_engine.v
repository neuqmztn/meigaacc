`timescale 1ns / 1ps

// ============================================================================
// 单 Head 引擎（FlashAttention 版，修复 stats_buffer 接口）
// ============================================================================

module single_head_engine #(
    parameter HEAD_ID       = 0,
    parameter NUM_CHUNKS    = 21,
    parameter TOKEN_BATCH   = 32,
    parameter CHUNK_SIZE    = 32,
    parameter HEAD_DIM      = 8,
    parameter DATA_WIDTH    = 8,
    parameter EXP_WIDTH     = 8,
    parameter SCORE_WIDTH   = 8,
    parameter ACCUM_WIDTH   = 24,

    // CE 配置
    parameter NUM_PE = 1,
    parameter PE_TYPE_0 = 2,
    parameter PE_TYPE_1 = 2,
    parameter ELEM_PE0 = 8,
    parameter ELEM_PE1 = 8,
    parameter CE_OUTPUT_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,

    // 控制
    input  wire start,
    output reg  done,
    output reg  busy,

    // Q 读
    output wire q_rd_en,
    input  wire [(TOKEN_BATCH*EXP_WIDTH)-1:0]           q_batch_exp,
    input  wire [(TOKEN_BATCH*HEAD_DIM*DATA_WIDTH)-1:0] q_batch_mant,

    // K 读
    output reg  k_rd_en,
    output reg  [4:0] k_rd_chunk_id,
    input  wire       k_rd_valid,
    input  wire [(CHUNK_SIZE*EXP_WIDTH)-1:0]            k_chunk_exp,
    input  wire [(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)-1:0]  k_chunk_mant,

    // V 读
    output reg  v_rd_en,
    output reg  [4:0] v_rd_chunk_id,
    input  wire       v_rd_valid,
    input  wire [(CHUNK_SIZE*EXP_WIDTH)-1:0]            v_chunk_exp,
    input  wire [(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)-1:0]  v_chunk_mant,

    // 最终输出读（已经做过 /l_new）
    input  wire final_rd_en,
    input  wire [4:0] final_rd_row,
    input  wire [2:0] final_rd_dim,
    output wire signed [ACCUM_WIDTH-1:0] final_rd_mant,
    output wire [EXP_WIDTH-1:0]         final_rd_exp,
    output wire                         final_rd_valid,

    // 调试
    output wire [4:0]  current_chunk_id,
    output wire [31:0] dbg_accum_wr_count
);

    // ---------------- 状态机 ----------------
    localparam IDLE          = 4'd0;
    localparam READ_Q        = 4'd1;
    localparam READ_K        = 4'd2;
    localparam WAIT_K_VALID  = 4'd3;
    localparam QK_COMPUTE    = 4'd4;
    localparam WAIT_QK       = 4'd5;
    localparam SOFTMAX       = 4'd6;
    localparam WAIT_SM       = 4'd7;
    localparam READ_V        = 4'd8;
    localparam WAIT_V_VALID  = 4'd9;
    localparam APPLY_V       = 4'd10;
    localparam WAIT_AV       = 4'd11;
    localparam NEXT_CHUNK    = 4'd12;
    localparam DONE_ST       = 4'd13;

    reg [3:0] state;

    reg [4:0] chunk_counter;
    reg       chunk_first;
    reg       chunk_last;

    reg q_data_valid;
    assign q_rd_en = !q_data_valid && (state == READ_Q);

    // 子模块控制
    reg  qk_start;
    wire qk_done;
    wire qk_busy;
    wire qk_valid;

    reg  softmax_start;
    wire softmax_done;
    wire softmax_busy;
    wire weights_valid;

    reg  apply_v_start;
    wire apply_v_done;
    wire apply_v_busy;
    wire apply_v_valid;

    // ----------------------------------------------------------------
    // Scores / Weights
    // ----------------------------------------------------------------
    // 来自 QK 的"每行一个指数"（行级 shared exponent）
    wire [TOKEN_BATCH*EXP_WIDTH-1:0]              scores_exp_row;
    // 广播后的"每个 score 一个指数"（给 softmax 用）
    wire [TOKEN_BATCH*CHUNK_SIZE*EXP_WIDTH-1:0]   scores_exp;
    // 每个 score 的尾数
    wire [TOKEN_BATCH*CHUNK_SIZE*SCORE_WIDTH-1:0] scores_mants;
    // softmax 后的权重
    wire [TOKEN_BATCH*CHUNK_SIZE*SCORE_WIDTH-1:0] weights;

    // Apply V 输出 -> Accumulator
    wire [4:0]                     apply_v_query;
    wire [2:0]                     apply_v_dim;
    wire signed [ACCUM_WIDTH-1:0]  apply_v_mant;
    wire [EXP_WIDTH-1:0]           apply_v_exp;
    wire                           apply_v_first_chunk;

    // ------------ Stats Buffer 相关 ------------
    wire [4:0] sm_max_rd_row;
    wire [4:0] sm_sum_rd_row;
    wire [4:0] norm_sum_rd_row;
    wire [4:0] stats_max_rd_row;
    wire [4:0] stats_sum_rd_row;

    wire signed [SCORE_WIDTH-1:0]  max_rd_value;
    wire signed [ACCUM_WIDTH-1:0]  sum_rd_value;

    wire        max_wr_en;
    wire [4:0]  max_wr_row;
    wire signed [SCORE_WIDTH-1:0]  max_wr_value;

    wire        sum_wr_en;
    wire [4:0]  sum_wr_row;
    wire signed [ACCUM_WIDTH-1:0]  sum_wr_value;

    // Softmax 是否在运行，用于 mux sum_rd_row
    //（softmax_busy 由 online_softmax_batch 提供）
    assign stats_max_rd_row = sm_max_rd_row;
    assign stats_sum_rd_row = (softmax_busy) ? sm_sum_rd_row : norm_sum_rd_row;

    // renorm 批量信息（按行）
    wire [TOKEN_BATCH-1:0]             renorm_flag_batch;
    wire [TOKEN_BATCH*SCORE_WIDTH-1:0] renorm_scale_batch;

    // Accumulator 写口
    wire                     accum_wr_en;
    wire [4:0]               accum_wr_row;
    wire [2:0]               accum_wr_dim_wire;
    wire signed [ACCUM_WIDTH-1:0] accum_wr_mant;
    wire [EXP_WIDTH-1:0]     accum_wr_exp;
    wire                     accum_wr_first_chunk;

    // 对某一行选 renorm_scale
    wire                     accum_renorm_en;
    wire signed [SCORE_WIDTH-1:0] accum_renorm_scale;

    assign accum_renorm_en =
        renorm_flag_batch[apply_v_query];

    assign accum_renorm_scale =
        renorm_scale_batch[apply_v_query*SCORE_WIDTH +: SCORE_WIDTH];

    // ----------------------------------------------------------------
    // QK 批量计算
    // ----------------------------------------------------------------
    qk_compute_batch #(
        .NUM_QUERIES    (TOKEN_BATCH),
        .CHUNK_SIZE     (CHUNK_SIZE),
        .HEAD_DIM       (HEAD_DIM),
        .DATA_WIDTH     (DATA_WIDTH),
        .EXP_WIDTH      (EXP_WIDTH),
        .SCORE_WIDTH    (SCORE_WIDTH),
        .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
        .NUM_PE         (NUM_PE),
        .PE_TYPE_0      (PE_TYPE_0),
        .PE_TYPE_1      (PE_TYPE_1),
        .ELEM_PE0       (ELEM_PE0),
        .ELEM_PE1       (ELEM_PE1)
    ) u_qk_batch (
        .clk           (clk),
        .rst_n         (rst_n),

        .start         (qk_start),
        .done          (qk_done),
        .busy          (qk_busy),

        .q_batch_exp   (q_batch_exp),
        .q_batch_mant  (q_batch_mant),

        .k_chunk_exp   (k_chunk_exp),
        .k_chunk_mant  (k_chunk_mant),

        .scores_valid      (qk_valid),
        // ★ 来自 QK 的"每行一个指数"
        .scores_exp_batch  (scores_exp_row),
        .scores_batch      (scores_mants)
    );

    // ----------------------------------------------------------------
    // 指数广播：行级 exponent → 每个 score 的 exponent
    // ----------------------------------------------------------------
    genvar gi, gj;
    generate
        for (gi = 0; gi < TOKEN_BATCH; gi = gi + 1) begin
            for (gj = 0; gj < CHUNK_SIZE; gj = gj + 1) begin
                assign scores_exp[(gi*CHUNK_SIZE + gj)*EXP_WIDTH +: EXP_WIDTH] =
                       scores_exp_row[gi*EXP_WIDTH +: EXP_WIDTH];
            end
        end
    endgenerate

    // ----------------------------------------------------------------
    // Softmax 批处理（修复 stats_buffer 接口）
    // ----------------------------------------------------------------
    online_softmax_batch #(
        .NUM_QUERIES  (TOKEN_BATCH),
        .CHUNK_SIZE   (CHUNK_SIZE),
        .SCORE_WIDTH  (SCORE_WIDTH),
        .EXP_WIDTH    (EXP_WIDTH),
        .ACCUM_WIDTH  (ACCUM_WIDTH)
    ) u_softmax_batch (
        .clk         (clk),
        .rst_n       (rst_n),

        .start       (softmax_start),
        .done        (softmax_done),
        .busy        (softmax_busy),

        .chunk_first (chunk_first),
        .chunk_last  (chunk_last),

        // ★ 这里接"每个 score 的指数"
        .scores_exp_batch (scores_exp),
        .scores_batch     (scores_mants),

        .weights_valid    (weights_valid),
        .weights_batch    (weights),

        // 与 stats_buffer 对接：读 m/l
        .max_rd_row   (sm_max_rd_row),
        .max_rd_value (max_rd_value),

        .sum_rd_row   (sm_sum_rd_row),
        .sum_rd_value (sum_rd_value),

        // 写回 m/l
        .max_wr_en    (max_wr_en),
        .max_wr_row   (max_wr_row),
        .max_wr_value (max_wr_value),

        .sum_wr_en    (sum_wr_en),
        .sum_wr_row   (sum_wr_row),
        .sum_wr_value (sum_wr_value),

        // 行级 renorm 信息
        .renorm_flag_batch (renorm_flag_batch),
        .renorm_scale_batch(renorm_scale_batch)
    );

    // ----------------------------------------------------------------
    // Apply V
    // ----------------------------------------------------------------
    apply_v_batch #(
        .NUM_QUERIES (TOKEN_BATCH),
        .CHUNK_SIZE  (CHUNK_SIZE),
        .HEAD_DIM    (HEAD_DIM),
        .DATA_WIDTH  (DATA_WIDTH),
        .EXP_WIDTH   (EXP_WIDTH),
        .SCORE_WIDTH (SCORE_WIDTH),
        .ACCUM_WIDTH (ACCUM_WIDTH)
    ) u_apply_v_batch (
        .clk         (clk),
        .rst_n       (rst_n),

        .start       (apply_v_start),
        .done        (apply_v_done),
        .busy        (apply_v_busy),

        .chunk_first (chunk_first),

        .weights_batch (weights),

        .v_chunk_exp   (v_chunk_exp),
        .v_chunk_mant  (v_chunk_mant),

        .output_valid       (apply_v_valid),
        .output_query       (apply_v_query),
        .output_dim         (apply_v_dim),
        .output_mant        (apply_v_mant),
        .output_exp         (apply_v_exp),
        .output_first_chunk (apply_v_first_chunk)
    );

    // ----------------------------------------------------------------
    // Stats Buffer（单 head）
    // ----------------------------------------------------------------
    softmax_stats_buffer #(
        .TOKEN_BATCH(TOKEN_BATCH),
        .SCORE_WIDTH(SCORE_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH)
    ) u_stats_buffer (
        .clk          (clk),
        .rst_n        (rst_n),

        .max_wr_en    (max_wr_en),
        .max_wr_row   (max_wr_row),
        .max_wr_value (max_wr_value),

        .max_rd_row   (stats_max_rd_row),
        .max_rd_value (max_rd_value),

        .sum_wr_en    (sum_wr_en),
        .sum_wr_row   (sum_wr_row),
        .sum_wr_value (sum_wr_value),

        .sum_rd_row   (stats_sum_rd_row),
        .sum_rd_value (sum_rd_value)
    );

    // ----------------------------------------------------------------
    // Flash 累加器
    // ----------------------------------------------------------------
    wire signed [ACCUM_WIDTH-1:0] acc_rd_mant;
    wire [EXP_WIDTH-1:0]          acc_rd_exp;
    wire                          acc_rd_valid;

    attention_bfp_accumulator_flash #(
        .TOKEN_BATCH (TOKEN_BATCH),
        .HEAD_DIM    (HEAD_DIM),
        .EXP_WIDTH   (EXP_WIDTH),
        .ACCUM_WIDTH (ACCUM_WIDTH),
        .SCORE_WIDTH (SCORE_WIDTH)
    ) u_accumulator (
        .clk         (clk),
        .rst_n       (rst_n),

        .wr_en       (accum_wr_en),
        .wr_row      (accum_wr_row),
        .wr_dim      (accum_wr_dim_wire),
        .wr_mant     (accum_wr_mant),
        .wr_exp      (accum_wr_exp),
        .first_chunk (accum_wr_first_chunk),

        .renorm_en   (accum_renorm_en),
        .renorm_scale(accum_renorm_scale),

        .rd_en       (final_rd_en),
        .rd_row      (final_rd_row),
        .rd_dim      (final_rd_dim),
        .rd_mant     (acc_rd_mant),
        .rd_exp      (acc_rd_exp),
        .rd_valid    (acc_rd_valid)
    );

    // ----------------------------------------------------------------
    // 最终归一化
    // ----------------------------------------------------------------
    reg [4:0] acc_rd_row_reg;
    reg [2:0] acc_rd_dim_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_rd_row_reg <= 5'd0;
            acc_rd_dim_reg <= 3'd0;
        end else if (final_rd_en) begin
            acc_rd_row_reg <= final_rd_row;
            acc_rd_dim_reg <= final_rd_dim;
        end
    end

    assign norm_sum_rd_row = acc_rd_row_reg;

    wire signed [ACCUM_WIDTH-1:0] norm_out_mant;
    wire [EXP_WIDTH-1:0]          norm_out_exp;
    wire                          norm_out_valid;

    attention_output_normalizer #(
        .TOKEN_BATCH (TOKEN_BATCH),
        .HEAD_DIM    (HEAD_DIM),
        .EXP_WIDTH   (EXP_WIDTH),
        .ACCUM_WIDTH (ACCUM_WIDTH),
        .SCORE_WIDTH (SCORE_WIDTH)
    ) u_normalizer (
        .clk       (clk),
        .rst_n     (rst_n),

        .in_valid  (acc_rd_valid),
        .in_row    (acc_rd_row_reg),
        .in_dim    (acc_rd_dim_reg),
        .in_mant   (acc_rd_mant),
        .in_exp    (acc_rd_exp),

        .sum_value (sum_rd_value),

        .out_valid (norm_out_valid),
        .out_row   (),   // 如有需要可以接出去
        .out_dim   (),
        .out_mant  (norm_out_mant),
        .out_exp   (norm_out_exp)
    );

    assign final_rd_mant  = norm_out_mant;
    assign final_rd_exp   = norm_out_exp;
    assign final_rd_valid = norm_out_valid;

    // ----------------------------------------------------------------
    // Accumulator 写入控制（直接转接 Apply V）
    // ----------------------------------------------------------------
    reg [31:0] total_accum_writes;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_accum_writes <= 32'd0;
        end else if (apply_v_valid) begin
            total_accum_writes <= total_accum_writes + 32'd1;
        end
    end

    assign accum_wr_en          = apply_v_valid;
    assign accum_wr_row         = apply_v_query;
    assign accum_wr_dim_wire    = apply_v_dim;
    assign accum_wr_mant        = apply_v_mant;
    assign accum_wr_exp         = apply_v_exp;
    assign accum_wr_first_chunk = apply_v_first_chunk;

    // ----------------------------------------------------------------
    // 主状态机（chunk 流程）
    // ----------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            chunk_counter <= 5'd0;
            chunk_first   <= 1'b0;
            chunk_last    <= 1'b0;
            done          <= 1'b0;
            busy          <= 1'b0;
            q_data_valid  <= 1'b0;

            k_rd_en       <= 1'b0;
            k_rd_chunk_id <= 5'd0;
            v_rd_en       <= 1'b0;
            v_rd_chunk_id <= 5'd0;

            qk_start      <= 1'b0;
            softmax_start <= 1'b0;
            apply_v_start <= 1'b0;
        end else begin
            k_rd_en       <= 1'b0;
            v_rd_en       <= 1'b0;
            qk_start      <= 1'b0;
            softmax_start <= 1'b0;
            apply_v_start <= 1'b0;

            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy         <= 1'b1;
                        chunk_counter<= 5'd0;
                        chunk_first  <= 1'b1;
                        chunk_last   <= (NUM_CHUNKS == 1);
                        q_data_valid <= 1'b0;
                        state        <= READ_Q;
                    end else begin
                        busy <= 1'b0;
                    end
                end

                READ_Q: begin
                    if (!q_data_valid) begin
                        q_data_valid <= 1'b1;
                    end
                    state <= READ_K;
                end

                READ_K: begin
                    k_rd_en       <= 1'b1;
                    k_rd_chunk_id <= chunk_counter;
                    state         <= WAIT_K_VALID;
                end

                WAIT_K_VALID: begin
                    if (k_rd_valid) begin
                        state <= QK_COMPUTE;
                    end
                end

                QK_COMPUTE: begin
                    qk_start <= 1'b1;
                    state    <= WAIT_QK;
                end

                WAIT_QK: begin
                    if (qk_done) begin
                        state <= SOFTMAX;
                    end
                end

                SOFTMAX: begin
                    softmax_start <= 1'b1;
                    state         <= WAIT_SM;
                end

                WAIT_SM: begin
                    if (softmax_done) begin
                        state <= READ_V;
                    end
                end

                READ_V: begin
                    v_rd_en       <= 1'b1;
                    v_rd_chunk_id <= chunk_counter;
                    state         <= WAIT_V_VALID;
                end

                WAIT_V_VALID: begin
                    if (v_rd_valid) begin
                        state <= APPLY_V;
                    end
                end

                APPLY_V: begin
                    apply_v_start <= 1'b1;
                    state         <= WAIT_AV;
                end

                WAIT_AV: begin
                    if (apply_v_done) begin
                        state <= NEXT_CHUNK;
                    end
                end

                NEXT_CHUNK: begin
                    if (chunk_counter < NUM_CHUNKS-1) begin
                        chunk_counter <= chunk_counter + 5'd1;
                        chunk_first   <= 1'b0;
                        chunk_last    <= (chunk_counter == NUM_CHUNKS-2);
                        state         <= READ_K;
                    end else begin
                        state <= DONE_ST;
                    end
                end

                DONE_ST: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    state<= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    assign current_chunk_id   = chunk_counter;
    assign dbg_accum_wr_count = total_accum_writes;

endmodule
