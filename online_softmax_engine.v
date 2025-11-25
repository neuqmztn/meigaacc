`timescale 1ns / 1ps

// ============================================================================
// online_softmax_engine（多 chunk 在线 Softmax 核心）
//
// 功能（FlashAttention 风格的统计部分）：
//   - 对同一行 scores 的多个 chunk 顺序处理，维护全局的：
//       m_new = max(m_prev, m_chunk)
//       l_new = l_prev * exp(m_prev - m_new) + Σ_j exp(score_j - m_new)
//   - 输出：
//       1) 写回 softmax_stats_buffer 的 m_new / l_new
//       2) 当前 chunk 中各列的 "分子"：exp(score_j - m_new)
//       3) 当 m_new > m_prev 且非首块时，给出 renorm_scale = exp(m_prev - m_new)
//          供下游对旧 O 做重标定（可选用）
//
// 说明：
//   - 该模块本身不做 / l_new，因此 weights_packed 不是最终权重，
//     而是 softmax 的未归一化分子（Q14 定点）。
//   - 真正的权重 w_j = exp(score_j - m_new) / l_new 需要在后级计算。
// ============================================================================

module online_softmax_engine #(
    parameter NUM_HEADS    = 4,
    parameter TOKEN_BATCH  = 32,
    parameter K_CHUNK_SIZE = 32,   // 每个 chunk 的 K 上限
    parameter DATA_WIDTH   = 8,    // 这里未用，可预留
    parameter EXP_WIDTH    = 8,    // 这里未用，可预留 BFP 指数
    parameter SCORE_WIDTH  = 16,   // score / 分子 使用位宽（建议 Q2.14）
    parameter ACCUM_WIDTH  = 24    // 统计 l_new 使用位宽
)(
    input  wire clk,
    input  wire rst_n,

    // 控制接口（由 batch wrapper 驱动）
    input  wire        start,        // 拉高一个周期，启动本行本 chunk 计算
    input  wire [1:0]  head_idx,     // 哪个 head 的统计（方便写回）
    input  wire [4:0]  row_idx,      // 哪一行 query（0..TOKEN_BATCH-1）
    input  wire [4:0]  chunk_id,     // 当前 chunk id（调试用）
    input  wire        chunk_first,  // 1：该行的第一个 chunk
    input  wire        chunk_last,   // 1：该行的最后一个 chunk（仅供上层参考）
    input  wire [5:0]  chunk_size,   // 本 chunk 内实际列数（<= K_CHUNK_SIZE）

    output reg         done,         // 本 chunk 计算完成（单周期脉冲）
    output reg         busy,         // 该 engine 正在处理

    // Scores 输入（每次处理一行的一个 chunk）
    input  wire [EXP_WIDTH-1:0]                scores_shared_exp,   // 保留接口，当前未使用
    input  wire [K_CHUNK_SIZE*SCORE_WIDTH-1:0] scores_mants_packed, // 扁平化 scores[j]，按列打包

    // 旧统计量（从 softmax_stats_buffer 读出）
    input  wire signed [SCORE_WIDTH-1:0]       max_rd_value,  // old m_prev
    input  wire signed [ACCUM_WIDTH-1:0]       sum_rd_value,  // old l_prev

    // 新统计量（写回 softmax_stats_buffer）
    output reg        max_wr_en,
    output reg  [1:0] max_wr_head,
    output reg  [4:0] max_wr_row,
    output reg  signed [SCORE_WIDTH-1:0] max_wr_value,  // m_new

    output reg        sum_wr_en,
    output reg  [1:0] sum_wr_head,
    output reg  [4:0] sum_wr_row,
    output reg  signed [ACCUM_WIDTH-1:0] sum_wr_value,  // l_new（截位）

    // 若 m_new > m_prev，则需对旧 O 乘 renorm_scale
    output reg        renorm_en,                      // 1：需要重标定
    output reg  signed [SCORE_WIDTH-1:0] renorm_scale, // α_prev = exp(m_prev - m_new)，Q14

    // 当前 chunk 输出的"分子"数组：exp(score_j - m_new)
    output reg        weights_valid,   // 当前 chunk 的分子数组就绪
    output wire [K_CHUNK_SIZE*SCORE_WIDTH-1:0] weights_packed
);

    // --------------------------------------------------------------------
    // 本地参数
    // --------------------------------------------------------------------
    localparam integer FRAC_BITS = 14;      // 约定 Q2.14
    localparam signed [SCORE_WIDTH-1:0] NEG_INF = -16'sh7FFF;
    localparam signed [SCORE_WIDTH-1:0] ONE_Q   = 16'sd1 <<< FRAC_BITS;
    localparam integer INTERNAL_ACC_WIDTH = ACCUM_WIDTH + 4;

    // 状态机
    localparam [2:0]
        S_IDLE      = 3'd0,
        S_LOAD      = 3'd1,
        S_FIND      = 3'd2,
        S_PREP      = 3'd3,
        S_CLEAR_EXP = 3'd4,
        S_EXP_SUM   = 3'd5,
        S_WRITE     = 3'd6,
        S_DONE      = 3'd7;

    reg [2:0] state;

    // --------------------------------------------------------------------
    // 内部寄存器 / 数组
    // --------------------------------------------------------------------
    // 展开 scores
    reg signed [SCORE_WIDTH-1:0] scores_array [0:K_CHUNK_SIZE-1];
    // 当前 chunk 对应 exp(score - m_new) 分子
    reg signed [SCORE_WIDTH-1:0] exp_array    [0:K_CHUNK_SIZE-1];

    // 统计量寄存器
    reg signed [SCORE_WIDTH-1:0] old_max;     // m_prev
    reg signed [SCORE_WIDTH-1:0] new_max;     // m_new
    reg signed [SCORE_WIDTH-1:0] chunk_max;   // m_chunk

    reg signed [INTERNAL_ACC_WIDTH-1:0] old_sum_internal; // 扩展后的 l_prev
    reg signed [INTERNAL_ACC_WIDTH-1:0] new_sum_internal; // l_new 内部累加

    // FlashAttention 的缩放因子
    reg signed [SCORE_WIDTH-1:0] alpha_prev;   // = exp(m_prev  - m_new)
    reg signed [SCORE_WIDTH-1:0] beta_chunk;   // = exp(m_chunk - m_new)

    // 循环指标
    reg [5:0] idx;

    // 用于 S_PREP
    reg signed [SCORE_WIDTH-1:0] m_new_next;
    reg signed [SCORE_WIDTH-1:0] dm_prev;
    reg signed [SCORE_WIDTH-1:0] dm_chunk;
    reg signed [SCORE_WIDTH-1:0] scale_prev_q;
    reg signed [SCORE_WIDTH-1:0] scale_chunk_q;
    reg signed [INTERNAL_ACC_WIDTH-1:0] old_sum_scaled;

    // 用于 S_EXP_SUM
    reg signed [SCORE_WIDTH-1:0] delta_local;
    reg signed [SCORE_WIDTH-1:0] exp_local;
    reg signed [2*SCORE_WIDTH-1:0] mult_tmp;
    reg signed [SCORE_WIDTH-1:0] exp_global;

    integer i;

    // --------------------------------------------------------------------
    // 解包 scores_mants_packed
    // --------------------------------------------------------------------
    always @(*) begin
        for (i = 0; i < K_CHUNK_SIZE; i = i + 1) begin
            scores_array[i] = scores_mants_packed[i*SCORE_WIDTH +: SCORE_WIDTH];
        end
    end

    // --------------------------------------------------------------------
    // 打包 exp_array -> weights_packed
    // --------------------------------------------------------------------
    genvar gv;
    generate
        for (gv = 0; gv < K_CHUNK_SIZE; gv = gv + 1) begin : GEN_PACK
            assign weights_packed[gv*SCORE_WIDTH +: SCORE_WIDTH] = exp_array[gv];
        end
    endgenerate

    // --------------------------------------------------------------------
    // exp 近似函数（Q14，针对 x <= 0 区域）:
    //   exp(x) ≈ 1 + x + x^2/2
    //   输入/输出均视为 Q2.14
    // --------------------------------------------------------------------
    function signed [SCORE_WIDTH-1:0] exp_approx_q14;
        input signed [SCORE_WIDTH-1:0] x;
        reg   signed [2*SCORE_WIDTH-1:0] x2;
        reg   signed [2*SCORE_WIDTH-1:0] term2;
        reg   signed [SCORE_WIDTH:0]     res_ext;
    begin
        if (x <= -16'sd8 <<< FRAC_BITS) begin
            exp_approx_q14 = {SCORE_WIDTH{1'b0}};
        end else if (x >= 0) begin
            exp_approx_q14 = ONE_Q;
        end else begin
            x2    = x * x;
            term2 = x2 >>> (FRAC_BITS+1); // x^2 / 2

            res_ext = {ONE_Q[15], ONE_Q} + {{1{x[15]}}, x} +
                      {term2[2*SCORE_WIDTH-1], term2[2*SCORE_WIDTH-1 -: SCORE_WIDTH]};

            exp_approx_q14 = res_ext[SCORE_WIDTH-1:0];
            if (exp_approx_q14[SCORE_WIDTH-1])
                exp_approx_q14 = {SCORE_WIDTH{1'b0}};
        end
    end
    endfunction

    // --------------------------------------------------------------------
    // 主状态机
    // --------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            busy            <= 1'b0;

            max_wr_en       <= 1'b0;
            sum_wr_en       <= 1'b0;
            max_wr_head     <= 2'd0;
            max_wr_row      <= 5'd0;
            max_wr_value    <= {SCORE_WIDTH{1'b0}};
            sum_wr_head     <= 2'd0;
            sum_wr_row      <= 5'd0;
            sum_wr_value    <= {ACCUM_WIDTH{1'b0}};

            renorm_en       <= 1'b0;
            renorm_scale    <= ONE_Q;
            weights_valid   <= 1'b0;

            old_max         <= NEG_INF;
            new_max         <= NEG_INF;
            chunk_max       <= NEG_INF;
            old_sum_internal<= {INTERNAL_ACC_WIDTH{1'b0}};
            new_sum_internal<= {INTERNAL_ACC_WIDTH{1'b0}};

            alpha_prev      <= ONE_Q;
            beta_chunk      <= ONE_Q;
            idx             <= 6'd0;

            for (i = 0; i < K_CHUNK_SIZE; i = i + 1) begin
                exp_array[i] <= {SCORE_WIDTH{1'b0}};
            end
        end else begin
            // 单周期脉冲默认清零
            done          <= 1'b0;
            max_wr_en     <= 1'b0;
            sum_wr_en     <= 1'b0;
            weights_valid <= 1'b0;
            renorm_en     <= 1'b0;

            case (state)
                //----------------------------------------------------------
                // S_IDLE：等待 start
                //----------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy             <= 1'b1;
                        idx              <= 6'd0;
                        chunk_max        <= NEG_INF;
                        new_sum_internal <= {INTERNAL_ACC_WIDTH{1'b0}};

                        // 读取旧统计量
                        if (chunk_first) begin
                            old_max         <= NEG_INF;
                            old_sum_internal<= {INTERNAL_ACC_WIDTH{1'b0}};
                        end else begin
                            old_max <= max_rd_value;
                            old_sum_internal <= {{(INTERNAL_ACC_WIDTH-ACCUM_WIDTH){sum_rd_value[ACCUM_WIDTH-1]}},
                                                  sum_rd_value};
                        end
                        state <= S_FIND;
                    end
                end

                //----------------------------------------------------------
                // S_FIND：遍历当前 chunk 找 chunk_max
                //----------------------------------------------------------
                S_FIND: begin
                    if (idx < chunk_size) begin
                        if (scores_array[idx] > chunk_max)
                            chunk_max <= scores_array[idx];
                        idx <= idx + 6'd1;
                    end else begin
                        state <= S_PREP;
                    end
                end

                //----------------------------------------------------------
                // S_PREP：计算 m_new / alpha_prev / beta_chunk / 初始化 new_sum
                //----------------------------------------------------------
                S_PREP: begin
                    if (chunk_first) begin
                        new_max         <= chunk_max;
                        alpha_prev      <= ONE_Q;
                        beta_chunk      <= ONE_Q;
                        new_sum_internal<= {INTERNAL_ACC_WIDTH{1'b0}};
                    end else begin
                        // m_new = max(old_max, chunk_max)
                        if (old_max > chunk_max)
                            m_new_next = old_max;
                        else
                            m_new_next = chunk_max;

                        dm_prev   = old_max   - m_new_next;
                        dm_chunk  = chunk_max - m_new_next;
                        scale_prev_q  = exp_approx_q14(dm_prev);
                        scale_chunk_q = exp_approx_q14(dm_chunk);

                        alpha_prev   <= scale_prev_q;
                        beta_chunk   <= scale_chunk_q;
                        new_max      <= m_new_next;

                        // l_prev * alpha_prev
                        old_sum_scaled =
                            (old_sum_internal *
                             {{(INTERNAL_ACC_WIDTH-SCORE_WIDTH){scale_prev_q[SCORE_WIDTH-1]}},
                               scale_prev_q}) >>> FRAC_BITS;
                        new_sum_internal <= old_sum_scaled;
                    end

                    idx   <= 6'd0;
                    state <= S_CLEAR_EXP;
                end

                //----------------------------------------------------------
                // S_CLEAR_EXP：清零 exp_array，防止 chunk_size < K_CHUNK_SIZE 时脏数据残留
                //----------------------------------------------------------
                S_CLEAR_EXP: begin
                    if (idx < K_CHUNK_SIZE[5:0]) begin
                        exp_array[idx] <= {SCORE_WIDTH{1'b0}};
                        idx <= idx + 6'd1;
                    end else begin
                        idx   <= 6'd0;
                        state <= S_EXP_SUM;
                    end
                end

                //----------------------------------------------------------
                // S_EXP_SUM：计算当前 chunk 的 exp(score - m_new) 并累加到 new_sum_internal
                //----------------------------------------------------------
                S_EXP_SUM: begin
                    if (idx < chunk_size) begin
                        // 先以 chunk_max 为基准算 exp(score - chunk_max)
                        delta_local = scores_array[idx] - chunk_max;
                        exp_local   = exp_approx_q14(delta_local);

                        if (chunk_first) begin
                            exp_global = exp_local; // m_new == chunk_max
                        end else begin
                            // exp(score - m_new) = exp(score - chunk_max) * beta_chunk
                            mult_tmp   = exp_local * beta_chunk;
                            exp_global = mult_tmp >>> FRAC_BITS;
                        end

                        exp_array[idx] <= exp_global;
                        new_sum_internal <= new_sum_internal +
                            {{(INTERNAL_ACC_WIDTH-SCORE_WIDTH){exp_global[SCORE_WIDTH-1]}},
                              exp_global};

                        idx <= idx + 6'd1;
                    end else begin
                        state <= S_WRITE;
                    end
                end

                //----------------------------------------------------------
                // S_WRITE：写回 (m_new, l_new)，并给出 renorm_scale
                //----------------------------------------------------------
                S_WRITE: begin
                    // 写 max
                    max_wr_en    <= 1'b1;
                    max_wr_head  <= head_idx;
                    max_wr_row   <= row_idx;
                    max_wr_value <= new_max;

                    // 写 sum（做简单饱和截位）
                    sum_wr_en    <= 1'b1;
                    sum_wr_head  <= head_idx;
                    sum_wr_row   <= row_idx;
                    if (new_sum_internal[INTERNAL_ACC_WIDTH-1]) begin
                        // 负溢出 -> 最小负值
                        sum_wr_value <= {1'b1, {(ACCUM_WIDTH-1){1'b0}}};
                    end else if (new_sum_internal >
                                 {{(INTERNAL_ACC_WIDTH-ACCUM_WIDTH){1'b0}},
                                   {1'b0, {(ACCUM_WIDTH-1){1'b1}}}}) begin
                        // 正溢出 -> 最大正值
                        sum_wr_value <= {1'b0, {(ACCUM_WIDTH-1){1'b1}}};
                    end else begin
                        sum_wr_value <= new_sum_internal[ACCUM_WIDTH-1:0];
                    end

                    // renorm：仅当非首块且 m_new > old_max 时，才需要对旧 O 乘 α_prev
                    if (!chunk_first && (new_max > old_max)) begin
                        renorm_en    <= 1'b1;
                        renorm_scale <= alpha_prev;
                    end else begin
                        renorm_en    <= 1'b0;
                        renorm_scale <= ONE_Q;
                    end

                    state <= S_DONE;
                end

                //----------------------------------------------------------
                // S_DONE：本 chunk 完成，weights_packed 中的分子就绪
                //----------------------------------------------------------
                S_DONE: begin
                    busy          <= 1'b0;
                    done          <= 1'b1;
                    weights_valid <= 1'b1;  // exp_array 已经填好
                    state         <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
