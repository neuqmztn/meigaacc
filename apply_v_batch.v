`timescale 1ns / 1ps

// ============================================================================
//   输入：每行每列分子 weights_batch = exp(score - m_new)，Q14
//   - 输入：V chunk（BFP）
//   - 输出：Z_chunk(q,d) = Σ_k exp(score_{q,k} - m_new)*V_{k,d}，以 BFP 表示
// ============================================================================

module apply_v_batch #(
    parameter NUM_QUERIES  = 32,
    parameter CHUNK_SIZE   = 32,
    parameter HEAD_DIM     = 8,
    parameter DATA_WIDTH   = 8,
    parameter EXP_WIDTH    = 8,
    parameter SCORE_WIDTH  = 16,
    parameter ACCUM_WIDTH  = 24
)(
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    output reg  done,
    output reg  busy,

    input  wire chunk_first,

    // weights_batch: 展平为 (q * CHUNK_SIZE + k) * SCORE_WIDTH
    input  wire [NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH-1:0] weights_batch,
    // v_chunk_exp : k * EXP_WIDTH
    input  wire [CHUNK_SIZE*EXP_WIDTH-1:0]               v_chunk_exp,
    // v_chunk_mant: (k * HEAD_DIM + d) * DATA_WIDTH
    input  wire [CHUNK_SIZE*HEAD_DIM*DATA_WIDTH-1:0]     v_chunk_mant,

    // 输出：逐个 (q,d) 输出 BFP 结果
    output reg                      output_valid,
    output reg  [4:0]               output_query,
    output reg  [2:0]               output_dim,
    output reg  signed [ACCUM_WIDTH-1:0] output_mant,
    output reg  [EXP_WIDTH-1:0]     output_exp,
    output reg                      output_first_chunk
);

    localparam integer FRAC_BITS = 14;

    // 索引计数器
    reg [4:0] q_idx;
    reg [4:0] k_idx;
    reg [2:0] d_idx;

    // 当前 (q,d) 的 BFP 累加寄存器
    reg signed [ACCUM_WIDTH-1:0] acc_mant;
    reg        [EXP_WIDTH-1:0]   acc_exp;

    // FSM 状态编码
    localparam ST_IDLE   = 2'd0;
    localparam ST_ACCUM  = 2'd1;
    localparam ST_OUTPUT = 2'd2;

    reg [1:0] st;

    // ------------------------------------------------------------------------
    // 访问展开后的权重/向量的函数
    // ------------------------------------------------------------------------
    function signed [SCORE_WIDTH-1:0] get_weight;
        input [4:0] q;
        input [4:0] k;
        reg   [SCORE_WIDTH-1:0] tmp;
        integer idx_flat;
    begin
        idx_flat = (q*CHUNK_SIZE + k)*SCORE_WIDTH;
        tmp = weights_batch[idx_flat +: SCORE_WIDTH];
        get_weight = tmp;
    end
    endfunction

    function [EXP_WIDTH-1:0] get_v_exp;
        input [4:0] k;
        integer idx_flat;
        reg [EXP_WIDTH-1:0] tmp;
    begin
        idx_flat = k*EXP_WIDTH;
        tmp = v_chunk_exp[idx_flat +: EXP_WIDTH];
        get_v_exp = tmp;
    end
    endfunction

    function signed [DATA_WIDTH-1:0] get_v_mant;
        input [4:0] k;
        input [2:0] d;
        integer idx_flat;
        reg [DATA_WIDTH-1:0] tmp;
    begin
        idx_flat = (k*HEAD_DIM + d)*DATA_WIDTH;
        tmp = v_chunk_mant[idx_flat +: DATA_WIDTH];
        get_v_mant = tmp;
    end
    endfunction

    // ------------------------------------------------------------------------
    // 主时序进程：FSM + 累加 + 输出
    // ------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 异步复位：所有寄存器给确定初值
            st               <= ST_IDLE;
            busy             <= 1'b0;
            done             <= 1'b0;

            q_idx            <= 5'd0;
            k_idx            <= 5'd0;
            d_idx            <= 3'd0;

            acc_mant         <= {ACCUM_WIDTH{1'b0}};
            acc_exp          <= {EXP_WIDTH{1'b0}};

            output_valid     <= 1'b0;
            output_query     <= 5'd0;
            output_dim       <= 3'd0;
            output_mant      <= {ACCUM_WIDTH{1'b0}};
            output_exp       <= {EXP_WIDTH{1'b0}};
            output_first_chunk <= 1'b0;
        end else begin
            // 默认行为：这些信号单周期脉冲
            done         <= 1'b0;
            output_valid <= 1'b0;

            case (st)
                //==========================================================
                // 空闲：等待 start
                //==========================================================
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy    <= 1'b1;
                        q_idx   <= 5'd0;
                        d_idx   <= 3'd0;
                        k_idx   <= 5'd0;
                        acc_mant<= {ACCUM_WIDTH{1'b0}};
                        acc_exp <= get_v_exp(5'd0);
                        st      <= ST_ACCUM;

                        output_first_chunk <= chunk_first;
                    end
                end

                //==========================================================
                // 累加：对当前 (q_idx, d_idx) 做 k=0..CHUNK_SIZE-1 的 BFP 累加
                //==========================================================
                ST_ACCUM: begin : ACCUM_BLK
                    reg signed [SCORE_WIDTH-1:0]           w_q14;
                    reg        [EXP_WIDTH-1:0]             v_e;
                    reg signed [DATA_WIDTH-1:0]            v_m;
                    reg signed [SCORE_WIDTH+DATA_WIDTH-1:0] prod_q;
                    reg signed [ACCUM_WIDTH-1:0]           prod_shifted;

                    w_q14       = get_weight(q_idx, k_idx);
                    v_e         = get_v_exp(k_idx);
                    v_m         = get_v_mant(k_idx, d_idx);
                    prod_q      = w_q14 * v_m;
                    prod_shifted= prod_q >>> FRAC_BITS;

                    if (k_idx == 0) begin
                        // 第一项，直接写入
                        acc_mant <= prod_shifted;
                        acc_exp  <= v_e;
                    end else begin
                        if (v_e >= acc_exp) begin : ALIGN_ACC_UP
                            integer diff;
                            reg signed [ACCUM_WIDTH-1:0] acc_shift;
                            diff = v_e - acc_exp;
                            if (diff >= ACCUM_WIDTH)
                                acc_shift = {ACCUM_WIDTH{1'b0}};
                            else
                                acc_shift = acc_mant >>> diff;
                            acc_mant <= acc_shift + prod_shifted;
                            acc_exp  <= v_e;
                        end else begin : ALIGN_NEW_UP
                            integer diff2;
                            reg signed [ACCUM_WIDTH-1:0] prod_shift2;
                            diff2 = acc_exp - v_e;
                            if (diff2 >= ACCUM_WIDTH)
                                prod_shift2 = {ACCUM_WIDTH{1'b0}};
                            else
                                prod_shift2 = prod_shifted >>> diff2;
                            acc_mant <= acc_mant + prod_shift2;
                            // acc_exp 保持不变
                        end
                    end

                    // k 计数
                    if (k_idx < CHUNK_SIZE-1) begin
                        k_idx <= k_idx + 5'd1;
                    end else begin
                        // 本 (q,d) 的所有 k 累加完毕，下一拍进入输出
                        st    <= ST_OUTPUT;
                    end
                end

                //==========================================================
                // 输出：打一拍输出当前 (q_idx, d_idx) 的结果
                //==========================================================
                ST_OUTPUT: begin
                    // ---- 输出当前累加结果 ----
                    output_valid <= 1'b1;
                    output_query <= q_idx;
                    output_dim   <= d_idx;
                    output_mant  <= acc_mant;
                    output_exp   <= acc_exp;

                    // 是否最后一个 (q,d)
                    if ((d_idx == HEAD_DIM-1) && (q_idx == NUM_QUERIES-1)) begin
                        // 整个 batch 结束
                        busy <= 1'b0;
                        done <= 1'b1;
                        st   <= ST_IDLE;
                    end else begin
                        // 还有下一个 (q,d) 需要处理
                        if (d_idx < HEAD_DIM-1) begin
                            // 同一个 query 的下一个 dim
                            d_idx    <= d_idx + 3'd1;
                        end else begin
                            // 下一个 query，从 dim=0 开始
                            d_idx    <= 3'd0;
                            q_idx    <= q_idx + 5'd1;
                        end

                        // 为下一个 (q_idx, d_idx) 初始化累加器
                        k_idx    <= 5'd0;
                        acc_mant <= {ACCUM_WIDTH{1'b0}};
                        acc_exp  <= get_v_exp(5'd0);
                        st       <= ST_ACCUM;
                    end
                end

                default: begin
                    st   <= ST_IDLE;
                    busy <= 1'b0;
                end
            endcase
        end
    end

endmodule
