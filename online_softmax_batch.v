`timescale 1ns / 1ps

// ============================================================================
// online_softmax_batch
// - 对 NUM_QUERIES 行做批量 online softmax（单 head）
// - 逐行调用 online_softmax_engine，维护 m/l，并输出每行分子 + renorm 信息
// - 与 softmax_stats_buffer 的接口只用 rd_row / rd_value / wr_*，无 rd_en / head
// ============================================================================

module online_softmax_batch #(
    parameter NUM_QUERIES   = 32,
    parameter CHUNK_SIZE    = 32,
    parameter SCORE_WIDTH   = 16,
    parameter EXP_WIDTH     = 8,
    parameter ACCUM_WIDTH   = 24,
    parameter NUM_HEADS     = 1,   // 单 head，这里保留参数只是占位
    parameter K_CHUNK_SIZE  = CHUNK_SIZE
)(
    input  wire clk,
    input  wire rst_n,

    // 控制
    input  wire start,
    output reg  done,
    output reg  busy,

    input  wire chunk_first,
    input  wire chunk_last,

    // Scores 批量输入：[row][k] 展开
    input  wire [NUM_QUERIES*CHUNK_SIZE*EXP_WIDTH-1:0]   scores_exp_batch,
    input  wire [NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH-1:0] scores_batch,

    // 分子批量输出：[row][k] 展开
    output reg                        weights_valid,
    output reg [NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH-1:0] weights_batch,

    // ---- 与 softmax_stats_buffer 对接（单 head）----
    // 读旧 m / l
    output reg  [4:0] max_rd_row,
    input  wire signed [SCORE_WIDTH-1:0] max_rd_value,

    output reg  [4:0] sum_rd_row,
    input  wire signed [ACCUM_WIDTH-1:0] sum_rd_value,

    // 写回新 m / l
    output reg        max_wr_en,
    output reg  [4:0] max_wr_row,
    output reg  signed [SCORE_WIDTH-1:0] max_wr_value,

    output reg        sum_wr_en,
    output reg  [4:0] sum_wr_row,
    output reg  signed [ACCUM_WIDTH-1:0] sum_wr_value,

    // 行级 renorm 输出
    output reg [NUM_QUERIES-1:0]             renorm_flag_batch,
    output reg [NUM_QUERIES*SCORE_WIDTH-1:0] renorm_scale_batch
);

    // 状态机
    localparam ST_IDLE = 2'd0;
    localparam ST_RUN  = 2'd1;
    localparam ST_WAIT = 2'd2;
    localparam ST_DONE = 2'd3;

    reg [1:0] state;
    reg [4:0] row_idx;

    // 与 online_softmax_engine 的接口
    reg  eng_start;
    wire eng_done;
    wire eng_busy;
    wire eng_weights_valid;

    wire signed [SCORE_WIDTH-1:0]  eng_max_wr_value;
    wire signed [ACCUM_WIDTH-1:0]  eng_sum_wr_value;
    wire        eng_max_wr_en;
    wire [1:0]  eng_max_wr_head;
    wire [4:0]  eng_max_wr_row;
    wire        eng_sum_wr_en;
    wire [1:0]  eng_sum_wr_head;
    wire [4:0]  eng_sum_wr_row;

    wire        eng_renorm_en;
    wire signed [SCORE_WIDTH-1:0] eng_renorm_scale;

    wire [K_CHUNK_SIZE*SCORE_WIDTH-1:0] eng_weights_packed;

    // 按行切片 scores
    wire [K_CHUNK_SIZE*SCORE_WIDTH-1:0] scores_row_mants;
    wire [K_CHUNK_SIZE*EXP_WIDTH-1:0]   scores_row_exp;

    assign scores_row_mants = scores_batch[
        row_idx*CHUNK_SIZE*SCORE_WIDTH +: K_CHUNK_SIZE*SCORE_WIDTH
    ];

    assign scores_row_exp = scores_exp_batch[
        row_idx*CHUNK_SIZE*EXP_WIDTH +: K_CHUNK_SIZE*EXP_WIDTH
    ];

    // 行级 online softmax engine
    online_softmax_engine #(
        .NUM_HEADS   (NUM_HEADS),
        .TOKEN_BATCH (NUM_QUERIES),
        .K_CHUNK_SIZE(K_CHUNK_SIZE),
        .DATA_WIDTH  (8),
        .EXP_WIDTH   (EXP_WIDTH),
        .SCORE_WIDTH (SCORE_WIDTH),
        .ACCUM_WIDTH (ACCUM_WIDTH)
    ) u_engine (
        .clk      (clk),
        .rst_n    (rst_n),

        .start    (eng_start),
        .head_idx (2'd0),                 // 单 head 固定 0
        .row_idx  (row_idx),
        .chunk_id (5'd0),
        .chunk_first(chunk_first),
        .chunk_last (chunk_last),
        .chunk_size (CHUNK_SIZE[5:0]),

        .done     (eng_done),
        .busy     (eng_busy),

        .scores_shared_exp  (scores_row_exp[EXP_WIDTH-1:0]),
        .scores_mants_packed(scores_row_mants),

        .max_rd_value (max_rd_value),
        .sum_rd_value (sum_rd_value),

        .max_wr_en    (eng_max_wr_en),
        .max_wr_head  (eng_max_wr_head),
        .max_wr_row   (eng_max_wr_row),
        .max_wr_value (eng_max_wr_value),

        .sum_wr_en    (eng_sum_wr_en),
        .sum_wr_head  (eng_sum_wr_head),
        .sum_wr_row   (eng_sum_wr_row),
        .sum_wr_value (eng_sum_wr_value),

        .renorm_en    (eng_renorm_en),
        .renorm_scale (eng_renorm_scale),

        .weights_valid(eng_weights_valid),
        .weights_packed(eng_weights_packed)
    );

    integer i;

    // 主控制
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= ST_IDLE;
            done     <= 1'b0;
            busy     <= 1'b0;
            row_idx  <= 5'd0;
            eng_start<= 1'b0;

            weights_valid <= 1'b0;
            weights_batch <= {NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH{1'b0}};

            max_rd_row   <= 5'd0;
            max_wr_en    <= 1'b0;
            max_wr_row   <= 5'd0;
            max_wr_value <= {SCORE_WIDTH{1'b0}};

            sum_rd_row   <= 5'd0;
            sum_wr_en    <= 1'b0;
            sum_wr_row   <= 5'd0;
            sum_wr_value <= {ACCUM_WIDTH{1'b0}};

            renorm_flag_batch  <= {NUM_QUERIES{1'b0}};
            renorm_scale_batch <= {NUM_QUERIES*SCORE_WIDTH{1'b0}};
        end else begin
            done         <= 1'b0;
            eng_start    <= 1'b0;
            max_wr_en    <= 1'b0;
            sum_wr_en    <= 1'b0;
            weights_valid<= 1'b0;

            case (state)
                // ------------------------------
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy    <= 1'b1;
                        row_idx <= 5'd0;

                        // 第一行：先把 rd_row 指到 0，给 engine 读旧 m/l
                        max_rd_row <= 5'd0;
                        sum_rd_row <= 5'd0;

                        renorm_flag_batch  <= {NUM_QUERIES{1'b0}};
                        renorm_scale_batch <= {NUM_QUERIES*SCORE_WIDTH{1'b0}};

                        state <= ST_RUN;
                    end
                end

                // ------------------------------
                // 对当前 row_idx 启动 engine
                ST_RUN: begin
                    eng_start <= 1'b1;

                    // 告诉 stats_buffer：我要读 row_idx 的 m/l
                    max_rd_row <= row_idx;
                    sum_rd_row <= row_idx;

                    state <= ST_WAIT;
                end

                // ------------------------------
                // 等待 engine 完成
                ST_WAIT: begin
                    // engine 在内部计算完之后，会产生写回 m/l 的脉冲
                    if (eng_max_wr_en) begin
                        max_wr_en    <= 1'b1;
                        max_wr_row   <= eng_max_wr_row;
                        max_wr_value <= eng_max_wr_value;
                    end

                    if (eng_sum_wr_en) begin
                        sum_wr_en    <= 1'b1;
                        sum_wr_row   <= eng_sum_wr_row;
                        sum_wr_value <= eng_sum_wr_value;
                    end

                    if (eng_done) begin
                        // 写回当前行的分子数组
                        weights_batch[row_idx*CHUNK_SIZE*SCORE_WIDTH +: CHUNK_SIZE*SCORE_WIDTH]
                            <= eng_weights_packed;

                        // 记录行级 renorm 信息
                        renorm_flag_batch[row_idx] <= eng_renorm_en;
                        renorm_scale_batch[row_idx*SCORE_WIDTH +: SCORE_WIDTH]
                            <= eng_renorm_scale;

                        if (row_idx == NUM_QUERIES-1) begin
                            state <= ST_DONE;
                        end else begin
                            row_idx   <= row_idx + 5'd1;
                            max_rd_row<= row_idx + 5'd1;
                            sum_rd_row<= row_idx + 5'd1;
                            state     <= ST_RUN;
                        end
                    end
                end

                // ------------------------------
                ST_DONE: begin
                    busy          <= 1'b0;
                    done          <= 1'b1;
                    weights_valid <= 1'b1;
                    state         <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
