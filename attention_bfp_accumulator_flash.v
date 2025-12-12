
// ============================================================================
// attention_bfp_accumulator_flash
// 维护 O_num = Σ exp(score - m_new) * V，跨 chunk：
//   O_num_new = O_num_prev * renorm_scale + Z_chunk
// 最终在外部使用 sum_wr_value(l_new) 做除法：O = O_num / l_new。
// 接口在原单 head 累加器基础上增加 renorm 控制信号。
// ============================================================================

module attention_bfp_accumulator_flash #(
    parameter TOKEN_BATCH = 32,
    parameter HEAD_DIM    = 8,
    parameter EXP_WIDTH   = 8,
    parameter ACCUM_WIDTH = 24,
    parameter SCORE_WIDTH = 16
)(
    input  wire clk,
    input  wire rst_n,

    // 写入接口（来自 apply_v_batch_flash）
    input  wire                    wr_en,
    input  wire [4:0]              wr_row,
    input  wire [2:0]              wr_dim,
    input  wire signed [ACCUM_WIDTH-1:0] wr_mant,
    input  wire [EXP_WIDTH-1:0]    wr_exp,
    input  wire                    first_chunk,

    // FlashAttention 统计控制信号（广播给所有 dim）
    input  wire                    renorm_en,
    input  wire signed [SCORE_WIDTH-1:0] renorm_scale, // Q14

    // 读接口（输出当前存储的 O_num，以 BFP 表示）
    input  wire                    rd_en,
    input  wire [4:0]              rd_row,
    input  wire [2:0]              rd_dim,
    output reg  signed [ACCUM_WIDTH-1:0] rd_mant,
    output reg  [EXP_WIDTH-1:0]    rd_exp,
    output reg                     rd_valid
);

    localparam integer FRAC_BITS = 14;

    // 存储 O_num
    reg signed [ACCUM_WIDTH-1:0] O_mant [0:TOKEN_BATCH-1][0:HEAD_DIM-1];
    reg [EXP_WIDTH-1:0]          O_exp  [0:TOKEN_BATCH-1][0:HEAD_DIM-1];

    integer i, j;

    // 写路径：在 first_chunk=1 时相当于清零重写；
    // 非首块：如果 renorm_en=1 则先对旧值做缩放，再与 wr_mant BFP 对齐累加。

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < TOKEN_BATCH; i = i + 1) begin
                for (j = 0; j < HEAD_DIM; j = j + 1) begin
                    O_mant[i][j] <= {ACCUM_WIDTH{1'b0}};
                    O_exp[i][j]  <= {EXP_WIDTH{1'b0}};
                end
            end
        end else begin
            if (wr_en && wr_row < TOKEN_BATCH && wr_dim < HEAD_DIM) begin:l
                reg signed [ACCUM_WIDTH-1:0] old_mant;
                reg [EXP_WIDTH-1:0]          old_exp;
                reg signed [ACCUM_WIDTH-1:0] old_scaled;
                reg signed [ACCUM_WIDTH-1:0] new_shift;
                reg signed [ACCUM_WIDTH-1:0] acc_sum;
                reg [EXP_WIDTH-1:0]          max_exp;
                integer diff;

                old_mant = O_mant[wr_row][wr_dim];
                old_exp  = O_exp[wr_row][wr_dim];

                if (first_chunk) begin
                    // 第一个 chunk：直接写入 Z_chunk
                    O_mant[wr_row][wr_dim] <= wr_mant;
                    O_exp[wr_row][wr_dim]  <= wr_exp;
                end else begin
                    // 旧值 * renorm_scale（若 renorm_en=1）
                    if (renorm_en) begin:m
                        reg signed [ACCUM_WIDTH+SCORE_WIDTH-1:0] mult_tmp;
                        mult_tmp  = old_mant * renorm_scale;
                        old_scaled= mult_tmp >>> FRAC_BITS;
                    end else begin
                        old_scaled= old_mant;
                    end

                    // 与新贡献 BFP 对齐累加
                    if (wr_exp >= old_exp) begin
                        max_exp = wr_exp;
                        diff    = wr_exp - old_exp;
                        if (diff >= ACCUM_WIDTH)
                            new_shift = {ACCUM_WIDTH{1'b0}};
                        else
                            new_shift = old_scaled >>> diff;
                        acc_sum = new_shift + wr_mant;
                    end else begin
                        max_exp = old_exp;
                        diff    = old_exp - wr_exp;
                        if (diff >= ACCUM_WIDTH)
                            new_shift = {ACCUM_WIDTH{1'b0}};
                        else
                            new_shift = wr_mant >>> diff;
                        acc_sum = old_scaled + new_shift;
                    end

                    O_mant[wr_row][wr_dim] <= acc_sum;
                    O_exp[wr_row][wr_dim]  <= max_exp;
                end
            end
        end
    end

    // 读路径：同步读出当前存储的 O_num
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_mant  <= {ACCUM_WIDTH{1'b0}};
            rd_exp   <= {EXP_WIDTH{1'b0}};
            rd_valid <= 1'b0;
        end else begin
            if (rd_en && rd_row < TOKEN_BATCH && rd_dim < HEAD_DIM) begin
                rd_mant  <= O_mant[rd_row][rd_dim];
                rd_exp   <= O_exp[rd_row][rd_dim];
                rd_valid <= 1'b1;
            end else begin
                rd_valid <= 1'b0;
            end
        end
    end

endmodule
