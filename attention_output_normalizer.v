`timescale 1ns / 1ps

// ============================================================================
// attention_output_normalizer
//   - 输入：O_num（BFP），sum_value = l_new
//   - 输出：O = O_num / l_new（仍为 BFP）
// ============================================================================

module attention_output_normalizer #(
    parameter TOKEN_BATCH = 32,
    parameter HEAD_DIM    = 8,
    parameter EXP_WIDTH   = 8,
    parameter ACCUM_WIDTH = 24,
    parameter SCORE_WIDTH = 16
)(
    input  wire clk,
    input  wire rst_n,

    input  wire        in_valid,
    input  wire [4:0]  in_row,
    input  wire [2:0]  in_dim,

    input  wire signed [ACCUM_WIDTH-1:0] in_mant,
    input  wire [EXP_WIDTH-1:0]          in_exp,

    input  wire signed [ACCUM_WIDTH-1:0] sum_value,

    output reg        out_valid,
    output reg [4:0]  out_row,
    output reg [2:0]  out_dim,
    output reg signed [ACCUM_WIDTH-1:0] out_mant,
    output reg [EXP_WIDTH-1:0]          out_exp
);

    localparam integer FRAC_BITS = 14;

    reg        stage1_valid;
    reg [4:0]  stage1_row;
    reg [2:0]  stage1_dim;
    reg signed [ACCUM_WIDTH-1:0] stage1_mant;
    reg [EXP_WIDTH-1:0]          stage1_exp;
    reg signed [ACCUM_WIDTH-1:0] stage1_sum;

    reg signed [ACCUM_WIDTH-1:0] div_result;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_valid <= 1'b0;
            stage1_row   <= 5'd0;
            stage1_dim   <= 3'd0;
            stage1_mant  <= {ACCUM_WIDTH{1'b0}};
            stage1_exp   <= {EXP_WIDTH{1'b0}};
            stage1_sum   <= {ACCUM_WIDTH{1'b0}};
        end else begin
            stage1_valid <= in_valid;
            if (in_valid) begin
                stage1_row  <= in_row;
                stage1_dim  <= in_dim;
                stage1_mant <= in_mant;
                stage1_exp  <= in_exp;
                stage1_sum  <= sum_value;
            end
        end
    end

    always @(*) begin
        if (stage1_sum != 0)
            div_result = (stage1_mant <<< FRAC_BITS) / stage1_sum;
        else
            div_result = {ACCUM_WIDTH{1'b0}};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_row   <= 5'd0;
            out_dim   <= 3'd0;
            out_mant  <= {ACCUM_WIDTH{1'b0}};
            out_exp   <= {EXP_WIDTH{1'b0}};
        end else begin
            out_valid <= stage1_valid;
            if (stage1_valid) begin
                out_row  <= stage1_row;
                out_dim  <= stage1_dim;
                out_mant <= div_result;
                if (stage1_exp >= FRAC_BITS[EXP_WIDTH-1:0])
                    out_exp <= stage1_exp - FRAC_BITS[EXP_WIDTH-1:0];
                else
                    out_exp <= {EXP_WIDTH{1'b0}};
            end
        end
    end

endmodule

