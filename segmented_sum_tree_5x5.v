`timescale 1ns / 1ps

module segmented_sum_tree_5x5 (
    input  wire signed [18:0] pp_AB_0,
    input  wire signed [18:0] pp_AB_1,
    input  wire signed [18:0] pp_AB_2,
    input  wire signed [18:0] pp_AB_3,
    input  wire signed [18:0] pp_AB_4,

    input  wire signed [18:0] pp_CD_0,
    input  wire signed [18:0] pp_CD_1,
    input  wire signed [18:0] pp_CD_2,
    input  wire signed [18:0] pp_CD_3,
    input  wire signed [18:0] pp_CD_4,

    input  wire [4:0]        skip_AB,
    input  wire [4:0]        skip_CD,

    output wire signed [18:0] result
);

    // ---------------------------------------------------------
    // 1) 根据 skip 标志屏蔽对应的部分积（为 0 不参与求和）
    // ---------------------------------------------------------
    wire signed [18:0] ab0_eff = skip_AB[0] ? 19'sd0 : pp_AB_0;
    wire signed [18:0] ab1_eff = skip_AB[1] ? 19'sd0 : pp_AB_1;
    wire signed [18:0] ab2_eff = skip_AB[2] ? 19'sd0 : pp_AB_2;
    wire signed [18:0] ab3_eff = skip_AB[3] ? 19'sd0 : pp_AB_3;
    wire signed [18:0] ab4_eff = skip_AB[4] ? 19'sd0 : pp_AB_4;

    wire signed [18:0] cd0_eff = skip_CD[0] ? 19'sd0 : pp_CD_0;
    wire signed [18:0] cd1_eff = skip_CD[1] ? 19'sd0 : pp_CD_1;
    wire signed [18:0] cd2_eff = skip_CD[2] ? 19'sd0 : pp_CD_2;
    wire signed [18:0] cd3_eff = skip_CD[3] ? 19'sd0 : pp_CD_3;
    wire signed [18:0] cd4_eff = skip_CD[4] ? 19'sd0 : pp_CD_4;

    // ---------------------------------------------------------
    // 2) 第一层：AB / CD 各自做本地求和
    //    （每个加法器建议用 DSP 来实现）
    // ---------------------------------------------------------
    (* use_dsp = "yes" *) wire signed [18:0] ab_sum0 = ab0_eff + ab1_eff;
    (* use_dsp = "yes" *) wire signed [18:0] ab_sum1 = ab2_eff + ab3_eff;
    wire  signed [18:0] ab_sum2 = ab4_eff; // 单独一项，直接透传

    (* use_dsp = "yes" *) wire signed [18:0] cd_sum0 = cd0_eff + cd1_eff;
    (* use_dsp = "yes" *) wire signed [18:0] cd_sum1 = cd2_eff + cd3_eff;
    wire  signed [18:0] cd_sum2 = cd4_eff;

    // ---------------------------------------------------------
    // 3) 第二层：合并 AB / CD 局部和
    // ---------------------------------------------------------
    (* use_dsp = "yes" *) wire signed [18:0] sum_AB = ab_sum0 + ab_sum1; // AB 前 4 组
    (* use_dsp = "yes" *) wire signed [18:0] sum_CD = cd_sum0 + cd_sum1; // CD 前 4 组

    // 剩余两项（AB 第 5 组 + CD 第 5 组）
    (* use_dsp = "yes" *) wire signed [18:0] sum_tail = ab_sum2 + cd_sum2;

    // ---------------------------------------------------------
    // 4) 最后一层：得到 10 组部分积的总和
    //   （这个表达式里有两个"+"，工具会拆成两级加法，
    //    也会尽量在 DSP 里实现）
    // ---------------------------------------------------------
    (* use_dsp = "yes" *) wire signed [18:0] result_tmp = sum_AB + sum_CD + sum_tail;

    assign result = result_tmp;

endmodule
