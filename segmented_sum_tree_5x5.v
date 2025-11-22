`timescale 1ns / 1ps

//==============================================================
// segmented_sum_tree_5x5.v - 分段求和树 (修复版)
// 
// 修改日志：
// 1. 输入/输出位宽从 17-bit 扩展至 19-bit，适配顶层改动。
// 2. 内部信号同步加宽。
//==============================================================

module segmented_sum_tree_5x5 (
    // AB 组部分积输入 (19-bit)
    input  wire signed [18:0] pp_AB_0,
    input  wire signed [18:0] pp_AB_1,
    input  wire signed [18:0] pp_AB_2,
    input  wire signed [18:0] pp_AB_3,
    input  wire signed [18:0] pp_AB_4,
    
    // CD 组部分积输入 (19-bit)
    input  wire signed [18:0] pp_CD_0,
    input  wire signed [18:0] pp_CD_1,
    input  wire signed [18:0] pp_CD_2,
    input  wire signed [18:0] pp_CD_3,
    input  wire signed [18:0] pp_CD_4,
    
    // Skip 控制信号
    input  wire [4:0] skip_AB,
    input  wire [4:0] skip_CD,
    
    // 最终结果输出 (19-bit)
    output wire signed [18:0] result
);

    //--------------------------------------------------------------
    // Stage 1: AB组内稀疏求和
    //--------------------------------------------------------------
    wire [2:0] ab_valid_count;
    wire signed [18:0] ab_sum; // 修改为 19-bit

    sparse_sum_5to1 u_ab_sum (
        .in0(pp_AB_0),
        .in1(pp_AB_1),
        .in2(pp_AB_2),
        .in3(pp_AB_3),
        .in4(pp_AB_4),
        .skip(skip_AB),
        .valid_count(ab_valid_count),
        .sum(ab_sum)
    );

    //--------------------------------------------------------------
    // Stage 2: CD组内稀疏求和
    //--------------------------------------------------------------
    wire [2:0] cd_valid_count;
    wire signed [18:0] cd_sum; // 修改为 19-bit

    sparse_sum_5to1 u_cd_sum (
        .in0(pp_CD_0),
        .in1(pp_CD_1),
        .in2(pp_CD_2),
        .in3(pp_CD_3),
        .in4(pp_CD_4),
        .skip(skip_CD),
        .valid_count(cd_valid_count),
        .sum(cd_sum)
    );

    //--------------------------------------------------------------
    // Stage 3: 组间合并
    //--------------------------------------------------------------
    wire ab_all_zero = (ab_valid_count == 3'd0);
    wire cd_all_zero = (cd_valid_count == 3'd0);

    // 结果计算：即便 AB+CD，19-bit 也足够容纳最大值 (130,050 < 262,143)
    assign result = ab_all_zero ? cd_sum :
                    (cd_all_zero ? ab_sum : (ab_sum + cd_sum));

endmodule