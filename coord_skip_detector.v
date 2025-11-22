`timescale 1ns / 1ps

module coord_skip_detector (
    input  wire [14:0] k_AB_packed,   // AB 的 5 组 Booth 码
    input  wire [14:0] k_CD_packed,   // CD 的 5 组 Booth 码
    output wire [9:0]  coord_skip_mode, // [9,7,5,3,1]=AB_valid，[8,6,4,2,0]=CD_valid
    output wire [3:0]  valid_count    // 全局有效 PP 数量 (0~10)
);

    //--------------------------------------------------------------
    // 1. 解包 Booth 码
    //--------------------------------------------------------------
    wire [2:0] k_AB [0:4];
    wire [2:0] k_CD [0:4];

    genvar i;
    generate
        for (i = 0; i < 5; i = i + 1) begin : unpack
            assign k_AB[i] = k_AB_packed[i*3 +: 3];
            assign k_CD[i] = k_CD_packed[i*3 +: 3];
        end
    endgenerate

    //--------------------------------------------------------------
    // 2. 对每一组 Booth 码做 skip 检测
    //--------------------------------------------------------------
    wire is_skip_AB [0:4];
    wire is_skip_CD [0:4];

    generate
        for (i = 0; i < 5; i = i + 1) begin : detect
            skip_detector u_skip_ab (
                .booth_code(k_AB[i]),
                .is_skip   (is_skip_AB[i])
            );

            skip_detector u_skip_cd (
                .booth_code(k_CD[i]),
                .is_skip   (is_skip_CD[i])
            );
        end
    endgenerate

    //--------------------------------------------------------------
    // 3. 生成每一组的 {AB_valid, CD_valid}
    //--------------------------------------------------------------
    wire [1:0] mode [0:4];  // mode[i][1]=AB_valid, mode[i][0]=CD_valid

    generate
        for (i = 0; i < 5; i = i + 1) begin : gen_mode
            assign mode[i] = {~is_skip_AB[i], ~is_skip_CD[i]};
        end
    endgenerate

    // 打包成 coord_skip_mode（和顶层的解码严格对应）
    assign coord_skip_mode = { mode[4], mode[3], mode[2], mode[1], mode[0] };

    //--------------------------------------------------------------
    // 4. 统计有效 PP 个数（AB+CD 的总个数）
    //--------------------------------------------------------------
    wire [3:0] sum0;
    wire [3:0] sum1;

    assign sum0 = mode[0][1] + mode[0][0] +
                  mode[1][1] + mode[1][0] +
                  mode[2][1] + mode[2][0];

    assign sum1 = mode[3][1] + mode[3][0] +
                  mode[4][1] + mode[4][0];

    assign valid_count = sum0 + sum1;

endmodule


//==============================================================
// 子模块：单个 skip 检测器
//==============================================================
module skip_detector (
    input  wire [2:0] booth_code,   // {sign, sel[1:0]}
    output wire       is_skip       // 1 = 可以跳过（部分积为 0）
);
    wire is_000 = ~(booth_code[2] | booth_code[1] | booth_code[0]);
    wire is_111 =  (booth_code[2] & booth_code[1] & booth_code[0]);
    assign is_skip = is_000 | is_111;
endmodule
