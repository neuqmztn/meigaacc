`timescale 1ns / 1ps
module coordinated_skip_mac_unit(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    
    input  wire        unsigned_mode_A,
    input  wire        unsigned_mode_B,
    input  wire        unsigned_mode_C,
    input  wire        unsigned_mode_D,
    
    input  wire signed [7:0]  A,
    input  wire signed [7:0]  B,
    input  wire signed [7:0]  C,
    input  wire signed [7:0]  D,
    
    output reg  signed [18:0] result,
    output reg         result_valid
);


    //==============================================================
    // Stage 0: 输入寄存
    //==============================================================
    reg signed [7:0] A_reg, B_reg, C_reg, D_reg;
    reg unsigned_mode_A_reg, unsigned_mode_B_reg;
    reg unsigned_mode_C_reg, unsigned_mode_D_reg;
    reg stage0_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            A_reg <= 8'sb0;
            B_reg <= 8'sb0;
            C_reg <= 8'sb0;
            D_reg <= 8'sb0;
            unsigned_mode_A_reg <= 1'b0;
            unsigned_mode_B_reg <= 1'b0;
            unsigned_mode_C_reg <= 1'b0;
            unsigned_mode_D_reg <= 1'b0;
            stage0_valid <= 1'b0;
        end else begin
            if (enable) begin
                A_reg <= A;
                B_reg <= B;
                C_reg <= C;
                D_reg <= D;
                unsigned_mode_A_reg <= unsigned_mode_A;
                unsigned_mode_B_reg <= unsigned_mode_B;
                unsigned_mode_C_reg <= unsigned_mode_C;
                unsigned_mode_D_reg <= unsigned_mode_D;
            end
            stage0_valid <= enable; 
        end
    end

    //==============================================================
    // Stage 1: 逻辑计算 (基于 Stage 0 寄存器)
    //==============================================================
    
    wire fast_zero_AB;
    wire fast_zero_CD;
    
    assign fast_zero_AB = (A_reg == 8'sd0) || (B_reg == 8'sd0);
    assign fast_zero_CD = (C_reg == 8'sd0) || (D_reg == 8'sd0);

    // 符号扩展
    wire signed [8:0] A_ext, C_ext;
    assign A_ext = unsigned_mode_A_reg ? {1'b0, A_reg} : {A_reg[7], A_reg};
    assign C_ext = unsigned_mode_C_reg ? {1'b0, C_reg} : {C_reg[7], C_reg};

    // Booth 编码
    wire [14:0] booth_AB_packed, booth_CD_packed;
    booth_encoder_universal u_booth_AB (.multiplicand(B_reg), .unsigned_mode(unsigned_mode_B_reg), .k_packed(booth_AB_packed));
    booth_encoder_universal u_booth_CD (.multiplicand(D_reg), .unsigned_mode(unsigned_mode_D_reg), .k_packed(booth_CD_packed));

    // Skip 检测
    wire [9:0] skip_mode;
    wire [3:0] valid_count;
    coord_skip_detector u_skip_detect (
        .k_AB_packed(booth_AB_packed), .k_CD_packed(booth_CD_packed),
        .coord_skip_mode(skip_mode), .valid_count(valid_count)
    );

    // 结合快速零检测生成最终 Skip 信号
    wire [4:0] skip_AB, skip_CD;
    assign skip_AB = fast_zero_AB ? 5'b11111 : ~{skip_mode[9], skip_mode[7], skip_mode[5], skip_mode[3], skip_mode[1]};
    assign skip_CD = fast_zero_CD ? 5'b11111 : ~{skip_mode[8], skip_mode[6], skip_mode[4], skip_mode[2], skip_mode[0]};

    // PP 生成 (9-in / 19-out)
    wire signed [18:0] pp_AB [0:4];
    wire signed [18:0] pp_CD [0:4];

    genvar i;
    generate
        for (i = 0; i < 5; i = i + 1) begin : pp_gen
            pp_generator_with_gating #(.SHIFT(i*2)) u_pp_AB (
                .multiplicand(skip_AB[i] ? 9'sd0 : A_ext),
                .booth_code(booth_AB_packed[i*3 +: 3]),
                .skip(skip_AB[i]),
                .pp_aligned(pp_AB[i])
            );
            pp_generator_with_gating #(.SHIFT(i*2)) u_pp_CD (
                .multiplicand(skip_CD[i] ? 9'sd0 : C_ext),
                .booth_code(booth_CD_packed[i*3 +: 3]),
                .skip(skip_CD[i]),
                .pp_aligned(pp_CD[i])
            );
        end
    endgenerate

    // 求和树 (19-bit)
    wire signed [18:0] sum_total;
    segmented_sum_tree_5x5 u_sum_tree (
        .pp_AB_0(pp_AB[0]), .pp_AB_1(pp_AB[1]), .pp_AB_2(pp_AB[2]), .pp_AB_3(pp_AB[3]), .pp_AB_4(pp_AB[4]),
        .pp_CD_0(pp_CD[0]), .pp_CD_1(pp_CD[1]), .pp_CD_2(pp_CD[2]), .pp_CD_3(pp_CD[3]), .pp_CD_4(pp_CD[4]),
        .skip_AB(skip_AB), .skip_CD(skip_CD),
        .result(sum_total)
    );

    //==============================================================
    // Stage 2: 中间寄存
    //==============================================================
    reg signed [18:0] sum_stage2;
    reg stage2_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_stage2 <= 19'sb0;
            stage2_valid <= 1'b0;
        end else begin
            sum_stage2 <= sum_total;
            stage2_valid <= stage0_valid; 
        end
    end

    //==============================================================
    // Stage 3: 输出寄存 (自动移位)
    //==============================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 19'sb0;
            result_valid <= 1'b0;
        end else begin
            result <= sum_stage2;
            result_valid <= stage2_valid;
        end
    end

endmodule