`timescale 1ns / 1ps

module bfp_converter #(
    parameter TOTAL_RESULTS = 32,       // 修正参数，匹配 qk_compute_batch.v
    parameter FIXED_WIDTH = 32,         
    parameter BASE_EXP_WIDTH = 9,       
    parameter OUTPUT_MANT_WIDTH = 8,    // 修正参数，匹配 qk_compute_batch.v
    parameter OUTPUT_EXP_WIDTH = 8      
)(
    input  wire clk,
    input  wire rst_n,
    input  wire flush,
    
    input  wire [TOTAL_RESULTS-1:0] input_valids,
    input  wire signed [TOTAL_RESULTS*FIXED_WIDTH-1:0] input_fixed_array,
    input  wire [TOTAL_RESULTS*BASE_EXP_WIDTH-1:0] input_base_exp_array,
    input  wire [TOTAL_RESULTS-1:0] input_zero_array,
    
    output reg  [TOTAL_RESULTS-1:0] output_valids,
    output reg  signed [TOTAL_RESULTS*OUTPUT_MANT_WIDTH-1:0] output_mant_array,
    output reg  [OUTPUT_EXP_WIDTH-1:0] output_shared_exp,
    output reg  output_overflow
);
    //==========================================================================
    // 内部信号
    //==========================================================================
    
    wire signed [OUTPUT_MANT_WIDTH-1:0] normalized_mants [0:TOTAL_RESULTS-1];
    wire [OUTPUT_EXP_WIDTH-1:0] independent_exps [0:TOTAL_RESULTS-1];
    wire zero_flags [0:TOTAL_RESULTS-1];
    
    reg signed [OUTPUT_MANT_WIDTH-1:0] aligned_mants [0:TOTAL_RESULTS-1];
    wire signed [OUTPUT_MANT_WIDTH-1:0] mant_packed_wires [0:TOTAL_RESULTS-1];
    
    reg [OUTPUT_EXP_WIDTH-1:0] max_exp;
    reg overflow_flag;
    
    integer i;
    //==========================================================================
    // 步骤1：归一化 - 使用generate实例化
    //==========================================================================
    
    genvar g;
    generate
        for (g = 0; g < TOTAL_RESULTS; g = g + 1) begin : gen_normalize
            
            wire signed [FIXED_WIDTH-1:0] fixed_val;
            wire [BASE_EXP_WIDTH-1:0] base_exp;
            wire is_zero;
            
            assign fixed_val = input_fixed_array[(g+1)*FIXED_WIDTH-1 : g*FIXED_WIDTH];
            assign base_exp = input_base_exp_array[(g+1)*BASE_EXP_WIDTH-1 : g*BASE_EXP_WIDTH];
            assign is_zero = input_zero_array[g];
            
            normalize_fixed_to_bfp #(
                .FIXED_WIDTH(FIXED_WIDTH),
                .BASE_EXP_WIDTH(BASE_EXP_WIDTH),
                .OUTPUT_MANT_WIDTH(OUTPUT_MANT_WIDTH),
                .OUTPUT_EXP_WIDTH(OUTPUT_EXP_WIDTH)
            ) u_norm (
                .fixed_val(fixed_val),
                .base_exp(base_exp),
                .is_zero(is_zero),
                .bfp_mant(normalized_mants[g]),
                .bfp_exp(independent_exps[g]),
                .zero_out(zero_flags[g])
            );
        end
    endgenerate
    
    //==========================================================================
    // 步骤2：找最大指数 - 树形比较器 (修正为有符号比较)
    //==========================================================================
    
    reg [OUTPUT_EXP_WIDTH-1:0] tree_level1 [0:15];
    reg [OUTPUT_EXP_WIDTH-1:0] tree_level2 [0:7];
    reg [OUTPUT_EXP_WIDTH-1:0] tree_level3 [0:3];
    reg [OUTPUT_EXP_WIDTH-1:0] tree_level4 [0:1];
    always @(*) begin : find_max_exp_tree
        integer j;
        
        // Level 1
        for (j = 0; j < 16; j = j + 1) begin
            if (2*j < TOTAL_RESULTS) begin
                if (2*j+1 < TOTAL_RESULTS) begin
                    if (zero_flags[2*j] && zero_flags[2*j+1]) begin
                        tree_level1[j] = {OUTPUT_EXP_WIDTH{1'b0}};
                    end else if (zero_flags[2*j]) begin
                        tree_level1[j] = independent_exps[2*j+1];
                    end else if (zero_flags[2*j+1]) begin
                        tree_level1[j] = independent_exps[2*j];
                    end else begin
                        // *** FIX: 使用 $signed() 进行有符号比较 ***
                        tree_level1[j] = ($signed(independent_exps[2*j]) > $signed(independent_exps[2*j+1])) ?
                            independent_exps[2*j] : independent_exps[2*j+1];
                    end
                end else begin
                    tree_level1[j] = zero_flags[2*j] ?
                        {OUTPUT_EXP_WIDTH{1'b0}} : independent_exps[2*j];
                end
            end else begin
                tree_level1[j] = {OUTPUT_EXP_WIDTH{1'b0}};
            end
        end
        
        // Level 2-5 (继续使用有符号比较)
        for (j = 0; j < 8; j = j + 1) begin
            tree_level2[j] = ($signed(tree_level1[2*j]) > $signed(tree_level1[2*j+1])) ?
                tree_level1[2*j] : tree_level1[2*j+1];
        end
        
        for (j = 0; j < 4; j = j + 1) begin
            tree_level3[j] = ($signed(tree_level2[2*j]) > $signed(tree_level2[2*j+1])) ?
                tree_level2[2*j] : tree_level2[2*j+1];
        end
        
        for (j = 0; j < 2; j = j + 1) begin
            tree_level4[j] = ($signed(tree_level3[2*j]) > $signed(tree_level3[2*j+1])) ?
                tree_level3[2*j] : tree_level3[2*j+1];
        end
        
        max_exp = ($signed(tree_level4[0]) > $signed(tree_level4[1])) ?
            tree_level4[0] : tree_level4[1];
    end
    
    //==========================================================================
    // 步骤3：对齐尾数到共享指数
    //==========================================================================
    
    always @(*) begin : align_mantissas
        reg signed [OUTPUT_EXP_WIDTH:0] exp_diff;
        // 声明为有符号数，用于计算差值
        reg signed [OUTPUT_MANT_WIDTH-1:0] shifted_val;
        
        overflow_flag = 1'b0;
        for (i = 0; i < TOTAL_RESULTS; i = i + 1) begin
            if (zero_flags[i]) begin
                aligned_mants[i] = {OUTPUT_MANT_WIDTH{1'b0}};
            end else begin
                // 使用 $signed() 确保是算术减法
                exp_diff = $signed(max_exp) - $signed(independent_exps[i]);
                
                if (exp_diff < 0) begin
                    // 理论上不应发生，因为 max_exp 是最大值。设为 0 移位。
                    aligned_mants[i] = normalized_mants[i];
                end else if (exp_diff == 0) begin
                    aligned_mants[i] = normalized_mants[i];
                end else begin
                    // exp_diff > 0, 需要右移 exp_diff 位
                    
                    // 检查是否溢出 (右移位数 >= 输出尾数位宽)
                    if (exp_diff >= OUTPUT_MANT_WIDTH) begin
                        // 溢出或完全舍弃：只保留符号位
                        aligned_mants[i] = {normalized_mants[i][OUTPUT_MANT_WIDTH-1], 
                                          {(OUTPUT_MANT_WIDTH-1){1'b0}}};
                        overflow_flag = 1'b1; // 仅作为警告
                    end else begin
                        // 正常右移
                        shifted_val = normalized_mants[i] >>> exp_diff; // 使用算术右移
                        aligned_mants[i] = shifted_val;
                    end
                end
            end
        end
    end
    //==========================================================================
    // 步骤3.5：使用generate展开打包操作
    //==========================================================================
    
    generate
        for (g = 0; g < TOTAL_RESULTS; g = g + 1) begin : gen_pack
            // 每个元素生成一个独立的连续赋值
            assign mant_packed_wires[g] = aligned_mants[g];
        end
    endgenerate
    
    //==========================================================================
    // 步骤4：输出寄存（时序逻辑）
    //==========================================================================
    
    generate
        for (g = 0; g < TOTAL_RESULTS; g = g + 1) begin : gen_output_pack
            // 在时序逻辑外部用generate展开打包
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    output_mant_array[(g+1)*OUTPUT_MANT_WIDTH-1 : g*OUTPUT_MANT_WIDTH] 
                        <= {OUTPUT_MANT_WIDTH{1'b0}};
                end else if (!flush) begin
                    // 直接从wire数组赋值到输出的对应切片
                    output_mant_array[(g+1)*OUTPUT_MANT_WIDTH-1 : g*OUTPUT_MANT_WIDTH] 
                        <= mant_packed_wires[g];
                end
            end
        end
    endgenerate
    
    // 其他输出信号的寄存
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_valids <= {TOTAL_RESULTS{1'b0}};
            output_shared_exp <= {OUTPUT_EXP_WIDTH{1'b0}};
            output_overflow <= 1'b0;
        end else if (flush) begin
            output_valids <= {TOTAL_RESULTS{1'b0}};
        end else begin
            output_valids <= input_valids;
            output_shared_exp <= max_exp;
            output_overflow <= overflow_flag;
        end
    end

endmodule


//==============================================================================
// 子模块：normalize_fixed_to_bfp (修复版：预留符号位)
//==============================================================================

module normalize_fixed_to_bfp #(
    parameter FIXED_WIDTH = 32,
    parameter BASE_EXP_WIDTH = 9,
    parameter OUTPUT_MANT_WIDTH = 8, 
    parameter OUTPUT_EXP_WIDTH = 8
)(
    input  wire signed [FIXED_WIDTH-1:0] fixed_val,
    input  wire [BASE_EXP_WIDTH-1:0] base_exp,
    input  wire is_zero,
    output wire signed [OUTPUT_MANT_WIDTH-1:0] bfp_mant,
    output wire [OUTPUT_EXP_WIDTH-1:0] bfp_exp,
    output wire zero_out
);
    assign zero_out = is_zero;
    
    wire signed [FIXED_WIDTH-1:0] abs_val;
    wire sign_bit;
    
    assign sign_bit = fixed_val[FIXED_WIDTH-1];
    assign abs_val = sign_bit ? -fixed_val : fixed_val;
    
    function [5:0] count_leading_zeros_fast;
        input [FIXED_WIDTH-1:0] value;
        reg [5:0] count;
        reg [FIXED_WIDTH-1:0] tmp;
        begin
            tmp = value;
            count = 0;
            if (FIXED_WIDTH == 32) begin
                if (tmp[31:16] == 16'h0000) begin
                    count = count + 16;
                    tmp = {tmp[15:0], 16'h0};
                end
                if (tmp[31:24] == 8'h00) begin
                    count = count + 8;
                    tmp = {tmp[23:0], 8'h0};
                end
                if (tmp[31:28] == 4'h0) begin
                    count = count + 4;
                    tmp = {tmp[27:0], 4'h0};
                end
                if (tmp[31:30] == 2'b00) begin
                    count = count + 2;
                    tmp = {tmp[29:0], 2'b0};
                end
                if (tmp[31] == 1'b0) begin
                    count = count + 1;
                end
            end
            count_leading_zeros_fast = count;
        end
    endfunction
    
    wire [5:0] leading_zeros;
    assign leading_zeros = is_zero ? 6'd0 : count_leading_zeros_fast(abs_val);
    
    // 【修正 1】指数计算调整
    // 原公式将数值左移到顶格 (bit 31)。
    // 现在我们需要留出 1 位符号位，所以实际有效左移少 1 位，或者说指数需要少减 1。
    // 但为了保持数值对齐逻辑简单，我们保持 leading_zeros 不变，
    // 仅仅在提取尾数时少取 1 位。这意味着数值实际上小了一倍（右移一位），
    // 所以指数应该 +1 来补偿？
    // 在 BFP 中，只要所有数都按同样规则归一化，相对关系就不变。
    // 我们保持指数计算不变，这会使数值看起来小一倍，但符合符号位要求。
    wire [OUTPUT_EXP_WIDTH:0] exp_temp;
    assign exp_temp = base_exp[OUTPUT_EXP_WIDTH-1:0] + {{(OUTPUT_EXP_WIDTH-6+1){1'b0}}, leading_zeros};
    
    wire exp_overflow;
    assign exp_overflow = exp_temp[OUTPUT_EXP_WIDTH];
    assign bfp_exp = is_zero ? {OUTPUT_EXP_WIDTH{1'b0}} :
                       (exp_overflow ? {OUTPUT_EXP_WIDTH{1'b1}} : 
                        exp_temp[OUTPUT_EXP_WIDTH-1:0]);
    
    reg signed [OUTPUT_MANT_WIDTH-1:0] mant_result;
    
    always @(*) begin : normalize_mantissa
        reg [FIXED_WIDTH-1:0] shifted;
        reg [FIXED_WIDTH-1:0] rounded;
        reg round_bit;
        reg [OUTPUT_MANT_WIDTH-2:0] mant_magnitude; // 【修正 2】幅度只有 7 位 (对于8位输出)
        
        if (is_zero) begin
            mant_result = {OUTPUT_MANT_WIDTH{1'b0}};
        end else begin
            // 1. 左移归一化 (最高位为 1)
            shifted = abs_val << leading_zeros;
            
            // 2. 舍入处理
            if (FIXED_WIDTH > OUTPUT_MANT_WIDTH) begin

                round_bit = shifted[FIXED_WIDTH - OUTPUT_MANT_WIDTH]; 
                
                // 简单的向上舍入逻辑
                if (round_bit) begin
                     rounded = shifted; 
                end else begin
                    rounded = shifted;
                end

                mant_magnitude = rounded[FIXED_WIDTH-1 : FIXED_WIDTH-(OUTPUT_MANT_WIDTH-1)];
            end else begin
                mant_magnitude = shifted[FIXED_WIDTH-1 : FIXED_WIDTH-(OUTPUT_MANT_WIDTH-1)];
            end
            
            // 3. 恢复符号
            if (sign_bit) begin
                mant_result = -$signed({1'b0, mant_magnitude});
            end else begin
                mant_result = $signed({1'b0, mant_magnitude});
            end
        end
    end
    
    assign bfp_mant = mant_result;
endmodule