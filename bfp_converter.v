`timescale 1ns / 1ps

//==============================================================================
// BFP Converter - 定点到共享指数BFP转换器
// 
// 版本：v1.4 - 使用generate展开解决常量表达式问题（最稳定方案）
// 修改日期：2025-11-14
//
// 主要特点：
// 1. 使用generate在编译时展开所有打包操作
// 2. 完全避免循环变量，所有索引都是常量
// 3. 最大兼容性，适用于所有综合工具
// 4. 代码简洁清晰，易于维护
//==============================================================================

module bfp_converter #(
    parameter TOTAL_RESULTS = 12,       
    parameter FIXED_WIDTH = 32,         
    parameter BASE_EXP_WIDTH = 9,       
    parameter OUTPUT_MANT_WIDTH = 16,   
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
    
    // 【方案2关键】使用wire数组在组合逻辑中打包
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
    // 步骤2：找最大指数 - 树形比较器
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
                        tree_level1[j] = (independent_exps[2*j] > independent_exps[2*j+1]) ? 
                                        independent_exps[2*j] : independent_exps[2*j+1];
                    end
                end else begin
                    tree_level1[j] = zero_flags[2*j] ? {OUTPUT_EXP_WIDTH{1'b0}} : independent_exps[2*j];
                end
            end else begin
                tree_level1[j] = {OUTPUT_EXP_WIDTH{1'b0}};
            end
        end
        
        // Level 2-5
        for (j = 0; j < 8; j = j + 1) begin
            tree_level2[j] = (tree_level1[2*j] > tree_level1[2*j+1]) ? 
                            tree_level1[2*j] : tree_level1[2*j+1];
        end
        
        for (j = 0; j < 4; j = j + 1) begin
            tree_level3[j] = (tree_level2[2*j] > tree_level2[2*j+1]) ? 
                            tree_level2[2*j] : tree_level2[2*j+1];
        end
        
        for (j = 0; j < 2; j = j + 1) begin
            tree_level4[j] = (tree_level3[2*j] > tree_level3[2*j+1]) ? 
                            tree_level3[2*j] : tree_level3[2*j+1];
        end
        
        max_exp = (tree_level4[0] > tree_level4[1]) ? tree_level4[0] : tree_level4[1];
    end
    
    //==========================================================================
    // 步骤3：对齐尾数到共享指数
    //==========================================================================
    
    always @(*) begin : align_mantissas
        reg [OUTPUT_EXP_WIDTH:0] shift_amt;
        reg signed [OUTPUT_MANT_WIDTH-1:0] shifted_val;
        
        overflow_flag = 1'b0;
        
        for (i = 0; i < TOTAL_RESULTS; i = i + 1) begin
            if (zero_flags[i]) begin
                aligned_mants[i] = {OUTPUT_MANT_WIDTH{1'b0}};
            end else begin
                shift_amt = max_exp - independent_exps[i];
                
                if (shift_amt == 0) begin
                    aligned_mants[i] = normalized_mants[i];
                end else if (shift_amt < OUTPUT_MANT_WIDTH) begin
                    shifted_val = normalized_mants[i] >>> shift_amt;
                    aligned_mants[i] = shifted_val;
                    if (shift_amt > 8) begin
                        overflow_flag = 1'b1;
                    end
                end else begin
                    aligned_mants[i] = {normalized_mants[i][OUTPUT_MANT_WIDTH-1], 
                                      {(OUTPUT_MANT_WIDTH-1){1'b0}}};
                    overflow_flag = 1'b1;
                end
            end
        end
    end
    
    //==========================================================================
    // 步骤3.5：【方案2】使用generate展开打包操作
    //==========================================================================
    // 
    // 关键优势：
    // 1. 所有索引都是编译时常量（genvar）
    // 2. 综合工具可以完全优化
    // 3. 没有任何循环变量的问题
    // 4. 清晰、简洁、高效
    //
    // 工作原理：
    // - 使用generate for循环在编译时展开
    // - 每个迭代生成一个assign语句
    // - 所有assign语句并行执行（纯组合逻辑）
    //==========================================================================
    
    generate
        for (g = 0; g < TOTAL_RESULTS; g = g + 1) begin : gen_pack
            // 每个元素生成一个独立的连续赋值
            // g是genvar，在编译时已知，不是循环变量
            assign mant_packed_wires[g] = aligned_mants[g];
        end
    endgenerate
    
    //==========================================================================
    // 步骤4：输出寄存（时序逻辑）
    //==========================================================================
    // 
    // 使用generate展开将wire数组打包到输出
    // 所有索引都是常量，完全符合Verilog-2001标准
    //==========================================================================
    
    generate
        for (g = 0; g < TOTAL_RESULTS; g = g + 1) begin : gen_output_pack
            // 在时序逻辑外部用generate展开打包
            // 这样在时序块中只需要简单赋值
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
// 子模块：normalize_fixed_to_bfp
//==============================================================================

module normalize_fixed_to_bfp #(
    parameter FIXED_WIDTH = 32,
    parameter BASE_EXP_WIDTH = 9,
    parameter OUTPUT_MANT_WIDTH = 16,
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
        reg [OUTPUT_MANT_WIDTH-1:0] mant_unsigned;
        
        if (is_zero) begin
            mant_result = {OUTPUT_MANT_WIDTH{1'b0}};
        end else begin
            shifted = abs_val << leading_zeros;
            
            if (FIXED_WIDTH > OUTPUT_MANT_WIDTH) begin
                round_bit = shifted[FIXED_WIDTH - OUTPUT_MANT_WIDTH - 1];
                
                if (round_bit && (|shifted[FIXED_WIDTH - OUTPUT_MANT_WIDTH - 2:0] || 
                                 shifted[FIXED_WIDTH - OUTPUT_MANT_WIDTH])) begin
                    rounded = shifted + (1 << (FIXED_WIDTH - OUTPUT_MANT_WIDTH));
                end else begin
                    rounded = shifted;
                end
                
                mant_unsigned = rounded[FIXED_WIDTH-1 : FIXED_WIDTH-OUTPUT_MANT_WIDTH];
            end else begin
                mant_unsigned = shifted[FIXED_WIDTH-1 : FIXED_WIDTH-OUTPUT_MANT_WIDTH];
            end
            
            if (sign_bit) begin
                mant_result = -$signed(mant_unsigned);
            end else begin
                mant_result = $signed(mant_unsigned);
            end
        end
    end
    
    assign bfp_mant = mant_result;

endmodule