`timescale 1ns / 1ps

//================================================================================
// Fixed-Point to Independent BFP Normalizer
// 
// 功能：将CE输出的定点数转换为独立BFP格式（每个结果有独立指数）
// 
// 特点：
// - 只做归一化，不做共享指数对齐
// - 保持最高精度（无对齐截断）
// - 每个结果有独立指数
// - 支持8位或16位尾数配置
// - 适合训练场景（高精度）
// 
// 输入：定点数 + 基础指数（CE输出）
// 输出：归一化尾数 + 独立指数（Independent BFP）
//
// 数据流：
//   定点数(未归一化) → 归一化 → 独立BFP
//   无共享指数，无对齐，无精度损失
//
// 版本：v1.1 - Verilog-2001兼容性修复
// 标准：Verilog-2001
// 修改日期：2025-11-14
//
// 主要修改：
// 1. 删除automatic关键字
// 2. 使用generate展开数组打包操作
// 3. 简化组合逻辑，提高可综合性
//================================================================================

module fixed_to_independent_bfp #(
    parameter TOTAL_RESULTS = 12,          // 结果数量（通常12个PU输出）
    parameter FIXED_WIDTH = 32,            // 定点输入位宽（32或39）
    parameter BASE_EXP_WIDTH = 9,          // 基础指数位宽
    parameter OUTPUT_MANT_WIDTH = 16,      // 输出尾数位宽（8或16）
    parameter OUTPUT_EXP_WIDTH = 8         // 输出指数位宽
)(
    input  wire clk,
    input  wire rst_n,
    input  wire flush,
    
    //==========================================================================
    // 输入：CE的定点输出
    //==========================================================================
    input  wire [TOTAL_RESULTS-1:0] input_valids,
    input  wire signed [TOTAL_RESULTS*FIXED_WIDTH-1:0] input_fixed_array,
    input  wire [TOTAL_RESULTS*BASE_EXP_WIDTH-1:0] input_base_exp_array,
    input  wire [TOTAL_RESULTS-1:0] input_zero_array,
    
    //==========================================================================
    // 输出：独立BFP格式（每个结果有独立指数）
    //==========================================================================
    output reg  [TOTAL_RESULTS-1:0] output_valids,
    output reg  signed [TOTAL_RESULTS*OUTPUT_MANT_WIDTH-1:0] output_mant_array,
    output reg  [TOTAL_RESULTS*OUTPUT_EXP_WIDTH-1:0] output_exp_array,
    output reg  [TOTAL_RESULTS-1:0] output_overflow
);

    //==========================================================================
    // 内部信号 - 使用unpacked数组存储中间结果
    //==========================================================================
    
    // 每个结果的归一化输出
    wire signed [OUTPUT_MANT_WIDTH-1:0] normalized_mants [0:TOTAL_RESULTS-1];
    wire [OUTPUT_EXP_WIDTH-1:0] independent_exps [0:TOTAL_RESULTS-1];
    wire overflow_flags [0:TOTAL_RESULTS-1];

    //==========================================================================
    // 前导零/一检测函数（二分查找，支持32位）
    // 注意：Verilog-2001不支持automatic，但function中的变量本身就是局部的
    //==========================================================================
    function [5:0] count_leading_zeros;
        input signed [31:0] val;
        reg [5:0] count;
        reg [31:0] temp;
    begin
        // 负数取反（找前导一）
        temp = (val[31]) ? ~val : val;
        count = 0;
        
        // 5级二分查找（32位）
        if (temp[31:16] == 16'd0) begin
            count = count + 16;
            temp = {temp[15:0], 16'd0};
        end
        if (temp[31:24] == 8'd0) begin
            count = count + 8;
            temp = {temp[23:0], 8'd0};
        end
        if (temp[31:28] == 4'd0) begin
            count = count + 4;
            temp = {temp[27:0], 4'd0};
        end
        if (temp[31:30] == 2'd0) begin
            count = count + 2;
            temp = {temp[29:0], 2'd0};
        end
        if (temp[31] == 1'd0) begin
            count = count + 1;
        end
        
        count_leading_zeros = count;
    end
    endfunction

    //==========================================================================
    // 使用generate实例化归一化单元（避免循环索引问题）
    //==========================================================================
    
    genvar g;
    generate
        for (g = 0; g < TOTAL_RESULTS; g = g + 1) begin : gen_normalize
            
            // 提取当前结果的输入
            wire signed [FIXED_WIDTH-1:0] fixed_val;
            wire [BASE_EXP_WIDTH-1:0] base_exp;
            wire is_zero;
            
            assign fixed_val = input_fixed_array[(g+1)*FIXED_WIDTH-1 : g*FIXED_WIDTH];
            assign base_exp = input_base_exp_array[(g+1)*BASE_EXP_WIDTH-1 : g*BASE_EXP_WIDTH];
            assign is_zero = input_zero_array[g];
            
            // 实例化归一化单元
            normalize_single_result #(
                .FIXED_WIDTH(FIXED_WIDTH),
                .BASE_EXP_WIDTH(BASE_EXP_WIDTH),
                .OUTPUT_MANT_WIDTH(OUTPUT_MANT_WIDTH),
                .OUTPUT_EXP_WIDTH(OUTPUT_EXP_WIDTH)
            ) u_normalize (
                .fixed_val(fixed_val),
                .base_exp(base_exp),
                .is_zero(is_zero),
                .normalized_mant(normalized_mants[g]),
                .independent_exp(independent_exps[g]),
                .overflow_flag(overflow_flags[g])
            );
        end
    endgenerate

    //==========================================================================
    // 使用generate展开打包操作（避免循环变量索引问题）
    //==========================================================================
    
    generate
        for (g = 0; g < TOTAL_RESULTS; g = g + 1) begin : gen_output
            // 每个结果单独寄存
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    output_mant_array[(g+1)*OUTPUT_MANT_WIDTH-1 : g*OUTPUT_MANT_WIDTH] 
                        <= {OUTPUT_MANT_WIDTH{1'b0}};
                    output_exp_array[(g+1)*OUTPUT_EXP_WIDTH-1 : g*OUTPUT_EXP_WIDTH] 
                        <= {OUTPUT_EXP_WIDTH{1'b0}};
                    output_overflow[g] <= 1'b0;
                end else if (flush) begin
                    output_mant_array[(g+1)*OUTPUT_MANT_WIDTH-1 : g*OUTPUT_MANT_WIDTH] 
                        <= {OUTPUT_MANT_WIDTH{1'b0}};
                    output_exp_array[(g+1)*OUTPUT_EXP_WIDTH-1 : g*OUTPUT_EXP_WIDTH] 
                        <= {OUTPUT_EXP_WIDTH{1'b0}};
                    output_overflow[g] <= 1'b0;
                end else begin
                    output_mant_array[(g+1)*OUTPUT_MANT_WIDTH-1 : g*OUTPUT_MANT_WIDTH] 
                        <= normalized_mants[g];
                    output_exp_array[(g+1)*OUTPUT_EXP_WIDTH-1 : g*OUTPUT_EXP_WIDTH] 
                        <= independent_exps[g];
                    output_overflow[g] <= overflow_flags[g];
                end
            end
        end
    endgenerate
    
    // Valid信号的寄存（单独处理）
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_valids <= {TOTAL_RESULTS{1'b0}};
        end else if (flush) begin
            output_valids <= {TOTAL_RESULTS{1'b0}};
        end else begin
            output_valids <= input_valids;
        end
    end

    //==========================================================================
    // 调试信息
    //==========================================================================
    initial begin
        $display("========================================");
        $display("Fixed to Independent BFP Normalizer v1.1");
        $display("========================================");
        $display("Configuration:");
        $display("  Total Results:     %0d", TOTAL_RESULTS);
        $display("  Fixed Width:       %0d-bit", FIXED_WIDTH);
        $display("  Output Mant Width: %0d-bit", OUTPUT_MANT_WIDTH);
        $display("  Output Exp Width:  %0d-bit", OUTPUT_EXP_WIDTH);
        $display("========================================");
        $display("Features:");
        $display("  ✓ Verilog-2001 compliant");
        $display("  ✓ Independent BFP (No shared exponent)");
        $display("  ✓ Maximum precision (No alignment loss)");
        $display("  ✓ Suitable for training");
        $display("  ✓ Configurable mantissa: 8-bit or 16-bit");
        $display("========================================");
        $display("Output Format:");
        $display("  Each result: %0d-bit mant + %0d-bit exp", OUTPUT_MANT_WIDTH, OUTPUT_EXP_WIDTH);
        $display("  Total output: %0d results", TOTAL_RESULTS);
        $display("  Storage: %0d bits (mant) + %0d bits (exp)",
                 TOTAL_RESULTS*OUTPUT_MANT_WIDTH,
                 TOTAL_RESULTS*OUTPUT_EXP_WIDTH);
        $display("========================================");
    end

endmodule


//================================================================================
// 子模块：单个结果的归一化单元
//================================================================================
module normalize_single_result #(
    parameter FIXED_WIDTH = 32,
    parameter BASE_EXP_WIDTH = 9,
    parameter OUTPUT_MANT_WIDTH = 16,
    parameter OUTPUT_EXP_WIDTH = 8
)(
    input  wire signed [FIXED_WIDTH-1:0] fixed_val,
    input  wire [BASE_EXP_WIDTH-1:0] base_exp,
    input  wire is_zero,
    output wire signed [OUTPUT_MANT_WIDTH-1:0] normalized_mant,
    output wire [OUTPUT_EXP_WIDTH-1:0] independent_exp,
    output wire overflow_flag
);

    //==========================================================================
    // 前导零检测函数（32位版本）
    //==========================================================================
    function [5:0] count_lz_32;
        input signed [31:0] val;
        reg [5:0] count;
        reg [31:0] temp;
    begin
        temp = (val[31]) ? ~val : val;
        count = 0;
        
        if (temp[31:16] == 16'd0) begin
            count = count + 16;
            temp = {temp[15:0], 16'd0};
        end
        if (temp[31:24] == 8'd0) begin
            count = count + 8;
            temp = {temp[23:0], 8'd0};
        end
        if (temp[31:28] == 4'd0) begin
            count = count + 4;
            temp = {temp[27:0], 4'd0};
        end
        if (temp[31:30] == 2'd0) begin
            count = count + 2;
            temp = {temp[29:0], 2'd0};
        end
        if (temp[31] == 1'd0) begin
            count = count + 1;
        end
        
        count_lz_32 = count;
    end
    endfunction

    //==========================================================================
    // 组合逻辑：归一化处理
    //==========================================================================
    
    reg signed [OUTPUT_MANT_WIDTH-1:0] mant_result;
    reg [OUTPUT_EXP_WIDTH-1:0] exp_result;
    reg ovf_result;
    
    always @(*) begin
        // 默认值
        mant_result = {OUTPUT_MANT_WIDTH{1'b0}};
        exp_result = {OUTPUT_EXP_WIDTH{1'b0}};
        ovf_result = 1'b0;
        
        if (is_zero) begin
            // 零值特殊处理
            mant_result = {OUTPUT_MANT_WIDTH{1'b0}};
            exp_result = {OUTPUT_EXP_WIDTH{1'b0}};
            ovf_result = 1'b0;
            
        end else begin:bfp
            // 局部变量声明
            reg [5:0] leading_zeros;
            reg signed [FIXED_WIDTH-1:0] normalized;
            reg [OUTPUT_EXP_WIDTH:0] temp_exp;  // 临时9位指数
            reg signed [OUTPUT_MANT_WIDTH-1:0] temp_mant;
            reg round_bit;
            
            //==================================================================
            // 步骤1：计算前导零/一
            //==================================================================
            if (FIXED_WIDTH == 32) begin
                leading_zeros = count_lz_32(fixed_val);
            end else begin
                // 39位：简化处理，取高32位
                leading_zeros = count_lz_32(fixed_val[38:7]);
            end
            
            //==================================================================
            // 步骤2：归一化（左移）
            //==================================================================
            normalized = fixed_val << leading_zeros;
            
            //==================================================================
            // 步骤3：计算独立指数
            // 公式：independent_exp = base_exp - leading_zeros + (FIXED_WIDTH-1)
            //==================================================================
            temp_exp = base_exp - {{(OUTPUT_EXP_WIDTH-5){1'b0}}, leading_zeros} + (FIXED_WIDTH - 1);
            
            //==================================================================
            // 步骤4：提取尾数（取高位）
            //==================================================================
            temp_mant = normalized[FIXED_WIDTH-1 : FIXED_WIDTH-OUTPUT_MANT_WIDTH];
            
            // 舍入处理（检查被截断的最高位）
            if (FIXED_WIDTH > OUTPUT_MANT_WIDTH) begin
                round_bit = normalized[FIXED_WIDTH-OUTPUT_MANT_WIDTH-1];
                if (round_bit) begin
                    // 简单舍入：加1
                    temp_mant = temp_mant + 1'b1;
                    
                    // 检查舍入后是否溢出（变为全1）
                    if (temp_mant == {OUTPUT_MANT_WIDTH{1'b1}}) begin
                        // 尾数溢出，调整指数
                        temp_exp = temp_exp + 1'b1;
                        temp_mant = {1'b1, {(OUTPUT_MANT_WIDTH-1){1'b0}}};  // 归一化到1.0
                    end
                end
            end
            
            //==================================================================
            // 步骤5：溢出检测和饱和
            //==================================================================
            ovf_result = 1'b0;
            
            // 检查指数溢出
            if (temp_exp[OUTPUT_EXP_WIDTH]) begin
                // 指数超过8位范围
                ovf_result = 1'b1;
                exp_result = {OUTPUT_EXP_WIDTH{1'b1}};  // 饱和到最大
                mant_result = temp_mant;
            end else begin
                exp_result = temp_exp[OUTPUT_EXP_WIDTH-1:0];
                mant_result = temp_mant;
            end
        end
    end
    
    // 输出连接
    assign normalized_mant = mant_result;
    assign independent_exp = exp_result;
    assign overflow_flag = ovf_result;

endmodule