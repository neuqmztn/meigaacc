`timescale 1ns / 1ps

module multi_head_bfp_converter #(
    parameter NUM_HEADS = 4,              // 头数
    parameter RESULTS_PER_HEAD = 8,       // 每头的结果数
    parameter FIXED_WIDTH = 32,           // 定点输入位宽（推荐32位）
    parameter BASE_EXP_WIDTH = 9,         // 基础指数位宽
    parameter OUTPUT_MANT_WIDTH = 8,      // 输出尾数位宽
    parameter OUTPUT_EXP_WIDTH = 8        // 输出指数位宽
)(
    input  wire clk,
    input  wire rst_n,
    input  wire flush,
    
    //==========================================================================
    // 输入：所有头的定点结果
    //==========================================================================
    input  wire [NUM_HEADS*RESULTS_PER_HEAD-1:0] input_valids,
    input  wire signed [NUM_HEADS*RESULTS_PER_HEAD*FIXED_WIDTH-1:0] input_fixed_array,
    input  wire [NUM_HEADS*RESULTS_PER_HEAD*BASE_EXP_WIDTH-1:0] input_base_exp_array,
    input  wire [NUM_HEADS*RESULTS_PER_HEAD-1:0] input_zero_array,
    
    //==========================================================================
    // 输出：每个头有独立的共享指数
    //==========================================================================
    output wire [NUM_HEADS*RESULTS_PER_HEAD-1:0] output_valids,
    output wire signed [NUM_HEADS*RESULTS_PER_HEAD*OUTPUT_MANT_WIDTH-1:0] output_mant_array,
    output wire [NUM_HEADS*OUTPUT_EXP_WIDTH-1:0] output_shared_exps,  // 每头一个共享指数
    output wire [NUM_HEADS-1:0] output_overflow                        // 每头一个溢出标志
);

//==============================================================================
// 参数检查
//==============================================================================

localparam TOTAL_RESULTS = NUM_HEADS * RESULTS_PER_HEAD;

genvar h;
generate
    for (h = 0; h < NUM_HEADS; h = h + 1) begin : gen_head_converters
        
        // 计算该头的数据索引范围
        localparam HEAD_START = h * RESULTS_PER_HEAD;
        localparam HEAD_END = (h + 1) * RESULTS_PER_HEAD - 1;
        
        // 提取该头的输入信号
        wire [RESULTS_PER_HEAD-1:0] head_input_valids;
        wire signed [RESULTS_PER_HEAD*FIXED_WIDTH-1:0] head_input_fixed;
        wire [RESULTS_PER_HEAD*BASE_EXP_WIDTH-1:0] head_input_base_exp;
        wire [RESULTS_PER_HEAD-1:0] head_input_zero;
        
        assign head_input_valids = input_valids[HEAD_END:HEAD_START];
        assign head_input_fixed = input_fixed_array[(HEAD_END+1)*FIXED_WIDTH-1 : HEAD_START*FIXED_WIDTH];
        assign head_input_base_exp = input_base_exp_array[(HEAD_END+1)*BASE_EXP_WIDTH-1 : HEAD_START*BASE_EXP_WIDTH];
        assign head_input_zero = input_zero_array[HEAD_END:HEAD_START];
        
        // 该头的输出信号
        wire [RESULTS_PER_HEAD-1:0] head_output_valids;
        wire signed [RESULTS_PER_HEAD*OUTPUT_MANT_WIDTH-1:0] head_output_mants;
        wire [OUTPUT_EXP_WIDTH-1:0] head_shared_exp;
        wire head_overflow;
        
        // 为该头实例化BFP转换器
        bfp_converter #(
            .TOTAL_RESULTS(RESULTS_PER_HEAD),
            .FIXED_WIDTH(FIXED_WIDTH),
            .BASE_EXP_WIDTH(BASE_EXP_WIDTH),
            .OUTPUT_MANT_WIDTH(OUTPUT_MANT_WIDTH),
            .OUTPUT_EXP_WIDTH(OUTPUT_EXP_WIDTH)
        ) u_head_converter (
            .clk(clk),
            .rst_n(rst_n),
            .flush(flush),
            
            // 该头的输入
            .input_valids(head_input_valids),
            .input_fixed_array(head_input_fixed),
            .input_base_exp_array(head_input_base_exp),
            .input_zero_array(head_input_zero),
            
            // 该头的输出
            .output_valids(head_output_valids),
            .output_mant_array(head_output_mants),
            .output_shared_exp(head_shared_exp),
            .output_overflow(head_overflow)
        );
        
        // 连接到输出总线
        assign output_valids[HEAD_END:HEAD_START] = head_output_valids;
        assign output_mant_array[(HEAD_END+1)*OUTPUT_MANT_WIDTH-1 : HEAD_START*OUTPUT_MANT_WIDTH] = head_output_mants;
        assign output_shared_exps[(h+1)*OUTPUT_EXP_WIDTH-1 : h*OUTPUT_EXP_WIDTH] = head_shared_exp;
        assign output_overflow[h] = head_overflow;
        
    end
endgenerate



endmodule