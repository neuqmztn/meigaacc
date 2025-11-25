`timescale 1ns / 1ps

//================================================================================
// Linear Compute Engine v5.0 - Scheme B (8-bit I/O, Multi-Exp)
//
// 功能：执行矩阵乘法 Y[32×32] = X[32×32] × W[32×32]
//
// 核心特性：
// 1. 标准 8-bit 输入 x 8-bit 权重 (硬件效率最高)
// 2. 输出 16-bit 结果 (适配 Accumulator 或 高精度 GELU)
// 3. ✅ 核心修复：输出 [32*8-1:0] 的独立指数数组，支持逐Token动态范围
//
// 适配模块：
// - compute_engine v8.4
// - bfp_converter v1.4
//================================================================================

module linear_compute_engine #(
    parameter TOKEN_CHUNK   = 32,
    parameter INPUT_DIM     = 32,
    parameter OUTPUT_DIM    = 32,
    
    // BFP 参数
    parameter BFP_EXP_W     = 8,
    parameter BFP_MANT_W    = 8,      // 输入/权重数据位宽 (标准 8-bit)
    parameter OUTPUT_MANT_W = 16,     // 输出结果位宽 (16-bit)
    
    // CE 配置 (默认 4x8=32 PUs)
    parameter G_OUT         = 4,
    parameter T_OUT         = 8,
    parameter CE_OUTPUT_WIDTH = 32,
    parameter CE_BASE_EXP_WIDTH = 9,
    
    // 底层 PE 配置
    parameter NUM_PE_PER_GROUP = 2,
    parameter PE_TYPE_0     = 0,
    parameter PE_TYPE_1     = 2,
    parameter ELEM_PE0      = 16,
    parameter ELEM_PE1      = 8
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire start,
    output reg  done,
    output reg  busy,
    
    //================================================================================
    // 输入数据接口 (8-bit)
    //================================================================================
    // X: Token 块 [TOKEN_CHUNK × INPUT_DIM]
    input  wire [BFP_EXP_W-1:0] x_exp, // 假设输入 Block 共享指数 (若上游也是独立指数需修改)
    input  wire [TOKEN_CHUNK*INPUT_DIM*BFP_MANT_W-1:0] x_mant,
    
    // W: 权重矩阵 [INPUT_DIM × OUTPUT_DIM]
    input  wire [OUTPUT_DIM*BFP_EXP_W-1:0] w_exp_array,  
    input  wire [INPUT_DIM*OUTPUT_DIM*BFP_MANT_W-1:0] w_mant,
    
    //================================================================================
    // 输出数据接口 (16-bit)
    //================================================================================
    // Y: 结果块 [TOKEN_CHUNK × OUTPUT_DIM]
    // ✅ 关键：32个独立指数，每个 Token 一个
    output reg  [TOKEN_CHUNK*BFP_EXP_W-1:0] y_exp,
    output reg  [TOKEN_CHUNK*OUTPUT_DIM*OUTPUT_MANT_W-1:0] y_mant
);

    //================================================================================
    // 内部参数与信号
    //================================================================================
    localparam TOTAL_ELEM = INPUT_DIM; // CE 一次处理一行向量点积
    
    // 状态机状态
    localparam S_IDLE    = 4'd0;
    localparam S_LOAD    = 4'd1;
    localparam S_EXTRACT = 4'd2;
    localparam S_SEND    = 4'd3;
    localparam S_WAIT    = 4'd4;
    localparam S_SAMPLE  = 4'd5;
    localparam S_CONV    = 4'd6;
    localparam S_SAVE    = 4'd7;
    localparam S_DONE    = 4'd8;
    
    reg [3:0] state;
    reg [5:0] token_cnt; // 0~31

    // 输入缓存
    reg [BFP_EXP_W-1:0] x_exp_buf;
    reg [TOKEN_CHUNK*INPUT_DIM*BFP_MANT_W-1:0] x_mant_buf;
    reg [OUTPUT_DIM*BFP_EXP_W-1:0] w_exp_buf;
    reg [INPUT_DIM*OUTPUT_DIM*BFP_MANT_W-1:0] w_mant_buf;

    // CE 接口信号
    reg ce_input_valid;
    reg [BFP_EXP_W-1:0] ce_exp_X;
    reg [TOTAL_ELEM*BFP_MANT_W-1:0] ce_mant_X;
    
    wire [G_OUT*T_OUT-1:0] ce_result_valids;
    wire signed [G_OUT*T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed;
    wire [G_OUT*T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp;
    wire [G_OUT*T_OUT-1:0] ce_result_zero;

    // CE 输出缓存 (确保时序安全)
    reg signed [G_OUT*T_OUT*CE_OUTPUT_WIDTH-1:0] ce_fixed_buf;
    reg [G_OUT*T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_base_exp_buf;
    reg [G_OUT*T_OUT-1:0] ce_zero_buf;
    reg ce_buf_valid;

    // BFP Converter 接口信号
    wire [G_OUT*T_OUT-1:0] conv_valids;
    wire signed [G_OUT*T_OUT*OUTPUT_MANT_W-1:0] conv_mants;
    wire [BFP_EXP_W-1:0] conv_shared_exp;
    
    // 结果缓存
    reg [TOKEN_CHUNK*OUTPUT_DIM*OUTPUT_MANT_W-1:0] res_buf_mant;
    reg [TOKEN_CHUNK*BFP_EXP_W-1:0] res_buf_exp;

    //================================================================================
    // 1. Compute Engine 实例化
    //================================================================================
    compute_engine #(
        .G_OUT(G_OUT),
        .T_OUT(T_OUT),
        .NUM_PE(NUM_PE_PER_GROUP),
        .PE_TYPE_0(PE_TYPE_0),
        .PE_TYPE_1(PE_TYPE_1),
        .EXP_WIDTH(BFP_EXP_W),
        .MANT_WIDTH(BFP_MANT_W),       // 8-bit 乘法器
        .INPUT_MANT_WIDTH(BFP_MANT_W), // 8-bit 输入
        .ELEM_PE0(ELEM_PE0),
        .ELEM_PE1(ELEM_PE1),
        .TOTAL_ELEM(TOTAL_ELEM),
        .FIXED_WIDTH(CE_OUTPUT_WIDTH),
        .FIFO_DEPTH(16)
    ) u_ce (
        .clk(clk),
        .rst_n(rst_n),
        .flush(1'b0),
        
        .input_valid(ce_input_valid),
        .input_ready(), // 假设 FIFO 足够大，简化流控
        
        .exp_X(ce_exp_X),
        .mant_X_block(ce_mant_X),
        
        .exp_W_array(w_exp_buf),
        .mant_W_blocks(w_mant_buf), // 直接传输整个 8-bit 权重矩阵
        
        .result_valids(ce_result_valids),
        .result_ready(1'b1), // 始终准备好接收到 Buffer
        
        .result_fixed_array(ce_result_fixed),
        .result_base_exp_array(ce_result_base_exp),
        .result_zero_array(ce_result_zero)
    );

    //================================================================================
    // 2. CE 输出采样缓存
    //================================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ce_fixed_buf <= 0;
            ce_base_exp_buf <= 0;
            ce_zero_buf <= 0;
            ce_buf_valid <= 0;
        end else begin
            if (state == S_WAIT && |ce_result_valids && !ce_buf_valid) begin
                ce_fixed_buf <= ce_result_fixed;
                ce_base_exp_buf <= ce_result_base_exp;
                ce_zero_buf <= ce_result_zero;
                ce_buf_valid <= 1'b1;
            end else if (state == S_SAVE) begin
                ce_buf_valid <= 1'b0;
            end
        end
    end

    //================================================================================
    // 3. BFP Converter 实例化 (输出 16-bit)
    //================================================================================
    bfp_converter #(
        .TOTAL_RESULTS(G_OUT*T_OUT),
        .FIXED_WIDTH(CE_OUTPUT_WIDTH),
        .BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH),
        .OUTPUT_MANT_WIDTH(OUTPUT_MANT_W), // ✅ 16-bit 输出
        .OUTPUT_EXP_WIDTH(BFP_EXP_W)
    ) u_converter (
        .clk(clk),
        .rst_n(rst_n),
        .flush(1'b0),
        
        .input_valids({(G_OUT*T_OUT){ce_buf_valid}}),
        .input_fixed_array(ce_fixed_buf),
        .input_base_exp_array(ce_base_exp_buf),
        .input_zero_array(ce_zero_buf),
        
        .output_valids(conv_valids),
        .output_mant_array(conv_mants),
        .output_shared_exp(conv_shared_exp),
        .output_overflow()
    );

    //================================================================================
    // 4. 主控制状态机
    //================================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 0;
            busy <= 0;
            token_cnt <= 0;
            ce_input_valid <= 0;
            res_buf_exp <= 0;
            res_buf_mant <= 0;
            // 缓存清零略...
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= S_LOAD;
                        busy <= 1;
                        token_cnt <= 0;
                    end
                end

                S_LOAD: begin
                    // 缓存输入数据
                    x_exp_buf <= x_exp;
                    x_mant_buf <= x_mant;
                    w_exp_buf <= w_exp_array;
                    w_mant_buf <= w_mant;
                    state <= S_EXTRACT;
                end

                S_EXTRACT: begin
                    // 提取当前 Token (8-bit)
                    ce_exp_X <= x_exp_buf; 
                    ce_mant_X <= x_mant_buf[token_cnt*INPUT_DIM*BFP_MANT_W +: INPUT_DIM*BFP_MANT_W];
                    state <= S_SEND;
                end

                S_SEND: begin
                    ce_input_valid <= 1;
                    state <= S_WAIT;
                end

                S_WAIT: begin
                    ce_input_valid <= 0;
                    // 等待 CE 完成并被缓存
                    if (ce_buf_valid) begin
                        state <= S_SAMPLE;
                    end
                end

                S_SAMPLE: begin
                    // 等待 Converter 准备好 (简化：假设下一拍就好)
                    state <= S_CONV;
                end

                S_CONV: begin
                    if (|conv_valids) begin
                        state <= S_SAVE;
                    end
                end

                S_SAVE: begin
                    // 保存当前 Token 结果
                    // 1. 尾数 (16-bit)
                    res_buf_mant[token_cnt*OUTPUT_DIM*OUTPUT_MANT_W +: OUTPUT_DIM*OUTPUT_MANT_W] 
                        <= conv_mants;
                    
                    // 2. ✅ 指数 (独立保存)
                    res_buf_exp[token_cnt*BFP_EXP_W +: BFP_EXP_W] 
                        <= conv_shared_exp;

                    if (token_cnt < TOKEN_CHUNK - 1) begin
                        token_cnt <= token_cnt + 1;
                        state <= S_EXTRACT;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done <= 1;
                    busy <= 0;
                    y_exp <= res_buf_exp;   // 输出整个指数数组
                    y_mant <= res_buf_mant; // 输出整个尾数块
                    
                    if (!start) state <= S_IDLE;
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule