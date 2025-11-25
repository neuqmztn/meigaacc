`timescale 1ns / 1ps

//================================================================================
// GELU Engine v4.1 (Scheme B - PWL Logic Version) - Verilog-2001
//
// 功能概述：
// 1. 高精度输入 -> PWL拟合(无LUT RAM) -> 压缩输出 (16-bit -> 8-bit)
// 2. 32个独立指数直接透传
//
// 算法改进：
// 使用 16段 分段线性逼近 (PWL) 替代大容量 LUT。
// 优势：不消耗 Block RAM，规避了初始化赋值问题，精度可控。
//
// 拟合区间 [-4.0, 4.0]，在此区间外：
// x > 4.0  -> y = x
// x < -4.0 -> y = 0
//================================================================================
module gelu_engine #(
    parameter TOKEN_CHUNK    = 32,
    parameter MATRIX_SIZE    = 32,
    parameter FEATURE_SIZE   = 32,
    
    parameter BFP_EXP_W      = 8,
    parameter INPUT_MANT_W   = 16,      // Q3.12 格式
    parameter OUTPUT_MANT_W  = 8,       // 输出压缩位宽
    parameter PIPELINE_STAGES = 4
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    output reg  busy,
    
    input  wire [TOKEN_CHUNK*BFP_EXP_W-1:0] in_exp,
    input  wire [MATRIX_SIZE*FEATURE_SIZE*INPUT_MANT_W-1:0] in_mant,
    
    output reg  [TOKEN_CHUNK*BFP_EXP_W-1:0] out_exp,
    output reg  [MATRIX_SIZE*FEATURE_SIZE*OUTPUT_MANT_W-1:0] out_mant
);

    localparam FIXED_ONE = 16'h1000; 

    // 状态定义
    localparam IDLE = 3'd0;
    localparam PROCESS = 3'd1;
    localparam DONE_STATE = 3'd2;
    reg [2:0] state;

    // 流水线寄存器 - 使用signed类型简化有符号运算
    reg signed [INPUT_MANT_W-1:0] pipe_data_s0 [0:FEATURE_SIZE-1];
    reg signed [INPUT_MANT_W-1:0] pipe_k_s1 [0:FEATURE_SIZE-1];
    reg signed [INPUT_MANT_W-1:0] pipe_b_s1 [0:FEATURE_SIZE-1];
    reg signed [INPUT_MANT_W-1:0] pipe_data_s1 [0:FEATURE_SIZE-1];
    reg signed [INPUT_MANT_W-1:0] pipe_res_s2 [0:FEATURE_SIZE-1];

    reg pipe_valid [0:PIPELINE_STAGES-1];
    reg [4:0] row_idx_pipe [0:PIPELINE_STAGES-1];
    
    // 计数器改为行计数 (0~31)
    reg [5:0] process_row_cnt; 
    
    // 修复1: 为每个流水线阶段使用独立的循环变量
    integer i_rst, j0, j1, k2, m3;

    // PWL 系数查找函数
    // 输入：16位有符号定点数 (Q3.12格式)
    // 输出：32位 {slope[15:0], intercept[15:0]}

    function [31:0] get_pwl_coef;
        input signed [INPUT_MANT_W-1:0] x;
        reg [15:0] slope;
        reg [15:0] intercept;
        reg [3:0] seg_idx;
        begin
            // 修复点：使用 $signed() 强制常数为有符号类型进行比较
            if (x >= $signed(16'h4000)) begin      // x >= 4.0
                slope     = FIXED_ONE;
                intercept = 16'd0;
            end 
            else if (x <= $signed(16'hC000)) begin // x <= -4.0
                slope     = 16'd0;
                intercept = 16'd0;
            end 
            else begin
                // 线性拟合区
                seg_idx = x[14:11]; 
                case (seg_idx)
                    // === 正数部分 ===
                    4'h0: begin slope = 16'h0B10; intercept = 16'h0000; end 
                    4'h1: begin slope = 16'h0FDC; intercept = 16'hFD9A; end 
                    4'h2: begin slope = 16'h11D7; intercept = 16'hFB9F; end // x=1.0 在此
                    4'h3: begin slope = 16'h11C6; intercept = 16'hFBB9; end 
                    4'h4: begin slope = 16'h116E; intercept = 16'hFC69; end 
                    4'h5: begin slope = 16'h0FE6; intercept = 16'h003D; end 
                    4'h6: begin slope = 16'h101A; intercept = 16'hFFA1; end 
                    4'h7: begin slope = 16'h1008; intercept = 16'hFFE0; end 

                    // === 负数部分 ===
                    4'h8: begin slope = 16'h0000; intercept = 16'h0000; end 
                    4'h9: begin slope = 16'hFFE4; intercept = 16'hFF9C; end 
                    4'hA: begin slope = 16'hFFA2; intercept = 16'hFED6; end 
                    4'hB: begin slope = 16'hFF0A; intercept = 16'hFD5A; end 
                    4'hC: begin slope = 16'hFE4A; intercept = 16'hFBDA; end 
                    4'hD: begin slope = 16'hFE16; intercept = 16'hFB8C; end 
                    4'hE: begin slope = 16'h0024; intercept = 16'hFD9A; end 
                    4'hF: begin slope = 16'h04EF; intercept = 16'h0000; end 
                    
                    default: begin slope = FIXED_ONE; intercept = 16'd0; end
                endcase
            end
            get_pwl_coef = {slope, intercept};
        end
    endfunction

    // 主逻辑
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            busy <= 1'b0;
            process_row_cnt <= 6'd0;
            out_exp <= {TOKEN_CHUNK*BFP_EXP_W{1'b0}};
            out_mant <= {MATRIX_SIZE*FEATURE_SIZE*OUTPUT_MANT_W{1'b0}};
            for(i_rst=0; i_rst<PIPELINE_STAGES; i_rst=i_rst+1) pipe_valid[i_rst] <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state <= PROCESS;
                        busy <= 1'b1;
                        process_row_cnt <= 6'd0;
                        out_exp <= in_exp;
                        for(i_rst=0; i_rst<PIPELINE_STAGES; i_rst=i_rst+1) pipe_valid[i_rst] <= 1'b0;
                    end
                end

                PROCESS: begin
                    //------------------------------------------------------------
                    // Stage 0: 并行数据加载 (Row Parallel Load) - 使用 j0
                    //------------------------------------------------------------
                    if (process_row_cnt < MATRIX_SIZE) begin
                        // 一次加载整行 32 个数据
                        for (j0=0; j0<FEATURE_SIZE; j0=j0+1) begin
                            pipe_data_s0[j0] <= $signed(in_mant[(process_row_cnt*FEATURE_SIZE + j0)*INPUT_MANT_W +: INPUT_MANT_W]);
                        end
                        row_idx_pipe[0] <= process_row_cnt[4:0];
                        pipe_valid[0] <= 1'b1;
                        process_row_cnt <= process_row_cnt + 1'b1;
                    end else begin
                        pipe_valid[0] <= 1'b0;
                        // 等待流水线排空
                        if (!pipe_valid[0] && !pipe_valid[1] && !pipe_valid[2] && !pipe_valid[3]) begin
                            state <= DONE_STATE;
                        end
                    end
                    
                    //------------------------------------------------------------
                    // Stage 1: 系数查找 - 使用 j1
                    //------------------------------------------------------------
                    if (pipe_valid[0]) begin
                        for(j1=0; j1<FEATURE_SIZE; j1=j1+1) begin
                            {pipe_k_s1[j1], pipe_b_s1[j1]} <= get_pwl_coef(pipe_data_s0[j1]);
                            pipe_data_s1[j1] <= pipe_data_s0[j1];
                        end
                        row_idx_pipe[1] <= row_idx_pipe[0];
                        pipe_valid[1] <= 1'b1;
                    end else begin
                        pipe_valid[1] <= 1'b0;
                    end
                    
                    //------------------------------------------------------------
                    // Stage 2: 乘加运算 - 使用 k2
                    // 修复2: 通过添加 32'sd0 强制32位乘法精度，防止截断
                    //------------------------------------------------------------
                    if (pipe_valid[1]) begin
                        for(k2=0; k2<FEATURE_SIZE; k2=k2+1) begin
                            // 强制32位运算：32'sd0 + (16bit * 16bit) 确保乘法结果保持32位
                            // 然后算术右移12位，再加上截距
                            pipe_res_s2[k2] <= ((32'sd0 + (pipe_data_s1[k2] * pipe_k_s1[k2])) >>> 12) 
                                            + pipe_b_s1[k2];
                        end
                        row_idx_pipe[2] <= row_idx_pipe[1];
                        pipe_valid[2] <= 1'b1;
                    end else begin
                        pipe_valid[2] <= 1'b0;
                    end

                    //------------------------------------------------------------
                    // Stage 3: 输出截断与并行写入 - 使用 m3
                    //------------------------------------------------------------
                    if (pipe_valid[2]) begin
                        for(m3=0; m3<FEATURE_SIZE; m3=m3+1) begin
                            if (INPUT_MANT_W > OUTPUT_MANT_W) begin
                                out_mant[(row_idx_pipe[2]*FEATURE_SIZE + m3)*OUTPUT_MANT_W +: OUTPUT_MANT_W]
                                    <= pipe_res_s2[m3][INPUT_MANT_W-1 -: OUTPUT_MANT_W];
                            end else begin
                                out_mant[(row_idx_pipe[2]*FEATURE_SIZE + m3)*OUTPUT_MANT_W +: OUTPUT_MANT_W]
                                    <= pipe_res_s2[m3][OUTPUT_MANT_W-1:0];
                            end
                        end
                        pipe_valid[3] <= 1'b1;
                    end else begin
                        pipe_valid[3] <= 1'b0;
                    end
                end
                
                DONE_STATE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule