`timescale 1ns / 1ps

//==============================================================
// GELU激活引擎 - 优化版本
//==============================================================

module gelu_engine #(
    parameter MATRIX_SIZE = 32,
    parameter FEATURE_SIZE = 32,
    parameter BFP_EXP_W = 8,
    parameter ACC_MANT_W = 15,
    parameter LUT_DEPTH = 512,
    parameter PIPELINE_STAGES = 4
)(
    input  wire clk,
    input  wire rst_n,
    
    input  wire start,
    output reg  done,
    
    // 输入矩阵 [32x32]
    input  wire [BFP_EXP_W-1:0] h_in_exp,
    input  wire [MATRIX_SIZE*FEATURE_SIZE*ACC_MANT_W-1:0] h_in_mant,
    
    // 输出矩阵 [32x32]
    output reg  [BFP_EXP_W-1:0] h_out_exp,
    output reg  [MATRIX_SIZE*FEATURE_SIZE*ACC_MANT_W-1:0] h_out_mant
);

//==============================================================
// GELU查找表
//==============================================================

// 双精度LUT：正负分开
reg [ACC_MANT_W-1:0] gelu_lut_pos [0:LUT_DEPTH/2-1];
reg [ACC_MANT_W-1:0] gelu_lut_neg [0:LUT_DEPTH/2-1];

// LUT初始化
initial begin:gelu
    integer i;
    real x, gelu_val;
    
    for (i = 0; i < LUT_DEPTH/2; i = i + 1) begin
        // 正数部分 [0, 8]
        x = i * 8.0 / (LUT_DEPTH/2);
        gelu_val = x * 0.5 * (1.0 + tanh(sqrt(2.0/3.14159) * (x + 0.044715 * x * x * x)));
        gelu_lut_pos[i] = $rtoi(gelu_val * (1 << (ACC_MANT_W-3)));
        
        // 负数部分 [-8, 0]
        x = -i * 8.0 / (LUT_DEPTH/2);
        gelu_val = x * 0.5 * (1.0 + tanh(sqrt(2.0/3.14159) * (x + 0.044715 * x * x * x)));
        gelu_lut_neg[i] = $rtoi(gelu_val * (1 << (ACC_MANT_W-3)));
    end
end

//==============================================================
// 流水线处理
//==============================================================

// 流水线寄存器
reg [ACC_MANT_W-1:0] pipe_reg [0:PIPELINE_STAGES-1][0:31];
reg [4:0] row_idx [0:PIPELINE_STAGES-1];
reg [4:0] col_idx [0:PIPELINE_STAGES-1];
reg pipe_valid [0:PIPELINE_STAGES-1];

// 处理计数器
reg [9:0] process_cnt;
reg [4:0] curr_row, curr_col;

// 状态机
reg [2:0] state;
localparam IDLE = 3'd0;
localparam PROCESS = 3'd1;
localparam FLUSH = 3'd2;
localparam DONE = 3'd3;
integer i;
//==============================================================
// 主处理逻辑
//==============================================================

always @(posedge clk or negedge rst_n) begin:gelu2
    if (!rst_n) begin
        state <= IDLE;
        done <= 1'b0;
        process_cnt <= 10'd0;
        curr_row <= 5'd0;
        curr_col <= 5'd0;
        h_out_exp <= {BFP_EXP_W{1'b0}};
        
        // 清空流水线
        for (i = 0; i < PIPELINE_STAGES; i = i + 1) begin
            pipe_valid[i] <= 1'b0;
            row_idx[i] <= 5'd0;
            col_idx[i] <= 5'd0;
        end
        
    end else begin
        case (state)
            IDLE: begin
                done <= 1'b0;
                if (start) begin
                    state <= PROCESS;
                    process_cnt <= 10'd0;
                    curr_row <= 5'd0;
                    curr_col <= 5'd0;
                    h_out_exp <= h_in_exp;  // GELU保持指数不变（简化）
                end
            end
            
            PROCESS: begin
                // Stage 0: 输入
                if (process_cnt < MATRIX_SIZE * FEATURE_SIZE) begin:process
                    reg [ACC_MANT_W-1:0] input_val;
                    
                    // 提取输入值
                    input_val = h_in_mant[(curr_row*FEATURE_SIZE + curr_col)*ACC_MANT_W +: ACC_MANT_W];
                    
                    // 送入流水线
                    pipe_reg[0][curr_col] <= input_val;
                    row_idx[0] <= curr_row;
                    col_idx[0] <= curr_col;
                    pipe_valid[0] <= 1'b1;
                    
                    // 更新索引
                    if (curr_col == FEATURE_SIZE - 1) begin
                        curr_col <= 5'd0;
                        curr_row <= curr_row + 1;
                    end else begin
                        curr_col <= curr_col + 1;
                    end
                    
                    process_cnt <= process_cnt + 1;
                end else begin
                    pipe_valid[0] <= 1'b0;
                    if (!pipe_valid[PIPELINE_STAGES-1]) begin
                        state <= DONE;
                    end
                end
                
                // Stage 1: 地址计算
                if (pipe_valid[0]) begin:adress
                    reg [8:0] lut_addr;
                    reg is_neg;
                    
                    is_neg = pipe_reg[0][col_idx[0]][ACC_MANT_W-1];
                    if (is_neg) begin
                        lut_addr = (-pipe_reg[0][col_idx[0]]) >> 6;
                    end else begin
                        lut_addr = pipe_reg[0][col_idx[0]] >> 6;
                    end
                    
                    pipe_reg[1][col_idx[0]] <= {is_neg, lut_addr};
                    row_idx[1] <= row_idx[0];
                    col_idx[1] <= col_idx[0];
                    pipe_valid[1] <= 1'b1;
                end else begin
                    pipe_valid[1] <= 1'b0;
                end
                
                // Stage 2: 查表
                if (pipe_valid[1]) begin:lut
                    reg [ACC_MANT_W-1:0] lut_val;
                    reg is_neg;
                    reg [8:0] addr;
                    
                    {is_neg, addr} = pipe_reg[1][col_idx[1]];
                    
                    if (is_neg) begin
                        lut_val = gelu_lut_neg[addr];
                    end else begin
                        lut_val = gelu_lut_pos[addr];
                    end
                    
                    pipe_reg[2][col_idx[1]] <= lut_val;
                    row_idx[2] <= row_idx[1];
                    col_idx[2] <= col_idx[1];
                    pipe_valid[2] <= 1'b1;
                end else begin
                    pipe_valid[2] <= 1'b0;
                end
                
                // Stage 3: 输出
                if (pipe_valid[2]) begin
                    h_out_mant[(row_idx[2]*FEATURE_SIZE + col_idx[2])*ACC_MANT_W +: ACC_MANT_W] 
                        <= pipe_reg[2][col_idx[2]];
                    
                    row_idx[3] <= row_idx[2];
                    col_idx[3] <= col_idx[2];
                    pipe_valid[3] <= 1'b1;
                end else begin
                    pipe_valid[3] <= 1'b0;
                end
            end
            
            DONE: begin
                done <= 1'b1;
                state <= IDLE;
            end
        endcase
    end
end

endmodule