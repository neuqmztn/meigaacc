`timescale 1ns / 1ps

module bfp_adder #(
    parameter EXP_WIDTH = 8,
    parameter MANT_WIDTH = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire enable,
    input  wire flush,
    
    input  wire sign_a, 
    input  wire [EXP_WIDTH-1:0] exp_a,
    input  wire [MANT_WIDTH-1:0] mant_a,
    input  wire zero_a,
    
    input  wire sign_b,
    input  wire [EXP_WIDTH-1:0] exp_b,
    input  wire [MANT_WIDTH-1:0] mant_b,
    input  wire zero_b,
    
    output reg sign_out,
    output reg [EXP_WIDTH-1:0] exp_out,
    output reg [MANT_WIDTH-1:0] mant_out,
    output reg zero_out
);

    //==============================================================
    // Stage 1: 输入对齐 (Combinational or Registered)
    // 为了时序更好，我们保持2级流水线
    //==============================================================
    
    reg [EXP_WIDTH-1:0] st1_max_exp;
    reg [MANT_WIDTH:0]  st1_mant_a_aligned; // 多1位用于进位
    reg [MANT_WIDTH:0]  st1_mant_b_aligned;
    reg                 st1_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st1_max_exp <= 0;
            st1_mant_a_aligned <= 0;
            st1_mant_b_aligned <= 0;
            st1_valid <= 0;
        end else if (flush) begin
            st1_valid <= 0;
        end else if (enable) begin
            st1_valid <= 1;
            
            // 简单的对齐逻辑：谁的指数小，谁右移
            if (exp_a >= exp_b) begin
                st1_max_exp <= exp_a;
                st1_mant_a_aligned <= {1'b0, mant_a}; // 扩展一位
                // 限制移位最大值，防止逻辑过于复杂
                if ((exp_a - exp_b) > MANT_WIDTH)
                    st1_mant_b_aligned <= 0;
                else
                    st1_mant_b_aligned <= {1'b0, mant_b} >> (exp_a - exp_b);
            end else begin
                st1_max_exp <= exp_b;
                st1_mant_b_aligned <= {1'b0, mant_b};
                if ((exp_b - exp_a) > MANT_WIDTH)
                    st1_mant_a_aligned <= 0;
                else
                    st1_mant_a_aligned <= {1'b0, mant_a} >> (exp_b - exp_a);
            end
        end else begin
            st1_valid <= 0;
        end
    end

    //==============================================================
    // Stage 2: 加法与溢出处理
    //==============================================================
    
    reg [MANT_WIDTH+1:0] sum_raw; // 17 bits
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            exp_out <= 0;
            mant_out <= 0;
            zero_out <= 1;
            sign_out <= 0;
        end else if (st1_valid) begin
            // 执行加法
            sum_raw = st1_mant_a_aligned + st1_mant_b_aligned;
            
            // 检查溢出 (Bit 16 is 1?)
            if (sum_raw[MANT_WIDTH]) begin
                // 发生溢出，右移1位，指数+1
                mant_out <= sum_raw[MANT_WIDTH:1];
                exp_out  <= st1_max_exp + 1;
            end else begin
                // 无溢出，保持原样
                mant_out <= sum_raw[MANT_WIDTH-1:0];
                exp_out  <= st1_max_exp;
            end
            
            zero_out <= (sum_raw == 0);
            sign_out <= 0; // 始终为正
        end
    end

endmodule