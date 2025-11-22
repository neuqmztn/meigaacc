`timescale 1ns / 1ps

//==============================================================
// BFP加法器 - 修复版
// 
// 关键修复：
// 1. 正确的尾数扩展位宽
// 2. 正确的指数补偿逻辑
// 3. 正确的截取方式
//==============================================================

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
// Stage 1: 输入寄存
//==============================================================
reg sign_a_r, sign_b_r;
reg [EXP_WIDTH-1:0] exp_a_r, exp_b_r;
reg [MANT_WIDTH-1:0] mant_a_r, mant_b_r;
reg zero_a_r, zero_b_r;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sign_a_r <= 1'b0;
        sign_b_r <= 1'b0;
        exp_a_r <= {EXP_WIDTH{1'b0}};
        exp_b_r <= {EXP_WIDTH{1'b0}};
        mant_a_r <= {MANT_WIDTH{1'b0}};
        mant_b_r <= {MANT_WIDTH{1'b0}};
        zero_a_r <= 1'b1;
        zero_b_r <= 1'b1;
    end else if (flush) begin
        zero_a_r <= 1'b1;
        zero_b_r <= 1'b1;
    end else if (enable) begin
        sign_a_r <= sign_a;
        sign_b_r <= sign_b;
        exp_a_r <= exp_a;
        exp_b_r <= exp_b;
        mant_a_r <= mant_a;
        mant_b_r <= mant_b;
        zero_a_r <= zero_a;
        zero_b_r <= zero_b;
    end
end

//==============================================================
// Stage 2: 组合逻辑计算
//==============================================================

// 特殊情况
wire both_zero = zero_a_r & zero_b_r;
wire only_a_valid = (~zero_a_r) & zero_b_r;
wire only_b_valid = zero_a_r & (~zero_b_r);

// 指数对齐
wire exp_a_larger = (exp_a_r > exp_b_r);
wire exp_equal = (exp_a_r == exp_b_r);
wire [EXP_WIDTH-1:0] exp_diff = exp_a_larger ? (exp_a_r - exp_b_r) : (exp_b_r - exp_a_r);
wire [EXP_WIDTH-1:0] exp_max = exp_a_larger ? exp_a_r : exp_b_r;

// ✅ 关键修复：使用3位保护位（不是4位！）
// 扩展格式：{mant[15:0], 3'b000} = 19-bit
wire [MANT_WIDTH+2:0] mant_a_ext = exp_a_larger ? {mant_a_r, 3'b000} : 
                                    (exp_diff < MANT_WIDTH+3) ? ({mant_a_r, 3'b000} >> exp_diff) : 
                                    {(MANT_WIDTH+3){1'b0}};

wire [MANT_WIDTH+2:0] mant_b_ext = exp_a_larger ? 
                                    (exp_diff < MANT_WIDTH+3) ? ({mant_b_r, 3'b000} >> exp_diff) : 
                                    {(MANT_WIDTH+3){1'b0}} :
                                    {mant_b_r, 3'b000};

// 加减法控制
wire sign_same = (sign_a_r == sign_b_r);
wire a_larger_mag = exp_a_larger || (exp_equal && (mant_a_r >= mant_b_r));

// ✅ 修复：使用20-bit来防止溢出
wire [MANT_WIDTH+3:0] mant_sum = sign_same ? 
                                  ({1'b0, mant_a_ext} + {1'b0, mant_b_ext}) :
                                  (a_larger_mag ? 
                                   ({1'b0, mant_a_ext} - {1'b0, mant_b_ext}) :
                                   ({1'b0, mant_b_ext} - {1'b0, mant_a_ext}));

// 结果符号
wire result_sign = sign_same ? sign_a_r : (a_larger_mag ? sign_a_r : sign_b_r);

// 前导零检测
function integer clz;
    input [MANT_WIDTH+3:0] value;
    integer i;
    reg found;
    begin
        clz = MANT_WIDTH + 4;
        found = 0;
        for (i = MANT_WIDTH+3; i >= 0; i = i - 1) begin
            if (!found && value[i] == 1'b1) begin
                clz = MANT_WIDTH + 3 - i;
                found = 1;
            end
        end
    end
endfunction

wire [EXP_WIDTH-1:0] leading_zeros = clz(mant_sum);

// 归一化
wire overflow = mant_sum[MANT_WIDTH+3];  // bit[19]

wire [MANT_WIDTH+3:0] mant_normalized;
wire [EXP_WIDTH:0] exp_normalized;

assign mant_normalized = overflow ? (mant_sum >> 1) : 
                         (leading_zeros > 0 && leading_zeros < MANT_WIDTH+4) ? 
                         (mant_sum << leading_zeros) : 
                         mant_sum;

// ✅ 关键修复：指数调整
assign exp_normalized = overflow ? ({1'b0, exp_max} + 9'd1) :
                        (leading_zeros > 0 && {1'b0, exp_max} >= {1'b0, leading_zeros}) ? 
                        ({1'b0, exp_max} - {1'b0, leading_zeros}) :
                        9'd0;

// ✅ 修复：正确的截取 [18:3]
// 因为扩展了3位，所以截取[MANT_WIDTH+2:3]
wire [MANT_WIDTH-1:0] mant_result = mant_normalized[MANT_WIDTH+2:3];

wire result_is_zero = (mant_sum == 0) || both_zero;

//==============================================================
// Stage 3: 输出寄存
//==============================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sign_out <= 1'b0;
        exp_out <= {EXP_WIDTH{1'b0}};
        mant_out <= {MANT_WIDTH{1'b0}};
        zero_out <= 1'b1;
    end else if (flush) begin
        sign_out <= 1'b0;
        exp_out <= {EXP_WIDTH{1'b0}};
        mant_out <= {MANT_WIDTH{1'b0}};
        zero_out <= 1'b1;
    end else if (enable) begin
        if (both_zero) begin
            sign_out <= 1'b0;
            exp_out <= {EXP_WIDTH{1'b0}};
            mant_out <= {MANT_WIDTH{1'b0}};
            zero_out <= 1'b1;
        end else if (only_a_valid) begin
            sign_out <= sign_a_r;
            exp_out <= exp_a_r;
            mant_out <= mant_a_r;
            zero_out <= 1'b0;
        end else if (only_b_valid) begin
            sign_out <= sign_b_r;
            exp_out <= exp_b_r;
            mant_out <= mant_b_r;
            zero_out <= 1'b0;
        end else if (result_is_zero) begin
            sign_out <= 1'b0;
            exp_out <= {EXP_WIDTH{1'b0}};
            mant_out <= {MANT_WIDTH{1'b0}};
            zero_out <= 1'b1;
        end else begin
            sign_out <= result_sign;
            exp_out <= exp_normalized[EXP_WIDTH-1:0];
            mant_out <= mant_result;
            zero_out <= 1'b0;
        end
    end
end

endmodule