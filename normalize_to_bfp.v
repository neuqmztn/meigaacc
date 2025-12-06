//==============================================================
// ✅ 改进的normalize_to_bfp模块
// 
// 主要改进:
// 1. 支持连续enable输入
// 2. 内部自动流水线管理
// 3. 无需外部复杂控制
//==============================================================

module normalize_to_bfp#(
    parameter INPUT_WIDTH = 39,
    parameter EXP_WIDTH = 8,
    parameter MANT_WIDTH = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire enable,          // ✅ 可以连续assert
    input  wire flush,
    
    input  wire signed [INPUT_WIDTH-1:0] int_value,
    input  wire [EXP_WIDTH:0] base_exp,
    
    output reg sign,
    output reg [EXP_WIDTH-1:0] exponent,
    output reg [MANT_WIDTH-1:0] mantissa,
    output reg is_zero
);

//==============================================================
// Stage 1: 输入寄存
//==============================================================

reg signed [INPUT_WIDTH-1:0] int_value_r;
reg [EXP_WIDTH:0] base_exp_r;
reg stage1_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        int_value_r <= 0;
        base_exp_r <= 0;
        stage1_valid <= 1'b0;
    end else if (flush) begin
        stage1_valid <= 1'b0;
    end else if (enable) begin
        int_value_r <= int_value;
        base_exp_r <= base_exp;
        stage1_valid <= 1'b1;
    end else begin
        stage1_valid <= 1'b0;
    end
end

//==============================================================
// Stage 2: 归一化计算
//==============================================================

reg sign_s2;
reg [EXP_WIDTH-1:0] exponent_s2;
reg [MANT_WIDTH-1:0] mantissa_s2;
reg is_zero_s2;
reg stage2_valid;

// 工作变量
reg [INPUT_WIDTH-1:0] abs_value;
integer msb_pos;
integer i;
reg signed [EXP_WIDTH+1:0] norm_exp;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sign_s2 <= 1'b0;
        exponent_s2 <= {EXP_WIDTH{1'b0}};
        mantissa_s2 <= {MANT_WIDTH{1'b0}};
        is_zero_s2 <= 1'b1;
        stage2_valid <= 1'b0;
    end else if (flush) begin
        stage2_valid <= 1'b0;
    end else if (stage1_valid) begin
        
        // 零值检测
        if (int_value_r == 0) begin
            sign_s2 <= 1'b0;
            exponent_s2 <= 8'd0;
            mantissa_s2 <= 16'd0;
            is_zero_s2 <= 1'b1;
        end else begin
            // 提取符号
            sign_s2 <= int_value_r[INPUT_WIDTH-1];
            
            // 取绝对值
            if (int_value_r[INPUT_WIDTH-1]) begin
                abs_value = -int_value_r;
            end else begin
                abs_value = int_value_r;
            end
            
            // 查找MSB
            msb_pos = -1;
            for (i = INPUT_WIDTH-1; i >= 0; i = i - 1) begin
                if (abs_value[i] == 1'b1 && msb_pos == -1) begin
                    msb_pos = i;
                end
            end
            
            if (msb_pos == -1) begin
                sign_s2 <= 1'b0;
                exponent_s2 <= 8'd0;
                mantissa_s2 <= 16'd0;
                is_zero_s2 <= 1'b1;
            end else begin
                is_zero_s2 <= 1'b0;
                
                // 计算指数
                norm_exp = $signed({1'b0, base_exp_r}) - $signed((MANT_WIDTH - 1 - msb_pos));
                
                // 饱和处理
                if (norm_exp > 127) begin
                    exponent_s2 <= 8'd127;
                end else if (norm_exp < -128) begin
                    exponent_s2 <= 8'd128;
                end else begin
                    exponent_s2 <= norm_exp[7:0];
                end
                
                // 提取尾数
                if (msb_pos >= MANT_WIDTH-1) begin
                    mantissa_s2 <= abs_value >> (msb_pos - (MANT_WIDTH-1));
                end else begin
                    mantissa_s2 <= abs_value << ((MANT_WIDTH-1) - msb_pos);
                end
            end
        end
        
        stage2_valid <= 1'b1;
    end else begin
        stage2_valid <= 1'b0;
    end
end

//==============================================================
// Stage 3: 输出寄存
//==============================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sign <= 1'b0;
        exponent <= {EXP_WIDTH{1'b0}};
        mantissa <= {MANT_WIDTH{1'b0}};
        is_zero <= 1'b1;
    end else if (flush) begin
        sign <= 1'b0;
        exponent <= {EXP_WIDTH{1'b0}};
        mantissa <= {MANT_WIDTH{1'b0}};
        is_zero <= 1'b1;
    end else if (stage2_valid) begin
        sign <= sign_s2;
        exponent <= exponent_s2;
        mantissa <= mantissa_s2;
        is_zero <= is_zero_s2;
    end
end

endmodule