`timescale 1ns / 1ps

//================================================================================
// BFP to Q4.12 Converter (Fixed Version)
// 
// 修复记录：
// 1. [FIX] Stage 1: 修正了 bfp_exp 的符号扩展逻辑，正确处理负指数。
// 2. [FIX] Stage 3: 修正了饱和判断中的符号扩展，防止将负的最大值误判为正数。
//================================================================================

module bfp_to_q412_converter #(
    parameter BFP_MANT_WIDTH = 16,      // BFP尾数位宽
    parameter BFP_EXP_WIDTH  = 8,       // BFP指数位宽
    parameter Q412_WIDTH     = 16,      // Q4.12位宽
    parameter Q412_FRAC_BITS = 12       // Q4.12小数位数
)(
    input  wire clk,
    input  wire rst_n,
    
    // 控制接口
    input  wire         valid_in,
    output reg          valid_out,
    
    // BFP输入
    input  wire signed [BFP_MANT_WIDTH-1:0] bfp_mant,
    input  wire signed [BFP_EXP_WIDTH-1:0]  bfp_exp,
    
    // Q4.12输出
    output reg  signed [Q412_WIDTH-1:0] q412_data,
    
    // 状态和调试
    output reg          overflow,
    output reg          underflow,
    output wire [2:0]   pipeline_stage
);

//================================================================================
// 内部参数
//================================================================================
localparam signed [Q412_WIDTH-1:0] Q412_MAX =  16'sd32767;  // +7.9998
localparam signed [Q412_WIDTH-1:0] Q412_MIN = -16'sd32768;  // -8.0

//================================================================================
// Stage 1: 移位量计算和特殊情况检测
//================================================================================

// Stage 1 寄存器声明 (必须在 always 块之前!)
reg signed [BFP_MANT_WIDTH-1:0] s1_mant;
reg signed [8:0]                s1_shift_amount;  // 9位以处理溢出
reg                             s1_shift_left;    // 1=左移, 0=右移
reg                             s1_is_zero;       // 输入为零
reg                             s1_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s1_mant         <= {BFP_MANT_WIDTH{1'b0}};
        s1_shift_amount <= 9'd0;
        s1_shift_left   <= 1'b0;
        s1_is_zero      <= 1'b0;
        s1_valid        <= 1'b0;
    end else begin
        s1_valid <= valid_in;
        
        if (valid_in) begin
            s1_mant <= bfp_mant;
            
            // 检测零值
            s1_is_zero <= (bfp_mant == 0);
            
            // [FIX 1] 修复：正确的符号扩展
            // bfp_exp 是 signed，Verilog 会自动进行符号扩展以匹配 9位的加法
            s1_shift_amount <= bfp_exp + 9'sd12;
            
            // 确定移位方向
            if (bfp_exp + 9'sd12 >= 9'sd0) begin
                s1_shift_left <= 1'b1;
            end else begin
                s1_shift_left <= 1'b0;
            end
        end
    end
end

//================================================================================
// Stage 2: 桶形移位器
//================================================================================

// Stage 2 寄存器声明
reg signed [47:0] s2_shifted;       // 扩展到48位防止溢出
reg               s2_is_zero;
reg               s2_valid;

// 移位逻辑（组合逻辑）
reg signed [47:0] shifted_result;
reg [5:0]         abs_shift_amount;

always @(*) begin
    // 提取移位量的绝对值（限制在0-31）
    if (s1_shift_left) begin
        if (s1_shift_amount > 9'sd31) begin
            // 移位量 >= 32，钳位到31
            abs_shift_amount = 6'd31;
        end else begin
            abs_shift_amount = s1_shift_amount[5:0];
        end
    end else begin
        // 处理负数移位量 (右移)
        if ((-s1_shift_amount) >= 32) begin
            abs_shift_amount = 6'd31;
        end else begin
            // 取反加一获得绝对值
            abs_shift_amount = (-s1_shift_amount[5:0]) & 6'h3F;
        end
    end
    
    // 执行移位
    if (s1_is_zero) begin
        shifted_result = 48'sd0;
    end else if (s1_shift_left) begin
        // 左移：扩展符号位到48位，然后左移
        shifted_result = $signed({{32{s1_mant[BFP_MANT_WIDTH-1]}}, s1_mant}) <<< abs_shift_amount;
    end else begin
        // 右移：扩展符号位，然后算术右移
        shifted_result = $signed({{32{s1_mant[BFP_MANT_WIDTH-1]}}, s1_mant}) >>> abs_shift_amount;
    end
end

// Stage 2 时序逻辑
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s2_shifted <= 48'sd0;
        s2_is_zero <= 1'b0;
        s2_valid   <= 1'b0;
    end else begin
        s2_shifted <= shifted_result;
        s2_is_zero <= s1_is_zero;
        s2_valid   <= s1_valid;
    end
end

//================================================================================
// Stage 3: 溢出检测、饱和处理、输出
//================================================================================

// 溢出检测逻辑（组合逻辑）
reg                     detect_overflow;
reg                     detect_underflow;
reg signed [Q412_WIDTH-1:0] saturated_result;

always @(*) begin
    detect_overflow  = 1'b0;
    detect_underflow = 1'b0;
    
    if (s2_is_zero) begin
        // 零值直接输出
        saturated_result = 16'sd0;
    end else begin
        // [FIX 2] 修复：正确的符号扩展比较
        // 必须将 Q412_MAX/MIN 的符号位扩展到 48 位，否则 Verilog 会将其视为无符号数比较
        
        // 检查正溢出 (Q412_MAX = 0x7FFF)
        if (s2_shifted > $signed({{32{Q412_MAX[15]}}, Q412_MAX})) begin
            saturated_result = Q412_MAX;
            detect_overflow = 1'b1;
        end 
        // 检查负溢出 (Q412_MIN = 0x8000)
        else if (s2_shifted < $signed({{32{Q412_MIN[15]}}, Q412_MIN})) begin
            saturated_result = Q412_MIN;
            detect_underflow = 1'b1;
        end 
        else begin
            // 正常范围，直接截取低16位
            saturated_result = s2_shifted[Q412_WIDTH-1:0];
        end
    end
end

// Stage 3 时序逻辑
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        q412_data  <= {Q412_WIDTH{1'b0}};
        overflow   <= 1'b0;
        underflow  <= 1'b0;
        valid_out  <= 1'b0;
    end else begin
        q412_data  <= saturated_result;
        overflow   <= detect_overflow;
        underflow  <= detect_underflow;
        valid_out  <= s2_valid;
    end
end

//================================================================================
// 调试接口
//================================================================================
reg [2:0] pipeline_stage_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pipeline_stage_reg <= 3'd0;
    end else begin
        // 跟踪流水线中的有效数据
        pipeline_stage_reg <= {s2_valid, s1_valid, valid_in};
    end
end

assign pipeline_stage = pipeline_stage_reg;

//================================================================================
// 仿真监控
//================================================================================
`ifdef SIMULATION

reg [31:0] conversion_count;
reg [31:0] overflow_count;
reg [31:0] underflow_count;

initial begin
    conversion_count = 0;
    overflow_count = 0;
    underflow_count = 0;
end

always @(posedge clk) begin
    if (valid_out) begin
        conversion_count = conversion_count + 1;
        
        if (overflow) begin
            overflow_count = overflow_count + 1;
            $display("[%0t] BFP->Q4.12 Overflow: raw_shifted=%h (dec %0d) -> saturated to %0d", 
                     $time, s2_shifted, s2_shifted, q412_data);
        end
        
        if (underflow) begin
            underflow_count = underflow_count + 1;
            $display("[%0t] BFP->Q4.12 Underflow: raw_shifted=%h (dec %0d) -> saturated to %0d", 
                     $time, s2_shifted, s2_shifted, q412_data);
        end
        
        // 每1000次转换显示统计
        if (conversion_count % 1000 == 0) begin
            $display("[%0t] Stats: Total=%0d, Ovf=%0d, Und=%0d",
                     $time, conversion_count, overflow_count, underflow_count);
        end
    end
end

`endif // SIMULATION

endmodule