`timescale 1ns / 1ps

module bfp_to_q412_vector_converter #(
    parameter DIM            = 32,      // 维度（8或32）
    parameter BFP_MANT_WIDTH = 16,      // BFP尾数位宽
    parameter BFP_EXP_WIDTH  = 8,       // BFP指数位宽
    parameter Q412_WIDTH     = 16,      // Q4.12位宽
    parameter Q412_FRAC_BITS = 12       // Q4.12小数位数
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 输入：BFP格式
    //==========================================================================
    input  wire                            valid_in,
    input  wire [BFP_EXP_WIDTH-1:0]        bfp_exp,     // 共享指数
    input  wire [DIM*BFP_MANT_WIDTH-1:0]   bfp_mant,    // DIM个尾数
    
    //==========================================================================
    // 输出：Q4.12格式
    //==========================================================================
    output reg  [DIM*Q412_WIDTH-1:0]       q412_data,   // DIM个Q4.12值
    output reg                             valid_out,
    
    //==========================================================================
    // 状态输出
    //==========================================================================
    output wire [DIM-1:0]                  overflow,    // 溢出标志（每维1bit）
    output wire [DIM-1:0]                  underflow    // 下溢标志（每维1bit）
);

//================================================================================
// 本地参数
//================================================================================
localparam BFP_BIAS = 127;  // BFP指数偏置（IEEE 754标准）
localparam SCALE_FACTOR = Q412_FRAC_BITS;  // Q4.12的缩放因子

//================================================================================
// 中间信号
//================================================================================
reg [DIM-1:0] overflow_reg;
reg [DIM-1:0] underflow_reg;

//================================================================================
// 逐维度转换逻辑（并行）
//================================================================================
genvar i;
generate
    for (i = 0; i < DIM; i = i + 1) begin : gen_converters
        
        // 提取当前维度的尾数
        wire signed [BFP_MANT_WIDTH-1:0] mant_i;
        assign mant_i = bfp_mant[i*BFP_MANT_WIDTH +: BFP_MANT_WIDTH];
        
        // 转换逻辑（组合逻辑）
        reg signed [31:0] converted_value;
        reg overflow_i;
        reg underflow_i;
        
        always @(*) begin:zhuan
            integer shift_amount;
            reg signed [31:0] temp;
            
            // 默认值
            converted_value = 32'sd0;
            overflow_i = 1'b0;
            underflow_i = 1'b0;
            
            // 计算移位量
            // shift = (bfp_exp - BFP_BIAS) + SCALE_FACTOR
            shift_amount = $signed({1'b0, bfp_exp}) - BFP_BIAS + SCALE_FACTOR;
            
            // 移位操作
            if (shift_amount >= 0) begin
                // 左移（放大）
                if (shift_amount < 16) begin
                    temp = $signed(mant_i) <<< shift_amount;
                end else begin
                    // 移位太大，可能溢出
                    temp = $signed(mant_i) <<< 15;
                    overflow_i = 1'b1;
                end
            end else begin
                // 右移（缩小）
                if (shift_amount > -16) begin
                    temp = $signed(mant_i) >>> (-shift_amount);
                end else begin
                    // 右移太多，接近0
                    temp = 32'sd0;
                    underflow_i = 1'b1;
                end
            end
            
            // 饱和处理
            if (temp > 32'sd32767) begin
                converted_value = 32'sd32767;
                overflow_i = 1'b1;
            end else if (temp < -32'sd32768) begin
                converted_value = -32'sd32768;
                overflow_i = 1'b1;
            end else begin
                converted_value = temp;
            end
        end
        
        // 时序逻辑：输出寄存
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                q412_data[i*Q412_WIDTH +: Q412_WIDTH] <= {Q412_WIDTH{1'b0}};
                overflow_reg[i] <= 1'b0;
                underflow_reg[i] <= 1'b0;
            end else if (valid_in) begin
                q412_data[i*Q412_WIDTH +: Q412_WIDTH] <= converted_value[Q412_WIDTH-1:0];
                overflow_reg[i] <= overflow_i;
                underflow_reg[i] <= underflow_i;
            end
        end
    end
endgenerate

//================================================================================
// Valid信号
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out <= 1'b0;
    end else begin
        valid_out <= valid_in;
    end
end

//================================================================================
// 状态输出
//================================================================================
assign overflow = overflow_reg;
assign underflow = underflow_reg;

endmodule