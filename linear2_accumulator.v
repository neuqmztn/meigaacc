`timescale 1ns / 1ps

//================================================================================
// Linear2 Accumulator - 完整版本（带指数对齐）
//
// 功能：累加 Linear2 的多次部分矩阵乘法结果
//
// 数学原理：
// Y[i,j] = Σ(k=0 to 127) H[i,k] × W2[k,j]
//
// 分块计算：
// Y = H[:,0:31]   × W2[0:31,:]   +  (Part 0)
//     H[:,32:63]  × W2[32:63,:]  +  (Part 1)
//     H[:,64:95]  × W2[64:95,:]  +  (Part 2)
//     H[:,96:127] × W2[96:127,:]    (Part 3)
//
// 关键：完整的 BFP 指数对齐
// - 选择最大指数作为共享指数
// - 对齐所有尾数到该指数
// - 累加对齐后的尾数
// - 处理溢出和精度损失
//
//================================================================================

module linear2_accumulator #(
    parameter TOKEN_CHUNK    = 32,
    parameter OUTPUT_DIM     = 32,
    parameter BFP_EXP_W      = 8,
    parameter INPUT_MANT_W   = 15,     // 输入尾数位宽
    parameter OUTPUT_MANT_W  = 15,     // 输出尾数位宽
    parameter NUM_CHUNKS     = 4       // 累加次数
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire clear,                 // 清空累加器（新token块）
    input  wire enable,                // 累加使能（每次Linear2完成）
    
    //================================================================================
    // 输入：部分结果
    //================================================================================
    input  wire [BFP_EXP_W-1:0] partial_exp,
    input  wire [TOKEN_CHUNK*OUTPUT_DIM*INPUT_MANT_W-1:0] partial_mant,
    
    //================================================================================
    // 输出：累加结果
    //================================================================================
    output reg  [BFP_EXP_W-1:0] result_exp,
    output reg  [TOKEN_CHUNK*OUTPUT_DIM*OUTPUT_MANT_W-1:0] result_mant,
    output reg  result_valid,           // 累加完成标志
    
    //================================================================================
    // 状态输出
    //================================================================================
    output reg  [2:0] accum_count       // 当前累加次数（0-3）
);

//================================================================================
// 内部信号
//================================================================================

localparam TOTAL_ELEMENTS = TOKEN_CHUNK * OUTPUT_DIM;

// 累加缓存（使用更大位宽防止溢出）
reg signed [OUTPUT_MANT_W-1:0] accum_buffer [0:TOTAL_ELEMENTS-1];
reg [BFP_EXP_W-1:0] accum_exp;

// 输入解包
wire signed [INPUT_MANT_W-1:0] partial_mant_unpacked [0:TOTAL_ELEMENTS-1];

// 对齐后的数据
reg signed [OUTPUT_MANT_W-1:0] partial_aligned [0:TOTAL_ELEMENTS-1];
reg signed [OUTPUT_MANT_W-1:0] accum_aligned [0:TOTAL_ELEMENTS-1];

// 指数对齐信号
reg signed [BFP_EXP_W:0] exp_diff;      // 9-bit signed
reg [BFP_EXP_W-1:0] new_shared_exp;
reg [5:0] shift_amount;                 // 最多右移 63 位

integer i;

//================================================================================
// 输入解包
//================================================================================

genvar g;
generate
    for (g = 0; g < TOTAL_ELEMENTS; g = g + 1) begin : gen_unpack
        assign partial_mant_unpacked[g] = 
            partial_mant[g*INPUT_MANT_W +: INPUT_MANT_W];
    end
endgenerate

//================================================================================
// 指数对齐逻辑（组合逻辑）
//================================================================================

always @(*) begin
    // 计算指数差（带符号）
    exp_diff = $signed({1'b0, partial_exp}) - $signed({1'b0, accum_exp});
    
    if (exp_diff >= 0) begin
        //====================================================================
        // partial 的指数更大或相等
        //====================================================================
        new_shared_exp = partial_exp;
        
        // 限制右移量
        if (exp_diff >= OUTPUT_MANT_W) begin
            shift_amount = OUTPUT_MANT_W;  // 完全右移（变为0）
        end else begin
            shift_amount = exp_diff[5:0];
        end
        
        // 对齐 accum_buffer（右移）
        for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
            if (shift_amount >= OUTPUT_MANT_W) begin
                // 右移太多，直接置0
                accum_aligned[i] = {OUTPUT_MANT_W{1'b0}};
            end else begin
                // 算术右移（保持符号位）
                accum_aligned[i] = accum_buffer[i] >>> shift_amount;
            end
            
            // partial 不需要对齐，但需要扩展位宽
            if (OUTPUT_MANT_W > INPUT_MANT_W) begin
                // 符号扩展
                partial_aligned[i] = {{(OUTPUT_MANT_W-INPUT_MANT_W){partial_mant_unpacked[i][INPUT_MANT_W-1]}},
                                     partial_mant_unpacked[i]};
            end else begin
                // 截断（理论上不应该发生）
                partial_aligned[i] = partial_mant_unpacked[i][INPUT_MANT_W-1 -: OUTPUT_MANT_W];
            end
        end
        
    end else begin
        //====================================================================
        // accum 的指数更大
        //====================================================================
        new_shared_exp = accum_exp;
        
        // 限制右移量（exp_diff 是负数）
        if (-exp_diff >= OUTPUT_MANT_W) begin
            shift_amount = OUTPUT_MANT_W;
        end else begin
            shift_amount = (-exp_diff) & 6'h3F;
        end
        
        // accum_buffer 不需要对齐
        for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
            accum_aligned[i] = accum_buffer[i];
            
            // 对齐 partial（右移）
            if (shift_amount >= INPUT_MANT_W) begin
                // 右移太多，直接置0
                partial_aligned[i] = {OUTPUT_MANT_W{1'b0}};
            end else begin:test
                // 先扩展，再右移
                reg signed [OUTPUT_MANT_W-1:0] partial_extended;
                partial_extended = {{(OUTPUT_MANT_W-INPUT_MANT_W){partial_mant_unpacked[i][INPUT_MANT_W-1]}},
                                   partial_mant_unpacked[i]};
                partial_aligned[i] = partial_extended >>> shift_amount;
            end
        end
    end
end

//================================================================================
// 累加逻辑（时序逻辑）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
            accum_buffer[i] <= {OUTPUT_MANT_W{1'b0}};
        end
        accum_exp     <= {BFP_EXP_W{1'b0}};
        accum_count   <= 3'd0;
        result_exp    <= {BFP_EXP_W{1'b0}};
        result_mant   <= {TOKEN_CHUNK*OUTPUT_DIM*OUTPUT_MANT_W{1'b0}};
        result_valid  <= 1'b0;
        
    end else begin
        
        //====================================================================
        // 清空累加器
        //====================================================================
        if (clear) begin
            for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
                accum_buffer[i] <= {OUTPUT_MANT_W{1'b0}};
            end
            accum_exp     <= {BFP_EXP_W{1'b0}};
            accum_count   <= 3'd0;
            result_valid  <= 1'b0;
            
        //====================================================================
        // 累加
        //====================================================================
        end else if (enable) begin
            
            if (accum_count == 3'd0) begin
                //------------------------------------------------------------
                // 第一次：直接赋值
                //------------------------------------------------------------
                for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
                    if (OUTPUT_MANT_W > INPUT_MANT_W) begin
                        // 符号扩展
                        accum_buffer[i] <= {{(OUTPUT_MANT_W-INPUT_MANT_W){partial_mant_unpacked[i][INPUT_MANT_W-1]}},
                                           partial_mant_unpacked[i]};
                    end else begin
                        accum_buffer[i] <= partial_mant_unpacked[i][INPUT_MANT_W-1 -: OUTPUT_MANT_W];
                    end
                end
                accum_exp <= partial_exp;
                accum_count <= 3'd1;
                result_valid <= 1'b0;
                
            end else if (accum_count < NUM_CHUNKS) begin
                //------------------------------------------------------------
                // 第2-4次：对齐后累加
                //------------------------------------------------------------
                
                // 更新共享指数
                accum_exp <= new_shared_exp;
                
                // 累加对齐后的尾数
                for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
                    accum_buffer[i] <= accum_aligned[i] + partial_aligned[i];
                end
                
                accum_count <= accum_count + 1'b1;
                
                //------------------------------------------------------------
                // 如果是最后一次累加，输出结果
                //------------------------------------------------------------
                if (accum_count == NUM_CHUNKS - 1) begin
                    result_valid <= 1'b1;
                end else begin
                    result_valid <= 1'b0;
                end
            end
        end else begin
            result_valid <= 1'b0;
        end
    end
end

//================================================================================
// 输出打包
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        result_exp <= {BFP_EXP_W{1'b0}};
        result_mant <= {TOKEN_CHUNK*OUTPUT_DIM*OUTPUT_MANT_W{1'b0}};
    end else if (result_valid) begin
        result_exp <= accum_exp;
        for (i = 0; i < TOTAL_ELEMENTS; i = i + 1) begin
            result_mant[i*OUTPUT_MANT_W +: OUTPUT_MANT_W] <= accum_buffer[i];
        end
    end
end

//================================================================================
// 调试信号
//================================================================================

`ifdef SIMULATION
always @(posedge clk) begin
    if (clear) begin
        $display("[%0t] Accum: Cleared for new token batch", $time);
    end
    
    if (enable) begin
        $display("[%0t] Accum: Chunk %0d/%0d - exp_partial=%0d, exp_accum=%0d, exp_diff=%0d, new_exp=%0d", 
                 $time, accum_count, NUM_CHUNKS, partial_exp, accum_exp, exp_diff, new_shared_exp);
        
        if (accum_count == 3'd0) begin
            $display("[%0t] Accum: First chunk - direct assignment", $time);
        end else begin
            $display("[%0t] Accum: Aligning and accumulating (shift_amount=%0d)", $time, shift_amount);
        end
    end
    
    if (result_valid) begin
        $display("[%0t] Accum: Result ready - exp=%0d, accumulated %0d chunks", 
                 $time, result_exp, NUM_CHUNKS);
    end
end
`endif

endmodule