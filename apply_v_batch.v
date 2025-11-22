`timescale 1ns / 1ps

//===================================================================================
// 批量Apply V模块
//
// 功能：
//   对32个query批量执行Apply V操作
//   Output[query] = Σ(Weights[query][k] × V[k]) for k=0..31
//
// 架构：串行循环32次
//   - 每次处理1个query
//   - 计算该query与32个V向量的加权和
//   - 输出累加到BFP累加器
//
// 数据格式：
//   - 输入Weights：32×32矩阵（定点数Q14格式）
//   - 输入V：32个向量，每个8维，每个向量有独立BFP指数
//   - 输出：32个向量，每个8维，BFP格式
//
// 时序：
//   32次迭代 × 单次计算 ≈ 32 × 50 cycles = 1600 cycles
//
// 注意：
//   - 每个Head Engine实例化1个本模块
//   - chunk_first时清零累加器，否则累加
//===================================================================================

module apply_v_batch #(
   parameter NUM_QUERIES    = 32,
    parameter CHUNK_SIZE     = 32,
    parameter HEAD_DIM       = 8,
    parameter DATA_WIDTH     = 8,
    parameter EXP_WIDTH      = 8,
    parameter SCORE_WIDTH    = 16,      // Weight位宽（Q14定点）
    parameter ACCUM_WIDTH    = 32      // ✅ 增加到32位（原24位）

)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 控制接口
    //===========================================================================
    input  wire start,
    input  wire chunk_first,            // 第一个chunk：清零累加器
    output reg  done,
    output reg  busy,
    
    //===========================================================================
    // Weights输入：32×32矩阵（定点数Q14格式）
    //===========================================================================
    input  wire [(NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH)-1:0] weights_batch,
    
    //===========================================================================
    // V输入：32个向量，每个有独立BFP指数
    //===========================================================================
    input  wire [(CHUNK_SIZE*EXP_WIDTH)-1:0] v_chunk_exp,
    input  wire [(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)-1:0] v_chunk_mant,
    
    //===========================================================================
    // 累加器接口（每个query输出8个维度）
    //===========================================================================
    output reg  output_valid,
    output reg  [4:0] output_query,      // 0-31
    output reg  [2:0] output_dim,        // 0-7
    output reg  signed [ACCUM_WIDTH-1:0] output_mant,
    output reg  [EXP_WIDTH-1:0] output_exp,
    output reg  output_first_chunk,
    
    //===========================================================================
    // 调试和错误报告
    //===========================================================================
    output reg  [31:0] dbg_precision_warnings,  // 精度警告计数
    output reg  [31:0] dbg_overflow_warnings     // 溢出警告计数
);

//===================================================================================
// 本地参数
//===================================================================================
localparam PRECISION_THRESHOLD = 16; 
// 定点数Q14的小数位数
localparam FRAC_BITS = 14;

// 中间计算位宽（weight×mant = 16×8 = 24，加上额外位用于累加）
localparam MULT_WIDTH = SCORE_WIDTH + DATA_WIDTH;  // 24-bit

// 状态机
localparam IDLE           = 4'd0;
localparam FIND_MAX_EXP   = 4'd1;  // ✅ 新增：找到32个V的最大指数
localparam INIT_ACCUM     = 4'd2;  // ✅ 新增：初始化累加器
localparam ACCUMULATE     = 4'd3;  // ✅ 修改：流水线累加
localparam NORMALIZE      = 4'd4;  // ✅ 新增：归一化和溢出检测
localparam OUTPUT_RESULT  = 4'd5;
localparam NEXT_DIM       = 4'd6;
localparam NEXT_QUERY     = 4'd7;
localparam DONE_ST        = 4'd8;

reg [3:0] state;
reg [5:0] query_idx;   // 0-31
reg [2:0] dim_idx;     // 0-7
reg [5:0] k_idx;       // 0-31 (V向量索引)

//===================================================================================
// 内部信号
//===================================================================================

// 当前query的weights（1行32列）
reg signed [SCORE_WIDTH-1:0] current_weights [0:CHUNK_SIZE-1];

// ✅ 最大指数和对齐后的V尾数
reg [EXP_WIDTH-1:0] max_exp;
reg signed [DATA_WIDTH-1:0] v_mant_current;
reg [EXP_WIDTH-1:0] v_exp_current;
reg [5:0] exp_diff;
reg signed [DATA_WIDTH-1:0] v_mant_aligned;

// ✅ 累加器（32-bit，有符号）
reg signed [ACCUM_WIDTH-1:0] dim_accum;

// ✅ 乘法结果（扩展精度）
reg signed [MULT_WIDTH-1:0] mult_result;

// ✅ 归一化后的结果
reg signed [ACCUM_WIDTH-1:0] normalized_mant;
reg [EXP_WIDTH-1:0] final_exp;

// 循环变量
integer i;

//===================================================================================
// ✅ 提取当前query的weights
//===================================================================================

always @(*) begin
    for (i = 0; i < CHUNK_SIZE; i = i + 1) begin
        current_weights[i] = weights_batch[
            (query_idx*CHUNK_SIZE + i)*SCORE_WIDTH +: SCORE_WIDTH
        ];
    end
end

//===================================================================================
// ✅ 主状态机 - 流水线式累加
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        query_idx <= 6'd0;
        dim_idx <= 3'd0;
        k_idx <= 6'd0;
        done <= 1'b0;
        busy <= 1'b0;
        output_valid <= 1'b0;
        
        dim_accum <= {ACCUM_WIDTH{1'b0}};
        max_exp <= {EXP_WIDTH{1'b0}};
        
        dbg_precision_warnings <= 32'd0;
        dbg_overflow_warnings <= 32'd0;
        
    end else begin
        // 默认：清除单周期脉冲
        output_valid <= 1'b0;
        
        case (state)
            //===================================================================
            // IDLE: 等待启动
            //===================================================================
            IDLE: begin
                done <= 1'b0;
                
                if (start) begin
                    busy <= 1'b1;
                    query_idx <= 6'd0;
                    dim_idx <= 3'd0;
                    output_first_chunk <= chunk_first;
                    state <= FIND_MAX_EXP;
                    
                    $display("[%0t] ApplyV_Opt: Started, chunk_first=%0b",
                             $time, chunk_first);
                end else begin
                    busy <= 1'b0;
                end
            end
            
            //===================================================================
            // ✅ FIND_MAX_EXP: 找到32个V向量的最大指数
            //===================================================================
            FIND_MAX_EXP: begin
                // 组合逻辑：找最大指数
                max_exp = v_chunk_exp[0*EXP_WIDTH +: EXP_WIDTH];
                
                for (i = 1; i < CHUNK_SIZE; i = i + 1) begin
                    if (v_chunk_exp[i*EXP_WIDTH +: EXP_WIDTH] > max_exp) begin
                        max_exp = v_chunk_exp[i*EXP_WIDTH +: EXP_WIDTH];
                    end
                end
                
                state <= INIT_ACCUM;
                
                $display("[%0t] ApplyV_Opt: Query %0d, Dim %0d - Max exp = %0d",
                         $time, query_idx, dim_idx, max_exp);
            end
            
            //===================================================================
            // ✅ INIT_ACCUM: 初始化累加器
            //===================================================================
            INIT_ACCUM: begin
                dim_accum <= {ACCUM_WIDTH{1'b0}};
                k_idx <= 6'd0;
                state <= ACCUMULATE;
            end
            
            //===================================================================
            // ✅ ACCUMULATE: 流水线累加（每周期累加1个V向量）
            //===================================================================
            ACCUMULATE: begin
                if (k_idx < CHUNK_SIZE) begin
                    //------------------------------------------------------------
                    // 1. 提取V[k][dim]的指数和尾数
                    //------------------------------------------------------------
                    v_exp_current = v_chunk_exp[k_idx*EXP_WIDTH +: EXP_WIDTH];
                    v_mant_current = $signed(
                        v_chunk_mant[(k_idx*HEAD_DIM + dim_idx)*DATA_WIDTH +: DATA_WIDTH]
                    );
                    
                    //------------------------------------------------------------
                    // 2. 计算指数差并对齐尾数
                    //------------------------------------------------------------
                    exp_diff = max_exp - v_exp_current;
                    
                    // 精度保护：检测大幅度右移
                    if (exp_diff > PRECISION_THRESHOLD) begin
                        dbg_precision_warnings <= dbg_precision_warnings + 1'b1;
                        $display("[%0t] WARNING: Large exp_diff=%0d for V[%0d], precision loss!",
                                 $time, exp_diff, k_idx);
                    end
                    
                    // 右移对齐（限制最大右移量）
                    if (exp_diff >= DATA_WIDTH) begin
                        v_mant_aligned = {DATA_WIDTH{1'b0}};  // 完全舍弃
                    end else begin
                        v_mant_aligned = v_mant_current >>> exp_diff;  // 算术右移
                    end
                    
                    //------------------------------------------------------------
                    // 3. 乘以权重: weight[k] × v_aligned
                    //------------------------------------------------------------
                    mult_result = current_weights[k_idx] * v_mant_aligned;
                    
                    //------------------------------------------------------------
                    // 4. 累加到dim_accum
                    //------------------------------------------------------------
                    dim_accum <= dim_accum + mult_result;
                    
                    //------------------------------------------------------------
                    // 5. 移到下一个V向量
                    //------------------------------------------------------------
                    k_idx <= k_idx + 6'd1;
                    
                    $display("[%0t] ApplyV_Opt: Q%0d D%0d K%0d: w=%0d v_exp=%0d v_mant=%0d aligned=%0d prod=%0d accum=%0d",
                             $time, query_idx, dim_idx, k_idx,
                             $signed(current_weights[k_idx]), v_exp_current, 
                             $signed(v_mant_current), $signed(v_mant_aligned),
                             $signed(mult_result), $signed(dim_accum) + $signed(mult_result));
                    
                end else begin
                    // 累加完成，进入归一化
                    state <= NORMALIZE;
                end
            end
            
            //===================================================================
            // ✅ NORMALIZE: 归一化和溢出检测
            //===================================================================
            NORMALIZE: begin
                //----------------------------------------------------------------
                // 1. 除以2^14进行Q14定点归一化
                //----------------------------------------------------------------
                normalized_mant = dim_accum >>> FRAC_BITS;
                
                //----------------------------------------------------------------
                // 2. 溢出检测（检查高位是否全是符号扩展）
                //----------------------------------------------------------------
                // 对于32-bit有符号数，归一化后如果超过24-bit范围，说明溢出
                if (normalized_mant > 32'sd8388607 || normalized_mant < -32'sd8388608) begin
                    // 超过24-bit有符号范围：[-2^23, 2^23-1]
                    dbg_overflow_warnings <= dbg_overflow_warnings + 1'b1;
                    
                    $display("[%0t] WARNING: Overflow detected! mant=%0d (before norm: %0d)",
                             $time, $signed(normalized_mant), $signed(dim_accum));
                    
                    // 饱和处理
                    if (normalized_mant > 0) begin
                        normalized_mant = 32'sd8388607;   // 正饱和
                    end else begin
                        normalized_mant = -32'sd8388608;  // 负饱和
                    end
                end
                
                //----------------------------------------------------------------
                // 3. 调整指数（✅ 修复版本）
                //----------------------------------------------------------------
                // 数学推导：
                // - V的值 = v_mant × 2^(max_exp)
                // - 权重(Q14) = weight_int × 2^(-14)
                // - 累加结果 = Σ(weight_int × v_aligned)，其中v_aligned已对齐到max_exp
                // - 因此dim_accum表示: dim_accum × 2^(max_exp-14)
                // - 归一化后: normalized_mant = dim_accum >>> 14 = (真实值 / 2^(max_exp))
                // - 所以输出应该是: normalized_mant × 2^(max_exp)
                // ✅ 结论: final_exp = max_exp (不需要加减FRAC_BITS)
                final_exp = max_exp;
                
                state <= OUTPUT_RESULT;
                
                $display("[%0t] ApplyV_Opt: Q%0d D%0d normalized: exp=%0d mant=%0d",
                         $time, query_idx, dim_idx, final_exp, $signed(normalized_mant));
            end
            
            //===================================================================
            // ✅ OUTPUT_RESULT: 输出结果
            //===================================================================
            OUTPUT_RESULT: begin
                output_valid <= 1'b1;
                output_query <= query_idx[4:0];
                output_dim <= dim_idx;
                output_mant <= normalized_mant;
                output_exp <= final_exp;
                
                state <= NEXT_DIM;
            end
            
            //===================================================================
            // NEXT_DIM: 移到下一个维度
            //===================================================================
            NEXT_DIM: begin
                if (dim_idx < HEAD_DIM - 1) begin
                    dim_idx <= dim_idx + 3'd1;
                    state <= FIND_MAX_EXP;
                end else begin
                    state <= NEXT_QUERY;
                end
            end
            
            //===================================================================
            // NEXT_QUERY: 移到下一个query
            //===================================================================
            NEXT_QUERY: begin
                if (query_idx < NUM_QUERIES - 1) begin
                    query_idx <= query_idx + 6'd1;
                    dim_idx <= 3'd0;
                    state <= FIND_MAX_EXP;
                end else begin
                    state <= DONE_ST;
                end
            end
            
            //===================================================================
            // DONE: 全部完成
            //===================================================================
            DONE_ST: begin
                done <= 1'b1;
                busy <= 1'b0;
                state <= IDLE;
                
                $display("[%0t] ApplyV_Opt: All %0d queries completed", $time, NUM_QUERIES);
                $display("           Precision warnings: %0d", dbg_precision_warnings);
                $display("           Overflow warnings: %0d", dbg_overflow_warnings);
            end
            
            default: state <= IDLE;
        endcase
    end
end



endmodule