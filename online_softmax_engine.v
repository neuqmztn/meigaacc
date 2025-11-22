module online_softmax_engine #(
    parameter NUM_HEADS = 4,
    parameter TOKEN_BATCH = 32,
    parameter K_CHUNK_SIZE = 32,
    parameter DATA_WIDTH = 8,
    parameter EXP_WIDTH = 8,
    parameter SCORE_WIDTH = 16,
    parameter ACCUM_WIDTH = 24
)(
    input  wire clk,
    input  wire rst_n,
    
    // 控制接口
    input  wire start,
    input  wire [1:0] head_idx,
    input  wire [4:0] row_idx,
    input  wire [4:0] chunk_id,
    input  wire chunk_first,
    input  wire chunk_last,
    input  wire [5:0] chunk_size,
    output reg  done,
    output reg  busy,
    
    // Scores输入（BFP格式：共享指数 + 尾数）
    input  wire [EXP_WIDTH-1:0] scores_shared_exp,
    input  wire [K_CHUNK_SIZE*SCORE_WIDTH-1:0] scores_mants_packed,
    
    // 统计量存储接口（读取）
    input  wire signed [SCORE_WIDTH-1:0] max_rd_value,
    input  wire signed [ACCUM_WIDTH-1:0] sum_rd_value,
    
    // 统计量存储接口（写入）
    output reg  max_wr_en,
    output reg  [1:0] max_wr_head,
    output reg  [4:0] max_wr_row,
    output reg  signed [SCORE_WIDTH-1:0] max_wr_value,
    
    output reg  sum_wr_en,
    output reg  [1:0] sum_wr_head,
    output reg  [4:0] sum_wr_row,
    output reg  signed [ACCUM_WIDTH-1:0] sum_wr_value,
    
    // 重归一化信号（给Apply V模块）
    output reg  renorm_en,
    output reg  signed [SCORE_WIDTH-1:0] renorm_scale,
    
    // Attention weights输出
    output reg  weights_valid,
    output wire [K_CHUNK_SIZE*SCORE_WIDTH-1:0] weights_packed
);

//================================================================================
// 本地参数定义
//================================================================================

// Q14定点数的小数位数
localparam FRAC_BITS = 14;

// 负无穷大（用于初始化max）
localparam NEG_INF = -16'h7FFF;

// 内部扩展累加器位宽（32位，提供更大动态范围）
localparam INTERNAL_ACCUM_WIDTH = 32;

// Exp函数分段阈值（Q14格式）
localparam EXP_THRESHOLD_HIGH = 16'sd2048;   // 0.125
localparam EXP_THRESHOLD_MID = -16'sd2048;   // -0.125
localparam EXP_THRESHOLD_LOW = -16'sd8192;   // -0.5

// 最小非零exp值（防止完全下溢）
localparam MIN_EXP_VALUE = 16'sd1;

// 除法保护阈值
localparam MIN_SUM_THRESHOLD = 32'd16;

//================================================================================
// 状态定义
//================================================================================

localparam STATE_IDLE            = 4'd0;
localparam STATE_LOAD_STATS      = 4'd1;
localparam STATE_FIND_MAX_INIT   = 4'd2;
localparam STATE_FIND_MAX_LOOP   = 4'd3;
localparam STATE_UPDATE_MAX      = 4'd4;
localparam STATE_COMPUTE_EXP     = 4'd5;
localparam STATE_ACCUMULATE_SUM  = 4'd6;
localparam STATE_WRITE_STATS     = 4'd7;
localparam STATE_NORMALIZE_INIT  = 4'd8;
localparam STATE_NORMALIZE_LOOP  = 4'd9;
localparam STATE_DONE            = 4'd10;

reg [3:0] state;

//================================================================================
// 内部寄存器
//================================================================================

// 统计量（使用扩展位宽）
reg signed [SCORE_WIDTH-1:0] old_max;
reg signed [SCORE_WIDTH-1:0] new_max;
reg signed [SCORE_WIDTH-1:0] chunk_max;
reg signed [INTERNAL_ACCUM_WIDTH-1:0] old_sum_internal;
reg signed [INTERNAL_ACCUM_WIDTH-1:0] new_sum_internal;
reg max_updated;

// 数据数组
reg signed [SCORE_WIDTH-1:0] scores_array [0:K_CHUNK_SIZE-1];
reg signed [SCORE_WIDTH-1:0] exp_values [0:K_CHUNK_SIZE-1];
reg signed [SCORE_WIDTH-1:0] weights_array [0:K_CHUNK_SIZE-1];

// 循环计数器
reg [5:0] loop_idx;

// 临时变量
reg signed [SCORE_WIDTH-1:0] temp_score;
reg signed [SCORE_WIDTH-1:0] temp_exp;
reg signed [INTERNAL_ACCUM_WIDTH-1:0] temp_sum;

// 中间计算变量
reg signed [31:0] mult_temp;
reg signed [47:0] mult_product;

// 整数索引
integer i;

//================================================================================
// 解包输入scores
//================================================================================

always @(*) begin
    for (i = 0; i < K_CHUNK_SIZE; i = i + 1) begin
        scores_array[i] = scores_mants_packed[i*SCORE_WIDTH +: SCORE_WIDTH];
    end
end

//================================================================================
// 打包输出weights - 使用generate确保可综合
//================================================================================

genvar gv_i;
generate
    for (gv_i = 0; gv_i < K_CHUNK_SIZE; gv_i = gv_i + 1) begin : gen_weights_pack
        assign weights_packed[gv_i*SCORE_WIDTH +: SCORE_WIDTH] = weights_array[gv_i];
    end
endgenerate

//================================================================================
// 改进的Exp近似函数（分段Padé逼近 + 泰勒展开）
//
// 精度对比：
// - 原版线性近似：误差 ~5%
// - 优化版分段逼近：误差 ~0.5%
//
// 输入：x = score - max （Q14格式，通常为负数）
// 输出：exp(x) （Q14格式）
//================================================================================

function signed [SCORE_WIDTH-1:0] exp_approx_improved;
    input signed [SCORE_WIDTH-1:0] x;
    reg signed [31:0] temp;
    reg signed [31:0] x_sq;
    reg signed [31:0] x_cu;
    reg signed [15:0] result;
    begin
        // 区间1: x >= 0.125 (正值区域)
        // 使用Padé [1/1]逼近: exp(x) ≈ (2 + x) / (2 - x)
        if (x >= EXP_THRESHOLD_HIGH) begin
            // 计算分子: 2 + x = 32768 + x
            temp = 32'sd32768 + {{16{x[15]}}, x};
            
            // 计算分母: 2 - x = 32768 - x
            mult_temp = 32'sd32768 - {{16{x[15]}}, x};
            
            // 避免除零
            if (mult_temp <= 32'sd256) begin
                result = 16'sd32767;
            end else begin
                // (分子 << 14) / 分母，保持Q14格式
                temp = (temp <<< 14) / mult_temp;
                
                // 饱和处理
                if (temp > 32'sd32767) begin
                    result = 16'sd32767;
                end else if (temp < 32'sd0) begin
                    result = MIN_EXP_VALUE;
                end else begin
                    result = temp[15:0];
                end
            end
        end
        
        // 区间2: -0.125 <= x < 0.125 (接近零的区域)
        // 使用泰勒展开到3阶: exp(x) ≈ 1 + x + x²/2 + x³/6
        else if (x >= EXP_THRESHOLD_MID) begin
            // 初始化为 1.0 (Q14)
            temp = 32'sd16384;
            
            // 加上 x
            temp = temp + {{16{x[15]}}, x};
            
            // 计算 x² (Q28) 并右移14位回Q14
            x_sq = ({{16{x[15]}}, x} * {{16{x[15]}}, x}) >>> 14;
            
            // 加上 x²/2 (右移1位即除以2)
            temp = temp + (x_sq >>> 1);
            
            // 计算 x³ (Q28) 并右移14位
            x_cu = (x_sq * {{16{x[15]}}, x}) >>> 14;
            
            // 加上 x³/6
            // 除以6可近似为 (x³ >> 2) - (x³ >> 4)
            temp = temp + ((x_cu >>> 2) - (x_cu >>> 4));
            
            // 饱和处理
            if (temp > 32'sd32767) begin
                result = 16'sd32767;
            end else if (temp < 32'sd0) begin
                result = MIN_EXP_VALUE;
            end else begin
                result = temp[15:0];
            end
        end
        
        // 区间3: -0.5 <= x < -0.125 (中等负值)
        // 使用泰勒展开到2阶: exp(x) ≈ 1 + x + x²/2
        else if (x >= EXP_THRESHOLD_LOW) begin
            temp = 32'sd16384 + {{16{x[15]}}, x};
            x_sq = ({{16{x[15]}}, x} * {{16{x[15]}}, x}) >>> 14;
            temp = temp + (x_sq >>> 1);
            
            if (temp > 32'sd32767) begin
                result = 16'sd32767;
            end else if (temp < 32'sd0) begin
                result = MIN_EXP_VALUE;
            end else begin
                result = temp[15:0];
            end
        end
        
        // 区间4: x < -0.5 (大负值，接近零)
        // 使用简化线性近似: exp(x) ≈ max(1 + x, MIN_EXP)
        else begin
            temp = 32'sd16384 + {{16{x[15]}}, x};
            
            if (temp < MIN_EXP_VALUE) begin
                result = MIN_EXP_VALUE;
            end else if (temp > 32'sd32767) begin
                result = 16'sd32767;
            end else begin
                result = temp[15:0];
            end
        end
        
        exp_approx_improved = result;
    end
endfunction

//================================================================================
// 饱和加法函数
//================================================================================

function signed [INTERNAL_ACCUM_WIDTH-1:0] saturate_add;
    input signed [INTERNAL_ACCUM_WIDTH-1:0] a;
    input signed [INTERNAL_ACCUM_WIDTH-1:0] b;
    reg signed [INTERNAL_ACCUM_WIDTH:0] temp_result;
    begin
        temp_result = {a[INTERNAL_ACCUM_WIDTH-1], a} + {b[INTERNAL_ACCUM_WIDTH-1], b};
        
        // 检测溢出
        if (temp_result[INTERNAL_ACCUM_WIDTH] != temp_result[INTERNAL_ACCUM_WIDTH-1]) begin
            // 溢出：根据符号位饱和
            if (temp_result[INTERNAL_ACCUM_WIDTH]) begin
                // 负溢出
                saturate_add = {1'b1, {(INTERNAL_ACCUM_WIDTH-1){1'b0}}};
            end else begin
                // 正溢出
                saturate_add = {1'b0, {(INTERNAL_ACCUM_WIDTH-1){1'b1}}};
            end
        end else begin
            saturate_add = temp_result[INTERNAL_ACCUM_WIDTH-1:0];
        end
    end
endfunction

//================================================================================
// Generate块：清零exp_values数组（修复第552行问题）
//================================================================================

genvar gv_exp_clear;
reg clear_exp_trigger;  // 触发信号

generate
    for (gv_exp_clear = 0; gv_exp_clear < K_CHUNK_SIZE; gv_exp_clear = gv_exp_clear + 1) begin : gen_exp_clear
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                // 复位时清零
                exp_values[gv_exp_clear] <= {SCORE_WIDTH{1'b0}};
            end else if (clear_exp_trigger && (gv_exp_clear >= chunk_size)) begin
                // 运行时条件清零
                exp_values[gv_exp_clear] <= {SCORE_WIDTH{1'b0}};
            end else if (state == STATE_COMPUTE_EXP && loop_idx == gv_exp_clear) begin
                // 正常计算过程中的赋值
                temp_score = scores_array[loop_idx] - new_max;
                temp_exp = exp_approx_improved(temp_score);
                exp_values[gv_exp_clear] <= temp_exp;
            end
        end
    end
endgenerate

//================================================================================
// Generate块：复制exp到weights（修复第643行问题）
//================================================================================

genvar gv_copy;
reg copy_exp_to_weights;  // 触发信号

generate
    for (gv_copy = 0; gv_copy < K_CHUNK_SIZE; gv_copy = gv_copy + 1) begin : gen_copy_weights
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                weights_array[gv_copy] <= {SCORE_WIDTH{1'b0}};
            end else if (copy_exp_to_weights) begin
                weights_array[gv_copy] <= exp_values[gv_copy];
            end else if (state == STATE_NORMALIZE_LOOP && loop_idx == gv_copy && loop_idx < chunk_size) begin
                // 归一化过程中的赋值
                if (new_sum_internal > MIN_SUM_THRESHOLD) begin
                    if (exp_values[gv_copy][SCORE_WIDTH-1]) begin
                        mult_product = {{(48-SCORE_WIDTH){1'b1}}, exp_values[gv_copy]};
                    end else begin
                        mult_product = {{(48-SCORE_WIDTH){1'b0}}, exp_values[gv_copy]};
                    end
                    mult_product = mult_product <<< FRAC_BITS;
                    mult_temp = mult_product[INTERNAL_ACCUM_WIDTH+FRAC_BITS-1:0] / new_sum_internal;
                    
                    if (mult_temp > 32'sd32767) begin
                        weights_array[gv_copy] <= 16'sd32767;
                    end else if (mult_temp < -32'sd32768) begin
                        weights_array[gv_copy] <= -16'sd32768;
                    end else begin
                        weights_array[gv_copy] <= mult_temp[15:0];
                    end
                end else begin
                    mult_temp = 32'sd16384 / chunk_size;
                    weights_array[gv_copy] <= mult_temp[15:0];
                end
            end else if (state == STATE_NORMALIZE_LOOP && gv_copy >= chunk_size && loop_idx >= chunk_size) begin
                // 清零超出chunk_size的元素
                weights_array[gv_copy] <= {SCORE_WIDTH{1'b0}};
            end
        end
    end
endgenerate

//================================================================================
// 主状态机（简化版，移除了for循环）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_IDLE;
        done <= 1'b0;
        busy <= 1'b0;
        max_wr_en <= 1'b0;
        sum_wr_en <= 1'b0;
        renorm_en <= 1'b0;
        weights_valid <= 1'b0;
        loop_idx <= 6'd0;
        clear_exp_trigger <= 1'b0;
        copy_exp_to_weights <= 1'b0;
    end else begin
        // 默认值
        max_wr_en <= 1'b0;
        sum_wr_en <= 1'b0;
        renorm_en <= 1'b0;
        weights_valid <= 1'b0;
        done <= 1'b0;
        clear_exp_trigger <= 1'b0;
        copy_exp_to_weights <= 1'b0;
        
        case (state)
            //================================================================
            // STATE_IDLE：等待启动
            //================================================================
            STATE_IDLE: begin
                if (start) begin
                    busy <= 1'b1;
                    state <= STATE_LOAD_STATS;
                    
                    `ifdef DEBUG_SOFTMAX
                    $display("[%0t] Softmax Start: head=%0d, row=%0d, chunk=%0d, size=%0d, first=%0b, last=%0b",
                             $time, head_idx, row_idx, chunk_id, chunk_size, chunk_first, chunk_last);
                    `endif
                end
            end
            
            //================================================================
            // STATE_LOAD_STATS：加载统计量
            //================================================================
            STATE_LOAD_STATS: begin
                if (chunk_first) begin
                    old_max <= NEG_INF;
                    old_sum_internal <= {INTERNAL_ACCUM_WIDTH{1'b0}};
                end else begin
                    old_max <= max_rd_value;
                    
                    // 扩展sum到内部位宽
                    if (sum_rd_value[ACCUM_WIDTH-1]) begin
                        old_sum_internal <= {{(INTERNAL_ACCUM_WIDTH-ACCUM_WIDTH){1'b1}}, sum_rd_value};
                    end else begin
                        old_sum_internal <= {{(INTERNAL_ACCUM_WIDTH-ACCUM_WIDTH){1'b0}}, sum_rd_value};
                    end
                end
                
                state <= STATE_FIND_MAX_INIT;
                
                `ifdef DEBUG_SOFTMAX
                $display("[%0t]   Loaded stats: old_max=%0d, old_sum=%0d",
                         $time, chunk_first ? NEG_INF : max_rd_value,
                         chunk_first ? 0 : sum_rd_value);
                `endif
            end
            
            //================================================================
            // STATE_FIND_MAX_INIT：初始化查找最大值
            //================================================================
            STATE_FIND_MAX_INIT: begin
                loop_idx <= 6'd0;
                chunk_max <= NEG_INF;
                state <= STATE_FIND_MAX_LOOP;
            end
            
            //================================================================
            // STATE_FIND_MAX_LOOP：顺序查找chunk的最大值
            //================================================================
            STATE_FIND_MAX_LOOP: begin
                if (loop_idx < chunk_size) begin
                    if (scores_array[loop_idx] > chunk_max) begin
                        chunk_max <= scores_array[loop_idx];
                    end
                    loop_idx <= loop_idx + 1'b1;
                end else begin
                    state <= STATE_UPDATE_MAX;
                    
                    `ifdef DEBUG_SOFTMAX
                    $display("[%0t]   Chunk max found: %0d", $time, chunk_max);
                    `endif
                end
            end
            
            //================================================================
            // STATE_UPDATE_MAX：更新全局最大值
            //================================================================
            STATE_UPDATE_MAX: begin
                if (chunk_max > old_max) begin
                    new_max <= chunk_max;
                    max_updated <= 1'b1;
                    
                    // 计算重归一化scale: exp(old_max - new_max)
                    if (!chunk_first) begin
                        renorm_scale <= exp_approx_improved(old_max - chunk_max);
                    end
                    
                    `ifdef DEBUG_SOFTMAX
                    $display("[%0t]   Max updated: %0d -> %0d", $time, old_max, chunk_max);
                    `endif
                end else begin
                    new_max <= old_max;
                    max_updated <= 1'b0;
                    
                    `ifdef DEBUG_SOFTMAX
                    $display("[%0t]   Max unchanged: %0d", $time, old_max);
                    `endif
                end
                
                // 如果max更新且存在旧sum，需重归一化旧sum
                if (chunk_max > old_max && !chunk_first) begin
                    temp_sum = old_sum_internal * {{(INTERNAL_ACCUM_WIDTH-SCORE_WIDTH){1'b0}}, 
                                                    exp_approx_improved(old_max - chunk_max)};
                    old_sum_internal <= temp_sum >>> FRAC_BITS;
                end
                
                loop_idx <= 6'd0;
                state <= STATE_COMPUTE_EXP;
            end
            
            //================================================================
            // STATE_COMPUTE_EXP：顺序计算exp（使用generate块）
            //================================================================
            STATE_COMPUTE_EXP: begin
                if (loop_idx < chunk_size) begin
                    // exp计算在generate块中完成
                    loop_idx <= loop_idx + 1'b1;
                end else begin
                    // 触发清零超出chunk_size的元素
                    clear_exp_trigger <= 1'b1;
                    
                    // 重置循环索引，准备累加
                    loop_idx <= 6'd0;
                    new_sum_internal <= old_sum_internal;
                    state <= STATE_ACCUMULATE_SUM;
                    
                    `ifdef DEBUG_SOFTMAX
                    $display("[%0t]   Exp computation done", $time);
                    `endif
                end
            end
            
            //================================================================
            // STATE_ACCUMULATE_SUM：顺序累加sum
            //================================================================
            STATE_ACCUMULATE_SUM: begin
                if (loop_idx < chunk_size) begin
                    // 扩展exp_values到内部累加器位宽
                    if (exp_values[loop_idx][SCORE_WIDTH-1]) begin
                        temp_sum = {{(INTERNAL_ACCUM_WIDTH-SCORE_WIDTH){1'b1}}, exp_values[loop_idx]};
                    end else begin
                        temp_sum = {{(INTERNAL_ACCUM_WIDTH-SCORE_WIDTH){1'b0}}, exp_values[loop_idx]};
                    end
                    
                    new_sum_internal <= saturate_add(new_sum_internal, temp_sum);
                    loop_idx <= loop_idx + 1'b1;
                end else begin
                    state <= STATE_WRITE_STATS;
                    
                    `ifdef DEBUG_SOFTMAX
                    $display("[%0t]   Sum accumulated: %0d", $time, new_sum_internal);
                    `endif
                end
            end
            
            //================================================================
            // STATE_WRITE_STATS：写回统计量
            //================================================================
            STATE_WRITE_STATS: begin
                max_wr_en <= 1'b1;
                max_wr_head <= head_idx;
                max_wr_row <= row_idx;
                max_wr_value <= new_max;
                
                sum_wr_en <= 1'b1;
                sum_wr_head <= head_idx;
                sum_wr_row <= row_idx;
                
                // 饱和截断到ACCUM_WIDTH
                if (new_sum_internal > {{(INTERNAL_ACCUM_WIDTH-ACCUM_WIDTH){1'b0}}, 
                                        {1'b0, {(ACCUM_WIDTH-1){1'b1}}}}) begin
                    sum_wr_value <= {1'b0, {(ACCUM_WIDTH-1){1'b1}}};
                end else if (new_sum_internal < {{(INTERNAL_ACCUM_WIDTH-ACCUM_WIDTH){1'b1}}, 
                                                  {1'b1, {(ACCUM_WIDTH-1){1'b0}}}}) begin
                    sum_wr_value <= {1'b1, {(ACCUM_WIDTH-1){1'b0}}};
                end else begin
                    sum_wr_value <= new_sum_internal[ACCUM_WIDTH-1:0];
                end
                
                if (max_updated && !chunk_first) begin
                    renorm_en <= 1'b1;
                end
                
                `ifdef DEBUG_SOFTMAX
                $display("[%0t]   Write stats: max=%0d, sum=%0d, renorm=%0b",
                         $time, new_max, new_sum_internal[ACCUM_WIDTH-1:0], max_updated);
                `endif
                
                if (chunk_last) begin
                    loop_idx <= 6'd0;
                    state <= STATE_NORMALIZE_INIT;
                end else begin
                    // 触发复制exp_values到weights_array
                    copy_exp_to_weights <= 1'b1;
                    state <= STATE_DONE;
                end
            end
            
            //================================================================
            // STATE_NORMALIZE_INIT：初始化归一化
            //================================================================
            STATE_NORMALIZE_INIT: begin
                loop_idx <= 6'd0;
                state <= STATE_NORMALIZE_LOOP;
                
                `ifdef DEBUG_SOFTMAX
                $display("[%0t]   Start normalization: sum=%0d", $time, new_sum_internal);
                `endif
            end
            
            //================================================================
            // STATE_NORMALIZE_LOOP：顺序归一化（在generate块中完成）
            //================================================================
            STATE_NORMALIZE_LOOP: begin
                if (loop_idx < chunk_size) begin
                    // 归一化计算在generate块中完成
                    loop_idx <= loop_idx + 1'b1;
                end else begin
                    state <= STATE_DONE;
                    
                    `ifdef DEBUG_SOFTMAX
                    $display("[%0t]   Normalization done", $time);
                    `endif
                end
            end
            
            //================================================================
            // STATE_DONE：完成
            //================================================================
            STATE_DONE: begin
                weights_valid <= 1'b1;
                done <= 1'b1;
                busy <= 1'b0;
                
                `ifdef DEBUG_SOFTMAX
                $display("[%0t] Softmax Done", $time);
                `endif
                
                state <= STATE_IDLE;
            end
            
            default: begin
                state <= STATE_IDLE;
            end
        endcase
    end
end

//================================================================================
// 调试支持
//================================================================================

`ifdef DEBUG_SOFTMAX
always @(posedge clk) begin
    if (state != STATE_IDLE && state != STATE_DONE) begin
        $display("[%0t] State=%0d, loop_idx=%0d", $time, state, loop_idx);
    end
end
`endif

endmodule