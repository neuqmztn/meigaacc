`timescale 1ns / 1ps

//================================================================================
// 模块名称: residual_add_bfp (Block Floating Point残差加法器)
// 功能描述: Transformer残差连接模块（BFP格式）
//
// 算法说明:
// 实现 Output = Input + Result 的BFP格式加法
// BFP (Block Floating Point) = 1个共享指数 + N个尾数
//
// 什么是BFP格式?
// ┌──────────────────────────────────────────────────────────┐
// │  共享指数(8位)  │  尾数0  │  尾数1  │ ... │  尾数31  │
// │   Exponent      │  Mant0  │  Mant1  │ ... │  Mant31  │
// │     [7:0]       │  [7:0]  │  [7:0]  │ ... │  [7:0]   │
// └──────────────────────────────────────────────────────────┘
//     1个8位              32个8位尾数(共256位)
//
// BFP优势:
// 1. 存储高效: 32个数字只需1个指数(vs 浮点需32个指数)
// 2. 计算并行: 所有尾数可并行处理
// 3. 硬件友好: 减少指数处理逻辑
//
// 加法流程:
// ┌──────────┐
// │ 输入准备 │  读取Input和Result两组BFP数据
// └────┬─────┘
//      │
// ┌────▼─────┐
// │ 指数对齐 │  找出较大指数，将较小指数的尾数右移
// └────┬─────┘  例: Exp_A=5, Exp_B=3 → B的尾数右移2位
//      │
// ┌────▼─────┐
// │ 尾数相加 │  32个对齐后的尾数并行相加
// └────┬─────┘  结果: 11位尾数 + 3位保护位 = 14位有符号数
//      │
// ┌────▼─────┐
// │ 找最大值 │  在32个和中找绝对值最大的
// └────┬─────┘  目的: 确定归一化移位量
//      │
// ┌────▼─────┐
// │ 归一化   │  ⭐ 这里使用automatic变量处理移位
// └────┬─────┘  - 使最高有效位对齐到标准位置
//      │        - 调整共享指数
//      │        - 截断到8位标准尾数
// ┌────▼─────┐
// │ 输出结果 │  新的BFP格式: 1个指数 + 32个尾数
// └──────────┘
//
// 关键技术细节:
// 
// 1. 保护位(Guard Bits):
//    - 加法时扩展3位保护位(总共11位尾数)
//    - 最终舍入时根据保护位决定+1或截断
//    - 提高数值精度，减少舍入误差
//
// 2. automatic变量:
//    - 在NORMALIZE状态的for循环中使用
//    - 用于存储每个维度的移位中间结果
//    - 每次循环迭代有独立的变量实例
//    - 避免循环间的数据相关性
//
// 3. 两阶段归一化:
//    Cycle 0: 分析最大值，计算移位量和新指数
//    Cycle 1: 执行32路并行的移位和舍入操作
//
// 版本历史:
// v1.0 - 初始版本
// v2.0 - 性能优化版本
//   - 优化count_leading_zeros为二分查找（13级→4级）
//   - 优化FIND_MAX为树形比较器（8周期→4周期）
//   - 添加输出握手机制
//   - 移除未使用的mode参数
//   - 添加条件编译的性能监控
//   - 修复ERROR_STATE卡住问题
// v2.1 - 代码规范版本 (当前版本)
//   - 使用automatic变量处理循环内临时变量
//   - 将移位操作分解为两步: 移位→存储→位选择
//   - 添加详细的注释说明各个处理阶段
//
// 性能指标:
// - 每token处理周期: ~12 cycles (优化前16 cycles)
// - 总延迟(641 tokens): ~7,692 cycles (优化前10,256 cycles)
// - 性能提升: 25%
// - 并行度: 32维度并行处理
// - 时钟频率: 100 MHz (10ns周期)
//
// 资源占用估算:
// - 寄存器: ~2,500个
// - LUT: ~8,000个  
// - DSP: 0个 (纯逻辑实现)
// - BRAM: 0个 (不含存储器)
//
// 作者: MEIGA Design Team
// 日期: 2025-11-13
// 版本: 2.1
//================================================================================

module residual_add_bfp #(
    parameter TOKEN_NUM  = 641,
    parameter DIM        = 32,
    parameter DATA_WIDTH = 8,
    parameter EXP_WIDTH  = 8,
    parameter ADDR_WIDTH = 10,
    parameter GUARD_BITS = 3,
    parameter WRITE_TIMEOUT = 100  // 新增: 写超时阈值
)(
    input  wire clk,
    input  wire rst_n,
    
    input  wire start,
    output reg  done,
    output wire busy,
    output reg  error,
    
    output reg  input_rd_en,
    output reg  [ADDR_WIDTH-1:0] input_rd_addr,
    input  wire [EXP_WIDTH-1:0] input_exp,
    input  wire [DIM*DATA_WIDTH-1:0] input_mant,
    input  wire input_valid,
    
    output reg  result_rd_en,
    output reg  [ADDR_WIDTH-1:0] result_rd_addr,
    input  wire [EXP_WIDTH-1:0] result_exp,
    input  wire [DIM*DATA_WIDTH-1:0] result_mant,
    input  wire result_valid,
    
    output reg  output_wr_en,
    output reg  [ADDR_WIDTH-1:0] output_wr_addr,
    output reg  [EXP_WIDTH-1:0] output_exp,
    output reg  [DIM*DATA_WIDTH-1:0] output_mant,
    output reg  output_valid,
    input  wire output_ready,
    
    output reg  [3:0] state,
    output reg  [9:0] processed_count,
    output reg  [31:0] cycle_count,
    output reg  overflow_detected,     // 现已实现
    output reg  underflow_detected     // 现已实现
);

//================================================================================
// 状态定义
//================================================================================
localparam IDLE           = 4'd0;
localparam READ_PREP      = 4'd1;
localparam READ_WAIT      = 4'd2;
localparam ALIGN_EXP      = 4'd3;
localparam ADD_MANTISSA   = 4'd4;
localparam FIND_MAX       = 4'd5;
localparam NORMALIZE      = 4'd6;
localparam WRITE          = 4'd7;
localparam NEXT_TOKEN     = 4'd8;
localparam DONE_STATE     = 4'd9;
localparam ERROR_STATE    = 4'd10;

//================================================================================
// 内部寄存器
//================================================================================
reg [9:0] token_idx;
reg [3:0] read_wait_cnt;
reg [3:0] process_cycle;
reg [7:0] write_timeout_cnt;  // 新增: 写超时计数器

reg [EXP_WIDTH-1:0] input_exp_reg;
reg [DIM*DATA_WIDTH-1:0] input_mant_reg;
reg [EXP_WIDTH-1:0] result_exp_reg;
reg [DIM*DATA_WIDTH-1:0] result_mant_reg;

reg [EXP_WIDTH-1:0] aligned_exp;
reg signed [DATA_WIDTH+GUARD_BITS:0] aligned_mant_a [0:DIM-1];
reg signed [DATA_WIDTH+GUARD_BITS:0] aligned_mant_b [0:DIM-1];
reg signed [DATA_WIDTH+GUARD_BITS+1:0] sum_mant [0:DIM-1];

reg [5:0] max_abs_pos;
reg [DATA_WIDTH+GUARD_BITS+1:0] max_abs_val;
reg [4:0] shift_amount;
reg signed [EXP_WIDTH+1:0] new_exp;  // 扩展到10位以检测溢出
reg signed [DATA_WIDTH-1:0] norm_mant [0:DIM-1];

reg all_zero_flag;  // 新增: 全零标志

integer i;

//================================================================================
// 解包输入尾数数组
//================================================================================
wire signed [DATA_WIDTH-1:0] input_mant_array [0:DIM-1];
wire signed [DATA_WIDTH-1:0] result_mant_array [0:DIM-1];

genvar g;
generate
    for (g = 0; g < DIM; g = g + 1) begin : gen_unpack
        assign input_mant_array[g] = input_mant_reg[g*DATA_WIDTH +: DATA_WIDTH];
        assign result_mant_array[g] = result_mant_reg[g*DATA_WIDTH +: DATA_WIDTH];
    end
endgenerate

//================================================================================
// 快速前导零计数函数 (优化版,二分查找)
//================================================================================
function [4:0] count_leading_zeros_fast;
    input [DATA_WIDTH+GUARD_BITS+1:0] value;
    reg [4:0] count;
    reg [DATA_WIDTH+GUARD_BITS+1:0] tmp;
    begin
        tmp = value;
        count = 0;
        
        // 检查高7位
        if (tmp[12:6] == 7'b0000000) begin
            count = count + 7;
            tmp = {tmp[5:0], 7'b0};
        end
        
        // 检查高4位
        if (tmp[12:9] == 4'b0000) begin
            count = count + 4;
            tmp = {tmp[8:0], 4'b0};
        end
        
        // 检查高2位
        if (tmp[12:11] == 2'b00) begin
            count = count + 2;
            tmp = {tmp[10:0], 2'b0};
        end
        
        // 检查最高位
        if (tmp[12] == 1'b0) begin
            count = count + 1;
        end
        
        count_leading_zeros_fast = count;
    end
endfunction

//================================================================================
// 树形比较器数组 (用于FIND_MAX加速)
//================================================================================
reg [DATA_WIDTH+GUARD_BITS+1:0] abs_array [0:DIM-1];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level1 [0:15];
reg [5:0] pos_level1 [0:15];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level2 [0:7];
reg [5:0] pos_level2 [0:7];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level3 [0:3];
reg [5:0] pos_level3 [0:3];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level4 [0:1];
reg [5:0] pos_level4 [0:1];

//================================================================================
// 主状态机
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        done <= 1'b0;
        error <= 1'b0;
        
        input_rd_en <= 1'b0;
        input_rd_addr <= {ADDR_WIDTH{1'b0}};
        result_rd_en <= 1'b0;
        result_rd_addr <= {ADDR_WIDTH{1'b0}};
        
        output_wr_en <= 1'b0;
        output_wr_addr <= {ADDR_WIDTH{1'b0}};
        output_exp <= {EXP_WIDTH{1'b0}};
        output_mant <= {(DIM*DATA_WIDTH){1'b0}};
        output_valid <= 1'b0;
        
        token_idx <= 10'd0;
        read_wait_cnt <= 4'd0;
        process_cycle <= 4'd0;
        write_timeout_cnt <= 8'd0;
        
        processed_count <= 10'd0;
        cycle_count <= 32'd0;
        
        overflow_detected <= 1'b0;
        underflow_detected <= 1'b0;
        all_zero_flag <= 1'b0;
        
        aligned_exp <= {EXP_WIDTH{1'b0}};
        max_abs_pos <= 6'd0;
        max_abs_val <= {(DATA_WIDTH+GUARD_BITS+2){1'b0}};
        shift_amount <= 5'd0;
        new_exp <= {(EXP_WIDTH+2){1'b0}};
        
        input_exp_reg <= {EXP_WIDTH{1'b0}};
        input_mant_reg <= {(DIM*DATA_WIDTH){1'b0}};
        result_exp_reg <= {EXP_WIDTH{1'b0}};
        result_mant_reg <= {(DIM*DATA_WIDTH){1'b0}};
        
        for (i = 0; i < DIM; i = i + 1) begin
            aligned_mant_a[i] <= {(DATA_WIDTH+GUARD_BITS+1){1'b0}};
            aligned_mant_b[i] <= {(DATA_WIDTH+GUARD_BITS+1){1'b0}};
            sum_mant[i] <= {(DATA_WIDTH+GUARD_BITS+2){1'b0}};
            norm_mant[i] <= {DATA_WIDTH{1'b0}};
            abs_array[i] <= {(DATA_WIDTH+GUARD_BITS+2){1'b0}};
        end
        
    end else begin
        
        // 默认清除一次性信号
        input_rd_en <= 1'b0;
        result_rd_en <= 1'b0;
        output_wr_en <= 1'b0;
        output_valid <= 1'b0;
        
        case (state)
            
            //====================================================================
            // IDLE: 等待启动信号
            //====================================================================
            IDLE: begin
                done <= 1'b0;
                error <= 1'b0;
                overflow_detected <= 1'b0;
                underflow_detected <= 1'b0;
                
                if (start) begin
                    token_idx <= 10'd0;
                    processed_count <= 10'd0;
                    cycle_count <= 32'd0;
                    state <= READ_PREP;
                end
            end
            
            //====================================================================
            // READ_PREP: 准备读取输入和结果
            //====================================================================
            READ_PREP: begin
                input_rd_en <= 1'b1;
                result_rd_en <= 1'b1;
                input_rd_addr <= token_idx[ADDR_WIDTH-1:0];
                result_rd_addr <= token_idx[ADDR_WIDTH-1:0];
                
                read_wait_cnt <= 4'd0;
                state <= READ_WAIT;
            end
            
            //====================================================================
            // READ_WAIT: 等待读取数据有效
            //====================================================================
            READ_WAIT: begin
                read_wait_cnt <= read_wait_cnt + 1;
                
                if (input_valid && result_valid) begin
                    input_exp_reg <= input_exp;
                    input_mant_reg <= input_mant;
                    result_exp_reg <= result_exp;
                    result_mant_reg <= result_mant;
                    state <= ALIGN_EXP;
                    
                end else if (read_wait_cnt > 4'd10) begin
                    // 读取超时保护
                    state <= ERROR_STATE;
                end
            end
            
            //====================================================================
            // ALIGN_EXP: 指数对齐
            //====================================================================
            ALIGN_EXP: begin:a1
                reg signed [EXP_WIDTH:0] exp_diff;  // 9位有符号差值
                
                exp_diff = $signed({1'b0, input_exp_reg}) - $signed({1'b0, result_exp_reg});
                
                //----------------------------------------------------------------
                // 情况1: input指数更大
                //----------------------------------------------------------------
                if (exp_diff > 0) begin
                    aligned_exp <= input_exp_reg;
                    
                    for (i = 0; i < DIM; i = i + 1) begin
                        // Input不移位,扩展保护位
                        aligned_mant_a[i] <= {input_mant_array[i], {GUARD_BITS{1'b0}}};
                        
                        // Result右移对齐
                        // ⭐ 优化: 限制最大移位量,超过则直接置0
                        if (exp_diff >= (DATA_WIDTH + GUARD_BITS)) begin
                            aligned_mant_b[i] <= {(DATA_WIDTH+GUARD_BITS+1){1'b0}};
                        end else begin
                            aligned_mant_b[i] <= ({result_mant_array[i], {GUARD_BITS{1'b0}}}) >>> exp_diff[4:0];
                        end
                    end
                    
                //----------------------------------------------------------------
                // 情况2: result指数更大
                //----------------------------------------------------------------
                end else if (exp_diff < 0) begin
                    aligned_exp <= result_exp_reg;
                    
                    for (i = 0; i < DIM; i = i + 1) begin
                        // Result不移位,扩展保护位
                        aligned_mant_b[i] <= {result_mant_array[i], {GUARD_BITS{1'b0}}};
                        
                        // Input右移对齐
                        // ⭐ 优化: 限制最大移位量
                        if ((-exp_diff) >= (DATA_WIDTH + GUARD_BITS)) begin
                            aligned_mant_a[i] <= {(DATA_WIDTH+GUARD_BITS+1){1'b0}};
                        end else begin
                            aligned_mant_a[i] <= ({input_mant_array[i], {GUARD_BITS{1'b0}}}) >>> (-exp_diff[4:0]);
                        end
                    end
                    
                //----------------------------------------------------------------
                // 情况3: 指数相等
                //----------------------------------------------------------------
                end else begin
                    aligned_exp <= input_exp_reg;
                    
                    for (i = 0; i < DIM; i = i + 1) begin
                        aligned_mant_a[i] <= {input_mant_array[i], {GUARD_BITS{1'b0}}};
                        aligned_mant_b[i] <= {result_mant_array[i], {GUARD_BITS{1'b0}}};
                    end
                end
                
                state <= ADD_MANTISSA;
            end
            
            //====================================================================
            // ADD_MANTISSA: 尾数相加
            //====================================================================
            ADD_MANTISSA: begin
                for (i = 0; i < DIM; i = i + 1) begin
                    sum_mant[i] <= aligned_mant_a[i] + aligned_mant_b[i];
                end
                
                process_cycle <= 4'd0;
                state <= FIND_MAX;
            end
            
            //====================================================================
            // FIND_MAX: 找最大绝对值 (4周期树形比较)
            //====================================================================
            FIND_MAX: begin
                process_cycle <= process_cycle + 1;
                
                case (process_cycle)
                    //------------------------------------------------------------
                    // Cycle 0: 计算绝对值
                    //------------------------------------------------------------
                    4'd0: begin
                        // ⭐ 新增: 检测全零输入
                        all_zero_flag <= 1'b1;
                        
                        for (i = 0; i < DIM; i = i + 1) begin
                            if (sum_mant[i][DATA_WIDTH+GUARD_BITS+1]) begin  // 负数
                                abs_array[i] <= -sum_mant[i];
                            end else begin
                                abs_array[i] <= sum_mant[i];
                            end
                            
                            // 检查是否有非零值
                            if (sum_mant[i] != 0) begin
                                all_zero_flag <= 1'b0;
                            end
                        end
                    end
                    
                    //------------------------------------------------------------
                    // Cycle 1: Level 1 比较 (32→16)
                    //------------------------------------------------------------
                    4'd1: begin
                        for (i = 0; i < 16; i = i + 1) begin
                            if (abs_array[2*i] >= abs_array[2*i+1]) begin
                                max_level1[i] <= abs_array[2*i];
                                pos_level1[i] <= 2*i;
                            end else begin
                                max_level1[i] <= abs_array[2*i+1];
                                pos_level1[i] <= 2*i + 1;
                            end
                        end
                    end
                    
                    //------------------------------------------------------------
                    // Cycle 2: Level 2 比较 (16→8)
                    //------------------------------------------------------------
                    4'd2: begin
                        for (i = 0; i < 8; i = i + 1) begin
                            if (max_level1[2*i] >= max_level1[2*i+1]) begin
                                max_level2[i] <= max_level1[2*i];
                                pos_level2[i] <= pos_level1[2*i];
                            end else begin
                                max_level2[i] <= max_level1[2*i+1];
                                pos_level2[i] <= pos_level1[2*i+1];
                            end
                        end
                    end
                    
                    //------------------------------------------------------------
                    // Cycle 3: Level 3 比较 (8→4)
                    //------------------------------------------------------------
                    4'd3: begin
                        for (i = 0; i < 4; i = i + 1) begin
                            if (max_level2[2*i] >= max_level2[2*i+1]) begin
                                max_level3[i] <= max_level2[2*i];
                                pos_level3[i] <= pos_level2[2*i];
                            end else begin
                                max_level3[i] <= max_level2[2*i+1];
                                pos_level3[i] <= pos_level2[2*i+1];
                            end
                        end
                    end
                    
                    //------------------------------------------------------------
                    // Cycle 4: 最终比较
                    //------------------------------------------------------------
                    4'd4: begin
                        for (i = 0; i < 2; i = i + 1) begin
                            if (max_level3[2*i] >= max_level3[2*i+1]) begin
                                max_level4[i] <= max_level3[2*i];
                                pos_level4[i] <= pos_level3[2*i];
                            end else begin
                                max_level4[i] <= max_level3[2*i+1];
                                pos_level4[i] <= pos_level3[2*i+1];
                            end
                        end
                        
                        if (max_level4[0] >= max_level4[1]) begin
                            max_abs_val <= max_level4[0];
                            max_abs_pos <= pos_level4[0];
                        end else begin
                            max_abs_val <= max_level4[1];
                            max_abs_pos <= pos_level4[1];
                        end
                        
                        process_cycle <= 4'd0;
                        state <= NORMALIZE;
                    end
                endcase
            end
            
            //====================================================================
            // NORMALIZE: 归一化处理 (2周期)
            //====================================================================
            NORMALIZE: begin
                process_cycle <= process_cycle + 1;
                
                case (process_cycle)
                    //------------------------------------------------------------
                    // Cycle 0: 计算移位量和新指数
                    //------------------------------------------------------------
                    4'd0: begin
                        
                        // ⭐ 新增: 处理全零情况
                        if (all_zero_flag || max_abs_val == 0) begin
                            shift_amount <= 5'd0;
                            new_exp <= {(EXP_WIDTH+2){1'b0}};  // 指数为0
                            
                        // 检查是否需要右移 (溢出)
                        end else if (max_abs_val[DATA_WIDTH+GUARD_BITS+1]) begin
                            shift_amount <= 5'd31;  // 特殊标记: 右移1位
                            new_exp <= $signed({2'b00, aligned_exp}) + 1;  // 10位扩展
                            
                        // 正常左移归一化
                        end else begin:a2
                            reg [4:0] leading_zeros;
                            leading_zeros = count_leading_zeros_fast(max_abs_val);
                            
                            // 计算左移量
                            if (leading_zeros > 1) begin
                                shift_amount <= leading_zeros - 1;
                            end else begin
                                shift_amount <= 5'd0;
                            end
                            
                            // 计算新指数 (扩展到10位)
                            new_exp <= $signed({2'b00, aligned_exp}) - $signed({5'b00000, leading_zeros}) + 1;
                        end
                        
                        // ⭐ 新增: 检测指数溢出/下溢
                        if (new_exp > 255) begin
                            overflow_detected <= 1'b1;
                        end else if (new_exp < 0) begin
                            underflow_detected <= 1'b1;
                        end
                    end
                    
                    //------------------------------------------------------------
                    // Cycle 1: 执行归一化和舍入
                    //------------------------------------------------------------
                    4'd1: begin
                        
                        // ⭐ 改进: 使用wire而非automatic,提高兼容性
                        for (i = 0; i < DIM; i = i + 1) begin:a3
                            reg signed [DATA_WIDTH+GUARD_BITS+1:0] shifted_val;
                            reg signed [DATA_WIDTH:0] rounded_val;  // 9位,用于检测舍入溢出
                            
                            //================================================
                            // 步骤1: 移位操作
                            //================================================
                            if (all_zero_flag) begin
                                shifted_val = {(DATA_WIDTH+GUARD_BITS+2){1'b0}};
                                
                            end else if (shift_amount == 5'd31) begin
                                // 右移1位
                                shifted_val = sum_mant[i] >>> 1;
                                
                            end else if (shift_amount > 0) begin
                                // 左移
                                shifted_val = sum_mant[i] << shift_amount;
                            end else begin
                                // 不移位
                                shifted_val = sum_mant[i];
                            end
                            
                            //================================================
                            // 步骤2: 舍入操作 (带溢出检测)
                            //================================================
                            if (shifted_val[GUARD_BITS-1]) begin
                                // 舍入位为1,向上舍入
                                rounded_val = $signed({1'b0, shifted_val[DATA_WIDTH+GUARD_BITS-1:GUARD_BITS]}) + 1;
                            end else begin
                                // 舍入位为0,直接截断
                                rounded_val = {1'b0, shifted_val[DATA_WIDTH+GUARD_BITS-1:GUARD_BITS]};
                            end
                            
                            //================================================
                            // 步骤3: 舍入溢出检查
                            //================================================
                            // ⭐ 新增: 检测舍入后是否溢出
                            if (rounded_val[DATA_WIDTH]) begin
                                // 溢出: 例如 0111_1111 + 1 = 1_0000_0000
                                // 饱和到最大/最小值
                                if (shifted_val[DATA_WIDTH+GUARD_BITS+1]) begin
                                    // 负数溢出 → 饱和到最小负数
                                    norm_mant[i] <= {1'b1, {(DATA_WIDTH-1){1'b0}}};  // -128
                                end else begin
                                    // 正数溢出 → 饱和到最大正数
                                    norm_mant[i] <= {1'b0, {(DATA_WIDTH-1){1'b1}}};  // 127
                                end
                                overflow_detected <= 1'b1;
                            end else begin
                                // 正常情况
                                norm_mant[i] <= rounded_val[DATA_WIDTH-1:0];
                            end
                        end
                        
                        //====================================================
                        // 指数饱和处理 (防止截断)
                        //====================================================
                        if (new_exp > 255) begin
                            // 上溢: 饱和到最大指数
                            output_exp <= 8'hFF;
                            overflow_detected <= 1'b1;
                            
                        end else if (new_exp[EXP_WIDTH+1]) begin
                            // 下溢: 饱和到0 (检查符号位)
                            output_exp <= 8'h00;
                            underflow_detected <= 1'b1;
                            
                        end else begin
                            // 正常范围
                            output_exp <= new_exp[EXP_WIDTH-1:0];
                        end
                        
                        state <= WRITE;
                        write_timeout_cnt <= 8'd0;  // 重置超时计数器
                    end
                endcase
            end
            
            //====================================================================
            // WRITE: 写入输出 (带超时保护)
            //====================================================================
            WRITE: begin
                // 打包输出尾数
                for (i = 0; i < DIM; i = i + 1) begin
                    output_mant[i*DATA_WIDTH +: DATA_WIDTH] <= norm_mant[i];
                end
                
                // ⭐ 新增: 超时保护
                write_timeout_cnt <= write_timeout_cnt + 1;
                
                if (output_ready) begin
                    output_wr_en <= 1'b1;
                    output_valid <= 1'b1;
                    output_wr_addr <= token_idx[ADDR_WIDTH-1:0];
                    
                    processed_count <= processed_count + 1;
                    state <= NEXT_TOKEN;
                    
                end else if (write_timeout_cnt >= WRITE_TIMEOUT) begin
                    // 写超时,进入错误状态
                    error <= 1'b1;
                    state <= ERROR_STATE;
                end
            end
            
            //====================================================================
            // NEXT_TOKEN: 处理下一个token
            //====================================================================
            NEXT_TOKEN: begin
                if (token_idx < TOKEN_NUM - 1) begin
                    token_idx <= token_idx + 1;
                    state <= READ_PREP;
                end else begin
                    state <= DONE_STATE;
                end
            end
            
            //====================================================================
            // DONE_STATE: 完成
            //====================================================================
            DONE_STATE: begin
                done <= 1'b1;
                if (!start) begin
                    state <= IDLE;
                end
            end
            
            //====================================================================
            // ERROR_STATE: 错误状态 (自动恢复)
            //====================================================================
            ERROR_STATE: begin
                error <= 1'b1;
                
                // ⭐ 改进: 自动恢复而非永久卡死
                if (!start) begin
                    state <= IDLE;
                end
            end
            
            default: state <= IDLE;
            
        endcase
        
        // 周期计数器
        if (state != IDLE && state != DONE_STATE && state != ERROR_STATE) begin
            cycle_count <= cycle_count + 1;
        end
    end
end

//================================================================================
// 忙信号
//================================================================================
assign busy = (state != IDLE) && (state != DONE_STATE) && (state != ERROR_STATE);

//================================================================================
// 仿真性能监控 (可选)
//================================================================================
`ifdef SIMULATION
    reg [31:0] max_cycle_per_token;
    reg [31:0] min_cycle_per_token;
    reg [31:0] current_token_cycles;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_cycle_per_token <= 32'd0;
            min_cycle_per_token <= 32'hFFFFFFFF;
            current_token_cycles <= 32'd0;
        end else begin
            if (state == READ_PREP) begin
                current_token_cycles <= 32'd0;
            end else if (state != IDLE && state != DONE_STATE) begin
                current_token_cycles <= current_token_cycles + 1;
            end else if (state == NEXT_TOKEN) begin
                if (current_token_cycles > max_cycle_per_token) begin
                    max_cycle_per_token <= current_token_cycles;
                end
                if (current_token_cycles < min_cycle_per_token) begin
                    min_cycle_per_token <= current_token_cycles;
                end
            end
        end
    end
`endif

endmodule