`timescale 1ns / 1ps

//================================================================================
// Delta Calculator - DFA Delta向量计算
//
// 功能说明：
// 计算 delta = B · error
// 
// 对于二分类：
//   delta[d] = B[d][0] × error（error是标量）
//   
// 对于多分类：
//   delta[d] = Σ(B[d][c] × error[c]) for c in 0..NUM_CLASSES-1
//
// 工作流程：
//   for each dimension d in 0..DIM-1:
//       accumulator = 0
//       for each class c in 0..NUM_CLASSES-1:
//           b_value = read_B_matrix(d, c)
//           accumulator += b_value × error[c]
//       delta[d] = accumulator
//
// 性能：
//   二分类（NUM_CLASSES=1）：DIM cycles（每维度1次MAC）
//   多分类（NUM_CLASSES=10）：DIM × 10 cycles（每维度10次MAC）
//
// 作者：MEIGA Team
// 日期：2025-11-19
// 版本：v1.0
//================================================================================

module delta_calculator #(
    parameter DIM = 32,              // 维度（8 or 32）
    parameter NUM_CLASSES = 1,       // 分类数（二分类=1, 多分类=10）
    parameter DATA_WIDTH = 16,       // Q4.12格式
    parameter DIM_ADDR_WIDTH = 5,    // log2(32) = 5
    parameter LAYER_ID = 4           // 层ID (0-4)，默认Layer 4
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire start,               // 开始计算（脉冲）
    output reg  done,                // 计算完成
    output reg  busy,                // 计算中
    
    //==========================================================================
    // 误差输入
    // 二分类：error只用error[0]（单个标量）
    // 多分类：error是完整向量[NUM_CLASSES × DATA_WIDTH]
    //==========================================================================
    input  wire [NUM_CLASSES*DATA_WIDTH-1:0] error_vector,
    
    //==========================================================================
    // B矩阵读取接口
    //==========================================================================
    output reg         b_rd_en,                  // 读使能
    output reg  [2:0]  b_layer_id,               // 层ID (0-4)
    output reg  [DIM_ADDR_WIDTH-1:0] b_row_addr,    // 行地址（维度）
    output reg  [3:0]  b_col_addr,               // 列地址（类别，0-9）
    input  wire [DATA_WIDTH-1:0] b_data,         // B矩阵元素
    input  wire        b_valid,                  // 数据有效
    
    //==========================================================================
    // Delta输出
    //==========================================================================
    output reg  [DIM*DATA_WIDTH-1:0] delta_vector,  // Delta向量
    output reg  delta_valid                      // Delta有效
);

//================================================================================
// 状态定义
//================================================================================
localparam IDLE         = 2'd0;
localparam READ_B       = 2'd1;  // 读取B矩阵
localparam COMPUTE_MAC  = 2'd2;  // 计算MAC
localparam OUTPUT_DELTA = 2'd3;  // 输出delta

reg [1:0] state_reg, state_next;

//================================================================================
// 计数器
//================================================================================
reg [DIM_ADDR_WIDTH-1:0] dim_cnt_reg, dim_cnt_next;       // 维度计数（0..DIM-1）
reg [3:0]                class_cnt_reg, class_cnt_next;   // 类别计数（0..NUM_CLASSES-1）

//================================================================================
// 累加器（每个维度一个）
//================================================================================
reg signed [31:0] accumulator_reg, accumulator_next;

//================================================================================
// Delta缓存
//================================================================================
reg [DATA_WIDTH-1:0] delta_buffer [0:DIM-1];

//================================================================================
// 误差向量拆分
//================================================================================
wire [DATA_WIDTH-1:0] error [0:NUM_CLASSES-1];
genvar g;
generate
    if (NUM_CLASSES == 1) begin : gen_binary_class
        // 二分类：只有一个误差值
        assign error[0] = error_vector[DATA_WIDTH-1:0];
    end else begin : gen_multi_class
        // 多分类：拆分误差向量
        for (g = 0; g < NUM_CLASSES; g = g + 1) begin : gen_error_split
            assign error[g] = error_vector[g*DATA_WIDTH +: DATA_WIDTH];
        end
    end
endgenerate

//================================================================================
// 状态机 - 时序逻辑
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg <= IDLE;
        dim_cnt_reg <= {DIM_ADDR_WIDTH{1'b0}};
        class_cnt_reg <= 4'd0;
        accumulator_reg <= 32'sd0;
    end else begin
        state_reg <= state_next;
        dim_cnt_reg <= dim_cnt_next;
        class_cnt_reg <= class_cnt_next;
        accumulator_reg <= accumulator_next;
    end
end

//================================================================================
// 状态机 - 组合逻辑
//================================================================================
always @(*) begin
    // 默认值
    state_next = state_reg;
    dim_cnt_next = dim_cnt_reg;
    class_cnt_next = class_cnt_reg;
    accumulator_next = accumulator_reg;
    
    b_rd_en = 1'b0;
    b_layer_id = LAYER_ID[2:0];      // 输出层ID
    b_row_addr = dim_cnt_reg;
    b_col_addr = class_cnt_reg;
    
    done = 1'b0;
    busy = (state_reg != IDLE);
    delta_valid = 1'b0;
    
    case (state_reg)
        //======================================================================
        // IDLE: 等待启动
        //======================================================================
        IDLE: begin
            if (start) begin
                state_next = READ_B;
                dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                class_cnt_next = 4'd0;
                accumulator_next = 32'sd0;
            end
        end
        
        //======================================================================
        // READ_B: 发出读B矩阵请求
        //======================================================================
        READ_B: begin
            b_rd_en = 1'b1;
            b_row_addr = dim_cnt_reg;
            b_col_addr = class_cnt_reg;
            state_next = COMPUTE_MAC;
        end
        
        //======================================================================
        // COMPUTE_MAC: 等待B矩阵数据并计算MAC
        //======================================================================
        COMPUTE_MAC: begin
            if (b_valid) begin
                // MAC: accumulator += B[d][c] × error[c]
                // Q4.12 × Q4.12 = Q8.24
                accumulator_next = accumulator_reg + 
                                  ($signed(b_data) * $signed(error[class_cnt_reg]));
                
                if (class_cnt_reg == NUM_CLASSES-1) begin
                    // 当前维度的所有类别都累加完毕
                    // 保存delta[dim]
                    // 移动到下一个维度
                    
                    if (dim_cnt_reg == DIM-1) begin
                        // 所有维度都计算完毕
                        state_next = OUTPUT_DELTA;
                    end else begin
                        // 继续下一个维度
                        dim_cnt_next = dim_cnt_reg + 1;
                        class_cnt_next = 4'd0;
                        state_next = READ_B;
                    end
                end else begin
                    // 继续当前维度的下一个类别
                    class_cnt_next = class_cnt_reg + 1;
                    state_next = READ_B;
                end
            end
        end
        
        //======================================================================
        // OUTPUT_DELTA: 输出delta向量
        //======================================================================
        OUTPUT_DELTA: begin
            delta_valid = 1'b1;
            done = 1'b1;
            state_next = IDLE;
        end
        
        default: state_next = IDLE;
    endcase
end

//================================================================================
// Delta缓存更新
//================================================================================
integer i;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < DIM; i = i + 1) begin
            delta_buffer[i] <= 16'd0;
        end
    end else begin
        if (state_reg == COMPUTE_MAC && b_valid && class_cnt_reg == NUM_CLASSES-1) begin
            // 当前维度的MAC计算完成，保存结果
            // 截断到Q4.12：取[27:12]
            delta_buffer[dim_cnt_reg] <= accumulator_next[27:12];
        end
        
        if (state_reg == COMPUTE_MAC && b_valid && class_cnt_reg == NUM_CLASSES-1 && 
            dim_cnt_reg != DIM-1) begin
            // 移动到下一个维度，清零累加器
            // 这个在时序逻辑中处理
        end
    end
end

// 累加器在维度切换时清零
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        accumulator_reg <= 32'sd0;
    end else begin
        if (state_reg == COMPUTE_MAC && b_valid && class_cnt_reg == NUM_CLASSES-1 && 
            dim_cnt_reg != DIM-1) begin
            // 当前维度完成，清零累加器准备下一个维度
            accumulator_reg <= 32'sd0;
        end else if (state_reg == IDLE && start) begin
            accumulator_reg <= 32'sd0;
        end else begin
            accumulator_reg <= accumulator_next;
        end
    end
end

//================================================================================
// Delta向量打包输出
//================================================================================
always @(*) begin:Da
    integer j;
    for (j = 0; j < DIM; j = j + 1) begin
        delta_vector[j*DATA_WIDTH +: DATA_WIDTH] = delta_buffer[j];
    end
end

endmodule