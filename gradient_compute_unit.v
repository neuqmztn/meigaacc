`timescale 1ns / 1ps

//================================================================================
// Gradient Compute Unit - DFA二分类版本 (Fixed Version)
//
// 修正说明：
// 1. NUM_TOKENS = 640（不包含CLS token，CLS仅用于分类输出）
// 2. 确保所有640个token的梯度都被正确计算
// 3. 优化了WAIT_DELTA的延迟逻辑
// 4. 增强了调试和验证功能
//
// 二分类DFA算法：
// • 输入：error_scalar (Q4.12标量，BCE误差)
// • B矩阵：N×1 (N=8或32)
// • 输出：640个token × N维度的梯度
//
// 计算公式：
//   delta[d]   = B[d] * error_scalar
//   grad[t][d] = delta[d] * activation[t][d]
//
// 定点格式：Q4.12
// • 所有输入/输出均为16-bit有符号定点（[-8, 8)）
// • 乘法：Q4.12 × Q4.12 = Q8.24
// • 对齐：右移12bit → 回到Q4.12
//
// 性能估计：
// • 对于DIM=8：每个token约需DIM个周期写梯度
// • 总周期 ≈ NUM_TOKENS × DIM + 若干控制开销
// • DIM=8:  约 640×8  = 5120 cycles
// • DIM=32: 约 640×32 = 20480 cycles
//
// 作者：MEIGA Team
// 日期：2025-12-04
// 版本：v2.0 (NUM_TOKENS=640)
//================================================================================

module gradient_compute_unit #(
    //==========================================================================
    // 基本参数
    //==========================================================================
    parameter LAYER_ID          = 0,            // 层ID (0-4)
    parameter DIM               = 8,            // 维度 (8 or 32)
    parameter NUM_CLASSES       = 1,            // 二分类 = 1
    parameter NUM_TOKENS        = 640,          // ⚠️ 640个token（不含CLS）
    
    //==========================================================================
    // 数据格式参数
    //==========================================================================
    parameter DATA_WIDTH        = 16,           // Q4.12，16-bit有符号定点
    parameter TOKEN_ADDR_WIDTH  = 10,           // 支持0-1023 token地址
    parameter DIM_ADDR_WIDTH    = 5             // 支持32维
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire                         start,          // 启动GCU
    output reg                          done,           // 全部梯度写完
    output wire                         busy,           // 正在工作
    
    //==========================================================================
    // 输入：error_scalar (来自顶层DFA控制器)
    //==========================================================================
    input  wire [DATA_WIDTH-1:0]        error_scalar,   // Q4.12 (signed)
    
    //==========================================================================
    // 输入：B矩阵 (DIM x 1)
    //   - 由外部B矩阵存储单元提供
    //   - 按行访问：b_row_addr = 0..DIM-1, b_col_addr = 0
    //==========================================================================
    output reg                          b_rd_en,        // 读使能
    output reg  [DIM_ADDR_WIDTH-1:0]    b_row_addr,     // 行地址 (维度index)
    output reg  [3:0]                   b_col_addr,     // 列地址 (固定为0)
    input  wire [DATA_WIDTH-1:0]        b_data,         // B[d][0]
    input  wire                         b_valid,        // 数据有效
    
    //==========================================================================
    // 输入：激活值LOB (Layer Output Buffer)
    //   - 提供每个token对应的激活向量
    //   - act_token_addr: 0..639
    //   - act_vector: DIM个Q4.12元素打包
    //==========================================================================
    output reg                          act_rd_en,      // 读使能
    output reg  [TOKEN_ADDR_WIDTH-1:0]  act_token_addr, // token地址
    input  wire [DIM*DATA_WIDTH-1:0]    act_vector,     // 激活向量(打包)
    input  wire                         act_valid,      // 数据有效
    
    //==========================================================================
    // 梯度写出接口
    //==========================================================================
    output reg         grad_wr_en,              // 写使能
    output reg  [TOKEN_ADDR_WIDTH-1:0] grad_token_addr, // Token地址 (0-639)
    output reg  [DIM_ADDR_WIDTH-1:0]   grad_dim_addr,   // 维度地址 (0 to DIM-1)
    output reg  [DATA_WIDTH-1:0]       grad_data,       // 梯度值（Q4.12）
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [2:0]                  state,           // 当前状态
    output wire [31:0]                 cycle_count,     // 周期计数
    output wire [31:0]                 token_count      // Token计数
);

//================================================================================
// 状态定义
//================================================================================
localparam IDLE         = 3'd0;
localparam COMPUTE_DELTA= 3'd1;  // 计算delta向量 = B × error
localparam WAIT_DELTA   = 3'd2;  // 等待delta计算完成（流水线延迟）
localparam COMPUTE_GRAD = 3'd3;  // 请求激活值
localparam WAIT_ACT     = 3'd4;  // 等待激活值返回
localparam WRITE_GRAD   = 3'd5;  // 写梯度
localparam DONE_STATE   = 3'd6;

reg [2:0] state_reg, state_next;

//================================================================================
// 计数器
//================================================================================
reg [TOKEN_ADDR_WIDTH-1:0] token_cnt_reg, token_cnt_next;
reg [DIM_ADDR_WIDTH-1:0]   dim_cnt_reg, dim_cnt_next;
reg [31:0]                 cycle_cnt_reg, cycle_cnt_next;

//================================================================================
// Delta缓存（N个元素，存储 delta = B × error）
//================================================================================
reg signed [DATA_WIDTH-1:0] delta_buffer [0:DIM-1];

//================================================================================
// 激活值缓存（暂存当前token的激活向量）
// ⚠️ 修复：声明为 signed，并在写入时用 $signed() 显式转换
//================================================================================
reg signed [DATA_WIDTH-1:0] activation_buffer [0:DIM-1];

//================================================================================
// 流水线寄存器（用于B矩阵数据的流水处理）
// ⚠️ 修复：b_data_reg 也声明为 signed，并用 $signed(b_data) 赋值
//================================================================================
reg signed [DATA_WIDTH-1:0]  b_data_reg;
reg                           b_valid_reg;
reg        [DIM_ADDR_WIDTH-1:0] b_dim_reg;

//================================================================================
// 调试寄存器（仿真统计）
//================================================================================
integer total_cycles_sim;
integer tokens_processed_sim;
integer gradients_written_sim;

integer i;

//================================================================================
// 状态机寄存器
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg      <= IDLE;
        token_cnt_reg  <= {TOKEN_ADDR_WIDTH{1'b0}};
        dim_cnt_reg    <= {DIM_ADDR_WIDTH{1'b0}};
        cycle_cnt_reg  <= 32'd0;
        
        b_data_reg     <= 16'd0;
        b_valid_reg    <= 1'b0;
        b_dim_reg      <= {DIM_ADDR_WIDTH{1'b0}};
    end else begin
        state_reg      <= state_next;
        token_cnt_reg  <= token_cnt_next;
        dim_cnt_reg    <= dim_cnt_next;
        cycle_cnt_reg  <= cycle_cnt_next;
        
        // B矩阵数据流水线（1周期延迟）
        if (b_valid) begin
            b_data_reg  <= $signed(b_data);   // ⚠️ 显式有符号转换
            b_valid_reg <= 1'b1;
            b_dim_reg   <= b_row_addr;
        end else begin
            b_valid_reg <= 1'b0;
        end
    end
end

//================================================================================
// Delta计算逻辑：delta_buffer[d] = B[d] * error_scalar （Q4.12）
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < DIM; i = i + 1) begin
            delta_buffer[i] <= 16'sd0;
        end
    end else begin
        // 仅当b_valid_reg=1时更新对应维度的delta
        if (b_valid_reg) begin
            // Q4.12 × Q4.12 = Q8.24 → 右移12bit → Q4.12
            delta_buffer[b_dim_reg] <= ($signed(b_data_reg) * $signed(error_scalar)) >>> 12;
        end
    end
end

//================================================================================
// 激活值锁存逻辑
// 从LOB返回的向量中提取各维度数据
//
// ⚠️ 修复：移除state_reg检查，只要act_valid=1就锁存
// ⚠️ 关键修复：写入 activation_buffer 时使用 $signed()，保证符号扩展正确
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < DIM; i = i + 1) begin
            activation_buffer[i] <= 16'sd0;
        end
    end else begin
        if (act_valid) begin
            // act_vector = {act[DIM-1], ..., act[1], act[0]}
            for (i = 0; i < DIM; i = i + 1) begin
                activation_buffer[i] <= $signed(act_vector[i*DATA_WIDTH +: DATA_WIDTH]);
            end
        end
    end
end

//================================================================================
// 状态机组合逻辑
//================================================================================
always @(*) begin
    // 默认保持
    state_next      = state_reg;
    token_cnt_next  = token_cnt_reg;
    dim_cnt_next    = dim_cnt_reg;
    cycle_cnt_next  = cycle_cnt_reg + 1;
    
    // 默认不使能
    b_rd_en         = 1'b0;
    b_row_addr      = {DIM_ADDR_WIDTH{1'b0}};
    b_col_addr      = 4'd0;
    
    act_rd_en       = 1'b0;
    act_token_addr  = {TOKEN_ADDR_WIDTH{1'b0}};
    
    grad_wr_en      = 1'b0;
    grad_token_addr = {TOKEN_ADDR_WIDTH{1'b0}};
    grad_dim_addr   = {DIM_ADDR_WIDTH{1'b0}};
    grad_data       = {DATA_WIDTH{1'b0}};
    
    case (state_reg)
        //======================================================================
        // IDLE: 等待启动
        //======================================================================
        IDLE: begin
            cycle_cnt_next = 32'd0;
            if (start) begin
                state_next     = COMPUTE_DELTA;
                token_cnt_next = {TOKEN_ADDR_WIDTH{1'b0}};
                dim_cnt_next   = {DIM_ADDR_WIDTH{1'b0}};
            end
        end
        
        //======================================================================
        // COMPUTE_DELTA: 读取B矩阵，计算delta向量
        //======================================================================
        COMPUTE_DELTA: begin
            // 请求读取B[dim_cnt_reg][0]
            b_rd_en    = 1'b1;
            b_row_addr = dim_cnt_reg;
            b_col_addr = 4'd0;
            
            // 维度计数：0..DIM-1
            if (dim_cnt_reg == (DIM-1)) begin
                dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                state_next   = WAIT_DELTA;
            end else begin
                dim_cnt_next = dim_cnt_reg + 1'b1;
            end
        end
        
        //======================================================================
        // WAIT_DELTA: 为delta计算预留一个流水延迟
        //======================================================================
        WAIT_DELTA: begin
            // 这里可以根据乘法器流水深度增加等待周期
            // 当前版本假设1周期延迟足够
            state_next = COMPUTE_GRAD;
        end
        
        //======================================================================
        // COMPUTE_GRAD: 请求激活值
        //======================================================================
        COMPUTE_GRAD: begin
            act_rd_en       = 1'b1;
            act_token_addr  = token_cnt_reg;
            state_next      = WAIT_ACT;
        end
        
        //======================================================================
        // WAIT_ACT: 等待激活值有效
        //======================================================================
        WAIT_ACT: begin
            if (act_valid) begin
                dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                state_next   = WRITE_GRAD;
            end
        end
        
        //======================================================================
        // WRITE_GRAD: 写出当前token的所有维度梯度
        //======================================================================
        WRITE_GRAD: begin
            grad_wr_en      = 1'b1;
            grad_token_addr = token_cnt_reg;
            grad_dim_addr   = dim_cnt_reg;
            
            // HW计算：grad = delta[d] * act[t][d]  (Q4.12 × Q4.12 → Q4.12)
            grad_data = ($signed(delta_buffer[dim_cnt_reg]) * 
                         $signed(activation_buffer[dim_cnt_reg])) >>> 12;
            
            if (dim_cnt_reg == (DIM-1)) begin
                dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                
                if (token_cnt_reg == (NUM_TOKENS-1)) begin
                    state_next     = DONE_STATE;
                end else begin
                    token_cnt_next = token_cnt_reg + 1'b1;
                    state_next     = COMPUTE_GRAD;
                end
            end else begin
                dim_cnt_next = dim_cnt_reg + 1'b1;
            end
        end
        
        //======================================================================
        // DONE_STATE: 所有token梯度已写完
        //======================================================================
        DONE_STATE: begin
            state_next = DONE_STATE;
        end
        
        default: begin
            state_next = IDLE;
        end
    endcase
end

//================================================================================
// 输出信号映射
//================================================================================
assign state        = state_reg;
assign cycle_count  = cycle_cnt_reg;
assign token_count  = token_cnt_reg;
assign busy         = (state_reg != IDLE) && (state_reg != DONE_STATE);

//================================================================================
// done 信号：在DONE_STATE拉高一个周期
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        done <= 1'b0;
    end else begin
        if (state_reg == DONE_STATE && state_next == DONE_STATE) begin
            done <= 1'b1;
        end else if (state_reg == DONE_STATE && state_next != DONE_STATE) begin
            done <= 1'b0;
        end else if (state_reg != DONE_STATE && state_next == DONE_STATE) begin
            done <= 1'b1;
        end else begin
            done <= 1'b0;
        end
    end
end

//================================================================================
// 仿真调试：统计信息（仅在仿真中有效）
//================================================================================
`ifdef SIMULATION
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        total_cycles_sim       <= 0;
        tokens_processed_sim   <= 0;
        gradients_written_sim  <= 0;
    end else begin
        if (state_reg != IDLE) begin
            total_cycles_sim <= total_cycles_sim + 1;
        end
        
        if (grad_wr_en) begin
            gradients_written_sim <= gradients_written_sim + 1;
        end
        
        if (state_reg == WRITE_GRAD && dim_cnt_reg == (DIM-1)) begin
            tokens_processed_sim <= tokens_processed_sim + 1;
        end
    end
end

// 状态变迁时打印信息
always @(posedge clk) begin
    if (!rst_n) begin
        // do nothing
    end else begin
        if (state_reg != state_next) begin
            case (state_next)
                IDLE: begin
                    $display("[%0t] GCU[%0d] → IDLE", $time, LAYER_ID);
                    if (state_reg == DONE_STATE) begin
                        $display("    ╰─ Total: cycles=%0d, tokens=%0d, gradients=%0d", 
                                 total_cycles_sim, tokens_processed_sim, gradients_written_sim);
                    end
                end
                
                COMPUTE_DELTA: 
                    $display("[%0t] GCU[%0d] → COMPUTE_DELTA (computing B×error)", $time, LAYER_ID);
                
                WAIT_DELTA: 
                    $display("[%0t] GCU[%0d] → WAIT_DELTA (pipeline delay)", $time, LAYER_ID);
                
                COMPUTE_GRAD: 
                    $display("[%0t] GCU[%0d] → COMPUTE_GRAD (request LOB activations, token=%0d)", 
                             $time, LAYER_ID, token_cnt_next);
                
                WAIT_ACT:
                    $display("[%0t] GCU[%0d] → WAIT_ACT (waiting LOB valid)", $time, LAYER_ID);
                
                WRITE_GRAD:
                    $display("[%0t] GCU[%0d] → WRITE_GRAD (token=%0d)", 
                             $time, LAYER_ID, token_cnt_next);
                
                DONE_STATE:
                    $display("[%0t] GCU[%0d] → DONE_STATE (all gradients written)", $time, LAYER_ID);
            endcase
        end
    end
end
`endif

endmodule
