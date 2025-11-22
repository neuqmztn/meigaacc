`timescale 1ns / 1ps

//================================================================================
// Gradient Compute Unit (GCU) v3.0 - 向量化接口版本
//
// 主要改进（v3.0）：
// ✅ 适配LOB v4.2的向量化读取接口
// ✅ 从逐维度读取 → 按token读取整个向量
// ✅ 性能提升：641次读取 vs 20,512次读取（32倍提速）
// ✅ 适配二分类系统（error是标量）
// ✅ 保留内部Delta计算（并行性优化）
//
// 功能说明：
// 计算单层的DFA梯度：grad = B · e · a
//
// 详细公式（二分类）：
// 对每个token t，每个维度d：
//   1. delta[d] = B[d][0] × error（error是标量）
//   2. gradient[t][d] = delta[d] × activation[t][d]
//
// 流水线设计（5级）：
//   Stage 1: 读取激活向量（从LOB）
//   Stage 2: 计算Delta（B × error）
//   Stage 3: 向量化乘法（delta × activation）
//   Stage 4: 写回梯度
//   Stage 5: 完成
//
// 参数化：
//   LAYER_ID: 0-4
//   DIM: 8 (Layer 0-3) or 32 (Layer 4)
//
// 作者：MEIGA Team
// 日期：2025-11-19
// 版本：v3.0 (向量化 + 二分类)
//================================================================================

module gradient_compute_unit #(
    parameter LAYER_ID          = 0,            // 层ID (0-4)
    parameter DIM               = 8,            // 维度 (8 or 32)
    parameter NUM_CLASSES       = 1,            // 二分类 = 1
    parameter NUM_TOKENS        = 641,          // Token数量
    parameter DATA_WIDTH        = 16,           // Q4.12格式
    parameter TOKEN_ADDR_WIDTH  = 10,           // log2(641) = 10
    parameter DIM_ADDR_WIDTH    = 5             // log2(32) = 5
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire        start,                   // 开始计算（脉冲）
    output reg         done,                    // 计算完成
    output reg         busy,                    // 计算中
    
    //==========================================================================
    // B矩阵读取接口
    //==========================================================================
    output reg         b_rd_en,                 // 读使能
    output reg  [DIM_ADDR_WIDTH-1:0] b_row_addr, // 行地址
    output reg  [3:0]  b_col_addr,              // 列地址 (固定为0，二分类)
    input  wire [DATA_WIDTH-1:0] b_data,        // B矩阵元素
    input  wire        b_valid,                 // 数据有效
    
    //==========================================================================
    // 误差输入（标量，二分类）
    //==========================================================================
    input  wire [DATA_WIDTH-1:0] error_scalar,  // BCE误差（单个Q4.12值）
    
    //==========================================================================
    // 激活值读取接口 - 向量化（从LOB v4.2）
    //==========================================================================
    output reg         act_rd_en,               // 读使能
    output reg  [TOKEN_ADDR_WIDTH-1:0] act_token_addr, // Token地址
    input  wire [DIM*DATA_WIDTH-1:0] act_vector,       // 整个激活向量
    input  wire        act_valid,               // 数据有效
    
    //==========================================================================
    // 梯度写出接口 - 向量化
    //==========================================================================
    output reg         grad_wr_en,              // 写使能
    output reg  [TOKEN_ADDR_WIDTH-1:0] grad_token_addr, // Token地址
    output reg  [DIM_ADDR_WIDTH-1:0] grad_dim_addr,     // 维度地址
    output reg  [DATA_WIDTH-1:0] grad_data,     // 梯度值（单个）
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [2:0]  state,                   // 当前状态
    output wire [31:0] cycle_count,             // 周期计数
    output wire [31:0] token_count              // Token计数
);

//================================================================================
// 状态定义
//================================================================================
localparam IDLE         = 3'd0;
localparam COMPUTE_DELTA= 3'd1;  // 计算delta = B·e（向量）
localparam WAIT_DELTA   = 3'd2;  // 等待delta计算完成
localparam COMPUTE_GRAD = 3'd3;  // 计算grad = delta·a（641个token）
localparam WAIT_ACT     = 3'd4;  // 等待激活值
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
// Delta缓存（每个维度一个）
//================================================================================
reg signed [DATA_WIDTH-1:0] delta_buffer [0:DIM-1];

//================================================================================
// 激活值缓存
//================================================================================
reg [DATA_WIDTH-1:0] activation_buffer [0:DIM-1];

//================================================================================
// 流水线寄存器
//================================================================================
reg [DATA_WIDTH-1:0] b_data_reg;
reg                  b_valid_reg;
reg [DIM_ADDR_WIDTH-1:0] b_dim_reg;

//================================================================================
// 状态机寄存器
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg <= IDLE;
        token_cnt_reg <= {TOKEN_ADDR_WIDTH{1'b0}};
        dim_cnt_reg <= {DIM_ADDR_WIDTH{1'b0}};
        cycle_cnt_reg <= 32'd0;
        
        b_data_reg <= 16'd0;
        b_valid_reg <= 1'b0;
        b_dim_reg <= {DIM_ADDR_WIDTH{1'b0}};
    end else begin
        state_reg <= state_next;
        token_cnt_reg <= token_cnt_next;
        dim_cnt_reg <= dim_cnt_next;
        cycle_cnt_reg <= cycle_cnt_next;
        
        // B矩阵数据流水线
        if (b_valid) begin
            b_data_reg <= b_data;
            b_valid_reg <= 1'b1;
            b_dim_reg <= b_row_addr;
        end else begin
            b_valid_reg <= 1'b0;
        end
    end
end

//================================================================================
// Delta计算（向量化）
// delta[d] = B[d][0] × error（二分类，标量乘法）
//================================================================================
integer i;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < DIM; i = i + 1) begin
            delta_buffer[i] <= 16'sd0;
        end
    end else begin
        if (b_valid_reg) begin
            // Q4.12 × Q4.12 = Q8.24，右移12位得到Q4.12
            delta_buffer[b_dim_reg] <= 
                ($signed(b_data_reg) * $signed(error_scalar)) >>> 12;
        end
    end
end

//================================================================================
// 激活值锁存（从LOB向量中提取）
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < DIM; i = i + 1) begin
            activation_buffer[i] <= 16'd0;
        end
    end else begin
        if (state_reg == WAIT_ACT && act_valid) begin
            // 从向量中提取各维度
            for (i = 0; i < DIM; i = i + 1) begin
                activation_buffer[i] <= act_vector[i*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
end

//================================================================================
// 状态机组合逻辑
//================================================================================
always @(*) begin
    // 默认值
    state_next = state_reg;
    token_cnt_next = token_cnt_reg;
    dim_cnt_next = dim_cnt_reg;
    cycle_cnt_next = cycle_cnt_reg;
    
    done = 1'b0;
    busy = 1'b0;
    
    b_rd_en = 1'b0;
    b_row_addr = {DIM_ADDR_WIDTH{1'b0}};
    b_col_addr = 4'd0;  // 二分类：固定为0
    
    act_rd_en = 1'b0;
    act_token_addr = {TOKEN_ADDR_WIDTH{1'b0}};
    
    grad_wr_en = 1'b0;
    grad_token_addr = {TOKEN_ADDR_WIDTH{1'b0}};
    grad_dim_addr = {DIM_ADDR_WIDTH{1'b0}};
    grad_data = 16'd0;
    
    case (state_reg)
        //======================================================================
        // IDLE: 等待启动
        //======================================================================
        IDLE: begin
            if (start) begin
                state_next = COMPUTE_DELTA;
                dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                token_cnt_next = {TOKEN_ADDR_WIDTH{1'b0}};
                cycle_cnt_next = 32'd0;
            end
        end
        
        //======================================================================
        // COMPUTE_DELTA: 计算delta向量
        // 逐维度读取B矩阵，计算 delta[d] = B[d][0] × error
        //======================================================================
        COMPUTE_DELTA: begin
            busy = 1'b1;
            cycle_cnt_next = cycle_cnt_reg + 32'd1;
            
            // 读取B矩阵
            b_rd_en = 1'b1;
            b_row_addr = dim_cnt_reg;
            b_col_addr = 4'd0;  // 二分类：只读第0列
            
            if (dim_cnt_reg < DIM[DIM_ADDR_WIDTH-1:0] - 1) begin
                dim_cnt_next = dim_cnt_reg + 1;
            end else begin
                state_next = WAIT_DELTA;
            end
        end
        
        //======================================================================
        // WAIT_DELTA: 等待delta计算完成（流水线延迟）
        //======================================================================
        WAIT_DELTA: begin
            busy = 1'b1;
            cycle_cnt_next = cycle_cnt_reg + 32'd1;
            
            // 等待2个周期让delta计算完成
            if (cycle_cnt_reg >= 32'd2) begin
                state_next = COMPUTE_GRAD;
                token_cnt_next = {TOKEN_ADDR_WIDTH{1'b0}};
                dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
            end
        end
        
        //======================================================================
        // COMPUTE_GRAD: 对每个token计算梯度
        // 发出激活值读取请求
        //======================================================================
        COMPUTE_GRAD: begin
            busy = 1'b1;
            cycle_cnt_next = cycle_cnt_reg + 32'd1;
            
            // 读取当前token的激活向量
            act_rd_en = 1'b1;
            act_token_addr = token_cnt_reg;
            
            state_next = WAIT_ACT;
        end
        
        //======================================================================
        // WAIT_ACT: 等待激活值返回
        //======================================================================
        WAIT_ACT: begin
            busy = 1'b1;
            
            if (act_valid) begin
                state_next = WRITE_GRAD;
                dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
            end
        end
        
        //======================================================================
        // WRITE_GRAD: 写回梯度（逐维度）
        // gradient[t][d] = delta[d] × activation[t][d]
        //======================================================================
        WRITE_GRAD: begin
            busy = 1'b1;
            cycle_cnt_next = cycle_cnt_reg + 32'd1;
            
            // 计算并写入当前维度的梯度
            grad_wr_en = 1'b1;
            grad_token_addr = token_cnt_reg;
            grad_dim_addr = dim_cnt_reg;
            
            // Q4.12 × Q4.12 = Q8.24，右移12位得到Q4.12
            grad_data = ($signed(delta_buffer[dim_cnt_reg]) * 
                        $signed(activation_buffer[dim_cnt_reg])) >>> 12;
            
            // 维度递增
            if (dim_cnt_reg < DIM[DIM_ADDR_WIDTH-1:0] - 1) begin
                dim_cnt_next = dim_cnt_reg + 1;
            end else begin
                // 当前token的所有维度写完
                dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                
                if (token_cnt_reg < NUM_TOKENS[TOKEN_ADDR_WIDTH-1:0] - 1) begin
                    // 移动到下一个token
                    token_cnt_next = token_cnt_reg + 1;
                    state_next = COMPUTE_GRAD;
                end else begin
                    // 所有token处理完成
                    state_next = DONE_STATE;
                end
            end
        end
        
        //======================================================================
        // DONE_STATE: 完成
        //======================================================================
        DONE_STATE: begin
            done = 1'b1;
            state_next = IDLE;
        end
        
        default: state_next = IDLE;
    endcase
end

//================================================================================
// 调试输出
//================================================================================
assign state = state_reg;
assign cycle_count = cycle_cnt_reg;
assign token_count = token_cnt_reg;

//================================================================================
// 性能监控
//================================================================================
`ifdef SIMULATION
reg [31:0] total_cycles;
reg [31:0] tokens_processed;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        total_cycles <= 32'd0;
        tokens_processed <= 32'd0;
    end else begin
        if (busy) begin
            total_cycles <= total_cycles + 1;
        end
        
        if (state_reg == WRITE_GRAD && dim_cnt_reg == DIM-1 && 
            (token_cnt_reg < NUM_TOKENS-1 || state_next == DONE_STATE)) begin
            tokens_processed <= tokens_processed + 1;
        end
    end
end

always @(posedge clk) begin
    if (state_reg != state_next) begin
        case (state_next)
            IDLE:          $display("[%0t] GCU[%0d] → IDLE", $time, LAYER_ID);
            COMPUTE_DELTA: $display("[%0t] GCU[%0d] → COMPUTE_DELTA", $time, LAYER_ID);
            WAIT_DELTA:    $display("[%0t] GCU[%0d] → WAIT_DELTA", $time, LAYER_ID);
            COMPUTE_GRAD:  $display("[%0t] GCU[%0d] → COMPUTE_GRAD (token=%0d)", 
                                   $time, LAYER_ID, token_cnt_reg);
            DONE_STATE:    $display("[%0t] GCU[%0d] → DONE (cycles=%0d, tokens=%0d)", 
                                   $time, LAYER_ID, total_cycles, tokens_processed);
        endcase
    end
end
`endif

endmodule