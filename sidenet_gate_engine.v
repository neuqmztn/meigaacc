`timescale 1ns / 1ps

//================================================================================
// Sidenet Gate Engine
//
// 功能：
// 实现MEIGA的门控融合机制（Algorithm 1）：
// - Layer 0: a_0 = ẑ_0 (直通模式)
// - Layer i>0: a_i = (1-γ_i)·ẑ_i + γ_i·â_(i-1)
//   其中 γ_i = sigmoid(α_i/T)，α_i是可学习参数
//
// 数据格式：16-bit BFP
// - 输入ẑ_i：641 tokens × 8 dim
// - 输入â_(i-1)：641 tokens × 8 dim  
// - 输出a_i：641 tokens × 8 dim
//
// 实现策略（简化版）：
// - γ_i预先计算并存储（避免实时sigmoid计算）
// - 使用定点乘法和BFP加法器
// - 逐token处理
//
//================================================================================

module sidenet_gate_engine #(
    parameter TOKEN_NUM      = 641,       // Token数量
    parameter COMPRESSED_DIM = 8,         // 压缩维度
    parameter DATA_WIDTH     = 16,        // 16-bit尾数
    parameter EXP_WIDTH      = 8,         // 8-bit指数
    parameter ADDR_WIDTH     = 10,        // 地址位宽
    parameter NUM_LAYERS     = 4          // Sidenet层数
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire start,
    input  wire is_layer0,                // Layer 0直通模式标志
    input  wire [1:0] layer_id,           // 当前层ID（用于选择γ参数）
    output reg  done,
    output reg  busy,
    
    //==========================================================================
    // Compressed Buffer读接口 - 读取ẑ_i
    //==========================================================================
    output reg  compressed_rd_en,
    output reg  [ADDR_WIDTH-1:0] compressed_rd_addr,
    input  wire [EXP_WIDTH-1:0] compressed_rd_exp,
    input  wire [COMPRESSED_DIM*DATA_WIDTH-1:0] compressed_rd_mant,
    input  wire compressed_rd_valid,
    
    //==========================================================================
    // Adapted Buffer读接口 - 读取â_(i-1) (仅Layer i>0)
    //==========================================================================
    output reg  adapted_rd_en,
    output reg  [ADDR_WIDTH-1:0] adapted_rd_addr,
    input  wire [EXP_WIDTH-1:0] adapted_rd_exp,
    input  wire [COMPRESSED_DIM*DATA_WIDTH-1:0] adapted_rd_mant,
    input  wire adapted_rd_valid,
    
    //==========================================================================
    // Gated Buffer写接口 - 写入a_i
    //==========================================================================
    output reg  gated_wr_en,
    output reg  [ADDR_WIDTH-1:0] gated_wr_addr,
    output reg  [EXP_WIDTH-1:0] gated_wr_exp,
    output reg  [COMPRESSED_DIM*DATA_WIDTH-1:0] gated_wr_mant,
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [3:0] dbg_state,
    output reg  [31:0] dbg_token_count
);

//================================================================================
// 参数定义 - 门控系数γ_i（预计算的sigmoid值）
// γ_i = sigmoid(α_i/T)，存储为16-bit定点数（0.16格式，范围0-1）
//================================================================================

// 示例值（实际应该从训练中获得）
// 格式：16'h8000 = 0.5, 16'hC000 = 0.75, 16'h4000 = 0.25
reg [15:0] gamma_params [0:NUM_LAYERS-1];

initial begin
    gamma_params[0] = 16'h0000;  // Layer 0: γ=0（不使用，直通）
    gamma_params[1] = 16'h4000;  // Layer 1: γ=0.25
    gamma_params[2] = 16'h6000;  // Layer 2: γ=0.375
    gamma_params[3] = 16'h8000;  // Layer 3: γ=0.5
end

//================================================================================
// 状态机定义
//================================================================================

localparam IDLE           = 4'd0;
localparam LOAD_COMPRESSED = 4'd1;
localparam WAIT_COMPRESSED = 4'd2;
localparam LOAD_ADAPTED   = 4'd3;
localparam WAIT_ADAPTED   = 4'd4;
localparam COMPUTE_GATE   = 4'd5;
localparam WRITE_RESULT   = 4'd6;
localparam NEXT_TOKEN     = 4'd7;
localparam DONE_STATE     = 4'd8;

// Layer 0特殊路径（直通）
localparam BYPASS_MODE    = 4'd9;

reg [3:0] state, next_state;

//================================================================================
// 内部信号
//================================================================================

// Token计数
reg [9:0] token_counter;

// 数据缓存
reg [EXP_WIDTH-1:0] compressed_exp_buf;
reg [COMPRESSED_DIM*DATA_WIDTH-1:0] compressed_mant_buf;

reg [EXP_WIDTH-1:0] adapted_exp_buf;
reg [COMPRESSED_DIM*DATA_WIDTH-1:0] adapted_mant_buf;

// 门控系数
reg [15:0] gamma;        // 当前层的γ值
reg [15:0] one_minus_gamma;  // 1-γ

// 加权结果
reg [EXP_WIDTH-1:0] weighted_compressed_exp;
reg [COMPRESSED_DIM*DATA_WIDTH-1:0] weighted_compressed_mant;

reg [EXP_WIDTH-1:0] weighted_adapted_exp;
reg [COMPRESSED_DIM*DATA_WIDTH-1:0] weighted_adapted_mant;

// BFP加法器接口
reg bfp_add_enable;
wire [EXP_WIDTH-1:0] bfp_add_exp_out;
wire [COMPRESSED_DIM*DATA_WIDTH-1:0] bfp_add_mant_out;
reg [2:0] dim_counter;  // 处理每个维度

//================================================================================
// γ参数加载
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        gamma <= 16'd0;
        one_minus_gamma <= 16'hFFFF;  // 1.0 (16.16格式)
    end else if (start) begin
        gamma <= gamma_params[layer_id];
        // 计算1-γ（简化：假设γ < 1.0）
        one_minus_gamma <= 16'hFFFF - gamma_params[layer_id];
    end
end

//================================================================================
// 状态转移
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

//================================================================================
// 下一状态逻辑
//================================================================================

always @(*) begin
    next_state = state;
    
    case (state)
        IDLE: begin
            if (start) begin
                if (is_layer0) begin
                    next_state = BYPASS_MODE;  // Layer 0直通
                end else begin
                    next_state = LOAD_COMPRESSED;
                end
            end
        end
        
        // Layer 0直通模式
        BYPASS_MODE: begin
            if (compressed_rd_valid) begin
                next_state = WRITE_RESULT;
            end
        end
        
        // 正常门控模式（Layer i>0）
        LOAD_COMPRESSED: begin
            next_state = WAIT_COMPRESSED;
        end
        
        WAIT_COMPRESSED: begin
            if (compressed_rd_valid) begin
                next_state = LOAD_ADAPTED;
            end
        end
        
        LOAD_ADAPTED: begin
            next_state = WAIT_ADAPTED;
        end
        
        WAIT_ADAPTED: begin
            if (adapted_rd_valid) begin
                next_state = COMPUTE_GATE;
            end
        end
        
        COMPUTE_GATE: begin
            // 计算完成后写入
            next_state = WRITE_RESULT;
        end
        
        WRITE_RESULT: begin
            next_state = NEXT_TOKEN;
        end
        
        NEXT_TOKEN: begin
            if (token_counter == TOKEN_NUM - 1) begin
                next_state = DONE_STATE;
            end else begin
                if (is_layer0) begin
                    next_state = BYPASS_MODE;
                end else begin
                    next_state = LOAD_COMPRESSED;
                end
            end
        end
        
        DONE_STATE: begin
            next_state = IDLE;
        end
        
        default: begin
            next_state = IDLE;
        end
    endcase
end

//================================================================================
// 输出逻辑和控制
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        done <= 1'b0;
        busy <= 1'b0;
        
        compressed_rd_en <= 1'b0;
        compressed_rd_addr <= {ADDR_WIDTH{1'b0}};
        
        adapted_rd_en <= 1'b0;
        adapted_rd_addr <= {ADDR_WIDTH{1'b0}};
        
        gated_wr_en <= 1'b0;
        gated_wr_addr <= {ADDR_WIDTH{1'b0}};
        gated_wr_exp <= {EXP_WIDTH{1'b0}};
        gated_wr_mant <= {COMPRESSED_DIM*DATA_WIDTH{1'b0}};
        
        token_counter <= 10'd0;
        dim_counter <= 3'd0;
        
        compressed_exp_buf <= {EXP_WIDTH{1'b0}};
        compressed_mant_buf <= {COMPRESSED_DIM*DATA_WIDTH{1'b0}};
        adapted_exp_buf <= {EXP_WIDTH{1'b0}};
        adapted_mant_buf <= {COMPRESSED_DIM*DATA_WIDTH{1'b0}};
        
        bfp_add_enable <= 1'b0;
        dbg_token_count <= 32'd0;
        
    end else begin
        // 默认值
        compressed_rd_en <= 1'b0;
        adapted_rd_en <= 1'b0;
        gated_wr_en <= 1'b0;
        bfp_add_enable <= 1'b0;
        done <= 1'b0;
        
        case (state)
            IDLE: begin
                busy <= 1'b0;
                token_counter <= 10'd0;
                dbg_token_count <= 32'd0;
            end
            
            // ========== Layer 0 直通模式 ==========
            BYPASS_MODE: begin
                busy <= 1'b1;
                compressed_rd_en <= 1'b1;
                compressed_rd_addr <= token_counter;
                
                if (compressed_rd_valid) begin
                    // 直接通过，无需计算
                    compressed_exp_buf <= compressed_rd_exp;
                    compressed_mant_buf <= compressed_rd_mant;
                end
            end
            
            // ========== 正常门控模式 ==========
            LOAD_COMPRESSED: begin
                busy <= 1'b1;
                compressed_rd_en <= 1'b1;
                compressed_rd_addr <= token_counter;
            end
            
            WAIT_COMPRESSED: begin
                if (compressed_rd_valid) begin
                    compressed_exp_buf <= compressed_rd_exp;
                    compressed_mant_buf <= compressed_rd_mant;
                end
            end
            
            LOAD_ADAPTED: begin
                adapted_rd_en <= 1'b1;
                adapted_rd_addr <= token_counter;
            end
            
            WAIT_ADAPTED: begin
                if (adapted_rd_valid) begin
                    adapted_exp_buf <= adapted_rd_exp;
                    adapted_mant_buf <= adapted_rd_mant;
                end
            end
            
            COMPUTE_GATE: begin
                // 简化实现：直接使用BFP加法器
                // a_i = (1-γ)·ẑ_i + γ·â_(i-1)
                // 这里简化为直接加权平均（实际需要定点乘法）
                
                // TODO: 实现正确的加权计算
                // 当前简化为直接相加
                bfp_add_enable <= 1'b1;
            end
            
            WRITE_RESULT: begin
                gated_wr_en <= 1'b1;
                gated_wr_addr <= token_counter;
                
                if (is_layer0) begin
                    // Layer 0: 直通
                    gated_wr_exp <= compressed_exp_buf;
                    gated_wr_mant <= compressed_mant_buf;
                end else begin
                    // Layer i>0: 门控融合结果
                    // 简化：取平均（TODO: 使用正确的加权）
                    gated_wr_exp <= bfp_add_exp_out;
                    gated_wr_mant <= bfp_add_mant_out;
                end
                
                dbg_token_count <= dbg_token_count + 1;
            end
            
            NEXT_TOKEN: begin
                token_counter <= token_counter + 1;
            end
            
            DONE_STATE: begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        endcase
    end
end

//================================================================================
// BFP加法器实例化（简化版，处理整个向量）
//================================================================================

// 注意：这里简化为单个BFP加法器
// 实际应该对每个维度分别处理

bfp_adder #(
    .EXP_WIDTH(EXP_WIDTH),
    .MANT_WIDTH(DATA_WIDTH)
) u_bfp_adder (
    .clk(clk),
    .rst_n(rst_n),
    
    .enable(bfp_add_enable),
    .flush(1'b0),
    
    // 输入A: (1-γ)·ẑ_i (简化：直接使用ẑ_i)
    .sign_a(1'b0),
    .exp_a(compressed_exp_buf),
    .mant_a(compressed_mant_buf[DATA_WIDTH-1:0]),  // 只处理第一个维度（简化）
    .zero_a(1'b0),
    
    // 输入B: γ·â_(i-1) (简化：直接使用â_(i-1))
    .sign_b(1'b0),
    .exp_b(adapted_exp_buf),
    .mant_b(adapted_mant_buf[DATA_WIDTH-1:0]),     // 只处理第一个维度（简化）
    .zero_b(1'b0),
    
    // 输出
    .sign_out(),
    .exp_out(bfp_add_exp_out),
    .mant_out(bfp_add_mant_out[DATA_WIDTH-1:0]),   // 只处理第一个维度（简化）
    .zero_out()
);

assign dbg_state = state;

//================================================================================
// 仿真信息输出
//================================================================================

`ifdef SIMULATION
always @(posedge clk) begin
    case (state)
        IDLE:         if (start) $display("[%0t] Gate Engine: Start (layer=%0d, is_layer0=%0b, γ=0x%h)", 
                                          $time, layer_id, is_layer0, gamma);
        BYPASS_MODE:  if (token_counter == 0) $display("[%0t] Gate Engine: BYPASS mode (Layer 0)", $time);
        COMPUTE_GATE: $display("[%0t] Gate Engine: Computing gate for token %0d", $time, token_counter);
        DONE_STATE:   $display("[%0t] Gate Engine: Done, processed %0d tokens", $time, dbg_token_count);
    endcase
end
`endif

//================================================================================
// TODO: 完整实现说明
//================================================================================

/*
完整实现需要：

1. 定点乘法器：
   - 计算(1-γ)·ẑ_i和γ·â_(i-1)
   - 输入：16-bit γ（0.16格式）× 16-bit尾数
   - 输出：16-bit加权尾数

2. 多维度处理：
   - 当前只处理第一个维度
   - 应该遍历所有8个维度
   - 可以串行处理或并行实例化8个加法器

3. 指数处理：
   - 加权后的指数调整
   - 考虑乘法带来的指数变化

4. 优化：
   - 批量处理多个token
   - 流水线设计
*/

endmodule