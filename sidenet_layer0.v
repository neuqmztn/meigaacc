`timescale 1ns / 1ps

//================================================================================
// SideNet Layer 0 - 初始层模块（简化版）
//
// 功能：
// Layer 0是SideNet的初始层，只执行压缩操作：
// 1. Compression：将backbone输出从32维压缩到8维，直接输出
//
// 数据流：
// Backbone输出 z_0 [641×32] 
//   → Compression Engine → Layer Output Buffer â_0 [641×8]
//
// 特殊性：
// - 最简化的处理，只有Compression
// - 不需要Gate和Adaptation
// - 为后续层提供初始压缩特征
//
//================================================================================

module sidenet_layer0 #(
    // ========== Token参数 ==========
    parameter TOKEN_NUM       = 641,
    parameter TOKEN_BATCH     = 32,
    parameter BATCH_NUM       = 21,
    
    // ========== 维度参数 ==========
    parameter BACKBONE_DIM    = 32,        // Backbone输出维度
    parameter SIDENET_DIM     = 8,         // SideNet压缩维度
    
    // ========== 数据格式参数 ==========
    parameter DATA_WIDTH      = 16,        // BFP尾数位宽
    parameter EXP_WIDTH       = 8,         // BFP指数位宽
    parameter ADDR_WIDTH      = 10         // 地址位宽
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire start,                     // 启动信号
    output wire done,                      // 完成信号
    output wire busy,                      // 忙碌标志
    output reg  error,                     // 错误标志
    
    //================================================================================
    // Backbone输出读取接口（从Layer Token Buffer）
    // 读取 z_0 [641×32]
    //================================================================================
    output wire backbone_rd_en,
    output wire [ADDR_WIDTH-1:0] backbone_rd_addr,
    input  wire [EXP_WIDTH-1:0] backbone_rd_exp,
    input  wire [BACKBONE_DIM*DATA_WIDTH-1:0] backbone_rd_mant,
    input  wire backbone_rd_valid,
    
    //================================================================================
    // Layer Output Buffer接口（直接写入Layer 0输出 â_0）
    //================================================================================
    output wire layer0_wr_en,
    output wire [ADDR_WIDTH-1:0] layer0_token_id,
    output wire [EXP_WIDTH-1:0] layer0_exp,
    output wire [SIDENET_DIM*DATA_WIDTH-1:0] layer0_mant,
    
    //================================================================================
    // 权重接口（只需要Compression权重）
    // ✅ 修改：支持每列独立共享指数（8个指数）
    //================================================================================
    output wire compress_weight_req,
    input  wire compress_weight_ready,
    input  wire [SIDENET_DIM*EXP_WIDTH-1:0] compress_weight_exp,  // 8×8 = 64位
    input  wire [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH-1:0] compress_weight_mant,
    
    //================================================================================
    // 调试接口
    //================================================================================
    output wire [3:0] dbg_state,
    output wire [31:0] dbg_compress_tokens
);

//================================================================================
// 状态机定义（简化版）
//================================================================================

localparam IDLE        = 4'd0;
localparam COMPRESS    = 4'd1;
localparam WAIT_COMP   = 4'd2;
localparam DONE_STATE  = 4'd3;

reg [3:0] state_reg;
reg [3:0] next_state;

//================================================================================
// 内部控制信号
//================================================================================

// Compression控制
reg compress_start;
wire compress_done;
wire compress_busy;

//================================================================================
// FSM控制逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_reg <= IDLE;
    end else begin
        state_reg <= next_state;
    end
end

always @(*) begin
    next_state = state_reg;
    
    case (state_reg)
        IDLE: begin
            if (start) next_state = COMPRESS;
        end
        
        COMPRESS: begin
            next_state = WAIT_COMP;
        end
        
        WAIT_COMP: begin
            if (compress_done) next_state = DONE_STATE;
        end
        
        DONE_STATE: begin
            next_state = IDLE;
        end
        
        default: next_state = IDLE;
    endcase
end

// 控制信号生成
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        compress_start <= 1'b0;
        error <= 1'b0;
    end else begin
        compress_start <= 1'b0;
        
        case (state_reg)
            COMPRESS: compress_start <= 1'b1;
        endcase
    end
end

assign done = (state_reg == DONE_STATE);
assign busy = (state_reg != IDLE) && (state_reg != DONE_STATE);
assign dbg_state = state_reg;

//================================================================================
// 模块实例化（只需要Compression Engine）
//================================================================================

// Compression Engine - 直接输出到Layer Output Buffer
sidenet_compression_engine #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .INPUT_DIM(BACKBONE_DIM),
    .OUTPUT_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_compression (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(compress_start),
    .done(compress_done),
    .busy(compress_busy),
    
    // Backbone读接口
    .input_rd_en(backbone_rd_en),
    .input_rd_addr(backbone_rd_addr),
    .input_rd_exp(backbone_rd_exp),
    .input_rd_mant(backbone_rd_mant),
    .input_rd_valid(backbone_rd_valid),
    
    // 权重接口
    .weight_req(compress_weight_req),
    .weight_ready(compress_weight_ready),
    .weight_exp(compress_weight_exp),
    .weight_mant(compress_weight_mant),
    
    // 直接输出到Layer Output Buffer（作为â_0）
    .result_wr_en(layer0_wr_en),
    .result_wr_addr(layer0_token_id),
    .result_wr_exp(layer0_exp),
    .result_wr_mant(layer0_mant),
    
    // 调试
    .dbg_state(),
    .dbg_token_count(dbg_compress_tokens)
);

endmodule