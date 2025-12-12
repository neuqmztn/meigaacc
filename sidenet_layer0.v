`timescale 1ns / 1ps

module sidenet_layer0 #(
    // ========== Token参数 ==========
    parameter TOKEN_NUM       = 640,
    parameter TOKEN_BATCH     = 32,
    parameter BATCH_NUM       = 20,
    
    // ========== 维度参数 ==========
    parameter BACKBONE_DIM    = 32,        
    parameter SIDENET_DIM     = 8,        
    
    // ========== 数据格式参数 ==========
    parameter DATA_WIDTH      = 16,        
    parameter EXP_WIDTH       = 8,         
    parameter ADDR_WIDTH      = 10      
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire start,                     
    output wire done,                    
    output wire busy,                     
    output reg  error,                    
    
    //================================================================================
    // Backbone输出读取接口（从Layer Token Buffer）
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
// 状态机定义
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
// 模块实例化
//================================================================================

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
    .token_rd_en(backbone_rd_en),
    .token_rd_addr(backbone_rd_addr),
    .token_rd_exp(backbone_rd_exp),
    .token_rd_mant(backbone_rd_mant),
    .token_rd_valid(backbone_rd_valid),
    
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