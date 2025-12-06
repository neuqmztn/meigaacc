`timescale 1ns / 1ps

module sidenet_layer4 #(
    // ========== Token参数 ==========
    parameter TOKEN_NUM       = 640,
    parameter TOKEN_BATCH     = 32,
    parameter BATCH_NUM       = 20,
    
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
    // 读取 z_4 [641×32]
    //================================================================================
    output wire backbone_rd_en,
    output wire [ADDR_WIDTH-1:0] backbone_rd_addr,
    input  wire [EXP_WIDTH-1:0] backbone_rd_exp,
    input  wire [BACKBONE_DIM*DATA_WIDTH-1:0] backbone_rd_mant,
    input  wire backbone_rd_valid,
    
    //================================================================================
    // Layer 3输出读取接口（从Layer Output Buffer）
    // 读取 â_3 [641×8] 用于Gate融合
    //================================================================================
    output wire layer3_rd_en,
    output wire [ADDR_WIDTH-1:0] layer3_token_id,
    input  wire [EXP_WIDTH-1:0] layer3_exp,
    input  wire [SIDENET_DIM*DATA_WIDTH-1:0] layer3_mant,
    input  wire layer3_valid,
    
    //================================================================================
    // 最终输出接口（写入Final Result Buffer）
    // 输出扩展后的32维特征
    //================================================================================
    output wire final_wr_en,
    output wire [ADDR_WIDTH-1:0] final_wr_addr,
    output wire [EXP_WIDTH-1:0] final_wr_exp,
    output wire [BACKBONE_DIM*DATA_WIDTH-1:0] final_wr_mant,
    
    //================================================================================
    // 权重接口
    // ✅ 修改：支持每列独立共享指数
    //================================================================================
    // Compression权重（8个指数）
    output wire compress_weight_req,
    input  wire compress_weight_ready,
    input  wire [SIDENET_DIM*EXP_WIDTH-1:0] compress_weight_exp,  // 8×8 = 64位
    input  wire [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH-1:0] compress_weight_mant,
    
    // Expand权重（32个指数）
    output wire expand_weight_req,
    input  wire expand_weight_ready,
    input  wire [BACKBONE_DIM*EXP_WIDTH-1:0] expand_weight_exp,  // 32×8 = 256位
    input  wire [SIDENET_DIM*BACKBONE_DIM*DATA_WIDTH-1:0] expand_weight_mant,
    
    //================================================================================
    // 调试接口
    //================================================================================
    output wire [3:0] dbg_state,
    output wire [31:0] dbg_compress_tokens,
    output wire [31:0] dbg_gate_tokens,
    output wire [31:0] dbg_expand_tokens
);

//================================================================================
// 状态机定义
//================================================================================

localparam IDLE        = 4'd0;
localparam COMPRESS    = 4'd1;
localparam WAIT_COMP   = 4'd2;
localparam GATE        = 4'd3;
localparam WAIT_GATE   = 4'd4;
localparam EXPAND      = 4'd5;
localparam WAIT_EXPAND = 4'd6;
localparam DONE_STATE  = 4'd7;

reg [3:0] state_reg;
reg [3:0] next_state;

//================================================================================
// 内部控制信号
//================================================================================

// Compression控制
reg compress_start;
wire compress_done;
wire compress_busy;

// Gate控制  
reg gate_start;
wire gate_done;
wire gate_busy;

// Expand控制
reg expand_start;
wire expand_done;
wire expand_busy;

//================================================================================
// 内部Buffer接口
//================================================================================

// Compressed Buffer接口
wire comp_buf_wr_en;
wire [ADDR_WIDTH-1:0] comp_buf_wr_addr;
wire [EXP_WIDTH-1:0] comp_buf_wr_exp;
wire [SIDENET_DIM*DATA_WIDTH-1:0] comp_buf_wr_mant;

wire comp_buf_rd_en;
wire [ADDR_WIDTH-1:0] comp_buf_rd_addr;
wire [EXP_WIDTH-1:0] comp_buf_rd_exp;
wire [SIDENET_DIM*DATA_WIDTH-1:0] comp_buf_rd_mant;
wire comp_buf_rd_valid;

// Gated Buffer接口
wire gate_buf_wr_en;
wire [ADDR_WIDTH-1:0] gate_buf_wr_addr;
wire [EXP_WIDTH-1:0] gate_buf_wr_exp;
wire [SIDENET_DIM*DATA_WIDTH-1:0] gate_buf_wr_mant;

wire gate_buf_rd_en;
wire [ADDR_WIDTH-1:0] gate_buf_rd_addr;
wire [EXP_WIDTH-1:0] gate_buf_rd_exp;
wire [SIDENET_DIM*DATA_WIDTH-1:0] gate_buf_rd_mant;
wire gate_buf_rd_valid;

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
            if (compress_done) next_state = GATE;
        end
        
        GATE: begin
            next_state = WAIT_GATE;
        end
        
        WAIT_GATE: begin
            if (gate_done) next_state = EXPAND;
        end
        
        EXPAND: begin
            next_state = WAIT_EXPAND;
        end
        
        WAIT_EXPAND: begin
            if (expand_done) next_state = DONE_STATE;
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
        gate_start <= 1'b0;
        expand_start <= 1'b0;
        error <= 1'b0;
    end else begin
        compress_start <= 1'b0;
        gate_start <= 1'b0;
        expand_start <= 1'b0;
        
        case (state_reg)
            COMPRESS: compress_start <= 1'b1;
            GATE: gate_start <= 1'b1;
            EXPAND: expand_start <= 1'b1;
        endcase
    end
end

assign done = (state_reg == DONE_STATE);
assign busy = (state_reg != IDLE) && (state_reg != DONE_STATE);
assign dbg_state = state_reg;

//================================================================================
// 模块实例化
//================================================================================

// 1. Compression Engine
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
    
    // 输出到Compressed Buffer
    .result_wr_en(comp_buf_wr_en),
    .result_wr_addr(comp_buf_wr_addr),
    .result_wr_exp(comp_buf_wr_exp),
    .result_wr_mant(comp_buf_wr_mant),
    
    // 调试
    .dbg_state(),
    .dbg_token_count(dbg_compress_tokens)
);

// 2. Compressed Buffer
sidenet_compressed_buffer #(
    .TOKEN_NUM(TOKEN_NUM),
    .COMPRESSED_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_compressed_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    .wr_en(comp_buf_wr_en),
    .wr_addr(comp_buf_wr_addr),
    .wr_exp(comp_buf_wr_exp),
    .wr_mant(comp_buf_wr_mant),
    .wr_ready(),
    
    .rd_en(comp_buf_rd_en),
    .rd_addr(comp_buf_rd_addr),
    .rd_exp(comp_buf_rd_exp),
    .rd_mant(comp_buf_rd_mant),
    .rd_valid(comp_buf_rd_valid)
);

// 3. Gate Engine (正常融合模式)
sidenet_gate_engine #(
    .TOKEN_NUM(TOKEN_NUM),
    .COMPRESSED_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .NUM_LAYERS(4)
) u_gate (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(gate_start),
    .is_layer0(1'b0),           // Layer 4不是Layer 0
    .layer_id(2'b11),           // Layer 4 (假设映射为3)
    .done(gate_done),
    .busy(gate_busy),
    
    // 从Compressed Buffer读取ẑ_4
    .compressed_rd_en(comp_buf_rd_en),
    .compressed_rd_addr(comp_buf_rd_addr),
    .compressed_rd_exp(comp_buf_rd_exp),
    .compressed_rd_mant(comp_buf_rd_mant),
    .compressed_rd_valid(comp_buf_rd_valid),
    
    // 从Layer Output Buffer读取â_3
    .adapted_rd_en(layer3_rd_en),
    .adapted_rd_addr(layer3_token_id),
    .adapted_rd_exp(layer3_exp),
    .adapted_rd_mant(layer3_mant),
    .adapted_rd_valid(layer3_valid),
    
    // 输出到Gated Buffer
    .gated_wr_en(gate_buf_wr_en),
    .gated_wr_addr(gate_buf_wr_addr),
    .gated_wr_exp(gate_buf_wr_exp),
    .gated_wr_mant(gate_buf_wr_mant),
    
    // 调试
    .dbg_state(),
    .dbg_token_count(dbg_gate_tokens)
);

// 4. Gated Buffer
sidenet_gated_buffer #(
    .TOKEN_NUM(TOKEN_NUM),
    .COMPRESSED_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_gated_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    .wr_en(gate_buf_wr_en),
    .wr_addr(gate_buf_wr_addr),
    .wr_exp(gate_buf_wr_exp),
    .wr_mant(gate_buf_wr_mant),
    .wr_ready(),
    
    .rd_en(gate_buf_rd_en),
    .rd_addr(gate_buf_rd_addr),
    .rd_exp(gate_buf_rd_exp),
    .rd_mant(gate_buf_rd_mant),
    .rd_valid(gate_buf_rd_valid)
);

// 5. Expand Engine (代替Adaptation)
sidenet_expand_engine #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .INPUT_DIM(SIDENET_DIM),
    .OUTPUT_DIM(BACKBONE_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .LAYER_WIDTH(2)
) u_expand (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(expand_start),
    .layer_id(2'b11),           // Layer 4
    .done(expand_done),
    .busy(expand_busy),
    
    // 直接从Gated Buffer读取（不是从Adapted Buffer）
    // 因为Layer 4的Gated Buffer输出直接进入Expand
    .adapted_rd_en(gate_buf_rd_en),
    .adapted_rd_layer(),        // 不使用，直接从Gated Buffer读
    .adapted_rd_addr(gate_buf_rd_addr),
    .adapted_rd_exp(gate_buf_rd_exp),
    .adapted_rd_mant(gate_buf_rd_mant),
    .adapted_rd_valid(gate_buf_rd_valid),
    
    // 权重接口
    .weight_req(expand_weight_req),
    .weight_ready(expand_weight_ready),
    .weight_exp(expand_weight_exp),
    .weight_mant(expand_weight_mant),
    
    // 输出到Final Result Buffer
    .result_wr_en(final_wr_en),
    .result_wr_addr(final_wr_addr),
    .result_wr_exp(final_wr_exp),
    .result_wr_mant(final_wr_mant),
    
    // 调试
    .dbg_state(),
    .dbg_token_count(dbg_expand_tokens)
);

endmodule