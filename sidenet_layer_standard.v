`timescale 1ns / 1ps

//================================================================================
// SideNet Layer Standard v2.1 - 标准层集成模块（Layer 1-3复用）
//
// 主要变更 (v2.1):
// - 删除FFN DRAM接口
// - 新增FFN权重接口（W1和W2）
// - FFN权重通过weight_controller统一管理
//
// 主要变更 (v2.0):
// - 删除KV Cache接口（内部封装在Attention模块中）
// - 删除LN1暂存接口（内部封装在Transformer模块中）
// - 修改Attention权重接口为数组格式（8个指数）
// - 新增独立的WO权重接口
// - 更新LayerNorm参数接口（ln_param_type编码）
//
// 功能：
// 实现SideNet标准层（Layer 1-3）的完整处理流程：
// Compression → Gate → Adaptation Transformer
//
// 数据流：
// Backbone输出 z_(i-1) [641×32] 
//   → Compression Engine → Compressed Buffer ẑ_i [641×8]
//   → Gate Engine (融合 â_(i-1)) → Gated Buffer a_i [641×8]
//   → Adaptation Transformer → Layer Output Buffer â_i [641×8]
//
// 时分复用：
// 本模块单个实例可被Layer 1、2、3复用，通过layer_id参数区分
//
// 架构特点：
// - 三阶段流水线：Compression → Gate → Adaptation
// - 内部临时Buffer：Compressed Buffer、Gated Buffer（单BANK复用）
// - 外部持久存储：Layer Output Buffer（4-BANK独立）
// - FSM控制：协调各阶段执行顺序
//
// 作者: MEIGA Team
// 日期: 2025-11-16
// 版本: v2.1
//================================================================================

module sidenet_layer_standard #(
    // ========== Token参数 ==========
    parameter TOKEN_NUM       = 640,
    parameter TOKEN_BATCH     = 32,
    parameter BATCH_NUM       = 20,
    
    // ========== 维度参数 ==========
    parameter BACKBONE_DIM    = 32,        // Backbone输出维度
    parameter SIDENET_DIM     = 8,         // SideNet压缩维度
    parameter NUM_HEADS       = 4,         // Attention头数
    parameter HEAD_DIM        = 2,         // 每个头的维度
    parameter D_FF            = 32,        // FFN隐藏层维度
    parameter FEATURE_CHUNK   = 8,         // FFN特征分块
    
    // ========== 数据格式参数 ==========
    parameter DATA_WIDTH      = 16,        // BFP尾数位宽
    parameter EXP_WIDTH       = 8,         // BFP指数位宽
    parameter ADDR_WIDTH      = 10,        // 地址位宽
    
    // ========== Attention参数 ==========
    parameter K_CHUNK_SIZE    = 32,
    parameter K_CHUNK_NUM     = 20,
    
    // ========== 层参数 ==========
    parameter NUM_LAYERS      = 4          // SideNet总层数
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire start,                     // 启动信号
    input  wire [1:0] layer_id,            // 当前层ID (1-3)
    output wire done,                      // 完成信号
    output wire busy,                      // 忙碌标志
    
    //================================================================================
    // Backbone输出读取接口（从Layer Token Buffer）
    // 读取 z_(i-1) [641×32]
    //================================================================================
    output wire backbone_rd_en,
    output wire [ADDR_WIDTH-1:0] backbone_rd_addr,
    input  wire [EXP_WIDTH-1:0] backbone_rd_exp,
    input  wire [BACKBONE_DIM*DATA_WIDTH-1:0] backbone_rd_mant,
    input  wire backbone_rd_valid,
    
    //================================================================================
    // Layer Output Buffer接口（读取前一层输出 â_(i-1)）
    // Gate Engine需要读取前一层的适配输出
    //================================================================================
    output wire prev_layer_rd_en,
    output wire [1:0] prev_layer_id,       // layer_id - 1
    output wire [ADDR_WIDTH-1:0] prev_layer_token_id,
    input  wire [EXP_WIDTH-1:0] prev_layer_exp,
    input  wire [SIDENET_DIM*DATA_WIDTH-1:0] prev_layer_mant,
    input  wire prev_layer_valid,
    
    //================================================================================
    // Layer Output Buffer接口（写入当前层输出 â_i）
    // Adaptation Transformer输出最终结果
    //================================================================================
    output wire curr_layer_wr_en,
    output wire [1:0] curr_layer_id,
    output wire [ADDR_WIDTH-1:0] curr_layer_token_id,
    output wire [EXP_WIDTH-1:0] curr_layer_exp,
    output wire [SIDENET_DIM*DATA_WIDTH-1:0] curr_layer_mant,
    
    //================================================================================
    // 权重接口（Compression权重）
    // 支持每列独立共享指数（8个指数）
    //================================================================================
    output wire compress_weight_req,
    input  wire compress_weight_ready,
    input  wire [SIDENET_DIM*EXP_WIDTH-1:0] compress_weight_exp,  // 8×8 = 64位
    input  wire [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH-1:0] compress_weight_mant,
    
    //================================================================================
    // 权重接口（QKV权重 - 修改 v2.0）
    // 改为数组格式，支持每列独立共享指数
    //================================================================================
    output wire qkv_weight_req,
    output wire [1:0] qkv_weight_type,     // 00=Q, 01=K, 10=V
    input  wire qkv_weight_ack,
    input  wire qkv_weight_valid,
    input  wire [SIDENET_DIM*EXP_WIDTH-1:0] qkv_weight_exp_array,        // 8×8 = 64位
    input  wire [SIDENET_DIM*SIDENET_DIM*DATA_WIDTH-1:0] qkv_weight_mant_blocks,
    
    //================================================================================
    // 权重接口（WO权重 - 新增 v2.0）
    // Output Projection专用接口
    //================================================================================
    output wire wo_weight_req,
    input  wire wo_weight_ready,
    input  wire [SIDENET_DIM*EXP_WIDTH-1:0] wo_weight_exp_array,          // 8×8 = 64位
    input  wire [SIDENET_DIM*SIDENET_DIM*DATA_WIDTH-1:0] wo_weight_mant,
    
    //================================================================================
    // FFN权重接口（新增 v2.1 - 通过weight_controller提供 W1/W2 完整矩阵）
    //================================================================================
    // Linear1权重接口
    output wire ffn_w1_weight_req,
    input  wire ffn_w1_weight_ready,
    input  wire [D_FF*EXP_WIDTH-1:0] ffn_w1_weight_exp_array,
    input  wire [SIDENET_DIM*D_FF*DATA_WIDTH-1:0] ffn_w1_weight_mant,
    
    // Linear2权重接口
    output wire ffn_w2_weight_req,
    input  wire ffn_w2_weight_ready,
    input  wire [SIDENET_DIM*EXP_WIDTH-1:0] ffn_w2_weight_exp_array,
    input  wire [D_FF*SIDENET_DIM*DATA_WIDTH-1:0] ffn_w2_weight_mant,
    
    //================================================================================
    // LayerNorm参数接口（修改 v2.0）
    // 使用统一的type编码区分4种参数
    //================================================================================
    output wire ln_param_req,
    output wire [1:0] ln_param_type,       // 00=LN1_gamma, 01=LN1_beta, 10=LN2_gamma, 11=LN2_beta
    input  wire [EXP_WIDTH-1:0] ln_param_exp,
    input  wire [SIDENET_DIM*DATA_WIDTH-1:0] ln_param_mant,
    input  wire ln_param_valid,
    
    //================================================================================
    // 调试接口
    //================================================================================
    output wire [3:0] dbg_fsm_state,
    output wire [3:0] dbg_compress_state,
    output wire [3:0] dbg_gate_state,
    output wire [3:0] dbg_transformer_state,
    output wire [3:0] dbg_att_state,
    output wire [31:0] dbg_cycle_count
);

//================================================================================
// 状态机定义
//================================================================================

localparam IDLE        = 4'd0;
localparam COMPRESSION = 4'd1;
localparam WAIT_COMP   = 4'd2;
localparam GATE        = 4'd3;
localparam WAIT_GATE   = 4'd4;
localparam ADAPTATION  = 4'd5;
localparam WAIT_ADAPT  = 4'd6;
localparam DONE_STATE  = 4'd7;

reg [3:0] state, next_state;
reg [31:0] cycle_counter;

//================================================================================
// 控制信号
//================================================================================

// Compression Engine
wire compress_start, compress_done, compress_busy;

// Gate Engine
wire gate_start, gate_done, gate_busy;
wire is_layer0;  // Layer 0标志（本模块不处理Layer 0，但保留接口）
// Adaptation Transformer
wire transformer_start, transformer_done, transformer_busy;

//================================================================================
// 内部Buffer信号
//================================================================================

// Compressed Buffer（Compression → Gate）
wire compressed_wr_en;
wire [ADDR_WIDTH-1:0] compressed_wr_addr;
wire [EXP_WIDTH-1:0] compressed_wr_exp;
wire [SIDENET_DIM*DATA_WIDTH-1:0] compressed_wr_mant;

wire compressed_rd_en;
wire [ADDR_WIDTH-1:0] compressed_rd_addr;
wire [EXP_WIDTH-1:0] compressed_rd_exp;
wire [SIDENET_DIM*DATA_WIDTH-1:0] compressed_rd_mant;
wire compressed_rd_valid;

// Gated Buffer（Gate → Adaptation）
wire gated_wr_en;
wire [ADDR_WIDTH-1:0] gated_wr_addr;
wire [EXP_WIDTH-1:0] gated_wr_exp;
wire [SIDENET_DIM*DATA_WIDTH-1:0] gated_wr_mant;

wire gated_rd_en;
wire [ADDR_WIDTH-1:0] gated_rd_addr;
wire [EXP_WIDTH-1:0] gated_rd_exp;
wire [SIDENET_DIM*DATA_WIDTH-1:0] gated_rd_mant;
wire gated_rd_valid;

//================================================================================
// FFN 权重适配器：将 weight_controller 提供的 W1/W2 完整矩阵
// 转换为 FFN 内核需要的按 chunk 的权重块接口
//================================================================================

// 传给 Adaptation Transformer / FFN 的统一权重接口
wire                        ffn_weight_req;
wire [1:0]                  ffn_weight_type;       // 00=W1, 01=W2
wire [1:0]                  ffn_weight_chunk_id;   // 0~3, 每块 FEATURE_CHUNK 列
wire                        ffn_weight_ready;
wire [SIDENET_DIM*EXP_WIDTH-1:0]           ffn_weight_exp_array;
wire [SIDENET_DIM*FEATURE_CHUNK*DATA_WIDTH-1:0] ffn_weight_mant;

// 内部寄存版本，便于组合逻辑赋值
reg  [SIDENET_DIM*EXP_WIDTH-1:0]           ffn_weight_exp_array_reg;
reg  [SIDENET_DIM*FEATURE_CHUNK*DATA_WIDTH-1:0] ffn_weight_mant_reg;

assign ffn_weight_exp_array = ffn_weight_exp_array_reg;
assign ffn_weight_mant      = ffn_weight_mant_reg;

// 将统一接口拆分为 W1 / W2 请求，连到 weight_controller
assign ffn_w1_weight_req = ffn_weight_req && (ffn_weight_type == 2'd0);
assign ffn_w2_weight_req = ffn_weight_req && (ffn_weight_type == 2'd1);

// 将 W1 / W2 ready 合成一个统一的 ready
assign ffn_weight_ready =
    ((ffn_weight_type == 2'd0) && ffn_w1_weight_ready) ||
    ((ffn_weight_type == 2'd1) && ffn_w2_weight_ready);

// W1 / W2 chunk 选择常数
localparam integer FFN_W1_CHUNK_MANT_BITS = SIDENET_DIM * FEATURE_CHUNK * DATA_WIDTH;

// 组合逻辑：根据 type / chunk_id，从 W1/W2 完整矩阵中抽取对应 8×8 block
integer ffn_row_idx;
integer ffn_feat_idx;
integer ffn_src_exp_idx;
integer ffn_src_row_idx;
integer ffn_dest_index;
integer ffn_src_index;
integer base_mant_bit;

always @(*) begin
    // 默认清零
    ffn_weight_exp_array_reg = {SIDENET_DIM*EXP_WIDTH{1'b0}};
    ffn_weight_mant_reg      = {SIDENET_DIM*FEATURE_CHUNK*DATA_WIDTH{1'b0}};
    base_mant_bit            = 0;
    ffn_src_exp_idx          = 0;
    ffn_src_row_idx          = 0;
    ffn_dest_index           = 0;
    ffn_src_index            = 0;

    case (ffn_weight_type)
        //==============================================================
        // W1: 8×32
        //  - 指数：32 个，按列组织，这里按 chunk 切成 4 组，每组 8 个
        //  - 尾数：按列优先展开，每列 8 个元素
        //==============================================================
        2'd0: begin
            // 指数：第 chunk_id 组的 8 个指数
            for (ffn_row_idx = 0; ffn_row_idx < SIDENET_DIM; ffn_row_idx = ffn_row_idx + 1) begin
                ffn_src_exp_idx = ffn_weight_chunk_id*SIDENET_DIM + ffn_row_idx;   // 0~31
                ffn_weight_exp_array_reg[ffn_row_idx*EXP_WIDTH +: EXP_WIDTH] =
                    ffn_w1_weight_exp_array[ffn_src_exp_idx*EXP_WIDTH +: EXP_WIDTH];
            end

            // 尾数：整块切片 (一个 chunk 是连续 8×8=64 个权重)
            base_mant_bit = ffn_weight_chunk_id * FFN_W1_CHUNK_MANT_BITS;
            ffn_weight_mant_reg =
                ffn_w1_weight_mant[base_mant_bit +: FFN_W1_CHUNK_MANT_BITS];
        end

        //==============================================================
        // W2: 32×8
        //  - 指数：8 个，对应 8 个输出维度（列）
        //  - 尾数：按列优先展开，每列 32 个元素
        //  - chunk 沿着 32 维隐藏层方向划分，每块 8 行
        //==============================================================
        2'd1: begin
            // 指数：对所有 chunk 相同，直接透传
            ffn_weight_exp_array_reg = ffn_w2_weight_exp_array;

            // 尾数：为当前 chunk_id / feature 内的每一个 (feature, dim) 选取对应元素
            for (ffn_feat_idx = 0; ffn_feat_idx < FEATURE_CHUNK; ffn_feat_idx = ffn_feat_idx + 1) begin
                ffn_src_row_idx = ffn_weight_chunk_id*FEATURE_CHUNK + ffn_feat_idx; // 0~31

                for (ffn_row_idx = 0; ffn_row_idx < SIDENET_DIM; ffn_row_idx = ffn_row_idx + 1) begin
                    // 目标下标：feature 在外层，dim 在内层
                    ffn_dest_index = (ffn_feat_idx*SIDENET_DIM + ffn_row_idx)*DATA_WIDTH;

                    // 源下标：W2 按列优先展开 -> 索引 = 列*32 + 行
                    ffn_src_index  = (ffn_row_idx*D_FF + ffn_src_row_idx)*DATA_WIDTH;

                    ffn_weight_mant_reg[ffn_dest_index +: DATA_WIDTH] =
                        ffn_w2_weight_mant[ffn_src_index +: DATA_WIDTH];
                end
            end
        end

        default: begin
            // 保持默认 0
        end
    endcase
end

//================================================================================
// 状态机：协调三个阶段的执行
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        cycle_counter <= 32'd0;
    end else begin
        state <= next_state;
        if (state != IDLE) begin
            cycle_counter <= cycle_counter + 1;
        end else begin
            cycle_counter <= 32'd0;
        end
    end
end

always @(*) begin
    case (state)
        IDLE: begin
            if (start) begin
                next_state = COMPRESSION;
            end else begin
                next_state = IDLE;
            end
        end
        
        COMPRESSION: begin
            next_state = WAIT_COMP;
        end
        
        WAIT_COMP: begin
            if (compress_done) begin
                next_state = GATE;
            end else begin
                next_state = WAIT_COMP;
            end
        end
        
        GATE: begin
            next_state = WAIT_GATE;
        end
        
        WAIT_GATE: begin
            if (gate_done) begin
                next_state = ADAPTATION;
            end else begin
                next_state = WAIT_GATE;
            end
        end
        
        ADAPTATION: begin
            next_state = WAIT_ADAPT;
        end
        
        WAIT_ADAPT: begin
            if (transformer_done) begin
                next_state = DONE_STATE;
            end else begin
                next_state = WAIT_ADAPT;
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

// 子模块启动信号生成
assign compress_start    = (state == COMPRESSION);
assign gate_start        = (state == GATE);
assign transformer_start = (state == ADAPTATION);

// 顶层控制输出
assign done = (state == DONE_STATE);
assign busy = (state != IDLE) && (state != DONE_STATE);

// Layer 0标志（本模块只处理Layer 1-3，Layer 0在专用模块中处理）
assign is_layer0 = 1'b0;

// Layer ID传递
assign curr_layer_id = layer_id;
assign prev_layer_id = layer_id - 2'b01;  // 前一层ID

//================================================================================
// 模块实例化
//================================================================================

//========== 1. Compression Engine ==========
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
    
    // 控制
    .start(compress_start),
    .done(compress_done),
    .busy(compress_busy),
    
    // Token输入：读取Backbone输出
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
    
    // 结果输出：写入Compressed Buffer
    .result_wr_en(compressed_wr_en),
    .result_wr_addr(compressed_wr_addr),
    .result_wr_exp(compressed_wr_exp),
    .result_wr_mant(compressed_wr_mant),
    
    // 调试
    .dbg_state(dbg_compress_state),
    .dbg_token_count()
);

//========== 2. Compressed Buffer ==========
sidenet_compressed_buffer #(
    .TOKEN_NUM(TOKEN_NUM),
    .COMPRESSED_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_compressed_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    // 写接口：来自Compression Engine
    .wr_en(compressed_wr_en),
    .wr_addr(compressed_wr_addr),
    .wr_exp(compressed_wr_exp),
    .wr_mant(compressed_wr_mant),
    .wr_ready(),
    
    // 读接口：供Gate Engine读取
    .rd_en(compressed_rd_en),
    .rd_addr(compressed_rd_addr),
    .rd_exp(compressed_rd_exp),
    .rd_mant(compressed_rd_mant),
    .rd_valid(compressed_rd_valid),
    
    // 调试
    .dbg_total_writes(),
    .dbg_total_reads(),
    .dbg_write_collisions()
);

//========== 3. Gate Engine ==========
sidenet_gate_engine #(
    .TOKEN_NUM(TOKEN_NUM),
    .COMPRESSED_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .NUM_LAYERS(NUM_LAYERS)
) u_gate (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制
    .start(gate_start),
    .is_layer0(is_layer0),
    .layer_id(layer_id),
    .done(gate_done),
    .busy(gate_busy),
    
    // Compressed Buffer读取：ẑ_i
    .compressed_rd_en(compressed_rd_en),
    .compressed_rd_addr(compressed_rd_addr),
    .compressed_rd_exp(compressed_rd_exp),
    .compressed_rd_mant(compressed_rd_mant),
    .compressed_rd_valid(compressed_rd_valid),
    
    // Layer Output Buffer读取：â_(i-1)
    .adapted_rd_en(prev_layer_rd_en),
    .adapted_rd_addr(prev_layer_token_id),
    .adapted_rd_exp(prev_layer_exp),
    .adapted_rd_mant(prev_layer_mant),
    .adapted_rd_valid(prev_layer_valid),
    
    // Gated Buffer写入：a_i
    .gated_wr_en(gated_wr_en),
    .gated_wr_addr(gated_wr_addr),
    .gated_wr_exp(gated_wr_exp),
    .gated_wr_mant(gated_wr_mant),
    
    // 调试
    .dbg_state(dbg_gate_state),
    .dbg_token_count()
);

//========== 4. Gated Buffer ==========
sidenet_gated_buffer #(
    .TOKEN_NUM(TOKEN_NUM),
    .COMPRESSED_DIM(SIDENET_DIM),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) u_gated_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    // 写接口：来自Gate Engine
    .wr_en(gated_wr_en),
    .wr_addr(gated_wr_addr),
    .wr_exp(gated_wr_exp),
    .wr_mant(gated_wr_mant),
    .wr_ready(),
    
    // 读接口：供Adaptation Transformer读取
    .rd_en(gated_rd_en),
    .rd_addr(gated_rd_addr),
    .rd_exp(gated_rd_exp),
    .rd_mant(gated_rd_mant),
    .rd_valid(gated_rd_valid),
    
    // 调试
    .dbg_total_writes(),
    .dbg_total_reads(),
    .dbg_write_collisions()
);

//========== 5. Adaptation Transformer (修改 v2.1, 适配统一 FFN 接口) ==========
sidenet_adaptation_transformer #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_BATCH(TOKEN_BATCH),
    .BATCH_NUM(BATCH_NUM),
    .DIM(SIDENET_DIM),
    .NUM_HEADS(NUM_HEADS),
    .HEAD_DIM(HEAD_DIM),
    .D_FF(D_FF),
    .FEATURE_CHUNK(FEATURE_CHUNK),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .K_CHUNK_SIZE(K_CHUNK_SIZE),
    .K_CHUNK_NUM(K_CHUNK_NUM)
) u_transformer (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制
    .start(transformer_start),
    .layer_id(layer_id),
    .done(transformer_done),
    .busy(transformer_busy),
    
    // Layer Token Buffer接口（读）- 从Gated Buffer读取
    .layer_token_rd_en(gated_rd_en),
    .layer_token_rd_addr(gated_rd_addr),
    .layer_token_rd_exp(gated_rd_exp),
    .layer_token_rd_mant(gated_rd_mant),
    .layer_token_rd_valid(gated_rd_valid),
    
    // Layer Token Buffer接口（写）- 写到Layer Output Buffer
    .layer_token_wr_en(curr_layer_wr_en),
    .layer_token_wr_addr(curr_layer_token_id),
    .layer_token_wr_exp(curr_layer_exp),
    .layer_token_wr_mant(curr_layer_mant),
    
    // QKV权重接口（修改 v2.0）- 数组格式
    .qkv_weight_req(qkv_weight_req),
    .qkv_weight_type(qkv_weight_type),
    .qkv_weight_ack(qkv_weight_ack),
    .qkv_weight_valid(qkv_weight_valid),
    .qkv_weight_exp_array(qkv_weight_exp_array),
    .qkv_weight_mant_blocks(qkv_weight_mant_blocks),
    
    // WO权重接口（新增 v2.0）
    .wo_weight_req(wo_weight_req),
    .wo_weight_ready(wo_weight_ready),
    .wo_weight_exp_array(wo_weight_exp_array),
    .wo_weight_mant(wo_weight_mant),
    
    // FFN权重接口（通过weight_controller统一管理）
    .ffn_weight_req(ffn_weight_req),
    .ffn_weight_type(ffn_weight_type),
    .ffn_weight_chunk_id(ffn_weight_chunk_id),
    .ffn_weight_ready(ffn_weight_ready),
    .ffn_weight_exp_array(ffn_weight_exp_array),
    .ffn_weight_mant(ffn_weight_mant),
    
    // LayerNorm参数接口（修改 v2.0）
    .ln_param_req(ln_param_req),
    .ln_param_type(ln_param_type),
    .ln_param_exp(ln_param_exp),
    .ln_param_mant(ln_param_mant),
    .ln_param_valid(ln_param_valid),
    
    // 调试
    .dbg_fsm_state(dbg_transformer_state),
    .dbg_att_state(dbg_att_state),
    .dbg_att_batch(),
    .dbg_ffn_state(),
    .dbg_result_rd_count(),
    .dbg_result_wr_count(),
    .dbg_ln1_rd_count(),
    .dbg_ln1_wr_count()
);

//================================================================================
// 调试输出
//================================================================================

assign dbg_fsm_state   = state;
assign dbg_cycle_count = cycle_counter;

endmodule
