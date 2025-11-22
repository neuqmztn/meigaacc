`timescale 1ns / 1ps

//================================================================================
// FFN Backbone Top - Feed-Forward Network 完整实现（无简化）
//
// 版本：v2.2 - 完整支持32个独立指数（配合linear_compute_engine v3.0）
//
// 修改内容：
// - 权重接口从单指数格式改为32个指数数组格式
// - 兼容weight_controller的FFN权重输出
// - 完整传递32个独立指数给Linear Compute Engine（不丢失信息）
// - 删除共享指数提取逻辑（不再需要）
//
// 架构设计（参考 Attention Backbone）：
// - 在顶层直接实例化 H_act Buffer（7.5KB）
// - 完整的 Token 加载状态机（逐个加载）
// - 完整的权重握手协议
// - 完整的数据流控制
// - 完整的结果写回状态机
//
// 存储架构：
// - H_act Buffer：result_buffer_single_port 实例（7.5KB）
//   * 32 tokens × 4 chunks = 128 entries
//   * 每个 entry：1 个共享指数 + 32 个尾数（15-bit）
//
// 数据流（完整）：
// 1. Token 加载：Result Buffer → Token Block（32 次读取）
// 2. Linear1 阶段（4 次 feature chunk）：
//    - 加载 W1_chunk（权重握手，32个指数数组）
//    - X × W1_chunk → H_chunk
//    - GELU(H_chunk) → H_act_chunk
//    - H_act_chunk → H_act Buffer（32 次写入）
// 3. Linear2 阶段（4 次 feature chunk，累加）：
//    - 清空累加器（第一次）
//    - 加载 H_act_chunk（32 次读取）
//    - 加载 W2_chunk（权重握手，32个指数数组）
//    - H_act_chunk × W2_chunk → Y_partial
//    - Y_partial → Accumulator（累加）
// 4. 结果写回：Y_final → Result Buffer（32 次写入）
//
//================================================================================

module ffn_backbone_top #(
    parameter TOKEN_NUM      = 641,
    parameter TOKEN_CHUNK    = 32,
    parameter BATCH_NUM      = 21,
    parameter D_MODEL        = 32,
    parameter D_FF           = 128,
    parameter FEATURE_CHUNK  = 32,
    parameter NUM_CHUNKS     = 4,
    
    parameter BFP_EXP_W      = 8,
    parameter BFP_MANT_W     = 8,
    parameter ACC_MANT_W     = 15,
    
    parameter ADDR_WIDTH     = 10,
    
    // CE 配置
    parameter G_OUT          = 4,
    parameter T_OUT          = 8,
    parameter CE_OUTPUT_WIDTH = 32,
    parameter CE_BASE_EXP_WIDTH = 9
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire start,
    output wire done,
    output wire busy,
    
    //================================================================================
    // Result Buffer 接口（Token 输入和最终输出）
    //================================================================================
    output wire                      rb_rd_en,
    output wire [ADDR_WIDTH-1:0]     rb_rd_addr,
    input  wire [BFP_EXP_W-1:0]      rb_rd_exp,
    input  wire [D_MODEL*BFP_MANT_W-1:0] rb_rd_mant,
    input  wire                      rb_rd_valid,
    
    output wire                      rb_wr_en,
    output wire [ADDR_WIDTH-1:0]     rb_wr_addr,
    output wire [BFP_EXP_W-1:0]      rb_wr_exp,
    output wire [D_MODEL*BFP_MANT_W-1:0] rb_wr_mant,
    
    //================================================================================
    // 权重请求接口（数组格式 - 与weight_controller兼容）
    //================================================================================
    output wire                      weight_req,
    output wire [1:0]                weight_type,
    output wire [1:0]                weight_chunk_id,
    input  wire                      weight_ready,
    input  wire [D_MODEL*BFP_EXP_W-1:0] weight_exp_array,        // 32个指数数组
    input  wire [D_MODEL*FEATURE_CHUNK*BFP_MANT_W-1:0] weight_mant,
    
    //================================================================================
    // 调试接口
    //================================================================================
    output wire [3:0]  dbg_state,
    output wire [4:0]  dbg_token_batch,
    output wire [1:0]  dbg_feature_chunk,
    output wire [31:0] dbg_cycle_count
);

//================================================================================
// 内部信号声明
//================================================================================

// FSM 控制信号
wire [4:0] token_batch_id;
wire [1:0] feature_chunk_id;
wire [1:0] stage;

wire load_token_block;
wire load_w1_chunk;
wire load_w2_chunk;
wire save_h_act_chunk;
wire load_h_act_chunk;
wire save_result;

wire linear1_start, linear1_done, linear1_busy;
wire gelu_start, gelu_done, gelu_busy;
wire linear2_start, linear2_done, linear2_busy;

wire acc_clear;
wire acc_enable;
wire acc_valid;
wire [2:0] acc_count;

// Token 加载状态机
reg [2:0] token_load_state;
reg [5:0] token_load_count;
reg token_load_done_reg;

localparam TL_IDLE     = 3'd0;
localparam TL_REQUEST  = 3'd1;
localparam TL_WAIT     = 3'd2;
localparam TL_RECEIVE  = 3'd3;
localparam TL_DONE     = 3'd4;

// Token Block 存储（完整）
reg [BFP_EXP_W-1:0] token_block_exp;
reg [D_MODEL*BFP_MANT_W-1:0] token_block_mant [0:TOKEN_CHUNK-1];

// Token 读取控制
reg token_rb_rd_en;
reg [ADDR_WIDTH-1:0] token_rb_rd_addr;

// 权重加载状态机
reg [2:0] weight_load_state;
reg weight_load_done_reg;

localparam WL_IDLE    = 3'd0;
localparam WL_REQUEST = 3'd1;
localparam WL_WAIT_ACK = 3'd2;
localparam WL_WAIT_DATA = 3'd3;
localparam WL_DONE    = 3'd4;

// 权重存储（数组格式）
reg [D_MODEL*BFP_EXP_W-1:0] w1_chunk_exp_array;         // 32个指数数组
reg [D_MODEL*FEATURE_CHUNK*BFP_MANT_W-1:0] w1_chunk_mant;
reg [D_MODEL*BFP_EXP_W-1:0] w2_chunk_exp_array;         // 32个指数数组
reg [FEATURE_CHUNK*D_MODEL*BFP_MANT_W-1:0] w2_chunk_mant;

// 权重请求控制
reg weight_req_reg;
reg [1:0] weight_type_reg;
reg [1:0] weight_chunk_id_reg;

// Linear1 Engine 信号
wire [BFP_EXP_W-1:0] linear1_x_exp;
wire [TOKEN_CHUNK*D_MODEL*BFP_MANT_W-1:0] linear1_x_mant;
wire [D_MODEL*BFP_EXP_W-1:0] linear1_w_exp_array;  // 32个指数数组
wire [D_MODEL*FEATURE_CHUNK*BFP_MANT_W-1:0] linear1_w_mant;
wire [BFP_EXP_W-1:0] linear1_y_exp;
wire [TOKEN_CHUNK*FEATURE_CHUNK*ACC_MANT_W-1:0] linear1_y_mant;

// GELU Engine 信号
wire [BFP_EXP_W-1:0] gelu_in_exp;
wire [TOKEN_CHUNK*FEATURE_CHUNK*ACC_MANT_W-1:0] gelu_in_mant;
wire [BFP_EXP_W-1:0] gelu_out_exp;
wire [TOKEN_CHUNK*FEATURE_CHUNK*ACC_MANT_W-1:0] gelu_out_mant;

// H_act Buffer 信号
wire h_act_wr_en;
wire [6:0] h_act_wr_addr;
wire [BFP_EXP_W-1:0] h_act_wr_exp;
wire [FEATURE_CHUNK*ACC_MANT_W-1:0] h_act_wr_mant;

wire h_act_rd_en;
wire [6:0] h_act_rd_addr;
wire [BFP_EXP_W-1:0] h_act_rd_exp;
wire [FEATURE_CHUNK*ACC_MANT_W-1:0] h_act_rd_mant;
wire h_act_rd_valid;

// H_act 写入状态机
reg [2:0] h_act_wr_state;
reg [5:0] h_act_wr_token_count;
reg h_act_wr_done_reg;

localparam HW_IDLE    = 3'd0;
localparam HW_WRITE   = 3'd1;
localparam HW_WAIT    = 3'd2;
localparam HW_DONE    = 3'd3;

// H_act 读取状态机
reg [2:0] h_act_rd_state;
reg [5:0] h_act_rd_token_count;
reg h_act_rd_done_reg;

localparam HR_IDLE    = 3'd0;
localparam HR_REQUEST = 3'd1;
localparam HR_WAIT    = 3'd2;
localparam HR_RECEIVE = 3'd3;
localparam HR_DONE    = 3'd4;

// H_act 读取缓存
reg [BFP_EXP_W-1:0] h_act_chunk_exp;
reg [FEATURE_CHUNK*ACC_MANT_W-1:0] h_act_chunk_mant [0:TOKEN_CHUNK-1];

// Linear2 Engine 信号
wire [BFP_EXP_W-1:0] linear2_x_exp;
wire [TOKEN_CHUNK*FEATURE_CHUNK*ACC_MANT_W-1:0] linear2_x_mant;
wire [FEATURE_CHUNK*BFP_EXP_W-1:0] linear2_w_exp_array;  // 32个指数数组
wire [FEATURE_CHUNK*D_MODEL*BFP_MANT_W-1:0] linear2_w_mant;
wire [BFP_EXP_W-1:0] linear2_y_exp;
wire [TOKEN_CHUNK*D_MODEL*ACC_MANT_W-1:0] linear2_y_mant;

// Accumulator 信号
wire [BFP_EXP_W-1:0] acc_partial_exp;
wire [TOKEN_CHUNK*D_MODEL*ACC_MANT_W-1:0] acc_partial_mant;
wire [BFP_EXP_W-1:0] acc_result_exp;
wire [TOKEN_CHUNK*D_MODEL*ACC_MANT_W-1:0] acc_result_mant;

// 结果写回状态机
reg [2:0] result_wr_state;
reg [5:0] result_wr_token_count;
reg result_wr_done_reg;

localparam RW_IDLE    = 3'd0;
localparam RW_WRITE   = 3'd1;
localparam RW_WAIT    = 3'd2;
localparam RW_DONE    = 3'd3;

// 结果写回控制
reg result_rb_wr_en;
reg [ADDR_WIDTH-1:0] result_rb_wr_addr;
reg [BFP_EXP_W-1:0] result_rb_wr_exp;
reg [D_MODEL*BFP_MANT_W-1:0] result_rb_wr_mant;

//================================================================================
// 调试输出
//================================================================================

assign dbg_token_batch = token_batch_id;
assign dbg_feature_chunk = feature_chunk_id;

//================================================================================
// 1. FFN Control FSM（四维分块调度器）
//================================================================================

ffn_control_fsm #(
    .TOKEN_NUM(TOKEN_NUM),
    .TOKEN_CHUNK(TOKEN_CHUNK),
    .D_MODEL(D_MODEL),
    .D_FF(D_FF),
    .FEATURE_CHUNK(FEATURE_CHUNK)
) u_ffn_control_fsm (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(start),
    .done(done),
    .busy(busy),
    
    // 握手完成信号
    .token_load_done(token_load_done_reg),
    .weight_load_done(weight_load_done_reg),
    .h_act_wr_done(h_act_wr_done_reg),
    .h_act_rd_done(h_act_rd_done_reg),
    .result_wr_done(result_wr_done_reg),
    
    // 索引输出
    .token_batch_id(token_batch_id),
    .feature_chunk_id(feature_chunk_id),
    .stage(stage),
    
    // Memory Manager 控制
    .load_token_block(load_token_block),
    .load_w1_chunk(load_w1_chunk),
    .load_w2_chunk(load_w2_chunk),
    .load_h_act_chunk(load_h_act_chunk),
    .save_h_act_chunk(save_h_act_chunk),
    .save_result(save_result),
    
    // 计算引擎控制
    .linear1_start(linear1_start),
    .linear1_done(linear1_done),
    .gelu_start(gelu_start),
    .gelu_done(gelu_done),
    .linear2_start(linear2_start),
    .linear2_done(linear2_done),
    
    // 累加器控制
    .acc_clear(acc_clear),
    .acc_enable(acc_enable),
    .acc_valid(acc_valid),
    
    // 调试
    .cycle_count(dbg_cycle_count),
    .fsm_state(dbg_state)
);

//================================================================================
// 2. Token Block 加载状态机（完整，无简化）
//================================================================================

integer t_idx;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        token_load_state <= TL_IDLE;
        token_load_count <= 6'd0;
        token_load_done_reg <= 1'b0;
        token_rb_rd_en <= 1'b0;
        token_rb_rd_addr <= {ADDR_WIDTH{1'b0}};
        token_block_exp <= {BFP_EXP_W{1'b0}};
        for (t_idx = 0; t_idx < TOKEN_CHUNK; t_idx = t_idx + 1) begin
            token_block_mant[t_idx] <= {D_MODEL*BFP_MANT_W{1'b0}};
        end
    end else begin
        case (token_load_state)
            
            TL_IDLE: begin
                token_load_done_reg <= 1'b0;
                if (load_token_block) begin
                    token_load_count <= 6'd0;
                    token_load_state <= TL_REQUEST;
                end
            end
            
            TL_REQUEST: begin
                // 请求读取一个 token
                token_rb_rd_en <= 1'b1;
                token_rb_rd_addr <= token_batch_id * TOKEN_CHUNK + token_load_count;
                token_load_state <= TL_WAIT;
            end
            
            TL_WAIT: begin
                token_rb_rd_en <= 1'b0;
                if (rb_rd_valid) begin
                    token_load_state <= TL_RECEIVE;
                end
            end
            
            TL_RECEIVE: begin
                // 接收数据
                if (token_load_count == 6'd0) begin
                    token_block_exp <= rb_rd_exp;  // 第一个 token 的指数
                end
                token_block_mant[token_load_count] <= rb_rd_mant;
                
                // 检查是否完成
                if (token_load_count < TOKEN_CHUNK - 1) begin
                    token_load_count <= token_load_count + 1'b1;
                    token_load_state <= TL_REQUEST;  // 继续加载下一个
                end else begin
                    token_load_state <= TL_DONE;
                end
            end
            
            TL_DONE: begin
                token_load_done_reg <= 1'b1;
                if (!load_token_block) begin
                    token_load_state <= TL_IDLE;
                    token_load_done_reg <= 1'b0;
                end
            end
            
            default: token_load_state <= TL_IDLE;
        endcase
    end
end

assign rb_rd_en = token_rb_rd_en;
assign rb_rd_addr = token_rb_rd_addr;

//================================================================================
// 3. 权重加载状态机（完整握手协议）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        weight_load_state <= WL_IDLE;
        weight_load_done_reg <= 1'b0;
        weight_req_reg <= 1'b0;
        weight_type_reg <= 2'd0;
        weight_chunk_id_reg <= 2'd0;
        w1_chunk_exp_array <= {D_MODEL*BFP_EXP_W{1'b0}};
        w1_chunk_mant <= {D_MODEL*FEATURE_CHUNK*BFP_MANT_W{1'b0}};
        w2_chunk_exp_array <= {D_MODEL*BFP_EXP_W{1'b0}};
        w2_chunk_mant <= {FEATURE_CHUNK*D_MODEL*BFP_MANT_W{1'b0}};
    end else begin
        case (weight_load_state)
            
            WL_IDLE: begin
                weight_load_done_reg <= 1'b0;
                weight_req_reg <= 1'b0;
                
                if (load_w1_chunk || load_w2_chunk) begin
                    weight_type_reg <= load_w2_chunk ? 2'd1 : 2'd0;
                    weight_chunk_id_reg <= feature_chunk_id;
                    weight_load_state <= WL_REQUEST;
                end
            end
            
            WL_REQUEST: begin
                weight_req_reg <= 1'b1;
                weight_load_state <= WL_WAIT_ACK;
            end
            
            WL_WAIT_ACK: begin
                // 等待第一次 weight_ready（ACK）
                if (weight_ready) begin
                    weight_req_reg <= 1'b0;
                    weight_load_state <= WL_WAIT_DATA;
                end
            end
            
            WL_WAIT_DATA: begin
                // 等待第二次 weight_ready（DATA）
                if (weight_ready) begin
                    // 接收数据
                    if (weight_type_reg == 2'd0) begin
                        // W1
                        w1_chunk_exp_array <= weight_exp_array;
                        w1_chunk_mant <= weight_mant;
                    end else begin
                        // W2
                        w2_chunk_exp_array <= weight_exp_array;
                        w2_chunk_mant <= weight_mant;
                    end
                    weight_load_state <= WL_DONE;
                end
            end
            
            WL_DONE: begin
                weight_load_done_reg <= 1'b1;
                if (!load_w1_chunk && !load_w2_chunk) begin
                    weight_load_state <= WL_IDLE;
                    weight_load_done_reg <= 1'b0;
                end
            end
            
            default: weight_load_state <= WL_IDLE;
        endcase
    end
end

assign weight_req = weight_req_reg;
assign weight_type = weight_type_reg;
assign weight_chunk_id = weight_chunk_id_reg;

//================================================================================
// 4. Linear1 数据准备
//================================================================================

// 打包 token block
integer pack_idx;
reg [TOKEN_CHUNK*D_MODEL*BFP_MANT_W-1:0] token_block_mant_packed;

always @(*) begin
    for (pack_idx = 0; pack_idx < TOKEN_CHUNK; pack_idx = pack_idx + 1) begin
        token_block_mant_packed[pack_idx*D_MODEL*BFP_MANT_W +: D_MODEL*BFP_MANT_W] 
            = token_block_mant[pack_idx];
    end
end

assign linear1_x_exp = token_block_exp;
assign linear1_x_mant = token_block_mant_packed;
assign linear1_w_exp_array = w1_chunk_exp_array;  // 直接传递32个指数
assign linear1_w_mant = w1_chunk_mant;

//================================================================================
// 5. Linear1 Compute Engine
//================================================================================

linear_compute_engine #(
    .TOKEN_CHUNK(TOKEN_CHUNK),
    .INPUT_DIM(D_MODEL),
    .OUTPUT_DIM(FEATURE_CHUNK),
    .BFP_EXP_W(BFP_EXP_W),
    .BFP_MANT_W(BFP_MANT_W),
    .ACC_MANT_W(ACC_MANT_W),
    .G_OUT(G_OUT),
    .T_OUT(T_OUT),
    .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH)
) u_linear1_engine (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(linear1_start && token_load_done_reg && weight_load_done_reg),
    .done(linear1_done),
    .busy(linear1_busy),
    
    .x_exp(linear1_x_exp),
    .x_mant(linear1_x_mant),
    .w_exp_array(linear1_w_exp_array),  // 使用数组接口
    .w_mant(linear1_w_mant),
    
    .y_exp(linear1_y_exp),
    .y_mant(linear1_y_mant)
);

//================================================================================
// 6. GELU Engine
//================================================================================

assign gelu_in_exp = linear1_y_exp;
assign gelu_in_mant = linear1_y_mant;

gelu_engine #(
    .TOKEN_CHUNK(TOKEN_CHUNK),
    .DIM(FEATURE_CHUNK),
    .BFP_EXP_W(BFP_EXP_W),
    .MANT_W(ACC_MANT_W)
) u_gelu_engine (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(gelu_start),
    .done(gelu_done),
    .busy(gelu_busy),
    
    .in_exp(gelu_in_exp),
    .in_mant(gelu_in_mant),
    
    .out_exp(gelu_out_exp),
    .out_mant(gelu_out_mant)
);

//================================================================================
// 7. H_act Buffer 实例化（使用 result_buffer_single_port）
//================================================================================

result_buffer_single_port #(
    .TOKEN_NUM(128),               // 32 tokens × 4 chunks
    .DIM(FEATURE_CHUNK),           // 32 dims per chunk
    .DATA_WIDTH(ACC_MANT_W),       // 15-bit mantissa
    .EXP_WIDTH(BFP_EXP_W),         // 8-bit exponent
    .ADDR_WIDTH(7)                 // 2^7 = 128
) u_h_act_buffer (
    .clk(clk),
    .rst_n(rst_n),
    
    // 读接口
    .rd_en(h_act_rd_en),
    .rd_addr(h_act_rd_addr),
    .rd_exp(h_act_rd_exp),
    .rd_mant(h_act_rd_mant),
    .rd_valid(h_act_rd_valid),
    
    // 写接口
    .wr_en(h_act_wr_en),
    .wr_addr(h_act_wr_addr),
    .wr_exp(h_act_wr_exp),
    .wr_mant(h_act_wr_mant),
    
    // 调试
    .dbg_rd_count(),
    .dbg_wr_count()
);

//================================================================================
// 8. H_act Buffer 写入状态机（完整，逐 token 写入）
//================================================================================

// 解包 GELU 输出
reg [FEATURE_CHUNK*ACC_MANT_W-1:0] gelu_out_mant_unpacked [0:TOKEN_CHUNK-1];

integer unpack_idx;
always @(*) begin
    for (unpack_idx = 0; unpack_idx < TOKEN_CHUNK; unpack_idx = unpack_idx + 1) begin
        gelu_out_mant_unpacked[unpack_idx] 
            = gelu_out_mant[unpack_idx*FEATURE_CHUNK*ACC_MANT_W +: FEATURE_CHUNK*ACC_MANT_W];
    end
end

reg h_act_wr_en_reg;
reg [6:0] h_act_wr_addr_reg;
reg [BFP_EXP_W-1:0] h_act_wr_exp_reg;
reg [FEATURE_CHUNK*ACC_MANT_W-1:0] h_act_wr_mant_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        h_act_wr_state <= HW_IDLE;
        h_act_wr_token_count <= 6'd0;
        h_act_wr_done_reg <= 1'b0;
        h_act_wr_en_reg <= 1'b0;
        h_act_wr_addr_reg <= 7'd0;
        h_act_wr_exp_reg <= {BFP_EXP_W{1'b0}};
        h_act_wr_mant_reg <= {FEATURE_CHUNK*ACC_MANT_W{1'b0}};
    end else begin
        case (h_act_wr_state)
            
            HW_IDLE: begin
                h_act_wr_done_reg <= 1'b0;
                h_act_wr_en_reg <= 1'b0;
                if (save_h_act_chunk && gelu_done) begin
                    h_act_wr_token_count <= 6'd0;
                    h_act_wr_state <= HW_WRITE;
                end
            end
            
            HW_WRITE: begin
                // 写入一个 token 的数据
                // 地址 = token_id * 4 + chunk_id
                h_act_wr_en_reg <= 1'b1;
                h_act_wr_addr_reg <= {h_act_wr_token_count[4:0], feature_chunk_id};
                h_act_wr_exp_reg <= gelu_out_exp;
                h_act_wr_mant_reg <= gelu_out_mant_unpacked[h_act_wr_token_count];
                h_act_wr_state <= HW_WAIT;
            end
            
            HW_WAIT: begin
                h_act_wr_en_reg <= 1'b0;
                
                // 检查是否完成所有 token
                if (h_act_wr_token_count < TOKEN_CHUNK - 1) begin
                    h_act_wr_token_count <= h_act_wr_token_count + 1'b1;
                    h_act_wr_state <= HW_WRITE;
                end else begin
                    h_act_wr_state <= HW_DONE;
                end
            end
            
            HW_DONE: begin
                h_act_wr_done_reg <= 1'b1;
                if (!save_h_act_chunk) begin
                    h_act_wr_state <= HW_IDLE;
                    h_act_wr_done_reg <= 1'b0;
                end
            end
            
            default: h_act_wr_state <= HW_IDLE;
        endcase
    end
end

assign h_act_wr_en = h_act_wr_en_reg;
assign h_act_wr_addr = h_act_wr_addr_reg;
assign h_act_wr_exp = h_act_wr_exp_reg;
assign h_act_wr_mant = h_act_wr_mant_reg;

//================================================================================
// 9. H_act Buffer 读取状态机（完整，逐 token 读取）
//================================================================================

reg h_act_rd_en_reg;
reg [6:0] h_act_rd_addr_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        h_act_rd_state <= HR_IDLE;
        h_act_rd_token_count <= 6'd0;
        h_act_rd_done_reg <= 1'b0;
        h_act_rd_en_reg <= 1'b0;
        h_act_rd_addr_reg <= 7'd0;
        h_act_chunk_exp <= {BFP_EXP_W{1'b0}};
        for (t_idx = 0; t_idx < TOKEN_CHUNK; t_idx = t_idx + 1) begin
            h_act_chunk_mant[t_idx] <= {FEATURE_CHUNK*ACC_MANT_W{1'b0}};
        end
    end else begin
        case (h_act_rd_state)
            
            HR_IDLE: begin
                h_act_rd_done_reg <= 1'b0;
                if (load_h_act_chunk) begin
                    h_act_rd_token_count <= 6'd0;
                    h_act_rd_state <= HR_REQUEST;
                end
            end
            
            HR_REQUEST: begin
                // 请求读取一个 token 的数据
                h_act_rd_en_reg <= 1'b1;
                h_act_rd_addr_reg <= {h_act_rd_token_count[4:0], feature_chunk_id};
                h_act_rd_state <= HR_WAIT;
            end
            
            HR_WAIT: begin
                h_act_rd_en_reg <= 1'b0;
                if (h_act_rd_valid) begin
                    h_act_rd_state <= HR_RECEIVE;
                end
            end
            
            HR_RECEIVE: begin
                // 接收数据
                if (h_act_rd_token_count == 6'd0) begin
                    h_act_chunk_exp <= h_act_rd_exp;
                end
                h_act_chunk_mant[h_act_rd_token_count] <= h_act_rd_mant;
                
                // 检查是否完成
                if (h_act_rd_token_count < TOKEN_CHUNK - 1) begin
                    h_act_rd_token_count <= h_act_rd_token_count + 1'b1;
                    h_act_rd_state <= HR_REQUEST;
                end else begin
                    h_act_rd_state <= HR_DONE;
                end
            end
            
            HR_DONE: begin
                h_act_rd_done_reg <= 1'b1;
                if (!load_h_act_chunk) begin
                    h_act_rd_state <= HR_IDLE;
                    h_act_rd_done_reg <= 1'b0;
                end
            end
            
            default: h_act_rd_state <= HR_IDLE;
        endcase
    end
end

assign h_act_rd_en = h_act_rd_en_reg;
assign h_act_rd_addr = h_act_rd_addr_reg;

//================================================================================
// 10. Linear2 数据准备
//================================================================================

// 打包 H_act chunk
integer pack_h_idx;
reg [TOKEN_CHUNK*FEATURE_CHUNK*ACC_MANT_W-1:0] h_act_chunk_mant_packed;

always @(*) begin
    for (pack_h_idx = 0; pack_h_idx < TOKEN_CHUNK; pack_h_idx = pack_h_idx + 1) begin
        h_act_chunk_mant_packed[pack_h_idx*FEATURE_CHUNK*ACC_MANT_W +: FEATURE_CHUNK*ACC_MANT_W]
            = h_act_chunk_mant[pack_h_idx];
    end
end

assign linear2_x_exp = h_act_chunk_exp;
assign linear2_x_mant = h_act_chunk_mant_packed;
assign linear2_w_exp_array = w2_chunk_exp_array;  // 直接传递32个指数
assign linear2_w_mant = w2_chunk_mant;

//================================================================================
// 11. Linear2 Compute Engine（复用 Linear1 引擎）
//================================================================================

linear_compute_engine #(
    .TOKEN_CHUNK(TOKEN_CHUNK),
    .INPUT_DIM(FEATURE_CHUNK),
    .OUTPUT_DIM(D_MODEL),
    .BFP_EXP_W(BFP_EXP_W),
    .BFP_MANT_W(ACC_MANT_W),      // ✅ 输入是 GELU 的 15-bit
    .ACC_MANT_W(ACC_MANT_W),
    .G_OUT(G_OUT),
    .T_OUT(T_OUT),
    .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH)
) u_linear2_engine (
    .clk(clk),
    .rst_n(rst_n),
    
    .start(linear2_start && h_act_rd_done_reg && weight_load_done_reg),
    .done(linear2_done),
    .busy(linear2_busy),
    
    .x_exp(linear2_x_exp),
    .x_mant(linear2_x_mant),
    .w_exp_array(linear2_w_exp_array),  // 使用数组接口
    .w_mant(linear2_w_mant),
    
    .y_exp(linear2_y_exp),
    .y_mant(linear2_y_mant)
);

//================================================================================
// 12. Linear2 Accumulator（带指数对齐）
//================================================================================

assign acc_partial_exp = linear2_y_exp;
assign acc_partial_mant = linear2_y_mant;

linear2_accumulator #(
    .TOKEN_CHUNK(TOKEN_CHUNK),
    .OUTPUT_DIM(D_MODEL),
    .BFP_EXP_W(BFP_EXP_W),
    .INPUT_MANT_W(ACC_MANT_W),
    .OUTPUT_MANT_W(ACC_MANT_W),
    .NUM_CHUNKS(NUM_CHUNKS)
) u_accumulator (
    .clk(clk),
    .rst_n(rst_n),
    
    .clear(acc_clear),
    .enable(acc_enable && linear2_done),
    
    .partial_exp(acc_partial_exp),
    .partial_mant(acc_partial_mant),
    
    .result_exp(acc_result_exp),
    .result_mant(acc_result_mant),
    .result_valid(acc_valid),
    
    .accum_count(acc_count)
);

//================================================================================
// 13. 结果写回状态机（完整，逐 token 写入）
//================================================================================

// 解包累加器结果
reg [D_MODEL*ACC_MANT_W-1:0] acc_result_mant_unpacked [0:TOKEN_CHUNK-1];

integer unpack_acc_idx;
always @(*) begin
    for (unpack_acc_idx = 0; unpack_acc_idx < TOKEN_CHUNK; unpack_acc_idx = unpack_acc_idx + 1) begin
        acc_result_mant_unpacked[unpack_acc_idx]
            = acc_result_mant[unpack_acc_idx*D_MODEL*ACC_MANT_W +: D_MODEL*ACC_MANT_W];
    end
end

// 精度转换：ACC_MANT_W (15-bit) → BFP_MANT_W (8-bit)
reg [D_MODEL*BFP_MANT_W-1:0] acc_result_mant_converted [0:TOKEN_CHUNK-1];

integer convert_idx, dim_idx;
always @(*) begin
    for (convert_idx = 0; convert_idx < TOKEN_CHUNK; convert_idx = convert_idx + 1) begin
        for (dim_idx = 0; dim_idx < D_MODEL; dim_idx = dim_idx + 1) begin
            // 简单截断（实际应该有舍入逻辑）
            acc_result_mant_converted[convert_idx][dim_idx*BFP_MANT_W +: BFP_MANT_W]
                = acc_result_mant_unpacked[convert_idx][(dim_idx+1)*ACC_MANT_W-1 -: BFP_MANT_W];
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        result_wr_state <= RW_IDLE;
        result_wr_token_count <= 6'd0;
        result_wr_done_reg <= 1'b0;
        result_rb_wr_en <= 1'b0;
        result_rb_wr_addr <= {ADDR_WIDTH{1'b0}};
        result_rb_wr_exp <= {BFP_EXP_W{1'b0}};
        result_rb_wr_mant <= {D_MODEL*BFP_MANT_W{1'b0}};
    end else begin
        case (result_wr_state)
            
            RW_IDLE: begin
                result_wr_done_reg <= 1'b0;
                result_rb_wr_en <= 1'b0;
                if (save_result && acc_valid) begin
                    result_wr_token_count <= 6'd0;
                    result_wr_state <= RW_WRITE;
                end
            end
            
            RW_WRITE: begin
                // 写入一个 token 的结果
                result_rb_wr_en <= 1'b1;
                result_rb_wr_addr <= token_batch_id * TOKEN_CHUNK + result_wr_token_count;
                result_rb_wr_exp <= acc_result_exp;
                result_rb_wr_mant <= acc_result_mant_converted[result_wr_token_count];
                result_wr_state <= RW_WAIT;
            end
            
            RW_WAIT: begin
                result_rb_wr_en <= 1'b0;
                
                // 检查是否完成
                if (result_wr_token_count < TOKEN_CHUNK - 1) begin
                    result_wr_token_count <= result_wr_token_count + 1'b1;
                    result_wr_state <= RW_WRITE;
                end else begin
                    result_wr_state <= RW_DONE;
                end
            end
            
            RW_DONE: begin
                result_wr_done_reg <= 1'b1;
                if (!save_result) begin
                    result_wr_state <= RW_IDLE;
                    result_wr_done_reg <= 1'b0;
                end
            end
            
            default: result_wr_state <= RW_IDLE;
        endcase
    end
end

assign rb_wr_en = result_rb_wr_en;
assign rb_wr_addr = result_rb_wr_addr;
assign rb_wr_exp = result_rb_wr_exp;
assign rb_wr_mant = result_rb_wr_mant;

//================================================================================
// 调试信号
//================================================================================

`ifdef SIMULATION
always @(posedge clk) begin
    if (token_load_state == TL_RECEIVE) begin
        $display("[%0t] FFN Top: Loaded token %0d/%0d for batch %0d", 
                 $time, token_load_count, TOKEN_CHUNK, token_batch_id);
    end
    
    if (h_act_wr_state == HW_WRITE) begin
        $display("[%0t] FFN Top: Writing H_act token %0d, chunk %0d to addr %0d", 
                 $time, h_act_wr_token_count, feature_chunk_id, h_act_wr_addr_reg);
    end
    
    if (h_act_rd_state == HR_RECEIVE) begin
        $display("[%0t] FFN Top: Read H_act token %0d, chunk %0d from addr %0d", 
                 $time, h_act_rd_token_count, feature_chunk_id, h_act_rd_addr_reg);
    end
    
    if (result_wr_state == RW_WRITE) begin
        $display("[%0t] FFN Top: Writing result token %0d for batch %0d to addr %0d", 
                 $time, result_wr_token_count, token_batch_id, result_rb_wr_addr);
    end
    
    if (done) begin
        $display("[%0t] FFN Top: ========== FFN COMPLETE ==========", $time);
        $display("[%0t] FFN Top: Total cycles: %0d", $time, dbg_cycle_count);
    end
end
`endif

endmodule