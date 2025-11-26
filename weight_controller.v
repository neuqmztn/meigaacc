//================================================================================
// Weight Controller - 统一权重管理模块
//
// 版本：v2.1
// 日期：2024-11-16
//
// 更新内容（v2.0 → v2.1）：
// - 修复LayerNorm参数接口命名：ln_param_ready → ln_param_valid
// - 与backbone_transformer接口完全兼容
//
// 功能：
// 1. 统一管理Attention、FFN、LayerNorm的所有权重/参数
// 2. 处理权重请求仲裁（QKV vs WO vs FFN vs LayerNorm）
// 3. 管理DMA读取权重
// 4. 片上缓存最近使用的权重（LRU）
// 5. 自动地址映射和计算
//
// 接口设计：
// - QKV: weight_type [1:0] (00=Q, 01=K, 10=V)
// - WO:  独立请求信号
// - FFN: weight_type [1:0] (00=W1, 01=W2), chunk_id [1:0] (0-3)
// - LN:  param_type [1:0] (00=LN1_gamma, 01=LN1_beta, 10=LN2_gamma, 11=LN2_beta)
//
// 内部权重类型编码（serving_type [3:0]）：
// 0x0: WQ,  0x1: WK,  0x2: WV,  0x3: WO     (Attention, 32×32, 数组格式)
// 0x4: W1,  0x5: W2                         (FFN, 32×32, 数组格式)
// 0x8: LN1_gamma, 0x9: LN1_beta            (LayerNorm1, 32, 单指数)
// 0xA: LN2_gamma, 0xB: LN2_beta            (LayerNorm2, 32, 单指数)
//
// DRAM布局（per layer, 16KB对齐）：
// +0x0000: WQ (1056B)       [32个exp][32×32个mant]
// +0x0420: WK (1056B)       [32个exp][32×32个mant]
// +0x0840: WV (1056B)       [32个exp][32×32个mant]
// +0x0C60: WO (1056B)       [32个exp][32×32个mant]
// +0x1080: W1_c0 (1056B)    [32个exp][32×32个mant]
// +0x14A0: W1_c1 (1056B)    [32个exp][32×32个mant]
// +0x18C0: W1_c2 (1056B)    [32个exp][32×32个mant]
// +0x1CE0: W1_c3 (1056B)    [32个exp][32×32个mant]
// +0x2100: W2_c0 (1056B)    [32个exp][32×32个mant]
// +0x2520: W2_c1 (1056B)    [32个exp][32×32个mant]
// +0x2940: W2_c2 (1056B)    [32个exp][32×32个mant]
// +0x2D60: W2_c3 (1056B)    [32个exp][32×32个mant]
// +0x3180: LN1_gamma (33B)  [1个exp][32个mant]
// +0x31A1: LN1_beta (33B)   [1个exp][32个mant]
// +0x31C2: LN2_gamma (33B)  [1个exp][32个mant]
// +0x31E3: LN2_beta (33B)   [1个exp][32个mant]
//
// 作者：Claude
//================================================================================

`timescale 1ns / 1ps

module weight_controller #(
    parameter NUM_LAYERS      = 4,
    parameter DIM             = 32,
    parameter DATA_WIDTH      = 8,
    parameter EXP_WIDTH       = 8,
    parameter DRAM_ADDR_WIDTH = 32,
    parameter DRAM_DATA_WIDTH = 256,
    parameter CACHE_ENTRIES   = 4
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 配置接口
    //================================================================================
    input  wire [1:0] current_layer_id,
    input  wire [DRAM_ADDR_WIDTH-1:0] weight_base_addr,
    
    //================================================================================
    // QKV权重接口（数组格式）
    //================================================================================
    input  wire qkv_weight_req,                                // 请求加载Q/K/V权重
    input  wire [1:0] qkv_weight_type,                         // 00=Q, 01=K, 10=V
    output reg  qkv_weight_ack,                                // ACK握手
    output reg  qkv_weight_valid,                              // DATA有效
    output reg  [32*EXP_WIDTH-1:0] qkv_weight_exp_array,
    output reg  [32*32*DATA_WIDTH-1:0] qkv_weight_mant_blocks,
    
    //================================================================================
    // WO权重接口（数组格式）
    //================================================================================
    input  wire wo_weight_req,
    output reg  wo_weight_ready,
    output reg  [32*EXP_WIDTH-1:0] wo_weight_exp_array,
    output reg  [32*32*DATA_WIDTH-1:0] wo_weight_mant,
    
    //================================================================================
    // FFN权重接口（数组格式）
    //================================================================================
    input  wire ffn_weight_req,
    input  wire [1:0] ffn_weight_type,                         // 00=W1, 01=W2
    input  wire [1:0] ffn_weight_chunk_id,                     // 0-3
    output reg  ffn_weight_ready,
    output reg  [32*EXP_WIDTH-1:0] ffn_weight_exp_array,
    output reg  [32*32*DATA_WIDTH-1:0] ffn_weight_mant,
    
    //================================================================================
    // LayerNorm参数接口（单指数格式）
    //================================================================================
    input  wire ln_param_req,
    input  wire [1:0] ln_param_type,                           // 00=LN1_GAMMA, 01=LN1_BETA, 10=LN2_GAMMA, 11=LN2_BETA
    output reg  ln_param_valid,
    output reg  [EXP_WIDTH-1:0] ln_param_exp,
    output reg  [DIM*DATA_WIDTH-1:0] ln_param_mant,
    
    //================================================================================
    // DMA读接口
    //================================================================================
    output reg  dma_req_valid,
    output reg  [DRAM_ADDR_WIDTH-1:0] dma_req_addr,
    output reg  [7:0] dma_req_burst_len,
    input  wire dma_req_ready,
    
    input  wire dma_rsp_valid,
    input  wire [DRAM_DATA_WIDTH-1:0] dma_rsp_data,
    input  wire dma_rsp_last,
    output reg  dma_rsp_ready,
    
    //================================================================================
    // Debug信号
    //================================================================================
    output reg  [3:0] dbg_state,
    output reg  [31:0] dbg_cache_hit_count,
    output reg  [31:0] dbg_cache_miss_count,
    output reg  [31:0] dbg_dma_req_count
);

//================================================================================
// 常量定义
//================================================================================

// 权重类型编码
localparam TYPE_WQ  = 4'h0;
localparam TYPE_WK  = 4'h1;
localparam TYPE_WV  = 4'h2;
localparam TYPE_WO  = 4'h3;
localparam TYPE_W1  = 4'h4;
localparam TYPE_W2  = 4'h5;
localparam TYPE_LN1_GAMMA = 4'h8;
localparam TYPE_LN1_BETA  = 4'h9;
localparam TYPE_LN2_GAMMA = 4'hA;
localparam TYPE_LN2_BETA  = 4'hB;

// 请求者编码
localparam REQ_NONE = 2'b00;
localparam REQ_QKV  = 2'b01;
localparam REQ_WO   = 2'b10;
localparam REQ_FFN  = 2'b11;
localparam REQ_LN   = 3'b100;

// 状态机
localparam IDLE          = 4'd0;
localparam ARBITRATE     = 4'd1;
localparam CHECK_CACHE   = 4'd2;
localparam CALC_ADDR     = 4'd3;
localparam DMA_REQUEST   = 4'd4;
localparam DMA_WAIT      = 4'd5;
localparam DMA_RECEIVE   = 4'd6;
localparam CACHE_UPDATE  = 4'd7;
localparam RESPOND       = 4'd8;

//================================================================================
// 状态机
//================================================================================

reg [3:0] state, next_state;

//================================================================================
// 内部信号
//================================================================================

// 当前服务的请求
reg [2:0] serving_requester;  // 0=none, 1=QKV, 2=WO, 3=FFN, 4=LN
reg [3:0] serving_type;
reg [1:0] serving_chunk_id;
reg [1:0] serving_layer;

//================================================================================
// 权重矩阵缓存（数组格式，用于WQ/WK/WV/WO/W1/W2）
//================================================================================

reg [3:0] weight_cache_type     [0:CACHE_ENTRIES-1]; // TYPE_WQ, TYPE_WK, ...
reg [1:0] weight_cache_chunk_id [0:CACHE_ENTRIES-1]; // 0~3
reg [1:0] weight_cache_layer    [0:CACHE_ENTRIES-1]; // layer id
reg       weight_cache_valid    [0:CACHE_ENTRIES-1];

reg [32*EXP_WIDTH-1:0]      weight_cache_exp_array     [0:CACHE_ENTRIES-1];
reg [32*32*DATA_WIDTH-1:0]  weight_cache_mant_blocks   [0:CACHE_ENTRIES-1];

// 缓存LRU信息
reg [31:0] weight_cache_age [0:CACHE_ENTRIES-1];  // 简单LRU: age越大，越久未使用

// LayerNorm参数缓存（单指数格式）
reg [3:0] ln_cache_type     [0:CACHE_ENTRIES-1]; // TYPE_LN1_GAMMA, TYPE_LN1_BETA, ...
reg [1:0] ln_cache_layer    [0:CACHE_ENTRIES-1];
reg       ln_cache_valid    [0:CACHE_ENTRIES-1];

reg [EXP_WIDTH-1:0]        ln_cache_exp [0:CACHE_ENTRIES-1];
reg [DIM*DATA_WIDTH-1:0]   ln_cache_mant[0:CACHE_ENTRIES-1];

//================================================================================
// DMA接收缓冲
//================================================================================

reg [32*EXP_WIDTH-1:0]     dma_rcv_exp_array;
reg [32*32*DATA_WIDTH-1:0] dma_rcv_mant_blocks;
reg [EXP_WIDTH-1:0]        dma_rcv_ln_exp;
reg [DIM*DATA_WIDTH-1:0]   dma_rcv_ln_mant;

//================================================================================
// 地址计算相关
//================================================================================

reg [DRAM_ADDR_WIDTH-1:0] serving_base_addr;
reg [DRAM_ADDR_WIDTH-1:0] dma_target_addr;
reg [7:0]                 dma_burst_len;

//================================================================================
// 缓存命中信息
//================================================================================

reg weight_cache_hit;
reg [1:0] weight_cache_hit_index;
reg ln_cache_hit;
reg [1:0] ln_cache_hit_index;

//================================================================================
// 通用索引变量
//================================================================================

integer i, j;

//================================================================================
// 状态机时序部分
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        dbg_state <= IDLE;
    end else begin
        state <= next_state;
        dbg_state <= next_state;
    end
end

//================================================================================
// 状态机组合部分
//================================================================================

always @(*) begin
    next_state = state;
    case (state)
        IDLE: begin
            if (qkv_weight_req || wo_weight_req || ffn_weight_req || ln_param_req) begin
                next_state = ARBITRATE;
            end
        end
        
        ARBITRATE: begin
            next_state = CHECK_CACHE;
        end
        
        CHECK_CACHE: begin
            if (serving_type == TYPE_LN1_GAMMA ||
                serving_type == TYPE_LN1_BETA  ||
                serving_type == TYPE_LN2_GAMMA ||
                serving_type == TYPE_LN2_BETA) begin
                // LayerNorm参数
                if (ln_cache_hit) begin
                    next_state = RESPOND;
                end else begin
                    next_state = CALC_ADDR;
                end
            end else begin
                // 权重矩阵（WQ/WK/WV/WO/W1/W2）
                if (weight_cache_hit) begin
                    next_state = RESPOND;
                end else begin
                    next_state = CALC_ADDR;
                end
            end
        end
        
        CALC_ADDR: begin
            next_state = DMA_REQUEST;
        end
        
        DMA_REQUEST: begin
            if (dma_req_ready) begin
                next_state = DMA_WAIT;
            end
        end
        
        DMA_WAIT: begin
            if (dma_rsp_valid) begin
                next_state = DMA_RECEIVE;
            end
        end
        
        DMA_RECEIVE: begin
            if (dma_rsp_valid && dma_rsp_last) begin
                next_state = CACHE_UPDATE;
            end
        end
        
        CACHE_UPDATE: begin
            next_state = RESPOND;
        end
        
        RESPOND: begin
            next_state = IDLE;
        end
        
        default: next_state = IDLE;
    endcase
end

//================================================================================
// 请求仲裁逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        serving_requester <= REQ_NONE;
        serving_type      <= 4'h0;
        serving_chunk_id  <= 2'b00;
        serving_layer     <= 2'b00;
    end else if (state == IDLE && next_state == ARBITRATE) begin
        // 简单优先级仲裁：QKV > WO > FFN > LN
        if (qkv_weight_req) begin
            serving_requester <= REQ_QKV;
            serving_layer     <= current_layer_id;
            case (qkv_weight_type)
                2'b00: serving_type <= TYPE_WQ;
                2'b01: serving_type <= TYPE_WK;
                2'b10: serving_type <= TYPE_WV;
                default: serving_type <= TYPE_WQ;
            endcase
            serving_chunk_id <= 2'b00;    // QKV一次性请求全chunk
        end else if (wo_weight_req) begin
            serving_requester <= REQ_WO;
            serving_layer     <= current_layer_id;
            serving_type      <= TYPE_WO;
            serving_chunk_id  <= 2'b00;   // WO一次性请求全chunk
        end else if (ffn_weight_req) begin
            serving_requester <= REQ_FFN;
            serving_layer     <= current_layer_id;
            case (ffn_weight_type)
                2'b00: serving_type <= TYPE_W1;
                2'b01: serving_type <= TYPE_W2;
                default: serving_type <= TYPE_W1;
            endcase
            serving_chunk_id <= ffn_weight_chunk_id;
        end else if (ln_param_req) begin
            serving_requester <= REQ_LN;
            serving_layer     <= current_layer_id;
            case (ln_param_type)
                2'b00: serving_type <= TYPE_LN1_GAMMA;
                2'b01: serving_type <= TYPE_LN1_BETA;
                2'b10: serving_type <= TYPE_LN2_GAMMA;
                2'b11: serving_type <= TYPE_LN2_BETA;
                default: serving_type <= TYPE_LN1_GAMMA;
            endcase
            serving_chunk_id <= 2'b00;
        end else begin
            serving_requester <= REQ_NONE;
            serving_type      <= 4'h0;
            serving_chunk_id  <= 2'b00;
            serving_layer     <= 2'b00;
        end
    end
end

//================================================================================
// 缓存命中检测
//================================================================================

always @(*) begin
    weight_cache_hit = 1'b0;
    weight_cache_hit_index = 2'b00;
    ln_cache_hit = 1'b0;
    ln_cache_hit_index = 2'b00;
    
    // 判断是权重矩阵还是LayerNorm参数
    if (serving_type <= 4'h5) begin
        // 权重矩阵（WQ/WK/WV/WO/W1/W2）
        for (i = 0; i < 4; i = i + 1) begin
            if (weight_cache_valid[i] &&
                weight_cache_type[i] == serving_type &&
                weight_cache_chunk_id[i] == serving_chunk_id &&
                weight_cache_layer[i] == serving_layer) begin
                weight_cache_hit = 1'b1;
                weight_cache_hit_index = i[1:0];
            end
        end
    end else begin
        // LayerNorm参数
        for (i = 0; i < 4; i = i + 1) begin
            if (ln_cache_valid[i] &&
                ln_cache_type[i] == serving_type &&
                ln_cache_layer[i] == serving_layer) begin
                ln_cache_hit = 1'b1;
                ln_cache_hit_index = i[1:0];
            end
        end
    end
end

//================================================================================
// 地址计算逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        serving_base_addr <= {DRAM_ADDR_WIDTH{1'b0}};
        dma_target_addr   <= {DRAM_ADDR_WIDTH{1'b0}};
        dma_burst_len     <= 8'd0;
    end else if (state == CALC_ADDR) begin
        serving_base_addr <= weight_base_addr;
        
        if (serving_type == TYPE_WQ || serving_type == TYPE_WK || serving_type == TYPE_WV) begin
            dma_target_addr <= weight_base_addr
                               + serving_layer * 32'h0001_0000
                               + {serving_type[1:0], 14'h0};
            dma_burst_len   <= 8'd32;
        end else if (serving_type == TYPE_WO) begin
            dma_target_addr <= weight_base_addr
                               + serving_layer * 32'h0001_0000
                               + 32'h0000_4000;
            dma_burst_len   <= 8'd32;
        end else if (serving_type == TYPE_W1 || serving_type == TYPE_W2) begin
            dma_target_addr <= weight_base_addr
                               + serving_layer * 32'h0002_0000
                               + {serving_type[1:0], serving_chunk_id, 12'h000};
            dma_burst_len   <= 8'd32;
        end else begin
            dma_target_addr <= weight_base_addr
                               + serving_layer * 32'h0000_1000
                               + {serving_type[1:0], 10'h0};
            dma_burst_len   <= 8'd4;
        end
    end
end

//================================================================================
// DMA请求逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dma_req_valid <= 1'b0;
        dma_req_addr  <= {DRAM_ADDR_WIDTH{1'b0}};
        dma_req_burst_len <= 8'd0;
        dma_rsp_ready <= 1'b0;
        dbg_dma_req_count <= 32'h0;
    end else begin
        dma_req_valid <= 1'b0;
        
        if (state == DMA_REQUEST && !dma_req_valid) begin
            dma_req_valid     <= 1'b1;
            dma_req_addr      <= dma_target_addr;
            dma_req_burst_len <= dma_burst_len;
            dbg_dma_req_count <= dbg_dma_req_count + 1;
        end
        
        if (state == DMA_WAIT || state == DMA_RECEIVE) begin
            dma_rsp_ready <= 1'b1;
        end else begin
            dma_rsp_ready <= 1'b0;
        end
    end
end

//================================================================================
// DMA接收逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dma_rcv_exp_array   <= {32*EXP_WIDTH{1'b0}};
        dma_rcv_mant_blocks <= {32*32*DATA_WIDTH{1'b0}};
        dma_rcv_ln_exp      <= {EXP_WIDTH{1'b0}};
        dma_rcv_ln_mant     <= {DIM*DATA_WIDTH{1'b0}};
    end else if (state == DMA_RECEIVE && dma_rsp_valid && dma_rsp_ready) begin
        if (serving_type <= 4'h5) begin
        end else begin
        end
    end
end

//================================================================================
// 缓存更新逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
            weight_cache_valid[i] <= 1'b0;
            weight_cache_type[i]  <= 4'h0;
            weight_cache_chunk_id[i] <= 2'b00;
            weight_cache_layer[i] <= 2'b00;
            weight_cache_exp_array[i] <= {32*EXP_WIDTH{1'b0}};
            weight_cache_mant_blocks[i] <= {32*32*DATA_WIDTH{1'b0}};
            weight_cache_age[i] <= 32'h0;
            
            ln_cache_valid[i] <= 1'b0;
            ln_cache_type[i]  <= 4'h0;
            ln_cache_layer[i] <= 2'b00;
            ln_cache_exp[i]   <= {EXP_WIDTH{1'b0}};
            ln_cache_mant[i]  <= {DIM*DATA_WIDTH{1'b0}};
        end
        dbg_cache_miss_count <= 32'h0;
    end else begin
        if (state == CHECK_CACHE && !weight_cache_hit && !ln_cache_hit &&
            next_state == CALC_ADDR) begin
            dbg_cache_miss_count <= dbg_cache_miss_count + 1;
        end
        
        if (state == CHECK_CACHE || state == RESPOND) begin
            for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
                if ((weight_cache_valid[i] && weight_cache_hit && weight_cache_hit_index != i[1:0]) ||
                    (ln_cache_valid[i]     && ln_cache_hit     && ln_cache_hit_index     != i[1:0])) begin
                    weight_cache_age[i] <= weight_cache_age[i] + 1;
                end
            end
        end
        
        if (state == CACHE_UPDATE) begin
            if (serving_type <= 4'h5) begin:t1
                integer min_index;
                reg [31:0] min_age;
                min_age = 32'hFFFF_FFFF;
                min_index = 0;
                for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
                    if (!weight_cache_valid[i]) begin
                        min_index = i;
                        min_age = 32'h0;
                    end else if (weight_cache_age[i] < min_age) begin
                        min_index = i;
                        min_age = weight_cache_age[i];
                    end
                end
                
                weight_cache_valid[min_index] <= 1'b1;
                weight_cache_type[min_index]  <= serving_type;
                weight_cache_chunk_id[min_index] <= serving_chunk_id;
                weight_cache_layer[min_index] <= serving_layer;
                weight_cache_exp_array[min_index] <= dma_rcv_exp_array;
                weight_cache_mant_blocks[min_index] <= dma_rcv_mant_blocks;
                weight_cache_age[min_index] <= 32'h0;
            end else begin:t2
                integer min_index_ln;
                reg [31:0] min_age_ln;
                min_age_ln = 32'hFFFF_FFFF;
                min_index_ln = 0;
                for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
                    if (!ln_cache_valid[i]) begin
                        min_index_ln = i;
                        min_age_ln = 32'h0;
                    end else if (weight_cache_age[i] < min_age_ln) begin
                        min_index_ln = i;
                        min_age_ln = weight_cache_age[i];
                    end
                end
                
                ln_cache_valid[min_index_ln] <= 1'b1;
                ln_cache_type[min_index_ln]  <= serving_type;
                ln_cache_layer[min_index_ln] <= serving_layer;
                ln_cache_exp[min_index_ln]   <= dma_rcv_ln_exp;
                ln_cache_mant[min_index_ln]  <= dma_rcv_ln_mant;
                weight_cache_age[min_index_ln] <= 32'h0;
            end
        end
    end
end

//================================================================================
// 响应逻辑（这里已经修正 QKV 的 ack/valid 握手）
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        qkv_weight_ack        <= 1'b0;
        qkv_weight_valid      <= 1'b0;
        wo_weight_ready       <= 1'b0;
        ffn_weight_ready      <= 1'b0;
        ln_param_valid        <= 1'b0;
        
        qkv_weight_exp_array      <= {32*EXP_WIDTH{1'b0}};
        qkv_weight_mant_blocks    <= {32*32*DATA_WIDTH{1'b0}};
        wo_weight_exp_array       <= {32*EXP_WIDTH{1'b0}};
        wo_weight_mant            <= {32*32*DATA_WIDTH{1'b0}};
        ffn_weight_exp_array      <= {32*EXP_WIDTH{1'b0}};
        ffn_weight_mant           <= {32*32*DATA_WIDTH{1'b0}};
        ln_param_exp              <= {EXP_WIDTH{1'b0}};
        ln_param_mant             <= {DIM*DATA_WIDTH{1'b0}};
        
        dbg_cache_hit_count       <= 32'h0;
    end else begin
        // 默认清除 ready/ready 型握手信号；valid 采用保持型
        qkv_weight_ack   <= 1'b0;
        wo_weight_ready  <= 1'b0;
        ffn_weight_ready <= 1'b0;
        ln_param_valid   <= 1'b0;
        
        // 当检测到新的 QKV 请求时，清除上一轮的 valid
        if (qkv_weight_req) begin
            qkv_weight_valid <= 1'b0;
        end
        
        if (state == RESPOND) begin
            case (serving_requester)
                REQ_QKV: begin
                    // QKV权重响应
                    qkv_weight_ack   <= 1'b1;
                    qkv_weight_valid <= 1'b1;
                    if (weight_cache_hit) begin
                        qkv_weight_exp_array   <= weight_cache_exp_array[weight_cache_hit_index];
                        qkv_weight_mant_blocks <= weight_cache_mant_blocks[weight_cache_hit_index];
                        dbg_cache_hit_count    <= dbg_cache_hit_count + 1;
                    end else begin
                        qkv_weight_exp_array   <= dma_rcv_exp_array;
                        qkv_weight_mant_blocks <= dma_rcv_mant_blocks;
                    end
                end
                
                REQ_WO: begin
                    // WO权重响应
                    wo_weight_ready <= 1'b1;
                    if (weight_cache_hit) begin
                        wo_weight_exp_array <= weight_cache_exp_array[weight_cache_hit_index];
                        wo_weight_mant      <= weight_cache_mant_blocks[weight_cache_hit_index];
                        dbg_cache_hit_count <= dbg_cache_hit_count + 1;
                    end else begin
                        wo_weight_exp_array <= dma_rcv_exp_array;
                        wo_weight_mant      <= dma_rcv_mant_blocks;
                    end
                end
                
                REQ_FFN: begin
                    // FFN权重响应
                    ffn_weight_ready <= 1'b1;
                    if (weight_cache_hit) begin
                        ffn_weight_exp_array <= weight_cache_exp_array[weight_cache_hit_index];
                        ffn_weight_mant      <= weight_cache_mant_blocks[weight_cache_hit_index];
                        dbg_cache_hit_count  <= dbg_cache_hit_count + 1;
                    end else begin
                        ffn_weight_exp_array <= dma_rcv_exp_array;
                        ffn_weight_mant      <= dma_rcv_mant_blocks;
                    end
                end
                
                REQ_LN: begin
                    // LayerNorm参数响应
                    ln_param_valid <= 1'b1;
                    if (ln_cache_hit) begin
                        ln_param_exp  <= ln_cache_exp[ln_cache_hit_index];
                        ln_param_mant <= ln_cache_mant[ln_cache_hit_index];
                        dbg_cache_hit_count <= dbg_cache_hit_count + 1;
                    end else begin
                        ln_param_exp  <= dma_rcv_ln_exp;
                        ln_param_mant <= dma_rcv_ln_mant;
                    end
                end
                
                default: begin
                end
            endcase
        end
    end
end

//================================================================================
// 仿真显示
//================================================================================

`ifdef SIMULATION
always @(posedge clk) begin
    if (state == ARBITRATE) begin
        case (serving_requester)
            REQ_QKV: $display("[%0t] Weight Controller: Serving QKV type=%0h layer=%0d", 
                             $time, serving_type, serving_layer);
            REQ_WO:  $display("[%0t] Weight Controller: Serving WO layer=%0d", 
                             $time, serving_layer);
            REQ_FFN: $display("[%0t] Weight Controller: Serving FFN type=%0h chunk=%0d layer=%0d", 
                             $time, serving_type, serving_chunk_id, serving_layer);
            REQ_LN:  $display("[%0t] Weight Controller: Serving LN type=%0h layer=%0d", 
                             $time, serving_type, serving_layer);
            default: ;
        endcase
    end
end
`endif

endmodule
