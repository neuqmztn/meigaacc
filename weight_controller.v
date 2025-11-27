//================================================================================
// Weight Controller - 统一权重管理模块
//
// 版本：v2.2_2001
// 日期：2025-11-27
//
// 更新内容（v2.2 → v2.2_2001）：
// - 完全符合Verilog-2001标准
// - 移除所有SystemVerilog特性（automatic变量、bit类型等）
// - 补全所有省略的代码
// - 所有临时变量改为模块级reg声明
//
// 作者：Claude (Assisted)
//================================================================================

`timescale 1ns / 1ps

module weight_controller #(
    parameter NUM_LAYERS        = 4,
    parameter DIM               = 32,
    parameter DATA_WIDTH        = 8,
    parameter EXP_WIDTH         = 8,
    parameter DRAM_ADDR_WIDTH   = 32,
    parameter DRAM_DATA_WIDTH   = 256,
    parameter CACHE_ENTRIES     = 4
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 配置接口
    //================================================================================
    input  wire [1:0] current_layer_id,  // 固定为2位，支持4层
    input  wire [DRAM_ADDR_WIDTH-1:0] weight_base_addr,
    
    //================================================================================
    // QKV权重接口（数组格式）
    //================================================================================
    input  wire qkv_weight_req,
    input  wire [1:0] qkv_weight_type,
    output reg  qkv_weight_ack,
    output reg  qkv_weight_valid,
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
    input  wire [1:0] ffn_weight_type,
    input  wire [1:0] ffn_weight_chunk_id,
    output reg  ffn_weight_ready,
    output reg  [32*EXP_WIDTH-1:0] ffn_weight_exp_array,
    output reg  [32*32*DATA_WIDTH-1:0] ffn_weight_mant,
    
    //================================================================================
    // LayerNorm参数接口（单指数格式）
    //================================================================================
    input  wire ln_param_req,
    input  wire [1:0] ln_param_type,
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
localparam REQ_NONE = 3'b000;
localparam REQ_QKV  = 3'b001;
localparam REQ_WO   = 3'b010;
localparam REQ_FFN  = 3'b011;
localparam REQ_LN   = 3'b100;

// 状态机
localparam IDLE           = 4'd0;
localparam ARBITRATE      = 4'd1;
localparam CHECK_CACHE    = 4'd2;
localparam CALC_ADDR      = 4'd3;
localparam DMA_REQUEST    = 4'd4;
localparam DMA_WAIT       = 4'd5;
localparam DMA_RECEIVE    = 4'd6;
localparam CACHE_UPDATE   = 4'd7;
localparam RESPOND        = 4'd8;

// DRAM布局相关常量
localparam LAYER_STRIDE   = 32'h4000;  // 16KB per layer
localparam WEIGHT_SIZE    = 32'd1056;  // 0x420 Bytes

// DMA 突发长度
localparam BURST_LEN_MATRIX = 8'd32;   // 33 transfers (AXI: N-1)
localparam BURST_LEN_LN     = 8'd1;    // 2 transfers (AXI: N-1)

//================================================================================
// 状态机和内部信号
//================================================================================

reg [3:0] state, next_state;

// 当前服务的请求
reg [2:0] serving_requester;
reg [3:0] serving_type;
reg [1:0] serving_chunk_id;
reg [1:0] serving_layer;

// 判断当前请求是否为权重矩阵类型
wire is_serving_matrix;
assign is_serving_matrix = (serving_type <= TYPE_W2);

//================================================================================
// 权重矩阵缓存 (Weight Cache)
//================================================================================

reg [3:0] weight_cache_type       [0:CACHE_ENTRIES-1];
reg [1:0] weight_cache_chunk_id   [0:CACHE_ENTRIES-1];
reg [1:0] weight_cache_layer      [0:CACHE_ENTRIES-1];
reg       weight_cache_valid      [0:CACHE_ENTRIES-1];

// Cache Data
reg [32*EXP_WIDTH-1:0]        weight_cache_exp_array    [0:CACHE_ENTRIES-1];
reg [32*32*DATA_WIDTH-1:0]    weight_cache_mant_blocks  [0:CACHE_ENTRIES-1];

// LRU Age
reg [31:0] weight_cache_age [0:CACHE_ENTRIES-1];

//================================================================================
// LayerNorm参数缓存 (LN Cache)
//================================================================================

reg [3:0] ln_cache_type       [0:CACHE_ENTRIES-1];
reg [1:0] ln_cache_layer      [0:CACHE_ENTRIES-1];
reg       ln_cache_valid      [0:CACHE_ENTRIES-1];

reg [EXP_WIDTH-1:0]           ln_cache_exp  [0:CACHE_ENTRIES-1];
reg [DIM*DATA_WIDTH-1:0]      ln_cache_mant [0:CACHE_ENTRIES-1];

// LRU Age
reg [31:0] ln_cache_age [0:CACHE_ENTRIES-1];

//================================================================================
// DMA接收缓冲与控制
//================================================================================

reg [32*EXP_WIDTH-1:0]        dma_rcv_exp_array;
reg [32*32*DATA_WIDTH-1:0]    dma_rcv_mant_blocks;
reg [EXP_WIDTH-1:0]           dma_rcv_ln_exp;
reg [DIM*DATA_WIDTH-1:0]      dma_rcv_ln_mant;

reg [5:0] dma_rcv_counter;

//================================================================================
// 地址计算相关
//================================================================================

reg [DRAM_ADDR_WIDTH-1:0] dma_target_addr;
reg [7:0]                 dma_burst_len_reg;

// 地址计算用临时变量（Verilog-2001: 声明在模块级）
reg [DRAM_ADDR_WIDTH-1:0] calc_layer_base_addr;
reg [DRAM_ADDR_WIDTH-1:0] calc_offset;

//================================================================================
// 缓存管理信号
//================================================================================

reg weight_cache_hit;
reg [1:0] weight_cache_hit_index;  // 固定2位，支持4个entries
reg ln_cache_hit;
reg [1:0] ln_cache_hit_index;

reg [1:0] replace_index;

//================================================================================
// LRU查找用临时变量（Verilog-2001: 声明在模块级）
//================================================================================

reg [31:0] lru_max_age;
reg [1:0]  lru_idx;
reg        lru_found_invalid;

//================================================================================
// RESPOND阶段数据源选择用临时变量
//================================================================================

reg [1:0] data_source_idx;
reg       cache_hit_flag;

//================================================================================
// 通用循环变量
//================================================================================

integer i;

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
            if (is_serving_matrix) begin
                if (weight_cache_hit) begin
                    next_state = RESPOND;
                end else begin
                    next_state = CALC_ADDR;
                end
            end else begin
                if (ln_cache_hit) begin
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
            if (dma_req_valid && dma_req_ready) begin
                next_state = DMA_WAIT;
            end
        end
        
        DMA_WAIT: begin
            if (dma_rsp_valid && dma_rsp_ready) begin
                if (dma_rsp_last) begin
                    next_state = CACHE_UPDATE;
                end else begin
                    next_state = DMA_RECEIVE;
                end
            end
        end
        
        DMA_RECEIVE: begin
            if (dma_rsp_valid && dma_rsp_ready && dma_rsp_last) begin
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
        serving_layer <= current_layer_id;
        
        if (qkv_weight_req) begin
            serving_requester <= REQ_QKV;
            case (qkv_weight_type)
                2'b00: serving_type <= TYPE_WQ;
                2'b01: serving_type <= TYPE_WK;
                2'b10: serving_type <= TYPE_WV;
                default: serving_type <= TYPE_WQ;
            endcase
            serving_chunk_id <= 2'b00;
        end else if (wo_weight_req) begin
            serving_requester <= REQ_WO;
            serving_type      <= TYPE_WO;
            serving_chunk_id  <= 2'b00;
        end else if (ffn_weight_req) begin
            serving_requester <= REQ_FFN;
            case (ffn_weight_type)
                2'b00: serving_type <= TYPE_W1;
                2'b01: serving_type <= TYPE_W2;
                default: serving_type <= TYPE_W1;
            endcase
            serving_chunk_id <= ffn_weight_chunk_id;
        end else if (ln_param_req) begin
            serving_requester <= REQ_LN;
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
    
    if (is_serving_matrix) begin
        for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
            if (weight_cache_valid[i] &&
                weight_cache_type[i] == serving_type &&
                weight_cache_chunk_id[i] == serving_chunk_id &&
                weight_cache_layer[i] == serving_layer) begin
                weight_cache_hit = 1'b1;
                weight_cache_hit_index = i[1:0];
            end
        end
    end else begin
        for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
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
        dma_target_addr <= {DRAM_ADDR_WIDTH{1'b0}};
        dma_burst_len_reg <= 8'd0;
        calc_layer_base_addr <= {DRAM_ADDR_WIDTH{1'b0}};
        calc_offset <= {DRAM_ADDR_WIDTH{1'b0}};
    end else if (state == CALC_ADDR) begin
        // 计算层的基地址
        calc_layer_base_addr <= weight_base_addr + (serving_layer * LAYER_STRIDE);
        
        // 计算权重偏移量
        case (serving_type)
            TYPE_WQ: calc_offset <= 32'h0000;
            TYPE_WK: calc_offset <= 32'h0420;
            TYPE_WV: calc_offset <= 32'h0840;
            TYPE_WO: calc_offset <= 32'h0C60;
            TYPE_W1: begin
                calc_offset <= 32'h1080 + (serving_chunk_id * WEIGHT_SIZE);
            end
            TYPE_W2: begin
                calc_offset <= 32'h2100 + (serving_chunk_id * WEIGHT_SIZE);
            end
            TYPE_LN1_GAMMA: calc_offset <= 32'h3180;
            TYPE_LN1_BETA:  calc_offset <= 32'h31A1;
            TYPE_LN2_GAMMA: calc_offset <= 32'h31C2;
            TYPE_LN2_BETA:  calc_offset <= 32'h31E3;
            default: calc_offset <= 32'h0000;
        endcase
        
        // 合成最终地址
        dma_target_addr <= calc_layer_base_addr + calc_offset;
        
        // 设置突发长度
        if (is_serving_matrix) begin
            dma_burst_len_reg <= BURST_LEN_MATRIX;
        end else begin
            dma_burst_len_reg <= BURST_LEN_LN;
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
        dbg_dma_req_count <= 32'd0;
    end else begin
        if (state == DMA_REQUEST) begin
            dma_req_valid     <= 1'b1;
            dma_req_addr      <= dma_target_addr;
            dma_req_burst_len <= dma_burst_len_reg;
            
            if (dma_req_ready) begin
                dma_req_valid <= 1'b0;
                dbg_dma_req_count <= dbg_dma_req_count + 32'd1;
            end
        end else begin
            dma_req_valid <= 1'b0;
        end
    end
end

//================================================================================
// DMA接收逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dma_rsp_ready        <= 1'b0;
        dma_rcv_counter      <= 6'd0;
        dma_rcv_exp_array    <= {(32*EXP_WIDTH){1'b0}};
        dma_rcv_mant_blocks  <= {(32*32*DATA_WIDTH){1'b0}};
        dma_rcv_ln_exp       <= {EXP_WIDTH{1'b0}};
        dma_rcv_ln_mant      <= {(DIM*DATA_WIDTH){1'b0}};
    end else begin
        dma_rsp_ready <= 1'b0;
        
        // DMA传输开始时重置计数器
        if (state == DMA_REQUEST && next_state == DMA_WAIT) begin
            dma_rcv_counter <= 6'd0;
        end
        
        // 接收数据
        if (state == DMA_WAIT || state == DMA_RECEIVE) begin
            dma_rsp_ready <= 1'b1;
            
            if (dma_rsp_valid && dma_rsp_ready) begin
                if (is_serving_matrix) begin
                    // 权重矩阵: [32个exp (32B)][32x32个mant (1024B)]
                    if (dma_rcv_counter == 6'd0) begin
                        // Transfer 0: Exponents (256bit)
                        dma_rcv_exp_array <= dma_rsp_data[32*EXP_WIDTH-1:0];
                    end else begin
                        // Transfer 1-32: Mantissas
                        // 使用 counter-1 作为索引
                        dma_rcv_mant_blocks[(dma_rcv_counter-6'd1)*DRAM_DATA_WIDTH +: DRAM_DATA_WIDTH] 
                            <= dma_rsp_data;
                    end
                end else begin
                    // LayerNorm: [1个exp (1B)][32个mant (32B)]
                    if (dma_rcv_counter == 6'd0) begin
                        // Transfer 0: Exp + Mant[0:30]
                        dma_rcv_ln_exp <= dma_rsp_data[7:0];
                        dma_rcv_ln_mant[247:0] <= dma_rsp_data[255:8];
                    end else if (dma_rcv_counter == 6'd1) begin
                        // Transfer 1: Mant[31]
                        dma_rcv_ln_mant[255:248] <= dma_rsp_data[7:0];
                    end
                end
                
                // 更新计数器
                if (!dma_rsp_last) begin
                    dma_rcv_counter <= dma_rcv_counter + 6'd1;
                end
            end
        end
    end
end

//================================================================================
// LRU Age 更新逻辑
//================================================================================

wire cache_access_event;
assign cache_access_event = (state == CHECK_CACHE && (weight_cache_hit || ln_cache_hit)) || 
                            (state == CACHE_UPDATE);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
            weight_cache_age[i] <= 32'd0;
            ln_cache_age[i] <= 32'd0;
        end
    end else if (cache_access_event) begin
        // Weight Cache Age Update
        for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
            if (weight_cache_valid[i]) begin
                // 默认增加Age
                weight_cache_age[i] <= weight_cache_age[i] + 32'd1;
                
                // 如果被访问，重置Age为0
                if (is_serving_matrix) begin
                    if (state == CHECK_CACHE && weight_cache_hit && weight_cache_hit_index == i[1:0]) begin
                        weight_cache_age[i] <= 32'd0;
                    end
                    if (state == CACHE_UPDATE && replace_index == i[1:0]) begin
                        weight_cache_age[i] <= 32'd0;
                    end
                end
            end
        end
        
        // LN Cache Age Update
        for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
            if (ln_cache_valid[i]) begin
                ln_cache_age[i] <= ln_cache_age[i] + 32'd1;
                
                if (!is_serving_matrix) begin
                    if (state == CHECK_CACHE && ln_cache_hit && ln_cache_hit_index == i[1:0]) begin
                        ln_cache_age[i] <= 32'd0;
                    end
                    if (state == CACHE_UPDATE && replace_index == i[1:0]) begin
                        ln_cache_age[i] <= 32'd0;
                    end
                end
            end
        end
    end
end

//================================================================================
// LRU替换索引计算
//================================================================================

wire start_cache_update;
assign start_cache_update = (state == DMA_WAIT || state == DMA_RECEIVE) && 
                            (next_state == CACHE_UPDATE);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        replace_index <= 2'b00;
        lru_max_age <= 32'd0;
        lru_idx <= 2'b00;
        lru_found_invalid <= 1'b0;
    end else if (start_cache_update) begin
        // 初始化
        lru_max_age <= 32'd0;
        lru_idx <= 2'b00;
        lru_found_invalid <= 1'b0;
        
        if (is_serving_matrix) begin
            // 查找Weight Cache的LRU索引
            for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
                if (!weight_cache_valid[i]) begin
                    lru_idx <= i[1:0];
                    lru_found_invalid <= 1'b1;
                end else if (!lru_found_invalid && weight_cache_age[i] > lru_max_age) begin
                    lru_max_age <= weight_cache_age[i];
                    lru_idx <= i[1:0];
                end
            end
        end else begin
            // 查找LN Cache的LRU索引
            for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
                if (!ln_cache_valid[i]) begin
                    lru_idx <= i[1:0];
                    lru_found_invalid <= 1'b1;
                end else if (!lru_found_invalid && ln_cache_age[i] > lru_max_age) begin
                    lru_max_age <= ln_cache_age[i];
                    lru_idx <= i[1:0];
                end
            end
        end
        
        replace_index <= lru_idx;
    end
end

//================================================================================
// 缓存数据更新逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 初始化Weight Cache
        for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
            weight_cache_valid[i] <= 1'b0;
            weight_cache_type[i] <= 4'h0;
            weight_cache_chunk_id[i] <= 2'b00;
            weight_cache_layer[i] <= 2'b00;
            weight_cache_exp_array[i] <= {(32*EXP_WIDTH){1'b0}};
            weight_cache_mant_blocks[i] <= {(32*32*DATA_WIDTH){1'b0}};
        end
        
        // 初始化LN Cache
        for (i = 0; i < CACHE_ENTRIES; i = i + 1) begin
            ln_cache_valid[i] <= 1'b0;
            ln_cache_type[i] <= 4'h0;
            ln_cache_layer[i] <= 2'b00;
            ln_cache_exp[i] <= {EXP_WIDTH{1'b0}};
            ln_cache_mant[i] <= {(DIM*DATA_WIDTH){1'b0}};
        end
        
        dbg_cache_miss_count <= 32'd0;
    end else begin
        // 统计Cache Miss
        if (state == CHECK_CACHE && next_state == CALC_ADDR) begin
            dbg_cache_miss_count <= dbg_cache_miss_count + 32'd1;
        end
        
        // 执行缓存更新
        if (state == CACHE_UPDATE) begin
            if (is_serving_matrix) begin
                weight_cache_valid[replace_index] <= 1'b1;
                weight_cache_type[replace_index] <= serving_type;
                weight_cache_chunk_id[replace_index] <= serving_chunk_id;
                weight_cache_layer[replace_index] <= serving_layer;
                weight_cache_exp_array[replace_index] <= dma_rcv_exp_array;
                weight_cache_mant_blocks[replace_index] <= dma_rcv_mant_blocks;
            end else begin
                ln_cache_valid[replace_index] <= 1'b1;
                ln_cache_type[replace_index] <= serving_type;
                ln_cache_layer[replace_index] <= serving_layer;
                ln_cache_exp[replace_index] <= dma_rcv_ln_exp;
                ln_cache_mant[replace_index] <= dma_rcv_ln_mant;
            end
        end
    end
end

//================================================================================
// 响应逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        qkv_weight_ack    <= 1'b0;
        qkv_weight_valid  <= 1'b0;
        wo_weight_ready   <= 1'b0;
        ffn_weight_ready  <= 1'b0;
        ln_param_valid    <= 1'b0;
        
        qkv_weight_exp_array   <= {(32*EXP_WIDTH){1'b0}};
        qkv_weight_mant_blocks <= {(32*32*DATA_WIDTH){1'b0}};
        wo_weight_exp_array    <= {(32*EXP_WIDTH){1'b0}};
        wo_weight_mant         <= {(32*32*DATA_WIDTH){1'b0}};
        ffn_weight_exp_array   <= {(32*EXP_WIDTH){1'b0}};
        ffn_weight_mant        <= {(32*32*DATA_WIDTH){1'b0}};
        ln_param_exp           <= {EXP_WIDTH{1'b0}};
        ln_param_mant          <= {(DIM*DATA_WIDTH){1'b0}};
        
        data_source_idx   <= 2'b00;
        cache_hit_flag    <= 1'b0;
        
        dbg_cache_hit_count <= 32'd0;
    end else begin
        // 默认清除脉冲型握手信号
        qkv_weight_ack   <= 1'b0;
        wo_weight_ready  <= 1'b0;
        ffn_weight_ready <= 1'b0;
        ln_param_valid   <= 1'b0;
        
        // QKV握手处理
        if (state == IDLE && next_state == ARBITRATE && qkv_weight_req) begin
            qkv_weight_ack <= 1'b1;
            qkv_weight_valid <= 1'b0;
        end
        
        // 响应阶段
        if (state == RESPOND) begin
            // 确定数据源
            if (is_serving_matrix) begin
                cache_hit_flag <= weight_cache_hit;
                data_source_idx <= weight_cache_hit ? weight_cache_hit_index : replace_index;
            end else begin
                cache_hit_flag <= ln_cache_hit;
                data_source_idx <= ln_cache_hit ? ln_cache_hit_index : replace_index;
            end
            
            // 统计Cache Hit
            if (cache_hit_flag) begin
                dbg_cache_hit_count <= dbg_cache_hit_count + 32'd1;
            end
            
            // 响应数据
            case (serving_requester)
                REQ_QKV: begin
                    qkv_weight_valid <= 1'b1;
                    qkv_weight_exp_array   <= weight_cache_exp_array[data_source_idx];
                    qkv_weight_mant_blocks <= weight_cache_mant_blocks[data_source_idx];
                end
                
                REQ_WO: begin
                    wo_weight_ready <= 1'b1;
                    wo_weight_exp_array <= weight_cache_exp_array[data_source_idx];
                    wo_weight_mant      <= weight_cache_mant_blocks[data_source_idx];
                end
                
                REQ_FFN: begin
                    ffn_weight_ready <= 1'b1;
                    ffn_weight_exp_array <= weight_cache_exp_array[data_source_idx];
                    ffn_weight_mant      <= weight_cache_mant_blocks[data_source_idx];
                end
                
                REQ_LN: begin
                    ln_param_valid <= 1'b1;
                    ln_param_exp  <= ln_cache_exp[data_source_idx];
                    ln_param_mant <= ln_cache_mant[data_source_idx];
                end
                
                default: begin
                    // Do nothing
                end
            endcase
        end
    end
end


endmodule