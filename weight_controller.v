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
    
    // 配置接口
    input  wire [1:0] current_layer_id,
    input  wire [DRAM_ADDR_WIDTH-1:0] weight_base_addr,
    
    // QKV权重接口
    input  wire qkv_weight_req,
    input  wire [1:0] qkv_weight_type,
    output reg  qkv_weight_ack,
    output reg  qkv_weight_valid,
    output wire [32*EXP_WIDTH-1:0] qkv_weight_exp_array,
    output wire [32*32*DATA_WIDTH-1:0] qkv_weight_mant_blocks, 
    
    // WO权重接口
    input  wire wo_weight_req,
    output reg  wo_weight_ready,
    output wire [32*EXP_WIDTH-1:0] wo_weight_exp_array,
    output wire [32*32*DATA_WIDTH-1:0] wo_weight_mant,
    
    // FFN权重接口
    input  wire ffn_weight_req,
    input  wire [1:0] ffn_weight_type,
    input  wire [1:0] ffn_weight_chunk_id,
    output reg  ffn_weight_ready,
    output wire [32*EXP_WIDTH-1:0] ffn_weight_exp_array,
    output wire [32*32*DATA_WIDTH-1:0] ffn_weight_mant,
    
    // LayerNorm参数接口
    input  wire ln_param_req,
    input  wire [1:0] ln_param_type,
    output reg  ln_param_valid,
    output wire [EXP_WIDTH-1:0] ln_param_exp,
    output wire [DIM*DATA_WIDTH-1:0] ln_param_mant,
    
    // DMA接口
    output reg  dma_req_valid,
    output reg  [DRAM_ADDR_WIDTH-1:0] dma_req_addr,
    output reg  [7:0] dma_req_burst_len,
    input  wire dma_req_ready,
    
    input  wire dma_rsp_valid,
    input  wire [DRAM_DATA_WIDTH-1:0] dma_rsp_data,
    input  wire dma_rsp_last,
    output reg  dma_rsp_ready,
    
    // Debug信号
    output reg  [3:0]  dbg_state,
    output reg  [31:0] dbg_cache_hit_count,
    output reg  [31:0] dbg_cache_miss_count,
    output reg  [31:0] dbg_dma_req_count
);

    //================================================================================
    // 常量定义
    //================================================================================
    localparam TYPE_WQ          = 4'h0;
    localparam TYPE_WK          = 4'h1;
    localparam TYPE_WV          = 4'h2;
    localparam TYPE_WO          = 4'h3;
    localparam TYPE_W1          = 4'h4;
    localparam TYPE_W2          = 4'h5;
    localparam TYPE_LN1_GAMMA   = 4'h8;
    localparam TYPE_LN1_BETA    = 4'h9;
    localparam TYPE_LN2_GAMMA   = 4'hA;
    localparam TYPE_LN2_BETA    = 4'hB;

    localparam IDLE     = 4'd0;
    localparam ARBITRATE = 4'd1;
    localparam CHECK_CACHE = 4'd2;
    localparam CALC_ADDR = 4'd3;
    localparam DMA_REQUEST = 4'd4;
    localparam DMA_RECEIVE = 4'd6;
    localparam RESPOND     = 4'd8;

    localparam LAYER_STRIDE      = 32'h4000;
    localparam WEIGHT_SIZE       = 32'd1056; 
    localparam BURST_LEN_MATRIX  = 8'd32;    // 33 transfers (0..32)
    localparam BURST_LEN_LN      = 8'd1;     // 2 transfers (0..1)

    //================================================================================
    // 内部信号定义
    //================================================================================
    reg [3:0] state, next_state;
    
    // 当前请求信息
    reg [2:0] serving_requester; 
    reg [3:0] serving_type;
    reg [1:0] serving_chunk_id;
    reg [1:0] serving_layer;
    wire is_serving_matrix = (serving_type <= TYPE_W2);

    // Cache 管理
    reg [1:0] target_cache_index;
    
    // DMA 控制
    reg [DRAM_ADDR_WIDTH-1:0] dma_target_addr;
    reg [7:0]                 dma_burst_len_reg;
    reg [5:0]                 dma_rcv_counter;
    
    reg [31:0] calculated_offset;

    integer i;

    // Tags 管理
    reg [3:0] weight_cache_type     [0:CACHE_ENTRIES-1];
    reg [1:0] weight_cache_chunk_id [0:CACHE_ENTRIES-1];
    reg [1:0] weight_cache_layer    [0:CACHE_ENTRIES-1];
    reg       weight_cache_valid    [0:CACHE_ENTRIES-1];
    reg [31:0] weight_cache_age     [0:CACHE_ENTRIES-1];

    reg [3:0] ln_cache_type     [0:CACHE_ENTRIES-1];
    reg [1:0] ln_cache_layer    [0:CACHE_ENTRIES-1];
    reg       ln_cache_valid    [0:CACHE_ENTRIES-1];
    reg [31:0] ln_cache_age     [0:CACHE_ENTRIES-1];

    //================================================================================
    // 实例化 32 个独立的 LUTRAM Bank (用于 Weight Matrix Mantissa)
    //================================================================================
    reg [31:0] bank_we;
    always @(*) begin
        bank_we = 32'd0;
        if (state == DMA_RECEIVE && dma_rsp_valid && dma_rsp_ready && is_serving_matrix) begin
            // 优化：使用位移操作替代动态索引，更加稳健
            if (dma_rcv_counter >= 1 && dma_rcv_counter <= 32) begin
                bank_we = (32'd1 << (dma_rcv_counter - 1));
            end
        end
    end

    wire [DRAM_DATA_WIDTH-1:0] bank_dout [0:31];
    
    
    generate
    genvar k;
        for (k = 0; k < 32; k = k + 1) begin : ram_banks
            wc_internal_lutram #(
                .BANK_DATA_WIDTH(256),
                .BANK_ADDR_WIDTH(2)
            ) u_bank (
                .clk(clk),
                .we(bank_we[k]),
                .waddr(target_cache_index),
                .din(dma_rsp_data),
                .raddr(target_cache_index),
                .dout(bank_dout[k])
            );
        end
    endgenerate

    //================================================================================
    // 非矩阵数据存储 (Exp 和 LN)
    // [重要修复]：移除 ram_style = "distributed"，因为下面有部分位宽写入操作
    //================================================================================
    // 指数存储 (虽然是全字读写，但为了统一性可以使用寄存器，或者保持 distributed)
    (* ram_style = "distributed" *) reg [32*EXP_WIDTH-1:0] exp_ram [0:CACHE_ENTRIES-1];
    
    // LN 数据存储 - [修复] 移除 ram_style，使用寄存器以支持 partial write
    reg [EXP_WIDTH-1:0]      ln_cache_exp  [0:CACHE_ENTRIES-1];
    reg [DIM*DATA_WIDTH-1:0] ln_cache_mant [0:CACHE_ENTRIES-1];

    //================================================================================
    // Cache Hit 检测
    //================================================================================
    reg [1:0] hit_index;
    reg       hit_flag;
    reg [1:0] replace_index;
    reg [31:0] max_age;
    reg       found_empty;

    always @(*) begin
        hit_flag = 1'b0;
        hit_index = 2'b00;
        if (is_serving_matrix) begin
            for (i=0; i<CACHE_ENTRIES; i=i+1) begin
                if (weight_cache_valid[i] && weight_cache_type[i] == serving_type && 
                    weight_cache_chunk_id[i] == serving_chunk_id && weight_cache_layer[i] == serving_layer) begin
                    hit_flag = 1'b1;
                    hit_index = i[1:0];
                end
            end
        end else begin
            for (i=0; i<CACHE_ENTRIES; i=i+1) begin
                if (ln_cache_valid[i] && ln_cache_type[i] == serving_type && ln_cache_layer[i] == serving_layer) begin
                    hit_flag = 1'b1;
                    hit_index = i[1:0];
                end
            end
        end

        // LRU
        replace_index = 2'b00;
        max_age = 0;
        found_empty = 1'b0;
        
        for (i=0; i<CACHE_ENTRIES; i=i+1) begin
            if (is_serving_matrix) begin
                if (!weight_cache_valid[i]) begin
                    if (!found_empty) begin
                        replace_index = i[1:0];
                        found_empty = 1'b1;
                    end
                end else if (!found_empty && weight_cache_age[i] >= max_age) begin
                    max_age = weight_cache_age[i];
                    replace_index = i[1:0];
                end
            end else begin
                if (!ln_cache_valid[i]) begin
                     if (!found_empty) begin
                        replace_index = i[1:0];
                        found_empty = 1'b1;
                     end
                end else if (!found_empty && ln_cache_age[i] >= max_age) begin
                    max_age = ln_cache_age[i];
                    replace_index = i[1:0];
                end
            end
        end
    end

    //================================================================================
    // 状态机
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

    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (qkv_weight_req || wo_weight_req || ffn_weight_req || ln_param_req)
                    next_state = ARBITRATE;
            end
            ARBITRATE:   next_state = CHECK_CACHE;
            CHECK_CACHE: begin
                if (hit_flag) next_state = RESPOND;
                else          next_state = CALC_ADDR;
            end
            CALC_ADDR:   next_state = DMA_REQUEST;
            DMA_REQUEST: if (dma_req_valid && dma_req_ready) next_state = DMA_RECEIVE;
            DMA_RECEIVE: if (dma_rsp_valid && dma_rsp_ready && dma_rsp_last) next_state = RESPOND;
            RESPOND:     next_state = IDLE;
            default:     next_state = IDLE;
        endcase
    end

    //================================================================================
    // 逻辑控制
    //================================================================================
    // 仲裁与请求捕获
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            serving_requester <= 0;
            serving_type <= 0;
            serving_chunk_id <= 0;
            serving_layer <= 0;
        end else if (state == IDLE && next_state == ARBITRATE) begin
            serving_layer <= current_layer_id;
            if (qkv_weight_req) begin
                serving_requester <= 1;
                serving_chunk_id <= 0;
                case(qkv_weight_type)
                    0: serving_type <= TYPE_WQ;
                    1: serving_type <= TYPE_WK;
                    2: serving_type <= TYPE_WV;
                    default: serving_type <= TYPE_WQ;
                endcase
            end else if (wo_weight_req) begin
                serving_requester <= 2;
                serving_type <= TYPE_WO;
                serving_chunk_id <= 0;
            end else if (ffn_weight_req) begin
                serving_requester <= 3;
                serving_type <= (ffn_weight_type == 0) ? TYPE_W1 : TYPE_W2;
                serving_chunk_id <= ffn_weight_chunk_id;
            end else if (ln_param_req) begin
                serving_requester <= 4;
                serving_chunk_id <= 0;
                case(ln_param_type)
                    0: serving_type <= TYPE_LN1_GAMMA;
                    1: serving_type <= TYPE_LN1_BETA;
                    2: serving_type <= TYPE_LN2_GAMMA;
                    3: serving_type <= TYPE_LN2_BETA;
                    default: serving_type <= TYPE_LN1_GAMMA;
                endcase
            end
        end
    end

    // Cache Index 锁定
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            target_cache_index <= 0;
            dbg_cache_hit_count <= 0;
            dbg_cache_miss_count <= 0;
        end else if (state == CHECK_CACHE) begin
            if (hit_flag) begin
                target_cache_index <= hit_index;
                dbg_cache_hit_count <= dbg_cache_hit_count + 1;
            end else begin
                target_cache_index <= replace_index;
                dbg_cache_miss_count <= dbg_cache_miss_count + 1;
            end
        end
    end

    // 地址计算 (组合逻辑)
    always @(*) begin
        case(serving_type)
            TYPE_WQ: calculated_offset = 32'h0000;
            TYPE_WK: calculated_offset = 32'h0420;
            TYPE_WV: calculated_offset = 32'h0840;
            TYPE_WO: calculated_offset = 32'h0C60;
            TYPE_W1: calculated_offset = 32'h1080 + (serving_chunk_id * WEIGHT_SIZE);
            TYPE_W2: calculated_offset = 32'h2100 + (serving_chunk_id * WEIGHT_SIZE);
            TYPE_LN1_GAMMA: calculated_offset = 32'h3180;
            TYPE_LN1_BETA:  calculated_offset = 32'h31A1;
            TYPE_LN2_GAMMA: calculated_offset = 32'h31C2;
            TYPE_LN2_BETA:  calculated_offset = 32'h31E3;
            default: calculated_offset = 0;
        endcase
    end

    // DMA 准备
    always @(posedge clk) begin
        if (state == CALC_ADDR) begin
            dma_target_addr <= weight_base_addr + (serving_layer * LAYER_STRIDE) + calculated_offset;
            dma_burst_len_reg <= is_serving_matrix ? BURST_LEN_MATRIX : BURST_LEN_LN;
        end
    end

    // DMA 请求发送
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_req_valid <= 0;
            dma_req_addr <= 0;
            dma_req_burst_len <= 0;
            dbg_dma_req_count <= 0;
        end else if (state == DMA_REQUEST) begin
            dma_req_valid <= 1;
            dma_req_addr <= dma_target_addr;
            dma_req_burst_len <= dma_burst_len_reg;
            if (dma_req_ready) begin
                dma_req_valid <= 0;
                dbg_dma_req_count <= dbg_dma_req_count + 1;
            end
        end else begin
            dma_req_valid <= 0;
        end
    end

    // DMA 数据接收与写入
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_rsp_ready <= 0;
            dma_rcv_counter <= 0;
            for(i=0; i<CACHE_ENTRIES; i=i+1) begin
                weight_cache_valid[i] <= 0;
                ln_cache_valid[i] <= 0;
            end
            // 复位 Memory 并非必须，但在 FPGA 中通常不复位大数组
        end else begin
            dma_rsp_ready <= (state == DMA_RECEIVE);
            if (state == DMA_REQUEST) dma_rcv_counter <= 0;

            if (state == DMA_RECEIVE && dma_rsp_valid && dma_rsp_ready) begin
                dma_rcv_counter <= dma_rcv_counter + 1;
                
                if (is_serving_matrix) begin
                    if (dma_rcv_counter == 0) begin
                        // 写入指数
                        exp_ram[target_cache_index] <= dma_rsp_data;
                    end
                    // 尾数由 u_bank 写入
                end else begin
                    // LN 数据：由于这里使用了切片写入，必须保证 ln_cache_mant 是寄存器
                    if (dma_rcv_counter == 0) begin
                        ln_cache_exp[target_cache_index] <= dma_rsp_data[7:0];
                        ln_cache_mant[target_cache_index][247:0] <= dma_rsp_data[255:8];
                    end else begin
                        ln_cache_mant[target_cache_index][255:248] <= dma_rsp_data[7:0];
                    end
                end

                if (dma_rsp_last) begin
                    if (is_serving_matrix) begin
                        weight_cache_valid[target_cache_index] <= 1'b1;
                        weight_cache_type[target_cache_index] <= serving_type;
                        weight_cache_chunk_id[target_cache_index] <= serving_chunk_id;
                        weight_cache_layer[target_cache_index] <= serving_layer;
                    end else begin
                        ln_cache_valid[target_cache_index] <= 1'b1;
                        ln_cache_type[target_cache_index] <= serving_type;
                        ln_cache_layer[target_cache_index] <= serving_layer;
                    end
                end
            end
        end
    end

    // Age 更新
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for(i=0; i<CACHE_ENTRIES; i=i+1) begin
                weight_cache_age[i] <= 0;
                ln_cache_age[i] <= 0;
            end
        end else if (state == RESPOND && next_state == IDLE) begin
            for(i=0; i<CACHE_ENTRIES; i=i+1) begin
                if (weight_cache_valid[i]) weight_cache_age[i] <= weight_cache_age[i] + 1;
                if (is_serving_matrix && i == target_cache_index) weight_cache_age[i] <= 0;

                if (ln_cache_valid[i]) ln_cache_age[i] <= ln_cache_age[i] + 1;
                if (!is_serving_matrix && i == target_cache_index) ln_cache_age[i] <= 0;
            end
        end
    end

    // 输出逻辑
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            qkv_weight_ack <= 0; qkv_weight_valid <= 0;
            wo_weight_ready <= 0; ffn_weight_ready <= 0; ln_param_valid <= 0;
        end else begin
            qkv_weight_ack <= (state==IDLE && next_state==ARBITRATE && qkv_weight_req);
            qkv_weight_valid <= 0; wo_weight_ready <= 0; ffn_weight_ready <= 0; ln_param_valid <= 0;

            if (state == RESPOND) begin
                case(serving_requester)
                    1: qkv_weight_valid <= 1;
                    2: wo_weight_ready <= 1;
                    3: ffn_weight_ready <= 1;
                    4: ln_param_valid <= 1;
                endcase
            end
        end
    end

    wire [32*32*DATA_WIDTH-1:0] full_mant_matrix;
    wire [32*EXP_WIDTH-1:0]     full_exp_array;

    genvar m;
    generate
        for (m = 0; m < 32; m = m + 1) begin : out_concat
            assign full_mant_matrix[m*256 +: 256] = bank_dout[m];
        end
    endgenerate

    assign full_exp_array = exp_ram[target_cache_index];

    assign qkv_weight_mant_blocks = full_mant_matrix;
    assign qkv_weight_exp_array   = full_exp_array;
    assign wo_weight_mant         = full_mant_matrix;
    assign wo_weight_exp_array    = full_exp_array;
    assign ffn_weight_mant        = full_mant_matrix;
    assign ffn_weight_exp_array   = full_exp_array;

    assign ln_param_exp           = ln_cache_exp[target_cache_index];
    assign ln_param_mant          = ln_cache_mant[target_cache_index];

endmodule