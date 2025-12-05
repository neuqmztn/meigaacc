`timescale 1ns / 1ps

//================================================================================
// Gradient Buffer v2 - 单写口 + 单读口（对齐 Weight Update Engine）
//
// 功能：
//   - 存储 5 个 layer 的梯度：
//       * Layer 0-3: NUM_TOKENS × DIM_SMALL (8)
//       * Layer 4  : NUM_TOKENS × DIM_LARGE (32)
//   - 写入：统一 1 路写口（来自单一 GCU 核心 + layer_id）
//   - 读取：统一 1 路标量读口，握手接口与 weight_update_engine 的 grad_rd_* 完全一致
//
// 说明：
//   - Q4.12 梯度位宽 = DATA_WIDTH = Q412_WIDTH。
//   - 内部使用 5 组二维扁平数组，分别存 0~4 层。
//   - 映射方式：
//       * 小层(0..3)：addr_small = token*DIM_SMALL + dim = {token,3'b000} + dim
//       * 大层(4)   ：addr_large = token*DIM_LARGE + dim = {token,5'b00000} + dim
//================================================================================
module gradient_buffer_v2 #(
    parameter NUM_TOKENS        = 641,
    parameter DIM_SMALL         = 8,    // Layer 0-3
    parameter DIM_LARGE         = 32,   // Layer 4
    parameter DATA_WIDTH        = 16,   // Q4.12
    parameter TOKEN_ADDR_WIDTH  = 10,   // 支持 0..640
    parameter DIM_ADDR_WIDTH    = 5     // 支持 0..31
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire                         clk,
    input  wire                         rst_n,

    //==========================================================================
    // 统一写口（来自 GCU）
    //==========================================================================
    input  wire                         wr_en,
    input  wire [2:0]                   wr_layer_id,      // 0..4
    input  wire [TOKEN_ADDR_WIDTH-1:0]  wr_token_id,      // 0..NUM_TOKENS-1
    input  wire [DIM_ADDR_WIDTH-1:0]    wr_dim_id,        // 0..7/31
    input  wire [DATA_WIDTH-1:0]        wr_data,          // Q4.12

    //==========================================================================
    // 梯度读取接口（直接对齐 weight_update_engine 的 grad_rd_*）
    //==========================================================================
    input  wire                         grad_rd_req,      // 读请求（1周期脉冲）
    input  wire [2:0]                   grad_rd_layer_id, // 层 ID (0..4)
    input  wire [TOKEN_ADDR_WIDTH-1:0]  grad_rd_token_id, // Token ID
    input  wire [DIM_ADDR_WIDTH-1:0]    grad_rd_dim_id,   // 维度 ID
    output reg                          grad_rd_valid,    // 读数据有效
    output reg  [DATA_WIDTH-1:0]        grad_rd_data      // 梯度数据(Q4.12)
);

    //==========================================================================
    // 层大小与地址宽度
    //==========================================================================
    // Layer 0-3: NUM_TOKENS × DIM_SMALL
    localparam integer SMALL_LAYER_SIZE   = NUM_TOKENS * DIM_SMALL;
    localparam integer SMALL_ADDR_WIDTH   = 13;   // 5128 < 2^13

    // Layer 4: NUM_TOKENS × DIM_LARGE
    localparam integer LARGE_LAYER_SIZE   = NUM_TOKENS * DIM_LARGE;
    localparam integer LARGE_ADDR_WIDTH   = 15;   // 20512 < 2^15

    //==========================================================================
    // 存储阵列（提示综合成 BRAM）
    //==========================================================================
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] grad_l0 [0:SMALL_LAYER_SIZE-1];
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] grad_l1 [0:SMALL_LAYER_SIZE-1];
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] grad_l2 [0:SMALL_LAYER_SIZE-1];
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] grad_l3 [0:SMALL_LAYER_SIZE-1];
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] grad_l4 [0:LARGE_LAYER_SIZE-1];

    // 写地址
    reg [SMALL_ADDR_WIDTH-1:0] wr_addr_small;
    reg [LARGE_ADDR_WIDTH-1:0] wr_addr_large;

    // 读地址（寄存一拍）
    reg [2:0]                  rd_layer_id_reg;
    reg [SMALL_ADDR_WIDTH-1:0] rd_addr_small_reg;
    reg [LARGE_ADDR_WIDTH-1:0] rd_addr_large_reg;

    //==========================================================================
    // 写地址组合逻辑
    //==========================================================================
    always @(*) begin
        // DIM_SMALL = 8 → token*8 = {token,3'b000}
        wr_addr_small = {wr_token_id, 3'b000} + wr_dim_id[2:0];
        // DIM_LARGE = 32 → token*32 = {token,5'b00000}
        wr_addr_large = {wr_token_id, 5'b00000} + wr_dim_id;
    end

    //==========================================================================
    // 写入逻辑（同步写）
    //==========================================================================
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 一般不需要复位 BRAM 内容，可选清零
            // for (i = 0; i < SMALL_LAYER_SIZE; i = i + 1) begin
            //     grad_l0[i] <= {DATA_WIDTH{1'b0}};
            //     grad_l1[i] <= {DATA_WIDTH{1'b0}};
            //     grad_l2[i] <= {DATA_WIDTH{1'b0}};
            //     grad_l3[i] <= {DATA_WIDTH{1'b0}};
            // end
            // for (i = 0; i < LARGE_LAYER_SIZE; i = i + 1) begin
            //     grad_l4[i] <= {DATA_WIDTH{1'b0}};
            // end
        end else if (wr_en) begin
            case (wr_layer_id)
                3'd0: begin
                    if (wr_addr_small < SMALL_LAYER_SIZE)
                        grad_l0[wr_addr_small] <= wr_data;
                end
                3'd1: begin
                    if (wr_addr_small < SMALL_LAYER_SIZE)
                        grad_l1[wr_addr_small] <= wr_data;
                end
                3'd2: begin
                    if (wr_addr_small < SMALL_LAYER_SIZE)
                        grad_l2[wr_addr_small] <= wr_data;
                end
                3'd3: begin
                    if (wr_addr_small < SMALL_LAYER_SIZE)
                        grad_l3[wr_addr_small] <= wr_data;
                end
                3'd4: begin
                    if (wr_addr_large < LARGE_LAYER_SIZE)
                        grad_l4[wr_addr_large] <= wr_data;
                end
                default: begin
                    // do nothing
                end
            endcase
        end
    end

    //==========================================================================
    // 读地址寄存：grad_rd_req 拉高 → 锁存 layer_id 和地址
    //   注意：这里实现的是 "请求后一拍返回" 的 1-cycle latency 协议
    //==========================================================================
    wire [SMALL_ADDR_WIDTH-1:0] rd_addr_small_next;
    wire [LARGE_ADDR_WIDTH-1:0] rd_addr_large_next;

    assign rd_addr_small_next = {grad_rd_token_id, 3'b000} + grad_rd_dim_id[2:0];
    assign rd_addr_large_next = {grad_rd_token_id, 5'b00000} + grad_rd_dim_id;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_layer_id_reg   <= 3'd0;
            rd_addr_small_reg <= {SMALL_ADDR_WIDTH{1'b0}};
            rd_addr_large_reg <= {LARGE_ADDR_WIDTH{1'b0}};
        end else if (grad_rd_req) begin
            rd_layer_id_reg   <= grad_rd_layer_id;
            rd_addr_small_reg <= rd_addr_small_next;
            rd_addr_large_reg <= rd_addr_large_next;
        end
    end

    //==========================================================================
    // 读数据与 valid 产生：
    //   - 当 cycle(n) grad_rd_req=1 → cycle(n+1) grad_rd_valid=1，data 输出
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            grad_rd_valid <= 1'b0;
            grad_rd_data  <= {DATA_WIDTH{1'b0}};
        end else begin
            // 默认拉低 valid
            grad_rd_valid <= 1'b0;

            // 对上一拍锁存的地址进行读取
            case (rd_layer_id_reg)
                3'd0: begin
                    if (rd_addr_small_reg < SMALL_LAYER_SIZE) begin
                        grad_rd_data  <= grad_l0[rd_addr_small_reg];
                        grad_rd_valid <= 1'b1;
                    end
                end
                3'd1: begin
                    if (rd_addr_small_reg < SMALL_LAYER_SIZE) begin
                        grad_rd_data  <= grad_l1[rd_addr_small_reg];
                        grad_rd_valid <= 1'b1;
                    end
                end
                3'd2: begin
                    if (rd_addr_small_reg < SMALL_LAYER_SIZE) begin
                        grad_rd_data  <= grad_l2[rd_addr_small_reg];
                        grad_rd_valid <= 1'b1;
                    end
                end
                3'd3: begin
                    if (rd_addr_small_reg < SMALL_LAYER_SIZE) begin
                        grad_rd_data  <= grad_l3[rd_addr_small_reg];
                        grad_rd_valid <= 1'b1;
                    end
                end
                3'd4: begin
                    if (rd_addr_large_reg < LARGE_LAYER_SIZE) begin
                        grad_rd_data  <= grad_l4[rd_addr_large_reg];
                        grad_rd_valid <= 1'b1;
                    end
                end
                default: begin
                    grad_rd_data  <= {DATA_WIDTH{1'b0}};
                    grad_rd_valid <= 1'b0;
                end
            endcase
        end
    end

endmodule
