`timescale 1ns / 1ps

//================================================================================
// Gradient Buffer - 梯度缓存
//
// 功能说明：
// 临时存储5个GCU计算出的梯度，供权重更新引擎读取
//
// 存储结构：
// • Layer 0-3: 641 tokens × 8 dim = 5128 gradients × 16bit
// • Layer 4:   641 tokens × 32 dim = 20512 gradients × 16bit
//
// 接口设计：
// • 写入：5个GCU独立写入端口
// • 读取：权重更新引擎统一读取
//
// 作者：MEIGA Team
// 日期：2025-01-18
// 版本：v1.0 (Phase 2/3)
//================================================================================

module gradient_buffer #(
    parameter NUM_TOKENS    = 641,
    parameter DIM_SMALL     = 8,             // Layer 0-3
    parameter DIM_LARGE     = 32,            // Layer 4
    parameter DATA_WIDTH    = 16,
    parameter TOKEN_ADDR_WIDTH = 10,
    parameter DIM_ADDR_WIDTH = 5
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // GCU 0 写入接口 (Layer 0)
    //==========================================================================
    input  wire        gcu0_wr_en,
    input  wire [TOKEN_ADDR_WIDTH-1:0] gcu0_token_addr,
    input  wire [DIM_ADDR_WIDTH-1:0]   gcu0_dim_addr,
    input  wire [DATA_WIDTH-1:0]       gcu0_data,
    
    //==========================================================================
    // GCU 1 写入接口 (Layer 1)
    //==========================================================================
    input  wire        gcu1_wr_en,
    input  wire [TOKEN_ADDR_WIDTH-1:0] gcu1_token_addr,
    input  wire [DIM_ADDR_WIDTH-1:0]   gcu1_dim_addr,
    input  wire [DATA_WIDTH-1:0]       gcu1_data,
    
    //==========================================================================
    // GCU 2 写入接口 (Layer 2)
    //==========================================================================
    input  wire        gcu2_wr_en,
    input  wire [TOKEN_ADDR_WIDTH-1:0] gcu2_token_addr,
    input  wire [DIM_ADDR_WIDTH-1:0]   gcu2_dim_addr,
    input  wire [DATA_WIDTH-1:0]       gcu2_data,
    
    //==========================================================================
    // GCU 3 写入接口 (Layer 3)
    //==========================================================================
    input  wire        gcu3_wr_en,
    input  wire [TOKEN_ADDR_WIDTH-1:0] gcu3_token_addr,
    input  wire [DIM_ADDR_WIDTH-1:0]   gcu3_dim_addr,
    input  wire [DATA_WIDTH-1:0]       gcu3_data,
    
    //==========================================================================
    // GCU 4 写入接口 (Layer 4)
    //==========================================================================
    input  wire        gcu4_wr_en,
    input  wire [TOKEN_ADDR_WIDTH-1:0] gcu4_token_addr,
    input  wire [DIM_ADDR_WIDTH-1:0]   gcu4_dim_addr,
    input  wire [DATA_WIDTH-1:0]       gcu4_data,
    
    //==========================================================================
    // 权重更新引擎读取接口（4路并行读取）
    //==========================================================================
    input  wire        rd_en,
    input  wire [2:0]  rd_layer_id,          // 0-4
    input  wire [9:0]  rd_addr,              // 线性地址
    output reg  [63:0] rd_data,              // 4个梯度并行输出
    output reg         rd_valid,
    
    //==========================================================================
    // 简单读取接口（单路，用于分类头权重更新）
    //==========================================================================
    input  wire        rd_simple_en,
    input  wire [2:0]  rd_simple_layer,      // 层ID (0-4)
    input  wire [TOKEN_ADDR_WIDTH-1:0] rd_simple_token,
    input  wire [DIM_ADDR_WIDTH-1:0]   rd_simple_dim,
    output reg  [DATA_WIDTH-1:0]       rd_simple_data,
    output reg         rd_simple_valid
);

//================================================================================
// 存储器实例化
//================================================================================
// Layer 0-3: 每层 641×8 = 5128个梯度
reg [DATA_WIDTH-1:0] grad_l0 [0:NUM_TOKENS*DIM_SMALL-1];
reg [DATA_WIDTH-1:0] grad_l1 [0:NUM_TOKENS*DIM_SMALL-1];
reg [DATA_WIDTH-1:0] grad_l2 [0:NUM_TOKENS*DIM_SMALL-1];
reg [DATA_WIDTH-1:0] grad_l3 [0:NUM_TOKENS*DIM_SMALL-1];

// Layer 4: 641×32 = 20512个梯度
reg [DATA_WIDTH-1:0] grad_l4 [0:NUM_TOKENS*DIM_LARGE-1];

//================================================================================
// 写入逻辑：计算线性地址
//================================================================================
// 线性地址 = token_id × dim + dim_id
wire [12:0] wr_addr_0 = {gcu0_token_addr, gcu0_dim_addr[2:0]};
wire [12:0] wr_addr_1 = {gcu1_token_addr, gcu1_dim_addr[2:0]};
wire [12:0] wr_addr_2 = {gcu2_token_addr, gcu2_dim_addr[2:0]};
wire [12:0] wr_addr_3 = {gcu3_token_addr, gcu3_dim_addr[2:0]};
wire [14:0] wr_addr_4 = {gcu4_token_addr, gcu4_dim_addr[4:0]};

// GCU 0 写入
always @(posedge clk) begin
    if (gcu0_wr_en) begin
        grad_l0[wr_addr_0] <= gcu0_data;
    end
end

// GCU 1 写入
always @(posedge clk) begin
    if (gcu1_wr_en) begin
        grad_l1[wr_addr_1] <= gcu1_data;
    end
end

// GCU 2 写入
always @(posedge clk) begin
    if (gcu2_wr_en) begin
        grad_l2[wr_addr_2] <= gcu2_data;
    end
end

// GCU 3 写入
always @(posedge clk) begin
    if (gcu3_wr_en) begin
        grad_l3[wr_addr_3] <= gcu3_data;
    end
end

// GCU 4 写入
always @(posedge clk) begin
    if (gcu4_wr_en) begin
        grad_l4[wr_addr_4] <= gcu4_data;
    end
end

//================================================================================
// 读取逻辑：4路并行读取
//================================================================================
// 将rd_addr映射到4个连续的梯度
wire [12:0] base_addr_small = {rd_addr, 2'b00};  // rd_addr × 4
wire [14:0] base_addr_large = {rd_addr, 2'b00};

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_data <= 64'd0;
        rd_valid <= 1'b0;
    end else if (rd_en) begin
        case (rd_layer_id)
            3'd0: begin
                rd_data <= {grad_l0[base_addr_small + 13'd3],
                           grad_l0[base_addr_small + 13'd2],
                           grad_l0[base_addr_small + 13'd1],
                           grad_l0[base_addr_small]};
                rd_valid <= 1'b1;
            end
            
            3'd1: begin
                rd_data <= {grad_l1[base_addr_small + 13'd3],
                           grad_l1[base_addr_small + 13'd2],
                           grad_l1[base_addr_small + 13'd1],
                           grad_l1[base_addr_small]};
                rd_valid <= 1'b1;
            end
            
            3'd2: begin
                rd_data <= {grad_l2[base_addr_small + 13'd3],
                           grad_l2[base_addr_small + 13'd2],
                           grad_l2[base_addr_small + 13'd1],
                           grad_l2[base_addr_small]};
                rd_valid <= 1'b1;
            end
            
            3'd3: begin
                rd_data <= {grad_l3[base_addr_small + 13'd3],
                           grad_l3[base_addr_small + 13'd2],
                           grad_l3[base_addr_small + 13'd1],
                           grad_l3[base_addr_small]};
                rd_valid <= 1'b1;
            end
            
            3'd4: begin
                rd_data <= {grad_l4[base_addr_large + 15'd3],
                           grad_l4[base_addr_large + 15'd2],
                           grad_l4[base_addr_large + 15'd1],
                           grad_l4[base_addr_large]};
                rd_valid <= 1'b1;
            end
            
            default: begin
                rd_data <= 64'd0;
                rd_valid <= 1'b0;
            end
        endcase
    end else begin
        rd_valid <= 1'b0;
    end
end

//================================================================================
// 简单读取逻辑（单路，用于分类头权重更新）
//================================================================================
// 计算线性地址
wire [13:0] rd_simple_addr_small = rd_simple_token * DIM_SMALL + rd_simple_dim;
wire [14:0] rd_simple_addr_large = rd_simple_token * DIM_LARGE + rd_simple_dim;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_simple_data <= {DATA_WIDTH{1'b0}};
        rd_simple_valid <= 1'b0;
    end else if (rd_simple_en) begin
        case (rd_simple_layer)
            3'd0: begin
                rd_simple_data <= grad_l0[rd_simple_addr_small];
                rd_simple_valid <= 1'b1;
            end
            3'd1: begin
                rd_simple_data <= grad_l1[rd_simple_addr_small];
                rd_simple_valid <= 1'b1;
            end
            3'd2: begin
                rd_simple_data <= grad_l2[rd_simple_addr_small];
                rd_simple_valid <= 1'b1;
            end
            3'd3: begin
                rd_simple_data <= grad_l3[rd_simple_addr_small];
                rd_simple_valid <= 1'b1;
            end
            3'd4: begin
                rd_simple_data <= grad_l4[rd_simple_addr_large];
                rd_simple_valid <= 1'b1;
            end
            default: begin
                rd_simple_data <= {DATA_WIDTH{1'b0}};
                rd_simple_valid <= 1'b0;
            end
        endcase
    end else begin
        rd_simple_valid <= 1'b0;
    end
end

endmodule