`timescale 1ns / 1ps

//================================================================================
// Sidenet Layer Output Buffer - v4.2 (统一按token读取)
//
// 版本历史：
// v1.0: 基础LOB，BFP存储
// v2.0: 新增5路GCU并行读接口  
// v3.0: 集成BFP→Q4.12转换器
// v4.0: 扩展支持Layer 4 (32维)
// v4.1: 尝试统一接口（部分按维度）
// v4.2: 完全统一！所有训练读接口都按token读取 + 向量化转换 ← 🎯 正确设计
//
// 主要改进（v4.2）：
// ✅ DFA读接口改为按token读取（GCU 0-4）
// ✅ 5个向量化转换器（GCU 0-3用8维，GCU 4用32维）
// ✅ 分类头也按token读取（已经是了）
// ✅ 性能提升：641次读取 vs 20,512次读取（32倍提速）
// ✅ 接口简洁统一
//
// 核心思想：
// - 推理：按token读，输出BFP
// - 训练：按token读，输出Q4.12向量
//
// 作者：MEIGA Team
// 日期：2025-11-19
// 版本：v4.2 (推荐使用)
//================================================================================

module sidenet_layer_output_buffer #(
    //==========================================================================
    // 基本参数
    //==========================================================================
    parameter TOKEN_NUM      = 641,         // Token数量
    parameter DIM_L0_L3      = 8,           // Layer 0-3维度
    parameter DIM_L4         = 32,          // Layer 4维度（Expand）
    parameter DATA_WIDTH     = 16,          // 尾数位宽 (16-bit BFP)
    parameter EXP_WIDTH      = 8,           // 指数位宽
    parameter NUM_LAYERS     = 5,           // 层数 (Layer 0-4)
    parameter ADDR_WIDTH     = 10,          // Token地址位宽
    parameter LAYER_WIDTH    = 3            // Layer地址位宽 (0-4)
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 写接口 - 自动适配维度
    //==========================================================================
    input  wire                             wr_en,
    input  wire [LAYER_WIDTH-1:0]           wr_layer_id,   // 0-4
    input  wire [ADDR_WIDTH-1:0]            wr_token_id,
    input  wire [EXP_WIDTH-1:0]             wr_exp,
    input  wire [DIM_L4*DATA_WIDTH-1:0]     wr_mant,       // 最大32维（512-bit）
    output wire                             wr_ready,
    output wire                             wr_error,
    
    //==========================================================================
    // 推理读接口 - 输出BFP格式
    // 按token读取，一次返回整个token
    //==========================================================================
    input  wire                             rd_infer_en,
    input  wire [LAYER_WIDTH-1:0]           rd_infer_layer_id,
    input  wire [ADDR_WIDTH-1:0]            rd_infer_token_id,
    output reg  [EXP_WIDTH-1:0]             rd_infer_exp,
    output reg  [DIM_L4*DATA_WIDTH-1:0]     rd_infer_mant,
    output reg                              rd_infer_valid,
    output wire                             rd_infer_error,
    
    //==========================================================================
    // DFA训练读接口 - GCU 0-3 (Layer 0-3)
    // 按token读取 + 向量化BFP→Q4.12转换
    // 一次返回整个token的8维Q4.12向量
    //==========================================================================
    input  wire                             rd_dfa0_en,
    input  wire [ADDR_WIDTH-1:0]            rd_dfa0_token_id,
    output wire [DIM_L0_L3*DATA_WIDTH-1:0]  rd_dfa0_data_q412,  // 8×16-bit = 128-bit
    output wire                             rd_dfa0_valid,
    
    input  wire                             rd_dfa1_en,
    input  wire [ADDR_WIDTH-1:0]            rd_dfa1_token_id,
    output wire [DIM_L0_L3*DATA_WIDTH-1:0]  rd_dfa1_data_q412,  // 8×16-bit = 128-bit
    output wire                             rd_dfa1_valid,
    
    input  wire                             rd_dfa2_en,
    input  wire [ADDR_WIDTH-1:0]            rd_dfa2_token_id,
    output wire [DIM_L0_L3*DATA_WIDTH-1:0]  rd_dfa2_data_q412,  // 8×16-bit = 128-bit
    output wire                             rd_dfa2_valid,
    
    input  wire                             rd_dfa3_en,
    input  wire [ADDR_WIDTH-1:0]            rd_dfa3_token_id,
    output wire [DIM_L0_L3*DATA_WIDTH-1:0]  rd_dfa3_data_q412,  // 8×16-bit = 128-bit
    output wire                             rd_dfa3_valid,
    
    //==========================================================================
    // DFA训练读接口 - GCU 4 (Layer 4)
    // 按token读取 + 向量化BFP→Q4.12转换
    // 一次返回整个token的32维Q4.12向量
    //==========================================================================
    input  wire                             rd_dfa4_en,
    input  wire [ADDR_WIDTH-1:0]            rd_dfa4_token_id,
    output wire [DIM_L4*DATA_WIDTH-1:0]     rd_dfa4_data_q412,  // 32×16-bit = 512-bit
    output wire                             rd_dfa4_valid,
    
    //==========================================================================
    // 分类头读接口
    // 固定读取Token 0 (CLS token)
    // 输出32维Q4.12向量
    //==========================================================================
    input  wire                             rd_cls_en,
    output wire [DIM_L4*DATA_WIDTH-1:0]     rd_cls_data_q412,   // 32×16-bit = 512-bit
    output wire                             rd_cls_valid,
    
    //==========================================================================
    // 调试和监控接口
    //==========================================================================
    output wire [31:0]                      dbg_wr_count_l0,
    output wire [31:0]                      dbg_wr_count_l1,
    output wire [31:0]                      dbg_wr_count_l2,
    output wire [31:0]                      dbg_wr_count_l3,
    output wire [31:0]                      dbg_wr_count_l4,
    output wire [31:0]                      dbg_infer_rd_count,
    output wire [31:0]                      dbg_dfa_rd_count,
    output wire [31:0]                      dbg_cls_rd_count,
    output wire [31:0]                      dbg_wr_error_count,
    output wire [31:0]                      dbg_rd_error_count
);

//================================================================================
// 本地参数定义
//================================================================================
localparam MANT_WIDTH_L0_L3 = DIM_L0_L3 * DATA_WIDTH;  // 128 bits
localparam MANT_WIDTH_L4    = DIM_L4 * DATA_WIDTH;     // 512 bits

//================================================================================
// 内部信号声明
//================================================================================
reg        wr_addr_error;
reg        wr_layer_error;
reg        rd_infer_addr_error;
reg        rd_infer_layer_error;
reg        rd_dfa0_addr_error;
reg        rd_dfa1_addr_error;
reg        rd_dfa2_addr_error;
reg        rd_dfa3_addr_error;
reg        rd_dfa4_addr_error;

reg [31:0] wr_count [0:NUM_LAYERS-1];
reg [31:0] infer_rd_count;
reg [31:0] dfa_rd_count;
reg [31:0] cls_rd_count;
reg [31:0] wr_error_count;
reg [31:0] rd_error_count;

//================================================================================
// 存储Bank声明
//================================================================================
// Bank 0-3: 8维
reg [EXP_WIDTH-1:0]      exp_bank0  [0:TOKEN_NUM-1];
reg [MANT_WIDTH_L0_L3-1:0] mant_bank0 [0:TOKEN_NUM-1];

reg [EXP_WIDTH-1:0]      exp_bank1  [0:TOKEN_NUM-1];
reg [MANT_WIDTH_L0_L3-1:0] mant_bank1 [0:TOKEN_NUM-1];

reg [EXP_WIDTH-1:0]      exp_bank2  [0:TOKEN_NUM-1];
reg [MANT_WIDTH_L0_L3-1:0] mant_bank2 [0:TOKEN_NUM-1];

reg [EXP_WIDTH-1:0]      exp_bank3  [0:TOKEN_NUM-1];
reg [MANT_WIDTH_L0_L3-1:0] mant_bank3 [0:TOKEN_NUM-1];

// Bank 4: 32维
reg [EXP_WIDTH-1:0]      exp_bank4  [0:TOKEN_NUM-1];
reg [MANT_WIDTH_L4-1:0]  mant_bank4 [0:TOKEN_NUM-1];

//================================================================================
// BFP→Q4.12转换器的中间信号（向量化）
//================================================================================
// GCU 0-3 (8维向量)
reg [EXP_WIDTH-1:0]       dfa0_bfp_exp, dfa1_bfp_exp, dfa2_bfp_exp, dfa3_bfp_exp;
reg [MANT_WIDTH_L0_L3-1:0] dfa0_bfp_mant, dfa1_bfp_mant, dfa2_bfp_mant, dfa3_bfp_mant;
reg                       dfa0_bfp_valid, dfa1_bfp_valid, dfa2_bfp_valid, dfa3_bfp_valid;

// GCU 4 (32维向量)
reg [EXP_WIDTH-1:0]       dfa4_bfp_exp;
reg [MANT_WIDTH_L4-1:0]   dfa4_bfp_mant;
reg                       dfa4_bfp_valid;

// 分类头 (32维向量)
reg [EXP_WIDTH-1:0]       cls_bfp_exp;
reg [MANT_WIDTH_L4-1:0]   cls_bfp_mant;
reg                       cls_bfp_valid;

//================================================================================
// 写接口逻辑
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin:xie
        integer i;
        for (i = 0; i < TOKEN_NUM; i = i + 1) begin
            exp_bank0[i]  <= {EXP_WIDTH{1'b0}};
            exp_bank1[i]  <= {EXP_WIDTH{1'b0}};
            exp_bank2[i]  <= {EXP_WIDTH{1'b0}};
            exp_bank3[i]  <= {EXP_WIDTH{1'b0}};
            exp_bank4[i]  <= {EXP_WIDTH{1'b0}};
            mant_bank0[i] <= {MANT_WIDTH_L0_L3{1'b0}};
            mant_bank1[i] <= {MANT_WIDTH_L0_L3{1'b0}};
            mant_bank2[i] <= {MANT_WIDTH_L0_L3{1'b0}};
            mant_bank3[i] <= {MANT_WIDTH_L0_L3{1'b0}};
            mant_bank4[i] <= {MANT_WIDTH_L4{1'b0}};
        end
    end 
    else if (wr_en && !wr_error) begin
        case (wr_layer_id)
            3'd0: begin
                exp_bank0[wr_token_id]  <= wr_exp;
                mant_bank0[wr_token_id] <= wr_mant[MANT_WIDTH_L0_L3-1:0];
            end
            3'd1: begin
                exp_bank1[wr_token_id]  <= wr_exp;
                mant_bank1[wr_token_id] <= wr_mant[MANT_WIDTH_L0_L3-1:0];
            end
            3'd2: begin
                exp_bank2[wr_token_id]  <= wr_exp;
                mant_bank2[wr_token_id] <= wr_mant[MANT_WIDTH_L0_L3-1:0];
            end
            3'd3: begin
                exp_bank3[wr_token_id]  <= wr_exp;
                mant_bank3[wr_token_id] <= wr_mant[MANT_WIDTH_L0_L3-1:0];
            end
            3'd4: begin
                exp_bank4[wr_token_id]  <= wr_exp;
                mant_bank4[wr_token_id] <= wr_mant[MANT_WIDTH_L4-1:0];
            end
        endcase
    end
end

assign wr_ready = rst_n;
assign wr_error = wr_addr_error || wr_layer_error;

//================================================================================
// 推理读接口（BFP格式，按token读取）
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_infer_exp   <= {EXP_WIDTH{1'b0}};
        rd_infer_mant  <= {MANT_WIDTH_L4{1'b0}};
        rd_infer_valid <= 1'b0;
    end 
    else if (rd_infer_en && !rd_infer_error) begin
        case (rd_infer_layer_id)
            3'd0: begin
                rd_infer_exp  <= exp_bank0[rd_infer_token_id];
                rd_infer_mant <= {{(MANT_WIDTH_L4-MANT_WIDTH_L0_L3){1'b0}}, 
                                  mant_bank0[rd_infer_token_id]};
            end
            3'd1: begin
                rd_infer_exp  <= exp_bank1[rd_infer_token_id];
                rd_infer_mant <= {{(MANT_WIDTH_L4-MANT_WIDTH_L0_L3){1'b0}}, 
                                  mant_bank1[rd_infer_token_id]};
            end
            3'd2: begin
                rd_infer_exp  <= exp_bank2[rd_infer_token_id];
                rd_infer_mant <= {{(MANT_WIDTH_L4-MANT_WIDTH_L0_L3){1'b0}}, 
                                  mant_bank2[rd_infer_token_id]};
            end
            3'd3: begin
                rd_infer_exp  <= exp_bank3[rd_infer_token_id];
                rd_infer_mant <= {{(MANT_WIDTH_L4-MANT_WIDTH_L0_L3){1'b0}}, 
                                  mant_bank3[rd_infer_token_id]};
            end
            3'd4: begin
                rd_infer_exp  <= exp_bank4[rd_infer_token_id];
                rd_infer_mant <= mant_bank4[rd_infer_token_id];
            end
        endcase
        rd_infer_valid <= 1'b1;
    end 
    else begin
        rd_infer_valid <= 1'b0;
    end
end

assign rd_infer_error = rd_infer_addr_error || rd_infer_layer_error;

//================================================================================
// DFA训练读接口 - GCU 0 (按token读取，8维向量)
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dfa0_bfp_exp   <= {EXP_WIDTH{1'b0}};
        dfa0_bfp_mant  <= {MANT_WIDTH_L0_L3{1'b0}};
        dfa0_bfp_valid <= 1'b0;
    end 
    else if (rd_dfa0_en && !rd_dfa0_addr_error) begin
        dfa0_bfp_exp   <= exp_bank0[rd_dfa0_token_id];
        dfa0_bfp_mant  <= mant_bank0[rd_dfa0_token_id];  // 完整8维
        dfa0_bfp_valid <= 1'b1;
    end 
    else begin
        dfa0_bfp_valid <= 1'b0;
    end
end

//================================================================================
// DFA训练读接口 - GCU 1 (按token读取，8维向量)
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dfa1_bfp_exp   <= {EXP_WIDTH{1'b0}};
        dfa1_bfp_mant  <= {MANT_WIDTH_L0_L3{1'b0}};
        dfa1_bfp_valid <= 1'b0;
    end 
    else if (rd_dfa1_en && !rd_dfa1_addr_error) begin
        dfa1_bfp_exp   <= exp_bank1[rd_dfa1_token_id];
        dfa1_bfp_mant  <= mant_bank1[rd_dfa1_token_id];  // 完整8维
        dfa1_bfp_valid <= 1'b1;
    end 
    else begin
        dfa1_bfp_valid <= 1'b0;
    end
end

//================================================================================
// DFA训练读接口 - GCU 2 (按token读取，8维向量)
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dfa2_bfp_exp   <= {EXP_WIDTH{1'b0}};
        dfa2_bfp_mant  <= {MANT_WIDTH_L0_L3{1'b0}};
        dfa2_bfp_valid <= 1'b0;
    end 
    else if (rd_dfa2_en && !rd_dfa2_addr_error) begin
        dfa2_bfp_exp   <= exp_bank2[rd_dfa2_token_id];
        dfa2_bfp_mant  <= mant_bank2[rd_dfa2_token_id];  // 完整8维
        dfa2_bfp_valid <= 1'b1;
    end 
    else begin
        dfa2_bfp_valid <= 1'b0;
    end
end

//================================================================================
// DFA训练读接口 - GCU 3 (按token读取，8维向量)
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dfa3_bfp_exp   <= {EXP_WIDTH{1'b0}};
        dfa3_bfp_mant  <= {MANT_WIDTH_L0_L3{1'b0}};
        dfa3_bfp_valid <= 1'b0;
    end 
    else if (rd_dfa3_en && !rd_dfa3_addr_error) begin
        dfa3_bfp_exp   <= exp_bank3[rd_dfa3_token_id];
        dfa3_bfp_mant  <= mant_bank3[rd_dfa3_token_id];  // 完整8维
        dfa3_bfp_valid <= 1'b1;
    end 
    else begin
        dfa3_bfp_valid <= 1'b0;
    end
end

//================================================================================
// DFA训练读接口 - GCU 4 (按token读取，32维向量)
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dfa4_bfp_exp   <= {EXP_WIDTH{1'b0}};
        dfa4_bfp_mant  <= {MANT_WIDTH_L4{1'b0}};
        dfa4_bfp_valid <= 1'b0;
    end 
    else if (rd_dfa4_en && !rd_dfa4_addr_error) begin
        dfa4_bfp_exp   <= exp_bank4[rd_dfa4_token_id];
        dfa4_bfp_mant  <= mant_bank4[rd_dfa4_token_id];  // 完整32维
        dfa4_bfp_valid <= 1'b1;
    end 
    else begin
        dfa4_bfp_valid <= 1'b0;
    end
end

//================================================================================
// 分类头读接口 (固定读取Token 0，32维向量)
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cls_bfp_exp   <= {EXP_WIDTH{1'b0}};
        cls_bfp_mant  <= {MANT_WIDTH_L4{1'b0}};
        cls_bfp_valid <= 1'b0;
    end 
    else if (rd_cls_en) begin
        // 固定读取Token 0 (CLS token)
        cls_bfp_exp   <= exp_bank4[10'd0];
        cls_bfp_mant  <= mant_bank4[10'd0];  // 完整32维
        cls_bfp_valid <= 1'b1;
    end 
    else begin
        cls_bfp_valid <= 1'b0;
    end
end

//================================================================================
// BFP→Q4.12向量化转换器实例化
//================================================================================

// GCU 0-3: 8维向量转换器
bfp_to_q412_vector_converter #(
    .DIM            (DIM_L0_L3),
    .BFP_MANT_WIDTH (DATA_WIDTH),
    .BFP_EXP_WIDTH  (EXP_WIDTH),
    .Q412_WIDTH     (DATA_WIDTH),
    .Q412_FRAC_BITS (12)
) u_conv_dfa0 (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (dfa0_bfp_valid),
    .bfp_exp    (dfa0_bfp_exp),
    .bfp_mant   (dfa0_bfp_mant),
    .q412_data  (rd_dfa0_data_q412),
    .valid_out  (rd_dfa0_valid),
    .overflow   (),
    .underflow  ()
);

bfp_to_q412_vector_converter #(
    .DIM            (DIM_L0_L3),
    .BFP_MANT_WIDTH (DATA_WIDTH),
    .BFP_EXP_WIDTH  (EXP_WIDTH),
    .Q412_WIDTH     (DATA_WIDTH),
    .Q412_FRAC_BITS (12)
) u_conv_dfa1 (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (dfa1_bfp_valid),
    .bfp_exp    (dfa1_bfp_exp),
    .bfp_mant   (dfa1_bfp_mant),
    .q412_data  (rd_dfa1_data_q412),
    .valid_out  (rd_dfa1_valid),
    .overflow   (),
    .underflow  ()
);

bfp_to_q412_vector_converter #(
    .DIM            (DIM_L0_L3),
    .BFP_MANT_WIDTH (DATA_WIDTH),
    .BFP_EXP_WIDTH  (EXP_WIDTH),
    .Q412_WIDTH     (DATA_WIDTH),
    .Q412_FRAC_BITS (12)
) u_conv_dfa2 (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (dfa2_bfp_valid),
    .bfp_exp    (dfa2_bfp_exp),
    .bfp_mant   (dfa2_bfp_mant),
    .q412_data  (rd_dfa2_data_q412),
    .valid_out  (rd_dfa2_valid),
    .overflow   (),
    .underflow  ()
);

bfp_to_q412_vector_converter #(
    .DIM            (DIM_L0_L3),
    .BFP_MANT_WIDTH (DATA_WIDTH),
    .BFP_EXP_WIDTH  (EXP_WIDTH),
    .Q412_WIDTH     (DATA_WIDTH),
    .Q412_FRAC_BITS (12)
) u_conv_dfa3 (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (dfa3_bfp_valid),
    .bfp_exp    (dfa3_bfp_exp),
    .bfp_mant   (dfa3_bfp_mant),
    .q412_data  (rd_dfa3_data_q412),
    .valid_out  (rd_dfa3_valid),
    .overflow   (),
    .underflow  ()
);

// GCU 4: 32维向量转换器
bfp_to_q412_vector_converter #(
    .DIM            (DIM_L4),
    .BFP_MANT_WIDTH (DATA_WIDTH),
    .BFP_EXP_WIDTH  (EXP_WIDTH),
    .Q412_WIDTH     (DATA_WIDTH),
    .Q412_FRAC_BITS (12)
) u_conv_dfa4 (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (dfa4_bfp_valid),
    .bfp_exp    (dfa4_bfp_exp),
    .bfp_mant   (dfa4_bfp_mant),
    .q412_data  (rd_dfa4_data_q412),
    .valid_out  (rd_dfa4_valid),
    .overflow   (),
    .underflow  ()
);

// 分类头: 32维向量转换器
bfp_to_q412_vector_converter #(
    .DIM            (DIM_L4),
    .BFP_MANT_WIDTH (DATA_WIDTH),
    .BFP_EXP_WIDTH  (EXP_WIDTH),
    .Q412_WIDTH     (DATA_WIDTH),
    .Q412_FRAC_BITS (12)
) u_conv_cls (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_in   (cls_bfp_valid),
    .bfp_exp    (cls_bfp_exp),
    .bfp_mant   (cls_bfp_mant),
    .q412_data  (rd_cls_data_q412),
    .valid_out  (rd_cls_valid),
    .overflow   (),
    .underflow  ()
);

//================================================================================
// 错误检测
//================================================================================
always @(*) begin
    // 写地址/层ID错误
    wr_addr_error  = (wr_token_id >= TOKEN_NUM);
    wr_layer_error = (wr_layer_id >= NUM_LAYERS);
    
    // 推理读错误
    rd_infer_addr_error  = (rd_infer_token_id >= TOKEN_NUM);
    rd_infer_layer_error = (rd_infer_layer_id >= NUM_LAYERS);
    
    // DFA读地址错误
    rd_dfa0_addr_error = (rd_dfa0_token_id >= TOKEN_NUM);
    rd_dfa1_addr_error = (rd_dfa1_token_id >= TOKEN_NUM);
    rd_dfa2_addr_error = (rd_dfa2_token_id >= TOKEN_NUM);
    rd_dfa3_addr_error = (rd_dfa3_token_id >= TOKEN_NUM);
    rd_dfa4_addr_error = (rd_dfa4_token_id >= TOKEN_NUM);
end

//================================================================================
// 计数器（调试用）
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin:ji
        integer i;
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            wr_count[i] <= 32'd0;
        end
        wr_error_count <= 32'd0;
        infer_rd_count <= 32'd0;
        dfa_rd_count   <= 32'd0;
        cls_rd_count   <= 32'd0;
        rd_error_count <= 32'd0;
    end else begin
        if (wr_en && !wr_error) begin
            wr_count[wr_layer_id] <= wr_count[wr_layer_id] + 1;
        end
        if (wr_en && wr_error) begin
            wr_error_count <= wr_error_count + 1;
        end
        if (rd_infer_en && rd_infer_valid) begin
            infer_rd_count <= infer_rd_count + 1;
        end
        if (rd_dfa0_valid || rd_dfa1_valid || rd_dfa2_valid || 
            rd_dfa3_valid || rd_dfa4_valid) begin
            dfa_rd_count <= dfa_rd_count + 1;
        end
        if (rd_cls_valid) begin
            cls_rd_count <= cls_rd_count + 1;
        end
        if ((rd_infer_en && rd_infer_error) ||
            (rd_dfa0_en && rd_dfa0_addr_error) ||
            (rd_dfa1_en && rd_dfa1_addr_error) ||
            (rd_dfa2_en && rd_dfa2_addr_error) ||
            (rd_dfa3_en && rd_dfa3_addr_error) ||
            (rd_dfa4_en && rd_dfa4_addr_error)) begin
            rd_error_count <= rd_error_count + 1;
        end
    end
end

//================================================================================
// 调试接口连接
//================================================================================
assign dbg_wr_count_l0 = wr_count[0];
assign dbg_wr_count_l1 = wr_count[1];
assign dbg_wr_count_l2 = wr_count[2];
assign dbg_wr_count_l3 = wr_count[3];
assign dbg_wr_count_l4 = wr_count[4];
assign dbg_infer_rd_count = infer_rd_count;
assign dbg_dfa_rd_count = dfa_rd_count;
assign dbg_cls_rd_count = cls_rd_count;
assign dbg_wr_error_count = wr_error_count;
assign dbg_rd_error_count = rd_error_count;

endmodule