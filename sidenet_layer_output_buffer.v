`timescale 1ns / 1ps

module sidenet_layer_output_buffer #(
    //==========================================================================
    // 基本参数
    //==========================================================================
    parameter TOKEN_NUM      = 640,         // Token数量
    parameter DIM_L0_L3      = 8,           // Layer 0-3维度
    parameter DIM_L4         = 32,          // Layer 4维度（Expand）
    parameter DATA_WIDTH     = 16,          // 尾数位宽 (16-bit BFP)
    parameter EXP_WIDTH      = 8,           // 指数位宽
    parameter NUM_LAYERS     = 5,           // 层数 (Layer 0-4)
    parameter ADDR_WIDTH     = 10,          // Token地址位宽
    parameter LAYER_WIDTH    = 4           // Layer地址位宽 (0-4)
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
localparam TOTAL_DEPTH      = TOKEN_NUM * NUM_LAYERS;

// Verilog-2001 版本的 clog2
function integer CLOG2;
    input integer value;
    integer i;
begin
    CLOG2 = 0;
    for (i = value - 1; i > 0; i = i >> 1)
        CLOG2 = CLOG2 + 1;
end
endfunction

localparam TOTAL_ADDR_W = CLOG2(TOTAL_DEPTH);

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

// 计数器
reg [31:0] wr_count       [0:NUM_LAYERS-1];
reg [31:0] wr_error_count;
reg [31:0] infer_rd_count;
reg [31:0] dfa_rd_count;
reg [31:0] cls_rd_count;
reg [31:0] rd_error_count;

// 打包 BRAM：指数 + 尾数
(* ram_style = "block" *) reg [EXP_WIDTH-1:0]      exp_mem  [0:TOTAL_DEPTH-1];
(* ram_style = "block" *) reg [MANT_WIDTH_L4-1:0]  mant_mem [0:TOTAL_DEPTH-1];

// BFP→Q4.12转换器的中间信号（向量化）

// GCU 0-3 (8维向量)
reg [EXP_WIDTH-1:0]           dfa0_bfp_exp, dfa1_bfp_exp, dfa2_bfp_exp, dfa3_bfp_exp;
reg [MANT_WIDTH_L0_L3-1:0]    dfa0_bfp_mant, dfa1_bfp_mant, dfa2_bfp_mant, dfa3_bfp_mant;
reg                           dfa0_bfp_valid, dfa1_bfp_valid, dfa2_bfp_valid, dfa3_bfp_valid;

// GCU 4 (32维向量)
reg [EXP_WIDTH-1:0]           dfa4_bfp_exp;
reg [MANT_WIDTH_L4-1:0]       dfa4_bfp_mant;
reg                           dfa4_bfp_valid;

// 分类头 (32维向量)
reg [EXP_WIDTH-1:0]           cls_bfp_exp;
reg [MANT_WIDTH_L4-1:0]       cls_bfp_mant;
reg                           cls_bfp_valid;

// BRAM 读多路复用控制
reg [2:0]                     rd_sel;      // 0:dfa0 1:dfa1 2:dfa2 3:dfa3 4:dfa4 5:infer 6:cls 7:none
reg [2:0]                     rd_sel_d;
reg [LAYER_WIDTH-1:0]         rd_layer;
reg [LAYER_WIDTH-1:0]         rd_layer_d;
reg [ADDR_WIDTH-1:0]          rd_token;
reg                           rd_do_read;

reg [TOTAL_ADDR_W-1:0]        rd_addr;
reg [TOTAL_ADDR_W-1:0]        wr_addr;

reg [EXP_WIDTH-1:0]           rd_exp_raw;
reg [MANT_WIDTH_L4-1:0]       rd_mant_raw;

//================================================================================
// 写接口逻辑
//================================================================================
integer i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 可选：初始化存储为0，方便仿真
        for (i = 0; i < TOTAL_DEPTH; i = i + 1) begin
            exp_mem[i]  <= {EXP_WIDTH{1'b0}};
            mant_mem[i] <= {MANT_WIDTH_L4{1'b0}};
        end
    end else if (wr_en && !wr_error) begin
        // 写地址 = layer_id * TOKEN_NUM + token_id
        wr_addr <= wr_layer_id * TOKEN_NUM + wr_token_id;

        // 写入指数
        exp_mem[wr_layer_id * TOKEN_NUM + wr_token_id] <= wr_exp;

        // 写入尾数：L0-3 只用低 8 维，高位补 0；L4 用完整 32 维
        case (wr_layer_id)
            4'd0, 4'd1, 4'd2, 4'd3: begin
                mant_mem[wr_layer_id * TOKEN_NUM + wr_token_id]
                    <= {{(MANT_WIDTH_L4-MANT_WIDTH_L0_L3){1'b0}},
                        wr_mant[MANT_WIDTH_L0_L3-1:0]};
            end
            default: begin // layer4
                mant_mem[wr_layer_id * TOKEN_NUM + wr_token_id]
                    <= wr_mant[MANT_WIDTH_L4-1:0];
            end
        endcase
    end
end

assign wr_ready = rst_n;
assign wr_error = wr_addr_error || wr_layer_error;

//================================================================================
// BRAM 读多路复用：优先级仲裁
//================================================================================
always @(*) begin
    // 默认值
    rd_sel     = 3'd7;
    rd_layer   = {LAYER_WIDTH{1'b0}};
    rd_token   = {ADDR_WIDTH{1'b0}};
    rd_do_read = 1'b0;

    // 简单优先级：dfa0 > dfa1 > dfa2 > dfa3 > dfa4 > infer > cls
    if (rd_dfa0_en && !rd_dfa0_addr_error) begin
        rd_sel     = 3'd0;
        rd_layer   = 4'd0;
        rd_token   = rd_dfa0_token_id;
        rd_do_read = 1'b1;
    end else if (rd_dfa1_en && !rd_dfa1_addr_error) begin
        rd_sel     = 3'd1;
        rd_layer   = 4'd1;
        rd_token   = rd_dfa1_token_id;
        rd_do_read = 1'b1;
    end else if (rd_dfa2_en && !rd_dfa2_addr_error) begin
        rd_sel     = 3'd2;
        rd_layer   = 4'd2;
        rd_token   = rd_dfa2_token_id;
        rd_do_read = 1'b1;
    end else if (rd_dfa3_en && !rd_dfa3_addr_error) begin
        rd_sel     = 3'd3;
        rd_layer   = 4'd3;
        rd_token   = rd_dfa3_token_id;
        rd_do_read = 1'b1;
    end else if (rd_dfa4_en && !rd_dfa4_addr_error) begin
        rd_sel     = 3'd4;
        rd_layer   = 4'd4;
        rd_token   = rd_dfa4_token_id;
        rd_do_read = 1'b1;
    end else if (rd_infer_en && !rd_infer_addr_error && !rd_infer_layer_error) begin
        rd_sel     = 3'd5;
        rd_layer   = rd_infer_layer_id;
        rd_token   = rd_infer_token_id;
        rd_do_read = 1'b1;
    end else if (rd_cls_en) begin
        // CLS: 固定 Layer4, Token0
        rd_sel     = 3'd6;
        rd_layer   = 4'd4;
        rd_token   = {ADDR_WIDTH{1'b0}};
        rd_do_read = 1'b1;
    end
end

//================================================================================
// BRAM 同步读 + 输出分发
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_sel_d      <= 3'd7;
        rd_layer_d    <= {LAYER_WIDTH{1'b0}};
        rd_exp_raw    <= {EXP_WIDTH{1'b0}};
        rd_mant_raw   <= {MANT_WIDTH_L4{1'b0}};

        dfa0_bfp_exp   <= {EXP_WIDTH{1'b0}};
        dfa1_bfp_exp   <= {EXP_WIDTH{1'b0}};
        dfa2_bfp_exp   <= {EXP_WIDTH{1'b0}};
        dfa3_bfp_exp   <= {EXP_WIDTH{1'b0}};
        dfa4_bfp_exp   <= {EXP_WIDTH{1'b0}};
        cls_bfp_exp    <= {EXP_WIDTH{1'b0}};

        dfa0_bfp_mant  <= {MANT_WIDTH_L0_L3{1'b0}};
        dfa1_bfp_mant  <= {MANT_WIDTH_L0_L3{1'b0}};
        dfa2_bfp_mant  <= {MANT_WIDTH_L0_L3{1'b0}};
        dfa3_bfp_mant  <= {MANT_WIDTH_L0_L3{1'b0}};
        dfa4_bfp_mant  <= {MANT_WIDTH_L4{1'b0}};
        cls_bfp_mant   <= {MANT_WIDTH_L4{1'b0}};

        dfa0_bfp_valid <= 1'b0;
        dfa1_bfp_valid <= 1'b0;
        dfa2_bfp_valid <= 1'b0;
        dfa3_bfp_valid <= 1'b0;
        dfa4_bfp_valid <= 1'b0;
        cls_bfp_valid  <= 1'b0;

        rd_infer_exp   <= {EXP_WIDTH{1'b0}};
        rd_infer_mant  <= {MANT_WIDTH_L4{1'b0}};
        rd_infer_valid <= 1'b0;
    end else begin
        rd_sel_d   <= rd_sel;
        rd_layer_d <= rd_layer;

        // 默认 valid 置 0（单拍脉冲）
        dfa0_bfp_valid <= 1'b0;
        dfa1_bfp_valid <= 1'b0;
        dfa2_bfp_valid <= 1'b0;
        dfa3_bfp_valid <= 1'b0;
        dfa4_bfp_valid <= 1'b0;
        cls_bfp_valid  <= 1'b0;
        rd_infer_valid <= 1'b0;

        // BRAM 同步读
        if (rd_do_read) begin
            rd_addr    <= rd_layer * TOKEN_NUM + rd_token;
            rd_exp_raw <= exp_mem[rd_layer * TOKEN_NUM + rd_token];
            rd_mant_raw<= mant_mem[rd_layer * TOKEN_NUM + rd_token];
        end

        // 上一拍的选择决定这拍的输出归属
        case (rd_sel_d)
            3'd0: begin
                dfa0_bfp_exp   <= rd_exp_raw;
                dfa0_bfp_mant  <= rd_mant_raw[MANT_WIDTH_L0_L3-1:0];
                dfa0_bfp_valid <= 1'b1;
            end
            3'd1: begin
                dfa1_bfp_exp   <= rd_exp_raw;
                dfa1_bfp_mant  <= rd_mant_raw[MANT_WIDTH_L0_L3-1:0];
                dfa1_bfp_valid <= 1'b1;
            end
            3'd2: begin
                dfa2_bfp_exp   <= rd_exp_raw;
                dfa2_bfp_mant  <= rd_mant_raw[MANT_WIDTH_L0_L3-1:0];
                dfa2_bfp_valid <= 1'b1;
            end
            3'd3: begin
                dfa3_bfp_exp   <= rd_exp_raw;
                dfa3_bfp_mant  <= rd_mant_raw[MANT_WIDTH_L0_L3-1:0];
                dfa3_bfp_valid <= 1'b1;
            end
            3'd4: begin
                dfa4_bfp_exp   <= rd_exp_raw;
                dfa4_bfp_mant  <= rd_mant_raw;
                dfa4_bfp_valid <= 1'b1;
            end
            3'd5: begin
                // 推理读接口（BFP 直出）
                rd_infer_exp  <= rd_exp_raw;
                // Layer0-3：只有 8 维，有效位在低 8 维
                if (rd_layer_d == 4'd4) begin
                    rd_infer_mant <= rd_mant_raw;
                end else begin
                    rd_infer_mant <= {{(MANT_WIDTH_L4-MANT_WIDTH_L0_L3){1'b0}},
                                      rd_mant_raw[MANT_WIDTH_L0_L3-1:0]};
                end
                rd_infer_valid <= 1'b1;
            end
            3'd6: begin
                // CLS：Layer4, Token0
                cls_bfp_exp   <= rd_exp_raw;
                cls_bfp_mant  <= rd_mant_raw;
                cls_bfp_valid <= 1'b1;
            end
            default: begin
                // no-op
            end
        endcase
    end
end

//================================================================================
// BFP→Q4.12 向量转换器实例
//================================================================================

// GCU 0-3: 8维向量
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

assign rd_infer_error = rd_infer_addr_error || rd_infer_layer_error;

//================================================================================
// 计数器（调试用）
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
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
assign dbg_wr_count_l0    = wr_count[0];
assign dbg_wr_count_l1    = wr_count[1];
assign dbg_wr_count_l2    = wr_count[2];
assign dbg_wr_count_l3    = wr_count[3];
assign dbg_wr_count_l4    = wr_count[4];
assign dbg_infer_rd_count = infer_rd_count;
assign dbg_dfa_rd_count   = dfa_rd_count;
assign dbg_cls_rd_count   = cls_rd_count;
assign dbg_wr_error_count = wr_error_count;
assign dbg_rd_error_count = rd_error_count;

endmodule
