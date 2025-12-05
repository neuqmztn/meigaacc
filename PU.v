`timescale 1ns / 1ps

//==============================================================
// PU (Processing Unit) - 终版（4MAC / 32bit 接口 + 32bit 输出截断）
//
// - PE_A / PE_B：输入 32 维 → 2×16 维（双打包器 / 双 PE）
// - PE_C / PE_D：输入 16 维 → 2×8 维 （双打包器 / 双 PE）
//
// 说明：
//   * 每个 PE 有自己独立的 Data Packer 实例（一个打包器对应一个 PE）。
//   * NUM_PE=1：只启用 PE0；NUM_PE=2：PE0+PE1 的结果在 PU 内相加。
//   * 输出仍为：INTERNAL_WIDTH → OUTPUT_WIDTH（带 GUARD_BITS 截断/舍入）。
//==============================================================

module PU #(
    parameter NUM_PE            = 2,   // 1 或 2 个 PE
    parameter PE_TYPE_0         = 0,   // 0=A, 1=B, 2=C, 3=D
    parameter PE_TYPE_1         = 0,   // 仅当 NUM_PE=2 时有意义，应与 PE_TYPE_0 一致或受控配置
    parameter EXP_WIDTH         = 8,
    parameter INPUT_MANT_WIDTH  = 8,   // A/C: 8; B/D: 16 (按 packer 内部约定)
    parameter ELEM_PE0          = 16,   // 保留参数
    parameter ELEM_PE1          = 16,
    parameter TOTAL_ELEM        = 32,   // A/B:32, C/D:16

    // 输出位宽优化参数
    parameter INTERNAL_WIDTH    = 39,
    parameter OUTPUT_WIDTH      = 32,
    parameter GUARD_BITS        = 7,
    parameter ENABLE_ROUNDING   = 1,

    // PE 内部结果位宽
    parameter PE_FINAL_WIDTH    = 38
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // 握手
    input  wire                          input_valid,
    output wire                          input_ready,
    output wire                          result_valid,
    input  wire                          result_ready,

    input  wire                          flush,

    // BFP 输入
    input  wire [EXP_WIDTH-1:0]          exp_X,
    input  wire [EXP_WIDTH-1:0]          exp_W,
    input  wire [TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_X_block,
    input  wire [TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_W_block,

    // 输出：定点和 + 基础指数（未归一化）
    output wire signed [OUTPUT_WIDTH-1:0] result_fixed,
    output wire [EXP_WIDTH:0]             result_base_exp,
    output wire                           result_zero
);

//==============================================================
// 流水线状态机：IDLE -> BUSY -> VALID
//==============================================================
localparam [1:0] IDLE  = 2'b00;
localparam [1:0] BUSY  = 2'b01;
localparam [1:0] VALID = 2'b10;

reg [1:0] state, state_next;

localparam integer PIPELINE_DEPTH = 7; // 经过测量：PE_A 延迟约 7 拍
reg [5:0] pipe_cnt;

// 接收寄存输入的寄存器
reg [EXP_WIDTH-1:0]          exp_X_r, exp_W_r;
reg [TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_X_r, mant_W_r;

// 输入 ready：空闲，或者结果已取走的 VALID 状态可以接新数据
assign input_ready  = (state == IDLE) || (state == VALID && result_ready);
// 输出 valid：仅在 VALID 状态
//assign result_valid = (state == VALID);
reg result_valid_r;
assign result_valid = result_valid_r;
// 用于驱动 PE 的 enable：BUSY 期间持续为 1；在接收首拍输入时也拉高一拍
wire pipeline_enable = (state == BUSY) || (state == IDLE && input_valid && input_ready);

// 内部flush：在接收新输入时清除PE的残留valid信号
// 两种情况都需要flush:
// 1. 从IDLE接收新输入 (第一次计算)
// 2. 从VALID接收新输入 (连续计算)
wire internal_flush = ((state == IDLE) || (state == VALID && result_ready)) 
                      && input_valid && input_ready;
wire combined_flush = flush || internal_flush;
//==============================================================
// PE 接口信号（统一 4MAC / 32bit）
//==============================================================
localparam integer PACK_WIDTH      = 64; // 最大配置：8个MAC，每个8bit
localparam integer HALF_PACK_WIDTH = 32; // 4个MAC×8bit
wire [PACK_WIDTH-1:0] pe0_x_a;
wire [PACK_WIDTH-1:0] pe0_x_b;
wire [PACK_WIDTH-1:0] pe0_w_a;
wire [PACK_WIDTH-1:0] pe0_w_b;

wire [PACK_WIDTH-1:0] pe1_x_a;
wire [PACK_WIDTH-1:0] pe1_x_b;
wire [PACK_WIDTH-1:0] pe1_w_a;
wire [PACK_WIDTH-1:0] pe1_w_b;

wire signed [PE_FINAL_WIDTH-1:0] pe0_result;
wire               pe0_valid;
wire signed [PE_FINAL_WIDTH-1:0] pe1_result;
wire               pe1_valid;
wire pe_all_valid;
assign pe_all_valid = (NUM_PE == 1) ? pe0_valid :
                                      (pe0_valid & pe1_valid);
//---------------- 状态与计数器 ----------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state    <= IDLE;
        pipe_cnt <= 6'd0;
    end else if (flush) begin
        state    <= IDLE;
        pipe_cnt <= 6'd0;
    end else begin
        state <= state_next;
        case (state)
            IDLE: begin
                if (input_valid && input_ready)
                    pipe_cnt <= 6'd1;
                else
                    pipe_cnt <= 6'd0;
            end
            BUSY: begin
                if (pipe_cnt != 0 && pipe_cnt < PIPELINE_DEPTH)
                    pipe_cnt <= pipe_cnt + 6'd1;
            end
            VALID: begin
                if (result_ready) begin
                    if (input_valid)
                        pipe_cnt <= 6'd1;
                    else
                        pipe_cnt <= 6'd0;
                end
            end
            default: pipe_cnt <= 6'd0;
        endcase
    end
end

always @(*) begin
    state_next = state;
    case (state)
        IDLE: begin
            if (input_valid && input_ready)
                state_next = BUSY;
        end
        BUSY: begin
            if (pe_all_valid)
                state_next = VALID;
        end
        VALID: begin
            if (result_ready) begin
                if (input_valid)
                    state_next = BUSY;
                else
                    state_next = IDLE;
            end
        end
        default: state_next = IDLE;
    endcase
end

//---------------- 输入寄存 ----------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        exp_X_r  <= {EXP_WIDTH{1'b0}};
        exp_W_r  <= {EXP_WIDTH{1'b0}};
        mant_X_r <= {TOTAL_ELEM*INPUT_MANT_WIDTH{1'b0}};
        mant_W_r <= {TOTAL_ELEM*INPUT_MANT_WIDTH{1'b0}};
    end else if (flush) begin
        exp_X_r  <= {EXP_WIDTH{1'b0}};
        exp_W_r  <= {EXP_WIDTH{1'b0}};
        mant_X_r <= {TOTAL_ELEM*INPUT_MANT_WIDTH{1'b0}};
        mant_W_r <= {TOTAL_ELEM*INPUT_MANT_WIDTH{1'b0}};
    end else if (input_valid && input_ready) begin
        exp_X_r  <= exp_X;
        exp_W_r  <= exp_W;
        mant_X_r <= mant_X_block;
        mant_W_r <= mant_W_block;
    end
end

//==============================================================
// 指数计算：PE 不参与，仅在 PU 层处理
// group_exp = exp_X + exp_W - 127
// 注意：需要延迟以匹配 PE 的 2 级流水线延迟
//==============================================================
wire [EXP_WIDTH:0] group_exp_comb;
assign group_exp_comb = {1'b0, exp_X_r} + {1'b0, exp_W_r} - 9'd127;

// 延迟 2 拍以匹配 PE 流水线（Stage1 + Stage2）
reg [EXP_WIDTH:0] group_exp_d1, group_exp_d2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        group_exp_d1 <= {(EXP_WIDTH+1){1'b0}};
        group_exp_d2 <= {(EXP_WIDTH+1){1'b0}};
    end else if (flush) begin
        group_exp_d1 <= {(EXP_WIDTH+1){1'b0}};
        group_exp_d2 <= {(EXP_WIDTH+1){1'b0}};
    end else if (pipeline_enable) begin
        group_exp_d1 <= group_exp_comb;
        group_exp_d2 <= group_exp_d1;
    end
end

wire [EXP_WIDTH:0] group_exp;
assign group_exp = group_exp_d2;


//==============================================================
// 一个打包器对应一个 PE：按 PE_TYPE_0 + NUM_PE 选择打包器拓扑
// A/B: 32 → 2×16; C/D: 16 → 2×8
//==============================================================
generate
 //====================== PE_A：INT8，一 packer 对应一 PE，8MAC ======================
    if (PE_TYPE_0 == 0) begin : gen_type_A
        // 每个 PE 多少元素：如果有 2 个 PE，就 TOTAL_ELEM/2；否则全给 PE0
        localparam integer ELEM_PER_PE_A  = (NUM_PE == 2) ? (TOTAL_ELEM/2) : TOTAL_ELEM;
        localparam integer WIDTH_PER_PE_A = ELEM_PER_PE_A * INPUT_MANT_WIDTH; // 应该是 16*8=128

        // 按 PE 切分 mant_X / mant_W
        wire [WIDTH_PER_PE_A-1:0] mant_X_pe0_A;
        wire [WIDTH_PER_PE_A-1:0] mant_W_pe0_A;
        wire [WIDTH_PER_PE_A-1:0] mant_X_pe1_A;
        wire [WIDTH_PER_PE_A-1:0] mant_W_pe1_A;

        // PE0：低半部分
        assign mant_X_pe0_A = mant_X_r[WIDTH_PER_PE_A-1:0];
        assign mant_W_pe0_A = mant_W_r[WIDTH_PER_PE_A-1:0];

        // PE1：高半部分（仅 NUM_PE==2 时使用）
        assign mant_X_pe1_A = (NUM_PE == 2) ?
                              mant_X_r[2*WIDTH_PER_PE_A-1:WIDTH_PER_PE_A] :
                              {WIDTH_PER_PE_A{1'b0}};
        assign mant_W_pe1_A = (NUM_PE == 2) ?
                              mant_W_r[2*WIDTH_PER_PE_A-1:WIDTH_PER_PE_A] :
                              {WIDTH_PER_PE_A{1'b0}};

        // -------- packer for PE0（16elem → 8MAC×INT8）--------
        data_packer_PE_A_single u_packer_A0 (
            .mant_X_vec      (mant_X_pe0_A), // 期望 WIDTH_PER_PE_A = 128
            .mant_W_vec      (mant_W_pe0_A),
            .x_data_a_packed (pe0_x_a),      // 8*8 = 64bit
            .x_data_b_packed (pe0_x_b),
            .w_data_a_packed (pe0_w_a),
            .w_data_b_packed (pe0_w_b)
        );

        // -------- packer for PE1（可选）--------
        if (NUM_PE == 2) begin : A_has_pe1_packer
            data_packer_PE_A_single u_packer_A1 (
                .mant_X_vec      (mant_X_pe1_A),
                .mant_W_vec      (mant_W_pe1_A),
                .x_data_a_packed (pe1_x_a),
                .x_data_b_packed (pe1_x_b),
                .w_data_a_packed (pe1_w_a),
                .w_data_b_packed (pe1_w_b)
            );
        end else begin : A_no_pe1_packer
            assign pe1_x_a = {PACK_WIDTH{1'b0}};
            assign pe1_x_b = {PACK_WIDTH{1'b0}};
            assign pe1_w_a = {PACK_WIDTH{1'b0}};
            assign pe1_w_b = {PACK_WIDTH{1'b0}};
        end

        // -------- PE 实例：NUM_MAC = 8，INT8 模式 --------
        PE #(
            .NUM_MAC       (8),
            .ADDER_MODE    (1'b0),                 // INT8
            .DATA_WIDTH    (INPUT_MANT_WIDTH),    
            .MAC_OUT_WIDTH (19),
            .FINAL_WIDTH   (PE_FINAL_WIDTH)
        ) u_pe0 (
            .clk             (clk),
            .rst_n           (rst_n),
            .enable          (pipeline_enable),
            .flush           (combined_flush),
            .x_data_a_packed (pe0_x_a),
            .x_data_b_packed (pe0_x_b),
            .w_data_a_packed (pe0_w_a),
            .w_data_b_packed (pe0_w_b),
            .pe_result       (pe0_result),
            .result_valid    (pe0_valid)
        );

        if (NUM_PE == 2) begin : A_has_pe1_PE
            PE #(
                .NUM_MAC       (8),
                .ADDER_MODE    (1'b0),             // INT8
                .DATA_WIDTH    (INPUT_MANT_WIDTH),
                .MAC_OUT_WIDTH (19),
                .FINAL_WIDTH   (PE_FINAL_WIDTH)
            ) u_pe1 (
                .clk             (clk),
                .rst_n           (rst_n),
                .enable          (pipeline_enable),
                .flush           (combined_flush),
                .x_data_a_packed (pe1_x_a),
                .x_data_b_packed (pe1_x_b),
                .w_data_a_packed (pe1_w_a),
                .w_data_b_packed (pe1_w_b),
                .pe_result       (pe1_result),
                .result_valid    (pe1_valid)
            );
        end else begin : A_no_pe1_PE
            assign pe1_result = {PE_FINAL_WIDTH{1'b0}};
            assign pe1_valid  = 1'b0;
        end
    //====================== PE_B：INT16，32 维 → 2×16，最多 2PE ======================
    end else if (PE_TYPE_0 == 1) begin : gen_type_B
        localparam integer TOTAL_WIDTH_B   = TOTAL_ELEM * INPUT_MANT_WIDTH;
        localparam integer WIDTH_PER_PE_B  = (NUM_PE == 2) ? (TOTAL_WIDTH_B/2) : TOTAL_WIDTH_B;
    
        wire [WIDTH_PER_PE_B-1:0] mant_X_pe0_B;
        wire [WIDTH_PER_PE_B-1:0] mant_W_pe0_B;
        wire [WIDTH_PER_PE_B-1:0] mant_X_pe1_B;
        wire [WIDTH_PER_PE_B-1:0] mant_W_pe1_B;
    
        assign mant_X_pe0_B = mant_X_r[WIDTH_PER_PE_B-1:0];
        assign mant_W_pe0_B = mant_W_r[WIDTH_PER_PE_B-1:0];
    
        assign mant_X_pe1_B = (NUM_PE == 2) ? mant_X_r[TOTAL_WIDTH_B-1:WIDTH_PER_PE_B]
                                            : {WIDTH_PER_PE_B{1'b0}};
        assign mant_W_pe1_B = (NUM_PE == 2) ? mant_W_r[TOTAL_WIDTH_B-1:WIDTH_PER_PE_B]
                                            : {WIDTH_PER_PE_B{1'b0}};
    
        // 这里改成 "每个 PE 一个单 packer"，并输出 64bit
        data_packer_PE_B u_packer_B0 (
            .mant_X_vec      (mant_X_pe0_B),   // 32bit (2×INT16)
            .mant_W_vec      (mant_W_pe0_B),
            .x_data_a_packed (pe0_x_a),        // 64bit
            .x_data_b_packed (pe0_x_b),
            .w_data_a_packed (pe0_w_a),
            .w_data_b_packed (pe0_w_b)
        );
    
        if (NUM_PE == 2) begin : B_has_pe1
            data_packer_PE_B u_packer_B1 (
                .mant_X_vec      (mant_X_pe1_B),
                .mant_W_vec      (mant_W_pe1_B),
                .x_data_a_packed (pe1_x_a),
                .x_data_b_packed (pe1_x_b),
                .w_data_a_packed (pe1_w_a),
                .w_data_b_packed (pe1_w_b)
            );
        end else begin : B_no_pe1_packer
            assign pe1_x_a = {PACK_WIDTH{1'b0}};
            assign pe1_x_b = {PACK_WIDTH{1'b0}};
            assign pe1_w_a = {PACK_WIDTH{1'b0}};
            assign pe1_w_b = {PACK_WIDTH{1'b0}};
        end
        PE #(
            .NUM_MAC      (8),
            .ADDER_MODE   (1'b1),
            .DATA_WIDTH   (8),
            .MAC_OUT_WIDTH(19),
            .FINAL_WIDTH  (PE_FINAL_WIDTH)
        ) u_pe0 (
            .clk             (clk),
            .rst_n           (rst_n),
            .enable          (pipeline_enable),
            .flush           (combined_flush),
            .x_data_a_packed (pe0_x_a),
            .x_data_b_packed (pe0_x_b),
            .w_data_a_packed (pe0_w_a),
            .w_data_b_packed (pe0_w_b),
            .pe_result       (pe0_result),
            .result_valid    (pe0_valid)
        );

        if (NUM_PE == 2) begin : has_pe1_B_PE
            PE #(
                .NUM_MAC      (8),
                .ADDER_MODE   (1'b1),
                .DATA_WIDTH   (8),
                .MAC_OUT_WIDTH(19),
                .FINAL_WIDTH  (PE_FINAL_WIDTH)
            ) u_pe1 (
                .clk             (clk),
                .rst_n           (rst_n),
                .enable          (pipeline_enable),
                .flush           (combined_flush),
                .x_data_a_packed (pe1_x_a),
                .x_data_b_packed (pe1_x_b),
                .w_data_a_packed (pe1_w_a),
                .w_data_b_packed (pe1_w_b),
                .pe_result       (pe1_result),
                .result_valid    (pe1_valid)
            );
        end else begin : no_pe1_B_PE
            assign pe1_result = {PE_FINAL_WIDTH{1'b0}};
            assign pe1_valid  = 1'b0;
        end

    //====================== PE_C：INT8，TOTAL_ELEM → 2×(TOTAL_ELEM/2)，最多 2PE ======================
    end else if (PE_TYPE_0 == 2) begin : gen_type_C
        // 根据 TOTAL_ELEM 自动推导宽度
        localparam integer TOTAL_WIDTH_C   = TOTAL_ELEM * INPUT_MANT_WIDTH;
        localparam integer WIDTH_PER_PE_C  = (NUM_PE == 2) ? (TOTAL_WIDTH_C/2) : TOTAL_WIDTH_C;

        wire [WIDTH_PER_PE_C-1:0] mant_X_pe0_C;
        wire [WIDTH_PER_PE_C-1:0] mant_W_pe0_C;
        wire [WIDTH_PER_PE_C-1:0] mant_X_pe1_C;
        wire [WIDTH_PER_PE_C-1:0] mant_W_pe1_C;

        // PE0：低半部分
        assign mant_X_pe0_C = mant_X_r[WIDTH_PER_PE_C-1:0];
        assign mant_W_pe0_C = mant_W_r[WIDTH_PER_PE_C-1:0];

        // PE1：高半部分（仅在 NUM_PE==2 时使用）
        assign mant_X_pe1_C = (NUM_PE == 2) ?
                              mant_X_r[TOTAL_WIDTH_C-1:WIDTH_PER_PE_C] :
                              {WIDTH_PER_PE_C{1'b0}};
        assign mant_W_pe1_C = (NUM_PE == 2) ?
                              mant_W_r[TOTAL_WIDTH_C-1:WIDTH_PER_PE_C] :
                              {WIDTH_PER_PE_C{1'b0}};

        // -------- packer for PE0 --------
        data_packer_PE_C u_packer_C0 (
            .mant_X_vec      (mant_X_pe0_C),
            .mant_W_vec      (mant_W_pe0_C),
            .x_data_a_packed (pe0_x_a[HALF_PACK_WIDTH-1:0]),
            .x_data_b_packed (pe0_x_b[HALF_PACK_WIDTH-1:0]),
            .w_data_a_packed (pe0_w_a[HALF_PACK_WIDTH-1:0]),
            .w_data_b_packed (pe0_w_b[HALF_PACK_WIDTH-1:0])
        );

        // -------- packer for PE1（可选） --------
        if (NUM_PE == 2) begin : C_has_pe1
            data_packer_PE_C u_packer_C1 (
                .mant_X_vec      (mant_X_pe1_C),
                .mant_W_vec      (mant_W_pe1_C),
                .x_data_a_packed (pe1_x_a[HALF_PACK_WIDTH-1:0]),
                .x_data_b_packed (pe1_x_b[HALF_PACK_WIDTH-1:0]),
                .w_data_a_packed (pe1_w_a[HALF_PACK_WIDTH-1:0]),
                .w_data_b_packed (pe1_w_b[HALF_PACK_WIDTH-1:0])
            );
        end else begin : C_no_pe1_packer
            assign pe1_x_a = 32'd0;
            assign pe1_x_b = 32'd0;
            assign pe1_w_a = 32'd0;
            assign pe1_w_b = 32'd0;
        end

        // -------- PE 实例 --------
        PE #(
            .NUM_MAC      (4),
            .ADDER_MODE   (1'b0),   // INT8
            .DATA_WIDTH   (8),
            .MAC_OUT_WIDTH(19),
            .FINAL_WIDTH  (PE_FINAL_WIDTH)
        ) u_pe0 (
            .clk             (clk),
            .rst_n           (rst_n),
            .enable          (pipeline_enable),
            .flush           (combined_flush),
            .x_data_a_packed (pe0_x_a[HALF_PACK_WIDTH-1:0]),
            .x_data_b_packed (pe0_x_b[HALF_PACK_WIDTH-1:0]),
            .w_data_a_packed (pe0_w_a[HALF_PACK_WIDTH-1:0]),
            .w_data_b_packed (pe0_w_b[HALF_PACK_WIDTH-1:0]),
            .pe_result       (pe0_result),
            .result_valid    (pe0_valid)
        );

        if (NUM_PE == 2) begin : has_pe1_C_PE
            PE #(
                .NUM_MAC      (4),
                .ADDER_MODE   (1'b0),   // INT8
                .DATA_WIDTH   (8),
                .MAC_OUT_WIDTH(19),
                .FINAL_WIDTH  (PE_FINAL_WIDTH)
            ) u_pe1 (
                .clk             (clk),
                .rst_n           (rst_n),
                .enable          (pipeline_enable),
                .flush           (combined_flush),
                .x_data_a_packed (pe1_x_a[HALF_PACK_WIDTH-1:0]),
                .x_data_b_packed (pe1_x_b[HALF_PACK_WIDTH-1:0]),
                .w_data_a_packed (pe1_w_a[HALF_PACK_WIDTH-1:0]),
                .w_data_b_packed (pe1_w_b[HALF_PACK_WIDTH-1:0]),
                .pe_result       (pe1_result),
                .result_valid    (pe1_valid)
            );
        end else begin : no_pe1_C_PE
            assign pe1_result = {PE_FINAL_WIDTH{1'b0}};
            assign pe1_valid  = 1'b0;
        end
    //====================== PE_D：INT16，16 维 → 2×8，最多 2PE ======================
end else begin : gen_type_D
        localparam integer TOTAL_WIDTH_D   = TOTAL_ELEM * INPUT_MANT_WIDTH;
        localparam integer WIDTH_PER_PE_D  = (NUM_PE == 2) ? (TOTAL_WIDTH_D/2) : TOTAL_WIDTH_D;

        wire [WIDTH_PER_PE_D-1:0] mant_X_pe0_D;
        wire [WIDTH_PER_PE_D-1:0] mant_W_pe0_D;
        wire [WIDTH_PER_PE_D-1:0] mant_X_pe1_D;
        wire [WIDTH_PER_PE_D-1:0] mant_W_pe1_D;

        assign mant_X_pe0_D = mant_X_r[WIDTH_PER_PE_D-1:0];
        assign mant_W_pe0_D = mant_W_r[WIDTH_PER_PE_D-1:0];

        assign mant_X_pe1_D = (NUM_PE == 2) ?
                              mant_X_r[TOTAL_WIDTH_D-1:WIDTH_PER_PE_D] :
                              {WIDTH_PER_PE_D{1'b0}};
        assign mant_W_pe1_D = (NUM_PE == 2) ?
                              mant_W_r[TOTAL_WIDTH_D-1:WIDTH_PER_PE_D] :
                              {WIDTH_PER_PE_D{1'b0}};

        // -------- packer for PE0 --------
        data_packer_PE_D u_packer_D0 (
            .mant_X_vec      (mant_X_pe0_D),
            .mant_W_vec      (mant_W_pe0_D),
            .x_data_a_packed (pe0_x_a[HALF_PACK_WIDTH-1:0]),
            .x_data_b_packed (pe0_x_b[HALF_PACK_WIDTH-1:0]),
            .w_data_a_packed (pe0_w_a[HALF_PACK_WIDTH-1:0]),
            .w_data_b_packed (pe0_w_b[HALF_PACK_WIDTH-1:0])
        );
        // -------- packer for PE1（可选） --------
        if (NUM_PE == 2) begin : D_has_pe1
            data_packer_PE_D u_packer_D1 (
                .mant_X_vec      (mant_X_pe1_D),
                .mant_W_vec      (mant_W_pe1_D),
                .x_data_a_packed (pe1_x_a),
                .x_data_b_packed (pe1_x_b),
                .w_data_a_packed (pe1_w_a),
                .w_data_b_packed (pe1_w_b)
            );
        end else begin : D_no_pe1_packer
            assign pe1_x_a = 32'd0;
            assign pe1_x_b = 32'd0;
            assign pe1_w_a = 32'd0;
            assign pe1_w_b = 32'd0;
        end

        // -------- PE 实例 --------
        PE #(
            .NUM_MAC      (4),
            .ADDER_MODE   (1'b1),   // INT16
            .DATA_WIDTH   (8),
            .MAC_OUT_WIDTH(19),
            .FINAL_WIDTH  (PE_FINAL_WIDTH)
        ) u_pe0 (
            .clk             (clk),
            .rst_n           (rst_n),
            .enable          (pipeline_enable),
            .flush           (combined_flush),
            .x_data_a_packed (pe0_x_a[HALF_PACK_WIDTH-1:0]),
            .x_data_b_packed (pe0_x_b[HALF_PACK_WIDTH-1:0]),
            .w_data_a_packed (pe0_w_a[HALF_PACK_WIDTH-1:0]),
            .w_data_b_packed (pe0_w_b[HALF_PACK_WIDTH-1:0]),
            .pe_result       (pe0_result),
            .result_valid    (pe0_valid)
        );

        if (NUM_PE == 2) begin : has_pe1_D_PE
            PE #(
                .NUM_MAC      (4),
                .ADDER_MODE   (1'b1),   // INT16
                .DATA_WIDTH   (8),
                .MAC_OUT_WIDTH(19),
                .FINAL_WIDTH  (PE_FINAL_WIDTH)
            ) u_pe1 (
            .clk             (clk),
            .rst_n           (rst_n),
            .enable          (pipeline_enable),
            .flush           (combined_flush),
            .x_data_a_packed (pe1_x_a[HALF_PACK_WIDTH-1:0]),
            .x_data_b_packed (pe1_x_b[HALF_PACK_WIDTH-1:0]),
            .w_data_a_packed (pe1_w_a[HALF_PACK_WIDTH-1:0]),
            .w_data_b_packed (pe1_w_b[HALF_PACK_WIDTH-1:0]),
            .pe_result       (pe1_result),
            .result_valid    (pe1_valid)
        );
        end else begin : no_pe1_D_PE
            assign pe1_result = {PE_FINAL_WIDTH{1'b0}};
            assign pe1_valid  = 1'b0;
        end
    end
endgenerate

//==============================================================
// 定点合并（0~2 个 PE 的输出合并成 group_sum，INTERNAL_WIDTH 位）
//==============================================================
wire signed [INTERNAL_WIDTH-1:0] pe0_ext;
wire signed [INTERNAL_WIDTH-1:0] pe1_ext;

assign pe0_ext = {{(INTERNAL_WIDTH-PE_FINAL_WIDTH){pe0_result[PE_FINAL_WIDTH-1]}},
                  pe0_result};

assign pe1_ext = {{(INTERNAL_WIDTH-PE_FINAL_WIDTH){pe1_result[PE_FINAL_WIDTH-1]}},
                  pe1_result};

wire signed [INTERNAL_WIDTH-1:0] group_sum;
wire group_sum_zero_flag;

generate
    if (NUM_PE == 2) begin : merge_two
        assign group_sum = pe0_ext + pe1_ext;
    end else begin : merge_single
        assign group_sum = pe0_ext;
    end
endgenerate

// zero 标志（基于 group_sum）
assign group_sum_zero_flag = (group_sum == {INTERNAL_WIDTH{1'b0}});

//==============================================================
// 截断/舍入到 OUTPUT_WIDTH 位
//==============================================================
localparam integer KEPT_WIDTH = INTERNAL_WIDTH - GUARD_BITS;

// 0.5 ULP 的舍入偏移
wire signed [INTERNAL_WIDTH-1:0] rounding_bias;
assign rounding_bias = (ENABLE_ROUNDING && (GUARD_BITS > 0)) ?
                       {{(INTERNAL_WIDTH-GUARD_BITS){1'b0}}, 1'b1, {(GUARD_BITS-1){1'b0}}} :
                       {INTERNAL_WIDTH{1'b0}};

wire signed [INTERNAL_WIDTH-1:0] rounded_sum;
assign rounded_sum = group_sum + rounding_bias;

wire signed [OUTPUT_WIDTH-1:0] truncated_sum;
assign truncated_sum = rounded_sum[INTERNAL_WIDTH-1:GUARD_BITS];

//==============================================================
// 输出寄存（定点数 + 基础指数）
//==============================================================
reg signed [OUTPUT_WIDTH-1:0] result_fixed_r;
reg [EXP_WIDTH:0]             result_base_exp_r;
reg                           result_zero_r;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        result_fixed_r    <= {OUTPUT_WIDTH{1'b0}};
        result_base_exp_r <= {(EXP_WIDTH+1){1'b0}};
        result_zero_r     <= 1'b1;
        result_valid_r    <= 1'b0;
    end else if (flush) begin
        result_fixed_r    <= {OUTPUT_WIDTH{1'b0}};
        result_base_exp_r <= {(EXP_WIDTH+1){1'b0}};
        result_zero_r     <= 1'b1;
        result_valid_r    <= 1'b0;
    end else begin
        // 1) 当 BUSY 且所有 PE 完成时，打一拍输出寄存，并置 valid=1
        if (state == BUSY && pe_all_valid) begin
            result_fixed_r    <= truncated_sum;
            result_base_exp_r <= group_exp;
            result_zero_r     <= group_sum_zero_flag;
            result_valid_r    <= 1'b1;
        end
        // 2) VALID 状态下，下游接收后清除 valid
        else if (state == VALID && result_ready) begin
            result_valid_r    <= 1'b0;
        end
        // 其余情况保持寄存器数值不变（包括结果数据）
    end
end

assign result_fixed    = result_fixed_r;
assign result_base_exp = result_base_exp_r;
assign result_zero     = result_zero_r;

endmodule