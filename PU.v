`timescale 1ns / 1ps

//==============================================================
// PU (Processing Unit) - 严谨版（4MAC / 32bit 接口 + 32bit 输出截断）
//
// 1) 每个 PE 内部固定包含 4 个 MAC（NUM_MAC=4），每个 MAC 计算 A*B + C*D。
// 2) 四个打包器：
//    - PE_A / PE_B：一个打包器负责一个 PU，输出两套 32bit 端口（pe0_* / pe1_*），可驱动 2 个同类型 PE。
//    - PE_C / PE_D：一个打包器负责一个 PE，输出一套 32bit 端口（x_data_*_packed），只驱动 PE0。
// 3) PU 不再做 64bit→32bit 截断，所有连到 PE 的 packed 端口统一为 32bit。
// 4) 内部累加使用 INTERNAL_WIDTH 位（默认 39），对接输出转换器时输出 OUTPUT_WIDTH 位（默认 32），
//    GUARD_BITS 为截断的低位数（默认 7），可选舍入或简单截断。
//==============================================================

module PU #(
    parameter NUM_PE            = 1,   // 1 或 2 个 PE
    parameter PE_TYPE_0         = 0,   // 0=A, 1=B, 2=C, 3=D
    parameter PE_TYPE_1         = 0,   // 仅当 NUM_PE=2 时有意义，应与 PE_TYPE_0 一致或受控配置
    parameter EXP_WIDTH         = 8,
    parameter INPUT_MANT_WIDTH  = 8,   // A/C: 8; B/D: 16
    parameter ELEM_PE0          = 0,   // 保留参数（由上层配置），本实现不按元素数切片
    parameter ELEM_PE1          = 0,
    parameter TOTAL_ELEM        = 0,   // 保留参数，应与具体模式对应 (A:16, B:4, C:8, D:2)

    // 输出位宽优化参数
    parameter INTERNAL_WIDTH    = 39,  // 内部累加位宽（应 ≥ PE 输出位宽 + 1）
    parameter OUTPUT_WIDTH      = 32,  // 对接转换器的输出位宽
    parameter GUARD_BITS        = 7,   // 截断位数，通常 = INTERNAL_WIDTH - OUTPUT_WIDTH
    parameter ENABLE_ROUNDING   = 1,   // 1=舍入, 0=截断

    // PE 内部结果位宽
    parameter PE_FINAL_WIDTH    = 38   // 单个 PE 的输出位宽
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

// 给一个保守的流水线深度参数，供计数使用；如有需要可以调大
localparam integer PIPELINE_DEPTH = 6;
reg [5:0] pipe_cnt;

// 接收寄存输入的寄存器
reg [EXP_WIDTH-1:0]          exp_X_r, exp_W_r;
reg [TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_X_r, mant_W_r;

// 输入 ready：空闲，或者结果已取走的 VALID 状态可以接新数据
assign input_ready  = (state == IDLE) || (state == VALID && result_ready);
// 输出 valid：仅在 VALID 状态
assign result_valid = (state == VALID);

// 用于驱动 PE 的 enable：BUSY 期间持续为 1；在接收首拍输入时也拉高一拍
wire pipeline_enable = (state == BUSY) || (state == IDLE && input_valid && input_ready);

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
                        pipe_cnt <= 6'd1;    // 接新一帧
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
            if (pipe_cnt >= PIPELINE_DEPTH)
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
//==============================================================
wire [EXP_WIDTH:0] group_exp;
assign group_exp = {1'b0, exp_X_r} + {1'b0, exp_W_r} - 9'd127;

//==============================================================
// PE 接口信号（统一 4MAC / 32bit）
//==============================================================

wire [31:0] pe0_x_a, pe0_x_b, pe0_w_a, pe0_w_b;
wire [31:0] pe1_x_a, pe1_x_b, pe1_w_a, pe1_w_b;

wire signed [PE_FINAL_WIDTH-1:0] pe0_result;
wire               pe0_valid;
wire signed [PE_FINAL_WIDTH-1:0] pe1_result;
wire               pe1_valid;

//==============================================================
// 一个打包器服务一个 PU：按 PE_TYPE_0 选择打包器 + PE 拓扑
//==============================================================

generate
    //====================== PE_A：INT8，16 元素，双 PE ======================
    if (PE_TYPE_0 == 0) begin : gen_type_A
        data_packer_PE_A u_packer_A (
            .mant_X_vec(mant_X_r),
            .mant_W_vec(mant_W_r),
            .pe0_x_data_a_packed(pe0_x_a),
            .pe0_x_data_b_packed(pe0_x_b),
            .pe0_w_data_a_packed(pe0_w_a),
            .pe0_w_data_b_packed(pe0_w_b),
            .pe1_x_data_a_packed(pe1_x_a),
            .pe1_x_data_b_packed(pe1_x_b),
            .pe1_w_data_a_packed(pe1_w_a),
            .pe1_w_data_b_packed(pe1_w_b)
        );

        PE #(
            .NUM_MAC      (4),
            .ADDER_MODE   (1'b0),
            .DATA_WIDTH   (8),
            .MAC_OUT_WIDTH(17),
            .FINAL_WIDTH  (PE_FINAL_WIDTH)
        ) u_pe0 (
            .clk             (clk),
            .rst_n           (rst_n),
            .enable          (pipeline_enable),
            .flush           (flush),
            .x_data_a_packed (pe0_x_a),
            .x_data_b_packed (pe0_x_b),
            .w_data_a_packed (pe0_w_a),
            .w_data_b_packed (pe0_w_b),
            .pe_result       (pe0_result),
            .result_valid    (pe0_valid)
        );

        if (NUM_PE == 2) begin : has_pe1_A
            PE #(
                .NUM_MAC      (4),
                .ADDER_MODE   (1'b0),
                .DATA_WIDTH   (8),
                .MAC_OUT_WIDTH(17),
                .FINAL_WIDTH  (PE_FINAL_WIDTH)
            ) u_pe1 (
                .clk             (clk),
                .rst_n           (rst_n),
                .enable          (pipeline_enable),
                .flush           (flush),
                .x_data_a_packed (pe1_x_a),
                .x_data_b_packed (pe1_x_b),
                .w_data_a_packed (pe1_w_a),
                .w_data_b_packed (pe1_w_b),
                .pe_result       (pe1_result),
                .result_valid    (pe1_valid)
            );
        end else begin : no_pe1_A
            assign pe1_result = {PE_FINAL_WIDTH{1'b0}};
            assign pe1_valid  = 1'b0;
        end

    //====================== PE_B：INT16，4 元素，双 PE ======================
    end else if (PE_TYPE_0 == 1) begin : gen_type_B
        data_packer_PE_B u_packer_B (
            .mant_X_vec(mant_X_r),
            .mant_W_vec(mant_W_r),
            .pe0_x_data_a_packed(pe0_x_a),
            .pe0_x_data_b_packed(pe0_x_b),
            .pe0_w_data_a_packed(pe0_w_a),
            .pe0_w_data_b_packed(pe0_w_b),
            .pe1_x_data_a_packed(pe1_x_a),
            .pe1_x_data_b_packed(pe1_x_b),
            .pe1_w_data_a_packed(pe1_w_a),
            .pe1_w_data_b_packed(pe1_w_b)
        );

        PE #(
            .NUM_MAC      (4),
            .ADDER_MODE   (1'b1),
            .DATA_WIDTH   (8),
            .MAC_OUT_WIDTH(17),
            .FINAL_WIDTH  (PE_FINAL_WIDTH)
        ) u_pe0 (
            .clk             (clk),
            .rst_n           (rst_n),
            .enable          (pipeline_enable),
            .flush           (flush),
            .x_data_a_packed (pe0_x_a),
            .x_data_b_packed (pe0_x_b),
            .w_data_a_packed (pe0_w_a),
            .w_data_b_packed (pe0_w_b),
            .pe_result       (pe0_result),
            .result_valid    (pe0_valid)
        );

        if (NUM_PE == 2) begin : has_pe1_B
            PE #(
                .NUM_MAC      (4),
                .ADDER_MODE   (1'b1),
                .DATA_WIDTH   (8),
                .MAC_OUT_WIDTH(17),
                .FINAL_WIDTH  (PE_FINAL_WIDTH)
            ) u_pe1 (
                .clk             (clk),
                .rst_n           (rst_n),
                .enable          (pipeline_enable),
                .flush           (flush),
                .x_data_a_packed (pe1_x_a),
                .x_data_b_packed (pe1_x_b),
                .w_data_a_packed (pe1_w_a),
                .w_data_b_packed (pe1_w_b),
                .pe_result       (pe1_result),
                .result_valid    (pe1_valid)
            );
        end else begin : no_pe1_B
            assign pe1_result = {PE_FINAL_WIDTH{1'b0}};
            assign pe1_valid  = 1'b0;
        end

    //====================== PE_C：INT8，8 元素，单 PE ======================
    end else if (PE_TYPE_0 == 2) begin : gen_type_C
        data_packer_PE_C u_packer_C (
            .mant_X_vec        (mant_X_r),
            .mant_W_vec        (mant_W_r),
            .x_data_a_packed   (pe0_x_a),
            .x_data_b_packed   (pe0_x_b),
            .w_data_a_packed   (pe0_w_a),
            .w_data_b_packed   (pe0_w_b)
        );

        PE #(
            .NUM_MAC      (4),
            .ADDER_MODE   (1'b0),
            .DATA_WIDTH   (8),
            .MAC_OUT_WIDTH(17),
            .FINAL_WIDTH  (PE_FINAL_WIDTH)
        ) u_pe0 (
            .clk             (clk),
            .rst_n           (rst_n),
            .enable          (pipeline_enable),
            .flush           (flush),
            .x_data_a_packed (pe0_x_a),
            .x_data_b_packed (pe0_x_b),
            .w_data_a_packed (pe0_w_a),
            .w_data_b_packed (pe0_w_b),
            .pe_result       (pe0_result),
            .result_valid    (pe0_valid)
        );

        assign pe1_result = {PE_FINAL_WIDTH{1'b0}};
        assign pe1_valid  = 1'b0;

    //====================== PE_D：INT16，2 元素，单 PE ======================
    end else begin : gen_type_D
        data_packer_PE_D u_packer_D (
            .mant_X_vec        (mant_X_r),
            .mant_W_vec        (mant_W_r),
            .x_data_a_packed   (pe0_x_a),
            .x_data_b_packed   (pe0_x_b),
            .w_data_a_packed   (pe0_w_a),
            .w_data_b_packed   (pe0_w_b)
        );

        PE #(
            .NUM_MAC      (4),
            .ADDER_MODE   (1'b1),
            .DATA_WIDTH   (8),
            .MAC_OUT_WIDTH(17),
            .FINAL_WIDTH  (PE_FINAL_WIDTH)
        ) u_pe0 (
            .clk             (clk),
            .rst_n           (rst_n),
            .enable          (pipeline_enable),
            .flush           (flush),
            .x_data_a_packed (pe0_x_a),
            .x_data_b_packed (pe0_x_b),
            .w_data_a_packed (pe0_w_a),
            .w_data_b_packed (pe0_w_b),
            .pe_result       (pe0_result),
            .result_valid    (pe0_valid)
        );

        assign pe1_result = {PE_FINAL_WIDTH{1'b0}};
        assign pe1_valid  = 1'b0;
    end
endgenerate

//==============================================================
// 定点合并（0~2 个 PE 的输出合并成 group_sum，INTERNAL_WIDTH 位）
//==============================================================

// 先把两个 PE 的 38bit 结果符号扩展到 INTERNAL_WIDTH 再相加
wire signed [INTERNAL_WIDTH-1:0] pe0_ext;
wire signed [INTERNAL_WIDTH-1:0] pe1_ext;

assign pe0_ext = {{(INTERNAL_WIDTH-PE_FINAL_WIDTH){pe0_result[PE_FINAL_WIDTH-1]}},
                  pe0_result};

assign pe1_ext = {{(INTERNAL_WIDTH-PE_FINAL_WIDTH){pe1_result[PE_FINAL_WIDTH-1]}},
                  pe1_result};

wire signed [INTERNAL_WIDTH-1:0] group_sum;

generate
    if (NUM_PE == 2 && (PE_TYPE_0 == 0 || PE_TYPE_0 == 1)) begin : merge_two
        assign group_sum = pe0_ext + pe1_ext;
    end else begin : merge_single
        assign group_sum = pe0_ext;
    end
endgenerate

//==============================================================
// 截断/舍入到 OUTPUT_WIDTH 位
//==============================================================

localparam integer KEPT_WIDTH = INTERNAL_WIDTH - GUARD_BITS;
initial begin
    // 简单一致性约束（编译期检查用，工具会 warning 但不影响综合）
    if (KEPT_WIDTH != OUTPUT_WIDTH) begin
        $display("WARNING: PU: INTERNAL_WIDTH - GUARD_BITS != OUTPUT_WIDTH");
    end
end

// 0.5 ULP 的舍入偏移：在保留位最低位上加 1（即第 GUARD_BITS-1 位上加 1）
wire signed [INTERNAL_WIDTH-1:0] rounding_bias;
assign rounding_bias = (ENABLE_ROUNDING && (GUARD_BITS > 0)) ?
                       {{(INTERNAL_WIDTH-GUARD_BITS){1'b0}}, 1'b1, {(GUARD_BITS-1){1'b0}}} :
                       {INTERNAL_WIDTH{1'b0}};

// 先加偏移，再截断（算术右移 GUARD_BITS）
wire signed [INTERNAL_WIDTH-1:0] rounded_sum;
assign rounded_sum = group_sum + rounding_bias;

wire signed [OUTPUT_WIDTH-1:0] truncated_sum;
// 保留高位 [INTERNAL_WIDTH-1 : GUARD_BITS]，宽度为 OUTPUT_WIDTH
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
    end else if (flush) begin
        result_fixed_r    <= {OUTPUT_WIDTH{1'b0}};
        result_base_exp_r <= {(EXP_WIDTH+1){1'b0}};
        result_zero_r     <= 1'b1;
    end else if (state == VALID && result_ready) begin
        result_fixed_r    <= truncated_sum;
        result_base_exp_r <= group_exp;
        result_zero_r     <= (group_sum == {INTERNAL_WIDTH{1'b0}});
    end
end

assign result_fixed    = result_fixed_r;
assign result_base_exp = result_base_exp_r;
assign result_zero     = result_zero_r;

endmodule
