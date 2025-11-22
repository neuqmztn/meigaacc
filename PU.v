`timescale 1ns / 1ps

//==============================================================
// PU (Processing Unit) - Fixed v3.1
//==============================================================

module PU #(
    // PE配置
    parameter NUM_PE    = 1,           // PE数量，1或2
    parameter PE_TYPE_0 = 0,           // PE0类型 (0=A:INT8x16, 1=B:INT16x4, 2=C:INT8x8, 3=D:INT16x2)
    parameter PE_TYPE_1 = 2,           // PE1类型
    
    // 数据位宽
    parameter EXP_WIDTH = 8,           // 指数位宽
    parameter INPUT_MANT_WIDTH = 8,    // 输入尾数位宽
    
    // 向量维度分配 (注意：需与 PE_TYPE 对应)
    parameter ELEM_PE0 = 16,           
    parameter ELEM_PE1 = 8,              
    parameter TOTAL_ELEM = 24,         
    
    // 输出位宽优化参数
    parameter INTERNAL_WIDTH = 39,     // 内部累加位宽
    parameter OUTPUT_WIDTH = 32,       // 输出位宽
    parameter GUARD_BITS = 7,          // 截断位数
    parameter ENABLE_ROUNDING = 1      // 1=舍入, 0=截断
)(
    input  wire clk,
    input  wire rst_n,
    input  wire flush,
    
    // 输入握手
    input  wire input_valid,
    output wire input_ready,
    
    // 输出握手
    output wire result_valid,
    input  wire result_ready,
    
    // BFP格式输入
    input  wire [EXP_WIDTH-1:0] exp_X,
    input  wire [EXP_WIDTH-1:0] exp_W,
    input  wire [TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_X_block,
    input  wire [TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_W_block,
    
    // 定点数输出
    output wire signed [OUTPUT_WIDTH-1:0] result_fixed,
    output wire [EXP_WIDTH:0] result_base_exp,
    output wire result_zero
);

    //==============================================================
    // 参数检查
    //==============================================================
    initial begin
        if (OUTPUT_WIDTH > INTERNAL_WIDTH) begin
            $error("ERROR: OUTPUT_WIDTH (%0d) > INTERNAL_WIDTH (%0d)", OUTPUT_WIDTH, INTERNAL_WIDTH);
            $finish;
        end
    end

    //==============================================================
    // 状态机与流水线控制
    //==============================================================
    localparam IDLE  = 2'b00;
    localparam BUSY  = 2'b01;
    localparam VALID = 2'b10;

    reg [1:0] state, state_next;
    // Latency: MAC(3) + PE(2) = 5 cycles. Pipeline depth 6 is safe.
    localparam PIPELINE_DEPTH = 6; 
    reg [$clog2(PIPELINE_DEPTH+1)-1:0] cycle_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else if (flush) state <= IDLE;
        else state <= state_next;
    end

    always @(*) begin
        state_next = state;
        case (state)
            IDLE: if (input_valid && input_ready) state_next = BUSY;
            BUSY: if (cycle_counter >= PIPELINE_DEPTH) state_next = VALID;
            VALID: if (result_ready) state_next = IDLE;
            default: state_next = IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) cycle_counter <= 0;
        else if (flush) cycle_counter <= 0;
        else begin
            case (state)
                IDLE: if (input_valid && input_ready) cycle_counter <= 1; else cycle_counter <= 0;
                BUSY: if (cycle_counter < PIPELINE_DEPTH) cycle_counter <= cycle_counter + 1;
                VALID: if (result_ready) cycle_counter <= 0;
                default: cycle_counter <= 0;
            endcase
        end
    end

    assign input_ready = (state == IDLE);
    assign result_valid = (state == VALID);
    
    // 流水线使能
    wire pipeline_enable = (state == BUSY) || (state == IDLE && input_valid && input_ready);

    //==============================================================
    // 输入寄存
    //==============================================================
    localparam TOTAL_WIDTH = TOTAL_ELEM * INPUT_MANT_WIDTH;
    reg [EXP_WIDTH-1:0] exp_X_r, exp_W_r;
    reg [TOTAL_WIDTH-1:0] mant_X_r, mant_W_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            exp_X_r <= 0; exp_W_r <= 0;
            mant_X_r <= 0; mant_W_r <= 0;
        end else if (input_valid && input_ready) begin
            exp_X_r <= exp_X; exp_W_r <= exp_W;
            mant_X_r <= mant_X_block; mant_W_r <= mant_W_block;
        end
    end

    wire [EXP_WIDTH:0] group_exp;
    assign group_exp = {1'b0, exp_X_r} + {1'b0, exp_W_r} - 9'd127;

    //==============================================================
    // PE0 处理逻辑
    //==============================================================
    wire [ELEM_PE0*INPUT_MANT_WIDTH-1:0] pe0_X;
    wire [ELEM_PE0*INPUT_MANT_WIDTH-1:0] pe0_W;
    assign pe0_X = mant_X_r[ELEM_PE0*INPUT_MANT_WIDTH-1 : 0];
    assign pe0_W = mant_W_r[ELEM_PE0*INPUT_MANT_WIDTH-1 : 0];

    wire [63:0] pe0_x_a, pe0_x_b, pe0_w_a, pe0_w_b;

    generate
        if (PE_TYPE_0 == 0) begin : pe0_packer_A
            data_packer_PE_A u_pack (.mant_X_vec(pe0_X), .mant_W_vec(pe0_W), .x_data_a_packed(pe0_x_a), .x_data_b_packed(pe0_x_b), .w_data_a_packed(pe0_w_a), .w_data_b_packed(pe0_w_b));
        end else if (PE_TYPE_0 == 1) begin : pe0_packer_B
            data_packer_PE_B u_pack (.mant_X_vec(pe0_X), .mant_W_vec(pe0_W), .x_data_a_packed(pe0_x_a), .x_data_b_packed(pe0_x_b), .w_data_a_packed(pe0_w_a), .w_data_b_packed(pe0_w_b));
        end else if (PE_TYPE_0 == 2) begin : pe0_packer_C
            data_packer_PE_C u_pack (.mant_X_vec(pe0_X), .mant_W_vec(pe0_W), .x_data_a_packed(pe0_x_a), .x_data_b_packed(pe0_x_b), .w_data_a_packed(pe0_w_a), .w_data_b_packed(pe0_w_b));
        end else begin : pe0_packer_D
            data_packer_PE_D u_pack (.mant_X_vec(pe0_X), .mant_W_vec(pe0_W), .x_data_a_packed(pe0_x_a), .x_data_b_packed(pe0_x_b), .w_data_a_packed(pe0_w_a), .w_data_b_packed(pe0_w_b));
        end
    endgenerate

    wire signed [37:0] pe0_result;
    wire pe0_valid;

    PE #(
        // 【修复】动态 MAC 数量和位宽
        .NUM_MAC((PE_TYPE_0 >= 2) ? 4 : 8), 
        .ADDER_MODE((PE_TYPE_0 == 1 || PE_TYPE_0 == 3) ? 1 : 0),
        .DATA_WIDTH(8),
        .MAC_OUT_WIDTH(19), // 必须是 19
        .FINAL_WIDTH(38)
    ) u_pe0 (
        .clk(clk), .rst_n(rst_n), .enable(pipeline_enable), .flush(flush),
        .x_data_a_packed(pe0_x_a), .x_data_b_packed(pe0_x_b),
        .w_data_a_packed(pe0_w_a), .w_data_b_packed(pe0_w_b),
        .pe_result(pe0_result), .result_valid(pe0_valid)
    );

    //==============================================================
    // PE1 处理逻辑
    //==============================================================
    wire signed [37:0] pe1_result;
    wire pe1_valid;

    generate
        if (NUM_PE == 2) begin : pe1_processing
            wire [ELEM_PE1*INPUT_MANT_WIDTH-1:0] pe1_X;
            wire [ELEM_PE1*INPUT_MANT_WIDTH-1:0] pe1_W;
            assign pe1_X = mant_X_r[(ELEM_PE0*INPUT_MANT_WIDTH) +: (ELEM_PE1*INPUT_MANT_WIDTH)];
            assign pe1_W = mant_W_r[(ELEM_PE0*INPUT_MANT_WIDTH) +: (ELEM_PE1*INPUT_MANT_WIDTH)];

            wire [63:0] pe1_x_a, pe1_x_b, pe1_w_a, pe1_w_b;
            if (PE_TYPE_1 == 0) begin : pe1_packer_A
                data_packer_PE_A u_pack (.mant_X_vec(pe1_X), .mant_W_vec(pe1_W), .x_data_a_packed(pe1_x_a), .x_data_b_packed(pe1_x_b), .w_data_a_packed(pe1_w_a), .w_data_b_packed(pe1_w_b));
            end else if (PE_TYPE_1 == 1) begin : pe1_packer_B
                data_packer_PE_B u_pack (.mant_X_vec(pe1_X), .mant_W_vec(pe1_W), .x_data_a_packed(pe1_x_a), .x_data_b_packed(pe1_x_b), .w_data_a_packed(pe1_w_a), .w_data_b_packed(pe1_w_b));
            end else if (PE_TYPE_1 == 2) begin : pe1_packer_C
                data_packer_PE_C u_pack (.mant_X_vec(pe1_X), .mant_W_vec(pe1_W), .x_data_a_packed(pe1_x_a), .x_data_b_packed(pe1_x_b), .w_data_a_packed(pe1_w_a), .w_data_b_packed(pe1_w_b));
            end else begin : pe1_packer_D
                data_packer_PE_D u_pack (.mant_X_vec(pe1_X), .mant_W_vec(pe1_W), .x_data_a_packed(pe1_x_a), .x_data_b_packed(pe1_x_b), .w_data_a_packed(pe1_w_a), .w_data_b_packed(pe1_w_b));
            end

            PE #(
                .NUM_MAC((PE_TYPE_1 >= 2) ? 4 : 8), 
                .ADDER_MODE((PE_TYPE_1 == 1 || PE_TYPE_1 == 3) ? 1 : 0),
                .DATA_WIDTH(8),
                .MAC_OUT_WIDTH(19),
                .FINAL_WIDTH(38)
            ) u_pe1 (
                .clk(clk), .rst_n(rst_n), .enable(pipeline_enable), .flush(flush),
                .x_data_a_packed(pe1_x_a), .x_data_b_packed(pe1_x_b),
                .w_data_a_packed(pe1_w_a), .w_data_b_packed(pe1_w_b),
                .pe_result(pe1_result), .result_valid(pe1_valid)
            );
        end else begin : pe1_zero
            assign pe1_result = 38'sd0;
            assign pe1_valid = 1'b0;
        end
    endgenerate

    //==============================================================
    // 结果累加与截断
    //==============================================================
    reg signed [INTERNAL_WIDTH-1:0] group_sum_internal;
    always @(*) begin
        if (NUM_PE == 2)
            group_sum_internal = {{1{pe0_result[37]}}, pe0_result} + {{1{pe1_result[37]}}, pe1_result};
        else
            group_sum_internal = {{1{pe0_result[37]}}, pe0_result};
    end

    reg signed [OUTPUT_WIDTH-1:0] group_sum_truncated;
    generate
        if (OUTPUT_WIDTH == INTERNAL_WIDTH) begin : no_truncation
            always @(*) group_sum_truncated = group_sum_internal;
        end else begin : with_truncation
            always @(*) begin:pu
                reg signed [INTERNAL_WIDTH-1:0] shifted_val;
                reg signed [OUTPUT_WIDTH-1:0] high_bits;
                reg round_bit;

                // 【修复】使用算术右移
                shifted_val = group_sum_internal >>> GUARD_BITS;
                high_bits = shifted_val[OUTPUT_WIDTH-1:0];
                round_bit = group_sum_internal[GUARD_BITS-1];

                if (ENABLE_ROUNDING == 1)
                    group_sum_truncated = high_bits + {{(OUTPUT_WIDTH-1){1'b0}}, round_bit};
                else
                    group_sum_truncated = high_bits;
            end
        end
    endgenerate

    // 指数调整
    reg [EXP_WIDTH:0] group_exp_adjusted;
    generate
        if (OUTPUT_WIDTH == INTERNAL_WIDTH)
            always @(*) group_exp_adjusted = group_exp;
        else
            always @(*) group_exp_adjusted = group_exp + GUARD_BITS;
    endgenerate

    //==============================================================
    // 输出寄存
    //==============================================================
    reg signed [OUTPUT_WIDTH-1:0] result_fixed_r;
    reg [EXP_WIDTH:0] result_base_exp_r;
    reg result_zero_r;

    initial begin
        result_fixed_r = 0; result_base_exp_r = 0; result_zero_r = 1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_fixed_r <= 0; result_base_exp_r <= 0; result_zero_r <= 1;
        end else if (flush) begin
            result_fixed_r <= 0; result_base_exp_r <= 0; result_zero_r <= 1;
        end else if (state == VALID && !result_ready) begin
            result_fixed_r <= result_fixed_r;
        end else if (pipeline_enable) begin
            result_fixed_r <= group_sum_truncated;
            result_base_exp_r <= group_exp_adjusted;
            result_zero_r <= (group_sum_internal == 0);
        end
    end

    assign result_fixed = result_fixed_r;
    assign result_base_exp = result_base_exp_r;
    assign result_zero = result_zero_r;

endmodule