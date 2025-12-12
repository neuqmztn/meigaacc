`timescale 1ns / 1ps

module gelu_engine #(
    parameter TOKEN_CHUNK     = 32,
    parameter MATRIX_SIZE     = 32,
    parameter FEATURE_SIZE    = 32,

    parameter BFP_EXP_W       = 8,
    parameter INPUT_MANT_W    = 16,      // Q3.12 格式
    parameter OUTPUT_MANT_W   = 8,       // 输出压缩位宽
    parameter PIPELINE_STAGES = 4
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    output reg  busy,

    input  wire [TOKEN_CHUNK*BFP_EXP_W-1:0]               in_exp,
    input  wire [MATRIX_SIZE*FEATURE_SIZE*INPUT_MANT_W-1:0] in_mant,

    output reg  [TOKEN_CHUNK*BFP_EXP_W-1:0]                 out_exp,
    output reg  [MATRIX_SIZE*FEATURE_SIZE*OUTPUT_MANT_W-1:0] out_mant
);

    // =========================================================================
    // 常量与状态
    // =========================================================================
    localparam [INPUT_MANT_W-1:0] FIXED_ONE = 16'h1000;  

    // FSM
    localparam IDLE       = 3'd0;
    localparam PROCESS    = 3'd1;
    localparam DONE_STATE = 3'd2;

    reg [2:0] state;

    // 流水线寄存器
    reg signed [INPUT_MANT_W-1:0] pipe_data_s0 [0:FEATURE_SIZE-1];

    reg signed [INPUT_MANT_W-1:0] pipe_k_s1    [0:FEATURE_SIZE-1];
    reg signed [INPUT_MANT_W-1:0] pipe_b_s1    [0:FEATURE_SIZE-1];
    reg signed [INPUT_MANT_W-1:0] pipe_data_s1 [0:FEATURE_SIZE-1];

    reg signed [INPUT_MANT_W-1:0] pipe_res_s2  [0:FEATURE_SIZE-1];

    // 有效标志流水线
    reg       pipe_valid   [0:PIPELINE_STAGES-1];
    reg [4:0] row_idx_pipe [0:PIPELINE_STAGES-1];

    // 行计数 (0 ~ MATRIX_SIZE-1)
    reg [5:0] process_row_cnt;

    // 循环变量
    integer i, j0, j1, k2, m3;

    // =========================================================================
    // PWL 系数查找函数
    //   输入：16 位有符号定点数 (Q3.12)
    //   输出：{slope[15:0], intercept[15:0]}
    // =========================================================================
    function [31:0] get_pwl_coef;
        input signed [INPUT_MANT_W-1:0] x;
        reg [15:0] slope;
        reg [15:0] intercept;
        reg [3:0] seg_idx;
        begin
            // 使用 $signed() 保证常数是有符号比较
            if (x >= $signed(16'h4000)) begin       // x >=  4.0
                slope     = FIXED_ONE;
                intercept = 16'd0;
            end
            else if (x <= $signed(16'hC000)) begin  // x <= -4.0
                slope     = 16'd0;
                intercept = 16'd0;
            end
            else begin
                // 线性拟合区
                seg_idx = x[14:11];
                case (seg_idx)
                    // === 正数部分 ===
                    4'h0: begin slope = 16'h0B10; intercept = 16'h0000; end
                    4'h1: begin slope = 16'h0FDC; intercept = 16'hFD9A; end
                    4'h2: begin slope = 16'h11D7; intercept = 16'hFB9F; end // x=1.0 在此
                    4'h3: begin slope = 16'h11C6; intercept = 16'hFBB9; end
                    4'h4: begin slope = 16'h116E; intercept = 16'hFC69; end
                    4'h5: begin slope = 16'h0FE6; intercept = 16'h003D; end
                    4'h6: begin slope = 16'h101A; intercept = 16'hFFA1; end
                    4'h7: begin slope = 16'h1008; intercept = 16'hFFE0; end

                    // === 负数部分 ===
                    4'h8: begin slope = 16'h0000; intercept = 16'h0000; end
                    4'h9: begin slope = 16'hFFE4; intercept = 16'hFF9C; end
                    4'hA: begin slope = 16'hFFA2; intercept = 16'hFED6; end
                    4'hB: begin slope = 16'hFF0A; intercept = 16'hFD5A; end
                    4'hC: begin slope = 16'hFE4A; intercept = 16'hFBDA; end
                    4'hD: begin slope = 16'hFE16; intercept = 16'hFB8C; end
                    4'hE: begin slope = 16'h0024; intercept = 16'hFD9A; end
                    4'hF: begin slope = 16'h04EF; intercept = 16'h0000; end

                    default: begin
                        slope     = FIXED_ONE;
                        intercept = 16'd0;
                    end
                endcase
            end
            get_pwl_coef = {slope, intercept};
        end
    endfunction

    // =========================================================================
    // 1) 控制 / FSM / 有效标志流水线
    //    这里只负责：state、busy、done、process_row_cnt、pipe_valid[]、
    //    row_idx_pipe[]、out_exp 的时序更新。
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= IDLE;
            done            <= 1'b0;
            busy            <= 1'b0;
            process_row_cnt <= 6'd0;
            out_exp         <= {TOKEN_CHUNK*BFP_EXP_W{1'b0}};

            for (i = 0; i < PIPELINE_STAGES; i = i + 1) begin
                pipe_valid[i]   <= 1'b0;
                row_idx_pipe[i] <= 5'd0;
            end
        end
        else begin
            // 缺省：done 为 0（只在 DONE_STATE 拉高 1 个周期）
            done <= 1'b0;

            case (state)
                // ------------------------------------------------------------
                // IDLE：等待 start
                // ------------------------------------------------------------
                IDLE: begin
                    busy            <= 1'b0;
                    process_row_cnt <= 6'd0;

                    // 进入新一轮处理
                    if (start) begin
                        state <= PROCESS;
                        busy  <= 1'b1;
                        out_exp <= in_exp;  // 指数直接透传

                        for (i = 0; i < PIPELINE_STAGES; i = i + 1) begin
                            pipe_valid[i]   <= 1'b0;
                            row_idx_pipe[i] <= 5'd0;
                        end
                    end
                end

                // ------------------------------------------------------------
                // PROCESS：按行推进流水线
                // ------------------------------------------------------------
                PROCESS: begin
                    busy <= 1'b1;

                    // Stage0 有效标志（装载行）：
                    // 只要还有未处理的行，就持续把 pipe_valid[0] 置 1
                    if (process_row_cnt < MATRIX_SIZE) begin
                        pipe_valid[0]   <= 1'b1;
                        row_idx_pipe[0] <= process_row_cnt[4:0];
                        process_row_cnt <= process_row_cnt + 1'b1;
                    end
                    else begin
                        pipe_valid[0] <= 1'b0;
                    end

                    // 其余级的 valid / 行号 直接顺着流水线传递
                    pipe_valid[1]   <= pipe_valid[0];
                    pipe_valid[2]   <= pipe_valid[1];
                    pipe_valid[3]   <= pipe_valid[2];

                    row_idx_pipe[1] <= row_idx_pipe[0];
                    row_idx_pipe[2] <= row_idx_pipe[1];
                    row_idx_pipe[3] <= row_idx_pipe[2];

                    // 当所有行已经送完，且流水线 4 级全部排空 → 进入 DONE
                    if ((process_row_cnt >= MATRIX_SIZE) &&
                        (pipe_valid[0] == 1'b0) &&
                        (pipe_valid[1] == 1'b0) &&
                        (pipe_valid[2] == 1'b0) &&
                        (pipe_valid[3] == 1'b0)) begin
                        state <= DONE_STATE;
                    end
                end

                // ------------------------------------------------------------
                // DONE_STATE：输出 done 脉冲 1 个周期，然后回到 IDLE
                // ------------------------------------------------------------
                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

    // =========================================================================
    // 2) Stage 0：按行加载输入数据 → pipe_data_s0[]
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j0 = 0; j0 < FEATURE_SIZE; j0 = j0 + 1) begin
                pipe_data_s0[j0] <= {INPUT_MANT_W{1'b0}};
            end
        end
        else begin
            // 只有在 PROCESS 状态下且有有效行输入时，才装载新数据
            if ((state == PROCESS) && (process_row_cnt < MATRIX_SIZE)) begin
                for (j0 = 0; j0 < FEATURE_SIZE; j0 = j0 + 1) begin
                    pipe_data_s0[j0] <= $signed(
                        in_mant[(process_row_cnt*FEATURE_SIZE + j0)*INPUT_MANT_W +: INPUT_MANT_W]
                    );
                end
            end
            // 其他情况保持原值（寄存器自然保持，不再写）
        end
    end

    // =========================================================================
    // 3) Stage 1：PWL 系数查找 → pipe_k_s1 / pipe_b_s1 + 数据直通 pipe_data_s1
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j1 = 0; j1 < FEATURE_SIZE; j1 = j1 + 1) begin
                pipe_k_s1[j1]    <= {INPUT_MANT_W{1'b0}};
                pipe_b_s1[j1]    <= {INPUT_MANT_W{1'b0}};
                pipe_data_s1[j1] <= {INPUT_MANT_W{1'b0}};
            end
        end
        else begin

            if (pipe_valid[0]) begin
                for (j1 = 0; j1 < FEATURE_SIZE; j1 = j1 + 1) begin
                    {pipe_k_s1[j1], pipe_b_s1[j1]} <= get_pwl_coef(pipe_data_s0[j1]);
                    pipe_data_s1[j1]               <= pipe_data_s0[j1];
                end
            end
        end
    end

    // =========================================================================
    // 4) Stage 2：乘加运算 → pipe_res_s2[]
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k2 = 0; k2 < FEATURE_SIZE; k2 = k2 + 1) begin
                pipe_res_s2[k2] <= {INPUT_MANT_W{1'b0}};
            end
        end
        else begin

            if (pipe_valid[1]) begin
                for (k2 = 0; k2 < FEATURE_SIZE; k2 = k2 + 1) begin

                    pipe_res_s2[k2] <=
                        ((32'sd0 + (pipe_data_s1[k2] * pipe_k_s1[k2])) >>> 12)
                        + pipe_b_s1[k2];
                end
            end
        end
    end

    // =========================================================================
    // 5) Stage 3：截断并写入输出 out_mant[]
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_mant <= {MATRIX_SIZE*FEATURE_SIZE*OUTPUT_MANT_W{1'b0}};
        end
        else begin
  
            if (pipe_valid[2]) begin
                for (m3 = 0; m3 < FEATURE_SIZE; m3 = m3 + 1) begin
                    if (INPUT_MANT_W > OUTPUT_MANT_W) begin

                        out_mant[(row_idx_pipe[2]*FEATURE_SIZE + m3)*OUTPUT_MANT_W +: OUTPUT_MANT_W]
                            <= pipe_res_s2[m3][INPUT_MANT_W-1 -: OUTPUT_MANT_W];
                    end
                    else begin
                        out_mant[(row_idx_pipe[2]*FEATURE_SIZE + m3)*OUTPUT_MANT_W +: OUTPUT_MANT_W]
                            <= pipe_res_s2[m3][OUTPUT_MANT_W-1:0];
                    end
                end
            end
        end
    end

endmodule
