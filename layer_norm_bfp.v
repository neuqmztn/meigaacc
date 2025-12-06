`timescale 1ns / 1ps

//==============================================================================
// Layer Norm BFP - Lite 版（只去均值 + 仿射变换，无查表）
//
// y_i = gamma_i * (x_i - mean) + beta_i
//
// - 不做方差 / sqrt / 1/sqrt，完全不用查表
// - 只需要加/乘/移位/比较
// - 接口与原 layer_norm_bfp 完全一致，方便替换
//==============================================================================

module layer_norm_bfp #(
    parameter TOKEN_NUM    = 641,
    parameter DIM          = 32,
    parameter DATA_WIDTH   = 8,
    parameter EXP_WIDTH    = 8,
    parameter ADDR_WIDTH   = 10,
    parameter GUARD_BITS   = 3,     // 内部运算保护位
    parameter ACCUM_WIDTH  = 24     // 统计累加器位宽
)(
    input  wire clk,
    input  wire rst_n,

    // 控制接口
    input  wire start,
    input  wire mode,         // 0 = LN1, 1 = LN2（当前未区分，预留）
    output reg  done,
    output wire busy,
    output reg  error,
    
    // 输入数据接口 (流式读取)
    output reg  input_rd_en,
    output reg  [ADDR_WIDTH-1:0] input_rd_addr,
    input  wire [EXP_WIDTH-1:0]  input_exp,
    input  wire [DIM*DATA_WIDTH-1:0] input_mant,
    input  wire input_valid,
    
    // 参数接口 (Gamma / Beta)
    output reg  [ADDR_WIDTH-1:0] param_addr,
    output reg  param_rd_gamma,
    output reg  param_rd_beta,
    input  wire [EXP_WIDTH-1:0]  param_exp,
    input  wire [DIM*DATA_WIDTH-1:0] param_mant,
    input  wire param_valid,
    
    // 输出数据接口 (写回)
    output reg  output_wr_en,
    output reg  [ADDR_WIDTH-1:0] output_wr_addr,
    output reg  [EXP_WIDTH-1:0]  output_exp,
    output reg  [DIM*DATA_WIDTH-1:0] output_mant,
    output reg  output_valid,
    input  wire output_ready,
    
    // 调试接口
    output reg  [9:0]  dbg_token_idx,
    output reg  [31:0] dbg_cycle_cnt
);

    //==========================================================================
    // 状态机
    //==========================================================================
    localparam S_IDLE        = 5'd0;
    localparam S_READ_PARAM  = 5'd1;
    localparam S_PREP_TOKEN  = 5'd2;
    localparam S_READ_INPUT  = 5'd3;
    localparam S_CALC_SUM    = 5'd4;
    localparam S_SUB_GAMMA   = 5'd5;
    localparam S_FIND_MAX    = 5'd6;
    localparam S_OUTPUT_NORM = 5'd7;
    localparam S_WRITE_OUT   = 5'd8;
    localparam S_NEXT_TOKEN  = 5'd9;
    localparam S_DONE        = 5'd10;

    reg [4:0] state, next_state;

    // 控制寄存器
    reg [9:0] token_idx;
    reg [31:0] cycle_cnt;

    // Busy 标志
    assign busy = (state != S_IDLE) && (state != S_DONE);

    //==========================================================================
    // 输入/参数缓存
    //==========================================================================

    // 当前 token 的 mantissa 向量（把 input_mant 存下来，避免后面读口被改）
    reg signed [DATA_WIDTH-1:0] token_mant  [0:DIM-1];

    // Gamma / Beta 向量（只存 mantissa，exp 暂时不用）
    reg signed [DATA_WIDTH-1:0] gamma_mant  [0:DIM-1];
    reg signed [DATA_WIDTH-1:0] beta_mant   [0:DIM-1];
    reg        [EXP_WIDTH-1:0]  gamma_exp_reg;
    reg        [EXP_WIDTH-1:0]  beta_exp_reg;
    reg                         params_loaded;

    // 均值 / 累加
    reg signed [ACCUM_WIDTH-1:0] sum_val;
    reg signed [DATA_WIDTH+GUARD_BITS-1:0] mean_val;

    // 仿射变换后的中间结果（高位精度）
    reg signed [39:0]           affine_val [0:DIM-1];
    reg signed [39:0]           max_abs_mant;
    reg [5:0]                   out_shift;
    reg [EXP_WIDTH-1:0]         out_exp_reg;
    reg signed [DATA_WIDTH-1:0] out_mant_arr [0:DIM-1];

    integer i;

    //==========================================================================
    // 主时序逻辑
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            next_state     <= S_IDLE;

            done           <= 1'b0;
            error          <= 1'b0;
            input_rd_en    <= 1'b0;
            input_rd_addr  <= {ADDR_WIDTH{1'b0}};
            param_addr     <= {ADDR_WIDTH{1'b0}};
            param_rd_gamma <= 1'b0;
            param_rd_beta  <= 1'b0;

            output_wr_en   <= 1'b0;
            output_wr_addr <= {ADDR_WIDTH{1'b0}};
            output_exp     <= {EXP_WIDTH{1'b0}};
            output_mant    <= {DIM*DATA_WIDTH{1'b0}};
            output_valid   <= 1'b0;

            token_idx      <= 10'd0;
            cycle_cnt      <= 32'd0;
            dbg_token_idx  <= 10'd0;
            dbg_cycle_cnt  <= 32'd0;

            params_loaded  <= 1'b0;
            gamma_exp_reg  <= {EXP_WIDTH{1'b0}};
            beta_exp_reg   <= {EXP_WIDTH{1'b0}};
            sum_val        <= {ACCUM_WIDTH{1'b0}};
            mean_val       <= {DATA_WIDTH+GUARD_BITS{1'b0}};
            max_abs_mant   <= 40'd0;
            out_shift      <= 6'd0;
            out_exp_reg    <= {EXP_WIDTH{1'b0}};
        end else begin
            state <= next_state;

            // cycle 计数
            if (state != S_IDLE && state != S_DONE)
                cycle_cnt <= cycle_cnt + 1;

            dbg_token_idx <= token_idx;
            dbg_cycle_cnt <= cycle_cnt;

            //------------------------------------------------------------------
            // 状态行为
            //------------------------------------------------------------------
            case (state)
                //==============================================================
                // S_IDLE：等待 start，若参数未加载先读参数
                //==============================================================
                S_IDLE: begin
                    done          <= 1'b0;
                    output_valid  <= 1'b0;
                    output_wr_en  <= 1'b0;
                    input_rd_en   <= 1'b0;
                    param_rd_gamma<= 1'b0;
                    param_rd_beta <= 1'b0;

                    if (start) begin
                        if (params_loaded) begin
                            token_idx  <= 10'd0;
                            next_state <= S_PREP_TOKEN;
                        end else begin
                            // 读取 Gamma
                            param_addr     <= {ADDR_WIDTH{1'b0}};
                            param_rd_gamma <= 1'b1;
                            next_state     <= S_READ_PARAM;
                        end
                    end else begin
                        next_state <= S_IDLE;
                    end
                end

                //==============================================================
                // S_READ_PARAM：先读 Gamma，再读 Beta
                //==============================================================
                S_READ_PARAM: begin
                    if (param_valid) begin
                        if (param_rd_gamma) begin
                            gamma_exp_reg <= param_exp;
                            for (i = 0; i < DIM; i = i + 1) begin
                                gamma_mant[i] <= param_mant[i*DATA_WIDTH +: DATA_WIDTH];
                            end
                            param_rd_gamma <= 1'b0;
                            param_rd_beta  <= 1'b1;
                            param_addr     <= {ADDR_WIDTH{1'b0}}; // 单行参数
                            next_state     <= S_READ_PARAM;
                        end else if (param_rd_beta) begin
                            beta_exp_reg <= param_exp;
                            for (i = 0; i < DIM; i = i + 1) begin
                                beta_mant[i] <= param_mant[i*DATA_WIDTH +: DATA_WIDTH];
                            end
                            param_rd_beta  <= 1'b0;
                            params_loaded  <= 1'b1;
                            token_idx      <= 10'd0;
                            next_state     <= S_PREP_TOKEN;
                        end else begin
                            next_state <= S_PREP_TOKEN;
                        end
                    end else begin
                        next_state <= S_READ_PARAM;
                    end
                end

                //==============================================================
                // 为当前 token 发起读请求
                //==============================================================
                S_PREP_TOKEN: begin
                    if (token_idx < TOKEN_NUM) begin
                        input_rd_en   <= 1'b1;
                        input_rd_addr <= token_idx[ADDR_WIDTH-1:0];
                        next_state    <= S_READ_INPUT;
                    end else begin
                        next_state    <= S_DONE;
                    end
                end

                //==============================================================
                // 等待 input_valid，把当前 token 的 32 维 mant 存下来
                //==============================================================
                S_READ_INPUT: begin
                    if (input_valid) begin
                        input_rd_en <= 1'b0;

                        // 把 input_mant 展开存到 token_mant[]
                        for (i = 0; i < DIM; i = i + 1) begin
                            token_mant[i] <= input_mant[i*DATA_WIDTH +: DATA_WIDTH];
                        end

                        // 清 sum，进下一个状态算均值
                        sum_val    <= {ACCUM_WIDTH{1'b0}};
                        next_state <= S_CALC_SUM;
                    end else begin
                        next_state <= S_READ_INPUT;
                    end
                end

                //==============================================================
                // S_CALC_SUM：一次性把 32 维加起来，得到 sum_val & mean_val
                //==============================================================
                S_CALC_SUM: begin:a
                    // 用临时变量做阻塞累加，再在该拍末尾写回寄存器
                    reg signed [ACCUM_WIDTH-1:0] sum_tmp;
                    reg signed [DATA_WIDTH-1:0]  x_i;
                    reg signed [ACCUM_WIDTH-1:0] x_ext;
                    integer k;

                    sum_tmp = {ACCUM_WIDTH{1'b0}};
                    for (k = 0; k < DIM; k = k + 1) begin
                        x_i   = token_mant[k];
                        x_ext = {{(ACCUM_WIDTH-DATA_WIDTH){x_i[DATA_WIDTH-1]}}, x_i};
                        sum_tmp = sum_tmp + x_ext;
                    end

                    sum_val  <= sum_tmp;
                    mean_val <= sum_tmp >>> 5;  // /32
                    next_state <= S_SUB_GAMMA;
                end

                //==============================================================
                // S_SUB_GAMMA：逐维做 (x_i - mean)*gamma_i + beta_i
                //==============================================================
                S_SUB_GAMMA: begin:c
                    integer k;
                    reg signed [DATA_WIDTH+GUARD_BITS:0] diff_val;
                    reg signed [31:0] mult_stage;
                    reg signed [39:0] beta_ext;

                    // 这里我们直接把 gamma/beta 当普通整数使用，
                    // gamma_mant=1, beta_mant=0 就是最简单的 LN-lite
                    for (k = 0; k < DIM; k = k + 1) begin
                        diff_val   = token_mant[k] - mean_val;   // 宽一些防止溢出

                        mult_stage = diff_val * gamma_mant[k];   // 32bit
                        beta_ext   = {{32{beta_mant[k][DATA_WIDTH-1]}}, beta_mant[k]};

                        affine_val[k] <= {{8{mult_stage[31]}}, mult_stage} + beta_ext;
                    end

                    // 本实现不真正使用 BFP 指数，这里简单沿用 gamma 的指数
                    out_exp_reg <= gamma_exp_reg;

                    next_state <= S_FIND_MAX;
                end

                //==============================================================
                // S_FIND_MAX：找当前 token 中绝对值最大值，确定 out_shift
                //==============================================================
                S_FIND_MAX: begin:highz
                    integer k;
                    reg signed [39:0] local_max;
                    reg signed [39:0] abs_temp;

                    local_max = 40'd0;
                    for (k = 0; k < DIM; k = k + 1) begin
                        abs_temp = (affine_val[k] < 0) ? -affine_val[k] : affine_val[k];
                        if (abs_temp > local_max)
                            local_max = abs_temp;
                    end
                    max_abs_mant <= local_max;

                    // 根据 local_max 决定需要右移多少位控制在 DATA_WIDTH 里
                    out_shift <= 6'd0;
                    for (k = 39; k >= (DATA_WIDTH-1); k = k - 1) begin
                        if (local_max[k]) begin
                            if (k > (DATA_WIDTH-2))
                                out_shift <= k - (DATA_WIDTH-2);
                            else
                                out_shift <= 6'd0;
                        end
                    end

                    next_state <= S_OUTPUT_NORM;
                end

                //==============================================================
                // S_OUTPUT_NORM：按 out_shift 归一化并饱和到 DATA_WIDTH
                //==============================================================
                S_OUTPUT_NORM: begin:z
                    integer k;
                    reg signed [39:0] shifted_val;
                    reg signed [DATA_WIDTH-1:0] sat_val;

                    // 可以选：out_exp_reg 保持不变 或 +out_shift，这里先保持不变
                    for (k = 0; k < DIM; k = k + 1) begin
                        shifted_val = affine_val[k] >>> out_shift;

                        // 饱和到 [-128,127]
                        if (shifted_val > $signed({1'b0, {(DATA_WIDTH-1){1'b1}}}))
                            sat_val = {1'b0, {(DATA_WIDTH-1){1'b1}}};
                        else if (shifted_val < $signed({1'b1, {(DATA_WIDTH-1){1'b0}}}))
                            sat_val = {1'b1, {(DATA_WIDTH-1){1'b0}}};
                        else
                            sat_val = shifted_val[DATA_WIDTH-1:0];

                        out_mant_arr[k] <= sat_val;
                    end

                    next_state <= S_WRITE_OUT;
                end

                //==============================================================
                // S_WRITE_OUT：写回当前 token
                //==============================================================
                S_WRITE_OUT: begin
                    if (output_ready) begin
                        output_wr_en   <= 1'b1;
                        output_wr_addr <= token_idx[ADDR_WIDTH-1:0];
                        output_exp     <= out_exp_reg;

                        for (i = 0; i < DIM; i = i + 1) begin
                            output_mant[i*DATA_WIDTH +: DATA_WIDTH] <= out_mant_arr[i];
                        end

                        output_valid   <= 1'b1;
                        next_state     <= S_NEXT_TOKEN;
                    end else begin
                        output_wr_en   <= 1'b0;
                        output_valid   <= 1'b0;
                        next_state     <= S_WRITE_OUT;
                    end
                end

                //==============================================================
                // S_NEXT_TOKEN：准备下一个 token
                //==============================================================
                S_NEXT_TOKEN: begin
                    output_wr_en  <= 1'b0;
                    output_valid  <= 1'b0;

                    token_idx <= token_idx + 1'b1;

                    if (token_idx + 1 < TOKEN_NUM)
                        next_state <= S_PREP_TOKEN;
                    else
                        next_state <= S_DONE;
                end

                //==============================================================
                // S_DONE：一次 LN 完成
                //==============================================================
                S_DONE: begin
                    done       <= 1'b1;
                    next_state <= S_IDLE;
                end

                default: begin
                    next_state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
