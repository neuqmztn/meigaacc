module layer_norm_bfp #(
    parameter TOKEN_NUM    = 641,
    parameter DIM          = 32,
    parameter DATA_WIDTH   = 8,
    parameter EXP_WIDTH    = 8,
    parameter ADDR_WIDTH   = 10,
    parameter GUARD_BITS   = 3,      
    parameter ACCUM_WIDTH  = 24      
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire start,
    input  wire mode,         // 0 = LN1 (使用参数组0), 1 = LN2 (使用参数组1)
    output reg  done,
    output wire busy,
    output reg  error,
    
    //==========================================================================
    // 输入/输出/参数接口 (保持不变)
    //==========================================================================
    output reg  input_rd_en,
    output reg  [ADDR_WIDTH-1:0] input_rd_addr,
    input  wire [EXP_WIDTH-1:0] input_exp,
    input  wire [DIM*DATA_WIDTH-1:0] input_mant,
    input  wire input_valid,
    
    output reg  param_rd_en,
    output reg  param_rd_gamma,
    input  wire [EXP_WIDTH-1:0] param_exp,
    input  wire [DIM*DATA_WIDTH-1:0] param_mant,
    input  wire param_valid,
    
    output reg  output_wr_en,
    output reg  [ADDR_WIDTH-1:0] output_wr_addr,
    output reg  [EXP_WIDTH-1:0] output_exp,
    output reg  [DIM*DATA_WIDTH-1:0] output_mant,
    output reg  output_valid,
    input  wire output_ready
);

    // 状态机定义
    localparam IDLE           = 5'd0;
    localparam LOAD_GAMMA     = 5'd1;
    localparam LOAD_BETA      = 5'd2;
    localparam READ_WAIT      = 5'd3;
    localparam PREP_TOKEN     = 5'd4;
    localparam CALC_SUM       = 5'd5;
    localparam CALC_VAR       = 5'd6;
    localparam ISQRT_NORM     = 5'd7;
    localparam ISQRT_CALC     = 5'd8; 
    localparam APPLY_GAMMA    = 5'd9;
    localparam ADD_BETA       = 5'd10;
    localparam FIND_MAX       = 5'd11;
    localparam OUTPUT_NORM    = 5'd12;
    localparam WRITE_OUT      = 5'd13;
    localparam NEXT_TOKEN     = 5'd14;
    localparam DONE_STATE     = 5'd15;

    //================================================================================
    // 内部寄存器
    //================================================================================
    reg [4:0] state;
    reg [9:0] token_idx;
    reg [3:0] wait_cnt;
    integer i;
    
    // ------------------- 参数存储 (双组: [0]对应LN1, [1]对应LN2) -------------------
    // 增加第一维度 [0:1]
    reg [EXP_WIDTH-1:0] gamma_exp_reg [0:1];
    reg signed [DATA_WIDTH-1:0] gamma_mant_arr [0:1][0:DIM-1]; // 2D Array
    
    reg [EXP_WIDTH-1:0] beta_exp_reg [0:1];
    reg signed [DATA_WIDTH-1:0] beta_mant_arr [0:1][0:DIM-1];  // 2D Array
    
    reg [1:0] params_loaded; // Bit 0: LN1 loaded, Bit 1: LN2 loaded
    reg active_mode;         // 锁存当前的 mode，防止运行中输入变化

    // ------------------- 数据与计算流水线 -------------------
    reg [EXP_WIDTH-1:0] curr_input_exp;
    reg signed [DATA_WIDTH-1:0] curr_input_mant [0:DIM-1];

    reg signed [ACCUM_WIDTH-1:0] sum_val;
    reg signed [ACCUM_WIDTH+DATA_WIDTH-1:0] sum_sq_val; 
    reg signed [DATA_WIDTH-1:0] mean_val;
    reg signed [ACCUM_WIDTH-1:0] var_val;

    reg [4:0] var_lzc;
    reg signed [DATA_WIDTH+2:0] inv_std_mant; 
    reg signed [EXP_WIDTH+1:0] inv_std_exp;

    reg signed [31:0] term_a_mant [0:DIM-1]; 
    reg signed [EXP_WIDTH+1:0] term_a_exp;
    reg signed [39:0] term_final_mant [0:DIM-1]; 
    reg signed [EXP_WIDTH+1:0] term_final_exp;
    reg signed [39:0] max_abs_val;

    //================================================================================
    // 辅助函数: CLZ
    //================================================================================
    function automatic [4:0] count_leading_zeros;
        input [23:0] val; 
        reg [4:0] cnt;
        reg [23:0] val_temp; 
        begin
            cnt = 0;
            val_temp = val;
            if (val_temp[23:8] == 16'b0) begin cnt = cnt + 16; val_temp = val_temp << 16; end
            if (val_temp[23:16] == 8'b0) begin cnt = cnt + 8;  val_temp = val_temp << 8;  end
            if (val_temp[23:20] == 4'b0) begin cnt = cnt + 4;  val_temp = val_temp << 4;  end
            if (val_temp[23:22] == 2'b0) begin cnt = cnt + 2;  val_temp = val_temp << 2;  end
            if (val_temp[23] == 1'b0)    cnt = cnt + 1;
            count_leading_zeros = cnt;
        end
    endfunction

    //================================================================================
    // 主逻辑
    //================================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            token_idx <= 0;
            input_rd_en <= 0;
            output_wr_en <= 0;
            output_valid <= 0;
            param_rd_en <= 0;
            done <= 0;
            error <= 0;
            params_loaded <= 2'b00; // 复位时两个参数组都标记为未加载
            active_mode <= 0;
        end else begin
            // 脉冲信号复位
            input_rd_en <= 0;
            output_wr_en <= 0;
            output_valid <= 0;
            param_rd_en <= 0;

            case (state)
                IDLE: begin
                    done <= 0;
                    token_idx <= 0;
                    if (start) begin
                        active_mode <= mode; // 锁存当前模式
                        // 检查当前模式对应的参数是否已加载
                        if (params_loaded[mode]) begin
                            state <= PREP_TOKEN;
                        end else begin
                            state <= LOAD_GAMMA; // 如果没加载，先去加载
                        end
                    end
                end

                //------------------------------------------------------------
                // 1. 加载参数 (根据 active_mode 加载到对应 Bank)
                //------------------------------------------------------------
                LOAD_GAMMA: begin
                    param_rd_en <= 1; 
                    param_rd_gamma <= 1;
                    state <= READ_WAIT; 
                    wait_cnt <= 0;
                end
                LOAD_BETA: begin
                    param_rd_en <= 1; 
                    param_rd_gamma <= 0;
                    state <= READ_WAIT; 
                    wait_cnt <= 0;
                end

                READ_WAIT: begin
                    wait_cnt <= wait_cnt + 1;
                    
                    if (param_valid) begin
                        if (param_rd_gamma) begin
                            // 写入对应 Bank
                            gamma_exp_reg[active_mode] <= param_exp;
                            for (i=0; i<DIM; i=i+1) 
                                gamma_mant_arr[active_mode][i] <= param_mant[i*DATA_WIDTH +: DATA_WIDTH];
                            state <= LOAD_BETA;
                        end else begin
                            // 写入对应 Bank
                            beta_exp_reg[active_mode] <= param_exp;
                            for (i=0; i<DIM; i=i+1) 
                                beta_mant_arr[active_mode][i] <= param_mant[i*DATA_WIDTH +: DATA_WIDTH];
                            
                            // 标记当前 mode 参数已加载
                            params_loaded[active_mode] <= 1'b1;
                            state <= PREP_TOKEN;
                        end
                    end 
                    else if (input_valid) begin
                        // 输入数据返回
                        curr_input_exp <= input_exp;
                        for (i=0; i<DIM; i=i+1) 
                            curr_input_mant[i] <= input_mant[i*DATA_WIDTH +: DATA_WIDTH];
                        state <= CALC_SUM;
                    end 
                    else if (wait_cnt > 12) begin
                        error <= 1; state <= DONE_STATE;
                    end
                end

                //------------------------------------------------------------
                // 2. 读取 Token 数据
                //------------------------------------------------------------
                PREP_TOKEN: begin
                    input_rd_en <= 1;
                    input_rd_addr <= token_idx;
                    state <= READ_WAIT;
                    wait_cnt <= 0;
                end

                //------------------------------------------------------------
                // 3. 计算流水线 (Sum, Var, InvSqrt) - 与之前版本相同
                //------------------------------------------------------------
                CALC_SUM: begin
                    begin : blk_sum
                        reg signed [ACCUM_WIDTH:0] temp_sum; 
                        reg signed [ACCUM_WIDTH+DATA_WIDTH:0] temp_sum_sq;
                        reg signed [DATA_WIDTH:0] val_ext;
                        
                        temp_sum = 0;
                        temp_sum_sq = 0;
                        for (i=0; i<DIM; i=i+1) begin
                            val_ext = curr_input_mant[i];
                            temp_sum = temp_sum + val_ext;
                            temp_sum_sq = temp_sum_sq + (val_ext * val_ext);
                        end
                        sum_val <= temp_sum[ACCUM_WIDTH-1:0];
                        sum_sq_val <= temp_sum_sq[ACCUM_WIDTH+DATA_WIDTH-1:0];
                    end
                    state <= CALC_VAR;
                end

                CALC_VAR: begin
                    mean_val <= sum_val >>> 5;
                    var_val <= (sum_sq_val >>> 5) - ((sum_val >>> 5) * (sum_val >>> 5));
                    state <= ISQRT_NORM;
                end

                ISQRT_NORM: begin
                    if (var_val <= 0) begin
                        inv_std_mant <= {1'b0, 10'h1FF}; 
                        inv_std_exp <= 0;
                    end else begin
                        var_lzc <= count_leading_zeros(var_val[23:0]);
                    end
                    state <= ISQRT_CALC;
                end

                ISQRT_CALC: begin // 无 ROM 线性逼近
                    if (var_val > 0) begin:q
                        reg [4:0] shift_amt;
                        reg [23:0] var_shifted;
                        reg [9:0] linear_approx;

                        shift_amt = (var_lzc[0]) ? (var_lzc - 1) : var_lzc;
                        var_shifted = var_val << shift_amt;
                        linear_approx = 10'd384 - ({1'b0, var_shifted[23:15]} >> 1);
                        
                        inv_std_mant <= {1'b0, linear_approx};
                        inv_std_exp <= shift_amt >> 1;
                    end
                    state <= APPLY_GAMMA;
                end

                //------------------------------------------------------------
                // 4. 应用仿射变换 (关键修改点：根据 active_mode 选择参数)
                //------------------------------------------------------------
                APPLY_GAMMA: begin
                    // 使用 active_mode 索引 gamma_exp_reg 和 gamma_mant_arr
                    term_a_exp <= curr_input_exp + gamma_exp_reg[active_mode] + inv_std_exp;
                    
                    for (i=0; i<DIM; i=i+1) begin:a
                        reg signed [DATA_WIDTH:0] diff_val;
                        diff_val = curr_input_mant[i] - mean_val;
                        // 选取对应的 Gamma 参数
                        term_a_mant[i] <= diff_val * inv_std_mant * gamma_mant_arr[active_mode][i];
                    end
                    state <= ADD_BETA;
                end

                ADD_BETA: begin
                    begin : align_block
                        reg signed [EXP_WIDTH+1:0] exp_diff;
                        reg signed [39:0] op_a, op_b;
                        
                        // 使用 active_mode 索引 beta_exp_reg
                        exp_diff = $signed(term_a_exp) - $signed({2'b0, beta_exp_reg[active_mode]});

                        if (exp_diff >= 0) begin
                            term_final_exp <= term_a_exp;
                            for (i=0; i<DIM; i=i+1) begin
                                op_a = term_a_mant[i];
                                // 选取对应的 Beta 参数
                                op_b = ($signed(beta_mant_arr[active_mode][i]) <<< 8); 
                                if (exp_diff < 30) op_b = op_b >>> exp_diff;
                                else op_b = 0;
                                term_final_mant[i] <= op_a + op_b;
                            end
                        end else begin
                            term_final_exp <= beta_exp_reg[active_mode];
                            for (i=0; i<DIM; i=i+1) begin
                                // 选取对应的 Beta 参数
                                op_b = ($signed(beta_mant_arr[active_mode][i]) <<< 8);
                                op_a = term_a_mant[i];
                                if ((-exp_diff) < 30) op_a = op_a >>> (-exp_diff);
                                else op_a = 0;
                                term_final_mant[i] <= op_a + op_b;
                            end
                        end
                    end
                    state <= FIND_MAX;
                end

                //------------------------------------------------------------
                // 5. 输出处理
                //------------------------------------------------------------
                FIND_MAX: begin
                    begin:z
                        reg signed [39:0] local_max, abs_temp;
                        local_max = 0;
                        for (i=0; i<DIM; i=i+1) begin
                            abs_temp = (term_final_mant[i] < 0) ? -term_final_mant[i] : term_final_mant[i];
                            if (abs_temp > local_max) local_max = abs_temp;
                        end
                        max_abs_val <= local_max;
                    end
                    state <= OUTPUT_NORM;
                end

                OUTPUT_NORM: begin
                    output_exp <= term_final_exp[EXP_WIDTH-1:0];
                    for (i=0; i<DIM; i=i+1) begin:s
                        reg signed [39:0] val_rnd;
                        reg signed [7:0] val_sat;
                        val_rnd = term_final_mant[i] + 40'd128; 
                        val_rnd = val_rnd >>> 8;
                        if (val_rnd > 127) val_sat = 8'd127;
                        else if (val_rnd < -128) val_sat = -8'd128;
                        else val_sat = val_rnd[7:0];
                        output_mant[i*DATA_WIDTH +: DATA_WIDTH] <= val_sat;
                    end
                    state <= WRITE_OUT;
                end

                WRITE_OUT: begin
                    output_wr_en <= 1; output_valid <= 1;
                    output_wr_addr <= token_idx;
                    if (output_ready) state <= NEXT_TOKEN;
                end

                NEXT_TOKEN: begin
                    if (token_idx < TOKEN_NUM - 1) begin
                        token_idx <= token_idx + 1;
                        state <= PREP_TOKEN;
                    end else begin
                        state <= DONE_STATE;
                    end
                end

                DONE_STATE: begin
                    done <= 1;
                    if (!start) state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

    assign busy = (state != IDLE) && (state != DONE_STATE);

endmodule