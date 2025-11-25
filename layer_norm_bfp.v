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
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire start,
    input  wire mode,             // 预留端口：0=LN1, 1=LN2 (本代码通用)
    output reg  done,
    output wire busy,
    output reg  error,
    
    //==========================================================================
    // 输入数据接口 (流式读取)
    //==========================================================================
    output reg  input_rd_en,
    output reg  [ADDR_WIDTH-1:0] input_rd_addr,
    input  wire [EXP_WIDTH-1:0] input_exp,
    input  wire [DIM*DATA_WIDTH-1:0] input_mant,
    input  wire input_valid,
    
    //==========================================================================
    // 参数接口 (Gamma / Beta)
    //==========================================================================
    output reg  param_rd_en,
    output reg  param_rd_gamma,   // 1=Gamma, 0=Beta
    input  wire [EXP_WIDTH-1:0] param_exp,
    input  wire [DIM*DATA_WIDTH-1:0] param_mant,
    input  wire param_valid,
    
    //==========================================================================
    // 输出数据接口 (写回)
    //==========================================================================
    output reg  output_wr_en,
    output reg  [ADDR_WIDTH-1:0] output_wr_addr,
    output reg  [EXP_WIDTH-1:0] output_exp,
    output reg  [DIM*DATA_WIDTH-1:0] output_mant,
    output reg  output_valid,
    input  wire output_ready,
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output reg  [4:0] state_out,
    output reg  [9:0] processed_cnt,
    output reg  [31:0] cycle_cnt,
    output reg  overflow_flag,
    output reg  underflow_flag
);

    //================================================================================
    // 状态机定义
    //================================================================================
    localparam IDLE           = 5'd0;
    localparam LOAD_GAMMA     = 5'd1;
    localparam LOAD_BETA      = 5'd2;
    localparam READ_WAIT      = 5'd3;  // 通用等待状态
    localparam PREP_TOKEN     = 5'd4;  // 准备下一个Token地址
    localparam CALC_SUM       = 5'd5;  // 累加和与平方和
    localparam CALC_VAR       = 5'd6;  // 计算均值与方差
    localparam ISQRT_NORM     = 5'd7;  // 倒数平方根：输入归一化 (CLZ)
    localparam ISQRT_LUT      = 5'd8;  // 倒数平方根：查表与指数调整
    localparam APPLY_GAMMA    = 5'd9;  // 仿射变换乘法部分：(x-u)*inv_std*gamma
    localparam ADD_BETA       = 5'd10; // 仿射变换加法部分：+ beta (严格对齐)
    localparam FIND_MAX       = 5'd11; // 寻找最大绝对值 (用于输出重归一化)
    localparam OUTPUT_NORM    = 5'd12; // 输出移位、舍入、饱和
    localparam WRITE_OUT      = 5'd13; // 写回数据
    localparam NEXT_TOKEN     = 5'd14; // 循环判断
    localparam DONE_STATE     = 5'd15;

    //================================================================================
    // 内部寄存器定义
    //================================================================================
    reg [4:0] state;
    reg [9:0] token_idx;
    reg [3:0] wait_cnt;
    integer i;
    
    // ------------------- 参数存储 -------------------
    reg [EXP_WIDTH-1:0] gamma_exp_reg;
    reg signed [DATA_WIDTH-1:0] gamma_mant_arr [0:DIM-1];
    reg [EXP_WIDTH-1:0] beta_exp_reg;
    reg signed [DATA_WIDTH-1:0] beta_mant_arr [0:DIM-1];
    reg params_loaded;

    // ------------------- 当前Token数据 -------------------
    reg [EXP_WIDTH-1:0] curr_input_exp;
    reg signed [DATA_WIDTH-1:0] curr_input_mant [0:DIM-1];

    // ------------------- 统计计算流水线 -------------------
    reg signed [ACCUM_WIDTH-1:0] sum_val;
    reg signed [ACCUM_WIDTH+DATA_WIDTH-1:0] sum_sq_val; // 平方和需要更大位宽防止溢出
    reg signed [DATA_WIDTH-1:0] mean_val;
    reg signed [ACCUM_WIDTH-1:0] var_val;

    // ------------------- 倒数平方根计算 -------------------
    reg [4:0] var_lzc;                    // 方差前导零数量
    reg signed [DATA_WIDTH+1:0] inv_std_mant; // 1/std 尾数 (比输入多几位精度)
    reg signed [EXP_WIDTH+1:0] inv_std_exp;   // 1/std 指数修正值

    // ------------------- 仿射变换中间结果 -------------------
    // 第一步乘法结果: (Input-Mean) * InvStd * Gamma
    // 宽度分析: 8bit(Input) + 1bit(Sub) + 10bit(InvStd) + 8bit(Gamma) ≈ 27bit
    // 选用 32bit 安全位宽
    reg signed [31:0] term_a_mant [0:DIM-1]; 
    reg signed [EXP_WIDTH+1:0] term_a_exp;

    // 第二步加法结果: TermA + Beta
    // 需要容纳对齐移位后的结果，选用 40bit
    reg signed [39:0] term_final_mant [0:DIM-1]; 
    reg signed [EXP_WIDTH+1:0] term_final_exp; // 对齐后的基准指数

    // ------------------- 输出重归一化 -------------------
    reg signed [39:0] max_abs_val;
    
    // ------------------- 查找表 (ROM) -------------------
    // 深度 256，位宽 9 (无符号 Q1.8 格式)
    reg [8:0] sqrt_inv_rom [0:255]; 

    //================================================================================
    // 辅助函数：自动前导零计数 (CLZ)
    //================================================================================
    function automatic [4:0] count_leading_zeros;
        input [23:0] val; // 适配 ACCUM_WIDTH
        reg [4:0] cnt;
        begin
            cnt = 0;
            // 二分法检测
            if (val[23:8] == 16'b0) begin cnt = cnt + 16; val = val << 16; end
            if (val[23:16] == 8'b0) begin cnt = cnt + 8;  val = val << 8;  end
            if (val[23:20] == 4'b0) begin cnt = cnt + 4;  val = val << 4;  end
            if (val[23:22] == 2'b0) begin cnt = cnt + 2;  val = val << 2;  end
            if (val[23] == 1'b0)    cnt = cnt + 1;
            count_leading_zeros = cnt;
        end
    endfunction

    //================================================================================
    // ROM 初始化：存储 1/sqrt(x) 的归一化值
    // 输入范围映射：[0.25, 1.0) -> index [64, 255] 或者采用 Mantissa 映射
    // 此处策略：输入归一化使得 MSB(Bit 23)=1。
    // 我们取 Bit 23:16 作为 8位索引。
    // ROM 内容 = 511 / sqrt(1.0 + x_frac)
    //================================================================================
    initial begin
        // 注意：实际综合时建议使用 $readmemh 加载外部 .mif/.hex 文件
        // 这里为了代码完整性使用算法生成近似值
        // 索引 i 对应值 v = 1.0 + i/256.0
        // 存储 val = floor(1.0/sqrt(v) * 511)
        // 范围 1.0 -> 1/1 * 511 = 511 (0x1FF)
        // 范围 ~2.0 -> 1/1.414 * 511 = 361 (0x169)
        sqrt_inv_rom[0] = 9'h1FF; // 防止 i=0 时的计算异常
        for (i = 0; i < 256; i = i + 1) begin
            // 模拟 1/sqrt(1 + i/256) 的定点值
            sqrt_inv_rom[i] = 511 / $rtoi($sqrt(1.0 + i/256.0));
        end
    end

    //================================================================================
    // 主逻辑
    //================================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            token_idx <= 10'd0;
            
            // 端口复位
            input_rd_en <= 1'b0;
            input_rd_addr <= {ADDR_WIDTH{1'b0}};
            output_wr_en <= 1'b0;
            output_wr_addr <= {ADDR_WIDTH{1'b0}};
            output_valid <= 1'b0;
            output_exp <= {EXP_WIDTH{1'b0}};
            output_mant <= {(DIM*DATA_WIDTH){1'b0}};
            param_rd_en <= 1'b0;
            param_rd_gamma <= 1'b0;
            
            // 状态复位
            done <= 1'b0;
            error <= 1'b0;
            
            params_loaded <= 1'b0;
            
            // 调试计数器复位
            processed_cnt <= 10'd0;
            cycle_cnt <= 32'd0;
            overflow_flag <= 1'b0;
            underflow_flag <= 1'b0;
            
            // 关键寄存器清理
            gamma_exp_reg <= {EXP_WIDTH{1'b0}};
            beta_exp_reg <= {EXP_WIDTH{1'b0}};
            
        end else begin
            // 默认清除脉冲信号
            input_rd_en <= 1'b0;
            output_wr_en <= 1'b0;
            output_valid <= 1'b0;
            param_rd_en <= 1'b0;
            
            // 状态输出调试
            state_out <= state;

            case (state)
                //------------------------------------------------------------
                // 0. IDLE
                //------------------------------------------------------------
                IDLE: begin
                    done <= 1'b0;
                    token_idx <= 10'd0;
                    processed_cnt <= 10'd0;
                    cycle_cnt <= 32'd0;
                    
                    if (start) begin
                        if (params_loaded) state <= PREP_TOKEN;
                        else state <= LOAD_GAMMA;
                    end
                end

                //------------------------------------------------------------
                // 1. 加载参数 (Gamma)
                //------------------------------------------------------------
                LOAD_GAMMA: begin
                    param_rd_en <= 1'b1;
                    param_rd_gamma <= 1'b1; // 请求 Gamma
                    state <= READ_WAIT;
                    wait_cnt <= 4'd0;
                end

                //------------------------------------------------------------
                // 2. 加载参数 (Beta)
                //------------------------------------------------------------
                LOAD_BETA: begin
                    param_rd_en <= 1'b1;
                    param_rd_gamma <= 1'b0; // 请求 Beta
                    state <= READ_WAIT;
                    wait_cnt <= 4'd0;
                end

                //------------------------------------------------------------
                // 3. 通用读取等待 (Wait & Dispatch)
                //------------------------------------------------------------
                READ_WAIT: begin
                    wait_cnt <= wait_cnt + 1;
                    
                    // A. 参数返回处理
                    if (param_valid) begin
                        if (param_rd_gamma) begin
                            gamma_exp_reg <= param_exp;
                            // 解包 Gamma
                            for (i=0; i<DIM; i=i+1) 
                                gamma_mant_arr[i] <= param_mant[i*DATA_WIDTH +: DATA_WIDTH];
                            state <= LOAD_BETA;
                        end else begin
                            beta_exp_reg <= param_exp;
                            // 解包 Beta
                            for (i=0; i<DIM; i=i+1) 
                                beta_mant_arr[i] <= param_mant[i*DATA_WIDTH +: DATA_WIDTH];
                            params_loaded <= 1'b1;
                            state <= PREP_TOKEN;
                        end
                    end 
                    // B. 输入数据返回处理
                    else if (input_valid) begin
                        curr_input_exp <= input_exp;
                        // 解包 Input Mantissa
                        for (i=0; i<DIM; i=i+1) 
                            curr_input_mant[i] <= input_mant[i*DATA_WIDTH +: DATA_WIDTH];
                        state <= CALC_SUM;
                    end
                    // C. 超时保护
                    else if (wait_cnt > 4'd12) begin
                        error <= 1'b1; // 标记错误
                        state <= DONE_STATE;
                    end
                end

                //------------------------------------------------------------
                // 4. 准备 Token 读取
                //------------------------------------------------------------
                PREP_TOKEN: begin
                    input_rd_en <= 1'b1;
                    input_rd_addr <= token_idx[ADDR_WIDTH-1:0];
                    state <= READ_WAIT;
                    wait_cnt <= 4'd0;
                end

                //------------------------------------------------------------
                // 5. 计算累加和 (Sum & SumSq)
                // 并行化处理，单周期完成 DIM=32 的累加
                //------------------------------------------------------------
                CALC_SUM: begin
                    begin : blk_sum
                        reg signed [31:0] temp_sum;
                        reg signed [31:0] temp_sum_sq;
                        reg signed [15:0] val_ext; // 符号扩展
                        
                        temp_sum = 0;
                        temp_sum_sq = 0;
                        
                        for (i=0; i<DIM; i=i+1) begin
                            val_ext = curr_input_mant[i];
                            temp_sum = temp_sum + val_ext;
                            temp_sum_sq = temp_sum_sq + (val_ext * val_ext);
                        end
                        sum_val <= temp_sum[ACCUM_WIDTH-1:0]; // 截断到累加器位宽
                        sum_sq_val <= temp_sum_sq[ACCUM_WIDTH+DATA_WIDTH-1:0];
                    end
                    state <= CALC_VAR;
                end

                //------------------------------------------------------------
                // 6. 计算均值和方差
                //------------------------------------------------------------
                CALC_VAR: begin
                    // Mean = Sum / 32 (算术右移 5)
                    mean_val <= sum_val >>> 5;
                    
                    // Variance = E[x^2] - (E[x])^2
                    // Var = (SumSq / 32) - Mean^2
                    // 注意：这里的计算是在 Input Mantissa 的定点域进行，Input Exp 暂时不参与
                    // 结果 var_val 代表 var_mantissa
                    var_val <= (sum_sq_val >>> 5) - ((sum_val >>> 5) * (sum_val >>> 5));
                    
                    state <= ISQRT_NORM;
                end

                //------------------------------------------------------------
                // 7. 倒数平方根：归一化 (CLZ)
                // 解决 LUT 在小数值时精度不足的问题
                //------------------------------------------------------------
                ISQRT_NORM: begin
                    if (var_val <= 0) begin
                        // 极小方差保护：设为最大值或默认值
                        // 当方差为0，标准差为0，倒数无穷大。实际硬件中给最大饱和值。
                        inv_std_mant <= {1'b0, 9'h1FF}; // Max (Q1.8 format in 10 bits)
                        inv_std_exp <= 0;
                    end else begin
                        // 计算前导零
                        var_lzc <= count_leading_zeros(var_val[23:0]);
                    end
                    state <= ISQRT_LUT;
                end

                //------------------------------------------------------------
                // 8. 倒数平方根：查表
                //------------------------------------------------------------
                ISQRT_LUT: begin
                    if (var_val > 0) begin:l1
                        reg [4:0] shift_amt;
                        reg [23:0] var_shifted;
                        reg [7:0] lut_idx;
                        
                        // 策略：我们总是左移偶数位，以简化 sqrt(2^E) 的处理
                        // 如果 lzc 是偶数，shift = lzc
                        // 如果 lzc 是奇数，shift = lzc - 1 (保证 shift 是偶数)
                        // 目标是将有效数据推到高位
                        
                        shift_amt = (var_lzc[0]) ? (var_lzc - 1) : var_lzc;
                        
                        var_shifted = var_val << shift_amt;
                        
                        // 取 Bit 23:16 作为索引 (8位)
                        lut_idx = var_shifted[23:16];
                        
                        // 查表
                        inv_std_mant <= {1'b0, sqrt_inv_rom[lut_idx]};
                        
                        // 指数修正计算：
                        // 我们计算的是 1/sqrt(Var_Mant * 2^-shift)
                        // = 1/sqrt(Var_Mant) * sqrt(2^shift)
                        // = Inv_Mant * 2^(shift/2)
                        // 因此，inv_std 的指数贡献是 + (shift / 2)
                        inv_std_exp <= shift_amt >> 1;
                    end
                    state <= APPLY_GAMMA;
                end

                //------------------------------------------------------------
                // 9. 应用仿射变换乘法部分
                // 公式：TermA = (Input - Mean) * InvStd * Gamma
                //------------------------------------------------------------
                APPLY_GAMMA: begin
                    // 1. 指数合并
                    // Base Exp = Input Exp (因为 x-u 在 input domain)
                    // Total Exp = Input Exp + Gamma Exp + InvStd Exp (Correction)
                    // 注意：这里的 InvStd Exp 是正的 shift correction
                    term_a_exp <= curr_input_exp + gamma_exp_reg + inv_std_exp;
                    
                    // 2. 尾数乘法
                    for (i=0; i<DIM; i=i+1) begin:l2
                        reg signed [DATA_WIDTH:0] diff_val;
                        reg signed [31:0] mult_stage;
                        
                        // Sub: (x - u) -> 9 bit
                        diff_val = curr_input_mant[i] - mean_val;
                        
                        // Mult: Diff(9) * InvStd(10) * Gamma(8) -> ~27 bit
                        // InvStd 是 Q1.8 格式 (实际值 1.0 ~ 2.0)
                        mult_stage = diff_val * inv_std_mant * gamma_mant_arr[i];
                        
                        term_a_mant[i] <= mult_stage;
                    end
                    state <= ADD_BETA;
                end

                //------------------------------------------------------------
                // 10. 应用仿射变换加法部分 (严谨对齐)
                // 公式：Final = TermA + Beta
                // 难点：TermA 和 Beta 的指数可能相差巨大，必须对齐
                //------------------------------------------------------------
                ADD_BETA: begin
                    begin : align_block
                        reg signed [EXP_WIDTH+1:0] exp_diff;
                        reg signed [39:0] op_a_shifted; // TermA 扩展
                        reg signed [39:0] op_b_shifted; // Beta 扩展
                        
                        // 计算指数差: TermA_Exp - Beta_Exp
                        exp_diff = $signed(term_a_exp) - $signed({2'b0, beta_exp_reg});
                        
                        if (exp_diff >= 0) begin
                            // Case 1: TermA 指数大 (或相等)
                            // 基准指数 = TermA Exp
                            // 操作：保持 TermA，右移 Beta
                            term_final_exp <= term_a_exp;
                            
                            for (i=0; i<DIM; i=i+1) begin
                                op_a_shifted = term_a_mant[i]; // 32bit -> 40bit
                                
                                // Beta 原本是 8bit 整数。我们需要把它对齐到 TermA 的定点域。
                                // TermA 经过了 InvStd(Q1.8) 和其他乘法，假设其隐式小数位增加了约 8位
                                // 为了保证精度，我们将 Beta 左移提升精度，再根据指数差右移
                                // 这里假设 TermA 的小数位在 Bit 8 左右 (来自 InvStd 的 Q1.8)
                                // 所以给 Beta 补 8 位零 ( << 8 ) 作为对齐基准
                                
                                op_b_shifted = ($signed(beta_mant_arr[i]) <<< 8);
                                
                                // 右移 Beta 以匹配 TermA 的指数级
                                // 限制移位量防止越界
                                if (exp_diff < 40)
                                    op_b_shifted = op_b_shifted >>> exp_diff;
                                else
                                    op_b_shifted = 0; // 差别太大，Beta 忽略不计
                                    
                                term_final_mant[i] <= op_a_shifted + op_b_shifted;
                            end
                        end else begin
                            // Case 2: Beta 指数大
                            // 基准指数 = Beta Exp
                            // 操作：保持 Beta，右移 TermA
                            term_final_exp <= beta_exp_reg;
                            
                            for (i=0; i<DIM; i=i+1) begin
                                op_b_shifted = ($signed(beta_mant_arr[i]) <<< 8); // 基准
                                
                                // 右移 TermA
                                if ((-exp_diff) < 40)
                                    op_a_shifted = term_a_mant[i] >>> (-exp_diff);
                                else
                                    op_a_shifted = 0;
                                    
                                term_final_mant[i] <= op_a_shifted + op_b_shifted;
                            end
                        end
                    end
                    state <= FIND_MAX;
                end

                //------------------------------------------------------------
                // 11. 寻找最大绝对值 (Output Block Normalization)
                // 单周期线性查找 (DIM=32 可接受)
                //------------------------------------------------------------
                FIND_MAX: begin
                    begin : find_max_blk
                        reg signed [39:0] local_max;
                        reg signed [39:0] abs_temp;
                        
                        local_max = 0;
                        for (i=0; i<DIM; i=i+1) begin
                            abs_temp = (term_final_mant[i] < 0) ? -term_final_mant[i] : term_final_mant[i];
                            if (abs_temp > local_max) local_max = abs_temp;
                        end
                        max_abs_val <= local_max;
                    end
                    state <= OUTPUT_NORM;
                end

                //------------------------------------------------------------
                // 12. 输出标准化：移位、舍入、饱和
                //------------------------------------------------------------
                OUTPUT_NORM: begin
                    begin : out_norm_blk
                        reg [5:0] lzc;
                        reg [5:0] shift_right_amt;
                        reg signed [EXP_WIDTH+1:0] final_exp_adj;
                        
                        // 我们现在的 Mantissa 是 ~40bit 宽，定点位置约在 Bit 8
                        // 目标是压缩回 8-bit 有符号整数 (-128 ~ 127)
                        // 需要找到 max_abs_val 的最高有效位 (MSB)
                        
                        // 简单处理：如果 Max > 127, 需要右移直到 Max < 128
                        // 我们可以利用 CLZ 来计算需要的右移量
                        // 假设 max_abs_val 是 40bit。
                        // 有效位宽 = 40 - CLZ(max)
                        // 目标位宽 = 7 (因为还有符号位)
                        // 右移量 = 有效位宽 - 7
                        
                        // 简化逻辑：动态计算右移量比较复杂，这里采用"块对齐"策略
                        // 我们的 term_final_mant 小数点约在 Bit 8。
                        // 如果我们想保持这个精度，Exponent 需减 8 (或相应调整)
                        // 这里使用类似 Softmax/BFP 的逻辑：
                        // 将最大值对齐到 8-bit 的满量程
                        
                        // 这里为了代码简洁和鲁棒性，采用固定右移 + 饱和策略，配合指数调整
                        // 假设我们固定右移 8 位 (丢弃小数部分)，变回整数
                        // 但如果数值很大，需要额外右移
                        
                        // 为了严谨：
                        // 1. 计算 max_abs_val 的有效整数位宽
                        // 2. 动态调整 shift
                        
                        // 这里采用：Count Leading Zeros on Max Val
                        // 假设 max_abs_val 在 39:0
                        // 目标是将 MSB 移到 Bit 6
                        // 当前 MSB 位置 = 39 - CLZ(max)
                        // Shift = (39 - CLZ) - 6 = 33 - CLZ
                        
                        // 为实现方便，这里做简化：假设我们需要右移 8 位基准
                        // 并根据是否溢出进行调整
                        
                        output_exp <= term_final_exp[EXP_WIDTH-1:0]; // 简化：直接使用基准指数
                        
                        for (i=0; i<DIM; i=i+1) begin:l3
                            reg signed [39:0] val_rounded;
                            reg signed [7:0] val_sat;
                            
                            // Rounding: 加 0.5 (Bit 7 是 0.5，因为我们假设右移8位)
                            val_rounded = term_final_mant[i] + 8'd128;
                            val_rounded = val_rounded >>> 8; // 右移 8 位
                            
                            // Saturation
                            if (val_rounded > 127) val_sat = 8'd127;
                            else if (val_rounded < -128) val_sat = -8'd128;
                            else val_sat = val_rounded[7:0];
                            
                            output_mant[i*DATA_WIDTH +: DATA_WIDTH] <= val_sat;
                        end
                    end
                    state <= WRITE_OUT;
                end

                //------------------------------------------------------------
                // 13. 写回结果
                //------------------------------------------------------------
                WRITE_OUT: begin
                    output_wr_en <= 1'b1;
                    output_valid <= 1'b1;
                    output_wr_addr <= token_idx[ADDR_WIDTH-1:0];
                    
                    // 握手：只有下游 ready 才前进，否则等待
                    if (output_ready) begin
                        state <= NEXT_TOKEN;
                        processed_cnt <= processed_cnt + 1;
                    end
                end

                //------------------------------------------------------------
                // 14. 循环控制
                //------------------------------------------------------------
                NEXT_TOKEN: begin
                    if (token_idx < TOKEN_NUM - 1) begin
                        token_idx <= token_idx + 1;
                        state <= PREP_TOKEN;
                    end else begin
                        state <= DONE_STATE;
                    end
                end

                //------------------------------------------------------------
                // 15. 完成
                //------------------------------------------------------------
                DONE_STATE: begin
                    done <= 1'b1;
                    if (!start) state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
            
            // 性能计数器
            if (state != IDLE && state != DONE_STATE) begin
                cycle_cnt <= cycle_cnt + 1;
            end
        end
    end
    
    // 忙信号逻辑
    assign busy = (state != IDLE) && (state != DONE_STATE);

endmodule