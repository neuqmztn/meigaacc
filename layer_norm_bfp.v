`timescale 1ns / 1ps
//================================================================================
// Layer Normalization Module with BFP Format Support
//
// 功能说明：
// 1. 实现Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
// 2. 支持BFP格式的统计量计算和归一化
// 3. 两遍处理：第一遍计算均值和方差，第二遍归一化
// 4. 包含平方根近似、除法近似、参数应用
//================================================================================

module layer_norm_bfp #(
    parameter TOKEN_NUM  = 641,          // Token总数
    parameter DIM        = 32,           // 特征维度
    parameter DATA_WIDTH = 8,            // BFP尾数位宽
    parameter EXP_WIDTH  = 8,            // BFP指数位宽
    parameter ADDR_WIDTH = 10,           // 地址位宽
    parameter ACCUM_WIDTH = 24,          // 累加器位宽
    parameter SQRT_LUT_DEPTH = 256       // 平方根查找表深度
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire start,
    input  wire mode,                     // 0=LN1, 1=LN2
    output reg  done,
    output wire busy,
    output reg  error,
    
    //==========================================================================
    // 输入接口: 从Residual Add读取
    //==========================================================================
    output reg  input_rd_en,
    output reg  [ADDR_WIDTH-1:0] input_rd_addr,
    input  wire [EXP_WIDTH-1:0] input_exp,
    input  wire [DIM*DATA_WIDTH-1:0] input_mant,
    input  wire input_valid,
    
    //==========================================================================
    // 输出接口: 写回到Input Buffer或输出
    //==========================================================================
    output reg  output_wr_en,
    output reg  [ADDR_WIDTH-1:0] output_wr_addr,
    output reg  [EXP_WIDTH-1:0] output_exp,
    output reg  [DIM*DATA_WIDTH-1:0] output_mant,
    output reg  output_valid,
    
    //==========================================================================
    // 参数接口: gamma和beta (从权重存储器读取)
    //==========================================================================
    output reg  param_rd_en,
    output reg  param_rd_gamma,          // 0=beta, 1=gamma
    input  wire [EXP_WIDTH-1:0] param_exp,
    input  wire [DIM*DATA_WIDTH-1:0] param_mant,
    input  wire param_valid,
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output reg  [3:0] state,
    output reg  [9:0] processed_tokens,
    output reg  [31:0] cycle_count,
    output reg  overflow_flag,
    output reg  underflow_flag
);

//================================================================================
// 状态定义
//================================================================================
localparam IDLE              = 4'd0;
localparam LOAD_PARAMS       = 4'd1;  // 加载gamma和beta参数
localparam PASS1_INIT        = 4'd2;  // 第一遍初始化
localparam PASS1_READ        = 4'd3;  // 第一遍读取
localparam PASS1_ACCUM       = 4'd4;  // 第一遍累加
localparam PASS1_COMPUTE     = 4'd5;  // 计算均值和方差
localparam PASS2_INIT        = 4'd6;  // 第二遍初始化
localparam PASS2_READ        = 4'd7;  // 第二遍读取
localparam PASS2_NORM        = 4'd8;  // 归一化计算
localparam PASS2_APPLY       = 4'd9;  // 应用gamma和beta
localparam PASS2_WRITE       = 4'd10; // 写回结果
localparam DONE_STATE        = 4'd11;
localparam ERROR_STATE       = 4'd12;

//================================================================================
// 内部存储
//================================================================================
// Token缓存（存储第一遍读取的数据）
reg [EXP_WIDTH-1:0] token_exp_buf [0:TOKEN_NUM-1];
reg [DIM*DATA_WIDTH-1:0] token_mant_buf [0:TOKEN_NUM-1];
reg [TOKEN_NUM-1:0] token_valid_buf;

// 参数存储
reg [EXP_WIDTH-1:0] gamma_exp;
reg [DIM*DATA_WIDTH-1:0] gamma_mant;
reg [EXP_WIDTH-1:0] beta_exp;
reg [DIM*DATA_WIDTH-1:0] beta_mant;
reg params_loaded;

// 统计量存储（每个token的均值和方差）
reg signed [ACCUM_WIDTH-1:0] mean_accum [0:TOKEN_NUM-1];
reg signed [ACCUM_WIDTH-1:0] var_accum [0:TOKEN_NUM-1];
reg [EXP_WIDTH-1:0] mean_exp [0:TOKEN_NUM-1];
reg [EXP_WIDTH-1:0] var_exp [0:TOKEN_NUM-1];

// 归一化中间结果
reg [EXP_WIDTH-1:0] std_inv_exp [0:TOKEN_NUM-1];
reg [DATA_WIDTH-1:0] std_inv_mant [0:TOKEN_NUM-1];

// 计数器和索引
reg [9:0] token_idx;
reg [4:0] dim_idx;
reg [3:0] wait_cnt;
reg [2:0] compute_cycle;

// 累加器
reg signed [ACCUM_WIDTH+4:0] sum_accum;
reg signed [ACCUM_WIDTH+8:0] sum_sq_accum;

//================================================================================
// 临时变量声明（修复：从过程块内移到模块级别）
//================================================================================
// PASS1_ACCUM 状态使用的临时变量
reg signed [DATA_WIDTH-1:0] accum_val;
reg signed [2*DATA_WIDTH-1:0] accum_val_sq;

// PASS1_COMPUTE 状态使用的临时变量
reg signed [ACCUM_WIDTH-1:0] compute_mean_val;
reg signed [2*ACCUM_WIDTH-1:0] compute_mean_sq;
reg signed [ACCUM_WIDTH-1:0] compute_avg_sq;

// PASS2_INIT 状态使用的临时变量
reg [ACCUM_WIDTH-1:0] init_var_plus_eps;
reg [7:0] init_lut_idx;

// PASS2_NORM 状态使用的临时变量
reg signed [DATA_WIDTH-1:0] norm_val [0:DIM-1];
reg signed [DATA_WIDTH-1:0] norm_centered_val;
reg signed [DATA_WIDTH*2-1:0] norm_scaled_val;

// PASS2_APPLY 状态使用的临时变量
reg signed [DATA_WIDTH-1:0] apply_final_val [0:DIM-1];
reg signed [DATA_WIDTH*2-1:0] apply_gamma_scaled;

//================================================================================
// 平方根倒数查找表（简化实现）
//================================================================================
reg [DATA_WIDTH-1:0] sqrt_inv_lut [0:SQRT_LUT_DEPTH-1];

// 初始化查找表（使用近似值）
initial begin: init_sqrt_lut
    integer k;
    for (k = 0; k < SQRT_LUT_DEPTH; k = k + 1) begin
        if (k == 0) begin
            sqrt_inv_lut[k] = 8'hFF; // 最大值（防止除零）
        end else begin
            // 简化的平方根倒数近似
            // 实际应该使用更精确的计算
            sqrt_inv_lut[k] = 255 / $rtoi($sqrt(k));
        end
    end
end

//================================================================================
// 主状态机
//================================================================================
integer i, j;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        done <= 1'b0;
        error <= 1'b0;
        token_idx <= 10'd0;
        dim_idx <= 5'd0;
        processed_tokens <= 10'd0;
        cycle_count <= 32'd0;
        wait_cnt <= 4'd0;
        compute_cycle <= 3'd0;
        
        input_rd_en <= 1'b0;
        input_rd_addr <= {ADDR_WIDTH{1'b0}};
        output_wr_en <= 1'b0;
        output_valid <= 1'b0;
        output_wr_addr <= {ADDR_WIDTH{1'b0}};
        output_exp <= {EXP_WIDTH{1'b0}};
        output_mant <= {DIM*DATA_WIDTH{1'b0}};
        
        param_rd_en <= 1'b0;
        param_rd_gamma <= 1'b0;
        params_loaded <= 1'b0;
        
        gamma_exp <= {EXP_WIDTH{1'b0}};
        gamma_mant <= {DIM*DATA_WIDTH{1'b0}};
        beta_exp <= {EXP_WIDTH{1'b0}};
        beta_mant <= {DIM*DATA_WIDTH{1'b0}};
        
        token_valid_buf <= {TOKEN_NUM{1'b0}};
        
        sum_accum <= {(ACCUM_WIDTH+5){1'b0}};
        sum_sq_accum <= {(ACCUM_WIDTH+9){1'b0}};
        
        overflow_flag <= 1'b0;
        underflow_flag <= 1'b0;
        
        // 初始化临时变量
        accum_val <= {DATA_WIDTH{1'b0}};
        accum_val_sq <= {(2*DATA_WIDTH){1'b0}};
        compute_mean_val <= {ACCUM_WIDTH{1'b0}};
        compute_mean_sq <= {(2*ACCUM_WIDTH){1'b0}};
        compute_avg_sq <= {ACCUM_WIDTH{1'b0}};
        init_var_plus_eps <= {ACCUM_WIDTH{1'b0}};
        init_lut_idx <= 8'd0;
        norm_centered_val <= {DATA_WIDTH{1'b0}};
        norm_scaled_val <= {(DATA_WIDTH*2){1'b0}};
        apply_gamma_scaled <= {(DATA_WIDTH*2){1'b0}};
        
        for (j = 0; j < DIM; j = j + 1) begin
            norm_val[j] <= {DATA_WIDTH{1'b0}};
            apply_final_val[j] <= {DATA_WIDTH{1'b0}};
        end
        
        // 清空缓存
        for (i = 0; i < TOKEN_NUM; i = i + 1) begin
            token_exp_buf[i] <= {EXP_WIDTH{1'b0}};
            token_mant_buf[i] <= {DIM*DATA_WIDTH{1'b0}};
            mean_accum[i] <= {ACCUM_WIDTH{1'b0}};
            var_accum[i] <= {ACCUM_WIDTH{1'b0}};
            mean_exp[i] <= {EXP_WIDTH{1'b0}};
            var_exp[i] <= {EXP_WIDTH{1'b0}};
            std_inv_exp[i] <= {EXP_WIDTH{1'b0}};
            std_inv_mant[i] <= {DATA_WIDTH{1'b0}};
        end
        
    end else begin
        // 默认信号
        input_rd_en <= 1'b0;
        output_wr_en <= 1'b0;
        output_valid <= 1'b0;
        param_rd_en <= 1'b0;
        
        case (state)
            //------------------------------------------------------------------
            IDLE: begin
                done <= 1'b0;
                error <= 1'b0;
                token_idx <= 10'd0;
                processed_tokens <= 10'd0;
                cycle_count <= 32'd0;
                overflow_flag <= 1'b0;
                underflow_flag <= 1'b0;
                
                if (start) begin
                    if (params_loaded) begin
                        state <= PASS1_INIT;
                    end else begin
                        state <= LOAD_PARAMS;
                    end
                end
            end
            
            //------------------------------------------------------------------
            LOAD_PARAMS: begin
                if (wait_cnt == 4'd0) begin
                    // 请求gamma参数
                    param_rd_en <= 1'b1;
                    param_rd_gamma <= 1'b1;
                    wait_cnt <= 4'd1;
                end else if (wait_cnt == 4'd1) begin
                    if (param_valid) begin
                        gamma_exp <= param_exp;
                        gamma_mant <= param_mant;
                        wait_cnt <= 4'd2;
                    end
                end else if (wait_cnt == 4'd2) begin
                    // 请求beta参数
                    param_rd_en <= 1'b1;
                    param_rd_gamma <= 1'b0;
                    wait_cnt <= 4'd3;
                end else if (wait_cnt == 4'd3) begin
                    if (param_valid) begin
                        beta_exp <= param_exp;
                        beta_mant <= param_mant;
                        params_loaded <= 1'b1;
                        wait_cnt <= 4'd0;
                        state <= PASS1_INIT;
                    end
                end
            end
            
            //------------------------------------------------------------------
            // 第一遍：计算均值和方差
            //------------------------------------------------------------------
            PASS1_INIT: begin
                token_idx <= 10'd0;
                token_valid_buf <= {TOKEN_NUM{1'b0}};
                state <= PASS1_READ;
            end
            
            //------------------------------------------------------------------
            PASS1_READ: begin
                if (wait_cnt == 4'd0) begin
                    // 发起读请求
                    input_rd_en <= 1'b1;
                    input_rd_addr <= token_idx[ADDR_WIDTH-1:0];
                    wait_cnt <= 4'd1;
                end else begin
                    wait_cnt <= wait_cnt + 1;
                    if (input_valid) begin
                        // 存储到缓存
                        token_exp_buf[token_idx] <= input_exp;
                        token_mant_buf[token_idx] <= input_mant;
                        token_valid_buf[token_idx] <= 1'b1;
                        
                        // 初始化累加器
                        sum_accum <= {(ACCUM_WIDTH+5){1'b0}};
                        sum_sq_accum <= {(ACCUM_WIDTH+9){1'b0}};
                        dim_idx <= 5'd0;
                        wait_cnt <= 4'd0;
                        state <= PASS1_ACCUM;
                    end else if (wait_cnt >= 4'd15) begin
                        // 读取超时
                        error <= 1'b1;
                        state <= ERROR_STATE;
                    end
                end
            end
            
            //------------------------------------------------------------------
            PASS1_ACCUM: begin
                // 累加所有维度
                if (dim_idx < DIM) begin
                    // 提取当前维度的值
                    accum_val = token_mant_buf[token_idx][dim_idx*DATA_WIDTH +: DATA_WIDTH];
                    
                    // 累加值和平方值
                    sum_accum <= sum_accum + accum_val;
                    accum_val_sq = accum_val * accum_val;
                    sum_sq_accum <= sum_sq_accum + accum_val_sq;
                    
                    dim_idx <= dim_idx + 1;
                end else begin
                    state <= PASS1_COMPUTE;
                    compute_cycle <= 3'd0;
                end
            end
            
            //------------------------------------------------------------------
            PASS1_COMPUTE: begin
                if (compute_cycle == 3'd0) begin
                    // 计算均值：mean = sum / DIM
                    // 使用移位近似除法（DIM=32 = 2^5）
                    mean_accum[token_idx] <= sum_accum >>> 5;
                    mean_exp[token_idx] <= token_exp_buf[token_idx];
                    compute_cycle <= 3'd1;
                    
                end else if (compute_cycle == 3'd1) begin
                    // 计算方差：var = sum_sq/DIM - mean^2
                    compute_mean_val = mean_accum[token_idx];
                    compute_mean_sq = compute_mean_val * compute_mean_val;
                    compute_avg_sq = sum_sq_accum >>> 5; // 除以DIM
                    
                    // 方差 = E[X^2] - E[X]^2
                    var_accum[token_idx] <= compute_avg_sq - compute_mean_sq[ACCUM_WIDTH-1:0];
                    var_exp[token_idx] <= token_exp_buf[token_idx];
                    
                    // 处理下一个token
                    if (token_idx < TOKEN_NUM - 1) begin
                        token_idx <= token_idx + 1;
                        state <= PASS1_READ;
                    end else begin
                        state <= PASS2_INIT;
                    end
                end
            end
            
            //------------------------------------------------------------------
            // 第二遍：应用归一化
            //------------------------------------------------------------------
            PASS2_INIT: begin
                token_idx <= 10'd0;
                processed_tokens <= 10'd0;
                
                // 预计算所有token的标准差倒数
                for (i = 0; i < TOKEN_NUM && i < 64; i = i + 1) begin
                    // 添加epsilon（小常数防止除零）
                    init_var_plus_eps = var_accum[i] + 1;
                    
                    // 简化：使用高8位作为查找表索引
                    init_lut_idx = init_var_plus_eps[ACCUM_WIDTH-1:ACCUM_WIDTH-8];
                    
                    // 查找平方根倒数
                    std_inv_mant[i] <= sqrt_inv_lut[init_lut_idx];
                    std_inv_exp[i] <= var_exp[i] >> 1; // 平方根指数减半
                end
                
                state <= PASS2_READ;
            end
            
            //------------------------------------------------------------------
            PASS2_READ: begin
                if (token_valid_buf[token_idx]) begin
                    dim_idx <= 5'd0;
                    state <= PASS2_NORM;
                end else begin
                    // 跳过无效token
                    if (token_idx < TOKEN_NUM - 1) begin
                        token_idx <= token_idx + 1;
                    end else begin
                        state <= DONE_STATE;
                    end
                end
            end
            
            //------------------------------------------------------------------
            PASS2_NORM: begin
                // 对每个维度进行归一化
                // 批量处理所有维度
                for (i = 0; i < DIM; i = i + 1) begin
                    // 中心化：x - mean
                    norm_centered_val = token_mant_buf[token_idx][i*DATA_WIDTH +: DATA_WIDTH] - 
                                   mean_accum[token_idx][DATA_WIDTH-1:0];
                    
                    // 缩放：(x - mean) / std
                    norm_scaled_val = norm_centered_val * std_inv_mant[token_idx];
                    norm_val[i] = norm_scaled_val[DATA_WIDTH*2-1:DATA_WIDTH]; // 取高位
                end
                
                // 临时存储归一化结果
                for (i = 0; i < DIM; i = i + 1) begin
                    token_mant_buf[token_idx][i*DATA_WIDTH +: DATA_WIDTH] <= norm_val[i];
                end
                
                state <= PASS2_APPLY;
            end
            
            //------------------------------------------------------------------
            PASS2_APPLY: begin
                // 应用gamma和beta参数
                for (i = 0; i < DIM; i = i + 1) begin
                    // y = norm * gamma + beta
                    apply_gamma_scaled = token_mant_buf[token_idx][i*DATA_WIDTH +: DATA_WIDTH] * 
                                   gamma_mant[i*DATA_WIDTH +: DATA_WIDTH];
                    apply_final_val[i] = apply_gamma_scaled[DATA_WIDTH*2-1:DATA_WIDTH] + 
                                   beta_mant[i*DATA_WIDTH +: DATA_WIDTH];
                end
                
                // 组装输出
                for (i = 0; i < DIM; i = i + 1) begin
                    output_mant[i*DATA_WIDTH +: DATA_WIDTH] <= apply_final_val[i];
                end
                
                // 计算输出指数
                output_exp <= token_exp_buf[token_idx];
                
                state <= PASS2_WRITE;
            end
            
            //------------------------------------------------------------------
            PASS2_WRITE: begin
                // 写出结果
                output_wr_en <= 1'b1;
                output_valid <= 1'b1;
                output_wr_addr <= token_idx[ADDR_WIDTH-1:0];
                
                processed_tokens <= processed_tokens + 1;
                
                // 处理下一个token
                if (token_idx < TOKEN_NUM - 1) begin
                    token_idx <= token_idx + 1;
                    state <= PASS2_READ;
                end else begin
                    state <= DONE_STATE;
                end
            end
            
            //------------------------------------------------------------------
            DONE_STATE: begin
                done <= 1'b1;
                if (!start) begin
                    state <= IDLE;
                end
            end
            
            //------------------------------------------------------------------
            ERROR_STATE: begin
                error <= 1'b1;
                if (!start) begin
                    state <= IDLE;
                end
            end
            
            default: state <= IDLE;
        endcase
        
        // 周期计数
        if (state != IDLE && state != DONE_STATE && state != ERROR_STATE) begin
            cycle_count <= cycle_count + 1;
        end
    end
end

assign busy = (state != IDLE) && (state != DONE_STATE) && (state != ERROR_STATE);

endmodule
