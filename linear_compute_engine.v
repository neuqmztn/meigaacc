`timescale 1ns / 1ps

//================================================================================
// Linear Compute Engine v3.0 - 支持32个独立权重指数
//
// 版本更新（v2.0 → v3.0）：
// ✅ 权重指数从单个共享指数改为32个独立指数数组
// ✅ 完整保留权重的指数信息（不再丢失）
// ✅ 兼容weight_controller的输出格式
//
// 之前版本历史：
// v2.0: 添加CE输出缓存寄存器以确保时序安全
// v1.0: 初始版本，使用单个共享指数
//
// 功能：执行矩阵乘法 Y[32×32] = X[32×32] × W[32×32]
//
// 关键特性：
// 1. 逐Token循环调用CE（32次）
// 2. CE输出定点数 → 缓存寄存器 → BFP转换器 → BFP格式
// 3. 假设权重已按列优先格式存储
// 4. 支持32个独立权重指数（每个输出维度一个指数）
// 5. 缓存寄存器确保时序安全
//
// 数据流：
// Token[1×32] + Weight[32×32] → CE(定点输出) → Buffer → BFP Converter → BFP结果
//
// 作者：Claude
// 日期：2024-11-16
// 参考：qkv_compute_engine.v, compression engine
//================================================================================

module linear_compute_engine #(
    parameter TOKEN_CHUNK   = 32,
    parameter INPUT_DIM     = 32,
    parameter OUTPUT_DIM    = 32,
    
    // BFP 参数
    parameter BFP_EXP_W     = 8,
    parameter BFP_MANT_W    = 8,
    parameter ACC_MANT_W    = 8,     // 输出尾数位宽
    
    // CE 配置
    parameter G_OUT         = 4,
    parameter T_OUT         = 8,
    parameter NUM_GROUPS    = 1,
    parameter NUM_PE_PER_GROUP = 2,
    parameter PE_TYPE_0     = 0,
    parameter PE_TYPE_1     = 2,
    parameter ELEM_PE0      = 16,
    parameter ELEM_PE1      = 8,
    parameter TOTAL_ELEM    = 32,
    
    // CE 输出配置
    parameter CE_OUTPUT_WIDTH = 32,      // CE定点输出位宽
    parameter CE_BASE_EXP_WIDTH = 9      // CE基础指数位宽
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire start,
    output reg  done,
    output reg  busy,
    
    //================================================================================
    // 输入数据接口（来自 Memory Manager）
    //================================================================================
    // X: Token 块 [TOKEN_CHUNK × INPUT_DIM]
    input  wire [BFP_EXP_W-1:0] x_exp,
    input  wire [TOKEN_CHUNK*INPUT_DIM*BFP_MANT_W-1:0] x_mant,
    
    // W: 权重矩阵 [INPUT_DIM × OUTPUT_DIM] - 列优先格式
    input  wire [OUTPUT_DIM*BFP_EXP_W-1:0] w_exp_array,  // 32个独立指数
    input  wire [INPUT_DIM*OUTPUT_DIM*BFP_MANT_W-1:0] w_mant,
    
    //================================================================================
    // 输出数据接口
    //================================================================================
    // Y: 结果块 [TOKEN_CHUNK × OUTPUT_DIM]
    output reg  [BFP_EXP_W-1:0] y_exp,
    output reg  [TOKEN_CHUNK*OUTPUT_DIM*ACC_MANT_W-1:0] y_mant
);

//================================================================================
// 本地参数
//================================================================================

localparam TOTAL_WIDTH = TOTAL_ELEM * BFP_MANT_W;
localparam TOTAL_PUS = G_OUT * T_OUT;

// 状态机 - ✅ 新增SAMPLE_CE状态
localparam STATE_IDLE          = 4'd0;
localparam STATE_LOAD_DATA     = 4'd1;
localparam STATE_EXTRACT_TOKEN = 4'd2;
localparam STATE_SEND_CE       = 4'd3;
localparam STATE_WAIT_CE       = 4'd4;
localparam STATE_SAMPLE_CE     = 4'd5;  // ✅ 新增：采样CE输出到缓存
localparam STATE_WAIT_CONV     = 4'd6;
localparam STATE_SAVE_RESULT   = 4'd7;
localparam STATE_DONE          = 4'd8;

//================================================================================
// 内部信号
//================================================================================

// 状态机
reg [3:0] state;
reg [5:0] token_counter;  // 0 到 TOKEN_CHUNK-1

// 输入数据缓存
reg [BFP_EXP_W-1:0] x_exp_cached;
reg [TOKEN_CHUNK*INPUT_DIM*BFP_MANT_W-1:0] x_mant_cached;
reg [OUTPUT_DIM*BFP_EXP_W-1:0] w_exp_array_cached;  // 32个指数数组
reg [INPUT_DIM*OUTPUT_DIM*BFP_MANT_W-1:0] w_mant_cached;

// 当前 Token 数据
reg [BFP_EXP_W-1:0] current_token_exp;
reg [INPUT_DIM*BFP_MANT_W-1:0] current_token_mant;

// CE 接口信号
reg ce_input_valid;
reg [BFP_EXP_W-1:0] ce_exp_X;
reg [TOTAL_WIDTH-1:0] ce_mant_X;
// 权重指数直接使用 w_exp_array_cached，不再需要单独的ce_exp_W

wire [G_OUT*T_OUT-1:0] ce_result_valids;
wire signed [G_OUT*T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_array;
wire [G_OUT*T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_array;
wire [G_OUT*T_OUT-1:0] ce_result_zero_array;

//================================================================================
// ✅ 新增：CE输出缓存寄存器（确保时序安全）
//================================================================================

reg signed [G_OUT*T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_buf;
reg [G_OUT*T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_buf;
reg [G_OUT*T_OUT-1:0] ce_result_zero_buf;
reg ce_result_cached;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ce_result_fixed_buf    <= {(G_OUT*T_OUT*CE_OUTPUT_WIDTH){1'b0}};
        ce_result_base_exp_buf <= {(G_OUT*T_OUT*CE_BASE_EXP_WIDTH){1'b0}};
        ce_result_zero_buf     <= {(G_OUT*T_OUT){1'b0}};
        ce_result_cached       <= 1'b0;
    end else begin
        // 状态机控制采样
        case (state)
            STATE_WAIT_CE: begin
                if (|ce_result_valids && !ce_result_cached) begin
                    // CE输出有效，立即采样缓存
                    ce_result_fixed_buf    <= ce_result_fixed_array;
                    ce_result_base_exp_buf <= ce_result_base_exp_array;
                    ce_result_zero_buf     <= ce_result_zero_array;
                    ce_result_cached       <= 1'b1;
                end
            end
            
            STATE_SAMPLE_CE: begin
                // 保持缓存，供下一个状态使用
            end
            
            STATE_WAIT_CONV: begin
                // 保持缓存，供BFP转换器使用
            end
            
            STATE_SAVE_RESULT: begin
                // 结果保存后清除缓存标志
                ce_result_cached <= 1'b0;
            end
        endcase
    end
end

//================================================================================
// BFP Converter 接口信号
//================================================================================

wire [G_OUT*T_OUT-1:0] converter_output_valids;
wire signed [G_OUT*T_OUT*ACC_MANT_W-1:0] converter_output_mants;
wire [BFP_EXP_W-1:0] converter_output_shared_exp;
wire converter_overflow;

// 结果缓存
reg [TOKEN_CHUNK*OUTPUT_DIM*ACC_MANT_W-1:0] result_buffer;
reg [BFP_EXP_W-1:0] result_exp_buffer;

//================================================================================
// Compute Engine 实例化
//================================================================================

compute_engine #(
    .G_OUT(G_OUT),
    .T_OUT(T_OUT),
    .NUM_PE(NUM_PE_PER_GROUP),
    .PE_TYPE_0(PE_TYPE_0),
    .PE_TYPE_1(PE_TYPE_1),
    .EXP_WIDTH(BFP_EXP_W),
    .MANT_WIDTH(ACC_MANT_W),
    .INPUT_MANT_WIDTH(BFP_MANT_W),
    .ELEM_PE0(ELEM_PE0),
    .ELEM_PE1(ELEM_PE1),
    .TOTAL_ELEM(TOTAL_ELEM),
    .OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
    .FIFO_DEPTH(16)
) u_ce (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    
    .input_valid(ce_input_valid),
    .input_ready(),
    
    .exp_X(ce_exp_X),
    .mant_X_block(ce_mant_X),
    
    .exp_W_array(w_exp_array_cached),  // 直接使用32个指数数组
    .mant_W_blocks(w_mant_cached),     // 直接使用列优先权重
    
    .result_valids(ce_result_valids),
    .result_ready(1'b1),                     // 始终就绪（有缓存保护）
    
    // 定点输出
    .result_fixed_array(ce_result_fixed_array),
    .result_base_exp_array(ce_result_base_exp_array),
    .result_zero_array(ce_result_zero_array)
);

//================================================================================
// BFP Converter 实例化
// ✅ 输入连接到缓存寄存器（而非直连CE）
//================================================================================

bfp_converter #(
    .TOTAL_RESULTS(G_OUT*T_OUT),
    .FIXED_WIDTH(CE_OUTPUT_WIDTH),
    .BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH),
    .OUTPUT_MANT_WIDTH(ACC_MANT_W),
    .OUTPUT_EXP_WIDTH(BFP_EXP_W)
) u_bfp_converter (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    
    // ✅ 使用缓存的数据（而非直连CE）
    .input_valids({(G_OUT*T_OUT){ce_result_cached}}),
    .input_fixed_array(ce_result_fixed_buf),
    .input_base_exp_array(ce_result_base_exp_buf),
    .input_zero_array(ce_result_zero_buf),
    
    // BFP格式输出
    .output_valids(converter_output_valids),
    .output_mant_array(converter_output_mants),
    .output_shared_exp(converter_output_shared_exp),
    .output_overflow(converter_overflow)
);

//================================================================================
// 主状态机
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state           <= STATE_IDLE;
        done            <= 1'b0;
        busy            <= 1'b0;
        token_counter   <= 6'd0;
        ce_input_valid  <= 1'b0;
        
        x_exp_cached    <= {BFP_EXP_W{1'b0}};
        x_mant_cached   <= {TOKEN_CHUNK*INPUT_DIM*BFP_MANT_W{1'b0}};
        w_exp_array_cached <= {OUTPUT_DIM*BFP_EXP_W{1'b0}};  // 32个指数
        w_mant_cached   <= {INPUT_DIM*OUTPUT_DIM*BFP_MANT_W{1'b0}};
        
        current_token_exp  <= {BFP_EXP_W{1'b0}};
        current_token_mant <= {INPUT_DIM*BFP_MANT_W{1'b0}};
        
        result_buffer     <= {TOKEN_CHUNK*OUTPUT_DIM*ACC_MANT_W{1'b0}};
        result_exp_buffer <= {BFP_EXP_W{1'b0}};
        
    end else begin
        case (state)
            
            //================================================================
            // IDLE: 等待启动
            //================================================================
            STATE_IDLE: begin
                done <= 1'b0;
                busy <= 1'b0;
                
                if (start) begin
                    busy          <= 1'b1;
                    token_counter <= 6'd0;
                    state         <= STATE_LOAD_DATA;
                end
            end
            
            //================================================================
            // LOAD_DATA: 缓存输入数据
            //================================================================
            STATE_LOAD_DATA: begin
                // 缓存整块输入
                x_exp_cached  <= x_exp;
                x_mant_cached <= x_mant;
                w_exp_array_cached <= w_exp_array;  // 加载32个指数数组
                w_mant_cached <= w_mant;  // 假设已经是列优先格式
                
                state <= STATE_EXTRACT_TOKEN;
            end
            
            //================================================================
            // EXTRACT_TOKEN: 提取当前 Token
            //================================================================
            STATE_EXTRACT_TOKEN: begin
                // 提取当前 token
                current_token_exp  <= x_exp_cached;
                current_token_mant <= x_mant_cached[token_counter*INPUT_DIM*BFP_MANT_W +: INPUT_DIM*BFP_MANT_W];
                
                state <= STATE_SEND_CE;
            end
            
            //================================================================
            // SEND_CE: 发送当前 Token 到 CE
            //================================================================
            STATE_SEND_CE: begin
                // 准备 CE 输入
                ce_input_valid <= 1'b1;
                ce_exp_X       <= current_token_exp;
                ce_mant_X      <= current_token_mant;
                // 权重指数通过w_exp_array_cached直接传递给CE
                
                state <= STATE_WAIT_CE;
            end
            
            //================================================================
            // WAIT_CE: 等待 CE 计算完成
            // ✅ 在此状态CE输出被采样到缓存寄存器
            //================================================================
            STATE_WAIT_CE: begin
                ce_input_valid <= 1'b0;
                
                // 等待缓存标志置位（由缓存寄存器逻辑控制）
                if (ce_result_cached) begin
                    state <= STATE_SAMPLE_CE;
                end
            end
            
            //================================================================
            // SAMPLE_CE: CE输出已采样到缓存
            // ✅ 新增状态：确保缓存数据稳定
            //================================================================
            STATE_SAMPLE_CE: begin
                // 缓存数据已稳定，可以进入转换状态
                state <= STATE_WAIT_CONV;
            end
            
            //================================================================
            // WAIT_CONV: 等待 BFP 转换完成
            // ✅ 转换器使用缓存的数据（稳定可靠）
            //================================================================
            STATE_WAIT_CONV: begin
                // 等待转换器输出有效
                if (|converter_output_valids) begin
                    state <= STATE_SAVE_RESULT;
                end
            end
            
            //================================================================
            // SAVE_RESULT: 保存当前 Token 的结果
            //================================================================
            STATE_SAVE_RESULT: begin
                // 保存结果到对应位置
                result_buffer[token_counter*OUTPUT_DIM*ACC_MANT_W +: OUTPUT_DIM*ACC_MANT_W] 
                    <= converter_output_mants[0 +: OUTPUT_DIM*ACC_MANT_W];
                
                // 更新结果指数（使用第一个 token 的指数）
                if (token_counter == 6'd0) begin
                    result_exp_buffer <= converter_output_shared_exp;
                end
                
                // 检查是否完成所有 token
                if (token_counter < TOKEN_CHUNK - 1) begin
                    token_counter <= token_counter + 1'b1;
                    state <= STATE_EXTRACT_TOKEN;
                end else begin
                    state <= STATE_DONE;
                end
            end
            
            //================================================================
            // DONE: 完成
            //================================================================
            STATE_DONE: begin
                done <= 1'b1;
                busy <= 1'b0;
                
                // 输出结果
                y_exp  <= result_exp_buffer;
                y_mant <= result_buffer;
                
                if (!start) begin
                    state <= STATE_IDLE;
                    done  <= 1'b0;
                end
            end
            
            default: state <= STATE_IDLE;
        endcase
    end
end

//================================================================================
// 调试信号（可选）
//================================================================================

`ifdef SIMULATION
initial begin
    $display("========================================");
    $display("Linear Compute Engine v2.0");
    $display("========================================");
    $display("Updates:");
    $display("  ✅ CE output buffer for timing safety");
    $display("  ✅ Added SAMPLE_CE state");
    $display("  ✅ BFP converter uses buffered data");
    $display("");
    $display("Configuration:");
    $display("  Token Chunk: %0d", TOKEN_CHUNK);
    $display("  Input Dim:   %0d", INPUT_DIM);
    $display("  Output Dim:  %0d", OUTPUT_DIM);
    $display("  CE Output:   %0d-bit fixed", CE_OUTPUT_WIDTH);
    $display("  BFP Output:  %0d-bit mant", ACC_MANT_W);
    $display("========================================");
end

always @(posedge clk) begin
    if (state == STATE_SEND_CE) begin
        $display("[%0t] Linear CE: Sending Token %0d to CE", $time, token_counter);
    end
    
    if (state == STATE_SAMPLE_CE) begin
        $display("[%0t] Linear CE: CE output sampled to buffer for Token %0d", 
                 $time, token_counter);
    end
    
    if (state == STATE_SAVE_RESULT) begin
        $display("[%0t] Linear CE: Saved result for Token %0d, exp=%0d", 
                 $time, token_counter, converter_output_shared_exp);
    end
    
    if (state == STATE_DONE) begin
        $display("[%0t] Linear CE: All %0d tokens processed", $time, TOKEN_CHUNK);
    end
    
    if (converter_overflow) begin
        $display("[%0t] Linear CE: WARNING - BFP converter overflow detected", $time);
    end
end
`endif

endmodule