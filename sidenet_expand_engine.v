`timescale 1ns / 1ps

//================================================================================
// Sidenet Expand Engine v2.1
//
// 功能说明：
// - 将Sidenet Transformer输出的8维特征扩展回32维
// - 计算：output_i = W_expand × â_i
// - 输入：â_i ∈ R^(N×8)，输出：output_i ∈ R^(N×32)
// - 权重：W_expand ∈ R^(8×32)，每列独立共享指数
//
// 设计对称于Compression Engine：
// - Compression: 32维 → 8维（降维）
// - Expansion:   8维 → 32维（升维）
//
// 数据格式：16-bit BFP (blockBFP)
// - 输入adapted：每个token有1个共享指数 + 8个尾数
// - 输出expanded：每个token有1个共享指数 + 32个尾数
// - 权重：32个共享指数（每列1个）+ 8×32个尾数
//
//================================================================================

module sidenet_expand_engine #(
    parameter TOKEN_NUM      = 641,       // 总token数
    parameter TOKEN_BATCH    = 32,        // 每批处理的token数（保留，实际逐个）
    parameter INPUT_DIM      = 8,         // 输入维度（压缩后）
    parameter OUTPUT_DIM     = 32,        // 输出维度（恢复到原始）
    parameter DATA_WIDTH     = 16,        // 16-bit尾数
    parameter EXP_WIDTH      = 8,         // 8-bit指数
    parameter ADDR_WIDTH     = 10,        // 地址位宽
    parameter LAYER_WIDTH    = 2,         // Layer ID位宽
    
    // CE配置参数
    parameter CE_OUTPUT_WIDTH = 32,       // CE输出位宽（32位定点）
    parameter CE_INTERNAL_WIDTH = 39,     // CE内部计算位宽
    parameter CE_GUARD_BITS = 7,          // CE截断保护位数
    parameter CE_ENABLE_ROUNDING = 1      // CE舍入使能
)(
    input  wire clk,
    input  wire rst_n,
    
    //============================================================================
    // 控制接口
    //============================================================================
    input  wire start,                    // 启动信号
    input  wire [LAYER_WIDTH-1:0] layer_id, // 当前层ID（用于读取adapted buffer）
    output reg  done,                     // 完成信号
    output reg  busy,                     // 忙碌标志
    
    //============================================================================
    // Adapted Buffer读接口 - 读取â_i
    //============================================================================
    output reg  adapted_rd_en,
    output reg  [LAYER_WIDTH-1:0] adapted_rd_layer,
    output reg  [ADDR_WIDTH-1:0] adapted_rd_addr,
    input  wire [EXP_WIDTH-1:0] adapted_rd_exp,
    input  wire [INPUT_DIM*DATA_WIDTH-1:0] adapted_rd_mant,  // 8×16 = 128位
    input  wire adapted_rd_valid,
    
    //============================================================================
    // 权重接口 - 从权重控制器获取W_expand
    // ✅ 每列独立共享指数（32个指数）
    //============================================================================
    output reg  weight_req,               // 权重请求
    input  wire weight_ready,             // 权重就绪
    input  wire [OUTPUT_DIM*EXP_WIDTH-1:0] weight_exp,  // 32×8 = 256位
    input  wire [INPUT_DIM*OUTPUT_DIM*DATA_WIDTH-1:0] weight_mant,  // 8×32×16
    
    //============================================================================
    // 扩展结果输出 - 写入Layer Output Buffer
    //============================================================================
    output reg  result_wr_en,
    output reg  [ADDR_WIDTH-1:0] result_wr_addr,
    output reg  [EXP_WIDTH-1:0] result_wr_exp,
    output reg  [OUTPUT_DIM*DATA_WIDTH-1:0] result_wr_mant,  // 32×16 = 512位
    
    //============================================================================
    // 调试接口
    //============================================================================
    output wire [3:0] dbg_state,
    output reg  [31:0] dbg_token_count
);

//================================================================================
// 本地参数
//================================================================================

localparam BATCH_NUM = (TOKEN_NUM + TOKEN_BATCH - 1) / TOKEN_BATCH;  // 21批
localparam LAST_BATCH_SIZE = TOKEN_NUM - (BATCH_NUM - 1) * TOKEN_BATCH;  // 最后一批1个
localparam CE_BASE_EXP_WIDTH = EXP_WIDTH + 1;  // CE基础指数位宽（9位）

//================================================================================
// 状态机定义
//================================================================================

localparam IDLE          = 4'd0;
localparam LOAD_WEIGHT   = 4'd1;
localparam WAIT_WEIGHT   = 4'd2;
localparam LOAD_ADAPTED  = 4'd3;
localparam WAIT_ADAPTED  = 4'd4;
localparam COMPUTE       = 4'd5;
localparam WAIT_COMPUTE  = 4'd6;
localparam BFP_CONVERT   = 4'd7;
localparam WRITE_RESULT  = 4'd8;
localparam NEXT_TOKEN    = 4'd9;
localparam NEXT_BATCH    = 4'd10;
localparam DONE_STATE    = 4'd11;

reg [3:0] state, next_state;

//================================================================================
// 内部信号
//================================================================================

// 批次和token计数
reg [4:0] current_batch;       // 0-20 (21批)
reg [4:0] batch_token_id;      // 0-31 (批内token)
reg [9:0] global_token_id;     // 0-640 (全局token)
reg [4:0] tokens_in_batch;     // 当前批的token数

// Layer ID锁存
reg [LAYER_WIDTH-1:0] current_layer_id;

// Adapted特征输入缓存
reg [EXP_WIDTH-1:0] adapted_exp_buf;
reg [INPUT_DIM*DATA_WIDTH-1:0] adapted_mant_buf;

// 权重缓存
reg [OUTPUT_DIM*EXP_WIDTH-1:0] weight_exp_buf;  // 32×8 = 256位
reg [INPUT_DIM*OUTPUT_DIM*DATA_WIDTH-1:0] weight_mant_buf;
reg weight_loaded;

//================================================================================
// Compute Engine信号定义
//================================================================================

// CE控制信号
reg ce_input_valid;
wire ce_input_ready;
wire [OUTPUT_DIM-1:0] ce_result_valids;  // 32位

// CE定点输出
wire signed [OUTPUT_DIM*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_array;  // 32×32 = 1024位
wire [OUTPUT_DIM*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_array;   // 32×9 = 288位
wire [OUTPUT_DIM-1:0] ce_result_zero_array;                          // 32位

// CE完成信号
wire ce_compute_done;
assign ce_compute_done = |ce_result_valids;

//================================================================================
// ✅ CE输出缓存寄存器（v2.1新增）
// 
// 作用：
// 1. 在CE输出有效时立即采样，避免数据丢失
// 2. 为BFP转换器提供稳定的输入
// 3. 解耦CE和BFP转换器，时序安全
//================================================================================

reg [OUTPUT_DIM*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_buf;     // 32×32 = 1024位
reg [OUTPUT_DIM*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_buf; // 32×9 = 288位
reg [OUTPUT_DIM-1:0] ce_result_zero_buf;                        // 32位
reg ce_result_cached;                                           // 缓存有效标志

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ce_result_fixed_buf <= {(OUTPUT_DIM*CE_OUTPUT_WIDTH){1'b0}};
        ce_result_base_exp_buf <= {(OUTPUT_DIM*CE_BASE_EXP_WIDTH){1'b0}};
        ce_result_zero_buf <= {OUTPUT_DIM{1'b0}};
        ce_result_cached <= 1'b0;
    end else begin
        case (state)
            WAIT_COMPUTE: begin
                // CE输出有效时，立即采样缓存
                if (ce_compute_done && !ce_result_cached) begin
                    ce_result_fixed_buf <= ce_result_fixed_array;
                    ce_result_base_exp_buf <= ce_result_base_exp_array;
                    ce_result_zero_buf <= ce_result_zero_array;
                    ce_result_cached <= 1'b1;
                end
            end
            
            BFP_CONVERT: begin
                // 保持缓存，供BFP转换器使用
            end
            
            WRITE_RESULT: begin
                // 结果已写入，清除缓存标志
                ce_result_cached <= 1'b0;
            end
            
            default: begin
                // 其他状态不改变缓存
            end
        endcase
    end
end

//================================================================================
// BFP转换器信号定义
//================================================================================

// BFP转换器启动信号（由状态机控制）
reg bfp_start;

// 转换器输出：共享指数BFP格式
wire [OUTPUT_DIM-1:0] bfp_valids;
wire signed [OUTPUT_DIM*DATA_WIDTH-1:0] bfp_mants;      // 32×16 = 512位
wire [EXP_WIDTH-1:0] bfp_shared_exp;                    // 8位共享指数
wire bfp_overflow;

// 完成信号
wire bfp_convert_done;
assign bfp_convert_done = |bfp_valids;

//================================================================================
// Layer ID锁存
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_layer_id <= {LAYER_WIDTH{1'b0}};
    end else if (start && !busy) begin
        current_layer_id <= layer_id;
    end
end

//================================================================================
// 状态转移
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

//================================================================================
// 下一状态逻辑
//================================================================================

always @(*) begin
    next_state = state;
    
    case (state)
        IDLE: begin
            if (start) begin
                if (weight_loaded) begin
                    next_state = LOAD_ADAPTED;
                end else begin
                    next_state = LOAD_WEIGHT;
                end
            end
        end
        
        LOAD_WEIGHT: begin
            next_state = WAIT_WEIGHT;
        end
        
        WAIT_WEIGHT: begin
            if (weight_ready) begin
                next_state = LOAD_ADAPTED;
            end
        end
        
        LOAD_ADAPTED: begin
            next_state = WAIT_ADAPTED;
        end
        
        WAIT_ADAPTED: begin
            if (adapted_rd_valid) begin
                next_state = COMPUTE;
            end
        end
        
        COMPUTE: begin
            next_state = WAIT_COMPUTE;
        end
        
        WAIT_COMPUTE: begin
            // ✅ 等待CE完成并缓存数据
            if (ce_result_cached) begin
                next_state = BFP_CONVERT;
            end
        end
        
        BFP_CONVERT: begin
            // ✅ 等待BFP转换完成
            if (bfp_convert_done) begin
                next_state = WRITE_RESULT;
            end
        end
        
        WRITE_RESULT: begin
            next_state = NEXT_TOKEN;
        end
        
        NEXT_TOKEN: begin
            if (batch_token_id == tokens_in_batch - 1) begin
                next_state = NEXT_BATCH;
            end else begin
                next_state = LOAD_ADAPTED;
            end
        end
        
        NEXT_BATCH: begin
            if (current_batch == BATCH_NUM - 1) begin
                next_state = DONE_STATE;
            end else begin
                next_state = LOAD_ADAPTED;
            end
        end
        
        DONE_STATE: begin
            next_state = IDLE;
        end
        
        default: begin
            next_state = IDLE;
        end
    endcase
end

//================================================================================
// 输出逻辑和控制
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        //------------------------------------------------------------------------
        // 复位所有寄存器
        //------------------------------------------------------------------------
        done <= 1'b0;
        busy <= 1'b0;
        
        adapted_rd_en <= 1'b0;
        adapted_rd_layer <= {LAYER_WIDTH{1'b0}};
        adapted_rd_addr <= {ADDR_WIDTH{1'b0}};
        
        weight_req <= 1'b0;
        weight_loaded <= 1'b0;
        
        result_wr_en <= 1'b0;
        result_wr_addr <= {ADDR_WIDTH{1'b0}};
        result_wr_exp <= {EXP_WIDTH{1'b0}};
        result_wr_mant <= {(OUTPUT_DIM*DATA_WIDTH){1'b0}};
        
        ce_input_valid <= 1'b0;
        bfp_start <= 1'b0;
        
        current_batch <= 5'd0;
        batch_token_id <= 5'd0;
        global_token_id <= 10'd0;
        tokens_in_batch <= 5'd0;
        
        adapted_exp_buf <= {EXP_WIDTH{1'b0}};
        adapted_mant_buf <= {(INPUT_DIM*DATA_WIDTH){1'b0}};
        weight_exp_buf <= {(OUTPUT_DIM*EXP_WIDTH){1'b0}};
        weight_mant_buf <= {(INPUT_DIM*OUTPUT_DIM*DATA_WIDTH){1'b0}};
        
        dbg_token_count <= 32'd0;
        
    end else begin
        //------------------------------------------------------------------------
        // 默认值（每周期更新）
        //------------------------------------------------------------------------
        adapted_rd_en <= 1'b0;
        weight_req <= 1'b0;
        result_wr_en <= 1'b0;
        ce_input_valid <= 1'b0;
        bfp_start <= 1'b0;
        done <= 1'b0;
        
        //------------------------------------------------------------------------
        // 状态机主逻辑
        //------------------------------------------------------------------------
        case (state)
            
            //====================================================================
            // IDLE：空闲状态
            //====================================================================
            IDLE: begin
                busy <= 1'b0;
                current_batch <= 5'd0;
                batch_token_id <= 5'd0;
                global_token_id <= 10'd0;
                dbg_token_count <= 32'd0;
            end
            
            //====================================================================
            // LOAD_WEIGHT：请求权重
            //====================================================================
            LOAD_WEIGHT: begin
                busy <= 1'b1;
                weight_req <= 1'b1;
            end
            
            //====================================================================
            // WAIT_WEIGHT：等待权重就绪，缓存权重
            //====================================================================
            WAIT_WEIGHT: begin
                if (weight_ready) begin
                    weight_exp_buf <= weight_exp;      // 缓存32个共享指数
                    weight_mant_buf <= weight_mant;    // 缓存8×32×16尾数
                    weight_loaded <= 1'b1;
                end
            end
            
            //====================================================================
            // LOAD_ADAPTED：请求读取当前adapted特征
            //====================================================================
            LOAD_ADAPTED: begin
                busy <= 1'b1;
                adapted_rd_en <= 1'b1;
                adapted_rd_layer <= current_layer_id;
                adapted_rd_addr <= global_token_id;
                
                // 计算当前批的token数
                if (current_batch == BATCH_NUM - 1) begin
                    tokens_in_batch <= LAST_BATCH_SIZE;
                end else begin
                    tokens_in_batch <= TOKEN_BATCH;
                end
            end
            
            //====================================================================
            // WAIT_ADAPTED：等待adapted读取完成，缓存数据
            //====================================================================
            WAIT_ADAPTED: begin
                if (adapted_rd_valid) begin
                    adapted_exp_buf <= adapted_rd_exp;    // 缓存1个共享指数
                    adapted_mant_buf <= adapted_rd_mant;  // 缓存8×16尾数
                end
            end
            
            //====================================================================
            // COMPUTE：启动CE计算
            //====================================================================
            COMPUTE: begin
                ce_input_valid <= 1'b1;  // 启动compute engine（单周期脉冲）
            end
            
            //====================================================================
            // WAIT_COMPUTE：等待CE计算完成并缓存
            // ✅ CE输出被自动采样到缓存寄存器
            //====================================================================
            WAIT_COMPUTE: begin
                // CE输出自动被缓存寄存器采样
                // 等待ce_result_cached变高
            end
            
            //====================================================================
            // BFP_CONVERT：启动BFP转换，等待转换完成
            // ✅ BFP转换器使用稳定的缓存数据
            //====================================================================
            BFP_CONVERT: begin
                if (!bfp_convert_done) begin
                    bfp_start <= 1'b1;  // 启动BFP转换器
                end
            end
            
            //====================================================================
            // WRITE_RESULT：写入扩展结果
            //====================================================================
            WRITE_RESULT: begin
                result_wr_en <= 1'b1;
                result_wr_addr <= global_token_id;
                result_wr_exp <= bfp_shared_exp;      // BFP共享指数（8位）
                result_wr_mant <= bfp_mants;          // BFP尾数数组（32×16位）
                
                dbg_token_count <= dbg_token_count + 1;
                
                // 调试信息
                if (global_token_id == 0 || global_token_id == TOKEN_NUM - 1) begin
                    $display("[%0t] Expand: Token %0d - exp=0x%h, mant[0]=0x%h", 
                             $time, global_token_id, bfp_shared_exp, 
                             bfp_mants[DATA_WIDTH-1:0]);
                end
            end
            
            //====================================================================
            // NEXT_TOKEN：更新token计数
            //====================================================================
            NEXT_TOKEN: begin
                batch_token_id <= batch_token_id + 1;
                global_token_id <= global_token_id + 1;
            end
            
            //====================================================================
            // NEXT_BATCH：更新批次计数
            //====================================================================
            NEXT_BATCH: begin
                current_batch <= current_batch + 1;
                batch_token_id <= 5'd0;
            end
            
            //====================================================================
            // DONE_STATE：完成状态
            //====================================================================
            DONE_STATE: begin
                busy <= 1'b0;
                done <= 1'b1;
                weight_loaded <= 1'b0;  // 下次重新加载权重
                
                $display("[%0t] Expand: Done, processed %0d tokens (layer=%0d)", 
                         $time, dbg_token_count, current_layer_id);
            end
            
        endcase
    end
end

//================================================================================
// Compute Engine实例化
//
// 配置说明：
// - G_OUT=1, T_OUT=32：单行输出，32列（32个输出维度）
// - TOTAL_ELEM=8：输入向量8维
// - 输入：adapted[8] × weight[8×32] → result[32]
// - 输出：32个定点结果（32位）+ 32个基础指数（9位）
//================================================================================

compute_engine #(
    // 矩阵维度配置
    .G_OUT(1),                        // 1行输出
    .T_OUT(OUTPUT_DIM),               // 32列（32个输出维度）
    
    // PE配置（处理8维向量）
    .NUM_PE(1),                       // 1个PE即可（小向量）
    .PE_TYPE_0(0),                    // PE0类型A
    .PE_TYPE_1(0),                    // 未使用
    
    // 数据位宽
    .EXP_WIDTH(EXP_WIDTH),            // 指数8位
    .INPUT_MANT_WIDTH(DATA_WIDTH),    // 输入尾数16位
    
    // 向量维度
    .ELEM_PE0(INPUT_DIM),             // PE0处理8个元素
    .ELEM_PE1(0),                     // 未使用
    .TOTAL_ELEM(INPUT_DIM),           // 总元素8个
    
    // PU位宽优化配置
    .INTERNAL_WIDTH(CE_INTERNAL_WIDTH),     // 内部39位
    .OUTPUT_WIDTH(CE_OUTPUT_WIDTH),         // 输出32位
    .GUARD_BITS(CE_GUARD_BITS),             // 保护位7位
    .ENABLE_ROUNDING(CE_ENABLE_ROUNDING) // 启用舍入
    

) u_compute_engine (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    
    //--------------------------------------------------------------------------
    // 输入握手
    //--------------------------------------------------------------------------
    .input_valid(ce_input_valid),     // 输入有效（状态机控制）
    .input_ready(ce_input_ready),     // 输入就绪（CE反馈）
    
    //--------------------------------------------------------------------------
    // 输入数据
    //--------------------------------------------------------------------------
    .exp_X(adapted_exp_buf),          // Adapted的共享指数（1个，8位）
    .mant_X_block(adapted_mant_buf),  // Adapted的尾数（8×16位）
 
    .exp_W_array(weight_exp_buf),     // 32个共享指数（32×8 = 256位）
    .mant_W_blocks(weight_mant_buf),  // 权重尾数（8×32×16位）
    
    //--------------------------------------------------------------------------
    // 输出握手
    //--------------------------------------------------------------------------
    .result_valids(ce_result_valids), // 输出有效信号（32位）
    .result_ready(1'b1),              // 始终就绪（输出被缓存寄存器采样）
    
    //--------------------------------------------------------------------------
    //  输出数据
    //--------------------------------------------------------------------------
    .result_fixed_array(ce_result_fixed_array),      // 定点累加结果（32×32位）
    .result_base_exp_array(ce_result_base_exp_array),// 基础指数（32×9位）
    .result_zero_array(ce_result_zero_array)         // 零标志（32位）
);

//================================================================================
// BFP转换器实例化
//
// 功能说明：
// - ✅ 输入：缓存寄存器中的稳定数据
// - 处理：归一化 → 找最大指数 → 对齐尾数
// - 输出：共享指数BFP格式（1个8位指数 + 32个16位尾数）
//================================================================================

bfp_converter #(
    .TOTAL_RESULTS(OUTPUT_DIM),       // 结果数量：32个
    .FIXED_WIDTH(CE_OUTPUT_WIDTH),    // 定点输入位宽：32位
    .BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH),// 基础指数位宽：9位
    .OUTPUT_MANT_WIDTH(DATA_WIDTH),   // 输出尾数位宽：16位
    .OUTPUT_EXP_WIDTH(EXP_WIDTH)      // 输出指数位宽：8位
) u_bfp_converter (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    
    //--------------------------------------------------------------------------
    // ✅ 输入：来自稳定的缓存寄存器
    //--------------------------------------------------------------------------
    .input_valids({OUTPUT_DIM{ce_result_cached}}),    // 使用缓存有效标志
    .input_fixed_array(ce_result_fixed_buf),          // 缓存的定点数
    .input_base_exp_array(ce_result_base_exp_buf),    // 缓存的基础指数
    .input_zero_array(ce_result_zero_buf),            // 缓存的零标志
    
    //--------------------------------------------------------------------------
    // 输出：共享指数BFP格式
    //--------------------------------------------------------------------------
    .output_valids(bfp_valids),                       // 有效标志（32位）
    .output_mant_array(bfp_mants),                    // BFP尾数数组（32×16位）
    .output_shared_exp(bfp_shared_exp),               // BFP共享指数（8位）
    .output_overflow(bfp_overflow)                    // 溢出标志
);

//================================================================================
// 调试信号输出
//================================================================================

assign dbg_state = state;

//================================================================================
// 仿真信息输出
//================================================================================

`ifdef SIMULATION

initial begin
    $display("========================================");
    $display("Sidenet Expand Engine v2.1");
    $display("========================================");
    $display("Configuration:");
    $display("  Input Dim:      %0d (compressed)", INPUT_DIM);
    $display("  Output Dim:     %0d (expanded)", OUTPUT_DIM);
    $display("  Tokens:         %0d", TOKEN_NUM);
    $display("  Data Width:     %0d-bit BFP", DATA_WIDTH);
    $display("  CE Output:      %0d-bit fixed-point", CE_OUTPUT_WIDTH);
    $display("");
    $display("New Features (v2.1):");
    $display("  ✅ CE output buffering");
    $display("  ✅ Timing safe");
    $display("  ✅ No data loss");
    $display("");
    $display("Weight Format:");
    $display("  - %0d shared exponents (per column)", OUTPUT_DIM);
    $display("  - %0d×%0d mantissas", INPUT_DIM, OUTPUT_DIM);
    $display("========================================");
end

// 监控CE握手
always @(posedge clk) begin
    if (ce_input_valid && !ce_input_ready) begin
        $display("[WARNING] @%0t CE input not ready when valid asserted!", $time);
    end
end

// 监控CE输出采样
always @(posedge clk) begin
    if (state == WAIT_COMPUTE && ce_compute_done && !ce_result_cached) begin
        $display("[INFO] @%0t Sampling CE output for token %0d", $time, global_token_id);
    end
end

// 监控BFP转换
always @(posedge clk) begin
    if (state == BFP_CONVERT && bfp_start) begin
        $display("[INFO] @%0t Starting BFP conversion for token %0d", $time, global_token_id);
    end
end

// 监控BFP溢出
always @(posedge clk) begin
    if (bfp_convert_done && bfp_overflow) begin
        $display("[WARNING] @%0t BFP conversion overflow for token %0d!", 
                 $time, global_token_id);
    end
end

// 监控批次处理
always @(posedge clk) begin
    if (state == LOAD_ADAPTED && global_token_id % 32 == 0) begin
        $display("[%0t] Expand: Processing batch %0d (tokens %0d-%0d), layer=%0d", 
                 $time, current_batch, global_token_id, global_token_id + tokens_in_batch - 1,
                 current_layer_id);
    end
end

`endif

//================================================================================
// 参数合法性检查
//================================================================================

initial begin
    // 检查维度配置
    if (OUTPUT_DIM < INPUT_DIM) begin
        $error("ERROR: OUTPUT_DIM (%0d) < INPUT_DIM (%0d) - not expansion!", 
               OUTPUT_DIM, INPUT_DIM);
        $finish;
    end
    
    // 检查CE位宽配置
    if (CE_OUTPUT_WIDTH > CE_INTERNAL_WIDTH) begin
        $error("ERROR: CE_OUTPUT_WIDTH (%0d) > CE_INTERNAL_WIDTH (%0d)", 
               CE_OUTPUT_WIDTH, CE_INTERNAL_WIDTH);
        $finish;
    end
    
    // 检查保护位配置
    if (CE_GUARD_BITS != (CE_INTERNAL_WIDTH - CE_OUTPUT_WIDTH)) begin
        $warning("WARNING: CE_GUARD_BITS (%0d) != INTERNAL-OUTPUT (%0d)", 
                 CE_GUARD_BITS, CE_INTERNAL_WIDTH - CE_OUTPUT_WIDTH);
    end
    
    // 检查Layer ID范围
    if (layer_id >= (1 << LAYER_WIDTH)) begin
        $error("ERROR: layer_id (%0d) exceeds LAYER_WIDTH (%0d)", 
               layer_id, LAYER_WIDTH);
    end
end

endmodule