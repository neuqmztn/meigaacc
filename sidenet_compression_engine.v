`timescale 1ns / 1ps

module sidenet_compression_engine #(
    parameter TOKEN_NUM      = 640,       // 总token数
    parameter TOKEN_BATCH    = 32,        // 每批处理的token数（保留，实际逐个）
    parameter INPUT_DIM      = 32,        // 输入维度
    parameter OUTPUT_DIM     = 8,         // 输出维度（压缩后）
    parameter DATA_WIDTH     = 16,        // 16-bit尾数
    parameter EXP_WIDTH      = 8,         // 8-bit指数
    parameter ADDR_WIDTH     = 10,        // 地址位宽
    
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
    output reg  done,                     // 完成信号
    output reg  busy,                     // 忙碌标志
    
    //============================================================================
    // Token输入接口 - 从Layer Token Buffer读取
    //============================================================================
    output reg  token_rd_en,
    output reg  [ADDR_WIDTH-1:0] token_rd_addr,
    input  wire [EXP_WIDTH-1:0] token_rd_exp,
    input  wire [INPUT_DIM*DATA_WIDTH-1:0] token_rd_mant,  // 32×16 = 512位
    input  wire token_rd_valid,
    
    //============================================================================
    // 权重接口 - 从权重控制器获取W_compress
    // ✅ 修改：每列独立共享指数（8个指数）
    //============================================================================
    output reg  weight_req,               // 权重请求
    input  wire weight_ready,             // 权重就绪
    input  wire [OUTPUT_DIM*EXP_WIDTH-1:0] weight_exp,  // 8×8 = 64位
    input  wire [INPUT_DIM*OUTPUT_DIM*DATA_WIDTH-1:0] weight_mant,  // 32×8×16
    
    //============================================================================
    // 压缩结果输出 
    //============================================================================
    output reg  result_wr_en,
    output reg  [ADDR_WIDTH-1:0] result_wr_addr,
    output reg  [EXP_WIDTH-1:0] result_wr_exp,
    output reg  [OUTPUT_DIM*DATA_WIDTH-1:0] result_wr_mant,  // 8×16 = 128位
    
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
localparam LOAD_TOKEN    = 4'd3;
localparam WAIT_TOKEN    = 4'd4;
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

// Token输入缓存
reg [EXP_WIDTH-1:0] token_exp_buf;
reg [INPUT_DIM*DATA_WIDTH-1:0] token_mant_buf;

// ✅ 权重缓存（修改：存储8个共享指数）
reg [OUTPUT_DIM*EXP_WIDTH-1:0] weight_exp_buf;  // 8×8 = 64位
reg [INPUT_DIM*OUTPUT_DIM*DATA_WIDTH-1:0] weight_mant_buf;
reg weight_loaded;

//================================================================================
// Compute Engine信号定义
//================================================================================

// CE控制信号
reg ce_input_valid;
wire ce_input_ready;
wire [OUTPUT_DIM-1:0] ce_result_valids;  // 8位

// ✅ CE定点输出（正确格式）
wire signed [OUTPUT_DIM*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_array;  // 8×32 = 256位
wire [OUTPUT_DIM*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_array;   // 8×9 = 72位
wire [OUTPUT_DIM-1:0] ce_result_zero_array;                          // 8位

//================================================================================
// BFP转换器信号定义
//================================================================================

// 转换器输出：共享指数BFP格式
wire [OUTPUT_DIM-1:0] bfp_valids;
wire signed [OUTPUT_DIM*DATA_WIDTH-1:0] bfp_mants;      // 8×16 = 128位
wire [EXP_WIDTH-1:0] bfp_shared_exp;                    // 8位共享指数
wire bfp_overflow;

// 完成信号
wire ce_compute_done;
wire bfp_convert_done;

assign ce_compute_done = |ce_result_valids;
assign bfp_convert_done = |bfp_valids;

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
                    next_state = LOAD_TOKEN;
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
                next_state = LOAD_TOKEN;
            end
        end
        
        LOAD_TOKEN: begin
            next_state = WAIT_TOKEN;
        end
        
        WAIT_TOKEN: begin
            if (token_rd_valid) begin
                next_state = COMPUTE;
            end
        end
        
        COMPUTE: begin
            next_state = WAIT_COMPUTE;
        end
        
        WAIT_COMPUTE: begin
            if (ce_compute_done) begin
                next_state = BFP_CONVERT;
            end
        end
        
        BFP_CONVERT: begin
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
                next_state = LOAD_TOKEN;
            end
        end
        
        NEXT_BATCH: begin
            if (current_batch == BATCH_NUM - 1) begin
                next_state = DONE_STATE;
            end else begin
                next_state = LOAD_TOKEN;
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
        
        token_rd_en <= 1'b0;
        token_rd_addr <= {ADDR_WIDTH{1'b0}};
        
        weight_req <= 1'b0;
        weight_loaded <= 1'b0;
        
        result_wr_en <= 1'b0;
        result_wr_addr <= {ADDR_WIDTH{1'b0}};
        result_wr_exp <= {EXP_WIDTH{1'b0}};
        result_wr_mant <= {(OUTPUT_DIM*DATA_WIDTH){1'b0}};
        
        ce_input_valid <= 1'b0;
        
        current_batch <= 5'd0;
        batch_token_id <= 5'd0;
        global_token_id <= 10'd0;
        tokens_in_batch <= 5'd0;
        
        token_exp_buf <= {EXP_WIDTH{1'b0}};
        token_mant_buf <= {(INPUT_DIM*DATA_WIDTH){1'b0}};
        weight_exp_buf <= {(OUTPUT_DIM*EXP_WIDTH){1'b0}};
        weight_mant_buf <= {(INPUT_DIM*OUTPUT_DIM*DATA_WIDTH){1'b0}};
        
        dbg_token_count <= 32'd0;
        
    end else begin
        //------------------------------------------------------------------------
        // 默认值（每周期更新）
        //------------------------------------------------------------------------
        token_rd_en <= 1'b0;
        weight_req <= 1'b0;
        result_wr_en <= 1'b0;
        ce_input_valid <= 1'b0;
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
                    weight_exp_buf <= weight_exp;      // 缓存8个共享指数
                    weight_mant_buf <= weight_mant;    // 缓存32×8×16尾数
                    weight_loaded <= 1'b1;
                end
            end
            
            //====================================================================
            // LOAD_TOKEN：请求读取当前token
            //====================================================================
            LOAD_TOKEN: begin
                busy <= 1'b1;
                token_rd_en <= 1'b1;
                token_rd_addr <= global_token_id;
                
                // 计算当前批的token数
                if (current_batch == BATCH_NUM - 1) begin
                    tokens_in_batch <= LAST_BATCH_SIZE;
                end else begin
                    tokens_in_batch <= TOKEN_BATCH;
                end
            end
            
            //====================================================================
            // WAIT_TOKEN：等待token读取完成，缓存token
            //====================================================================
            WAIT_TOKEN: begin
                if (token_rd_valid) begin
                    token_exp_buf <= token_rd_exp;    // 缓存1个共享指数
                    token_mant_buf <= token_rd_mant;  // 缓存32×16尾数
                end
            end
            
            //====================================================================
            // COMPUTE：启动CE计算
            //====================================================================
            COMPUTE: begin
                ce_input_valid <= 1'b1;  // 启动compute engine（单周期脉冲）
            end
            
            //====================================================================
            // WAIT_COMPUTE：等待CE计算完成
            //====================================================================
            WAIT_COMPUTE: begin
                // 等待ce_result_valids变高
                // 状态机将自动跳转到BFP_CONVERT
            end
            
            //====================================================================
            // BFP_CONVERT：等待BFP转换完成
            //====================================================================
            BFP_CONVERT: begin
                // 等待bfp_valids变高
                // BFP转换器延迟1-2周期
            end
            
            //====================================================================
            // WRITE_RESULT：写入压缩结果
            //====================================================================
            WRITE_RESULT: begin
                result_wr_en <= 1'b1;
                result_wr_addr <= global_token_id;
                result_wr_exp <= bfp_shared_exp;      // BFP共享指数（8位）
                result_wr_mant <= bfp_mants;          // BFP尾数数组（8×16位）
                
                dbg_token_count <= dbg_token_count + 1;
                
                // 调试信息
                if (global_token_id == 0 || global_token_id == TOKEN_NUM - 1) begin
                    $display("[%0t] Compression: Token %0d - exp=0x%h, mant[0]=0x%h", 
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
                
                $display("[%0t] Compression: Done, processed %0d tokens", 
                         $time, dbg_token_count);
            end
            
        endcase
    end
end

//================================================================================
// Compute Engine实例化
//
// 配置说明：
// - G_OUT=1, T_OUT=8：单行输出，8列（8个输出维度）
// - TOTAL_ELEM=32：输入向量32维
// - 输入：token[32] × weight[32×8] → result[8]
// - 输出：8个定点结果（32位）+ 8个基础指数（9位）
//================================================================================

compute_engine #(
    // 矩阵维度配置
    .G_OUT(1),                        // 1行输出
    .T_OUT(OUTPUT_DIM),               // 8列（8个输出维度）
    
    // PE配置（处理32维向量）
    .NUM_PE(2),                       // 2个PE
    .PE_TYPE_0(0),                    // PE0类型A
    .PE_TYPE_1(2),                    // PE1类型C
    
    // 数据位宽
    .EXP_WIDTH(EXP_WIDTH),            // 指数8位
    .INPUT_MANT_WIDTH(DATA_WIDTH),    // 输入尾数16位
    
    // 向量维度
    .ELEM_PE0(16),                    // PE0处理16个元素
    .ELEM_PE1(16),                    // PE1处理16个元素
    .TOTAL_ELEM(INPUT_DIM),           // 总元素32个
    
    // PU位宽优化配置
    .INTERNAL_WIDTH(CE_INTERNAL_WIDTH),     // 内部39位
    .OUTPUT_WIDTH(CE_OUTPUT_WIDTH),         // 输出32位
    .GUARD_BITS(CE_GUARD_BITS),             // 保护位7位
    .ENABLE_ROUNDING(CE_ENABLE_ROUNDING)  // 启用舍入
    
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
    .exp_X(token_exp_buf),            // Token的共享指数（1个，8位）
    .mant_X_block(token_mant_buf),    // Token的尾数（32×16位）

    .exp_W_array(weight_exp_buf),     // 8个共享指数（8×8 = 64位）
    .mant_W_blocks(weight_mant_buf),  // 权重尾数（32×8×16位）
    
    //--------------------------------------------------------------------------
    // 输出握手
    //--------------------------------------------------------------------------
    .result_valids(ce_result_valids), // 输出有效信号（8位）
    .result_ready(1'b1),              // 始终就绪（直接连转换器）
    
    //--------------------------------------------------------------------------
    // 输出数据
    //--------------------------------------------------------------------------
    .result_fixed_array(ce_result_fixed_array),      // 定点累加结果（8×32位）
    .result_base_exp_array(ce_result_base_exp_array),// 基础指数（8×9位）
    .result_zero_array(ce_result_zero_array)         // 零标志（8位）
);

//================================================================================
// BFP转换器实例化
//
// 功能说明：
// - 输入：CE输出的定点数（8个32位） + 基础指数（8个9位）
// - 处理：归一化 → 找最大指数 → 对齐尾数
// - 输出：共享指数BFP格式（1个8位指数 + 8个16位尾数）
//================================================================================

bfp_converter #(
    .TOTAL_RESULTS(OUTPUT_DIM),       // 结果数量：8个
    .FIXED_WIDTH(CE_OUTPUT_WIDTH),    // 定点输入位宽：32位
    .BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH),// 基础指数位宽：9位
    .OUTPUT_MANT_WIDTH(DATA_WIDTH),   // 输出尾数位宽：16位
    .OUTPUT_EXP_WIDTH(EXP_WIDTH)      // 输出指数位宽：8位
) u_bfp_converter (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    
    //--------------------------------------------------------------------------
    // 输入：CE的定点输出
    //--------------------------------------------------------------------------
    .input_valids(ce_result_valids),                  // 有效标志（8位）
    .input_fixed_array(ce_result_fixed_array),        // 32位定点数（8个）
    .input_base_exp_array(ce_result_base_exp_array),  // 9位基础指数（8个）
    .input_zero_array(ce_result_zero_array),          // 零标志（8位）
    
    //--------------------------------------------------------------------------
    // 输出：共享指数BFP格式
    //--------------------------------------------------------------------------
    .output_valids(bfp_valids),                       // 有效标志（8位）
    .output_mant_array(bfp_mants),                    // BFP尾数数组（8×16位）
    .output_shared_exp(bfp_shared_exp),               // BFP共享指数（8位）
    .output_overflow(bfp_overflow)                    // 溢出标志
);

//================================================================================
// 调试信号输出
//================================================================================

assign dbg_state = state;


endmodule