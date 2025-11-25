`timescale 1ns / 1ps

//================================================================================
// Output Projection - BFP版本 v2.0
//
// 改进点（v2.0）：
//   - 适配4个独立的head engine内部accumulator
//   - 直接从4个head的final_rd接口读取
//   - 支持并行读取提高效率
//
// 功能：
// - 从4个head engine的内部accumulator读取输出（BFP格式）
// - 对齐指数并concat成32维向量
// - 使用Compute Engine计算 concat × W_O
// - 使用BFP转换器转换回BFP格式
// - 输出到Result Buffer
//
// 数据流：
//   4× Head Engine Internal Accumulator (4 heads × 8 dims each)
//   → Read & Buffer
//   → Exponent Align
//   → Concat (32 dims)
//   → Compute Engine (32×32 matrix multiply)
//   → BFP Converter
//   → Result Buffer
//
// 版本：v2.0
// 日期：2024-11-15
//================================================================================

module output_projection #(
    parameter NUM_HEADS      = 4,
    parameter TOKEN_BATCH    = 32,
    parameter HEAD_DIM       = 8,
    parameter DIM            = 32,
    parameter DATA_WIDTH     = 8,
    parameter EXP_WIDTH      = 8,
    parameter ACCUM_WIDTH    = 24,
    
    // CE配置
    parameter CE_OUTPUT_WIDTH = 32,
    parameter CE_INTERNAL_WIDTH = 39,
    parameter CE_GUARD_BITS = 7,
    parameter CE_ENABLE_ROUNDING = 1
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 控制接口
    //===========================================================================
    input  wire start,
    input  wire [5:0] batch_id,
    input  wire [4:0] tokens_in_batch,
    output reg  done,
    output reg  busy,
    
    //===========================================================================
    // 4个Head Engine的Accumulator读取接口
    //===========================================================================
    // Head 0 读取
    output reg  accum_rd_en_h0,
    output reg  [4:0] accum_rd_row_h0,
    output reg  [2:0] accum_rd_dim_h0,
    input  wire signed [ACCUM_WIDTH-1:0] accum_rd_mant_h0,
    input  wire [EXP_WIDTH-1:0] accum_rd_exp_h0,
    input  wire accum_rd_valid_h0,
    
    // Head 1 读取
    output reg  accum_rd_en_h1,
    output reg  [4:0] accum_rd_row_h1,
    output reg  [2:0] accum_rd_dim_h1,
    input  wire signed [ACCUM_WIDTH-1:0] accum_rd_mant_h1,
    input  wire [EXP_WIDTH-1:0] accum_rd_exp_h1,
    input  wire accum_rd_valid_h1,
    
    // Head 2 读取
    output reg  accum_rd_en_h2,
    output reg  [4:0] accum_rd_row_h2,
    output reg  [2:0] accum_rd_dim_h2,
    input  wire signed [ACCUM_WIDTH-1:0] accum_rd_mant_h2,
    input  wire [EXP_WIDTH-1:0] accum_rd_exp_h2,
    input  wire accum_rd_valid_h2,
    
    // Head 3 读取
    output reg  accum_rd_en_h3,
    output reg  [4:0] accum_rd_row_h3,
    output reg  [2:0] accum_rd_dim_h3,
    input  wire signed [ACCUM_WIDTH-1:0] accum_rd_mant_h3,
    input  wire [EXP_WIDTH-1:0] accum_rd_exp_h3,
    input  wire accum_rd_valid_h3,
    
    //===========================================================================
    // W_O权重接口
    //===========================================================================
    output reg  weight_req,
    input  wire weight_ready,
    // ✅ 修改：使用数组格式（每列一个指数）
    input  wire [DIM*EXP_WIDTH-1:0] weight_exp_array,
    input  wire [DIM*DIM*DATA_WIDTH-1:0] weight_mant,
    
    //===========================================================================
    // 结果输出接口（BFP格式到Result Buffer）
    //===========================================================================
    output reg  result_wr_en,
    output reg  [9:0] result_wr_addr,
    output reg  [EXP_WIDTH-1:0] result_exp,
    output reg  [DIM*DATA_WIDTH-1:0] result_mant,
    
    //===========================================================================
    // 调试接口
    //===========================================================================
    output wire [3:0] dbg_state
);

//================================================================================
// 本地参数
//================================================================================

localparam BATCH_NUM = (641 + TOKEN_BATCH - 1) / TOKEN_BATCH;
localparam CE_BASE_EXP_WIDTH = EXP_WIDTH + 1;

//================================================================================
// 状态机定义
//================================================================================

localparam STATE_IDLE          = 4'd0;
localparam STATE_LOAD_WEIGHT   = 4'd1;
localparam STATE_WAIT_WEIGHT   = 4'd2;
localparam STATE_READ_HEADS    = 4'd3;   // 并行读取4个head的所有维度
localparam STATE_WAIT_READ     = 4'd4;   // 等待读取完成
localparam STATE_ALIGN_CONCAT  = 4'd5;   // 指数对齐并concat
localparam STATE_SEND_CE       = 4'd6;   // 发送给CE
localparam STATE_WAIT_CE       = 4'd7;   // 等待CE完成
localparam STATE_BFP_CONVERT   = 4'd8;   // BFP转换
localparam STATE_WRITE_RESULT  = 4'd9;   // 写入结果
localparam STATE_NEXT_TOKEN    = 4'd10;  // 下一个token
localparam STATE_DONE          = 4'd11;  // 完成

reg [3:0] state;

assign dbg_state = state;

//================================================================================
// 内部寄存器
//================================================================================

// Token计数
reg [4:0] token_idx;
reg [9:0] global_token_addr;

// 维度计数（并行读取4个head的同一维度）
reg [2:0] dim_idx;
reg [3:0] read_wait_counter;  // 等待读取valid的计数器

// 权重缓存
// ✅ 修改：缓存指数数组
reg [DIM*EXP_WIDTH-1:0] weight_exp_array_buf;
reg [DIM*DIM*DATA_WIDTH-1:0] weight_mant_buf;
reg weight_loaded;

// 读取的多头数据（BFP格式）
reg signed [ACCUM_WIDTH-1:0] heads_mant [0:NUM_HEADS-1][0:HEAD_DIM-1];
reg [EXP_WIDTH-1:0] heads_exp [0:NUM_HEADS-1];

// 指数对齐后的数据
reg [EXP_WIDTH-1:0] max_exp;
reg signed [ACCUM_WIDTH-1:0] aligned_mants [0:NUM_HEADS-1][0:HEAD_DIM-1];

// Concat后的32维向量
reg signed [ACCUM_WIDTH-1:0] concat_mant [0:DIM-1];
reg [EXP_WIDTH-1:0] concat_exp;

//================================================================================
// Compute Engine信号
//================================================================================

reg ce_input_valid;
wire ce_input_ready;
wire [DIM-1:0] ce_result_valids;

wire signed [DIM*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_array;
wire [DIM*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_array;
wire [DIM-1:0] ce_result_zero_array;

//================================================================================
// BFP转换器信号
//================================================================================

wire [DIM-1:0] bfp_valids;
wire signed [DIM*DATA_WIDTH-1:0] bfp_mants;
wire [EXP_WIDTH-1:0] bfp_shared_exp;
wire bfp_overflow;

wire ce_compute_done;
wire bfp_convert_done;

//================================================================================
// Compute Engine实例化
//================================================================================

compute_engine #(
    .G_OUT(4),
    .T_OUT(8),
    .NUM_PE(2),
    .PE_TYPE_0(0),
    .PE_TYPE_1(0),
    .EXP_WIDTH(EXP_WIDTH),
    .INPUT_MANT_WIDTH(ACCUM_WIDTH),
    .ELEM_PE0(16),
    .ELEM_PE1(16),
    .TOTAL_ELEM(32),
    .TOTAL_WIDTH(32*ACCUM_WIDTH),
    .INTERNAL_WIDTH(CE_INTERNAL_WIDTH),
    .OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
    .GUARD_BITS(CE_GUARD_BITS),
    .ENABLE_ROUNDING(CE_ENABLE_ROUNDING),
    .FIFO_DEPTH(16),
    .HANDSHAKE_TIMEOUT(100)
) u_compute_engine (
    .clk(clk),
    .rst_n(rst_n),
    .flush(1'b0),
    
    .input_valid(ce_input_valid),
    .input_ready(ce_input_ready),
    
    // 输入X（concat后的32维向量）
    .exp_X(concat_exp),
    .mant_X_block({concat_mant[31], concat_mant[30], concat_mant[29], concat_mant[28],
                   concat_mant[27], concat_mant[26], concat_mant[25], concat_mant[24],
                   concat_mant[23], concat_mant[22], concat_mant[21], concat_mant[20],
                   concat_mant[19], concat_mant[18], concat_mant[17], concat_mant[16],
                   concat_mant[15], concat_mant[14], concat_mant[13], concat_mant[12],
                   concat_mant[11], concat_mant[10], concat_mant[9], concat_mant[8],
                   concat_mant[7], concat_mant[6], concat_mant[5], concat_mant[4],
                   concat_mant[3], concat_mant[2], concat_mant[1], concat_mant[0]}),
    
    // 权重W_O
    // ✅ 修改：使用指数数组（每列一个指数）
    .exp_W_array(weight_exp_array_buf),
    .mant_W_blocks(weight_mant_buf),
    
    // 输出
    .result_valids(ce_result_valids),
    .result_ready(1'b1),
    .result_fixed_array(ce_result_fixed_array),
    .result_base_exp_array(ce_result_base_exp_array),
    .result_zero_array(ce_result_zero_array)
);

assign ce_compute_done = &ce_result_valids;  // 所有输出都valid

//================================================================================
// BFP转换器实例化
//================================================================================

fixed_to_independent_bfp #(
    .NUM_ELEMENTS(DIM),
    .INPUT_WIDTH(CE_OUTPUT_WIDTH),
    .INPUT_BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH),
    .OUTPUT_MANT_WIDTH(DATA_WIDTH),
    .OUTPUT_EXP_WIDTH(EXP_WIDTH)
) u_bfp_converter (
    .clk(clk),
    .rst_n(rst_n),
    
    .input_valids(ce_result_valids),
    .input_fixed_values(ce_result_fixed_array),
    .input_base_exps(ce_result_base_exp_array),
    .input_zeros(ce_result_zero_array),
    
    .output_valids(bfp_valids),
    .output_mants(bfp_mants),
    .output_shared_exp(bfp_shared_exp),
    .output_overflow(bfp_overflow)
);

assign bfp_convert_done = &bfp_valids;

//================================================================================
// 主状态机
//================================================================================

integer h, d;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_IDLE;
        done <= 1'b0;
        busy <= 1'b0;
        token_idx <= 5'd0;
        global_token_addr <= 10'd0;
        dim_idx <= 3'd0;
        read_wait_counter <= 4'd0;
        
        weight_loaded <= 1'b0;
        weight_exp_array_buf <= {DIM*EXP_WIDTH{1'b0}};
        weight_mant_buf <= {DIM*DIM*DATA_WIDTH{1'b0}};
        
        accum_rd_en_h0 <= 1'b0;
        accum_rd_en_h1 <= 1'b0;
        accum_rd_en_h2 <= 1'b0;
        accum_rd_en_h3 <= 1'b0;
        
        weight_req <= 1'b0;
        result_wr_en <= 1'b0;
        ce_input_valid <= 1'b0;
        
    end else begin
        // 默认值
        accum_rd_en_h0 <= 1'b0;
        accum_rd_en_h1 <= 1'b0;
        accum_rd_en_h2 <= 1'b0;
        accum_rd_en_h3 <= 1'b0;
        weight_req <= 1'b0;
        result_wr_en <= 1'b0;
        done <= 1'b0;
        
        case (state)
            //====================================================================
            // STATE_IDLE：等待启动
            //====================================================================
            STATE_IDLE: begin
                if (start) begin
                    busy <= 1'b1;
                    done <= 1'b0;
                    token_idx <= 5'd0;
                    global_token_addr <= batch_id * TOKEN_BATCH;
                    dim_idx <= 3'd0;
                    
                    if (!weight_loaded) begin
                        state <= STATE_LOAD_WEIGHT;
                    end else begin
                        state <= STATE_READ_HEADS;
                    end
                    
                    $display("\n[%0t] ========================================", $time);
                    $display("[%0t] Output Projection v2.0: Started", $time);
                    $display("           batch=%0d tokens=%0d", batch_id, tokens_in_batch);
                end else begin
                    busy <= 1'b0;
                end
            end
            
            //====================================================================
            // STATE_LOAD_WEIGHT：请求W_O权重
            //====================================================================
            STATE_LOAD_WEIGHT: begin
                weight_req <= 1'b1;
                state <= STATE_WAIT_WEIGHT;
            end
            
            //====================================================================
            // STATE_WAIT_WEIGHT：等待权重就绪
            //====================================================================
            STATE_WAIT_WEIGHT: begin
                if (weight_ready) begin
                    weight_req <= 1'b0;
                    // ✅ 修改：加载指数数组
                    weight_exp_array_buf <= weight_exp_array;
                    weight_mant_buf <= weight_mant;
                    weight_loaded <= 1'b1;
                    state <= STATE_READ_HEADS;
                    
                    $display("[%0t]   Weight loaded (array format)", $time);
                end
            end
            
            //====================================================================
            // STATE_READ_HEADS：并行读取4个head的当前维度
            //====================================================================
            STATE_READ_HEADS: begin
                if (dim_idx < HEAD_DIM) begin
                    // 同时向4个head发出读请求（读取相同的维度）
                    accum_rd_en_h0 <= 1'b1;
                    accum_rd_row_h0 <= token_idx;
                    accum_rd_dim_h0 <= dim_idx;
                    
                    accum_rd_en_h1 <= 1'b1;
                    accum_rd_row_h1 <= token_idx;
                    accum_rd_dim_h1 <= dim_idx;
                    
                    accum_rd_en_h2 <= 1'b1;
                    accum_rd_row_h2 <= token_idx;
                    accum_rd_dim_h2 <= dim_idx;
                    
                    accum_rd_en_h3 <= 1'b1;
                    accum_rd_row_h3 <= token_idx;
                    accum_rd_dim_h3 <= dim_idx;
                    
                    read_wait_counter <= 4'd0;
                    state <= STATE_WAIT_READ;
                    
                    $display("[%0t]   Reading dim=%0d from 4 heads (token=%0d)",
                             $time, dim_idx, token_idx);
                    
                end else begin
                    // 所有维度读取完成
                    dim_idx <= 3'd0;
                    state <= STATE_ALIGN_CONCAT;
                    
                    $display("[%0t]   All dimensions read for token=%0d",
                             $time, token_idx);
                end
            end
            
            //====================================================================
            // STATE_WAIT_READ：等待4个head的读取完成
            //====================================================================
            STATE_WAIT_READ: begin
                // 检查所有4个head是否都返回valid
                if (accum_rd_valid_h0 && accum_rd_valid_h1 && 
                    accum_rd_valid_h2 && accum_rd_valid_h3) begin
                    
                    // 缓存读取的数据
                    heads_mant[0][dim_idx] <= accum_rd_mant_h0;
                    heads_mant[1][dim_idx] <= accum_rd_mant_h1;
                    heads_mant[2][dim_idx] <= accum_rd_mant_h2;
                    heads_mant[3][dim_idx] <= accum_rd_mant_h3;
                    
                    // 只在第一个维度时缓存指数（所有维度共享）
                    if (dim_idx == 3'd0) begin
                        heads_exp[0] <= accum_rd_exp_h0;
                        heads_exp[1] <= accum_rd_exp_h1;
                        heads_exp[2] <= accum_rd_exp_h2;
                        heads_exp[3] <= accum_rd_exp_h3;
                        
                        $display("[%0t]     Exps: h0=%0d h1=%0d h2=%0d h3=%0d",
                                 $time, accum_rd_exp_h0, accum_rd_exp_h1,
                                 accum_rd_exp_h2, accum_rd_exp_h3);
                    end
                    
                    // 移到下一个维度
                    dim_idx <= dim_idx + 3'd1;
                    state <= STATE_READ_HEADS;
                    
                end else begin
                    // 超时保护
                    read_wait_counter <= read_wait_counter + 4'd1;
                    if (read_wait_counter > 4'd10) begin
                        $display("[%0t] ERROR: Read timeout! valids=%04b",
                                 $time, {accum_rd_valid_h3, accum_rd_valid_h2,
                                        accum_rd_valid_h1, accum_rd_valid_h0});
                        state <= STATE_IDLE;
                    end
                end
            end
            
            //====================================================================
            // STATE_ALIGN_CONCAT：指数对齐并concat
            //====================================================================
            STATE_ALIGN_CONCAT: begin
                // 组合逻辑已完成对齐和concat
                state <= STATE_SEND_CE;
                
                $display("[%0t]   Align & Concat: max_exp=%0d concat_exp=%0d",
                         $time, max_exp, concat_exp);
            end
            
            //====================================================================
            // STATE_SEND_CE：发送给CE计算
            //====================================================================
            STATE_SEND_CE: begin
                if (ce_input_ready && !ce_input_valid) begin
                    ce_input_valid <= 1'b1;
                    state <= STATE_WAIT_CE;
                    
                    $display("[%0t]   Sent to CE", $time);
                end
            end
            
            //====================================================================
            // STATE_WAIT_CE：等待CE完成
            //====================================================================
            STATE_WAIT_CE: begin
                ce_input_valid <= 1'b0;
                
                if (ce_compute_done) begin
                    state <= STATE_BFP_CONVERT;
                    
                    $display("[%0t]   CE done", $time);
                end
            end
            
            //====================================================================
            // STATE_BFP_CONVERT：等待BFP转换完成
            //====================================================================
            STATE_BFP_CONVERT: begin
                if (bfp_convert_done) begin
                    state <= STATE_WRITE_RESULT;
                    
                    $display("[%0t]   BFP convert done: exp=%0d",
                             $time, bfp_shared_exp);
                end
            end
            
            //====================================================================
            // STATE_WRITE_RESULT：写入结果
            //====================================================================
            STATE_WRITE_RESULT: begin
                result_wr_en <= 1'b1;
                result_wr_addr <= global_token_addr + token_idx;
                result_exp <= bfp_shared_exp;
                result_mant <= bfp_mants;
                
                state <= STATE_NEXT_TOKEN;
                
                $display("[%0t]   Write result: addr=%0d exp=%0d",
                         $time, global_token_addr + token_idx, bfp_shared_exp);
            end
            
            //====================================================================
            // STATE_NEXT_TOKEN：下一个token
            //====================================================================
            STATE_NEXT_TOKEN: begin
                if (token_idx < tokens_in_batch - 1) begin
                    token_idx <= token_idx + 5'd1;
                    dim_idx <= 3'd0;
                    state <= STATE_READ_HEADS;
                    
                    $display("[%0t]   Moving to token=%0d", $time, token_idx + 1);
                end else begin
                    state <= STATE_DONE;
                end
            end
            
            //====================================================================
            // STATE_DONE：完成
            //====================================================================
            STATE_DONE: begin
                done <= 1'b1;
                busy <= 1'b0;
                state <= STATE_IDLE;
                
                $display("[%0t] Output Projection v2.0: Completed", $time);
                $display("[%0t] ========================================\n", $time);
            end
            
            default: state <= STATE_IDLE;
        endcase
    end
end

//================================================================================
// 指数对齐和Concat逻辑（组合逻辑）
//================================================================================

always @(*) begin
    // 1. 找到最大指数
    max_exp = heads_exp[0];
    for (h = 1; h < NUM_HEADS; h = h + 1) begin
        if (heads_exp[h] > max_exp) begin
            max_exp = heads_exp[h];
        end
    end
    
    // 2. 对齐各个head的尾数
    for (h = 0; h < NUM_HEADS; h = h + 1) begin
        for (d = 0; d < HEAD_DIM; d = d + 1) begin
            if (max_exp > heads_exp[h]) begin
                // 右移对齐
                aligned_mants[h][d] = heads_mant[h][d] >>> (max_exp - heads_exp[h]);
            end else begin
                aligned_mants[h][d] = heads_mant[h][d];
            end
        end
    end
    
    // 3. Concat：[head0_dim0..7, head1_dim0..7, head2_dim0..7, head3_dim0..7]
    for (h = 0; h < NUM_HEADS; h = h + 1) begin
        for (d = 0; d < HEAD_DIM; d = d + 1) begin
            concat_mant[h * HEAD_DIM + d] = aligned_mants[h][d];
        end
    end
    
    // 4. 使用最大指数作为concat的共享指数
    concat_exp = max_exp;
end

//================================================================================
// 初始化
//================================================================================

`ifdef SIMULATION
initial begin
    $display("========================================");
    $display("Output Projection v2.0 Initialized");
    $display("  Architecture: 4 independent head accumulators");
    $display("  Heads: %0d × Dims: %0d = Total: %0d", NUM_HEADS, HEAD_DIM, DIM);
    $display("  Accumulator width: %0d bits", ACCUM_WIDTH);
    $display("  Output width: %0d bits", DATA_WIDTH);
    $display("  Reading strategy: Parallel read from 4 heads");
    $display("========================================");
end
`endif

endmodule