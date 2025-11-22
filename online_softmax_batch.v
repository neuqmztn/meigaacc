module online_softmax_batch #(
    parameter NUM_HEADS      = 4,
    parameter NUM_QUERIES    = 32,
    parameter CHUNK_SIZE     = 32,
    parameter DATA_WIDTH     = 8,
    parameter EXP_WIDTH      = 8,
    parameter SCORE_WIDTH    = 16,
    parameter ACCUM_WIDTH    = 24
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 控制接口
    //===========================================================================
    input  wire start,
    input  wire [1:0] head_idx,        // 当前head编号
    input  wire chunk_first,           // 是否第一个chunk
    input  wire chunk_last,            // 是否最后一个chunk
    output reg  done,
    output reg  busy,
    
    //===========================================================================
    // Scores输入：32行 × 32列
    // 每行有独立的BFP指数
    //===========================================================================
    input  wire [(NUM_QUERIES*EXP_WIDTH)-1:0] scores_exp_batch,
    input  wire [(NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH)-1:0] scores_batch,
    
    //===========================================================================
    // Weights输出：32行 × 32列
    //===========================================================================
    output reg  weights_valid,
    output reg  [(NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH)-1:0] weights_batch,
    
    //===========================================================================
    // 统计量存储接口（连接到本head的私有stats_buffer）
    //===========================================================================
    // Max读取
    output wire max_rd_en,
    output wire [1:0] max_rd_head,
    output wire [4:0] max_rd_row,
    input  wire signed [SCORE_WIDTH-1:0] max_rd_value,
    
    // Max写入
    output wire max_wr_en,
    output wire [1:0] max_wr_head,
    output wire [4:0] max_wr_row,
    output wire signed [SCORE_WIDTH-1:0] max_wr_value,
    
    // Sum读取
    output wire sum_rd_en,
    output wire [1:0] sum_rd_head,
    output wire [4:0] sum_rd_row,
    input  wire signed [ACCUM_WIDTH-1:0] sum_rd_value,
    
    // Sum写入
    output wire sum_wr_en,
    output wire [1:0] sum_wr_head,
    output wire [4:0] sum_wr_row,
    output wire signed [ACCUM_WIDTH-1:0] sum_wr_value
);

//===================================================================================
// 本地参数
//===================================================================================

// 状态机
localparam IDLE     = 2'd0;
localparam PROCESS  = 2'd1;
localparam WAIT     = 2'd2;
localparam DONE_ST  = 2'd3;

reg [1:0] state;
reg [5:0] query_idx;  // 0-31

//===================================================================================
// Online Softmax Engine 控制信号
//===================================================================================

reg softmax_start;
wire softmax_done;
wire softmax_busy;

// 当前query的输入
reg [EXP_WIDTH-1:0] current_scores_exp;
reg [(CHUNK_SIZE*SCORE_WIDTH)-1:0] current_scores_mants;
reg [4:0] current_row_idx;

// 当前query的输出
wire weights_valid_single;
wire [(CHUNK_SIZE*SCORE_WIDTH)-1:0] weights_single;

// 重归一化信号（暂未使用）
wire renorm_en;
wire signed [SCORE_WIDTH-1:0] renorm_scale;

// 从online_softmax_engine输出的head信号（不使用，因为buffer是单head的）
wire [1:0] softmax_max_wr_head;
wire [1:0] softmax_sum_wr_head;

//===================================================================================
// 提取当前query的scores
//===================================================================================

always @(*) begin
    // 提取第query_idx行的指数和尾数
    current_scores_exp = scores_exp_batch[query_idx*EXP_WIDTH +: EXP_WIDTH];
    current_scores_mants = scores_batch[query_idx*CHUNK_SIZE*SCORE_WIDTH +: CHUNK_SIZE*SCORE_WIDTH];
    current_row_idx = query_idx[4:0];
end

//===================================================================================
// 实例化Online Softmax Engine（处理单个query）
//===================================================================================

online_softmax_engine #(
    .NUM_HEADS(NUM_HEADS),
    .TOKEN_BATCH(NUM_QUERIES),
    .K_CHUNK_SIZE(CHUNK_SIZE),
    .DATA_WIDTH(DATA_WIDTH),
    .EXP_WIDTH(EXP_WIDTH),
    .SCORE_WIDTH(SCORE_WIDTH),
    .ACCUM_WIDTH(ACCUM_WIDTH)
) u_softmax_engine (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制
    .start(softmax_start),
    .head_idx(head_idx),
    .row_idx(current_row_idx),
    .chunk_id(5'd0),              // chunk管理在外部
    .chunk_first(chunk_first),
    .chunk_last(chunk_last),
    .chunk_size(6'd32),
    .done(softmax_done),
    .busy(softmax_busy),
    
    // Scores输入（当前query）
    .scores_shared_exp(current_scores_exp),
    .scores_mants_packed(current_scores_mants),
    
    // 统计量读取
    .max_rd_value(max_rd_value),
    .sum_rd_value(sum_rd_value),
    
    // 统计量写入
    .max_wr_en(max_wr_en),
    .max_wr_head(softmax_max_wr_head),  // 接收但不使用
    .max_wr_row(max_wr_row),
    .max_wr_value(max_wr_value),
    
    .sum_wr_en(sum_wr_en),
    .sum_wr_head(softmax_sum_wr_head),  // 接收但不使用
    .sum_wr_row(sum_wr_row),
    .sum_wr_value(sum_wr_value),
    
    // 重归一化信号
    .renorm_en(renorm_en),
    .renorm_scale(renorm_scale),
    
    // Weights输出（当前query）
    .weights_valid(weights_valid_single),
    .weights_packed(weights_single)
);

// 统计量读取地址（组合逻辑，由engine内部控制）
assign max_rd_row = current_row_idx;
assign sum_rd_row = current_row_idx;

// 统计量读取使能（当softmax engine运行时有效）
assign max_rd_en = softmax_busy;
assign sum_rd_en = softmax_busy;

// Head索引（读取和写入都使用当前head）
assign max_rd_head = head_idx;
assign sum_rd_head = head_idx;
assign max_wr_head = softmax_max_wr_head;
assign sum_wr_head = softmax_sum_wr_head;

//===================================================================================
// 主状态机：循环32次调用softmax engine
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        query_idx <= 6'd0;
        done <= 1'b0;
        busy <= 1'b0;
        softmax_start <= 1'b0;
        weights_valid <= 1'b0;
        weights_batch <= {(NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH){1'b0}};
        
    end else begin
        // 默认：清除单周期脉冲
        softmax_start <= 1'b0;
        weights_valid <= 1'b0;
        
        case (state)
            //===================================================================
            // IDLE: 等待启动
            //===================================================================
            IDLE: begin
                done <= 1'b0;
                
                if (start) begin
                    busy <= 1'b1;
                    query_idx <= 6'd0;
                    state <= PROCESS;
                    
                    $display("[%0t] Softmax_Batch: Started, head=%0d, chunk_first=%0b, chunk_last=%0b",
                             $time, head_idx, chunk_first, chunk_last);
                end else begin
                    busy <= 1'b0;
                end
            end
            
            //===================================================================
            // PROCESS: 启动当前query的softmax计算
            //===================================================================
            PROCESS: begin
                softmax_start <= 1'b1;
                state <= WAIT;
                
                $display("[%0t] Softmax_Batch: Processing query %0d/%0d",
                         $time, query_idx, NUM_QUERIES-1);
            end
            
            //===================================================================
            // WAIT: 等待当前query的softmax完成
            //===================================================================
            WAIT: begin
                if (softmax_done) begin
                    // 保存当前query的weights结果
                    weights_batch[query_idx*CHUNK_SIZE*SCORE_WIDTH +: CHUNK_SIZE*SCORE_WIDTH] <= weights_single;
                    
                    $display("[%0t] Softmax_Batch: Query %0d done",
                             $time, query_idx);
                    
                    // 检查是否所有query都完成
                    if (query_idx < NUM_QUERIES - 1) begin
                        query_idx <= query_idx + 6'd1;
                        state <= PROCESS;
                    end else begin
                        state <= DONE_ST;
                    end
                end
            end
            
            //===================================================================
            // DONE: 全部完成
            //===================================================================
            DONE_ST: begin
                done <= 1'b1;
                busy <= 1'b0;
                weights_valid <= 1'b1;
                state <= IDLE;
                
                $display("[%0t] Softmax_Batch: All %0d queries completed",
                         $time, NUM_QUERIES);
            end
            
            default: state <= IDLE;
        endcase
    end
end

//===================================================================================
// 参数检查
//===================================================================================

initial begin
    if (NUM_QUERIES != 32) begin
        $display("WARNING: NUM_QUERIES=%0d, expected 32", NUM_QUERIES);
    end
    if (CHUNK_SIZE != 32) begin
        $display("WARNING: CHUNK_SIZE=%0d, expected 32", CHUNK_SIZE);
    end
    
    $display("========================================");
    $display("Online Softmax Batch Module");
    $display("========================================");
    $display("Configuration:");
    $display("  Queries: %0d", NUM_QUERIES);
    $display("  Chunk size: %0d", CHUNK_SIZE);
    $display("  Score width: %0d", SCORE_WIDTH);
    $display("  Architecture: Serial (32 iterations)");
    $display("========================================");
end

endmodule