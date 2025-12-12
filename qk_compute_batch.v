`timescale 1ns / 1ps

module qk_compute_batch #(
    parameter NUM_QUERIES     = 32,       
    parameter CHUNK_SIZE      = 32,       
    parameter HEAD_DIM        = 8,       
    parameter DATA_WIDTH      = 8,        
    parameter EXP_WIDTH       = 8,       
    parameter SCORE_WIDTH     = 8,        
    parameter CE_OUTPUT_WIDTH = 32,      

    parameter NUM_PE          = 2,
    parameter PE_TYPE_0       = 0,        
    parameter PE_TYPE_1       = 0, 
    parameter ELEM_PE0        = 16,
    parameter ELEM_PE1        = 16 
)(
    input  wire clk,
    input  wire rst_n,
    
    input  wire start,
    output reg  done,
    output reg  busy,
    
    input  wire [(NUM_QUERIES*EXP_WIDTH)-1:0] q_batch_exp,
    input  wire [(NUM_QUERIES*HEAD_DIM*DATA_WIDTH)-1:0] q_batch_mant,
    input  wire [(CHUNK_SIZE*EXP_WIDTH)-1:0] k_chunk_exp,
    input  wire [(CHUNK_SIZE*HEAD_DIM*DATA_WIDTH)-1:0] k_chunk_mant,
    
    output reg  scores_valid,
    output reg  [(NUM_QUERIES*EXP_WIDTH)-1:0] scores_exp_batch,
    output reg  [(NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH)-1:0] scores_batch
);

    //===================================================================================
    // 内部参数与信号
    //===================================================================================
    localparam TOTAL_ELEM      = HEAD_DIM;
    localparam CE_BASE_EXP_WIDTH = EXP_WIDTH + 1;
    localparam G_OUT = 1;      
    localparam T_OUT = CHUNK_SIZE; 

    localparam IDLE         = 3'd0;
    localparam SEND_REQ     = 3'd1; 
    localparam COLLECT      = 3'd2; 
    localparam CONVERT      = 3'd3; 
    localparam FINISH       = 3'd4;

    reg [2:0] state;
    reg [5:0] query_idx;

    // CE 接口信号
    reg ce_input_valid;
    wire ce_input_ready;
    wire [T_OUT-1:0] ce_result_valids;
    wire ce_result_ready; 
    
    // CE 结果缓存
    // 用于暂存从CE出来的结果，直到凑齐32个
    reg signed [T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_buf;
    reg [T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_buf;
    reg [T_OUT-1:0] ce_result_zero_buf;
    reg [T_OUT-1:0] collected_mask;     // 记录哪些位置已经收到了结果
    wire all_collected;                 // 标记是否收齐
    
    assign all_collected = &collected_mask;

    // 当前处理的Q向量
    reg [EXP_WIDTH-1:0] current_q_exp;
    reg [(HEAD_DIM*DATA_WIDTH)-1:0] current_q_mant;
    
    // CE 原始输出
    wire signed [T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_array;
    wire [T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_array;
    wire [T_OUT-1:0] ce_result_zero_array;

    // BFP 转换器信号
    wire [EXP_WIDTH-1:0] bfp_shared_exp;
    wire signed [(CHUNK_SIZE*SCORE_WIDTH)-1:0] bfp_mants_packed;
    wire [CHUNK_SIZE-1:0] bfp_output_valids;

    //===================================================================================
    // 逻辑实现
    //===================================================================================

    // 1. 提取当前 Query 的 Q 向量
    always @(*) begin
        current_q_exp = q_batch_exp[query_idx*EXP_WIDTH +: EXP_WIDTH];
        current_q_mant = q_batch_mant[query_idx*HEAD_DIM*DATA_WIDTH +: HEAD_DIM*DATA_WIDTH];
    end

    // 2. 生成 ce_result_ready
    assign ce_result_ready = (state == COLLECT && !all_collected);

    // 3. 结果收集逻辑 (Core Fix)
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ce_result_fixed_buf    <= {(T_OUT*CE_OUTPUT_WIDTH){1'b0}};
            ce_result_base_exp_buf <= {(T_OUT*CE_BASE_EXP_WIDTH){1'b0}};
            ce_result_zero_buf     <= {T_OUT{1'b0}};
            collected_mask         <= {T_OUT{1'b0}};
        end else begin
            // 每次开始新一轮 Query 计算时，清空 mask
            if (state == SEND_REQ && ce_input_valid && ce_input_ready) begin
                collected_mask <= {T_OUT{1'b0}};
            end
            
            // 在收集阶段，把有效的结果填入 Buffer
            if (state == COLLECT) begin
                for (i = 0; i < T_OUT; i = i + 1) begin
                    // 如果 CE 该通道有效，且我还没收集过这个位置
                    if (ce_result_valids[i] && !collected_mask[i]) begin
                        ce_result_fixed_buf[i*CE_OUTPUT_WIDTH +: CE_OUTPUT_WIDTH] 
                            <= ce_result_fixed_array[i*CE_OUTPUT_WIDTH +: CE_OUTPUT_WIDTH];
                        ce_result_base_exp_buf[i*CE_BASE_EXP_WIDTH +: CE_BASE_EXP_WIDTH] 
                            <= ce_result_base_exp_array[i*CE_BASE_EXP_WIDTH +: CE_BASE_EXP_WIDTH];
                        ce_result_zero_buf[i] <= ce_result_zero_array[i];
                        collected_mask[i] <= 1'b1;
                    end
                end
            end
        end
    end

    // 4. 实例化 Compute Engine
    compute_engine #(
        .G_OUT(G_OUT),              
        .T_OUT(T_OUT),              
        .NUM_PE(NUM_PE),            
        .PE_TYPE_0(PE_TYPE_0),
        .PE_TYPE_1(PE_TYPE_1),
        .EXP_WIDTH(EXP_WIDTH),
        .INPUT_MANT_WIDTH(DATA_WIDTH),
        .ELEM_PE0(ELEM_PE0),
        .ELEM_PE1(ELEM_PE1),
        .TOTAL_ELEM(TOTAL_ELEM),    
        .OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
        .INTERNAL_WIDTH(39),
        .GUARD_BITS(7),
        .ENABLE_ROUNDING(1),
        .HANDSHAKE_TIMEOUT(255)
    ) u_ce (
        .clk(clk),
        .rst_n(rst_n),
        .flush(1'b0),
        
        .input_valid(ce_input_valid),
        .input_ready(ce_input_ready),
        .exp_X(current_q_exp),
        .mant_X_block(current_q_mant),
        .exp_W_array(k_chunk_exp),
        .mant_W_blocks(k_chunk_mant),
        
        .result_valids(ce_result_valids),
        .result_ready(ce_result_ready),
        .result_fixed_array(ce_result_fixed_array),
        .result_base_exp_array(ce_result_base_exp_array),
        .result_zero_array(ce_result_zero_array)
    );

    // 5. BFP 转换器
    bfp_converter #(
        .TOTAL_RESULTS(CHUNK_SIZE),           
        .FIXED_WIDTH(CE_OUTPUT_WIDTH),        
        .BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH),   
        .OUTPUT_MANT_WIDTH(SCORE_WIDTH),      
        .OUTPUT_EXP_WIDTH(EXP_WIDTH)          
    ) u_bfp_converter (
        .clk(clk),
        .rst_n(rst_n),
        .flush(1'b0),

        .input_valids({T_OUT{all_collected && (state == COLLECT)}}), 
        .input_fixed_array(ce_result_fixed_buf),
        .input_base_exp_array(ce_result_base_exp_buf),
        .input_zero_array(ce_result_zero_buf),
        
        .output_valids(bfp_output_valids),
        .output_mant_array(bfp_mants_packed),
        .output_shared_exp(bfp_shared_exp),
        .output_overflow() 
    );

    // 6. 主状态机
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            query_idx <= 6'd0;
            done <= 1'b0;
            busy <= 1'b0;
            ce_input_valid <= 1'b0;
            scores_valid <= 1'b0;
            scores_exp_batch <= {(NUM_QUERIES*EXP_WIDTH){1'b0}};
            scores_batch <= {(NUM_QUERIES*CHUNK_SIZE*SCORE_WIDTH){1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        done <= 1'b0;
                        scores_valid <= 1'b0;
                        busy <= 1'b1;
                        query_idx <= 6'd0;
                        state <= SEND_REQ;
                    end else begin
                        busy <= 1'b0;
                    end
                end
                
                SEND_REQ: begin
                    ce_input_valid <= 1'b1;
                    // 等待 CE 接收输入请求
                    if (ce_input_valid && ce_input_ready) begin
                        ce_input_valid <= 1'b0;
                        state <= COLLECT;
                    end
                end
                
                COLLECT: begin
                    ce_input_valid <= 1'b0;
                    if (all_collected) begin
                        state <= CONVERT;
                    end
                end
                
                CONVERT: begin

                    if (|bfp_output_valids) begin
                        scores_exp_batch[query_idx*EXP_WIDTH +: EXP_WIDTH] <= bfp_shared_exp;
                        scores_batch[query_idx*CHUNK_SIZE*SCORE_WIDTH +: CHUNK_SIZE*SCORE_WIDTH] <= bfp_mants_packed;

                        if (query_idx < NUM_QUERIES - 1) begin
                            query_idx <= query_idx + 6'd1;
                            state <= SEND_REQ;
                        end else begin
                            state <= FINISH;
                        end
                    end
                end
                
                FINISH: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    scores_valid <= 1'b1;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule