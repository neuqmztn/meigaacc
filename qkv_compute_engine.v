`timescale 1ns / 1ps

module qkv_compute_engine #(
    // Token配置
    parameter TOKEN_NUM      = 640,
    parameter TOKEN_BATCH    = 32,
    parameter DIM            = 32,
    
    // Attention配置
    parameter NUM_HEADS      = 4,
    parameter HEAD_DIM       = 8,
    
    // 数据格式
    parameter DATA_WIDTH     = 8,
    parameter EXP_WIDTH      = 8,
    parameter MANT_WIDTH     = 8,
    
    // CE矩阵配置
    parameter G_OUT          = 4,
    parameter T_OUT          = 8,

    parameter TOTAL_ELEM     = 32, 
    parameter TOTAL_WIDTH    = 256,
    
    // CE输出格式
    parameter CE_OUTPUT_WIDTH = 32,
    parameter CE_BASE_EXP_WIDTH = 9
)(
    input  wire clk,
    input  wire rst_n,
    
    //------------ 控制接口 ------------
    input  wire start,
    input  wire [1:0] compute_mode,
    input  wire [5:0] batch_id,
    input  wire [4:0] tokens_in_batch,
    input  wire enable_shared_exp,
    output reg  done,
    output reg  busy,
    output reg  error,
    
    //------------ Token读取接口 ------------
    output reg  token_rd_en,
    output reg  [9:0] token_rd_addr,
    input  wire [EXP_WIDTH-1:0] token_rd_exp,
    input  wire [DIM*DATA_WIDTH-1:0] token_rd_mant,
    
    //------------ 权重读取接口 ------------
    output reg  weight_req,
    output reg  [1:0] weight_type,
    input  wire weight_ack,
    input  wire weight_valid,
    input  wire [G_OUT*T_OUT*EXP_WIDTH-1:0] weight_exp_array,
    input  wire [G_OUT*T_OUT*TOTAL_WIDTH-1:0] weight_mant_blocks,
    
    //------------ Q矩阵存储接口 ------------
    output reg  storage_wr_en,
    output reg  [1:0] storage_matrix_type,
    output reg  [1:0] storage_head_id,
    output reg  [4:0] storage_token_id,
    output reg  [EXP_WIDTH-1:0] storage_shared_exp,
    output reg  [HEAD_DIM*DATA_WIDTH-1:0] storage_mant_packed,
    output reg  storage_overflow_flag,
    
    //------------ KV Cache接口 ------------
    output reg  kv_wr_en,
    output reg  kv_wr_type,
    output reg  [1:0] kv_wr_head,
    output reg  [9:0] kv_wr_token,
    output reg  [EXP_WIDTH-1:0] kv_wr_exp,
    output reg  [HEAD_DIM*DATA_WIDTH-1:0] kv_wr_mant,
    
    //------------ 状态输出 ------------
    output reg  all_kv_written
);
    //================================================================================
    // 状态机定义
    //================================================================================
    localparam STATE_IDLE           = 4'h0;
    localparam STATE_REQ_WEIGHT     = 4'h1;
    localparam STATE_WAIT_WEIGHT    = 4'h2;
    localparam STATE_LOAD_TOKEN     = 4'h3;
    localparam STATE_SEND_CE        = 4'h4;
    localparam STATE_WAIT_ACK       = 4'hB; 
    localparam STATE_WAIT_CE        = 4'h5;
    localparam STATE_WAIT_CONVERTER = 4'h6;
    localparam STATE_SAVE_RESULT    = 4'h7;
    localparam STATE_CLEAR_WR       = 4'h8;
    localparam STATE_NEXT_TOKEN     = 4'h9;
    localparam STATE_DONE           = 4'hA;

    reg [3:0] state;

    reg [9:0] token_counter;
    reg [9:0] global_token_idx;
    reg [4:0] batch_token_idx;
    reg [1:0] head_counter;
    reg [9:0] total_tokens;
    reg [9:0] kv_written_count;
    reg is_mode_q, is_mode_k, is_mode_v;
    reg [2:0] weight_loaded_flags;
    reg [G_OUT*T_OUT*EXP_WIDTH-1:0] weight_exp_cached;
    reg [G_OUT*T_OUT*TOTAL_WIDTH-1:0] weight_mant_cached;
    reg [EXP_WIDTH-1:0] token_exp_cached;
    reg [TOTAL_WIDTH-1:0] token_mant_cached;
    reg ce_input_valid;
    reg [EXP_WIDTH-1:0] ce_exp_X;
    reg [TOTAL_WIDTH-1:0] ce_mant_X;
    wire ce_input_ready;
    reg  ce_result_ready;
    wire [G_OUT*T_OUT-1:0] ce_result_valids;
    wire signed [G_OUT*T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_array;
    wire [G_OUT*T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_array;
    wire [G_OUT*T_OUT-1:0] ce_result_zero_array;
    reg signed [G_OUT*T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_buf;
    reg [G_OUT*T_OUT*CE_BASE_EXP_WIDTH-1:0] ce_result_base_exp_buf;
    reg [G_OUT*T_OUT-1:0] ce_result_zero_buf;
    reg ce_result_cached;
    reg [G_OUT*T_OUT-1:0] collected_mask;
    wire all_collected = &collected_mask;
    wire [G_OUT*T_OUT-1:0] converter_output_valids;
    wire signed [G_OUT*T_OUT*MANT_WIDTH-1:0] converter_output_mants;
    wire [G_OUT*EXP_WIDTH-1:0] converter_output_shared_exps;
    wire [G_OUT-1:0] converter_alignment_overflow;

    // CE输出收集逻辑
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ce_result_fixed_buf    <= {(G_OUT*T_OUT*CE_OUTPUT_WIDTH){1'b0}};
            ce_result_base_exp_buf <= {(G_OUT*T_OUT*CE_BASE_EXP_WIDTH){1'b0}};
            ce_result_zero_buf     <= {(G_OUT*T_OUT){1'b0}};
            ce_result_cached       <= 1'b0;
            collected_mask         <= {(G_OUT*T_OUT){1'b0}};
        end else begin
            if (state != STATE_WAIT_CE && state != STATE_WAIT_CONVERTER) begin
                 ce_result_cached <= 1'b0;
                 collected_mask   <= {(G_OUT*T_OUT){1'b0}};
            end
            case (state)
                STATE_WAIT_CE: begin
                    for (i = 0; i < G_OUT*T_OUT; i = i + 1) begin
                        if (ce_result_valids[i] && !collected_mask[i]) begin
                            ce_result_fixed_buf[i*CE_OUTPUT_WIDTH +: CE_OUTPUT_WIDTH] 
                                <= ce_result_fixed_array[i*CE_OUTPUT_WIDTH +: CE_OUTPUT_WIDTH];
                            ce_result_base_exp_buf[i*CE_BASE_EXP_WIDTH +: CE_BASE_EXP_WIDTH] 
                                <= ce_result_base_exp_array[i*CE_BASE_EXP_WIDTH +: CE_BASE_EXP_WIDTH];
                            ce_result_zero_buf[i] <= ce_result_zero_array[i];
                            collected_mask[i] <= 1'b1;
                        end
                    end
                    if (&collected_mask) begin
                          ce_result_cached <= 1'b1;
                    end
                end
            endcase
        end
    end
    
    //================================================================================
    // 实例化：恢复高性能 32 元素配置
    //================================================================================
    
    compute_engine #(
        .G_OUT(G_OUT),
        .T_OUT(T_OUT),
        .NUM_PE(2), 

        .PE_TYPE_0(0), 
        .PE_TYPE_1(0), 
        .EXP_WIDTH(EXP_WIDTH), 
        .INPUT_MANT_WIDTH(MANT_WIDTH),
        
   
        .ELEM_PE0(16),  
        .ELEM_PE1(16),  
        .TOTAL_ELEM(TOTAL_ELEM), 
        
        .INTERNAL_WIDTH(39), 
        .OUTPUT_WIDTH(CE_OUTPUT_WIDTH),
        .GUARD_BITS(7), 
        .ENABLE_ROUNDING(1),
        .HANDSHAKE_TIMEOUT(255) 
    ) u_compute_engine (
        .clk(clk), 
        .rst_n(rst_n), 
        .flush(1'b0),
        .input_valid(ce_input_valid), 
        .input_ready(ce_input_ready),
        .exp_X(ce_exp_X), 
        .mant_X_block(ce_mant_X),
        .exp_W_array(weight_exp_cached), 
        .mant_W_blocks(weight_mant_cached),
        .result_valids(ce_result_valids), 
        .result_ready(ce_result_ready),
        .result_fixed_array(ce_result_fixed_array),
        .result_base_exp_array(ce_result_base_exp_array),
        .result_zero_array(ce_result_zero_array)
    );

    multi_head_bfp_converter #(
        .NUM_HEADS(G_OUT), .RESULTS_PER_HEAD(T_OUT),
        .FIXED_WIDTH(CE_OUTPUT_WIDTH), .BASE_EXP_WIDTH(CE_BASE_EXP_WIDTH),
        .OUTPUT_MANT_WIDTH(MANT_WIDTH), .OUTPUT_EXP_WIDTH(EXP_WIDTH)
    ) u_multi_head_bfp_converter (
        .clk(clk), .rst_n(rst_n), .flush(1'b0),
        .input_valids({(G_OUT*T_OUT){ce_result_cached}}),
        .input_fixed_array(ce_result_fixed_buf),
        .input_base_exp_array(ce_result_base_exp_buf),
        .input_zero_array(ce_result_zero_buf),
        .output_valids(converter_output_valids),
        .output_mant_array(converter_output_mants),
        .output_shared_exps(converter_output_shared_exps),
        .output_overflow(converter_alignment_overflow)
    );
// =====================================================
// DEBUG: 在 STATE_WAIT_CE 阶段打印 CE 的 valid / mask 信息
// =====================================================
always @(posedge clk) begin
    if (rst_n && state == STATE_WAIT_CE) begin
        $display("[%0t][QKV WAIT_CE] ce_result_valids = %b", 
                 $time, ce_result_valids);
        $display("[%0t][QKV WAIT_CE] collected_mask   = %b", 
                 $time, collected_mask);
        $display("[%0t][QKV WAIT_CE] ce_result_ready  = %b", 
                 $time, ce_result_ready);
        $display("--------------------------------------------------");
    end
end

    //================================================================================
    // 主状态机 
    //================================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= STATE_IDLE;
            done               <= 1'b0;
            busy               <= 1'b0;
            error              <= 1'b0;
            token_counter      <= 10'd0;
            global_token_idx   <= 10'd0;
            batch_token_idx    <= 5'd0;
            head_counter       <= 2'd0;
            total_tokens       <= 10'd0;
            is_mode_q          <= 1'b0;
            is_mode_k          <= 1'b0;
            is_mode_v          <= 1'b0;
            weight_loaded_flags <= 3'b000;
            weight_exp_cached   <= {G_OUT*T_OUT*EXP_WIDTH{1'b0}};
            weight_mant_cached  <= {G_OUT*T_OUT*TOTAL_WIDTH{1'b0}};
            token_exp_cached    <= {EXP_WIDTH{1'b0}};
            token_mant_cached   <= {TOTAL_WIDTH{1'b0}};
            kv_written_count    <= 10'd0;
            all_kv_written      <= 1'b0;
            ce_input_valid      <= 1'b0;
            ce_exp_X            <= {EXP_WIDTH{1'b0}};
            ce_mant_X           <= {TOTAL_WIDTH{1'b0}};
            ce_result_ready     <= 1'b0;
            token_rd_en         <= 1'b0;
            token_rd_addr       <= 10'd0;
            weight_req          <= 1'b0;
            weight_type         <= 2'd0;
            storage_wr_en       <= 1'b0;
            storage_matrix_type <= 2'd0;
            storage_head_id     <= 2'd0;
            storage_token_id    <= 5'd0;
            storage_shared_exp  <= {EXP_WIDTH{1'b0}};
            storage_mant_packed <= {HEAD_DIM*DATA_WIDTH{1'b0}};
            storage_overflow_flag <= 1'b0;
            kv_wr_en            <= 1'b0;
            kv_wr_type          <= 1'b0;
            kv_wr_head          <= 2'd0;
            kv_wr_token         <= 10'd0;
            kv_wr_exp           <= {EXP_WIDTH{1'b0}};
            kv_wr_mant          <= {HEAD_DIM*DATA_WIDTH{1'b0}};
        end else begin
            case (state)
                STATE_IDLE: begin
                    done <= 1'b0;
                    busy <= 1'b0;
                    error <= 1'b0;
                    if (start) begin
                        busy  <= 1'b1;
                        is_mode_q <= (compute_mode == 2'b00);
                        is_mode_k <= (compute_mode == 2'b01);
                        is_mode_v <= (compute_mode == 2'b10);
                        token_counter   <= 10'd0;
                        batch_token_idx <= 5'd0;
                        if (compute_mode == 2'b00) begin
                            global_token_idx <= batch_id * TOKEN_BATCH;
                            total_tokens     <= tokens_in_batch;
                        end else begin
                            global_token_idx <= 10'd0;
                            total_tokens     <= TOKEN_NUM;
                        end
                        if (weight_loaded_flags[compute_mode]) begin
                            state <= STATE_LOAD_TOKEN;
                        end else begin
                            state <= STATE_REQ_WEIGHT;
                        end
                    end
                end
                
                STATE_REQ_WEIGHT: begin
                    if (!weight_req) begin
                        weight_req  <= 1'b1;
                        weight_type <= compute_mode;
                    end
                    if (weight_ack) begin
                        weight_req <= 1'b0;
                        state      <= STATE_WAIT_WEIGHT;
                    end
                end
                
                STATE_WAIT_WEIGHT: begin
                    if (weight_valid) begin
                        weight_exp_cached  <= weight_exp_array;
                        weight_mant_cached <= weight_mant_blocks;
                        weight_loaded_flags[compute_mode] <= 1'b1;
                        state              <= STATE_LOAD_TOKEN;
                    end
                end
                
                STATE_LOAD_TOKEN: begin
                    case (token_counter)
                        10'd0: begin
                            token_rd_en   <= 1'b1;
                            token_rd_addr <= global_token_idx;
                            token_counter <= token_counter + 1;
                        end
                        10'd1: begin
                            token_rd_en   <= 1'b0;
                            token_counter <= token_counter + 1;
                        end
                        10'd2: begin
                            token_exp_cached  <= token_rd_exp;
                            token_mant_cached <= token_rd_mant;
                            token_counter     <= 10'd0;
                            state             <= STATE_SEND_CE;
                        end
                    endcase
                end

                STATE_SEND_CE: begin
                    ce_input_valid <= 1'b1;
                    ce_exp_X       <= token_exp_cached;
                    ce_mant_X      <= token_mant_cached;
                    
                    if (ce_input_ready) begin
                        state <= STATE_WAIT_ACK;
                    end
                end

                STATE_WAIT_ACK: begin
                     ce_input_valid <= 1'b0; 
                     state <= STATE_WAIT_CE; 
                end
                
                STATE_WAIT_CE: begin
                    ce_input_valid <= 1'b0; 
                    if (&collected_mask) begin
                        ce_result_ready <= 1'b0;
                        state <= STATE_WAIT_CONVERTER;
                    end else begin
                        ce_result_ready <= 1'b1;
                    end
                end
                
                STATE_WAIT_CONVERTER: begin
                    ce_result_ready <= 1'b0;
                    if (|converter_output_valids) begin
                        state        <= STATE_SAVE_RESULT;
                        head_counter <= 2'd0;
                    end
                end
                
                STATE_SAVE_RESULT: begin
                    if (is_mode_q) begin
                        storage_wr_en         <= 1'b1;
                        storage_matrix_type   <= compute_mode;
                        storage_head_id       <= head_counter;
                        storage_token_id      <= batch_token_idx;
                        storage_shared_exp    <= converter_output_shared_exps[head_counter*EXP_WIDTH +: EXP_WIDTH];
                        storage_mant_packed   <= converter_output_mants[head_counter*T_OUT*MANT_WIDTH +: T_OUT*MANT_WIDTH];
                        storage_overflow_flag <= converter_alignment_overflow[head_counter];
                        if (head_counter < NUM_HEADS - 1) begin
                            head_counter <= head_counter + 1'b1;
                        end else begin
                            state <= STATE_CLEAR_WR;
                        end
                    end else begin
                        kv_wr_en    <= 1'b1;
                        kv_wr_type  <= compute_mode[0];
                        kv_wr_head  <= head_counter;
                        kv_wr_token <= global_token_idx;
                        kv_wr_exp   <= converter_output_shared_exps[head_counter*EXP_WIDTH +: EXP_WIDTH];
                        kv_wr_mant  <= converter_output_mants[head_counter*T_OUT*MANT_WIDTH +: T_OUT*MANT_WIDTH];
                        
                        if (head_counter < NUM_HEADS - 1) begin
                            head_counter <= head_counter + 1'b1;
                        end else begin
                            kv_written_count <= kv_written_count + 1'b1;
                            state <= STATE_CLEAR_WR;
                        end
                    end
                end
                
                STATE_CLEAR_WR: begin
                    storage_wr_en <= 1'b0;
                    kv_wr_en      <= 1'b0;
                    state         <= STATE_NEXT_TOKEN;
                end
                
                STATE_NEXT_TOKEN: begin
                    global_token_idx <= global_token_idx + 1'b1;
                    if (is_mode_q) begin
                        batch_token_idx <= batch_token_idx + 1'b1;
                    end
                    if (global_token_idx >= total_tokens - 1) begin
                        state <= STATE_DONE;
                    end else begin
                        state <= STATE_LOAD_TOKEN;
                    end
                end
                
                STATE_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= STATE_IDLE;
                    if (!is_mode_q && kv_written_count >= TOKEN_NUM) begin
                        all_kv_written <= 1'b1;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule