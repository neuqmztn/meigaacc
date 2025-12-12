`timescale 1ns / 1ps
module linear_compute_engine #(
    parameter TOKEN_CHUNK   = 32,
    parameter INPUT_DIM     = 32,
    parameter OUTPUT_DIM    = 32,
 
    parameter BFP_EXP_W     = 8,
    parameter BFP_MANT_W    = 8,     
    parameter OUTPUT_MANT_W = 16,     
    
    parameter G_OUT         = 4,
    parameter T_OUT         = 8,
    parameter CE_OUTPUT_WIDTH = 32,

    parameter NUM_PE_PER_GROUP = 1,
    parameter PE_TYPE_0     = 2,      // PE_A/PE_B/PE_C/PE_D
    parameter PE_TYPE_1     = 2,
    parameter ELEM_PE0      = 8,
    parameter ELEM_PE1      = 8
)(
    input  wire clk,
    input  wire rst_n,
    
    // 控制信号
    input  wire start,
    output reg  done,
    output reg  busy,
    
    //================================================================================
    // 输入数据接口 (8-bit)
    //================================================================================
    input  wire [TOKEN_CHUNK*BFP_EXP_W-1:0]                 x_exp,
    input  wire [TOKEN_CHUNK*INPUT_DIM*BFP_MANT_W-1:0]      x_mant,
    
    // W: 权重矩阵 [INPUT_DIM × OUTPUT_DIM]
    input  wire [OUTPUT_DIM*BFP_EXP_W-1:0]                  w_exp_array,  
    input  wire [INPUT_DIM*OUTPUT_DIM*BFP_MANT_W-1:0]       w_mant,
    
    //================================================================================
    // 输出数据
    //================================================================================
    output reg  [TOKEN_CHUNK*BFP_EXP_W-1:0]                 y_exp,
    output reg  [TOKEN_CHUNK*OUTPUT_DIM*OUTPUT_MANT_W-1:0]  y_mant
);

    //============================================================================
    // 内部参数与状态机定义
    //============================================================================
    localparam TOTAL_TOKENS = TOKEN_CHUNK;
    localparam TOTAL_ELEM   = INPUT_DIM; 
    localparam TOTAL_RESULTS = OUTPUT_DIM; 

    // 状态机状态
    localparam S_IDLE    = 4'd0;
    localparam S_LOAD    = 4'd1;
    localparam S_EXTRACT = 4'd2;
    localparam S_SEND    = 4'd3;
    localparam S_WAIT    = 4'd4;
    localparam S_SAVE    = 4'd5;
    localparam S_CONV    = 4'd6;
    localparam S_DONE    = 4'd7;
    
    reg [3:0] state;
    reg [5:0] token_cnt; // 0~31

    reg [TOKEN_CHUNK*BFP_EXP_W-1:0]            x_exp_buf;
    reg [TOKEN_CHUNK*INPUT_DIM*BFP_MANT_W-1:0] x_mant_buf;
    reg [OUTPUT_DIM*BFP_EXP_W-1:0]             w_exp_buf;
    reg [INPUT_DIM*OUTPUT_DIM*BFP_MANT_W-1:0]  w_mant_buf;

    //============================================================================
    // Compute Engine 接口信号
    //============================================================================
    reg                         ce_input_valid;
    wire                        ce_input_ready;
    
    reg  [BFP_EXP_W-1:0]        ce_exp_X;
    reg  [INPUT_DIM*BFP_MANT_W-1:0] ce_mant_X;
    
    reg  [OUTPUT_DIM*BFP_EXP_W-1:0]             ce_exp_W;
    reg  [INPUT_DIM*OUTPUT_DIM*BFP_MANT_W-1:0]  ce_mant_W;

    wire [G_OUT*T_OUT-1:0]                       ce_result_valids;
    wire signed [G_OUT*T_OUT*CE_OUTPUT_WIDTH-1:0] ce_result_fixed_array;
    wire [G_OUT*T_OUT*(BFP_EXP_W+1)-1:0]        ce_result_base_exp_array;
    wire [G_OUT*T_OUT-1:0]                      ce_result_zero_array;

    //============================================================================
    // BFP Converter 接口信号
    //============================================================================
    wire [TOTAL_RESULTS-1:0]                       bfp_valids;
    wire signed [TOTAL_RESULTS*OUTPUT_MANT_W-1:0]  bfp_mant_array;
    wire [BFP_EXP_W-1:0]                           bfp_shared_exp;
    wire                                           bfp_overflow_unused;

    reg  [OUTPUT_DIM*OUTPUT_MANT_W-1:0] conv_mants;
    reg  [BFP_EXP_W-1:0]                conv_shared_exp;

    reg [TOKEN_CHUNK*OUTPUT_DIM*OUTPUT_MANT_W-1:0] res_buf_mant;
    reg [TOKEN_CHUNK*BFP_EXP_W-1:0]                res_buf_exp;

    //================================================================================
    // 1. Compute Engine
    //================================================================================
    compute_engine #(
        .G_OUT(G_OUT),
        .T_OUT(T_OUT),

        .NUM_PE(NUM_PE_PER_GROUP),
        .PE_TYPE_0(PE_TYPE_0),
        .PE_TYPE_1(PE_TYPE_1),

        .EXP_WIDTH(BFP_EXP_W),
        .INPUT_MANT_WIDTH(BFP_MANT_W),

        .ELEM_PE0(ELEM_PE0),
        .ELEM_PE1(ELEM_PE1),
        .TOTAL_ELEM(TOTAL_ELEM),

        .OUTPUT_WIDTH(CE_OUTPUT_WIDTH)
    ) u_compute_engine (
        .clk   (clk),
        .rst_n (rst_n),
        .flush (1'b0),

        .input_valid (ce_input_valid),
        .input_ready (ce_input_ready),

        .exp_X          (ce_exp_X),
        .mant_X_block   (ce_mant_X),
        .exp_W_array    (w_exp_buf),
        .mant_W_blocks  (w_mant_buf),

        .result_valids        (ce_result_valids),
        .result_ready         (1'b1),  
        .result_fixed_array   (ce_result_fixed_array),
        .result_base_exp_array(ce_result_base_exp_array),
        .result_zero_array    (ce_result_zero_array)
    );

    //================================================================================
    // 2. BFP Converter
    //================================================================================
    bfp_converter #(
        .TOTAL_RESULTS    (TOTAL_RESULTS),          
        .FIXED_WIDTH      (CE_OUTPUT_WIDTH),         
        .BASE_EXP_WIDTH   (BFP_EXP_W + 1),         
        .OUTPUT_MANT_WIDTH(OUTPUT_MANT_W),          
        .OUTPUT_EXP_WIDTH (BFP_EXP_W)                
    ) u_bfp_converter (
        .clk              (clk),
        .rst_n            (rst_n),
        .flush            (1'b0),

        .input_valids     (ce_result_valids),
        .input_fixed_array(ce_result_fixed_array),
        .input_base_exp_array(ce_result_base_exp_array),
        .input_zero_array (ce_result_zero_array),

        .output_valids    (bfp_valids),
        .output_mant_array(bfp_mant_array),
        .output_shared_exp(bfp_shared_exp),
        .output_overflow  (bfp_overflow_unused)
    );

    //================================================================================
    // 3. 状态机
    //================================================================================
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            token_cnt       <= 0;
            busy            <= 0;
            done            <= 0;

            ce_input_valid  <= 0;
            
            x_exp_buf       <= 0;
            x_mant_buf      <= 0;
            w_exp_buf       <= 0;
            w_mant_buf      <= 0;
            
            res_buf_mant    <= 0;
            res_buf_exp     <= 0;
            y_exp           <= 0;
            y_mant          <= 0;

            conv_mants      <= 0;
            conv_shared_exp <= 0;
        end else begin
            done           <= 1'b0;
            ce_input_valid <= 1'b0;

            case (state)
                //============================================================
                // IDLE: 等待 start
                //============================================================
                S_IDLE: begin
                    busy      <= 0;
                    token_cnt <= 0;
                    if (start) begin
                        busy  <= 1;
                        state <= S_LOAD;
                    end
                end

                //============================================================
                // LOAD: 缓存输入 X / W
                //============================================================
                S_LOAD: begin
                    x_exp_buf  <= x_exp;        
                    x_mant_buf <= x_mant;       
                    w_exp_buf  <= w_exp_array;
                    w_mant_buf <= w_mant;
                    state      <= S_EXTRACT;
                end

                //============================================================
                // EXTRACT: 提取当前 token 的一行输入
                //============================================================
                S_EXTRACT: begin
                    ce_exp_X  <= x_exp_buf[token_cnt*BFP_EXP_W +: BFP_EXP_W];
                    ce_mant_X <= x_mant_buf[token_cnt*INPUT_DIM*BFP_MANT_W +: INPUT_DIM*BFP_MANT_W];
                    state <= S_SEND;
                end

                //============================================================
                // SEND: 向 CE 发送一行数据（input_valid / input_ready 握手）
                //============================================================
                S_SEND: begin
                    if (ce_input_ready) begin
                        ce_input_valid <= 1'b1;   // 单拍 valid
                        state          <= S_WAIT;
                    end
                end

                //============================================================
                // WAIT: 等待 CE + BFP Converter 输出一整行结果
                //============================================================
                S_WAIT: begin
                    if (|bfp_valids) begin
                        conv_mants      <= bfp_mant_array;
                        conv_shared_exp <= bfp_shared_exp;
                        state           <= S_SAVE;
                    end
                end

                //============================================================
                // SAVE: 写回当前 token 的输出到结果缓存
                //============================================================
                S_SAVE: begin
                    res_buf_mant[token_cnt*OUTPUT_DIM*OUTPUT_MANT_W +: OUTPUT_DIM*OUTPUT_MANT_W]
                        <= conv_mants;
                    res_buf_exp[token_cnt*BFP_EXP_W +: BFP_EXP_W]
                        <= conv_shared_exp;
                    
                    if (token_cnt == TOTAL_TOKENS-1) begin
                        state <= S_CONV;
                    end else begin
                        token_cnt <= token_cnt + 1;
                        state     <= S_EXTRACT;
                    end
                end

                //============================================================
                // CONV: 所有 token 计算完成，准备整体输出
                //============================================================
                S_CONV: begin
                    state <= S_DONE;
                end

                //============================================================
                // DONE: 一次块计算完成，输出结果
                //============================================================
                S_DONE: begin
                    done   <= 1;
                    busy   <= 0;
                    y_exp  <= res_buf_exp;
                    y_mant <= res_buf_mant;
                    
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
