`timescale 1ns / 1ps

//==============================================================
// Compute Engine - v9.2 (Final Fix)
// 
// Fix Log:
// 1. [Compilation] Fixed "handshake_timeout is not declared" error.
// 2. [Deadlock] Fixed handshake slip in ST_WAIT_READY by setting input_done immediately.
// 3. [Config] Restored 32-element support (INT8x16 packer).
//==============================================================

module compute_engine #(
    // ========== Matrix Dimensions ==========
    parameter G_OUT = 4,               
    parameter T_OUT = 8,               
    
    // ========== PE Configuration ==========
    parameter NUM_PE = 2,              
    parameter PE_TYPE_0 = 0,           // 0 = INT8x16 (Matches your 128-bit Packer)
    parameter PE_TYPE_1 = 0,           // 0 = INT8x16
    
    // ========== Data Widths ==========
    parameter EXP_WIDTH = 8,           
    parameter INPUT_MANT_WIDTH = 8,    
    
    // ========== Vector Dimensions ==========
    parameter ELEM_PE0 = 16,           // 16 elements per PE
    parameter ELEM_PE1 = 16,            
    parameter TOTAL_ELEM = 32,         // Total 32 elements (16+16)
    
    // ========== Optimization ==========
    parameter INTERNAL_WIDTH = 39,     
    parameter OUTPUT_WIDTH = 32,       
    parameter GUARD_BITS = 7,          
    parameter ENABLE_ROUNDING = 1,     
    
    // ========== Timeout ==========
    parameter HANDSHAKE_TIMEOUT = 255  
)(
    input  wire clk,
    input  wire rst_n,
    input  wire flush,
    
    // ========== Input Handshake ==========
    input  wire input_valid,
    output wire input_ready,
    
    // ========== Input Data ==========
    input  wire [EXP_WIDTH-1:0] exp_X,
    input  wire [TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_X_block,
    
    input  wire [G_OUT*T_OUT*EXP_WIDTH-1:0] exp_W_array,
    input  wire [G_OUT*T_OUT*TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_W_blocks,
    
    // ========== Output Handshake ==========
    output wire [G_OUT*T_OUT-1:0] result_valids,
    input  wire result_ready,
    
    // ========== Output Data ==========
    output wire signed [G_OUT*T_OUT*OUTPUT_WIDTH-1:0] result_fixed_array,
    output wire [G_OUT*T_OUT*(EXP_WIDTH+1)-1:0] result_base_exp_array,
    output wire [G_OUT*T_OUT-1:0] result_zero_array
);
    //==============================================================
    // Local Parameters
    //==============================================================
    localparam TOTAL_WIDTH = TOTAL_ELEM * INPUT_MANT_WIDTH;
    localparam TOTAL_PUS = G_OUT * T_OUT;

    //==============================================================
    // Global Ready Signal
    //==============================================================
    wire [G_OUT-1:0] col_ready_array;
    assign input_ready = &col_ready_array;

    //==============================================================
    // PU Connection Arrays
    //==============================================================
    wire [TOTAL_PUS-1:0] pu_input_valid_array;
    wire [TOTAL_PUS-1:0] pu_input_ready_array;
    wire [TOTAL_PUS-1:0] pu_result_valid_array;
    wire [TOTAL_PUS-1:0] pu_result_ready_array;

    wire signed [TOTAL_PUS*OUTPUT_WIDTH-1:0] pu_result_fixed_array;
    wire [TOTAL_PUS*(EXP_WIDTH+1)-1:0] pu_result_base_exp_array;
    wire [TOTAL_PUS-1:0] pu_result_zero_array;

    //==============================================================
    // Column Control Logic
    //==============================================================
    genvar g_col;
    generate
        for (g_col = 0; g_col < G_OUT; g_col = g_col + 1) begin : columns
            
            // Cache Registers
            reg [TOTAL_WIDTH-1:0] cached_mant_X;
            reg [EXP_WIDTH-1:0]   cached_exp_X;
            reg [T_OUT*EXP_WIDTH-1:0]   cached_exp_W;
            reg [T_OUT*TOTAL_WIDTH-1:0] cached_mant_W;
            
            reg cache_valid;

            // FSM States
            localparam ST_IDLE       = 3'b000;
            localparam ST_WAIT_READY = 3'b010;
            localparam ST_PROCESSING = 3'b011;
            localparam ST_SWITCHING  = 3'b100;
            
            reg [2:0] state;
            reg [$clog2(T_OUT)-1:0] active_pu_idx;
            reg input_done;
            reg [$clog2(HANDSHAKE_TIMEOUT+1)-1:0] handshake_timeout_cnt;
            
            // [FIX 1] Define the missing wire
            wire handshake_timeout = (handshake_timeout_cnt >= HANDSHAKE_TIMEOUT);

            assign col_ready_array[g_col] = (state == ST_IDLE);

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    state <= ST_IDLE;
                    cache_valid <= 0;
                    active_pu_idx <= 0;
                    input_done <= 0;
                    handshake_timeout_cnt <= 0;
                    cached_mant_X <= 0;
                    cached_exp_X <= 0;
                    cached_exp_W <= 0; cached_mant_W <= 0;
                end else if (flush) begin
                    state <= ST_IDLE;
                    cache_valid <= 0;
                    active_pu_idx <= 0;
                    input_done <= 0;
                    handshake_timeout_cnt <= 0;
                end else begin
                    case (state)
                        ST_IDLE: begin
                            if (input_valid && input_ready) begin
                                cached_exp_X  <= exp_X;
                                cached_mant_X <= mant_X_block;
                                cached_exp_W  <= exp_W_array[ g_col * (T_OUT*EXP_WIDTH) +: (T_OUT*EXP_WIDTH) ];
                                cached_mant_W <= mant_W_blocks[ g_col * (T_OUT*TOTAL_WIDTH) +: (T_OUT*TOTAL_WIDTH) ];
                                cache_valid <= 1;
                                active_pu_idx <= 0;
                                input_done <= 0;
                                handshake_timeout_cnt <= 0;
                                state <= ST_WAIT_READY;
                            end
                        end
                        
                        // [修改 1]：不立即置 input_done，确保 Valid 信号维持
                        ST_WAIT_READY: begin
                            if (pu_input_ready_array[g_col*T_OUT + active_pu_idx]) begin
                                state <= ST_PROCESSING;
                                input_done <= 0; // 保持为 0，让 Valid 继续拉高
                                handshake_timeout_cnt <= 0;
                            end else if (handshake_timeout) begin
                                // 超时保护
                                state <= ST_IDLE;
                                cache_valid <= 0;
                                handshake_timeout_cnt <= 0;
                            end else begin
                                handshake_timeout_cnt <= handshake_timeout_cnt + 1;
                            end
                        end
                        
                        ST_PROCESSING: begin
                            // [修改 2]：检测到 PU 变忙 (Ready变低) 后，才认为输入完成
                            if (!input_done) begin
                                if (!pu_input_ready_array[g_col*T_OUT + active_pu_idx]) begin
                                    // PU 已经变忙，握手成功，现在可以撤销 Valid
                                    input_done <= 1;
                                    handshake_timeout_cnt <= 0; 
                                end else begin
                                    // PU 还没变忙，继续计数，防止 PU 彻底挂死
                                    handshake_timeout_cnt <= handshake_timeout_cnt + 1;
                                    
                                    // 如果在这里超时，说明 Valid 送了很久 PU 也没反应，强制退出
                                    if (handshake_timeout) begin
                                        state <= ST_IDLE;
                                        cache_valid <= 0;
                                    end
                                end
                            end else begin
                                // input_done == 1，正常等待结果
                                if (result_valids[g_col*T_OUT + active_pu_idx] && result_ready) begin
                                    state <= ST_SWITCHING;
                                    input_done <= 0;
                                    handshake_timeout_cnt <= 0;
                                    if (active_pu_idx < T_OUT-1) begin
                                        active_pu_idx <= active_pu_idx + 1;
                                    end else begin
                                        active_pu_idx <= 0;
                                    end
                                end
                            end
                        end
                        
                        ST_SWITCHING: begin
                            if (active_pu_idx == 0) begin
                                state <= ST_IDLE;
                                cache_valid <= 0;
                            end else begin
                                state <= ST_WAIT_READY;
                            end
                        end
                        
                        default: state <= ST_IDLE;
                    endcase
                end
            end

            // Instantiate PUs
            genvar t_pu;
            for (t_pu = 0; t_pu < T_OUT; t_pu = t_pu + 1) begin : pu_pipeline
                localparam PU_IDX = g_col * T_OUT + t_pu;
                
                wire [EXP_WIDTH-1:0]   pu_exp_W;
                wire [TOTAL_WIDTH-1:0] pu_mant_W;
                assign pu_exp_W  = cached_exp_W[t_pu*EXP_WIDTH +: EXP_WIDTH];
                assign pu_mant_W = cached_mant_W[t_pu*TOTAL_WIDTH +: TOTAL_WIDTH];

                wire this_pu_active;
                assign this_pu_active = cache_valid && (active_pu_idx == t_pu);
                
                // Drop Valid immediately when input_done is high to prevent double-triggering
                assign pu_input_valid_array[PU_IDX] = this_pu_active && 
                                                      ((state == ST_PROCESSING) || (state == ST_WAIT_READY)) && 
                                                      !input_done;
                                                      
                assign pu_result_ready_array[PU_IDX] = result_ready;
                assign result_valids[PU_IDX] = pu_result_valid_array[PU_IDX];
                
                PU #(
                    .NUM_PE(NUM_PE),
                    .PE_TYPE_0(PE_TYPE_0),
                    .PE_TYPE_1(PE_TYPE_1),
                    .EXP_WIDTH(EXP_WIDTH),
                    .INPUT_MANT_WIDTH(INPUT_MANT_WIDTH),
                    .ELEM_PE0(ELEM_PE0),
                    .ELEM_PE1(ELEM_PE1),
                    .TOTAL_ELEM(TOTAL_ELEM),
                    .INTERNAL_WIDTH(INTERNAL_WIDTH),
                    .OUTPUT_WIDTH(OUTPUT_WIDTH),
                    .GUARD_BITS(GUARD_BITS),
                    .ENABLE_ROUNDING(ENABLE_ROUNDING)
                ) u_pu (
                    .clk(clk),
                    .rst_n(rst_n),
                    .flush(flush),
                    .input_valid(pu_input_valid_array[PU_IDX]),
                    .input_ready(pu_input_ready_array[PU_IDX]),
                    .exp_X(cached_exp_X),
                    .mant_X_block(cached_mant_X),
                    .exp_W(pu_exp_W),
                    .mant_W_block(pu_mant_W),
                    .result_valid(pu_result_valid_array[PU_IDX]),
                    .result_ready(pu_result_ready_array[PU_IDX]),
                    .result_fixed(pu_result_fixed_array[(PU_IDX+1)*OUTPUT_WIDTH-1 : PU_IDX*OUTPUT_WIDTH]),
                    .result_base_exp(pu_result_base_exp_array[(PU_IDX+1)*(EXP_WIDTH+1)-1 : PU_IDX*(EXP_WIDTH+1)]),
                    .result_zero(pu_result_zero_array[PU_IDX])
                );
            end
        end
    endgenerate

    // Output Connection
    assign result_fixed_array = pu_result_fixed_array;
    assign result_base_exp_array = pu_result_base_exp_array;
    assign result_zero_array = pu_result_zero_array;

    initial begin
        $display("========================================");
        $display("Compute Engine v9.2 - Handshake Fixed");
        $display("========================================");
    end

endmodule