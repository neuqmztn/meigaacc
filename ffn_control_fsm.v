`timescale 1ns / 1ps

module ffn_control_fsm #(
    parameter TOKEN_NUM       = 641,
    parameter TOKEN_CHUNK     = 32,
    parameter D_MODEL         = 32,
    parameter D_FF            = 128,
    parameter FEATURE_CHUNK   = 32
)(
    input  wire clk,
    input  wire rst_n,
    
    // 控制接口
    input  wire start,
    output reg  done,
    output reg  busy,
    
    // 握手完成信号
    input  wire token_load_done,
    input  wire weight_load_done,
    input  wire h_act_wr_done,
    input  wire h_act_rd_done,
    input  wire result_wr_done,
    
    // 索引输出
    output reg  [4:0] token_batch_id,
    output reg  [1:0] feature_chunk_id,
    output reg  [1:0] stage,
    
    // Memory Manager 控制信号 (电平信号/握手信号)
    output reg  load_token_block,
    output reg  load_w1_chunk,
    output reg  load_w2_chunk,
    output reg  load_h_act_chunk,
    output reg  save_h_act_chunk,
    output reg  save_result,
    
    // 计算引擎控制信号 (脉冲信号)
    output reg  linear1_start,
    input  wire linear1_done,
    
    output reg  gelu_start,
    input  wire gelu_done,
    
    output reg  linear2_start,
    input  wire linear2_done,
    
    // Linear2 累加器控制信号 (脉冲信号)
    output reg  acc_clear,
    output reg  acc_enable,
    input  wire acc_valid,
    
    // 调试输出
    output wire [31:0] cycle_count,
    output wire [3:0]  fsm_state
);

//================================================================================
// 本地参数
//================================================================================
localparam TOKEN_BATCHES  = (TOKEN_NUM + TOKEN_CHUNK - 1) / TOKEN_CHUNK;
localparam FEATURE_CHUNKS = (D_FF + FEATURE_CHUNK - 1) / FEATURE_CHUNK;

// 状态定义
localparam STATE_IDLE             = 4'd0;
localparam STATE_LOAD_TOKEN       = 4'd1;
// Linear1
localparam STATE_LOAD_W1          = 4'd2;
localparam STATE_LINEAR1_COMPUTE  = 4'd3;
localparam STATE_GELU_COMPUTE     = 4'd4;
localparam STATE_SAVE_H_ACT       = 4'd5;
// Linear2
localparam STATE_CLEAR_ACCUM      = 4'd6;
localparam STATE_LOAD_H_ACT       = 4'd7;
localparam STATE_LOAD_W2          = 4'd8;
localparam STATE_LINEAR2_COMPUTE  = 4'd9;
localparam STATE_ACCUMULATE       = 4'd10;
localparam STATE_WAIT_ACC_DONE    = 4'd11;
// End
localparam STATE_SAVE_RESULT      = 4'd12;
localparam STATE_DONE             = 4'd13;

//================================================================================
// 内部信号
//================================================================================
reg [3:0] state;
reg [4:0] token_count;
reg [1:0] feature_count;
reg [31:0] cycle_counter;

//================================================================================
// 主状态机
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_IDLE;
        done <= 1'b0;
        busy <= 1'b0;
        token_batch_id <= 5'd0;
        feature_chunk_id <= 2'd0;
        stage <= 2'd0;
        token_count <= 5'd0;
        feature_count <= 2'd0;
        cycle_counter <= 32'd0;
        
        // 所有输出信号复位
        load_token_block <= 1'b0;
        load_w1_chunk <= 1'b0;
        load_w2_chunk <= 1'b0;
        load_h_act_chunk <= 1'b0;
        save_h_act_chunk <= 1'b0;
        save_result <= 1'b0;
        linear1_start <= 1'b0;
        gelu_start <= 1'b0;
        linear2_start <= 1'b0;
        acc_clear <= 1'b0;
        acc_enable <= 1'b0;
        
    end else begin
        // 周期计数
        if (busy) cycle_counter <= cycle_counter + 1;
        
        acc_clear     <= 1'b0; 
        acc_enable    <= 1'b0;
        // linear1/2/gelu_start 使用握手逻辑(等待done)，所以保持原有逻辑，不在此处默认置0

        case (state)
            
            STATE_IDLE: begin
                done <= 1'b0;
                cycle_counter <= 32'd0;
                if (start) begin
                    busy <= 1'b1;
                    token_count <= 5'd0;
                    feature_count <= 2'd0;
                    state <= STATE_LOAD_TOKEN;
                end
            end
            
            STATE_LOAD_TOKEN: begin
                token_batch_id <= token_count;
                if (!load_token_block) begin
                    load_token_block <= 1'b1;
                end else if (token_load_done) begin
                    load_token_block <= 1'b0;
                    feature_count <= 2'd0;
                    stage <= 2'd0; // Linear1
                    state <= STATE_LOAD_W1;
                end
            end
            
            // --- Linear1 Loop ---
            STATE_LOAD_W1: begin
                feature_chunk_id <= feature_count;
                if (!load_w1_chunk) begin
                    load_w1_chunk <= 1'b1;
                end else if (weight_load_done) begin
                    load_w1_chunk <= 1'b0;
                    state <= STATE_LINEAR1_COMPUTE;
                end
            end
            
            STATE_LINEAR1_COMPUTE: begin
                if (!linear1_start) begin
                    linear1_start <= 1'b1;
                end else begin
                    linear1_start <= 1'b0; // 手动拉低，产生脉冲
                    if (linear1_done) state <= STATE_GELU_COMPUTE;
                end
            end
            
            STATE_GELU_COMPUTE: begin
                if (!gelu_start) begin
                    gelu_start <= 1'b1;
                end else begin
                    gelu_start <= 1'b0;
                    if (gelu_done) state <= STATE_SAVE_H_ACT;
                end
            end
            
            STATE_SAVE_H_ACT: begin
                if (!save_h_act_chunk) begin
                    save_h_act_chunk <= 1'b1;
                end else if (h_act_wr_done) begin
                    save_h_act_chunk <= 1'b0;
                    if (feature_count < FEATURE_CHUNKS - 1) begin
                        feature_count <= feature_count + 1;
                        state <= STATE_LOAD_W1;
                    end else begin
                        feature_count <= 2'd0;
                        stage <= 2'd2; // Linear2
                        state <= STATE_CLEAR_ACCUM;
                    end
                end
            end
            
            // --- Linear2 Loop ---
            STATE_CLEAR_ACCUM: begin
                // 关键修改：只置 1。因为顶部有默认置 0，下一周期自动变 0
                acc_clear <= 1'b1; 
                state <= STATE_LOAD_H_ACT;
            end
            
            STATE_LOAD_H_ACT: begin
                feature_chunk_id <= feature_count;
                if (!load_h_act_chunk) begin
                    load_h_act_chunk <= 1'b1;
                end else if (h_act_rd_done) begin
                    load_h_act_chunk <= 1'b0;
                    state <= STATE_LOAD_W2;
                end
            end
            
            STATE_LOAD_W2: begin
                if (!load_w2_chunk) begin
                    load_w2_chunk <= 1'b1;
                end else if (weight_load_done) begin
                    load_w2_chunk <= 1'b0;
                    state <= STATE_LINEAR2_COMPUTE;
                end
            end
            
            STATE_LINEAR2_COMPUTE: begin
                if (!linear2_start) begin
                    linear2_start <= 1'b1;
                end else begin
                    linear2_start <= 1'b0;
                    if (linear2_done) state <= STATE_ACCUMULATE;
                end
            end
            
            STATE_ACCUMULATE: begin
                // 关键修改：只置 1。因为顶部有默认置 0，下一周期自动变 0
                acc_enable <= 1'b1; 
                
                if (feature_count < FEATURE_CHUNKS - 1) begin
                    feature_count <= feature_count + 1;
                    state <= STATE_LOAD_H_ACT;
                end else begin
                    state <= STATE_WAIT_ACC_DONE;
                end
            end
            
            STATE_WAIT_ACC_DONE: begin
                if (acc_valid) state <= STATE_SAVE_RESULT;
            end
            
            // --- Save Result ---
            STATE_SAVE_RESULT: begin
                if (!save_result) begin
                    save_result <= 1'b1;
                end else if (result_wr_done) begin
                    save_result <= 1'b0;
                    if (token_count < TOKEN_BATCHES - 1) begin
                        token_count <= token_count + 1;
                        state <= STATE_LOAD_TOKEN;
                    end else begin
                        state <= STATE_DONE;
                    end
                end
            end
            
            STATE_DONE: begin
                done <= 1'b1;
                busy <= 1'b0;
                if (!start) begin
                    state <= STATE_IDLE;
                    done <= 1'b0;
                end
            end
            
            default: state <= STATE_IDLE;
        endcase
    end
end

// 输出赋值
assign cycle_count = cycle_counter;
assign fsm_state = state;

endmodule