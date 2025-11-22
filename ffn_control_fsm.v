`timescale 1ns / 1ps

//================================================================================
// FFN Control FSM - 四维分块调度器
//
// 功能：协调 FFN 各阶段的执行和数据流
//
// 四维分块：
// 1. Token Batch 维度：641 tokens → 21 blocks × 32 tokens
// 2. Feature Chunk 维度：128 hidden → 4 blocks × 32
// 3. Stage 维度：Linear1 → GELU → Linear2
// 4. Accumulation 维度：Linear2 需要4次累加
//
// 控制流程（每个 token batch）：
// 1. 加载Token块
// 2. Linear1阶段（循环4次feature chunks）
//    - 加载W1_chunk
//    - 计算 H_chunk
//    - GELU
//    - 保存 H_act_chunk
// 3. Linear2阶段（循环4次feature chunks，累加）
//    - 清空累加器（第一次）
//    - 加载 H_act_chunk
//    - 加载W2_chunk
//    - 计算 Y_partial
//    - 累加
// 4. 保存结果
//
//================================================================================

module ffn_control_fsm #(
    parameter TOKEN_NUM     = 641,
    parameter TOKEN_CHUNK   = 32,
    parameter D_MODEL       = 32,
    parameter D_FF          = 128,
    parameter FEATURE_CHUNK = 32
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
    // 握手完成信号（来自顶层状态机）
    //================================================================================
    input  wire token_load_done,        // Token 加载完成
    input  wire weight_load_done,       // 权重加载完成
    input  wire h_act_wr_done,          // H_act 写入完成
    input  wire h_act_rd_done,          // H_act 读取完成
    input  wire result_wr_done,         // 结果写回完成,
    
    //================================================================================
    // 四维分块索引输出
    //================================================================================
    output reg  [4:0] token_batch_id,    // 0-20 (21 batches)
    output reg  [1:0] feature_chunk_id,  // 0-3  (4 chunks)
    output reg  [1:0] stage,             // 0=Linear1, 1=GELU, 2=Linear2
    
    //================================================================================
    // Memory Manager 控制信号
    //================================================================================
    output reg  load_token_block,        // 加载Token块
    output reg  load_w1_chunk,           // 加载W1权重块
    output reg  load_w2_chunk,           // 加载W2权重块
    output reg  load_h_act_chunk,        // 加载中间结果块
    output reg  save_h_act_chunk,        // 保存中间结果块
    output reg  save_result,             // 保存最终结果
    
    //================================================================================
    // 计算引擎控制信号
    //================================================================================
    output reg  linear1_start,
    input  wire linear1_done,
    
    output reg  gelu_start,
    input  wire gelu_done,
    
    output reg  linear2_start,
    input  wire linear2_done,
    
    //================================================================================
    // Linear2 累加器控制信号
    //================================================================================
    output reg  acc_clear,               // 清空累加器
    output reg  acc_enable,              // 累加使能
    input  wire acc_valid,               // 累加完成标志
    
    //================================================================================
    // 调试输出
    //================================================================================
    output wire [31:0] cycle_count,
    output wire [3:0]  fsm_state
);

//================================================================================
// 本地参数
//================================================================================

localparam TOKEN_BATCHES  = (TOKEN_NUM + TOKEN_CHUNK - 1) / TOKEN_CHUNK;  // 21
localparam FEATURE_CHUNKS = (D_FF + FEATURE_CHUNK - 1) / FEATURE_CHUNK;   // 4

// 状态定义
localparam STATE_IDLE             = 4'd0;
localparam STATE_LOAD_TOKEN       = 4'd1;

// Linear1 阶段
localparam STATE_LOAD_W1          = 4'd2;
localparam STATE_LINEAR1_COMPUTE  = 4'd3;
localparam STATE_GELU_COMPUTE     = 4'd4;
localparam STATE_SAVE_H_ACT       = 4'd5;

// Linear2 阶段
localparam STATE_CLEAR_ACCUM      = 4'd6;
localparam STATE_LOAD_H_ACT       = 4'd7;
localparam STATE_LOAD_W2          = 4'd8;
localparam STATE_LINEAR2_COMPUTE  = 4'd9;
localparam STATE_ACCUMULATE       = 4'd10;
localparam STATE_WAIT_ACC_DONE    = 4'd11;

// 结束阶段
localparam STATE_SAVE_RESULT      = 4'd12;
localparam STATE_DONE             = 4'd13;

//================================================================================
// 内部信号
//================================================================================

// 状态机
reg [3:0] state;

// 计数器
reg [4:0] token_count;      // 当前token batch (0-20)
reg [1:0] feature_count;    // 当前feature chunk (0-3)

// 周期计数器
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
        
        // 控制信号初始化
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
        if (busy) begin
            cycle_counter <= cycle_counter + 1;
        end
        
        case (state)
            
            //================================================================
            // IDLE: 等待启动
            //================================================================
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
            
            //================================================================
            // LOAD_TOKEN: 加载Token块（每个batch一次）
            //================================================================
            STATE_LOAD_TOKEN: begin
                token_batch_id <= token_count;
                
                if (!load_token_block) begin
                    // 发起加载请求
                    load_token_block <= 1'b1;
                end else begin
                    // 已发起请求，等待完成
                    // token_load_done 信号由顶层模块提供
                    // 当32个token全部加载完成后置1
                    if (token_load_done) begin
                        load_token_block <= 1'b0;
                        feature_count <= 2'd0;  // 重置feature计数
                        stage <= 2'd0;          // Linear1阶段
                        state <= STATE_LOAD_W1;
                    end
                end
            end
            
            //================================================================
            // Linear1 阶段（循环4次）
            //================================================================
            STATE_LOAD_W1: begin
                feature_chunk_id <= feature_count;
                
                if (!load_w1_chunk) begin
                    load_w1_chunk <= 1'b1;
                end else begin
                    // 等待权重加载完成
                    if (weight_load_done) begin
                        load_w1_chunk <= 1'b0;
                        state <= STATE_LINEAR1_COMPUTE;
                    end
                end
            end
            
            STATE_LINEAR1_COMPUTE: begin
                if (!linear1_start) begin
                    linear1_start <= 1'b1;
                end else begin
                    linear1_start <= 1'b0;
                    
                    if (linear1_done) begin
                        state <= STATE_GELU_COMPUTE;
                    end
                end
            end
            
            STATE_GELU_COMPUTE: begin
                if (!gelu_start) begin
                    gelu_start <= 1'b1;
                end else begin
                    gelu_start <= 1'b0;
                    
                    if (gelu_done) begin
                        state <= STATE_SAVE_H_ACT;
                    end
                end
            end
            
            STATE_SAVE_H_ACT: begin
                if (!save_h_act_chunk) begin
                    save_h_act_chunk <= 1'b1;
                end else begin
                    // 等待 H_act 写入完成
                    if (h_act_wr_done) begin
                        save_h_act_chunk <= 1'b0;
                        
                        // 检查是否完成所有feature chunks
                        if (feature_count < FEATURE_CHUNKS - 1) begin
                            feature_count <= feature_count + 1;
                            state <= STATE_LOAD_W1;  // 下一个feature chunk
                        end else begin
                            // Linear1+GELU完成，开始Linear2
                            feature_count <= 2'd0;
                            stage <= 2'd2;  // Linear2阶段
                            state <= STATE_CLEAR_ACCUM;
                        end
                    end
                end
            end
            
            //================================================================
            // Linear2 阶段（循环4次，需要累加）
            //================================================================
            STATE_CLEAR_ACCUM: begin
                acc_clear <= 1'b1;
                
                acc_clear <= 1'b0;
                state <= STATE_LOAD_H_ACT;
            end
            
            STATE_LOAD_H_ACT: begin
                feature_chunk_id <= feature_count;
                
                if (!load_h_act_chunk) begin
                    load_h_act_chunk <= 1'b1;
                end else begin
                    // 等待 H_act 读取完成
                    if (h_act_rd_done) begin
                        load_h_act_chunk <= 1'b0;
                        state <= STATE_LOAD_W2;
                    end
                end
            end
            
            STATE_LOAD_W2: begin
                if (!load_w2_chunk) begin
                    load_w2_chunk <= 1'b1;
                end else begin
                    // 等待权重加载完成
                    if (weight_load_done) begin
                        load_w2_chunk <= 1'b0;
                        state <= STATE_LINEAR2_COMPUTE;
                    end
                end
            end
            
            STATE_LINEAR2_COMPUTE: begin
                if (!linear2_start) begin
                    linear2_start <= 1'b1;
                end else begin
                    linear2_start <= 1'b0;
                    
                    if (linear2_done) begin
                        state <= STATE_ACCUMULATE;
                    end
                end
            end
            
            STATE_ACCUMULATE: begin
                acc_enable <= 1'b1;
                
                acc_enable <= 1'b0;
                
                // 检查是否完成所有feature chunks
                if (feature_count < FEATURE_CHUNKS - 1) begin
                    feature_count <= feature_count + 1;
                    state <= STATE_LOAD_H_ACT;  // 下一个feature chunk
                end else begin
                    // 等待累加完成
                    state <= STATE_WAIT_ACC_DONE;
                end
            end
            
            STATE_WAIT_ACC_DONE: begin
                // 等待累加器输出有效信号
                if (acc_valid) begin
                    state <= STATE_SAVE_RESULT;
                end
            end
            
            //================================================================
            // 保存结果，检查是否完成所有token batches
            //================================================================
            STATE_SAVE_RESULT: begin
                if (!save_result) begin
                    save_result <= 1'b1;
                end else begin
                    // 等待结果写回完成
                    if (result_wr_done) begin
                        save_result <= 1'b0;
                        
                        // 检查是否完成所有token batches
                        if (token_count < TOKEN_BATCHES - 1) begin
                            token_count <= token_count + 1;
                            state <= STATE_LOAD_TOKEN;  // 下一个token batch
                        end else begin
                            state <= STATE_DONE;
                        end
                    end
                end
            end
            
            //================================================================
            // 完成
            //================================================================
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

//================================================================================
// 输出赋值
//================================================================================

assign cycle_count = cycle_counter;
assign fsm_state = state;

//================================================================================
// 调试信号（可选）
//================================================================================

`ifdef SIMULATION
always @(posedge clk) begin
    if (state == STATE_LOAD_TOKEN) begin
        $display("[%0t] FFN FSM: Loading Token Batch %0d/%0d", 
                 $time, token_count, TOKEN_BATCHES);
    end
    
    if (state == STATE_LOAD_W1) begin
        $display("[%0t] FFN FSM: Linear1 - Feature Chunk %0d/%0d", 
                 $time, feature_count, FEATURE_CHUNKS);
    end
    
    if (state == STATE_CLEAR_ACCUM) begin
        $display("[%0t] FFN FSM: Linear2 - Clearing Accumulator", $time);
    end
    
    if (state == STATE_LOAD_H_ACT) begin
        $display("[%0t] FFN FSM: Linear2 - Feature Chunk %0d/%0d", 
                 $time, feature_count, FEATURE_CHUNKS);
    end
    
    if (state == STATE_WAIT_ACC_DONE) begin
        $display("[%0t] FFN FSM: Waiting for accumulator completion", $time);
    end
    
    if (state == STATE_SAVE_RESULT) begin
        $display("[%0t] FFN FSM: Saving result for Token Batch %0d", 
                 $time, token_count);
    end
    
    if (state == STATE_DONE) begin
        $display("[%0t] FFN FSM: All %0d token batches processed in %0d cycles", 
                 $time, TOKEN_BATCHES, cycle_counter);
    end
end
`endif

endmodule