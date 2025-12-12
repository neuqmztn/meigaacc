module attention_batch_controller #(
    parameter NUM_BATCHES = 20,
    parameter NUM_HEADS   = 4,
    parameter TOKEN_BATCH = 32
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 顶层控制接口
    //===========================================================================
    input  wire start,
    output reg  done,
    output reg  busy,
    
    //===========================================================================
    // QKV计算引擎控制
    //===========================================================================
    output reg  qkv_start,
    output reg  [1:0] qkv_compute_mode,
    output reg  [5:0] qkv_batch_id,
    output reg  [4:0] qkv_tokens_in_batch,
    input  wire qkv_done,
    input  wire qkv_busy,
    
    //===========================================================================
    // 4个Head引擎控制
    //===========================================================================
    output reg  [NUM_HEADS-1:0] heads_start,
    input  wire [NUM_HEADS-1:0] heads_done,
    input  wire [NUM_HEADS-1:0] heads_busy,
    
    //===========================================================================
    // Output Projection控制
    //===========================================================================
    output reg  output_proj_start,
    input  wire output_proj_done,
    input  wire output_proj_busy,
    
    //===========================================================================
    // 状态输出
    //===========================================================================
    output reg  [6:0] current_batch,
    output reg  first_batch_flag,
    
    //===========================================================================
    // 调试接口
    //===========================================================================
    output reg  [3:0] current_state,
    output reg  [31:0] cycle_count,
    output reg  [31:0] batch_cycle_count
);

//===================================================================================
// 状态机定义
//===================================================================================

localparam IDLE              = 4'd0;
localparam COMPUTE_Q         = 4'd1;
localparam WAIT_Q            = 4'd2;
localparam COMPUTE_K         = 4'd3;
localparam WAIT_K            = 4'd4;
localparam COMPUTE_V         = 4'd5;
localparam WAIT_V            = 4'd6;
localparam START_HEADS       = 4'd7;
localparam WAIT_HEADS        = 4'd8;
localparam START_OUTPUT_PROJ = 4'd9;
localparam WAIT_OUTPUT_PROJ  = 4'd10;
localparam NEXT_BATCH        = 4'd11;
localparam DONE_ST           = 4'd12;

reg [3:0] state;
reg [4:0] batch_counter;

//===================================================================================
// ✅ Done信号锁存寄存器（核心改进）
//===================================================================================

reg qkv_done_latch;                      // 锁存QKV完成状态
reg [NUM_HEADS-1:0] heads_done_latch;    // 锁存每个Head完成状态
reg output_proj_done_latch;              // 锁存Output Projection完成状态

//===================================================================================
// ✅ QKV done锁存逻辑
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        qkv_done_latch <= 1'b0;
    end else begin
        if (qkv_start) begin
            // 启动新计算时清除锁存
            qkv_done_latch <= 1'b0;
        end else if (qkv_done) begin
            // 检测到done信号，立即锁存
            qkv_done_latch <= 1'b1;
        end
        // 否则保持当前值
    end
end

//===================================================================================
// ✅ Heads done锁存逻辑（每个head独立）
//===================================================================================

genvar i;
generate
    for (i = 0; i < NUM_HEADS; i = i + 1) begin : head_done_latch_gen
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                heads_done_latch[i] <= 1'b0;
            end else begin
                if (heads_start[i]) begin
                    // 启动时清除锁存
                    heads_done_latch[i] <= 1'b0;
                end else if (heads_done[i]) begin
                    // 检测到done，立即锁存
                    heads_done_latch[i] <= 1'b1;
                end
            end
        end
    end
endgenerate

//===================================================================================
// ✅ Output Projection done锁存逻辑
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        output_proj_done_latch <= 1'b0;
    end else begin
        if (output_proj_start) begin
            output_proj_done_latch <= 1'b0;
        end else if (output_proj_done) begin
            output_proj_done_latch <= 1'b1;
        end
    end
end

//===================================================================================
// Batch计数器和标志
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_batch <= 5'd0;
        first_batch_flag <= 1'b0;
    end else begin
        current_batch <= batch_counter;
        first_batch_flag <= (batch_counter == 5'd0);
    end
end

//===================================================================================
// 周期计数器
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cycle_count <= 32'd0;
        batch_cycle_count <= 32'd0;
    end else if (busy) begin
        cycle_count <= cycle_count + 1'b1;
        
        if (state == COMPUTE_Q && batch_cycle_count != 32'd0) begin
            batch_cycle_count <= 32'd0;
        end else begin
            batch_cycle_count <= batch_cycle_count + 1'b1;
        end
    end
end

//===================================================================================
// 主状态机 - 使用锁存的done信号
//===================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        current_state <= 4'd0;
        batch_counter <= 5'd0;
        done <= 1'b0;
        busy <= 1'b0;
        
        qkv_start <= 1'b0;
        qkv_compute_mode <= 2'b00;
        qkv_batch_id <= 6'd0;
        qkv_tokens_in_batch <= 5'd0;
        
        heads_start <= {NUM_HEADS{1'b0}};
        output_proj_start <= 1'b0;
        
    end else begin
        current_state <= state;
        qkv_start <= 1'b0;
        heads_start <= {NUM_HEADS{1'b0}};
        output_proj_start <= 1'b0;
        
        case (state)
            //===================================================================
            // IDLE: 等待启动
            //===================================================================
            IDLE: begin
                done <= 1'b0;
                
                if (start) begin
                    busy <= 1'b1;
                    batch_counter <= 5'd0;
                    state <= COMPUTE_Q;
                    
                    $display("========================================");
                    $display("[%0t] Batch Controller v2.2: Started", $time);
                    $display("  With done signal latching mechanism");
                    $display("========================================");
                end else begin
                    busy <= 1'b0;
                end
            end
            
            //===================================================================
            // COMPUTE_Q: 启动Q矩阵计算
            //===================================================================
            COMPUTE_Q: begin
                qkv_start <= 1'b1;
                qkv_compute_mode <= 2'b00;
                qkv_batch_id <= {1'b0, batch_counter};
                
                if (batch_counter < NUM_BATCHES - 1) begin
                    qkv_tokens_in_batch <= TOKEN_BATCH;
                end else begin
                    qkv_tokens_in_batch <= 5'd1;
                end
                
                state <= WAIT_Q;
                $display("[%0t] Batch %0d: Start Q computation", $time, batch_counter);
            end
            
            //===================================================================
            // WAIT_Q: 等待Q计算完成（使用锁存信号）
            //===================================================================
            WAIT_Q: begin
                if (qkv_done_latch) begin  // ✅ 使用锁存信号
                    if (batch_counter == 5'd0) begin
                        state <= COMPUTE_K;
                        $display("[%0t] Batch %0d: Q done → Computing K", $time, batch_counter);
                    end else begin
                        state <= START_HEADS;
                        $display("[%0t] Batch %0d: Q done → Starting heads", $time, batch_counter);
                    end
                end
            end
            
            //===================================================================
            // COMPUTE_K: 启动K矩阵计算
            //===================================================================
            COMPUTE_K: begin
                qkv_start <= 1'b1;
                qkv_compute_mode <= 2'b01;
                qkv_batch_id <= 6'd0;
                qkv_tokens_in_batch <= 5'd0;
                
                state <= WAIT_K;
                $display("[%0t] Batch 0: Start K computation (all tokens)", $time);
            end
            
            //===================================================================
            // WAIT_K: 等待K计算完成（使用锁存信号）
            //===================================================================
            WAIT_K: begin
                if (qkv_done_latch) begin  // ✅ 使用锁存信号
                    state <= COMPUTE_V;
                    $display("[%0t] Batch 0: K done → Computing V", $time);
                end
            end
            
            //===================================================================
            // COMPUTE_V: 启动V矩阵计算
            //===================================================================
            COMPUTE_V: begin
                qkv_start <= 1'b1;
                qkv_compute_mode <= 2'b10;
                qkv_batch_id <= 6'd0;
                qkv_tokens_in_batch <= 5'd0;
                
                state <= WAIT_V;
                $display("[%0t] Batch 0: Start V computation (all tokens)", $time);
            end
            
            //===================================================================
            // WAIT_V: 等待V计算完成（使用锁存信号）
            //===================================================================
            WAIT_V: begin
                if (qkv_done_latch) begin  // ✅ 使用锁存信号
                    state <= START_HEADS;
                    $display("[%0t] Batch 0: V done → K/V ready", $time);
                end
            end
            
            //===================================================================
            // START_HEADS: 启动所有Head引擎
            //===================================================================
            START_HEADS: begin
                heads_start <= {NUM_HEADS{1'b1}};
                state <= WAIT_HEADS;
                $display("[%0t] Batch %0d: Starting %0d heads", $time, batch_counter, NUM_HEADS);
            end
            
            //===================================================================
            // WAIT_HEADS: 等待所有Head完成（使用锁存信号）
            //===================================================================
            WAIT_HEADS: begin
                if (heads_done_latch == {NUM_HEADS{1'b1}}) begin  // ✅ 使用锁存信号
                    state <= START_OUTPUT_PROJ;
                    $display("[%0t] Batch %0d: All heads done", $time, batch_counter);
                end
                
                // 调试：打印各个head的状态
                if (cycle_count[7:0] == 8'd0) begin
                    $display("[%0t]   Head status: done=%04b latch=%04b busy=%04b",
                             $time, heads_done, heads_done_latch, heads_busy);
                end
            end
            
            //===================================================================
            // START_OUTPUT_PROJ: 启动Output Projection
            //===================================================================
            START_OUTPUT_PROJ: begin
                output_proj_start <= 1'b1;
                state <= WAIT_OUTPUT_PROJ;
                $display("[%0t] Batch %0d: Starting output projection", $time, batch_counter);
            end
            
            //===================================================================
            // WAIT_OUTPUT_PROJ: 等待Output Projection完成（使用锁存信号）
            //===================================================================
            WAIT_OUTPUT_PROJ: begin
                if (output_proj_done_latch) begin  // ✅ 使用锁存信号
                    state <= NEXT_BATCH;
                    $display("[%0t] Batch %0d: Output proj done", $time, batch_counter);
                    $display("[%0t]   Batch cycles: %0d\n", $time, batch_cycle_count);
                end
            end
            
            //===================================================================
            // NEXT_BATCH: 移到下一个batch
            //===================================================================
            NEXT_BATCH: begin
                if (batch_counter < NUM_BATCHES - 1) begin
                    batch_counter <= batch_counter + 5'd1;
                    state <= COMPUTE_Q;
                end else begin
                    state <= DONE_ST;
                end
            end
            
            //===================================================================
            // DONE: 全部完成
            //===================================================================
            DONE_ST: begin
                done <= 1'b1;
                busy <= 1'b0;
                state <= IDLE;
                
                $display("========================================");
                $display("[%0t] ALL BATCHES COMPLETED!", $time);
                $display("  Total cycles: %0d", cycle_count);
                $display("  Average cycles/batch: %0d", cycle_count / NUM_BATCHES);
                $display("========================================");
            end
            
            default: state <= IDLE;
        endcase
    end
end


endmodule