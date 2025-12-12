`timescale 1ns / 1ps

module meiga_control #(
    parameter NUM_LAYERS         = 4,    // Backbone层数
    parameter NUM_SIDENET_LAYERS = 5,    // Sidenet层数 (0-4)
    parameter ENABLE_LAYER_0     = 1,    // 层级使能控制
    parameter ENABLE_LAYER_1     = 1,
    parameter ENABLE_LAYER_2     = 1,
    parameter ENABLE_LAYER_3     = 1,
    parameter ENABLE_LAYER_4     = 1,
    parameter TIMEOUT_CYCLES     = 100000  // 超时周期数
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 系统控制接口
    //==========================================================================
    input  wire start,              // 启动信号（推理模式）
    output reg  done,               // 完成信号
    output reg  busy,               // 忙状态
    output reg  error,              // 错误标志
    
    //==========================================================================
    // 训练模式控制
    //==========================================================================
    input  wire train_mode,         // 训练模式标志（1=训练，0=推理）
    input  wire train_start_req,    // 训练启动请求
    output reg  train_start_pulse,  // 给DFA的训练启动脉冲
    input  wire train_done,         // DFA训练完成信号
    input  wire train_active,       // DFA训练活动标志
    
    //==========================================================================
    // 层级控制输出
    //==========================================================================
    output reg  [1:0] current_backbone_layer,   // 当前Backbone层 (0-3)
    output reg  [2:0] current_sidenet_layer,    // 当前Sidenet层 (0-4)
    
    //==========================================================================
    // 启动信号输出 (单周期脉冲)
    //==========================================================================
    output reg  backbone_start,     // Backbone启动脉冲
    output reg  sidenet_start,      // Sidenet启动脉冲
    
    //==========================================================================
    // 完成信号输入
    //==========================================================================
    input  wire backbone_done,      // Backbone完成信号
    input  wire backbone_busy,      // Backbone忙状态
    input  wire sidenet_done,       // Sidenet完成信号
    input  wire sidenet_busy,       // Sidenet忙状态
    
    //==========================================================================
    // Bank控制
    //==========================================================================
    output reg  bank_swap_pulse,    // Bank切换脉冲 (单周期)
    
    //==========================================================================
    // 加载/存储控制
    //==========================================================================
    output reg  load_input_start,   // 输入加载启动
    input  wire load_input_done,    // 输入加载完成
    output reg  store_output_start, // 输出存储启动
    input  wire store_output_done,  // 输出存储完成
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [3:0] dbg_state,              // 当前主状态
    output wire [1:0] dbg_current_layer,      // 当前层级 (0-3)
    output wire       dbg_backbone_done_latch, // 调试：backbone done锁存状态
    output wire       dbg_sidenet_done_latch   // 调试：sidenet done锁存状态
);

//================================================================================
// 状态定义
//================================================================================
// 主状态机
localparam STATE_IDLE         = 4'd0;   // 空闲状态
localparam STATE_LOAD_INPUT   = 4'd1;   // 加载输入数据
localparam STATE_WAIT_LOAD    = 4'd2;   // 等待加载完成
localparam STATE_LAYER_0      = 4'd3;   // Layer 0处理
localparam STATE_LAYER_1      = 4'd4;   // Layer 1处理
localparam STATE_LAYER_2      = 4'd5;   // Layer 2处理
localparam STATE_LAYER_3      = 4'd6;   // Layer 3处理
localparam STATE_LAYER_4      = 4'd7;   // Layer 4处理 (仅Sidenet)
localparam STATE_STORE_OUTPUT = 4'd8;   // 存储输出数据
localparam STATE_WAIT_STORE   = 4'd9;   // 等待存储完成
localparam STATE_DONE         = 4'd10;  // 完成状态
localparam STATE_TRAINING     = 4'd11;  // 训练状态（新增）
localparam STATE_ERROR        = 4'd15;  // 错误状态

// 层内子状态机 (用于Layer 0-3)
localparam LAYER_IDLE         = 3'd0;
localparam LAYER_START_BOTH   = 3'd1;
localparam LAYER_WAIT_BOTH    = 3'd2;
localparam LAYER_BANK_SWAP    = 3'd3;
localparam LAYER_DONE         = 3'd4;

// Layer 4子状态机
localparam LAYER4_IDLE        = 3'd0;
localparam LAYER4_START_SN    = 3'd1;
localparam LAYER4_WAIT_SN     = 3'd2;
localparam LAYER4_DONE        = 3'd3;

//================================================================================
// 内部信号
//================================================================================
reg [3:0] state, state_next;
reg [2:0] layer_substate, layer_substate_next;

//================================================================================
// Done信号内部锁存器
//================================================================================
reg backbone_done_latch;
reg sidenet_done_latch;
reg load_done_latch;
reg store_done_latch;

//================================================================================
// 超时计数器
//================================================================================
reg [31:0] timeout_counter;
reg timeout_error;

//================================================================================
// 调试输出
//================================================================================
assign dbg_state = state;
assign dbg_current_layer = current_backbone_layer;
assign dbg_backbone_done_latch = backbone_done_latch;
assign dbg_sidenet_done_latch = sidenet_done_latch;

//================================================================================
// Done信号锁存逻辑
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        backbone_done_latch <= 1'b0;
        sidenet_done_latch <= 1'b0;
        load_done_latch <= 1'b0;
        store_done_latch <= 1'b0;
    end else begin
        // Backbone done锁存
        if (backbone_start) begin
            backbone_done_latch <= 1'b0;
        end else if (backbone_done) begin
            backbone_done_latch <= 1'b1;
        end
        
        // Sidenet done锁存
        if (sidenet_start) begin
            sidenet_done_latch <= 1'b0;
        end else if (sidenet_done) begin
            sidenet_done_latch <= 1'b1;
        end
        
        // Load done锁存
        if (load_input_start) begin
            load_done_latch <= 1'b0;
        end else if (load_input_done) begin
            load_done_latch <= 1'b1;
        end
        
        // Store done锁存
        if (store_output_start) begin
            store_done_latch <= 1'b0;
        end else if (store_output_done) begin
            store_done_latch <= 1'b1;
        end
    end
end

//================================================================================
// 主状态机
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_IDLE;
        layer_substate <= LAYER_IDLE;
    end else begin
        state <= state_next;
        layer_substate <= layer_substate_next;
    end
end

//================================================================================
// 主状态机组合逻辑
//================================================================================
always @(*) begin
    state_next = state;
    layer_substate_next = layer_substate;
    
    case (state)
        //--------------------------------------------------------------------
        // IDLE: 等待启动信号
        //--------------------------------------------------------------------
        STATE_IDLE: begin
            if (train_start_req && train_mode) begin
                // 训练模式启动
                state_next = STATE_TRAINING;
            end else if (start && !train_mode) begin
                // 推理模式启动
                state_next = STATE_LOAD_INPUT;
            end
        end
        
        //--------------------------------------------------------------------
        // TRAINING: 训练状态（新增）
        //--------------------------------------------------------------------
        STATE_TRAINING: begin
            if (train_done) begin
                state_next = STATE_IDLE;
            end else if (timeout_error) begin
                state_next = STATE_ERROR;
            end
        end
        
        //--------------------------------------------------------------------
        // LOAD_INPUT: 加载输入数据
        //--------------------------------------------------------------------
        STATE_LOAD_INPUT: begin
            state_next = STATE_WAIT_LOAD;
        end
        
        STATE_WAIT_LOAD: begin
            if (load_done_latch) begin
                if (ENABLE_LAYER_0) begin
                    state_next = STATE_LAYER_0;
                    layer_substate_next = LAYER_IDLE;
                end else begin
                    state_next = STATE_DONE;
                end
            end else if (timeout_error) begin
                state_next = STATE_ERROR;
            end
        end
        
        //--------------------------------------------------------------------
        // LAYER_0: Layer 0处理
        //--------------------------------------------------------------------
        STATE_LAYER_0: begin
            case (layer_substate)
                LAYER_IDLE: begin
                    layer_substate_next = LAYER_START_BOTH;
                end
                
                LAYER_START_BOTH: begin
                    layer_substate_next = LAYER_WAIT_BOTH;
                end
                
                LAYER_WAIT_BOTH: begin
                    if (backbone_done_latch && sidenet_done_latch) begin
                        layer_substate_next = LAYER_BANK_SWAP;
                    end else if (timeout_error) begin
                        state_next = STATE_ERROR;
                        layer_substate_next = LAYER_IDLE;
                    end
                end
                
                LAYER_BANK_SWAP: begin
                    layer_substate_next = LAYER_DONE;
                end
                
                LAYER_DONE: begin
                    if (ENABLE_LAYER_1)
                        state_next = STATE_LAYER_1;
                    else if (ENABLE_LAYER_4)
                        state_next = STATE_LAYER_4;
                    else
                        state_next = STATE_STORE_OUTPUT;
                    layer_substate_next = LAYER_IDLE;
                end
                
                default: layer_substate_next = LAYER_IDLE;
            endcase
        end
        
        //--------------------------------------------------------------------
        // LAYER_1: Layer 1处理
        //--------------------------------------------------------------------
        STATE_LAYER_1: begin
            case (layer_substate)
                LAYER_IDLE: begin
                    layer_substate_next = LAYER_START_BOTH;
                end
                
                LAYER_START_BOTH: begin
                    layer_substate_next = LAYER_WAIT_BOTH;
                end
                
                LAYER_WAIT_BOTH: begin
                    if (backbone_done_latch && sidenet_done_latch) begin
                        layer_substate_next = LAYER_BANK_SWAP;
                    end else if (timeout_error) begin
                        state_next = STATE_ERROR;
                        layer_substate_next = LAYER_IDLE;
                    end
                end
                
                LAYER_BANK_SWAP: begin
                    layer_substate_next = LAYER_DONE;
                end
                
                LAYER_DONE: begin
                    if (ENABLE_LAYER_2)
                        state_next = STATE_LAYER_2;
                    else if (ENABLE_LAYER_4)
                        state_next = STATE_LAYER_4;
                    else
                        state_next = STATE_STORE_OUTPUT;
                    layer_substate_next = LAYER_IDLE;
                end
                
                default: layer_substate_next = LAYER_IDLE;
            endcase
        end
        
        //--------------------------------------------------------------------
        // LAYER_2: Layer 2处理
        //--------------------------------------------------------------------
        STATE_LAYER_2: begin
            case (layer_substate)
                LAYER_IDLE: begin
                    layer_substate_next = LAYER_START_BOTH;
                end
                
                LAYER_START_BOTH: begin
                    layer_substate_next = LAYER_WAIT_BOTH;
                end
                
                LAYER_WAIT_BOTH: begin
                    if (backbone_done_latch && sidenet_done_latch) begin
                        layer_substate_next = LAYER_BANK_SWAP;
                    end else if (timeout_error) begin
                        state_next = STATE_ERROR;
                        layer_substate_next = LAYER_IDLE;
                    end
                end
                
                LAYER_BANK_SWAP: begin
                    layer_substate_next = LAYER_DONE;
                end
                
                LAYER_DONE: begin
                    if (ENABLE_LAYER_3)
                        state_next = STATE_LAYER_3;
                    else if (ENABLE_LAYER_4)
                        state_next = STATE_LAYER_4;
                    else
                        state_next = STATE_STORE_OUTPUT;
                    layer_substate_next = LAYER_IDLE;
                end
                
                default: layer_substate_next = LAYER_IDLE;
            endcase
        end
        
        //--------------------------------------------------------------------
        // LAYER_3: Layer 3处理
        //--------------------------------------------------------------------
        STATE_LAYER_3: begin
            case (layer_substate)
                LAYER_IDLE: begin
                    layer_substate_next = LAYER_START_BOTH;
                end
                
                LAYER_START_BOTH: begin
                    layer_substate_next = LAYER_WAIT_BOTH;
                end
                
                LAYER_WAIT_BOTH: begin
                    if (backbone_done_latch && sidenet_done_latch) begin
                        layer_substate_next = LAYER_BANK_SWAP;
                    end else if (timeout_error) begin
                        state_next = STATE_ERROR;
                        layer_substate_next = LAYER_IDLE;
                    end
                end
                
                LAYER_BANK_SWAP: begin
                    layer_substate_next = LAYER_DONE;
                end
                
                LAYER_DONE: begin
                    if (ENABLE_LAYER_4)
                        state_next = STATE_LAYER_4;
                    else
                        state_next = STATE_STORE_OUTPUT;
                    layer_substate_next = LAYER_IDLE;
                end
                
                default: layer_substate_next = LAYER_IDLE;
            endcase
        end
        
        //--------------------------------------------------------------------
        // LAYER_4: Layer 4处理 (仅Sidenet)
        //--------------------------------------------------------------------
        STATE_LAYER_4: begin
            case (layer_substate)
                LAYER4_IDLE: begin
                    layer_substate_next = LAYER4_START_SN;
                end
                
                LAYER4_START_SN: begin
                    layer_substate_next = LAYER4_WAIT_SN;
                end
                
                LAYER4_WAIT_SN: begin
                    if (sidenet_done_latch) begin
                        layer_substate_next = LAYER4_DONE;
                    end else if (timeout_error) begin
                        state_next = STATE_ERROR;
                        layer_substate_next = LAYER4_IDLE;
                    end
                end
                
                LAYER4_DONE: begin
                    state_next = STATE_STORE_OUTPUT;
                    layer_substate_next = LAYER4_IDLE;
                end
                
                default: layer_substate_next = LAYER4_IDLE;
            endcase
        end
        
        //--------------------------------------------------------------------
        // STORE_OUTPUT: 存储输出
        //--------------------------------------------------------------------
        STATE_STORE_OUTPUT: begin
            state_next = STATE_WAIT_STORE;
        end
        
        STATE_WAIT_STORE: begin
            if (store_done_latch) begin
                state_next = STATE_DONE;
            end else if (timeout_error) begin
                state_next = STATE_ERROR;
            end
        end
        
        //--------------------------------------------------------------------
        // DONE: 完成状态
        //--------------------------------------------------------------------
        STATE_DONE: begin
            state_next = STATE_IDLE;
        end
        
        //--------------------------------------------------------------------
        // ERROR: 错误状态
        //--------------------------------------------------------------------
        STATE_ERROR: begin
            // 保持在错误状态，直到复位
            state_next = STATE_ERROR;
        end
        
        default: begin
            state_next = STATE_IDLE;
        end
    endcase
end

//================================================================================
// 控制信号生成
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        backbone_start <= 1'b0;
        sidenet_start <= 1'b0;
        bank_swap_pulse <= 1'b0;
        load_input_start <= 1'b0;
        store_output_start <= 1'b0;
        train_start_pulse <= 1'b0;
        current_backbone_layer <= 2'b0;
        current_sidenet_layer <= 3'b0;
        done <= 1'b0;
        busy <= 1'b0;
        error <= 1'b0;
    end else begin
        // 默认：所有脉冲信号清零
        backbone_start <= 1'b0;
        sidenet_start <= 1'b0;
        bank_swap_pulse <= 1'b0;
        load_input_start <= 1'b0;
        store_output_start <= 1'b0;
        train_start_pulse <= 1'b0;
        done <= 1'b0;
        
        case (state)
            STATE_IDLE: begin
                busy <= 1'b0;
            end
            
            //----------------------------------------------------------------
            // 训练状态控制（新增）
            //----------------------------------------------------------------
            STATE_TRAINING: begin
                busy <= 1'b1;
                // 进入训练状态时发出train_start脉冲
                if (state != state_next) begin
                    train_start_pulse <= 1'b1;
                end
            end
            
            //----------------------------------------------------------------
            // 推理状态控制
            //----------------------------------------------------------------
            STATE_LOAD_INPUT: begin
                busy <= 1'b1;
                load_input_start <= 1'b1;
            end
            
            STATE_WAIT_LOAD: begin
                busy <= 1'b1;
            end
            
            STATE_LAYER_0: begin
                busy <= 1'b1;
                current_backbone_layer <= 2'd0;
                current_sidenet_layer <= 3'd0;
                
                case (layer_substate)
                    LAYER_START_BOTH: begin
                        backbone_start <= 1'b1;
                        sidenet_start <= 1'b1;
                    end
                    
                    LAYER_BANK_SWAP: begin
                        bank_swap_pulse <= 1'b1;
                    end
                endcase
            end
            
            STATE_LAYER_1: begin
                busy <= 1'b1;
                current_backbone_layer <= 2'd1;
                current_sidenet_layer <= 3'd1;
                
                case (layer_substate)
                    LAYER_START_BOTH: begin
                        backbone_start <= 1'b1;
                        sidenet_start <= 1'b1;
                    end
                    
                    LAYER_BANK_SWAP: begin
                        bank_swap_pulse <= 1'b1;
                    end
                endcase
            end
            
            STATE_LAYER_2: begin
                busy <= 1'b1;
                current_backbone_layer <= 2'd2;
                current_sidenet_layer <= 3'd2;
                
                case (layer_substate)
                    LAYER_START_BOTH: begin
                        backbone_start <= 1'b1;
                        sidenet_start <= 1'b1;
                    end
                    
                    LAYER_BANK_SWAP: begin
                        bank_swap_pulse <= 1'b1;
                    end
                endcase
            end
            
            STATE_LAYER_3: begin
                busy <= 1'b1;
                current_backbone_layer <= 2'd3;
                current_sidenet_layer <= 3'd3;
                
                case (layer_substate)
                    LAYER_START_BOTH: begin
                        backbone_start <= 1'b1;
                        sidenet_start <= 1'b1;
                    end
                    
                    LAYER_BANK_SWAP: begin
                        bank_swap_pulse <= 1'b1;
                    end
                endcase
            end
            
            STATE_LAYER_4: begin
                busy <= 1'b1;
                current_sidenet_layer <= 3'd4;
                
                case (layer_substate)
                    LAYER4_START_SN: begin
                        sidenet_start <= 1'b1;
                    end
                endcase
            end
            
            STATE_STORE_OUTPUT: begin
                busy <= 1'b1;
                store_output_start <= 1'b1;
            end
            
            STATE_WAIT_STORE: begin
                busy <= 1'b1;
            end
            
            STATE_DONE: begin
                busy <= 1'b0;
                done <= 1'b1;
            end
            
            STATE_ERROR: begin
                busy <= 1'b0;
                error <= 1'b1;
            end
        endcase
    end
end

//================================================================================
// 超时检测
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        timeout_counter <= 32'b0;
        timeout_error <= 1'b0;
    end else begin
        if (state == STATE_IDLE || state == STATE_DONE || state == STATE_ERROR) begin
            timeout_counter <= 32'b0;
            timeout_error <= 1'b0;
        end else begin
            if (timeout_counter >= TIMEOUT_CYCLES) begin
                timeout_error <= 1'b1;
            end else begin
                timeout_counter <= timeout_counter + 1;
            end
        end
    end
end

endmodule