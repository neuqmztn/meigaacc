//================================================================================
// Module: meiga_control (Improved Version with Internal Latching)
// Description: MEIGA系统顶层控制FSM - 改进版
//              管理Backbone和Sidenet的层级处理、Bank切换和数据流控制
// 
// Author: MEIGA Design Team
// Date: 2025-11-12
// Version: 2.0 (Added internal done signal latching)
// Standard: Verilog 2001
//
// Key Improvements:
//   - 内部锁存done信号，兼容脉冲和电平两种信号类型
//   - 增强的鲁棒性，不依赖外部模块的done信号保持时长
//   - 更清晰的接口文档和时序说明
//   - 实现了超时错误检测机制
//
// Architecture:
//   - 主状态机: 管理系统级流程(加载、层级、存储)
//   - 层内子状态机: 管理每层的Backbone和Sidenet并行执行
//   - Done信号锁存器: 捕获外部done脉冲/电平，确保可靠检测
//   - 控制信号生成: 产生所有模块的启动和控制信号
//
// Features:
//   - 支持4层Backbone + 5层Sidenet的协调调度
//   - Backbone和Sidenet并行执行
//   - 自动Bank切换管理
//   - 可配置的层级使能控制
//   - 超时错误检测
//   - 调试接口支持
//================================================================================

module meiga_control #(
    parameter NUM_LAYERS         = 4,    // Backbone层数
    parameter NUM_SIDENET_LAYERS = 5,    // Sidenet层数 (0-4)
    parameter ENABLE_LAYER_0     = 1,    // 层级使能控制
    parameter ENABLE_LAYER_1     = 1,
    parameter ENABLE_LAYER_2     = 1,
    parameter ENABLE_LAYER_3     = 1,
    parameter ENABLE_LAYER_4     = 1,
    parameter TIMEOUT_CYCLES     = 100000  // 超时周期数 (可配置)
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 系统控制接口
    //==========================================================================
    input  wire start,              // 启动信号 (脉冲或电平均可)
    output reg  done,               // 完成信号
    output reg  busy,               // 忙状态
    output reg  error,              // 错误标志
    
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
    // 接口协议说明：
    //   - done信号可以是脉冲(单周期)或电平(多周期保持)
    //   - 本模块内部会锁存done信号，确保可靠检测
    //   - 建议外部模块在完成后保持done为高直到下一个start信号
    //   - busy信号用于指示模块工作状态
    //==========================================================================
    input  wire backbone_done,      // Backbone完成信号 (脉冲/电平均可)
    input  wire backbone_busy,      // Backbone忙状态
    input  wire sidenet_done,       // Sidenet完成信号 (脉冲/电平均可)
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
localparam STATE_ERROR        = 4'd15;  // 错误状态

// 层内子状态机 (用于Layer 0-3)
localparam LAYER_IDLE         = 3'd0;   // 层空闲
localparam LAYER_START_BOTH   = 3'd1;   // 启动Backbone和Sidenet
localparam LAYER_WAIT_BOTH    = 3'd2;   // 等待两者完成
localparam LAYER_BANK_SWAP    = 3'd3;   // 执行Bank切换
localparam LAYER_DONE         = 3'd4;   // 层完成

// Layer 4子状态机 (特殊：仅Sidenet)
localparam LAYER4_IDLE        = 3'd0;   // Layer 4空闲
localparam LAYER4_START_SN    = 3'd1;   // 启动Sidenet
localparam LAYER4_WAIT_SN     = 3'd2;   // 等待Sidenet完成
localparam LAYER4_DONE        = 3'd3;   // Layer 4完成

//================================================================================
// 内部信号
//================================================================================
reg [3:0] state, state_next;               // 主状态机
reg [2:0] layer_substate, layer_substate_next;  // 层内子状态机

//================================================================================
// Done信号内部锁存器 - 核心改进
// 设计思想：
//   1. 捕获外部done信号（无论是脉冲还是电平）
//   2. 在层开始时清除锁存
//   3. 检测到done信号后保持锁存状态
//   4. 确保FSM能够可靠地检测到完成条件
//================================================================================
reg backbone_done_latch;    // Backbone完成信号锁存
reg sidenet_done_latch;     // Sidenet完成信号锁存
reg load_done_latch;        // Load完成信号锁存
reg store_done_latch;       // Store完成信号锁存

//================================================================================
// 超时计数器 - 用于错误检测
//================================================================================
reg [31:0] timeout_counter;
reg timeout_error;  // 超时错误标志

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
        //----------------------------------------------------------------------
        // Backbone Done 锁存控制
        //----------------------------------------------------------------------
        case (state)
            // 在Layer 0-3的START_BOTH状态清除锁存
            STATE_LAYER_0, STATE_LAYER_1, STATE_LAYER_2, STATE_LAYER_3: begin
                if (layer_substate == LAYER_START_BOTH) begin
                    backbone_done_latch <= 1'b0;  // 新层开始，清除锁存
                end else if (backbone_done) begin
                    backbone_done_latch <= 1'b1;  // 捕获done信号
                end
                // 否则保持当前值
            end
            
            // 其他状态下保持或清除
            STATE_IDLE: begin
                backbone_done_latch <= 1'b0;
            end
            
            default: begin
                if (backbone_done) begin
                    backbone_done_latch <= 1'b1;
                end
            end
        endcase
        
        //----------------------------------------------------------------------
        // Sidenet Done 锁存控制
        //----------------------------------------------------------------------
        case (state)
            // 在Layer 0-3的START_BOTH状态清除锁存
            STATE_LAYER_0, STATE_LAYER_1, STATE_LAYER_2, STATE_LAYER_3: begin
                if (layer_substate == LAYER_START_BOTH) begin
                    sidenet_done_latch <= 1'b0;  // 新层开始，清除锁存
                end else if (sidenet_done) begin
                    sidenet_done_latch <= 1'b1;  // 捕获done信号
                end
            end
            
            // Layer 4只有Sidenet
            STATE_LAYER_4: begin
                if (layer_substate == LAYER4_START_SN) begin
                    sidenet_done_latch <= 1'b0;
                end else if (sidenet_done) begin
                    sidenet_done_latch <= 1'b1;
                end
            end
            
            // 其他状态下保持或清除
            STATE_IDLE: begin
                sidenet_done_latch <= 1'b0;
            end
            
            default: begin
                if (sidenet_done) begin
                    sidenet_done_latch <= 1'b1;
                end
            end
        endcase
        
        //----------------------------------------------------------------------
        // Load Done 锁存控制
        //----------------------------------------------------------------------
        if (state == STATE_LOAD_INPUT) begin
            load_done_latch <= 1'b0;  // 开始加载，清除锁存
        end else if (load_input_done) begin
            load_done_latch <= 1'b1;  // 捕获完成信号
        end else if (state == STATE_IDLE) begin
            load_done_latch <= 1'b0;  // 空闲时清除
        end
        
        //----------------------------------------------------------------------
        // Store Done 锁存控制
        //----------------------------------------------------------------------
        if (state == STATE_STORE_OUTPUT) begin
            store_done_latch <= 1'b0;  // 开始存储，清除锁存
        end else if (store_output_done) begin
            store_done_latch <= 1'b1;  // 捕获完成信号
        end else if (state == STATE_IDLE) begin
            store_done_latch <= 1'b0;  // 空闲时清除
        end
    end
end

//================================================================================
// 状态机：状态寄存器更新
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
// 状态机：下一状态逻辑 (使用锁存后的done信号)
//================================================================================
always @(*) begin
    // 默认保持当前状态
    state_next = state;
    layer_substate_next = layer_substate;
    
    case (state)
        //--------------------------------------------------------------------
        // IDLE: 等待启动信号
        //--------------------------------------------------------------------
        STATE_IDLE: begin
            if (start) begin
                state_next = STATE_LOAD_INPUT;
                layer_substate_next = LAYER_IDLE;
            end
        end
        
        //--------------------------------------------------------------------
        // LOAD_INPUT: 启动输入加载
        //--------------------------------------------------------------------
        STATE_LOAD_INPUT: begin
            state_next = STATE_WAIT_LOAD;
        end
        
        //--------------------------------------------------------------------
        // WAIT_LOAD: 等待输入加载完成 (使用锁存信号)
        //--------------------------------------------------------------------
        STATE_WAIT_LOAD: begin
            if (load_done_latch) begin
                // 根据ENABLE参数跳转到第一个使能的层
                if (ENABLE_LAYER_0)
                    state_next = STATE_LAYER_0;
                else if (ENABLE_LAYER_1)
                    state_next = STATE_LAYER_1;
                else if (ENABLE_LAYER_2)
                    state_next = STATE_LAYER_2;
                else if (ENABLE_LAYER_3)
                    state_next = STATE_LAYER_3;
                else if (ENABLE_LAYER_4)
                    state_next = STATE_LAYER_4;
                else
                    state_next = STATE_STORE_OUTPUT;
                    
                layer_substate_next = LAYER_IDLE;
            end
            // 超时检测
            else if (timeout_error) begin
                state_next = STATE_ERROR;
            end
        end
        
        //--------------------------------------------------------------------
        // LAYER_0: Layer 0处理 (Backbone + Sidenet并行)
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
                    // 关键改进：使用锁存后的信号
                    if (backbone_done_latch && sidenet_done_latch) begin
                        layer_substate_next = LAYER_BANK_SWAP;
                    end
                    // 超时检测
                    else if (timeout_error) begin
                        state_next = STATE_ERROR;
                        layer_substate_next = LAYER_IDLE;
                    end
                end
                
                LAYER_BANK_SWAP: begin
                    layer_substate_next = LAYER_DONE;
                end
                
                LAYER_DONE: begin
                    // 跳转到下一层
                    if (ENABLE_LAYER_1)
                        state_next = STATE_LAYER_1;
                    else if (ENABLE_LAYER_2)
                        state_next = STATE_LAYER_2;
                    else if (ENABLE_LAYER_3)
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
        // LAYER_1: Layer 1处理 (Backbone + Sidenet并行)
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
                    // 使用锁存后的信号
                    if (backbone_done_latch && sidenet_done_latch) begin
                        layer_substate_next = LAYER_BANK_SWAP;
                    end
                    else if (timeout_error) begin
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
                    else if (ENABLE_LAYER_3)
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
        // LAYER_2: Layer 2处理 (Backbone + Sidenet并行)
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
                    end
                    else if (timeout_error) begin
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
        // LAYER_3: Layer 3处理 (Backbone + Sidenet并行)
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
                    end
                    else if (timeout_error) begin
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
        // LAYER_4: Layer 4处理 (仅Sidenet，特殊处理)
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
                    // 使用锁存后的信号
                    if (sidenet_done_latch) begin
                        layer_substate_next = LAYER4_DONE;
                    end
                    else if (timeout_error) begin
                        state_next = STATE_ERROR;
                        layer_substate_next = LAYER_IDLE;
                    end
                end
                
                LAYER4_DONE: begin
                    state_next = STATE_STORE_OUTPUT;
                    layer_substate_next = LAYER_IDLE;
                end
                
                default: layer_substate_next = LAYER4_IDLE;
            endcase
        end
        
        //--------------------------------------------------------------------
        // STORE_OUTPUT: 启动输出存储
        //--------------------------------------------------------------------
        STATE_STORE_OUTPUT: begin
            state_next = STATE_WAIT_STORE;
        end
        
        //--------------------------------------------------------------------
        // WAIT_STORE: 等待输出存储完成 (使用锁存信号)
        //--------------------------------------------------------------------
        STATE_WAIT_STORE: begin
            if (store_done_latch) begin
                state_next = STATE_DONE;
            end
            else if (timeout_error) begin
                state_next = STATE_ERROR;
            end
        end
        
        //--------------------------------------------------------------------
        // DONE: 完成状态
        //--------------------------------------------------------------------
        STATE_DONE: begin
            // 等待start信号释放后返回IDLE
            // 或者添加超时强制返回机制
            if (!start) begin
                state_next = STATE_IDLE;
            end
        end
        
        //--------------------------------------------------------------------
        // ERROR: 错误状态
        //--------------------------------------------------------------------
        STATE_ERROR: begin
            // 需要复位或start释放后才能恢复
            if (!start) begin
                state_next = STATE_IDLE;
            end
        end
        
        //--------------------------------------------------------------------
        // DEFAULT: 默认返回IDLE
        //--------------------------------------------------------------------
        default: begin
            state_next = STATE_IDLE;
            layer_substate_next = LAYER_IDLE;
        end
    endcase
end

//================================================================================
// 控制信号生成：组合逻辑
//================================================================================
always @(*) begin
    // 默认值：所有控制信号为0
    backbone_start = 1'b0;
    sidenet_start = 1'b0;
    bank_swap_pulse = 1'b0;
    load_input_start = 1'b0;
    store_output_start = 1'b0;
    current_backbone_layer = 2'd0;
    current_sidenet_layer = 3'd0;
    
    case (state)
        //--------------------------------------------------------------------
        // LOAD_INPUT: 发出加载启动信号
        //--------------------------------------------------------------------
        STATE_LOAD_INPUT: begin
            load_input_start = 1'b1;
        end
        
        //--------------------------------------------------------------------
        // LAYER_0: Layer 0控制信号
        //--------------------------------------------------------------------
        STATE_LAYER_0: begin
            current_backbone_layer = 2'd0;
            current_sidenet_layer = 3'd0;
            
            case (layer_substate)
                LAYER_START_BOTH: begin
                    // 同时启动Backbone和Sidenet
                    backbone_start = 1'b1;
                    sidenet_start = 1'b1;
                end
                
                LAYER_BANK_SWAP: begin
                    // 发出Bank切换脉冲
                    bank_swap_pulse = 1'b1;
                end
                
                default: begin
                    // 其他子状态保持默认值
                end
            endcase
        end
        
        //--------------------------------------------------------------------
        // LAYER_1: Layer 1控制信号
        //--------------------------------------------------------------------
        STATE_LAYER_1: begin
            current_backbone_layer = 2'd1;
            current_sidenet_layer = 3'd1;
            
            case (layer_substate)
                LAYER_START_BOTH: begin
                    backbone_start = 1'b1;
                    sidenet_start = 1'b1;
                end
                
                LAYER_BANK_SWAP: begin
                    bank_swap_pulse = 1'b1;
                end
                
                default: begin
                end
            endcase
        end
        
        //--------------------------------------------------------------------
        // LAYER_2: Layer 2控制信号
        //--------------------------------------------------------------------
        STATE_LAYER_2: begin
            current_backbone_layer = 2'd2;
            current_sidenet_layer = 3'd2;
            
            case (layer_substate)
                LAYER_START_BOTH: begin
                    backbone_start = 1'b1;
                    sidenet_start = 1'b1;
                end
                
                LAYER_BANK_SWAP: begin
                    bank_swap_pulse = 1'b1;
                end
                
                default: begin
                end
            endcase
        end
        
        //--------------------------------------------------------------------
        // LAYER_3: Layer 3控制信号
        //--------------------------------------------------------------------
        STATE_LAYER_3: begin
            current_backbone_layer = 2'd3;
            current_sidenet_layer = 3'd3;
            
            case (layer_substate)
                LAYER_START_BOTH: begin
                    backbone_start = 1'b1;
                    sidenet_start = 1'b1;
                end
                
                LAYER_BANK_SWAP: begin
                    bank_swap_pulse = 1'b1;
                end
                
                default: begin
                end
            endcase
        end
        
        //--------------------------------------------------------------------
        // LAYER_4: Layer 4控制信号 (仅Sidenet)
        //--------------------------------------------------------------------
        STATE_LAYER_4: begin
            current_sidenet_layer = 3'd4;
            // current_backbone_layer保持默认0 (不使用)
            
            case (layer_substate)
                LAYER4_START_SN: begin
                    // 仅启动Sidenet
                    sidenet_start = 1'b1;
                end
                
                default: begin
                end
            endcase
        end
        
        //--------------------------------------------------------------------
        // STORE_OUTPUT: 发出存储启动信号
        //--------------------------------------------------------------------
        STATE_STORE_OUTPUT: begin
            store_output_start = 1'b1;
        end
        
        //--------------------------------------------------------------------
        // 其他状态：保持默认值
        //--------------------------------------------------------------------
        default: begin
        end
    endcase
end

//================================================================================
// 状态输出寄存器：时序逻辑
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        done <= 1'b0;
        busy <= 1'b0;
        error <= 1'b0;
    end else begin
        // done信号：仅在DONE状态为高
        done <= (state == STATE_DONE);
        
        // busy信号：除了IDLE和DONE状态外都为高
        busy <= (state != STATE_IDLE && state != STATE_DONE && state != STATE_ERROR);
        
        // error信号：仅在ERROR状态为高
        error <= (state == STATE_ERROR);
    end
end

//================================================================================
// 超时检测逻辑 (改进版 - 实际触发错误)
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        timeout_counter <= 32'd0;
        timeout_error <= 1'b0;
    end else begin
        // 在WAIT状态下计数
        if (state == STATE_WAIT_LOAD || 
            state == STATE_WAIT_STORE ||
            (state >= STATE_LAYER_0 && state <= STATE_LAYER_3 && layer_substate == LAYER_WAIT_BOTH) ||
            (state == STATE_LAYER_4 && layer_substate == LAYER4_WAIT_SN)) begin
            
            if (timeout_counter < TIMEOUT_CYCLES) begin
                timeout_counter <= timeout_counter + 1'b1;
                timeout_error <= 1'b0;
            end else begin
                // 超时触发错误
                timeout_error <= 1'b1;
            end
        end else begin
            // 其他状态清除计数器
            timeout_counter <= 32'd0;
            timeout_error <= 1'b0;
        end
    end
end

//================================================================================
// 调试输出
//================================================================================
assign dbg_state = state;

// dbg_current_layer: 根据当前状态计算层级
assign dbg_current_layer = (state >= STATE_LAYER_0 && state <= STATE_LAYER_3) ? 
                           (state - STATE_LAYER_0) : 2'd0;

// 新增：输出锁存状态用于调试
assign dbg_backbone_done_latch = backbone_done_latch;
assign dbg_sidenet_done_latch = sidenet_done_latch;

//================================================================================
// 断言 (仅用于仿真，综合时会被忽略)
//================================================================================
// synthesis translate_off
always @(posedge clk) begin
    // 检查状态合法性
    if (state > STATE_ERROR && state != STATE_IDLE && state != STATE_DONE) begin
        $display("ERROR: Invalid state %d at time %t", state, $time);
    end
    
    // 检查启动信号是否为单周期脉冲
    if (backbone_start) begin
        @(posedge clk);
        if (backbone_start && layer_substate != LAYER_START_BOTH) begin
            $display("WARNING: backbone_start not a single-cycle pulse at time %t", $time);
        end
    end
    
    // 检查锁存逻辑是否正常工作
    if ((state >= STATE_LAYER_0 && state <= STATE_LAYER_3) && 
        layer_substate == LAYER_WAIT_BOTH) begin
        if (backbone_done && !backbone_done_latch) begin
            $display("INFO: Backbone done captured at time %t", $time);
        end
        if (sidenet_done && !sidenet_done_latch) begin
            $display("INFO: Sidenet done captured at time %t", $time);
        end
    end
end
// synthesis translate_on

endmodule