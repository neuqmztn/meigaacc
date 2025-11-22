`timescale 1ns / 1ps

//==============================================================
// Compute Engine - v9.0 (No-FIFO Optimized)
// 
// 修改日志：
// 1. 移除输入端冗余的 FIFO，采用直接寄存器握手 (Skid Buffer 机制)。
// 2. 修复 W 权重数据的切片逻辑，支持任意 T_OUT 参数，不再硬编码。
// 3. 完美适配 PU v3.1 的 32-bit 截断输出接口。
//==============================================================

module compute_engine #(
    // ========== 矩阵维度 ==========
    parameter G_OUT = 4,               // 输出组数（列数）
    parameter T_OUT = 3,               // 每组的PU数量（行数）
    
    // ========== PE配置 ==========
    parameter NUM_PE = 1,              // 每个PU的PE数量
    parameter PE_TYPE_0 = 0,           // PE0类型
    parameter PE_TYPE_1 = 2,           // PE1类型
    
    // ========== 数据位宽 ==========
    parameter EXP_WIDTH = 8,           // 指数位宽
    parameter INPUT_MANT_WIDTH = 8,    // 输入尾数位宽
    
    // ========== 向量维度 ==========
    parameter ELEM_PE0 = 16,           
    parameter ELEM_PE1 = 8,            
    parameter TOTAL_ELEM = 24,         
    
    // ========== PU位宽优化配置 ==========
    parameter INTERNAL_WIDTH = 39,     // PU内部计算位宽
    parameter OUTPUT_WIDTH = 32,       // PU输出位宽
    parameter GUARD_BITS = 7,          // 截断保护位数
    parameter ENABLE_ROUNDING = 1,     // 启用舍入
    
    // ========== 其他配置 ==========
    parameter HANDSHAKE_TIMEOUT = 100  // 握手超时周期数
)(
    input  wire clk,
    input  wire rst_n,
    input  wire flush,
    
    // ========== 输入握手 ==========
    input  wire input_valid,
    output wire input_ready,
    
    // ========== 输入数据 ==========
    // 输入 X 是广播给所有 Column 的
    input  wire [EXP_WIDTH-1:0] exp_X,
    input  wire [TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_X_block,
    
    // 输入 W 是分发给不同 Column 和 PU 的
    // 宽度 = G_OUT * T_OUT * WIDTH
    input  wire [G_OUT*T_OUT*EXP_WIDTH-1:0] exp_W_array,
    input  wire [G_OUT*T_OUT*TOTAL_ELEM*INPUT_MANT_WIDTH-1:0] mant_W_blocks,
    
    // ========== 输出握手 ==========
    output wire [G_OUT*T_OUT-1:0] result_valids,
    input  wire result_ready,
    
    // ========== 输出数据 ==========
    output wire signed [G_OUT*T_OUT*OUTPUT_WIDTH-1:0] result_fixed_array,
    output wire [G_OUT*T_OUT*(EXP_WIDTH+1)-1:0] result_base_exp_array,
    output wire [G_OUT*T_OUT-1:0] result_zero_array
);

    //==============================================================
    // 本地参数计算
    //==============================================================
    localparam TOTAL_WIDTH = TOTAL_ELEM * INPUT_MANT_WIDTH;
    localparam TOTAL_PUS = G_OUT * T_OUT;

    //==============================================================
    // 全局握手控制 (Global Handshake)
    //==============================================================
    // 只有当所有列 (Column) 都准备好接收新任务时，CE 才 Ready
    wire [G_OUT-1:0] col_ready_array;
    assign input_ready = &col_ready_array;

    //==============================================================
    // PU 连接信号阵列
    //==============================================================
    wire [TOTAL_PUS-1:0] pu_input_valid_array;
    wire [TOTAL_PUS-1:0] pu_input_ready_array;
    wire [TOTAL_PUS-1:0] pu_result_valid_array;
    wire [TOTAL_PUS-1:0] pu_result_ready_array;

    wire signed [TOTAL_PUS*OUTPUT_WIDTH-1:0] pu_result_fixed_array;
    wire [TOTAL_PUS*(EXP_WIDTH+1)-1:0] pu_result_base_exp_array;
    wire [TOTAL_PUS-1:0] pu_result_zero_array;

    //==============================================================
    // 列控制逻辑 (Column Control Logic)
    //==============================================================
    genvar g_col;
    generate
        for (g_col = 0; g_col < G_OUT; g_col = g_col + 1) begin : columns
            
            //------------------------------------------------------
            // Cache 寄存器 (直接锁存输入)
            //------------------------------------------------------
            reg [TOTAL_WIDTH-1:0] cached_mant_X;
            reg [EXP_WIDTH-1:0]   cached_exp_X;
            
            // 缓存该列下所有 PU (T_OUT个) 的权重
            reg [T_OUT*EXP_WIDTH-1:0]   cached_exp_W;
            reg [T_OUT*TOTAL_WIDTH-1:0] cached_mant_W;
            
            reg cache_valid;

            //------------------------------------------------------
            // 状态机定义
            //------------------------------------------------------
            localparam ST_IDLE       = 3'b000;
            // 移除 ST_LOADING，因为直接锁存后即可进入 WAIT_READY，节省 1 cycle
            localparam ST_WAIT_READY = 3'b010;
            localparam ST_PROCESSING = 3'b011;
            localparam ST_SWITCHING  = 3'b100;
            
            reg [2:0] state;
            reg [$clog2(T_OUT)-1:0] active_pu_idx; // 当前激活的 PU 索引
            
            //------------------------------------------------------
            // 握手与超时控制
            //------------------------------------------------------
            reg input_done;
            reg [$clog2(HANDSHAKE_TIMEOUT+1)-1:0] handshake_timeout_cnt;
            wire handshake_timeout = (handshake_timeout_cnt >= HANDSHAKE_TIMEOUT);

            //------------------------------------------------------
            // 当前列 Ready 信号
            //------------------------------------------------------
            assign col_ready_array[g_col] = (state == ST_IDLE);

            //------------------------------------------------------
            // 状态机逻辑
            //------------------------------------------------------
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    state <= ST_IDLE;
                    cache_valid <= 0;
                    active_pu_idx <= 0;
                    input_done <= 0;
                    handshake_timeout_cnt <= 0;
                    // Registers reset
                    cached_mant_X <= 0; cached_exp_X <= 0;
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
                            // 握手成功：Input Valid=1 且 全局 Ready=1
                            if (input_valid && input_ready) begin
                                // 1. 锁存广播数据 X
                                cached_exp_X  <= exp_X;
                                cached_mant_X <= mant_X_block;
                                
                                // 2. 锁存分发数据 W (参数化切片)
                                // 提取属于当前 Column (g_col) 的 T_OUT 组权重
                                cached_exp_W  <= exp_W_array[ g_col * (T_OUT*EXP_WIDTH) +: (T_OUT*EXP_WIDTH) ];
                                cached_mant_W <= mant_W_blocks[ g_col * (T_OUT*TOTAL_WIDTH) +: (T_OUT*TOTAL_WIDTH) ];
                                
                                // 3. 状态跳转
                                cache_valid <= 1;
                                active_pu_idx <= 0;
                                input_done <= 0;
                                handshake_timeout_cnt <= 0;
                                state <= ST_WAIT_READY;
                            end
                        end
                        
                        // 这里的状态逻辑与 v8.4 保持一致，确保 PU 切换正确
                        ST_WAIT_READY: begin
                            // 获取当前激活 PU 的 ready 信号
                            if (pu_input_ready_array[g_col*T_OUT + active_pu_idx]) begin
                                state <= ST_PROCESSING;
                                handshake_timeout_cnt <= 0;
                            end else if (handshake_timeout) begin
                                // 超时复位
                                state <= ST_IDLE;
                                cache_valid <= 0;
                                handshake_timeout_cnt <= 0;
                            end else begin
                                handshake_timeout_cnt <= handshake_timeout_cnt + 1;
                            end
                        end
                        
                        ST_PROCESSING: begin
                            // 等待握手完成 (Input accepted)
                            if (pu_input_valid_array[g_col*T_OUT + active_pu_idx] && 
                                pu_input_ready_array[g_col*T_OUT + active_pu_idx]) begin
                                input_done <= 1;
                                handshake_timeout_cnt <= 0;
                            end else if (!input_done) begin
                                if (handshake_timeout) begin
                                    input_done <= 1; // 强制完成
                                    handshake_timeout_cnt <= 0;
                                end else begin
                                    handshake_timeout_cnt <= handshake_timeout_cnt + 1;
                                end
                            end
                            
                            // 等待输出完成 (Output accepted)
                            if (result_valids[g_col*T_OUT + active_pu_idx] && result_ready && input_done) begin
                                state <= ST_SWITCHING;
                                input_done <= 0;
                                handshake_timeout_cnt <= 0;
                                
                                // 切换到下一个 PU
                                if (active_pu_idx < T_OUT-1) begin
                                    active_pu_idx <= active_pu_idx + 1;
                                end else begin
                                    active_pu_idx <= 0; // 循环结束
                                end
                            end
                        end
                        
                        ST_SWITCHING: begin
                            if (active_pu_idx == 0) begin
                                // 所有 PU 都处理完了，回到 IDLE 接收新数据
                                state <= ST_IDLE;
                                cache_valid <= 0;
                            end else begin
                                // 处理下一个 PU
                                state <= ST_WAIT_READY;
                            end
                        end
                        
                        default: state <= ST_IDLE;
                    endcase
                end
            end

            //==========================================================
            // PU 阵列实例化 (每个 Column 内有 T_OUT 个 PU)
            //==========================================================
            genvar t_pu;
            for (t_pu = 0; t_pu < T_OUT; t_pu = t_pu + 1) begin : pu_pipeline
                
                localparam PU_IDX = g_col * T_OUT + t_pu;
                
                // 从 Cache 中切片出当前 PU 的权重
                wire [EXP_WIDTH-1:0]   pu_exp_W;
                wire [TOTAL_WIDTH-1:0] pu_mant_W;
                assign pu_exp_W  = cached_exp_W[t_pu*EXP_WIDTH +: EXP_WIDTH];
                assign pu_mant_W = cached_mant_W[t_pu*TOTAL_WIDTH +: TOTAL_WIDTH];
                
                // 激活信号：仅当状态机指向当前 PU 时才有效
                wire this_pu_active;
                assign this_pu_active = cache_valid && (active_pu_idx == t_pu);
                
                assign pu_input_valid_array[PU_IDX] = this_pu_active && 
                                                      ((state == ST_PROCESSING) || (state == ST_WAIT_READY)) && 
                                                      !input_done;
                                                      
                assign pu_result_ready_array[PU_IDX] = result_ready;
                assign result_valids[PU_IDX] = pu_result_valid_array[PU_IDX];
                
                // 实例化 PU (v3.1 接口)
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

    //==============================================================
    // 全局输出连接
    //==============================================================
    assign result_fixed_array = pu_result_fixed_array;
    assign result_base_exp_array = pu_result_base_exp_array;
    assign result_zero_array = pu_result_zero_array;

    //==============================================================
    // 调试与检查
    //==============================================================
    initial begin
        $display("========================================");
        $display("Compute Engine v9.0 - No FIFO Optimization");
        $display("========================================");
        $display("Config:");
        $display("  G_OUT (Cols): %0d, T_OUT (Rows): %0d", G_OUT, T_OUT);
        $display("  Total PUs:    %0d", TOTAL_PUS);
        $display("  Output Width: %0d bits", OUTPUT_WIDTH);
        $display("  FIFO Removed: YES (Direct Latched)");
        $display("========================================");
    end

endmodule