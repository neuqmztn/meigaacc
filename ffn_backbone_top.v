`timescale 1ns / 1ps

module ffn_backbone_top #(
    //============================================================================
    // 全局架构参数
    //============================================================================
    parameter TOKEN_NUM      = 641,    // 总 Token 数
    parameter TOKEN_CHUNK    = 32,     // 一次处理的 Token 块大小
    parameter BATCH_NUM      = 21,     // Token Batch 数 (641/32向上取整)
    parameter D_MODEL        = 32,     // 模型维度
    parameter D_FF           = 128,    // FFN 隐藏层维度
    parameter FEATURE_CHUNK  = 32,     // 特征分块大小
    parameter NUM_CHUNKS     = 4,      // FFN 分块数 (D_FF / FEATURE_CHUNK)
    
    //============================================================================
    // Scheme B 精度配置参数
    //============================================================================
    parameter BFP_EXP_W      = 8,      // 指数位宽
    parameter BFP_MANT_W     = 8,      // 基础输入/权重尾数位宽 (8-bit)
    parameter ACC_MANT_W     = 16,     // 高精度计算位宽 (Linear Out, Accum) (16-bit)
    parameter INTER_MANT_W   = 8,      // 中间 H_act Buffer 存储位宽 (8-bit)
    
    parameter ADDR_WIDTH     = 5,     // 外部 Buffer 地址位宽
    
    //============================================================================
    // 计算引擎 (CE) 配置参数
    //============================================================================
    parameter G_OUT          = 4,      // CE 输出组数
    parameter T_OUT          = 8,      // 每组输出包含的元素数
    parameter CE_OUTPUT_WIDTH = 32,    // CE 内部定点输出位宽
    parameter CE_BASE_EXP_WIDTH = 9    // CE 内部基指数位宽
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire start,                 // FFN 开始信号
    output wire done,                  // FFN 完成信号
    output wire busy,                  // FFN 忙信号
    
    //================================================================================
    // 外部 Result Buffer 接口 (读输入 Token / 写最终结果) - 8-bit
    //================================================================================
    output wire                      rb_rd_en,
    output wire [ADDR_WIDTH-1:0]     rb_rd_addr,
    input  wire [BFP_EXP_W-1:0]      rb_rd_exp,
    input  wire [D_MODEL*BFP_MANT_W-1:0] rb_rd_mant,
    input  wire                      rb_rd_valid,
    
    output wire                      rb_wr_en,
    output wire [ADDR_WIDTH-1:0]     rb_wr_addr,
    output wire [BFP_EXP_W-1:0]      rb_wr_exp,
    output wire [D_MODEL*BFP_MANT_W-1:0] rb_wr_mant,
    
    //================================================================================
    // 外部权重存储接口 (读 W1 / W2) - 8-bit
    //================================================================================
    output wire                      weight_req,       // 权重请求
    output wire [1:0]                weight_type,      // 0: W1, 1: W2
    output wire [1:0]                weight_chunk_id,  // 权重块 ID
    input  wire                      weight_ready,     // 权重数据就绪
    input  wire [D_MODEL*BFP_EXP_W-1:0] weight_exp_array, // 权重指数数组 (每列一个)
    input  wire [D_MODEL*FEATURE_CHUNK*BFP_MANT_W-1:0] weight_mant// 权重尾数数据块

);

    //================================================================================
    // 内部信号与寄存器声明
    //================================================================================

    //--- FSM 控制信号 ---
    wire [4:0] token_batch_id;
    wire [1:0] feature_chunk_id;
    wire [1:0] stage;
    wire load_token_block, load_w1_chunk, load_w2_chunk;
    wire save_h_act_chunk, load_h_act_chunk, save_result;
    wire linear1_start, linear1_done, linear1_busy;
    wire gelu_start, gelu_done, gelu_busy;
    wire linear2_start, linear2_done, linear2_busy;
    wire acc_clear, acc_enable, acc_valid;
    wire [2:0] acc_count;

    //--- 各子模块完成握手信号寄存器 ---
    reg token_load_done_reg, weight_load_done_reg;
    reg h_act_wr_done_reg, h_act_rd_done_reg, result_wr_done_reg;

    //--- Token 数据缓存 (8-bit) ---
    // 输入 Token Block 通常共享一个指数 (基于上层 LayerNorm 输出)
    reg [TOKEN_CHUNK*BFP_EXP_W-1:0] token_block_exp;
    reg [D_MODEL*BFP_MANT_W-1:0] token_block_mant_mem [0:TOKEN_CHUNK-1];
    reg [TOKEN_CHUNK*D_MODEL*BFP_MANT_W-1:0] token_block_mant_packed;
    //--- 权重数据缓存 (8-bit) ---
    reg [D_MODEL*BFP_EXP_W-1:0] w1_chunk_exp_array;
    reg [D_MODEL*FEATURE_CHUNK*BFP_MANT_W-1:0] w1_chunk_mant;
    reg [FEATURE_CHUNK*BFP_EXP_W-1:0] w2_chunk_exp_array; // W2 为转置视角，维度互换
    reg [FEATURE_CHUNK*D_MODEL*BFP_MANT_W-1:0] w2_chunk_mant;

    //--- Linear1 引擎信号 (8-bit In -> 16-bit Out) ---
    wire [TOKEN_CHUNK*BFP_EXP_W-1:0] linear1_y_exp;         // 输出 32 个独立指数
    wire [TOKEN_CHUNK*FEATURE_CHUNK*ACC_MANT_W-1:0] linear1_y_mant; // 输出 16-bit 高精度尾数

    //--- GELU 引擎信号 (16-bit In -> 8-bit Out) ---
    wire [TOKEN_CHUNK*BFP_EXP_W-1:0] gelu_in_exp;
    wire [TOKEN_CHUNK*FEATURE_CHUNK*ACC_MANT_W-1:0] gelu_in_mant;
    wire [TOKEN_CHUNK*BFP_EXP_W-1:0] gelu_out_exp;          // 透传 32 个独立指数
    wire [TOKEN_CHUNK*FEATURE_CHUNK*INTER_MANT_W-1:0] gelu_out_mant; // 输出 8-bit 压缩尾数

    //--- H_act Buffer 接口信号 (8-bit 串行) ---
    wire h_act_wr_en, h_act_rd_en, h_act_rd_valid;
    wire [6:0] h_act_wr_addr, h_act_rd_addr; // Buffer深度 128，地址 7-bit
    wire [BFP_EXP_W-1:0] h_act_wr_exp, h_act_rd_exp;
    wire [FEATURE_CHUNK*INTER_MANT_W-1:0] h_act_wr_mant, h_act_rd_mant; // 单个Token的8-bit数据

    //--- H_act Write FSM 信号 (并行转串行) ---
    reg [2:0] h_act_wr_state;
    reg [5:0] h_act_wr_token_count;
    reg h_act_wr_en_reg;
    reg [6:0] h_act_wr_addr_reg;
    reg [BFP_EXP_W-1:0] h_act_wr_exp_reg;
    reg [FEATURE_CHUNK*INTER_MANT_W-1:0] h_act_wr_mant_reg;
    // 解包数组
    reg [FEATURE_CHUNK*INTER_MANT_W-1:0] gelu_out_mant_unpacked [0:TOKEN_CHUNK-1];
    reg [BFP_EXP_W-1:0] gelu_out_exp_unpacked [0:TOKEN_CHUNK-1];

    //--- H_act Read FSM 信号 (串行转并行) ---
    reg [2:0] h_act_rd_state;
    reg [5:0] h_act_rd_token_count;
    reg h_act_rd_en_reg;
    reg [6:0] h_act_rd_addr_reg;
    // 并行化缓存
    reg [TOKEN_CHUNK*BFP_EXP_W-1:0] h_act_chunk_exp_array; // 恢复出的 32 个指数
    reg [FEATURE_CHUNK*INTER_MANT_W-1:0] h_act_chunk_mant_mem [0:TOKEN_CHUNK-1];
    reg [TOKEN_CHUNK*FEATURE_CHUNK*INTER_MANT_W-1:0] h_act_chunk_mant_packed;

    //--- Linear2 引擎信号 (8-bit In -> 16-bit Out) ---
    wire [TOKEN_CHUNK*BFP_EXP_W-1:0] linear2_x_exp;      // 输入 32 个独立指数
    wire [TOKEN_CHUNK*FEATURE_CHUNK*INTER_MANT_W-1:0] linear2_x_mant; // 输入 8-bit 尾数
    wire [FEATURE_CHUNK*BFP_EXP_W-1:0] linear2_w_exp_array;
    wire [FEATURE_CHUNK*D_MODEL*BFP_MANT_W-1:0] linear2_w_mant;
    wire [TOKEN_CHUNK*BFP_EXP_W-1:0] linear2_y_exp;      // 输出 32 个独立指数
    wire [TOKEN_CHUNK*D_MODEL*ACC_MANT_W-1:0] linear2_y_mant; // 输出 16-bit 部分和

    //--- Accumulator 信号 (16-bit) ---
    wire [TOKEN_CHUNK*BFP_EXP_W-1:0] acc_result_exp;     // 最终 32 个独立指数
    wire [TOKEN_CHUNK*D_MODEL*ACC_MANT_W-1:0] acc_result_mant; // 最终 16-bit 累加结果

    //--- 结果写回 FSM 信号 (16-bit 转 8-bit 串行) ---
    reg [2:0] result_wr_state;
    reg [5:0] result_wr_token_count;
    reg result_rb_wr_en_reg;
    reg [ADDR_WIDTH-1:0] result_rb_wr_addr_reg;
    reg [BFP_EXP_W-1:0] result_rb_wr_exp_reg;
    reg [D_MODEL*BFP_MANT_W-1:0] result_rb_wr_mant_reg;
    // 解包数组
    reg [D_MODEL*ACC_MANT_W-1:0] acc_result_mant_unpacked [0:TOKEN_CHUNK-1];
    reg [BFP_EXP_W-1:0] acc_result_exp_unpacked [0:TOKEN_CHUNK-1];

    //--- 循环变量 ---
    integer i_pack, i_unpack_mant, i_unpack_exp;
    integer i_pack_h, i_unpack_acc_m, i_unpack_acc_e, i_trunc;

    //================================================================================
    // 1. FFN 主控制状态机实例
    //================================================================================
    ffn_control_fsm #(
        .TOKEN_NUM(TOKEN_NUM), .TOKEN_CHUNK(TOKEN_CHUNK),
        .D_MODEL(D_MODEL), .D_FF(D_FF), .FEATURE_CHUNK(FEATURE_CHUNK)
    ) u_ffn_control_fsm (
        .clk(clk), .rst_n(rst_n),
        .start(start), .done(done), .busy(busy),
        // 各子任务完成信号
        .token_load_done(token_load_done_reg),
        .weight_load_done(weight_load_done_reg),
        .h_act_wr_done(h_act_wr_done_reg),
        .h_act_rd_done(h_act_rd_done_reg),
        .result_wr_done(result_wr_done_reg),
        // 调度信息输出
        .token_batch_id(token_batch_id), .feature_chunk_id(feature_chunk_id), .stage(stage),
        // 任务触发信号
        .load_token_block(load_token_block), .load_w1_chunk(load_w1_chunk),
        .load_w2_chunk(load_w2_chunk), .load_h_act_chunk(load_h_act_chunk),
        .save_h_act_chunk(save_h_act_chunk), .save_result(save_result),
        // 引擎控制信号
        .linear1_start(linear1_start), .linear1_done(linear1_done),
        .gelu_start(gelu_start), .gelu_done(gelu_done),
        .linear2_start(linear2_start), .linear2_done(linear2_done),
        .acc_clear(acc_clear), .acc_enable(acc_enable), .acc_valid(acc_valid),
        // 调试信息
        .cycle_count(dbg_cycle_count), .fsm_state(dbg_state)
    );

    //================================================================================
    // 2. Token 加载逻辑 (完整实现)
    //================================================================================
    reg [2:0] token_load_state;
    reg [5:0] token_load_count;
    reg token_rb_rd_en_reg;
    reg [ADDR_WIDTH-1:0] token_rb_rd_addr_reg;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            token_load_state <= 0;
            token_load_count <= 0;
            token_load_done_reg <= 0;
            token_rb_rd_en_reg <= 0;
            token_rb_rd_addr_reg <= 0;
            token_block_exp <= 0;
            // token_block_mant_mem 清零省略，依赖写入
        end else begin
            case(token_load_state)
                3'd0: begin // IDLE
                    token_load_done_reg <= 0;
                    if(load_token_block) begin
                        token_load_count <= 0;
                        token_load_state <= 3'd1;
                    end
                end
                3'd1: begin // REQUEST
                    token_rb_rd_en_reg <= 1;
                    // 计算绝对地址：Batch起始地址 + 当前Token偏移
                    token_rb_rd_addr_reg <= token_batch_id * TOKEN_CHUNK + token_load_count;
                    token_load_state <= 3'd2;
                end
                3'd2: begin // WAIT FOR VALID
                    token_rb_rd_en_reg <= 0;
                    if(rb_rd_valid) begin
                        token_load_state <= 3'd3;
                    end
                end
                3'd3: begin // RECEIVE DATA
                    // 假设一个 Block 内的所有 Token 共享一个指数 (通常由上层 LN 决定)
                    token_block_exp[token_load_count*BFP_EXP_W +: BFP_EXP_W] <= rb_rd_exp;
                    // 存储尾数
                    token_block_mant_mem[token_load_count] <= rb_rd_mant;
                    
                    if(token_load_count < TOKEN_CHUNK - 1) begin
                        token_load_count <= token_load_count + 1;
                        token_load_state <= 3'd1; // 继续请求下一个
                    end else begin
                        token_load_state <= 3'd4; // 全部加载完成
                    end
                end
                3'd4: begin // DONE
                    token_load_done_reg <= 1;
                    // 等待 FSM 撤销请求信号
                    if(!load_token_block) begin
                        token_load_state <= 3'd0;
                        token_load_done_reg <= 0;
                    end
                end
            endcase
        end
    end

    // 连接 Result Buffer 读接口
    assign rb_rd_en = token_rb_rd_en_reg;
    assign rb_rd_addr = token_rb_rd_addr_reg;

    // Token 数据打包 (Array -> Vector) 供 Linear1 使用
    always @(*) begin
        for (i_pack = 0; i_pack < TOKEN_CHUNK; i_pack = i_pack + 1) begin
            token_block_mant_packed[i_pack*D_MODEL*BFP_MANT_W +: D_MODEL*BFP_MANT_W] 
                = token_block_mant_mem[i_pack];
        end
    end

    //================================================================================
    // 3. 权重加载逻辑 (完整实现)
    //================================================================================
    reg [2:0] weight_load_state;
    reg weight_req_reg;
    reg [1:0] weight_type_reg, weight_chunk_id_reg;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            weight_load_state <= 0;
            weight_load_done_reg <= 0;
            weight_req_reg <= 0;
            weight_type_reg <= 0;
            weight_chunk_id_reg <= 0;
            w1_chunk_exp_array <= 0; w1_chunk_mant <= 0;
            w2_chunk_exp_array <= 0; w2_chunk_mant <= 0;
        end else begin
            case(weight_load_state)
                3'd0: begin // IDLE
                    weight_load_done_reg <= 0;
                    if(load_w1_chunk || load_w2_chunk) begin
                        // 设置请求类型和 Chunk ID
                        weight_type_reg <= load_w2_chunk ? 2'd1 : 2'd0;
                        weight_chunk_id_reg <= feature_chunk_id;
                        weight_req_reg <= 1;
                        weight_load_state <= 3'd1;
                    end
                end
                3'd1: begin // WAIT FOR ACK
                    if(weight_ready) begin
                        // 外部存储器已接收请求，可以撤销 req
                        weight_req_reg <= 0;
                        weight_load_state <= 3'd2;
                    end
                end
                3'd2: begin // WAIT FOR DATA & LATCH
                    // 假设 weight_ready 在数据有效时再次拉高 (或保持高)
                    if(weight_ready) begin
                        if(weight_type_reg == 2'd0) begin // W1
                            w1_chunk_exp_array <= weight_exp_array;
                            w1_chunk_mant <= weight_mant;
                        end else begin // W2
                            w2_chunk_exp_array <= weight_exp_array; // W2 Exp 维度是 FEATURE_CHUNK
                            w2_chunk_mant <= weight_mant;
                        end
                        weight_load_state <= 3'd3;
                    end
                end
                3'd3: begin // DONE
                    weight_load_done_reg <= 1;
                    if(!load_w1_chunk && !load_w2_chunk) begin
                        weight_load_state <= 3'd0;
                        weight_load_done_reg <= 0;
                    end
                end
            endcase
        end
    end

    // 连接权重接口
    assign weight_req = weight_req_reg;
    assign weight_type = weight_type_reg;
    assign weight_chunk_id = weight_chunk_id_reg;

    //================================================================================
    // 4. Linear1 计算引擎实例 (8-bit In -> 16-bit Out)
    //================================================================================
    linear_compute_engine #(
        .TOKEN_CHUNK(TOKEN_CHUNK),
        .INPUT_DIM(D_MODEL),
        .OUTPUT_DIM(FEATURE_CHUNK),
        .BFP_EXP_W(BFP_EXP_W),
        .BFP_MANT_W(BFP_MANT_W),       // 输入/权重均为 8-bit
        .OUTPUT_MANT_W(ACC_MANT_W),    // 输出扩展为 16-bit
        .G_OUT(G_OUT), .T_OUT(T_OUT), .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH)
    ) u_linear1_engine (
        .clk(clk), .rst_n(rst_n),
        // 仅在数据加载完成后启动计算
        .start(linear1_start && token_load_done_reg && weight_load_done_reg),
        .done(linear1_done), .busy(linear1_busy),
        
        .x_exp(token_block_exp),       // 输入 Block 共享指数
        .x_mant(token_block_mant_packed),
        .w_exp_array(w1_chunk_exp_array),
        .w_mant(w1_chunk_mant),
        
        .y_exp(linear1_y_exp),         // 输出 32 个独立指数
        .y_mant(linear1_y_mant)        // 输出 16-bit 高精度尾数
    );

    //================================================================================
    // 5. GELU 引擎实例 (16-bit In -> 8-bit Out 压缩)
    //================================================================================
    assign gelu_in_exp = linear1_y_exp;   // 指数透传
    assign gelu_in_mant = linear1_y_mant; // 16-bit 输入
    
    gelu_engine #(
        .TOKEN_CHUNK(TOKEN_CHUNK), .FEATURE_SIZE(FEATURE_CHUNK),
        .BFP_EXP_W(BFP_EXP_W),
        .INPUT_MANT_W(ACC_MANT_W),     // 16-bit 高精度输入
        .OUTPUT_MANT_W(INTER_MANT_W)   // 8-bit 压缩输出
    ) u_gelu_engine (
        .clk(clk), .rst_n(rst_n),
        .start(gelu_start), .done(gelu_done), .busy(gelu_busy),
        .in_exp(gelu_in_exp),
        .in_mant(gelu_in_mant),
        .out_exp(gelu_out_exp),        // 透传 32 个独立指数
        .out_mant(gelu_out_mant)       // 8-bit 压缩数据
    );

    //================================================================================
    // 6. H_act Buffer 写入状态机 (并行 -> 串行打包)
    //================================================================================
    
    // 组合逻辑：解包并行数据到数组
    always @(*) begin
        for (i_unpack_mant = 0; i_unpack_mant < TOKEN_CHUNK; i_unpack_mant = i_unpack_mant + 1) begin
            gelu_out_mant_unpacked[i_unpack_mant] 
                = gelu_out_mant[i_unpack_mant*FEATURE_CHUNK*INTER_MANT_W +: FEATURE_CHUNK*INTER_MANT_W];
        end
        for (i_unpack_exp = 0; i_unpack_exp < TOKEN_CHUNK; i_unpack_exp = i_unpack_exp + 1) begin
            gelu_out_exp_unpacked[i_unpack_exp] 
                = gelu_out_exp[i_unpack_exp*BFP_EXP_W +: BFP_EXP_W];
        end
    end

    // 时序逻辑：串行写入 FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            h_act_wr_state <= 0; h_act_wr_token_count <= 0; h_act_wr_done_reg <= 0;
            h_act_wr_en_reg <= 0; h_act_wr_addr_reg <= 0;
            h_act_wr_exp_reg <= 0; h_act_wr_mant_reg <= 0;
        end else begin
            case (h_act_wr_state)
                3'd0: begin // IDLE
                    h_act_wr_done_reg <= 0; h_act_wr_en_reg <= 0;
                    if (save_h_act_chunk && gelu_done) begin
                        h_act_wr_token_count <= 0;
                        h_act_wr_state <= 3'd1;
                    end
                end
                3'd1: begin // WRITE CYCLE
                    h_act_wr_en_reg <= 1;
                    // 地址映射：使用 Feature Chunk ID 作为基地址偏移，Token Count 作为内部偏移
                    // 假设 Buffer 深度 128，映射方式为 {chunk_id[1:0], token_id[4:0]}
                    h_act_wr_addr_reg <= {feature_chunk_id, h_act_wr_token_count[4:0]}; 
                    
                    // 写入当前 Token 的独立指数和尾数
                    h_act_wr_exp_reg <= gelu_out_exp_unpacked[h_act_wr_token_count];
                    h_act_wr_mant_reg <= gelu_out_mant_unpacked[h_act_wr_token_count];
                    
                    h_act_wr_state <= 3'd2;
                end
                3'd2: begin // LOOP CONTROL
                    h_act_wr_en_reg <= 0; // 撤销写使能，准备下一拍
                    if (h_act_wr_token_count < TOKEN_CHUNK - 1) begin
                        h_act_wr_token_count <= h_act_wr_token_count + 1;
                        h_act_wr_state <= 3'd1; // 继续写下一个
                    end else begin
                        h_act_wr_state <= 3'd3; // 全部写完
                    end
                end
                3'd3: begin // DONE
                    h_act_wr_done_reg <= 1;
                    if (!save_h_act_chunk) begin 
                        h_act_wr_state <= 3'd0; 
                        h_act_wr_done_reg <= 0; 
                    end
                end
            endcase
        end
    end
    
    // 连接 Buffer 写接口
    assign h_act_wr_en = h_act_wr_en_reg;
    assign h_act_wr_addr = h_act_wr_addr_reg;
    assign h_act_wr_exp = h_act_wr_exp_reg;
    assign h_act_wr_mant = h_act_wr_mant_reg;

    //================================================================================
    // 7. H_act Buffer 实例 (8-bit 单端口 SRAM)
    //================================================================================
    result_buffer_single_port #(
        .TOKEN_NUM(128),               // 总容量 32 tokens * 4 chunks
        .DIM(FEATURE_CHUNK),           // 数据维度
        .DATA_WIDTH(INTER_MANT_W),     // 8-bit 数据位宽
        .EXP_WIDTH(BFP_EXP_W),         // 8-bit 指数位宽
        .ADDR_WIDTH(7)                 // 7-bit 地址线 (覆盖 0-127)
    ) u_h_act_buffer (
        .clk(clk), .rst_n(rst_n),
        // 读端口
        .rd_en(h_act_rd_en), .rd_addr(h_act_rd_addr),
        .rd_exp(h_act_rd_exp), .rd_mant(h_act_rd_mant), .rd_valid(h_act_rd_valid),
        // 写端口
        .wr_en(h_act_wr_en), .wr_addr(h_act_wr_addr),
        .wr_exp(h_act_wr_exp), .wr_mant(h_act_wr_mant),
        // 调试接口
        .dbg_rd_count(), .dbg_wr_count()
    );

    //================================================================================
    // 8. H_act Buffer 读取状态机 (串行 -> 并行解包)
    //================================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            h_act_rd_state <= 0; h_act_rd_token_count <= 0; h_act_rd_done_reg <= 0;
            h_act_rd_en_reg <= 0; h_act_rd_addr_reg <= 0;
            h_act_chunk_exp_array <= 0;
            // mem 清零省略
        end else begin
            case (h_act_rd_state)
                3'd0: begin // IDLE
                    h_act_rd_done_reg <= 0;
                    if (load_h_act_chunk) begin
                        h_act_rd_token_count <= 0;
                        h_act_rd_state <= 3'd1;
                    end
                end
                3'd1: begin // REQUEST
                    h_act_rd_en_reg <= 1;
                    // 地址映射需与写入一致：{chunk_id[1:0], token_id[4:0]}
                    h_act_rd_addr_reg <= {feature_chunk_id, h_act_rd_token_count[4:0]};
                    h_act_rd_state <= 3'd2;
                end
                3'd2: begin // WAIT FOR VALID
                    h_act_rd_en_reg <= 0;
                    if (h_act_rd_valid) begin
                        h_act_rd_state <= 3'd3;
                    end
                end
                3'd3: begin // RECEIVE & LATCH
                    // 将读出的指数填入数组对应位置
                    h_act_chunk_exp_array[h_act_rd_token_count*BFP_EXP_W +: BFP_EXP_W] 
                        <= h_act_rd_exp;
                    // 将读出的尾数存入内存数组
                    h_act_chunk_mant_mem[h_act_rd_token_count] <= h_act_rd_mant;
                    
                    if (h_act_rd_token_count < TOKEN_CHUNK - 1) begin
                        h_act_rd_token_count <= h_act_rd_token_count + 1;
                        h_act_rd_state <= 3'd1; // 请求下一个
                    end else begin
                        h_act_rd_state <= 3'd4; // 全部读完
                    end
                end
                3'd4: begin // DONE
                    h_act_rd_done_reg <= 1;
                    if (!load_h_act_chunk) begin 
                        h_act_rd_state <= 3'd0; 
                        h_act_rd_done_reg <= 0; 
                    end
                end
            endcase
        end
    end
    
    // 连接 Buffer 读接口
    assign h_act_rd_en = h_act_rd_en_reg;
    assign h_act_rd_addr = h_act_rd_addr_reg;

    // 组合逻辑：数据打包 (Array -> Vector) 供 Linear2 使用
    always @(*) begin
        for (i_pack_h = 0; i_pack_h < TOKEN_CHUNK; i_pack_h = i_pack_h + 1) begin
            h_act_chunk_mant_packed[i_pack_h*FEATURE_CHUNK*INTER_MANT_W +: FEATURE_CHUNK*INTER_MANT_W]
                = h_act_chunk_mant_mem[i_pack_h];
        end
    end

    //================================================================================
    // 9. Linear2 计算引擎实例 (8-bit In -> 16-bit Out)
    //================================================================================
    // Linear2 的输入来自 H_act Read FSM 恢复出的并行数据
    assign linear2_x_exp = h_act_chunk_exp_array;
    assign linear2_x_mant = h_act_chunk_mant_packed;
    // 权重输入
    assign linear2_w_exp_array = w2_chunk_exp_array;
    assign linear2_w_mant = w2_chunk_mant;

    linear_compute_engine #(
        .TOKEN_CHUNK(TOKEN_CHUNK), .INPUT_DIM(FEATURE_CHUNK), .OUTPUT_DIM(D_MODEL),
        .BFP_EXP_W(BFP_EXP_W),
        .BFP_MANT_W(INTER_MANT_W),     // 8-bit 输入 (来自 Buffer)
        .OUTPUT_MANT_W(ACC_MANT_W),    // 16-bit 输出 (部分和)
        .G_OUT(G_OUT), .T_OUT(T_OUT), .CE_OUTPUT_WIDTH(CE_OUTPUT_WIDTH)
    ) u_linear2_engine (
        .clk(clk), .rst_n(rst_n),
        // 仅在激活值和权重都加载完成后启动
        .start(linear2_start && h_act_rd_done_reg && weight_load_done_reg),
        .done(linear2_done), .busy(linear2_busy),
        
        .x_exp(linear2_x_exp),         // 输入 32 个独立指数
        .x_mant(linear2_x_mant),       // 输入 8-bit 数据
        .w_exp_array(linear2_w_exp_array),
        .w_mant(linear2_w_mant),
        
        .y_exp(linear2_y_exp),         // 输出 32 个独立指数
        .y_mant(linear2_y_mant)        // 输出 16-bit 部分和
    );

    //================================================================================
    // 10. Linear2 累加器实例 (16-bit 累加)
    //================================================================================
    linear2_accumulator #(
        .TOKEN_CHUNK(TOKEN_CHUNK), .OUTPUT_DIM(D_MODEL),
        .BFP_EXP_W(BFP_EXP_W),
        .INPUT_MANT_W(ACC_MANT_W),     // 16-bit 输入
        .OUTPUT_MANT_W(ACC_MANT_W),    // 16-bit 输出
        .NUM_CHUNKS(NUM_CHUNKS)        // 累加次数 (通常为 4)
    ) u_accumulator (
        .clk(clk), .rst_n(rst_n),
        .clear(acc_clear),             // FSM 控制清零
        .enable(acc_enable),           // FSM 控制使能
        
        .partial_exp(linear2_y_exp),   // 输入 32 个独立指数用于对齐
        .partial_mant(linear2_y_mant), // 输入 16-bit 部分和
        
        .result_exp(acc_result_exp),   // 输出最终 32 个指数
        .result_mant(acc_result_mant), // 输出最终 16-bit 结果
        .result_valid(acc_valid),      // 累加完成指示
        .debug_accum_count(acc_count)
    );

    //================================================================================
    // 11. 结果写回状态机 (16-bit 截断 -> 8-bit 串行写入)
    //================================================================================
    
    // 组合逻辑：解包并行累加结果到数组
    always @(*) begin
        for (i_unpack_acc_m = 0; i_unpack_acc_m < TOKEN_CHUNK; i_unpack_acc_m = i_unpack_acc_m + 1) begin
            acc_result_mant_unpacked[i_unpack_acc_m] 
                = acc_result_mant[i_unpack_acc_m*D_MODEL*ACC_MANT_W +: D_MODEL*ACC_MANT_W];
        end
        for (i_unpack_acc_e = 0; i_unpack_acc_e < TOKEN_CHUNK; i_unpack_acc_e = i_unpack_acc_e + 1) begin
            acc_result_exp_unpacked[i_unpack_acc_e] 
                = acc_result_exp[i_unpack_acc_e*BFP_EXP_W +: BFP_EXP_W];
        end
    end

    // 时序逻辑：串行写回 FSM
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            result_wr_state <= 0; result_wr_token_count <= 0; result_wr_done_reg <= 0;
            result_rb_wr_en_reg <= 0; result_rb_wr_addr_reg <= 0;
            result_rb_wr_exp_reg <= 0; result_rb_wr_mant_reg <= 0;
        end else begin
            case(result_wr_state)
                3'd0: begin // IDLE
                    result_wr_done_reg <= 0; result_rb_wr_en_reg <= 0;
                    if(save_result && acc_valid) begin
                        result_wr_token_count <= 0;
                        result_wr_state <= 3'd1;
                    end
                end
                3'd1: begin // WRITE CYCLE & TRUNCATE
                    result_rb_wr_en_reg <= 1;
                    // 计算绝对地址：Batch起始地址 + 当前Token偏移
                    result_rb_wr_addr_reg <= token_batch_id * TOKEN_CHUNK + result_wr_token_count;
                    
                    // 写入当前 Token 的独立指数
                    result_rb_wr_exp_reg <= acc_result_exp_unpacked[result_wr_token_count];
                    
                    // 位宽截断 (16-bit -> 8-bit) ---
                    // 遍历当前 Token 的所有维度，取每个 16-bit 数据的高 8 位
                    for(i_trunc = 0; i_trunc < D_MODEL; i_trunc = i_trunc + 1) begin
                        // 提取高 8 位 (简单截断，未做舍入)
                        result_rb_wr_mant_reg[i_trunc*BFP_MANT_W +: BFP_MANT_W] 
                            <= acc_result_mant_unpacked[result_wr_token_count][(i_trunc+1)*ACC_MANT_W-1 -: BFP_MANT_W];
                    end
                    
                    result_wr_state <= 3'd2;
                end
                3'd2: begin // LOOP CONTROL
                    result_rb_wr_en_reg <= 0;
                    if(result_wr_token_count < TOKEN_CHUNK-1) begin
                        result_wr_token_count <= result_wr_token_count + 1;
                        result_wr_state <= 3'd1;
                    end else begin
                        result_wr_state <= 3'd3;
                    end
                end
                3'd3: begin // DONE
                    result_wr_done_reg <= 1;
                    if(!save_result) begin 
                        result_wr_state <= 3'd0; 
                        result_wr_done_reg <= 0; 
                    end
                end
            endcase
        end
    end

    // 连接外部 Result Buffer 写接口
    assign rb_wr_en = result_rb_wr_en_reg;
    assign rb_wr_addr = result_rb_wr_addr_reg;
    assign rb_wr_exp = result_rb_wr_exp_reg;
    assign rb_wr_mant = result_rb_wr_mant_reg;

endmodule