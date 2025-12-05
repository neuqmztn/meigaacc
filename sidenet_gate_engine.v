`timescale 1ns / 1ps

module sidenet_gate_engine #(
    parameter TOKEN_NUM      = 641,       // Token数量
    parameter COMPRESSED_DIM = 8,         // 压缩维度
    parameter DATA_WIDTH     = 16,        // 16-bit尾数
    parameter EXP_WIDTH      = 8,         // 8-bit指数
    parameter ADDR_WIDTH     = 10,        // 地址位宽
    parameter NUM_LAYERS     = 4          // 层数
)(
    input  wire clk,
    input  wire rst_n,
    
    //--------------------------------------------------------------------------
    // 控制接口
    //--------------------------------------------------------------------------
    input  wire start,
    input  wire is_layer0,
    input  wire [1:0] layer_id,
    output reg  done,
    output reg  busy,
    
    //--------------------------------------------------------------------------
    // 数据读写接口
    //--------------------------------------------------------------------------
    // Compressed Buffer (Input Z)
    output reg  compressed_rd_en,
    output reg  [ADDR_WIDTH-1:0] compressed_rd_addr,
    input  wire [EXP_WIDTH-1:0] compressed_rd_exp,
    input  wire [COMPRESSED_DIM*DATA_WIDTH-1:0] compressed_rd_mant,
    input  wire compressed_rd_valid,
    
    // Adapted Buffer (Input A)
    output reg  adapted_rd_en,
    output reg  [ADDR_WIDTH-1:0] adapted_rd_addr,
    input  wire [EXP_WIDTH-1:0] adapted_rd_exp,
    input  wire [COMPRESSED_DIM*DATA_WIDTH-1:0] adapted_rd_mant,
    input  wire adapted_rd_valid,
    
    // Gated Buffer (Output Result)
    output reg  gated_wr_en,
    output reg  [ADDR_WIDTH-1:0] gated_wr_addr,
    output reg  [EXP_WIDTH-1:0] gated_wr_exp,
    output reg  [COMPRESSED_DIM*DATA_WIDTH-1:0] gated_wr_mant,

    // 调试信号
    output wire [3:0] dbg_state,
    output reg  [31:0] dbg_token_count
);

    //==========================================================================
    // 参数与状态定义
    //==========================================================================
    localparam S_IDLE        = 4'd0;
    localparam S_LOAD_Z      = 4'd1;
    localparam S_WAIT_Z      = 4'd2;
    localparam S_LOAD_A      = 4'd3;
    localparam S_WAIT_A      = 4'd4;
    localparam S_EXECUTE     = 4'd5; // 流水线计算阶段
    localparam S_ALIGN       = 4'd6; // 统一对齐阶段 (Re-alignment)
    localparam S_WRITE       = 4'd7;
    localparam S_NEXT        = 4'd8;
    localparam S_DONE        = 4'd9;
    localparam S_BYPASS      = 4'd10;

    reg [3:0] state, next_state;
    reg [9:0] token_cnt;

    // Gamma 表
    reg [15:0] gamma_params [0:NUM_LAYERS-1];
    reg [15:0] curr_gamma, curr_one_minus_gamma;

    initial begin
        gamma_params[0] = 16'h0000;
        gamma_params[1] = 16'h4000; // 0.25
        gamma_params[2] = 16'h6000; // 0.375
        gamma_params[3] = 16'h8000; // 0.50
    end

    //==========================================================================
    // 数据缓存区
    //==========================================================================
    // 输入缓存
    reg [EXP_WIDTH-1:0] in_z_exp;
    reg [COMPRESSED_DIM*DATA_WIDTH-1:0] in_z_mant;
    reg [EXP_WIDTH-1:0] in_a_exp;
    reg [COMPRESSED_DIM*DATA_WIDTH-1:0] in_a_mant;

    // 中间结果缓存 (Temp Results)
    // 必须存储每个维度的独立指数和尾数，等待最后统一对齐
    reg [DATA_WIDTH-1:0] temp_mant_arr [0:COMPRESSED_DIM-1];
    reg [EXP_WIDTH-1:0]  temp_exp_arr  [0:COMPRESSED_DIM-1];
    reg [EXP_WIDTH-1:0]  global_max_exp; // 当前Token所有维度中的最大指数

    // 流水线计数器
    reg [3:0] cnt_issue;   // 发射计数
    reg [3:0] cnt_collect; // 收集计数

    //==========================================================================
    // 乘法器与加法器接口信号
    //==========================================================================
    reg  bfp_en;
    wire bfp_valid_out; // 加法器输出有效指示
    
    // 乘法器输出（加法器输入）
    wire [15:0] mult_z_res;
    wire [15:0] mult_a_res;
    
    // 加法器输出
    wire [EXP_WIDTH-1:0] bfp_res_exp;
    wire [DATA_WIDTH-1:0] bfp_res_mant;

    // 辅助移位寄存器，用于生成 valid 信号 (模拟 Pipeline Delay)
    // 假设 bfp_adder 延迟为 2 周期
    reg [2:0] pipe_delay_sr; 

    //==========================================================================
    // 核心计算逻辑 1: 加权乘法 (Combinational)
    //==========================================================================
    // 1. 切片：获取当前发射维度的数据
    wire [15:0] slice_z = in_z_mant[cnt_issue*16 +: 16];
    wire [15:0] slice_a = in_a_mant[cnt_issue*16 +: 16];

    // 2. 乘法：Fixed Point 0.16 * 0.16 -> 0.32
    wire [31:0] prod_z = slice_z * curr_one_minus_gamma;
    wire [31:0] prod_a = slice_a * curr_gamma;

    // 3. 截断：取高16位 (保持 0.16 格式)
    assign mult_z_res = prod_z[31:16];
    assign mult_a_res = prod_a[31:16];

    //==========================================================================
    // 核心计算逻辑 2: BFP 对齐与打包 (Combinational for S_ALIGN)
    //==========================================================================
    reg [COMPRESSED_DIM*DATA_WIDTH-1:0] aligned_mant_vector;
    integer i;
    reg [EXP_WIDTH-1:0] shift_amount;
    reg [DATA_WIDTH-1:0] shifted_mant;

    always @(*) begin
        aligned_mant_vector = {COMPRESSED_DIM*DATA_WIDTH{1'b0}};
        
        // 遍历所有维度，根据 global_max_exp 进行右移对齐
        for (i = 0; i < COMPRESSED_DIM; i = i + 1) begin
            // 计算位移量：最大指数 - 当前维度指数
            if (global_max_exp >= temp_exp_arr[i])
                shift_amount = global_max_exp - temp_exp_arr[i];
            else
                shift_amount = 0; // 理论上不可能发生，除非逻辑错误
                
            // 执行右移
            shifted_mant = temp_mant_arr[i] >> shift_amount;
            
            // 拼接到输出向量
            // 注意：in_z_mant[0] 对应低位，所以这里保持一致
            aligned_mant_vector[i*16 +: 16] = shifted_mant;
        end
    end

    //==========================================================================
    // 状态机控制
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else state <= next_state;
    end

    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE: 
                if (start) begin
                    if (is_layer0) next_state = S_BYPASS;
                    else next_state = S_LOAD_Z;
                end
            
            S_BYPASS: if (compressed_rd_valid) next_state = S_WRITE;

            S_LOAD_Z: next_state = S_WAIT_Z;
            S_WAIT_Z: if (compressed_rd_valid) next_state = S_LOAD_A;
            
            S_LOAD_A: next_state = S_WAIT_A;
            S_WAIT_A: if (adapted_rd_valid) next_state = S_EXECUTE;
            
            S_EXECUTE: 
                // 当收集完所有8个结果
                if (cnt_collect == COMPRESSED_DIM) next_state = S_ALIGN;
                
            S_ALIGN: 
                // 单周期组合逻辑完成对齐
                next_state = S_WRITE;
                
            S_WRITE: next_state = S_NEXT;
            
            S_NEXT: 
                if (token_cnt == TOKEN_NUM - 1) next_state = S_DONE;
                else begin
                    if (is_layer0) next_state = S_BYPASS;
                    else next_state = S_LOAD_Z;
                end
                
            S_DONE: next_state = S_IDLE;
            default: next_state = S_IDLE;
        endcase
    end

    //==========================================================================
    // 数据路径与时序逻辑
    //==========================================================================
    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            token_cnt <= 10'd0;
            compressed_rd_en <= 1'b0;
            adapted_rd_en <= 1'b0;
            gated_wr_en <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            
            curr_gamma <= 16'd0;
            curr_one_minus_gamma <= 16'd0;
            
            cnt_issue <= 4'd0;
            cnt_collect <= 4'd0;
            bfp_en <= 1'b0;
            pipe_delay_sr <= 3'd0;
            
            global_max_exp <= 8'd0;
            // 清空中间存储数组 (Reset temp arrays)
            for (j=0; j<COMPRESSED_DIM; j=j+1) begin
                temp_mant_arr[j] <= 16'd0;
                temp_exp_arr[j] <= 8'd0;
            end
            
            dbg_token_count <= 32'd0;
        end else begin
            // 默认脉冲复位
            compressed_rd_en <= 1'b0;
            adapted_rd_en <= 1'b0;
            gated_wr_en <= 1'b0;
            bfp_en <= 1'b0;
            done <= 1'b0;
            
            // 移位寄存器每拍更新，用于追踪 Pipeline Delay
            pipe_delay_sr <= {pipe_delay_sr[1:0], bfp_en};

            case (state)
                S_IDLE: begin
                    token_cnt <= 10'd0;
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        curr_gamma <= gamma_params[layer_id];
                        curr_one_minus_gamma <= 16'hFFFF - gamma_params[layer_id];
                    end
                end

                // --- Layer 0 直通 ---
                S_BYPASS: begin
                    compressed_rd_en <= 1'b1;
                    compressed_rd_addr <= token_cnt;
                    if (compressed_rd_valid) begin
                        // 暂存直通数据到输出寄存器 (复用 gated_wr_* 在 Write 状态输出)
                        // 这里简化为直接在 S_WRITE 中使用 buffer 数据，或者需要额外的寄存器
                        // 为了逻辑清晰，我们在 S_WRITE 处理，这里只负责读
                        in_z_exp <= compressed_rd_exp;
                        in_z_mant <= compressed_rd_mant;
                    end
                end

                // --- Layer > 0 加载 ---
                S_LOAD_Z: begin
                    compressed_rd_en <= 1'b1;
                    compressed_rd_addr <= token_cnt;
                end
                S_WAIT_Z: begin
                    if (compressed_rd_valid) begin
                        in_z_exp <= compressed_rd_exp;
                        in_z_mant <= compressed_rd_mant;
                    end
                end
                S_LOAD_A: begin
                    adapted_rd_en <= 1'b1;
                    adapted_rd_addr <= token_cnt;
                end
                S_WAIT_A: begin
                    if (adapted_rd_valid) begin
                        in_a_exp <= adapted_rd_exp;
                        in_a_mant <= adapted_rd_mant;
                        // 准备计算
                        cnt_issue <= 4'd0;
                        cnt_collect <= 4'd0;
                        global_max_exp <= 8'd0; // Reset max exp
                        pipe_delay_sr <= 3'd0;
                    end
                end

                // --- 核心计算 (流水线) ---
                S_EXECUTE: begin
                    // 1. 发射逻辑 (Issue)
                    if (cnt_issue < COMPRESSED_DIM) begin
                        bfp_en <= 1'b1;
                        cnt_issue <= cnt_issue + 1'b1;
                    end

                    // 2. 收集逻辑 (Collect)
                    // bfp_valid_out 为高表示加法器输出有效
                    if (bfp_valid_out) begin
                        // 存储中间结果
                        temp_mant_arr[cnt_collect] <= bfp_res_mant;
                        temp_exp_arr[cnt_collect] <= bfp_res_exp;

                        // **关键步骤：动态更新最大指数**
                        if (cnt_collect == 0) begin
                            global_max_exp <= bfp_res_exp;
                        end else begin
                            if (bfp_res_exp > global_max_exp)
                                global_max_exp <= bfp_res_exp;
                        end

                        cnt_collect <= cnt_collect + 1'b1;
                    end
                end

                // --- 统一对齐与写回 ---
                // S_ALIGN 是组合逻辑状态，这里不需要时序操作，直接跳到写
                
                S_WRITE: begin
                    gated_wr_en <= 1'b1;
                    gated_wr_addr <= token_cnt;
                    
                    if (is_layer0) begin
                        // 直通模式直接写回输入 Z
                        gated_wr_exp <= in_z_exp;
                        gated_wr_mant <= in_z_mant;
                    end else begin
                        // 门控模式写回对齐后的结果
                        gated_wr_exp <= global_max_exp;     // 这一组的公共指数
                        gated_wr_mant <= aligned_mant_vector; // 对齐后的尾数向量
                    end
                    
                    dbg_token_count <= dbg_token_count + 1;
                end

                S_NEXT: token_cnt <= token_cnt + 1;
                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    // valid 信号生成 (延迟2拍)
    // pipe_delay_sr[0] -> delay 1
    // pipe_delay_sr[1] -> delay 2 (matches bfp_adder latency)
    assign bfp_valid_out = pipe_delay_sr[1];

    assign dbg_state = state;

    //==========================================================================
    // BFP 加法器实例化
    //==========================================================================
    bfp_adder #(
        .EXP_WIDTH(EXP_WIDTH),
        .MANT_WIDTH(DATA_WIDTH)
    ) u_bfp_adder (
        .clk      (clk),
        .rst_n    (rst_n),
        .enable   (bfp_en),
        .flush    (state == S_WAIT_A), // 计算前清空
        
        // Input A (Weighted Z)
        .sign_a   (1'b0),          // 假设为正数/幅值
        .exp_a    (in_z_exp),      // 原始指数
        .mant_a   (mult_z_res),    // 乘法后的尾数
        .zero_a   (mult_z_res == 0),
        
        // Input B (Weighted A)
        .sign_b   (1'b0),
        .exp_b    (in_a_exp),      // 原始指数
        .mant_b   (mult_a_res),    // 乘法后的尾数
        .zero_b   (mult_a_res == 0),
        
        // Output
        .sign_out (),
        .exp_out  (bfp_res_exp),
        .mant_out (bfp_res_mant),
        .zero_out ()
    );

endmodule