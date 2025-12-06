`timescale 1ns / 1ps

//================================================================================
// DFA Matrix Bank v2 - 仅用于 DFA 训练 (Layer0~3, Nx1, 二分类)
//
// - 支持 4 个小层：layer_id = 0..3，每层 8 维 → 总 32 个元素
// - 接口：单读口 rd_en + layer_id + row_addr + col_addr → rd_data/rd_valid
// - 初始化：沿用原 LFSR 逻辑，产生 [-0.1,0.1] 区间随机数
//
// 说明：
//   * 不再包含 GCU0..4 / delta 的多路读口；
//   * expand / layer4 不在这里训练，故不存储 B4；
//   * 读协议：
//       cycle N : rd_en=1, layer_id, row_addr 给出
//       cycle N+1 : rd_valid=1, rd_data 输出
//================================================================================
module dfa_matrix_bank_v2 #(
    parameter NUM_LAYERS  = 4,              // 仅 layer0~3
    parameter LAYER_DIM   = 8,              // 每层 8 维
    parameter DATA_WIDTH  = 16,             // Q4.12
    parameter LFSR_SEED   = 32'hACE1_BABE   // LFSR 初始种子
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire                   clk,
    input  wire                   rst_n,

    //==========================================================================
    // 初始化控制接口
    //==========================================================================
    input  wire                   init_start,      // 开始初始化
    output reg                    init_done,       // 初始化完成
    output reg                    matrices_exist,  // 矩阵已存在

    // 调试
    output wire [3:0]             init_state,
    output wire [31:0]            init_counter,

    //==========================================================================
    // 统一读口（供 gcu_core_v2 使用）
    //==========================================================================
    input  wire                   rd_en,
    input  wire [1:0]             layer_id,    // 0..3
    input  wire [2:0]             row_addr,    // 
    output reg  [DATA_WIDTH-1:0]  rd_data,
    output reg                    rd_valid
);

    //==========================================================================
    // 常量与存储器
    //==========================================================================
    localparam integer TOTAL_ELEMS   = NUM_LAYERS * LAYER_DIM;  // 4*8=32
    localparam integer ADDR_WIDTH    = 6; // log2(32) = 5，这里略放宽一点

    // BRAM 风格存储
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] b_mem [0:TOTAL_ELEMS-1];

    // 计算行地址实际使用的下标（只用低 3 位）
    wire [2:0] row_index = row_addr[2:0];

    // 打包地址：addr = layer_id * LAYER_DIM + row_index
    wire [ADDR_WIDTH-1:0] addr_next =
        (layer_id * LAYER_DIM[ADDR_WIDTH-1:0]) + row_index;

    // 读地址寄存
    reg [ADDR_WIDTH-1:0] rd_addr_reg;
    reg                  rd_pending_reg;

    //==========================================================================
    // LFSR 随机数生成器（沿用原版）
    //==========================================================================
    reg  [31:0] lfsr_reg;
    wire [31:0] lfsr_next;

    // LFSR反馈多项式：x^32 + x^22 + x^2 + x + 1
    assign lfsr_next = {lfsr_reg[30:0],
                        lfsr_reg[31] ^ lfsr_reg[21] ^ lfsr_reg[1] ^ lfsr_reg[0]};

    // 从 LFSR 输出生成 B 矩阵元素（范围[-0.1, 0.1]）
    wire [15:0] b_element;
    wire [15:0] lfsr_16bit;
    wire [15:0] normalized;
    wire [19:0] scaled_temp;

    assign lfsr_16bit = lfsr_reg[15:0];
    // 归一化到[-32768, 32767]然后转换到[-1, 1]
    assign normalized = lfsr_16bit - 16'd32768;
    // 乘以13然后右移约 7 位得到 ~0.1 的缩放
    assign scaled_temp = $signed(normalized) * $signed(5'd13);
    // 这里与原代码保持一致，直接取 [19:4]
    assign b_element = scaled_temp[19:4];

    //==========================================================================
    // 初始化状态机（简单版本）
    //==========================================================================
    localparam INIT_IDLE  = 4'd0;
    localparam INIT_RUN   = 4'd1;
    localparam INIT_DONE  = 4'd2;

    reg [3:0] init_state_reg, init_state_next;
    reg [7:0] init_cnt_reg,   init_cnt_next;   // 足够覆盖 0..31

    // 状态寄存器
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            init_state_reg  <= INIT_IDLE;
            init_cnt_reg    <= 8'd0;
            lfsr_reg        <= LFSR_SEED;
            matrices_exist  <= 1'b0;
            init_done       <= 1'b0;
        end else begin
            init_state_reg  <= init_state_next;
            init_cnt_reg    <= init_cnt_next;

            // LFSR 更新：仅在填充阶段更新
            if (init_state_reg == INIT_RUN) begin
                lfsr_reg <= lfsr_next;
            end

            if (init_state_reg == INIT_DONE) begin
                matrices_exist <= 1'b1;
                init_done      <= 1'b1;
            end else begin
                init_done      <= 1'b0;
            end
        end
    end

    // 状态转移 & 计数
    always @(*) begin
        init_state_next = init_state_reg;
        init_cnt_next   = init_cnt_reg;

        case (init_state_reg)
            INIT_IDLE: begin
                if (init_start && !matrices_exist) begin
                    init_state_next = INIT_RUN;
                    init_cnt_next   = 8'd0;
                end
            end

            INIT_RUN: begin
                if (init_cnt_reg < TOTAL_ELEMS[7:0]) begin
                    init_cnt_next = init_cnt_reg + 8'd1;
                end else begin
                    init_state_next = INIT_DONE;
                    init_cnt_next   = 8'd0;
                end
            end

            INIT_DONE: begin
                // 回到 IDLE，允许下一次重新 init（如果外面想重置，可以清 matrices_exist）
                init_state_next = INIT_IDLE;
            end

            default: begin
                init_state_next = INIT_IDLE;
            end
        endcase
    end

    // 存储器写入（按线性地址 0..TOTAL_ELEMS-1 写入）
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 一般不需要清零 BRAM 内容
        end else if (init_state_reg == INIT_RUN &&
                     init_cnt_reg < TOTAL_ELEMS[7:0]) begin
            b_mem[init_cnt_reg] <= b_element;
        end
    end

    //==========================================================================
    // 读逻辑：1-cycle latency
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_addr_reg     <= {ADDR_WIDTH{1'b0}};
            rd_pending_reg  <= 1'b0;
            rd_data         <= {DATA_WIDTH{1'b0}};
            rd_valid        <= 1'b0;
        end else begin
            // 默认拉低 valid
            rd_valid <= 1'b0;

            // 接收读请求
            if (rd_en && matrices_exist &&
                (layer_id < NUM_LAYERS)) begin        
                rd_addr_reg    <= addr_next;
                rd_pending_reg <= 1'b1;
            end else begin
                rd_pending_reg <= 1'b0;
            end

            // 下一拍输出数据
            if (rd_pending_reg) begin
                rd_data  <= b_mem[rd_addr_reg];
                rd_valid <= 1'b1;
            end
        end
    end

    //==========================================================================
    // 调试输出
    //==========================================================================
    assign init_state   = init_state_reg;
    assign init_counter = {24'd0, init_cnt_reg};

endmodule
