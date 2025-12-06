`timescale 1ns / 1ps

//================================================================================
// DFA Matrix Bank - B矩阵生成与存储 (二分类优化版 - 已修复地址逻辑)
// 存储映射：
// • B₀-B₃: 8×1 = 8 elements each
// • B₄: 32×1 = 32 elements
// • 总计：112 elements
//
// 修正说明：
// 1. 读取逻辑修改为直接映射行地址 (Row Address -> Mem Address)
// 2. 初始化写入计数器限制修正为实际深度 (80->8, 320->32)
//================================================================================

module dfa_matrix_bank #(
    parameter NUM_CLASSES       = 1,        // 分类数（二分类=1）
    parameter LAYER0_DIM        = 8,        // Layer 0-3维度
    parameter LAYER4_DIM        = 32,       // Layer 4维度
    parameter DATA_WIDTH        = 16,       // 数据位宽 (Q4.12)
    parameter LFSR_SEED         = 32'hACE1_BABE  // LFSR初始种子
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 初始化控制接口
    //==========================================================================
    input  wire         init_start,       // 开始初始化
    output reg          init_done,        // 初始化完成
    output reg          matrices_exist,   // 矩阵已存在标志
    
    //==========================================================================
    // GCU 0 读接口 (Layer 0)
    //==========================================================================
    input  wire         gcu0_rd_en,
    input  wire [2:0]   gcu0_row_addr,    // 0-7
    input  wire [3:0]   gcu0_col_addr,    // 0-9 (二分类模式下忽略此输入)
    output reg  [DATA_WIDTH-1:0] gcu0_data,
    output reg          gcu0_valid,
    
    //==========================================================================
    // GCU 1 读接口 (Layer 1)
    //==========================================================================
    input  wire         gcu1_rd_en,
    input  wire [2:0]   gcu1_row_addr,
    input  wire [3:0]   gcu1_col_addr,
    output reg  [DATA_WIDTH-1:0] gcu1_data,
    output reg          gcu1_valid,
    
    //==========================================================================
    // GCU 2 读接口 (Layer 2)
    //==========================================================================
    input  wire         gcu2_rd_en,
    input  wire [2:0]   gcu2_row_addr,
    input  wire [3:0]   gcu2_col_addr,
    output reg  [DATA_WIDTH-1:0] gcu2_data,
    output reg          gcu2_valid,
    
    //==========================================================================
    // GCU 3 读接口 (Layer 3)
    //==========================================================================
    input  wire         gcu3_rd_en,
    input  wire [2:0]   gcu3_row_addr,
    input  wire [3:0]   gcu3_col_addr,
    output reg  [DATA_WIDTH-1:0] gcu3_data,
    output reg          gcu3_valid,
    
    //==========================================================================
    // GCU 4 读接口 (Layer 4)
    //==========================================================================
    input  wire         gcu4_rd_en,
    input  wire [4:0]   gcu4_row_addr,    // 0-31
    input  wire [3:0]   gcu4_col_addr,    // 0-9
    output reg  [DATA_WIDTH-1:0] gcu4_data,
    output reg          gcu4_valid,
    
    //==========================================================================
    // Delta Calculator 读接口
    //==========================================================================
    input  wire         delta_rd_en,
    input  wire [2:0]   delta_layer_id,   // 层ID (0-4)
    input  wire [4:0]   delta_row_addr,   // 行地址 (0-31)
    input  wire [3:0]   delta_col_addr,   // 列地址 (0-9)
    output reg  [DATA_WIDTH-1:0] delta_data,
    output reg          delta_valid,
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [3:0]   init_state,       // 初始化状态
    output wire [31:0]  init_counter      // 初始化计数器
);

//================================================================================
// 初始化状态机
//================================================================================
localparam INIT_IDLE    = 4'd0;
localparam INIT_B0      = 4'd1;
localparam INIT_B1      = 4'd2;
localparam INIT_B2      = 4'd3;
localparam INIT_B3      = 4'd4;
localparam INIT_B4      = 4'd5;
localparam INIT_DONE    = 4'd6;

reg [3:0]  init_state_reg, init_state_next;
reg [8:0]  init_cnt_reg, init_cnt_next;      // 最大32（Layer 4）

//================================================================================
// LFSR随机数生成器
//================================================================================
reg [31:0] lfsr_reg;
wire [31:0] lfsr_next;

// LFSR反馈多项式：x^32 + x^22 + x^2 + x + 1
assign lfsr_next = {lfsr_reg[30:0], lfsr_reg[31] ^ lfsr_reg[21] ^ lfsr_reg[1] ^ lfsr_reg[0]};

// 从LFSR输出生成B矩阵元素（范围[-0.1, 0.1]）
wire [15:0] b_element;
wire [15:0] lfsr_16bit;
wire [15:0] normalized;
wire [19:0] scaled_temp;

assign lfsr_16bit = lfsr_reg[15:0];
// 归一化到[-32768, 32767]然后转换到[-1, 1]
assign normalized = lfsr_16bit - 16'd32768;
// 乘以13然后右移7位得到约0.1的缩放 (13/128 ≈ 0.1016)
assign scaled_temp = $signed(normalized) * $signed(5'd13);
// 算术右移7位
assign b_element = scaled_temp[19:4];  // 取[19:4]相当于右移4位，再配合除以8得到总共右移7位

//================================================================================
// 存储器：BRAM实现（二分类优化）
//================================================================================
// B0-B3: 每个8×1 = 8个元素
reg [DATA_WIDTH-1:0] b0_mem [0:7];
reg [DATA_WIDTH-1:0] b1_mem [0:7];
reg [DATA_WIDTH-1:0] b2_mem [0:7];
reg [DATA_WIDTH-1:0] b3_mem [0:7];

// B4: 32×1 = 32个元素
reg [DATA_WIDTH-1:0] b4_mem [0:31];

//================================================================================
// 初始化状态机
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        init_state_reg <= INIT_IDLE;
        init_cnt_reg <= 9'd0;
        lfsr_reg <= LFSR_SEED;
        matrices_exist <= 1'b0;
        init_done <= 1'b0;
    end else begin
        init_state_reg <= init_state_next;
        init_cnt_reg <= init_cnt_next;
        
        // LFSR更新
        if (init_state_reg != INIT_IDLE && init_state_reg != INIT_DONE) begin
            lfsr_reg <= lfsr_next;
        end
        
        // 初始化完成标志
        if (init_state_reg == INIT_DONE) begin
            matrices_exist <= 1'b1;
            init_done <= 1'b1;
        end else begin
            init_done <= 1'b0;
        end
    end
end

// 状态机组合逻辑和存储器写入
always @(*) begin
    init_state_next = init_state_reg;
    init_cnt_next = init_cnt_reg;
    
    case (init_state_reg)
        INIT_IDLE: begin
            if (init_start && !matrices_exist) begin
                init_state_next = INIT_B0;
                init_cnt_next = 9'd0;
            end
        end
        
        INIT_B0: begin
            // 初始化B0（8个元素）
            if (init_cnt_reg < 9'd8) begin
                init_cnt_next = init_cnt_reg + 9'd1;
            end else begin
                init_state_next = INIT_B1;
                init_cnt_next = 9'd0;
            end
        end
        
        INIT_B1: begin
            // 初始化B1（8个元素）
            if (init_cnt_reg < 9'd8) begin
                init_cnt_next = init_cnt_reg + 9'd1;
            end else begin
                init_state_next = INIT_B2;
                init_cnt_next = 9'd0;
            end
        end
        
        INIT_B2: begin
            // 初始化B2（8个元素）
            if (init_cnt_reg < 9'd8) begin
                init_cnt_next = init_cnt_reg + 9'd1;
            end else begin
                init_state_next = INIT_B3;
                init_cnt_next = 9'd0;
            end
        end
        
        INIT_B3: begin
            // 初始化B3（8个元素）
            if (init_cnt_reg < 9'd8) begin
                init_cnt_next = init_cnt_reg + 9'd1;
            end else begin
                init_state_next = INIT_B4;
                init_cnt_next = 9'd0;
            end
        end
        
        INIT_B4: begin
            // 初始化B4（32个元素）
            if (init_cnt_reg < 9'd32) begin
                init_cnt_next = init_cnt_reg + 9'd1;
            end else begin
                init_state_next = INIT_DONE;
                init_cnt_next = 9'd0;
            end
        end
        
        INIT_DONE: begin
            init_state_next = INIT_IDLE;
        end
        
        default: begin
            init_state_next = INIT_IDLE;
        end
    endcase
end

// 存储器写入逻辑
// 修正：循环次数从80/320改为8/32，匹配N x 1结构
always @(posedge clk) begin
    case (init_state_reg)
        INIT_B0: begin
            if (init_cnt_reg < 9'd8) begin
                b0_mem[init_cnt_reg[2:0]] <= b_element;
            end
        end
        INIT_B1: begin
            if (init_cnt_reg < 9'd8) begin
                b1_mem[init_cnt_reg[2:0]] <= b_element;
            end
        end
        INIT_B2: begin
            if (init_cnt_reg < 9'd8) begin
                b2_mem[init_cnt_reg[2:0]] <= b_element;
            end
        end
        INIT_B3: begin
            if (init_cnt_reg < 9'd8) begin
                b3_mem[init_cnt_reg[2:0]] <= b_element;
            end
        end
        INIT_B4: begin
            if (init_cnt_reg < 9'd32) begin
                b4_mem[init_cnt_reg[4:0]] <= b_element;
            end
        end
    endcase
end

//================================================================================
// 读取逻辑 - 5个并行端口
//================================================================================
// 修正：二分类(Nx1)模式下，内存索引直接等于行地址(Row Address)，忽略列地址
// 地址计算：addr = row (因为 col 恒为 0)

wire [2:0] gcu0_addr_opt = gcu0_row_addr;
wire [2:0] gcu1_addr_opt = gcu1_row_addr;
wire [2:0] gcu2_addr_opt = gcu2_row_addr;
wire [2:0] gcu3_addr_opt = gcu3_row_addr;
wire [4:0] gcu4_addr_opt = gcu4_row_addr;

// GCU 0读取（B0）
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        gcu0_data <= 16'd0;
        gcu0_valid <= 1'b0;
    end else if (gcu0_rd_en && matrices_exist) begin
        gcu0_data <= b0_mem[gcu0_addr_opt];
        gcu0_valid <= 1'b1;
    end else begin
        gcu0_valid <= 1'b0;
    end
end

// GCU 1读取（B1）
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        gcu1_data <= 16'd0;
        gcu1_valid <= 1'b0;
    end else if (gcu1_rd_en && matrices_exist) begin
        gcu1_data <= b1_mem[gcu1_addr_opt];
        gcu1_valid <= 1'b1;
    end else begin
        gcu1_valid <= 1'b0;
    end
end

// GCU 2读取（B2）
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        gcu2_data <= 16'd0;
        gcu2_valid <= 1'b0;
    end else if (gcu2_rd_en && matrices_exist) begin
        gcu2_data <= b2_mem[gcu2_addr_opt];
        gcu2_valid <= 1'b1;
    end else begin
        gcu2_valid <= 1'b0;
    end
end

// GCU 3读取（B3）
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        gcu3_data <= 16'd0;
        gcu3_valid <= 1'b0;
    end else if (gcu3_rd_en && matrices_exist) begin
        gcu3_data <= b3_mem[gcu3_addr_opt];
        gcu3_valid <= 1'b1;
    end else begin
        gcu3_valid <= 1'b0;
    end
end

// GCU 4读取（B4）
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        gcu4_data <= 16'd0;
        gcu4_valid <= 1'b0;
    end else if (gcu4_rd_en && matrices_exist) begin
        gcu4_data <= b4_mem[gcu4_addr_opt];
        gcu4_valid <= 1'b1;
    end else begin
        gcu4_valid <= 1'b0;
    end
end

//================================================================================
// Delta Calculator 读取逻辑
//================================================================================
// 修正：Delta读取也改为直接映射行地址
wire [4:0] delta_addr_opt = delta_row_addr;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        delta_data <= 16'd0;
        delta_valid <= 1'b0;
    end else if (delta_rd_en && matrices_exist) begin
        case (delta_layer_id)
            3'd0: begin
                // B0矩阵 (取低3位地址)
                delta_data <= b0_mem[delta_addr_opt[2:0]];
            end
            3'd1: begin
                // B1矩阵
                delta_data <= b1_mem[delta_addr_opt[2:0]];
            end
            3'd2: begin
                // B2矩阵
                delta_data <= b2_mem[delta_addr_opt[2:0]];
            end
            3'd3: begin
                // B3矩阵
                delta_data <= b3_mem[delta_addr_opt[2:0]];
            end
            3'd4: begin
                // B4矩阵 (32深，使用完整5位地址)
                delta_data <= b4_mem[delta_addr_opt];
            end
            default: begin
                delta_data <= 16'd0;
            end
        endcase
        delta_valid <= 1'b1;
    end else begin
        delta_valid <= 1'b0;
    end
end

//================================================================================
// 调试输出
//================================================================================
assign init_state = init_state_reg;
assign init_counter = {23'd0, init_cnt_reg};

endmodule