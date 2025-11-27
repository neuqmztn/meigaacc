`timescale 1ns / 1ps

//================================================================================
// Sidenet Weight Storage v3.1
// 
// 主要变更 (v3.1):
// - 添加FFN权重存储 (W1和W2)
// - W1: 8×32矩阵, 32个列指数, 32 bursts
// - W2: 32×8矩阵, 8个列指数, 16 bursts
//
// 主要变更 (v3.0):
// - 批量读写替代逐权重访问 (对齐backbone设计)
// - 支持多指数BFP: 每列共享一个指数
// - 列优先存储: 便于并行计算引擎访问
//
// 存储组织 (以Compression 32×8为例):
//   矩阵: Y = X × W，其中W是32×8权重矩阵
//   
//   W = [w00  w01  w02  ...  w07]  ← 32行
//       [w10  w11  w12  ...  w17]
//       [w20  w21  w22  ...  w27]
//       ...
//       [w310 w311 w312 ... w317]
//        ^    ^    ^         ^
//        列0  列1  列2       列7
//
//   每列独立共享指数:
//   - 列0: exp[0] 对应 w00, w10, w20, ..., w310 (32个权重)
//   - 列1: exp[1] 对应 w01, w11, w21, ..., w311
//   - ...
//   - 列7: exp[7] 对应 w07, w17, w27, ..., w317
//
// 列优先存储布局:
//   SRAM按列组织，每列的权重连续存储
//   
//   列0数据: [w00, w10, w20, ..., w150, w160, ..., w310] (32个)
//   列1数据: [w01, w11, w21, ..., w151, w161, ..., w311]
//   ...
//   列7数据: [w07, w17, w27, ..., w157, w167, ..., w317]
//
// Burst传输格式 (256 bits = 16个16-bit权重):
//   每列需要2个burst (32个权重 / 16 = 2)
//   
//   Compression total: 8列 × 2 burst/列 = 16 bursts
//   - Burst 0:  列0前半 [w00...w150]
//   - Burst 1:  列0后半 [w160...w310]
//   - Burst 2:  列1前半 [w01...w151]
//   - Burst 3:  列1后半 [w161...w311]
//   - ...
//   - Burst 14: 列7前半 [w07...w157]
//   - Burst 15: 列7后半 [w167...w317]
//
//   Attention total: 8列 × 4 burst/列 = 4 bursts (8×8矩阵)
//   Expand total: 32列 × 2 burst/列 = 64 bursts? 不对，Expand是8×32
//   
//   等等，Expand是8行32列，应该是32个输出维度，每个输出维度8个权重
//   所以Expand只需要1个burst/列 (8个权重 < 16)，但实际会用满16个位置
//
// 双BANK设计:
//   - 推理时读当前BANK (例如BANK A)
//   - 训练时写备用BANK (例如BANK B)
//   - Epoch结束后切换BANK
//
// 作者: MEIGA Team
// 日期: 2025-11-16
// 版本: v3.0
//================================================================================

module sidenet_weight_storage #(
    parameter NUM_LAYERS       = 5,
    parameter BACKBONE_DIM     = 32,
    parameter SIDENET_DIM      = 8,
    parameter DATA_WIDTH       = 16,
    parameter EXP_WIDTH        = 8,
    parameter DRAM_DATA_WIDTH  = 256,
    
    // 每个burst能装多少个权重
    parameter WEIGHTS_PER_BURST = DRAM_DATA_WIDTH / DATA_WIDTH,  // 256/16 = 16
    
    // 最大指数数组位宽 (Expand需要32个指数)
    parameter MAX_EXP_ARRAY_WIDTH = BACKBONE_DIM * EXP_WIDTH     // 32×8 = 256
)(
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // BANK切换
    //==========================================================================
    input  wire bank_swap,
    output wire current_read_bank,    // 0=A, 1=B
    output wire current_write_bank,   // 0=A, 1=B
    
    //==========================================================================
    // 读接口 - 批量读取
    //==========================================================================
    input  wire rd_en,
    input  wire [2:0] rd_layer_id,        // 0-4
    input  wire [3:0] rd_weight_type,     // 见下方定义
    input  wire [5:0] rd_burst_idx,       // Burst索引 (0-63，不同权重类型范围不同)
    
    output reg  rd_valid,
    output reg  [MAX_EXP_ARRAY_WIDTH-1:0] rd_exp_array,    // 完整指数数组
    output reg  [DRAM_DATA_WIDTH-1:0] rd_data_burst,       // 一个数据burst
    
    //==========================================================================
    // 写接口 - 批量写入
    //==========================================================================
    input  wire wr_en,
    input  wire [2:0] wr_layer_id,
    input  wire [3:0] wr_weight_type,
    input  wire [5:0] wr_burst_idx,
    input  wire [MAX_EXP_ARRAY_WIDTH-1:0] wr_exp_array,
    input  wire [DRAM_DATA_WIDTH-1:0] wr_data_burst,
    output reg  wr_ready,
    
    //==========================================================================
    // DMA初始化 - 从DRAM加载权重
    //==========================================================================
    input  wire dma_init_en,
    input  wire dma_init_bank,            // 0=A, 1=B
    input  wire [2:0] dma_init_layer,
    input  wire [3:0] dma_init_weight_type,
    input  wire [5:0] dma_init_burst_idx,
    input  wire [DRAM_DATA_WIDTH-1:0] dma_init_data,
    
    //==========================================================================
    // 调试
    //==========================================================================
    output wire [31:0] dbg_read_count,
    output wire [31:0] dbg_write_count,
    output wire [31:0] dbg_bank_swap_count,

    //==========================================================================
    // DFA / WUE 按元素读接口（只读当前 read bank）
    //==========================================================================
    input  wire        dfa_rd_en,
    input  wire [2:0]  dfa_rd_layer_id,
    input  wire [3:0]  dfa_rd_weight_type,
    input  wire [4:0]  dfa_rd_col_id,   // 0..31
    input  wire [4:0]  dfa_rd_row_id,   // 0..31
    output reg         dfa_rd_valid,
    output reg  [EXP_WIDTH-1:0]  dfa_rd_exp,
    output reg  [DATA_WIDTH-1:0] dfa_rd_mant
);

//================================================================================
// 权重类型定义
//================================================================================
// 注意: 编码保持与controller一致
localparam WEIGHT_COMPRESS = 4'd0;  // 32×8, 8个指数
localparam WEIGHT_GATE     = 4'd1;  // 16×8, 8个指数 (暂未使用)
localparam WEIGHT_ATT_WQ   = 4'd2;  // 8×8, 8个指数
localparam WEIGHT_ATT_WK   = 4'd3;
localparam WEIGHT_ATT_WV   = 4'd4;
localparam WEIGHT_ATT_WO   = 4'd5;
localparam WEIGHT_FFN_W1   = 4'd6;  // 8×32, 32个指数 (FFN Linear1)
localparam WEIGHT_FFN_W2   = 4'd7;  // 32×8, 8个指数 (FFN Linear2)
localparam WEIGHT_EXPAND   = 4'd8;  // 8×32, 32个指数 (Layer 4专用)

//================================================================================
// 存储结构 - BANK A
//================================================================================
// 组织原则: 按层、按权重类型、按列存储
// 每列的数据连续存储，便于PU并行访问

// Layer 0 - Compression (32×8矩阵)
// 8列，每列32个权重，分2个burst存储
reg [EXP_WIDTH-1:0]  bank_a_l0_compress_exp [0:7];                    // 8个列指数
reg [DATA_WIDTH-1:0] bank_a_l0_compress_col [0:7][0:1][0:15];        // [列][burst][权重]
//                                             ^^^  ^^^  ^^^^
//                                             8列  2 burst/列  16 weights/burst

// Layer 1-3 - Compression (每层32×8)
reg [EXP_WIDTH-1:0]  bank_a_l13_compress_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_a_l13_compress_col [0:2][0:7][0:1][0:15];  // [层][列][burst][权重]

// Layer 1-3 - Attention WQ/WK/WV/WO (每个8×8)
// 8列，每列8个权重，占1个burst (前8个位置，后8个位置空置)
reg [EXP_WIDTH-1:0]  bank_a_l13_wq_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_a_l13_wq_col [0:2][0:7][0:15];             // [层][列][权重]
//                                        ^^^  ^^^  ^^^^
//                                        3层  8列  单burst足够(只用前8个)

reg [EXP_WIDTH-1:0]  bank_a_l13_wk_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_a_l13_wk_col [0:2][0:7][0:15];

reg [EXP_WIDTH-1:0]  bank_a_l13_wv_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_a_l13_wv_col [0:2][0:7][0:15];

reg [EXP_WIDTH-1:0]  bank_a_l13_wo_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_a_l13_wo_col [0:2][0:7][0:15];

// Layer 1-3 - FFN W1 (8×32矩阵)
// 32列，每列8个权重，占1个burst
reg [EXP_WIDTH-1:0]  bank_a_l13_ffn_w1_exp [0:2][0:31];       // [层][列] - 32个指数
reg [DATA_WIDTH-1:0] bank_a_l13_ffn_w1_col [0:2][0:31][0:15]; // [层][列][权重]
//                                            ^^^  ^^^^  ^^^^
//                                            3层  32列  单burst(只用前8个)

// Layer 1-3 - FFN W2 (32×8矩阵)
// 8列，每列32个权重，占2个burst
reg [EXP_WIDTH-1:0]  bank_a_l13_ffn_w2_exp [0:2][0:7];              // [层][列] - 8个指数
reg [DATA_WIDTH-1:0] bank_a_l13_ffn_w2_col [0:2][0:7][0:1][0:15];  // [层][列][burst][权重]
//                                            ^^^  ^^^  ^^^  ^^^^
//                                            3层  8列  2burst 16weights/burst

// Layer 4 - Compression (32×8)
reg [EXP_WIDTH-1:0]  bank_a_l4_compress_exp [0:7];
reg [DATA_WIDTH-1:0] bank_a_l4_compress_col [0:7][0:1][0:15];

// Layer 4 - Expand (8×32矩阵)
// 32列，每列8个权重，占1个burst
reg [EXP_WIDTH-1:0]  bank_a_l4_expand_exp [0:31];                     // 32个列指数
reg [DATA_WIDTH-1:0] bank_a_l4_expand_col [0:31][0:15];              // [列][权重]
//                                          ^^^^^  ^^^^
//                                          32列   单burst足够(只用前8个)

//================================================================================
// 存储结构 - BANK B (结构同BANK A)
//================================================================================
reg [EXP_WIDTH-1:0]  bank_b_l0_compress_exp [0:7];
reg [DATA_WIDTH-1:0] bank_b_l0_compress_col [0:7][0:1][0:15];

reg [EXP_WIDTH-1:0]  bank_b_l13_compress_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_b_l13_compress_col [0:2][0:7][0:1][0:15];

reg [EXP_WIDTH-1:0]  bank_b_l13_wq_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_b_l13_wq_col [0:2][0:7][0:15];

reg [EXP_WIDTH-1:0]  bank_b_l13_wk_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_b_l13_wk_col [0:2][0:7][0:15];

reg [EXP_WIDTH-1:0]  bank_b_l13_wv_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_b_l13_wv_col [0:2][0:7][0:15];

reg [EXP_WIDTH-1:0]  bank_b_l13_wo_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_b_l13_wo_col [0:2][0:7][0:15];

// Layer 1-3 - FFN W1 (8×32矩阵)
reg [EXP_WIDTH-1:0]  bank_b_l13_ffn_w1_exp [0:2][0:31];
reg [DATA_WIDTH-1:0] bank_b_l13_ffn_w1_col [0:2][0:31][0:15];

// Layer 1-3 - FFN W2 (32×8矩阵)
reg [EXP_WIDTH-1:0]  bank_b_l13_ffn_w2_exp [0:2][0:7];
reg [DATA_WIDTH-1:0] bank_b_l13_ffn_w2_col [0:2][0:7][0:1][0:15];

// Layer 4 - Compression (32×8)
reg [EXP_WIDTH-1:0]  bank_b_l4_compress_exp [0:7];
reg [DATA_WIDTH-1:0] bank_b_l4_compress_col [0:7][0:1][0:15];

reg [EXP_WIDTH-1:0]  bank_b_l4_expand_exp [0:31];
reg [DATA_WIDTH-1:0] bank_b_l4_expand_col [0:31][0:15];

//================================================================================
// BANK切换逻辑
//================================================================================
reg bank_select_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bank_select_reg <= 1'b0;
    end else if (bank_swap) begin
        bank_select_reg <= ~bank_select_reg;
    end
end

assign current_read_bank  = bank_select_reg;
assign current_write_bank = ~bank_select_reg;

//================================================================================
// 调试计数
//================================================================================
reg [31:0] read_count;
reg [31:0] write_count;
reg [31:0] bank_swap_count;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        read_count      <= 32'd0;
        write_count     <= 32'd0;
        bank_swap_count <= 32'd0;
    end else begin
        if (rd_en) read_count <= read_count + 1;
        if (wr_en) write_count <= write_count + 1;
        if (bank_swap) bank_swap_count <= bank_swap_count + 1;
    end
end

assign dbg_read_count      = read_count;
assign dbg_write_count     = write_count;
assign dbg_bank_swap_count = bank_swap_count;

//================================================================================
// 读逻辑
//
// burst_idx含义 (以Compression为例):
//   Burst 0-1:   列0的数据 (burst_idx[3:1]=0, burst_idx[0]=0/1)
//   Burst 2-3:   列1的数据 (burst_idx[3:1]=1, burst_idx[0]=0/1)
//   ...
//   Burst 14-15: 列7的数据 (burst_idx[3:1]=7, burst_idx[0]=0/1)
//
// 对于Attention (8×8):
//   Burst 0: 列0 (burst_idx[2:0]=0)
//   Burst 1: 列1 (burst_idx[2:0]=1)
//   ...
//   Burst 7: 列7 (burst_idx[2:0]=7)
//
// 对于Expand (8×32):
//   Burst 0: 列0 (burst_idx[4:0]=0)
//   ...
//   Burst 31: 列31 (burst_idx[4:0]=31)
//================================================================================

integer rd_i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_valid <= 1'b0;
        rd_exp_array <= {MAX_EXP_ARRAY_WIDTH{1'b0}};
        rd_data_burst <= {DRAM_DATA_WIDTH{1'b0}};
    end else begin
        rd_valid <= rd_en;
        
        if (rd_en) begin
            // 默认清零
            rd_exp_array <= {MAX_EXP_ARRAY_WIDTH{1'b0}};
            rd_data_burst <= {DRAM_DATA_WIDTH{1'b0}};
            
            if (current_read_bank == 1'b0) begin
                // 从BANK A读取
                case (rd_layer_id)
                    3'd0: begin
                        if (rd_weight_type == WEIGHT_COMPRESS) begin
                            // Compression 32×8: burst_idx[3:1]=列号, burst_idx[0]=列内burst号
                            // 提取列号和列内burst号
                            // 读取指数数组 (8个指数)
                            for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH] 
                                    <= bank_a_l0_compress_exp[rd_i];
                            end
                            // 读取数据burst
                            for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                    <= bank_a_l0_compress_col[rd_burst_idx[3:1]][rd_burst_idx[0]][rd_i];
                            end
                        end
                    end
                    
                    3'd1, 3'd2, 3'd3: begin
                        case (rd_weight_type)
                            WEIGHT_COMPRESS: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_a_l13_compress_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_a_l13_compress_col[rd_layer_id-1][rd_burst_idx[3:1]][rd_burst_idx[0]][rd_i];
                                end
                            end
                            
                            WEIGHT_ATT_WQ: begin
                                // Attention 8×8: burst_idx[2:0]=列号, 单burst足够
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_a_l13_wq_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_a_l13_wq_col[rd_layer_id-1][rd_burst_idx[2:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_ATT_WK: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_a_l13_wk_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_a_l13_wk_col[rd_layer_id-1][rd_burst_idx[2:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_ATT_WV: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_a_l13_wv_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_a_l13_wv_col[rd_layer_id-1][rd_burst_idx[2:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_ATT_WO: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_a_l13_wo_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_a_l13_wo_col[rd_layer_id-1][rd_burst_idx[2:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_FFN_W1: begin
                                // FFN W1: 8×32, burst_idx[4:0]=列号 (0-31)
                                // 读取32个列指数
                                for (rd_i = 0; rd_i < 32; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_a_l13_ffn_w1_exp[rd_layer_id-1][rd_i];
                                end
                                // 读取数据burst (该列的8个权重)
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_a_l13_ffn_w1_col[rd_layer_id-1][rd_burst_idx[4:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_FFN_W2: begin
                                // FFN W2: 32×8, burst_idx[3:1]=列号, burst_idx[0]=列内burst号
                                // 读取8个列指数
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_a_l13_ffn_w2_exp[rd_layer_id-1][rd_i];
                                end
                                // 读取数据burst
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_a_l13_ffn_w2_col[rd_layer_id-1][rd_burst_idx[3:1]][rd_burst_idx[0]][rd_i];
                                end
                            end
                        endcase
                    end
                    
                    3'd4: begin
                        case (rd_weight_type)
                            WEIGHT_COMPRESS: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_a_l4_compress_exp[rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_a_l4_compress_col[rd_burst_idx[3:1]][rd_burst_idx[0]][rd_i];
                                end
                            end
                            
                            WEIGHT_EXPAND: begin
                                // Expand 8×32: burst_idx[4:0]=列号, 单burst足够
                                for (rd_i = 0; rd_i < 32; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_a_l4_expand_exp[rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_a_l4_expand_col[rd_burst_idx[4:0]][rd_i];
                                end
                            end
                        endcase
                    end
                endcase
                
            end else begin
                // 从BANK B读取 (逻辑同BANK A)
                case (rd_layer_id)
                    3'd0: begin
                        if (rd_weight_type == WEIGHT_COMPRESS) begin
                            for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH] 
                                    <= bank_b_l0_compress_exp[rd_i];
                            end
                            for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                    <= bank_b_l0_compress_col[rd_burst_idx[3:1]][rd_burst_idx[0]][rd_i];
                            end
                        end
                    end
                    
                    3'd1, 3'd2, 3'd3: begin
                        case (rd_weight_type)
                            WEIGHT_COMPRESS: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_b_l13_compress_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_b_l13_compress_col[rd_layer_id-1][rd_burst_idx[3:1]][rd_burst_idx[0]][rd_i];
                                end
                            end
                            
                            WEIGHT_ATT_WQ: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_b_l13_wq_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_b_l13_wq_col[rd_layer_id-1][rd_burst_idx[2:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_ATT_WK: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_b_l13_wk_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_b_l13_wk_col[rd_layer_id-1][rd_burst_idx[2:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_ATT_WV: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_b_l13_wv_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_b_l13_wv_col[rd_layer_id-1][rd_burst_idx[2:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_ATT_WO: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_b_l13_wo_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_b_l13_wo_col[rd_layer_id-1][rd_burst_idx[2:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_FFN_W1: begin
                                // FFN W1: 8×32, burst_idx[4:0]=列号 (0-31)
                                for (rd_i = 0; rd_i < 32; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_b_l13_ffn_w1_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_b_l13_ffn_w1_col[rd_layer_id-1][rd_burst_idx[4:0]][rd_i];
                                end
                            end
                            
                            WEIGHT_FFN_W2: begin
                                // FFN W2: 32×8, burst_idx[3:1]=列号, burst_idx[0]=列内burst号
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_b_l13_ffn_w2_exp[rd_layer_id-1][rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_b_l13_ffn_w2_col[rd_layer_id-1][rd_burst_idx[3:1]][rd_burst_idx[0]][rd_i];
                                end
                            end
                        endcase
                    end
                    
                    3'd4: begin
                        case (rd_weight_type)
                            WEIGHT_COMPRESS: begin
                                for (rd_i = 0; rd_i < 8; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_b_l4_compress_exp[rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_b_l4_compress_col[rd_burst_idx[3:1]][rd_burst_idx[0]][rd_i];
                                end
                            end
                            
                            WEIGHT_EXPAND: begin
                                for (rd_i = 0; rd_i < 32; rd_i = rd_i + 1) begin
                                    rd_exp_array[rd_i*EXP_WIDTH +: EXP_WIDTH]
                                        <= bank_b_l4_expand_exp[rd_i];
                                end
                                for (rd_i = 0; rd_i < 16; rd_i = rd_i + 1) begin
                                    rd_data_burst[rd_i*DATA_WIDTH +: DATA_WIDTH]
                                        <= bank_b_l4_expand_col[rd_burst_idx[4:0]][rd_i];
                                end
                            end
                        endcase
                    end
                endcase
            end
        end
    end
end
//================================================================================
// DFA / WUE 按元素读逻辑
// - 根据 (dfa_rd_layer_id, dfa_rd_weight_type, dfa_rd_col_id, dfa_rd_row_id)
//   从当前 read bank 读取单个权重的 BFP (exp, mant)
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dfa_rd_valid <= 1'b0;
        dfa_rd_exp   <= {EXP_WIDTH{1'b0}};
        dfa_rd_mant  <= {DATA_WIDTH{1'b0}};
    end else begin
        dfa_rd_valid <= 1'b0;
        dfa_rd_exp   <= {EXP_WIDTH{1'b0}};
        dfa_rd_mant  <= {DATA_WIDTH{1'b0}};

        if (dfa_rd_en) begin
            dfa_rd_valid <= 1'b1;

            // 默认清零，防止case没命中
            dfa_rd_exp   <= {EXP_WIDTH{1'b0}};
            dfa_rd_mant  <= {DATA_WIDTH{1'b0}};

            // 选择当前读BANK
            if (current_read_bank == 1'b0) begin
                //==========================
                //        BANK A
                //==========================
                case (dfa_rd_layer_id)
                    //======================================================
                    // Layer 0
                    //======================================================
                    3'd0: begin
                        case (dfa_rd_weight_type)
                            // Layer 0 - Compression (32×8)
                            WEIGHT_COMPRESS: begin
                                // 列指数
                                dfa_rd_exp <= bank_a_l0_compress_exp[dfa_rd_col_id[2:0]];
                                // 行权重: 2 个 burst, 每个 16 行
                                if (dfa_rd_row_id[4] == 1'b0) begin
                                    dfa_rd_mant <= bank_a_l0_compress_col
                                                   [dfa_rd_col_id[2:0]][0][dfa_rd_row_id[3:0]];
                                end else begin
                                    dfa_rd_mant <= bank_a_l0_compress_col
                                                   [dfa_rd_col_id[2:0]][1][dfa_rd_row_id[3:0]];
                                end
                            end

                            default: begin
                                dfa_rd_exp  <= {EXP_WIDTH{1'b0}};
                                dfa_rd_mant <= {DATA_WIDTH{1'b0}};
                            end
                        endcase
                    end

                    //======================================================
                    // Layer 1-3
                    //======================================================
                    3'd1, 3'd2, 3'd3: begin
                        case (dfa_rd_weight_type)
                            // Layer 1-3 - Compression (32×8)
                            WEIGHT_COMPRESS: begin
                                dfa_rd_exp <= bank_a_l13_compress_exp
                                              [dfa_rd_layer_id-1][dfa_rd_col_id[2:0]];
                                if (dfa_rd_row_id[4] == 1'b0) begin
                                    dfa_rd_mant <= bank_a_l13_compress_col
                                                   [dfa_rd_layer_id-1]
                                                   [dfa_rd_col_id[2:0]][0][dfa_rd_row_id[3:0]];
                                end else begin
                                    dfa_rd_mant <= bank_a_l13_compress_col
                                                   [dfa_rd_layer_id-1]
                                                   [dfa_rd_col_id[2:0]][1][dfa_rd_row_id[3:0]];
                                end
                            end

                            // Layer 1-3 - FFN W1 (8×32)
                            WEIGHT_FFN_W1: begin
                                // 列指数：32 列
                                dfa_rd_exp <= bank_a_l13_ffn_w1_exp
                                              [dfa_rd_layer_id-1][dfa_rd_col_id[4:0]];
                                // 行：8 行，对应 idx 0..7
                                dfa_rd_mant <= bank_a_l13_ffn_w1_col
                                               [dfa_rd_layer_id-1]
                                               [dfa_rd_col_id[4:0]][dfa_rd_row_id[2:0]];
                            end

                            // Layer 1-3 - FFN W2 (32×8)
                            WEIGHT_FFN_W2: begin
                                dfa_rd_exp <= bank_a_l13_ffn_w2_exp
                                              [dfa_rd_layer_id-1][dfa_rd_col_id[2:0]];
                                if (dfa_rd_row_id[4] == 1'b0) begin
                                    dfa_rd_mant <= bank_a_l13_ffn_w2_col
                                                   [dfa_rd_layer_id-1]
                                                   [dfa_rd_col_id[2:0]][0][dfa_rd_row_id[3:0]];
                                end else begin
                                    dfa_rd_mant <= bank_a_l13_ffn_w2_col
                                                   [dfa_rd_layer_id-1]
                                                   [dfa_rd_col_id[2:0]][1][dfa_rd_row_id[3:0]];
                                end
                            end

                            default: begin
                                dfa_rd_exp  <= {EXP_WIDTH{1'b0}};
                                dfa_rd_mant <= {DATA_WIDTH{1'b0}};
                            end
                        endcase
                    end

                    //======================================================
                    // Layer 4
                    //======================================================
                    3'd4: begin
                        case (dfa_rd_weight_type)
                            // Layer 4 - Compression (32×8)
                            WEIGHT_COMPRESS: begin
                                dfa_rd_exp <= bank_a_l4_compress_exp[dfa_rd_col_id[2:0]];
                                if (dfa_rd_row_id[4] == 1'b0) begin
                                    dfa_rd_mant <= bank_a_l4_compress_col
                                                   [dfa_rd_col_id[2:0]][0][dfa_rd_row_id[3:0]];
                                end else begin
                                    dfa_rd_mant <= bank_a_l4_compress_col
                                                   [dfa_rd_col_id[2:0]][1][dfa_rd_row_id[3:0]];
                                end
                            end

                            // Layer 4 - Expand (8×32)
                            WEIGHT_EXPAND: begin
                                dfa_rd_exp <= bank_a_l4_expand_exp[dfa_rd_col_id[4:0]];
                                dfa_rd_mant <= bank_a_l4_expand_col
                                               [dfa_rd_col_id[4:0]][dfa_rd_row_id[2:0]];
                            end

                            default: begin
                                dfa_rd_exp  <= {EXP_WIDTH{1'b0}};
                                dfa_rd_mant <= {DATA_WIDTH{1'b0}};
                            end
                        endcase
                    end

                    default: begin
                        dfa_rd_exp  <= {EXP_WIDTH{1'b0}};
                        dfa_rd_mant <= {DATA_WIDTH{1'b0}};
                    end
                endcase
            end else begin
                //==========================
                //        BANK B
                //==========================
                case (dfa_rd_layer_id)
                    //======================================================
                    // Layer 0
                    //======================================================
                    3'd0: begin
                        case (dfa_rd_weight_type)
                            WEIGHT_COMPRESS: begin
                                dfa_rd_exp <= bank_b_l0_compress_exp[dfa_rd_col_id[2:0]];
                                if (dfa_rd_row_id[4] == 1'b0) begin
                                    dfa_rd_mant <= bank_b_l0_compress_col
                                                   [dfa_rd_col_id[2:0]][0][dfa_rd_row_id[3:0]];
                                end else begin
                                    dfa_rd_mant <= bank_b_l0_compress_col
                                                   [dfa_rd_col_id[2:0]][1][dfa_rd_row_id[3:0]];
                                end
                            end

                            default: begin
                                dfa_rd_exp  <= {EXP_WIDTH{1'b0}};
                                dfa_rd_mant <= {DATA_WIDTH{1'b0}};
                            end
                        endcase
                    end

                    //======================================================
                    // Layer 1-3
                    //======================================================
                    3'd1, 3'd2, 3'd3: begin
                        case (dfa_rd_weight_type)
                            WEIGHT_COMPRESS: begin
                                dfa_rd_exp <= bank_b_l13_compress_exp
                                              [dfa_rd_layer_id-1][dfa_rd_col_id[2:0]];
                                if (dfa_rd_row_id[4] == 1'b0) begin
                                    dfa_rd_mant <= bank_b_l13_compress_col
                                                   [dfa_rd_layer_id-1]
                                                   [dfa_rd_col_id[2:0]][0][dfa_rd_row_id[3:0]];
                                end else begin
                                    dfa_rd_mant <= bank_b_l13_compress_col
                                                   [dfa_rd_layer_id-1]
                                                   [dfa_rd_col_id[2:0]][1][dfa_rd_row_id[3:0]];
                                end
                            end

                            WEIGHT_FFN_W1: begin
                                dfa_rd_exp <= bank_b_l13_ffn_w1_exp
                                              [dfa_rd_layer_id-1][dfa_rd_col_id[4:0]];
                                dfa_rd_mant <= bank_b_l13_ffn_w1_col
                                               [dfa_rd_layer_id-1]
                                               [dfa_rd_col_id[4:0]][dfa_rd_row_id[2:0]];
                            end

                            WEIGHT_FFN_W2: begin
                                dfa_rd_exp <= bank_b_l13_ffn_w2_exp
                                              [dfa_rd_layer_id-1][dfa_rd_col_id[2:0]];
                                if (dfa_rd_row_id[4] == 1'b0) begin
                                    dfa_rd_mant <= bank_b_l13_ffn_w2_col
                                                   [dfa_rd_layer_id-1]
                                                   [dfa_rd_col_id[2:0]][0][dfa_rd_row_id[3:0]];
                                end else begin
                                    dfa_rd_mant <= bank_b_l13_ffn_w2_col
                                                   [dfa_rd_layer_id-1]
                                                   [dfa_rd_col_id[2:0]][1][dfa_rd_row_id[3:0]];
                                end
                            end

                            default: begin
                                dfa_rd_exp  <= {EXP_WIDTH{1'b0}};
                                dfa_rd_mant <= {DATA_WIDTH{1'b0}};
                            end
                        endcase
                    end

                    //======================================================
                    // Layer 4
                    //======================================================
                    3'd4: begin
                        case (dfa_rd_weight_type)
                            WEIGHT_COMPRESS: begin
                                dfa_rd_exp <= bank_b_l4_compress_exp[dfa_rd_col_id[2:0]];
                                if (dfa_rd_row_id[4] == 1'b0) begin
                                    dfa_rd_mant <= bank_b_l4_compress_col
                                                   [dfa_rd_col_id[2:0]][0][dfa_rd_row_id[3:0]];
                                end else begin
                                    dfa_rd_mant <= bank_b_l4_compress_col
                                                   [dfa_rd_col_id[2:0]][1][dfa_rd_row_id[3:0]];
                                end
                            end

                            WEIGHT_EXPAND: begin
                                dfa_rd_exp  <= bank_b_l4_expand_exp[dfa_rd_col_id[4:0]];
                                dfa_rd_mant <= bank_b_l4_expand_col
                                               [dfa_rd_col_id[4:0]][dfa_rd_row_id[2:0]];
                            end

                            default: begin
                                dfa_rd_exp  <= {EXP_WIDTH{1'b0}};
                                dfa_rd_mant <= {DATA_WIDTH{1'b0}};
                            end
                        endcase
                    end

                    default: begin
                        dfa_rd_exp  <= {EXP_WIDTH{1'b0}};
                        dfa_rd_mant <= {DATA_WIDTH{1'b0}};
                    end
                endcase
            end
        end
    end
end

//================================================================================
// 写逻辑 - 写入备用BANK
//
// 注意: 每次写入都会更新指数数组，虽然指数对所有burst相同
//       这样设计简化了控制逻辑，代价是重复写入指数
//================================================================================

integer wr_i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_ready <= 1'b0;
    end else begin
        wr_ready <= wr_en;
        
        if (wr_en) begin
            if (current_write_bank == 1'b0) begin
                // 写BANK A
                case (wr_layer_id)
                    3'd0: begin
                        if (wr_weight_type == WEIGHT_COMPRESS) begin
                            // 写指数
                            for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                bank_a_l0_compress_exp[wr_i]
                                    <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                            end
                            // 写数据
                            for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                bank_a_l0_compress_col[wr_burst_idx[3:1]][wr_burst_idx[0]][wr_i]
                                    <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                            end
                        end
                    end
                    
                    3'd1, 3'd2, 3'd3: begin
                        case (wr_weight_type)
                            WEIGHT_COMPRESS: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_a_l13_compress_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_a_l13_compress_col[wr_layer_id-1][wr_burst_idx[3:1]][wr_burst_idx[0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_ATT_WQ: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_a_l13_wq_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_a_l13_wq_col[wr_layer_id-1][wr_burst_idx[2:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_ATT_WK: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_a_l13_wk_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_a_l13_wk_col[wr_layer_id-1][wr_burst_idx[2:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_ATT_WV: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_a_l13_wv_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_a_l13_wv_col[wr_layer_id-1][wr_burst_idx[2:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_ATT_WO: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_a_l13_wo_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_a_l13_wo_col[wr_layer_id-1][wr_burst_idx[2:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_FFN_W1: begin
                                // FFN W1: 8×32, burst_idx[4:0]=列号
                                // 写入32个列指数
                                for (wr_i = 0; wr_i < 32; wr_i = wr_i + 1) begin
                                    bank_a_l13_ffn_w1_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                // 写入数据
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_a_l13_ffn_w1_col[wr_layer_id-1][wr_burst_idx[4:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_FFN_W2: begin
                                // FFN W2: 32×8, burst_idx[3:1]=列号, burst_idx[0]=列内burst号
                                // 写入8个列指数
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_a_l13_ffn_w2_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                // 写入数据
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_a_l13_ffn_w2_col[wr_layer_id-1][wr_burst_idx[3:1]][wr_burst_idx[0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                        endcase
                    end
                    
                    3'd4: begin
                        case (wr_weight_type)
                            WEIGHT_COMPRESS: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_a_l4_compress_exp[wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_a_l4_compress_col[wr_burst_idx[3:1]][wr_burst_idx[0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_EXPAND: begin
                                for (wr_i = 0; wr_i < 32; wr_i = wr_i + 1) begin
                                    bank_a_l4_expand_exp[wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_a_l4_expand_col[wr_burst_idx[4:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                        endcase
                    end
                endcase
                
            end else begin
                // 写BANK B (逻辑同BANK A)
                case (wr_layer_id)
                    3'd0: begin
                        if (wr_weight_type == WEIGHT_COMPRESS) begin
                            for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                bank_b_l0_compress_exp[wr_i]
                                    <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                            end
                            for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                bank_b_l0_compress_col[wr_burst_idx[3:1]][wr_burst_idx[0]][wr_i]
                                    <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                            end
                        end
                    end
                    
                    3'd1, 3'd2, 3'd3: begin
                        case (wr_weight_type)
                            WEIGHT_COMPRESS: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_b_l13_compress_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_b_l13_compress_col[wr_layer_id-1][wr_burst_idx[3:1]][wr_burst_idx[0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_ATT_WQ: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_b_l13_wq_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_b_l13_wq_col[wr_layer_id-1][wr_burst_idx[2:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_ATT_WK: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_b_l13_wk_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_b_l13_wk_col[wr_layer_id-1][wr_burst_idx[2:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_ATT_WV: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_b_l13_wv_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_b_l13_wv_col[wr_layer_id-1][wr_burst_idx[2:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_ATT_WO: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_b_l13_wo_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_b_l13_wo_col[wr_layer_id-1][wr_burst_idx[2:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_FFN_W1: begin
                                // FFN W1: 8×32, burst_idx[4:0]=列号
                                for (wr_i = 0; wr_i < 32; wr_i = wr_i + 1) begin
                                    bank_b_l13_ffn_w1_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_b_l13_ffn_w1_col[wr_layer_id-1][wr_burst_idx[4:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_FFN_W2: begin
                                // FFN W2: 32×8, burst_idx[3:1]=列号, burst_idx[0]=列内burst号
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_b_l13_ffn_w2_exp[wr_layer_id-1][wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_b_l13_ffn_w2_col[wr_layer_id-1][wr_burst_idx[3:1]][wr_burst_idx[0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                        endcase
                    end
                    
                    3'd4: begin
                        case (wr_weight_type)
                            WEIGHT_COMPRESS: begin
                                for (wr_i = 0; wr_i < 8; wr_i = wr_i + 1) begin
                                    bank_b_l4_compress_exp[wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_b_l4_compress_col[wr_burst_idx[3:1]][wr_burst_idx[0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                            
                            WEIGHT_EXPAND: begin
                                for (wr_i = 0; wr_i < 32; wr_i = wr_i + 1) begin
                                    bank_b_l4_expand_exp[wr_i]
                                        <= wr_exp_array[wr_i*EXP_WIDTH +: EXP_WIDTH];
                                end
                                for (wr_i = 0; wr_i < 16; wr_i = wr_i + 1) begin
                                    bank_b_l4_expand_col[wr_burst_idx[4:0]][wr_i]
                                        <= wr_data_burst[wr_i*DATA_WIDTH +: DATA_WIDTH];
                                end
                            end
                        endcase
                    end
                endcase
            end
        end
    end
end

//================================================================================
// DMA初始化 (可选，用于从外部DRAM加载权重)
//
// DMA传输格式假设与正常读写相同:
// - 使用相同的burst_idx编码
// - 使用dma_init_data直接写入，不区分指数和数据
//   (调用者需要先传指数burst，再传数据burst)
//================================================================================
endmodule