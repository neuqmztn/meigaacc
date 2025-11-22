`timescale 1ns / 1ps

//================================================================================
// Attention BFP Accumulator v2.0 - 支持每维独立指数的累加器
//
// 功能：
// - 存储多头attention的中间结果（BFP格式）
// - 支持跨chunk的指数对齐累加
// - ✅ v2.0更新：每个维度有独立指数（不再共享）
//
// 存储容量：
// - 4 heads × 32 tokens × 8 dims × 24-bit = 3072 bytes (尾数)
// - 4 heads × 32 tokens × 8 dims × 8-bit = 1024 bytes (指数) ← 增加8倍
// - 总计：4.0 KB (原3.2 KB)
//
// BFP格式说明：
// - ✅ v2.0：每个维度有独立指数
// - 优点：保持最高精度，支持不同维度的不同量级
// - 累加时自动对齐指数
//
// 版本：v2.0 - 支持每维独立指数
// 日期：2024-11-18
//================================================================================

module attention_bfp_accumulator #(
    parameter NUM_HEADS    = 4,
    parameter TOKEN_BATCH  = 32,
    parameter HEAD_DIM     = 8,
    parameter MANT_WIDTH   = 24,     // 尾数位宽
    parameter EXP_WIDTH    = 8       // 指数位宽
)(
    input  wire clk,
    input  wire rst_n,
    
    //===========================================================================
    // 写端口 - 从apply_v写入
    //===========================================================================
    input  wire wr_en,
    input  wire [1:0] wr_head,
    input  wire [4:0] wr_row,           // token索引
    input  wire [2:0] wr_dim,           // 维度索引
    input  wire signed [MANT_WIDTH-1:0] wr_mant,
    input  wire [EXP_WIDTH-1:0] wr_exp,
    input  wire wr_first_chunk,         // 是否第一个chunk（清零累加器）
    
    //===========================================================================
    // 读端口 - 供apply_v读取旧值或output_projection读取
    //===========================================================================
    input  wire rd_en,
    input  wire [1:0] rd_head,
    input  wire [4:0] rd_row,
    input  wire [2:0] rd_dim,
    output reg signed [MANT_WIDTH-1:0] rd_mant,
    output reg [EXP_WIDTH-1:0] rd_exp,
    output reg rd_valid,
    
    //===========================================================================
    // 调试接口
    //===========================================================================
    output reg [31:0] dbg_wr_count,
    output reg [31:0] dbg_rd_count
);

//================================================================================
// 参数计算
//================================================================================

localparam TOTAL_TOKENS = NUM_HEADS * TOKEN_BATCH;
localparam TOTAL_ELEMENTS = NUM_HEADS * TOKEN_BATCH * HEAD_DIM;

//================================================================================
// 存储阵列
//================================================================================

// 尾数存储（3D数组：head × token × dim）
(* ram_style = "block" *) 
reg signed [MANT_WIDTH-1:0] mant_mem [0:NUM_HEADS-1][0:TOKEN_BATCH-1][0:HEAD_DIM-1];

// ✅ v2.0：指数存储（3D数组：head × token × dim）
// 每个维度有独立指数，不再共享
(* ram_style = "distributed" *) 
reg [EXP_WIDTH-1:0] exp_mem [0:NUM_HEADS-1][0:TOKEN_BATCH-1][0:HEAD_DIM-1];

//================================================================================
// 写逻辑 - 支持指数对齐累加
//================================================================================

// 读取旧值（用于累加）
reg signed [MANT_WIDTH-1:0] old_mant;
reg [EXP_WIDTH-1:0] old_exp;

// 指数对齐信号
reg signed [EXP_WIDTH:0] exp_diff;        // 9-bit signed
reg [EXP_WIDTH-1:0] new_shared_exp;
reg [5:0] shift_amount;                   // 右移量（最多63）

// 对齐后的尾数
reg signed [MANT_WIDTH-1:0] aligned_old_mant;
reg signed [MANT_WIDTH-1:0] aligned_new_mant;
reg signed [MANT_WIDTH-1:0] sum_mant;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dbg_wr_count <= 32'd0;
        
    end else if (wr_en) begin
        dbg_wr_count <= dbg_wr_count + 1'b1;
        
        if (wr_first_chunk) begin
            //====================================================================
            // 第一个chunk：直接写入，清空累加器
            //====================================================================
            mant_mem[wr_head][wr_row][wr_dim] <= wr_mant;
            
            // ✅ v2.0：每个维度都更新自己的指数（不再只在dim=0时更新）
            exp_mem[wr_head][wr_row][wr_dim] <= wr_exp;
            
            $display("[%0t] BFP_Accum WR: head=%0d row=%0d dim=%0d FIRST_CHUNK exp=%0d mant=%0d",
                     $time, wr_head, wr_row, wr_dim, wr_exp, $signed(wr_mant));
            
        end else begin
            //====================================================================
            // 后续chunk：指数对齐后累加
            //====================================================================
            
            // 1. 读取旧值（✅ v2.0：读取对应维度的指数）
            old_mant = mant_mem[wr_head][wr_row][wr_dim];
            old_exp = exp_mem[wr_head][wr_row][wr_dim];  // ✅ 使用对应维度的指数
            
            // 2. 计算指数差（带符号）
            exp_diff = $signed({1'b0, wr_exp}) - $signed({1'b0, old_exp});
            
            if (exp_diff >= 0) begin
                //------------------------------------------------------------
                // 新值指数 >= 旧值指数
                //------------------------------------------------------------
                new_shared_exp = wr_exp;
                
                // 限制右移量
                if (exp_diff >= MANT_WIDTH) begin
                    shift_amount = MANT_WIDTH;  // 完全右移（变为0）
                end else begin
                    shift_amount = exp_diff[5:0];
                end
                
                // 对齐旧值（右移）
                if (shift_amount >= MANT_WIDTH) begin
                    aligned_old_mant = {MANT_WIDTH{1'b0}};
                end else begin
                    aligned_old_mant = old_mant >>> shift_amount;  // 算术右移
                end
                
                // 新值不需要对齐
                aligned_new_mant = wr_mant;
                
            end else begin
                //------------------------------------------------------------
                // 旧值指数 > 新值指数
                //------------------------------------------------------------
                new_shared_exp = old_exp;
                
                // 计算右移量（exp_diff是负数）
                if (-exp_diff >= MANT_WIDTH) begin
                    shift_amount = MANT_WIDTH;
                end else begin
                    shift_amount = (-exp_diff) & 6'h3F;
                end
                
                // 旧值不需要对齐
                aligned_old_mant = old_mant;
                
                // 对齐新值（右移）
                if (shift_amount >= MANT_WIDTH) begin
                    aligned_new_mant = {MANT_WIDTH{1'b0}};
                end else begin
                    aligned_new_mant = wr_mant >>> shift_amount;
                end
            end
            
            // 3. 累加对齐后的尾数
            sum_mant = aligned_old_mant + aligned_new_mant;
            mant_mem[wr_head][wr_row][wr_dim] <= sum_mant;
            
            // 4. ✅ v2.0：每个维度都更新自己的指数
            exp_mem[wr_head][wr_row][wr_dim] <= new_shared_exp;
            
            $display("[%0t] BFP_Accum WR: head=%0d row=%0d dim=%0d ACCUMULATE",
                     $time, wr_head, wr_row, wr_dim);
            $display("           old: exp=%0d mant=%0d", old_exp, $signed(old_mant));
            $display("           new: exp=%0d mant=%0d", wr_exp, $signed(wr_mant));
            $display("           exp_diff=%0d shift=%0d", $signed(exp_diff), shift_amount);
            $display("           aligned_old=%0d aligned_new=%0d", 
                     $signed(aligned_old_mant), $signed(aligned_new_mant));
            $display("           result: exp=%0d mant=%0d", new_shared_exp, $signed(sum_mant));
        end
    end
end

//================================================================================
// 读逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_mant <= {MANT_WIDTH{1'b0}};
        rd_exp <= {EXP_WIDTH{1'b0}};
        rd_valid <= 1'b0;
        dbg_rd_count <= 32'd0;
        
    end else if (rd_en) begin
        rd_mant <= mant_mem[rd_head][rd_row][rd_dim];
        rd_exp <= exp_mem[rd_head][rd_row][rd_dim];  // ✅ v2.0：读取对应维度的指数
        rd_valid <= 1'b1;
        dbg_rd_count <= dbg_rd_count + 1'b1;
        
        $display("[%0t] BFP_Accum RD: head=%0d row=%0d dim=%0d => exp=%0d mant=%0d",
                 $time, rd_head, rd_row, rd_dim, 
                 exp_mem[rd_head][rd_row][rd_dim], $signed(mant_mem[rd_head][rd_row][rd_dim]));
                 
    end else begin
        rd_valid <= 1'b0;
    end
end

//================================================================================
// 初始化（仿真）
//================================================================================

`ifdef SIMULATION
integer h, t, d;
initial begin
    for (h = 0; h < NUM_HEADS; h = h + 1) begin
        for (t = 0; t < TOKEN_BATCH; t = t + 1) begin
            for (d = 0; d < HEAD_DIM; d = d + 1) begin
                // ✅ v2.0：初始化每个维度的指数
                exp_mem[h][t][d] = 8'd0;
                mant_mem[h][t][d] = {MANT_WIDTH{1'b0}};
            end
        end
    end
    $display("========================================");
    $display("BFP Accumulator v2.0 Initialized");
    $display("  Heads: %0d", NUM_HEADS);
    $display("  Tokens per head: %0d", TOKEN_BATCH);
    $display("  Dimensions: %0d", HEAD_DIM);
    $display("  Mantissa width: %0d bits", MANT_WIDTH);
    $display("  Exponent width: %0d bits", EXP_WIDTH);
    $display("  ✅ Each dimension has independent exponent");
    $display("========================================");
end
`endif

endmodule