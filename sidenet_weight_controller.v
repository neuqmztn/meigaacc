`timescale 1ns / 1ps

//================================================================================
// Sidenet Weight Controller v3.2 - 添加FFN权重接口
// 
// 功能:
// 1. 从Storage批量读取权重 (burst传输)
// 2. 缓存和重组权重数据
// 3. 为各层提供完整的权重矩阵
// 4. 支持DFA训练时的权重写回
//
// 主要变更 (v3.2):
// - 新增FFN W1和W2权重接口
// - W1: 8×32矩阵，32个列指数，32 bursts
// - W2: 32×8矩阵，8个列指数，16 bursts
//
// 主要变更 (v3.1):
// - 新增独立的WO权重输出接口
// - WO权重使用独立缓存和输出端口
// - QKV权重仍使用原有接口
//
// 作者: MEIGA Team  
// 日期: 2025-11-16
// 版本: v3.2
//================================================================================

module sidenet_weight_controller #(
    parameter NUM_LAYERS       = 5,
    parameter BACKBONE_DIM     = 32,
    parameter SIDENET_DIM      = 8,
    parameter D_FF             = 32,  // FFN隐藏层维度
    parameter DATA_WIDTH       = 16,
    parameter EXP_WIDTH        = 8,
    parameter DRAM_DATA_WIDTH  = 256,
    parameter MAX_EXP_ARRAY_WIDTH = BACKBONE_DIM * EXP_WIDTH
)(
    input  wire clk,
    input  wire rst_n,
    
    //============================================================================
    // 模式控制
    //============================================================================
    input  wire training_mode,
    input  wire forward_phase,
    input  wire [2:0] current_layer_id,
    
    //============================================================================
    // Layer 0 权重接口
    //============================================================================
    input  wire layer0_compress_weight_req,
    output reg  layer0_compress_weight_ready,
    output wire [SIDENET_DIM*EXP_WIDTH-1:0] layer0_compress_weight_exp_array,
    output wire [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH-1:0] layer0_compress_weight_mant,
    
    //============================================================================
    // Layer Standard (1-3) 权重接口
    //============================================================================
    input  wire layer_std_compress_weight_req,
    output reg  layer_std_compress_weight_ready,
    output wire [SIDENET_DIM*EXP_WIDTH-1:0] layer_std_compress_weight_exp_array,
    output wire [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH-1:0] layer_std_compress_weight_mant,
    
    // QKV权重接口 (Q/K/V，不含WO)
    input  wire layer_std_att_weight_req,
    input  wire [1:0] layer_std_att_weight_type,  // 00=Q, 01=K, 10=V (11保留)
    output reg  layer_std_att_weight_ready,
    output wire [SIDENET_DIM*EXP_WIDTH-1:0] layer_std_att_weight_exp_array,
    output wire [SIDENET_DIM*SIDENET_DIM*DATA_WIDTH-1:0] layer_std_att_weight_mant,
    
    // WO权重接口 (独立) - 新增 v3.1
    input  wire layer_std_wo_weight_req,
    output reg  layer_std_wo_weight_ready,
    output wire [SIDENET_DIM*EXP_WIDTH-1:0] layer_std_wo_weight_exp_array,
    output wire [SIDENET_DIM*SIDENET_DIM*DATA_WIDTH-1:0] layer_std_wo_weight_mant,
    
    // FFN权重接口 (新增 v3.2)
    // Linear1权重接口
    input  wire layer_std_ffn_w1_weight_req,
    output reg  layer_std_ffn_w1_weight_ready,
    output wire [D_FF*EXP_WIDTH-1:0] layer_std_ffn_w1_weight_exp_array,     // 32×8=256位
    output wire [SIDENET_DIM*D_FF*DATA_WIDTH-1:0] layer_std_ffn_w1_weight_mant,
    
    // Linear2权重接口
    input  wire layer_std_ffn_w2_weight_req,
    output reg  layer_std_ffn_w2_weight_ready,
    output wire [SIDENET_DIM*EXP_WIDTH-1:0] layer_std_ffn_w2_weight_exp_array,  // 8×8=64位
    output wire [D_FF*SIDENET_DIM*DATA_WIDTH-1:0] layer_std_ffn_w2_weight_mant,
    
    //============================================================================
    // Layer 4 权重接口
    //============================================================================
    input  wire layer4_compress_weight_req,
    output reg  layer4_compress_weight_ready,
    output wire [SIDENET_DIM*EXP_WIDTH-1:0] layer4_compress_weight_exp_array,
    output wire [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH-1:0] layer4_compress_weight_mant,
    
    input  wire layer4_expand_weight_req,
    output reg  layer4_expand_weight_ready,
    output wire [BACKBONE_DIM*EXP_WIDTH-1:0] layer4_expand_weight_exp_array,
    output wire [SIDENET_DIM*BACKBONE_DIM*DATA_WIDTH-1:0] layer4_expand_weight_mant,
    
    //============================================================================
    // DFA 权重更新接口
    //============================================================================
    input  wire dfa_weight_update_req,
    input  wire [2:0] dfa_layer_id,
    input  wire [3:0] dfa_weight_type,
    input  wire [MAX_EXP_ARRAY_WIDTH-1:0] dfa_weight_exp_array,
    input  wire [DRAM_DATA_WIDTH-1:0] dfa_weight_data,
    input  wire [5:0] dfa_weight_burst_idx,
    output reg  dfa_weight_update_ready,
    
    //============================================================================
    // Storage 读接口
    //============================================================================
    output reg  storage_rd_en,
    output reg  [2:0] storage_rd_layer_id,
    output reg  [3:0] storage_rd_weight_type,
    output reg  [5:0] storage_rd_burst_idx,
    input  wire storage_rd_valid,
    input  wire [MAX_EXP_ARRAY_WIDTH-1:0] storage_rd_exp_array,
    input  wire [DRAM_DATA_WIDTH-1:0] storage_rd_data_burst,
    
    //============================================================================
    // Storage 写接口
    //============================================================================
    output reg  storage_wr_en,
    output reg  [2:0] storage_wr_layer_id,
    output reg  [3:0] storage_wr_weight_type,
    output reg  [5:0] storage_wr_burst_idx,
    output reg  [MAX_EXP_ARRAY_WIDTH-1:0] storage_wr_exp_array,
    output reg  [DRAM_DATA_WIDTH-1:0] storage_wr_data_burst,
    input  wire storage_wr_ready,
    
    //============================================================================
    // 调试
    //============================================================================
    output wire [3:0] dbg_state,
    output reg  [31:0] dbg_read_count,
    output reg  [31:0] dbg_write_count
);

//================================================================================
// 权重类型定义
//================================================================================
localparam WEIGHT_COMPRESS = 4'd0;
localparam WEIGHT_ATT_WQ   = 4'd2;
localparam WEIGHT_ATT_WK   = 4'd3;
localparam WEIGHT_ATT_WV   = 4'd4;
localparam WEIGHT_ATT_WO   = 4'd5;
localparam WEIGHT_FFN_W1   = 4'd6;  // 8×32 (FFN Linear1)
localparam WEIGHT_FFN_W2   = 4'd7;  // 32×8 (FFN Linear2)
localparam WEIGHT_EXPAND   = 4'd8;

//================================================================================
// 状态机
//================================================================================
localparam IDLE            = 4'd0;
localparam ARBITRATE       = 4'd1;
localparam READ_BURSTS     = 4'd2;
localparam WAIT_BURST      = 4'd3;
localparam ASSEMBLE        = 4'd4;
localparam READY_OUT       = 4'd5;
localparam WRITE_BURST     = 4'd6;

reg [3:0] state;
assign dbg_state = state;

//================================================================================
// 内部信号
//================================================================================
reg [2:0] serving_requester;  // 0=none, 1=L0, 2=L_std_comp, 3=L_std_att, 4=L4_comp, 5=L4_exp, 7=L_std_wo (新增)
reg [3:0] serving_weight_type;
reg [5:0] burst_counter;
reg [5:0] total_bursts;

//================================================================================
// Burst缓存 - 最多32个burst (Expand需要)
//================================================================================
reg [MAX_EXP_ARRAY_WIDTH-1:0] cached_exp_array;
reg [DRAM_DATA_WIDTH-1:0] burst_buffer [0:31];

//================================================================================
// 重组后的权重缓存
//================================================================================
// Compression: 8个指数 + 32×8个权重
reg [SIDENET_DIM*EXP_WIDTH-1:0] compress_exp_array;
reg [BACKBONE_DIM*SIDENET_DIM*DATA_WIDTH-1:0] compress_weight_mant;

// Attention (QKV): 8个指数 + 8×8个权重
reg [SIDENET_DIM*EXP_WIDTH-1:0] attention_exp_array;
reg [SIDENET_DIM*SIDENET_DIM*DATA_WIDTH-1:0] attention_weight_mant;

// WO: 8个指数 + 8×8个权重 (新增 v3.1)
reg [SIDENET_DIM*EXP_WIDTH-1:0] wo_exp_array;
reg [SIDENET_DIM*SIDENET_DIM*DATA_WIDTH-1:0] wo_weight_mant;

// FFN W1: 32个指数 + 8×32个权重 (新增 v3.2)
reg [D_FF*EXP_WIDTH-1:0] ffn_w1_exp_array;
reg [SIDENET_DIM*D_FF*DATA_WIDTH-1:0] ffn_w1_weight_mant;

// FFN W2: 8个指数 + 32×8个权重 (新增 v3.2)
reg [SIDENET_DIM*EXP_WIDTH-1:0] ffn_w2_exp_array;
reg [D_FF*SIDENET_DIM*DATA_WIDTH-1:0] ffn_w2_weight_mant;

// Expand: 32个指数 + 8×32个权重
reg [BACKBONE_DIM*EXP_WIDTH-1:0] expand_exp_array;
reg [SIDENET_DIM*BACKBONE_DIM*DATA_WIDTH-1:0] expand_weight_mant;

//================================================================================
// 输出连接
//================================================================================
assign layer0_compress_weight_exp_array = compress_exp_array;
assign layer0_compress_weight_mant = compress_weight_mant;

assign layer_std_compress_weight_exp_array = compress_exp_array;
assign layer_std_compress_weight_mant = compress_weight_mant;

assign layer_std_att_weight_exp_array = attention_exp_array;
assign layer_std_att_weight_mant = attention_weight_mant;

assign layer_std_wo_weight_exp_array = wo_exp_array;  // 新增 v3.1
assign layer_std_wo_weight_mant = wo_weight_mant;      // 新增 v3.1

assign layer_std_ffn_w1_weight_exp_array = ffn_w1_exp_array;  // 新增 v3.2
assign layer_std_ffn_w1_weight_mant = ffn_w1_weight_mant;      // 新增 v3.2

assign layer_std_ffn_w2_weight_exp_array = ffn_w2_exp_array;  // 新增 v3.2
assign layer_std_ffn_w2_weight_mant = ffn_w2_weight_mant;      // 新增 v3.2

assign layer4_compress_weight_exp_array = compress_exp_array;
assign layer4_compress_weight_mant = compress_weight_mant;

assign layer4_expand_weight_exp_array = expand_exp_array;
assign layer4_expand_weight_mant = expand_weight_mant;

//================================================================================
// Burst数量查找
//================================================================================
function [5:0] get_burst_count;
    input [3:0] weight_type;
    begin
        case (weight_type)
            WEIGHT_COMPRESS: get_burst_count = 16;  // 32×8, 8列×2 burst/列
            WEIGHT_ATT_WQ:   get_burst_count = 8;   // 8×8, 8列×1 burst/列
            WEIGHT_ATT_WK:   get_burst_count = 8;
            WEIGHT_ATT_WV:   get_burst_count = 8;
            WEIGHT_ATT_WO:   get_burst_count = 8;
            WEIGHT_FFN_W1:   get_burst_count = 32;  // 8×32, 32列×1 burst/列
            WEIGHT_FFN_W2:   get_burst_count = 16;  // 32×8, 8列×2 burst/列
            WEIGHT_EXPAND:   get_burst_count = 32;  // 8×32, 32列×1 burst/列
            default:         get_burst_count = 0;
        endcase
    end
endfunction

//================================================================================
// 主状态机
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        burst_counter <= 6'd0;
        total_bursts <= 6'd0;
        serving_requester <= 3'd0;
        serving_weight_type <= 4'd0;
        
        storage_rd_en <= 1'b0;
        storage_rd_layer_id <= 3'd0;
        storage_rd_weight_type <= 4'd0;
        storage_rd_burst_idx <= 6'd0;
        
        storage_wr_en <= 1'b0;
        storage_wr_layer_id <= 3'd0;
        storage_wr_weight_type <= 4'd0;
        storage_wr_burst_idx <= 6'd0;
        storage_wr_exp_array <= {MAX_EXP_ARRAY_WIDTH{1'b0}};
        storage_wr_data_burst <= {DRAM_DATA_WIDTH{1'b0}};
        
        layer0_compress_weight_ready <= 1'b0;
        layer_std_compress_weight_ready <= 1'b0;
        layer_std_att_weight_ready <= 1'b0;
        layer_std_wo_weight_ready <= 1'b0;  // 新增 v3.1
        layer_std_ffn_w1_weight_ready <= 1'b0;  // 新增 v3.2
        layer_std_ffn_w2_weight_ready <= 1'b0;  // 新增 v3.2
        layer4_compress_weight_ready <= 1'b0;
        layer4_expand_weight_ready <= 1'b0;
        dfa_weight_update_ready <= 1'b0;
        
        dbg_read_count <= 32'd0;
        dbg_write_count <= 32'd0;
        
    end else begin
        
        // 默认清除单周期信号
        storage_rd_en <= 1'b0;
        storage_wr_en <= 1'b0;
        layer0_compress_weight_ready <= 1'b0;
        layer_std_compress_weight_ready <= 1'b0;
        layer_std_att_weight_ready <= 1'b0;
        layer_std_wo_weight_ready <= 1'b0;  // 新增 v3.1
        layer_std_ffn_w1_weight_ready <= 1'b0;  // 新增 v3.2
        layer_std_ffn_w2_weight_ready <= 1'b0;  // 新增 v3.2
        layer4_compress_weight_ready <= 1'b0;
        layer4_expand_weight_ready <= 1'b0;
        dfa_weight_update_ready <= 1'b0;
        
        case (state)
            //================================================================
            // IDLE: 等待请求
            //================================================================
            IDLE: begin
                if (dfa_weight_update_req && training_mode && !forward_phase) begin
                    // DFA写回优先
                    state <= WRITE_BURST;
                    serving_requester <= 3'd6;
                end else begin
                    state <= ARBITRATE;
                end
            end
            
            //================================================================
            // ARBITRATE: 仲裁请求
            //================================================================
            ARBITRATE: begin
                // 简单优先级仲裁
                if (layer0_compress_weight_req) begin
                    serving_requester <= 3'd1;
                    serving_weight_type <= WEIGHT_COMPRESS;
                    storage_rd_layer_id <= 3'd0;
                    storage_rd_weight_type <= WEIGHT_COMPRESS;
                    total_bursts <= get_burst_count(WEIGHT_COMPRESS);
                    state <= READ_BURSTS;
                    burst_counter <= 6'd0;
                    
                end else if (layer_std_compress_weight_req) begin
                    serving_requester <= 3'd2;
                    serving_weight_type <= WEIGHT_COMPRESS;
                    storage_rd_layer_id <= current_layer_id;
                    storage_rd_weight_type <= WEIGHT_COMPRESS;
                    total_bursts <= get_burst_count(WEIGHT_COMPRESS);
                    state <= READ_BURSTS;
                    burst_counter <= 6'd0;
                    
                end else if (layer_std_att_weight_req) begin
                    serving_requester <= 3'd3;
                    // 根据att_weight_type确定具体类型 (只处理Q/K/V)
                    case (layer_std_att_weight_type)
                        2'd0: serving_weight_type <= WEIGHT_ATT_WQ;
                        2'd1: serving_weight_type <= WEIGHT_ATT_WK;
                        2'd2: serving_weight_type <= WEIGHT_ATT_WV;
                        default: serving_weight_type <= WEIGHT_ATT_WQ;  // 防御性编程
                    endcase
                    storage_rd_layer_id <= current_layer_id;
                    storage_rd_weight_type <= WEIGHT_ATT_WQ + {2'b0, layer_std_att_weight_type};
                    total_bursts <= 6'd8;
                    state <= READ_BURSTS;
                    burst_counter <= 6'd0;
                    
                // 新增 v3.1: WO独立请求处理
                end else if (layer_std_wo_weight_req) begin
                    serving_requester <= 3'd7;  // 新的requester ID
                    serving_weight_type <= WEIGHT_ATT_WO;
                    storage_rd_layer_id <= current_layer_id;
                    storage_rd_weight_type <= WEIGHT_ATT_WO;
                    total_bursts <= 6'd8;
                    state <= READ_BURSTS;
                    burst_counter <= 6'd0;
                    
                // 新增 v3.2: FFN权重请求处理
                end else if (layer_std_ffn_w1_weight_req) begin
                    serving_requester <= 4'd8;  // FFN W1 requester ID
                    serving_weight_type <= WEIGHT_FFN_W1;
                    storage_rd_layer_id <= current_layer_id;
                    storage_rd_weight_type <= WEIGHT_FFN_W1;
                    total_bursts <= 6'd32;
                    state <= READ_BURSTS;
                    burst_counter <= 6'd0;
                    
                end else if (layer_std_ffn_w2_weight_req) begin
                    serving_requester <= 4'd9;  // FFN W2 requester ID
                    serving_weight_type <= WEIGHT_FFN_W2;
                    storage_rd_layer_id <= current_layer_id;
                    storage_rd_weight_type <= WEIGHT_FFN_W2;
                    total_bursts <= 6'd16;
                    state <= READ_BURSTS;
                    burst_counter <= 6'd0;
                    
                end else if (layer4_compress_weight_req) begin
                    serving_requester <= 3'd4;
                    serving_weight_type <= WEIGHT_COMPRESS;
                    storage_rd_layer_id <= 3'd4;
                    storage_rd_weight_type <= WEIGHT_COMPRESS;
                    total_bursts <= get_burst_count(WEIGHT_COMPRESS);
                    state <= READ_BURSTS;
                    burst_counter <= 6'd0;
                    
                end else if (layer4_expand_weight_req) begin
                    serving_requester <= 3'd5;
                    serving_weight_type <= WEIGHT_EXPAND;
                    storage_rd_layer_id <= 3'd4;
                    storage_rd_weight_type <= WEIGHT_EXPAND;
                    total_bursts <= get_burst_count(WEIGHT_EXPAND);
                    state <= READ_BURSTS;
                    burst_counter <= 6'd0;
                    
                end else begin
                    state <= IDLE;
                end
            end
            
            //================================================================
            // READ_BURSTS: 发起burst读取
            //================================================================
            READ_BURSTS: begin
                storage_rd_en <= 1'b1;
                storage_rd_burst_idx <= burst_counter;
                state <= WAIT_BURST;
            end
            
            //================================================================
            // WAIT_BURST: 等待burst返回
            //================================================================
            WAIT_BURST: begin
                if (storage_rd_valid) begin
                    // 第一个burst获取指数数组
                    if (burst_counter == 0) begin
                        cached_exp_array <= storage_rd_exp_array;
                    end
                    
                    // 缓存数据burst
                    burst_buffer[burst_counter] <= storage_rd_data_burst;
                    
                    dbg_read_count <= dbg_read_count + 1;
                    
                    // 判断是否读完
                    if (burst_counter == total_bursts - 1) begin
                        state <= ASSEMBLE;
                    end else begin
                        burst_counter <= burst_counter + 1;
                        state <= READ_BURSTS;
                    end
                end
            end
            
            //================================================================
            // ASSEMBLE: 重组权重数据
            //================================================================
            ASSEMBLE: begin
                // 重组在组合逻辑中进行 (见下方always块)
                state <= READY_OUT;
            end
            
            //================================================================
            // READY_OUT: 输出给请求方
            //================================================================
            READY_OUT: begin
                case (serving_requester)
                    3'd1: layer0_compress_weight_ready <= 1'b1;
                    3'd2: layer_std_compress_weight_ready <= 1'b1;
                    3'd3: layer_std_att_weight_ready <= 1'b1;
                    3'd4: layer4_compress_weight_ready <= 1'b1;
                    3'd5: layer4_expand_weight_ready <= 1'b1;
                    3'd7: layer_std_wo_weight_ready <= 1'b1;  // 新增 v3.1
                    4'd8: layer_std_ffn_w1_weight_ready <= 1'b1;  // 新增 v3.2
                    4'd9: layer_std_ffn_w2_weight_ready <= 1'b1;  // 新增 v3.2
                endcase
                state <= IDLE;
            end
            
            //================================================================
            // WRITE_BURST: DFA写回
            //================================================================
            WRITE_BURST: begin
                storage_wr_en <= 1'b1;
                storage_wr_layer_id <= dfa_layer_id;
                storage_wr_weight_type <= dfa_weight_type;
                storage_wr_burst_idx <= dfa_weight_burst_idx;
                storage_wr_exp_array <= dfa_weight_exp_array;
                storage_wr_data_burst <= dfa_weight_data;
                
                if (storage_wr_ready) begin
                    dfa_weight_update_ready <= 1'b1;
                    dbg_write_count <= dbg_write_count + 1;
                    state <= IDLE;
                end
            end
            
            default: state <= IDLE;
        endcase
    end
end

//================================================================================
// 数据重组逻辑 - Compression
//
// 从burst buffer重组为计算引擎格式
// Burst组织: 列优先，每列占2个burst
//   Burst 0-1:   列0 (32个权重)
//   Burst 2-3:   列1
//   ...
//   Burst 14-15: 列7
//
// 目标格式: [列][行] = compress_weight_mant[col*32*16 + row*16 +: 16]
//================================================================================

integer asm_col, asm_row;

always @(posedge clk) begin
    if (state == ASSEMBLE && serving_weight_type == WEIGHT_COMPRESS) begin
        // 提取指数 (8个)
        for (asm_col = 0; asm_col < 8; asm_col = asm_col + 1) begin
            compress_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH] 
                <= cached_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH];
        end
        
        // 重组权重 (32×8)
        for (asm_col = 0; asm_col < 8; asm_col = asm_col + 1) begin
            // 列asm_col的数据来自burst (asm_col*2) 和 (asm_col*2+1)
            
            // 前16个权重 (行0-15) - 来自第一个burst
            for (asm_row = 0; asm_row < 16; asm_row = asm_row + 1) begin
                compress_weight_mant[(asm_col*32 + asm_row)*DATA_WIDTH +: DATA_WIDTH]
                    <= burst_buffer[asm_col*2][asm_row*DATA_WIDTH +: DATA_WIDTH];
            end
            
            // 后16个权重 (行16-31) - 来自第二个burst
            for (asm_row = 0; asm_row < 16; asm_row = asm_row + 1) begin
                compress_weight_mant[(asm_col*32 + 16 + asm_row)*DATA_WIDTH +: DATA_WIDTH]
                    <= burst_buffer[asm_col*2 + 1][asm_row*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
end

//================================================================================
// 数据重组逻辑 - Attention (QKV Only)
//
// Burst组织: 每列占1个burst
//   Burst 0: 列0 (8个权重)
//   Burst 1: 列1
//   ...
//   Burst 7: 列7
//
// 目标格式: [列][行] = attention_weight_mant[col*8*16 + row*16 +: 16]
//
// 修改 v3.1: 只处理WQ/WK/WV，WO使用独立逻辑
//================================================================================

always @(posedge clk) begin
    if (state == ASSEMBLE && (serving_weight_type >= WEIGHT_ATT_WQ && serving_weight_type <= WEIGHT_ATT_WV)) begin
        // 提取指数 (8个)
        for (asm_col = 0; asm_col < 8; asm_col = asm_col + 1) begin
            attention_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH]
                <= cached_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH];
        end
        
        // 重组权重 (8×8)
        for (asm_col = 0; asm_col < 8; asm_col = asm_col + 1) begin
            // 列asm_col的数据来自burst asm_col
            for (asm_row = 0; asm_row < 8; asm_row = asm_row + 1) begin
                attention_weight_mant[(asm_col*8 + asm_row)*DATA_WIDTH +: DATA_WIDTH]
                    <= burst_buffer[asm_col][asm_row*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
end

//================================================================================
// 数据重组逻辑 - WO (新增 v3.1)
//
// Burst组织: 每列占1个burst (同Attention)
//   Burst 0: 列0 (8个权重)
//   Burst 1: 列1
//   ...
//   Burst 7: 列7
//
// 目标格式: [列][行] = wo_weight_mant[col*8*16 + row*16 +: 16]
//================================================================================

always @(posedge clk) begin
    if (state == ASSEMBLE && serving_weight_type == WEIGHT_ATT_WO) begin
        // 提取指数 (8个)
        for (asm_col = 0; asm_col < 8; asm_col = asm_col + 1) begin
            wo_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH]
                <= cached_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH];
        end
        
        // 重组权重 (8×8)
        for (asm_col = 0; asm_col < 8; asm_col = asm_col + 1) begin
            // 列asm_col的数据来自burst asm_col
            for (asm_row = 0; asm_row < 8; asm_row = asm_row + 1) begin
                wo_weight_mant[(asm_col*8 + asm_row)*DATA_WIDTH +: DATA_WIDTH]
                    <= burst_buffer[asm_col][asm_row*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
end

//================================================================================
// 数据重组逻辑 - FFN W1 (新增 v3.2)
//
// Burst组织: 每列占1个burst
//   Burst 0:  列0 (8个权重)
//   Burst 1:  列1
//   ...
//   Burst 31: 列31
//
// 目标格式: [列][行] = ffn_w1_weight_mant[col*8*16 + row*16 +: 16]
// 矩阵: 8×32 (8行32列)
//================================================================================

always @(posedge clk) begin
    if (state == ASSEMBLE && serving_weight_type == WEIGHT_FFN_W1) begin
        // 提取指数 (32个)
        for (asm_col = 0; asm_col < 32; asm_col = asm_col + 1) begin
            ffn_w1_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH]
                <= cached_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH];
        end
        
        // 重组权重 (8×32)
        for (asm_col = 0; asm_col < 32; asm_col = asm_col + 1) begin
            // 列asm_col的数据来自burst asm_col
            for (asm_row = 0; asm_row < 8; asm_row = asm_row + 1) begin
                ffn_w1_weight_mant[(asm_col*8 + asm_row)*DATA_WIDTH +: DATA_WIDTH]
                    <= burst_buffer[asm_col][asm_row*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
end

//================================================================================
// 数据重组逻辑 - FFN W2 (新增 v3.2)
//
// Burst组织: 列优先，每列占2个burst
//   Burst 0-1:   列0 (32个权重)
//   Burst 2-3:   列1
//   ...
//   Burst 14-15: 列7
//
// 目标格式: [列][行] = ffn_w2_weight_mant[col*32*16 + row*16 +: 16]
// 矩阵: 32×8 (32行8列)
//================================================================================

always @(posedge clk) begin
    if (state == ASSEMBLE && serving_weight_type == WEIGHT_FFN_W2) begin
        // 提取指数 (8个)
        for (asm_col = 0; asm_col < 8; asm_col = asm_col + 1) begin
            ffn_w2_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH]
                <= cached_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH];
        end
        
        // 重组权重 (32×8) - 类似Compression
        for (asm_col = 0; asm_col < 8; asm_col = asm_col + 1) begin
            // 前16个权重 (行0-15)
            for (asm_row = 0; asm_row < 16; asm_row = asm_row + 1) begin
                ffn_w2_weight_mant[(asm_col*32 + asm_row)*DATA_WIDTH +: DATA_WIDTH]
                    <= burst_buffer[asm_col*2][asm_row*DATA_WIDTH +: DATA_WIDTH];
            end
            
            // 后16个权重 (行16-31)
            for (asm_row = 0; asm_row < 16; asm_row = asm_row + 1) begin
                ffn_w2_weight_mant[(asm_col*32 + 16 + asm_row)*DATA_WIDTH +: DATA_WIDTH]
                    <= burst_buffer[asm_col*2 + 1][asm_row*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
end

//================================================================================
// 数据重组逻辑 - Expand
//
// Burst组织: 每列占1个burst
//   Burst 0:  列0 (8个权重)
//   Burst 1:  列1
//   ...
//   Burst 31: 列31
//
// 目标格式: [列][行] = expand_weight_mant[col*8*16 + row*16 +: 16]
//================================================================================

always @(posedge clk) begin
    if (state == ASSEMBLE && serving_weight_type == WEIGHT_EXPAND) begin
        // 提取指数 (32个)
        for (asm_col = 0; asm_col < 32; asm_col = asm_col + 1) begin
            expand_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH]
                <= cached_exp_array[asm_col*EXP_WIDTH +: EXP_WIDTH];
        end
        
        // 重组权重 (8×32)
        for (asm_col = 0; asm_col < 32; asm_col = asm_col + 1) begin
            // 列asm_col的数据来自burst asm_col
            for (asm_row = 0; asm_row < 8; asm_row = asm_row + 1) begin
                expand_weight_mant[(asm_col*8 + asm_row)*DATA_WIDTH +: DATA_WIDTH]
                    <= burst_buffer[asm_col][asm_row*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
end

endmodule