`timescale 1ns / 1ps

module transformer_block_ctrl_fsm #(
    parameter TOKEN_NUM    = 641,
    parameter TOKEN_BATCH  = 32,
    parameter DIM          = 32,
    parameter ADDR_WIDTH   = 10
)(
    input  wire clk,
    input  wire rst_n,
    
    //================================================================================
    // 控制接口
    //================================================================================
    input  wire start,                          // 开始处理当前层
    output reg  done,                           // 层处理完成
    output reg  busy,                           // 忙碌标志
    
    //================================================================================
    // 子模块状态接口
    //================================================================================
    // Attention
    output reg  attention_start,
    input  wire attention_done,
    input  wire attention_busy,
    
    // FFN
    output reg  ffn_start,
    input  wire ffn_done,
    input  wire ffn_busy,
    
    // Residual Add 1
    output reg  residual1_start,
    input  wire residual1_done,
    
    // LayerNorm 1
    output reg  layernorm1_start,
    input  wire layernorm1_done,
    
    // Residual Add 2
    output reg  residual2_start,
    input  wire residual2_done,
    
    // LayerNorm 2
    output reg  layernorm2_start,
    input  wire layernorm2_done,
    
    //================================================================================
    // Layer Token Buffer控制信号
    //================================================================================
    output reg  bank_swap_req,                  // 请求切换BANK（层完成时）
    
    // Port A读控制（Attention和Residual使用）
    output reg  porta_rd_req,
    output reg  [ADDR_WIDTH-1:0] porta_rd_addr,
    
    // LayerNorm1暂存控制
    output reg  ln1_save_req,
    output reg  [ADDR_WIDTH-1:0] ln1_save_addr,
    
    // LayerNorm1读取控制（Residual 2使用）
    output reg  ln1_load_req,
    output reg  [ADDR_WIDTH-1:0] ln1_load_addr,
    
    // 层输出写控制（LayerNorm2输出）
    output reg  layer_out_wr_req,
    output reg  [ADDR_WIDTH-1:0] layer_out_wr_addr,
    
    //================================================================================
    // Result Buffer控制信号
    //================================================================================
    output reg  result_rd_req,
    output reg  [ADDR_WIDTH-1:0] result_rd_addr,
    
    output reg  result_wr_req,
    output reg  [ADDR_WIDTH-1:0] result_wr_addr
);

//================================================================================
// 状态定义
//================================================================================

localparam IDLE         = 4'd0;
localparam ATTENTION    = 4'd1;
localparam WAIT_ATT     = 4'd2;
localparam RESIDUAL_1   = 4'd3;
localparam WAIT_RES1    = 4'd4;
localparam LAYERNORM_1  = 4'd5;
localparam WAIT_LN1     = 4'd6;
localparam FFN          = 4'd7;
localparam WAIT_FFN     = 4'd8;
localparam RESIDUAL_2   = 4'd9;
localparam WAIT_RES2    = 4'd10;
localparam LAYERNORM_2  = 4'd11;
localparam WAIT_LN2     = 4'd12;
localparam DONE_STATE   = 4'd13;

reg [3:0] state, next_state;

//================================================================================
// 状态转移
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

//================================================================================
// 下一状态逻辑
//================================================================================

always @(*) begin
    next_state = state;
    
    case (state)
        IDLE: begin
            if (start) begin
                next_state = ATTENTION;
            end
        end
        
        ATTENTION: begin
            next_state = WAIT_ATT;
        end
        
        WAIT_ATT: begin
            if (attention_done) begin
                next_state = RESIDUAL_1;
            end
        end
        
        RESIDUAL_1: begin
            next_state = WAIT_RES1;
        end
        
        WAIT_RES1: begin
            if (residual1_done) begin
                next_state = LAYERNORM_1;
            end
        end
        
        LAYERNORM_1: begin
            next_state = WAIT_LN1;
        end
        
        WAIT_LN1: begin
            if (layernorm1_done) begin
                next_state = FFN;
            end
        end
        
        FFN: begin
            next_state = WAIT_FFN;
        end
        
        WAIT_FFN: begin
            if (ffn_done) begin
                next_state = RESIDUAL_2;
            end
        end
        
        RESIDUAL_2: begin
            next_state = WAIT_RES2;
        end
        
        WAIT_RES2: begin
            if (residual2_done) begin
                next_state = LAYERNORM_2;
            end
        end
        
        LAYERNORM_2: begin
            next_state = WAIT_LN2;
        end
        
        WAIT_LN2: begin
            if (layernorm2_done) begin
                next_state = DONE_STATE;
            end
        end
        
        DONE_STATE: begin
            next_state = IDLE;
        end
        
        default: begin
            next_state = IDLE;
        end
    endcase
end

//================================================================================
// 输出控制逻辑
//================================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        done <= 1'b0;
        busy <= 1'b0;
        
        attention_start <= 1'b0;
        ffn_start <= 1'b0;
        residual1_start <= 1'b0;
        layernorm1_start <= 1'b0;
        residual2_start <= 1'b0;
        layernorm2_start <= 1'b0;
        
        bank_swap_req <= 1'b0;
        
        porta_rd_req <= 1'b0;
        porta_rd_addr <= {ADDR_WIDTH{1'b0}};
        
        ln1_save_req <= 1'b0;
        ln1_save_addr <= {ADDR_WIDTH{1'b0}};
        
        ln1_load_req <= 1'b0;
        ln1_load_addr <= {ADDR_WIDTH{1'b0}};
        
        layer_out_wr_req <= 1'b0;
        layer_out_wr_addr <= {ADDR_WIDTH{1'b0}};
        
        result_rd_req <= 1'b0;
        result_rd_addr <= {ADDR_WIDTH{1'b0}};
        
        result_wr_req <= 1'b0;
        result_wr_addr <= {ADDR_WIDTH{1'b0}};
        

        
    end else begin
        // 默认值
        done <= 1'b0;
        attention_start <= 1'b0;
        ffn_start <= 1'b0;
        residual1_start <= 1'b0;
        layernorm1_start <= 1'b0;
        residual2_start <= 1'b0;
        layernorm2_start <= 1'b0;
        bank_swap_req <= 1'b0;
        
        case (state)
            IDLE: begin
                busy <= 1'b0;
            end
            
            ATTENTION: begin
                busy <= 1'b1;
                attention_start <= 1'b1;
            end
            
            WAIT_ATT: begin
                busy <= 1'b1;
            end
            
            RESIDUAL_1: begin
                busy <= 1'b1;
                residual1_start <= 1'b1;
            end
            
            WAIT_RES1: begin
                busy <= 1'b1;
            end
            
            LAYERNORM_1: begin
                busy <= 1'b1;
                layernorm1_start <= 1'b1;
            end
            
            WAIT_LN1: begin
                busy <= 1'b1;
            end
            
            FFN: begin
                busy <= 1'b1;
                ffn_start <= 1'b1;
            end
            
            WAIT_FFN: begin
                busy <= 1'b1;
            end
            
            RESIDUAL_2: begin
                busy <= 1'b1;
                residual2_start <= 1'b1;
            end
            
            WAIT_RES2: begin
                busy <= 1'b1;
            end
            
            LAYERNORM_2: begin
                busy <= 1'b1;
                layernorm2_start <= 1'b1;
            end
            
            WAIT_LN2: begin
                busy <= 1'b1;
            end
            
            DONE_STATE: begin
                busy <= 1'b0;
                done <= 1'b1;
                bank_swap_req <= 1'b1;  // 请求切换BANK
            end
            
            default: begin
                busy <= 1'b0;
            end
        endcase
    end
end

//================================================================================
// 调试输出
//================================================================================

assign dbg_state = state;

endmodule