`timescale 1ns / 1ps

module gcu_core_v2 #(
    // 基本参数
    parameter NUM_TOKENS        = 640,          
    parameter DIM_SMALL         = 8,          
    parameter DIM_LARGE         = 32,         
    
    // 数据格式参数
    parameter DATA_WIDTH        = 16,           
    parameter TOKEN_ADDR_WIDTH  = 10,          
    parameter DIM_ADDR_WIDTH    = 5           
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire                         start,         
    input  wire [1:0]                   layer_id,       
    input  wire [DATA_WIDTH-1:0]        error_scalar,   
    
    output reg                          done,          
    output wire                         busy,           
    
    //==========================================================================
    // B 矩阵读接口（直接挂到 dfa_matrix_bank_v2）
    //==========================================================================
    output reg                          mb_rd_en,
    output reg  [1:0]                   mb_layer_id,    // 0..3
    output reg  [DIM_ADDR_WIDTH-1:0]    mb_row_addr,    // 维度 index
    output reg  [3:0]                   mb_col_addr,    // 固定 0
    input  wire [DATA_WIDTH-1:0]        mb_rd_data,
    input  wire                         mb_rd_valid,
    
    //==========================================================================
    // 激活值读接口（沿用原 gradient_compute_unit 风格）
    //==========================================================================
    output reg                          act_rd_en,
    output reg  [TOKEN_ADDR_WIDTH-1:0]  act_token_addr,
    input  wire [DIM_SMALL*DATA_WIDTH-1:0] act_vector,  
    input  wire                         act_valid,
    
    //==========================================================================
    // 梯度写出接口（统一口）
    //==========================================================================
    output reg                          grad_valid,
    output reg  [TOKEN_ADDR_WIDTH-1:0]  grad_token_addr,
    output reg  [DIM_ADDR_WIDTH-1:0]    grad_dim_addr,
    output reg  [DATA_WIDTH-1:0]        grad_data
);

    //==========================================================================
    // 状态机定义
    //==========================================================================
    localparam IDLE          = 3'd0;
    localparam COMPUTE_DELTA = 3'd1;   // B × error_scalar，填 delta_buffer
    localparam WAIT_DELTA    = 3'd2;   // 等待所有 delta 写完
    localparam COMPUTE_GRAD  = 3'd3;   // 请求当前 token 激活向量
    localparam WAIT_ACT      = 3'd4;   // 等待 act_valid
    localparam WRITE_GRAD    = 3'd5;   // 串行写出当前 token 的各维梯度
    localparam DONE_STATE    = 3'd6;

    reg [2:0] state_reg, state_next;

    // 维度 / token 计数
    reg [TOKEN_ADDR_WIDTH-1:0] token_cnt_reg, token_cnt_next;
    reg [DIM_ADDR_WIDTH-1:0]   dim_cnt_reg,   dim_cnt_next;

    // 对应 layer 的有效维度数（本版 layer0~3 一律 8）
    reg [DIM_ADDR_WIDTH-1:0]   active_dim_reg;

    // B 数据寄存与 delta_buffer
    reg [DATA_WIDTH-1:0]       b_data_reg;
    reg [DIM_ADDR_WIDTH-1:0]   b_dim_reg;
    reg                        b_valid_reg;

    reg signed [DATA_WIDTH-1:0] delta_buffer [0:DIM_LARGE-1];
    reg signed [DATA_WIDTH-1:0] activation_buffer [0:DIM_LARGE-1];

    integer i;

    //==========================================================================
    // busy
    //==========================================================================
    assign busy = (state_reg != IDLE) && (state_reg != DONE_STATE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg      <= IDLE;
            token_cnt_reg  <= {TOKEN_ADDR_WIDTH{1'b0}};
            dim_cnt_reg    <= {DIM_ADDR_WIDTH{1'b0}};
      
            active_dim_reg <= DIM_SMALL[DIM_ADDR_WIDTH-1:0];
        end else begin
            // 状态和计数器更新
            state_reg      <= state_next;
            token_cnt_reg  <= token_cnt_next;
            dim_cnt_reg    <= dim_cnt_next;

            if (state_reg == IDLE && start) begin
                if (layer_id <= 2'd3)
                    active_dim_reg <= DIM_SMALL[DIM_ADDR_WIDTH-1:0];
                else
                    active_dim_reg <= {DIM_ADDR_WIDTH{1'b0}}; // 不支持 layer4 及以上
            end
        end
    end

    //==========================================================================
    // B 侧寄存：把 mb_rd_data / mb_rd_valid 锁存为 b_data_reg / b_valid_reg
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            b_data_reg  <= {DATA_WIDTH{1'b0}};
            b_dim_reg   <= {DIM_ADDR_WIDTH{1'b0}};
            b_valid_reg <= 1'b0;
        end else begin
            if (mb_rd_valid) begin
                b_data_reg  <= mb_rd_data;
                b_dim_reg   <= mb_row_addr;
                b_valid_reg <= 1'b1;
            end else begin
                b_valid_reg <= 1'b0;
            end
        end
    end

    //==========================================================================
    // delta_buffer[d] = B[d] * error_scalar （Q4.12 × Q4.12 → Q4.12）
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DIM_LARGE; i = i + 1) begin
                delta_buffer[i] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            if (b_valid_reg) begin
                // 只在有效维度范围内写入
                if (b_dim_reg < active_dim_reg) begin
                    delta_buffer[b_dim_reg] <= 
                        ($signed(b_data_reg) * $signed(error_scalar)) >>> 12;
                end
            end
        end
    end

    //==========================================================================
    // 激活值锁存：从 act_vector 中拆出前 active_dim_reg 个维度
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DIM_LARGE; i = i + 1) begin
                activation_buffer[i] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            if (act_valid) begin
                for (i = 0; i < DIM_SMALL; i = i + 1) begin
                    if (i < active_dim_reg) begin
                        activation_buffer[i] <= 
                            $signed(act_vector[i*DATA_WIDTH +: DATA_WIDTH]);
                    end
                end
            end
        end
    end

    //==========================================================================
    // 主状态机组合逻辑
    //==========================================================================
    always @(*) begin
        // 默认保持
        state_next     = state_reg;
        token_cnt_next = token_cnt_reg;
        dim_cnt_next   = dim_cnt_reg;

        // 默认输出
        mb_rd_en      = 1'b0;
        mb_layer_id   = layer_id;
        mb_row_addr   = {DIM_ADDR_WIDTH{1'b0}};
        mb_col_addr   = 4'd0;

        act_rd_en       = 1'b0;
        act_token_addr  = {TOKEN_ADDR_WIDTH{1'b0}};

        grad_valid      = 1'b0;
        grad_token_addr = {TOKEN_ADDR_WIDTH{1'b0}};
        grad_dim_addr   = {DIM_ADDR_WIDTH{1'b0}};
        grad_data       = {DATA_WIDTH{1'b0}};

        case (state_reg)
            //------------------------------------------------------------------
            // IDLE：等待 start
            //------------------------------------------------------------------
            IDLE: begin

                if (start) begin
                    token_cnt_next = {TOKEN_ADDR_WIDTH{1'b0}};
                    dim_cnt_next   = {DIM_ADDR_WIDTH{1'b0}};
       
                    state_next     = COMPUTE_DELTA;
                end
            end

            //------------------------------------------------------------------
            // COMPUTE_DELTA：串行读 B 矩阵，填 delta_buffer[0..active_dim-1]
            //------------------------------------------------------------------
            COMPUTE_DELTA: begin
 
                mb_rd_en    = 1'b1;
                mb_layer_id = layer_id;
                mb_row_addr = dim_cnt_reg;
                mb_col_addr = 4'd0;

                if (dim_cnt_reg == (active_dim_reg - 1'b1)) begin
                    dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                    state_next   = WAIT_DELTA;
                end else begin
                    dim_cnt_next = dim_cnt_reg + 1'b1;
                end
            end

            //------------------------------------------------------------------
            // WAIT_DELTA
            //------------------------------------------------------------------
            WAIT_DELTA: begin

                state_next = COMPUTE_GRAD;
            end

            //------------------------------------------------------------------
            // COMPUTE_GRAD
            //------------------------------------------------------------------
            COMPUTE_GRAD: begin
                act_rd_en       = 1'b1;
                act_token_addr  = token_cnt_reg;  // 0..639

                state_next      = WAIT_ACT;
            end

            //------------------------------------------------------------------
            // WAIT_ACT
            //------------------------------------------------------------------
            WAIT_ACT: begin
                if (act_valid) begin
                    dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                    state_next   = WRITE_GRAD;
                end
            end

            //------------------------------------------------------------------
            // WRITE_GRAD
            //------------------------------------------------------------------
            WRITE_GRAD: begin
                grad_valid      = 1'b1;
                grad_token_addr = token_cnt_reg;
                grad_dim_addr   = dim_cnt_reg;

                grad_data = ($signed(delta_buffer[dim_cnt_reg]) * $signed(activation_buffer[dim_cnt_reg])) >>> 12;

                if (dim_cnt_reg == (active_dim_reg - 1'b1)) begin
                    dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
    
                    if (token_cnt_reg == (NUM_TOKENS-1)) begin

                        token_cnt_next = {TOKEN_ADDR_WIDTH{1'b0}};
                        state_next     = DONE_STATE;
                    end else begin

                        token_cnt_next = token_cnt_reg + 1'b1;
                        state_next     = COMPUTE_GRAD;
                    end
                end else begin
                    // 当前 token 下一维
                    dim_cnt_next = dim_cnt_reg + 1'b1;
                end
            end

            //------------------------------------------------------------------
            // DONE_STATE
            //------------------------------------------------------------------
            DONE_STATE: begin

                if (!start) begin
                    state_next = IDLE;
                end
            end

            default: begin
                state_next = IDLE;
            end
        endcase
    end

    //==========================================================================
    // done 信号
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
        end else begin
            done <= (state_reg == DONE_STATE);
        end
    end

endmodule