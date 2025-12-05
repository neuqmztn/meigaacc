`timescale 1ns / 1ps

//================================================================================
// gcu_core_v2 - Unified Gradient Compute Unit (Layer 0~3, single instance)
//
// 特点：
//   - 逻辑上支持 4 个 layer（0~3），物理上只有一个 GCU 实例。
//   - 对于 layer0~3，DIM=8，使用统一结构；layer4 不在本模块中实现。
//   - 直接从 dfa_matrix_bank 读取 B 矩阵：mb_layer_id + mb_row_addr + mb_col_addr。
//   - 自己向 LOB 请求 activations：act_rd_en / act_token_addr / act_vector / act_valid。
//   - 输出统一的梯度流：grad_valid + grad_token_addr + grad_dim_addr + grad_data。
//   - 每次 start：完成「当前 layer 的 NUM_TOKENS × DIM」全部梯度计算。
//================================================================================
module gcu_core_v2 #(
    // 基本参数
    parameter NUM_TOKENS        = 640,          // 0..639
    parameter DIM_SMALL         = 8,            // layer0~3 的维度
    parameter DIM_LARGE         = 32,           // 预留（本版不使用）
    
    // 数据格式参数
    parameter DATA_WIDTH        = 16,           // Q4.12 有符号定点
    parameter TOKEN_ADDR_WIDTH  = 10,           // 支持 0..1023
    parameter DIM_ADDR_WIDTH    = 5             // 支持 0..31
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire                         start,          // 启动当前 layer 的梯度计算
    input  wire [1:0]                   layer_id,       // 0..3，对应 B 矩阵与激活缓冲
    input  wire [DATA_WIDTH-1:0]        error_scalar,   // BCE 误差标量 Q4.12
    
    output reg                          done,           // 当前 layer 所有 token 梯度完成
    output wire                         busy,           // 非 IDLE = 1
    
    //==========================================================================
    // B 矩阵读接口（直接挂到 dfa_matrix_bank_v2）
    //   - 读的是 B[layer_id][row][col]
    //   - 对于 DFA：col 固定为 0
    //==========================================================================
    output reg                          mb_rd_en,
    output reg  [1:0]                   mb_layer_id,    // 0..3
    output reg  [DIM_ADDR_WIDTH-1:0]    mb_row_addr,    // 维度 index
    output reg  [3:0]                   mb_col_addr,    // 固定 0
    input  wire [DATA_WIDTH-1:0]        mb_rd_data,
    input  wire                         mb_rd_valid,
    
    //==========================================================================
    // 激活值读接口（沿用原 gradient_compute_unit 风格）
    //   - LOB 根据 act_rd_en / act_token_addr 提供 act_vector
    //   - act_vector 总是 32 维打满，GCU 内部只用前 8 维
    //==========================================================================
    output reg                          act_rd_en,
    output reg  [TOKEN_ADDR_WIDTH-1:0]  act_token_addr,
    input  wire [DIM_LARGE*DATA_WIDTH-1:0] act_vector,  // {dim31,...,dim0}
    input  wire                         act_valid,
    
    //==========================================================================
    // 梯度写出接口（统一口）
    //   - 顶层根据 layer_id 把它路由进 gradient_buffer.gcuX_* 端口
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

    //==========================================================================
    // 状态寄存器
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg      <= IDLE;
            token_cnt_reg  <= {TOKEN_ADDR_WIDTH{1'b0}};
            dim_cnt_reg    <= {DIM_ADDR_WIDTH{1'b0}};
            active_dim_reg <= DIM_SMALL[DIM_ADDR_WIDTH-1:0];
        end else begin
            state_reg     <= state_next;
            token_cnt_reg <= token_cnt_next;
            dim_cnt_reg   <= dim_cnt_next;
        end
    end

    // 每次 start 时，根据 layer_id 设置 active_dim_reg
    // 目前 layer0~3 一律 8，layer>3 不支持（active_dim_reg=0）
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_dim_reg <= DIM_SMALL[DIM_ADDR_WIDTH-1:0];
        end else if (state_reg == IDLE && start) begin
            if (layer_id <= 2'd3)
                active_dim_reg <= DIM_SMALL[DIM_ADDR_WIDTH-1:0];
            else
                active_dim_reg <= {DIM_ADDR_WIDTH{1'b0}}; // 不支持 layer4 及以上
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
    //   - 这里直接使用组合乘法 + >>> 12，和你原来 GCU 一致。
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
                for (i = 0; i < DIM_LARGE; i = i + 1) begin
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
                if (start && (active_dim_reg != 0)) begin
                    token_cnt_next = {TOKEN_ADDR_WIDTH{1'b0}};
                    dim_cnt_next   = {DIM_ADDR_WIDTH{1'b0}};
                    state_next     = COMPUTE_DELTA;
                end
            end

            //------------------------------------------------------------------
            // COMPUTE_DELTA：串行读 B 矩阵，填 delta_buffer[0..active_dim-1]
            //------------------------------------------------------------------
            COMPUTE_DELTA: begin
                // 请求 B[layer_id][dim_cnt_reg][0]
                mb_rd_en    = 1'b1;
                mb_layer_id = layer_id;
                mb_row_addr = dim_cnt_reg;
                mb_col_addr = 4'd0;

                // 计数维度
                if (dim_cnt_reg == (active_dim_reg - 1'b1)) begin
                    dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                    state_next   = WAIT_DELTA;
                end else begin
                    dim_cnt_next = dim_cnt_reg + 1'b1;
                end
            end

            //------------------------------------------------------------------
            // WAIT_DELTA：给 delta_buffer 填充留一点延迟
            //------------------------------------------------------------------
            WAIT_DELTA: begin
                // 简单给一个 1 周期缓冲，之后开始请求激活向量
                state_next = COMPUTE_GRAD;
            end

            //------------------------------------------------------------------
            // COMPUTE_GRAD：对当前 token 请求 activations
            //------------------------------------------------------------------
            COMPUTE_GRAD: begin
                act_rd_en       = 1'b1;
                act_token_addr  = token_cnt_reg;  // 0..639

                state_next      = WAIT_ACT;
            end

            //------------------------------------------------------------------
            // WAIT_ACT：等 act_valid 把 activation_buffer 写满
            //------------------------------------------------------------------
            WAIT_ACT: begin
                if (act_valid) begin
                    dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                    state_next   = WRITE_GRAD;
                end
            end

            //------------------------------------------------------------------
            // WRITE_GRAD：写当前 token 的 active_dim_reg 个梯度
            //------------------------------------------------------------------
            WRITE_GRAD: begin
                grad_valid      = 1'b1;
                grad_token_addr = token_cnt_reg;
                grad_dim_addr   = dim_cnt_reg;

                // grad = delta[d] * act[t][d]  (Q4.12 × Q4.12 → Q4.12)
                grad_data = ($signed(delta_buffer[dim_cnt_reg]) *
                             $signed(activation_buffer[dim_cnt_reg])) >>> 12;

                if (dim_cnt_reg == (active_dim_reg - 1'b1)) begin
                    dim_cnt_next = {DIM_ADDR_WIDTH{1'b0}};
                    // 当前 token 最后一维
                    if (token_cnt_reg == (NUM_TOKENS-1)) begin
                        // 所有 token 都完成
                        token_cnt_next = {TOKEN_ADDR_WIDTH{1'b0}};
                        state_next     = DONE_STATE;
                    end else begin
                        // 下一个 token
                        token_cnt_next = token_cnt_reg + 1'b1;
                        state_next     = COMPUTE_GRAD;
                    end
                end else begin
                    // 当前 token 下一维
                    dim_cnt_next = dim_cnt_reg + 1'b1;
                end
            end

            //------------------------------------------------------------------
            // DONE_STATE：拉高 done，一个或多个周期
            //------------------------------------------------------------------
            DONE_STATE: begin
                // 外部拉低 start 之后，可以回到 IDLE
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
    // done 信号：在 DONE_STATE 期间为 1
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
        end else begin
            done <= (state_reg == DONE_STATE);
        end
    end

endmodule
