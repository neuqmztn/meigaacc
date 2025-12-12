`timescale 1ns / 1ps

module gradient_buffer_v2 #(
    parameter NUM_TOKENS        = 640,
    parameter DIM_SMALL         = 8,
    parameter DIM_LARGE         = 32,
    parameter DATA_WIDTH        = 16,
    parameter TOKEN_ADDR_WIDTH  = 10,
    parameter DIM_ADDR_WIDTH    = 5
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //======================================================================
    // 写接口
    //======================================================================
    input  wire                         wr_en,
    input  wire [2:0]                   wr_layer_id,   // 0..4
    input  wire [TOKEN_ADDR_WIDTH-1:0]  wr_token_id,
    input  wire [DIM_ADDR_WIDTH-1:0]    wr_dim_id,
    input  wire [DATA_WIDTH-1:0]        wr_data,

    //======================================================================
    // 读接口
    //======================================================================
    input  wire                         grad_rd_req,
    input  wire [2:0]                   grad_rd_layer_id,
    input  wire [TOKEN_ADDR_WIDTH-1:0]  grad_rd_token_id,
    input  wire [DIM_ADDR_WIDTH-1:0]    grad_rd_dim_id,
    output reg                          grad_rd_valid,
    output reg  [DATA_WIDTH-1:0]        grad_rd_data
);

    // 每个 layer 的总元素个数
    localparam integer SMALL_LAYER_SIZE = NUM_TOKENS * DIM_SMALL; // 640*8
    localparam integer LARGE_LAYER_SIZE = NUM_TOKENS * DIM_LARGE; // 640*32

    // 地址宽度（打包 token_id + dim_id）
    localparam integer SMALL_ADDR_WIDTH = TOKEN_ADDR_WIDTH + $clog2(DIM_SMALL);
    localparam integer LARGE_ADDR_WIDTH = TOKEN_ADDR_WIDTH + $clog2(DIM_LARGE);

    //==========================================================================
    // 存储阵列：5 块 BRAM
    //==========================================================================
    (* ram_style="block" *)
    reg [DATA_WIDTH-1:0] grad_l0 [0:SMALL_LAYER_SIZE-1];
    (* ram_style="block" *)
    reg [DATA_WIDTH-1:0] grad_l1 [0:SMALL_LAYER_SIZE-1];
    (* ram_style="block" *)
    reg [DATA_WIDTH-1:0] grad_l2 [0:SMALL_LAYER_SIZE-1];
    (* ram_style="block" *)
    reg [DATA_WIDTH-1:0] grad_l3 [0:SMALL_LAYER_SIZE-1];
    (* ram_style="block" *)
    reg [DATA_WIDTH-1:0] grad_l4 [0:LARGE_LAYER_SIZE-1];

    //==========================================================================
    // 写端地址打包
    //==========================================================================
    wire [SMALL_ADDR_WIDTH-1:0] wr_addr_small;
    wire [LARGE_ADDR_WIDTH-1:0] wr_addr_large;

    assign wr_addr_small = {wr_token_id, wr_dim_id[$clog2(DIM_SMALL)-1:0]};
    assign wr_addr_large = {wr_token_id, wr_dim_id[$clog2(DIM_LARGE)-1:0]};

    //==========================================================================
    // 写端逻辑（关键修改：移除异步复位，仅保留同步写）
    //==========================================================================
    always @(posedge clk) begin
        // BRAM 不需要复位内容，仅响应写使能
        if (wr_en) begin
            case (wr_layer_id)
                3'd0: grad_l0[wr_addr_small] <= wr_data;
                3'd1: grad_l1[wr_addr_small] <= wr_data;
                3'd2: grad_l2[wr_addr_small] <= wr_data;
                3'd3: grad_l3[wr_addr_small] <= wr_data;
                3'd4: grad_l4[wr_addr_large] <= wr_data;
                default: ; // do nothing
            endcase
        end
    end

    //==========================================================================
    // 读端：地址寄存 + 两级流水
    //==========================================================================

    // 0 → 1 拍：锁存 layer_id / 地址
    reg [2:0]                    rd_layer_id_reg1;
    reg [SMALL_ADDR_WIDTH-1:0]   rd_addr_small_reg;
    reg [LARGE_ADDR_WIDTH-1:0]   rd_addr_large_reg;

    // 1 → 2 拍：layer_id 再打一拍，供 MUX 使用
    reg [2:0] rd_layer_id_reg2;
    
    // valid pipeline
    reg rd_valid_d1, rd_valid_d2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_layer_id_reg1  <= 3'd0;
            rd_layer_id_reg2  <= 3'd0;
            rd_addr_small_reg <= {SMALL_ADDR_WIDTH{1'b0}};
            rd_addr_large_reg <= {LARGE_ADDR_WIDTH{1'b0}};
            rd_valid_d1       <= 1'b0;
            rd_valid_d2       <= 1'b0;
        end else begin
            // stage 0: latch request
            if (grad_rd_req) begin
                rd_layer_id_reg1  <= grad_rd_layer_id;
                rd_addr_small_reg <= {grad_rd_token_id, grad_rd_dim_id[$clog2(DIM_SMALL)-1:0]};
                rd_addr_large_reg <= {grad_rd_token_id, grad_rd_dim_id[$clog2(DIM_LARGE)-1:0]};
            end

            // stage 1: layer_id pipeline
            rd_layer_id_reg2 <= rd_layer_id_reg1;
            
            // valid pipeline
            rd_valid_d1 <= grad_rd_req;
            rd_valid_d2 <= rd_valid_d1;
        end
    end

    // 第一层：各 BRAM 各自 read（无复位，标准 RAM 读模板）
    reg [DATA_WIDTH-1:0] rd_data_l0_reg;
    reg [DATA_WIDTH-1:0] rd_data_l1_reg;
    reg [DATA_WIDTH-1:0] rd_data_l2_reg;
    reg [DATA_WIDTH-1:0] rd_data_l3_reg;
    reg [DATA_WIDTH-1:0] rd_data_l4_reg;

    always @(posedge clk) begin
        rd_data_l0_reg <= grad_l0[rd_addr_small_reg];
        rd_data_l1_reg <= grad_l1[rd_addr_small_reg];
        rd_data_l2_reg <= grad_l2[rd_addr_small_reg];
        rd_data_l3_reg <= grad_l3[rd_addr_small_reg];
    end

    always @(posedge clk) begin
        rd_data_l4_reg <= grad_l4[rd_addr_large_reg];
    end

    // 第二层：MUX 选择 + 有效位输出
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            grad_rd_data  <= {DATA_WIDTH{1'b0}};
            grad_rd_valid <= 1'b0;
        end else begin
            // data mux
            case (rd_layer_id_reg2)
                3'd0: grad_rd_data <= rd_data_l0_reg;
                3'd1: grad_rd_data <= rd_data_l1_reg;
                3'd2: grad_rd_data <= rd_data_l2_reg;
                3'd3: grad_rd_data <= rd_data_l3_reg;
                3'd4: grad_rd_data <= rd_data_l4_reg;
                default: grad_rd_data <= {DATA_WIDTH{1'b0}};
            endcase

            // valid：延迟两拍
            grad_rd_valid <= rd_valid_d2;
        end
    end

endmodule