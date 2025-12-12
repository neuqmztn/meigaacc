`timescale 1ns / 1ps


module PE #(
    parameter NUM_MAC        = 8,    // MAC 数量 (4 或 8)
    parameter ADDER_MODE     = 0,    // 0=INT8, 1=INT16
    parameter DATA_WIDTH     = 8,    // 输入数据位宽
    parameter MAC_OUT_WIDTH  = 19,  
    parameter FINAL_WIDTH    = 38    
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire        flush,
    
    // 打包后的输入数据 (来自 Data Packer)
    input  wire [NUM_MAC*DATA_WIDTH-1:0]    x_data_a_packed,
    input  wire [NUM_MAC*DATA_WIDTH-1:0]    x_data_b_packed,
    input  wire [NUM_MAC*DATA_WIDTH-1:0]    w_data_a_packed,
    input  wire [NUM_MAC*DATA_WIDTH-1:0]    w_data_b_packed,
    
    // 结果输出
    output reg  signed [FINAL_WIDTH-1:0]    pe_result,
    output reg                              result_valid
);

    //=============================================================================
    // 1. 动态符号模式控制 
    //=============================================================================
    // INT16 模式下，输入被拆分为 High(Signed) 和 Low(Unsigned)。
    // 根据 Packer_PE_B 的打包顺序：
    // MAC 0/4: X_L * W_L -> Unsigned * Unsigned
    // MAC 1/5: X_L * W_H -> Unsigned * Signed
    // MAC 2/6: X_H * W_L -> Signed   * Unsigned
    // MAC 3/7: X_H * W_H -> Signed   * Signed
    
    wire [7:0] mode_A_vec; // 控制 X 输入 (Port A/C)
    wire [7:0] mode_B_vec; // 控制 W 输入 (Port B/D)

    genvar k;
    generate
        for (k = 0; k < 8; k = k + 1) begin : gen_modes
            if (ADDER_MODE == 0) begin
                // INT8 模式：默认全为有符号
                assign mode_A_vec[k] = 1'b0;
                assign mode_B_vec[k] = 1'b0;
            end else begin
                // INT16 模式：根据 MAC 索引动态分配
                
                // Port A (X): 索引 % 4 < 2 为 Low (Unsigned)，否则为 High (Signed)
                // MAC 0,1,4,5 -> Unsigned; MAC 2,3,6,7 -> Signed
                assign mode_A_vec[k] = ((k % 4) < 2) ? 1'b1 : 1'b0;
                
                // Port B (W): 索引 % 2 == 0 为 Low (Unsigned)，否则为 High (Signed)
                // MAC 0,2,4,6 -> Unsigned; MAC 1,3,5,7 -> Signed
                assign mode_B_vec[k] = ((k % 2) == 0) ? 1'b1 : 1'b0;
            end
        end
    endgenerate

    //=============================================================================
    // 2. MAC 阵列实例化
    //=============================================================================
    wire signed [18:0] mac_out [0:7];
    wire [7:0] mac_valid;

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_mac_array
            if (i < NUM_MAC) begin : active_mac
                // 实例化 coordinated_skip_mac_unit
                coordinated_skip_mac_unit u_mac (
                    .clk(clk), .rst_n(rst_n), .enable(enable),
                    
                    // 输入数据 (根据索引切片)
                    .A(x_data_a_packed[i*8 +: 8]),
                    .B(w_data_a_packed[i*8 +: 8]),
                    .C(x_data_b_packed[i*8 +: 8]),
                    .D(w_data_b_packed[i*8 +: 8]),
                    
                    // 符号控制 (应用上述逻辑)
                    // C/D 通道与 A/B 通道共享逻辑，因为 Packer 结构一致
                    .unsigned_mode_A(mode_A_vec[i]),
                    .unsigned_mode_B(mode_B_vec[i]),
                    .unsigned_mode_C(mode_A_vec[i]), // Port C 对应 X (同 Port A)
                    .unsigned_mode_D(mode_B_vec[i]), // Port D 对应 W (同 Port B)
                    
                    // 结果输出
                    .result(mac_out[i]),
                    .result_valid(mac_valid[i])
                );
            end else begin : dummy_mac
                // 未使用的 MAC 接口置零
                assign mac_out[i] = 19'sd0;
                assign mac_valid[i] = 1'b0;
            end
        end
    endgenerate

    //=============================================================================
    // 3. 路由层 (Routing Layer) - 映射到加法树
    //=============================================================================
    // 目标是将 MAC 输出路由到三个路径：
    // Path A -> 最终左移 16 位 (High*High)
    // Path B -> 最终左移 8 位  (High*Low + Low*High)
    // Path C -> 最终左移 0 位  (Low*Low)
    
    wire signed [20:0] adder_input [0:7]; // 扩展到位宽以防止中间溢出

    generate
        if (NUM_MAC == 8 && ADDER_MODE == 0) begin : PE_A_ROUTING
            // PE_A (INT8): 一一对应，无乱序
            for (i = 0; i < 8; i = i + 1) begin : loop_pe_a
                assign adder_input[i] = {{2{mac_out[i][18]}}, mac_out[i]};
            end
            
        end else if (NUM_MAC == 8 && ADDER_MODE == 1) begin : PE_B_ROUTING
            // PE_B (INT16): 关键路由修正
            
            // Path A (Adder 0,1) -> 需要 HH (MAC 3, 7)
            assign adder_input[0] = {{2{mac_out[3][18]}}, mac_out[3]}; 
            assign adder_input[1] = {{2{mac_out[7][18]}}, mac_out[7]};
            
            // Path B (Adder 2,3) -> 需要 LH, HL (MAC 1, 2, 5, 6)
            assign adder_input[2] = {{2{mac_out[1][18]}}, mac_out[1]};
            assign adder_input[3] = {{2{mac_out[2][18]}}, mac_out[2]};
            assign adder_input[4] = {{2{mac_out[5][18]}}, mac_out[5]};
            assign adder_input[5] = {{2{mac_out[6][18]}}, mac_out[6]};
            
            // Path C (Adder 4) -> 需要 LL (MAC 0, 4) -> 放入 Adder input 6,7
            assign adder_input[6] = {{2{mac_out[0][18]}}, mac_out[0]};
            assign adder_input[7] = {{2{mac_out[4][18]}}, mac_out[4]};

        end else if (NUM_MAC == 4 && ADDER_MODE == 0) begin : PE_C_ROUTING
            // PE_C (INT8 Half): 仅使用前 4 个
            assign adder_input[0] = {{2{mac_out[0][18]}}, mac_out[0]};
            assign adder_input[1] = {{2{mac_out[1][18]}}, mac_out[1]};
            assign adder_input[2] = {{2{mac_out[2][18]}}, mac_out[2]};
            assign adder_input[3] = {{2{mac_out[3][18]}}, mac_out[3]};
            // 后半部分置零
            assign adder_input[4] = 21'sd0;
            assign adder_input[5] = 21'sd0;
            assign adder_input[6] = 21'sd0;
            assign adder_input[7] = 21'sd0;

        end else begin : PE_D_ROUTING
            // PE_D (INT16 Half): 仅处理一组 INT16 (MAC 0-3)
            
            // Path A (HH) -> MAC 3
            assign adder_input[0] = {{2{mac_out[3][18]}}, mac_out[3]};
            assign adder_input[1] = 21'sd0;
            
            // Path B (LH, HL) -> MAC 1, 2
            assign adder_input[2] = {{2{mac_out[1][18]}}, mac_out[1]};
            assign adder_input[3] = {{2{mac_out[2][18]}}, mac_out[2]};
            assign adder_input[4] = 21'sd0;
            assign adder_input[5] = 21'sd0;
            
            // Path C (LL) -> MAC 0
            assign adder_input[6] = {{2{mac_out[0][18]}}, mac_out[0]};
            assign adder_input[7] = 21'sd0;
        end
    endgenerate

    //=============================================================================
    // Stage 1: Level 1 加法器 (21bit -> 22bit)
    //=============================================================================
    wire signed [21:0] add1_out, add2_out, add3_out, add4_out;
    
    // 分组求和：
    // add1 -> 指向 Path A (High)
    // add2, add3 -> 指向 Path B (Mid)
    // add4 -> 指向 Path C (Low)
    
    assign add1_out = adder_input[0] + adder_input[1];
    assign add2_out = adder_input[2] + adder_input[3];
    assign add3_out = adder_input[4] + adder_input[5];
    assign add4_out = adder_input[6] + adder_input[7];

    reg signed [21:0] add1_out_r, add2_out_r, add3_out_r, add4_out_r;
    reg               stage1_valid;

    // 初始化
    initial begin
        add1_out_r   = 22'sd0;
        add2_out_r   = 22'sd0;
        add3_out_r   = 22'sd0;
        add4_out_r   = 22'sd0;
        stage1_valid = 1'b0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            add1_out_r   <= 22'sd0;
            add2_out_r   <= 22'sd0;
            add3_out_r   <= 22'sd0;
            add4_out_r   <= 22'sd0;
            stage1_valid <= 1'b0;
        end else if (flush) begin
            add1_out_r   <= 22'sd0;
            add2_out_r   <= 22'sd0;
            add3_out_r   <= 22'sd0;
            add4_out_r   <= 22'sd0;
            stage1_valid <= 1'b0;
        end else if (enable) begin
            add1_out_r   <= add1_out;
            add2_out_r   <= add2_out;
            add3_out_r   <= add3_out;
            add4_out_r   <= add4_out;
            // 使用 MAC 0 的 valid 信号作为流水线有效标志 (假设所有 MAC 同步)
            stage1_valid <= mac_valid[0]; 
        end
    end

    //=============================================================================
    // Stage 2: 移位对齐 + 最终求和
    //=============================================================================
    
    // --- Path A (处理 High 部分) ---
    wire signed [37:0] path_a;
    generate
        if (ADDER_MODE == 0) begin : PATH_A_BYPASS
            // INT8: 正常符号扩展
            assign path_a = {{16{add1_out_r[21]}}, add1_out_r};
        end else begin : PATH_A_SHIFT
            // INT16: HH 需要左移 16 位
            assign path_a = {add1_out_r, 16'd0};
        end
    endgenerate

    // --- Path B (处理 Mid 部分) ---
    wire signed [22:0] add5_out;
    assign add5_out = add2_out_r + add3_out_r; // 合并 Mid 部分
    
    wire signed [30:0] path_b;
    generate
        if (ADDER_MODE == 0) begin : PATH_B_BYPASS
            // INT8: 正常符号扩展
            assign path_b = {{8{add5_out[22]}}, add5_out};
        end else begin : PATH_B_SHIFT
            // INT16: LH/HL 需要左移 8 位
            assign path_b = {add5_out, 8'd0};
        end
    endgenerate

    // --- Path C (处理 Low 部分) ---
    wire signed [21:0] path_c;
    assign path_c = add4_out_r; // Low 部分无需合并，直接通过

    // --- 最终加法树 ---
    wire signed [37:0] level3_sum;
    
    // 将三条路径对齐相加：
    assign level3_sum = path_a + 
                        {{7{path_b[30]}}, path_b} + 
                        {{16{path_c[21]}}, path_c};

    //=============================================================================
    // 输出寄存器
    //=============================================================================
    initial begin
        pe_result    = {FINAL_WIDTH{1'b0}};
        result_valid = 1'b0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pe_result    <= {FINAL_WIDTH{1'b0}};
            result_valid <= 1'b0;
        end else if (flush) begin
            pe_result    <= {FINAL_WIDTH{1'b0}};
            result_valid <= 1'b0;
        end else if (enable) begin
            pe_result    <= level3_sum;
            result_valid <= stage1_valid;
        end
    end


endmodule