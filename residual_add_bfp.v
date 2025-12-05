`timescale 1ns / 1ps

module residual_add_bfp #(
    parameter TOKEN_NUM  = 640,
    parameter DIM        = 32,
    parameter DATA_WIDTH = 8,
    parameter EXP_WIDTH  = 8,
    parameter ADDR_WIDTH = 10,
    parameter GUARD_BITS = 3
)(
    input  wire clk,
    input  wire rst_n,
    
    input  wire start,
    output reg  done,
    output wire busy,
    output reg  error,
    
    // Input Port
    output reg  input_rd_en,
    output reg  [ADDR_WIDTH-1:0] input_rd_addr,
    input  wire [EXP_WIDTH-1:0] input_exp,
    input  wire [DIM*DATA_WIDTH-1:0] input_mant,
    input  wire input_valid,
    
    // Result Port
    output reg  result_rd_en,
    output reg  [ADDR_WIDTH-1:0] result_rd_addr,
    input  wire [EXP_WIDTH-1:0] result_exp,
    input  wire [DIM*DATA_WIDTH-1:0] result_mant,
    input  wire result_valid,
    
    // Output Port
    output reg  output_wr_en,
    output reg  [ADDR_WIDTH-1:0] output_wr_addr,
    output reg  [EXP_WIDTH-1:0] output_exp,
    output reg  [DIM*DATA_WIDTH-1:0] output_mant,
    output reg  output_valid,
    input  wire output_ready,
    
    // Debug / Status
    output reg  [3:0] state,
    output reg  [9:0] processed_count,
    output reg  [31:0] cycle_count,
    output reg  overflow_detected,
    output reg  underflow_detected
);

//================================================================================
// 状态定义
//================================================================================
localparam IDLE           = 4'd0;
localparam READ_PREP      = 4'd1;
localparam READ_WAIT      = 4'd2;
localparam ALIGN_EXP      = 4'd3;
localparam ADD_MANTISSA   = 4'd4;
localparam FIND_MAX       = 4'd5;
localparam NORMALIZE      = 4'd6;
localparam WRITE          = 4'd7;
localparam NEXT_TOKEN     = 4'd8;
localparam DONE_STATE     = 4'd9;
localparam ERROR_STATE    = 4'd10;

//================================================================================
// 内部信号与寄存器
//================================================================================
reg [9:0] token_idx;
reg [3:0] read_wait_cnt;
reg [3:0] process_cycle;

// 数据寄存器
reg [EXP_WIDTH-1:0] input_exp_reg;
reg [DIM*DATA_WIDTH-1:0] input_mant_reg;
reg [EXP_WIDTH-1:0] result_exp_reg;
reg [DIM*DATA_WIDTH-1:0] result_mant_reg;

// 计算过程寄存器
reg [EXP_WIDTH-1:0] aligned_exp;
reg signed [DATA_WIDTH+GUARD_BITS:0] aligned_mant_a [0:DIM-1];
reg signed [DATA_WIDTH+GUARD_BITS:0] aligned_mant_b [0:DIM-1];
reg signed [DATA_WIDTH+GUARD_BITS+1:0] sum_mant [0:DIM-1];

// 归一化寄存器
reg [5:0] max_abs_pos;
reg [DATA_WIDTH+GUARD_BITS+1:0] max_abs_val;
reg [4:0] shift_amount;
reg signed [EXP_WIDTH+1:0] new_exp;
reg signed [DATA_WIDTH-1:0] norm_mant [0:DIM-1];

reg all_zero_flag;

integer i;

//================================================================================
// 解包输入尾数数组
//================================================================================
wire signed [DATA_WIDTH-1:0] input_mant_array [0:DIM-1];
wire signed [DATA_WIDTH-1:0] result_mant_array [0:DIM-1];

genvar g;
generate
    for (g = 0; g < DIM; g = g + 1) begin : gen_unpack
        assign input_mant_array[g] = input_mant_reg[g*DATA_WIDTH +: DATA_WIDTH];
        assign result_mant_array[g] = result_mant_reg[g*DATA_WIDTH +: DATA_WIDTH];
    end
endgenerate

//================================================================================
// 前导零计数函数 
//================================================================================
function automatic [4:0] count_leading_zeros_fast;
    input [DATA_WIDTH+GUARD_BITS+1:0] value;
    reg [4:0] count;
    reg [DATA_WIDTH+GUARD_BITS+1:0] tmp;
    begin
        tmp = value;
        count = 0;
        
        // 检查高7位 (Bit 12 down to 6)
        if (tmp[12:6] == 7'b0000000) begin
            count = count + 7;
            tmp = {tmp[5:0], 7'b0};
        end
        // 检查高4位
        if (tmp[12:9] == 4'b0000) begin
            count = count + 4;
            tmp = {tmp[8:0], 4'b0};
        end
        // 检查高2位
        if (tmp[12:11] == 2'b00) begin
            count = count + 2;
            tmp = {tmp[10:0], 2'b0};
        end
        // 检查最高位
        if (tmp[12] == 1'b0) begin
            count = count + 1;
        end
        
        count_leading_zeros_fast = count;
    end
endfunction

//================================================================================
// 树形比较器数组寄存器
//================================================================================
reg [DATA_WIDTH+GUARD_BITS+1:0] abs_array [0:31];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level1 [0:15];
reg [5:0] pos_level1 [0:15];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level2 [0:7];
reg [5:0] pos_level2 [0:7];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level3 [0:3];
reg [5:0] pos_level3 [0:3];


//================================================================================
// 主状态机
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        done <= 1'b0;
        error <= 1'b0;
        
        // 控制信号复位
        input_rd_en <= 1'b0;
        input_rd_addr <= {ADDR_WIDTH{1'b0}};
        result_rd_en <= 1'b0;
        result_rd_addr <= {ADDR_WIDTH{1'b0}};
        
        output_wr_en <= 1'b0;
        output_wr_addr <= {ADDR_WIDTH{1'b0}};
        output_valid <= 1'b0;
        output_exp <= {EXP_WIDTH{1'b0}};
        output_mant <= {(DIM*DATA_WIDTH){1'b0}};
        
        // 状态计数器复位
        token_idx <= 10'd0;
        read_wait_cnt <= 4'd0;
        process_cycle <= 4'd0;
        
        processed_count <= 10'd0;
        cycle_count <= 32'd0;
        
        overflow_detected <= 1'b0;
        underflow_detected <= 1'b0;
        all_zero_flag <= 1'b0;
        
        // 数据路径寄存器复位 (可选，为了代码整洁这里省略大数组的显式复位)
        input_exp_reg <= {EXP_WIDTH{1'b0}};
        result_exp_reg <= {EXP_WIDTH{1'b0}};
        aligned_exp <= {EXP_WIDTH{1'b0}};
        max_abs_val <= 0;
        shift_amount <= 0;
        new_exp <= 0;
        
    end else begin
        
        // 默认清除一次性控制信号
        input_rd_en <= 1'b0;
        result_rd_en <= 1'b0;
        output_wr_en <= 1'b0;
        output_valid <= 1'b0;
        
        case (state)
            
            //----------------------------------------------------------------
            // IDLE: 等待启动
            //----------------------------------------------------------------
            IDLE: begin
                done <= 1'b0;
                error <= 1'b0;
                overflow_detected <= 1'b0;
                underflow_detected <= 1'b0;
                
                if (start) begin
                    token_idx <= 10'd0;
                    processed_count <= 10'd0;
                    cycle_count <= 32'd0;
                    state <= READ_PREP;
                end
            end
            
            //----------------------------------------------------------------
            // READ_PREP: 发起读请求
            //----------------------------------------------------------------
            READ_PREP: begin
                input_rd_en <= 1'b1;
                result_rd_en <= 1'b1;
                input_rd_addr <= token_idx[ADDR_WIDTH-1:0];
                result_rd_addr <= token_idx[ADDR_WIDTH-1:0];
                
                read_wait_cnt <= 4'd0;
                state <= READ_WAIT;
            end
            
            //----------------------------------------------------------------
            // READ_WAIT: 等待数据返回
            //----------------------------------------------------------------
            READ_WAIT: begin
                read_wait_cnt <= read_wait_cnt + 1;
                
                if (input_valid && result_valid) begin
                    input_exp_reg <= input_exp;
                    input_mant_reg <= input_mant;
                    result_exp_reg <= result_exp;
                    result_mant_reg <= result_mant;
                    state <= ALIGN_EXP;
                end else if (read_wait_cnt > 4'd12) begin
                    // 超时保护
                    state <= ERROR_STATE;
                end
            end
            
            //----------------------------------------------------------------
            // ALIGN_EXP: 指数对齐
            //----------------------------------------------------------------
            ALIGN_EXP: begin
                // 使用 begin 块定义局部变量
                begin : align_block
                    reg signed [EXP_WIDTH:0] exp_diff_temp;
                    exp_diff_temp = $signed({1'b0, input_exp_reg}) - $signed({1'b0, result_exp_reg});
                    
                    if (exp_diff_temp > 0) begin
                        // Input 指数大
                        aligned_exp <= input_exp_reg;
                        for (i = 0; i < DIM; i = i + 1) begin
                            aligned_mant_a[i] <= {input_mant_array[i], {GUARD_BITS{1'b0}}};
                            
                            if (exp_diff_temp >= (DATA_WIDTH + GUARD_BITS)) begin
                                aligned_mant_b[i] <= 0;
                            end else begin
                                aligned_mant_b[i] <= ({result_mant_array[i], {GUARD_BITS{1'b0}}}) >>> exp_diff_temp[4:0];
                            end
                        end
                    end else if (exp_diff_temp < 0) begin
                        // Result 指数大
                        aligned_exp <= result_exp_reg;
                        for (i = 0; i < DIM; i = i + 1) begin
                            aligned_mant_b[i] <= {result_mant_array[i], {GUARD_BITS{1'b0}}};
                            
                            if ((-exp_diff_temp) >= (DATA_WIDTH + GUARD_BITS)) begin
                                aligned_mant_a[i] <= 0;
                            end else begin
                                aligned_mant_a[i] <= ({input_mant_array[i], {GUARD_BITS{1'b0}}}) >>> (-exp_diff_temp);
                            end
                        end
                    end else begin
                        // 指数相等
                        aligned_exp <= input_exp_reg;
                        for (i = 0; i < DIM; i = i + 1) begin
                            aligned_mant_a[i] <= {input_mant_array[i], {GUARD_BITS{1'b0}}};
                            aligned_mant_b[i] <= {result_mant_array[i], {GUARD_BITS{1'b0}}};
                        end
                    end
                end
                
                state <= ADD_MANTISSA;
            end
            
            //----------------------------------------------------------------
            // ADD_MANTISSA: 尾数相加
            //----------------------------------------------------------------
            ADD_MANTISSA: begin
                for (i = 0; i < DIM; i = i + 1) begin
                    sum_mant[i] <= aligned_mant_a[i] + aligned_mant_b[i];
                end
                process_cycle <= 4'd0;
                state <= FIND_MAX;
            end
            
            //----------------------------------------------------------------
            // FIND_MAX: 树形查找最大绝对值 (5级流水)
            //----------------------------------------------------------------
            FIND_MAX: begin
                process_cycle <= process_cycle + 1;
                
                case (process_cycle)
                    // Cycle 0: 计算绝对值并检测全零
                    4'd0: begin
                        all_zero_flag <= 1'b1;
                        // 修改循环上限为 32
                        for (i = 0; i < 32; i = i + 1) begin
                            if (i < DIM) begin
                                // 有效数据范围：正常计算绝对值
                                if (sum_mant[i][DATA_WIDTH+GUARD_BITS+1]) 
                                    abs_array[i] <= -sum_mant[i];
                                else 
                                    abs_array[i] <= sum_mant[i];
                                
                                // 全零检测只看有效数据
                                if (sum_mant[i] != 0) 
                                    all_zero_flag <= 1'b0;
                            end else begin
                                // 无效数据范围：补零 (Padding)
                                abs_array[i] <= 0;
                            end
                        end
                    end
                                        
                    // Cycle 1: Level 1 (32 -> 16)
                    4'd1: begin
                         for (i = 0; i < 16; i = i + 1) begin
                            if (abs_array[2*i] >= abs_array[2*i+1]) begin
                                max_level1[i] <= abs_array[2*i];
                                pos_level1[i] <= 2*i;
                            end else begin
                                max_level1[i] <= abs_array[2*i+1];
                                pos_level1[i] <= 2*i + 1;
                            end
                        end
                    end
                    
                    // Cycle 2: Level 2 (16 -> 8)
                    4'd2: begin
                        for (i = 0; i < 8; i = i + 1) begin
                            if (max_level1[2*i] >= max_level1[2*i+1]) begin
                                max_level2[i] <= max_level1[2*i];
                                pos_level2[i] <= pos_level1[2*i];
                            end else begin
                                max_level2[i] <= max_level1[2*i+1];
                                pos_level2[i] <= pos_level1[2*i+1];
                            end
                        end
                    end
                    
                    // Cycle 3: Level 3 (8 -> 4)
                    4'd3: begin
                        for (i = 0; i < 4; i = i + 1) begin
                            if (max_level2[2*i] >= max_level2[2*i+1]) begin
                                max_level3[i] <= max_level2[2*i];
                                pos_level3[i] <= pos_level2[2*i];
                            end else begin
                                max_level3[i] <= max_level2[2*i+1];
                                pos_level3[i] <= pos_level2[2*i+1];
                            end
                        end
                    end
                    
                    // Cycle 4: Final Comparison (4 -> 1) 组合逻辑直接得出结果
                    4'd4: begin
                        begin : final_compare
                            reg [DATA_WIDTH+GUARD_BITS+1:0] temp_max0, temp_max1;
                            reg [5:0] temp_pos0, temp_pos1;
                            
                            // 前半部分比较 (0 vs 1)
                            if (max_level3[0] >= max_level3[1]) begin 
                                temp_max0 = max_level3[0]; 
                                temp_pos0 = pos_level3[0]; 
                            end else begin 
                                temp_max0 = max_level3[1]; 
                                temp_pos0 = pos_level3[1]; 
                            end
                            
                            // 后半部分比较 (2 vs 3)
                            if (max_level3[2] >= max_level3[3]) begin 
                                temp_max1 = max_level3[2]; 
                                temp_pos1 = pos_level3[2]; 
                            end else begin 
                                temp_max1 = max_level3[3]; 
                                temp_pos1 = pos_level3[3]; 
                            end
                            
                            // 最终比较
                            if (temp_max0 >= temp_max1) begin
                                max_abs_val <= temp_max0;
                                max_abs_pos <= temp_pos0;
                            end else begin
                                max_abs_val <= temp_max1;
                                max_abs_pos <= temp_pos1;
                            end
                        end
                        
                        process_cycle <= 0;
                        state <= NORMALIZE;
                    end
                endcase
            end
            
            //----------------------------------------------------------------
            // NORMALIZE: 归一化 (2周期)
            //----------------------------------------------------------------
            NORMALIZE: begin
                process_cycle <= process_cycle + 1;
                
                case (process_cycle)
                    // Cycle 0: 计算移位量和新指数
                    4'd0: begin
                         begin : calc_shift
                             reg [4:0] leading_zeros;
                             
                             if (all_zero_flag || max_abs_val == 0) begin
                                shift_amount <= 5'd0;
                                new_exp <= {(EXP_WIDTH+2){1'b0}};
                             end else if (max_abs_val[DATA_WIDTH+GUARD_BITS+1]) begin
                                // 溢出位为1，需要右移
                                shift_amount <= 5'd31; // 特殊标记：31代表右移1位
                                new_exp <= $signed({2'b00, aligned_exp}) + 1;
                             end else begin
                                // 正常左移归一化
                                leading_zeros = count_leading_zeros_fast(max_abs_val);
                                
                                if (leading_zeros > 1) 
                                    shift_amount <= leading_zeros - 1;
                                else 
                                    shift_amount <= 5'd0;
                                
                                new_exp <= $signed({2'b00, aligned_exp}) - $signed({5'b00000, leading_zeros}) + 1;
                             end
                         end
                         
                         // 预先检测溢出状态
                         if (new_exp > 255) overflow_detected <= 1'b1;
                         else if (new_exp < 0) underflow_detected <= 1'b1;
                    end
                    
                    // Cycle 1: 执行移位、舍入、饱和
                    4'd1: begin
                        for (i = 0; i < DIM; i = i + 1) begin
                            begin : norm_logic
                                reg signed [DATA_WIDTH+GUARD_BITS+1:0] temp_shifted;
                                reg signed [DATA_WIDTH:0] temp_rounded;
                                
                                // 1. 移位
                                if (all_zero_flag) begin
                                    temp_shifted = 0;
                                end else if (shift_amount == 5'd31) begin
                                    // 右移1位
                                    temp_shifted = sum_mant[i] >>> 1;
                                end else if (shift_amount > 0) begin
                                    // 左移
                                    temp_shifted = sum_mant[i] << shift_amount;
                                end else begin
                                    temp_shifted = sum_mant[i];
                                end
                                
                                // 2. 舍入
                                if (temp_shifted[GUARD_BITS-1]) begin
                                    // 向上舍入
                                    temp_rounded = $signed({1'b0, temp_shifted[DATA_WIDTH+GUARD_BITS-1:GUARD_BITS]}) + 1;
                                end else begin
                                    // 截断
                                    temp_rounded = {1'b0, temp_shifted[DATA_WIDTH+GUARD_BITS-1:GUARD_BITS]};
                                end
                                
                                // 3. 饱和 (处理舍入导致的溢出)
                                if (temp_rounded[DATA_WIDTH]) begin
                                    // 发生溢出 (例如 127+1 = 128 -> 溢出为负)
                                    if (temp_shifted[DATA_WIDTH+GUARD_BITS+1]) 
                                        norm_mant[i] <= {1'b1, {(DATA_WIDTH-1){1'b0}}}; // 负数饱和 -128
                                    else 
                                        norm_mant[i] <= {1'b0, {(DATA_WIDTH-1){1'b1}}}; // 正数饱和 127
                                    
                                    overflow_detected <= 1'b1;
                                end else begin
                                    norm_mant[i] <= temp_rounded[DATA_WIDTH-1:0];
                                end
                            end
                        end
                        
                        // 4. 指数输出饱和
                        if (new_exp > 255) output_exp <= 8'hFF;
                        else if (new_exp[EXP_WIDTH+1]) output_exp <= 8'h00; // 下溢为0
                        else output_exp <= new_exp[EXP_WIDTH-1:0];
                        
                        state <= WRITE;
                    end
                endcase
            end
            
            //----------------------------------------------------------------
            // WRITE: 等待 Output Ready 并写入
            //----------------------------------------------------------------
            WRITE: begin
                // 尾数打包
                for (i = 0; i < DIM; i = i + 1) begin
                    output_mant[i*DATA_WIDTH +: DATA_WIDTH] <= norm_mant[i];
                end
                
                // 只有当下游 Ready 时才写入并跳转
                if (output_ready) begin
                    output_wr_en <= 1'b1;
                    output_valid <= 1'b1;
                    output_wr_addr <= token_idx[ADDR_WIDTH-1:0];
                    
                    processed_count <= processed_count + 1;
                    state <= NEXT_TOKEN;
                end
                // 若 !output_ready，保持在 WRITE 状态等待，不报错
            end
            
            //----------------------------------------------------------------
            // NEXT_TOKEN: 循环处理下一个 Token
            //----------------------------------------------------------------
            NEXT_TOKEN: begin
                if (token_idx < TOKEN_NUM - 1) begin
                    token_idx <= token_idx + 1;
                    state <= READ_PREP;
                end else begin
                    state <= DONE_STATE;
                end
            end
            
            //----------------------------------------------------------------
            // DONE_STATE: 完成
            //----------------------------------------------------------------
            DONE_STATE: begin
                done <= 1'b1;
                if (!start) state <= IDLE;
            end
            
            //----------------------------------------------------------------
            // ERROR_STATE: 错误处理
            //----------------------------------------------------------------
            ERROR_STATE: begin
                error <= 1'b1;
                if (!start) state <= IDLE;
            end
            
            default: state <= IDLE;
        endcase
        
        // 性能计数器
        if (state != IDLE && state != DONE_STATE && state != ERROR_STATE) begin
            cycle_count <= cycle_count + 1;
        end
    end
end

//================================================================================
// 忙信号输出
//================================================================================
assign busy = (state != IDLE) && (state != DONE_STATE) && (state != ERROR_STATE);

endmodule