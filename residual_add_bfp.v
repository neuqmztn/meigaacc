`timescale 1ns / 1ps

module residual_add_bfp #(
    parameter TOKEN_NUM  = 640,
    parameter DIM        = 32,
    parameter DATA_WIDTH = 8,
    parameter EXP_WIDTH  = 8,
    parameter ADDR_WIDTH = 5,
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
// State Definitions
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
// Internal Signals & Registers
//================================================================================
reg [9:0] token_idx;
reg [3:0] read_wait_cnt;
reg [3:0] process_cycle;

// Data Registers
reg [EXP_WIDTH-1:0] input_exp_reg;
reg [DIM*DATA_WIDTH-1:0] input_mant_reg;
reg [EXP_WIDTH-1:0] result_exp_reg;
reg [DIM*DATA_WIDTH-1:0] result_mant_reg;

// Calculation Registers
reg [EXP_WIDTH-1:0] aligned_exp;
reg signed [DATA_WIDTH+GUARD_BITS:0] aligned_mant_a [0:DIM-1];
reg signed [DATA_WIDTH+GUARD_BITS:0] aligned_mant_b [0:DIM-1];
reg signed [DATA_WIDTH+GUARD_BITS+1:0] sum_mant      [0:DIM-1];

// Normalization Registers
reg [DATA_WIDTH+GUARD_BITS+1:0] max_abs_val;
reg [4:0] shift_amount;
reg signed [EXP_WIDTH+1:0] new_exp;
reg signed [DATA_WIDTH-1:0] norm_mant   [0:DIM-1];

reg all_zero_flag;

integer i;

//================================================================================
// Unpack Input Mantissa Arrays
//================================================================================
wire signed [DATA_WIDTH-1:0] input_mant_array  [0:DIM-1];
wire signed [DATA_WIDTH-1:0] result_mant_array [0:DIM-1];

genvar g;
generate
    for (g = 0; g < DIM; g = g + 1) begin : gen_unpack
        assign input_mant_array[g]  = input_mant_reg[g*DATA_WIDTH +: DATA_WIDTH];
        assign result_mant_array[g] = result_mant_reg[g*DATA_WIDTH +: DATA_WIDTH];
    end
endgenerate

//================================================================================
// Leading Zero Count Function
//================================================================================
function automatic [4:0] count_leading_zeros_fast;
    input [DATA_WIDTH+GUARD_BITS+1:0] value;
    reg [4:0] count;
    reg [DATA_WIDTH+GUARD_BITS+1:0] tmp;
    begin
        tmp   = value;
        count = 0;
        
        // Check upper 7 bits (Bit 12 down to 6)
        if (tmp[12:6] == 7'b0000000) begin
            count = count + 7;
            tmp   = {tmp[5:0], 7'b0};
        end
        // Check upper 4 bits
        if (tmp[12:9] == 4'b0000) begin
            count = count + 4;
            tmp   = {tmp[8:0], 4'b0};
        end
        // Check upper 2 bits
        if (tmp[12:11] == 2'b00) begin
            count = count + 2;
            tmp   = {tmp[10:0], 2'b0};
        end
        // Check MSB
        if (tmp[12] == 1'b0) begin
            count = count + 1;
        end
        
        count_leading_zeros_fast = count;
    end
endfunction

//================================================================================
// Tree Comparator Registers
//================================================================================
reg [DATA_WIDTH+GUARD_BITS+1:0] abs_array  [0:31];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level1 [0:15];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level2 [0:7];
reg [DATA_WIDTH+GUARD_BITS+1:0] max_level3 [0:3];

//================================================================================
// Main State Machine
//================================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state  <= IDLE;
        done   <= 1'b0;
        error  <= 1'b0;
        
        // Control Signals Reset
        input_rd_en   <= 1'b0;
        input_rd_addr <= {ADDR_WIDTH{1'b0}};
        result_rd_en  <= 1'b0;
        result_rd_addr<= {ADDR_WIDTH{1'b0}};
        
        output_wr_en   <= 1'b0;
        output_wr_addr <= {ADDR_WIDTH{1'b0}};
        output_valid   <= 1'b0;
        output_exp     <= {EXP_WIDTH{1'b0}};
        output_mant    <= {(DIM*DATA_WIDTH){1'b0}};
        
        // Counters Reset
        token_idx      <= 10'd0;
        read_wait_cnt  <= 4'd0;
        process_cycle  <= 4'd0;
        
        processed_count   <= 10'd0;
        cycle_count       <= 32'd0;
        
        overflow_detected <= 1'b0;
        underflow_detected<= 1'b0;
        all_zero_flag     <= 1'b0;
        
        // Data path reset
        input_exp_reg  <= {EXP_WIDTH{1'b0}};
        result_exp_reg <= {EXP_WIDTH{1'b0}};
        aligned_exp    <= {EXP_WIDTH{1'b0}};
        max_abs_val    <= { (DATA_WIDTH+GUARD_BITS+2){1'b0} };
        shift_amount   <= 5'd0;
        new_exp        <= { (EXP_WIDTH+2){1'b0} };
        
        // Arrays reset
        for (i = 0; i < DIM; i = i + 1) begin
            aligned_mant_a[i] <= { (DATA_WIDTH+GUARD_BITS+1){1'b0} };
            aligned_mant_b[i] <= { (DATA_WIDTH+GUARD_BITS+1){1'b0} };
            sum_mant[i]       <= { (DATA_WIDTH+GUARD_BITS+2){1'b0} };
            norm_mant[i]      <= { DATA_WIDTH{1'b0} };
        end
        for (i = 0; i < 32; i = i + 1) begin
            abs_array[i]  <= { (DATA_WIDTH+GUARD_BITS+2){1'b0} };
        end
        for (i = 0; i < 16; i = i + 1) begin
            max_level1[i] <= { (DATA_WIDTH+GUARD_BITS+2){1'b0} };
        end
        for (i = 0; i < 8; i = i + 1) begin
            max_level2[i] <= { (DATA_WIDTH+GUARD_BITS+2){1'b0} };
        end
        for (i = 0; i < 4; i = i + 1) begin
            max_level3[i] <= { (DATA_WIDTH+GUARD_BITS+2){1'b0} };
        end
        
    end else begin
        // Default clear for one-shot signals
        input_rd_en   <= 1'b0;
        result_rd_en  <= 1'b0;
        output_wr_en  <= 1'b0;
        output_valid  <= 1'b0;
        
        case (state)
            //----------------------------------------------------------------
            // IDLE
            //----------------------------------------------------------------
            IDLE: begin
                done               <= 1'b0;
                error              <= 1'b0;
                overflow_detected  <= 1'b0;
                underflow_detected <= 1'b0;
                
                if (start) begin
                    token_idx      <= 10'd0;
                    processed_count<= 10'd0;
                    cycle_count    <= 32'd0;
                    state          <= READ_PREP;
                end
            end
            
            //----------------------------------------------------------------
            // READ_PREP
            //----------------------------------------------------------------
            READ_PREP: begin
                input_rd_en   <= 1'b1;
                result_rd_en  <= 1'b1;
                input_rd_addr <= token_idx[ADDR_WIDTH-1:0];
                result_rd_addr<= token_idx[ADDR_WIDTH-1:0];
                
                read_wait_cnt <= 4'd0;
                state         <= READ_WAIT;
            end
            
            //----------------------------------------------------------------
            // READ_WAIT
            //----------------------------------------------------------------
            READ_WAIT: begin
                read_wait_cnt <= read_wait_cnt + 1'b1;
                
                if (input_valid && result_valid) begin
                    input_exp_reg  <= input_exp;
                    input_mant_reg <= input_mant;
                    result_exp_reg <= result_exp;
                    result_mant_reg<= result_mant;
                    state          <= ALIGN_EXP;
                end else if (read_wait_cnt > 4'd12) begin
                    state          <= ERROR_STATE;
                end
            end
            
            //----------------------------------------------------------------
            // ALIGN_EXP
            //----------------------------------------------------------------
            ALIGN_EXP: begin
                begin : align_block
                    reg signed [EXP_WIDTH:0] exp_diff_temp;
                    integer j;
                    
                    exp_diff_temp = $signed({1'b0, input_exp_reg}) - 
                                    $signed({1'b0, result_exp_reg});
                    
                    if (exp_diff_temp > 0) begin
                        // Input Exp is larger
                        aligned_exp <= input_exp_reg;
                        for (j = 0; j < DIM; j = j + 1) begin
                            aligned_mant_a[j] <= {input_mant_array[j], {GUARD_BITS{1'b0}}};
                            
                            if (exp_diff_temp >= (DATA_WIDTH + GUARD_BITS)) begin
                                aligned_mant_b[j] <= { (DATA_WIDTH+GUARD_BITS+1){1'b0} };
                            end else begin
                                aligned_mant_b[j] <= 
                                    $signed({result_mant_array[j], {GUARD_BITS{1'b0}}}) >>> exp_diff_temp[4:0];
                            end
                        end
                    end else if (exp_diff_temp < 0) begin
                        // Result Exp is larger
                        aligned_exp <= result_exp_reg;
                        for (j = 0; j < DIM; j = j + 1) begin
                            aligned_mant_b[j] <= {result_mant_array[j], {GUARD_BITS{1'b0}}};
                            
                            if ((-exp_diff_temp) >= (DATA_WIDTH + GUARD_BITS)) begin
                                aligned_mant_a[j] <= { (DATA_WIDTH+GUARD_BITS+1){1'b0} };
                            end else begin
                                aligned_mant_a[j] <= 
                                    $signed({input_mant_array[j], {GUARD_BITS{1'b0}}}) >>> (-exp_diff_temp);
                            end
                        end
                    end else begin
                        // Exponents equal
                        aligned_exp <= input_exp_reg;
                        for (j = 0; j < DIM; j = j + 1) begin
                            aligned_mant_a[j] <= {input_mant_array[j],  {GUARD_BITS{1'b0}}};
                            aligned_mant_b[j] <= {result_mant_array[j], {GUARD_BITS{1'b0}}};
                        end
                    end
                end
                
                state <= ADD_MANTISSA;
            end
            
            //----------------------------------------------------------------
            // ADD_MANTISSA
            //----------------------------------------------------------------
            ADD_MANTISSA: begin
                for (i = 0; i < DIM; i = i + 1) begin
                    sum_mant[i] <= aligned_mant_a[i] + aligned_mant_b[i];
                end
                process_cycle <= 4'd0;
                state         <= FIND_MAX;
            end
            
            //----------------------------------------------------------------
            // FIND_MAX
            //----------------------------------------------------------------
            FIND_MAX: begin
                process_cycle <= process_cycle + 1'b1;
                
                case (process_cycle)
                    // Cycle 0: Calculate Abs & all_zero_flag（改写为单次赋值）
                    4'd0: begin : calc_abs_and_zero
                        reg local_all_zero;
                        integer k;
                        
                        local_all_zero = 1'b1;
                        for (k = 0; k < 32; k = k + 1) begin
                            if (k < DIM) begin
                                if (sum_mant[k][DATA_WIDTH+GUARD_BITS+1]) 
                                    abs_array[k] <= -sum_mant[k];
                                else 
                                    abs_array[k] <= sum_mant[k];
                                
                                if (sum_mant[k] != 0) 
                                    local_all_zero = 1'b0;
                            end else begin
                                abs_array[k] <= { (DATA_WIDTH+GUARD_BITS+2){1'b0} };
                            end
                        end
                        all_zero_flag <= local_all_zero;
                    end
                                        
                    // Cycle 1: Level 1 (32 -> 16)
                    4'd1: begin
                        for (i = 0; i < 16; i = i + 1) begin
                            if (abs_array[2*i] >= abs_array[2*i+1])
                                max_level1[i] <= abs_array[2*i];
                            else
                                max_level1[i] <= abs_array[2*i+1];
                        end
                    end
                    
                    // Cycle 2: Level 2 (16 -> 8)
                    4'd2: begin
                        for (i = 0; i < 8; i = i + 1) begin
                            if (max_level1[2*i] >= max_level1[2*i+1])
                                max_level2[i] <= max_level1[2*i];
                            else
                                max_level2[i] <= max_level1[2*i+1];
                        end
                    end
                    
                    // Cycle 3: Level 3 (8 -> 4)
                    4'd3: begin
                        for (i = 0; i < 4; i = i + 1) begin
                            if (max_level2[2*i] >= max_level2[2*i+1])
                                max_level3[i] <= max_level2[2*i];
                            else
                                max_level3[i] <= max_level2[2*i+1];
                        end
                    end
                    
                    // Cycle 4: Final Comparison (4 -> 1)
                    4'd4: begin
                        begin : final_compare
                            reg [DATA_WIDTH+GUARD_BITS+1:0] temp_max0, temp_max1;
                            
                            // 0 vs 1
                            if (max_level3[0] >= max_level3[1]) 
                                temp_max0 = max_level3[0]; 
                            else 
                                temp_max0 = max_level3[1]; 
                            
                            // 2 vs 3
                            if (max_level3[2] >= max_level3[3]) 
                                temp_max1 = max_level3[2]; 
                            else 
                                temp_max1 = max_level3[3]; 
                            
                            // Final
                            if (temp_max0 >= temp_max1)
                                max_abs_val <= temp_max0;
                            else
                                max_abs_val <= temp_max1;
                        end
                        
                        process_cycle <= 4'd0;
                        state         <= NORMALIZE;
                    end
                endcase
            end
            
            //----------------------------------------------------------------
            // NORMALIZE
            //----------------------------------------------------------------
            NORMALIZE: begin
                process_cycle <= process_cycle + 1'b1;
                
                case (process_cycle)
                    // Cycle 0: Calculate shift amount and new exp
                    4'd0: begin
                        begin : calc_shift
                            reg [4:0] leading_zeros;
                            
                            if (all_zero_flag || max_abs_val == 0) begin
                                shift_amount <= 5'd0;
                                new_exp      <= {(EXP_WIDTH+2){1'b0}};
                            end else if (max_abs_val[DATA_WIDTH+GUARD_BITS+1]) begin
                                // Overflow bit set, right shift needed
                                shift_amount <= 5'd31; // 31 => right shift 1
                                new_exp      <= $signed({2'b00, aligned_exp}) + 1;
                            end else begin
                                // Normal left shift
                                leading_zeros = count_leading_zeros_fast(max_abs_val);
                                
                                if (leading_zeros > 1) 
                                    shift_amount <= leading_zeros - 1;
                                else 
                                    shift_amount <= 5'd0;
                                
                                new_exp <= $signed({2'b00, aligned_exp}) - 
                                           $signed({5'b00000, leading_zeros}) + 1;
                            end
                        end
                        
                        // exponent overflow / underflow flag
                        if (new_exp > 255) 
                            overflow_detected <= 1'b1;
                        else if (new_exp < 0) 
                            underflow_detected <= 1'b1;
                    end
                    
                    // Cycle 1: Shift, Round, Saturate
                    4'd1: begin : norm_cycle
                        reg local_overflow;
                        integer t;
                        
                        local_overflow = overflow_detected; // 保留之前的标志
                        
                        for (t = 0; t < DIM; t = t + 1) begin:w
                            reg signed [DATA_WIDTH+GUARD_BITS+1:0] temp_shifted;
                            reg signed [DATA_WIDTH:0]              temp_rounded;
                            
                            // 1. Shift
                            if (all_zero_flag) begin
                                temp_shifted = { (DATA_WIDTH+GUARD_BITS+2){1'b0} };
                            end else if (shift_amount == 5'd31) begin
                                temp_shifted = sum_mant[t] >>> 1;
                            end else if (shift_amount > 0) begin
                                temp_shifted = sum_mant[t] <<< shift_amount;
                            end else begin
                                temp_shifted = sum_mant[t];
                            end
                            
                            // 2. Round
                            if (temp_shifted[GUARD_BITS-1]) begin
                                temp_rounded = $signed({1'b0,
                                               temp_shifted[DATA_WIDTH+GUARD_BITS-1:GUARD_BITS]}) + 1;
                            end else begin
                                temp_rounded = {1'b0,
                                               temp_shifted[DATA_WIDTH+GUARD_BITS-1:GUARD_BITS]};
                            end
                            
                            // 3. Saturate
                            if (temp_rounded[DATA_WIDTH]) begin
                                if (temp_shifted[DATA_WIDTH+GUARD_BITS+1]) 
                                    norm_mant[t] <= {1'b1, {(DATA_WIDTH-1){1'b0}}}; // -128
                                else 
                                    norm_mant[t] <= {1'b0, {(DATA_WIDTH-1){1'b1}}}; // 127
                                
                                local_overflow = 1'b1;   // 只在局部变量上打标记
                            end else begin
                                norm_mant[t] <= temp_rounded[DATA_WIDTH-1:0];
                            end
                        end
                        
                        // 把局部 overflow 写回寄存器（单次赋值）
                        overflow_detected <= local_overflow;
                        
                        // 4. Exponent Saturation
                        if (new_exp > 255) 
                            output_exp <= 8'hFF;
                        else if (new_exp[EXP_WIDTH+1]) 
                            output_exp <= 8'h00;
                        else 
                            output_exp <= new_exp[EXP_WIDTH-1:0];
                        
                        state <= WRITE;
                    end
                endcase
            end
            
            //----------------------------------------------------------------
            // WRITE
            //----------------------------------------------------------------
            WRITE: begin
                for (i = 0; i < DIM; i = i + 1) begin
                    output_mant[i*DATA_WIDTH +: DATA_WIDTH] <= norm_mant[i];
                end
                
                if (output_ready) begin
                    output_wr_en   <= 1'b1;
                    output_valid   <= 1'b1;
                    output_wr_addr <= token_idx[ADDR_WIDTH-1:0];
                    
                    processed_count <= processed_count + 1'b1;
                    state           <= NEXT_TOKEN;
                end
            end
            
            //----------------------------------------------------------------
            // NEXT_TOKEN
            //----------------------------------------------------------------
            NEXT_TOKEN: begin
                if (token_idx < TOKEN_NUM - 1) begin
                    token_idx <= token_idx + 1'b1;
                    state     <= READ_PREP;
                end else begin
                    state     <= DONE_STATE;
                end
            end
            
            //----------------------------------------------------------------
            // DONE
            //----------------------------------------------------------------
            DONE_STATE: begin
                done <= 1'b1;
                if (!start) state <= IDLE;
            end
            
            //----------------------------------------------------------------
            // ERROR
            //----------------------------------------------------------------
            ERROR_STATE: begin
                error <= 1'b1;
                if (!start) state <= IDLE;
            end
            
            default: state <= IDLE;
        endcase
        
        // Performance Counter
        if (state != IDLE && state != DONE_STATE && state != ERROR_STATE) begin
            cycle_count <= cycle_count + 1'b1;
        end
    end
end

assign busy = (state != IDLE) && (state != DONE_STATE) && (state != ERROR_STATE);

endmodule
