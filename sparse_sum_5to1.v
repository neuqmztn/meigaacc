`timescale 1ns / 1ps

//==============================================================
// sparse_sum_5to1.v - 5-Input Sparse Aware Adder (Fixed)
//==============================================================

module sparse_sum_5to1 (
    input  wire signed [18:0] in0,
    input  wire signed [18:0] in1,
    input  wire signed [18:0] in2,
    input  wire signed [18:0] in3,
    input  wire signed [18:0] in4,
    input  wire [4:0] skip,              // Skip flags (1=skip)
    output wire [2:0] valid_count,       // Valid input count
    output wire signed [18:0] sum        // Sum result
);
    //--------------------------------------------------------------
    // Step 1: Count valid inputs (FIXED)
    //--------------------------------------------------------------
    // We concatenate {2'b0, bit} to force the adder to use 3-bit precision
    // preventing 1+1=0 overflow errors.
    assign valid_count = {2'b0, ~skip[0]} + 
                         {2'b0, ~skip[1]} + 
                         {2'b0, ~skip[2]} + 
                         {2'b0, ~skip[3]} + 
                         {2'b0, ~skip[4]};

    //--------------------------------------------------------------
    // Step 2: 3-Stage Adder Tree
    //--------------------------------------------------------------
    
    // --- Level 1: 5 inputs -> 3 outputs ---
    // 19-bit inputs -> 20-bit intermediate sums
    wire signed [19:0] level1_sum0; // in0 + in1
    wire signed [19:0] level1_sum1; // in2 + in3
    wire signed [19:0] level1_pass; // in4 pass-through

    assign level1_sum0 = {in0[18], in0} + {in1[18], in1};
    assign level1_sum1 = {in2[18], in2} + {in3[18], in3};
    assign level1_pass = {in4[18], in4};

    // --- Level 2: 3 inputs -> 2 outputs ---
    // 20-bit inputs -> 21-bit intermediate sums
    wire signed [20:0] level2_sum0; // level1_sum0 + level1_sum1
    wire signed [20:0] level2_pass; // level1_pass pass-through

    assign level2_sum0 = {level1_sum0[19], level1_sum0} + {level1_sum1[19], level1_sum1};
    assign level2_pass = {level1_pass[19], level1_pass};

    // --- Level 3: 2 inputs -> 1 output ---
    // 21-bit inputs -> 22-bit result
    wire signed [21:0] level3_sum;
    assign level3_sum = {level2_sum0[20], level2_sum0} + {level2_pass[20], level2_pass};

    //--------------------------------------------------------------
    // Step 3: Output Truncation
    //--------------------------------------------------------------
    // The maximum result fits within 19 bits.
    assign sum = level3_sum[18:0];

endmodule