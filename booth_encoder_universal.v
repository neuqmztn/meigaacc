`timescale 1ns / 1ps

module booth_encoder_universal (
    input  wire [7:0] multiplicand,   
    input  wire       unsigned_mode,  
    output wire [14:0] k_packed      
);


    wire [10:0] B_ext;

    assign B_ext = unsigned_mode ?
                   {2'b00,      multiplicand, 1'b0} :
                   {multiplicand[7], multiplicand[7], multiplicand, 1'b0};

    wire [2:0] window [0:4];

    assign window[0] = B_ext[ 2: 0];  // bit [2:0]
    assign window[1] = B_ext[ 4: 2];  // bit [4:2]
    assign window[2] = B_ext[ 6: 4];  // bit [6:4]
    assign window[3] = B_ext[ 8: 6];  // bit [8:6]
    assign window[4] = B_ext[10: 8];  // bit [10:8]


    wire [2:0] code [0:4];

    genvar i;
    generate
        for (i = 0; i < 5; i = i + 1) begin : gen_enc
            booth_encode_3bits u_enc (
                .bits(window[i]),
                .code(code[i])
            );
        end
    endgenerate

    // 打包输出：{code4, code3, code2, code1, code0}
    assign k_packed = {code[4], code[3], code[2], code[1], code[0]};

endmodule

module booth_encode_3bits (
    input  wire [2:0] bits,
    output reg  [2:0] code
);

    always @(*) begin
        case (bits)
            3'b000: code = 3'b000;
            3'b001: code = 3'b001;
            3'b010: code = 3'b001;
            3'b011: code = 3'b010;
            3'b100: code = 3'b110;
            3'b101: code = 3'b101;
            3'b110: code = 3'b101;
            3'b111: code = 3'b000;
            default: code = 3'b000;
        endcase
    end
endmodule
