//================================================================================
// 子模块：单 Bank LUTRAM
//================================================================================
module  wc_internal_lutram #(
    parameter BANK_DATA_WIDTH = 256,
    parameter BANK_ADDR_WIDTH = 2
)(
    input  wire clk,
    input  wire we,
    input  wire [BANK_ADDR_WIDTH-1:0] waddr,
    input  wire [BANK_DATA_WIDTH-1:0] din,
    input  wire [BANK_ADDR_WIDTH-1:0] raddr,
    output reg [BANK_DATA_WIDTH-1:0] dout
);
    // 强制使用 Distributed RAM
    // 深度=4, 位宽=256 -> Vivado 会用约 40-64 个 LUT (RAM32M) 实现
    (* ram_style = "distributed" *) reg [BANK_DATA_WIDTH-1:0] mem [0:(1<<BANK_ADDR_WIDTH)-1];

    always @(posedge clk) begin
        if (we) mem[waddr] <= din;
        dout <= mem[raddr];  // 同步读
    end

endmodule