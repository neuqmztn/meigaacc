module sparse_sum_4to1 (
    input  wire signed [16:0] in0,
    input  wire signed [16:0] in1,
    input  wire signed [16:0] in2,
    input  wire signed [16:0] in3,
    input  wire [3:0] skip,
    output wire [2:0] valid_count,
    output reg  signed [16:0] sum
);

// 计算有效输入数量
wire [2:0] count_0, count_1, count_2, count_3;
assign count_0 = skip[0] ? 3'd0 : 3'd1;
assign count_1 = skip[1] ? 3'd0 : 3'd1;
assign count_2 = skip[2] ? 3'd0 : 3'd1;
assign count_3 = skip[3] ? 3'd0 : 3'd1;

assign valid_count = count_0 + count_1 + count_2 + count_3;

// 根据有效数量和skip模式选择最优求和路径
always @(*) begin
    case (valid_count)
        3'd0: begin
            // 全部为零
            sum = 17'sd0;
        end
        
        3'd1: begin
            // 单个有效，直接传递
            if (!skip[0])      sum = in0;
            else if (!skip[1]) sum = in1;
            else if (!skip[2]) sum = in2;
            else               sum = in3;
        end
        
        3'd2: begin
            // 两个有效，单级加法
            // 修复：确保skip模式与求和操作匹配
            casez (skip)
                4'b??00: sum = in0 + in1;  // skip[1:0]=00 → in0,in1有效
                4'b?0?0: sum = in0 + in2;  // skip[2]=0,skip[0]=0 → in0,in2有效
                4'b?00?: sum = in1 + in2;  // 修复！skip[2:1]=00 → in1,in2有效
                4'b0??0: sum = in0 + in3;  // 修复！skip[3]=0,skip[0]=0 → in0,in3有效
                4'b0?0?: sum = in1 + in3;  // skip[3]=0,skip[1]=0 → in1,in3有效
                4'b00??: sum = in2 + in3;  // skip[3:2]=00 → in2,in3有效
                default: sum = 17'sd0;
            endcase
        end
        
        3'd3: begin
            // 三个有效，两级加法
            if (skip[0])      sum = in1 + in2 + in3;
            else if (skip[1]) sum = in0 + in2 + in3;
            else if (skip[2]) sum = in0 + in1 + in3;
            else              sum = in0 + in1 + in2;
        end
        
        default: begin
            // 全部有效，完整求和
            sum = in0 + in1 + in2 + in3;
        end
    endcase
end

endmodule