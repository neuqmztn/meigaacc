`timescale 1ns / 1ps

module classification_subsystem #(
    parameter DIM = 32,              
    parameter DATA_WIDTH = 16      
)(
    //==========================================================================
    // 时钟和复位
    //==========================================================================
    input  wire clk,
    input  wire rst_n,
    
    //==========================================================================
    // 控制接口
    //==========================================================================
    input  wire start,             
    input  wire train_mode,          // 1=训练模式，0=推理模式
    output wire done,                // 计算完成
    output wire busy,                // 计算中
    
    //==========================================================================
    // 标签输入（仅训练模式）
    //==========================================================================
    input  wire true_label,          // 真实标签（0或1）
    
    //==========================================================================
    // LOB读取接口 - CLS token
    //==========================================================================
    output wire lob_rd_cls_en,
    input  wire [DIM*DATA_WIDTH-1:0] lob_cls_data_q412,
    input  wire lob_cls_valid,
    
    //==========================================================================
    // 权重加载接口（初始化用）
    //==========================================================================
    input  wire weight_load_en,
    input  wire [4:0] weight_load_addr,
    input  wire [DATA_WIDTH-1:0] weight_load_data,
    input  wire [DATA_WIDTH-1:0] bias_load_data,
    
    //==========================================================================
    // 分类结果输出
    //==========================================================================
    output wire result_valid,
    output wire [DATA_WIDTH-1:0] prob,
    output wire predicted_class,
    
    //==========================================================================
    // 误差输出（仅训练模式）
    //==========================================================================
    output wire [DATA_WIDTH-1:0] error,
    output wire error_valid,
    
    //==========================================================================
    // 调试接口
    //==========================================================================
    output wire [3:0] state,
    output wire [DATA_WIDTH-1:0] debug_logit
);

//================================================================================
// 内部信号：权重读取
//================================================================================
wire [4:0] weight_rd_addr;
wire [DATA_WIDTH-1:0] weight_data;
wire [DATA_WIDTH-1:0] bias;

//================================================================================
// 模块实例化：Classification Weight Storage
//================================================================================
classification_weight_storage #(
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH)
) u_cls_weight_storage (
    .clk(clk),
    .rst_n(rst_n),
    
    // 读接口（推理和训练都用）
    .rd_addr(weight_rd_addr),
    .rd_data(weight_data),
    .bias(bias),
    
    // 加载接口（初始化用）
    .init_en(weight_load_en),         
    .init_addr(weight_load_addr),     
    .init_data(weight_load_data),     
    .init_bias_en(weight_load_en),    
    .init_bias(bias_load_data)
);

//================================================================================
// 模块实例化：Classification Error Module
//================================================================================
classification_error_module #(
    .DIM(DIM),
    .DATA_WIDTH(DATA_WIDTH)
) u_cls_error_module (
    .clk(clk),
    .rst_n(rst_n),
    
    // 控制
    .start(start),
    .train_mode(train_mode),
    .done(done),
    .busy(busy),
    
    // 标签输入
    .true_label(true_label),
    
    // LOB读取接口
    .lob_rd_cls_en(lob_rd_cls_en),
    .lob_cls_data_q412(lob_cls_data_q412),
    .lob_cls_valid(lob_cls_valid),
    
    // 权重读取接口
    .weight_addr(weight_rd_addr),
    .weight_data(weight_data),
    .bias(bias),
    
    // 分类结果输出
    .prob(prob),
    .predicted_class(predicted_class),
    .result_valid(result_valid),
    
    // 误差输出
    .error(error),
    .error_valid(error_valid),
    
    // 调试
    .state(state),
    .debug_logit(debug_logit)
);

endmodule