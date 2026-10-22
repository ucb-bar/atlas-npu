// Simulation top: atlas_soc with its host port flattened, so the C++ driver (atlas_sim.cpp) can
// play the host word by word without packed structs. The UART TSI target is not in this model;
// the fabric, AtlasCore and the DRAM are the FPGA build's.
module atlas_soc_sim
    import tl_pkg::*;
#(
    parameter int DRAM_LINE_W = 17
) (
    input  logic        clk,
    input  logic        rst,
    input  logic        h_a_valid,
    output logic        h_a_ready,
    input  logic        h_a_write,
    input  logic [31:0] h_a_address,
    input  logic [31:0] h_a_data,
    output logic        h_d_valid,
    input  logic        h_d_ready,
    output logic        h_d_denied,
    output logic [31:0] h_d_data,
    output logic        halted
);
    a_t a;
    d_t d;
    assign a = '{opcode: h_a_write ? A_PUT_FULL : A_GET, address: h_a_address, mask: 4'hF,
                 data: h_a_data};
    assign h_d_denied = d.denied;
    assign h_d_data   = d.data;

    atlas_soc #(.DRAM_LINE_W(DRAM_LINE_W)) u_soc (
        .clk, .rst,
        .a_valid(h_a_valid), .a_ready(h_a_ready), .a,
        .d_valid(h_d_valid), .d_ready(h_d_ready), .d,
        .dropped(1'b0), .bad_cmd(1'b0), .denied(), .halted, .user_leds(),
        .ext_a_valid(), .ext_a_ready(1'b0), .ext_a(), .ext_d_valid(1'b0), .ext_d_ready(), .ext_d('0),
        .ext_dma_a_valid(), .ext_dma_a_ready(1'b0), .ext_dma_a_opcode(), .ext_dma_a_source(),
        .ext_dma_a_address(), .ext_dma_a_mask(), .ext_dma_a_data(), .ext_dma_d_valid(1'b0),
        .ext_dma_d_ready(), .ext_dma_d_opcode('0), .ext_dma_d_source('0), .ext_dma_d_data('0),
        .ext_dram_ready(1'b1));
endmodule
