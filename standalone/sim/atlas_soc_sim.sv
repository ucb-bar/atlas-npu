// Simulation top: atlas_soc with its host port flattened, so the C++ driver (atlas_sim.cpp) can
// play the host word by word without packed structs. The TSI targets (UART, Ethernet) are not in
// this model: the second host port is tied off. The fabric, AtlasCore and the DRAM are the FPGA
// build's.
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
        .e_a_valid(1'b0), .e_a_ready(), .e_a('0), .e_d_valid(), .e_d_ready(1'b1), .e_d(),
        .dropped(1'b0), .bad_cmd(1'b0), .denied(), .halted, .user_leds(),
        .eth_pcs_status('0), .eth_phy_ready(1'b0), .eth_phy_id_ok(1'b0), .eth_gmii_alive(1'b0),
        .eth_synced(1'b0), .eth_n_exec('0), .eth_n_dup('0), .eth_n_drop('0), .eth_rx_dropped('0),
        .eth_rx_bad('0), .local_ip(),
        .ext_a_valid(), .ext_a_ready(1'b0), .ext_a(), .ext_d_valid(1'b0), .ext_d_ready(), .ext_d('0),
        .ext_dma_a_valid(), .ext_dma_a_ready(1'b0), .ext_dma_a_opcode(), .ext_dma_a_source(),
        .ext_dma_a_address(), .ext_dma_a_mask(), .ext_dma_a_data(), .ext_dma_d_valid(1'b0),
        .ext_dma_d_ready(), .ext_dma_d_opcode('0), .ext_dma_d_source('0), .ext_dma_d_data('0),
        .ext_dram_ready(1'b1));
endmodule
