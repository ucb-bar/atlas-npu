// Simulation top for the Ethernet TSI co-simulation (eth_cosim.cpp): eth_udp_stack and
// atlas_tsi_soc, with the GMII pins flat. clk is the core clock, gmii_clk the 125 MHz GMII clock
// (on the board: the PCS/PMA core's clk125_out). The UART is idle (rxd held at 1), the DRAM is
// atlas_dram, and the Ethernet status reads as a 1 Gb/s full-duplex link that is up, with the PHY
// ID matched and the GMII clock running.
module atlas_eth_sim
    import tl_pkg::*;
#(
    parameter int          DRAM_LINE_W = 17,
    parameter logic [31:0] ETH_IP      = 32'hAC14_0112,
    parameter logic [47:0] ETH_MAC     = 48'h02_00_41_54_4C_53,
    parameter logic [15:0] ETH_PORT    = 16'h5453
) (
    input  logic        clk,
    input  logic        rst,
    input  logic        gmii_clk,
    input  logic        gmii_rst,
    input  logic [7:0]  gmii_rxd,
    input  logic        gmii_rx_dv,
    input  logic        gmii_rx_er,
    output logic [7:0]  gmii_txd,
    output logic        gmii_tx_en,
    output logic        gmii_tx_er,
    output logic [31:0] local_ip,   // sys_regs ETH_IP
    output logic [47:0] local_mac,
    output logic        halted,
    output logic        activity,
    output logic        error
);
    logic [31:0] rx_data, tx_data;
    logic        rx_valid, rx_ready, rx_last, rx_abort, tx_valid, tx_ready, tx_last;
    logic [15:0] rx_dropped, rx_bad;

    assign local_mac = ETH_MAC;

    eth_udp_stack #(.LOCAL_MAC(ETH_MAC), .PORT(ETH_PORT)) u_eth (
        .gmii_clk, .gmii_rst, .gmii_clk_en(1'b1),
        .gmii_rxd, .gmii_rx_dv, .gmii_rx_er, .gmii_txd, .gmii_tx_en, .gmii_tx_er,
        .clk, .rst, .local_ip,
        .rx_data, .rx_valid, .rx_ready, .rx_last, .rx_abort,
        .tx_data, .tx_valid, .tx_ready, .tx_last,
        .rx_dropped, .rx_bad);

    atlas_tsi_soc #(.DRAM_LINE_W(DRAM_LINE_W), .ETH_IP(ETH_IP), .ETH_MAC(ETH_MAC),
                    .ETH_PORT(ETH_PORT)) u_soc (
        .clk, .rst, .rxd(1'b1), .txd(), .activity, .error, .halted, .user_leds(),
        .rx_data, .rx_valid, .rx_ready, .rx_last, .rx_abort,
        .tx_data, .tx_valid, .tx_ready, .tx_last,
        .pcs_status(16'h1883), .phy_ready(1'b1), .phy_id_ok(1'b1), .gmii_alive(1'b1),
        .rx_dropped, .rx_bad, .local_ip,
        .ext_a_valid(), .ext_a_ready(1'b0), .ext_a(), .ext_d_valid(1'b0), .ext_d_ready(), .ext_d('0),
        .ext_dma_a_valid(), .ext_dma_a_ready(1'b0), .ext_dma_a_opcode(), .ext_dma_a_source(),
        .ext_dma_a_address(), .ext_dma_a_mask(), .ext_dma_a_data(), .ext_dma_d_valid(1'b0),
        .ext_dma_d_ready(), .ext_dma_d_opcode('0), .ext_dma_d_source('0), .ext_dma_d_data('0),
        .ext_dram_ready(1'b1));
endmodule
