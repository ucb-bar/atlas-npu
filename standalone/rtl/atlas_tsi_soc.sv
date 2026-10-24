// TSI targets -> atlas_soc. Two hosts share the fabric:
//   uart_tsi  the UART pins: the PyUARTSI / Chipyard UART-TSI protocol
//   eth_tsi   TSI over UDP (docs/tsi-over-udp.md): the datagram streams of eth_udp_stack, core clock
// Without Ethernet, hold rx_valid at 0 and tx_ready at 1.
module atlas_tsi_soc
    import tl_pkg::*;
#(
    parameter int CLK_HZ      = 50_000_000,
    parameter int BAUD        = 921_600,
    parameter int DRAM_LINE_W = 17,
    parameter int DRAM_LAT    = 8,
    parameter bit EXT_DRAM    = 1'b0,
    parameter logic [31:0] ETH_IP   = 32'hAC14_0112,  // 172.20.1.18, reset value of sys_regs ETH_IP
    parameter logic [47:0] ETH_MAC  = 48'h02_00_41_54_4C_53,  // as eth_udp_stack LOCAL_MAC
    parameter logic [15:0] ETH_PORT = 16'h5453                // as eth_udp_stack PORT
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       rxd,
    output logic       txd,
    output logic       activity,  // toggles when a TSI command or datagram completes
    output logic       error,     // a link error or denied access has occurred
    output logic       halted,
    output logic [2:0] user_leds,

    // Ethernet TSI: datagrams with the 8-byte prefix, see eth_udp_stack (core clock)
    input  logic [31:0] rx_data,
    input  logic        rx_valid,
    output logic        rx_ready,
    input  logic        rx_last,
    input  logic        rx_abort,    // with rx_last: the datagram was cut short
    output logic [31:0] tx_data,
    output logic        tx_valid,
    input  logic        tx_ready,
    output logic        tx_last,
    input  logic [15:0] pcs_status,  // PCS/PMA status_vector (synchronized here)
    input  logic        phy_ready,   // PHY bring-up done (MDIO writes sent)
    input  logic        phy_id_ok,   // PHY ID read over MDIO matched
    input  logic        gmii_alive,  // GMII clock running
    input  logic [15:0] rx_dropped,
    input  logic [15:0] rx_bad,
    output logic [31:0] local_ip,    // sys_regs ETH_IP

    // External DRAM (EXT_DRAM), see atlas_soc
    output logic          ext_a_valid,
    input  logic          ext_a_ready,
    output wa_t           ext_a,
    input  logic          ext_d_valid,
    output logic          ext_d_ready,
    input  wd_t           ext_d,
    output logic          ext_dma_a_valid,
    input  logic          ext_dma_a_ready,
    output logic [2:0]    ext_dma_a_opcode,
    output logic [5:0]    ext_dma_a_source,
    output logic [63:0]   ext_dma_a_address,
    output logic [31:0]   ext_dma_a_mask,
    output logic [255:0]  ext_dma_a_data,
    input  logic          ext_dma_d_valid,
    output logic          ext_dma_d_ready,
    input  logic [2:0]    ext_dma_d_opcode,
    input  logic [5:0]    ext_dma_d_source,
    input  logic [255:0]  ext_dma_d_data,
    input  logic          ext_dram_ready
);
    localparam int CLKS_PER_BIT = (CLK_HZ + BAUD / 2) / BAUD;

    logic a_valid, a_ready, d_valid, d_ready;
    a_t   a;
    d_t   d;
    logic dropped, bad_cmd, done, denied;

    uart_tsi #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tsi (
        .clk, .rst, .rxd, .txd,
        .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
        .dropped, .bad_cmd, .done);

    logic        e_a_valid, e_a_ready, e_d_valid, e_d_ready;
    wa_t         e_a;
    wd_t         e_d;
    logic        eth_synced, eth_done;
    logic [31:0] eth_n_exec, eth_n_dup, eth_n_drop;

    eth_tsi u_eth (
        .clk, .rst,
        .rx_data, .rx_valid, .rx_ready, .rx_last, .rx_abort,
        .tx_data, .tx_valid, .tx_ready, .tx_last,
        .a_valid(e_a_valid), .a_ready(e_a_ready), .a(e_a),
        .d_valid(e_d_valid), .d_ready(e_d_ready), .d(e_d),
        .synced(eth_synced), .n_exec(eth_n_exec), .n_dup(eth_n_dup), .n_drop(eth_n_drop),
        .done(eth_done));

    // The PCS/PMA status comes from the GMII clock domain; its bits are independent.
    (* ASYNC_REG = "TRUE" *) logic [1:0][15:0] pcs_sync;
    always_ff @(posedge clk) pcs_sync <= {pcs_sync[0], pcs_status};

    atlas_soc #(.CLK_HZ(CLK_HZ), .BAUD(BAUD), .DRAM_LINE_W(DRAM_LINE_W), .DRAM_LAT(DRAM_LAT),
                .EXT_DRAM(EXT_DRAM), .ETH(1'b1), .ETH_IP(ETH_IP), .ETH_MAC(ETH_MAC),
                .ETH_PORT(ETH_PORT)) u_soc (
        .clk, .rst, .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
        .e_a_valid, .e_a_ready, .e_a, .e_d_valid, .e_d_ready, .e_d,
        .dropped, .bad_cmd, .denied, .halted, .user_leds,
        .eth_pcs_status(pcs_sync[1]), .eth_phy_ready(phy_ready), .eth_phy_id_ok(phy_id_ok),
        .eth_gmii_alive(gmii_alive), .eth_synced, .eth_n_exec, .eth_n_dup, .eth_n_drop,
        .eth_rx_dropped(rx_dropped), .eth_rx_bad(rx_bad), .local_ip,
        .ext_a_valid, .ext_a_ready, .ext_a,
        .ext_d_valid, .ext_d_ready, .ext_d, .ext_dma_a_valid, .ext_dma_a_ready, .ext_dma_a_opcode,
        .ext_dma_a_source, .ext_dma_a_address, .ext_dma_a_mask, .ext_dma_a_data, .ext_dma_d_valid,
        .ext_dma_d_ready, .ext_dma_d_opcode, .ext_dma_d_source, .ext_dma_d_data, .ext_dram_ready);

    logic denied_seen;
    always_ff @(posedge clk) begin
        if (rst) begin
            activity    <= 1'b0;
            denied_seen <= 1'b0;
        end else begin
            if (done || eth_done) activity <= ~activity;
            if (denied) denied_seen <= 1'b1;
        end
    end
    assign error = dropped | bad_cmd | denied_seen;
endmodule
