// VCU118 top of the standalone Atlas SoC. Its DRAM is the board's DDR4 (C1 bank, 2 GiB).
//
//   sys_clock_p/n  250 MHz differential (E12/D12) -> MMCM (VCO 1000 MHz) -> CLK_MHZ design clock
//   reset          CPU_RESET pushbutton (L19), active high
//   uart_rxd/txd   CP2105 UART (AW25/BB21), the TSI link at BAUD 8N1; RTS# (BB22) held asserted
//   c0_ddr4_*      DDR4 C1 component memory, through the DDR4 MIG (IP vcu118mig,
//                  fpga/vcu118_ddr4_mig.tcl); its reference clock is sys_clock, as in the
//                  Chipyard VCU118 shell
//   phy_*          DP83867 Ethernet PHY (RJ45, 1000BASE-T), SGMII over LVDS (vcu118_sgmii). TSI over
//                  UDP port 21587 at ETH_IP (eth_udp_stack, eth_tsi; docs/tsi-over-udp.md).
//                  phy_int_n is not used.
//   led[0]         heartbeat (about 1.5 Hz)
//   led[1]         TSI activity: toggles per completed command (UART) or datagram (Ethernet)
//   led[2]         link error or denied access
//   led[3]         MMCM locked and DDR4 calibrated
//   led[4]         Atlas halted
//   led[7:5]       user LEDs (sys_regs LEDS)
//
// atlas_ddr4_dram bridges the SoC's DRAM ports (core clock) to the MIG's AXI4 port (200 MHz).
// (The SoC's on-chip DRAM, atlas_dram, is only the Verilator model's memory.)
module vcu118_atlas_top
    import tl_pkg::*;
#(
    parameter int          CLK_MHZ = 50,
    parameter int          BAUD    = 921_600,
    parameter logic [31:0] ETH_IP  = 32'hAC14_0112,  // 172.20.1.18, sys_regs ETH_IP reset value
    parameter logic [47:0] ETH_MAC = 48'h02_00_41_54_4C_53
) (
    input  logic        sys_clock_p,
    input  logic        sys_clock_n,
    input  logic        reset,
    input  logic        uart_rxd,
    output logic        uart_txd,
    output logic        uart_rtsn,
    output logic [7:0]  led,

    input  logic        phy_sgmii_rx_p,
    input  logic        phy_sgmii_rx_n,
    output logic        phy_sgmii_tx_p,
    output logic        phy_sgmii_tx_n,
    input  logic        phy_sgmii_clk_p,
    input  logic        phy_sgmii_clk_n,
    output logic        phy_reset_n,
    input  logic        phy_int_n,
    inout  wire         phy_mdio,
    output logic        phy_mdc,

    output logic [16:0] c0_ddr4_adr,
    output logic        c0_ddr4_bg,
    output logic [1:0]  c0_ddr4_ba,
    output logic        c0_ddr4_reset_n,
    output logic        c0_ddr4_act_n,
    output logic        c0_ddr4_ck_c,
    output logic        c0_ddr4_ck_t,
    output logic        c0_ddr4_cke,
    output logic        c0_ddr4_cs_n,
    output logic        c0_ddr4_odt,
    inout  wire  [63:0] c0_ddr4_dq,
    inout  wire  [7:0]  c0_ddr4_dqs_c,
    inout  wire  [7:0]  c0_ddr4_dqs_t,
    inout  wire  [7:0]  c0_ddr4_dm_dbi_n
);
    logic clk_in, clk_fb, clk_mmcm, clk, locked;

    IBUFDS u_ibufds (.I(sys_clock_p), .IB(sys_clock_n), .O(clk_in));

    MMCME4_BASE #(
        .CLKIN1_PERIOD    (4.0),
        .DIVCLK_DIVIDE    (1),
        .CLKFBOUT_MULT_F  (4.0),
        .CLKOUT0_DIVIDE_F (1000.0 / CLK_MHZ)
    ) u_mmcm (
        .CLKIN1   (clk_in),
        .CLKFBIN  (clk_fb),
        .CLKFBOUT (clk_fb),
        .CLKOUT0  (clk_mmcm),
        .LOCKED   (locked),
        .RST      (1'b0),
        .PWRDWN   (1'b0),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
        .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6());

    BUFG u_bufg (.I(clk_mmcm), .O(clk));

    // Reset: held until the MMCM locks and for 256 cycles after, and while CPU_RESET is pressed.
    // CPU_RESET does not reset the MIG (that would recalibrate and disturb DRAM contents). It does
    // restart the Ethernet PHY bring-up (the link goes down and renegotiates).
    (* ASYNC_REG = "TRUE" *) logic [1:0] btn_sync;
    logic [8:0] por;
    logic       rst;
    always_ff @(posedge clk) btn_sync <= {btn_sync[0], reset};
    always_ff @(posedge clk or negedge locked) begin
        if (!locked)          por <= '0;
        else if (btn_sync[1]) por <= '0;
        else if (!por[8])     por <= por + 1'b1;
    end
    always_ff @(posedge clk) rst <= !por[8];

    // The MIG's own reset: power-on only.
    logic [7:0] mig_por = '0;
    always_ff @(posedge clk_in) if (!(&mig_por)) mig_por <= mig_por + 1'b1;
    wire mig_rst = !(&mig_por);

    // ── Ethernet: PHY, PCS/PMA, MAC/IP/UDP ──────────────────────────────────
    localparam logic [15:0] ETH_PORT = 16'h5453;

    logic        gmii_clk, gmii_rst, gmii_clk_en, gmii_rx_dv, gmii_rx_er, gmii_tx_en, gmii_tx_er;
    logic [7:0]  gmii_rxd, gmii_txd;
    logic [15:0] pcs_status, rx_dropped, rx_bad;
    logic        phy_ready;
    logic [31:0] local_ip, eth_rx_data, eth_tx_data;
    logic        eth_rx_valid, eth_rx_ready, eth_rx_last, eth_tx_valid, eth_tx_ready, eth_tx_last;

    vcu118_sgmii #(.CLK_HZ(CLK_MHZ * 1_000_000)) u_sgmii (
        .clk, .rst,
        .phy_sgmii_rx_p, .phy_sgmii_rx_n, .phy_sgmii_tx_p, .phy_sgmii_tx_n,
        .phy_sgmii_clk_p, .phy_sgmii_clk_n, .phy_reset_n, .phy_mdio, .phy_mdc,
        .gmii_clk, .gmii_rst, .gmii_clk_en, .gmii_rxd, .gmii_rx_dv, .gmii_rx_er,
        .gmii_txd, .gmii_tx_en, .gmii_tx_er,
        .status_vector(pcs_status), .phy_ready);

    eth_udp_stack #(.LOCAL_MAC(ETH_MAC), .PORT(ETH_PORT)) u_eth (
        .gmii_clk, .gmii_rst, .gmii_clk_en, .gmii_rxd, .gmii_rx_dv, .gmii_rx_er,
        .gmii_txd, .gmii_tx_en, .gmii_tx_er,
        .clk, .rst, .local_ip,
        .rx_data(eth_rx_data), .rx_valid(eth_rx_valid), .rx_ready(eth_rx_ready),
        .rx_last(eth_rx_last),
        .tx_data(eth_tx_data), .tx_valid(eth_tx_valid), .tx_ready(eth_tx_ready),
        .tx_last(eth_tx_last),
        .rx_dropped, .rx_bad);

    // ── SoC ─────────────────────────────────────────────────────────────────
    logic          activity, error, halted, calibrated;
    logic [2:0]    user_leds;
    logic          ext_a_valid, ext_a_ready, ext_d_valid, ext_d_ready;
    wa_t           ext_a;
    wd_t           ext_d;
    logic          dma_a_valid, dma_a_ready, dma_d_valid, dma_d_ready;
    logic [2:0]    dma_a_opcode, dma_d_opcode;
    logic [5:0]    dma_a_source, dma_d_source;
    logic [63:0]   dma_a_address;
    logic [31:0]   dma_a_mask;
    logic [255:0]  dma_a_data, dma_d_data;

    atlas_tsi_soc #(.CLK_HZ(CLK_MHZ * 1_000_000), .BAUD(BAUD), .EXT_DRAM(1'b1), .ETH_IP(ETH_IP),
                    .ETH_MAC(ETH_MAC), .ETH_PORT(ETH_PORT)) u_soc (
        .clk, .rst, .rxd(uart_rxd), .txd(uart_txd),
        .activity, .error, .halted, .user_leds,
        .rx_data(eth_rx_data), .rx_valid(eth_rx_valid), .rx_ready(eth_rx_ready),
        .rx_last(eth_rx_last),
        .tx_data(eth_tx_data), .tx_valid(eth_tx_valid), .tx_ready(eth_tx_ready),
        .tx_last(eth_tx_last),
        .pcs_status, .phy_ready, .rx_dropped, .rx_bad, .local_ip,
        .ext_a_valid, .ext_a_ready, .ext_a, .ext_d_valid, .ext_d_ready, .ext_d,
        .ext_dma_a_valid(dma_a_valid), .ext_dma_a_ready(dma_a_ready),
        .ext_dma_a_opcode(dma_a_opcode), .ext_dma_a_source(dma_a_source),
        .ext_dma_a_address(dma_a_address), .ext_dma_a_mask(dma_a_mask), .ext_dma_a_data(dma_a_data),
        .ext_dma_d_valid(dma_d_valid), .ext_dma_d_ready(dma_d_ready),
        .ext_dma_d_opcode(dma_d_opcode), .ext_dma_d_source(dma_d_source),
        .ext_dma_d_data(dma_d_data), .ext_dram_ready(calibrated));

    // ── DDR4: bridge and MIG ────────────────────────────────────────────────
    logic          ui_clk, ui_rst, calib_done, aresetn;
    logic [3:0]    awid, arid, bid, rid;
    logic [30:0]   awaddr, araddr;
    logic [7:0]    awlen, arlen;
    logic [2:0]    awsize, arsize, awprot, arprot;
    logic [1:0]    awburst, arburst, bresp, rresp;
    logic          awlock, arlock, wlast, rlast;
    logic [3:0]    awcache, arcache, awqos, arqos;
    logic          awvalid, awready, wvalid, wready, bvalid, bready;
    logic          arvalid, arready, rvalid, rready;
    logic [255:0]  wdata, rdata;
    logic [31:0]   wstrb;

    atlas_ddr4_dram #(.SW(6), .AXI_AW(31)) u_dram (
        .clk, .rst,
        .a_valid(ext_a_valid), .a_ready(ext_a_ready), .a(ext_a),
        .d_valid(ext_d_valid), .d_ready(ext_d_ready), .d(ext_d),
        .dma_a_valid, .dma_a_ready, .dma_a_opcode, .dma_a_source, .dma_a_address, .dma_a_mask,
        .dma_a_data, .dma_d_valid, .dma_d_ready, .dma_d_opcode, .dma_d_source, .dma_d_data,
        .calibrated,
        .ui_clk, .ui_rst, .calib_done, .aresetn,
        .awid, .awaddr, .awlen, .awsize, .awburst, .awlock, .awcache, .awprot, .awqos, .awvalid,
        .awready, .wdata, .wstrb, .wlast, .wvalid, .wready, .bid, .bresp, .bvalid, .bready,
        .arid, .araddr, .arlen, .arsize, .arburst, .arlock, .arcache, .arprot, .arqos, .arvalid,
        .arready, .rid, .rdata, .rresp, .rlast, .rvalid, .rready);

    vcu118mig u_mig (
        .c0_ddr4_adr, .c0_ddr4_bg, .c0_ddr4_ba, .c0_ddr4_reset_n, .c0_ddr4_act_n,
        .c0_ddr4_ck_c, .c0_ddr4_ck_t, .c0_ddr4_cke, .c0_ddr4_cs_n, .c0_ddr4_odt,
        .c0_ddr4_dq, .c0_ddr4_dqs_c, .c0_ddr4_dqs_t, .c0_ddr4_dm_dbi_n,
        .c0_sys_clk_i            (clk_in),
        .sys_rst                 (mig_rst),
        .c0_ddr4_ui_clk          (ui_clk),
        .c0_ddr4_ui_clk_sync_rst (ui_rst),
        .c0_init_calib_complete  (calib_done),
        .c0_ddr4_aresetn         (aresetn),
        .c0_ddr4_s_axi_awid      (awid),    .c0_ddr4_s_axi_awaddr  (awaddr),
        .c0_ddr4_s_axi_awlen     (awlen),   .c0_ddr4_s_axi_awsize  (awsize),
        .c0_ddr4_s_axi_awburst   (awburst), .c0_ddr4_s_axi_awlock  (awlock),
        .c0_ddr4_s_axi_awcache   (awcache), .c0_ddr4_s_axi_awprot  (awprot),
        .c0_ddr4_s_axi_awqos     (awqos),   .c0_ddr4_s_axi_awvalid (awvalid),
        .c0_ddr4_s_axi_awready   (awready),
        .c0_ddr4_s_axi_wdata     (wdata),   .c0_ddr4_s_axi_wstrb   (wstrb),
        .c0_ddr4_s_axi_wlast     (wlast),   .c0_ddr4_s_axi_wvalid  (wvalid),
        .c0_ddr4_s_axi_wready    (wready),
        .c0_ddr4_s_axi_bready    (bready),  .c0_ddr4_s_axi_bid     (bid),
        .c0_ddr4_s_axi_bresp     (bresp),   .c0_ddr4_s_axi_bvalid  (bvalid),
        .c0_ddr4_s_axi_arid      (arid),    .c0_ddr4_s_axi_araddr  (araddr),
        .c0_ddr4_s_axi_arlen     (arlen),   .c0_ddr4_s_axi_arsize  (arsize),
        .c0_ddr4_s_axi_arburst   (arburst), .c0_ddr4_s_axi_arlock  (arlock),
        .c0_ddr4_s_axi_arcache   (arcache), .c0_ddr4_s_axi_arprot  (arprot),
        .c0_ddr4_s_axi_arqos     (arqos),   .c0_ddr4_s_axi_arvalid (arvalid),
        .c0_ddr4_s_axi_arready   (arready),
        .c0_ddr4_s_axi_rready    (rready),  .c0_ddr4_s_axi_rid     (rid),
        .c0_ddr4_s_axi_rdata     (rdata),   .c0_ddr4_s_axi_rresp   (rresp),
        .c0_ddr4_s_axi_rlast     (rlast),   .c0_ddr4_s_axi_rvalid  (rvalid));

    logic [24:0] beat;
    always_ff @(posedge clk) beat <= beat + 1'b1;

    assign uart_rtsn = 1'b0;
    assign led       = {user_leds, halted, locked && calibrated, error, activity, beat[24]};
endmodule
