// VCU118 Ethernet PHY: the DP83867 1000BASE-T PHY over SGMII (LVDS, 6-wire with the PHY's
// 625 MHz clock) through the PCS/PMA IP gig_ethernet_pcs_pma_0 (fpga/vcu118_sgmii_pcs.tcl).
// Follows verilog-ethernet example/VCU118/fpga_1g/rtl/fpga.v (MIT, upstream 77320a9).
//
// Bring-up in the core clock, after rst:
//   1. phy_reset_n low for 1 ms, then 2 ms for the PHY to leave reset (MDIO needs 195 us).
//   2. MDIO writes to PHY address 3; extended registers through REGCR (0x0D) and ADDAR (0x0E):
//        CFG4          (0x031) = 0x0070  SGMII autonegotiation timer 11 ms
//        SGMIICTL1     (0x0D3) = 0x4000  6-wire SGMII: the PHY drives the 625 MHz clock
//        10M_SGMII_CFG (0x16F) = 0x0015  10 Mb/s over SGMII
//   3. 1 ms after the last write the PCS/PMA core leaves reset; phy_ready = 1.
// The core's clocks (clk125_out = gmii_clk) run only once the PHY drives the 625 MHz clock.
//
// gmii_* and status_vector are in the gmii_clk domain (gmii_rst). gmii_clk_en is 1 at 1 Gb/s and
// pulses at 10/100 Mb/s. status_vector is the core's (bit 0 link status, 1 link sync, 11:10 speed,
// 12 duplex). phy_ready is in the clk domain.
module vcu118_sgmii #(
    parameter int CLK_HZ = 50_000_000
) (
    input  logic        clk,
    input  logic        rst,

    input  logic        phy_sgmii_rx_p,
    input  logic        phy_sgmii_rx_n,
    output logic        phy_sgmii_tx_p,
    output logic        phy_sgmii_tx_n,
    input  logic        phy_sgmii_clk_p,
    input  logic        phy_sgmii_clk_n,
    output logic        phy_reset_n,
    inout  wire         phy_mdio,
    output logic        phy_mdc,

    output logic        gmii_clk,
    output logic        gmii_rst,
    output logic        gmii_clk_en,
    output logic [7:0]  gmii_rxd,
    output logic        gmii_rx_dv,
    output logic        gmii_rx_er,
    input  logic [7:0]  gmii_txd,
    input  logic        gmii_tx_en,
    input  logic        gmii_tx_er,

    output logic [15:0] status_vector,
    output logic        phy_ready
);
    localparam int MS       = CLK_HZ / 1000;
    localparam int PRESCALE = (CLK_HZ + 4_999_999) / 5_000_000 - 1;  // MDC <= 2.5 MHz
    localparam int NW       = 12;                                     // MDIO writes

    // ── Bring-up sequence ───────────────────────────────────────────────────
    typedef enum logic [2:0] {S_RESET, S_WAIT, S_MDIO, S_SETTLE, S_RUN} state_t;
    state_t                      state;
    logic [$clog2(2*MS+1)-1:0]   timer;
    logic [3:0]                  idx;
    logic                        cmd_valid, cmd_ready, mdio_busy;
    logic                        pcs_reset = 1'b1;  // also from configuration until rst
    logic [4:0]                  cmd_reg;
    logic [15:0]                 cmd_data, ext_reg, ext_val;

    // Write idx: REGCR = 0x001F, ADDAR = register, REGCR = 0x401F, ADDAR = value.
    always_comb begin
        case (idx[3:2])
            2'd0:    {ext_reg, ext_val} = {16'h0031, 16'h0070};
            2'd1:    {ext_reg, ext_val} = {16'h00D3, 16'h4000};
            default: {ext_reg, ext_val} = {16'h016F, 16'h0015};
        endcase
        case (idx[1:0])
            2'd0:    {cmd_reg, cmd_data} = {5'h0D, 16'h001F};
            2'd1:    {cmd_reg, cmd_data} = {5'h0E, ext_reg};
            2'd2:    {cmd_reg, cmd_data} = {5'h0D, 16'h401F};
            default: {cmd_reg, cmd_data} = {5'h0E, ext_val};
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state       <= S_RESET;
            timer       <= '0;
            idx         <= '0;
            cmd_valid   <= 1'b0;
            phy_reset_n <= 1'b0;
            pcs_reset   <= 1'b1;
        end else begin
            timer <= timer + 1'b1;
            case (state)
                S_RESET: if (timer == MS) begin
                    phy_reset_n <= 1'b1;
                    timer       <= '0;
                    state       <= S_WAIT;
                end
                S_WAIT: if (timer == 2 * MS) begin
                    cmd_valid <= 1'b1;
                    state     <= S_MDIO;
                end
                S_MDIO: if (cmd_ready) begin
                    idx <= idx + 1'b1;
                    if (idx == NW - 1) begin
                        cmd_valid <= 1'b0;
                        state     <= S_SETTLE;
                    end
                end
                S_SETTLE: begin
                    if (mdio_busy) timer <= '0;
                    else if (timer == MS) begin
                        pcs_reset <= 1'b0;
                        state     <= S_RUN;
                    end
                end
                default: ;
            endcase
        end
    end
    assign phy_ready = state == S_RUN;

    logic mdio_i, mdio_o, mdio_t;

    mdio_master u_mdio (
        .clk, .rst,
        .cmd_phy_addr(5'h03), .cmd_reg_addr(cmd_reg), .cmd_data, .cmd_opcode(2'b01),
        .cmd_valid, .cmd_ready,
        .data_out(), .data_out_valid(), .data_out_ready(1'b1),
        .mdc_o(phy_mdc), .mdio_i, .mdio_o, .mdio_t, .busy(mdio_busy),
        .prescale(8'(PRESCALE)));

    assign mdio_i   = phy_mdio;
    assign phy_mdio = mdio_t ? 1'bz : mdio_o;

    // ── PCS/PMA core ────────────────────────────────────────────────────────
    gig_ethernet_pcs_pma_0 u_pcs (
        .txp_0(phy_sgmii_tx_p), .txn_0(phy_sgmii_tx_n),
        .rxp_0(phy_sgmii_rx_p), .rxn_0(phy_sgmii_rx_n),
        .refclk625_p(phy_sgmii_clk_p), .refclk625_n(phy_sgmii_clk_n),
        .reset(pcs_reset),
        .clk125_out(gmii_clk), .clk312_out(), .rst_125_out(gmii_rst),
        .tx_logic_reset(), .rx_logic_reset(), .tx_locked(), .rx_locked(),
        .tx_pll_clk_out(), .rx_pll_clk_out(),
        .sgmii_clk_r_0(), .sgmii_clk_f_0(), .sgmii_clk_en_0(gmii_clk_en),
        .speed_is_10_100_0(status_vector[11:10] != 2'b10),
        .speed_is_100_0(status_vector[11:10] == 2'b01),
        .gmii_txd_0(gmii_txd), .gmii_tx_en_0(gmii_tx_en), .gmii_tx_er_0(gmii_tx_er),
        .gmii_rxd_0(gmii_rxd), .gmii_rx_dv_0(gmii_rx_dv), .gmii_rx_er_0(gmii_rx_er),
        .gmii_isolate_0(),
        // autonegotiation on; SGMII: link up, acknowledge, full duplex, 1 Gb/s
        .configuration_vector_0(5'b10000),
        .an_interrupt_0(), .an_adv_config_vector_0(16'hD801), .an_restart_config_0(1'b0),
        .status_vector_0(status_vector), .signal_detect_0(1'b1),
        .tx_bsc_rst_out(), .rx_bsc_rst_out(), .tx_bs_rst_out(), .rx_bs_rst_out(),
        .tx_rst_dly_out(), .rx_rst_dly_out(), .tx_bsc_en_vtc_out(), .rx_bsc_en_vtc_out(),
        .tx_bs_en_vtc_out(), .rx_bs_en_vtc_out(), .riu_clk_out(), .riu_addr_out(),
        .riu_wr_data_out(), .riu_wr_en_out(), .riu_nibble_sel_out(),
        .riu_rddata_1(16'b0), .riu_valid_1(1'b0), .riu_prsnt_1(1'b0),
        .riu_rddata_2(16'b0), .riu_valid_2(1'b0), .riu_prsnt_2(1'b0),
        .riu_rddata_3(16'b0), .riu_valid_3(1'b0), .riu_prsnt_3(1'b0),
        .rx_btval_1(), .rx_btval_2(), .rx_btval_3(),
        .tx_dly_rdy_1(1'b1), .rx_dly_rdy_1(1'b1), .rx_vtc_rdy_1(1'b1), .tx_vtc_rdy_1(1'b1),
        .tx_dly_rdy_2(1'b1), .rx_dly_rdy_2(1'b1), .rx_vtc_rdy_2(1'b1), .tx_vtc_rdy_2(1'b1),
        .tx_dly_rdy_3(1'b1), .rx_dly_rdy_3(1'b1), .rx_vtc_rdy_3(1'b1), .tx_vtc_rdy_3(1'b1),
        .tx_rdclk_out());
endmodule
