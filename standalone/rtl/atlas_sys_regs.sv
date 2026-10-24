// System registers of the standalone Atlas SoC (word device at 0x0000_0000).
//
//   0x000  ID          RO  0x4154_4C53 ("ATLS")
//   0x004  SCRATCH     RW
//   0x008  FLAGS       RO  [0] TSI receive queue overflowed, [1] bad TSI cmd, [2] a denied access occurred,
//                          [3] DRAM ready (DDR4 calibrated; always 1 in the Verilator model)
//   0x00C  DENIED      RO  number of denied (unmapped) accesses
//   0x010  CYCLES      RO  free-running cycle counter
//   0x014  CLK_HZ      RO
//   0x018  BAUD        RO
//   0x01C  DRAM_BYTES  RO  size of the DRAM (aliased over 0x8000_0000 up): 2 GiB of DDR4 on the board
//   0x020  ATLAS       RO  [0] halted, [1] MXU0 data busy, [2] MXU0 compute busy, [3] MXU1 data busy,
//                          [4] MXU1 compute busy, [5] XLU busy, [6] LSU busy, [15:8] DMA channel busy
//   0x024  FEATURES    RO  [0] hardware interlock (DELAY not required), [1] systolic-array MXU,
//                          [2] inner-product-tree MXU, [3] NVFP4 MXU (AtlasCore io_features)
//   0x028  LEDS        RW  [2:0] user LEDs
//   0x02C  RUN_CYCLES  RW  cycles the Atlas scalar core has spent running (not halted); any write clears it
//   0x030  ETH_IP      RW  the FPGA's IPv4 address (TSI over UDP)
//   0x034  ETH_STATUS  RO  [15:0] PCS/PMA status_vector (0 link status, 1 link sync, 11:10 speed,
//                          12 duplex; 0 while the GMII clock is stopped), [16] PHY bring-up done
//                          (MDIO writes sent), [17] Ethernet TSI session open, [18] Ethernet present
//                          in this build, [19] PHY ID matched (DP83867), [20] GMII clock running
//   0x038  ETH_RUN     RO  datagrams run
//   0x03C  ETH_DUP     RO  datagrams acknowledged DUP (not run again)
//   0x040  ETH_DROP    RO  datagrams dropped or not run
//   0x044  ETH_PORT    RO  UDP port
//   0x048  ETH_MAC_LO  RO  MAC address [31:0]
//   0x04C  ETH_MAC_HI  RO  MAC address [47:32]
//   0x050  ETH_RXERR   RO  [15:0] frames dropped before the TSI target, [31:16] frames with a bad FCS
//                          or a GMII error
//   0x054  ETH_RXFRAMES RO [15:0] frames at the GMII input, [31:16] frames out of the MAC
//
// In a build without Ethernet (ETH = 0, the word-level Verilator model) ETH_STATUS and the
// counters read 0.
module atlas_sys_regs
    import tl_pkg::*;
#(
    parameter int CLK_HZ     = 50_000_000,
    parameter int BAUD       = 921_600,
    parameter logic [31:0] DRAM_BYTES = 32'(4 << 20),
    parameter bit          ETH        = 1'b0,
    parameter logic [31:0] ETH_IP     = 32'hAC14_0112,  // 172.20.1.18
    parameter logic [47:0] ETH_MAC    = 48'h02_00_41_54_4C_53,
    parameter logic [15:0] ETH_PORT   = 16'h5453
) (
    input  logic        clk,
    input  logic        rst,
    input  logic        a_valid,
    output logic        a_ready,
    input  a_t          a,
    output logic        d_valid,
    input  logic        d_ready,
    output d_t          d,

    input  logic        dropped,
    input  logic        bad_cmd,
    input  logic        denied,
    input  logic [15:0] atlas_status,
    input  logic [3:0]  features,
    input  logic        dram_ready,
    output logic [2:0]  user_leds,

    input  logic [15:0] eth_pcs_status,
    input  logic        eth_phy_ready,
    input  logic        eth_phy_id_ok,
    input  logic        eth_gmii_alive,
    input  logic        eth_synced,
    input  logic [31:0] eth_n_exec,
    input  logic [31:0] eth_n_dup,
    input  logic [31:0] eth_n_drop,
    input  logic [15:0] eth_rx_dropped,
    input  logic [15:0] eth_rx_bad,
    input  logic [31:0] eth_rx_frames,
    output logic [31:0] eth_ip
);
    logic [31:0] scratch, denied_count, cycles, run_cycles, rdata;
    logic        hit;

    wire [9:0] word = a.address[11:2];
    wire       wr   = a.opcode != A_GET;

    always_comb begin
        hit = 1'b1;
        case (word)
            10'h00:  rdata = 32'h4154_4C53;
            10'h01:  rdata = scratch;
            10'h02:  rdata = {28'd0, dram_ready, denied_count != 0, bad_cmd, dropped};
            10'h03:  rdata = denied_count;
            10'h04:  rdata = cycles;
            10'h05:  rdata = CLK_HZ;
            10'h06:  rdata = BAUD;
            10'h07:  rdata = DRAM_BYTES;
            10'h08:  rdata = {16'd0, atlas_status};
            10'h09:  rdata = {28'd0, features};
            10'h0A:  rdata = {29'd0, user_leds};
            10'h0B:  rdata = run_cycles;
            10'h0C:  rdata = eth_ip;
            10'h0D:  rdata = {11'd0, eth_gmii_alive, eth_phy_id_ok, ETH, eth_synced, eth_phy_ready,
                              eth_gmii_alive ? eth_pcs_status : 16'd0};
            10'h0E:  rdata = eth_n_exec;
            10'h0F:  rdata = eth_n_dup;
            10'h10:  rdata = eth_n_drop;
            10'h11:  rdata = {16'd0, ETH_PORT};
            10'h12:  rdata = ETH_MAC[31:0];
            10'h13:  rdata = {16'd0, ETH_MAC[47:32]};
            10'h14:  rdata = {eth_rx_bad, eth_rx_dropped};
            10'h15:  rdata = eth_rx_frames;
            default: begin
                rdata = '0;
                hit   = 1'b0;
            end
        endcase
    end

    assign a_ready = !d_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            d_valid      <= 1'b0;
            scratch      <= '0;
            denied_count <= '0;
            cycles       <= '0;
            run_cycles   <= '0;
            user_leds    <= '0;
            eth_ip       <= ETH_IP;
        end else begin
            cycles <= cycles + 1'b1;
            if (!atlas_status[0]) run_cycles <= run_cycles + 1'b1;
            if (denied) denied_count <= denied_count + 1'b1;
            if (d_valid && d_ready) d_valid <= 1'b0;
            if (a_valid && a_ready) begin
                d_valid <= 1'b1;
                d       <= '{opcode: wr ? D_ACK : D_ACK_DATA, denied: !hit, data: wr ? '0 : rdata};
                if (wr && word == 10'h01) scratch <= apply_mask(scratch, a.data, a.mask);
                if (wr && word == 10'h0A && a.mask[0]) user_leds <= a.data[2:0];
                if (wr && word == 10'h0B) run_cycles <= '0;
                if (wr && word == 10'h0C) eth_ip <= apply_mask(eth_ip, a.data, a.mask);
            end
        end
    end
endmodule
