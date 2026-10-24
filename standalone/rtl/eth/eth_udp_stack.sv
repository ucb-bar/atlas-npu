// UDP/IPv4 endpoint for TSI over UDP (docs/tsi-over-udp.md) on the vendored verilog-ethernet.
//
//   GMII -> eth_mac_1g_fifo -> eth_axis_rx -> udp_complete -> filter + prefix -> datagram FIFO
//        -> 8 to 32 bits, core clock -> rx_*
//   tx_* -> 32 to 8 bits, GMII clock -> prefix to UDP header -> udp_complete -> eth_axis_tx
//        -> eth_mac_1g_fifo -> GMII
//
// The stack runs in the GMII clock; axis_async_fifo_adapter crosses to the core clock both ways.
// A datagram on the core side is the prefix (peer IP, {payload bytes, peer port}), then the UDP
// payload, little-endian; a reply is the prefix (peer IP, {16'h0, peer port}), then the payload.
// rx_abort (only with rx_valid and rx_last) marks a datagram cut short by a GMII-side reset: that
// last word is not part of the datagram.
//
// Receive never waits on the core: the datagram FIFO takes whole datagrams and drops one that does
// not fit, so ARP and other traffic keep flowing while eth_tsi is busy or the core is in reset.
// udp_complete does not check the destination IP; the filter passes UDP to local_ip:PORT that is
// not an IP fragment. Non-UDP IP (ping) is discarded. Subnet mask 0 makes arp.v look up every
// destination itself (no gateway), so the FPGA ARPs for the peer whatever its address; any ARP
// from the peer also fills the cache. A reply with more than 1472 payload bytes or none is dropped
// (udp_checksum_gen buffers the payload, 2048 bytes).
module eth_udp_stack #(
    parameter logic [47:0] LOCAL_MAC = 48'h02_00_41_54_4C_53,
    parameter logic [15:0] PORT      = 16'h5453,
    parameter int RX_FIFO_DEPTH = 32768,  // datagram FIFO bytes (whole datagrams; drops when full)
    parameter int TX_FIFO_DEPTH = 4096,   // tx FIFO from the core and MAC tx FIFO bytes
    parameter int CLK_HZ = 50_000_000     // core clock (sizes the GMII reset stretch)
) (
    input  logic        gmii_clk,
    input  logic        gmii_rst,
    input  logic        gmii_clk_en,
    input  logic [7:0]  gmii_rxd,
    input  logic        gmii_rx_dv,
    input  logic        gmii_rx_er,
    output logic [7:0]  gmii_txd,
    output logic        gmii_tx_en,
    output logic        gmii_tx_er,

    input  logic        clk,
    input  logic        rst,
    input  logic [31:0] local_ip,
    output logic [31:0] rx_data,
    output logic        rx_valid,
    input  logic        rx_ready,
    output logic        rx_last,
    output logic        rx_abort,
    input  logic [31:0] tx_data,
    input  logic        tx_valid,
    output logic        tx_ready,
    input  logic        tx_last,
    output logic [15:0] rx_dropped,  // frames dropped before the core (see Counters)
    output logic [15:0] rx_bad       // frames with a bad FCS or a GMII error
);
    localparam int MAC_RX_DEPTH = 8192;    // the stack drains it at line rate; covers ARP stalls
    localparam int MAX_PAYLOAD  = 1472;    // one 1500-byte IP packet
    localparam int ARP_RETRY    = 31_250_000;   // 250 ms at 125 MHz
    localparam int ARP_TIMEOUT  = 125_000_000;  // 1 s after the last retry

    // ── Reset and local IP in the GMII clock ────────────────────────────────
    // Each crossing FIFO passes a reset of one side to the other through a two-flop synchronizer.
    // A GMII-side reset that ends before the core side has seen it leaves the two sides out of
    // step (stale words read, pointers apart), so grst lasts at least 8 core cycles at CLK_HZ.
    localparam int GRST_N = 1 + (8 * 125_000_000 + CLK_HZ - 1) / CLK_HZ;

    logic        grst;
    logic [31:0] ip;

    sync_reset #(.N(GRST_N)) u_grst (.clk(gmii_clk), .rst(gmii_rst), .out(grst));

    eth_cdc_word #(.W(32)) u_ip (
        .sclk(clk), .sdata(local_ip), .dclk(gmii_clk), .ddata(ip));

    // ── MAC ─────────────────────────────────────────────────────────────────
    logic [7:0] mrx_data, mtx_data;
    logic       mrx_valid, mrx_ready, mrx_last, mrx_user;
    logic       mtx_valid, mtx_ready, mtx_last, mtx_user;
    logic       mac_bad, mac_overflow, mac_drop_bad;

    eth_mac_1g_fifo #(
        .ENABLE_PADDING(1),
        .MIN_FRAME_LENGTH(64),
        .TX_FIFO_DEPTH(TX_FIFO_DEPTH),
        .TX_FRAME_FIFO(1),
        .RX_FIFO_DEPTH(MAC_RX_DEPTH),
        .RX_FRAME_FIFO(1),
        .RX_DROP_BAD_FRAME(1),
        .RX_DROP_WHEN_FULL(1)
    ) u_mac (
        .rx_clk(gmii_clk), .rx_rst(grst), .tx_clk(gmii_clk), .tx_rst(grst),
        .logic_clk(gmii_clk), .logic_rst(grst),
        .tx_axis_tdata(mtx_data), .tx_axis_tkeep(1'b1), .tx_axis_tvalid(mtx_valid),
        .tx_axis_tready(mtx_ready), .tx_axis_tlast(mtx_last), .tx_axis_tuser(mtx_user),
        .rx_axis_tdata(mrx_data), .rx_axis_tkeep(), .rx_axis_tvalid(mrx_valid),
        .rx_axis_tready(mrx_ready), .rx_axis_tlast(mrx_last), .rx_axis_tuser(mrx_user),
        .gmii_rxd, .gmii_rx_dv, .gmii_rx_er, .gmii_txd, .gmii_tx_en, .gmii_tx_er,
        .rx_clk_enable(gmii_clk_en), .tx_clk_enable(gmii_clk_en),
        .rx_mii_select(1'b0), .tx_mii_select(1'b0),
        .tx_error_underflow(), .tx_fifo_overflow(), .tx_fifo_bad_frame(), .tx_fifo_good_frame(),
        .rx_error_bad_frame(mac_bad), .rx_error_bad_fcs(),
        .rx_fifo_overflow(mac_overflow), .rx_fifo_bad_frame(mac_drop_bad), .rx_fifo_good_frame(),
        .cfg_ifg(8'd12), .cfg_tx_enable(1'b1), .cfg_rx_enable(1'b1));

    // ── Ethernet framing ────────────────────────────────────────────────────
    logic        erx_hdr_valid, erx_hdr_ready, etx_hdr_valid, etx_hdr_ready;
    logic [47:0] erx_dest_mac, erx_src_mac, etx_dest_mac, etx_src_mac;
    logic [15:0] erx_type, etx_type;
    logic [7:0]  erx_data, etx_data;
    logic        erx_valid, erx_ready, erx_last, erx_user;
    logic        etx_valid, etx_ready, etx_last, etx_user;

    eth_axis_rx u_eth_rx (
        .clk(gmii_clk), .rst(grst),
        .s_axis_tdata(mrx_data), .s_axis_tkeep(1'b1), .s_axis_tvalid(mrx_valid),
        .s_axis_tready(mrx_ready), .s_axis_tlast(mrx_last), .s_axis_tuser(mrx_user),
        .m_eth_hdr_valid(erx_hdr_valid), .m_eth_hdr_ready(erx_hdr_ready),
        .m_eth_dest_mac(erx_dest_mac), .m_eth_src_mac(erx_src_mac), .m_eth_type(erx_type),
        .m_eth_payload_axis_tdata(erx_data), .m_eth_payload_axis_tkeep(),
        .m_eth_payload_axis_tvalid(erx_valid), .m_eth_payload_axis_tready(erx_ready),
        .m_eth_payload_axis_tlast(erx_last), .m_eth_payload_axis_tuser(erx_user),
        .busy(), .error_header_early_termination());

    eth_axis_tx u_eth_tx (
        .clk(gmii_clk), .rst(grst),
        .s_eth_hdr_valid(etx_hdr_valid), .s_eth_hdr_ready(etx_hdr_ready),
        .s_eth_dest_mac(etx_dest_mac), .s_eth_src_mac(etx_src_mac), .s_eth_type(etx_type),
        .s_eth_payload_axis_tdata(etx_data), .s_eth_payload_axis_tkeep(1'b1),
        .s_eth_payload_axis_tvalid(etx_valid), .s_eth_payload_axis_tready(etx_ready),
        .s_eth_payload_axis_tlast(etx_last), .s_eth_payload_axis_tuser(etx_user),
        .m_axis_tdata(mtx_data), .m_axis_tkeep(), .m_axis_tvalid(mtx_valid),
        .m_axis_tready(mtx_ready), .m_axis_tlast(mtx_last), .m_axis_tuser(mtx_user),
        .busy());

    // ── IPv4, ARP, UDP ──────────────────────────────────────────────────────
    logic        urx_hdr_valid, urx_hdr_ready, utx_hdr_valid, utx_hdr_ready;
    logic [31:0] urx_src_ip, urx_dest_ip;
    logic [2:0]  urx_flags;
    logic [12:0] urx_frag;
    logic [15:0] urx_src_port, urx_dest_port, urx_length;
    logic [7:0]  urx_data, utx_data;
    logic        urx_valid, urx_ready, urx_last, urx_user;
    logic        utx_valid, utx_ready, utx_last, utx_user;
    logic [31:0] utx_dest_ip;
    logic [15:0] utx_dest_port;
    logic        ip_bad_hdr, ip_bad_sum, ip_short_hdr, udp_short_hdr;

    udp_complete #(
        .ARP_REQUEST_RETRY_INTERVAL(ARP_RETRY),
        .ARP_REQUEST_TIMEOUT(ARP_TIMEOUT),
        .UDP_CHECKSUM_GEN_ENABLE(1)
    ) u_udp (
        .clk(gmii_clk), .rst(grst),
        .s_eth_hdr_valid(erx_hdr_valid), .s_eth_hdr_ready(erx_hdr_ready),
        .s_eth_dest_mac(erx_dest_mac), .s_eth_src_mac(erx_src_mac), .s_eth_type(erx_type),
        .s_eth_payload_axis_tdata(erx_data), .s_eth_payload_axis_tvalid(erx_valid),
        .s_eth_payload_axis_tready(erx_ready), .s_eth_payload_axis_tlast(erx_last),
        .s_eth_payload_axis_tuser(erx_user),
        .m_eth_hdr_valid(etx_hdr_valid), .m_eth_hdr_ready(etx_hdr_ready),
        .m_eth_dest_mac(etx_dest_mac), .m_eth_src_mac(etx_src_mac), .m_eth_type(etx_type),
        .m_eth_payload_axis_tdata(etx_data), .m_eth_payload_axis_tvalid(etx_valid),
        .m_eth_payload_axis_tready(etx_ready), .m_eth_payload_axis_tlast(etx_last),
        .m_eth_payload_axis_tuser(etx_user),
        // raw IP: nothing to send, discard what arrives (ICMP and other protocols)
        .s_ip_hdr_valid(1'b0), .s_ip_hdr_ready(), .s_ip_dscp(6'd0), .s_ip_ecn(2'd0),
        .s_ip_length(16'd0), .s_ip_ttl(8'd0), .s_ip_protocol(8'd0), .s_ip_source_ip(32'd0),
        .s_ip_dest_ip(32'd0), .s_ip_payload_axis_tdata(8'd0), .s_ip_payload_axis_tvalid(1'b0),
        .s_ip_payload_axis_tready(), .s_ip_payload_axis_tlast(1'b0), .s_ip_payload_axis_tuser(1'b0),
        .m_ip_hdr_valid(), .m_ip_hdr_ready(1'b1), .m_ip_eth_dest_mac(), .m_ip_eth_src_mac(),
        .m_ip_eth_type(), .m_ip_version(), .m_ip_ihl(), .m_ip_dscp(), .m_ip_ecn(), .m_ip_length(),
        .m_ip_identification(), .m_ip_flags(), .m_ip_fragment_offset(), .m_ip_ttl(),
        .m_ip_protocol(), .m_ip_header_checksum(), .m_ip_source_ip(), .m_ip_dest_ip(),
        .m_ip_payload_axis_tdata(), .m_ip_payload_axis_tvalid(), .m_ip_payload_axis_tready(1'b1),
        .m_ip_payload_axis_tlast(), .m_ip_payload_axis_tuser(),
        // UDP
        .s_udp_hdr_valid(utx_hdr_valid), .s_udp_hdr_ready(utx_hdr_ready),
        .s_udp_ip_dscp(6'd0), .s_udp_ip_ecn(2'd0), .s_udp_ip_ttl(8'd64),
        .s_udp_ip_source_ip(ip), .s_udp_ip_dest_ip(utx_dest_ip),
        .s_udp_source_port(PORT), .s_udp_dest_port(utx_dest_port),
        .s_udp_length(16'd0), .s_udp_checksum(16'd0),  // computed by udp_checksum_gen
        .s_udp_payload_axis_tdata(utx_data), .s_udp_payload_axis_tvalid(utx_valid),
        .s_udp_payload_axis_tready(utx_ready), .s_udp_payload_axis_tlast(utx_last),
        .s_udp_payload_axis_tuser(utx_user),
        .m_udp_hdr_valid(urx_hdr_valid), .m_udp_hdr_ready(urx_hdr_ready),
        .m_udp_eth_dest_mac(), .m_udp_eth_src_mac(), .m_udp_eth_type(), .m_udp_ip_version(),
        .m_udp_ip_ihl(), .m_udp_ip_dscp(), .m_udp_ip_ecn(), .m_udp_ip_length(),
        .m_udp_ip_identification(), .m_udp_ip_flags(urx_flags),
        .m_udp_ip_fragment_offset(urx_frag), .m_udp_ip_ttl(), .m_udp_ip_protocol(),
        .m_udp_ip_header_checksum(), .m_udp_ip_source_ip(urx_src_ip),
        .m_udp_ip_dest_ip(urx_dest_ip), .m_udp_source_port(urx_src_port),
        .m_udp_dest_port(urx_dest_port), .m_udp_length(urx_length), .m_udp_checksum(),
        .m_udp_payload_axis_tdata(urx_data), .m_udp_payload_axis_tvalid(urx_valid),
        .m_udp_payload_axis_tready(urx_ready), .m_udp_payload_axis_tlast(urx_last),
        .m_udp_payload_axis_tuser(urx_user),
        .ip_rx_busy(), .ip_tx_busy(), .udp_rx_busy(), .udp_tx_busy(),
        .ip_rx_error_header_early_termination(ip_short_hdr),
        .ip_rx_error_payload_early_termination(),
        .ip_rx_error_invalid_header(ip_bad_hdr),
        .ip_rx_error_invalid_checksum(ip_bad_sum),
        .ip_tx_error_payload_early_termination(), .ip_tx_error_arp_failed(),
        .udp_rx_error_header_early_termination(udp_short_hdr),
        .udp_rx_error_payload_early_termination(),
        .udp_tx_error_payload_early_termination(),
        .local_mac(LOCAL_MAC), .local_ip(ip), .gateway_ip(32'd0), .subnet_mask(32'd0),
        .clear_arp_cache(1'b0));

    // ── Receive: filter, prefix, datagram FIFO, clock crossing ──────────────
    // The datagram FIFO keeps whole frames and is always ready: a datagram that does not fit, or
    // is marked bad (payload shorter than the UDP length), is dropped there, so nothing upstream
    // ever waits on the core. A small streaming FIFO (8 -> 32 bits) crosses to the core clock. A
    // core reset drops the datagram it cuts. A GMII reset ends the datagram being delivered with
    // an extra word of stale data marked rx_abort (the FIFO's bad-frame tuser).
    typedef enum logic [1:0] {R_IDLE, R_PREFIX, R_DATA, R_SKIP} rstate_t;
    rstate_t     rs;
    logic [2:0]  rcnt;
    logic [63:0] rpre;
    logic        other_port;
    logic [7:0]  q_data, f_data;
    logic        q_valid, q_last, q_overflow, q_bad, f_valid, f_ready, f_last, x_overflow;

    wire for_us = urx_dest_ip == ip && urx_frag == '0 && !urx_flags[0];  // flags[0] = MF
    assign urx_hdr_ready = rs == R_IDLE;
    assign urx_ready     = rs == R_DATA || rs == R_SKIP;
    assign q_valid = rs == R_PREFIX || (rs == R_DATA && urx_valid);
    assign q_data  = rs == R_PREFIX ? rpre[7:0] : urx_data;
    assign q_last  = rs == R_DATA && urx_last;

    always_ff @(posedge gmii_clk) begin
        other_port <= 1'b0;
        if (grst) begin
            rs <= R_IDLE;
        end else begin
            case (rs)
                R_IDLE: if (urx_hdr_valid) begin
                    rpre <= {urx_length - 16'd8, urx_src_port, urx_src_ip};
                    rcnt <= '0;
                    rs   <= for_us && urx_dest_port == PORT ? R_PREFIX : R_SKIP;
                    other_port <= urx_dest_ip == ip && !(for_us && urx_dest_port == PORT);
                end
                R_PREFIX: begin
                    rpre <= rpre >> 8;
                    rcnt <= rcnt + 1'b1;
                    if (rcnt == 3'd7) rs <= R_DATA;
                end
                R_DATA: if (urx_valid && urx_last) rs <= R_IDLE;
                R_SKIP: if (urx_valid && urx_last) rs <= R_IDLE;
            endcase
        end
    end

    axis_fifo #(
        .DEPTH(RX_FIFO_DEPTH),
        .DATA_WIDTH(8),
        .ID_ENABLE(0),
        .DEST_ENABLE(0),
        .USER_ENABLE(1),
        .USER_WIDTH(1),
        .FRAME_FIFO(1),
        .DROP_OVERSIZE_FRAME(1),
        .DROP_BAD_FRAME(1),
        .DROP_WHEN_FULL(1)
    ) u_rxq (
        .clk(gmii_clk), .rst(grst),
        .s_axis_tdata(q_data), .s_axis_tkeep(1'b1), .s_axis_tvalid(q_valid), .s_axis_tready(),
        .s_axis_tlast(q_last), .s_axis_tid(8'd0), .s_axis_tdest(8'd0), .s_axis_tuser(urx_user),
        .m_axis_tdata(f_data), .m_axis_tkeep(), .m_axis_tvalid(f_valid), .m_axis_tready(f_ready),
        .m_axis_tlast(f_last), .m_axis_tid(), .m_axis_tdest(), .m_axis_tuser(),
        .pause_req(1'b0), .pause_ack(),
        .status_depth(), .status_depth_commit(), .status_overflow(q_overflow),
        .status_bad_frame(q_bad), .status_good_frame());

    axis_async_fifo_adapter #(
        .DEPTH(256),
        .S_DATA_WIDTH(8),
        .M_DATA_WIDTH(32),
        .ID_ENABLE(0),
        .DEST_ENABLE(0),
        .USER_ENABLE(1),
        .USER_WIDTH(1),
        .FRAME_FIFO(0)
    ) u_rxx (
        .s_clk(gmii_clk), .s_rst(grst),
        .s_axis_tdata(f_data), .s_axis_tkeep(1'b1), .s_axis_tvalid(f_valid),
        .s_axis_tready(f_ready), .s_axis_tlast(f_last), .s_axis_tid(8'd0), .s_axis_tdest(8'd0),
        .s_axis_tuser(1'b0),
        .m_clk(clk), .m_rst(rst),
        .m_axis_tdata(rx_data), .m_axis_tkeep(), .m_axis_tvalid(rx_valid),
        .m_axis_tready(rx_ready), .m_axis_tlast(rx_last), .m_axis_tid(), .m_axis_tdest(),
        .m_axis_tuser(rx_abort),
        .s_pause_req(1'b0), .s_pause_ack(), .m_pause_req(1'b0), .m_pause_ack(),
        .s_status_depth(), .s_status_depth_commit(), .s_status_overflow(x_overflow),
        .s_status_bad_frame(), .s_status_good_frame(),
        .m_status_depth(), .m_status_depth_commit(), .m_status_overflow(),
        .m_status_bad_frame(), .m_status_good_frame());

    // ── Transmit: tx FIFO, prefix to UDP header ─────────────────────────────
    typedef enum logic [1:0] {T_PREFIX, T_HDR, T_DATA, T_SKIP} tstate_t;
    tstate_t     ts;
    logic [2:0]  tcnt;
    logic [63:0] tpre;
    logic [10:0] tlen;
    logic [7:0]  t_data;
    logic        t_valid, t_ready, t_last, t_user;

    axis_async_fifo_adapter #(
        .DEPTH(TX_FIFO_DEPTH),
        .S_DATA_WIDTH(32),
        .S_KEEP_ENABLE(1),
        .M_DATA_WIDTH(8),
        .ID_ENABLE(0),
        .DEST_ENABLE(0),
        .USER_ENABLE(1),
        .USER_WIDTH(1),
        .FRAME_FIFO(0)
    ) u_txq (
        .s_clk(clk), .s_rst(rst),
        .s_axis_tdata(tx_data), .s_axis_tkeep(4'hF), .s_axis_tvalid(tx_valid),
        .s_axis_tready(tx_ready), .s_axis_tlast(tx_last), .s_axis_tid(8'd0), .s_axis_tdest(8'd0),
        .s_axis_tuser(1'b0),
        .m_clk(gmii_clk), .m_rst(grst),
        .m_axis_tdata(t_data), .m_axis_tkeep(), .m_axis_tvalid(t_valid),
        .m_axis_tready(t_ready), .m_axis_tlast(t_last), .m_axis_tid(), .m_axis_tdest(),
        .m_axis_tuser(t_user),
        .s_pause_req(1'b0), .s_pause_ack(), .m_pause_req(1'b0), .m_pause_ack(),
        .s_status_depth(), .s_status_depth_commit(), .s_status_overflow(),
        .s_status_bad_frame(), .s_status_good_frame(),
        .m_status_depth(), .m_status_depth_commit(), .m_status_overflow(),
        .m_status_bad_frame(), .m_status_good_frame());

    // An over-long reply is cut at MAX_PAYLOAD and marked bad (the MAC tx FIFO drops it).
    wire t_cut = tlen == 11'(MAX_PAYLOAD - 1);
    assign utx_hdr_valid = ts == T_HDR;
    assign utx_dest_ip   = tpre[31:0];
    assign utx_dest_port = tpre[47:32];
    assign utx_data  = t_data;
    assign utx_valid = ts == T_DATA && t_valid;
    assign utx_last  = t_last || t_cut;
    assign utx_user  = t_user || (t_cut && !t_last);
    assign t_ready   = ts == T_PREFIX || ts == T_SKIP || (ts == T_DATA && utx_ready);

    always_ff @(posedge gmii_clk) begin
        if (grst) begin
            ts <= T_PREFIX;
            tcnt <= '0;
        end else begin
            case (ts)
                T_PREFIX: if (t_valid) begin
                    tpre <= {t_data, tpre[63:8]};
                    tcnt <= tcnt + 1'b1;
                    if (t_last) tcnt <= '0;  // no payload: drop
                    else if (tcnt == 3'd7) ts <= T_HDR;
                end
                T_HDR: if (utx_hdr_ready) begin
                    tlen <= '0;
                    ts   <= T_DATA;
                end
                T_DATA: if (t_valid && utx_ready) begin
                    tlen <= tlen + 1'b1;
                    if (t_last) ts <= T_PREFIX;
                    else if (t_cut) ts <= T_SKIP;
                end
                T_SKIP: if (t_valid && t_last) ts <= T_PREFIX;
            endcase
        end
    end

    // ── Counters ────────────────────────────────────────────────────────────
    // rx_dropped: MAC FIFO full or bad frame, IP header errors, UDP without payload, UDP to
    // local_ip on another port or fragmented, datagram FIFO full or short payload, cut by a core
    // reset. Both wrap at 16 bits and reset with the GMII side.
    logic [15:0] n_drop, n_bad;

    always_ff @(posedge gmii_clk) begin
        if (grst) begin
            n_drop <= '0;
            n_bad  <= '0;
        end else begin
            n_bad  <= n_bad + 16'(mac_bad);
            n_drop <= n_drop + 16'(mac_overflow) + 16'(mac_drop_bad) + 16'(ip_bad_hdr)
                    + 16'(ip_bad_sum) + 16'(ip_short_hdr) + 16'(udp_short_hdr) + 16'(other_port)
                    + 16'(q_overflow) + 16'(q_bad) + 16'(x_overflow);
        end
    end

    eth_cdc_word #(.W(32)) u_cnt (
        .sclk(gmii_clk), .sdata({n_bad, n_drop}), .dclk(clk), .ddata({rx_bad, rx_dropped}));
endmodule

// Carries a word to another clock domain. The source holds a copy, toggles req and waits for the
// ack (two-flop synchronizers both ways), so the copy is stable whenever the destination takes it.
// It runs continuously: the output follows the input a few cycles of both clocks late. No reset:
// it starts from the initial values.
// hold -> dreg is not timed (build.tcl makes the clocks asynchronous groups). The destination
// takes hold two dclk cycles after the synchronized req edge (req_q), so that path may be up to
// about two dclk periods slower than req -> req_d[0] before old and new bits can mix.
module eth_cdc_word #(
    parameter int W = 32
) (
    input  logic         sclk,
    input  logic [W-1:0] sdata,
    input  logic         dclk,
    output logic [W-1:0] ddata
);
    logic [W-1:0] hold = '0, dreg = '0;
    logic req = 1'b0, ack = 1'b0;
    (* ASYNC_REG = "TRUE" *) logic [1:0] ack_s = '0;  // ack in the source clock
    (* ASYNC_REG = "TRUE" *) logic [1:0] req_d = '0;  // req in the destination clock
    logic req_q = 1'b0;                                // req_d[1] one dclk later

    assign ddata = dreg;

    always_ff @(posedge sclk) begin
        ack_s <= {ack_s[0], ack};
        if (req == ack_s[1]) begin
            hold <= sdata;
            req  <= !req;
        end
    end

    always_ff @(posedge dclk) begin
        req_d <= {req_d[0], req};
        req_q <= req_d[1];
        if (req_q != ack) begin
            dreg  <= hold;
            ack   <= req_q;
        end
    end
endmodule
