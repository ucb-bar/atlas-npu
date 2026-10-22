// DRAM in the VCU118's DDR4 (C1 bank, through the Xilinx DDR4 MIG's AXI4 port), with the
// ports of atlas_dram:
//
//   port A  the host fabric (wa_t/wd_t), one transaction at a time
//   port B  AtlasCore's DMA client: single-beat Get and PutFullData, up to QD requests in flight,
//           answered in order with their source IDs
//
// Both ports take the line from address bits [AXI_AW-1:5], so the 2 GiB DRAM window
// (0x8000_0000 up) and DMA's 64-bit {dma.base, x[rs]} addresses alias onto the memory.
//
// Requests cross to the MIG's user clock in arrival order. There, a read issues only when no
// write is outstanding and a write only when no read is, so the memory stays sequential even
// though AXI orders reads and writes independently; responses then return in request order.
// Each 32-byte line is one 256-bit AXI beat. Nothing issues before calibration completes.
module atlas_ddr4_dram
    import tl_pkg::*;
#(
    parameter int SW     = 6,   // DMA source ID width
    parameter int AXI_AW = 31,  // 2 GiB
    parameter int QD     = 32   // DMA requests in flight
) (
    input  logic                clk,
    input  logic                rst,

    input  logic                a_valid,
    output logic                a_ready,
    input  wa_t                 a,
    output logic                d_valid,
    input  logic                d_ready,
    output wd_t                 d,

    input  logic                dma_a_valid,
    output logic                dma_a_ready,
    input  logic [2:0]          dma_a_opcode,
    input  logic [SW-1:0]       dma_a_source,
    input  logic [63:0]         dma_a_address,
    input  logic [31:0]         dma_a_mask,
    input  logic [255:0]        dma_a_data,
    output logic                dma_d_valid,
    input  logic                dma_d_ready,
    output logic [2:0]          dma_d_opcode,
    output logic [SW-1:0]       dma_d_source,
    output logic [255:0]        dma_d_data,

    output logic                calibrated,   // calib_done in the core clock domain

    // AXI4 master, MIG user clock domain
    input  logic                ui_clk,
    input  logic                ui_rst,
    input  logic                calib_done,
    output logic                aresetn,
    output logic [3:0]          awid,
    output logic [AXI_AW-1:0]   awaddr,
    output logic [7:0]          awlen,
    output logic [2:0]          awsize,
    output logic [1:0]          awburst,
    output logic                awlock,
    output logic [3:0]          awcache,
    output logic [2:0]          awprot,
    output logic [3:0]          awqos,
    output logic                awvalid,
    input  logic                awready,
    output logic [255:0]        wdata,
    output logic [31:0]         wstrb,
    output logic                wlast,
    output logic                wvalid,
    input  logic                wready,
    input  logic [3:0]          bid,
    input  logic [1:0]          bresp,
    input  logic                bvalid,
    output logic                bready,
    output logic [3:0]          arid,
    output logic [AXI_AW-1:0]   araddr,
    output logic [7:0]          arlen,
    output logic [2:0]          arsize,
    output logic [1:0]          arburst,
    output logic                arlock,
    output logic [3:0]          arcache,
    output logic [2:0]          arprot,
    output logic [3:0]          arqos,
    output logic                arvalid,
    input  logic                arready,
    input  logic [3:0]          rid,
    input  logic [255:0]        rdata,
    input  logic [1:0]          rresp,
    input  logic                rlast,
    input  logic                rvalid,
    output logic                rready
);
    localparam int LW = AXI_AW - 5;
    localparam int QW = $clog2(QD + 1);

    typedef struct packed {
        logic          host;
        logic          write;
        logic [LW-1:0] line;
        logic [31:0]   mask;
        logic [255:0]  data;
        logic [SW-1:0] src;
    } req_t;

    typedef struct packed {
        logic          host;
        logic          write;
        logic [SW-1:0] src;
    } tag_t;

    typedef struct packed {
        tag_t          tag;
        logic [255:0]  data;
    } rsp_t;

    // ── Resets: either domain's reset clears both sides (and the MIG's AXI port) ────────
    (* ASYNC_REG = "TRUE" *) logic [2:0] ui_rst_c;   // ui_rst in the core domain
    (* ASYNC_REG = "TRUE" *) logic [2:0] rst_u;      // rst in the user clock domain
    (* ASYNC_REG = "TRUE" *) logic [1:0] cal_c;
    always_ff @(posedge clk)    ui_rst_c <= {ui_rst_c[1:0], ui_rst};
    always_ff @(posedge ui_clk) rst_u    <= {rst_u[1:0], rst};
    always_ff @(posedge clk)    cal_c    <= {cal_c[0], calib_done};
    logic crst, urst;
    always_ff @(posedge clk)    crst <= rst || ui_rst_c[2];
    always_ff @(posedge ui_clk) urst <= ui_rst || rst_u[2];
    always_ff @(posedge ui_clk) aresetn <= !urst;
    assign calibrated = cal_c[1];

    // ── Core side: requests in, responses out ───────────────────────────────────────────
    req_t          req;
    logic          req_valid, req_ready;
    rsp_t          rsp;
    logic          rsp_valid, rsp_ready;
    logic          h_busy;
    logic [QW-1:0] inflight;

    wire host_take = a_valid && !h_busy;
    assign req_valid   = host_take || (dma_a_valid && inflight < QW'(QD));
    assign a_ready     = !h_busy && req_ready;
    assign dma_a_ready = !host_take && req_ready && inflight < QW'(QD);
    always_comb begin
        if (host_take)
            req = '{host: 1'b1, write: a.opcode != A_GET, line: a.address[AXI_AW-1:5],
                    mask: a.mask, data: a.data, src: '0};
        else
            req = '{host: 1'b0, write: dma_a_opcode != 3'd4, line: dma_a_address[AXI_AW-1:5],
                    mask: dma_a_mask, data: dma_a_data, src: dma_a_source};
    end

    wire dma_take = dma_a_valid && dma_a_ready;
    wire dma_done = dma_d_valid && dma_d_ready;
    always_ff @(posedge clk) begin
        if (crst) begin
            h_busy   <= 1'b0;
            inflight <= '0;
        end else begin
            if (a_valid && a_ready) h_busy <= 1'b1;
            else if (d_valid && d_ready) h_busy <= 1'b0;
            inflight <= inflight + QW'(dma_take) - QW'(dma_done);
        end
    end

    assign d_valid      = rsp_valid && rsp.tag.host;
    assign d            = '{opcode: rsp.tag.write ? D_ACK : D_ACK_DATA, denied: 1'b0,
                            data: rsp.tag.write ? '0 : rsp.data};
    assign dma_d_valid  = rsp_valid && !rsp.tag.host;
    assign dma_d_opcode = rsp.tag.write ? 3'd0 : 3'd1;
    assign dma_d_source = rsp.tag.src;
    assign dma_d_data   = rsp.tag.write ? '0 : rsp.data;
    assign rsp_ready    = rsp.tag.host ? d_ready : dma_d_ready;

    // ── Clock crossings ─────────────────────────────────────────────────────────────────
    req_t q;
    logic q_valid, q_ready;
    rsp_t r_in;
    logic r_in_valid, r_in_ready;

    async_fifo #(.W($bits(req_t)), .AW(4)) u_req (
        .wclk(clk), .wrst(crst), .wdata(req), .wvalid(req_valid), .wready(req_ready),
        .rclk(ui_clk), .rrst(urst), .rdata(q), .rvalid(q_valid), .rready(q_ready));

    async_fifo #(.W($bits(rsp_t)), .AW(6)) u_rsp (
        .wclk(ui_clk), .wrst(urst), .wdata(r_in), .wvalid(r_in_valid), .wready(r_in_ready),
        .rclk(clk), .rrst(crst), .rdata(rsp), .rvalid(rsp_valid), .rready(rsp_ready));

    // ── User clock side: AXI4 ───────────────────────────────────────────────────────────
    localparam int OW = $clog2(QD + 2) + 1;
    logic [OW-1:0] rd_out, wr_out;      // issued, response not yet received
    logic          aw_pend, w_pend;
    tag_t          t_head;
    logic          t_valid, t_ready;
    logic          t_push, t_room;
    tag_t          t_in;
    assign t_in = '{host: q.host, write: q.write, src: q.src};

    wire can_read  = q_valid && !q.write && wr_out == 0 && (!arvalid || arready);
    wire can_write = q_valid && q.write && rd_out == 0 && (!aw_pend || awready) && (!w_pend || wready);
    assign q_ready = calib_done && t_room && (can_read || can_write);
    wire issue_r   = q_valid && q_ready && !q.write;
    wire issue_w   = q_valid && q_ready && q.write;
    assign t_push  = issue_r || issue_w;

    sync_fifo #(.W($bits(tag_t)), .DEPTH(64)) u_tags (
        .clk(ui_clk), .rst(urst),
        .wdata(t_in), .wvalid(t_push), .wready(t_room),
        .rdata(t_head), .rvalid(t_valid), .rready(t_ready));

    wire r_fire = rvalid && rready;
    wire b_fire = bvalid && bready;
    assign rready     = r_in_ready && t_valid && !t_head.write;
    assign bready     = r_in_ready && t_valid && t_head.write;
    assign t_ready    = r_fire || b_fire;
    assign r_in_valid = t_ready;
    assign r_in       = '{tag: t_head, data: r_fire ? rdata : '0};

    always_ff @(posedge ui_clk) begin
        if (urst) begin
            arvalid <= 1'b0;
            aw_pend <= 1'b0;
            w_pend  <= 1'b0;
            rd_out  <= '0;
            wr_out  <= '0;
        end else begin
            if (issue_r) begin
                arvalid <= 1'b1;
                araddr  <= {q.line, 5'b0};
            end else if (arready) begin
                arvalid <= 1'b0;
            end
            if (issue_w) begin
                aw_pend <= 1'b1;
                w_pend  <= 1'b1;
                awaddr  <= {q.line, 5'b0};
                wdata   <= q.data;
                wstrb   <= q.mask;
            end else begin
                if (awready) aw_pend <= 1'b0;
                if (wready)  w_pend  <= 1'b0;
            end
            rd_out <= rd_out + OW'(issue_r) - OW'(r_fire);
            wr_out <= wr_out + OW'(issue_w) - OW'(b_fire);
        end
    end

    assign awvalid = aw_pend;
    assign wvalid  = w_pend;
    assign awid    = '0;
    assign awlen   = '0;
    assign awsize  = 3'd5;     // 32 bytes: one full 256-bit beat
    assign awburst = 2'b01;
    assign awlock  = 1'b0;
    assign awcache = 4'b0011;
    assign awprot  = '0;
    assign awqos   = '0;
    assign wlast   = 1'b1;
    assign arid    = '0;
    assign arlen   = '0;
    assign arsize  = 3'd5;
    assign arburst = 2'b01;
    assign arlock  = 1'b0;
    assign arcache = 4'b0011;
    assign arprot  = '0;
    assign arqos   = '0;
endmodule
