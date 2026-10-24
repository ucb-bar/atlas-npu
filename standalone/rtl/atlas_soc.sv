// Standalone Atlas SoC: AtlasCore, its DRAM ports, and a TL-UL host fabric. A host (the UART
// TSI target on the board, the simulation driver in Verilator) replaces the CPU: it loads the
// kernel into IMEM and the data into DRAM, starts the core through EXEC_CONTROL, polls DBG0, and
// reads the results back. A second host port on the beat fabric (e_*, the Ethernet TSI target)
// shares the fabric with the first through tl_arbiter; tie it off (e_a_valid 0) when unused.
//
//   0x0000_0000 - 0x0000_0FFF  sys_regs    link identification, status, Atlas busy bits
//   0x0002_0000 - 0x0003_FFFF  Atlas IMEM  128 KiB (reads while the core runs are denied)
//   0x0004_0000 - 0x0004_0FFF  Atlas CSR   CYCLE, INSTCNT, STATUS, ILLEGAL_PC, DBG0, DBG1, EXEC_CONTROL
//   0x2000_0000 - 0x2017_FFFF  Atlas VMEM  1.5 MiB (0x2018_0000 - 0x201F_FFFF is denied)
//   0x8000_0000 - 0xFFFF_FFFF  DRAM        EXT_DRAM (the VCU118): the ext_* ports, 2 GiB of DDR4;
//                                          otherwise (the Verilator model) 2^DRAM_LINE_W x 32 B
//                                          of atlas_dram, aliased over the window
//
// The DMA engine reaches the same DRAM through the second memory port. With EXT_DRAM both DRAM
// ports leave the SoC (atlas_ddr4_dram in vcu118_atlas_top). Unmapped accesses get a denied
// response, and so do accesses a device cannot serve now (an IMEM read while the core runs, a
// DRAM access before the DDR4 is calibrated), so that no request holds the fabric.
module atlas_soc
    import tl_pkg::*;
#(
    parameter int CLK_HZ      = 50_000_000,
    parameter int BAUD        = 921_600,
    parameter int DRAM_LINE_W = 17,
    parameter int DRAM_LAT    = 8,
    parameter bit EXT_DRAM    = 1'b0,
    parameter bit          ETH      = 1'b0,  // Ethernet TSI present (sys_regs ETH_STATUS[18])
    parameter logic [31:0] ETH_IP   = 32'hAC14_0112,
    parameter logic [47:0] ETH_MAC  = 48'h02_00_41_54_4C_53,
    parameter logic [15:0] ETH_PORT = 16'h5453
) (
    input  logic       clk,
    input  logic       rst,

    // Host: one 32-bit TL-UL word at a time.
    input  logic       a_valid,
    output logic       a_ready,
    input  a_t         a,
    output logic       d_valid,
    input  logic       d_ready,
    output d_t         d,

    // Second host (Ethernet TSI): one 32-byte beat at a time.
    input  logic       e_a_valid,
    output logic       e_a_ready,
    input  wa_t        e_a,
    output logic       e_d_valid,
    input  logic       e_d_ready,
    output wd_t        e_d,

    input  logic       dropped,   // TSI link status, reported in sys_regs
    input  logic       bad_cmd,
    output logic       denied,    // pulses for every denied response
    output logic       halted,    // Atlas scalar core halted
    output logic [2:0] user_leds,

    // Ethernet status (core clock), reported in sys_regs
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
    output logic [31:0] local_ip,   // sys_regs ETH_IP

    // External DRAM (EXT_DRAM): the host port and the DMA port of atlas_dram
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
    localparam int N  = 5;
    localparam int SW = 6;  // DMA source ID width (DmaParams.tagBits)
    localparam logic [N-1:0][31:0] BASE = {32'h8000_0000, 32'h2000_0000, 32'h0004_0000,
                                           32'h0002_0000, 32'h0000_0000};
    localparam logic [N-1:0][31:0] MASK = {32'h7FFF_FFFF, 32'h001F_FFFF, 32'h0000_0FFF,
                                           32'h0001_FFFF, 32'h0000_0FFF};
    localparam int SYS = 0, IMEM = 1, CSR = 2, VMEM = 3, DRAM = 4;

    // ── Host fabric ─────────────────────────────────────────────────────────
    // Host 0: the 32-bit port through tl_widen. Host 1: the beat port.
    logic [1:0] h_a_valid, h_a_ready, h_d_valid, h_d_ready;
    wa_t        h_a [2];
    wd_t        h_d;

    tl_widen u_widen (
        .clk, .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
        .w_a_valid(h_a_valid[0]), .w_a_ready(h_a_ready[0]), .w_a(h_a[0]),
        .w_d_valid(h_d_valid[0]), .w_d_ready(h_d_ready[0]), .w_d(h_d));

    assign h_a_valid[1] = e_a_valid;
    assign e_a_ready    = h_a_ready[1];
    assign h_a[1]       = e_a;
    assign e_d_valid    = h_d_valid[1];
    assign h_d_ready[1] = e_d_ready;
    assign e_d          = h_d;

    logic   wa_valid, wa_ready, wd_valid, wd_ready;
    wa_t    wa;
    wd_t    wd;

    tl_arbiter u_arb (
        .clk, .rst,
        .host_a_valid(h_a_valid), .host_a_ready(h_a_ready), .host_a(h_a),
        .host_d_valid(h_d_valid), .host_d_ready(h_d_ready), .host_d(h_d),
        .a_valid(wa_valid), .a_ready(wa_ready), .a(wa),
        .d_valid(wd_valid), .d_ready(wd_ready), .d(wd));

    logic [N-1:0] x_a_valid, x_a_ready, x_d_valid, x_d_ready;
    wa_t          dev_a;
    wd_t          x_d [N];

    tl_xbar #(.N(N), .BASE(BASE), .MASK(MASK)) u_xbar (
        .clk, .rst,
        .a_valid(wa_valid), .a_ready(wa_ready), .a(wa),
        .d_valid(wd_valid), .d_ready(wd_ready), .d(wd),
        .dev_a_valid(x_a_valid), .dev_a_ready(x_a_ready), .dev_a,
        .dev_d_valid(x_d_valid), .dev_d_ready(x_d_ready), .dev_d(x_d),
        .denied);

    // A request a device cannot serve now is answered here, denied (reads return 0), instead of
    // waiting at the device with the fabric held for both hosts: an IMEM Get while the core runs
    // (InstrMem accepts a Get only while the core is halted; the core starts only through a CSR
    // write, which cannot pass while a request holds the fabric) and, with EXT_DRAM, a DRAM
    // access before the DDR4 is calibrated.
    logic [N-1:0] dev_a_valid, dev_a_ready, dev_d_valid, dev_d_ready, closed, refused;
    wd_t          dev_d [N];
    d_op_e        refused_op;

    always_comb begin
        closed       = '0;
        closed[IMEM] = !halted && dev_a.opcode == A_GET;
        closed[DRAM] = EXT_DRAM && !ext_dram_ready;
    end

    for (genvar k = 0; k < N; k++) begin : g_gate
        assign dev_a_valid[k] = x_a_valid[k] && !closed[k];
        assign x_a_ready[k]   = closed[k] || dev_a_ready[k];
        assign x_d_valid[k]   = refused[k] || dev_d_valid[k];
        assign dev_d_ready[k] = x_d_ready[k] && !refused[k];
        assign x_d[k]         = refused[k] ? '{opcode: refused_op, denied: 1'b1, data: '0} : dev_d[k];
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            refused <= '0;
        end else begin
            refused <= (refused & ~x_d_ready) | (x_a_valid & closed);
            if (|(x_a_valid & closed)) refused_op <= dev_a.opcode == A_GET ? D_ACK_DATA : D_ACK;
        end
    end

    // Word devices sit behind tl_narrow.
    logic       n_a_valid [3], n_a_ready [3], n_d_valid [3], n_d_ready [3];
    a_t         n_a [3];
    d_t         n_d [3];

    for (genvar k = 0; k < 3; k++) begin : g_narrow
        tl_narrow u_narrow (
            .clk, .rst,
            .a_valid(dev_a_valid[k]), .a_ready(dev_a_ready[k]), .a(dev_a),
            .d_valid(dev_d_valid[k]), .d_ready(dev_d_ready[k]), .d(dev_d[k]),
            .n_a_valid(n_a_valid[k]), .n_a_ready(n_a_ready[k]), .n_a(n_a[k]),
            .n_d_valid(n_d_valid[k]), .n_d_ready(n_d_ready[k]), .n_d(n_d[k]));
    end

    // ── AtlasCore ───────────────────────────────────────────────────────────
    logic         imem_a_valid, imem_a_ready, imem_d_valid, imem_d_ready, imem_d_denied;
    logic [2:0]   imem_a_opcode, imem_d_opcode;
    logic [3:0]   imem_a_size, imem_a_mask;
    logic [31:0]  imem_a_address, imem_a_data, imem_d_data;

    logic         csr_a_valid, csr_a_ready, csr_d_valid, csr_d_ready, csr_d_denied;
    logic [2:0]   csr_a_opcode, csr_d_opcode;
    logic [3:0]   csr_a_size, csr_a_mask;
    logic [31:0]  csr_a_address, csr_a_data, csr_d_data;

    logic         vmem_a_valid, vmem_a_ready, vmem_d_valid, vmem_d_ready, vmem_d_denied;
    logic [2:0]   vmem_a_opcode, vmem_d_opcode;
    logic [3:0]   vmem_a_size;
    logic [31:0]  vmem_a_address, vmem_a_mask;
    logic [255:0] vmem_a_data, vmem_d_data;

    logic          dma_a_valid, dma_a_ready, dma_d_valid, dma_d_ready;
    logic [2:0]    dma_a_opcode, dma_d_opcode;
    logic [SW-1:0] dma_a_source, dma_d_source;
    logic [63:0]   dma_a_address;
    logic [31:0]   dma_a_mask;
    logic [255:0]  dma_a_data, dma_d_data;

    logic          dbg_mxu0_data, dbg_mxu0_comp, dbg_mxu1_data, dbg_mxu1_comp, dbg_xlu, dbg_lsu;
    logic [3:0]    features;
    logic [7:0]    dbg_dma;

    atlas_tl_word u_imem_port (
        .a_valid(n_a_valid[IMEM]), .a_ready(n_a_ready[IMEM]), .a(n_a[IMEM]),
        .d_valid(n_d_valid[IMEM]), .d_ready(n_d_ready[IMEM]), .d(n_d[IMEM]),
        .tl_a_valid(imem_a_valid), .tl_a_ready(imem_a_ready), .tl_a_opcode(imem_a_opcode),
        .tl_a_size(imem_a_size), .tl_a_address(imem_a_address), .tl_a_mask(imem_a_mask),
        .tl_a_data(imem_a_data), .tl_d_valid(imem_d_valid), .tl_d_ready(imem_d_ready),
        .tl_d_opcode(imem_d_opcode), .tl_d_denied(imem_d_denied), .tl_d_data(imem_d_data));

    atlas_tl_word u_csr_port (
        .a_valid(n_a_valid[CSR]), .a_ready(n_a_ready[CSR]), .a(n_a[CSR]),
        .d_valid(n_d_valid[CSR]), .d_ready(n_d_ready[CSR]), .d(n_d[CSR]),
        .tl_a_valid(csr_a_valid), .tl_a_ready(csr_a_ready), .tl_a_opcode(csr_a_opcode),
        .tl_a_size(csr_a_size), .tl_a_address(csr_a_address), .tl_a_mask(csr_a_mask),
        .tl_a_data(csr_a_data), .tl_d_valid(csr_d_valid), .tl_d_ready(csr_d_ready),
        .tl_d_opcode(csr_d_opcode), .tl_d_denied(csr_d_denied), .tl_d_data(csr_d_data));

    atlas_tl_beat u_vmem_port (
        .clk, .rst,
        .a_valid(dev_a_valid[VMEM]), .a_ready(dev_a_ready[VMEM]), .a(dev_a),
        .d_valid(dev_d_valid[VMEM]), .d_ready(dev_d_ready[VMEM]), .d(dev_d[VMEM]),
        .tl_a_valid(vmem_a_valid), .tl_a_ready(vmem_a_ready), .tl_a_opcode(vmem_a_opcode),
        .tl_a_size(vmem_a_size), .tl_a_address(vmem_a_address), .tl_a_mask(vmem_a_mask),
        .tl_a_data(vmem_a_data), .tl_d_valid(vmem_d_valid), .tl_d_ready(vmem_d_ready),
        .tl_d_opcode(vmem_d_opcode), .tl_d_denied(vmem_d_denied), .tl_d_data(vmem_d_data));

    AtlasCore u_atlas (
        .clock (clk),
        .reset (rst),

        .io_imemTL_a_ready        (imem_a_ready),
        .io_imemTL_a_valid        (imem_a_valid),
        .io_imemTL_a_bits_opcode  (imem_a_opcode),
        .io_imemTL_a_bits_param   (3'd0),
        .io_imemTL_a_bits_size    (imem_a_size),
        .io_imemTL_a_bits_source  (1'b0),
        .io_imemTL_a_bits_address (imem_a_address),
        .io_imemTL_a_bits_mask    (imem_a_mask),
        .io_imemTL_a_bits_data    (imem_a_data),
        .io_imemTL_a_bits_corrupt (1'b0),
        .io_imemTL_d_ready        (imem_d_ready),
        .io_imemTL_d_valid        (imem_d_valid),
        .io_imemTL_d_bits_opcode  (imem_d_opcode),
        .io_imemTL_d_bits_param   (),
        .io_imemTL_d_bits_size    (),
        .io_imemTL_d_bits_source  (),
        .io_imemTL_d_bits_sink    (),
        .io_imemTL_d_bits_denied  (imem_d_denied),
        .io_imemTL_d_bits_data    (imem_d_data),
        .io_imemTL_d_bits_corrupt (),

        .io_csrTL_a_ready         (csr_a_ready),
        .io_csrTL_a_valid         (csr_a_valid),
        .io_csrTL_a_bits_opcode   (csr_a_opcode),
        .io_csrTL_a_bits_param    (3'd0),
        .io_csrTL_a_bits_size     (csr_a_size),
        .io_csrTL_a_bits_source   (1'b0),
        .io_csrTL_a_bits_address  (csr_a_address),
        .io_csrTL_a_bits_mask     (csr_a_mask),
        .io_csrTL_a_bits_data     (csr_a_data),
        .io_csrTL_a_bits_corrupt  (1'b0),
        .io_csrTL_d_ready         (csr_d_ready),
        .io_csrTL_d_valid         (csr_d_valid),
        .io_csrTL_d_bits_opcode   (csr_d_opcode),
        .io_csrTL_d_bits_param    (),
        .io_csrTL_d_bits_size     (),
        .io_csrTL_d_bits_source   (),
        .io_csrTL_d_bits_sink     (),
        .io_csrTL_d_bits_denied   (csr_d_denied),
        .io_csrTL_d_bits_data     (csr_d_data),
        .io_csrTL_d_bits_corrupt  (),

        .io_dmaTL_a_ready         (dma_a_ready),
        .io_dmaTL_a_valid         (dma_a_valid),
        .io_dmaTL_a_bits_opcode   (dma_a_opcode),
        .io_dmaTL_a_bits_param    (),
        .io_dmaTL_a_bits_size     (),
        .io_dmaTL_a_bits_source   (dma_a_source),
        .io_dmaTL_a_bits_address  (dma_a_address),
        .io_dmaTL_a_bits_mask     (dma_a_mask),
        .io_dmaTL_a_bits_data     (dma_a_data),
        .io_dmaTL_a_bits_corrupt  (),
        .io_dmaTL_d_ready         (dma_d_ready),
        .io_dmaTL_d_valid         (dma_d_valid),
        .io_dmaTL_d_bits_opcode   (dma_d_opcode),
        .io_dmaTL_d_bits_param    (2'd0),
        .io_dmaTL_d_bits_size     (4'd5),
        .io_dmaTL_d_bits_source   (dma_d_source),
        .io_dmaTL_d_bits_sink     (1'b0),
        .io_dmaTL_d_bits_denied   (1'b0),
        .io_dmaTL_d_bits_data     (dma_d_data),
        .io_dmaTL_d_bits_corrupt  (1'b0),

        .io_vmemTL_a_ready        (vmem_a_ready),
        .io_vmemTL_a_valid        (vmem_a_valid),
        .io_vmemTL_a_bits_opcode  (vmem_a_opcode),
        .io_vmemTL_a_bits_param   (3'd0),
        .io_vmemTL_a_bits_size    (vmem_a_size),
        .io_vmemTL_a_bits_source  (1'b0),
        .io_vmemTL_a_bits_address (vmem_a_address),
        .io_vmemTL_a_bits_mask    (vmem_a_mask),
        .io_vmemTL_a_bits_data    (vmem_a_data),
        .io_vmemTL_a_bits_corrupt (1'b0),
        .io_vmemTL_d_ready        (vmem_d_ready),
        .io_vmemTL_d_valid        (vmem_d_valid),
        .io_vmemTL_d_bits_opcode  (vmem_d_opcode),
        .io_vmemTL_d_bits_param   (),
        .io_vmemTL_d_bits_size    (),
        .io_vmemTL_d_bits_source  (),
        .io_vmemTL_d_bits_sink    (),
        .io_vmemTL_d_bits_denied  (vmem_d_denied),
        .io_vmemTL_d_bits_data    (vmem_d_data),
        .io_vmemTL_d_bits_corrupt (),

        .io_halted                (halted),
        .io_features              (features),
        .io_dbg_mxu0DataBusy      (dbg_mxu0_data),
        .io_dbg_mxu0CompBusy      (dbg_mxu0_comp),
        .io_dbg_mxu1DataBusy      (dbg_mxu1_data),
        .io_dbg_mxu1CompBusy      (dbg_mxu1_comp),
        .io_dbg_xluBusy           (dbg_xlu),
        .io_dbg_dmaBusy_0         (dbg_dma[0]),
        .io_dbg_dmaBusy_1         (dbg_dma[1]),
        .io_dbg_dmaBusy_2         (dbg_dma[2]),
        .io_dbg_dmaBusy_3         (dbg_dma[3]),
        .io_dbg_dmaBusy_4         (dbg_dma[4]),
        .io_dbg_dmaBusy_5         (dbg_dma[5]),
        .io_dbg_dmaBusy_6         (dbg_dma[6]),
        .io_dbg_dmaBusy_7         (dbg_dma[7]),
        .io_dbg_lsuBusy           (dbg_lsu),
        .io_dbg_scaleRegs_0       (), .io_dbg_scaleRegs_1  (), .io_dbg_scaleRegs_2  (),
        .io_dbg_scaleRegs_3       (), .io_dbg_scaleRegs_4  (), .io_dbg_scaleRegs_5  (),
        .io_dbg_scaleRegs_6       (), .io_dbg_scaleRegs_7  (), .io_dbg_scaleRegs_8  (),
        .io_dbg_scaleRegs_9       (), .io_dbg_scaleRegs_10 (), .io_dbg_scaleRegs_11 (),
        .io_dbg_scaleRegs_12      (), .io_dbg_scaleRegs_13 (), .io_dbg_scaleRegs_14 (),
        .io_dbg_scaleRegs_15      (), .io_dbg_scaleRegs_16 (), .io_dbg_scaleRegs_17 (),
        .io_dbg_scaleRegs_18      (), .io_dbg_scaleRegs_19 (), .io_dbg_scaleRegs_20 (),
        .io_dbg_scaleRegs_21      (), .io_dbg_scaleRegs_22 (), .io_dbg_scaleRegs_23 (),
        .io_dbg_scaleRegs_24      (), .io_dbg_scaleRegs_25 (), .io_dbg_scaleRegs_26 (),
        .io_dbg_scaleRegs_27      (), .io_dbg_scaleRegs_28 (), .io_dbg_scaleRegs_29 (),
        .io_dbg_scaleRegs_30      (), .io_dbg_scaleRegs_31 (),
        .io_dbg_dbg0              (),
        .io_dbg_dbg1              ());

    // ── System registers ────────────────────────────────────────────────────
    localparam logic [31:0] DRAM_BYTES = EXT_DRAM ? 32'h8000_0000 : 32'(32 << DRAM_LINE_W);
    atlas_sys_regs #(.CLK_HZ(CLK_HZ), .BAUD(BAUD), .DRAM_BYTES(DRAM_BYTES), .ETH(ETH),
                     .ETH_IP(ETH_IP), .ETH_MAC(ETH_MAC), .ETH_PORT(ETH_PORT)) u_sys (
        .clk, .rst,
        .a_valid(n_a_valid[SYS]), .a_ready(n_a_ready[SYS]), .a(n_a[SYS]),
        .d_valid(n_d_valid[SYS]), .d_ready(n_d_ready[SYS]), .d(n_d[SYS]),
        .dropped, .bad_cmd, .denied, .features, .dram_ready(EXT_DRAM ? ext_dram_ready : 1'b1),
        .atlas_status({dbg_dma, 1'b0, dbg_lsu, dbg_xlu, dbg_mxu1_comp, dbg_mxu1_data,
                       dbg_mxu0_comp, dbg_mxu0_data, halted}),
        .user_leds,
        .eth_pcs_status, .eth_phy_ready, .eth_phy_id_ok, .eth_gmii_alive, .eth_synced,
        .eth_n_exec, .eth_n_dup, .eth_n_drop, .eth_rx_dropped, .eth_rx_bad, .eth_rx_frames,
        .eth_ip(local_ip));

    // ── DRAM ────────────────────────────────────────────────────────────────
    if (EXT_DRAM) begin : g_ext
        assign ext_a_valid          = dev_a_valid[DRAM];
        assign dev_a_ready[DRAM]    = ext_a_ready;
        assign ext_a                = dev_a;
        assign dev_d_valid[DRAM]    = ext_d_valid;
        assign ext_d_ready          = dev_d_ready[DRAM];
        assign dev_d[DRAM]          = ext_d;
        assign ext_dma_a_valid      = dma_a_valid;
        assign dma_a_ready          = ext_dma_a_ready;
        assign ext_dma_a_opcode     = dma_a_opcode;
        assign ext_dma_a_source     = dma_a_source;
        assign ext_dma_a_address    = dma_a_address;
        assign ext_dma_a_mask       = dma_a_mask;
        assign ext_dma_a_data       = dma_a_data;
        assign dma_d_valid          = ext_dma_d_valid;
        assign ext_dma_d_ready      = dma_d_ready;
        assign dma_d_opcode         = ext_dma_d_opcode;
        assign dma_d_source         = ext_dma_d_source;
        assign dma_d_data           = ext_dma_d_data;
    end else begin : g_onchip
        atlas_dram #(.LINE_W(DRAM_LINE_W), .LAT(DRAM_LAT), .SW(SW)) u_dram (
            .clk, .rst,
            .a_valid(dev_a_valid[DRAM]), .a_ready(dev_a_ready[DRAM]), .a(dev_a),
            .d_valid(dev_d_valid[DRAM]), .d_ready(dev_d_ready[DRAM]), .d(dev_d[DRAM]),
            .dma_a_valid, .dma_a_ready, .dma_a_opcode, .dma_a_source, .dma_a_address,
            .dma_a_mask, .dma_a_data,
            .dma_d_valid, .dma_d_ready, .dma_d_opcode, .dma_d_source, .dma_d_data);
        assign ext_a_valid = 1'b0;
        assign ext_a = '0;
        assign ext_d_ready = 1'b0;
        assign ext_dma_a_valid = 1'b0;
        assign {ext_dma_a_opcode, ext_dma_a_source, ext_dma_a_address, ext_dma_a_mask, ext_dma_a_data} = '0;
        assign ext_dma_d_ready = 1'b0;
    end
endmodule
