// Standalone Atlas SoC: AtlasCore, an on-chip DRAM, and a TL-UL host fabric. A host (the UART
// TSI target on the board, the simulation driver in Verilator) replaces the CPU: it loads the
// kernel into IMEM and the data into DRAM, starts the core through EXEC_CONTROL, polls DBG0, and
// reads the results back.
//
//   0x0000_0000 - 0x0000_0FFF  sys_regs    link identification, status, Atlas busy bits
//   0x0002_0000 - 0x0003_FFFF  Atlas IMEM  128 KiB (reads stall while the core runs)
//   0x0004_0000 - 0x0004_0FFF  Atlas CSR   CYCLE, INSTCNT, STATUS, ILLEGAL_PC, DBG0, DBG1, EXEC_CONTROL
//   0x2000_0000 - 0x2017_FFFF  Atlas VMEM  1.5 MiB (0x2018_0000 - 0x201F_FFFF is denied)
//   0x8000_0000 - 0xFFFF_FFFF  DRAM        2^DRAM_LINE_W x 32 B, aliased over the window
//
// The DMA engine reaches the same DRAM through the second memory port. Unmapped accesses get a
// denied response.
module atlas_soc
    import tl_pkg::*;
#(
    parameter int CLK_HZ      = 50_000_000,
    parameter int BAUD        = 921_600,
    parameter int DRAM_LINE_W = 17,
    parameter int DRAM_LAT    = 8
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

    input  logic       dropped,   // TSI link status, reported in sys_regs
    input  logic       bad_cmd,
    output logic       denied,    // pulses for every denied response
    output logic       halted,    // Atlas scalar core halted
    output logic [2:0] user_leds
);
    localparam int N  = 5;
    localparam int SW = 6;  // DMA source ID width (DmaParams.tagBits)
    localparam logic [N-1:0][31:0] BASE = {32'h8000_0000, 32'h2000_0000, 32'h0004_0000,
                                           32'h0002_0000, 32'h0000_0000};
    localparam logic [N-1:0][31:0] MASK = {32'h7FFF_FFFF, 32'h001F_FFFF, 32'h0000_0FFF,
                                           32'h0001_FFFF, 32'h0000_0FFF};
    localparam int SYS = 0, IMEM = 1, CSR = 2, VMEM = 3, DRAM = 4;

    // ── Host fabric ─────────────────────────────────────────────────────────
    logic   wa_valid, wa_ready, wd_valid, wd_ready;
    wa_t    wa;
    wd_t    wd;

    tl_widen u_widen (
        .clk, .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
        .w_a_valid(wa_valid), .w_a_ready(wa_ready), .w_a(wa),
        .w_d_valid(wd_valid), .w_d_ready(wd_ready), .w_d(wd));

    logic [N-1:0] dev_a_valid, dev_a_ready, dev_d_valid, dev_d_ready;
    wa_t          dev_a;
    wd_t          dev_d [N];

    tl_xbar #(.N(N), .BASE(BASE), .MASK(MASK)) u_xbar (
        .clk, .rst,
        .a_valid(wa_valid), .a_ready(wa_ready), .a(wa),
        .d_valid(wd_valid), .d_ready(wd_ready), .d(wd),
        .dev_a_valid, .dev_a_ready, .dev_a, .dev_d_valid, .dev_d_ready, .dev_d,
        .denied);

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
    atlas_sys_regs #(.CLK_HZ(CLK_HZ), .BAUD(BAUD), .DRAM_BYTES(32 << DRAM_LINE_W)) u_sys (
        .clk, .rst,
        .a_valid(n_a_valid[SYS]), .a_ready(n_a_ready[SYS]), .a(n_a[SYS]),
        .d_valid(n_d_valid[SYS]), .d_ready(n_d_ready[SYS]), .d(n_d[SYS]),
        .dropped, .bad_cmd, .denied, .features,
        .atlas_status({dbg_dma, 1'b0, dbg_lsu, dbg_xlu, dbg_mxu1_comp, dbg_mxu1_data,
                       dbg_mxu0_comp, dbg_mxu0_data, halted}),
        .user_leds);

    // ── DRAM ────────────────────────────────────────────────────────────────
    atlas_dram #(.LINE_W(DRAM_LINE_W), .LAT(DRAM_LAT), .SW(SW)) u_dram (
        .clk, .rst,
        .a_valid(dev_a_valid[DRAM]), .a_ready(dev_a_ready[DRAM]), .a(dev_a),
        .d_valid(dev_d_valid[DRAM]), .d_ready(dev_d_ready[DRAM]), .d(dev_d[DRAM]),
        .dma_a_valid, .dma_a_ready, .dma_a_opcode, .dma_a_source, .dma_a_address,
        .dma_a_mask, .dma_a_data,
        .dma_d_valid, .dma_d_ready, .dma_d_opcode, .dma_d_source, .dma_d_data);
endmodule
