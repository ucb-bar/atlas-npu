// On-chip stand-in for DRAM: 2^LINE_W lines of 32 bytes in a two-port memory.
//
//   port A  the host fabric (wa_t/wd_t), one transaction at a time
//   port B  AtlasCore's DMA client: single-beat Get and PutFullData, up to QD requests in flight,
//           answered in order with their source IDs
//
// Both ports take the line index from address bits [LINE_W+4:5], so the whole DRAM window
// (0x8000_0000 up) and DMA's 64-bit {dma.base, x[rs]} addresses alias onto the memory.
//
// The FPGA build uses UltraRAM through XPM (true dual port, no_change write mode, byte writes);
// simulation uses a behavioral array with the same read latency.
module atlas_dram
    import tl_pkg::*;
#(
    parameter int LINE_W = 17,  // 2^17 lines x 32 B = 4 MiB
    parameter int LAT    = 8,   // read latency: UltraRAM cascade pipeline stages
    parameter int SW     = 6    // DMA source ID width
) (
    input  logic          clk,
    input  logic          rst,

    input  logic          a_valid,
    output logic          a_ready,
    input  wa_t           a,
    output logic          d_valid,
    input  logic          d_ready,
    output wd_t           d,

    input  logic          dma_a_valid,
    output logic          dma_a_ready,
    input  logic [2:0]    dma_a_opcode,
    input  logic [SW-1:0] dma_a_source,
    input  logic [63:0]   dma_a_address,
    input  logic [31:0]   dma_a_mask,
    input  logic [255:0]  dma_a_data,
    output logic          dma_d_valid,
    input  logic          dma_d_ready,
    output logic [2:0]    dma_d_opcode,
    output logic [SW-1:0] dma_d_source,
    output logic [255:0]  dma_d_data
);
    localparam int QD = 16;  // DMA requests in flight (> LAT); a power of two for sync_fifo
    localparam int CW = $clog2(LAT + 1);
    localparam int QW = $clog2(QD + 1);

    logic              ram_a_en, ram_b_en;
    logic [31:0]       ram_a_we, ram_b_we;
    logic [LINE_W-1:0] ram_a_addr, ram_b_addr;
    logic [255:0]      ram_a_q, ram_b_q;

    // ── Port A: host ────────────────────────────────────────────────────────
    logic          h_busy, h_get;
    logic [CW-1:0] h_left;  // cycles until the read data is valid

    wire h_fire = a_valid && a_ready;
    assign a_ready    = !h_busy;
    assign ram_a_en   = h_fire;
    assign ram_a_we   = a.opcode == A_GET ? '0 : a.mask;
    assign ram_a_addr = a.address[LINE_W+4:5];

    always_ff @(posedge clk) begin
        if (rst) begin
            h_busy  <= 1'b0;
            d_valid <= 1'b0;
        end else begin
            if (h_fire) begin
                h_busy <= 1'b1;
                h_get  <= a.opcode == A_GET;
                h_left <= CW'(LAT);
            end else if (h_busy && !d_valid) begin
                h_left <= h_left - 1'b1;
                if (h_left == CW'(1)) begin
                    d_valid <= 1'b1;
                    d       <= '{opcode: h_get ? D_ACK_DATA : D_ACK, denied: 1'b0,
                                 data: h_get ? ram_a_q : '0};
                end
            end
            if (d_valid && d_ready) begin
                d_valid <= 1'b0;
                h_busy  <= 1'b0;
            end
        end
    end

    // ── Port B: DMA ─────────────────────────────────────────────────────────
    logic [QW-1:0]             b_out;  // accepted and not yet answered
    logic [LAT-1:0]            p_v, p_get;
    logic [SW-1:0]             p_src [LAT];

    wire b_get  = dma_a_opcode == 3'd4;
    wire b_fire = dma_a_valid && dma_a_ready;
    wire b_done = dma_d_valid && dma_d_ready;

    assign dma_a_ready = b_out < QW'(QD);
    assign ram_b_en    = b_fire;
    assign ram_b_we    = b_get ? '0 : dma_a_mask;
    assign ram_b_addr  = dma_a_address[LINE_W+4:5];

    always_ff @(posedge clk) begin
        if (rst) begin
            b_out <= '0;
            p_v   <= '0;
        end else begin
            b_out <= b_out + QW'(b_fire) - QW'(b_done);
            p_v   <= {p_v[LAT-2:0], b_fire};
        end
        p_get    <= {p_get[LAT-2:0], b_get};
        p_src[0] <= dma_a_source;
        for (int i = 1; i < LAT; i++) p_src[i] <= p_src[i-1];
    end

    sync_fifo #(.W(3 + SW + 256), .DEPTH(QD)) u_resp (
        .clk, .rst,
        .wdata ({p_get[LAT-1] ? 3'd1 : 3'd0, p_src[LAT-1], p_get[LAT-1] ? ram_b_q : 256'd0}),
        .wvalid(p_v[LAT-1]),
        .wready(),
        .rdata ({dma_d_opcode, dma_d_source, dma_d_data}),
        .rvalid(dma_d_valid),
        .rready(dma_d_ready));

    // ── Memory ──────────────────────────────────────────────────────────────
`ifdef VERILATOR
    // public: atlas_sim.cpp's B command loads files straight into the array (a backdoor preload)
    logic [255:0] mem [2**LINE_W] /*verilator public_flat_rw*/;
    logic [255:0] qa [LAT], qb [LAT];

    always_ff @(posedge clk) begin
        if (ram_a_en) begin
            qa[0] <= mem[ram_a_addr];
            for (int i = 0; i < 32; i++)
                if (ram_a_we[i]) mem[ram_a_addr][8*i +: 8] <= a.data[8*i +: 8];
        end
        if (ram_b_en) begin
            qb[0] <= mem[ram_b_addr];
            for (int i = 0; i < 32; i++)
                if (ram_b_we[i]) mem[ram_b_addr][8*i +: 8] <= dma_a_data[8*i +: 8];
        end
        for (int k = 1; k < LAT; k++) begin
            qa[k] <= qa[k-1];
            qb[k] <= qb[k-1];
        end
    end
    assign ram_a_q = qa[LAT-1];
    assign ram_b_q = qb[LAT-1];
`else
    xpm_memory_tdpram #(
        .ADDR_WIDTH_A        (LINE_W),
        .ADDR_WIDTH_B        (LINE_W),
        .AUTO_SLEEP_TIME     (0),
        .BYTE_WRITE_WIDTH_A  (8),
        .BYTE_WRITE_WIDTH_B  (8),
        .CASCADE_HEIGHT      (0),
        .CLOCKING_MODE       ("common_clock"),
        .ECC_MODE            ("no_ecc"),
        .MEMORY_INIT_FILE    ("none"),
        .MEMORY_INIT_PARAM   ("0"),
        .MEMORY_OPTIMIZATION ("true"),
        .MEMORY_PRIMITIVE    ("ultra"),
        .MEMORY_SIZE         (256 * (2 ** LINE_W)),
        .MESSAGE_CONTROL     (0),
        .READ_DATA_WIDTH_A   (256),
        .READ_DATA_WIDTH_B   (256),
        .READ_LATENCY_A      (LAT),
        .READ_LATENCY_B      (LAT),
        .READ_RESET_VALUE_A  ("0"),
        .READ_RESET_VALUE_B  ("0"),
        .RST_MODE_A          ("SYNC"),
        .RST_MODE_B          ("SYNC"),
        .SIM_ASSERT_CHK      (0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT        (0),
        .WAKEUP_TIME         ("disable_sleep"),
        .WRITE_DATA_WIDTH_A  (256),
        .WRITE_DATA_WIDTH_B  (256),
        .WRITE_MODE_A        ("no_change"),
        .WRITE_MODE_B        ("no_change"),
        .WRITE_PROTECT       (1)
    ) u_ram (
        .clka(clk), .clkb(clk), .rsta(1'b0), .rstb(1'b0),
        .ena(ram_a_en), .enb(ram_b_en), .regcea(1'b1), .regceb(1'b1),
        .wea(ram_a_we), .web(ram_b_we), .addra(ram_a_addr), .addrb(ram_b_addr),
        .dina(a.data), .dinb(dma_a_data), .douta(ram_a_q), .doutb(ram_b_q),
        .injectdbiterra(1'b0), .injectdbiterrb(1'b0), .injectsbiterra(1'b0), .injectsbiterrb(1'b0),
        .sleep(1'b0), .dbiterra(), .dbiterrb(), .sbiterra(), .sbiterrb());
`endif
endmodule
