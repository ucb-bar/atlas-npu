// UART TSI target -> atlas_soc. The host speaks the PyUARTSI / Chipyard UART-TSI protocol.
module atlas_tsi_soc
    import tl_pkg::*;
#(
    parameter int CLK_HZ      = 50_000_000,
    parameter int BAUD        = 921_600,
    parameter int DRAM_LINE_W = 17,
    parameter int DRAM_LAT    = 8,
    parameter bit EXT_DRAM    = 1'b0
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       rxd,
    output logic       txd,
    output logic       activity,  // toggles when a TSI command completes
    output logic       error,     // a link error or denied access has occurred
    output logic       halted,
    output logic [2:0] user_leds,

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

    atlas_soc #(.CLK_HZ(CLK_HZ), .BAUD(BAUD), .DRAM_LINE_W(DRAM_LINE_W), .DRAM_LAT(DRAM_LAT),
                .EXT_DRAM(EXT_DRAM)) u_soc (
        .clk, .rst, .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
        .dropped, .bad_cmd, .denied, .halted, .user_leds, .ext_a_valid, .ext_a_ready, .ext_a,
        .ext_d_valid, .ext_d_ready, .ext_d, .ext_dma_a_valid, .ext_dma_a_ready, .ext_dma_a_opcode,
        .ext_dma_a_source, .ext_dma_a_address, .ext_dma_a_mask, .ext_dma_a_data, .ext_dma_d_valid,
        .ext_dma_d_ready, .ext_dma_d_opcode, .ext_dma_d_source, .ext_dma_d_data, .ext_dram_ready);

    logic denied_seen;
    always_ff @(posedge clk) begin
        if (rst) begin
            activity    <= 1'b0;
            denied_seen <= 1'b0;
        end else begin
            if (done) activity <= ~activity;
            if (denied) denied_seen <= 1'b1;
        end
    end
    assign error = dropped | bad_cmd | denied_seen;
endmodule
