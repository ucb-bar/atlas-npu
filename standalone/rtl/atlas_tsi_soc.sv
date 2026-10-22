// UART TSI target -> atlas_soc. The host speaks the PyUARTSI / Chipyard UART-TSI protocol.
module atlas_tsi_soc
    import tl_pkg::*;
#(
    parameter int CLK_HZ      = 50_000_000,
    parameter int BAUD        = 921_600,
    parameter int DRAM_LINE_W = 17,
    parameter int DRAM_LAT    = 8
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       rxd,
    output logic       txd,
    output logic       activity,  // toggles when a TSI command completes
    output logic       error,     // a link error or denied access has occurred
    output logic       halted,
    output logic [2:0] user_leds
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

    atlas_soc #(.CLK_HZ(CLK_HZ), .BAUD(BAUD), .DRAM_LINE_W(DRAM_LINE_W), .DRAM_LAT(DRAM_LAT)) u_soc (
        .clk, .rst, .a_valid, .a_ready, .a, .d_valid, .d_ready, .d,
        .dropped, .bad_cmd, .denied, .halted, .user_leds);

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
