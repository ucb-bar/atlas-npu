// Dual-clock FIFO with valid/ready on both sides: gray-coded pointers cross through two-flop
// synchronizers, the storage is LUT RAM read asynchronously (the first word falls through).
// Depth 2^AW, AW >= 2. Both resets must be asserted together (see atlas_ddr4_dram).
module async_fifo #(
    parameter int W  = 8,
    parameter int AW = 4
) (
    input  logic         wclk,
    input  logic         wrst,
    input  logic [W-1:0] wdata,
    input  logic         wvalid,
    output logic         wready,

    input  logic         rclk,
    input  logic         rrst,
    output logic [W-1:0] rdata,
    output logic         rvalid,
    input  logic         rready
);
    logic [W-1:0] mem [2**AW];
    logic [AW:0]  wbin, wgray, rbin, rgray;
    (* ASYNC_REG = "TRUE" *) logic [AW:0] rgray_w1, rgray_w2;  // rgray in the write domain
    (* ASYNC_REG = "TRUE" *) logic [AW:0] wgray_r1, wgray_r2;  // wgray in the read domain

    function automatic logic [AW:0] gray(logic [AW:0] b);
        return b ^ (b >> 1);
    endfunction

    // ── Write side ──────────────────────────────────────────────────────────
    assign wready = wgray != {~rgray_w2[AW:AW-1], rgray_w2[AW-2:0]};
    wire push = wvalid && wready;

    always_ff @(posedge wclk)
        if (push) mem[wbin[AW-1:0]] <= wdata;

    always_ff @(posedge wclk) begin
        if (wrst) begin
            wbin     <= '0;
            wgray    <= '0;
            rgray_w1 <= '0;
            rgray_w2 <= '0;
        end else begin
            if (push) begin
                wbin  <= wbin + 1'b1;
                wgray <= gray(wbin + 1'b1);
            end
            rgray_w1 <= rgray;
            rgray_w2 <= rgray_w1;
        end
    end

    // ── Read side ───────────────────────────────────────────────────────────
    assign rvalid = rgray != wgray_r2;
    assign rdata  = mem[rbin[AW-1:0]];
    wire pop = rvalid && rready;

    always_ff @(posedge rclk) begin
        if (rrst) begin
            rbin     <= '0;
            rgray    <= '0;
            wgray_r1 <= '0;
            wgray_r2 <= '0;
        end else begin
            if (pop) begin
                rbin  <= rbin + 1'b1;
                rgray <= gray(rbin + 1'b1);
            end
            wgray_r1 <= wgray;
            wgray_r2 <= wgray_r1;
        end
    end
endmodule
