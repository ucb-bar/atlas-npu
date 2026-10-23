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
module atlas_sys_regs
    import tl_pkg::*;
#(
    parameter int CLK_HZ     = 50_000_000,
    parameter int BAUD       = 921_600,
    parameter logic [31:0] DRAM_BYTES = 32'(4 << 20)
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
    output logic [2:0]  user_leds
);
    logic [31:0] scratch, denied_count, cycles, run_cycles, rdata;
    logic        hit;

    wire [9:0] word = a.address[11:2];
    wire       wr   = a.opcode != A_GET;

    always_comb begin
        hit = 1'b1;
        case (word)
            10'h0:   rdata = 32'h4154_4C53;
            10'h1:   rdata = scratch;
            10'h2:   rdata = {28'd0, dram_ready, denied_count != 0, bad_cmd, dropped};
            10'h3:   rdata = denied_count;
            10'h4:   rdata = cycles;
            10'h5:   rdata = CLK_HZ;
            10'h6:   rdata = BAUD;
            10'h7:   rdata = DRAM_BYTES;
            10'h8:   rdata = {16'd0, atlas_status};
            10'h9:   rdata = {28'd0, features};
            10'hA:   rdata = {29'd0, user_leds};
            10'hB:   rdata = run_cycles;
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
        end else begin
            cycles <= cycles + 1'b1;
            if (!atlas_status[0]) run_cycles <= run_cycles + 1'b1;
            if (denied) denied_count <= denied_count + 1'b1;
            if (d_valid && d_ready) d_valid <= 1'b0;
            if (a_valid && a_ready) begin
                d_valid <= 1'b1;
                d       <= '{opcode: wr ? D_ACK : D_ACK_DATA, denied: !hit, data: wr ? '0 : rdata};
                if (wr && word == 10'h1) scratch <= apply_mask(scratch, a.data, a.mask);
                if (wr && word == 10'hA && a.mask[0]) user_leds <= a.data[2:0];
                if (wr && word == 10'hB) run_cycles <= '0;
            end
        end
    end
endmodule
