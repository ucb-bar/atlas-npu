// One TL-UL host to N devices on the beat fabric, one transaction in flight. Device k owns the
// addresses where (address & ~MASK[k]) == BASE[k]. An address no device owns gets a denied
// response (reads return 0), so a stray access cannot stall the link.
module tl_xbar
    import tl_pkg::*;
#(
    parameter int                 N    = 2,
    parameter logic [N-1:0][31:0] BASE = '0,
    parameter logic [N-1:0][31:0] MASK = '0
) (
    input  logic         clk,
    input  logic         rst,

    input  logic         a_valid,
    output logic         a_ready,
    input  wa_t          a,
    output logic         d_valid,
    input  logic         d_ready,
    output wd_t          d,

    output logic [N-1:0] dev_a_valid,
    input  logic [N-1:0] dev_a_ready,
    output wa_t          dev_a,
    input  logic [N-1:0] dev_d_valid,
    output logic [N-1:0] dev_d_ready,
    input  wd_t          dev_d [N],

    output logic         denied  // pulses for every denied response
);
    localparam int SW = $clog2(N + 1);
    localparam logic [SW-1:0] NONE = SW'(N);

    logic          busy;
    logic [SW-1:0] sel, hit;
    wd_t           miss_d;

    always_comb begin
        hit = NONE;
        for (int k = N - 1; k >= 0; k--)
            if ((a.address & ~MASK[k]) == BASE[k]) hit = SW'(k);
    end

    assign dev_a = a;
    always_comb begin
        a_ready = !busy && hit == NONE;
        d_valid = busy && sel == NONE;
        d       = miss_d;
        for (int k = 0; k < N; k++) begin
            dev_a_valid[k] = !busy && hit == SW'(k) && a_valid;
            dev_d_ready[k] = busy && sel == SW'(k) && d_ready;
            if (!busy && hit == SW'(k)) a_ready = dev_a_ready[k];
            if (busy && sel == SW'(k)) begin
                d_valid = dev_d_valid[k];
                d       = dev_d[k];
            end
        end
    end

    assign denied = d_valid && d_ready && d.denied;

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
        end else if (!busy) begin
            if (a_valid && a_ready) begin
                busy   <= 1'b1;
                sel    <= hit;
                miss_d <= '{opcode: a.opcode == A_GET ? D_ACK_DATA : D_ACK, denied: 1'b1, data: '0};
            end
        end else if (d_valid && d_ready) begin
            busy <= 1'b0;
        end
    end
endmodule
