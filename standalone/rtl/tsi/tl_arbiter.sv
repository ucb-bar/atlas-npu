// Two hosts share the beat fabric, one transaction in flight. A host keeps the fabric from the
// cycle it presents a request until its response is accepted. When both request, the host that
// did not go last wins. Host 0 is UART TSI (after tl_widen), host 1 is Ethernet TSI.
//
// From ucb-bar/nvfp4 0c611e1 (fpga/tsi/tl_arbiter.sv).
module tl_arbiter
    import tl_pkg::*;
(
    input  logic       clk,
    input  logic       rst,

    input  logic [1:0] host_a_valid,
    output logic [1:0] host_a_ready,
    input  wa_t        host_a [2],
    output logic [1:0] host_d_valid,
    input  logic [1:0] host_d_ready,
    output wd_t        host_d,

    output logic       a_valid,
    input  logic       a_ready,
    output wa_t        a,
    input  logic       d_valid,
    output logic       d_ready,
    input  wd_t        d
);
    logic locked, outstanding, owner, last_owner, select_host;

    assign select_host = locked ? owner :
                         host_a_valid[1] && (!host_a_valid[0] || !last_owner);
    assign a       = host_a[select_host];
    assign a_valid = !outstanding && host_a_valid[select_host];
    assign d_ready = outstanding && host_d_ready[owner];
    assign host_d  = d;

    always_comb begin
        host_a_ready = '0;
        host_d_valid = '0;
        if (!outstanding) host_a_ready[select_host] = a_ready;
        if (outstanding)  host_d_valid[owner] = d_valid;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            locked      <= 1'b0;
            outstanding <= 1'b0;
            owner       <= 1'b0;
            last_owner  <= 1'b1;
        end else begin
            if (!locked && a_valid) begin
                locked <= 1'b1;
                owner  <= select_host;
            end
            if (a_valid && a_ready) outstanding <= 1'b1;
            if (d_valid && d_ready) begin
                locked      <= 1'b0;
                outstanding <= 1'b0;
                last_owner  <= owner;
            end
        end
    end
endmodule
