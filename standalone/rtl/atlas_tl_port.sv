// Adapters from the TL-UL fabric (rtl/tsi/tl_pkg.sv) to AtlasCore's TileLink manager ports.
//
//   atlas_tl_word  one 32-bit word (a_t/d_t, after tl_narrow) to a 32-bit port: IMEM, CSR
//   atlas_tl_beat  one 32-byte beat (wa_t/wd_t) to the 256-bit VMEM port. A request at or past
//                  LIMIT bytes into the window is answered here, denied, so a stray address never
//                  reaches Vmem's unimplemented banks.
//
// Both carry one transaction at a time, as the fabric does. Requests use source 0; a Get reads
// the whole word or beat, a Put is PutFullData when every byte is enabled, else PutPartialData.
module atlas_tl_word
    import tl_pkg::*;
(
    input  logic        a_valid,
    output logic        a_ready,
    input  a_t          a,
    output logic        d_valid,
    input  logic        d_ready,
    output d_t          d,

    output logic        tl_a_valid,
    input  logic        tl_a_ready,
    output logic [2:0]  tl_a_opcode,
    output logic [3:0]  tl_a_size,
    output logic [31:0] tl_a_address,
    output logic [3:0]  tl_a_mask,
    output logic [31:0] tl_a_data,
    input  logic        tl_d_valid,
    output logic        tl_d_ready,
    input  logic [2:0]  tl_d_opcode,
    input  logic        tl_d_denied,
    input  logic [31:0] tl_d_data
);
    wire get = a.opcode == A_GET;

    assign tl_a_valid   = a_valid;
    assign a_ready      = tl_a_ready;
    assign tl_a_opcode  = get ? 3'd4 : a.mask == 4'hF ? 3'd0 : 3'd1;
    assign tl_a_size    = 4'd2;
    assign tl_a_address = {a.address[31:2], 2'b00};
    assign tl_a_mask    = get ? 4'hF : a.mask;
    assign tl_a_data    = a.data;

    assign d_valid    = tl_d_valid;
    assign tl_d_ready = d_ready;
    assign d          = '{opcode: tl_d_opcode == 3'd1 ? D_ACK_DATA : D_ACK,
                          denied: tl_d_denied, data: tl_d_data};
endmodule

module atlas_tl_beat
    import tl_pkg::*;
#(
    parameter logic [31:0] WINDOW = 32'h001F_FFFF,  // address bits below the device base
    parameter logic [31:0] LIMIT  = 32'h0018_0000   // implemented bytes from the base
) (
    input  logic         clk,
    input  logic         rst,
    input  logic         a_valid,
    output logic         a_ready,
    input  wa_t          a,
    output logic         d_valid,
    input  logic         d_ready,
    output wd_t          d,

    output logic         tl_a_valid,
    input  logic         tl_a_ready,
    output logic [2:0]   tl_a_opcode,
    output logic [3:0]   tl_a_size,
    output logic [31:0]  tl_a_address,
    output logic [31:0]  tl_a_mask,
    output logic [255:0] tl_a_data,
    input  logic         tl_d_valid,
    output logic         tl_d_ready,
    input  logic [2:0]   tl_d_opcode,
    input  logic         tl_d_denied,
    input  logic [255:0] tl_d_data
);
    wire get      = a.opcode == A_GET;
    wire in_range = (a.address & WINDOW) < LIMIT;

    logic miss, miss_get;  // a denied response for an out-of-range request is pending

    assign tl_a_valid   = a_valid && in_range && !miss;
    assign a_ready      = !miss && (in_range ? tl_a_ready : 1'b1);
    assign tl_a_opcode  = get ? 3'd4 : a.mask == 32'hFFFF_FFFF ? 3'd0 : 3'd1;
    assign tl_a_size    = 4'd5;
    assign tl_a_address = {a.address[31:5], 5'b0};
    assign tl_a_mask    = get ? 32'hFFFF_FFFF : a.mask;
    assign tl_a_data    = a.data;

    assign d_valid    = miss || tl_d_valid;
    assign tl_d_ready = d_ready && !miss;
    assign d          = miss ? '{opcode: miss_get ? D_ACK_DATA : D_ACK, denied: 1'b1, data: '0}
                             : '{opcode: tl_d_opcode == 3'd1 ? D_ACK_DATA : D_ACK,
                                 denied: tl_d_denied, data: tl_d_data};

    always_ff @(posedge clk) begin
        if (rst) begin
            miss <= 1'b0;
        end else if (a_valid && a_ready && !in_range) begin
            miss     <= 1'b1;
            miss_get <= get;
        end else if (miss && d_ready) begin
            miss <= 1'b0;
        end
    end
endmodule
