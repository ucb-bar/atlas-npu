// A 32-bit device on the beat fabric. Each word of the request with a nonempty byte mask goes to
// the device in turn, lowest address first. The response holds the words read (other words 0)
// and is denied if any word was denied or the mask is empty.
module tl_narrow
  import tl_pkg::*;
(
  input  logic clk,
  input  logic rst,
  input  logic a_valid,
  output logic a_ready,
  input  wa_t  a,
  output logic d_valid,
  input  logic d_ready,
  output wd_t  d,
  output logic n_a_valid,
  input  logic n_a_ready,
  output a_t   n_a,
  input  logic n_d_valid,
  output logic n_d_ready,
  input  d_t   n_d
);
  wa_t        req;
  logic [7:0] todo;
  logic       active, waiting;  // serving a request; a word is at the device
  logic [2:0] w;                // lowest word still to send

  always_comb begin
    w = '0;
    for (int i = 7; i >= 0; i--) if (todo[i]) w = 3'(i);
  end

  assign a_ready   = !active && !d_valid;
  assign n_a_valid = active && !waiting && todo != 0;
  assign n_a       = '{opcode: req.opcode, address: {req.address[31:5], w, 2'b00},
                       mask: req.mask[4*w +: 4], data: req.data[32*w +: 32]};
  assign n_d_ready = waiting;

  always_ff @(posedge clk) begin
    if (rst) begin
      active  <= 1'b0;
      waiting <= 1'b0;
      d_valid <= 1'b0;
    end else begin
      if (d_valid && d_ready) d_valid <= 1'b0;
      if (a_valid && a_ready) begin
        req    <= a;
        todo   <= words_of(a.mask);
        active <= 1'b1;
        d      <= '{opcode: a.opcode == A_GET ? D_ACK_DATA : D_ACK, denied: a.mask == 0, data: '0};
      end
      if (n_a_valid && n_a_ready) waiting <= 1'b1;
      if (n_d_valid && n_d_ready) begin
        waiting            <= 1'b0;
        todo[w]            <= 1'b0;
        d.data[32*w +: 32] <= n_d.data;
        if (n_d.denied) d.denied <= 1'b1;
      end
      if (active && !waiting && todo == 0) begin
        active  <= 1'b0;
        d_valid <= 1'b1;
      end
    end
  end
endmodule
