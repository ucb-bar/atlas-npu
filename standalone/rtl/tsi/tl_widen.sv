// A 32-bit host on the beat fabric: the word goes in its own lane of the beat, and the response
// returns that lane. The fabric has one transaction in flight.
module tl_widen
  import tl_pkg::*;
(
  input  logic clk,
  input  logic a_valid,
  output logic a_ready,
  input  a_t   a,
  output logic d_valid,
  input  logic d_ready,
  output d_t   d,
  output logic w_a_valid,
  input  logic w_a_ready,
  output wa_t  w_a,
  input  logic w_d_valid,
  output logic w_d_ready,
  input  wd_t  w_d
);
  logic [2:0] lane;  // word lane of the request in flight
  assign w_a_valid = a_valid;
  assign a_ready = w_a_ready;
  assign w_a = '{opcode: a.opcode, address: a.address,
                 mask: 32'(a.mask) << (4 * a.address[4:2]), data: {8{a.data}}};
  assign d_valid = w_d_valid;
  assign w_d_ready = d_ready;
  assign d = '{opcode: w_d.opcode, denied: w_d.denied, data: w_d.data[32 * lane +: 32]};
  always_ff @(posedge clk) if (a_valid && a_ready) lane <= a.address[4:2];
endmodule
