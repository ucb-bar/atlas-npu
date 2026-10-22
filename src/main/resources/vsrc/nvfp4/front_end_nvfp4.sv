`ifndef FRONT_END_NVFP4_SV
`define FRONT_END_NVFP4_SV
import dotprod_pkg::*;

module front_end_nvfp4 (
  input  logic [3:0]    x,
  output e2m1_decoded_t d
);
  logic [1:0] ee;
  logic       m;

  always_comb begin
    ee = x[2:1];
    m  = x[0];
    d.sign = x[3];
    case ({ee, m})
      3'b000: d.mag_int = 4'd0;   // 0.0
      3'b001: d.mag_int = 4'd1;   // 0.5
      3'b010: d.mag_int = 4'd2;   // 1.0
      3'b011: d.mag_int = 4'd3;   // 1.5
      3'b100: d.mag_int = 4'd4;   // 2.0
      3'b101: d.mag_int = 4'd6;   // 3.0
      3'b110: d.mag_int = 4'd8;   // 4.0
`ifdef BUG_E2M1
      3'b111: d.mag_int = 4'd8;   // BUG: 6.0 pattern mapped to 4 instead of 12
`else
      3'b111: d.mag_int = 4'd12;  // 6.0
`endif
    endcase
  end
endmodule
`endif
