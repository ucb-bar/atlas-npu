`ifndef ALIGN_NVFP4_SV
`define ALIGN_NVFP4_SV
import dotprod_pkg::*;
// Sign-extend each 9-bit NVFP4 element product into a 13-bit accumulator lane.
module align_nvfp4 (
  input  nvfp4_product_t                   prod [NVFP4_BLOCK],
  output logic signed [NVFP4_INNER_W-1:0]  wide [NVFP4_BLOCK]
);
  always_comb
    foreach (prod[i]) begin
`ifdef BUG_ALIGN
      wide[i] = NVFP4_INNER_W'(prod[i].prod) <<< 1;  // BUG: shift left by 1 (multiply by 2)
`else
      wide[i] = NVFP4_INNER_W'(prod[i].prod);  // sign-extend
`endif
    end
endmodule
`endif
