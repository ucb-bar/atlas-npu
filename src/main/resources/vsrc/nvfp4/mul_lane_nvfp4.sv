`ifndef MUL_LANE_NVFP4_SV
`define MUL_LANE_NVFP4_SV
import dotprod_pkg::*;

module mul_lane_nvfp4 (
  input  logic [3:0]      a,
  input  logic [3:0]      b,
  output nvfp4_product_t  product
);
  e2m1_decoded_t       da;
  e2m1_decoded_t       db;
  logic signed [8:0]   mag;

  front_end_nvfp4 fe_a (.x(a), .d(da));
  front_end_nvfp4 fe_b (.x(b), .d(db));

  always_comb begin
    mag = 9'(da.mag_int) * 9'(db.mag_int);
    product.prod = (da.sign ^ db.sign) ? -mag : mag;
  end
endmodule
`endif
