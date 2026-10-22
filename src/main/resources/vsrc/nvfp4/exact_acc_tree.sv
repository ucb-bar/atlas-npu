`ifndef EXACT_ACC_TREE_SV
`define EXACT_ACC_TREE_SV
import dotprod_pkg::*;
// Exact balanced adder tree, parameterized by lane count N (power of two) and
// width W. Default N=N_LANES, W=ACC_W keep the INT8/BF16 instances identical.
// For N=8: 3 levels, identical pairwise structure to the original hardcoded tree.
// For N=16 (NVFP4): 4 levels summing all 16 lanes exactly.
module exact_acc_tree #(parameter int N = N_LANES, parameter int W = ACC_W) (
  input  logic signed [W-1:0] wide [N],
  output logic signed [W-1:0] sum
);
  localparam int LEVELS = $clog2(N);
  // node[l] has N entries; only the first N>>l are used at level l.
  // node[0] = inputs; node[LEVELS][0] = final sum.
  genvar l, i;
  generate
    logic signed [W-1:0] node [LEVELS+1][N];
    for (i = 0; i < N; i++) begin : g_in
      assign node[0][i] = wide[i];
    end
    for (l = 0; l < LEVELS; l++) begin : g_lvl
      for (i = 0; i < (N >> (l+1)); i++) begin : g_add
        assign node[l+1][i] = node[l][2*i] + node[l][2*i+1];
      end
    end
  endgenerate
  assign sum = node[LEVELS][0];
endmodule
`endif
