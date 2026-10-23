// 64-element block-scaled FP4 fused dot product with FP32 accumulate:
//   d = C + sum over blocks g = 0..3 of (sum of 16 E2M1 products in block g) x sA_g x sB_g
// with the arithmetic of NVIDIA tensor cores, as measured on RTX 5090 (mma.sync) and Jetson Thor
// (tcgen05) and modeled by ucb-bar/nvfp4@0c611e1:tcgen05/nvfp4_tcgen05_check.py (rule "rtx5090"):
//   - The sum of each 16-element block is exact. This part is the block datapath of
//     sashakatne/nvfp4-dotprod-formal-dv (mul_lane_nvfp4, align_nvfp4, exact_acc_tree), in LUTs.
//   - The scales, the alignment, the cuts toward zero and the special values are fp4_fedp_acc.
// ucb-bar/nvfp4@0c611e1:fpga/pe/fp4_fedp_dsp.sv (not in Atlas) computes the same outputs for a
// 2 x N grid with the products in DSP slices.
//
// mode: 0 NVFP4 (UE4M3 per 16), 1 MXFP4 (E8M0 per 32), 2 FP4 (no scale), 3 MXFP4 (E8M0 per 16).
// Byte g of sfa / sfb scales block g (mode 1: bytes 0 and 1 scale elements 0-31 and 32-63).
// Fixed latency: d is valid 5 cycles after in_vld. No back-pressure.
module fp4_fedp
    import dotprod_pkg::*;
(
    input  logic         clk,
    input  logic         rst,
    input  logic         in_vld,
    input  logic [255:0] a,     // element k is a[4k+3:4k]
    input  logic [255:0] b,
    input  logic [1:0]   mode,
    input  logic [31:0]  sfa,
    input  logic [31:0]  sfb,
    input  logic [31:0]  c,
    output logic         out_vld,
    output logic [31:0]  d
);
    logic signed [NVFP4_INNER_W-1:0] sigma [4], s1_sigma [4];

    for (genvar g = 0; g < 4; g++) begin : g_blk
        nvfp4_product_t                  prod [NVFP4_BLOCK];
        logic signed [NVFP4_INNER_W-1:0] wide [NVFP4_BLOCK];
        for (genvar i = 0; i < NVFP4_BLOCK; i++) begin : g_mul
            mul_lane_nvfp4 u_mul (.a(a[4*(16*g+i) +: 4]), .b(b[4*(16*g+i) +: 4]), .product(prod[i]));
        end
        align_nvfp4 u_align (.prod, .wide);
        exact_acc_tree #(.N(NVFP4_BLOCK), .W(NVFP4_INNER_W)) u_tree (.wide, .sum(sigma[g]));
    end

    logic all_zero;
    always_comb begin
        all_zero = 1'b1;
        for (int k = 0; k < 64; k++)
            if (a[4*k +: 3] != 3'd0 && b[4*k +: 3] != 3'd0) all_zero = 1'b0;
    end

    always_ff @(posedge clk) s1_sigma <= sigma;

    fp4_fedp_acc u_acc (.clk, .rst, .in_vld, .mode, .sfa, .sfb, .c, .zero(all_zero),
                        .sigma(s1_sigma), .out_vld, .d);
endmodule
