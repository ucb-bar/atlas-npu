// From sashakatne/nvfp4-dotprod-formal-dv 125db59 (Apache-2.0, LICENSE.nvfp4-dotprod-formal-dv).
// Modified for Atlas: the four simulation-reference `include lines are removed; the
// synthesizable types and parameters are unchanged.
`ifndef DOTPROD_PKG_SV
`define DOTPROD_PKG_SV
package dotprod_pkg;
  typedef enum logic [1:0] { FMT_INT8 = 2'd0, FMT_BF16 = 2'd1, FMT_NVFP4 = 2'd2 } fmt_e;

  parameter int N_LANES    = 8;
  parameter int INT8_W     = 8;
  parameter int PROD_W     = 16;   // signed 8x8 product
  parameter int ACC_W      = 24;   // exact 8-lane sum headroom (needs >=19b: max sum 8*128*128=131072)
  parameter int INT8_OUT_W = 32;   // saturating output width
  parameter int BF16_W     = 16;
  parameter int BF16_SIG_W = 8;    // hidden 1 + 7-bit stored mantissa
  parameter int BF16_PROD_W = 16;  // exact 8x8 significand product
  parameter int ACC_BF16_W = 56;   // fixed-point exact BF16 accumulator
  parameter int FP32_W     = 32;
  parameter int BF16_EXP_LO = 119;
  parameter int BF16_EXP_HI = 134;
  parameter int BF16_ACC_FRAC_BITS = 30; // accumulator LSB has weight 2^-30
  parameter logic [FP32_W-1:0] FP32_QNAN = 32'h7FC0_0000;

  typedef struct packed {
    logic sat;
    logic invalid;
    logic is_nan;
    logic is_inf;
  } dotprod_status_t;

  typedef struct packed {
    logic                      sign;
    logic signed [9:0]         q;
    logic [BF16_SIG_W-1:0]     sig;
    logic                      is_zero;
    logic                      is_inf;
    logic                      is_nan;
    logic                      is_oor;  // normal operand with exp outside [119,134]
  } bf16_decoded_t;

  typedef struct packed {
    logic                       p_sign;
    logic signed [9:0]          q;
    logic [BF16_PROD_W-1:0]     p;
    logic                       is_nan;
    logic                       is_inf;
    logic                       is_zero;
    logic                       invalid;
  } bf16_product_t;

  // Pre-round decomposition of the BF16 dot-product: the linear fixed-point
  // accumulation plus the resolved special-case outcome, BEFORE the RNE round.
  // Exposed so the top AG proof can assert the (linear) datapath equivalence
  // without a rounder network on either side of the miter; the rounded-result
  // equivalence then follows by transitivity with the standalone rounder proof.
  typedef struct packed {
    logic signed [ACC_BF16_W-1:0] acc;
    logic                         special_valid;
    logic [FP32_W-1:0]            special_result;
    dotprod_status_t              special_status;
  } bf16_preround_t;

  parameter int NVFP4_BLOCK   = 16;
  parameter int E2M1_W        = 4;
  parameter int UE4M3_W       = 8;
  parameter int NVFP4_INNER_W = 13;
  parameter logic [UE4M3_W-1:0] UE4M3_NAN = 8'h7F;

  typedef struct packed { logic sign; logic [3:0] mag_int; } e2m1_decoded_t;
  typedef struct packed { logic signed [8:0] prod; } nvfp4_product_t;
  typedef struct packed {
    logic signed [NVFP4_INNER_W-1:0] inner_sum;
    logic [7:0]                      scale_sig;
    logic signed [6:0]               scale_exp;
    logic                            scale_is_nan;
  } nvfp4_preround_t;

  // Saturate a wide signed value to INT8_OUT_W bits; assert sat when clamped.
  function automatic logic signed [INT8_OUT_W-1:0] sat_cast
      (input logic signed [ACC_W-1:0] v, output logic sat);
    localparam logic signed [INT8_OUT_W-1:0] MAXV = {1'b0, {(INT8_OUT_W-1){1'b1}}};
    localparam logic signed [INT8_OUT_W-1:0] MINV = {1'b1, {(INT8_OUT_W-1){1'b0}}};
    logic signed [INT8_OUT_W-1:0] ext;
    ext = INT8_OUT_W'(v);          // sign-extend (ACC_W <= INT8_OUT_W so exact here)
    if (v > MAXV)      begin sat = 1'b1; sat_cast = MAXV; end
    else if (v < MINV) begin sat = 1'b1; sat_cast = MINV; end
    else               begin sat = 1'b0; sat_cast = ext;  end
  endfunction

endpackage
`endif
