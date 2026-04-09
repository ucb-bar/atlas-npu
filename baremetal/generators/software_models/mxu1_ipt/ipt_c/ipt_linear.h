#ifndef IPT_LINEAR_H
#define IPT_LINEAR_H

/* ipt_linear.h — C translation of IPTLinearRTLFunction.__call__
 *
 * Mirrors the Python tiling loop in iptlinear_functional.py exactly:
 *   - outer loop over output tiles (groups of numLanes output rows)
 *   - inner loop over K-tiles (groups of vecLen input columns)
 *   - per-batch compute_now calls accumulating into psum
 *
 * Memory convention
 * -----------------
 *   w_e4m3  : uint8_t[out_features][in_features]  (row-major)
 *   b_e4m3  : uint8_t[out_features], or NULL
 *   x_e4m3  : uint8_t[batch][in_features]          (row-major)
 *   out_bits: uint16_t[batch][out_features]         (caller-allocated)
 *
 * Call ipt_linear_init_luts() once before first use.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "fp_formats.h"
#include "converters.h"
#include "params_and_requests.h"
#include "inner_product_trees_model.h"


/* Call once at program start (after any other LUT inits). */
static inline void ipt_linear_init_luts(void)
{
    atlas_fp_init_lut();
    atlas_acc_init_lut();
}


/* ipt_linear_call
 *
 * Parameters
 * ----------
 *   p            : model params (numLanes, vecLen, latency, …)
 *   x_e4m3       : activations,  uint8_t[batch * in_features]  row-major
 *   w_e4m3       : weights,       uint8_t[out_features * in_features] row-major
 *   b_e4m3       : biases,        uint8_t[out_features], or NULL
 *   batch        : number of input rows
 *   in_features  : K dimension
 *   out_features : N dimension
 *   scale_exp    : scalar applied to every lane's scaleExp array
 *   out_fmt_sel  : OutBF16 or OutE4M3
 *   out_bits     : caller-allocated uint16_t[batch * out_features], row-major
 *                  written by this function
 */
static inline void ipt_linear_call(
    const InnerProductTreeParams *p,
    const uint8_t  *x_e4m3,
    const uint8_t  *w_e4m3,
    const uint8_t  *b_e4m3,        /* may be NULL */
    int             batch,
    int             in_features,
    int             out_features,
    int             scale_exp,
    OutputFmtSel    out_fmt_sel,
    uint16_t       *out_bits)       /* [batch * out_features] */
{
    const int num_lanes = p->numLanes;
    const int vec_len   = p->vecLen;
    const int num_k_tiles = (in_features + vec_len - 1) / vec_len;

    /* Scratch buffers for a single compute call */
    uint8_t  *act_buf   = (uint8_t  *)calloc((size_t)vec_len,   1);
    uint8_t  *bias_buf  = (uint8_t  *)calloc((size_t)num_lanes, 1);
    uint16_t *psum_buf  = (uint16_t *)calloc((size_t)num_lanes, sizeof(uint16_t));
    int      *sexp_buf  = (int      *)malloc((size_t)num_lanes  * sizeof(int));
    uint16_t *lane_out  = (uint16_t *)malloc((size_t)num_lanes  * sizeof(uint16_t));
    assert(act_buf && bias_buf && psum_buf && sexp_buf && lane_out);

    for (int li = 0; li < num_lanes; li++)
        sexp_buf[li] = scale_exp;

    /* psum per batch row: uint16_t[batch * num_lanes] */
    uint16_t *psum_all = (uint16_t *)calloc((size_t)(batch * num_lanes),
                                             sizeof(uint16_t));
    assert(psum_all);

    /* ------------------------------------------------------------------
     * Outer loop: output tiles
     * ------------------------------------------------------------------ */
    for (int out_base = 0; out_base < out_features; out_base += num_lanes)
    {
        int lane_count = out_features - out_base;
        if (lane_count > num_lanes) lane_count = num_lanes;

        /* Reset per-batch psum to zero for this output tile */
        memset(psum_all, 0, (size_t)(batch * num_lanes) * sizeof(uint16_t));

        /* Create a fresh model for this output tile — mirrors Python's
         * "dut = InnerProductTreesModel(self.p)" inside the out_base loop */
        IPTModel *dut = ipt_model_init(p);

        /* ------------------------------------------------------------------
         * Inner loop: K-tiles
         * ------------------------------------------------------------------ */
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++)
        {
            int k0         = k_tile * vec_len;
            int k1         = k0 + vec_len;
            if (k1 > in_features) k1 = in_features;
            int tile_width = k1 - k0;

            /* Load weights for all lanes */
            for (int lane = 0; lane < num_lanes; lane++)
            {
                uint8_t w_row[/* vec_len, stack-alloc via VLA or heap */1];
                /* Use heap to avoid VLA-size issues */
                uint8_t *w_row_h = (uint8_t *)malloc((size_t)vec_len);
                assert(w_row_h);
                (void)w_row;

                if (lane < lane_count)
                {
                    /* w_e4m3 row for output neuron (out_base + lane) */
                    const uint8_t *src = w_e4m3 +
                        (size_t)(out_base + lane) * (size_t)in_features + k0;
                    memcpy(w_row_h, src, (size_t)tile_width);
                    if (tile_width < vec_len)
                        memset(w_row_h + tile_width, 0,
                               (size_t)(vec_len - tile_width));
                }
                else
                {
                    memset(w_row_h, 0, (size_t)vec_len);
                }

                WeightLoadReq wr;
                wr.weightsDma     = w_row_h;
                wr.weightsDma_len = vec_len;
                wr.laneIdx        = lane;
                wr.last           = (lane == num_lanes - 1);
                ipt_load_weights(dut, &wr);
                free(w_row_h);
            }

            /* Determine addend_sel and bias for this k_tile */
            AddendSel addend_sel;
            if (k_tile == 0 && b_e4m3 != NULL)
            {
                addend_sel = AddendSel_UseBias;
                memset(bias_buf, 0, (size_t)num_lanes);
                for (int lane = 0; lane < lane_count; lane++)
                    bias_buf[lane] = b_e4m3[out_base + lane];
            }
            else if (k_tile == 0)
            {
                addend_sel = AddendSel_UseAct;
                memset(bias_buf, 0, (size_t)num_lanes);
            }
            else
            {
                addend_sel = AddendSel_UsePsum;
                memset(bias_buf, 0, (size_t)num_lanes);
            }

            /* Per-batch compute */
            for (int b_idx = 0; b_idx < batch; b_idx++)
            {
                /* Build activation vector for this k_tile */
                const uint8_t *x_row = x_e4m3 +
                    (size_t)b_idx * (size_t)in_features + k0;
                memcpy(act_buf, x_row, (size_t)tile_width);
                if (tile_width < vec_len)
                    memset(act_buf + tile_width, 0,
                           (size_t)(vec_len - tile_width));

                /* Fetch psum for this batch row */
                uint16_t *psum_row = psum_all + (size_t)b_idx * num_lanes;
                memcpy(psum_buf, psum_row,
                       (size_t)num_lanes * sizeof(uint16_t));

                ComputeReq cr;
                cr.act          = act_buf;
                cr.act_len      = vec_len;
                cr.bias         = bias_buf;
                cr.bias_len     = num_lanes;
                cr.psum         = psum_buf;
                cr.psum_len     = num_lanes;
                cr.scaleExp     = sexp_buf;
                cr.scaleExp_len = num_lanes;
                cr.addendSel    = addend_sel;
                cr.outFmtSel    = out_fmt_sel;

                bool ok = ipt_compute_now(dut, &cr, lane_out);
                assert(ok);

                /* Write lane_out back as next psum */
                memcpy(psum_row, lane_out,
                       (size_t)num_lanes * sizeof(uint16_t));
            }
        } /* k_tile */

        ipt_model_free(dut);

        /* Write final psum (= output) into out_bits */
        for (int b_idx = 0; b_idx < batch; b_idx++)
        {
            uint16_t *psum_row = psum_all + (size_t)b_idx * num_lanes;
            uint16_t *out_row  = out_bits +
                (size_t)b_idx * out_features + out_base;
            memcpy(out_row, psum_row,
                   (size_t)lane_count * sizeof(uint16_t));
        }
    } /* out_base */

    free(psum_all);
    free(act_buf);
    free(bias_buf);
    free(psum_buf);
    free(sexp_buf);
    free(lane_out);
}


#endif /* IPT_LINEAR_H */