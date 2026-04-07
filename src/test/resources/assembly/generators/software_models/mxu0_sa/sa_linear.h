#ifndef SA_LINEAR_H
#define SA_LINEAR_H

/* sa_linear.h -- header-only C tiling loop for the systolic array.
 *
 * Computes y = x @ W^T + b  (same contract as ipt_linear.h / F.linear)
 * using the systolic array model (systolic_array_model.h).
 *
 * Geometry mapping
 * ----------------
 *   SA rows  == the K-tile width (number of input features processed per
 *               sa_compute_now call).  Matches SystolicArrayParams.rows.
 *   SA cols  == the output tile width (output features per call).
 *               Matches SystolicArrayParams.cols.
 *
 * Tiling strategy
 * ---------------
 *   Outer loop : output tiles  -- groups of cols output neurons.
 *   Inner loop : K-tiles       -- groups of rows input features.
 *   Per batch  : sa_compute_now accumulates into psum across K-tiles.
 *
 *   k_tile == 0, bias present  -> SA_UseBias  (addend = bias column)
 *   k_tile == 0, no bias       -> SA_UseZero  (addend = 0)
 *   k_tile >  0                -> SA_UsePsum  (addend = previous output)
 *
 * Memory layout (all row-major, C-contiguous)
 * --------------------------------------------
 *   x_e4m3   : uint8_t  [batch * in_features]
 *   w_e4m3   : uint8_t  [out_features * in_features]
 *   b_e4m3   : uint8_t  [out_features]  or NULL
 *   out_bits : uint16_t [batch * out_features]  -- written by this function
 *
 * Call sa_linear_init_luts() once before first use.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "fp_formats.h"
#include "converters.h"
#include "systolic_array_model.h"

static inline void sa_linear_init_luts(void)
{
    atlas_fp_init_lut();
    atlas_acc_init_lut();
}

/* sa_linear_call
 *
 * Parameters
 * ----------
 *   p            : SA geometry (rows = K-tile width, cols = output tile width)
 *   x_e4m3       : uint8_t  [batch * in_features]
 *   w_e4m3       : uint8_t  [out_features * in_features]
 *   b_e4m3       : uint8_t  [out_features], or NULL
 *   batch        : number of input rows
 *   in_features  : K dimension
 *   out_features : N dimension
 *   scale_exp    : scalar power-of-two exponent for OutputConvStage
 *   out_fmt_sel  : OutBF16 or OutE4M3
 *   out_bits     : uint16_t [batch * out_features]  -- caller allocates
 */
static inline void sa_linear_call(
    const SystolicArrayParams *p,
    const uint8_t *x_e4m3,
    const uint8_t *w_e4m3,
    const uint8_t *b_e4m3,
    int batch,
    int in_features,
    int out_features,
    int scale_exp,
    OutputFmtSel out_fmt_sel,
    uint16_t *out_bits)
{
    const int rows = p->rows; /* K-tile width        */
    const int cols = p->cols; /* output tile width   */
    const int num_k_tiles = (in_features + rows - 1) / rows;

    /* Scratch: activation row (zero-padded to rows), bias/psum for one tile */
    uint8_t *act_buf = (uint8_t *)calloc((size_t)rows, 1);
    uint8_t *bias_buf = (uint8_t *)calloc((size_t)cols, 1);
    uint16_t *psum_buf = (uint16_t *)calloc((size_t)cols, sizeof(uint16_t));
    uint16_t *tile_out = (uint16_t *)malloc((size_t)cols * sizeof(uint16_t));
    assert(act_buf && bias_buf && psum_buf && tile_out);

    /* psum accumulator per (batch_row, output_tile_col):
     * uint16_t [batch * cols]  — reset to zero per output tile */
    uint16_t *psum_all = (uint16_t *)calloc((size_t)(batch * cols),
                                            sizeof(uint16_t));
    assert(psum_all);

    /* -----------------------------------------------------------------------
     * Outer loop: output tiles  (cols output neurons each)
     * --------------------------------------------------------------------- */
    for (int out_base = 0; out_base < out_features; out_base += cols)
    {
        int col_count = out_features - out_base;
        if (col_count > cols)
            col_count = cols;

        /* Reset psum for this output tile */
        memset(psum_all, 0, (size_t)(batch * cols) * sizeof(uint16_t));

        /* Fresh SA model for this output tile (weight double-buffer buf 0) */
        SAModel *dut = sa_model_init(p);

        /* Load weights for all cols in this output tile.
         * w_e4m3 layout: row = output neuron, col = input feature.
         * SA weight buf layout: wbuf[0][col_j][row_i] = weight for
         *   output neuron (out_base + col_j), input feature (k_tile*rows + row_i).
         * We pre-load one K-tile's weights per SA col, then swap K-tiles.
         * Since the SA reuses the same weight buffer within a K-tile loop,
         * we load all cols' weights for each K-tile before computing. */

        /* -----------------------------------------------------------------------
         * Inner loop: K-tiles  (rows input features each)
         * --------------------------------------------------------------------- */
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++)
        {
            int k0 = k_tile * rows;
            int k1 = k0 + rows;
            if (k1 > in_features)
                k1 = in_features;
            int tile_width = k1 - k0; /* may be < rows on last tile */

            /* Load weights for this K-tile into buf 0, one col at a time.
             * wbuf[0][col_j][row_i] = w_e4m3[(out_base+col_j)*in_features + k0+row_i]
             */
            uint8_t *w_col = (uint8_t *)malloc((size_t)rows);
            assert(w_col);
            for (int cj = 0; cj < cols; cj++)
            {
                if (cj < col_count)
                {
                    const uint8_t *src = w_e4m3 +
                                         (size_t)(out_base + cj) * (size_t)in_features + k0;
                    memcpy(w_col, src, (size_t)tile_width);
                    if (tile_width < rows)
                        memset(w_col + tile_width, 0, (size_t)(rows - tile_width));
                }
                else
                {
                    memset(w_col, 0, (size_t)rows);
                }

                SA_WeightLoadReq wr;
                wr.weights = w_col;
                wr.weights_len = rows;
                wr.weight_buf_write_sel = 0;
                wr.col_idx = cj;
                sa_load_weights(dut, &wr);
            }
            free(w_col);

            /* Choose addend source */
            SA_AddendSel addend_sel;
            if (k_tile == 0 && b_e4m3 != NULL)
            {
                addend_sel = SA_UseBias;
                memset(bias_buf, 0, (size_t)cols);
                for (int cj = 0; cj < col_count; cj++)
                    bias_buf[cj] = b_e4m3[out_base + cj];
            }
            else if (k_tile == 0)
            {
                addend_sel = SA_UseZero;
                memset(bias_buf, 0, (size_t)cols);
            }
            else
            {
                addend_sel = SA_UsePsum;
                memset(bias_buf, 0, (size_t)cols);
            }

            /* Per-batch compute */
            for (int b_idx = 0; b_idx < batch; b_idx++)
            {
                /* Build activation row for this K-tile (zero-pad if needed) */
                const uint8_t *x_row = x_e4m3 +
                                       (size_t)b_idx * (size_t)in_features + k0;
                memcpy(act_buf, x_row, (size_t)tile_width);
                if (tile_width < rows)
                    memset(act_buf + tile_width, 0, (size_t)(rows - tile_width));

                /* Fetch psum for this batch row */
                uint16_t *psum_row = psum_all + (size_t)b_idx * cols;
                memcpy(psum_buf, psum_row, (size_t)cols * sizeof(uint16_t));

                SA_ComputeReq cr;
                cr.activation_row = act_buf;
                cr.bias = bias_buf;
                cr.psum = psum_buf;
                cr.addend_sel = addend_sel;
                cr.weight_buf_read_sel = false; /* always buf 0 */
                cr.scale_exp = scale_exp;
                cr.out_fmt_sel = out_fmt_sel;

                sa_compute_now(dut, &cr, tile_out);

                /* Write result back as psum for next K-tile */
                memcpy(psum_row, tile_out, (size_t)cols * sizeof(uint16_t));
            }
        } /* k_tile */

        sa_model_free(dut);

        /* Copy final psum (= output) into out_bits */
        for (int b_idx = 0; b_idx < batch; b_idx++)
        {
            uint16_t *psum_row = psum_all + (size_t)b_idx * cols;
            uint16_t *out_row = out_bits + (size_t)b_idx * out_features + out_base;
            memcpy(out_row, psum_row, (size_t)col_count * sizeof(uint16_t));
        }
    } /* out_base */

    free(psum_all);
    free(act_buf);
    free(bias_buf);
    free(psum_buf);
    free(tile_out);
}

#endif /* SA_LINEAR_H */