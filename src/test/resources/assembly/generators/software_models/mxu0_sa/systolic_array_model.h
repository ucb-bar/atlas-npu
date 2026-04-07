#ifndef SYSTOLIC_ARRAY_MODEL_H
#define SYSTOLIC_ARRAY_MODEL_H

/* systolic_array_model.h
 * ----------------------
 * Header-only C model of SystolicArray with useE4M3FMA=True.
 *
 * Mirrors systolic_array_model.py (Python golden model) exactly.
 *
 * Architecture (useE4M3FMA path):
 *   - inT  = E4M3  (activations, weights, bias)
 *   - outT = BF16  (psum, macQ, output)
 *   - PE:  E4M3Mul (13-bit E5M7 product) -> E4M3ProdAddBF16 -> BF16
 *   - Activations flow right across each row
 *   - Addend (bias/psum/zero) flows down from row 0
 *   - Output of last row -> OutputConvStage -> outputRow
 *
 * Memory ownership
 * ----------------
 *   sa_model_init()  allocates; sa_model_free()  releases.
 *   sa_model_step()  returns a malloc'd uint16_t[cols] when output is valid;
 *                    caller must free() it.
 *
 * Dependencies: fp_formats.h, converters.h
 *   (converters.h must have atlas_acc_init_lut() called before use)
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "fp_formats.h"
#include "converters.h" /* e4m3_mul_to_prod, output_conv_stage */

/* =========================================================================
 * SystolicArrayParams
 * ========================================================================= */

typedef struct
{
    int rows; /* default 32 */
    int cols; /* default 16 */
} SystolicArrayParams;

static inline SystolicArrayParams SA_default_params(void)
{
    SystolicArrayParams p = {.rows = 32, .cols = 32};
    return p;
}

/* =========================================================================
 * Enums
 * ========================================================================= */

typedef enum
{
    SA_UseBias = 0,
    SA_UsePsum = 1,
    SA_UseZero = 2
} SA_AddendSel;

/* =========================================================================
 * E4M3 -> BF16 conversion
 * (used for bias: RTL converts inT E4M3 -> outT BF16 before the PE addend)
 * ========================================================================= */

static inline uint16_t sa_e4m3_to_bf16(uint8_t byte)
{
    int sign = (byte >> 7) & 1;
    int exp4 = (byte >> 3) & 0xF;
    int mant3 = byte & 0x7;

    if (exp4 == 0)
        return (uint16_t)(sign << 15); /* subnormal/zero -> BF16 zero */

    int bf16_exp = exp4 + 120; /* exp4 - 7 + 127 */
    int bf16_mant = (mant3 << 4) & 0x7F;
    return (uint16_t)((sign << 15) | (bf16_exp << 7) | bf16_mant);
}

/* =========================================================================
 * E4M3ProdAddBF16
 *
 * Translates E4M3ProdAddBF16.scala exactly.
 *   prod13  : 13-bit E5M7 (S1 E5 M7, bias=13)
 *   addend16: BF16 IEEE
 *   returns : BF16 IEEE (16-bit)
 * ========================================================================= */

#define _SA_SIG8 8
#define _SA_GRS 3
#define _SA_SIG11 11
#define _SA_EXP_BIAS_OFFSET 114 /* BF16_bias(127) - E5M7_bias(13) = 114 */

static inline uint16_t sa_e4m3_prod_add_bf16(uint16_t prod13, uint16_t addend16)
{
    prod13 &= 0x1FFF;
    addend16 &= 0xFFFF;

    /* 1. Unpack product */
    int p_sign = (prod13 >> 12) & 1;
    int p_exp5 = (prod13 >> 7) & 0x1F;
    int p_man7 = prod13 & 0x7F;
    int p_zero = (p_exp5 == 0);
    int p_sig8 = p_zero ? 0 : (0x80 | p_man7);
    int p_exp8 = (p_exp5 + _SA_EXP_BIAS_OFFSET) & 0xFF;

    /* 2. Unpack BF16 addend */
    int a_sign = (addend16 >> 15) & 1;
    int a_exp8 = (addend16 >> 7) & 0xFF;
    int a_man7 = addend16 & 0x7F;
    int a_zero = (a_exp8 == 0);
    int a_sig8 = a_zero ? 0 : (0x80 | a_man7);

    /* 3. Zero fast paths */
    if (p_zero)
        return addend16;
    if (a_zero)
        return (uint16_t)(((p_sign << 15) | (p_exp8 << 7) | p_man7) & 0xFFFF);

    /* 4. Extend to 11 bits */
    int p_sig11 = p_sig8 << _SA_GRS;
    int a_sig11 = a_sig8 << _SA_GRS;

    /* 5. Compare and swap */
    int p_gt_a = (p_exp8 > a_exp8) || ((p_exp8 == a_exp8) && (p_sig11 >= a_sig11));
    int e_max = p_gt_a ? p_exp8 : a_exp8;
    int e_min = p_gt_a ? a_exp8 : p_exp8;
    int s_max = p_gt_a ? p_sign : a_sign;
    int s_min = p_gt_a ? a_sign : p_sign;
    int sig_max = p_gt_a ? p_sig11 : a_sig11;
    int sig_min = p_gt_a ? a_sig11 : p_sig11;

    int exp_diff = e_max - e_min;
    int shift_amt = (exp_diff > 11) ? 11 : exp_diff;

    /* 6. Shift-right-jam */
    int shifted_out = (sig_min >> shift_amt) & 0x7FF;
    int shift_mask = (1 << shift_amt) - 1;
    int sticky = (sig_min & shift_mask) ? 1 : 0;
    int sig_min_aligned = shifted_out | sticky;

    /* 7. Add or subtract (12-bit) */
    int same_sign = (s_max == s_min);
    int sig_sum = same_sign ? (sig_max + sig_min_aligned)
                            : (sig_max - sig_min_aligned);
    sig_sum &= 0xFFF;

    int cancel_zero = (sig_sum == 0);

    /* 8. Normalize — addition carry */
    int carry_out = (sig_sum >> _SA_SIG11) & 1;
    int sig_after_add_norm = carry_out ? (sig_sum >> 1) : (sig_sum & 0x7FF);
    int sticky_after_add_norm = carry_out ? (sticky | (sig_sum & 1)) : sticky;
    int e_after_add_norm = carry_out ? (e_max + 1) : e_max;

    /* 9. Normalize — subtraction leading zeros
     * lzc = number of leading zeros in sig_after_add_norm (SIG11 bits) */
    int lzc = _SA_SIG11;
    if (sig_after_add_norm > 0)
    {
        /* bit_length - 1 = index of highest set bit */
        int tmp = sig_after_add_norm;
        int bl = 0;
        while (tmp)
        {
            bl++;
            tmp >>= 1;
        }
        lzc = _SA_SIG11 - bl;
    }

    int need_left_shift = (!carry_out) && (!((sig_after_add_norm >> (_SA_SIG11 - 1)) & 1));
    int left_shift_amt = need_left_shift
                             ? (lzc < e_after_add_norm ? lzc : e_after_add_norm)
                             : 0;
    int sig_norm = need_left_shift
                       ? ((sig_after_add_norm << left_shift_amt) & 0x7FF)
                       : sig_after_add_norm;
    int e_norm = e_after_add_norm - left_shift_amt;

    /* 10. Round-to-nearest-even */
    int G = (sig_norm >> 2) & 1;
    int R = (sig_norm >> 1) & 1;
    int S = (sig_norm & 1) | sticky_after_add_norm;
    int mant_lsb = (sig_norm >> 3) & 1;
    int mantissa7 = (sig_norm >> 3) & 0x7F;
    int round_up = G && (R || S || mant_lsb);
    int mantissa_rounded = mantissa7 + round_up;
    int round_overflow = (mantissa_rounded >> 7) & 1;
    int mantissa_out = round_overflow ? 0 : (mantissa_rounded & 0x7F);
    int e_rounded = round_overflow ? (e_norm + 1) : e_norm;

    /* 11. Overflow / underflow clamp */
    int underflow = (e_norm <= 0);
    int overflow = (e_rounded > 254);
    int e_out, m_out;
    if (overflow)
    {
        e_out = 254;
        m_out = 0x7F;
    }
    else if (underflow)
    {
        e_out = 0;
        m_out = 0;
    }
    else
    {
        e_out = e_rounded & 0xFF;
        m_out = mantissa_out & 0x7F;
    }

    /* 12. Output mux */
    if (cancel_zero)
        return 0;
    return (uint16_t)(((s_max << 15) | (e_out << 7) | m_out) & 0xFFFF);
}

/* =========================================================================
 * PE compute  (E4M3FMA path)
 *   act, weight : raw E4M3 bytes
 *   addend      : BF16 IEEE bits
 *   returns     : BF16 IEEE bits
 * ========================================================================= */

static inline uint16_t sa_pe_compute(uint8_t act, uint8_t weight, uint16_t addend)
{
    uint16_t prod13 = e4m3_mul_to_prod(act, weight);
    return sa_e4m3_prod_add_bf16(prod13, addend);
}

/* =========================================================================
 * Output queue (singly-linked)
 * ========================================================================= */

typedef struct _SAOutNode
{
    bool valid;
    uint16_t *data; /* malloc'd [cols], or NULL */
    struct _SAOutNode *next;
} SAOutNode;

typedef struct
{
    SAOutNode *head;
    SAOutNode *tail;
    int size;
} SAOutQueue;

static inline void sa_outq_init(SAOutQueue *q, int latency)
{
    q->head = q->tail = NULL;
    q->size = 0;
    for (int i = 0; i < latency; i++)
    {
        SAOutNode *n = (SAOutNode *)calloc(1, sizeof *n);
        assert(n);
        n->valid = false;
        n->data = NULL;
        n->next = NULL;
        if (q->tail)
            q->tail->next = n;
        else
            q->head = n;
        q->tail = n;
        q->size++;
    }
}

static inline void sa_outq_append(SAOutQueue *q, bool valid, uint16_t *data)
{
    SAOutNode *n = (SAOutNode *)calloc(1, sizeof *n);
    assert(n);
    n->valid = valid;
    n->data = data;
    n->next = NULL;
    if (q->tail)
        q->tail->next = n;
    else
        q->head = n;
    q->tail = n;
    q->size++;
}

static inline bool sa_outq_popleft(SAOutQueue *q, uint16_t **data_out)
{
    assert(q->head);
    SAOutNode *n = q->head;
    q->head = n->next;
    if (!q->head)
        q->tail = NULL;
    q->size--;
    bool valid = n->valid;
    *data_out = n->data;
    free(n);
    return valid;
}

static inline void sa_outq_free(SAOutQueue *q)
{
    while (q->head)
    {
        uint16_t *d = NULL;
        sa_outq_popleft(q, &d);
        free(d);
    }
}

/* =========================================================================
 * SystolicArrayModel
 * ========================================================================= */

typedef struct
{
    SystolicArrayParams p;

    /* wbuf[buf][col][row] -> E4M3 byte */
    uint8_t ***wbuf; /* [2][cols][rows] */

    SAOutQueue out_queue;
} SAModel;

static inline uint8_t ***sa_alloc_wbuf(int cols, int rows)
{
    uint8_t ***buf = (uint8_t ***)calloc(2, sizeof(uint8_t **));
    assert(buf);
    for (int b = 0; b < 2; b++)
    {
        buf[b] = (uint8_t **)calloc((size_t)cols, sizeof(uint8_t *));
        assert(buf[b]);
        for (int c = 0; c < cols; c++)
        {
            buf[b][c] = (uint8_t *)calloc((size_t)rows, 1);
            assert(buf[b][c]);
        }
    }
    return buf;
}

static inline void sa_free_wbuf(uint8_t ***buf, int cols)
{
    if (!buf)
        return;
    for (int b = 0; b < 2; b++)
    {
        for (int c = 0; c < cols; c++)
            free(buf[b][c]);
        free(buf[b]);
    }
    free(buf);
}

static inline SAModel *sa_model_init(const SystolicArrayParams *p_in)
{
    SAModel *m = (SAModel *)calloc(1, sizeof *m);
    assert(m);
    m->p = p_in ? *p_in : SA_default_params();
    m->wbuf = sa_alloc_wbuf(m->p.cols, m->p.rows);
    int latency = m->p.rows + m->p.cols - 1;
    sa_outq_init(&m->out_queue, latency);
    return m;
}

static inline void sa_model_free(SAModel *m)
{
    if (!m)
        return;
    sa_free_wbuf(m->wbuf, m->p.cols);
    sa_outq_free(&m->out_queue);
    free(m);
}

static inline void sa_model_reset(SAModel *m)
{
    int cols = m->p.cols;
    int rows = m->p.rows;
    int latency = rows + cols - 1;
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < cols; c++)
            memset(m->wbuf[b][c], 0, (size_t)rows);
    sa_outq_free(&m->out_queue);
    sa_outq_init(&m->out_queue, latency);
}

/* -------------------------------------------------------------------------
 * Weight load
 * ------------------------------------------------------------------------- */

typedef struct
{
    const uint8_t *weights; /* rows E4M3 bytes */
    int weights_len;
    int weight_buf_write_sel; /* 0 or 1 */
    int col_idx;
} SA_WeightLoadReq;

static inline bool sa_load_weights(SAModel *m, const SA_WeightLoadReq *req)
{
    if (req->weights_len != m->p.rows)
        return false;
    if (req->col_idx < 0 || req->col_idx >= m->p.cols)
        return false;
    int buf = req->weight_buf_write_sel & 1;
    memcpy(m->wbuf[buf][req->col_idx], req->weights, (size_t)m->p.rows);
    return true;
}

/* -------------------------------------------------------------------------
 * Compute request
 * ------------------------------------------------------------------------- */

typedef struct
{
    const uint8_t *activation_row; /* rows E4M3 bytes  */
    const uint8_t *bias;           /* cols E4M3 bytes  */
    const uint16_t *psum;          /* cols BF16 bits   */
    SA_AddendSel addend_sel;
    bool weight_buf_read_sel;
    int scale_exp;
    OutputFmtSel out_fmt_sel;
} SA_ComputeReq;

/* -------------------------------------------------------------------------
 * Core: run the PE grid combinationally and return cols BF16/E4M3 values.
 * out_data must be caller-allocated uint16_t[cols].
 * ------------------------------------------------------------------------- */

static inline void sa_compute_now(const SAModel *m,
                                  const SA_ComputeReq *req,
                                  uint16_t *out_data)
{
    const SystolicArrayParams *p = &m->p;
    int rows = p->rows;
    int cols = p->cols;

    uint8_t **wbuf_sel = m->wbuf[req->weight_buf_read_sel ? 1 : 0];

    /* Build per-column addends for row 0 */
    uint16_t *addend = (uint16_t *)malloc((size_t)cols * sizeof(uint16_t));
    assert(addend);
    for (int j = 0; j < cols; j++)
    {
        switch (req->addend_sel)
        {
        case SA_UseBias:
            addend[j] = sa_e4m3_to_bf16(req->bias[j]);
            break;
        case SA_UsePsum:
            addend[j] = req->psum[j];
            break;
        default:
            addend[j] = 0;
            break;
        }
    }

    /* Propagate through PE grid */
    uint16_t *next_addend = (uint16_t *)malloc((size_t)cols * sizeof(uint16_t));
    assert(next_addend);

    for (int i = 0; i < rows; i++)
    {
        uint8_t act = req->activation_row[i] & 0xFF;
        for (int j = 0; j < cols; j++)
        {
            uint8_t weight = wbuf_sel[j][i];
            next_addend[j] = sa_pe_compute(act, weight, addend[j]);
        }
        /* swap buffers */
        uint16_t *tmp = addend;
        addend = next_addend;
        next_addend = tmp;
    }

    /* addend now holds macQ of last row; apply OutputConvStage */
    for (int j = 0; j < cols; j++)
    {
        uint32_t conv = output_conv_stage(addend[j], req->out_fmt_sel, req->scale_exp);
        out_data[j] = (uint16_t)(conv & 0xFFFF);
    }

    free(addend);
    free(next_addend);
}

/* -------------------------------------------------------------------------
 * Step: advance one clock cycle.
 *
 * Returns SA_StepResult.  If out_valid is true, out_bits is malloc'd
 * uint16_t[cols]; caller must free() it.
 * ------------------------------------------------------------------------- */

typedef struct
{
    bool out_valid;
    uint16_t *out_bits; /* [cols], or NULL */
} SA_StepResult;

static inline SA_StepResult sa_model_step(SAModel *m,
                                          const SA_ComputeReq *compute_req,
                                          const SA_WeightLoadReq *weight_load_req)
{
    if (weight_load_req != NULL)
        sa_load_weights(m, weight_load_req);

    uint16_t *produced = NULL;
    if (compute_req != NULL)
    {
        produced = (uint16_t *)malloc((size_t)m->p.cols * sizeof(uint16_t));
        assert(produced);
        sa_compute_now(m, compute_req, produced);
    }

    sa_outq_append(&m->out_queue, produced != NULL, produced);

    uint16_t *popped = NULL;
    bool valid = sa_outq_popleft(&m->out_queue, &popped);

    SA_StepResult r;
    r.out_valid = valid;
    r.out_bits = valid ? popped : NULL;
    if (!valid)
        free(popped);
    return r;
}

#endif /* SYSTOLIC_ARRAY_MODEL_H */