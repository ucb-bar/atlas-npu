#ifndef INNER_PRODUCT_TREES_MODEL_H
#define INNER_PRODUCT_TREES_MODEL_H

/* inner_product_trees_model.h — C translation of the inner-product tree model
 *   (AnchorAccumulationTreeModel + InnerProductTreesModel)
 *
 * Depends on: fp_formats.h, converters.h, params_and_requests.h
 *
 * Memory ownership
 * ----------------
 *  - Call ipt_model_init()  to allocate; call ipt_model_free() when done.
 *  - StepResult.out_bits is malloc'd by ipt_model_step(); caller must free().
 *  - ComputeReq / WeightLoadReq arrays are borrowed (not copied) for the
 *    duration of the call only.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "fp_formats.h"
#include "converters.h"
#include "params_and_requests.h"


/* ---------------------------------------------------------------------------
 * Output-queue node (singly-linked, intrusive)
 * Mirrors Python's collections.deque([None] * latency)
 * ------------------------------------------------------------------------- */

typedef struct OutQueueNode {
    bool      valid;        /* false == Python's None */
    uint16_t *data;         /* malloc'd array of length numLanes, or NULL */
    struct OutQueueNode *next;
} OutQueueNode;

typedef struct {
    OutQueueNode *head;     /* popleft end  */
    OutQueueNode *tail;     /* append end   */
    int           size;
} OutQueue;

static inline void outq_init(OutQueue *q, int latency) {
    q->head = q->tail = NULL;
    q->size = 0;
    for (int i = 0; i < latency; i++) {
        OutQueueNode *n = (OutQueueNode *)calloc(1, sizeof *n);
        assert(n);
        n->valid = false;
        n->data  = NULL;
        n->next  = NULL;
        if (q->tail) q->tail->next = n;
        else         q->head       = n;
        q->tail = n;
        q->size++;
    }
}

/* Append a slot to the tail.  data may be NULL (valid=false). */
static inline void outq_append(OutQueue *q, bool valid, uint16_t *data) {
    OutQueueNode *n = (OutQueueNode *)calloc(1, sizeof *n);
    assert(n);
    n->valid = valid;
    n->data  = data;
    n->next  = NULL;
    if (q->tail) q->tail->next = n;
    else         q->head       = n;
    q->tail = n;
    q->size++;
}

/* Pop from the head; caller owns *data_out (free it when done). */
static inline bool outq_popleft(OutQueue *q, uint16_t **data_out) {
    assert(q->head);
    OutQueueNode *n = q->head;
    q->head = n->next;
    if (!q->head) q->tail = NULL;
    q->size--;
    bool valid = n->valid;
    *data_out  = n->data;
    free(n);
    return valid;
}

static inline void outq_free(OutQueue *q) {
    while (q->head) {
        uint16_t *d = NULL;
        outq_popleft(q, &d);
        free(d);
    }
}


/* ---------------------------------------------------------------------------
 * AnchorAccumulationTreeModel  (per-lane helper — no heap allocation)
 * ------------------------------------------------------------------------- */

typedef struct {
    /* cached constants, all derived from InnerProductTreeParams */
    int sentinel;
    int anchor_headroom;
    int int_width;

    /* bias format cache */
    int bias_mant_bits;
    int bias_exp_mask;
    int bias_zero_mask;
    int bias_bias;

    /* psum format cache */
    int psum_mant_bits;
    int psum_exp_mask;
    int psum_frac_mask;
    int psum_bias;
} AnchorAccumTree;

static inline void aat_init(AnchorAccumTree *t, const InnerProductTreeParams *p) {
    const AtlasFPType *bfmt = IPT_biasFmt();
    const AtlasFPType *pfmt = IPT_psumFmt();

    t->sentinel        = -(1 << (IPT_expWorkWidth() - 2));
    t->anchor_headroom = IPT_anchorHeadroom(p);
    t->int_width       = IPT_intWidth(p);

    t->bias_mant_bits  = bfmt->mantissaBits;
    t->bias_exp_mask   = (1 << bfmt->expWidth) - 1;
    t->bias_zero_mask  = (1 << (bfmt->expWidth + 1)) - 1;
    t->bias_bias       = bfmt->ieeeBias;

    t->psum_mant_bits  = pfmt->mantissaBits;
    t->psum_exp_mask   = (1 << pfmt->expWidth) - 1;
    t->psum_frac_mask  = (1 << pfmt->mantissaBits) - 1;
    t->psum_bias       = pfmt->ieeeBias;
}

/* Returns the unbiased exponent of a product word, or sentinel for zero. */
static inline int aat_prod_unb_exp(const AnchorAccumTree *t, uint16_t prod) {
    int exp_bits = (prod >> 7) & 0x1F;
    return (exp_bits == 0) ? t->sentinel : (exp_bits - _E4M3_PROD_BIAS);
}

/* compute_lane: act[] and weights[] are vecLen long (vecLen from params). */
static inline uint32_t aat_compute_lane(
        const AnchorAccumTree *t,
        const InnerProductTreeParams *p,
        const uint8_t  *act,
        const uint8_t  *weights,    /* already selected buf */
        uint8_t         bias,
        uint16_t        psum,
        int             scale_exp,
        AddendSel       addend_sel,
        OutputFmtSel    out_fmt_sel)
{
    int vec_len   = p->vecLen;
    int sentinel  = t->sentinel;
    int int_width = t->int_width;

    /* --- Stage 0: multiply, track max product exponent --- */
    /* Stack-allocate up to 256 products; use heap for larger vecLen. */
    uint16_t  prod_stack[256];
    uint16_t *prod_s0 = (vec_len <= 256) ? prod_stack
                       : (uint16_t *)malloc((size_t)vec_len * sizeof(uint16_t));
    assert(prod_s0);

    int max_prod_exp = sentinel;
    for (int i = 0; i < vec_len; i++) {
        uint16_t prod = e4m3_mul_to_prod(act[i], weights[i]);
        prod_s0[i] = prod;
        int pe = aat_prod_unb_exp(t, prod);
        if (pe > max_prod_exp) max_prod_exp = pe;
    }

    /* --- Addend exponent --- */
    int addend_exp = sentinel;

    if (addend_sel == AddendSel_UseBias) {
        int bef = (bias >> t->bias_mant_bits) & t->bias_exp_mask;
        int bz  = (bias >> t->bias_mant_bits) & t->bias_zero_mask;
        if (bz != 0)
            addend_exp = bef - t->bias_bias;
    } else if (addend_sel == AddendSel_UsePsum) {
        int pef  = (psum >> t->psum_mant_bits) & t->psum_exp_mask;
        int pfrc = psum & t->psum_frac_mask;
        if (!(pef == 0 && pfrc == 0))
            addend_exp = pef - t->psum_bias;
    }

    /* --- Anchor --- */
    int max_exp = (max_prod_exp >= addend_exp) ? max_prod_exp : addend_exp;
    int anchor  = max_exp + t->anchor_headroom;

    /* --- Accumulate products --- */
    int32_t prod_sum = 0;
    for (int i = 0; i < vec_len; i++)
        prod_sum = wrap_signed(prod_sum +
                   e4m3_prod_to_aligned_int(prod_s0[i], anchor, int_width),
                   int_width);

    if (prod_s0 != prod_stack) free(prod_s0);

    /* --- Addend --- */
    int32_t addend_int = 0;
    if (addend_sel == AddendSel_UseBias)
        addend_int = ieee_to_aligned_int(bias, IPT_biasFmt(), anchor, int_width);
    else if (addend_sel == AddendSel_UsePsum)
        addend_int = ieee_to_aligned_int(psum, IPT_psumFmt(), anchor, int_width);

    /* --- Final sum → BF16 → output format --- */
    int32_t  total     = wrap_signed(prod_sum + addend_int, int_width);
    uint16_t bf16_res  = aligned_int_to_bf16(total, anchor, int_width);
    return output_conv_stage(bf16_res, out_fmt_sel, scale_exp);
}


/* ---------------------------------------------------------------------------
 * InnerProductTreesModel
 * ------------------------------------------------------------------------- */

typedef struct {
    InnerProductTreeParams p;

    bool      wEn;
    uint8_t **wbuf0;    /* [numLanes][vecLen] */
    uint8_t **wbuf1;    /* [numLanes][vecLen] */

    AnchorAccumTree *lanes; /* [numLanes] */

    OutQueue  out_queue;
} IPTModel;


/* Allocate wbuf0 / wbuf1 (internal helper) */
static inline uint8_t **_alloc_wbuf(int num_lanes, int vec_len) {
    uint8_t **buf = (uint8_t **)calloc((size_t)num_lanes, sizeof(uint8_t *));
    assert(buf);
    for (int i = 0; i < num_lanes; i++) {
        buf[i] = (uint8_t *)calloc((size_t)vec_len, 1);
        assert(buf[i]);
    }
    return buf;
}

static inline void _free_wbuf(uint8_t **buf, int num_lanes) {
    if (!buf) return;
    for (int i = 0; i < num_lanes; i++) free(buf[i]);
    free(buf);
}

/* ipt_model_init — allocate and initialise the model */
static inline IPTModel *ipt_model_init(const InnerProductTreeParams *p_in) {
    IPTModel *m = (IPTModel *)calloc(1, sizeof *m);
    assert(m);

    InnerProductTreeParams p = p_in ? *p_in : InnerProductTreeParams_default();
    assert(InnerProductTreeParams_valid(&p));
    m->p   = p;
    m->wEn = false;

    m->wbuf0 = _alloc_wbuf(p.numLanes, p.vecLen);
    m->wbuf1 = _alloc_wbuf(p.numLanes, p.vecLen);

    m->lanes = (AnchorAccumTree *)malloc((size_t)p.numLanes * sizeof(AnchorAccumTree));
    assert(m->lanes);
    for (int i = 0; i < p.numLanes; i++)
        aat_init(&m->lanes[i], &p);

    outq_init(&m->out_queue, IPT_latency(&p));
    return m;
}

/* ipt_model_free */
static inline void ipt_model_free(IPTModel *m) {
    if (!m) return;
    _free_wbuf(m->wbuf0, m->p.numLanes);
    _free_wbuf(m->wbuf1, m->p.numLanes);
    free(m->lanes);
    outq_free(&m->out_queue);
    free(m);
}

/* buf_read_sel property */
static inline bool ipt_buf_read_sel(const IPTModel *m) {
    return !m->wEn;
}

/* ipt_model_reset */
static inline void ipt_model_reset(IPTModel *m) {
    const InnerProductTreeParams *p = &m->p;
    m->wEn = false;

    for (int i = 0; i < p->numLanes; i++) {
        memset(m->wbuf0[i], 0, (size_t)p->vecLen);
        memset(m->wbuf1[i], 0, (size_t)p->vecLen);
    }

    outq_free(&m->out_queue);
    outq_init(&m->out_queue, IPT_latency(p));
}

/* ipt_load_weights — returns false on validation error */
static inline bool ipt_load_weights(IPTModel *m, const WeightLoadReq *req) {
    const InnerProductTreeParams *p = &m->p;

    if (req->weightsDma_len != p->vecLen)  return false;
    if (req->laneIdx < 0 || req->laneIdx >= p->numLanes) return false;

    uint8_t *row = m->wEn ? m->wbuf1[req->laneIdx]
                          : m->wbuf0[req->laneIdx];
    for (int i = 0; i < p->vecLen; i++)
        row[i] = req->weightsDma[i] & 0xFF;

    if (req->last)
        m->wEn = !m->wEn;

    return true;
}

/* ipt_compute_now — fills out_data[numLanes]; caller allocates array.
 * Returns false on validation error. */
static inline bool ipt_compute_now(IPTModel *m,
                                   const ComputeReq *req,
                                   uint16_t *out_data)
{
    const InnerProductTreeParams *p = &m->p;

    if (req->act_len     != p->vecLen)   return false;
    if (req->bias_len    != p->numLanes) return false;
    if (req->psum_len    != p->numLanes) return false;
    if (req->scaleExp_len!= p->numLanes) return false;

    bool          buf_sel    = ipt_buf_read_sel(m);
    AddendSel     addend_sel = req->addendSel;
    OutputFmtSel  out_fmt    = req->outFmtSel;

    for (int li = 0; li < p->numLanes; li++) {
        uint8_t *weights = buf_sel ? m->wbuf1[li] : m->wbuf0[li];
        uint32_t result  = aat_compute_lane(
            &m->lanes[li], p,
            req->act,
            weights,
            req->bias[li] & 0xFF,
            req->psum[li] & 0xFFFF,
            req->scaleExp[li],
            addend_sel,
            out_fmt
        );
        out_data[li] = (uint16_t)(result & 0xFFFF);
    }
    return true;
}

/* ipt_model_step — main clock-edge function.
 *
 * Either/both of compute_req and weight_load_req may be NULL.
 * Returns a StepResult; if out_valid is true, out_bits is malloc'd and the
 * caller must free() it.
 */
static inline StepResult ipt_model_step(IPTModel *m,
                                        const ComputeReq    *compute_req,
                                        const WeightLoadReq *weight_load_req)
{
    if (weight_load_req != NULL)
        ipt_load_weights(m, weight_load_req);

    uint16_t *produced = NULL;
    if (compute_req != NULL) {
        produced = (uint16_t *)malloc((size_t)m->p.numLanes * sizeof(uint16_t));
        assert(produced);
        if (!ipt_compute_now(m, compute_req, produced)) {
            free(produced);
            produced = NULL;
        }
    }

    outq_append(&m->out_queue, produced != NULL, produced);

    uint16_t *popped = NULL;
    bool valid = outq_popleft(&m->out_queue, &popped);

    StepResult r;
    r.out_valid = valid;
    r.out_bits  = valid ? popped : NULL;
    if (!valid) free(popped);   /* popped is NULL anyway, but be explicit */
    return r;
}


#endif