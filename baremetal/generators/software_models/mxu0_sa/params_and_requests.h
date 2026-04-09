#ifndef PARAMS_AND_REQUESTS_H
#define PARAMS_AND_REQUESTS_H

/* params_and_requests.h — C translation of params_and_requests.py
 *
 * Depends on fp_formats.h, converters.h for AtlasFPType, AddendSel, OutputFmtSel, etc.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <assert.h>
#include "fp_formats.h"
#include "converters.h"


/* ---------------------------------------------------------------------------
 * _PIPELINE_DEPTH_TO_CUTS
 *   Python uses frozenset; in C we represent a cut-set as a uint8_t bitmask
 *   where bit i means cut i is present.  Depth index is 1-based (depth 1..5).
 * ------------------------------------------------------------------------- */

static const uint8_t _PIPELINE_DEPTH_TO_CUTS[6] = {
    0,              /* unused (depth 0) */
    0x00,           /* depth 1: {}           */
    0x02,           /* depth 2: {1}          */
    0x05,           /* depth 3: {0,2}        */
    0x07,           /* depth 4: {0,1,2}      */
    0x0F,           /* depth 5: {0,1,2,3}    */
};

/* Popcount helper (number of set bits = number of pipe cuts) */
static inline int _popcount8(uint8_t x) {
    int n = 0;
    while (x) { n += x & 1; x >>= 1; }
    return n;
}

/* log2 ceiling helper: returns number of bits needed to represent x */
static inline int _bit_length(unsigned int x) {
    if (x == 0) return 0;
    int n = 0;
    while (x) { n++; x >>= 1; }
    return n;
}

/* Maximum exponent width across all formats (mirrors _MAX_EXP_WIDTH) */
#define _MAX_EXP_WIDTH_VAL  8   /* max(E4M3.expWidth=4, E4M3ProdFmt.expWidth=5,
                                        E4M3.expWidth=4, BF16.expWidth=8,
                                        BF16.expWidth=8) = 8 */

/* ---------------------------------------------------------------------------
 * InnerProductTreeParams
 * ------------------------------------------------------------------------- */

typedef struct {
    int     numLanes;       /* default 16  */
    int     vecLen;         /* default 32  */
    int     accumIntWidth;  /* default 0   */
    uint8_t pipelineCuts;   /* bitmask of cuts in {0..3} */
} InnerProductTreeParams;

/* Default initialiser — mirrors InnerProductTreeParams() in Python */
static inline InnerProductTreeParams InnerProductTreeParams_default(void) {
    InnerProductTreeParams p;
    p.numLanes      = 16;
    p.vecLen        = 32;
    p.accumIntWidth = 0;
    p.pipelineCuts  = 0x00;   /* empty set */
    return p;
}

/* Validate pipelineCuts: each set bit must be in positions 0..3 */
static inline bool InnerProductTreeParams_valid(const InnerProductTreeParams *p) {
    return (p->pipelineCuts & ~0x0Fu) == 0;
}

/* Properties */
static inline const AtlasFPType *IPT_inputFmt (void) { return &E4M3; }
static inline const AtlasFPType *IPT_biasFmt  (void) { return &E4M3; }
static inline const AtlasFPType *IPT_psumFmt  (void) { return &BF16; }
static inline const AtlasFPType *IPT_outputFmt(void) { return &BF16; }

static inline int IPT_anchorHeadroom(const InnerProductTreeParams *p) {
    return _bit_length((unsigned)(p->vecLen + 1)) + 1;
}

static inline int IPT_intWidth(const InnerProductTreeParams *p) {
    if (p->accumIntWidth > 0)
        return p->accumIntWidth;
    /* E4M3ProdFmt.sigWidth = 8, anchorHeadroom, + 15 */
    return _E4M3_PROD_SIG_WIDTH + IPT_anchorHeadroom(p) + 15;
}

static inline int IPT_expWorkWidth(void) {
    return _MAX_EXP_WIDTH_VAL + 4;   /* 12 */
}

static inline int IPT_numPipeCuts(const InnerProductTreeParams *p) {
    return _popcount8(p->pipelineCuts);
}

static inline int IPT_latency(const InnerProductTreeParams *p) {
    return IPT_numPipeCuts(p) + 1;
}

/* withPipelineDepth: returns a new params struct with the cut-set for depth */
static inline InnerProductTreeParams IPT_withPipelineDepth(
        int depth,
        const InnerProductTreeParams *base)
{
    assert(depth >= 1 && depth <= 5);
    InnerProductTreeParams out = (base != NULL) ? *base
                                                : InnerProductTreeParams_default();
    out.pipelineCuts = _PIPELINE_DEPTH_TO_CUTS[depth];
    return out;
}


/* ---------------------------------------------------------------------------
 * ComputeReq
 *   Callers own the pointed-to arrays; lengths are determined by the params
 *   that the model was constructed with (vecLen for act; numLanes for the rest).
 * ------------------------------------------------------------------------- */

typedef struct {
    const uint8_t  *act;        /* length: vecLen   */
    const uint8_t  *bias;       /* length: numLanes */
    const uint16_t *psum;       /* length: numLanes */
    const int      *scaleExp;   /* length: numLanes */
    int             act_len;
    int             bias_len;
    int             psum_len;
    int             scaleExp_len;
    AddendSel       addendSel;
    OutputFmtSel    outFmtSel;
} ComputeReq;


/* ---------------------------------------------------------------------------
 * WeightLoadReq
 * ------------------------------------------------------------------------- */

typedef struct {
    const uint8_t *weightsDma;  /* length: vecLen */
    int            weightsDma_len;
    int            laneIdx;
    bool           last;
} WeightLoadReq;


/* ---------------------------------------------------------------------------
 * StepResult
 *   out_bits is heap-allocated by the model when out_valid is true; the
 *   caller is responsible for freeing it.  NULL when out_valid is false.
 * ------------------------------------------------------------------------- */

typedef struct {
    bool      out_valid;
    uint16_t *out_bits;   /* length: numLanes, or NULL */
} StepResult;


#endif /* PARAMS_AND_REQUESTS_H */