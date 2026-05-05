# mxu1_more_push_pop.S — VMATPUSH.ACC.BF16 + VMATPOP.FP8 with non-trivial scaling
# @TIMEOUT 20000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_mxu1_more_push_pop.py
#
# Test flow:
#   1. Load E4M3 weights (all 1.0 = 0x38) into TRF m4
#   2. Load E4M3 activations (all 1.0 = 0x38) into TRF m0
#   3. Load BF16 bias lo (all 8.0) into TRF m6 (lanes 0–15)
#   4. Load BF16 bias hi (all 8.0) into TRF m7 (lanes 16–31)
#   5. Push weights into MXU1 weight FIFO
#   6. Push bias as BF16 accumulator seed via VMATPUSH.ACC.BF16.MXU1
#      (reads both m6 and m7 to fill all 32 accumulator lanes)
#   7. VMATMUL.ACC: result = act × wgt^T + bias = 32.0 + 8.0 = 40.0
#   8. Pop as FP8 with SELI scale 0x7C → 2^(124-127) = 2^(-3) = 0.125
#        40.0 × 0.125 = 5.0 → E4M3 = 0x4A
#
# DRAM layout (32-byte beats):
#   Weights:      word offsets   0..31   (1024 B, E4M3)
#   Activations:  word offsets  32..63   (1024 B, E4M3)
#   Bias lo:      word offsets  64..95   (1024 B, BF16 lanes 0–15)
#   Bias hi:      word offsets  96..127  (1024 B, BF16 lanes 16–31)
#   Output:       word offsets 128..159  (1024 B, E4M3)

    ADDI  x28, x0, 1
    LI    x5, 0x00000001
    DMA.CONFIG  x5, 0
    LI    x2, 1024

# ── Load weights DRAM[0] → VMEM[0] → TRF m4 ──
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0

# ── Load activations DRAM[1024] → VMEM[256] → TRF m0 ──
    LI    x7, 256
    LI    x1, 1024
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0

# ── Load bias lo DRAM[2048] → VMEM[512] → TRF m6 ──
    LI    x7, 512
    LI    x1, 2048
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 512
    VLOAD 6, x7, 0

# ── Load bias hi DRAM[3072] → VMEM[768] → TRF m7 ──
    LI    x7, 768
    LI    x1, 3072
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 768
    VLOAD 7, x7, 0

    DELAY 40

# ── Push weights into MXU1 weight buffer ──
    VMATPUSH.W.MXU1  0, 4
    DELAY 40
    
# ── Push BF16 bias as accumulator seed (reads m6 + m7) ──
    VMATPUSH.ACC.BF16.MXU1  0, 6
    DELAY 40

# ── Matmul with accumulate: y = (act × wgt^T) + acc ──
    VMATMUL.ACC.MXU1  0, 0, 0
    DELAY 120

# ── Load E8M0 scale into scale register 0 ──
#    0x7C = 124 → 2^(124-127) = 2^(-3) = 0.125
    SELI  0, 0x7C

# ── Pop as FP8 using scale register 0, accumulator 0 ──
    VMATPOP.FP8.MXU1  2, 0, 0
    DELAY 40

# ── Store result: TRF m2 → VMEM[1024] → DRAM[4096] ──
    LI    x8, 1024
    VSTORE 2, x8, 0
    DELAY 40
    LI    x8, 1024
    LI    x3, 4096
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
    