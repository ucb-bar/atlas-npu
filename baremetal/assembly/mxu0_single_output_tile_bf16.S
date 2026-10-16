# mxu0_single_output_tile_bf16.S — C_tile = A_tile @ B_tile + bias (K=2048)
#                                  weight double-buffer with overlap
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu0_single_output_tile_bf16.py 42 1
#
# Overlap patterns demonstrated:
#   [a] weight push + acc push — prologue pushes bias and first weights
#       concurrently using Slot A (port 0) and Slot B (port 1)
#   [b] compute + weight push — main loop pre-loads one tile ahead
#       and overlaps MATMUL.ACC with next weight push
#
# TRF bank allocation:
#   0      activations
#   4 → 5  even weight tiles (raw → transposed)
#   6 → 7  odd weight tiles (raw → transposed)
#   8      bias (FP8)
#   2, 3   BF16 result (pop output)
#
# DRAM layout (matches gen_mxu0_single_output_tile_bf16.py):
#   0x00000  activations     (64 tiles × 1024B = 64KB)
#   0x10000  weights         (64 tiles × 1024B = 64KB)
#   0x20000  bias FP8        (1024B)
#   0x20400  result bank 2   (1024B BF16, lanes 0-15)
#   0x20800  result bank 3   (1024B BF16, lanes 16-31)

    LI    x10, 0             # tile counter
    LI    x11, 64            # total K tiles
    LI    x12, 1024          # bytes per tile
    LI    x16, 0x90000000    # act DRAM offset (FPGA base + 0)
    LI    x17, 0x90010000    # wgt DRAM offset (FPGA base + 0x10000)
    LI    x18, 0x90020000    # bias DRAM offset (FPGA base + 0x20000)
    LI    x23, 0x90020400    # result DRAM offset (FPGA base + 0x20400)
    LI    x24, 1024          # result bytes per bank
    LI    x30, 1             # const 1
    LI    x5, 0x00000000
    dma.config x5, ch0

# VMEM staging: 0=weight, 256=activation, 512=bias/result, 768=result

# ═══════════════════════════════════════════════════════════════
# PROLOGUE: Bias + first weight push (OVERLAPPED [a])
# ═══════════════════════════════════════════════════════════════

    # Load bias → VMEM[512] → TRF 8
    LI    x6, 512
    dma.load x6, x18, x12, ch3
    dma.wait ch3
    LI    x6, 512
    vload m8, 0(x6)
    DELAY 33

    # Load weights[0] → VMEM[0] → TRF 4 → transpose → TRF 5
    ADDI  x6, x0, 0
    dma.load x6, x17, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m4, 0(x6)
    DELAY 33
    vtrpose.xlu m5, m4
    DELAY 64

    # Overlap-safe prologue push pair
    vmatpush.weight.mxu0 w0, m5
    vmatpush.acc.fp8.mxu0 acc0, m8
    DELAY 40

    ADD   x17, x17, x12
    ADDI  x10, x10, 1

# ═══════════════════════════════════════════════════════════════
# PRE-LOAD: weights[1] → TRF 6 → 7 (for double-buffer pipeline)
# ═══════════════════════════════════════════════════════════════
    ADDI  x6, x0, 0
    dma.load x6, x17, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m6, 0(x6)
    DELAY 33
    vtrpose.xlu m7, m6
    DELAY 64

    ADD   x17, x17, x12
    ADDI  x10, x10, 1

# ═══════════════════════════════════════════════════════════════
# FIRST MATMUL: act[0] + wbuf 0 → acc 0, push wbuf 1 (OVERLAPPED [b])
# ═══════════════════════════════════════════════════════════════
    LI    x7, 256
    dma.load x7, x16, x12, ch0
    dma.wait ch0
    LI    x7, 256
    vload m0, 0(x7)
    DELAY 33
    ADD   x16, x16, x12

    # Overlap-safe first compute/push
    vmatmul.acc.mxu0 acc0, m0, w0
    vmatpush.weight.mxu0 w1, m7
    DELAY 40

    LI    x29, 1             # next matmul uses wbuf 1

# ═══════════════════════════════════════════════════════════════
# MAIN LOOP — tiles 2..63
# ═══════════════════════════════════════════════════════════════
# Loop invariant:
#   - wbuf x29 was just filled by a push (concurrently with previous matmul)
#   - Pre-load weights for opposite wbuf into TRF 4→5 (if x29=1) or 6→7 (if x29=0)
#   - x10 = next tile to pre-load weights for

tile_loop:
    BEQ   x10, x11, epilogue
    NOP

    # Pre-load weights[x10] for the OPPOSITE buffer
    BEQ   x29, x0, preload_for_buf1
    NOP

preload_for_buf0:
    # Next matmul uses wbuf 1 → pre-load for wbuf 0: TRF 4 → 5
    ADDI  x6, x0, 0
    dma.load x6, x17, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m4, 0(x6)
    DELAY 33
    vtrpose.xlu m5, m4
    DELAY 64
    BEQ   x0, x0, do_matmul
    NOP

preload_for_buf1:
    # Next matmul uses wbuf 0 → pre-load for wbuf 1: TRF 6 → 7
    ADDI  x6, x0, 0
    dma.load x6, x17, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m6, 0(x6)
    DELAY 33
    vtrpose.xlu m7, m6
    DELAY 64

do_matmul:
    ADD   x17, x17, x12
    ADDI  x10, x10, 1

    # Load activations → TRF 0
    LI    x7, 256
    dma.load x7, x16, x12, ch0
    dma.wait ch0
    LI    x7, 256
    vload m0, 0(x7)
    DELAY 33
    ADD   x16, x16, x12

    # ── OVERLAP [b]: compute + weight push ──
    BEQ   x29, x0, matmul_buf0
    NOP

matmul_buf1:
    # Matmul with wbuf 1 + push pre-loaded weights to wbuf 0
    vmatmul.acc.mxu0 acc0, m0, w1      # Slot A: acc 0, acts TRF 0, wbuf 1
    vmatpush.weight.mxu0 w0, m5         # Slot B: wbuf 0 ← TRF 5  (OVERLAPPED)
    DELAY 40
    BEQ   x0, x0, advance
    NOP

matmul_buf0:
    # Matmul with wbuf 0 + push pre-loaded weights to wbuf 1
    vmatmul.acc.mxu0 acc0, m0, w0      # Slot A: acc 0, acts TRF 0, wbuf 0
    vmatpush.weight.mxu0 w1, m7         # Slot B: wbuf 1 ← TRF 7  (OVERLAPPED)
    DELAY 40

advance:
    XOR   x29, x29, x30            # toggle buffer
    BEQ   x0, x0, tile_loop
    NOP

# ═══════════════════════════════════════════════════════════════
# EPILOGUE — last tile: just matmul, no pre-loaded weights
# ═══════════════════════════════════════════════════════════════
epilogue:
    LI    x7, 256
    dma.load x7, x16, x12, ch0
    dma.wait ch0
    LI    x7, 256
    vload m0, 0(x7)
    DELAY 33

    BEQ   x29, x0, last_buf0
    NOP
last_buf1:
    vmatmul.acc.mxu0 acc0, m0, w1
    BEQ   x0, x0, do_pop
    NOP
last_buf0:
    vmatmul.acc.mxu0 acc0, m0, w0

do_pop:
    DELAY 95
    vmatpop.bf16.acc.mxu0 m2, acc0
    DELAY 31

    LI    x8, 512
    vstore m2, 0(x8)
    DELAY 33

    LI    x9, 768
    vstore m3, 0(x9)
    DELAY 33

    LI    x8, 512
    dma.store x23, x8, x24, ch2
    dma.wait ch2
    LI    x9, 768
    ADD   x25, x23, x24
    dma.store x25, x9, x24, ch3
    dma.wait ch3

    ADDI  x30, x0, 1
    CSRRW x0, 0xC10, x30
    ECALL
    
