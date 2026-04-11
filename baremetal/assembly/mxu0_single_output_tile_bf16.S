# mxu0_single_output_tile_bf16.S — C_tile = A_tile @ B_tile + bias (K=2048)
#                                  weight double-buffer with overlap
# @TIMEOUT 500000
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
    LI    x16, 0             # act DRAM offset
    LI    x17, 65536         # wgt DRAM offset  (0x10000)
    LI    x18, 131072        # bias DRAM offset (0x20000)
    LI    x23, 132096        # result DRAM offset (0x20400)
    LI    x24, 1024          # result bytes per bank
    LI    x30, 1             # const 1
    LI    x5,  0x90000000
    DMA.CONFIG  x5, 0

# VMEM staging: 0=weight, 256=activation, 512=bias/result, 768=result

# ═══════════════════════════════════════════════════════════════
# PROLOGUE: Bias + first weight push (OVERLAPPED [a])
# ═══════════════════════════════════════════════════════════════

    # Load bias → VMEM[512] → TRF 8
    LI    x6, 512
    DMA.LOAD  x6, x18, x12, 3
    DMA.WAIT  3
    LI    x6, 512
    VLOAD 8, x6, 0
    DELAY 40

    # Load weights[0] → VMEM[0] → TRF 4 → transpose → TRF 5
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40
    VTRPOSE.XLU 5, 4
    DELAY 80

    # Conservative: do not overlap first two MXU commands
    VMATPUSH.W.MXU0       0, 5
    DELAY 40
    VMATPUSH.ACC.FP8.MXU0 0, 8
    DELAY 40

    ADD   x17, x17, x12
    ADDI  x10, x10, 1

# ═══════════════════════════════════════════════════════════════
# PRE-LOAD: weights[1] → TRF 6 → 7 (for double-buffer pipeline)
# ═══════════════════════════════════════════════════════════════
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 40
    VTRPOSE.XLU 7, 6
    DELAY 80

    ADD   x17, x17, x12
    ADDI  x10, x10, 1

# ═══════════════════════════════════════════════════════════════
# FIRST MATMUL: act[0] + wbuf 0 → acc 0, push wbuf 1 (OVERLAPPED [b])
# ═══════════════════════════════════════════════════════════════
    LI    x7, 256
    DMA.LOAD  x7, x16, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    DELAY 40
    ADD   x16, x16, x12

    # Conservative first compute/push
    VMATMUL.ACC.MXU0  0, 0, 0
    DELAY 40
    VMATPUSH.W.MXU0   1, 7
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
    NOP

    # Pre-load weights[x10] for the OPPOSITE buffer
    BEQ   x29, x0, preload_for_buf1
    NOP
    NOP

preload_for_buf0:
    # Next matmul uses wbuf 1 → pre-load for wbuf 0: TRF 4 → 5
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40
    VTRPOSE.XLU 5, 4
    DELAY 80
    BEQ   x0, x0, do_matmul
    NOP
    NOP

preload_for_buf1:
    # Next matmul uses wbuf 0 → pre-load for wbuf 1: TRF 6 → 7
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 40
    VTRPOSE.XLU 7, 6
    DELAY 80

do_matmul:
    ADD   x17, x17, x12
    ADDI  x10, x10, 1

    # Load activations → TRF 0
    LI    x7, 256
    DMA.LOAD  x7, x16, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    DELAY 40
    ADD   x16, x16, x12

    # ── OVERLAP [b]: compute + weight push ──
    BEQ   x29, x0, matmul_buf0
    NOP
    NOP

matmul_buf1:
    # Matmul with wbuf 1 + push pre-loaded weights to wbuf 0
    VMATMUL.ACC.MXU0  0, 0, 1      # Slot A: acc 0, acts TRF 0, wbuf 1
    DELAY 40
    VMATPUSH.W.MXU0   0, 5         # Slot B: wbuf 0 ← TRF 5  (OVERLAPPED)
    DELAY 40
    BEQ   x0, x0, advance
    NOP
    NOP

matmul_buf0:
    # Matmul with wbuf 0 + push pre-loaded weights to wbuf 1
    VMATMUL.ACC.MXU0  0, 0, 0      # Slot A: acc 0, acts TRF 0, wbuf 0
    DELAY 40
    VMATPUSH.W.MXU0   1, 7         # Slot B: wbuf 1 ← TRF 7  (OVERLAPPED)
    DELAY 40

advance:
    XOR   x29, x29, x30            # toggle buffer
    BEQ   x0, x0, tile_loop
    NOP
    NOP

# ═══════════════════════════════════════════════════════════════
# EPILOGUE — last tile: just matmul, no pre-loaded weights
# ═══════════════════════════════════════════════════════════════
epilogue:
    LI    x7, 256
    DMA.LOAD  x7, x16, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    DELAY 40

    BEQ   x29, x0, last_buf0
    NOP
    NOP
last_buf1:
    VMATMUL.ACC.MXU0  0, 0, 1
    BEQ   x0, x0, do_pop
    NOP
    NOP
last_buf0:
    VMATMUL.ACC.MXU0  0, 0, 0

do_pop:
    DELAY 120
    VMATPOP.BF16.MXU0  2, 0
    DELAY 40

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 40

    LI    x9, 768
    VSTORE 3, x9, 0
    DELAY 40

    LI    x8, 512
    DMA.STORE x23, x8, x24, 2
    DMA.WAIT  2
    LI    x9, 768
    ADD   x25, x23, x24
    DMA.STORE x25, x9, x24, 3
    DMA.WAIT  3

    ADDI  x30, x0, 1
    CSRRW x0, 0xC10, x30
    ECALL
    
