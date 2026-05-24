# smolvla_matmul_mxu1_to_high.S — JAL-to-upper-IMEM variant of smolvla_matmul_mxu1
#
# Identical body to smolvla_matmul_mxu1.S through the final DMA.STORE + DMA.WAIT,
# then replaces the pass: {ADDI x1,x0,1 ; CSRRW x0,0xC10,x1 ; ECALL} block with
# a word-indexed JALR to byte 0x10000 (word 0x4000) so the next kernel staged
# into the upper IMEM half runs without halting.
#
# Used by tests/saturn_atlas_quad_phase_large.c. Atlas has 1 delay slot
# after JAL/JALR (PcControl.scala); the NOP after JALR absorbs it. The
# fail: ECALL block is dead code (no branches into it), retained as a safety
# net in case a later edit adds an explicit jump to fail:.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
#
# DRAM layout (same as smolvla_matmul_mxu1):
#   [0x0000..0x03FF]  A FP8 (32×32)             (1024 B)
#   [0x0400..0x07FF]  B FP8 (32×32)             (1024 B)
#   [0x0800..0x0BFF]  Output cols  0-15  BF16    (1024 B)
#   [0x0C00..0x0FFF]  Output cols 16-31  BF16    (1024 B)
#
# Mreg allocation:  m0 = A   m4 = B   m5 = B^T (XLU)   m2,m3 = C

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load B DRAM[0x400] → VMEM[0] → TRF m4 ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40

    # ── Load A DRAM[0x000] → VMEM[256] → TRF m0 ──
    LI    x7, 256
    LI    x1, 0x90000000
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    DELAY 40

    # ── Transpose B so MXU computes A @ B, not A @ B^T ──
    VTRPOSE.XLU 5, 4
    DELAY 80

    VMATPUSH.W.MXU1  0, 5
    DELAY 40
    VMATMUL.MXU1  0, 0, 0
    DELAY 120
    VMATPOP.BF16.MXU1  2, 0
    DELAY 40

    # ── Store m2 (cols 0-15) → VMEM[512] → DRAM[0x800] ──
    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 40

    # ── Store m3 (cols 16-31) → VMEM[768] → DRAM[0xC00] ──
    LI    x9, 768
    VSTORE 3, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x3, 0x90000800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    LI    x9, 768
    LI    x3, 0x90000C00
    DMA.STORE x3, x9, x2, 1
    DMA.WAIT  1

to_high:
    # Hop to the kernel staged at byte 0x10000 (word 0x4000) without halting.
    # One NOP delay slot covers Atlas's JAL/JALR latency.
    LI    x1, 0x4000
    JALR  x0, x1, 0
    NOP

fail:
    CSRRW x0, 0xC10, x28
    ECALL
