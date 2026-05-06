# smolvla_rms_norm.S — port of npu-model smolvla_rms_norm
# Tests: y = X * rsqrt(mean(X^2) + eps) on seeded random BF16 32x32 tile
# @TIMEOUT 30000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_smolvla_rms_norm.py
#
# Atlas's VREDSUM.ROW.BF16 reduces the full 32-lane (m, m+1) pair in a
# single op, so we skip npu-model's two-bank split (row-sum bank 0 +
# row-sum bank 1 then add) — one VREDSUM produces the full row sum
# broadcast across both output banks.
#
# Constants (via VLI.ALL):
#   1/32 = 0x3D00    eps ≈ 1e-6 = 0x3586
#
# DRAM layout:
#   [0x0000..0x03FF]  Input X bank 0    (1024 B)
#   [0x0400..0x07FF]  Input X bank 1    (1024 B)
#   [0x0800..0x0FFF]  Output Y          (2048 B, 2 banks)
#
# Mreg allocation (8 pair-pairs, some reused):
#   m0,m1   = X                   (kept)
#   m2,m3   = X^2                 (temp)
#   m4,m5   = row-sum(X^2) → inv_rms (reused)
#   m6,m7   = inv_dim = 1/32 → Y      (reused)
#   m8,m9   = eps                 (const)
#   m10,m11 = mean(X^2)
#   m12,m13 = var + eps
#   m14,m15 = sqrt(var + eps)

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Input X → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ── (m2, m3) = X^2 (VSQUARE, not VMUL X,X — MRF bank has one read port) ──
    VSQUARE.BF16 2, 0
    DELAY 64

    # ── (m4, m5) = row-sum(X^2) broadcast ──
    VREDSUM.ROW.BF16 4, 2
    DELAY 37

    # ── Constants: inv_dim and eps ──
    VLI.ALL 6, 0x3D00
    DELAY 63
    VLI.ALL 8, 0x3586
    DELAY 63

    # ── (m10, m11) = sum * inv_dim = mean(X^2) ──
    VMUL.BF16 10, 4, 6
    DELAY 64

    # ── (m12, m13) = mean + eps ──
    VADD.BF16 12, 10, 8
    DELAY 64

    # ── (m14, m15) = sqrt(mean + eps) ──
    VSQRT 14, 12
    DELAY 64

    # ── (m4, m5) = 1 / sqrt(mean + eps) = inv_rms ──
    VRECIP.BF16 4, 14
    DELAY 64

    # ── (m6, m7) = X * inv_rms = Y ──
    VMUL.BF16 6, 0, 4
    DELAY 64

    LI    x8, 512
    VSTORE 6, x8, 0
    DELAY 33
    VSTORE 7, x8, 8
    DELAY 33

    LI    x8, 512
    LI    x3, 0x0800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x0C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
