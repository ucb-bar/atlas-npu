# smolvla_rms_norm.S — port of npu-model smolvla_rms_norm
# Tests: y = X * rsqrt(mean(X^2) + eps) on seeded random BF16 32x32 tile
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
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

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

    # ── Load Input X → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 256
    LI    x1, 0x90000400
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ── (m2, m3) = X^2 (VSQUARE, not VMUL X,X — MRF bank has one read port) ──
    vsquare.bf16 m2, m0
    DELAY 64

    # ── (m4, m5) = row-sum(X^2) broadcast ──
    vredsum.row.bf16 m4, m2
    DELAY 37

    # ── Constants: inv_dim and eps ──
    vli.all m6, 0x3D00
    DELAY 63
    vli.all m8, 0x3586
    DELAY 63

    # ── (m10, m11) = sum * inv_dim = mean(X^2) ──
    vmul.bf16 m10, m4, m6
    DELAY 64

    # ── (m12, m13) = mean + eps ──
    vadd.bf16 m12, m10, m8
    DELAY 64

    # ── (m14, m15) = sqrt(mean + eps) ──
    vsqrt.bf16 m14, m12
    DELAY 64

    # ── (m4, m5) = 1 / sqrt(mean + eps) = inv_rms ──
    vrecip.bf16 m4, m14
    DELAY 64

    # ── (m6, m7) = X * inv_rms = Y ──
    vmul.bf16 m6, m0, m4
    DELAY 64

    LI    x8, 512
    vstore m6, 0(x8)
    DELAY 33
    vstore m7, 8(x8)
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90000800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 768
    LI    x3, 0x90000C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL
