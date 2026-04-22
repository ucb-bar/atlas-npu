# smolvla_parameterized_rms_norm.S — tiled y = X * rsqrt(mean(X^2) + eps), 128x32 BF16
# Port of jrmills20 parameterized_rms_norm.py. N fixed at 32; M scales via
# tile loop. Each 32-row group is independent. VLI constants hoisted.
# @TIMEOUT 2500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_parameterized_rms_norm.py
#
# Shape: M=512, N=32 → 16 groups.
# DRAM layout:
#   [0x00000..0x07FFF]  Input X   (16 groups x 2048B BF16)
#   [0x08000..0x0FFFF]  Output Y  (16 groups x 2048B BF16)
#
# Constants (via VLI.ALL, hoisted):
#   1/32 = 0x3D00    eps ≈ 1e-6 = 0x3586
#
# Mreg allocation (per group, m6..m9 invariant across tiles):
#   m0,m1   = X                    m2,m3   = X^2
#   m4,m5   = rowsum(X^2)          m6,m7   = inv_dim=1/32 (hoisted, invariant)
#   m8,m9   = eps (hoisted, invariant)
#   m10,m11 = mean(X^2) → inv_rms (reused within tile only)
#   m12,m13 = mean + eps → Y (reused within tile only; stored)
#   m14,m15 = sqrt(mean + eps)
#
# Note: stock smolvla_rms_norm.S aliases Y onto (m6, m7) to save mregs, but
# that clobbers the hoisted inv_dim across tile-loop iterations. We keep
# Y in (m12, m13) so the invariants survive.

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x22, 0x8000

    # ── Invariant constants ──
    VLI.ALL 6, 0x3D00          # 1/32
    DELAY 150
    VLI.ALL 8, 0x3586          # eps = 1e-6
    DELAY 150

    ADDI  x11, x0, 16
    ADDI  x10, x0, 0

tile_loop:
    SLLI  x20, x10, 11

    # ── Load X ──
    ADDI  x1, x20, 0
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x1, x20, 1024
    LI    x6, 256
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ── RMS-norm chain ──
    VSQUARE.BF16 2, 0
    DELAY 150
    VREDSUM.ROW.BF16 4, 2
    DELAY 150
    VMUL.BF16 10, 4, 6         # mean(X^2) = sum * inv_dim
    DELAY 150
    VADD.BF16 12, 10, 8        # mean + eps
    DELAY 150
    VSQRT 14, 12               # sqrt(mean + eps)
    DELAY 150
    VRECIP.BF16 10, 14         # inv_rms (reuses m10,m11; m6,m7 stays as inv_dim)
    DELAY 150
    VMUL.BF16 12, 0, 10        # Y = X * inv_rms (reuses m12,m13)
    DELAY 150

    LI    x8, 512
    VSTORE 12, x8, 0
    DELAY 40
    VSTORE 13, x8, 8
    DELAY 40

    # ── DMA out ──
    ADD   x3, x22, x20
    LI    x8, 512
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    ADDI  x3, x3, 1024
    LI    x8, 768
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    ADDI  x10, x10, 1
    BNE   x10, x11, tile_loop
    NOP
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
