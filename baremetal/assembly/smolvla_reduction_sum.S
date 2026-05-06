# smolvla_reduction_sum.S — port of npu-model smolvla_reduction_sum
# Tests: VREDSUM.ROW.BF16 on seeded random BF16 32x32 tile
#   y[i,:] = broadcast(sum_{j=0..31} X[i, j])
# @TIMEOUT 30000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_smolvla_reduction_sum.py
#
# Atlas's VREDSUM.ROW.BF16 is a one-shot pair op: it reduces all 32
# lanes of (m, m+1) per row and broadcasts the scalar back to all 16
# lanes of both output banks. npu-model splits the reduction into a
# halves-add then a 16-lane reduction; semantics differ on BF16 round
# edges, but the harness reference is atlas's VectorEngineModel.
#
# DRAM layout:
#   [0x0000..0x03FF]  Input X bank 0    (1024 B)
#   [0x0400..0x07FF]  Input X bank 1    (1024 B)
#   [0x0800..0x0FFF]  Output            (2048 B, both banks identical)
#
# Mreg allocation:  m0,m1 = X    m4,m5 = row sums (broadcast)

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

    # ── (m4,m5) = row-sum broadcast over (m0,m1) ──
    VREDSUM.ROW.BF16 4, 0
    DELAY 37

    LI    x8, 512
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
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
