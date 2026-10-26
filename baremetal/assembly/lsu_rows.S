# lsu_rows.S — vload and vstore at row counts M (vsetm)
# @TIMEOUT 20000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_lsu_rows.py
#
# vload of M rows writes rows 0..M-1 of m[vd] from M VMEM lines, vstore writes M lines from rows
# 0..M-1; the other rows and lines keep their values. Each LSU path is busy for M + 2 cycles, so a
# second vload (vstore) right after a first one issues M + 3 cycles after it.
#
# DBG0 = 1 if all checks pass, else the number of the failed check:
#   2  two back-to-back vloads take 29 cycles fewer at M = 3 than at M = 32, 31 fewer at M = 1
#   3  the same for two vstores
# The DRAM checks (generator): vload m8 at M = 3, vload m10 at M = 1, vstore m9 at M = 5 into R1,
# vstore m11 at M = 31 into R2.
#
# DRAM (32-byte words): T at 0, the start values of m8..m11 at 32 + 32 (r - 8), R1 at 160, R2 at
# 192; the results R1, R2, m8..m11 at 1024.
# VMEM: T at KiB 0, m8..m11 at KiB 1..4, R1 at KiB 5, R2 at KiB 6; the results from KiB 5 (R1, R2,
# then m8..m11 at KiB 7..10); KiB 12..14 for the timed transfers.

    LI    x5, 0
    dma.config x5, ch0
    LI    x2, 4096
    LI    x6, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    LI    x6, 1024
    LI    x1, 0x90001000
    LI    x3, 3072
    dma.load x6, x1, x3, ch1
    dma.wait ch0
    dma.wait ch1
    vload m8, 8(x0)
    vload m9, 16(x0)
    vload m10, 24(x0)
    vload m11, 32(x0)
    vload m12, 0(x0)
    LI    x9, 12288

# ── 2: two vloads ──
    ADDI  x28, x0, 2
    vstore m12, 96(x0)
    lw    x13, 0(x9)
    ADDI  x13, x13, 0
    vsetm 32
    CSRRS x10, 0xC00, x0
    vload m13, 104(x0)
    vload m13, 112(x0)
    CSRRS x11, 0xC00, x0
    SUB   x20, x11, x10
    vstore m12, 96(x0)
    lw    x13, 0(x9)
    ADDI  x13, x13, 0
    vsetm 3
    CSRRS x10, 0xC00, x0
    vload m13, 104(x0)
    vload m13, 112(x0)
    CSRRS x11, 0xC00, x0
    SUB   x21, x11, x10
    vstore m12, 96(x0)
    lw    x13, 0(x9)
    ADDI  x13, x13, 0
    vsetm 1
    CSRRS x10, 0xC00, x0
    vload m13, 104(x0)
    vload m13, 112(x0)
    CSRRS x11, 0xC00, x0
    SUB   x22, x11, x10
    SUB   x14, x20, x21
    ADDI  x15, x0, 29
    BNE   x14, x15, fail
    SUB   x14, x20, x22
    ADDI  x15, x0, 31
    BNE   x14, x15, fail

# ── 3: two vstores ──
    ADDI  x28, x0, 3
    vstore m12, 96(x0)
    lw    x13, 0(x9)
    ADDI  x13, x13, 0
    vsetm 32
    CSRRS x10, 0xC00, x0
    vstore m13, 104(x0)
    vstore m13, 112(x0)
    CSRRS x11, 0xC00, x0
    SUB   x20, x11, x10
    vstore m12, 96(x0)
    lw    x13, 0(x9)
    ADDI  x13, x13, 0
    vsetm 3
    CSRRS x10, 0xC00, x0
    vstore m13, 104(x0)
    vstore m13, 112(x0)
    CSRRS x11, 0xC00, x0
    SUB   x21, x11, x10
    vstore m12, 96(x0)
    lw    x13, 0(x9)
    ADDI  x13, x13, 0
    vsetm 1
    CSRRS x10, 0xC00, x0
    vstore m13, 104(x0)
    vstore m13, 112(x0)
    CSRRS x11, 0xC00, x0
    SUB   x22, x11, x10
    SUB   x14, x20, x21
    ADDI  x15, x0, 29
    BNE   x14, x15, fail
    SUB   x14, x20, x22
    ADDI  x15, x0, 31
    BNE   x14, x15, fail

# ── data ──
    vsetm 3
    vload m8, 0(x0)
    vsetm 1
    vload m10, 0(x0)
    vsetm 5
    vstore m9, 40(x0)
    vsetm 31
    vstore m11, 48(x0)
    vsetm 32
    vstore m8, 56(x0)
    vstore m9, 64(x0)
    vstore m10, 72(x0)
    vstore m11, 80(x0)
    LI    x6, 1280
    LI    x1, 0x90008000
    dma.store x1, x6, x2, ch0
    LI    x6, 2304
    LI    x1, 0x90009000
    LI    x3, 2048
    dma.store x1, x6, x3, ch1
    dma.wait ch0
    dma.wait ch1

    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL
