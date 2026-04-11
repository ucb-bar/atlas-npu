# dma_overlap.S — Multiple in-flight DMA transfers with early/late waits
# @TIMEOUT 20000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_overlap.py
#
# Tests three scenarios:
#
#   1. EARLY WAIT (transfer definitely still in flight):
#      Issue two DMA loads on ch0 and ch1 back-to-back,
#      then DMA.WAIT ch0 immediately — no intervening work,
#      so ch0 is certainly still transferring.
#
#   2. LATE WAIT (transfer almost certainly finished):
#      Issue DMA load on ch2, then execute ~50 scalar ALU ops
#      (no VMEM or DMA hazards), then DMA.WAIT ch2.
#      The transfer completes long before the wait.
#
#   3. OVERLAPPED STORES:
#      Issue three DMA stores on ch0/ch1/ch2 back-to-back.
#      DMA.WAIT ch0 immediately (early), do scalar work,
#      then DMA.WAIT ch1/ch2 (late).
#
# Data integrity check:
#   Input regions have distinct byte patterns (0xAA, 0xBB, 0xCC).
#   Each is loaded to VMEM, round-tripped through TRF, stored to
#   a different VMEM region, then DMA-stored back to DRAM.
#   The generator checks output regions match input patterns.
#
# DRAM layout (32-byte beats):
#   Input A:   word offsets   0..31   (1024 B, all 0xAA)
#   Input B:   word offsets  32..63   (1024 B, all 0xBB)
#   Input C:   word offsets  64..95   (1024 B, all 0xCC)
#   Output A:  word offsets  96..127  (1024 B, expect 0xAA)
#   Output B:  word offsets 128..159  (1024 B, expect 0xBB)
#   Output C:  word offsets 160..191  (1024 B, expect 0xCC)

    ADDI  x28, x0, 1
    LI    x5, 0x90000000
    DMA.CONFIG  x5, 0
    DMA.CONFIG  x5, 1
    DMA.CONFIG  x5, 2
    LI    x2, 1024

# ═══════════════════════════════════════════════════════════════════
# Phase 1: Two concurrent loads — EARLY WAIT
# ═══════════════════════════════════════════════════════════════════
# Issue both loads back-to-back on separate channels.
# Neither has completed when we immediately wait on ch0.

    ADDI  x6, x0, 0          # VMEM[0]
    ADDI  x1, x0, 0          # DRAM offset 0
    DMA.LOAD  x6, x1, x2, 0  # ch0: Input A → VMEM[0]

    LI    x7, 256             # VMEM[256]
    LI    x1, 1024            # DRAM offset 1024
    DMA.LOAD  x7, x1, x2, 1  # ch1: Input B → VMEM[256]

    # EARLY WAIT: ch0 is definitely still in flight —
    # we issued it just 2 instructions ago.
    DMA.WAIT  0

    # ch1 was issued even more recently, also still in flight.
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════════
# Phase 2: Load with scalar overlap — LATE WAIT
# ═══════════════════════════════════════════════════════════════════
# Issue ch2 load, then burn ~50 cycles of scalar ALU work.
# The 1024-byte DMA completes well within that window.

    LI    x7, 512             # VMEM[512]
    LI    x1, 2048            # DRAM offset 2048
    DMA.LOAD  x7, x1, x2, 2  # ch2: Input C → VMEM[512]

    # Scalar work — no VMEM or DMA hazards, just register arithmetic.
    # This gives ch2 plenty of time to finish.
    ADDI  x10, x0, 0
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1

    # LATE WAIT: ch2 has almost certainly finished by now.
    DMA.WAIT  2

# ═══════════════════════════════════════════════════════════════════
# Round-trip data through TRF to prove VMEM contents are correct
# ═══════════════════════════════════════════════════════════════════

    # VMEM[0] → TRF m0 → VMEM[768]
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    LI    x8, 768
    VSTORE 0, x8, 0
    DELAY 40

    # VMEM[256] → TRF m2 → VMEM[1024]
    LI    x7, 256
    VLOAD 2, x7, 0
    DELAY 40
    LI    x8, 1024
    VSTORE 2, x8, 0
    DELAY 40

    # VMEM[512] → TRF m4 → VMEM[1280]
    LI    x7, 512
    VLOAD 4, x7, 0
    DELAY 40
    LI    x8, 1280
    VSTORE 4, x8, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════════
# Phase 3: Three concurrent stores — mix of early and late waits
# ═══════════════════════════════════════════════════════════════════

    LI    x8, 768
    LI    x3, 3072
    DMA.STORE x3, x8, x2, 0   # ch0: VMEM[768] → Output A

    LI    x8, 1024
    LI    x3, 4096
    DMA.STORE x3, x8, x2, 1   # ch1: VMEM[1024] → Output B

    LI    x8, 1280
    LI    x3, 5120
    DMA.STORE x3, x8, x2, 2   # ch2: VMEM[1280] → Output C

    # EARLY WAIT: ch0 issued 4 instructions ago, likely still storing.
    DMA.WAIT  0

    # Burn scalar cycles so ch1 and ch2 finish.
    ADDI  x10, x0, 0
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1
    ADDI  x10, x10, 1

    # LATE WAIT: ch1 and ch2 have almost certainly finished.
    DMA.WAIT  1
    DMA.WAIT  2

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
