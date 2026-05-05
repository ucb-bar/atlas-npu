# vpu_midbank_recip_sqrt_edge_negimm.S
# @TIMEOUT 250000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_vpu_midbank_recip_sqrt_edge_negimm.py
#
# Coverage intent:
#   - Class-directed BF16 edge inputs for VRECIP.BF16 and VSQRT.
#   - Mid-bank MREG activity in m40..m45, including the cold m41 region.
#   - Negative-imm, high-range VLOAD/VSTORE addressing with 1 KiB-aligned
#     effective VMEM bases.
#
# DRAM byte layout (generator-backed):
#   [0x0000 .. 0x07FF]  BF16 edge-class input tensor pair   (2048 B)
#   [0x0800 .. 0x0FFF]  VRECIP.BF16 output tensor pair      (2048 B)
#   [0x1000 .. 0x17FF]  VSQRT output tensor pair            (2048 B)
#
# VMEM word-address layout:
#   input_lo   = 0xDC00  (byte 0x37000)
#   input_hi   = 0xDD00  (byte 0x37400)  -- VLOAD base, imm -8 / 0
#   recip_lo   = 0xE400  (byte 0x39000)
#   recip_hi   = 0xE500  (byte 0x39400)  -- VSTORE base, imm -8 / 0
#   sqrt_lo    = 0xEC00  (byte 0x3B000)
#   sqrt_hi    = 0xED00  (byte 0x3B400)  -- VSTORE base, imm -8 / 0
#
# MREG allocation:
#   m40,m41  input BF16 pair
#   m42,m43  recip output BF16 pair
#   m44,m45  sqrt output BF16 pair

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0

    LI    x12, 0x800          # 2048 B

    # DMA.LOAD input pair to high-range VMEM staging.
    LI    x6,  0xDC00         # input_lo word address
    LI    x10, 0x0000         # DRAM byte offset
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0

    # Negative-offset VLOAD into mid-bank MREGs.
    LI    x20, 0xDD00         # input_hi word address
    VLOAD 40, x20, -8         # -> VMEM word 0xDC00
    DELAY 40
    VLOAD 41, x20, 0          # -> VMEM word 0xDD00
    DELAY 40

    # Edge-class VPU approximators.
    VRECIP.BF16 42, 40
    DELAY 150

    VSQRT 44, 40
    DELAY 150

    # Spill recip result with negative-offset VSTORE, then DMA.STORE to DRAM.
    LI    x21, 0xE500         # recip_hi word address
    VSTORE 42, x21, -8        # -> VMEM word 0xE400
    DELAY 40
    VSTORE 43, x21, 0         # -> VMEM word 0xE500
    DELAY 40

    LI    x7,  0xE400         # recip_lo word address for DMA.STORE source
    LI    x11, 0x0800         # DRAM byte offset for recip output
    DMA.STORE x11, x7, x12, 1
    DMA.WAIT 1

    # Spill sqrt result with negative-offset VSTORE, then DMA.STORE to DRAM.
    LI    x22, 0xED00         # sqrt_hi word address
    VSTORE 44, x22, -8        # -> VMEM word 0xEC00
    DELAY 40
    VSTORE 45, x22, 0         # -> VMEM word 0xED00
    DELAY 40

    LI    x8,  0xEC00         # sqrt_lo word address for DMA.STORE source
    LI    x13, 0x1000         # DRAM byte offset for sqrt output
    DMA.STORE x13, x8, x12, 2
    DMA.WAIT 2

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
