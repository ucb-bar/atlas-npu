# dma_stress.S — DMA stress: copy 16 matrices via loop
# @TIMEOUT 300000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_stress.py 1300
#
# Each iteration: DRAM → VMEM[0] → DRAM

    LI    x2, 0              # loop counter
    LI    x3, 16             # count
    LI    x11, 1024          # bytes per matrix
    LI    x16, 0x90000000  # src DRAM byte offset
    LI    x17, 0x90004000  # dst DRAM byte offset
    LI    x5, 0x00000000
    dma.config x5, ch0

    ADDI  x6, x0, 0         # VMEM word address = 0 (staging)

loop:
    BEQ   x2, x3, done

    # DRAM[base+x16] → VMEM[0]
    dma.load x6, x16, x11, ch0
    dma.wait ch0

    # VMEM[0] → DRAM[base+x17]
    dma.store x17, x6, x11, ch2
    dma.wait ch2

    ADD   x16, x16, x11
    ADD   x17, x17, x11
    ADDI  x2, x2, 1
    JAL   x0, loop

done:
    ADDI  x30, x0, 1
    CSRRW x0, 0x800, x30
    ECALL
    
