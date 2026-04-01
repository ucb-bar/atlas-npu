# dma_stress.S — DMA stress: copy 16 matrices via loop
# @TIMEOUT 300000
# @PYTHON_GEN gen_dma_stress.py 1300
#
# Each iteration: DRAM → VMEM[0] → DRAM

    LI    x2, 0              # loop counter
    LI    x3, 16             # count
    LI    x11, 1024          # bytes per matrix
    LI    x16, 0             # src DRAM byte offset
    LI    x17, 16384         # dst DRAM byte offset
    LI    x5,  0x60000000
    DMA.CONFIG  x5, 0

    ADDI  x6, x0, 0         # VMEM word address = 0 (staging)

loop:
    BEQ   x2, x3, done
    NOP
    NOP

    # DRAM[base+x16] → VMEM[0]
    DMA.LOAD  x6, x16, x11, 0
    DMA.WAIT  0

    # VMEM[0] → DRAM[base+x17]
    DMA.STORE x17, x6, x11, 2
    DMA.WAIT  2

    ADD   x16, x16, x11
    ADD   x17, x17, x11
    ADDI  x2, x2, 1
    JAL   x0, loop
    NOP
    NOP

done:
    ADDI  x30, x0, 1
    CSRRW x0, 0xC10, x30
    JAL   x0, 0
    NOP
    NOP
    