# Resident IMEM dispatcher. See README.md for the host protocol and kernel ABI.
# Load at word 0. Reserve words [0, 256); slots start at 256 and 16640.
# DBG1 command: sequence[29:0] << 1 | slot. Sequence zero is reserved.
# DBG0 acknowledgment: command << 1 | done. Initial READY acknowledgment is 1.
# x28-x31 are reserved. Kernels must preserve x28 and x31 (reserve all four).

_start:
    ADDI x28, x0, 0
    ADDI x29, x0, 1
    CSRW x29, 0xC10

poll:
    CSRR x29, 0xC11
    SRLI x30, x29, 1
    BEQ x30, x0, poll
    NOP
    BEQ x29, x28, poll
    NOP
    ADDI x28, x29, 0
    SLLI x29, x28, 1
    CSRW x29, 0xC10
    ANDI x30, x28, 1
    SLLI x30, x30, 14
    ADDI x30, x30, 256
    JALR x31, x30, 0
    NOP

# Atlas JALR links to the delay slot (call PC + 1). The kernel returns with
# JALR x0, x31, 1 / NOP, landing here after its own delay slot has executed.
complete:
    SLLI x29, x28, 1
    ORI x29, x29, 1
    CSRW x29, 0xC10
    JAL x0, poll
    NOP
