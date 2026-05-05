# saturn_atlas_pi0.S — pi0-reflective cooperative kernel.
#
# Self-dispatching on dbg1 (single IMEM-resident program covers both phases).
#
# VMEM layout (32 words per tile, 4 tiles):
#   VMEM[byte   0..127 ]  Q  tile   (32 int32 words)
#   VMEM[byte 128..255 ]  K  tile   (32 int32 words)
#   VMEM[byte 256..383 ]  Q'XOR K'  (32 int32 words, phase-1 result)
#   VMEM[byte       384]  checksum  (1 word)
#
# Phase 0 (dbg1 == 0):
#   Seed Q[i] = (i+1) * 113,  K[i] = (i+1) * 97  for i in 0..31.
#   Write dbg0 = 1, ECALL.
#
# Between phases, Saturn (host) does pairwise Q15 rotation on both Q and K
# (16 pairs × 2 tiles) and writes Q', K' back in place.
#
# Phase 1 (dbg1 == 1):
#   For i in 0..31:  result[i] = Q'[i] XOR K'[i]; checksum ^= result[i].
#   Store result tile to VMEM[256..383], checksum to VMEM[384].
#   Write dbg0 = 2, ECALL.
#
# @TIMEOUT 200000

_start:
    CSRR  x1, 0xC11              # x1 = dbg1 (phase selector)
    BEQ   x1, x0, phase_0
    NOP
    JAL   x0, phase_1
    NOP

# ── Phase 0: seed Q[i] = (i+1)*113, K[i] = (i+1)*97 ────────────────────────
phase_0:
    ADDI  x2, x0, 113            # Q stride
    ADDI  x3, x0, 0              # Q byte addr = 0
    ADDI  x4, x0, 0              # i
    ADDI  x5, x0, 32             # count
    ADDI  x6, x0, 0              # val (will be val += 113 before first store)

p0_q_loop:
    ADD   x6, x6, x2             # val += 113
    SW    x6, x3, 0              # VMEM[addr] = val
    DELAY 1
    ADDI  x3, x3, 4
    ADDI  x4, x4, 1
    BLT   x4, x5, p0_q_loop
    NOP

    ADDI  x2, x0, 97             # K stride
    ADDI  x3, x0, 128            # K byte addr = 128
    ADDI  x4, x0, 0
    ADDI  x6, x0, 0              # val

p0_k_loop:
    ADD   x6, x6, x2             # val += 97
    SW    x6, x3, 0
    DELAY 1
    ADDI  x3, x3, 4
    ADDI  x4, x4, 1
    BLT   x4, x5, p0_k_loop
    NOP

    ADDI  x1, x0, 1
    CSRW  x1, 0xC10              # dbg0 = 1 (phase 0 done)
    ECALL

# ── Phase 1: result[i] = Q'[i] XOR K'[i]; running checksum ─────────────────
phase_1:
    ADDI  x3, x0, 0              # Q byte addr
    ADDI  x7, x0, 128            # K byte addr
    ADDI  x8, x0, 256            # result byte addr
    ADDI  x4, x0, 0              # i
    ADDI  x5, x0, 32             # count
    ADDI  x9, x0, 0              # checksum accumulator

p1_loop:
    LW    x10, x3, 0             # Q'[i]
    DELAY 2
    LW    x11, x7, 0             # K'[i]
    DELAY 2
    XOR   x12, x10, x11          # result
    XOR   x9,  x9,  x12          # checksum ^= result
    SW    x12, x8, 0             # VMEM[256 + i*4] = result
    DELAY 1
    ADDI  x3, x3, 4
    ADDI  x7, x7, 4
    ADDI  x8, x8, 4
    ADDI  x4, x4, 1
    BLT   x4, x5, p1_loop
    NOP

    SW    x9, x8, 0              # VMEM[384] = checksum
    DELAY 1

    ADDI  x1, x0, 2
    CSRW  x1, 0xC10              # dbg0 = 2 (PASS)
    ECALL
