#!/usr/bin/env python3
"""
assembler.py — Atlas SALU assembler (Python port of SaluAssembler.scala + RV32Encode.scala).

Two-pass assembler: resolves labels, encodes all Atlas ISA instructions to 32-bit words.
Outputs hex text (one word per line) or raw binary.

Usage:
    python3 scripts/assembler.py src/test/resources/assembly/alu_basic.S
    python3 scripts/assembler.py input.S --out-hex program.hex
    python3 scripts/assembler.py input.S --out-bin program.bin
    python3 scripts/assembler.py input.S --out-hex program.hex --out-bin program.bin
    python3 scripts/assembler.py input.S --out-c header.c
"""

import argparse
import os
import re
import struct
import sys



# ── Standard R/I/B/U/J type encoders ───────────────────────────

def _mask(val, bits):
    return val & ((1 << bits) - 1)

def r_type(f7, rs2, rs1, f3, rd, op):
    return (
        (_mask(f7, 7) << 25) | (_mask(rs2, 5) << 20) | (_mask(rs1, 5) << 15) |
        (_mask(f3, 3) << 12) | (_mask(rd, 5) << 7) | _mask(op, 7)
    )

def i_type(imm, rs1, f3, rd, op):
    return (
        (_mask(imm, 12) << 20) | (_mask(rs1, 5) << 15) |
        (_mask(f3, 3) << 12) | (_mask(rd, 5) << 7) | _mask(op, 7)
    )

def b_type(imm, rs2, rs1, f3, op):
    i = _mask(imm, 13)
    return (
        (((i >> 12) & 1) << 31) | (((i >> 5) & 0x3F) << 25) | (_mask(rs2, 5) << 20) |
        (_mask(rs1, 5) << 15) | (_mask(f3, 3) << 12) | (((i >> 1) & 0xF) << 8) |
        (((i >> 11) & 1) << 7) | _mask(op, 7)
    )

def u_type(imm, rd, op):
    return (_mask(imm, 20) << 12) | (_mask(rd, 5) << 7) | _mask(op, 7)

def j_type(imm, rd, op):
    i = _mask(imm, 21)
    return (
        (((i >> 20) & 1) << 31) | (((i >> 1) & 0x3FF) << 21) |
        (((i >> 11) & 1) << 20) | (((i >> 12) & 0xFF) << 12) |
        (_mask(rd, 5) << 7) | _mask(op, 7)
    )

def s_type(imm, rs2, rs1, f3, op):
    i = _mask(imm, 12)
    return (
        (((i >> 5) & 0x7F) << 25) | (_mask(rs2, 5) << 20) | (_mask(rs1, 5) << 15) |
        (_mask(f3, 3) << 12) | ((i & 0x1F) << 7) | _mask(op, 7)
    )

# VR format: funct7[31:25] | vs2[24:19] | vs1[18:13] | vd[12:7] | opcode[6:0]
def vr_type(f7, vs2, vs1, vd, op):
    return (
        (_mask(f7, 7) << 25) | (_mask(vs2, 6) << 19) | (_mask(vs1, 6) << 13) |
        (_mask(vd, 6) << 7) | _mask(op, 7)
    )

# VLS format: imm12[31:20] | rs1[19:15] | f2[14:13] | vd[12:7] | opcode[6:0]
def vls_type(imm12, rs1, f2, vd, op):
    return (
        (_mask(imm12, 12) << 20) | (_mask(rs1, 5) << 15) |
        (_mask(f2, 2) << 13) | (_mask(vd, 6) << 7) | _mask(op, 7)
    )


# ─── RV32I instructions ─────────────────────────────────────────────────────

def ADD(rd, rs1, rs2):   return r_type(0x00, rs2, rs1, 0, rd, 0x33)
def SUB(rd, rs1, rs2):   return r_type(0x20, rs2, rs1, 0, rd, 0x33)
def AND(rd, rs1, rs2):   return r_type(0x00, rs2, rs1, 7, rd, 0x33)
def OR(rd, rs1, rs2):    return r_type(0x00, rs2, rs1, 6, rd, 0x33)
def XOR(rd, rs1, rs2):   return r_type(0x00, rs2, rs1, 4, rd, 0x33)
def SLL(rd, rs1, rs2):   return r_type(0x00, rs2, rs1, 1, rd, 0x33)
def SRL(rd, rs1, rs2):   return r_type(0x00, rs2, rs1, 5, rd, 0x33)
def SRA(rd, rs1, rs2):   return r_type(0x20, rs2, rs1, 5, rd, 0x33)
def SLT(rd, rs1, rs2):   return r_type(0x00, rs2, rs1, 2, rd, 0x33)
def SLTU(rd, rs1, rs2):  return r_type(0x00, rs2, rs1, 3, rd, 0x33)

def ADDI(rd, rs1, imm):  return i_type(imm, rs1, 0, rd, 0x13)
def ANDI(rd, rs1, imm):  return i_type(imm, rs1, 7, rd, 0x13)
def ORI(rd, rs1, imm):   return i_type(imm, rs1, 6, rd, 0x13)
def XORI(rd, rs1, imm):  return i_type(imm, rs1, 4, rd, 0x13)
def SLTI(rd, rs1, imm):  return i_type(imm, rs1, 2, rd, 0x13)
def SLTIU(rd, rs1, imm): return i_type(imm, rs1, 3, rd, 0x13)
def SLLI(rd, rs1, sh):   return i_type(sh & 0x1F, rs1, 1, rd, 0x13)
def SRLI(rd, rs1, sh):   return i_type(sh & 0x1F, rs1, 5, rd, 0x13)
def SRAI(rd, rs1, sh):   return i_type((1 << 10) | (sh & 0x1F), rs1, 5, rd, 0x13)

def LUI(rd, imm20):      return u_type(imm20, rd, 0x37)
def AUIPC(rd, imm20):    return u_type(imm20, rd, 0x17)

# Branches — callers pass WORD offsets; encoder doubles them to fit B-type
# (which drops bit[0]). Hardware divides decoded imm by 2 to recover word offset.

def BEQ(rs1, rs2, off):  return b_type(off * 2, rs2, rs1, 0, 0x63)
def BNE(rs1, rs2, off):  return b_type(off * 2, rs2, rs1, 1, 0x63)
def BLT(rs1, rs2, off):  return b_type(off * 2, rs2, rs1, 4, 0x63)
def BGE(rs1, rs2, off):  return b_type(off * 2, rs2, rs1, 5, 0x63)
def BLTU(rs1, rs2, off): return b_type(off * 2, rs2, rs1, 6, 0x63)
def BGEU(rs1, rs2, off): return b_type(off * 2, rs2, rs1, 7, 0x63)


# Jumps — callers pass WORD offsets; encoder doubles for J-type encoding.
def JAL(rd, off):        return j_type(off * 2, rd, 0x6F)
def JALR(rd, rs1, off):  return i_type(off, rs1, 0, rd, 0x67)

def NOP():               return ADDI(0, 0, 0)
def FENCE():             return i_type(0, 0, 0, 0, 0x0F)

# ─── CSR ─────────────────────────────────────────────────────────────────────

def CSRRW(rd, csr, rs1):  return i_type(csr, rs1, 1, rd, 0x73)
def CSRRS(rd, csr, rs1):  return i_type(csr, rs1, 2, rd, 0x73)
def CSRRC(rd, csr, rs1):  return i_type(csr, rs1, 3, rd, 0x73)
def CSRRWI(rd, csr, z):   return i_type(csr, z & 0x1F, 5, rd, 0x73)
def CSRRSI(rd, csr, z):   return i_type(csr, z & 0x1F, 6, rd, 0x73)
def CSRRCI(rd, csr, z):   return i_type(csr, z & 0x1F, 7, rd, 0x73)

# ─── Scalar loads (opcode 0x03) ──────────────────────────────────────────────

def LB(rd, rs1, imm):     return i_type(imm, rs1, 0, rd, 0x03)
def LH(rd, rs1, imm):     return i_type(imm, rs1, 1, rd, 0x03)
def LW(rd, rs1, imm):     return i_type(imm, rs1, 2, rd, 0x03)
def LBU(rd, rs1, imm):    return i_type(imm, rs1, 4, rd, 0x03)
def LHU(rd, rs1, imm):    return i_type(imm, rs1, 5, rd, 0x03)
def SELD(rd, rs1, imm):   return i_type(imm, rs1, 6, rd, 0x03)
def SELI(rd, imm):        return i_type(imm, 0, 7, rd, 0x03)

# ─── Scalar stores (opcode 0x23) ─────────────────────────────────────────────

def SB(rs2, rs1, imm):    return s_type(imm, rs2, rs1, 0, 0x23)
def SH(rs2, rs1, imm):    return s_type(imm, rs2, rs1, 1, 0x23)
def SW(rs2, rs1, imm):    return s_type(imm, rs2, rs1, 2, 0x23)

# ─── Tensor load/store (VLS format, opcode 0x07) ─────────────────────────────

def VLOAD(vd, rs1, imm12):   return vls_type(imm12, rs1, 0, vd, 0x07)
def VSTORE(vs, rs1, imm12):  return vls_type(imm12, rs1, 1, vs, 0x07)

# ─── DMA R-type (opcode 0x7B = 1111011) ──────────────────────────────────────
# LOAD:  rd=vmem_reg, rs1=dram_off_reg, rs2=size_reg, funct3=ch
# STORE: rd=dram_off_reg, rs1=vmem_reg, rs2=size_reg, funct3=ch
def DMA_LOAD(rd, rs1, rs2, ch):   return r_type(0x00, rs2, rs1, ch, rd, 0x7B)
def DMA_STORE(rd, rs1, rs2, ch):  return r_type(0x01, rs2, rs1, ch, rd, 0x7B)

# ─── DMA I-type (opcode 0x7F = 1111111) ──────────────────────────────────────
# funct3 = channel
def DMA_CONFIG(rs1, ch):            return i_type(0x000, rs1, ch, 0, 0x7F)
def DMA_WAIT(ch):                   return i_type(0x020, 0, ch, 0, 0x7F)

# ─── MXU (opcode 0x77 = 1110111, VR format) ──────────────────────────────────
# funct7 selects operation, vs2/vs1/vd are 6-bit register indices

def VMATPUSH_W_MXU0(wSlot, vs):          return vr_type(0x00, 0, vs, wSlot, 0x77)
def VMATPUSH_W_MXU1(wSlot, vs):          return vr_type(0x01, 0, vs, wSlot, 0x77)
def VMATPUSH_AFP8_MXU0(accSel, vs):      return vr_type(0x02, 0, vs, accSel, 0x77)
def VMATPUSH_AFP8_MXU1(accSel, vs):      return vr_type(0x03, 0, vs, accSel, 0x77)
def VMATPUSH_ABF16_MXU0(accSel, vs):     return vr_type(0x04, 0, vs, accSel, 0x77)
def VMATPUSH_ABF16_MXU1(accSel, vs):     return vr_type(0x05, 0, vs, accSel, 0x77)
def VMATPOP_FP8_MXU0(vd, scaleReg, acc): return vr_type(0x06, acc, scaleReg, vd, 0x77)
def VMATPOP_FP8_MXU1(vd, scaleReg, acc): return vr_type(0x07, acc, scaleReg, vd, 0x77)
def VMATPOP_BF16_MXU0(vd, accSel):       return vr_type(0x08, accSel, 0, vd, 0x77)
def VMATPOP_BF16_MXU1(vd, accSel):       return vr_type(0x09, accSel, 0, vd, 0x77)
def VMATMUL_MXU0(accSel, vs1, wSlot):    return vr_type(0x0A, wSlot, vs1, accSel, 0x77)
def VMATMUL_MXU1(accSel, vs1, wSlot):    return vr_type(0x0B, wSlot, vs1, accSel, 0x77)
def VMATMUL_ACC_MXU0(accSel, vs1, wSlot):return vr_type(0x0C, wSlot, vs1, accSel, 0x77)
def VMATMUL_ACC_MXU1(accSel, vs1, wSlot):return vr_type(0x0D, wSlot, vs1, accSel, 0x77)

# ─── VPU (opcode 0x57 = 1010111, VR format) ──────────────────────────────────

def VADD_BF16(vd, vs1, vs2):    return vr_type(0x00, vs2, vs1, vd, 0x57)
def VREDSUM_BF16(vd, vs1):      return vr_type(0x01, 0, vs1, vd, 0x57)
def VSUB_BF16(vd, vs1, vs2):    return vr_type(0x02, vs2, vs1, vd, 0x57)
def VMIN_BF16(vd, vs1, vs2):    return vr_type(0x04, vs2, vs1, vd, 0x57)
def VMAX_BF16(vd, vs1, vs2):    return vr_type(0x06, vs2, vs1, vd, 0x57)
def VMUL_BF16(vd, vs1, vs2):    return vr_type(0x24, vs2, vs1, vd, 0x57)
def VMOV(vd, vs):               return vr_type(0x40, 0, vs, vd, 0x57)
def VRECIP_BF16(vd, vs):        return vr_type(0x41, 0, vs, vd, 0x57)
def VEXP(vd, vs):               return vr_type(0x42, 0, vs, vd, 0x57)
def VRELU(vd, vs):              return vr_type(0x44, 0, vs, vd, 0x57)

# ─── XLU (opcode 0x6B = 1101011, VR format) ──────────────────────────────────

def VTRPOSE_XLU(vd, vs1):  return vr_type(0x00, 0, vs1, vd, 0x6B)
def VREDMAX_XLU(vd, vs1):  return vr_type(0x01, 0, vs1, vd, 0x6B)
def VREDSUM_XLU(vd, vs1):  return vr_type(0x02, 0, vs1, vd, 0x6B)


# ─── Parsing helpers ────────────────────────────────────────────────────────

def parse_reg(s):
    s = s.strip().rstrip(",").lower()
    if not s.startswith("x"):
        raise ValueError(f"Invalid register: {s}")
    n = int(s[1:])
    if not 0 <= n <= 31:
        raise ValueError(f"Register out of range: {s}")
    return n

def parse_imm(s):
    s = s.strip().rstrip(",")
    if s.startswith("-0x") or s.startswith("-0X"):
        return -int(s[3:], 16)
    if s.startswith("0x") or s.startswith("0X"):
        return int(s[2:], 16)
    return int(s)

def expand_li(rd, value):
    """Expand LI pseudo-instruction into LUI+ADDI (32-bit)."""
    v = value & 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    if -2048 <= v <= 2047:
        return [ADDI(rd, 0, v & 0xFFF)]
    lo12 = v & 0xFFF
    if lo12 & 0x800:
        lo12 -= 0x1000
    hi20 = ((v - lo12) >> 12) & 0xFFFFF
    if lo12 == 0:
        return [LUI(rd, hi20)]
    return [LUI(rd, hi20), ADDI(rd, rd, lo12 & 0xFFF)]

def strip_comment(line):
    idx = line.find("#")
    return line[:idx].strip() if idx >= 0 else line.strip()

def tokenize(line):
    return [t for t in re.split(r"[\s,]+", strip_comment(line)) if t]


# ─── Two-pass assembler ─────────────────────────────────────────────────────

def assemble(source):
    raw_lines = source.splitlines()
    lines = [strip_comment(l) for l in raw_lines]
    lines = [l for l in lines if l]

    # Pass 1: collect labels (addresses are WORD indices)
    labels = {}
    addr = 0
    for line in lines:
        if line.endswith(":"):
            labels[line[:-1].strip()] = addr
        else:
            toks = tokenize(line)
            if toks:
                if toks[0].upper() == "LI":
                    addr += len(expand_li(0, parse_imm(toks[2])))
                else:
                    addr += 1

    # Pass 2: emit
    code = []
    pc = 0

    # Resolve label to word offset from current pc
    def resolve(s):
        if s in labels:
            return labels[s] - pc
        return parse_imm(s)

    for line in lines:
        if line.endswith(":"):
            continue
        toks = tokenize(line)
        if not toks:
            continue

        mnem = toks[0].upper()
        p = toks

        if mnem == "NOP":
            code.append(NOP()); pc += 1
        elif mnem == "LI":
            e = expand_li(parse_reg(p[1]), parse_imm(p[2]))
            code.extend(e); pc += len(e)

        # RV32I R-type
        elif mnem == "ADD":   code.append(ADD(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "SUB":   code.append(SUB(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "AND":   code.append(AND(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "OR":    code.append(OR(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "XOR":   code.append(XOR(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "SLL":   code.append(SLL(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "SRL":   code.append(SRL(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "SRA":   code.append(SRA(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "SLT":   code.append(SLT(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "SLTU":  code.append(SLTU(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]))); pc += 1

        # RV32I I-type
        elif mnem == "ADDI":  code.append(ADDI(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "ANDI":  code.append(ANDI(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "ORI":   code.append(ORI(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "XORI":  code.append(XORI(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "SLTI":  code.append(SLTI(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "SLTIU": code.append(SLTIU(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "SLLI":  code.append(SLLI(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "SRLI":  code.append(SRLI(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "SRAI":  code.append(SRAI(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1

        # U-type
        elif mnem == "LUI":   code.append(LUI(parse_reg(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "AUIPC": code.append(AUIPC(parse_reg(p[1]), parse_imm(p[2]))); pc += 1

        # Branches
        elif mnem == "BEQ":   code.append(BEQ(parse_reg(p[1]), parse_reg(p[2]), resolve(p[3]))); pc += 1
        elif mnem == "BNE":   code.append(BNE(parse_reg(p[1]), parse_reg(p[2]), resolve(p[3]))); pc += 1
        elif mnem == "BLT":   code.append(BLT(parse_reg(p[1]), parse_reg(p[2]), resolve(p[3]))); pc += 1
        elif mnem == "BGE":   code.append(BGE(parse_reg(p[1]), parse_reg(p[2]), resolve(p[3]))); pc += 1
        elif mnem == "BLTU":  code.append(BLTU(parse_reg(p[1]), parse_reg(p[2]), resolve(p[3]))); pc += 1
        elif mnem == "BGEU":  code.append(BGEU(parse_reg(p[1]), parse_reg(p[2]), resolve(p[3]))); pc += 1

        # Jumps
        elif mnem == "JAL":   code.append(JAL(parse_reg(p[1]), resolve(p[2]))); pc += 1
        elif mnem == "JALR":  code.append(JALR(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "FENCE": code.append(FENCE()); pc += 1

        # CSR
        elif mnem == "CSRW":   code.append(CSRRW(0, parse_imm(p[2]), parse_reg(p[1]))); pc += 1
        elif mnem == "CSRR":   code.append(CSRRS(parse_reg(p[1]), parse_imm(p[2]), 0)); pc += 1
        elif mnem == "CSRRW":  code.append(CSRRW(parse_reg(p[1]), parse_imm(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "CSRRS":  code.append(CSRRS(parse_reg(p[1]), parse_imm(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "CSRRC":  code.append(CSRRC(parse_reg(p[1]), parse_imm(p[2]), parse_reg(p[3]))); pc += 1
        elif mnem == "CSRRWI": code.append(CSRRWI(parse_reg(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "CSRRSI": code.append(CSRRSI(parse_reg(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "CSRRCI": code.append(CSRRCI(parse_reg(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1

        # Scalar loads/stores
        elif mnem == "LB":    code.append(LB(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "LH":    code.append(LH(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "LW":    code.append(LW(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "LBU":   code.append(LBU(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "LHU":   code.append(LHU(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "SELD":  code.append(SELD(parse_imm(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "SB":    code.append(SB(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "SH":    code.append(SH(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "SW":    code.append(SW(parse_reg(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1

        # Tensor load/store
        elif mnem == "VLOAD":  code.append(VLOAD(parse_imm(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VSTORE": code.append(VSTORE(parse_imm(p[1]), parse_reg(p[2]), parse_imm(p[3]))); pc += 1

        # Scale
        elif mnem == "SELI":  code.append(SELI(parse_imm(p[1]), parse_imm(p[2]))); pc += 1

        # DMA
        # DMA.LOAD  rd(vmem), rs1(dram_off), rs2(size), channel
        elif mnem == "DMA.LOAD":   code.append(DMA_LOAD(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]), parse_imm(p[4]))); pc += 1
        # DMA.STORE rd(dram_off), rs1(vmem), rs2(size), channel
        elif mnem == "DMA.STORE":  code.append(DMA_STORE(parse_reg(p[1]), parse_reg(p[2]), parse_reg(p[3]), parse_imm(p[4]))); pc += 1
        # DMA.CONFIG addr_reg, channel
        elif mnem == "DMA.CONFIG": code.append(DMA_CONFIG(parse_reg(p[1]), parse_imm(p[2]))); pc += 1
        # DMA.WAIT channel
        elif mnem == "DMA.WAIT":   code.append(DMA_WAIT(parse_imm(p[1]))); pc += 1

        # MXU push/pop/matmul
        # VMATPUSH.W.MXU0 wSlot, vs
        elif mnem == "VMATPUSH.W.MXU0":      code.append(VMATPUSH_W_MXU0(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VMATPUSH.W.MXU1":      code.append(VMATPUSH_W_MXU1(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        # VMATPUSH.ACC.FP8.MXU0 accSel, vs
        elif mnem == "VMATPUSH.ACC.FP8.MXU0":  code.append(VMATPUSH_AFP8_MXU0(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VMATPUSH.ACC.FP8.MXU1":  code.append(VMATPUSH_AFP8_MXU1(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        # VMATPUSH.ACC.BF16.MXU0 accSel, vs
        elif mnem == "VMATPUSH.ACC.BF16.MXU0": code.append(VMATPUSH_ABF16_MXU0(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VMATPUSH.ACC.BF16.MXU1": code.append(VMATPUSH_ABF16_MXU1(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        # VMATPOP.FP8.MXU0 vd, scaleReg, accSel
        elif mnem == "VMATPOP.FP8.MXU0":  code.append(VMATPOP_FP8_MXU0(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VMATPOP.FP8.MXU1":  code.append(VMATPOP_FP8_MXU1(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        # VMATPOP.BF16.MXU0 vd, accSel
        elif mnem == "VMATPOP.BF16.MXU0": code.append(VMATPOP_BF16_MXU0(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VMATPOP.BF16.MXU1": code.append(VMATPOP_BF16_MXU1(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        # VMATMUL.MXU0 accSel, vs1, wSlot
        elif mnem == "VMATMUL.MXU0":      code.append(VMATMUL_MXU0(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VMATMUL.MXU1":      code.append(VMATMUL_MXU1(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        # VMATMUL.ACC.MXU0 accSel, vs1, wSlot
        elif mnem == "VMATMUL.ACC.MXU0":  code.append(VMATMUL_ACC_MXU0(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VMATMUL.ACC.MXU1":  code.append(VMATMUL_ACC_MXU1(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1

        # VPU
        # VADD.BF16 vd, vs1, vs2
        elif mnem == "VADD.BF16":    code.append(VADD_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VSUB.BF16":    code.append(VSUB_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VMUL.BF16":    code.append(VMUL_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VMIN.BF16":    code.append(VMIN_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VMAX.BF16":    code.append(VMAX_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VREDSUM.BF16": code.append(VREDSUM_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VMOV":         code.append(VMOV(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VRECIP.BF16":  code.append(VRECIP_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VEXP":         code.append(VEXP(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VRELU":        code.append(VRELU(parse_imm(p[1]), parse_imm(p[2]))); pc += 1

        # XLU
        elif mnem == "VTRPOSE.XLU":  code.append(VTRPOSE_XLU(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VREDMAX.XLU":  code.append(VREDMAX_XLU(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VREDSUM.XLU":  code.append(VREDSUM_XLU(parse_imm(p[1]), parse_imm(p[2]))); pc += 1

        else:
            raise ValueError(f"Unknown mnemonic: {mnem} (line: {line})")

    return code

def emit_c_file(code, test_name="test"):
    n = len(code)
    array_body = ""
    for i, word in enumerate(code):
        comma = "," if i < n - 1 else ""
        array_body += f"    0x{word & 0xFFFFFFFF:08X}{comma}\n"

    return f"""\
/* Auto-generated from {test_name} by assembler.py */
#include <stdio.h>
#include <stdint.h>

static inline uint32_t mmio_read32(uintptr_t a) {{ return *(volatile uint32_t *)a; }}
static inline void mmio_write32(uintptr_t a, uint32_t v) {{ *(volatile uint32_t *)a = v; }}

/* Atlas memory map */
#define ATLAS_IMEM_BASE   0x00020000UL
#define ATLAS_CSR_BASE    0x00030000UL

#define CSR_CYCLE         (ATLAS_CSR_BASE + 0x00)
#define CSR_INSTCNT       (ATLAS_CSR_BASE + 0x04)
#define CSR_STATUS        (ATLAS_CSR_BASE + 0x08)
#define CSR_ILLEGAL_PC    (ATLAS_CSR_BASE + 0x0C)
#define CSR_DBG0          (ATLAS_CSR_BASE + 0x10)
#define CSR_DBG1          (ATLAS_CSR_BASE + 0x14)
#define CSR_SOFT_RESET    (ATLAS_CSR_BASE + 0x18)

#define ATLAS_PROGRAM_LEN {n}

static const uint32_t atlas_program[ATLAS_PROGRAM_LEN] = {{
{array_body}}};

int main(void)
{{
    uint32_t i;
    int fail = 0;

    printf("Writing program to ATLAS IMEM (%u words) ...\\n", ATLAS_PROGRAM_LEN);
    for (i = 0; i < ATLAS_PROGRAM_LEN; i++) {{
        mmio_write32(ATLAS_IMEM_BASE + i * 4, atlas_program[i]);
    }}
    asm volatile ("fence" ::: "memory");

    printf("Reading back IMEM and verifying ...\\n");
    for (i = 0; i < ATLAS_PROGRAM_LEN; i++) {{
        uint32_t got = mmio_read32(ATLAS_IMEM_BASE + i * 4);
        if (got != atlas_program[i]) {{
            printf("MISMATCH word[%u]: expected 0x%08x, got 0x%08x\\n",
                   i, atlas_program[i], got);
            fail++;
        }}
    }}

    if (fail) {{
        printf("FAIL: %d IMEM mismatches\\n", fail);
        return 1;
    }}

    printf("IMEM readback OK\\n");

    printf("Issuing soft-reset to start Atlas core ...\\n");
    mmio_write32(CSR_SOFT_RESET, 1);
    asm volatile ("fence" ::: "memory");

    printf("Waiting for Atlas core to execute ...\\n");
    uint32_t dbg0 = 0;
    for (i = 0; i < 200000000U; i++) {{
        dbg0 = mmio_read32(CSR_DBG0);
        if (dbg0 != 0) break;
    }}

    uint32_t cycles   = mmio_read32(CSR_CYCLE);
    uint32_t instcnt  = mmio_read32(CSR_INSTCNT);
    uint32_t status   = mmio_read32(CSR_STATUS);

    printf("  DBG0    = %u\\n", dbg0);
    printf("  cycles  = %u\\n", cycles);
    printf("  instcnt = %u\\n", instcnt);
    printf("  status  = 0x%08x\\n", status);

    if (dbg0 == 0) {{
        printf("FAIL: Atlas core did not complete (timeout)\\n");
        return 1;
    }} else if (dbg0 == 1) {{
        printf("PASS: {test_name} passed\\n");
        return 0;
    }} else {{
        printf("FAIL: {test_name} test %u failed\\n", dbg0);
        return 1;
    }}
}}
"""


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Atlas SALU assembler — .S to hex/binary/C"
    )
    parser.add_argument("input", help="Input .S assembly file")
    parser.add_argument("--out-hex", default=None, help="Output hex file (one 32-bit word per line)")
    parser.add_argument("--out-bin", default=None, help="Output raw binary file (little-endian 32-bit words)")
    parser.add_argument("--out-c", default=None, help="Output C file with program[] array")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        source = f.read()

    code = assemble(source)

    if args.out_hex:
        with open(args.out_hex, "w") as f:
            for word in code:
                f.write(f"{word & 0xFFFFFFFF:08x}\n")
        print(f"Wrote {len(code)} words to {args.out_hex}")

    if args.out_bin:
        with open(args.out_bin, "wb") as f:
            for word in code:
                f.write(struct.pack("<I", word & 0xFFFFFFFF))
        print(f"Wrote {len(code)} words ({len(code) * 4} bytes) to {args.out_bin}")

    if args.out_c:
        test_name = os.path.splitext(os.path.basename(args.input))[0]
        c_source = emit_c_file(code, test_name)
        with open(args.out_c, "w") as f:
            f.write(c_source)
        print(f"Wrote C file to {args.out_c} ({len(code)} instructions)")

    if not args.out_hex and not args.out_bin and not args.out_c:
        print(f"Assembled {len(code)} instructions from {args.input}")
        print()
        for i, word in enumerate(code):
            print(f"  [{i:4d}]  0x{word & 0xFFFFFFFF:08x}")


if __name__ == "__main__":
    main()
