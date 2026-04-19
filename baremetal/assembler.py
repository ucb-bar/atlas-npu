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

# VI format: imm16[31:16] | f3[15:13] | vd[12:7] | opcode[6:0]
def vi_type(imm16, f3, vd, op):
    return (
        (_mask(imm16, 16) << 16) | (_mask(f3, 3) << 13) |
        (_mask(vd, 6) << 7) | _mask(op, 7)
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

# Hardware DELAY instruction (opcode=JALR 0x67, funct3=1, I-type)
def DELAY_INSN(imm):     return i_type(imm, 0, 1, 0, 0x67)

def NOP():               return ADDI(0, 0, 0)
def FENCE():             return i_type(0, 0, 0, 0, 0x0F)
def ECALL():             return 0x00000073
def EBREAK():            return 0x00100073

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
# funct7 values from Instructions.scala

# Two-operand element-wise arithmetic
def VADD_BF16(vd, vs1, vs2):    return vr_type(0x00, vs2, vs1, vd, 0x57)  # 57/00
def VSUB_BF16(vd, vs1, vs2):    return vr_type(0x02, vs2, vs1, vd, 0x57)  # 57/02
def VMUL_BF16(vd, vs1, vs2):    return vr_type(0x03, vs2, vs1, vd, 0x57)  # 57/03

# Two-operand element-wise min/max
def VMIN_BF16(vd, vs1, vs2):    return vr_type(0x04, vs2, vs1, vd, 0x57)  # 57/04
def VMAX_BF16(vd, vs1, vs2):    return vr_type(0x06, vs2, vs1, vd, 0x57)  # 57/06

# Sublane (column) reductions: result[i,j] = op(src[:,j])
def VREDSUM_BF16(vd, vs1):      return vr_type(0x01, 0, vs1, vd, 0x57)    # 57/01
def VREDMIN_BF16(vd, vs1):      return vr_type(0x05, 0, vs1, vd, 0x57)    # 57/05
def VREDMAX_BF16(vd, vs1):      return vr_type(0x07, 0, vs1, vd, 0x57)    # 57/07

# Lane (row) reductions: result[i,j] = op(src[i,:])
def VREDSUM_ROW_BF16(vd, vs1):  return vr_type(0x21, 0, vs1, vd, 0x57)    # 57/21
def VREDMIN_ROW_BF16(vd, vs1):  return vr_type(0x24, 0, vs1, vd, 0x57)    # 57/24
def VREDMAX_ROW_BF16(vd, vs1):  return vr_type(0x26, 0, vs1, vd, 0x57)    # 57/26

# Unary ops
def VMOV(vd, vs):               return vr_type(0x40, 0, vs, vd, 0x57)     # 57/40
def VRECIP_BF16(vd, vs):        return vr_type(0x41, 0, vs, vd, 0x57)     # 57/41
def VEXP(vd, vs):               return vr_type(0x42, 0, vs, vd, 0x57)     # 57/42
def VEXP2(vd, vs):              return vr_type(0x43, 0, vs, vd, 0x57)     # 57/43
def VSQUARE_BF16(vd, vs):       return vr_type(0x46, 0, vs, vd, 0x57)     # 57/46
def VCUBE_BF16(vd, vs):         return vr_type(0x47, 0, vs, vd, 0x57)     # 57/47
def VFP8PACK(vd, vs2, es1):     return vr_type(0x44, vs2, es1, vd, 0x57)  # 57/44
def VFP8UNPACK(vd, vs2, es1):   return vr_type(0x45, vs2, es1, vd, 0x57)  # 57/45
def VRELU(vd, vs):              return vr_type(0x48, 0, vs, vd, 0x57)     # 57/48
def VSIN(vd, vs):               return vr_type(0x49, 0, vs, vd, 0x57)     # 57/49
def VCOS(vd, vs):               return vr_type(0x4A, 0, vs, vd, 0x57)     # 57/4A
def VTANH(vd, vs):              return vr_type(0x4B, 0, vs, vd, 0x57)     # 57/4B
def VLOG2(vd, vs):              return vr_type(0x4C, 0, vs, vd, 0x57)     # 57/4C
def VSQRT(vd, vs):              return vr_type(0x4D, 0, vs, vd, 0x57)     # 57/4D

# ─── XLU (opcode 0x6B = 1101011, VR format) ──────────────────────────────────

def VTRPOSE_XLU(vd, vs1):  return vr_type(0x00, 0, vs1, vd, 0x6B)

# ─── VLI immediate tensor fill (opcode 0x5F) ──────────────────────────────────
def VLI_ALL(vd, imm16): return vi_type(imm16, 0, vd, 0x5F)
def VLI_ROW(vd, imm16): return vi_type(imm16, 1, vd, 0x5F)
def VLI_COL(vd, imm16): return vi_type(imm16, 2, vd, 0x5F)
def VLI_ONE(vd, imm16): return vi_type(imm16, 3, vd, 0x5F)


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
        elif mnem == "ECALL":
            code.append(ECALL()); pc += 1
        elif mnem == "EBREAK":
            code.append(EBREAK()); pc += 1
        elif mnem == "DELAY":
            # Hardware DELAY instruction (opcode=JALR, funct3=1)
            code.append(DELAY_INSN(parse_imm(p[1]))); pc += 1
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
        elif mnem == "VLI.ALL": code.append(VLI_ALL(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VLI.ROW": code.append(VLI_ROW(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VLI.COL": code.append(VLI_COL(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VLI.ONE": code.append(VLI_ONE(parse_imm(p[1]), parse_imm(p[2]))); pc += 1

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

        # VPU — element-wise binary
        # VADD.BF16 vd, vs1, vs2
        elif mnem == "VADD.BF16":    code.append(VADD_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VSUB.BF16":    code.append(VSUB_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VMUL.BF16":    code.append(VMUL_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VMIN.BF16":    code.append(VMIN_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VMAX.BF16":    code.append(VMAX_BF16(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1

        # VPU — sublane (column) reductions
        elif mnem == "VREDSUM.BF16": code.append(VREDSUM_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VREDMIN.BF16": code.append(VREDMIN_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VREDMAX.BF16": code.append(VREDMAX_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1

        # VPU — lane (row) reductions
        elif mnem == "VREDSUM.ROW.BF16": code.append(VREDSUM_ROW_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VREDMIN.ROW.BF16": code.append(VREDMIN_ROW_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VREDMAX.ROW.BF16": code.append(VREDMAX_ROW_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1

        # VPU — unary ops
        elif mnem == "VMOV":         code.append(VMOV(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VRECIP.BF16":  code.append(VRECIP_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VEXP":         code.append(VEXP(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VEXP2":        code.append(VEXP2(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VSQUARE.BF16": code.append(VSQUARE_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VCUBE.BF16":   code.append(VCUBE_BF16(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VFP8PACK":     code.append(VFP8PACK(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VFP8UNPACK":   code.append(VFP8UNPACK(parse_imm(p[1]), parse_imm(p[2]), parse_imm(p[3]))); pc += 1
        elif mnem == "VRELU":        code.append(VRELU(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VSIN":         code.append(VSIN(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VCOS":         code.append(VCOS(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VTANH":        code.append(VTANH(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VLOG2":        code.append(VLOG2(parse_imm(p[1]), parse_imm(p[2]))); pc += 1
        elif mnem == "VSQRT":        code.append(VSQRT(parse_imm(p[1]), parse_imm(p[2]))); pc += 1

        # XLU
        elif mnem == "VTRPOSE.XLU":  code.append(VTRPOSE_XLU(parse_imm(p[1]), parse_imm(p[2]))); pc += 1

        else:
            raise ValueError(f"Unknown mnemonic: {mnem} (line: {line})")

    return code

def _hex_to_words(hex_str):
    """Convert an arbitrarily wide hex string to a list of 32-bit words, LSW first."""
    val = int(hex_str, 16)
    if val == 0:
        return [0]
    words = []
    while val > 0:
        words.append(val & 0xFFFFFFFF)
        val >>= 32
    return words


def _parse_golden_json(path, dram_base, beat_bytes):
    """Parse golden JSON → (preload_addrs[], preload_words[], check_addrs[], check_words[])."""
    import json
    with open(path) as f:
        data = json.load(f)

    words_per_beat = beat_bytes // 4

    def _expand(entries, key):
        """Convert beat-level entries to sorted per-word (byte_addr, u32) pairs."""
        pairs = []
        for e in entries:
            beat_off = e["word_offset"]
            beat_addr = dram_base + beat_off * beat_bytes
            raw = _hex_to_words(e[key])
            while len(raw) < words_per_beat:
                raw.append(0)
            for i, word in enumerate(raw[:words_per_beat]):
                pairs.append((beat_addr + i * 4, word))
        pairs.sort(key=lambda p: p[0])
        return pairs

    preload_pairs = _expand(data.get("dram_preloads", []), "data")
    check_pairs   = _expand(data.get("dram_checks", []),   "expected")

    def _split(pairs):
        if not pairs:
            return ([], [])
        addrs = []
        words = []
        for addr, word in pairs:
            addrs.append(addr)
            words.append(word)
        return (addrs, words)

    p_addrs, p_words = _split(preload_pairs)
    c_addrs, c_words = _split(check_pairs)
    return p_addrs, p_words, c_addrs, c_words


def _parse_inline_golden(source, beat_bytes):
    """Parse inline @DRAM annotations from assembly comments.

    Supported directives:
      # @DRAM_BASE   <addr>
      # @DRAM        <beat_offset> <hex_payload>
      # @CHECK_DRAM  <beat_offset> <hex_payload>

    Returns:
      (dram_base, preload_addrs, preload_words, check_addrs, check_words)
      or None if no inline preload/check directives were found.
    """
    dram_base = 0x90000000
    words_per_beat = beat_bytes // 4

    dram_base_re = re.compile(r"^\s*#\s*@DRAM_BASE\s+(\S+)\s*$", re.IGNORECASE)
    dram_re = re.compile(r"^\s*#\s*@DRAM\s+(\d+)\s+(0x[0-9a-fA-F]+)\s*$", re.IGNORECASE)
    check_re = re.compile(r"^\s*#\s*@CHECK_DRAM\s+(\d+)\s+(0x[0-9a-fA-F]+)\s*$", re.IGNORECASE)

    preload_pairs = []
    check_pairs = []

    for raw_line in source.splitlines():
        m = dram_base_re.match(raw_line)
        if m:
            dram_base = int(m.group(1), 0)
            continue

        m = dram_re.match(raw_line)
        if m:
            beat_off = int(m.group(1), 0)
            beat_addr = dram_base + beat_off * beat_bytes
            raw = _hex_to_words(m.group(2))
            while len(raw) < words_per_beat:
                raw.append(0)
            for i, word in enumerate(raw[:words_per_beat]):
                preload_pairs.append((beat_addr + i * 4, word))
            continue

        m = check_re.match(raw_line)
        if m:
            beat_off = int(m.group(1), 0)
            beat_addr = dram_base + beat_off * beat_bytes
            raw = _hex_to_words(m.group(2))
            while len(raw) < words_per_beat:
                raw.append(0)
            for i, word in enumerate(raw[:words_per_beat]):
                check_pairs.append((beat_addr + i * 4, word))

    if not preload_pairs and not check_pairs:
        return None

    preload_pairs.sort(key=lambda p: p[0])
    check_pairs.sort(key=lambda p: p[0])

    def _split(pairs):
        return ([addr for addr, _ in pairs], [word for _, word in pairs])

    p_addrs, p_words = _split(preload_pairs)
    c_addrs, c_words = _split(check_pairs)
    return dram_base, p_addrs, p_words, c_addrs, c_words


def _fmt_array(words, name, ctype="uint32_t", indent="    "):
    """Format a C array initializer."""
    lines = []
    for i in range(0, len(words), 8):
        chunk = words[i:i+8]
        vals = ", ".join(f"0x{w & 0xFFFFFFFF:08X}" for w in chunk)
        lines.append(f"{indent}{vals},")
    body = "\n".join(lines)
    return f"static const {ctype} {name}[] = {{\n{body}\n}};\n"


def emit_c_file(code, test_name="test", golden=None):
    """Generate a complete baremetal C test.

    Args:
        code: list of 32-bit instruction words
        test_name: used in printf messages
        golden: None, or (preload_addrs, preload_words, check_addrs, check_words)
    """
    n = len(code)
    array_body = ""
    for i, word in enumerate(code):
        comma = "," if i < n - 1 else ""
        array_body += f"    0x{word & 0xFFFFFFFF:08X}{comma}\n"

    has_golden = golden is not None and (golden[1] or golden[3])

    # header & helpers & memory map
    out = f"""\
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
#define CSR_EXEC_CONTROL  (ATLAS_CSR_BASE + 0x18)

#define ATLAS_EXEC_STOP   0U
#define ATLAS_EXEC_START  1U

#define ATLAS_PROGRAM_LEN {n}

static const uint32_t atlas_program[ATLAS_PROGRAM_LEN] = {{
{array_body}}};
"""

    # golden model data
    if has_golden:
        p_addrs, p_words, c_addrs, c_words = golden
        if p_words:
            out += f"#define PRELOAD_WORDS {len(p_words)}\n"
            out += _fmt_array(p_addrs, "preload_addrs", ctype="uintptr_t")
            out += _fmt_array(p_words, "preload_data")
        if c_words:
            out += f"#define CHECK_WORDS   {len(c_words)}\n"
            out += _fmt_array(c_addrs, "check_addrs", ctype="uintptr_t")
            out += _fmt_array(c_words, "check_expected")

    # main()
    out += f"""
int main(void)
{{
    uint32_t i;
    int fail = 0;
"""

    if has_golden and golden[1]:
        out += """
    printf("Writing preload data to DRAM (%u words) ...\\n", PRELOAD_WORDS);
    for (i = 0; i < PRELOAD_WORDS; i++) {
        mmio_write32(preload_addrs[i], preload_data[i]);
    }
    asm volatile ("fence" ::: "memory");
"""

    out += f"""
    printf("Stopping Atlas core before programming IMEM ...\\n");
    mmio_write32(CSR_EXEC_CONTROL, ATLAS_EXEC_STOP);
    mmio_write32(CSR_DBG0, 0);
    asm volatile ("fence" ::: "memory");

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

    printf("Issuing Atlas START command ...\\n");
    mmio_write32(CSR_EXEC_CONTROL, ATLAS_EXEC_START);
    asm volatile ("fence" ::: "memory");

    printf("Waiting for Atlas core to execute ...\\n");
    uint32_t dbg0 = 0;
    for (i = 0; i < 200000000U; i++) {{
        dbg0 = mmio_read32(CSR_DBG0);
        if (dbg0 != 0) break;
    }}

    printf("Issuing Atlas STOP command ...\\n");
    mmio_write32(CSR_EXEC_CONTROL, ATLAS_EXEC_STOP);
    asm volatile ("fence" ::: "memory");

    uint32_t cycles   = mmio_read32(CSR_CYCLE);
    uint32_t instcnt  = mmio_read32(CSR_INSTCNT);
    uint32_t status   = mmio_read32(CSR_STATUS);

    printf("  DBG0    = %u\\n", dbg0);
    printf("  mcycles = %u\\n", cycles);
    printf("  minstret = %u\\n", instcnt);
    printf("  status  = 0x%08x\\n", status);

    if (dbg0 == 0) {{
        printf("FAIL: Atlas core did not complete (timeout)\\n");
        return 1;
    }}
"""

    if has_golden and golden[3]:
        out += f"""
    printf("Verifying DRAM results (%u words) ...\\n", CHECK_WORDS);
    for (i = 0; i < CHECK_WORDS; i++) {{
        uint32_t got = mmio_read32(check_addrs[i]);
        if (got != check_expected[i]) {{
            printf("  DRAM MISMATCH word[%u] @ 0x%08x: expected 0x%08x, got 0x%08x\\n",
                   i, (uint32_t)check_addrs[i], check_expected[i], got);
            fail++;
        }}
    }}

    if (fail) {{
        printf("FAIL: %d DRAM mismatches\\n", fail);
        return 1;
    }}

    printf("PASS: {test_name} — all DRAM checks passed\\n");
    return 0;
}}
"""
    else:
        out += f"""
    if (dbg0 == 1) {{
        printf("PASS: {test_name} passed\\n");
        return 0;
    }} else {{
        printf("FAIL: {test_name} test %u failed\\n", dbg0);
        return 1;
    }}
}}
"""

    return out


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Atlas SALU assembler — .S to hex/binary/C"
    )
    parser.add_argument("input", help="Input .S assembly file")
    parser.add_argument("--out-hex", default=None, help="Output hex file (one 32-bit word per line)")
    parser.add_argument("--out-bin", default=None, help="Output raw binary file (little-endian 32-bit words)")
    parser.add_argument("--out-c", default=None, help="Output C file with full baremetal test")
    parser.add_argument("--golden-json", default=None,
                        help="Golden JSON from gen_*.py (dram_preloads + dram_checks)")
    parser.add_argument("--dram-base", default="0x90000000",
                        help="DRAM base address for golden data (default 0x90000000)")
    parser.add_argument("--beat-bytes", type=int, default=32,
                        help="DMA beat width in bytes (default 32)")
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
        golden = None
        if args.golden_json:
            dram_base = int(args.dram_base, 0)
            golden = _parse_golden_json(args.golden_json, dram_base, args.beat_bytes)
            n_pre = len(golden[1])
            n_chk = len(golden[3])
            print(f"Golden data: {n_pre} preload words, {n_chk} check words")
        else:
            inline = _parse_inline_golden(source, args.beat_bytes)
            if inline is not None:
                inline_dram_base, p_addrs, p_words, c_addrs, c_words = inline
                golden = (p_addrs, p_words, c_addrs, c_words)
                print(
                    f"Inline golden data @DRAM_BASE 0x{inline_dram_base:08X}: "
                    f"{len(p_words)} preload words, {len(c_words)} check words"
                )
        c_source = emit_c_file(code, test_name, golden=golden)
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
    
