#!/usr/bin/env python3
"""
random_gen.py — Comprehensive random test generator for Atlas.

Architecture (per document 24):
  1. Maintains full shadow state (DRAM, VMEM, TRF, xregs, eregs, MXU).
  2. Every generated instruction executes against the shadow model.
  3. MXU matmuls use RTL-accurate reference models (SA + IPT).
  4. Non-constant init tiles catch addressing/transpose bugs.
  5. Harvest phase stores selected state to DRAM and emits exact checks.
  6. Coverage tracking rejects weak tests.

Assumes this script lives at:
  chipyard/generators/sp26-atlas-acc/baremetal/random_permutations/random_gen.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# ── Import RTL-accurate MXU models ──────────────────────────────────
GENERATORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generators")
sys.path.insert(0, GENERATORS_DIR)

from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.ipt_rtl_linear import (
    IPTLinearRTLFunction,
    float_to_e4m3_bytes,
    e4m3_bytes_to_float,
    quant_bf16_tensor,
    torch_float_to_bf16_bits,
    torch_bf16_bits_to_float,
    _E4M3_FLOAT_LUT,
)
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

# ═════════════════════════════════════════════════════════════════════
#  Constants
# ═════════════════════════════════════════════════════════════════════

DMA_BASE   = 0x90000000
TILE_BYTES = 1024
TILE_ROWS  = 32
TILE_COLS  = 32
BEAT_BYTES = 32
BEATS_PER_TILE = TILE_BYTES // BEAT_BYTES

# DRAM regions (byte offsets, each 1024B = 32 beats)
NUM_INIT_TILES = 4
INIT_DRAM_OFF  = [i * TILE_BYTES for i in range(NUM_INIT_TILES)]  # 0, 1024, 2048, 3072
HARVEST_DRAM_BASE = NUM_INIT_TILES * TILE_BYTES  # 4096

# VMEM layout (word offsets)
INIT_VMEM_OFFS  = [i * (TILE_BYTES // 4) for i in range(NUM_INIT_TILES)]  # 0, 256, 512, 768
INIT_VLOAD_IMMS = [off // 32 for off in INIT_VMEM_OFFS] # = [0, 8, 16, 24]. 8 << 5 = 256 → VMEM[256] = tile 1
SCRATCH_VMEM    = NUM_INIT_TILES * (TILE_BYTES // 4)  # 1024

# Reserved registers
REG_VMEM_BASE  = 8
REG_VMEM_STORE = 9
REG_DMA_SIZE   = 12
RESERVED_XREGS = {0, REG_VMEM_BASE, REG_VMEM_STORE, REG_DMA_SIZE, 28, 29, 30, 31}
XREG_POOL = [r for r in range(1, 28) if r not in RESERVED_XREGS]
TRF_POOL  = list(range(0, 28))  # reserve 28-31 for harvest

# Instruction config
EXCLUDED = {
    "ECALL", "EBREAK", "NOP", "FENCE",
    "BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU", "JAL", "JALR",
    "LI", "LUI", "AUIPC",
    "LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW", "SELD",
    "DMA.CONFIG", "DMA.LOAD", "DMA.STORE", "DMA.WAIT",
    "VLI.ALL", "VLI.ROW", "VLI.COL", "VLI.ONE",
}

FAMILIES = {
    "salu": ["ADD", "SUB", "AND", "OR", "XOR", "SLL", "SRL", "SRA",
             "SLT", "SLTU", "ADDI", "ANDI", "ORI", "XORI", "SLTI",
             "SLTIU", "SLLI", "SRLI", "SRAI"],
    "xlu":  ["VTRPOSE.XLU"],
    "mxu":  ["VMATPUSH.W.MXU0", "VMATPUSH.W.MXU1",
             "VMATPUSH.ACC.FP8.MXU0", "VMATPUSH.ACC.FP8.MXU1",
             "VMATPUSH.ACC.BF16.MXU0", "VMATPUSH.ACC.BF16.MXU1",
             "VMATMUL.MXU0", "VMATMUL.MXU1",
             "VMATMUL.ACC.MXU0", "VMATMUL.ACC.MXU1",
             "VMATPOP.FP8.MXU0", "VMATPOP.FP8.MXU1",
             "VMATPOP.BF16.MXU0", "VMATPOP.BF16.MXU1"],
    "lsu":  ["VLOAD", "VSTORE", "SELI"],
}

LATENCY = {
    "ADD": 1, "SUB": 1, "AND": 1, "OR": 1, "XOR": 1,
    "SLL": 1, "SRL": 1, "SRA": 1, "SLT": 1, "SLTU": 1,
    "ADDI": 1, "ANDI": 1, "ORI": 1, "XORI": 1, "SLTI": 1,
    "SLTIU": 1, "SLLI": 1, "SRLI": 1, "SRAI": 1,
    "VLOAD": 40, "VSTORE": 40, "SELI": 1, "VTRPOSE.XLU": 80,
    "VMATPUSH.W.MXU0": 40, "VMATPUSH.W.MXU1": 40,
    "VMATPUSH.ACC.FP8.MXU0": 40, "VMATPUSH.ACC.FP8.MXU1": 40,
    "VMATPUSH.ACC.BF16.MXU0": 40, "VMATPUSH.ACC.BF16.MXU1": 40,
    "VMATMUL.MXU0": 120, "VMATMUL.MXU1": 120,
    "VMATMUL.ACC.MXU0": 120, "VMATMUL.ACC.MXU1": 120,
    "VMATPOP.FP8.MXU0": 40, "VMATPOP.FP8.MXU1": 40,
    "VMATPOP.BF16.MXU0": 40, "VMATPOP.BF16.MXU1": 40,
}

ENGINE = {}
for m in FAMILIES.get("salu", []):  ENGINE[m] = "salu"
for m in FAMILIES.get("lsu",  []):  ENGINE[m] = "lsu"
for m in FAMILIES.get("xlu",  []):  ENGINE[m] = "xlu"
for m in FAMILIES.get("mxu",  []):  ENGINE[m] = "mxu0" if "MXU0" in m else "mxu1"

# ═════════════════════════════════════════════════════════════════════
#  E4M3 / BF16 helpers
# ═════════════════════════════════════════════════════════════════════

def _e4m3_decode(b: int) -> float:
    return _E4M3_FLOAT_LUT[b & 0xFF].item()

def _tile_bytes_to_float(tile: np.ndarray) -> np.ndarray:
    """Decode a (32,32) uint8 E4M3 tile to float32."""
    t = torch.from_numpy(tile.astype(np.int64) & 0xFF)
    return _E4M3_FLOAT_LUT[t].numpy()

def _float_to_tile_bytes(arr: np.ndarray) -> np.ndarray:
    """Encode float32 (32,32) → uint8 E4M3."""
    return float_to_e4m3_bytes(torch.from_numpy(arr.astype(np.float32))).numpy()

def _bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 array to BF16 precision."""
    return quant_bf16_tensor(torch.from_numpy(x.astype(np.float32))).numpy()

def _float_to_bf16_bits_np(x: np.ndarray) -> np.ndarray:
    """Float32 → BF16 bit patterns as uint16."""
    return torch_float_to_bf16_bits(torch.from_numpy(x.astype(np.float32))).numpy().astype(np.uint16)

# ═════════════════════════════════════════════════════════════════════
#  Non-constant init tiles (position-dependent patterns)
# ═════════════════════════════════════════════════════════════════════

# Pool of valid E4M3 normal values with distinct float values
_E4M3_POOL = [0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
              0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
              0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47]

def _make_init_tile(seed: int) -> np.ndarray:
    """Generate a 32×32 E4M3 tile with position-dependent values.
    Row r, col c gets a value that depends on both r and c,
    so transpose produces a visibly different tile."""
    rng = random.Random(seed)
    pool = _E4M3_POOL[:]
    rng.shuffle(pool)
    tile = np.zeros((TILE_ROWS, TILE_COLS), dtype=np.uint8)
    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            tile[r, c] = pool[(r * 7 + c * 3 + seed) % len(pool)]
    return tile

def make_init_tiles(base_seed: int) -> List[np.ndarray]:
    """Generate NUM_INIT_TILES distinct non-constant tiles."""
    tiles = [_make_init_tile(base_seed + i) for i in range(NUM_INIT_TILES - 1)]
    tiles.append(np.zeros((TILE_ROWS, TILE_COLS), dtype=np.uint8))  # last tile = zeros
    return tiles

# ═════════════════════════════════════════════════════════════════════
#  Shadow State
# ═════════════════════════════════════════════════════════════════════

class ShadowState:
    """Full machine shadow state for reference execution."""

    def __init__(self, init_tiles: List[np.ndarray]):
        self.dram = {}          # byte_offset → np.ndarray (1024B tiles)
        self.vmem = {}          # word_offset → np.ndarray (1024B tiles)
        self.trf  = {}          # bank_id → np.ndarray (32×32 uint8)
        self.xregs = [0] * 32   # scalar registers
        self.eregs = [0] * 32   # scale registers
        self.wslots = {}        # "mxu0:0" → np.ndarray (32×32 uint8)
        self.accs   = {}        # "mxu0:0" → np.ndarray (32×32 float32, BF16-precision)

        # MXU reference models
        self.mxu_models = {
            "mxu0": SARTLLinearFunction(rows=32, cols=32, out_fmt_sel=OutputFmtSel.OutBF16),
            "mxu1": IPTLinearRTLFunction(vec_len=32, num_lanes=32, pipeline_depth=1,
                                         out_fmt_sel=OutputFmtSel.OutBF16),
        }

        # Preload DRAM with init tiles
        for i, tile in enumerate(init_tiles):
            self.dram[INIT_DRAM_OFF[i]] = tile.copy()

    # ── Scalar ALU ────────────────────────────────────────────────

    def _i32(self, v):
        v &= 0xFFFFFFFF
        return v - 0x100000000 if v >= 0x80000000 else v

    def exec_add(self, rd, rs1, rs2):
        self.xregs[rd] = self._i32(self.xregs[rs1] + self.xregs[rs2])
    def exec_sub(self, rd, rs1, rs2):
        self.xregs[rd] = self._i32(self.xregs[rs1] - self.xregs[rs2])
    def exec_and(self, rd, rs1, rs2):
        self.xregs[rd] = self.xregs[rs1] & self.xregs[rs2]
    def exec_or(self, rd, rs1, rs2):
        self.xregs[rd] = self.xregs[rs1] | self.xregs[rs2]
    def exec_xor(self, rd, rs1, rs2):
        self.xregs[rd] = self.xregs[rs1] ^ self.xregs[rs2]
    def exec_sll(self, rd, rs1, rs2):
        self.xregs[rd] = self._i32(self.xregs[rs1] << (self.xregs[rs2] & 0x1F))
    def exec_srl(self, rd, rs1, rs2):
        self.xregs[rd] = (self.xregs[rs1] & 0xFFFFFFFF) >> (self.xregs[rs2] & 0x1F)
    def exec_sra(self, rd, rs1, rs2):
        v = self.xregs[rs1]
        self.xregs[rd] = self._i32(v >> (self.xregs[rs2] & 0x1F))
    def exec_slt(self, rd, rs1, rs2):
        self.xregs[rd] = 1 if self.xregs[rs1] < self.xregs[rs2] else 0
    def exec_sltu(self, rd, rs1, rs2):
        self.xregs[rd] = 1 if (self.xregs[rs1] & 0xFFFFFFFF) < (self.xregs[rs2] & 0xFFFFFFFF) else 0

    def exec_addi(self, rd, rs1, imm):
        self.xregs[rd] = self._i32(self.xregs[rs1] + imm)
    def exec_andi(self, rd, rs1, imm):
        self.xregs[rd] = self.xregs[rs1] & (imm & 0xFFFFFFFF)
    def exec_ori(self, rd, rs1, imm):
        self.xregs[rd] = self.xregs[rs1] | (imm & 0xFFFFFFFF)
    def exec_xori(self, rd, rs1, imm):
        self.xregs[rd] = self.xregs[rs1] ^ (imm & 0xFFFFFFFF)
    def exec_slti(self, rd, rs1, imm):
        self.xregs[rd] = 1 if self.xregs[rs1] < imm else 0
    def exec_sltiu(self, rd, rs1, imm):
        self.xregs[rd] = 1 if (self.xregs[rs1] & 0xFFFFFFFF) < (imm & 0xFFFFFFFF) else 0
    def exec_slli(self, rd, rs1, imm):
        self.xregs[rd] = self._i32(self.xregs[rs1] << (imm & 0x1F))
    def exec_srli(self, rd, rs1, imm):
        self.xregs[rd] = (self.xregs[rs1] & 0xFFFFFFFF) >> (imm & 0x1F)
    def exec_srai(self, rd, rs1, imm):
        self.xregs[rd] = self._i32(self.xregs[rs1] >> (imm & 0x1F))

    def exec_seli(self, ed, val):
        self.eregs[ed] = val & 0xFF

    # ── DMA / LSU ─────────────────────────────────────────────────

    def exec_dma_load(self, vmem_off, dram_off, size):
        """Copy from DRAM to VMEM."""
        if dram_off in self.dram:
            self.vmem[vmem_off] = self.dram[dram_off].copy()

    def exec_dma_store(self, dram_off, vmem_off, size):
        """Copy from VMEM to DRAM."""
        if vmem_off in self.vmem:
            self.dram[dram_off] = self.vmem[vmem_off].copy()

    def exec_vload(self, vd, vmem_off):
        """Load from VMEM to TRF."""
        if vmem_off in self.vmem:
            self.trf[vd] = self.vmem[vmem_off].copy()

    def exec_vstore(self, vs, vmem_off):
        """Store from TRF to VMEM."""
        if vs in self.trf:
            self.vmem[vmem_off] = self.trf[vs].copy()

    # ── XLU ───────────────────────────────────────────────────────

    def exec_vtrpose(self, vd, vs):
        """Transpose a tile. Input is 32×32 bytes, transposed element-wise."""
        if vs in self.trf:
            self.trf[vd] = self.trf[vs].T.copy()

    # ── MXU ───────────────────────────────────────────────────────

    def exec_push_w(self, unit, slot, vs):
        key = f"{unit}:{slot}"
        if vs in self.trf:
            self.wslots[key] = self.trf[vs].copy()

    def exec_push_acc_fp8(self, unit, acc, vs):
        key = f"{unit}:{acc}"
        if vs in self.trf:
            self.accs[key] = _bf16_round(_tile_bytes_to_float(self.trf[vs]))

    def exec_push_acc_bf16(self, unit, acc, vs):
        """Push BF16 from two consecutive TRFs (vs, vs+1) into accumulator."""
        key = f"{unit}:{acc}"
        vs2 = vs + 1 if vs < 63 else vs - 1
        # Each TRF holds 1024 bytes = 32 rows × 16 BF16 values (32 bytes/row)
        # vs → lanes 0..15, vs+1 → lanes 16..31
        acc_tile = np.zeros((TILE_ROWS, TILE_COLS), dtype=np.float32)
        if vs in self.trf:
            raw = self.trf[vs].view(np.uint8).reshape(TILE_ROWS, TILE_COLS)
            for r in range(TILE_ROWS):
                for c in range(16):
                    bf16_bits = int(raw[r, c*2]) | (int(raw[r, c*2+1]) << 8)
                    acc_tile[r, c] = torch_bf16_bits_to_float(
                        torch.tensor([bf16_bits], dtype=torch.int32)).item()
        if vs2 in self.trf:
            raw = self.trf[vs2].view(np.uint8).reshape(TILE_ROWS, TILE_COLS)
            for r in range(TILE_ROWS):
                for c in range(16):
                    bf16_bits = int(raw[r, c*2]) | (int(raw[r, c*2+1]) << 8)
                    acc_tile[r, 16+c] = torch_bf16_bits_to_float(
                        torch.tensor([bf16_bits], dtype=torch.int32)).item()
        self.accs[key] = acc_tile

    def exec_matmul(self, unit, acc, vs, wslot):
        """acc = act × wgt^T (overwrite)"""
        wkey = f"{unit}:{wslot}"
        akey = f"{unit}:{acc}"
        if vs not in self.trf or wkey not in self.wslots:
            self.accs[akey] = np.zeros((TILE_ROWS, TILE_COLS), dtype=np.float32)
            return
        act_f = _tile_bytes_to_float(self.trf[vs]).astype(np.float32)
        wgt_f = _tile_bytes_to_float(self.wslots[wkey]).astype(np.float32)
        act_t = torch.from_numpy(act_f)
        wgt_t = torch.from_numpy(wgt_f)
        result = self.mxu_models[unit](act_t, wgt_t, scale_exp=0).numpy()
        self.accs[akey] = result.astype(np.float32)

    def exec_matmul_acc(self, unit, acc, vs, wslot):
        """acc += act × wgt^T (accumulate)"""
        wkey = f"{unit}:{wslot}"
        akey = f"{unit}:{acc}"
        if vs not in self.trf or wkey not in self.wslots:
            return
        act_f = _tile_bytes_to_float(self.trf[vs]).astype(np.float32)
        wgt_f = _tile_bytes_to_float(self.wslots[wkey]).astype(np.float32)
        act_t = torch.from_numpy(act_f)
        wgt_t = torch.from_numpy(wgt_f)
        new = self.mxu_models[unit](act_t, wgt_t, scale_exp=0).numpy().astype(np.float32)
        if akey in self.accs:
            self.accs[akey] = _bf16_round(self.accs[akey] + new)
        else:
            self.accs[akey] = new

    def exec_pop_bf16(self, unit, vd, acc):
        """Pop acc as BF16 into two TRFs (vd, vd+1)."""
        akey = f"{unit}:{acc}"
        vd2 = vd + 1 if vd < 63 else vd - 1
        if akey not in self.accs:
            return
        acc_data = self.accs[akey]
        # Lower 16 cols → vd (as 32 rows × 16 BF16 = 32 rows × 32 bytes)
        lo = np.zeros((TILE_ROWS, TILE_COLS), dtype=np.uint8)
        hi = np.zeros((TILE_ROWS, TILE_COLS), dtype=np.uint8)
        bits = _float_to_bf16_bits_np(acc_data)
        for r in range(TILE_ROWS):
            for c in range(16):
                lo[r, c*2]   = bits[r, c] & 0xFF
                lo[r, c*2+1] = (bits[r, c] >> 8) & 0xFF
            for c in range(16):
                hi[r, c*2]   = bits[r, 16+c] & 0xFF
                hi[r, c*2+1] = (bits[r, 16+c] >> 8) & 0xFF
        self.trf[vd]  = lo
        self.trf[vd2] = hi

    def exec_pop_fp8(self, unit, vd, ereg, acc):
        """Pop acc as E4M3 with scale into one TRF."""
        akey = f"{unit}:{acc}"
        if akey not in self.accs:
            return
        # Scale: multiply by 2^(ereg_val - 127)
        scale_exp = self.eregs[ereg] - 127
        scaled = self.accs[akey] * (2.0 ** scale_exp)
        self.trf[vd] = _float_to_tile_bytes(scaled)

# ═════════════════════════════════════════════════════════════════════
#  Hardware timing tracker (unchanged from before)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class TimingState:
    engine_busy: Dict[str, int] = field(default_factory=dict)
    write_dist:  Dict[str, int] = field(default_factory=dict)
    live_trfs:   Set[int] = field(default_factory=set)
    live_xregs:  Set[int] = field(default_factory=lambda: {0})
    live_eregs:  Set[int] = field(default_factory=set)
    live_wslots: Set[str] = field(default_factory=set)
    live_accs:   Set[str] = field(default_factory=set)

    def tick(self, n=1):
        for d in (self.write_dist, self.engine_busy):
            expired = [k for k, v in d.items() if v <= n]
            for k in d: d[k] -= n
            for k in expired: del d[k]

    def delay_needed(self, reads, engine=""):
        mx = max((self.write_dist.get(r, 0) for r in reads), default=0)
        if engine and engine in self.engine_busy:
            mx = max(mx, self.engine_busy[engine])
        return mx

    def emit_delay(self, reads, engine, body, margin=0):
        d = self.delay_needed(reads, engine) + margin
        if d > 0:
            body.append(f"    DELAY {d}")
            self.tick(d)

    def record(self, engine, latency, writes, body, asm):
        body.append(asm)
        for w in writes:
            self.write_dist[w] = latency
            if   w.startswith("trf:"):   self.live_trfs.add(int(w.split(":")[1]))
            elif w.startswith("xreg:"):  self.live_xregs.add(int(w.split(":")[1]))
            elif w.startswith("ereg:"):  self.live_eregs.add(int(w.split(":")[1]))
            elif w.startswith("wslot:"): self.live_wslots.add(w)
            elif w.startswith("acc:"):   self.live_accs.add(w)
        self.engine_busy[engine] = latency
        self.tick(1)

# ═════════════════════════════════════════════════════════════════════
#  Instruction generators (asm + shadow exec + timing)
# ═════════════════════════════════════════════════════════════════════

def _pick_live_xreg(rng, ts):
    live = sorted(ts.live_xregs & set(XREG_POOL + [0]))
    return rng.choice(live) if live else 0

def _pick_live_trf(rng, ts):
    live = sorted(ts.live_trfs & set(TRF_POOL))
    return rng.choice(live) if live else rng.choice(TRF_POOL)

def _mxu(m): return "mxu0" if "MXU0" in m else "mxu1"

def gen_insn(mnem, rng, ts, ss, body, margin):
    """Generate one instruction: emit asm, execute shadow, update timing."""
    engine = ENGINE.get(mnem, "salu")

    # ── SALU R-type ──
    if mnem in ("ADD","SUB","AND","OR","XOR","SLL","SRL","SRA","SLT","SLTU"):
        rd = rng.choice(XREG_POOL)
        rs1, rs2 = _pick_live_xreg(rng, ts), _pick_live_xreg(rng, ts)
        reads = {f"xreg:{rs1}", f"xreg:{rs2}"}
        writes = {f"xreg:{rd}"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 1, writes, body, f"    {mnem:8s} x{rd}, x{rs1}, x{rs2}")
        getattr(ss, f"exec_{mnem.lower()}")(rd, rs1, rs2)
        return

    # ── SALU I-type ──
    if mnem in ("ADDI","ANDI","ORI","XORI","SLTI","SLTIU","SLLI","SRLI","SRAI"):
        rd = rng.choice(XREG_POOL)
        rs1 = _pick_live_xreg(rng, ts)
        imm = rng.randint(0, 31) if mnem in ("SLLI","SRLI","SRAI") else rng.randint(-32, 63)
        reads, writes = {f"xreg:{rs1}"}, {f"xreg:{rd}"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 1, writes, body, f"    {mnem:8s} x{rd}, x{rs1}, {imm}")
        getattr(ss, f"exec_{mnem.lower()}")(rd, rs1, imm)
        return

    # ── SELI ──
    if mnem == "SELI":
        ed = rng.randint(0, 7)
        val = rng.choice([0x7F, 0x7C, 0x7E, 0x80, 0x74])
        ts.emit_delay(set(), engine, body, margin)
        ts.record(engine, 1, {f"ereg:{ed}"}, body, f"    SELI      {ed}, 0x{val:02X}")
        ss.exec_seli(ed, val)
        return

    # ── VLOAD ──
    if mnem == "VLOAD":
        vd = rng.choice(TRF_POOL)
        tile = rng.randint(0, NUM_INIT_TILES - 1)
        imm = INIT_VLOAD_IMMS[tile]
        reads = {f"xreg:{REG_VMEM_BASE}", "vmem:tensor"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 40, {f"trf:{vd}"}, body,
                  f"    VLOAD     {vd}, x{REG_VMEM_BASE}, {imm}")
        ss.exec_vload(vd, INIT_VMEM_OFFS[tile])
        return

    # ── VSTORE ──
    if mnem == "VSTORE":
        vs = _pick_live_trf(rng, ts)
        reads = {f"trf:{vs}", f"xreg:{REG_VMEM_STORE}"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 40, {"vmem:tensor"}, body,
                  f"    VSTORE    {vs}, x{REG_VMEM_STORE}, 0")
        ss.exec_vstore(vs, SCRATCH_VMEM)
        return

    # ── VTRPOSE.XLU ──
    if mnem == "VTRPOSE.XLU":
        vd, vs = rng.choice(TRF_POOL), _pick_live_trf(rng, ts)
        reads = {f"trf:{vs}"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 80, {f"trf:{vd}"}, body,
                  f"    VTRPOSE.XLU      {vd}, {vs}")
        ss.exec_vtrpose(vd, vs)
        return

    # ── MXU push weight ──
    if mnem.startswith("VMATPUSH.W."):
        u = _mxu(mnem)
        ws, vs = rng.randint(0, 1), _pick_live_trf(rng, ts)
        reads = {f"trf:{vs}"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 40, {f"wslot:{u}:{ws}"}, body,
                  f"    {mnem:30s} {ws}, {vs}")
        ss.exec_push_w(u, ws, vs)
        return

    # ── MXU push acc FP8 ──
    if "PUSH.ACC.FP8" in mnem:
        u = _mxu(mnem)
        ac, vs = rng.randint(0, 1), _pick_live_trf(rng, ts)
        reads = {f"trf:{vs}"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 40, {f"acc:{u}:{ac}"}, body,
                  f"    {mnem:30s} {ac}, {vs}")
        ss.exec_push_acc_fp8(u, ac, vs)
        return

    # ── MXU push acc BF16 ──
    if "PUSH.ACC.BF16" in mnem:
        u = _mxu(mnem)
        ac, vs = rng.randint(0, 1), _pick_live_trf(rng, ts)
        vs2 = vs + 1 if vs < 31 else vs - 1
        reads = {f"trf:{vs}", f"trf:{vs2}"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 40, {f"acc:{u}:{ac}"}, body,
                  f"    {mnem:30s} {ac}, {vs}")
        ss.exec_push_acc_bf16(u, ac, vs)
        return

    # ── MXU matmul / matmul.acc ──
    if "MATMUL" in mnem and "POP" not in mnem and "PUSH" not in mnem:
        u = _mxu(mnem)
        ac, vs = rng.randint(0, 1), _pick_live_trf(rng, ts)
        live_ws = [w for w in ts.live_wslots if w.startswith(u)]
        ws = int(rng.choice(live_ws).split(":")[-1]) if live_ws else 0
        reads = {f"trf:{vs}", f"wslot:{u}:{ws}"}
        if "ACC" in mnem: reads.add(f"acc:{u}:{ac}")
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 120, {f"acc:{u}:{ac}"}, body,
                  f"    {mnem:30s} {ac}, {vs}, {ws}")
        if "ACC" in mnem:
            ss.exec_matmul_acc(u, ac, vs, ws)
        else:
            ss.exec_matmul(u, ac, vs, ws)
        return

    # ── MXU pop BF16 ──
    if "POP.BF16" in mnem:
        u = _mxu(mnem)
        vd = rng.choice(TRF_POOL)
        live_ac = [a for a in ts.live_accs if a.startswith(u)]
        ac = int(rng.choice(live_ac).split(":")[-1]) if live_ac else 0
        vd2 = vd + 1 if vd < 31 else vd - 1
        reads = {f"acc:{u}:{ac}"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 40, {f"trf:{vd}", f"trf:{vd2}"}, body,
                  f"    {mnem:30s} {vd}, {ac}")
        ss.exec_pop_bf16(u, vd, ac)
        return

    # ── MXU pop FP8 ──
    if "POP.FP8" in mnem:
        u = _mxu(mnem)
        vd = rng.choice(TRF_POOL)
        live_ac = [a for a in ts.live_accs if a.startswith(u)]
        ac = int(rng.choice(live_ac).split(":")[-1]) if live_ac else 0
        live_er = sorted(ts.live_eregs)
        er = rng.choice(live_er) if live_er else 0
        reads = {f"acc:{u}:{ac}", f"ereg:{er}"}
        ts.emit_delay(reads, engine, body, margin)
        ts.record(engine, 40, {f"trf:{vd}"}, body,
                  f"    {mnem:30s} {vd}, {er}, {ac}")
        ss.exec_pop_fp8(u, vd, er, ac)
        return

# ═════════════════════════════════════════════════════════════════════
#  Auto-init (ensures reads don't access uninitialized resources)
# ═════════════════════════════════════════════════════════════════════

def auto_init_trf(trf_id, rng, ts, ss, body):
    tile = rng.randint(0, NUM_INIT_TILES - 1)
    imm = INIT_VLOAD_IMMS[tile]
    ts.emit_delay(set(), "lsu", body)
    ts.record("lsu", 40, {f"trf:{trf_id}"}, body,
              f"    VLOAD     {trf_id}, x{REG_VMEM_BASE}, {imm}  # auto-init")
    ss.exec_vload(trf_id, INIT_VMEM_OFFS[tile])
    body.append("    DELAY 40")
    ts.tick(40)

def auto_init_wslot(res, rng, ts, ss, body):
    parts = res.split(":")
    unit, slot = parts[1], int(parts[2])
    if not ts.live_trfs:
        auto_init_trf(0, rng, ts, ss, body)
    src = rng.choice(sorted(ts.live_trfs))
    ts.emit_delay({f"trf:{src}"}, unit, body)
    push = f"VMATPUSH.W.{'MXU0' if unit == 'mxu0' else 'MXU1'}"
    ts.record(unit, 40, {res}, body, f"    {push:30s} {slot}, {src}  # auto-init")
    ss.exec_push_w(unit, slot, src)
    body.append("    DELAY 40"); ts.tick(40)

def auto_init_acc(res, rng, ts, ss, body):
    parts = res.split(":")
    unit, acc = parts[1], int(parts[2])
    if not ts.live_trfs:
        auto_init_trf(0, rng, ts, ss, body)
    src = rng.choice(sorted(ts.live_trfs))
    ts.emit_delay({f"trf:{src}"}, unit, body)
    push = f"VMATPUSH.ACC.FP8.{'MXU0' if unit == 'mxu0' else 'MXU1'}"
    ts.record(unit, 40, {res}, body, f"    {push:30s} {acc}, {src}  # auto-init")
    ss.exec_push_acc_fp8(unit, acc, src)
    body.append("    DELAY 40"); ts.tick(40)

def ensure_resources(mnem, reads, rng, ts, ss, body):
    """Auto-initialize any uninitialized resources before an instruction."""
    for r in sorted(reads):
        if r.startswith("trf:"):
            t = int(r.split(":")[1])
            if t not in ts.live_trfs:
                auto_init_trf(t, rng, ts, ss, body)
        elif r.startswith("ereg:"):
            e = int(r.split(":")[1])
            if e not in ts.live_eregs:
                body.append(f"    SELI  {e}, 0x7F  # auto-init")
                ts.live_eregs.add(e); ss.exec_seli(e, 0x7F); ts.tick(1)
        elif r.startswith("wslot:") and r not in ts.live_wslots:
            auto_init_wslot(r, rng, ts, ss, body)
        elif r.startswith("acc:") and r not in ts.live_accs:
            auto_init_acc(r, rng, ts, ss, body)

# ═════════════════════════════════════════════════════════════════════
#  Harvest: store modeled state to DRAM and generate checks
# ═════════════════════════════════════════════════════════════════════

def _pack_beat(data_bytes):
    """Pack 32 bytes LE into hex string."""
    w = 0
    for i, b in enumerate(data_bytes):
        w |= (int(b) & 0xFF) << (8 * i)
    return f"0x{w:064x}"

def _tile_to_beats(tile):
    """Convert 32×32 uint8 tile to 32 beat hex strings."""
    flat = tile.reshape(TILE_ROWS, TILE_COLS)
    return [_pack_beat(flat[r, :]) for r in range(TILE_ROWS)]

def generate_harvest(ss, body, init_tiles):
    """Harvest live state: store TRFs to DRAM, return (asm_lines, dram_checks)."""
    checks = []
    harvest_off = HARVEST_DRAM_BASE
    harvest_items = []

    body.append("")
    body.append("# ═══════════════════════════════════════════════════════════")
    body.append("# Harvest: store modeled state to DRAM for verification")
    body.append("# ═══════════════════════════════════════════════════════════")
    body.append("    DELAY 120    # drain all pipelines")

    # Harvest live TRFs (up to 4)
    live_trfs = sorted(ss.trf.keys())[:4]
    for trf_id in live_trfs:
        tile = ss.trf[trf_id]
        beats = _tile_to_beats(tile)
        word_off = harvest_off // BEAT_BYTES

        body.append(f"# ── Harvest TRF {trf_id} → DRAM[0x{harvest_off:X}] ──")
        body.append(f"    VSTORE {trf_id}, x{REG_VMEM_STORE}, 0")
        body.append(f"    DELAY 40")
        body.append(f"    LI    x10, {SCRATCH_VMEM}")
        body.append(f"    LI    x11, {harvest_off}")
        body.append(f"    DMA.STORE x11, x10, x{REG_DMA_SIZE}, 0")
        body.append(f"    DMA.WAIT  0")

        for i, beat_hex in enumerate(beats):
            checks.append({"word_offset": word_off + i, "expected": beat_hex})

        harvest_items.append(f"trf:{trf_id}")
        harvest_off += TILE_BYTES

    return checks, harvest_items

# ═════════════════════════════════════════════════════════════════════
#  Golden data (preloads from init tiles)
# ═════════════════════════════════════════════════════════════════════

def make_preloads(init_tiles):
    preloads = []
    for ti, tile in enumerate(init_tiles):
        beats = _tile_to_beats(tile)
        base_wo = INIT_DRAM_OFF[ti] // BEAT_BYTES
        for i, beat_hex in enumerate(beats):
            preloads.append({"word_offset": base_wo + i, "data": beat_hex})
    return preloads

# ═════════════════════════════════════════════════════════════════════
#  Main generator
# ═════════════════════════════════════════════════════════════════════

def build_pool(families=None):
    if families:
        pool = []
        for f in families:
            pool.extend(FAMILIES.get(f, []))
    else:
        pool = [m for fam in FAMILIES.values() for m in fam]
    return [m for m in pool if m not in EXCLUDED]

def _reads_for(mnem, rng, ts):
    """Peek at what resources a random instantiation of mnem would read."""
    u = _mxu(mnem) if "MXU" in mnem else ""
    if mnem in ("ADD","SUB","AND","OR","XOR","SLL","SRL","SRA","SLT","SLTU"):
        return {f"xreg:{_pick_live_xreg(rng, ts)}", f"xreg:{_pick_live_xreg(rng, ts)}"}
    if mnem in ("ADDI","ANDI","ORI","XORI","SLTI","SLTIU","SLLI","SRLI","SRAI"):
        return {f"xreg:{_pick_live_xreg(rng, ts)}"}
    if mnem == "SELI": return set()
    if mnem == "VLOAD": return {f"xreg:{REG_VMEM_BASE}"}
    if mnem == "VSTORE": return {f"trf:{_pick_live_trf(rng, ts)}"}
    if mnem == "VTRPOSE.XLU": return {f"trf:{_pick_live_trf(rng, ts)}"}
    if "PUSH.W" in mnem: return {f"trf:{_pick_live_trf(rng, ts)}"}
    if "PUSH.ACC" in mnem: return {f"trf:{_pick_live_trf(rng, ts)}"}
    if "MATMUL" in mnem and "POP" not in mnem:
        ws = [w for w in ts.live_wslots if w.startswith(u)]
        r = {f"trf:{_pick_live_trf(rng, ts)}", f"wslot:{u}:{int(rng.choice(ws).split(':')[-1]) if ws else 0}"}
        if "ACC" in mnem: r.add(f"acc:{u}:{rng.randint(0,1)}")
        return r
    if "POP" in mnem:
        r = {f"acc:{u}:{rng.randint(0,1)}"}
        if "FP8" in mnem:
            er = sorted(ts.live_eregs)
            r.add(f"ereg:{rng.choice(er) if er else 0}")
        return r
    return set()


def generate_test(num_insns, seed, families=None, margin=0):
    rng = random.Random(seed)
    pool = build_pool(families)
    init_tiles = make_init_tiles(seed)

    ss = ShadowState(init_tiles)
    ts = TimingState()

    # ── Preamble ──
    preamble = [
        "    ADDI  x28, x0, 1", "",
        "# ── DMA + VLOAD preamble ──",
        f"    LI    x5, 0x{DMA_BASE:08X}",
        "    DMA.CONFIG  x5, 0",
        f"    LI    x{REG_DMA_SIZE}, {TILE_BYTES}",
        f"    ADDI  x{REG_VMEM_BASE}, x0, 0",
        f"    LI    x{REG_VMEM_STORE}, {SCRATCH_VMEM}",
    ]
    ts.live_xregs.update([5, REG_VMEM_BASE, REG_VMEM_STORE, REG_DMA_SIZE])
    ss.xregs[REG_VMEM_BASE] = 0
    ss.xregs[REG_VMEM_STORE] = SCRATCH_VMEM
    ss.xregs[REG_DMA_SIZE] = TILE_BYTES

    for ti in range(NUM_INIT_TILES):
        preamble.append(f"    LI    x10, {INIT_VMEM_OFFS[ti]}")
        preamble.append(f"    LI    x11, {INIT_DRAM_OFF[ti]}")
        preamble.append(f"    DMA.LOAD  x10, x11, x{REG_DMA_SIZE}, 0")
        preamble.append(f"    DMA.WAIT  0")
        ss.exec_dma_load(INIT_VMEM_OFFS[ti], INIT_DRAM_OFF[ti], TILE_BYTES)
    ts.live_xregs.update([10, 11])

    init_trfs = rng.sample(TRF_POOL[:16], min(4, len(TRF_POOL)))
    for i, t in enumerate(init_trfs):
        imm = INIT_VLOAD_IMMS[i % NUM_INIT_TILES]
        preamble.append(f"    VLOAD {t}, x{REG_VMEM_BASE}, {imm}")
        preamble.append("    DELAY 40")
        ts.live_trfs.add(t)
        ss.exec_vload(t, INIT_VMEM_OFFS[i % NUM_INIT_TILES])
    ts.engine_busy["lsu"] = 1

    for xr in [1, 2, 3, 4, 5]:
        val = rng.randint(1, 100)
        preamble.append(f"    ADDI  x{xr}, x0, {val}")
        ts.live_xregs.add(xr); ss.xregs[xr] = val

    preamble.append("    SELI  0, 0x7F")
    ts.live_eregs.add(0); ss.exec_seli(0, 0x7F)
    preamble.append("")

    # ── Random body ──
    body = ["# ── Random instruction sequence ──"]

    for _ in range(num_insns):
        # Save RNG state so _reads_for peek doesn't consume randomness
        rng_state = rng.getstate()
        mnem = rng.choice(pool)
        peek_reads = _reads_for(mnem, random.Random(rng.randint(0, 2**32)), ts)
        rng.setstate(rng_state)
        mnem = rng.choice(pool)  # re-pick same mnemonic

        # Auto-init uninitialized resources
        ensure_resources(mnem, peek_reads, rng, ts, ss, body)

        # Generate + execute
        gen_insn(mnem, rng, ts, ss, body, margin)

    # ── Harvest ──
    checks, harvest_items = generate_harvest(ss, body, init_tiles)

    return preamble, body, init_tiles, checks, harvest_items


# ═════════════════════════════════════════════════════════════════════
#  File emission
# ═════════════════════════════════════════════════════════════════════

def emit_asm(name, preamble, body, seed, timeout=100000):
    lines = [
        f"# {name} — random Atlas test with shadow-modeled expected values",
        f"# seed: {seed}",
        f"# @TIMEOUT {timeout}",
        f"# @DRAM_BASE 0x{DMA_BASE:08X}",
        f"# @PYTHON_GEN gen_{name}.py",
        "",
    ]
    lines.extend(preamble)
    lines.append("")
    lines.extend(body)
    lines.extend([
        "", "pass:",
        "    ADDI  x1, x0, 1",
        "    CSRRW x0, 0xC10, x1",
        "    ECALL",
        "", "fail:",
        "    CSRRW x0, 0xC10, x28",
        "    ECALL", "",
    ])
    return "\n".join(lines)


def emit_gen_script(name, init_tiles, checks, timeout=100000):
    """Emit a self-contained Python golden generator."""
    preloads = make_preloads(init_tiles)
    payload = {"dram_preloads": preloads, "dram_checks": checks, "timeout": timeout}
    payload_json = json.dumps(payload, indent=2)
    # Use triple-quoted raw string to avoid escaping issues
    return (
        f'#!/usr/bin/env python3\n'
        f'"""Golden data for {name} — shadow-modeled expected values."""\n'
        f'import json, sys\n'
        f'def main():\n'
        f'    payload = json.loads(r\'\'\'{payload_json}\'\'\')\n'
        f'    print(json.dumps(payload, indent=2))\n'
        f'if __name__ == "__main__":\n'
        f'    main()\n'
    )


def main():
    ap = argparse.ArgumentParser(description="Comprehensive random Atlas test generator.")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--families", default=None, help="salu,xlu,mxu,lsu")
    ap.add_argument("--margin", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=100000)
    ap.add_argument("--asm-dir", default=None)
    ap.add_argument("--gen-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fams = args.families.split(",") if args.families else None

    if args.dry_run:
        p, b, tiles, checks, items = generate_test(args.num, args.seed, fams, args.margin)
        print(emit_asm("random_dry_run", p, b, args.seed, args.timeout))
        print(f"\n# Harvest: {len(items)} items, {len(checks)} DRAM check beats", file=sys.stderr)
        return

    asm_dir = Path(args.asm_dir) if args.asm_dir else None
    gen_dir = Path(args.gen_dir) if args.gen_dir else None
    if asm_dir: asm_dir.mkdir(parents=True, exist_ok=True)
    if gen_dir: gen_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for i in range(args.count):
        seed = args.seed + i
        name = f"random_{seed:05d}"
        p, b, tiles, checks, items = generate_test(args.num, seed, fams, args.margin)
        if asm_dir:
            (asm_dir / f"{name}.S").write_text(emit_asm(name, p, b, seed, args.timeout))
        if gen_dir:
            (gen_dir / f"gen_{name}.py").write_text(emit_gen_script(name, tiles, checks, args.timeout))
        names.append(name)
        print(f"  {name}  ({len(checks)} checks, {len(items)} harvested)")

    pool = build_pool(fams)
    print(f"Generated {args.count} tests ({len(pool)} mnemonics)")


if __name__ == "__main__":
    main()
