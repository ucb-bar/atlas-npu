#!/usr/bin/env python3
"""
Validate IPT functional model (intWidth=32, 32×32 MXU) against RTL outputs.

Usage:
    python test_functional_vs_rtl.py <resource_dir>

    resource_dir should contain:
        mxu_vectors.txt                  -> single-beat BF16 stimulus
        ipt_rtl_vector_outputs.txt       -> RTL BF16 outputs
        mxu_vectors_e4m3.txt             -> single-beat E4M3 stimulus
        ipt_rtl_vector_e4m3_outputs.txt  -> RTL E4M3 outputs
        mxu_full_matmul_vectors.txt      -> multi-tile matmul stimulus
        ipt_rtl_full_matmul_outputs.txt  -> RTL full-matmul outputs

    example <resource_dir>:
        /tools/C/kellytou/chipyard/generators/sp26-atlas-acc/src/test/resources/mxu_test_vectors

All constants hardcoded for the current InnerProductTreeParams:
    vecLen=32, numLanes=32, tileRows=32
    intWidth=32, anchorHeadroom=7, sentinel=-1024
"""
from __future__ import annotations

import math
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Constants (match atlas.common.InnerProductTreeParams defaults)
VEC_LEN = 32
NUM_LANES = 32
TILE_ROWS = 32
INT_WIDTH = 32          # sigWidth(8) + anchorHeadroom(7) + 17
ANCHOR_HR = 7
PROD_BIAS = 13
SENTINEL = -1024
INT_MASK = (1 << INT_WIDTH) - 1       # 0xFFFF_FFFF
INT_SIGN = 1 << (INT_WIDTH - 1)       # 0x8000_0000


# Build E4M3 product LUT  (S1-E5-M7, bias=13, 13-bit output)
def _build_lut() -> list[int]:
    lut = [0] * 65536
    for a in range(256):
        a_sign = (a >> 7) & 1
        a_exp = (a >> 3) & 0xF
        a_man = a & 0x7
        for b in range(256):
            b_sign = (b >> 7) & 1
            b_exp = (b >> 3) & 0xF
            b_man = b & 0x7
            out_sign = a_sign ^ b_sign
            idx = (a << 8) | b
            if a_exp == 0 or b_exp == 0:
                lut[idx] = out_sign << 12
                continue
            a_sig = 8 | a_man
            b_sig = 8 | b_man
            prod_sig = a_sig * b_sig
            need_shift = (prod_sig >> 7) & 1
            out_man = (prod_sig & 0x7F) if need_shift else ((prod_sig & 0x3F) << 1)
            out_exp = a_exp + b_exp + need_shift - 1
            lut[idx] = (out_sign << 12) | ((out_exp & 0x1F) << 7) | out_man
    return lut


LUT = _build_lut()


# Core arithmetic (pure Python, matches RTL with intWidth=32)
def _wrap(v: int) -> int:
    v &= INT_MASK
    if v & INT_SIGN:
        v -= 1 << INT_WIDTH
    return v


def _prod_to_aligned(prod: int, anchor: int) -> int:
    exp_bits = (prod >> 7) & 0x1F
    if exp_bits == 0:
        return 0
    sign = (prod >> 12) & 1
    man = prod & 0x7F
    # left_pad = INT_WIDTH - sig_width = 32 - 8 = 24
    sig_wide = (0x80 | man) << 24
    rshift = anchor - (exp_bits - PROD_BIAS)
    shifted = sig_wide << (-rshift) if rshift < 0 else sig_wide >> rshift
    mag = shifted & INT_MASK
    return _wrap(-mag if sign else mag)


def _ieee_to_aligned(ieee: int, mant_bits: int, ieee_bias: int, anchor: int) -> int:
    """Generic IEEE-to-aligned for E4M3-bias or BF16-psum."""
    width_total = 1 + {3: 4, 7: 8}[mant_bits] + mant_bits  # 8 or 16
    sign = (ieee >> (width_total - 1)) & 1
    exp_mask = (1 << {3: 4, 7: 8}[mant_bits]) - 1
    exp_field = (ieee >> mant_bits) & exp_mask
    frac = ieee & ((1 << mant_bits) - 1)
    if exp_field == 0 and frac == 0:
        return 0
    unb_exp = exp_field - ieee_bias
    full_sig = frac if exp_field == 0 else ((1 << mant_bits) | frac)
    rshift = anchor - unb_exp - (INT_WIDTH - 1 - mant_bits)
    if rshift >= INT_WIDTH:
        mag = 0
    elif rshift >= 0:
        mag = full_sig >> rshift
    elif rshift > -INT_WIDTH:
        mag = full_sig << (-rshift)
    else:
        mag = 0
    mag &= INT_MASK
    return _wrap(-mag if sign else mag)


def _bias_to_aligned(bits: int, anchor: int) -> int:
    return _ieee_to_aligned(bits & 0xFF, mant_bits=3, ieee_bias=7, anchor=anchor)


def _psum_to_aligned(bits: int, anchor: int) -> int:
    return _ieee_to_aligned(bits & 0xFFFF, mant_bits=7, ieee_bias=127, anchor=anchor)


def _f64_to_bf16_rne(f: float) -> int:
    """Convert float64 directly to BF16 with a single RNE rounding step.

    This avoids the double-rounding problem of float64→float32→BF16.
    The RTL uses HardFloat INToRecFN which does a single round from the
    full-precision integer significand to BF16, so the Python model must
    do the same.
    """
    if f == 0.0:
        return 0x8000 if math.copysign(1.0, f) < 0 else 0

    d_bits = struct.unpack(">Q", struct.pack(">d", f))[0]
    sign = (d_bits >> 63) & 1
    d_exp = (d_bits >> 52) & 0x7FF
    d_frac = d_bits & ((1 << 52) - 1)

    if d_exp == 0x7FF:
        if d_frac:
            return 0x7FC0            # NaN → canonical BF16 NaN
        return 0xFF80 if sign else 0x7F80  # ±Inf

    if d_exp == 0:
        return 0                     # float64 subnormal → zero in BF16

    # Unbiased exponent (double bias = 1023, BF16 bias = 127)
    unb_exp = d_exp - 1023
    full_sig = (1 << 52) | d_frac   # 53 bits: 1.frac

    # Round 53-bit significand to 8-bit (1 implicit + 7 mantissa).
    # Drop the bottom 45 bits with RNE.
    shift = 52 - 7                   # = 45
    top8 = full_sig >> shift         # in [128, 255]
    guard = (full_sig >> (shift - 1)) & 1
    sticky = 1 if (full_sig & ((1 << (shift - 1)) - 1)) else 0
    lsb = top8 & 1
    top8 += guard & (sticky | lsb)   # RNE

    if top8 >= 256:                  # carry out of rounding
        top8 >>= 1
        unb_exp += 1

    biased_exp = unb_exp + 127
    if biased_exp >= 0xFF:
        return 0xFF7F if sign else 0x7F7F   # clamp to max finite (matches RTL sanitize)
    if biased_exp <= 0:
        return 0                             # underflow → zero (subnormals flushed)

    return (sign << 15) | (biased_exp << 7) | (top8 & 0x7F)


def _aligned_to_bf16(int_in: int, anchor: int) -> int:
    """Aligned-int to BF16, matching RTL AlignedIntToIEEE.

    Uses float64 as an exact intermediate (32-bit int fits in 52-bit
    mantissa), then a single RNE round to BF16 — no float32 detour.
    """
    val = _wrap(int_in)
    if val == 0:
        return 0
    f = math.ldexp(float(val), anchor - (INT_WIDTH - 1))
    return _f64_to_bf16_rne(f)


def _sanitize_bf16(bits: int) -> int:
    bits &= 0xFFFF
    exp = (bits >> 7) & 0xFF
    if exp == 0:
        return bits if (bits & 0x7F) == 0 else 0
    if exp == 0xFF:
        if bits & 0x7F:
            return 0
        return 0xFF7F if (bits & 0x8000) else 0x7F7F
    return bits


def _rshift4_rne(x: int) -> int:
    trunc = (x >> 4) & 0xF
    guard = (x >> 3) & 1
    sticky = 1 if (x & 0x7) != 0 else 0
    return trunc + (guard & (sticky | (trunc & 1)))


def _bf16_to_e4m3(bf16: int, scale_exp: int) -> int:
    bf16 &= 0xFFFF
    sign = (bf16 >> 15) & 1
    exp_bf16 = (bf16 >> 7) & 0xFF
    frac_bf16 = bf16 & 0x7F
    if exp_bf16 == 0:
        return 0
    if exp_bf16 == 0xFF:
        if frac_bf16:
            return 0
        return 0xFE if sign else 0x7E
    scaled_unb = exp_bf16 - 127 + scale_exp
    rn = _rshift4_rne(0x80 | frac_bf16)
    if rn == 16:
        final_exp = scaled_unb + 1
        norm_mant = 0
    else:
        final_exp = scaled_unb
        norm_mant = (rn - 8) & 0x7
    if final_exp > 8:
        return 0xFE if sign else 0x7E
    if final_exp >= -6:
        return ((sign & 1) << 7) | (((final_exp + 7) & 0xF) << 3) | (norm_mant & 0x7)
    return 0


# Compute one lane (single dot-product beat)
def compute_lane(
    act: list[int],          # VEC_LEN activation bytes
    wgt: list[int],          # VEC_LEN weight bytes
    bias_byte: int,          # E4M3 bias byte
    psum_bf16: int,          # BF16 psum bits
    addend_sel: int,         # 0=UseAct(none), 1=UseBias, 2=UsePsum
    out_fmt_sel: int = 0,    # 0=OutBF16, 1=OutE4M3
    scale_exp: int = 0,      # signed scale for E4M3 output
) -> int:
    vec_len = len(act)

    # Pass 1 — products + max exponent
    prods = []
    max_pe = SENTINEL
    for i in range(vec_len):
        a = act[i] & 0xFF
        w = wgt[i] & 0xFF
        prod = LUT[(a << 8) | w]
        prods.append(prod)
        eb = (prod >> 7) & 0x1F
        pe = SENTINEL if eb == 0 else (eb - PROD_BIAS)
        if pe > max_pe:
            max_pe = pe

    # Addend exponent
    addend_exp = SENTINEL
    b_bits = bias_byte & 0xFF
    p_bits = psum_bf16 & 0xFFFF

    if addend_sel == 1:
        bf = (b_bits >> 3) & 0xF
        if ((b_bits >> 3) & 0x1F) != 0:
            addend_exp = bf - 7
    elif addend_sel == 2:
        pef = (p_bits >> 7) & 0xFF
        pfr = p_bits & 0x7F
        if not (pef == 0 and pfr == 0):
            addend_exp = pef - 127

    anchor = max(max_pe, addend_exp) + ANCHOR_HR

    # Pass 2 — accumulate products with full precision.
    # RTL uses treeReduce with +& (width-extending add), so intermediate
    # sums can exceed intWidth. Only the final (prodSum + addend) result
    # is truncated to intWidth bits.
    prod_sum = 0
    for i in range(vec_len):
        prod_sum += _prod_to_aligned(prods[i], anchor)

    if addend_sel == 1:
        addend_int = _bias_to_aligned(b_bits, anchor)
    elif addend_sel == 2:
        addend_int = _psum_to_aligned(p_bits, anchor)
    else:
        addend_int = 0

    total_int = _wrap(prod_sum + addend_int)
    bf16_bits = _aligned_to_bf16(total_int, anchor)

    # Output stage
    s = _sanitize_bf16(bf16_bits)
    if out_fmt_sel == 0:
        return s & 0xFFFF
    return _bf16_to_e4m3(s, scale_exp) & 0xFF


# BF16 helpers
def bf16_to_float(b: int) -> float:
    return struct.unpack(">f", struct.pack(">I", (b & 0xFFFF) << 16))[0]


# File parsers
@dataclass
class SingleBeatCase:
    case_id: int
    case_type: str
    addend_sel: int
    act: list[int]
    wgt: list[list[int]]   # [lane][vec_len]
    bias: list[int]        # [num_lanes]
    psum: list[int]        # [num_lanes] BF16 bits
    scale_exp: int = 0     # signed, for E4M3 output
    exp_bf16: list[int] = field(default_factory=list)
    exp_e4m3: list[int] = field(default_factory=list)


def parse_single_beat_vectors(path: Path, is_e4m3: bool = False) -> list[SingleBeatCase]:
    """Parse mxu_vectors.txt or mxu_vectors_e4m3.txt."""
    cases: list[SingleBeatCase] = []
    current: Optional[SingleBeatCase] = None

    def flush():
        nonlocal current
        if current is not None:
            cases.append(current)
            current = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            flush()
            parts = line[1:].strip().split()
            current = SingleBeatCase(
                case_id=int(parts[0]),
                case_type=parts[1] if len(parts) > 1 else "",
                addend_sel=0,
                act=[],
                wgt=[None] * NUM_LANES,  # type: ignore
                bias=[0] * NUM_LANES,
                psum=[0] * NUM_LANES,
            )
            continue

        if current is None:
            continue

        parts = line.split()
        key = parts[0]

        if key == "sel":
            current.addend_sel = int(parts[1])
        elif key == "act":
            current.act = [int(x, 16) for x in parts[1:]]
        elif key == "wgt":
            lane = int(parts[1])
            current.wgt[lane] = [int(x, 16) for x in parts[2:]]
        elif key == "bias":
            current.bias = [int(x, 16) for x in parts[1:]]
        elif key == "psum":
            current.psum = [int(x, 16) for x in parts[1:]]
        elif key == "scale_exp":
            val = int(parts[1], 16)
            # sign-extend from 8-bit
            current.scale_exp = val - 256 if val >= 128 else val
        elif key == "exp":
            current.exp_bf16 = [int(x, 16) for x in parts[1:]]
        elif key == "exp_e4m3":
            current.exp_e4m3 = [int(x, 16) for x in parts[1:]]
        elif key == "recon_bf16":
            pass  # informational only

    flush()
    return cases


@dataclass
class FullMatmulCase:
    case_id: int
    num_rows: int
    num_tiles: int
    wgt: list[list[list[int]]]   # [tile][lane][vec_len]
    act: list[list[list[int]]]   # [tile][row][vec_len]


def parse_full_matmul_vectors(path: Path) -> list[FullMatmulCase]:
    """Parse mxu_full_matmul_vectors.txt."""
    cases: list[FullMatmulCase] = []
    current: Optional[FullMatmulCase] = None
    nr = 32
    nt = 1

    def flush():
        nonlocal current
        if current is not None:
            cases.append(current)
            current = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            flush()
            parts = line[1:].strip().split()
            cid = int(parts[0])
            current = FullMatmulCase(case_id=cid, num_rows=nr, num_tiles=nt, wgt=[], act=[])
            continue

        if current is None:
            parts = line.split()
            if parts[0] == "num_rows":
                nr = int(parts[1])
            elif parts[0] == "num_tiles":
                nt = int(parts[1])
            continue

        parts = line.split()
        key = parts[0]

        if key == "num_rows":
            current.num_rows = int(parts[1])
            nr = current.num_rows
        elif key == "num_tiles":
            current.num_tiles = int(parts[1])
            nt = current.num_tiles
        elif key == "wgt":
            tile = int(parts[1])
            lane = int(parts[2])
            data = [int(x, 16) for x in parts[3:]]
            while len(current.wgt) <= tile:
                current.wgt.append([None] * NUM_LANES)  # type: ignore
            current.wgt[tile][lane] = data
        elif key == "act":
            tile = int(parts[1])
            row = int(parts[2])
            data = [int(x, 16) for x in parts[3:]]
            while len(current.act) <= tile:
                current.act.append([])
            while len(current.act[tile]) <= row:
                current.act[tile].append(None)  # type: ignore
            current.act[tile][row] = data

    flush()
    return cases


@dataclass
class RTLOutput:
    case_id: int
    row: int
    lane: int
    actual_hex: int


def parse_rtl_outputs_bf16(path: Path) -> list[RTLOutput]:
    """Parse ipt_rtl_vector_outputs.txt or ipt_rtl_full_matmul_outputs.txt (BF16 format)."""
    results = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # ipt_rtl_vector_outputs.txt:       case_id case_type row lane actual_hex expected_hex ...
        # ipt_rtl_full_matmul_outputs.txt:  case_id           row lane actual_hex expected_hex ...
        # Detect format by trying to parse
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        # Find actual_hex (starts with 0x)
        hex_indices = [i for i, p in enumerate(parts) if p.startswith("0x")]
        if len(hex_indices) < 1:
            continue
        actual_idx = hex_indices[0]
        # row and lane are the two integers before actual_hex
        # For vector outputs: case_id type row lane 0xAAAA ...
        # For matmul outputs: case_id row lane 0xAAAA ...
        lane_idx = actual_idx - 1
        row_idx = actual_idx - 2
        try:
            row = int(parts[row_idx])
            lane = int(parts[lane_idx])
        except (ValueError, IndexError):
            continue

        actual = int(parts[actual_idx], 16)
        results.append(RTLOutput(cid, row, lane, actual))
    return results


def parse_rtl_outputs_e4m3(path: Path) -> list[RTLOutput]:
    """Parse ipt_rtl_vector_e4m3_outputs.txt."""
    results = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        hex_indices = [i for i, p in enumerate(parts) if p.startswith("0x")]
        if len(hex_indices) < 1:
            continue
        actual_idx = hex_indices[0]
        lane_idx = actual_idx - 1
        row_idx = actual_idx - 2
        try:
            row = int(parts[row_idx])
            lane = int(parts[lane_idx])
        except (ValueError, IndexError):
            continue
        actual = int(parts[actual_idx], 16)
        results.append(RTLOutput(cid, row, lane, actual))
    return results


# Comparison helpers. Functional Model needs to match RTL exactly.
BF16_ABS_TOL = 0x0000 # 0x0040
BF16_REL_TOL = 0.00   # 0 % relative


def bf16_match(model: int, rtl: int) -> tuple[bool, float, float]:
    """Compare two BF16 values.  Returns (ok, abs_err, rel_err)."""
    m_mag = model & 0x7FFF
    r_mag = rtl & 0x7FFF
    if m_mag <= BF16_ABS_TOL and r_mag <= BF16_ABS_TOL:
        return True, 0.0, 0.0
    mf = bf16_to_float(model)
    rf = bf16_to_float(rtl)
    ae = abs(mf - rf)
    re = ae / max(abs(rf), 1e-30)
    return model == rtl or re < BF16_REL_TOL, ae, re


def e4m3_match(model: int, rtl: int) -> bool:
    return (model & 0xFF) == (rtl & 0xFF)


# Test: single-beat BF16
def test_single_beat_bf16(stim_path: Path, rtl_path: Path) -> tuple[int, int]:
    print(f"\n{'='*72}")
    print(f"  TEST: Single-beat BF16")
    print(f"  Stimulus : {stim_path}")
    print(f"  RTL      : {rtl_path}")
    print(f"{'='*72}")

    cases = parse_single_beat_vectors(stim_path)
    rtl_outputs = parse_rtl_outputs_bf16(rtl_path)

    # Group RTL outputs by (case_id, row, lane)
    rtl_map: dict[tuple[int, int, int], int] = {}
    for o in rtl_outputs:
        rtl_map[(o.case_id, o.row, o.lane)] = o.actual_hex

    total_pass = 0
    total_fail = 0

    for tc in cases:
        # Compute model output for one beat (same for all rows)
        model_out = []
        for lane in range(NUM_LANES):
            v = compute_lane(
                tc.act, tc.wgt[lane],
                tc.bias[lane], tc.psum[lane],
                tc.addend_sel, out_fmt_sel=0,
            )
            model_out.append(v)

        # Compare against RTL for every row present in RTL outputs
        rows_tested = set()
        case_pass = 0
        case_fail = 0
        for lane in range(NUM_LANES):
            for row in range(TILE_ROWS):
                key = (tc.case_id, row, lane)
                if key not in rtl_map:
                    continue
                rows_tested.add(row)
                rtl_val = rtl_map[key]
                ok, ae, re = bf16_match(model_out[lane], rtl_val)
                if ok:
                    case_pass += 1
                else:
                    case_fail += 1
                    mf = bf16_to_float(model_out[lane])
                    rf = bf16_to_float(rtl_val)
                    print(
                        f"  FAIL case={tc.case_id} ({tc.case_type}) row={row} lane={lane}: "
                        f"model=0x{model_out[lane]:04x}({mf:.6g}) "
                        f"rtl=0x{rtl_val:04x}({rf:.6g}) "
                        f"rel_err={re:.4e}"
                    )

        n_rows = len(rows_tested) if rows_tested else 1
        total = case_pass + case_fail
        status = "PASS" if case_fail == 0 else "FAIL"
        print(
            f"  Case {tc.case_id:3d} ({tc.case_type:8s}): "
            f"{case_pass}/{total} [{status}] "
            f"({n_rows} rows)"
        )
        total_pass += case_pass
        total_fail += case_fail

    return total_pass, total_fail


# Test: single-beat E4M3
def test_single_beat_e4m3(stim_path: Path, rtl_path: Path) -> tuple[int, int]:
    print(f"\n{'='*72}")
    print(f"  TEST: Single-beat E4M3")
    print(f"  Stimulus : {stim_path}")
    print(f"  RTL      : {rtl_path}")
    print(f"{'='*72}")

    cases = parse_single_beat_vectors(stim_path, is_e4m3=True)
    rtl_outputs = parse_rtl_outputs_e4m3(rtl_path)

    rtl_map: dict[tuple[int, int, int], int] = {}
    for o in rtl_outputs:
        rtl_map[(o.case_id, o.row, o.lane)] = o.actual_hex

    total_pass = 0
    total_fail = 0

    for tc in cases:
        model_out = []
        for lane in range(NUM_LANES):
            v = compute_lane(
                tc.act, tc.wgt[lane],
                tc.bias[lane], tc.psum[lane],
                tc.addend_sel,
                out_fmt_sel=1,
                scale_exp=tc.scale_exp,
            )
            model_out.append(v)

        case_pass = 0
        case_fail = 0
        rows_tested = set()
        for lane in range(NUM_LANES):
            for row in range(TILE_ROWS):
                key = (tc.case_id, row, lane)
                if key not in rtl_map:
                    continue
                rows_tested.add(row)
                rtl_val = rtl_map[key]
                ok = e4m3_match(model_out[lane], rtl_val)
                if ok:
                    case_pass += 1
                else:
                    case_fail += 1
                    print(
                        f"  FAIL case={tc.case_id} ({tc.case_type}) row={row} lane={lane}: "
                        f"model=0x{model_out[lane]:02x} rtl=0x{rtl_val:02x}"
                    )

        n_rows = len(rows_tested) if rows_tested else 1
        total = case_pass + case_fail
        status = "PASS" if case_fail == 0 else "FAIL"
        print(
            f"  Case {tc.case_id:3d} ({tc.case_type:8s}): "
            f"{case_pass}/{total} [{status}] "
            f"({n_rows} rows)"
        )
        total_pass += case_pass
        total_fail += case_fail

    return total_pass, total_fail


# Test: full matmul (multi-tile accumulation)
def test_full_matmul(stim_path: Path, rtl_path: Path) -> tuple[int, int]:
    print(f"\n{'='*72}")
    print(f"  TEST: Full Matmul (multi-tile)")
    print(f"  Stimulus : {stim_path}")
    print(f"  RTL      : {rtl_path}")
    print(f"{'='*72}")

    cases = parse_full_matmul_vectors(stim_path)
    rtl_outputs = parse_rtl_outputs_bf16(rtl_path)

    rtl_map: dict[tuple[int, int, int], int] = {}
    for o in rtl_outputs:
        rtl_map[(o.case_id, o.row, o.lane)] = o.actual_hex

    total_pass = 0
    total_fail = 0

    for tc in cases:
        nt = tc.num_tiles
        nr = tc.num_rows
        print(f"  Case {tc.case_id}: {nr} rows × {nt} tiles ...")
        t0 = time.time()

        # model_psum[row][lane] = BF16 bits after all tiles
        model_psum = [[0] * NUM_LANES for _ in range(nr)]

        for tile_idx in range(nt):
            if tile_idx % 64 == 0:
                elapsed = time.time() - t0
                print(f"    tile {tile_idx}/{nt}  ({elapsed:.1f}s)", end="\r")

            tile_wgt = tc.wgt[tile_idx]   # [lane][vec_len]
            tile_act = tc.act[tile_idx]   # [row][vec_len]
            first_tile = (tile_idx == 0)

            for row in range(nr):
                for lane in range(NUM_LANES):
                    if first_tile:
                        # No addend on first tile
                        addend_sel = 0
                        psum_bits = 0
                    else:
                        # Accumulate previous psum
                        addend_sel = 2
                        psum_bits = model_psum[row][lane]

                    result = compute_lane(
                        tile_act[row], tile_wgt[lane],
                        0, psum_bits,
                        addend_sel, out_fmt_sel=0,
                    )
                    model_psum[row][lane] = result

        elapsed = time.time() - t0
        print(f"    Computed {nt} tiles in {elapsed:.1f}s                    ")

        case_pass = 0
        case_fail = 0
        for row in range(nr):
            for lane in range(NUM_LANES):
                key = (tc.case_id, row, lane)
                if key not in rtl_map:
                    continue
                rtl_val = rtl_map[key]
                ok, ae, re = bf16_match(model_psum[row][lane], rtl_val)
                if ok:
                    case_pass += 1
                else:
                    case_fail += 1
                    mf = bf16_to_float(model_psum[row][lane])
                    rf = bf16_to_float(rtl_val)
                    print(
                        f"  FAIL case={tc.case_id} row={row} lane={lane}: "
                        f"model=0x{model_psum[row][lane]:04x}({mf:.6g}) "
                        f"rtl=0x{rtl_val:04x}({rf:.6g}) "
                        f"rel_err={re:.4e}"
                    )

        total = case_pass + case_fail
        status = "PASS" if case_fail == 0 else "FAIL"
        print(f"  Case {tc.case_id}: {case_pass}/{total} [{status}]")
        total_pass += case_pass
        total_fail += case_fail

    return total_pass, total_fail


# Main
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <resource_dir>")
        print(f"  resource_dir should contain the stimulus and RTL output files.")
        sys.exit(1)

    rd = Path(sys.argv[1])
    if not rd.is_dir():
        print(f"ERROR: {rd} is not a directory")
        sys.exit(1)

    grand_pass = 0
    grand_fail = 0

    # ── Single-beat BF16 ──
    stim = rd / "mxu_vectors.txt"
    rtl = rd / "ipt_rtl_vector_outputs.txt"
    if stim.exists() and rtl.exists():
        p, f = test_single_beat_bf16(stim, rtl)
        grand_pass += p
        grand_fail += f
    else:
        missing = [x.name for x in (stim, rtl) if not x.exists()]
        print(f"\n  SKIP single-beat BF16 — missing: {', '.join(missing)}")

    # ── Single-beat E4M3 ──
    stim = rd / "mxu_vectors_e4m3.txt"
    rtl = rd / "ipt_rtl_vector_e4m3_outputs.txt"
    if stim.exists() and rtl.exists():
        p, f = test_single_beat_e4m3(stim, rtl)
        grand_pass += p
        grand_fail += f
    else:
        missing = [x.name for x in (stim, rtl) if not x.exists()]
        print(f"\n  SKIP single-beat E4M3 — missing: {', '.join(missing)}")

    # ── Full matmul ──
    stim = rd / "mxu_full_matmul_vectors.txt"
    rtl = rd / "ipt_rtl_full_matmul_outputs.txt"
    if stim.exists() and rtl.exists():
        p, f = test_full_matmul(stim, rtl)
        grand_pass += p
        grand_fail += f
    else:
        missing = [x.name for x in (stim, rtl) if not x.exists()]
        print(f"\n  SKIP full matmul — missing: {', '.join(missing)}")

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"  GRAND TOTAL: {grand_pass} passed, {grand_fail} failed")
    if grand_fail > 0:
        print(f"  STATUS: *** FAIL ***")
    else:
        print(f"  STATUS: ALL PASS")
    print(f"{'='*72}")

    sys.exit(1 if grand_fail > 0 else 0)


if __name__ == "__main__":
    main()
