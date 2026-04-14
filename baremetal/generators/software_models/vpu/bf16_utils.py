"""BF16 bit-level helpers.

Pulls round-to-nearest-even / round-trip primitives from the existing IPT
module and adds the bitwise sign-magnitude compare helpers that mirror
sp26FPUnits.VectorParam.compareReturnMax / compareReturnMin
(VectorParam.scala:98-110). Using Python's built-in max()/min() here would
silently diverge from the RTL on ties, ±0, and the bitwise NaN ordering.

There is intentionally no `bf16_add` / `bf16_sub` here. AddSubSumVec.scala
runs every add/sub/rsum through `VectorAddRecFN(8, BF16.sigWidth + 16 = 24)`
— i.e. FP32-precision adders — and extracts the BF16 result via a *raw*
upper-16-bit slice (`fNFromRecFN(8, 24, v)(31, 16)`), NOT a second
RNE-to-BF16 rounding pass. A naive `bf16_to_f32 -> f32_add -> f32_to_bf16_rne`
helper double-rounds on the half-ULP cases that AddRecFN has already pinned
down; use `fp32_bits_from_bf16 + fp32_bits_add/sub + bf16_upper_half_of_fp32_bits`
instead. `bf16_mul` is fine — `MulRec.scala` lives entirely at BF16 precision
(`MulRawFN(8,8)` + `RoundRawFNToRecFN(8,8,0)`), which the FP32-mul + RNE
round-trip matches exactly.
"""

from __future__ import annotations

import struct

from software_models.mxu1_ipt.fp_formats import (  # type: ignore
    BF16_MAX_NEG,
    BF16_MAX_POS,
    bf16_bits_to_f32,
    f32_to_bf16_bits_rne,
    float_to_u32,
    u32_to_float,
)

__all__ = [
    "BF16_MAX_NEG",
    "BF16_MAX_POS",
    "bf16_bits_to_f32",
    "f32_to_bf16_bits_rne",
    "float_to_u32",
    "u32_to_float",
    "bf16_ordered_key",
    "compare_return_max",
    "compare_return_min",
    "bf16_mul",
    "bf16_neg",
    "bf16_sign",
    "bf16_exp_field",
    "bf16_mant_field",
    "bf16_is_zero",
    "bf16_is_sub",
    "bf16_is_inf",
    "bf16_is_nan",
    "bf16_upper_half_of_fp32_bits",
    "fp32_bits_add",
    "fp32_bits_sub",
    "fp32_bits_from_bf16",
]


def bf16_ordered_key(bits: int) -> int:
    """Bitwise sign-magnitude -> total-order transform.

    Mirrors the Chisel expression:
        aOrdered = Mux(a(15) === 1.U, ~a, a ^ 0x8000.U)
    so that 0x0000 (+0) maps to 0x8000, 0x8000 (-0) maps to 0x7FFF, and both
    NaN and Inf land at the top/bottom of the ordering per their bit pattern.
    """
    bits &= 0xFFFF
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return bits ^ 0x8000


def compare_return_max(a: int, b: int) -> int:
    """Bitwise max with b-on-tie (VectorParam.scala:105-110)."""
    return a & 0xFFFF if bf16_ordered_key(a) > bf16_ordered_key(b) else b & 0xFFFF


def compare_return_min(a: int, b: int) -> int:
    """Bitwise min with b-on-tie (VectorParam.scala:98-103)."""
    return a & 0xFFFF if bf16_ordered_key(a) < bf16_ordered_key(b) else b & 0xFFFF


def bf16_sign(bits: int) -> int:
    return (bits >> 15) & 1


def bf16_exp_field(bits: int) -> int:
    return (bits >> 7) & 0xFF


def bf16_mant_field(bits: int) -> int:
    return bits & 0x7F


def bf16_is_zero(bits: int) -> bool:
    return (bits & 0x7FFF) == 0


def bf16_is_sub(bits: int) -> bool:
    return bf16_exp_field(bits) == 0 and bf16_mant_field(bits) != 0


def bf16_is_inf(bits: int) -> bool:
    return bf16_exp_field(bits) == 0xFF and bf16_mant_field(bits) == 0


def bf16_is_nan(bits: int) -> bool:
    return bf16_exp_field(bits) == 0xFF and bf16_mant_field(bits) != 0


def bf16_neg(bits: int) -> int:
    return (bits ^ 0x8000) & 0xFFFF


def _bf16_to_fp32_bits(bits: int) -> int:
    """Zero-pad BF16 -> FP32 bit pattern."""
    return (bits & 0xFFFF) << 16


def _fp32_bits_to_f32(bits: int) -> float:
    return u32_to_float(bits & 0xFFFFFFFF)


def _f32_to_fp32_bits(x: float) -> int:
    return float_to_u32(x)


def fp32_bits_from_bf16(bits: int) -> int:
    """Alias for _bf16_to_fp32_bits exposed to ColAddVec and friends."""
    return _bf16_to_fp32_bits(bits)


def bf16_upper_half_of_fp32_bits(fp32_bits: int) -> int:
    """Extract BF16 as the upper 16 bits of an FP32 bit pattern.

    Matches VectorEngine.scala:320-322 where ColAddVec's widened-recFN result
    is converted via `fNFromRecFN(8, 24, res)(31, 16)` — a *raw slice* with
    no second rounding pass. Do NOT replace with f32_to_bf16_bits_rne: that
    re-rounds and will diverge from the RTL on the mantissa-boundary cases
    that an FP32 AddRecFN has already rounded once.
    """
    return (fp32_bits >> 16) & 0xFFFF


def _f32_to_fp32_bits_clamped(s: float) -> int:
    """Pack a Python FP64 into FP32 bits with overflow clamped to ±inf.

    `struct.pack('<f', s)` raises OverflowError for finite FP64 values
    outside FP32 range — but the RTL AddRecFN saturates such results to
    ±inf, so we mirror that here. Inf and NaN inputs pack fine via struct.
    """
    try:
        packed = struct.pack("<f", s)
    except OverflowError:
        return (0x80000000 if s < 0 else 0) | 0x7F800000
    return struct.unpack("<I", packed)[0] & 0xFFFFFFFF


def fp32_bits_add(a_bits: int, b_bits: int) -> int:
    """Add two FP32 bit patterns, returning the FP32 bit pattern of the sum.

    The Python intermediate is FP64, but for any FP32+FP32 add the exact
    sum fits in 53 mantissa bits, so the FP64 add is exact and the cast
    back through `struct('<f')` is the same single RNE-to-FP32 rounding
    that VectorAddRecFN(8, 24) performs in hardware. No double-rounding.
    Overflow is clamped to ±inf to match AddRecFN.
    """
    a = _fp32_bits_to_f32(a_bits & 0xFFFFFFFF)
    b = _fp32_bits_to_f32(b_bits & 0xFFFFFFFF)
    return _f32_to_fp32_bits_clamped(a + b)


def fp32_bits_sub(a_bits: int, b_bits: int) -> int:
    """Subtract two FP32 bit patterns. Same rounding properties as add.

    AddSubSumVec.scala flips `subOp` on the AddRecFN, which negates b's
    sign before adding. We mirror that with a real FP32 subtraction; the
    result is bit-identical because IEEE-754 add(a, -b) == sub(a, b).
    """
    a = _fp32_bits_to_f32(a_bits & 0xFFFFFFFF)
    b = _fp32_bits_to_f32(b_bits & 0xFFFFFFFF)
    return _f32_to_fp32_bits_clamped(a - b)


def bf16_mul(a: int, b: int) -> int:
    """BF16 multiply matching MulRec.scala (MulRawFN + RoundRawFNToRecFN, BF16).

    BF16 inputs zero-pad to FP32. The exact product of two 9-effective-bit
    FP32 numbers has at most 18 significand bits, well within FP32's 24,
    so the FP32 multiply is exact and `f32_to_bf16_bits_rne` performs the
    one and only rounding — same as `RoundRawFNToRecFN(8, 8, 0)` does.

    Overflow path: BF16's 8-bit exponent matches FP32's, so two BF16 max
    operands produce a product (~2^254) far outside FP32 range. The RTL
    saturates such results to BF16 ±inf, which matches IEEE-754 multiply
    overflow; clamp the FP64 intermediate to FP32 ±inf before rounding so
    `f32_to_bf16_bits_rne` doesn't blow up packing the FP64 into FP32.
    """
    af = bf16_bits_to_f32(a)
    bf = bf16_bits_to_f32(b)
    p = af * bf
    if p > _FP32_MAX:
        p = float("inf")
    elif p < -_FP32_MAX:
        p = float("-inf")
    return f32_to_bf16_bits_rne(p)


_FP32_MAX = 3.4028234663852886e38
