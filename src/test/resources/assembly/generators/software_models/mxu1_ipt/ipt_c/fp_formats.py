from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import struct


_pack = struct.pack
_unpack = struct.unpack


@dataclass(frozen=True)
class AtlasFPType:
    name: str
    ieeeWidth: int
    expWidth: int
    mantissaBits: int
    ieeeBias: int

    @property
    def sigWidth(self) -> int:
        return 1 + self.mantissaBits


E4M3 = AtlasFPType("E4M3", ieeeWidth=8, expWidth=4, mantissaBits=3, ieeeBias=7)
BF16 = AtlasFPType("BF16", ieeeWidth=16, expWidth=8, mantissaBits=7, ieeeBias=127)


@dataclass(frozen=True)
class E4M3ProdFmtDesc:
    width: int = 13
    expWidth: int = 5
    mantissaBits: int = 7
    bias: int = 13

    @property
    def sigWidth(self) -> int:
        return 1 + self.mantissaBits


E4M3ProdFmt = E4M3ProdFmtDesc()


class AddendSel(Enum):
    UseAct = 0
    UseBias = 1
    UsePsum = 2


class OutputFmtSel(Enum):
    OutBF16 = 0
    OutE4M3 = 1


BF16_MAX_POS = 0x7F7F
BF16_MAX_NEG = 0xFF7F
E4M3_MAX_POS = 0x7E
E4M3_MAX_NEG = 0xFE


F32_SIGN_MASK = 0x80000000
F32_EXP_MASK = 0x7F800000
F32_FRAC_MASK = 0x007FFFFF

_E4M3_BIAS = E4M3.ieeeBias


@dataclass(frozen=True)
class DecodedFloat:
    sign: int
    exp_field: int
    frac: int
    is_zero: bool
    is_sub: bool
    is_inf: bool
    is_nan: bool
    unb_exp: int | None
    sig: float | None
    value: float | None


def u32_to_float(x: int) -> float:
    return _unpack(">f", _pack(">I", x & 0xFFFFFFFF))[0]


def float_to_u32(x: float) -> int:
    return _unpack(">I", _pack(">f", float(x)))[0]


def sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def clamp_signed(value: int, bits: int) -> int:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def wrap_signed(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        value -= 1 << bits
    return value


def _decode_e4m3_uncached(bits: int) -> DecodedFloat:
    bits &= 0xFF
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    frac = bits & 0x7

    if exp == 0:
        if frac == 0:
            v = -0.0 if sign else 0.0
            return DecodedFloat(sign, exp, frac, True, False, False, False, None, None, v)

        unb_exp = 1 - _E4M3_BIAS
        sig = frac * 0.125
        mag = sig * (2.0 ** unb_exp)
        v = -mag if sign else mag
        return DecodedFloat(sign, exp, frac, False, True, False, False, unb_exp, sig, v)

    if exp == 0xF:
        if frac == 0:
            v = float("-inf") if sign else float("inf")
            return DecodedFloat(sign, exp, frac, False, False, True, False, None, None, v)
        return DecodedFloat(sign, exp, frac, False, False, False, True, None, None, float("nan"))

    unb_exp = exp - _E4M3_BIAS
    sig = 1.0 + frac * 0.125
    mag = sig * (2.0 ** unb_exp)
    v = -mag if sign else mag
    return DecodedFloat(sign, exp, frac, False, False, False, False, unb_exp, sig, v)


_E4M3_DECODE_LUT = tuple(_decode_e4m3_uncached(i) for i in range(256))


def decode_e4m3(bits: int) -> DecodedFloat:
    return _E4M3_DECODE_LUT[bits & 0xFF]


def encode_e4m3_normal(sign: int, unb_exp: int, mant3: int) -> int:
    return ((sign & 1) << 7) | (((unb_exp + _E4M3_BIAS) & 0xF) << 3) | (mant3 & 0x7)


def f32_to_bf16_bits_rne(x: float) -> int:
    u = float_to_u32(x)
    exp = (u >> 23) & 0xFF

    if exp == 0xFF:
        frac = u & 0x7FFFFF
        if frac != 0:
            return 0x7FC0
        return (u >> 16) & 0xFFFF

    upper = u >> 16
    rounded = u + 0x7FFF + (upper & 1)
    return (rounded >> 16) & 0xFFFF


def bf16_bits_to_f32(bits: int) -> float:
    return u32_to_float((bits & 0xFFFF) << 16)


def round_right_shift4_rne(x: int) -> int:
    trunc = (x >> 4) & 0xF
    guard = (x >> 3) & 1
    sticky = int((x & 0x7) != 0)
    return trunc + (guard & (sticky | (trunc & 1)))


def sanitize_bf16(bits: int) -> int:
    bits &= 0xFFFF
    exp = (bits >> 7) & 0xFF

    if exp == 0:
        frac = bits & 0x7F
        return bits if frac == 0 else 0

    if exp == 0xFF:
        frac = bits & 0x7F
        if frac != 0:
            return 0
        return BF16_MAX_NEG if (bits & 0x8000) else BF16_MAX_POS

    return bits
