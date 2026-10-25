"""FP8 E4M3 <-> BF16 conversions of FP8Pack.scala and FP8Unpack.scala, one value at a time.

They are npu_model's Blackwell conversions (npu_model/configs/cvt.py, ISA v2 section 5.2).
e is the raw e-register value; both conversions multiply by 2^(e - 127) and round once.
"""

from __future__ import annotations

from npu_model.configs import cvt


def bf16_to_e4m3_byte(bf16: int, e: int) -> int:
    """E4M3_RN_SATFINITE(x * 2^(e - 127)) of BF16 bits x."""
    return int(cvt.e4m3_rn_satfinite((int(bf16) & 0xFFFF) << 16, int(e)))


def e4m3_byte_to_bf16(fp8: int, e: int) -> int:
    """BF16_RNE(x * 2^(e - 127)) of E4M3 code x."""
    return int(cvt.bf16_rne(cvt.e4m3_to_f32(int(fp8) & 0xFF), int(e)))
