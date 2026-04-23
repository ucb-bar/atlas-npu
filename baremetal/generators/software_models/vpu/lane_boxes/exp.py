"""lane_boxes/exp.py — funct model of `ExpLane.scala`.

`Exp` is the BF16 vector exp / exp2 lane box. Both paths are bit-exact
with the RTL.

`exp(x)` delegates to the standalone FPEX BF16 model under
`dependencies/fpex/model/bf16_exp_model.py`, which captures the
RawFloat → Q(m,n) → LUT → HardFloat round-trip exactly.

`exp2(x)` is the same RTL datapath as `exp`, except the
`qmn.mul(rln2)` step is skipped (see `ExpLane.scala:64-67`:
`exp2KRVec = qmnVec.map(_.getKR)`). Upstream FPEX does not yet export
an `exp2_bf16_bits` entry point, so `_build_local_exp2_bf16_bits`
synthesizes one here by reusing the FPEX module's private helpers
(`_raw_to_recfn_bf16`, `_recfn_to_bf16`, `LUT`, constants). If a
future FPEX release ships `exp2_bf16_bits`, we defer to it
automatically.

Visible latency assumption: 1 cycle (`ExpLane.scala:77` says
`numIntermediateStages = 1`, no extra LUT-output register downstream
because the LUT register and the commonState register both latch on
the same cycle K rising edge).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Callable, Optional

from ..vector_params import VectorParams


@dataclass
class FPEXReq:
    xVec: list[int]                     # 16 BF16 bit patterns
    isBase2: bool = False               # False → exp(x), True → 2^x
    laneMask: int = 0xFFFF


@dataclass
class FPEXResp:
    result: list[int]                   # 16 BF16 bit patterns


def _build_local_exp2_bf16_bits(module) -> Callable[[int, int], int]:
    # Mirror `exp_bf16_bits` in dependencies/fpex/model/bf16_exp_model.py,
    # but skip the `mul(rln2)` step — matching the RTL's `isBase2` path
    # (`ExpLane.scala:64-67`). Everything else (qmnFromRawFloat, LUT
    # interpolation, HardFloat rounding) is identical. Crucially, the
    # RTL hardcodes `expFPIsInf(x, false.B)` (line 47) with the
    # ln(max_finite) threshold regardless of `isBase2`, so we use the
    # same threshold here — matching the RTL's bit-level behaviour, even
    # in the (88.7, 128] range where it pre-empts a finite exp2 result.
    _s = module._s
    _mask = module._mask
    _raw_to_recfn_bf16 = module._raw_to_recfn_bf16
    _recfn_to_bf16 = module._recfn_to_bf16
    LUT = module.LUT

    QMN_WIDTH = module.QMN_WIDTH
    QMN_N = module.QMN_N
    QMN_M = module.QMN_M
    BF16_EXP_WIDTH = module.BF16_EXP_WIDTH
    BF16_FRAC_WIDTH = module.BF16_FRAC_WIDTH
    BF16_SIG_WIDTH = module.BF16_SIG_WIDTH
    MAX_X_EXP = module.MAX_X_EXP
    MAX_X_SIG = module.MAX_X_SIG
    LUT_ENTRIES = module.LUT_ENTRIES
    LUT_VAL_N = module.LUT_VAL_N
    LUT_ADDR_BITS = module.LUT_ADDR_BITS
    LUT_TOP_ENDPOINT = module.LUT_TOP_ENDPOINT
    R_LOW_BITS = module.R_LOW_BITS
    ROUND_NEAR_EVEN = module.ROUND_NEAR_EVEN

    def exp2_bf16_bits(x_bits: int, rounding_mode: int = ROUND_NEAR_EVEN) -> int:
        x_bits &= 0xFFFF

        sign = (x_bits >> 15) & 1
        exp = (x_bits >> BF16_FRAC_WIDTH) & _mask(BF16_EXP_WIDTH)
        frac = x_bits & _mask(BF16_FRAC_WIDTH)

        is_zero = exp == 0 and frac == 0
        is_subnorm = exp == 0 and frac != 0
        is_inf = exp == 0xFF and frac == 0
        is_nan = exp == 0xFF and frac != 0

        exp_fp_overflow = (sign == 0) and (
            (exp > MAX_X_EXP) or (exp == MAX_X_EXP and frac > MAX_X_SIG)
        )

        if is_nan:
            is_sig_nan = ((frac >> (BF16_FRAC_WIDTH - 1)) & 1) == 0
            nan_frac = (
                ((1 if is_sig_nan else 0) << (BF16_FRAC_WIDTH - 1))
                | _mask(BF16_FRAC_WIDTH - 1)
            )
            return (sign << 15) | (0xFF << BF16_FRAC_WIDTH) | nan_frac
        if is_zero or is_subnorm:
            return 0x3F80  # 2^0 = +1.0
        if is_inf and sign == 1:
            return 0x0000  # 2^-inf = +0
        if (is_inf and sign == 0) or exp_fp_overflow:
            return 0x7F80  # +inf

        raw_sig = (1 << BF16_FRAC_WIDTH) | frac
        shift = int(exp) - 122  # qmnN + unbiasedExp - (sigWidth-1)
        if shift < 0:
            mag = raw_sig >> (-shift)
        else:
            mag = raw_sig << shift

        qmn_value = _s(-mag if sign else mag, QMN_WIDTH)

        # isBase2 path: `qmn.getKR` directly (no mul(rln2)).
        k = _s(qmn_value >> QMN_N, QMN_M)
        r = qmn_value & _mask(QMN_N)

        addr = (r >> R_LOW_BITS) & _mask(LUT_ADDR_BITS)
        r_lower = r & _mask(R_LOW_BITS)
        y0 = LUT[addr]
        y1 = LUT_TOP_ENDPOINT if addr == (LUT_ENTRIES - 1) else LUT[addr + 1]
        delta = y1 - y0
        delta_frac = (delta * r_lower) >> R_LOW_BITS
        pow2r = y0 + delta_frac

        sig_with_gr = pow2r >> (LUT_VAL_N - (BF16_SIG_WIDTH - 1) - 2)
        pre_sig = sig_with_gr & _mask(BF16_SIG_WIDTH + 2)
        sticky = (pow2r & _mask(LUT_VAL_N - (BF16_SIG_WIDTH - 1) - 2)) != 0
        raw_out_sig = pre_sig | (1 if sticky else 0)
        raw_out_s_exp = _s(k + (1 << BF16_EXP_WIDTH), 10)

        rec = _raw_to_recfn_bf16(raw_out_s_exp, raw_out_sig, rounding_mode & 0x7)
        return _recfn_to_bf16(rec)

    return exp2_bf16_bits


def _load_exact_exp_bf16_bits():
    model_path = (
        Path(__file__).resolve().parents[5]
        / "dependencies"
        / "fpex"
        / "model"
        / "bf16_exp_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_atlas_exact_bf16_exp_model",
        model_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load exact BF16 exp model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    exp_fn = module.exp_bf16_bits
    exp2_fn = getattr(module, "exp2_bf16_bits", None) or _build_local_exp2_bf16_bits(module)
    return exp_fn, exp2_fn


_EXACT_EXP_BF16_BITS, _EXACT_EXP2_BF16_BITS = _load_exact_exp_bf16_bits()


class Exp:
    """Mirror of `class Exp` in `ExpLane.scala`.

    Mirrors the shared LUT-assisted RTL datapath for natural and base-2
    exponentiation.
    """

    LATENCIES: dict[str, int] = {"exp": 1, "exp2": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: FPEXReq) -> FPEXResp:
        n = self.p.num_lanes
        if len(req.xVec) != n:
            raise ValueError(f"xVec must have {n} lanes, got {len(req.xVec)}")

        out: list[int] = []
        for i in range(n):
            lane_en = bool((req.laneMask >> i) & 1)
            if not lane_en:
                out.append(0x0000)
                continue

            bits = req.xVec[i] & 0xFFFF
            if req.isBase2:
                out.append(_EXACT_EXP2_BF16_BITS(bits))
            else:
                out.append(_EXACT_EXP_BF16_BITS(bits))
        return FPEXResp(result=out)

    def step(self, op_name: str, req: Optional[FPEXReq]) -> Optional[FPEXResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"Exp has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()
