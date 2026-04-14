"""Top-level VPU functional-model dispatcher.

`VectorEngineModel(p).execute(op, ...)` is a one-shot functional call
that routes a `VPUOp` to the correct `lane_box`, runs it through
`compute_now` (or `pack_two_rows` / `unpack_both_rows` for the phased
FP8 lane_boxes), and returns the final BF16 lane vector. This is the
reference the golden generator and the torch adapter both use.

Functional entry point only — `cycle_step()` is an intentional stub.
The model ships final-output match only and should not be used as a
cycle-accurate driver.

`exp` / `exp2` still route through a small Python-`math` fallback
pending HardFloat-faithful lane_box ports. Search for
`_MATH_FALLBACK_OPS` to find the seam.

`fp8pack` / `fp8unpack` are phased 2→1 / 1→2 and don't fit the
single-pulse vector file format, so `gen_vectors.py` skips them in
file output. The functional entry point below still supports them:
fp8pack takes two BF16 rows via (`a_vec`, `b_vec`) and returns 16
UInt16 packed slots; fp8unpack takes one packed row via `a_vec` and
returns 32 BF16 bits (low row followed by high row). The engine-layer
E8M0 clamp from `VectorEngine.scala:44-51` is applied here to match
RTL.

`fp8` (the dead enum value between `csum` and `fp8pack` in `VPUOp`)
is unwired in `VectorEngine.scala`. `execute("fp8", ...)` raises
`NotImplementedError`.
"""

from __future__ import annotations

import math
import struct

from . import bf16_utils as fp
from .vector_params import VectorParams
from .vpu_op import VPUOp

from .lane_boxes.add_sub_sum_vec import AddSubSumVec, AddSubSumReq
from .lane_boxes.col_add_vec import ColAddVec, ColAddReq
from .lane_boxes.exp import Exp, FPEXReq
from .lane_boxes.fp8_pack import FP8Pack, FP8PackReq
from .lane_boxes.fp8_unpack import FP8Unpack, FP8UnpackReq
from .lane_boxes.log import Log, LogReq
from .lane_boxes.mov import Mov, MovReq
from .lane_boxes.mul_rec import MulRec, MulReq
from .lane_boxes.pair_wise_max import PairWiseMax, PairWiseMaxReq
from .lane_boxes.pair_wise_min import PairWiseMin, PairWiseMinReq
from .lane_boxes.rcp import Rcp, RcpReq
from .lane_boxes.relu import Relu, ReluReq
from .lane_boxes.row_max import RowMax, RowMaxReq
from .lane_boxes.row_min import RowMin, RowMinReq
from .lane_boxes.sin_cos_vec import SinCosVec, SinCosVecReq
from .lane_boxes.sqrt import Sqrt, SqrtReq
from .lane_boxes.square_cube_vec import SquareCubeVec, SquareCubeReq
from .lane_boxes.tanh_rec import TanhRec, TanhReq, TanhResp
from .lane_boxes.vector_load_imm import VectorLoadImm, VLIReq


_FP32_PLUS_ZERO = 0


def _e8m0_to_scale_exp_clamped(scale_e8m0: int) -> int:
    """Mirror of `VectorEngine.scala:44-51`: E8M0 byte → clamped int8
    scale exponent. Identical arithmetic: subtract 127, then clip into
    [-128, 127]. `scale_e8m0 == 0` yields -127, `scale_e8m0 == 0xFE`
    yields 127, `scale_e8m0 == 0xFF` yields 128 pre-clamp → clamped to
    127. Caller is responsible for accepting the 8-bit input; this
    helper does NOT validate."""
    wide = (scale_e8m0 & 0xFF) - 127
    if wide > 127:
        return 127
    if wide < -128:
        return -128
    return wide


# Single-row pointwise binary ops dispatched via `aVec` + `bVec`.
_BIN_POINTWISE = {"add", "sub", "mul", "pairmax", "pairmin"}
# Single-row pointwise unary ops dispatched via `aVec` only.
_UNARY_POINTWISE = {
    "rcp", "sqrt", "sin", "cos", "tanh", "log", "exp", "exp2",
    "square", "cube", "relu", "mov",
}
# Row reductions: 16-lane input → 16-lane scalar broadcast.
_ROW_REDUCE = {"rsum", "rmax", "rmin"}
# Col reductions: 32-lane input → 32-lane scalar broadcast.
_COL_REDUCE = {"csum", "cmax", "cmin"}
_VLI_OPS = {"vliAll", "vliRow", "vliCol", "vliOne"}
_MATH_FALLBACK_OPS = {"exp", "exp2"}


class VectorEngineModel:
    """Functional dispatcher used by `gen_vectors.py`. Stateless from the
    caller's perspective — every `execute()` call resets nothing and shares
    nothing with prior calls. Lane boxes are instantiated once per engine
    so per-op queues exist, but `execute()` always uses `compute_now`."""

    def __init__(self, p: VectorParams):
        self.p = p
        self.add_sub_sum = AddSubSumVec(p)
        self.col_add = ColAddVec(p)
        self.exp_box = Exp(p)
        self.fp8_pack = FP8Pack(p)
        self.fp8_unpack = FP8Unpack(p)
        self.log_box = Log(p)
        self.mov = Mov(p)
        self.mul = MulRec(p)
        self.pair_wise_max = PairWiseMax(p)
        self.pair_wise_min = PairWiseMin(p)
        self.rcp = Rcp(p)
        self.relu = Relu(p)
        self.row_max = RowMax(p)
        self.row_min = RowMin(p)
        self.sin_cos = SinCosVec(p)
        self.sqrt = Sqrt(p)
        self.square_cube = SquareCubeVec(p)
        self.tanh = TanhRec(p)
        self.vli = VectorLoadImm(p)

    # ----------------------------------------------------------------
    #  Public functional entry point
    # ----------------------------------------------------------------

    def execute(
        self,
        op: str,
        a_vec: list[int] | None = None,
        b_vec: list[int] | None = None,
        imm: int = 0,
        scale_e8m0: int = 0,
        row_idx: int = 0,
    ) -> list[int]:
        """Route `op` to the matching lane_box and return the result as a
        list of BF16 bit patterns.

        Lane-count semantics:
          - Pointwise / row-reduction ops:  `a_vec` is `num_lanes` long,
            output is `num_lanes` long.
          - Col reductions (csum/cmax/cmin): `a_vec` is `2 * num_lanes`
            long (two stacked rows), output is `2 * num_lanes` long
            with the scalar broadcast across every slot — matches the
            block layout the Scala col_reduce family driver expects.
          - VLI ops: `a_vec` ignored; `imm` carries the 16-bit pattern,
            `row_idx` selects the current write row. `vliRow` and
            `vliOne` only fire on row 0 (`rowIdx == 0` in
            `VectorLoadImm.scala`); pass `row_idx > 0` to model the
            "other rows stay zero" semantics the lane box already
            implements. `vliAll` and `vliCol` ignore `row_idx`.
          - `fp8pack`:   `a_vec` = 16 BF16 low row, `b_vec` = 16 BF16 high
            row, `scale_e8m0` = raw E8M0 byte. Returns 16 UInt16 packed
            slots (low byte at bits[7:0], high byte at bits[15:8]).
          - `fp8unpack`: `a_vec` = 16 UInt16 packed slots, `scale_e8m0` =
            raw E8M0 byte. Returns 32 BF16 bit patterns: lanes [0..15] are
            the low half, lanes [16..31] are the high half.
          - `fp8`:       unwired dead enum. Raises
            NotImplementedError.
        """
        n = self.p.num_lanes

        if op == "fp8":
            raise NotImplementedError(
                "VPUOp.fp8 is unwired in current VectorEngine; use "
                "fp8pack / fp8unpack instead."
            )

        if op == "fp8pack":
            return self._exec_fp8_pack(
                list(a_vec or []),
                list(b_vec or []),
                scale_e8m0,
            )

        if op == "fp8unpack":
            return self._exec_fp8_unpack(list(a_vec or []), scale_e8m0)

        if op in _UNARY_POINTWISE:
            return self._exec_unary(op, list(a_vec or []))

        if op in _BIN_POINTWISE:
            return self._exec_binary(op, list(a_vec or []), list(b_vec or []))

        if op in _ROW_REDUCE:
            return self._exec_row_reduce(op, list(a_vec or []))

        if op in _COL_REDUCE:
            return self._exec_col_reduce(op, list(a_vec or []))

        if op in _VLI_OPS:
            r = self.vli.compute_now(VLIReq(op=op, imm=imm, rowIdx=row_idx))
            return r.result

        raise ValueError(f"VectorEngineModel.execute: unknown op {op!r}")

    def cycle_step(self, *args, **kwargs):
        raise NotImplementedError(
            "Cycle-accurate driver is a stub; only execute() is supported."
        )

    # ----------------------------------------------------------------
    #  Per-op helpers
    # ----------------------------------------------------------------

    def _exec_unary(self, op: str, a: list[int]) -> list[int]:
        if op in _MATH_FALLBACK_OPS:
            return _legacy_math_fallback(op, a)

        if op == "rcp":
            return self.rcp.compute_now(RcpReq(aVec=a)).result
        if op == "sqrt":
            return self.sqrt.compute_now(SqrtReq(aVec=a)).result
        if op == "sin":
            return self.sin_cos.compute_now(SinCosVecReq(xVec=a, cos=False)).result
        if op == "cos":
            return self.sin_cos.compute_now(SinCosVecReq(xVec=a, cos=True)).result
        if op == "log":
            return self.log_box.compute_now(LogReq(aVec=a)).result
        if op == "tanh":
            return self.tanh.compute_now(TanhReq(xVec=a)).result
        if op == "square":
            return self.square_cube.compute_now(
                SquareCubeReq(aVec=a, isCube=False)
            ).result
        if op == "cube":
            return self.square_cube.compute_now(
                SquareCubeReq(aVec=a, isCube=True)
            ).result
        if op == "relu":
            return self.relu.compute_now(ReluReq(aVec=a)).result
        if op == "mov":
            return self.mov.compute_now(MovReq(aVec=a)).result
        raise AssertionError(f"unhandled unary op {op!r}")

    def _exec_binary(self, op: str, a: list[int], b: list[int]) -> list[int]:
        if op == "add":
            return self.add_sub_sum.compute_now(
                AddSubSumReq(aVec=a, bVec=b, isSub=False, isSum=False)
            ).result
        if op == "sub":
            return self.add_sub_sum.compute_now(
                AddSubSumReq(aVec=a, bVec=b, isSub=True, isSum=False)
            ).result
        if op == "mul":
            return self.mul.compute_now(MulReq(aVec=a, bVec=b)).result
        if op == "pairmax":
            return self.pair_wise_max.compute_now(
                PairWiseMaxReq(aVec=a, bVec=b)
            ).result
        if op == "pairmin":
            return self.pair_wise_min.compute_now(
                PairWiseMinReq(aVec=a, bVec=b)
            ).result
        raise AssertionError(f"unhandled binary op {op!r}")

    def _exec_row_reduce(self, op: str, a: list[int]) -> list[int]:
        if op == "rsum":
            return self.add_sub_sum.compute_now(
                AddSubSumReq(aVec=a, isSum=True)
            ).result
        if op == "rmax":
            return self.row_max.compute_now(RowMaxReq(aVec=a)).result
        if op == "rmin":
            return self.row_min.compute_now(RowMinReq(aVec=a)).result
        raise AssertionError(f"unhandled row-reduce op {op!r}")

    def _exec_col_reduce(self, op: str, a: list[int]) -> list[int]:
        """Reduce a 2-row × 16-col input to a single scalar broadcast
        32 ways: csum/cmax/cmin emit 32 expected slots, all populated
        with the same scalar (`sum`/`max`/`min` over every BF16 input).

        Reductions go through the real lane_boxes:
          - csum: `ColAddVec.compute_now` per row, then a final
            cross-column sum done via the same FP32 adder helper.
          - cmax/cmin: `PairWiseMax`/`Min` to fold rows column-wise,
            then a column-wise reduction via `compare_return_max/min`.

        Both paths use bit-exact FP32 / sign-magnitude semantics
        matching the RTL — csum in particular goes through the Chisel
        `AddRecFN(8, 24)` widened-recFN adder, not an FP64 accumulator.
        """
        n = self.p.num_lanes
        if len(a) != 2 * n:
            raise ValueError(
                f"col-reduce {op}: aVec must have {2 * n} lanes, got {len(a)}"
            )
        rows = [a[0:n], a[n:2 * n]]

        if op == "csum":
            acc = [_FP32_PLUS_ZERO] * n
            for row in rows:
                resp = self.col_add.compute_now(ColAddReq(aVec=row, bVec=acc))
                acc = resp.result
            scalar_fp32 = _FP32_PLUS_ZERO
            for col_fp32 in acc:
                scalar_fp32 = fp.fp32_bits_add(scalar_fp32, col_fp32)
            scalar_bf16 = fp.bf16_upper_half_of_fp32_bits(scalar_fp32)
            return [scalar_bf16] * (2 * n)

        if op in ("cmax", "cmin"):
            box = self.pair_wise_max if op == "cmax" else self.pair_wise_min
            req_cls = PairWiseMaxReq if op == "cmax" else PairWiseMinReq
            cmp_fn = (
                fp.compare_return_max if op == "cmax" else fp.compare_return_min
            )
            acc = list(rows[0])
            for row in rows[1:]:
                resp = box.compute_now(req_cls(aVec=acc, bVec=row))
                acc = resp.result
            scalar = acc[0]
            for v in acc[1:]:
                scalar = cmp_fn(scalar, v)
            return [scalar] * (2 * n)

        raise AssertionError(f"unhandled col-reduce op {op!r}")

    def stream_col_reduce(
        self, op: str, rows: list[list[int]]
    ) -> list[int]:
        """Stream N rows of BF16 (each `num_lanes` long) through the
        cross-row col reduction `op` and return the final `num_lanes`
        BF16 accumulator. Mirrors the engine's streaming loop for the
        torch adapter, which needs true per-column output rather than
        the scalar-broadcast shape the file-based golden format uses.

        - `csum`: widened-FP32 accumulator per column, seeded with
          `FP32 +0.0` on row 0 (matches `VectorEngine.scala:319`
          `Mux(isFirstEntryColSum, zeroVecWide, csum.io.resp.bits.result)`).
          Narrowing to BF16 is a raw upper-16-bit slice, NOT an RNE
          re-round, matching `fNFromRecFN(8, 24, _)(31, 16)`.
        - `cmax`/`cmin`: 16-lane compare accumulator, seeded with row
          0's own lanes (matches `VectorEngine.scala:277-282`
          `Mux(isFirstEntryCMax, aVec, cmax.io.resp.bits.result)`).
        """
        n = self.p.num_lanes
        if not rows:
            raise ValueError("stream_col_reduce: rows must be non-empty")
        for i, r in enumerate(rows):
            if len(r) != n:
                raise ValueError(
                    f"stream_col_reduce row {i}: expected {n} lanes, got {len(r)}"
                )

        if op == "csum":
            acc = [_FP32_PLUS_ZERO] * n
            for row in rows:
                resp = self.col_add.compute_now(ColAddReq(aVec=row, bVec=acc))
                acc = resp.result
            return [fp.bf16_upper_half_of_fp32_bits(v) for v in acc]

        if op in ("cmax", "cmin"):
            box = self.pair_wise_max if op == "cmax" else self.pair_wise_min
            req_cls = PairWiseMaxReq if op == "cmax" else PairWiseMinReq
            acc = list(rows[0])
            for row in rows[1:]:
                resp = box.compute_now(req_cls(aVec=acc, bVec=row))
                acc = resp.result
            return acc

        raise ValueError(f"stream_col_reduce: unknown op {op!r}")

    def _exec_fp8_pack(
        self, a_vec: list[int], b_vec: list[int], scale_e8m0: int
    ) -> list[int]:
        """Two BF16 rows → 16 UInt16 packed slots. Uses the pure
        `FP8Pack.pack_two_rows` helper so no phase state is mutated on
        the shared instance (the cycle-accurate test machinery owns
        the `step()` entry point). Engine-layer clamp mirrors
        `VectorEngine.scala:243-244`."""
        n = self.p.num_lanes
        if len(a_vec) != n:
            raise ValueError(
                f"fp8pack: a_vec (low row) must have {n} lanes, got {len(a_vec)}"
            )
        if len(b_vec) != n:
            raise ValueError(
                f"fp8pack: b_vec (high row) must have {n} lanes, got {len(b_vec)}"
            )
        exp_shift = _e8m0_to_scale_exp_clamped(scale_e8m0)
        resp = self.fp8_pack.pack_two_rows(
            FP8PackReq(xVec=a_vec, expShift=exp_shift),
            FP8PackReq(xVec=b_vec, expShift=exp_shift),
        )
        return list(resp.result)

    def _exec_fp8_unpack(
        self, a_vec: list[int], scale_e8m0: int
    ) -> list[int]:
        """One 16-slot packed row → 32 BF16 lanes (low row [0..15] then
        high row [16..31]). Uses the pure `FP8Unpack.unpack_both_rows`
        helper; phase state on the shared instance is untouched.
        Engine-layer clamp mirrors `VectorEngine.scala:243-244`."""
        n = self.p.num_lanes
        if len(a_vec) != n:
            raise ValueError(
                f"fp8unpack: a_vec must have {n} packed slots, got {len(a_vec)}"
            )
        exp_shift = _e8m0_to_scale_exp_clamped(scale_e8m0)
        low_resp, high_resp = self.fp8_unpack.unpack_both_rows(
            FP8UnpackReq(xVec=a_vec, expShift=exp_shift)
        )
        return list(low_resp.result) + list(high_resp.result)


# --------------------------------------------------------------------
#  Temporary math fallback for exp / exp2.
#  Drop when the HardFloat-faithful Exp lane box lands.
# --------------------------------------------------------------------

def _legacy_math_fallback(op: str, a_vec: list[int]) -> list[int]:
    """Python-`math` based `exp` / `exp2` fallback. The real `Exp`
    lane box is not yet HardFloat-faithful; route through this until
    it lands, then delete."""
    out: list[int] = []
    for bits in a_vec:
        x = _bf16_bits_to_f32(bits)
        try:
            if op == "exp":
                y = math.exp(x)
            elif op == "exp2":
                y = 2.0 ** x
            else:
                raise AssertionError(f"_legacy_math_fallback: bad op {op!r}")
        except OverflowError:
            y = float("inf") if x > 0 else 0.0
        except ValueError:
            y = float("nan")
        out.append(_f32_to_bf16_bits_rne(y))
    return out


def _bf16_bits_to_f32(bits: int) -> float:
    """Zero-pad BF16 → FP32 → Python float."""
    return struct.unpack(
        ">f", struct.pack(">I", (bits & 0xFFFF) << 16)
    )[0]


def _f32_to_bf16_bits_rne(x: float) -> int:
    """FP32 → BF16 with RNE truncation plus canonical NaN/inf encoding.
    Inlined here rather than using `bf16_utils.f32_to_bf16_bits_rne`
    because the IPT base helper has slightly different NaN payload
    handling; these paths feed the file-based golden which needs
    byte-stable NaN bits."""
    if math.isnan(x):
        return 0x7FC0
    if x == float("inf"):
        return 0x7F80
    if x == float("-inf"):
        return 0xFF80
    fp32_int = struct.unpack(">I", struct.pack(">f", x))[0]
    lower_16 = fp32_int & 0xFFFF
    bit_16 = (fp32_int >> 16) & 1
    if lower_16 > 0x8000 or (lower_16 == 0x8000 and bit_16 == 1):
        fp32_int = (fp32_int + 0x8000) & 0xFFFFFFFF
    return (fp32_int >> 16) & 0xFFFF
