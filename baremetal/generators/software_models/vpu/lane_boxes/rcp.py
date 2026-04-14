"""lane_boxes/rcp.py — funct model of `Rcp.scala`.

Single-port, latency-1 LUT-backed reciprocal. The Scala module wraps
`RcpLUT(numLanes, addrBits=sigWidth-1, m=BF16T.lutValM, n=BF16T.lutValN)`
and delays the valid bit by one cycle (`reqValid := io.req.valid`).

Per-lane data flow (`Rcp.scala:34-44`):
  - `lutInputExp` = `aVec[i][14:7]`        (8-bit biased exponent)
  - `neg`         = `aVec[i][15]`          (BF16 sign bit, propagated)
  - `lutInputMant`= `aVec[i][6:0]`         (used as the 7-bit LUT addr)
  - `lut.io.ren`  = `laneMask[i] & req.valid`

Inside `RcpLUT` (`RcpLUT.scala:27-34`):
  - `out := lutFixedToBf16Rcp(lut(raddr), exp, raddr, m, n, neg)`

Special cases come entirely from `lutFixedToBf16Rcp`:
  - input ±inf, NaN → signed zero
  - input ±0, subnormal, or `expUnsigned >= 255` → signed inf
  - normal → table value, normalized via the helper

Disabled lanes (`!laneEnable[i]`) hold the previous register value
(`when (en) { results := out }`). Like the other LUT-backed lane_boxes
we model that as 0x0000 on the cycle of disabled output, since the
funct model has no register-retention semantics across compute_now()
calls. For `step()` the per-op queue makes this consistent: a disabled
lane on cycle K appears as 0x0000 in the output for cycle K+1.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..vector_params import VectorParams
from ..lut_sources.lut_params import lut_fixed_to_bf16_rcp
from ..lut_sources.lut_tables import gen_rcp_lut


_RCP_ADDR_BITS = 7
_RCP_M = 1
_RCP_N = 16


@dataclass
class RcpReq:
    aVec: list[int]                     # 16 BF16 bit patterns
    laneMask: int = 0xFFFF              # numLanes-bit mask
    roundingMode: int = 0               # ignored — LUT path is fixed-rounding


@dataclass
class RcpResp:
    result: list[int]                   # 16 BF16 bit patterns


def rcp_bf16(bf16: int, lut: list[int]) -> int:
    """Pure per-lane `1/x` mirror.

    Mirrors the combinational portion of `RcpLUT.scala:27-34` plus the
    `lutFixedToBf16Rcp` helper. Caller is responsible for masking out
    disabled lanes.
    """
    bf16 &= 0xFFFF
    sign = (bf16 >> 15) & 1
    exp = (bf16 >> 7) & 0xFF
    mant = bf16 & 0x7F
    raw = lut[mant]
    return lut_fixed_to_bf16_rcp(
        raw, exp, mant, lut_val_m=_RCP_M, lut_val_n=_RCP_N, neg=sign
    )


class Rcp:
    """Mirror of `class Rcp` in
    src/main/scala/atlas/vector/laneBoxes/Rcp.scala.

    Visible latency = 1 (one register stage on `res` plus one on
    `reqValid`, both on the same cycle in a single-stage pipeline)."""

    LATENCIES: dict[str, int] = {"rcp": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._lut = gen_rcp_lut(
            addr_bits=_RCP_ADDR_BITS, m=_RCP_M, n=_RCP_N,
            minimum=1.0, maximum=2.0,
        )
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: RcpReq) -> RcpResp:
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes, got {len(req.aVec)}")
        out: list[int] = []
        for i in range(n):
            lane_en = bool((req.laneMask >> i) & 1)
            if not lane_en:
                out.append(0x0000)
            else:
                out.append(rcp_bf16(req.aVec[i], self._lut))
        return RcpResp(result=out)

    def step(self, op_name: str, req: Optional[RcpReq]) -> Optional[RcpResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"Rcp has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()
