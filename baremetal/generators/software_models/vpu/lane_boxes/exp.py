"""lane_boxes/exp.py — placeholder for `ExpLane.scala` (`class Exp`).

`Exp` is the BF16 vector exp / exp2 lane box. The Scala module wraps
`ExLUT(numLanes, lutAddrBits, lutValM, lutValN)` plus per-lane
`RoundRawFNToRecFN` rounding modules and uses HardFloat
`rawFloatFromFN`/`fNFromRecFN` for the input decode + output
conversion (`ExpLane.scala:32-140`). The hard part isn't the LUT —
it's that the input is converted to a `RawFloat`, scaled into a
Q(qmnM, qmnN) fixed-point value via `qmnFromRawFloat`, then the
result of the LUT-interp is fed back through `rawFloatFromQmnK` and
RNE-rounded to BF16 via the Berkeley HardFloat round module.

That round module is currently only mirrored end-to-end in the
arithmetic spine via `f32_to_bf16_bits_rne` (which goes BF16 → FP32 →
RNE → BF16). Porting the full `RawFloat` ↔ Q(m,n) round-trip is
pending future work.

This skeleton exists so that:
  - tests can import the class and skip the real compute check,
  - the engine wiring can register the op name with the dispatcher,
  - the request / response shape matches the Scala bundle so future
    tests don't need to be rewritten.

Visible latency assumption: 1 cycle (`ExpLane.scala:77` says
`numIntermediateStages = 1`, no extra LUT-output register downstream
because the LUT register and the commonState register both latch on
the same cycle K rising edge).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..vector_params import VectorParams


@dataclass
class FPEXReq:
    xVec: list[int]                     # 16 BF16 bit patterns
    isBase2: bool = False               # False → exp(x), True → 2^x
    laneMask: int = 0xFFFF


@dataclass
class FPEXResp:
    result: list[int]                   # 16 BF16 bit patterns


class Exp:
    """Placeholder — see module docstring."""

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
        raise NotImplementedError(
            "Exp / exp2 funct model is a placeholder pending the "
            "HardFloat RawFloat ↔ Q(m,n) round-trip port."
        )

    def step(self, op_name: str, req: Optional[FPEXReq]) -> Optional[FPEXResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"Exp has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        if req is not None:
            raise NotImplementedError(
                "Exp.step with a live request is a placeholder; see compute_now."
            )
        # Drain only path (req=None) is safe: returns whatever was
        # queued earlier (which is also None if no work has been done).
        q = self._queues[op_name]
        q.append(None)
        return q.popleft()
