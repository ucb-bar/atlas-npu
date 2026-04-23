"""lane_boxes/exp.py — funct model of `ExpLane.scala`.

`Exp` is the BF16 vector exp / exp2 lane box. Both paths are bit-exact
with the RTL by delegating to the standalone FPEX BF16 model under
`dependencies/fpex/model/bf16_exp_model.py`, which captures the
RawFloat → Q(m,n) → LUT → HardFloat-round-trip exactly.

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
    return module.exp_bf16_bits, module.exp2_bf16_bits


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
