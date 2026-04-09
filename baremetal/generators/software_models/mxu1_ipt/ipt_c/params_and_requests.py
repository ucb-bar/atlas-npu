from __future__ import annotations

from dataclasses import dataclass

from pi0_inout_c.ipt_mxu_model.fp_formats import (
    E4M3,
    BF16,
    E4M3ProdFmt,
    AtlasFPType,
    AddendSel,
    OutputFmtSel,
)


_PIPELINE_DEPTH_TO_CUTS = {
    1: frozenset(),
    2: frozenset({1}),
    3: frozenset({0, 2}),
    4: frozenset({0, 1, 2}),
    5: frozenset({0, 1, 2, 3}),
}

_MAX_EXP_WIDTH = max(
    E4M3.expWidth,
    E4M3ProdFmt.expWidth,
    E4M3.expWidth,  # biasFmt
    BF16.expWidth,  # psumFmt
    BF16.expWidth,  # outputFmt
)


@dataclass(frozen=True)
class InnerProductTreeParams:
    numLanes: int = 16
    vecLen: int = 32
    accumIntWidth: int = 0
    pipelineCuts: frozenset[int] = frozenset()

    def __post_init__(self):
        for c in self.pipelineCuts:
            if c < 0 or c > 3:
                raise ValueError(
                    f"pipelineCuts must be in {{0..3}}, got {self.pipelineCuts}"
                )

    @property
    def inputFmt(self) -> AtlasFPType:
        return E4M3

    @property
    def biasFmt(self) -> AtlasFPType:
        return E4M3

    @property
    def psumFmt(self) -> AtlasFPType:
        return BF16

    @property
    def outputFmt(self) -> AtlasFPType:
        return BF16

    @property
    def anchorHeadroom(self) -> int:
        return (self.vecLen + 1).bit_length() + 1

    @property
    def intWidth(self) -> int:
        accum_int_width = self.accumIntWidth
        if accum_int_width > 0:
            return accum_int_width
        return E4M3ProdFmt.sigWidth + self.anchorHeadroom + 15

    @property
    def expWorkWidth(self) -> int:
        return _MAX_EXP_WIDTH + 4

    @property
    def numPipeCuts(self) -> int:
        return len(self.pipelineCuts)

    @property
    def latency(self) -> int:
        return len(self.pipelineCuts) + 1

    @staticmethod
    def withPipelineDepth(
        depth: int, base: "InnerProductTreeParams | None" = None
    ) -> "InnerProductTreeParams":
        if base is None:
            base = InnerProductTreeParams()
        if depth < 1 or depth > 5:
            raise ValueError(f"depth must be 1..5, got {depth}")

        return InnerProductTreeParams(
            numLanes=base.numLanes,
            vecLen=base.vecLen,
            accumIntWidth=base.accumIntWidth,
            pipelineCuts=_PIPELINE_DEPTH_TO_CUTS[depth],
        )


@dataclass
class ComputeReq:
    act: list[int]
    bias: list[int]
    psum: list[int]
    scaleExp: list[int]
    addendSel: AddendSel
    outFmtSel: OutputFmtSel


@dataclass
class WeightLoadReq:
    weightsDma: list[int]
    laneIdx: int
    last: bool


@dataclass
class StepResult:
    out_valid: bool
    out_bits: list[int] | None
