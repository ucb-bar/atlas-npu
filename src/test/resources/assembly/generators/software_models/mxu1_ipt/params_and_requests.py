from __future__ import annotations

from dataclasses import dataclass

from .fp_formats import (
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
    E4M3.expWidth,   # biasFmt
    BF16.expWidth,   # psumFmt
    BF16.expWidth,   # outputFmt
)

def _log2ceil(x: int) -> int:
    assert x > 0
    if x == 1:
        return 1
    return (x - 1).bit_length()

@dataclass(frozen=True)
class InnerProductTreeParams:
    numLanes: int = 32
    vecLen: int = 32
    tileRows: int = 32
    accumIntWidth: int = 0
    pipelineCuts: frozenset[int] = frozenset()

    def __post_init__(self):
        assert self.numLanes >= 1
        assert self.vecLen >= 1
        assert self.tileRows >= 1
        for c in self.pipelineCuts:
            if c < 0 or c > 3:
                raise ValueError(
                    f"pipelineCuts must be in {{0..3}}, got {self.pipelineCuts}"
                )

    # ── Format accessors (match Scala InnerProductTreeParams) ──

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

    # ── Derived constants ──

    @property
    def anchorHeadroom(self) -> int:
        """BigInt(vecLen + 1).bitLength + 1"""
        total = self.vecLen + 1
        return total.bit_length() + 1

    @property
    def intWidth(self) -> int:
        """Matches Scala: mulFmt.sigWidth + anchorHeadroom + 17  (when accumIntWidth == 0)"""
        if self.accumIntWidth > 0:
            return self.accumIntWidth
        return E4M3ProdFmt.sigWidth + self.anchorHeadroom + 17

    @property
    def expWorkWidth(self) -> int:
        return _MAX_EXP_WIDTH + 4

    @property
    def numPipeCuts(self) -> int:
        return len(self.pipelineCuts)

    @property
    def latency(self) -> int:
        return self.numPipeCuts + 1

    @property
    def tileRowBits(self) -> int:
        return _log2ceil(self.tileRows)

    @staticmethod
    def withPipelineDepth(
        depth: int,
        base: "InnerProductTreeParams | None" = None,
    ) -> "InnerProductTreeParams":
        if base is None:
            base = InnerProductTreeParams()
        if depth < 1 or depth > 5:
            raise ValueError(f"depth must be 1..5, got {depth}")
        return InnerProductTreeParams(
            numLanes=base.numLanes,
            vecLen=base.vecLen,
            tileRows=base.tileRows,
            accumIntWidth=base.accumIntWidth,
            pipelineCuts=_PIPELINE_DEPTH_TO_CUTS[depth],
        )

    def __repr__(self) -> str:
        cuts = (
            "none"
            if not self.pipelineCuts
            else ",".join(str(c) for c in sorted(self.pipelineCuts))
        )
        return (
            f"InnerProductTreeParams("
            f"in={self.inputFmt.name}, psum={self.psumFmt.name}, out={self.outputFmt.name}, "
            f"{self.numLanes}x{self.vecLen}, tile={self.tileRows}, "
            f"intW={self.intWidth}, headroom={self.anchorHeadroom}, cuts={{{cuts}}})"
        )