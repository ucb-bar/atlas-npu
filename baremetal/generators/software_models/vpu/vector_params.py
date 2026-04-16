"""VectorParams — mirror of atlas.common.VpuParams + sp26FPUnits.AtlasFPType.

Only the fields the funct model actually reads are carried over. If the Scala
side grows a new field that the funct model needs, add it here and update the
Tier-0 canary so schema drift fails loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VectorParams:
    num_lanes: int = 16
    word_width: int = 16              # BF16 lane width
    rows_per_register: int = 32       # one physical tensor register = 32 rows

    # BF16 format (from sp26FPUnits.AtlasFPType.BF16)
    bf16_exp_width: int = 8
    bf16_sig_width: int = 8           # 1 hidden + 7 explicit mantissa bits
    bf16_bias: int = 127

    # Widened accumulator used by ColAddVec (ColAddVec.scala:21-23).
    # sigWidth = BF16.sigWidth + 16 = 24, which coincides with IEEE binary32.
    col_add_sig_width: int = 24

    # FP8 E4M3 format
    fp8_exp_width: int = 4
    fp8_sig_width: int = 4            # 1 hidden + 3 explicit mantissa bits
    fp8_bias: int = 7

    # Q-format parameters for SinCosVec (sp26-fp-units BF16 AtlasFPType). These
    # are the defaults used everywhere in the current Chisel tree; if BF16T
    # changes on the Scala side, update here and let the canary fail.
    qmn_m: int = 2
    qmn_n: int = 10
    lut_addr_bits: int = 7
    lut_val_m: int = 1
    lut_val_n: int = 16

    @property
    def qmn_total(self) -> int:
        return self.qmn_m + self.qmn_n

    @property
    def col_add_rec_width(self) -> int:
        # expWidth + computeSigWidth + 1, matching the HardFloat recFN layout.
        return self.bf16_exp_width + self.col_add_sig_width + 1
