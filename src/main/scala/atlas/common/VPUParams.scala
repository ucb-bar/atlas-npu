package atlas.common

import fpex.FPType

case class VPUParams(
  BF16T: FPType = FPType.BF16T,
  wordWidth: Int = FPType.BF16T.wordWidth,
  expWidth: Int = FPType.BF16T.expWidth,
  sigWidth: Int = FPType.BF16T.sigWidth,
  numLanes: Int = 16,
  tagWidth: Int = 16,
  numOpMod: Int = 8
)
