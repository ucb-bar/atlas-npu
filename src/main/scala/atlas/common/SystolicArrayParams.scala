package atlas.common

import sp26FPUnits.AtlasFPType

case class SystolicArrayParams(
  inT: AtlasFPType  = AtlasFPType.E4M3,  // Format of activation element, weight element, and bias
  outT: AtlasFPType = AtlasFPType.BF16,  // Format of output element, partial sum

  // Geometry
  rows: Int = 32,    // Number of rows of PEs
  cols: Int = 16,    // Number of cols of PEs

  // Use FP32 for vertical accumulation (reduces rounding error vs BF16 chain)
  useFP32Accumulation: Boolean = false,

  // Use custom E4M3 FMA (E4M3Mul + E4M3ProdAddBF16) instead of HardFloat path. Requires inT.ieeeWidth == 8.
  useE4M3FMA: Boolean = false
) {
  require(rows > 1, s"Systolic array must have >1 rows, got $rows")
  require(cols > 1, s"Systolic array must have >1 rows, got $cols")
  require(!(useFP32Accumulation && useE4M3FMA), "useFP32Accumulation and useE4M3FMA are mutually exclusive")
  require(!useE4M3FMA || inT.ieeeWidth == 8, "useE4M3FMA requires inT.ieeeWidth == 8 (e.g. E4M3)")

}
