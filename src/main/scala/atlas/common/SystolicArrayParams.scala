package atlas.common

import sp26FPUnits.AtlasFPType

case class SystolicArrayParams(
  inT: AtlasFPType  = AtlasFPType.E4M3,  // Format of activation element, weight element, and bias
  outT: AtlasFPType = AtlasFPType.BF16,  // Format of output element, partial sum

  // Geometry
  rows: Int = 32,   // Number of rows of PEs
  cols: Int = 16    // Number of cols of PEs
) {
  require(rows > 1, s"Systolic array must have >1 rows, got $rows")
  require(cols > 1, s"Systolic array must have >1 rows, got $cols")

}
