/*
Bf16Compare.scala - shared BF16 comparison for SystolicArray tests

Collects abs, rel, and ULP error. Pass/fail based on relative error only:
  - Both near zero (|value| <= absTolerance in magnitude bits): pass
  - Otherwise: pass iff relErr <= relTolerance (default 1%)
*/

package atlas.sa

object Bf16Compare {

  def bf16BitsToFloat(bits: Int): Float = {
    val fp32Bits = (bits & 0xFFFF) << 16
    java.lang.Float.intBitsToFloat(fp32Bits)
  }

  case class Bf16Errs(absErr: Double, relErr: Double, ulpDiff: Int)

  val defaultAbsTolerance = 0x0040
  val defaultRelTolerance = 0.01
  val defaultRelEps       = 1e-8

  /** Compare actual vs expected BF16. Returns (pass, errs). Pass = (both near zero) OR (relErr <= relTol). */
  def compare(
    a16: Int,
    e16: Int,
    absTol: Int = defaultAbsTolerance,
    relTol: Double = defaultRelTolerance,
    relEps: Double = defaultRelEps,
  ): (Boolean, Bf16Errs) = {
    val a   = a16 & 0xFFFF
    val e   = e16 & 0xFFFF
    val aMag = a & 0x7FFF
    val eMag = e & 0x7FFF
    val aF   = bf16BitsToFloat(a).toDouble
    val eF   = bf16BitsToFloat(e).toDouble
    val absErr = math.abs(aF - eF)
    val denom  = math.max(math.abs(eF), relEps)
    val relErr = if (denom > 0) absErr / denom else 0.0
    val ulpDiff = math.abs(aMag - eMag)

    val ok = (aMag <= absTol && eMag <= absTol) || (relErr <= relTol)
    (ok, Bf16Errs(absErr, relErr, ulpDiff))
  }
}
