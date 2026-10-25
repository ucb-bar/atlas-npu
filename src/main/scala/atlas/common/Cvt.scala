// ============================================================================
// Cvt.scala — FP8 and BF16 conversions of the FP8 path, as the NVIDIA
// Blackwell conversion instructions give them (ISA v2, section 5.2).
//
// x * G(e) = x * 2^(e - 127) is exact, and each conversion rounds once, to
// nearest even, with subnormals. x is FP32 bits (a BF16 x is FP32 x << 16),
// e is 0..255. npu_model configs/cvt.py states the same rules; the RTX 5090
// vectors in baremetal/generators/fp8_silicon check both.
// ============================================================================

package atlas.common

import chisel3._
import chisel3.util._

object Cvt {

  /** |x| * G(e) rounded to nearest even with `mbits` fraction bits and `emin` the
    * smallest normal exponent, as (exponent field << mbits) + fraction, without an
    * upper bound. Also returns the sign, Inf or NaN (exponent field 255), and NaN.
    */
  private def round(x: UInt, e: UInt, mbits: Int, emin: Int): (UInt, Bool, Bool, Bool) = {
    val field = x(30, 23)
    val sig   = Cat(field =/= 0.U, x(22, 0))
    val lz    = PriorityEncoder(Reverse(sig))
    val norm  = (sig << lz)(23, 0)
    // Exponent of the leading one of |x| * G(e): max(field, 1) - 127 - lz + e - 127.
    val lead = Mux(field === 0.U, 1.U, field).zext +& e.zext -& lz.zext -& 254.S
    // Below the smallest normal, the quantum grows; 2 more bits leave only sticky bits.
    val below = emin.S -& lead
    val drop  = Mux(below <= 0.S, 0.U, Mux(below >= (mbits + 2).S, (mbits + 2).U, below.asUInt))
    val sh     = Cat(norm, 0.U(26.W)) >> ((23 - mbits).U +& drop)
    val q      = sh(49, 26)
    val n      = q +& (sh(25) && (sh(24, 0).orR || q(0)))
    val field0 = Mux(lead >= emin.S, (lead -& emin.S).asUInt << mbits, 0.U)
    val code   = Mux(sig === 0.U, 0.U, field0 +& n)
    (code, x(31), field === 255.U, field === 255.U && x(22, 0) =/= 0.U)
  }

  /** E4M3_RN_SATFINITE(x * G(e)): above 448 and +-Inf give +-448, NaN gives 0x7F. */
  def e4m3RnSatfinite(x: UInt, e: UInt): UInt = {
    val (code, sign, special, nan) = round(x, e, 3, -6)
    Mux(nan, "h7f".U(8.W), Cat(sign, Mux(special || code > 0x7e.U, 0x7e.U, code)(6, 0)))
  }

  /** BF16_RNE(x * G(e)): overflow gives +-Inf, NaN gives 0x7FFF. */
  def bf16Rne(x: UInt, e: UInt): UInt = {
    val (code, sign, special, nan) = round(x, e, 7, -126)
    Mux(nan, "h7fff".U(16.W), Cat(sign, Mux(special || code > 0x7f80.U, 0x7f80.U, code)(14, 0)))
  }

  /** FP32 bits of an E4M3 code, exact; 0x7F and 0xFF give 0x7FFFFFFF. */
  def e4m3ToF32(code: UInt): UInt = {
    val field = code(6, 3)
    val m     = code(2, 0)
    val p     = Log2(m)
    val mag = Mux(field =/= 0.U, Cat(field +& 120.U, m, 0.U(20.W)),
      Mux(m === 0.U, 0.U, Cat(118.U(8.W) + p, (m << (3.U - p))(2, 0), 0.U(20.W))))
    Mux(code(6, 0) === "h7f".U, "h7fffffff".U(32.W), Cat(code(7), mag(30, 0)))
  }
}
