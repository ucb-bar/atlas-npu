/*
FormatConverters.scala: converts floating-point numbers to anchor-aligned integers and vice versa.

Modules:
1) RecFNUtils         helper functions for HardFloat recoded floating-point fields.
2) IEEEToAlignedInt   IEEE floating-point to signed integer, aligned to anchor.
3) AlignedIntToIEEE   anchor-aligned signed integer to IEEE floating-point. uses
                      HardFloat INToRecFN, then adjusts the exponent to undo the
                      anchor scaling, and converts back to IEEE.

All widths (integer, exponent, significand) are derived from InnerProductTreeParams
and the standalone E4M3ProdFmt constants.
*/

package atlas.ipt

import chisel3._
import chisel3.util._
import sp26FPUnits.hardfloat._
import sp26FPUnits.{AtlasFPType, E4M3ProdFmt}

// RecFNUtils
object RecFNUtils {
  // bit width of a recFN number
  def width(fmt: AtlasFPType): Int = fmt.recFNWidth

  // split a recFN number into (sign, recodedExponent, mantissa)
  def fields(fmt: AtlasFPType, rec: UInt): (Bool, UInt, UInt) = {
    val w = fmt.recFNWidth
    val ew = fmt.expWidth
    val sw = fmt.sigWidth
    (rec(w - 1), rec(w - 2, sw - 1), rec(sw - 2, 0))
  }

  // true when the recFN number represents +/- 0 or a subnormal
  def isZero(fmt: AtlasFPType, recExp: UInt): Bool =
    recExp(fmt.expWidth, fmt.expWidth - 2) === 0.U

  // converts recoded exponent into an unbiased signed exponent. recFN bias = 2^ew
  def unbiasedExp(fmt: AtlasFPType, recExp: UInt): SInt = {
    val bias = 1 << fmt.expWidth
    recExp.zext - bias.S
  }

  // builds a full significand by adding the implicit leading bit (sw total bits)
  def fullSig(fmt: AtlasFPType, recExp: UInt, mantissa: UInt): UInt =
    Cat(!isZero(fmt, recExp), mantissa)
}


// IEEEToAlignedInt
// works for any IEEE-like format (E4M3, FP16, BF16, FP32, ...)
// note: subnormals are treated with exponent = -bias (not 1-bias).
// this is a negligible approximation in the context of anchor accumulation.
class IEEEToAlignedInt(fmt: AtlasFPType, intWidth: Int, expW: Int) extends Module {
  val io = IO(new Bundle {
    val ieee      = Input(UInt(fmt.ieeeWidth.W))
    val anchorExp = Input(SInt(expW.W))
    val intOut    = Output(SInt(intWidth.W))
  })

  private val shiftBits = log2Ceil(intWidth + 1) max 1

  // IEEE field extraction
  val sign     = io.ieee(fmt.ieeeWidth - 1)
  val expField = io.ieee(fmt.ieeeWidth - 2, fmt.mantissaBits)
  val frac     = io.ieee(fmt.mantissaBits - 1, 0)

  val isZero  = expField === 0.U && frac === 0.U
  val unbExp  = expField.zext -& fmt.ieeeBias.S(expW.W)
  val fullSig = Cat(expField =/= 0.U, frac) // fmt.sigWidth bits

  // shiftRight = anchorExp - unbExp - (intWidth - 1 - mantissaBits)
  val shiftRight = (io.anchorExp -& unbExp -& (intWidth - 1 - fmt.mantissaBits).S(expW.W))

  val sigWide   = fullSig.pad(intWidth.max(fmt.sigWidth))
  val magnitude = Wire(UInt(intWidth.W))
  when(shiftRight >= intWidth.S) {
    magnitude := 0.U
  }.elsewhen(shiftRight >= 0.S) {
    magnitude := (sigWide >> shiftRight(shiftBits - 1, 0))(intWidth - 1, 0)
  }.elsewhen(shiftRight >= (-(intWidth - 1)).S) {
    val negShift = (0.S(expW.W) - shiftRight)(shiftBits - 1, 0)
    magnitude := (sigWide << negShift)(intWidth - 1, 0)
  }.otherwise {
    magnitude := 0.U
  }

  io.intOut := Mux(isZero, 0.S(intWidth.W),
               Mux(sign, -(magnitude.asSInt), magnitude.asSInt))
}


// AlignedIntToIEEE
class AlignedIntToIEEE(intWidth: Int, outFmt: AtlasFPType, expW: Int) extends Module {
  val io = IO(new Bundle {
    val intIn     = Input(SInt(intWidth.W))
    val anchorExp = Input(SInt(expW.W))
    val ieeeOut   = Output(UInt(outFmt.ieeeWidth.W))
  })

  // step 1: INToRecFN (treat the integer as a regular signed integer and convert to recFN)
  val conv = Module(new INToRecFN(intWidth, outFmt.expWidth, outFmt.sigWidth))
  conv.io.signedIn       := true.B
  conv.io.in             := io.intIn.asUInt
  conv.io.roundingMode   := consts.round_near_even
  conv.io.detectTininess := consts.tininess_afterRounding

  val rawRecFN = conv.io.out
  val (rawSign, rawRecExp, rawMantissa) = RecFNUtils.fields(outFmt, rawRecFN)
  val rawIsZero = RecFNUtils.isZero(outFmt, rawRecExp)

  // step 2: adjust the recFN exponent by (anchorExp - (intWidth-1)) to undo the anchor scaling
  val adjRecExp = rawRecExp.zext +& (io.anchorExp -& (intWidth - 1).S(expW.W))
  val recExpBits = outFmt.expWidth + 1

  // step 3: clamp the adjusted exponent to the output format's valid range
  val expClamped = Wire(UInt(recExpBits.W))
  when(adjRecExp <= 0.S) {
    expClamped := 0.U
  }.elsewhen(adjRecExp >= ((1 << recExpBits) - 2).S) {
    // recFN infinity exponent pattern
    expClamped := Cat("b110".U(3.W), 0.U((recExpBits - 3).max(0).W))
  }.otherwise {
    expClamped := adjRecExp(recExpBits - 1, 0)
  }

  val adjRecFN = Cat(rawSign, expClamped, rawMantissa)
  val finalRecFN = Mux(io.intIn === 0.S || rawIsZero,
                       0.U(outFmt.recFNWidth.W), adjRecFN)

  // step 4: fNFromRecFN: convert recFN to IEEE
  io.ieeeOut := fNFromRecFN(outFmt.expWidth, outFmt.sigWidth, finalRecFN)
}

/*
Convert an E4M3 lossless product (13-bit custom float) to an
anchor-aligned two's-complement integer.

The product format is S(1) E(5, bias=13) M(7) as produced by E4M3Mul.

Notes:
- The barrel shifter is fed an 8-bit significand (vs. the wider
  significand of an FP16/FP32 recFN path), so fewer mux stages
  are active and fewer wires toggle.
- Zero products bypass the shifter entirely (output clamped to 0).
*/
class E4M3ProdToAlignedInt(intW: Int, expW: Int) extends Module {
  val io = IO(new Bundle {
    val prod      = Input(UInt(E4M3ProdFmt.width.W))
    val anchorExp = Input(SInt(expW.W))
    val intOut    = Output(SInt(intW.W))
  })

  val sign    = io.prod(12)
  val expBits = io.prod(11, 7)
  val man     = io.prod(6, 0)
  val isZero  = expBits === 0.U

  val sig = Cat(1.U(1.W), man)

  val unbExp = (expBits.zext - E4M3ProdFmt.bias.S).pad(expW).asSInt
  val rshift = (io.anchorExp - unbExp).asUInt

  val sigWide  = Cat(sig, 0.U((intW - E4M3ProdFmt.sigWidth).W))  // intW bits
  val shifted  = (sigWide >> rshift)(intW - 1, 0)                 // intW bits

  val magnitude = shifted.asSInt
  val signed    = Mux(sign, -magnitude, magnitude)

  io.intOut := Mux(isZero, 0.S(intW.W), signed)
}
