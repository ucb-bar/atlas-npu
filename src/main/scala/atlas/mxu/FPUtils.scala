/*
FPUtils.scala — All floating-point conversion utilities for both MXU datapaths.

Everything lives in one object with a readable functional API:

  ┌─────────────────────────────────────────────────────────────────┐
  │  HardFloat recoding (SA PE architectures)                       │
  │    FPUtils.recode(bits, fmt)                                    │
  │    FPUtils.decode(bits, fmt)                                    │
  │    FPUtils.convert(bits, inFmt, outFmt)                         │
  ├─────────────────────────────────────────────────────────────────┤
  │  FP8 ↔ BF16 data-movement (push / pop paths)                    │
  │    FPUtils.e4m3ToBf16(fp8)                                      │
  │    FPUtils.bf16ToE4m3(bf16, scaleE8M0)                          │
  │    FPUtils.e4m3ToBf16Row(fp8s)                                  │
  │    FPUtils.bf16ToE4m3Row(bf16s, scaleE8M0)                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  Anchor-aligned converters (IPT accumulation tree)              │
  │    FPUtils.ieeeToAlignedInt(ieee, anchor, fmt, intW, expW)      │
  │    FPUtils.alignedIntToIEEE(intIn, anchor, outFmt, intW, expW)  │
  │    FPUtils.e4m3ProdToAlignedInt(prod, anchor, intW, expW)       │
  ├─────────────────────────────────────────────────────────────────┤
  │  RecFN helpers (HardFloat recoded-format field access)          │
  │    FPUtils.RecFN.fields(fmt, rec)                               │
  │    FPUtils.RecFN.isZero(fmt, recExp)                            │
  │    FPUtils.RecFN.unbiasedExp(fmt, recExp)                       │
  │    FPUtils.RecFN.fullSig(fmt, recExp, mantissa)                 │
  ├─────────────────────────────────────────────────────────────────┤
  │  Converter banks (shared hardware for sequencer FSMs)           │
  │    FPUtils.E4M3ToBF16Bank(n)                                    │
  │    FPUtils.BF16ToE4M3Bank(n)                                    │
  └─────────────────────────────────────────────────────────────────┘
*/

package atlas.mxu

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._

object FPUtils {

  // ═══════════════════════════════════════════════════════════════════
  //  RecFN helpers — HardFloat recoded-format field access
  // ═══════════════════════════════════════════════════════════════════

  object RecFN {
    /** Bit width of a recFN number for the given format. */
    def width(fmt: AtlasFPType): Int = fmt.recFNWidth

    /** Split a recFN number into (sign, recodedExponent, mantissa). */
    def fields(fmt: AtlasFPType, rec: UInt): (Bool, UInt, UInt) = {
      val w  = fmt.recFNWidth
      val sw = fmt.sigWidth
      (rec(w - 1), rec(w - 2, sw - 1), rec(sw - 2, 0))
    }

    /** True when the recFN number represents ±0 or a subnormal. */
    def isZero(fmt: AtlasFPType, recExp: UInt): Bool =
      recExp(fmt.expWidth, fmt.expWidth - 2) === 0.U

    /** Convert recoded exponent to an unbiased signed exponent (recFN bias = 2^ew). */
    def unbiasedExp(fmt: AtlasFPType, recExp: UInt): SInt = {
      val bias = 1 << fmt.expWidth
      recExp.zext - bias.S
    }

    /** Full significand with implicit leading bit restored (sigWidth total bits). */
    def fullSig(fmt: AtlasFPType, recExp: UInt, mantissa: UInt): UInt =
      Cat(!isZero(fmt, recExp), mantissa)
  }

  // ═══════════════════════════════════════════════════════════════════
  //  HardFloat recoding utilities
  // ═══════════════════════════════════════════════════════════════════

  /** Workaround: E4M3 is not natively supported by Berkeley HardFloat,
    * so we route through E5M3 for recoding. */
  private def getEffectiveType(t: AtlasFPType): (Int, Int) = {
    if (t.ieeeWidth == 8) (AtlasFPType.E5M3.expWidth, AtlasFPType.E5M3.sigWidth)
    else (t.expWidth, t.sigWidth)
  }

  /** Recode from IEEE standard format → Berkeley HardFloat recoded format. */
  def recodeIEEEToHardFloat(bits: UInt, t: AtlasFPType): UInt = {
    if (t.ieeeWidth == 8) {
      AtlasFPType.E5M3.recode(fp8ToE5M3(bits, false.B))
    } else {
      t.recode(bits)
    }
  }

  /** Decode from Berkeley HardFloat recoded format → IEEE standard format. */
  def decodeHardFloatToIEEE(bits: UInt, t: AtlasFPType): UInt = {
    fNFromRecFN(t.expWidth, t.sigWidth, bits)
  }

  /** Convert between two HardFloat recoded formats. */
  def convertHardFloat(bits: UInt, inT: AtlasFPType, outT: AtlasFPType): UInt = {
    val (inExp, inSig) = getEffectiveType(inT)
    val converter = Module(new RecFNToRecFN(inExp, inSig, outT.expWidth, outT.sigWidth))
    converter.io.in              := bits
    converter.io.roundingMode    := consts.round_near_even
    converter.io.detectTininess  := consts.tininess_afterRounding
    converter.io.exceptionFlags  := DontCare
    converter.io.out
  }

  // ═══════════════════════════════════════════════════════════════════
  //  FP8 ↔ BF16 data-movement converters
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Dequantize one E4M3 (FP8) value to BF16.
   *
   * Subnormals and NaN (exp=15, mant=7) are flushed to signed zero.
   * Normal values are re-biased: bf16Exp = e4m3Exp + 120, mantissa zero-extended.
   */
  def e4m3ToBf16(fp8: UInt): UInt = {
    val sign = fp8(7)
    val exp  = fp8(6, 3)
    val mant = fp8(2, 0)

    val isZero = (exp === 0.U) && (mant === 0.U)
    val isSub  = (exp === 0.U) && (mant =/= 0.U)
    val isNaN  = (exp === 0xF.U) && (mant === 0x7.U)

    val bf16 = Wire(UInt(16.W))
    when(isZero || isSub || isNaN) {
      bf16 := Cat(sign, 0.U(15.W))
    }.otherwise {
      val bf16Exp  = (exp +& 120.U)(7, 0)
      val bf16Mant = Cat(mant, 0.U(4.W))
      bf16 := Cat(sign, bf16Exp, bf16Mant)
    }
    bf16
  }

  /**
   * Quantize one BF16 value to E4M3 (FP8) with E8M0 block scale.
   *
   * The E8M0 unsigned byte is converted to a signed exponent,
   * clamped to [-128, 127], then forwarded to the BF16ScaleToE4M3 core.
   */
  def bf16ToE4m3(bf16: UInt, scaleE8M0: UInt): UInt = {
    val scaleExpWide = scaleE8M0.zext -& 127.S(9.W)
    val scaleExp     = Wire(SInt(8.W))
    when(scaleExpWide > 127.S)       { scaleExp := 127.S }
    .elsewhen(scaleExpWide < (-128).S) { scaleExp := (-128).S }
    .otherwise                         { scaleExp := scaleExpWide(7, 0).asSInt }

    val conv = Module(new BF16ScaleToE4M3())
    conv.io.bf16In   := bf16
    conv.io.scaleExp := scaleExp
    conv.io.e4m3Out(7, 0)
  }

  /** Dequantize a row of E4M3 values to BF16. */
  def e4m3ToBf16Row(fp8s: Seq[UInt]): Vec[UInt] =
    VecInit(fp8s.map(e4m3ToBf16))

  /** Quantize a row of BF16 values to E4M3 with shared E8M0 scale. */
  def bf16ToE4m3Row(bf16s: Seq[UInt], scaleE8M0: UInt): Vec[UInt] =
    VecInit(bf16s.map(bf16ToE4m3(_, scaleE8M0)))

  // ═══════════════════════════════════════════════════════════════════
  //  Anchor-aligned converters (IPT accumulation tree)
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Convert an IEEE floating-point value to an anchor-aligned signed integer.
   *
   * Works for any IEEE-like format (E4M3, FP16, BF16, FP32, …).
   * Subnormals are treated with exponent = −bias (not 1−bias) — a negligible
   * approximation in the context of anchor accumulation.
   *
   * @param ieee      IEEE-encoded floating-point input
   * @param anchorExp anchor exponent (signed)
   * @param fmt       floating-point format descriptor
   * @param intWidth  output integer width
   * @param expW      exponent work-register width
   */
  def ieeeToAlignedInt(ieee: UInt, anchorExp: SInt,
                       fmt: AtlasFPType, intWidth: Int, expW: Int): SInt = {
    val shiftBits = log2Ceil(intWidth + 1) max 1

    val sign     = ieee(fmt.ieeeWidth - 1)
    val expField = ieee(fmt.ieeeWidth - 2, fmt.mantissaBits)
    val frac     = ieee(fmt.mantissaBits - 1, 0)

    val isZero  = expField === 0.U && frac === 0.U
    val unbExp  = expField.zext -& fmt.ieeeBias.S(expW.W)
    val fullSig = Cat(expField =/= 0.U, frac)

    val shiftRight = anchorExp -& unbExp -& (intWidth - 1 - fmt.mantissaBits).S(expW.W)

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

    Mux(isZero, 0.S(intWidth.W),
      Mux(sign, -(magnitude.asSInt), magnitude.asSInt))
  }

  /**
   * Convert an anchor-aligned signed integer back to IEEE floating-point.
   *
   * Uses HardFloat's INToRecFN to convert the integer to recoded FP,
   * then adjusts the exponent by (anchorExp − (intWidth−1)) to undo the
   * anchor scaling, and converts back to IEEE.
   *
   * @param intIn     signed integer input
   * @param anchorExp anchor exponent (signed)
   * @param outFmt    output floating-point format descriptor
   * @param intWidth  input integer width
   * @param expW      exponent work-register width
   */
  def alignedIntToIEEE(intIn: SInt, anchorExp: SInt,
                       outFmt: AtlasFPType, intWidth: Int, expW: Int): UInt = {
    // Step 1: integer → recoded FP
    val conv = Module(new INToRecFN(intWidth, outFmt.expWidth, outFmt.sigWidth))
    conv.io.signedIn       := true.B
    conv.io.in             := intIn.asUInt
    conv.io.roundingMode   := consts.round_near_even
    conv.io.detectTininess := consts.tininess_afterRounding

    val rawRecFN = conv.io.out
    val (rawSign, rawRecExp, rawMantissa) = RecFN.fields(outFmt, rawRecFN)
    val rawIsZero = RecFN.isZero(outFmt, rawRecExp)

    // Step 2: adjust exponent to undo anchor scaling
    val adjRecExp  = rawRecExp.zext +& (anchorExp -& (intWidth - 1).S(expW.W))
    val recExpBits = outFmt.expWidth + 1

    // Step 3: clamp to valid range
    val expClamped = Wire(UInt(recExpBits.W))
    when(adjRecExp <= 0.S) {
      expClamped := 0.U
    }.elsewhen(adjRecExp >= ((1 << recExpBits) - 2).S) {
      expClamped := Cat("b110".U(3.W), 0.U((recExpBits - 3).max(0).W))
    }.otherwise {
      expClamped := adjRecExp(recExpBits - 1, 0)
    }

    val adjRecFN   = Cat(rawSign, expClamped, rawMantissa)
    val finalRecFN = Mux(intIn === 0.S || rawIsZero,
                         0.U(outFmt.recFNWidth.W), adjRecFN)

    // Step 4: recoded FP → IEEE
    fNFromRecFN(outFmt.expWidth, outFmt.sigWidth, finalRecFN)
  }

  /**
   * Convert an E4M3 lossless product (13-bit custom float) to an
   * anchor-aligned two's-complement integer.
   *
   * The product format is S(1) E(5, bias=13) M(7) as produced by E4M3Mul.
   * The barrel shifter is fed an 8-bit significand — fewer mux stages
   * and fewer wire toggles than the wider recFN path.
   * Zero products bypass the shifter entirely.
   *
   * @param prod      13-bit E4M3 product (from E4M3Mul)
   * @param anchorExp anchor exponent (signed)
   * @param intW      output integer width
   * @param expW      exponent work-register width
   */
  def e4m3ProdToAlignedInt(prod: UInt, anchorExp: SInt, intW: Int, expW: Int): SInt = {
    val sign    = prod(12)
    val expBits = prod(11, 7)
    val man     = prod(6, 0)
    val isZero  = expBits === 0.U

    val sig    = Cat(1.U(1.W), man)
    val unbExp = (expBits.zext - E4M3ProdFmt.bias.S).pad(expW).asSInt
    val rshift = (anchorExp - unbExp).asUInt

    val sigWide   = Cat(sig, 0.U((intW - E4M3ProdFmt.sigWidth).W))
    val shifted   = (sigWide >> rshift)(intW - 1, 0)
    val magnitude = shifted.asSInt
    val signed    = Mux(sign, -magnitude, magnitude)

    Mux(isZero, 0.S(intW.W), signed)
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Converter banks (shared hardware for sequencer FSMs)
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Shared n-wide E4M3 → BF16 dequantizer bank.
   *
   * Instantiate at module level.  Call `apply` from one or more
   * mutually-exclusive FSM paths; Chisel last-connect semantics
   * produce correct mux logic on the input wires.
   *
   * {{{
   *   val deq = FPUtils.E4M3ToBF16Bank(32)      // module level
   *   io.accLoad.bits.data := deq(fp8Row)        // in FSM path
   * }}}
   */
  class E4M3ToBF16Bank(n: Int) {
    private val ins = Wire(Vec(n, UInt(8.W)))
    ins := 0.U.asTypeOf(Vec(n, UInt(8.W)))
    val out: Vec[UInt] = VecInit(ins.map(e4m3ToBf16))

    def apply(fp8s: Seq[UInt]): Vec[UInt] = {
      require(fp8s.length == n, s"E4M3ToBF16Bank($n) given ${fp8s.length} inputs")
      fp8s.zipWithIndex.foreach { case (v, i) => ins(i) := v }
      out
    }
  }

  object E4M3ToBF16Bank {
    def apply(n: Int): E4M3ToBF16Bank = new E4M3ToBF16Bank(n)
  }

  /**
   * Shared n-wide BF16 → E4M3 quantizer bank with E8M0 scaling.
   *
   * {{{
   *   val quant = FPUtils.BF16ToE4M3Bank(32)            // module level
   *   val fp8Row = quant(bf16Row, cmd.scaleE8M0)         // in FSM path
   * }}}
   */
  class BF16ToE4M3Bank(n: Int) {
    private val bf16In  = Wire(Vec(n, UInt(16.W)))
    private val scaleIn = Wire(UInt(8.W))
    bf16In  := 0.U.asTypeOf(Vec(n, UInt(16.W)))
    scaleIn := 0.U
    val out: Vec[UInt] = VecInit(bf16In.map(bf16ToE4m3(_, scaleIn)))

    def apply(bf16s: Seq[UInt], scaleE8M0: UInt): Vec[UInt] = {
      require(bf16s.length == n, s"BF16ToE4M3Bank($n) given ${bf16s.length} inputs")
      bf16s.zipWithIndex.foreach { case (v, i) => bf16In(i) := v }
      scaleIn := scaleE8M0
      out
    }
  }

  object BF16ToE4M3Bank {
    def apply(n: Int): BF16ToE4M3Bank = new BF16ToE4M3Bank(n)
  }
}
