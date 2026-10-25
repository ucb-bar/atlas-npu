/*
FPUtils.scala — Floating-point conversion utilities of the systolic-array MXU (mxu0).

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
