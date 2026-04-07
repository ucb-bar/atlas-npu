// ============================================================================
// RVFP8Test.scala — BF16/FP8 conversion tests for the RV-style vector
// datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.RVFP8Test
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome 
import sp26FPUnits.AtlasFPType.BF16

class RVFP8Test extends AnyFlatSpec with Matchers {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("RVFP8Test=FAILED")
    } else if (outcome.isSucceeded) {
      println("RVFP8Test=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  // ---- BF16 <-> Float helpers ----
  def fpTobf(f: Float): Int = {
    val x = java.lang.Float.floatToRawIntBits(f)
    val lsb = (x >>> 16) & 1
    val roundingBias = 0x7FFF + lsb
    ((x + roundingBias) >>> 16) & 0xFFFF
  }

  def bfTofp(b: Int): Float =
    java.lang.Float.intBitsToFloat((b & 0xFFFF) << 16)

  // ---- FP8 E4M3 golden helpers ----
  private def packE4M3(sign: Int, exp: Int, mant: Int): Int =
    ((sign & 1) << 7) | ((exp & 0xF) << 3) | (mant & 0x7)

  private def goldenFp8ByteFromBf16(bf16: Int, expShift: Int): Int = {
    val sign = (bf16 >>> 15) & 1
    val expBF = (bf16 >>> 7) & 0xFF
    val mantBF = bf16 & 0x7F

    val isZero = (expBF == 0) && (mantBF == 0)
    if (isZero) return 0

    val expAdj = expBF - 120 + expShift
    val expClamped =
      if (expAdj <= 0) 0
      else if (expAdj >= 15) 15
      else expAdj

    val mantFP8 = (mantBF >>> 4) & 0x7
    packE4M3(sign, expClamped, mantFP8)
  }

  private def goldenPacked16FromBf16(bf16: Int, expShift: Int, leftAlign: Boolean): Int = {
    val fp8b = goldenFp8ByteFromBf16(bf16, expShift)
    if (leftAlign) ((fp8b & 0xFF) << 8) else (fp8b & 0xFF)
  }

  private def driveReq(
      dut: FP8,
      dataBf16: Seq[Int],
      expShift: Int,
      leftAlign: Boolean,
      tag: Int = 1,
      whichBank: Int = 2,
      wRow: Int = 3
  ): Unit = {
    dut.io.req.valid.poke(true.B)
    dut.io.resp.ready.poke(true.B)
    dut.io.req.bits.tag.poke(tag.U)
    dut.io.req.bits.whichBank.poke(whichBank.U)
    dut.io.req.bits.wRow.poke(wRow.U)
    dut.io.req.bits.laneMask.poke("hFFFF".U)
    dut.io.req.bits.expShift.poke(expShift.S)
    dut.io.req.bits.leftAlign.poke(leftAlign.B)

    dataBf16.zipWithIndex.foreach { case (bits, i) =>
      dut.io.req.bits.xVec(i).poke((bits & 0xFFFF).U)
    }
  }

  private def readOut(dut: FP8): Seq[Int] = {
    (0 until 16).map(i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF)
  }

  behavior of "FP8 (BF16 -> FP8 E4M3 packer)"

  it should "convert BF16 lanes into FP8 bytes with expShift=0 and right alignment" in {
    simulate(new FP8(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val expShift = 0
      val leftAlign = false

      val dataF = Seq(
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 3.25f, 10.0f,
        0.25f, -2.5f, 6.0f, 0.0f, 0.75f, -0.125f, 12.0f, 100.0f
      )

      val dataBf16 = dataF.map(fpTobf)
      val expectedPacked16 = dataBf16.map(b => goldenPacked16FromBf16(b, expShift, leftAlign))

      driveReq(dut, dataBf16, expShift, leftAlign)

      var cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1)
        cycles += 1
      }
      (dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) shouldBe true

      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val actualPacked16 = readOut(dut)

      println(f"lane0 bf16=0x${dataBf16.head}%04X -> packed16=0x${actualPacked16.head}%04X expected=0x${expectedPacked16.head}%04X")

      actualPacked16.zip(expectedPacked16).zipWithIndex.foreach { case ((act, exp), i) =>
        withClue(f"lane $i: in=0x${dataBf16(i)}%04X act=0x$act%04X exp=0x$exp%04X\n") {
          act shouldBe exp
        }
      }
    }
  }

  it should "left-align FP8 when leftAlign=true" in {
    simulate(new FP8(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val expShift = 0
      val leftAlign = true

      val dataF = Seq.fill(16)(1.0f)
      val dataBf16 = dataF.map(fpTobf)
      val expected = dataBf16.map(b => goldenPacked16FromBf16(b, expShift, leftAlign))

      driveReq(dut, dataBf16, expShift, leftAlign)

      var cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1)
        cycles += 1
      }
      (dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) shouldBe true

      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val actual = readOut(dut)

      println(f"leftAlign lane0 bf16=0x${dataBf16.head}%04X -> packed16=0x${actual.head}%04X expected=0x${expected.head}%04X")

      actual.zip(expected).zipWithIndex.foreach { case ((a, e), i) =>
        withClue(f"left-align lane $i: act=0x$a%04X exp=0x$e%04X\n") {
          a shouldBe e
        }
      }
    }
  }

  it should "shift exponent correctly with positive expShift" in {
    simulate(new FP8(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val expShift = 2
      val leftAlign = false

      val dataF = Seq.fill(16)(1.0f)
      val dataBf16 = dataF.map(fpTobf)
      val expected = dataBf16.map(b => goldenPacked16FromBf16(b, expShift, leftAlign))

      driveReq(dut, dataBf16, expShift, leftAlign)

      var cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1)
        cycles += 1
      }
      (dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) shouldBe true

      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val actual = readOut(dut)

      println(f"expShift=+2 lane0 bf16=0x${dataBf16.head}%04X -> packed16=0x${actual.head}%04X expected=0x${expected.head}%04X")

      actual.zip(expected).zipWithIndex.foreach { case ((a, e), i) =>
        withClue(f"expShift(+2) lane $i: act=0x$a%04X exp=0x$e%04X\n") {
          a shouldBe e
        }
      }
    }
  }

  it should "underflow to zero with sufficiently negative expShift" in {
    simulate(new FP8(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val expShift = -10
      val leftAlign = false

      val dataF = Seq.fill(16)(1.0f)
      val dataBf16 = dataF.map(fpTobf)
      val expected = dataBf16.map(b => goldenPacked16FromBf16(b, expShift, leftAlign))

      driveReq(dut, dataBf16, expShift, leftAlign)

      var cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1)
        cycles += 1
      }
      (dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) shouldBe true

      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val actual = readOut(dut)

      println(f"expShift=-10 lane0 bf16=0x${dataBf16.head}%04X -> packed16=0x${actual.head}%04X expected=0x${expected.head}%04X")

      actual.zip(expected).zipWithIndex.foreach { case ((a, e), i) =>
        withClue(f"underflow lane $i: act=0x$a%04X exp=0x$e%04X\n") {
          a shouldBe e
        }
      }
    }
  }

  it should "clamp overflow exponent to 15" in {
    simulate(new FP8(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val expShift = 10
      val leftAlign = false

      val dataF = Seq.fill(16)(1000.0f)
      val dataBf16 = dataF.map(fpTobf)
      val expected = dataBf16.map(b => goldenPacked16FromBf16(b, expShift, leftAlign))

      driveReq(dut, dataBf16, expShift, leftAlign)

      var cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1)
        cycles += 1
      }
      (dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) shouldBe true

      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val actual = readOut(dut)

      println(f"overflow lane0 bf16=0x${dataBf16.head}%04X -> packed16=0x${actual.head}%04X expected=0x${expected.head}%04X")

      actual.zip(expected).zipWithIndex.foreach { case ((a, e), i) =>
        withClue(f"overflow lane $i: act=0x$a%04X exp=0x$e%04X\n") {
          a shouldBe e
        }
      }
    }
  }

  it should "convert 33.5 (0x4206) for both right and left alignments" in {
    simulate(new FP8(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      // Note: 33.5f as an integer is exactly 0x4206 in BF16
      val testVal = fpTobf(33.5f) 
      val dataBf16 = Seq.fill(16)(testVal)

      driveReq(dut, dataBf16, expShift = 0, leftAlign = false)
      
      var cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1)
        cycles += 1
      }
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      var actualRight = readOut(dut)
      println(f"right-aligned: BF16=0x${dataBf16.head}%04X => output=0x${actualRight.head}%04X (expected=0x0060)")
      actualRight.head shouldBe 0x0060

      dut.clock.step(2)

      driveReq(dut, dataBf16, expShift = 0, leftAlign = true)

      cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1)
        cycles += 1
      }
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      var actualLeft = readOut(dut)
      println(f"left-aligned:  BF16=0x${dataBf16.head}%04X => output=0x${actualLeft.head}%04X (expected=0x6000)")
      actualLeft.head shouldBe 0x6000
    }
  }

  it should "handle the exact boundary between minimum normal and flush-to-zero underflow" in {
    simulate(new FP8(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      // 1.0f in BF16 is 0x3F80, unbiased exp is 0
      val testVal = fpTobf(1.0f) 
      val dataBf16 = Seq.fill(16)(testVal)

      // exact min normal (expShift = -6) => finalExpAdjusted = -6, expFP8 = -6 + 7 = 1
      // E4M3: sign=0, exp=0001, mant=000 -> 0x08 right-aligned: 0x0008
      driveReq(dut, dataBf16, expShift = -6, leftAlign = false)
      
      var cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1); cycles += 1
      }
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      var actualMinNormal = readOut(dut)
      println(f"Min Normal:  BF16=0x${dataBf16.head}%04X -> Output=0x${actualMinNormal.head}%04X (Expected=0x0008)")
      actualMinNormal.head shouldBe 0x0008

      dut.clock.step(2)

      // below min normal (expShift = -7) => flush subnorms to 0x0000
      driveReq(dut, dataBf16, expShift = -7, leftAlign = false)

      cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1); cycles += 1
      }
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      var actualUnderflow = readOut(dut)
      println(f"underflow:   BF16=0x${dataBf16.head}%04X -> output=0x${actualUnderflow.head}%04X (expected=0x0000)")
      actualUnderflow.head shouldBe 0x0000
    }
  }

  it should "handle the exact boundary between max normal and overflow clamping" in {
    simulate(new FP8(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      // 1.0f in BF16 => unbiased exp = 0
      val dataBf16 = Seq.fill(16)(fpTobf(1.0f))
      
      // exact max normal (expShift = 8) => expect finalExpAdjusted = 8, expFP8 = 8 + 7 = 15
      // E4M3 format: sign=0, exp=1111, mant=000 => 0x78, right-aligned: 0x0078
      driveReq(dut, dataBf16, expShift = 8, leftAlign = false)
      
      var cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1); cycles += 1
      }
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      var actualMaxNormal = readOut(dut)
      println(f"max normal:  BF16=0x${dataBf16.head}%04X => output=0x${actualMaxNormal.head}%04X (expected=0x0078)")
      actualMaxNormal.head shouldBe 0x0078

      dut.clock.step(2)

      // overflow expected bc expShift = 9
      // bc > 8, clamps to E4M3_MAX_POS (0x7E) => 0x007E
      driveReq(dut, dataBf16, expShift = 9, leftAlign = false)

      cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1); cycles += 1
      }
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      var actualOverflow = readOut(dut)
      println(f"overflow:    BF16=0x${dataBf16.head}%04X => output=0x${actualOverflow.head}%04X (expecteds=0x007E)")
      actualOverflow.head shouldBe 0x007E
    }
  }

  it should "ignore expShift for special values (zero, inf, nan)" in {
    simulate(new FP8(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val bf16Zero  = 0x0000 // +0.0
      val bf16PosInf= 0x7F80 // +inf
      val bf16NegInf= 0xFF80 // -inf
      val bf16NaN   = 0x7FC0 // nan

      val dataBf16 = Seq(
        bf16Zero, bf16PosInf, bf16NegInf, bf16NaN,
        bf16Zero, bf16PosInf, bf16NegInf, bf16NaN,
        bf16Zero, bf16PosInf, bf16NegInf, bf16NaN,
        bf16Zero, bf16PosInf, bf16NegInf, bf16NaN
      )

      val expectedOutput = Seq(
        0x0000, 0x007E, 0x00FE, 0x0000,
        0x0000, 0x007E, 0x00FE, 0x0000,
        0x0000, 0x007E, 0x00FE, 0x0000,
        0x0000, 0x007E, 0x00FE, 0x0000
      )

      // big expShift = +50 -> should be completely ignored.
      driveReq(dut, dataBf16, expShift = 50, leftAlign = false)

      var cycles = 0
      while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < 10) {
        dut.clock.step(1)
        cycles += 1
      }
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val actual = readOut(dut)

      actual.zip(expectedOutput).zipWithIndex.foreach { case ((act, exp), i) =>
        withClue(f"value lane $i: in=0x${dataBf16(i)}%04X act=0x$act%04X exp=0x$exp%04X\n") {
          act shouldBe exp
        }
      }
    }
  }
}
