package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome 
import sp26FPUnits.AtlasFPType.BF16

class RVSquareCubeTest extends AnyFlatSpec with Matchers {

    //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("RVSquareCubeTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("RVSquareCubeTest=PASSED")
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

  // Golden model: matches your MulRec flow by rounding inputs to BF16 first,
  // doing float math, then rounding result back to BF16.
  private def goldenSquare16(aVec: Seq[Float]): Seq[Int] = {
    require(aVec.length == 16)
    aVec.map { a =>
      val aB = bfTofp(fpTobf(a))
      fpTobf(aB * aB)
    }
  }

  private def goldenCube16(aVec: Seq[Float]): Seq[Int] = {
    require(aVec.length == 16)
    aVec.map { a =>
      val aB = bfTofp(fpTobf(a))
      fpTobf(aB * aB * aB)
    }
  }

  private def driveReq(
      dut: SquareCubeRec,
      dataBf16: Seq[Int],
      isCube: Boolean,
      tag: Int = 1,
      whichBank: Int = 2,
      wRow: Int = 3
  ): Unit = {
    dut.io.req.valid.poke(true.B)
    dut.io.resp.ready.poke(true.B)

    dut.io.req.bits.tag.poke(tag.U)
    dut.io.req.bits.whichBank.poke(whichBank.U)
    dut.io.req.bits.wRow.poke(wRow.U)
    dut.io.req.bits.roundingMode.poke(0.U)
    dut.io.req.bits.laneMask.poke("hFFFF".U)
    dut.io.req.bits.isCube.poke(isCube.B)

    dataBf16.zipWithIndex.foreach { case (bits, i) =>
      dut.io.req.bits.aVec(i).poke((bits & 0xFFFF).U)
    }
  }

  private def waitForReqFire(dut: SquareCubeRec, maxCycles: Int = 20): Unit = {
    var cycles = 0
    while (!(dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) && cycles < maxCycles) {
      dut.clock.step(1)
      cycles += 1
    }
    (dut.io.req.valid.peek().litToBoolean && dut.io.req.ready.peek().litToBoolean) shouldBe true
    dut.clock.step(1) // complete handshake
    dut.io.req.valid.poke(false.B)
  }

  private def waitForRespValid(dut: SquareCubeRec, maxCycles: Int = 200): Unit = {
    var cycles = 0
    while (!dut.io.resp.valid.peek().litToBoolean && cycles < maxCycles) {
      dut.clock.step(1)
      cycles += 1
    }
    dut.io.resp.valid.expect(true.B)
  }

  private def readOut(dut: SquareCubeRec): Seq[Int] = {
    (0 until 16).map(i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF)
  }

  behavior of "SquareCubeRec (BF16)"

  it should "compute square on all lanes" in {
    simulate(new SquareCubeRec(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val dataF = Seq(
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 3.25f, -2.5f,
        0.25f, -0.75f, 6.0f, 10.0f, -0.125f, 12.0f, 100.0f, -4.0f
      )

      val dataBf16 = dataF.map(fpTobf)
      val expected = goldenSquare16(dataF)

      driveReq(dut, dataBf16, isCube = false)
      waitForReqFire(dut)
      waitForRespValid(dut)

      val actual = readOut(dut)

      println("---- Square ----")
      actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
        val inF  = bfTofp(dataBf16(i))
        val actF = bfTofp(act)
        val expF = bfTofp(exp)
        println(f"lane=$i%2d in=0x${dataBf16(i)}%04X ($inF%f) act=0x$act%04X ($actF%f) exp=0x$exp%04X ($expF%f)")
      }

      actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
        withClue(f"square lane $i: in=0x${dataBf16(i)}%04X act=0x$act%04X exp=0x$exp%04X\n") {
          act shouldBe exp
        }
      }
    }
  }

  it should "compute cube on all lanes" in {
    simulate(new SquareCubeRec(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val dataF = Seq(
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 3.0f, -2.0f,
        0.25f, -0.75f, 4.0f, 5.0f, -0.125f, 6.0f, 10.0f, -3.5f
      )

      val dataBf16 = dataF.map(fpTobf)
      val expected = goldenCube16(dataF)

      driveReq(dut, dataBf16, isCube = true)
      waitForReqFire(dut)
      waitForRespValid(dut)

      val actual = readOut(dut)

      println("---- Cube ----")
      actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
        val inF  = bfTofp(dataBf16(i))
        val actF = bfTofp(act)
        val expF = bfTofp(exp)
        println(f"lane=$i%2d in=0x${dataBf16(i)}%04X ($inF%f) act=0x$act%04X ($actF%f) exp=0x$exp%04X ($expF%f)")
      }

      actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
        withClue(f"cube lane $i: in=0x${dataBf16(i)}%04X act=0x$act%04X exp=0x$exp%04X\n") {
          act shouldBe exp
        }
      }
    }
  }

  it should "preserve metadata fields tag/whichBank/wRow/laneMask" in {
    simulate(new SquareCubeRec(BF16, numLanes = 16, tagWidth = 16)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val dataF = Seq.fill(16)(2.0f)
      val dataBf16 = dataF.map(fpTobf)

      driveReq(dut, dataBf16, isCube = false, tag = 9, whichBank = 4, wRow = 7)
      waitForReqFire(dut)
      waitForRespValid(dut)

      dut.io.resp.bits.tag.expect(9.U)
      dut.io.resp.bits.whichBank.expect(4.U)
      dut.io.resp.bits.wRow.expect(7.U)
      dut.io.resp.bits.laneMask.expect("hFFFF".U)
    }
  }
}