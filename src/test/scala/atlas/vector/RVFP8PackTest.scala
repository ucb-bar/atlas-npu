// ============================================================================
// RVFP8PackTest.scala — BF16→FP8 pack tests for the RV-style vector datapath.
//
// FP8Pack is 2→1 phased: two sequential `req.valid` pulses (16 BF16 lanes
// each) produce one `resp.valid` pulse carrying 16×UInt(16.W), where each
// 16-bit slot holds two packed FP8 bytes. Output layout (FP8Pack.scala:122-125):
//
//   result(j)     = Cat(lowHalf(2j+1),  lowHalf(2j))   // first-pulse bytes
//   result(j+8)   = Cat(fp8Bytes(2j+1), fp8Bytes(2j))  // second-pulse bytes
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.RVFP8PackTest
// ============================================================================
package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import sp26FPUnits.AtlasFPType.BF16

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import atlas.common._
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator factory — fresh persistent workspace per test_name (each
// `it should` block gets its own workdir so multi-test files don't trip on
// stale NFS file handles during cleanup between simulate() calls).
// ============================================================================

class RVFP8PackTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

  private def makeSim(testName: String): Simulator[VcsBackend] with PeekPokeAPI =
    new Simulator[VcsBackend] with PeekPokeAPI {
      private val runDir: Path = {
        val rootDirStr = sys.env.getOrElse("MILL_WORKSPACE_ROOT", "/tmp")
        val baseDir = Paths.get(rootDirStr)
        val p = baseDir.resolve("tmp").resolve(testName)
        Files.createDirectories(p)
        p.toAbsolutePath
      }
      override val backend: VcsBackend   = VcsBackend.initializeFromProcessEnvironment()
      override val tag: String           = testName
      override val workspacePath: String = runDir.toString
      override val commonCompilationSettings: CommonCompilationSettings =
        CommonCompilationSettings(
          availableParallelism =
            CommonCompilationSettings.AvailableParallelism.UpTo(Runtime.getRuntime.availableProcessors())
        )
      override val backendSpecificCompilationSettings: Backend.CompilationSettings = {
        val cov = Backend.CoverageSettings(
          line = true, cond = true, branch = true, fsm = true, tgl = true
        )
        Backend.CompilationSettings(
          coverageSettings  = cov,
          coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
          simulationSettings = Backend.SimulationSettings(
            coverageSettings  = cov,
            coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
            coverageName      = Some(Backend.CoverageName(s"${testName}_coverage"))
          )
        )
      }
    }


  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("RVFP8PackTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("RVFP8PackTest=PASSED")
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

  // ---- FP8 E4M3 golden helpers matching FP8Pack.scala ----
  private def packE4M3(sign: Int, exp: Int, mant: Int): Int =
    ((sign & 1) << 7) | ((exp & 0xF) << 3) | (mant & 0x7)

  private def roundRightShift4RNE(mant8: Int): Int = {
    val trunc  = (mant8 >>> 4) & 0xF
    val guard  = (mant8 >>> 3) & 0x1
    val sticky = mant8 & 0x7
    val lsb    = trunc & 0x1
    val inc    = (guard == 1) && ((sticky != 0) || (lsb == 1))
    trunc + (if (inc) 1 else 0)
  }

  private def goldenFp8ByteFromBf16(bf16: Int, expShift: Int): Int = {
    val sign   = (bf16 >>> 15) & 0x1
    val expBF  = (bf16 >>> 7) & 0xFF
    val mantBF = bf16 & 0x7F

    val isZero = (expBF == 0) && (mantBF == 0)
    val isSub  = (expBF == 0) && (mantBF != 0)
    val isInf  = (expBF == 0xFF) && (mantBF == 0)
    val isNaN  = (expBF == 0xFF) && (mantBF != 0)

    val E4M3_MAX_POS = 0x7E
    val E4M3_MAX_NEG = 0xFE

    if (isZero || isSub || isNaN) {
      0
    } else if (isInf) {
      if (sign == 1) E4M3_MAX_NEG else E4M3_MAX_POS
    } else {
      val unbExp = expBF - 127
      val expAdjusted = unbExp - expShift

      val mant8 = (1 << 7) | mantBF
      val roundedSig = roundRightShift4RNE(mant8)
      val normCarry = roundedSig == 16

      val finalExpAdjusted =
        if (normCarry) expAdjusted + 1
        else expAdjusted

      val mantFP8 =
        if (normCarry) 0
        else (roundedSig - 8) & 0x7

      if (finalExpAdjusted > 8) {
        if (sign == 1) E4M3_MAX_NEG else E4M3_MAX_POS
      } else if (finalExpAdjusted >= -6) {
        val expFP8 = finalExpAdjusted + 7
        val packed = packE4M3(sign, expFP8, mantFP8)
        if ((packed & 0x7F) == 0x7F) {
          if (sign == 1) E4M3_MAX_NEG else E4M3_MAX_POS
        } else {
          packed
        }
      } else {
        0
      }
    }
  }

  // ---- DUT helpers for the 2→1 phased interface ----

  private def pokeRow(dut: FP8Pack, dataBf16: Seq[Int], expShift: Int): Unit = {
    dut.io.req.valid.poke(true.B)
    dut.io.req.bits.expShift.poke(expShift.S)
    dataBf16.zipWithIndex.foreach { case (bits, i) =>
      dut.io.req.bits.xVec(i).poke((bits & 0xFFFF).U)
    }
  }

  /** Drive two 16-lane BF16 rows through FP8Pack and return the resulting
    * 32 FP8 bytes in the order they were submitted (first pulse bytes
    * 0..15, then second pulse bytes 16..31).
    *
    * Timing: each `req.valid` pulse takes one clock. The second pulse also
    * triggers the `respBitsReg` latch, so after the second step `resp.valid`
    * is asserted and can be peeked immediately.
    */
  private def drivePairAndRead(
      dut: FP8Pack,
      firstRow: Seq[Int],
      secondRow: Seq[Int],
      expShift: Int
  ): Seq[Int] = {
    require(firstRow.length == 16 && secondRow.length == 16,
      "FP8Pack expects exactly 16 BF16 lanes per pulse")

    // First pulse: loads lowHalf, phase → true
    pokeRow(dut, firstRow, expShift)
    dut.clock.step(1)

    // Second pulse: pack and latch into respBitsReg, phase → false
    pokeRow(dut, secondRow, expShift)
    dut.clock.step(1)

    dut.io.req.valid.poke(false.B)
    dut.io.resp.valid.expect(true.B,
      "FP8Pack.resp.valid should assert on cycle after the second req pulse")

    val words = (0 until 16).map { i =>
      dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF
    }
    val firstBytes  = (0 until 8).flatMap { j =>
      val w = words(j)
      Seq(w & 0xFF, (w >>> 8) & 0xFF)
    }
    val secondBytes = (0 until 8).flatMap { j =>
      val w = words(j + 8)
      Seq(w & 0xFF, (w >>> 8) & 0xFF)
    }
    firstBytes ++ secondBytes
  }

  private def resetDrain(dut: FP8Pack): Unit = {
    dut.reset.poke(true.B)
    dut.clock.step(5)
    dut.reset.poke(false.B)
    dut.clock.step(1)
  }

  private def expectPackedMatches(
      firstRow: Seq[Int],
      secondRow: Seq[Int],
      expShift: Int,
      actual: Seq[Int],
      label: String
  ): Unit = {
    val allInputs = firstRow ++ secondRow
    val expected  = allInputs.map(b => goldenFp8ByteFromBf16(b, expShift))
    actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
      withClue(f"$label byte $i: in=0x${allInputs(i)}%04X act=0x$act%02X exp=0x$exp%02X\n") {
        act shouldBe exp
      }
    }
  }

  behavior of "FP8Pack"

  it should "convert 32 BF16 lanes into 32 FP8 bytes with expShift=0" in {
    makeSim("RVFP8PackTest_baseline").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val firstF = Seq(
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 3.25f, 10.0f,
        0.25f, -2.5f, 6.0f, 0.0f, 0.75f, -0.125f, 12.0f, 100.0f
      )
      val secondF = Seq(
        -0.5f, 4.0f, -8.0f, 0.125f, 16.0f, -32.0f, 0.375f, 1.75f,
        -0.0625f, 2.5f, -6.0f, 48.0f, 0.875f, -3.5f, 96.0f, -0.25f
      )

      val firstBf16  = firstF.map(fpTobf)
      val secondBf16 = secondF.map(fpTobf)

      val actual = drivePairAndRead(dut, firstBf16, secondBf16, expShift = 0)
      expectPackedMatches(firstBf16, secondBf16, expShift = 0, actual, "baseline")
    }
  }

  it should "shift exponent correctly with positive expShift" in {
    makeSim("RVFP8PackTest_posShift").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val expShift = 2
      val row = Seq.fill(16)(fpTobf(1.0f))

      val actual = drivePairAndRead(dut, row, row, expShift)
      expectPackedMatches(row, row, expShift, actual, "expShift(+2)")
    }
  }

  it should "underflow to zero with sufficiently positive expShift" in {
    makeSim("RVFP8PackTest_underflow").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      // Large positive expShift pushes the packed result below E4M3 normal range.
      val expShift = 10
      val row = Seq.fill(16)(fpTobf(1.0f))

      val actual = drivePairAndRead(dut, row, row, expShift)
      expectPackedMatches(row, row, expShift, actual, "underflow")
      actual.foreach(_ shouldBe 0)
    }
  }

  it should "clamp overflow to max finite with sufficiently negative expShift" in {
    makeSim("RVFP8PackTest_overflow").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      // Large negative expShift overflows past max finite E4M3.
      val expShift = -10
      val row = Seq.fill(16)(fpTobf(1000.0f))

      val actual = drivePairAndRead(dut, row, row, expShift)
      expectPackedMatches(row, row, expShift, actual, "overflow")
      actual.foreach(_ shouldBe 0x7E) // positive max finite
    }
  }

  it should "handle the exact boundary between minimum normal and flush-to-zero underflow" in {
    makeSim("RVFP8PackTest_minNormBoundary").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val row = Seq.fill(16)(fpTobf(1.0f))

      // expShift = +6 divides by 64 → minimum normal E4M3 (byte 0x08).
      val minNormal = drivePairAndRead(dut, row, row, expShift = 6)
      minNormal.foreach(_ shouldBe 0x08)

      // expShift = +7 divides by 128 → below min normal, flush to zero.
      val underflow = drivePairAndRead(dut, row, row, expShift = 7)
      underflow.foreach(_ shouldBe 0x00)
    }
  }

  it should "handle the exact boundary between max normal and overflow clamping" in {
    makeSim("RVFP8PackTest_maxNormBoundary").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val row = Seq.fill(16)(fpTobf(1.0f))

      // expShift = -8 multiplies by 256 → max E4M3 normal (byte 0x78).
      val maxNormal = drivePairAndRead(dut, row, row, expShift = -8)
      maxNormal.foreach(_ shouldBe 0x78)

      // expShift = -9 multiplies by 512 → overflow, clamp to max finite 0x7E.
      val overflow = drivePairAndRead(dut, row, row, expShift = -9)
      overflow.foreach(_ shouldBe 0x7E)
    }
  }

  it should "ignore expShift for special values (zero, inf, nan)" in {
    makeSim("RVFP8PackTest_specials").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val bf16Zero   = 0x0000
      val bf16PosInf = 0x7F80
      val bf16NegInf = 0xFF80
      val bf16NaN    = 0x7FC0

      val pattern = Seq(
        bf16Zero, bf16PosInf, bf16NegInf, bf16NaN,
        bf16Zero, bf16PosInf, bf16NegInf, bf16NaN,
        bf16Zero, bf16PosInf, bf16NegInf, bf16NaN,
        bf16Zero, bf16PosInf, bf16NegInf, bf16NaN
      )

      val actual = drivePairAndRead(dut, pattern, pattern, expShift = 50)
      val expected = Seq(
        0x00, 0x7E, 0xFE, 0x00,
        0x00, 0x7E, 0xFE, 0x00,
        0x00, 0x7E, 0xFE, 0x00,
        0x00, 0x7E, 0xFE, 0x00
      )
      val fullExpected = expected ++ expected

      actual.zip(fullExpected).zipWithIndex.foreach { case ((a, e), i) =>
        withClue(f"special byte $i: in=0x${(pattern ++ pattern)(i)}%04X act=0x$a%02X exp=0x$e%02X\n") {
          a shouldBe e
        }
      }
    }
  }

  it should "clamp would-be E4M3 NaN encoding to max finite" in {
    makeSim("RVFP8PackTest_clampNaN").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val expShift = 0
      // 480.0 is above max-finite E4M3 (448.0) and previously could land on 0x7F.
      val row = Seq.fill(16)(fpTobf(480.0f))

      val actual = drivePairAndRead(dut, row, row, expShift)
      expectPackedMatches(row, row, expShift, actual, "clamp-NaN")
      actual.foreach(_ shouldBe 0x7E)
    }
  }
}
