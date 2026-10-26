// ============================================================================
// RVFP8PackTest.scala — BF16→FP8 pack tests for the RV-style vector datapath.
//
// FP8Pack is 2→1 phased: two sequential `req.valid` pulses (16 BF16 lanes
// each) produce one `resp.valid` pulse carrying 16×UInt(16.W), where each
// 16-bit slot holds two packed FP8 bytes. Each byte is
// E4M3_RN_SATFINITE(x * 2^(scale - 127)) (VectorTestUtils.goldenE4m3).
// Output layout (FP8Pack.scala):
//
//   result(j)     = Cat(lowHalf(2j+1),  lowHalf(2j))   // first-pulse bytes
//   result(j+8)   = Cat(fp8Bytes(2j+1), fp8Bytes(2j))  // second-pulse bytes
//
// RUN: (from atlas-npu)
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
import atlas.vector.VectorTestUtils.goldenE4m3
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
          line = true, cond = true, branch = true, fsm = true, tgl = true, assert = true
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

  // ---- BF16 helper ----
  def fpTobf(f: Float): Int = {
    val x = java.lang.Float.floatToRawIntBits(f)
    val lsb = (x >>> 16) & 1
    val roundingBias = 0x7FFF + lsb
    ((x + roundingBias) >>> 16) & 0xFFFF
  }

  // ---- DUT helpers for the 2→1 phased interface ----

  private def pokeRow(dut: FP8Pack, dataBf16: Seq[Int], e: Int): Unit = {
    dut.io.req.valid.poke(true.B)
    dut.io.req.bits.scale.poke(e.U)
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
      e: Int
  ): Seq[Int] = {
    require(firstRow.length == 16 && secondRow.length == 16,
      "FP8Pack expects exactly 16 BF16 lanes per pulse")

    // First pulse: loads lowHalf, phase → true
    pokeRow(dut, firstRow, e)
    dut.clock.step(1)

    // Second pulse: pack and latch into respBitsReg, phase → false
    pokeRow(dut, secondRow, e)
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
      e: Int,
      actual: Seq[Int],
      label: String
  ): Unit = {
    val allInputs = firstRow ++ secondRow
    val expected  = allInputs.map(b => goldenE4m3(b, e))
    actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
      withClue(f"$label byte $i: in=0x${allInputs(i)}%04X act=0x$act%02X exp=0x$exp%02X\n") {
        act shouldBe exp
      }
    }
  }

  behavior of "FP8Pack"

  it should "convert 32 BF16 lanes into 32 FP8 bytes at e = 127" in {
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

      val actual = drivePairAndRead(dut, firstBf16, secondBf16, e = 127)
      expectPackedMatches(firstBf16, secondBf16, e = 127, actual, "baseline")
    }.result
  }

  it should "multiply by 2^(e - 127)" in {
    makeSim("RVFP8PackTest_scale").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val row = Seq.fill(16)(fpTobf(1.0f))

      val actual = drivePairAndRead(dut, row, row, e = 129)
      expectPackedMatches(row, row, e = 129, actual, "e = 129")
      actual.foreach(_ shouldBe 0x48) // 4.0
    }.result
  }

  it should "keep E4M3 subnormals and round below them to nearest even" in {
    makeSim("RVFP8PackTest_subnormal").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val row = Seq.fill(16)(fpTobf(1.0f))

      // 2^-6 is the smallest normal (0x08); 2^-7 and 2^-9 are subnormals.
      drivePairAndRead(dut, row, row, e = 121).foreach(_ shouldBe 0x08)
      drivePairAndRead(dut, row, row, e = 120).foreach(_ shouldBe 0x04)
      drivePairAndRead(dut, row, row, e = 118).foreach(_ shouldBe 0x01)

      // 2^-10 is the tie between 0 and 2^-9: nearest even gives zero, with the sign of x.
      val signs = Seq.tabulate(16)(i => fpTobf(if (i % 2 == 0) 1.0f else -1.0f))
      val tie   = drivePairAndRead(dut, signs, signs, e = 117)
      expectPackedMatches(signs, signs, e = 117, tie, "tie")
      tie.zipWithIndex.foreach { case (b, i) => b shouldBe (if (i % 2 == 0) 0x00 else 0x80) }
    }.result
  }

  it should "saturate above 448 to max finite" in {
    makeSim("RVFP8PackTest_overflow").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val row = Seq.fill(16)(fpTobf(1.0f))

      // e = 135 multiplies by 256: max E4M3 normal exponent (byte 0x78).
      drivePairAndRead(dut, row, row, e = 135).foreach(_ shouldBe 0x78)

      // e = 136 multiplies by 512: saturates to 0x7E (448).
      drivePairAndRead(dut, row, row, e = 136).foreach(_ shouldBe 0x7E)

      // 480 and the tie 464 give 448, never the NaN code 0x7F.
      val big = Seq.tabulate(16)(i => fpTobf(if (i % 2 == 0) 480.0f else -464.0f))
      val sat = drivePairAndRead(dut, big, big, e = 127)
      expectPackedMatches(big, big, e = 127, sat, "saturate")
      sat.zipWithIndex.foreach { case (b, i) => b shouldBe (if (i % 2 == 0) 0x7E else 0xFE) }
    }.result
  }

  it should "use 2^128 for e = 255" in {
    makeSim("RVFP8PackTest_e255").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      // BF16 subnormal 2^-128 times 2^128 is 1.0.
      val row = Seq.fill(16)(0x0020)
      drivePairAndRead(dut, row, row, e = 255).foreach(_ shouldBe 0x38)
    }.result
  }

  it should "give signed zero, 448 for Inf and 0x7F for NaN" in {
    makeSim("RVFP8PackTest_specials").simulate(new FP8Pack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)

      val pattern = Seq(0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x0001)
      val row     = pattern ++ pattern

      val actual   = drivePairAndRead(dut, row, row, e = 177)
      val expected = Seq(0x00, 0x80, 0x7E, 0xFE, 0x7F, 0x7F, 0x7F, 0x00)

      actual.zip(Seq.fill(4)(expected).flatten).zipWithIndex.foreach { case ((a, e), i) =>
        withClue(f"special byte $i: in=0x${(row ++ row)(i)}%04X act=0x$a%02X exp=0x$e%02X\n") {
          a shouldBe e
        }
      }
      expectPackedMatches(row, row, e = 177, actual, "specials")
    }.result
  }
}
