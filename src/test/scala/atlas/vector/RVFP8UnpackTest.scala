// ============================================================================
// RVFP8UnpackTest.scala — FP8→BF16 tests for the RV-style vector datapath.
//
// FP8Unpack is 1→2 phased: one `req.valid` pulse carrying 16×UInt(16.W)
// (= 32 packed FP8 bytes) produces two consecutive `resp.valid` pulses.
// First pulse carries the low 16 bytes (byte 0..15) as BF16, second pulse
// carries the high 16 bytes (byte 16..31) as BF16. See FP8Unpack.scala
// for the byte layout: `inputBuf := xVec.asUInt`, then
//   fp8Bytes(i) = inputBuf(8*i+7, 8*i)         (sLow)
//   fp8Bytes(i) = inputBuf(128 + 8*i+7, ...)   (sHigh)
// so xVec(i)[7:0] is an even byte and xVec(i)[15:8] is the following odd
// byte — pack inputs as xVec(i) = (bytes(2i+1) << 8) | bytes(2i).
// Each byte converts as BF16_RNE(x * 2^(scale - 127))
// (VectorTestUtils.goldenBf16FromE4m3).
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.RVFP8UnpackTest
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
import atlas.vector.VectorTestUtils.{goldenBf16FromE4m3, goldenE4m3}
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator factory — fresh persistent workspace per test_name (each
// `it should` block gets its own workdir so multi-test files don't trip on
// stale NFS file handles during cleanup between simulate() calls).
// ============================================================================

class RVFP8UnpackTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

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


  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("RVFP8UnpackTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("RVFP8UnpackTest=PASSED")
    }
    outcome
  }

  // ---- BF16 helper for the round-trip test ----
  private def fpTobf(f: Float): Int = {
    val x = java.lang.Float.floatToRawIntBits(f)
    val lsb = (x >>> 16) & 1
    val roundingBias = 0x7FFF + lsb
    ((x + roundingBias) >>> 16) & 0xFFFF
  }

  // ---- DUT helpers for the 1→2 phased interface ----

  private def resetDrain(dut: FP8Unpack): Unit = {
    dut.reset.poke(true.B)
    dut.clock.step(5)
    dut.reset.poke(false.B)
    dut.clock.step(1)
  }

  /** Drive one 32-byte FP8 row. Packs bytes little-endian into xVec such
    * that bytes(2i) occupies xVec(i)[7:0] and bytes(2i+1) occupies
    * xVec(i)[15:8], matching FP8Unpack's `xVec.asUInt` layout.
    */
  private def drivePackedRow(
      dut: FP8Unpack,
      bytes: Seq[Int],
      e: Int
  ): Unit = {
    require(bytes.length == 32, "FP8Unpack expects exactly 32 FP8 bytes per row")
    dut.io.req.valid.poke(true.B)
    dut.io.req.bits.scale.poke(e.U)
    for (i <- 0 until 16) {
      val lo = bytes(2 * i)     & 0xFF
      val hi = bytes(2 * i + 1) & 0xFF
      dut.io.req.bits.xVec(i).poke(((hi << 8) | lo).U)
    }
    dut.clock.step(1)
    dut.io.req.valid.poke(false.B)
  }

  /** Spin for the first resp.valid after drivePackedRow, read both beats,
    * and return 32 BF16 values in byte-order (low 16 then high 16).
    */
  private def readTwoBeats(dut: FP8Unpack): Seq[Int] = {
    var spins = 0
    while (!dut.io.resp.valid.peek().litToBoolean && spins < 20) {
      dut.clock.step(1)
      spins += 1
    }
    dut.io.resp.valid.expect(true.B,
      "FP8Unpack.resp.valid should assert within a few cycles of req")

    val low = (0 until 16).map { i =>
      dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF
    }
    dut.clock.step(1)
    dut.io.resp.valid.expect(true.B,
      "FP8Unpack.resp.valid should remain asserted for the high-half beat")
    val high = (0 until 16).map { i =>
      dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF
    }
    dut.clock.step(1)
    low ++ high
  }

  private def runUnpackCase(
      bytes: Seq[Int],
      e: Int,
      label: String,
      testName: String
  ): Seq[Int] = {
    makeSim(testName).simulate(new FP8Unpack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)
      drivePackedRow(dut, bytes, e)
      val actual = readTwoBeats(dut)
      val expected = bytes.map(b => goldenBf16FromE4m3(b, e))
      actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
        withClue(f"$label byte $i: in=0x${bytes(i)}%02X act=0x$act%04X exp=0x$exp%04X\n") {
          act shouldBe exp
        }
      }
      actual
    }.result
  }

  behavior of "FP8Unpack"

  it should "convert 32 FP8 bytes into 32 BF16 values at e = 127" in {
    val bytes = Seq(
      0x00, 0x30, 0x38, 0x3C, 0x40, 0xB8, 0x45, 0x52,
      0x28, 0xC2, 0x4C, 0x00, 0x34, 0xA0, 0x54, 0x64,
      0x3C, 0x44, 0x48, 0x4A, 0x50, 0xD0, 0x58, 0x60,
      0x20, 0xA8, 0x2C, 0x30, 0x35, 0xB0, 0x55, 0x70
    )
    runUnpackCase(bytes, e = 127, "baseline", "RVFP8UnpackTest_baseline")
  }

  it should "multiply by 2^(e - 127)" in {
    // 0x38 = 1.0 in E4M3
    val bytes = Seq.fill(32)(0x38)
    runUnpackCase(bytes, e = 129, "e = 129", "RVFP8UnpackTest_scale").foreach(_ shouldBe 0x4080)
  }

  it should "map zero FP8 to zero BF16 at any e" in {
    val bytes = Seq.fill(32)(0x00)
    runUnpackCase(bytes, e = 177, "zero", "RVFP8UnpackTest_zero")
  }

  it should "round-trip pack then unpack with only pack-side quantization loss" in {
    val dataF = Seq(
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 3.25f, 10.0f,
      0.25f, -2.5f, 6.0f, 0.0f, 0.75f, -0.125f, 12.0f, 100.0f,
      -0.5f, 4.0f, -8.0f, 0.125f, 16.0f, -32.0f, 0.375f, 1.75f,
      -0.0625f, 2.5f, -6.0f, 48.0f, 0.875f, -3.5f, 96.0f, -0.25f
    )
    val bf16  = dataF.map(fpTobf)
    val bytes = bf16.map(b => goldenE4m3(b, 127))
    runUnpackCase(bytes, e = 127, "roundtrip", "RVFP8UnpackTest_roundtrip")
  }

  it should "keep subnormals and signed zero, and give 0x7FFF for NaN" in {
    //  0x00/0x80 = +-0, 0x01/0x81 = +-2^-9, 0x7F/0xFF = NaN
    val specials = Seq(0x00, 0x80, 0x01, 0x81, 0x7F, 0xFF, 0x38, 0xB8)
    val bytes    = specials ++ specials ++ specials ++ specials
    val actual   = runUnpackCase(bytes, e = 127, "specials", "RVFP8UnpackTest_specials")
    actual.take(8) shouldBe Seq(0x0000, 0x8000, 0x3B00, 0xBB00, 0x7FFF, 0x7FFF, 0x3F80, 0xBF80)
  }

  it should "give Inf on overflow and round BF16 subnormals to nearest even" in {
    // At e = 255, 448 * 2^128 overflows. At e = 0, 2^-6 * 2^-127 is the smallest BF16
    // subnormal, and 2^-9 * 2^-127 rounds to zero.
    val bytes = Seq.tabulate(32)(i => Seq(0x7E, 0xFE, 0x08, 0x88, 0x01, 0x81, 0x09, 0x0C)(i % 8))
    runUnpackCase(bytes, e = 255, "e = 255", "RVFP8UnpackTest_e255").take(2) shouldBe Seq(0x7F80, 0xFF80)
    runUnpackCase(bytes, e = 0, "e = 0", "RVFP8UnpackTest_e0").drop(2).take(4) shouldBe
      Seq(0x0001, 0x8001, 0x0000, 0x8000)
  }
}
