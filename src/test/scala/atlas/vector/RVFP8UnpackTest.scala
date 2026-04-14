// ============================================================================
// RVFP8UnpackTest.scala — FP8→BF16 tests for the RV-style vector datapath.
//
// FP8Unpack is 1→2 phased: one `req.valid` pulse carrying 16×UInt(16.W)
// (= 32 packed FP8 bytes) produces two consecutive `resp.valid` pulses.
// First pulse carries the low 16 bytes (byte 0..15) as BF16, second pulse
// carries the high 16 bytes (byte 16..31) as BF16. See FP8Unpack.scala:57-64
// for the byte layout: `inputBuf := xVec.asUInt`, then
//   fp8Bytes(i) = inputBuf(8*i+7, 8*i)         (sLow)
//   fp8Bytes(i) = inputBuf(128 + 8*i+7, ...)   (sHigh)
// so xVec(i)[7:0] is an even byte and xVec(i)[15:8] is the following odd
// byte — pack inputs as xVec(i) = (bytes(2i+1) << 8) | bytes(2i).
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


  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("RVFP8UnpackTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("RVFP8UnpackTest=PASSED")
    }
    outcome
  }

  // ---- FP8 E4M3 → BF16 golden (matches FP8Unpack.scala) ----
  private def goldenBf16FromFp8Byte(fp8: Int, expShift: Int): Int = {
    val sign    = (fp8 >>> 7) & 0x1
    val expFP8  = (fp8 >>> 3) & 0xF
    val mantFP8 = fp8 & 0x7

    val isZero = (expFP8 == 0) && (mantFP8 == 0)
    val isSub  = (expFP8 == 0) && (mantFP8 != 0)
    val isNaN  = (expFP8 == 0xF) && (mantFP8 == 0x7)
    if (isZero || isSub || isNaN) return (sign << 15) & 0xFFFF

    val unbExpFP8  = expFP8 - 7
    val unbExpBF16 = unbExpFP8 + expShift
    val expBF16    = unbExpBF16 + 127

    if (expBF16 <= 0) {
      (sign << 15) & 0xFFFF
    } else if (expBF16 >= 255) {
      if (sign == 1) 0xFF7F else 0x7F7F
    } else {
      val fracBF16 = (mantFP8 & 0x7) << 4
      ((sign & 0x1) << 15) | ((expBF16 & 0xFF) << 7) | (fracBF16 & 0x7F)
    }
  }

  // ---- BF16 helpers for round-trip test ----
  private def fpTobf(f: Float): Int = {
    val x = java.lang.Float.floatToRawIntBits(f)
    val lsb = (x >>> 16) & 1
    val roundingBias = 0x7FFF + lsb
    ((x + roundingBias) >>> 16) & 0xFFFF
  }

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

      val finalExpAdjusted = if (normCarry) expAdjusted + 1 else expAdjusted
      val mantFP8 = if (normCarry) 0 else (roundedSig - 8) & 0x7

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
      expShift: Int
  ): Unit = {
    require(bytes.length == 32, "FP8Unpack expects exactly 32 FP8 bytes per row")
    dut.io.req.valid.poke(true.B)
    dut.io.req.bits.expShift.poke(expShift.S)
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
      expShift: Int,
      label: String,
      testName: String
  ): Unit = {
    makeSim(testName).simulate(new FP8Unpack(BF16, numLanes = 16)) { module =>
      val dut = module.wrapped
      resetDrain(dut)
      drivePackedRow(dut, bytes, expShift)
      val actual = readTwoBeats(dut)
      val expected = bytes.map(b => goldenBf16FromFp8Byte(b, expShift))
      actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
        withClue(f"$label byte $i: in=0x${bytes(i)}%02X act=0x$act%04X exp=0x$exp%04X\n") {
          act shouldBe exp
        }
      }
    }
  }

  behavior of "FP8Unpack"

  it should "convert 32 FP8 bytes into 32 BF16 values with expShift=0" in {
    val bytes = Seq(
      0x00, 0x30, 0x38, 0x3C, 0x40, 0xB8, 0x45, 0x52,
      0x28, 0xC2, 0x4C, 0x00, 0x34, 0xA0, 0x54, 0x64,
      0x3C, 0x44, 0x48, 0x4A, 0x50, 0xD0, 0x58, 0x60,
      0x20, 0xA8, 0x2C, 0x30, 0x35, 0xB0, 0x55, 0x70
    )
    runUnpackCase(bytes, expShift = 0, "baseline", "RVFP8UnpackTest_baseline")
  }

  it should "undo positive expShift correctly" in {
    // 0x38 ≈ 1.0 in E4M3
    val bytes = Seq.fill(32)(0x38)
    runUnpackCase(bytes, expShift = 2, "expShift(+2)", "RVFP8UnpackTest_posShift")
  }

  it should "map zero FP8 to zero BF16 regardless of expShift" in {
    val bytes = Seq.fill(32)(0x00)
    runUnpackCase(bytes, expShift = 50, "zero", "RVFP8UnpackTest_zero")
  }

  it should "round-trip pack then unpack with only pack-side quantization loss" in {
    val dataF = Seq(
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 3.25f, 10.0f,
      0.25f, -2.5f, 6.0f, 0.0f, 0.75f, -0.125f, 12.0f, 100.0f,
      -0.5f, 4.0f, -8.0f, 0.125f, 16.0f, -32.0f, 0.375f, 1.75f,
      -0.0625f, 2.5f, -6.0f, 48.0f, 0.875f, -3.5f, 96.0f, -0.25f
    )
    val bf16  = dataF.map(fpTobf)
    val bytes = bf16.map(b => goldenFp8ByteFromBf16(b, 0))
    runUnpackCase(bytes, expShift = 0, "roundtrip", "RVFP8UnpackTest_roundtrip")
  }

  it should "flush FP8 NaN/subnormal and preserve signed zero" in {
    //  0x00/0x80 = ±0, 0x01/0x81 = ±subnormal → ±0, 0x7F/0xFF = ±NaN → ±0
    val specials = Seq(0x00, 0x80, 0x01, 0x81, 0x7F, 0xFF, 0x38, 0xB8)
    val bytes    = specials ++ specials ++ specials ++ specials
    runUnpackCase(bytes, expShift = 0, "specials", "RVFP8UnpackTest_specials")
  }
}
