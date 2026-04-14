// ============================================================================
// DivSqrtBlockTest.scala — BF16 divide/sqrt block tests for the vector
// datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.DivSqrtBlockTest
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import sp26FPUnits._
import sp26FPUnits.hardfloat._
import sp26FPUnits.AtlasFPType._
import org.scalatest.Outcome

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

class DivSqrtBlockTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

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
      println("DivSqrtBlockTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("DivSqrtBlockTest=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  // Helper: Convert Double to BF16 bit pattern (top 16 bits of Float)
  def doubleToBF16Int(d: Double): Int = {
    val bits = java.lang.Float.floatToIntBits(d.toFloat)
    (bits >>> 16) & 0xFFFF
  }

  // Helper: Convert BF16 bit pattern back to Double
  def bf16IntToDouble(i: Int): Double = {
    java.lang.Float.intBitsToFloat(i << 16).toDouble
  }

  // Use the parameters defined in your project
  val fptype = AtlasFPType.BF16 
  val numLanes = 4 // Testing with 4 lanes for brevity, can be 16
  val tagWidth = 8

  behavior of "DivSqrtRec"

  it should "compute vectorized reciprocal across multiple lanes" in {
    makeSim("DivSqrtBlockTest_recip").simulate(new DivSqrtRec(fptype, numLanes, tagWidth)) { module =>
      val dut = module.wrapped

      // DivSqrtRec is a reciprocal/sqrt unit: when isSqrt=false it ignores
      // bVec and computes 1 / aVec per lane (see DivSqrtRec.scala:67-68).
      val aValues = Seq(10.0, 1.0, 144.0, 3.14)

      println(s"\n>>> STARTING VECTORIZED RECIPROCAL")

      // Let the HardFloat div/sqrt lanes settle before issuing a request.
      // DivSqrtRec is Valid-only (software-scheduled); its internal `allReady`
      // is a debug signal, not a handshake, so we drain a few cycles instead
      // of polling.
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)

      dut.io.req.valid.poke(true.B)
      dut.io.req.bits.isSqrt.poke(false.B) // Reciprocal
      dut.io.req.bits.laneMask.poke("b1111".U)
      dut.io.req.bits.tag.poke(0xAB.U)

      for (i <- 0 until numLanes) {
        dut.io.req.bits.aVec(i).poke(doubleToBF16Int(aValues(i)).U)
      }

      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      var cycles = 0
      while (!dut.io.resp.valid.peek().litToBoolean && cycles < 100) {
        dut.clock.step(1)
        cycles += 1
      }

      val resp = dut.io.resp.bits
      println(f"Response received in $cycles cycles")

      for (i <- 0 until numLanes) {
        val rawRes = resp.result(i).peek().litValue.toInt
        val actual = bf16IntToDouble(rawRes)
        val expected = 1 / aValues(i)

        println(f"Lane $i: 1 / ${aValues(i)} = $actual%.4f (Expected: $expected%.4f)")

        // BF16 ~2-3 decimal digits of precision; loose tolerance.
        math.abs(actual - expected) should be < 0.05
      }
    }
  }

  it should "compute vectorized square root" in {
    makeSim("DivSqrtBlockTest_sqrt").simulate(new DivSqrtRec(fptype, numLanes, tagWidth)) { module =>
      val dut = module.wrapped
      val aValues = Seq(4.0, 9.0, 2.0, 0.25)

      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)

      dut.io.req.valid.poke(true.B)
      dut.io.req.bits.isSqrt.poke(true.B)
      dut.io.req.bits.laneMask.poke("b1111".U)

      for (i <- 0 until numLanes) {
        dut.io.req.bits.aVec(i).poke(doubleToBF16Int(aValues(i)).U)
      }

      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      while (!dut.io.resp.valid.peek().litToBoolean) dut.clock.step(1)

      println(s"\n>>> STARTING VECTORIZED SQRT")
      for (i <- 0 until numLanes) {
        val actual = bf16IntToDouble(dut.io.resp.bits.result(i).peek().litValue.toInt)
        val expected = math.sqrt(aValues(i))
        println(f"Lane $i: sqrt(${aValues(i)}) = $actual%.4f (Expected: $expected%.4f)")
        math.abs(actual - expected) should be < 0.05
      }
    }
  }
}
