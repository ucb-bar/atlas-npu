// ============================================================================
// RVExpTest.scala — BF16 exponent vector tests for the RV-style vector
// datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.RVExpTest
// ============================================================================
package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import sp26FPUnits._
import VectorTestUtils._
import os.Source.WritableSource

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import atlas.common._
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsRVExpSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "RVExpTest"

  private val runDir: Path = {
    val rootDirStr = sys.env.getOrElse("MILL_WORKSPACE_ROOT", "/tmp")
    val baseDir = Paths.get(rootDirStr)
    val p = baseDir.resolve("tmp").resolve(test_name)
    Files.createDirectories(p)
    p.toAbsolutePath
  }

  override val backend: VcsBackend   = VcsBackend.initializeFromProcessEnvironment()
  override val tag: String           = test_name
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
        coverageName      = Some(Backend.CoverageName(s"${test_name}_coverage"))
      )
    )
  }
}

class RVExpTest extends AnyFlatSpec with HasSinCosParams with Matchers with PeekPokeAPI {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("RVExpTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("RVExpTest=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  val inputSize = 10
  val maxInput = 10
  val minInput = -10
  val intputArray = Array.fill(inputSize)(randomBF16Vec(16, maxInput, minInput))
  val expectedOutputArray = intputArray.map { arr => goldenExpVector(arr, false) }

  it should "Verify correctness of ExpVec" in {
    PersistentVcsRVExpSimulator.simulate(new Exp(AtlasFPType.BF16)) { module =>
      val dut = module.wrapped

      println("The following tests old EXP")
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      dut.io.req.bits.laneMask.poke("hFFFF".U) // enable all lanes
      for (i <- 0 until inputSize) {
        dut.io.req.valid.poke(true.B)
        dut.io.req.bits.isBase2.poke(false.B) // Test natural exponent first
        intputArray(i).zipWithIndex.foreach { case (value, idx) =>
          dut.io.req.bits.xVec(idx).poke(BigInt(value & 0xFFFF))
        }

        dut.clock.step(1)
        val outputFire = dut.io.resp.valid.peek().litValue == 1
        if (outputFire) {
          val bf16Shorts = Array.tabulate(16) { k => dut.io.resp.bits.result(k).peek().litValue.toShort }

          printBF16Array(bf16Shorts, false)
          printBF16Array(expectedOutputArray(i), true)
          checkVectorTolerance(bf16Shorts, expectedOutputArray(i), true, false)
          println("")
        }
      }
    }
  }
}
