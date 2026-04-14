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

object PersistentVcsPairwiseMaxSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "PairwiseMaxTest"

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
      line = true, cond = true, branch = true, fsm = true, tgl = true
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

class PairwiseMaxTest extends AnyFlatSpec with HasSinCosParams with Matchers with PeekPokeAPI {
  it should "Verify function of PairwiseMax" in {
    PersistentVcsPairwiseMaxSimulator.simulate(new PairWiseMax(AtlasFPType.BF16)) { module =>
      val dut = module.wrapped
      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)
      
      println("The following are PairwiseMax")
      for (i <- 0 until 10) {
        // Randonmize input data and compute expected output
        val inputData1 = randomBF16Vec(16, 100, -100)
        val inputData2 = randomBF16Vec(16, 100, -100)
        val expectedOutput = goldenPairwiseMaxVector(inputData1, inputData2)

        dut.io.req.valid.poke(true.B)
        dut.io.req.bits.laneMask.poke(0xFFFF)  // Enable all lanes
        inputData1.zipWithIndex.foreach { case (value, idx) =>
          dut.io.req.bits.aVec(idx).poke(BigInt(value & 0xFFFF))
        }
        inputData2.zipWithIndex.foreach { case (value, idx) =>
          dut.io.req.bits.bVec(idx).poke(BigInt(value & 0xFFFF))
        }

        dut.clock.step(1)
        val outputFire = dut.io.resp.valid.peek().litValue == 1
        if (outputFire) {
          val bf16Shorts = Array.tabulate(16) { k =>
            dut.io.resp.bits.result(k).peek().litValue.toShort
          }
          println(s"Cycle = ${i+1}")
          println("Inputs")
          printBF16Array(inputData1, false)
          printBF16Array(inputData2, false)
          println("Output")
          printBF16Array(bf16Shorts, false)
          printBF16Array(expectedOutput, true)
          checkVectorTolerance(bf16Shorts, expectedOutput, true)
          println("")
        }
      }
    }
  }
}
