package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import sp26FPUnits._
import VectorTestUtils._
import os.Source.WritableSource
import VectorTestUtils._

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsSqrtSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "SqrtTest"

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

class SqrtTest extends AnyFlatSpec with HasSinCosParams with Matchers with PeekPokeAPI {
  it should "Print values of SqrtLUT ranging from 0.5 to 1.0" in {
    PersistentVcsSqrtSimulator.simulate(new Sqrt(AtlasFPType.BF16)) { module =>
      val dut = module.wrapped
      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)
      
      dut.io.req.bits.roundingMode.poke(0.U)
      dut.io.req.bits.laneMask.poke("hFFFF".U) // enable all lanes
      
      for (i <- 0 until 100) {
        // Randonmize input data and compute expected output
        val inputData = randomBF16Vec(16, 10000, 0)
        val expectedOutput = goldenRcpSqrtVector(inputData, true)

        dut.io.req.valid.poke(true.B)
        inputData.zipWithIndex.foreach { case (value, idx) =>
          dut.io.req.bits.aVec(idx).poke(BigInt(value & 0xFFFF))
        }

        dut.clock.step(1)
        val outputFire = dut.io.resp.valid.peek().litValue == 1
        if (outputFire) {
          val bf16Shorts = Array.tabulate(16) { k =>
            dut.io.resp.bits.result(k).peek().litValue.toShort
          }

          printBF16Array(bf16Shorts, false)
          printBF16Array(expectedOutput, true)
          checkVectorTolerance(bf16Shorts, expectedOutput, true)
          println("")
        }
      }
    }
  }
}
