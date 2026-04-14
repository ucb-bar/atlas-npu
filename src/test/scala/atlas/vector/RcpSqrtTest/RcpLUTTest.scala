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
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsRcpLUTSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "RcpLUTTest"

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

class RcpLUTTest extends AnyFlatSpec with HasSinCosParams with Matchers with PeekPokeAPI {
  it should "Print values of RcpLUT ranging from 0.5 to 1.0" in {
    val rawExp = 128
    val isOddExp = (((rawExp - 127) & 1) === 1)
    PersistentVcsRcpLUTSimulator.simulate(new RcpLUT(16, 7, 9, 12)) { module =>
      val dut = module.wrapped
      for (i <- 0 until 8) {
        for (k <- 0 until 16) {
          
          dut.io.raddr(k).poke((i * 16 + k).U)
          dut.io.ren(k).poke(true.B)
          // Actual exp = raw exp - bias 
          // Examples: 
          // 255 - 127 = 128 (max normal exponent)
          // 128 - 127 = 1 (exponent for 2.0)
          // 127 - 127 = 0 (exponent for 1.0)
          dut.io.exp(k).poke(rawExp.asUInt) // raw exponent
          
          dut.io.neg(k).poke(true.B) // odd exponent to test shifting
        }

        dut.clock.step(1)

        val bf16Shorts = Array.tabulate(16) { k =>
          dut.io.rdata(k).peek().litValue.toShort
        }

        printBF16Array(bf16Shorts, false)
      }
    }
  }
}