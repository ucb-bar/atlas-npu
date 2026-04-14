// ============================================================================
// VectorEngineTopVLIVectorTest.scala
//
// Runs the `vli` family (vliOne, vliRow, vliCol, vliAll) from
// `/vpu_test_vectors/vpu_vli_vectors.txt`.
//
// RUN:
//    mill atlas.test.testOnly atlas.vector.VectorEngineTopVLIVectorTest
// ============================================================================
package atlas.vector

import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsVectorEngineTopVLIVectorSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "VectorEngineTopVLIVectorTest"

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

class VectorEngineTopVLIVectorTest
    extends AnyFlatSpec
    with Matchers
    with VpuVectorTestSupport {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed)         println("VectorEngineTopVLIVectorTest=FAILED")
    else if (outcome.isSucceeded) println("VectorEngineTopVLIVectorTest=PASSED")
    outcome
  }

  behavior of "VectorEngineTop vli family"

  it should "match Python golden for vliOne/vliRow/vliCol/vliAll" in {
    runVectorFamilyTest("vli", PersistentVcsVectorEngineTopVLIVectorSimulator)
  }
}
