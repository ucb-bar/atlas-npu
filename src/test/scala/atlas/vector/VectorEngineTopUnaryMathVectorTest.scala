// ============================================================================
// VectorEngineTopUnaryMathVectorTest.scala
//
// Runs the `unary_math` family (sqrt, log, exp, exp2, square, cube, sin,
// cos, tanh) from `/vpu_test_vectors/vpu_unary_math_vectors.txt`.
//
// RUN:
//    mill atlas.test.testOnly atlas.vector.VectorEngineTopUnaryMathVectorTest
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

object PersistentVcsVectorEngineTopUnaryMathVectorSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "VectorEngineTopUnaryMathVectorTest"

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

class VectorEngineTopUnaryMathVectorTest
    extends AnyFlatSpec
    with Matchers
    with VpuVectorTestSupport {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed)         println("VectorEngineTopUnaryMathVectorTest=FAILED")
    else if (outcome.isSucceeded) println("VectorEngineTopUnaryMathVectorTest=PASSED")
    outcome
  }

  behavior of "VectorEngineTop unary_math family"

  it should "match Python golden for sqrt/log/exp/exp2/square/cube/sin/cos/tanh" in {
    runVectorFamilyTest("unary_math", PersistentVcsVectorEngineTopUnaryMathVectorSimulator)
  }
}
