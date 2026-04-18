package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import atlas.common.VpuParams

import org.scalatest.Outcome 

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import java.nio.file.{Files, Path, Paths}

object PersistentVcsFSMOverlapTracking1Simulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val testName = "FSMOverlapTracking1"

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

class FSMOverlapTracking1 extends AnyFlatSpec with Matchers with PeekPokeAPI {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("FSMOverlapTracking1=FAILED")
    } else if (outcome.isSucceeded) {
      println("FSMOverlapTracking1=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  private val p = VpuParams()

  private def driveDefaults(dut: VectorFSM): Unit = {
    dut.io.in.instFire.poke(false.B)
    dut.io.in.instType.poke(VPUOp.add)
    dut.io.in.instReadBank1.poke(0.U)
    dut.io.in.instReadBank2.poke(0.U)
    dut.io.in.instWriteBank.poke(0.U)
    dut.io.in.instImm.poke(0.S)
    dut.io.in.instPackScaleE8M0.poke(127.U)
    dut.io.in.instUnpackScaleE8M0.poke(127.U)
    dut.io.in.dataInFire1.poke(false.B)
    dut.io.in.dataInFire2.poke(false.B)
    dut.io.in.dataOutFire1.poke(false.B)
    dut.io.in.dataOutFire2.poke(false.B)
    dut.io.in.divSqrtReady.poke(false.B)
  }

  behavior of "VectorFSM overlap + tracking"

  it should "collapse duplicate active reads for aliased two-input ops" in {
    PersistentVcsFSMOverlapTracking1Simulator.simulate(new VectorFSM(p)) { module =>
      val dut = module.wrapped
      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)
      driveDefaults(dut)

      dut.io.in.instFire.poke(true.B)
      dut.io.in.instType.poke(VPUOp.add)
      dut.io.in.instReadBank1.poke(16.U)
      dut.io.in.instReadBank2.poke(16.U)
      dut.io.in.instWriteBank.poke(24.U)
      dut.clock.step(1)

      driveDefaults(dut)
      dut.io.out.VEReady.expect(false.B)

      dut.io.out.activeReads(0).valid.expect(true.B)
      dut.io.out.activeReads(0).bits.expect(16.U)
      dut.io.out.activeReads(1).valid.expect(true.B)
      dut.io.out.activeReads(1).bits.expect(17.U)
      dut.io.out.activeReads(2).valid.expect(false.B)
      dut.io.out.activeReads(3).valid.expect(false.B)
    }
  }
}
