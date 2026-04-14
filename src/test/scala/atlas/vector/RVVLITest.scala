// ============================================================================
// Tests Vector Load Immediates
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

class RVVLITest extends AnyFlatSpec with Matchers with PeekPokeAPI {

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
      println("RVVLITest=FAILED")
    } else if (outcome.isSucceeded) {
      println("RVVLITest=PASSED")
    }
    outcome
  }

  def fpTobf(f: Float): Int = {
    val x = java.lang.Float.floatToRawIntBits(f)
    val lsb = (x >>> 16) & 1
    val roundingBias = 0x7FFF + lsb
    ((x + roundingBias) >>> 16) & 0xFFFF
  }

  private def driveReq(
      dut: VectorLoadImm,
      op: VPUOp.Type,
      immBits: Int,
      rowIdx: Int
  ): Unit = {
    dut.io.req.valid.poke(true.B)
    dut.io.req.bits.op.poke(op)
    dut.io.req.bits.imm.poke((immBits & 0xFFFF).S)
    dut.io.req.bits.rowIdx.poke(rowIdx.U)
  }

  private def readOut(dut: VectorLoadImm): Seq[Int] = {
    (0 until 16).map(i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF)
  }

  behavior of "VectorLoadImm"

  it should "do vliAll and vliRow correctly" in {
    makeSim("RVVLITest_vliAllRow").simulate(new VectorLoadImm(BF16, numLanes = 16, rowIdxWidth = 5)) { module =>
      val dut = module.wrapped
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val immAll = fpTobf(1.5f)
      val immRow = fpTobf(-2.25f)

      driveReq(dut, VPUOp.vliAll, immAll, rowIdx = 7)
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val outAll = readOut(dut)
      outAll.foreach(_ shouldBe immAll)

      dut.clock.step(1)

      driveReq(dut, VPUOp.vliRow, immRow, rowIdx = 0)
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val outRow0 = readOut(dut)
      outRow0.foreach(_ shouldBe immRow)

      dut.clock.step(1)

      driveReq(dut, VPUOp.vliRow, immRow, rowIdx = 3)
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val outRow3 = readOut(dut)
      outRow3.foreach(_ shouldBe 0)
    }
  }

  it should "do vliCol and vliOne correctly" in {
    makeSim("RVVLITest_vliColOne").simulate(new VectorLoadImm(BF16, numLanes = 16, rowIdxWidth = 5)) { module =>
      val dut = module.wrapped
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val immCol = fpTobf(3.75f)
      val immOne = fpTobf(-0.5f)

      driveReq(dut, VPUOp.vliCol, immCol, rowIdx = 4)
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val outCol = readOut(dut)
      outCol.head shouldBe immCol
      outCol.tail.foreach(_ shouldBe 0)

      dut.clock.step(1)

      driveReq(dut, VPUOp.vliOne, immOne, rowIdx = 0)
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val outOne0 = readOut(dut)
      outOne0.head shouldBe immOne
      outOne0.tail.foreach(_ shouldBe 0)

      dut.clock.step(1)

      driveReq(dut, VPUOp.vliOne, immOne, rowIdx = 5)
      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      val outOne5 = readOut(dut)
      outOne5.foreach(_ shouldBe 0)
    }
  }

  it should "only write row 0 for vliRow and zero all other rows" in {
    makeSim("RVVLITest_vliRowOnly").simulate(new VectorLoadImm(BF16, numLanes = 16, rowIdxWidth = 5)) { module =>
      val dut = module.wrapped
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      val imm = fpTobf(2.0f)

      for (row <- 0 until 16) {
        driveReq(dut, VPUOp.vliRow, imm, rowIdx = row)

        var cycles = 0
        cycles = 1
        dut.clock.step(1)

        dut.clock.step(1)
        dut.io.req.valid.poke(false.B)

        val out = readOut(dut)

        if (row == 0) {
          out.foreach { value =>
            value shouldBe imm
          }
        } else {
          out.foreach { value =>
            value shouldBe 0
          }
        }

        dut.clock.step(1)
      }
    }
  }
}
