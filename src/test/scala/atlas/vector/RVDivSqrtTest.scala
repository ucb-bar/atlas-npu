// ============================================================================
// RVDivSqrtTest.scala — BF16 divide/sqrt vector tests for the RV-style vector
// datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.RVDivSqrtTest
// ============================================================================
package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common.VpuParams
import svsim.CommonCompilationSettings.Timescale.Unit.s
import sp26FPUnits.hardfloat._
import sp26FPUnits.AtlasFPType.BF16
import atlas.vector.VectorTestUtils.randomBF16Vec
import VectorTestUtils._
import atlas.vector.VPUOp.rcp

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import atlas.common._
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsRVDivSqrtSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "RVDivSqrtTest"

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

class RVDivSqrtTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

    //----------- CI/CD INCLUDE --------------
    override def withFixture(test: NoArgTest): Outcome = {
        val outcome = super.withFixture(test)
        if (outcome.isFailed) {
        println("RVDivSqrtTest=FAILED")
        } else if (outcome.isSucceeded) {
        println("RVDivSqrtTest=PASSED")
        }
        outcome
    }
   //----------- CI/CD INCLUDE --------------
    val p = VpuParams()

    behavior of s"VectorEngine (BF16) Rcp/Sqrt lanes=${16}"
    it should "Verify the correctness of Rcp/Sqrt" in {
        PersistentVcsRVDivSqrtSimulator.simulate(new DivSqrtRec(BF16)) { module =>
            val dut = module.wrapped

            dut.reset.poke(true.B)
            dut.clock.step(3)
            dut.reset.poke(false.B)
            dut.clock.step(1)

            // Test data
            val sqrtData = randomBF16Vec(16, 100, 0)
            val rcpData = randomBF16Vec(16, 100, -100)

            // Expected values from golden model
            val expectedSqrt = goldenRcpSqrtVector(sqrtData, true)
            val expectedRcp = goldenRcpSqrtVector(rcpData, false)

            // Poke inputs for sqrt
            dut.io.req.valid.poke(true.B)
            dut.io.req.bits.tag.poke(0.U)
            dut.io.req.bits.isSqrt.poke(true.B)
            dut.io.req.bits.laneMask.poke(0xFFFF.U)
            sqrtData.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
            rcpData.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.bVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 7) {
                // Print values when output fires
                val outputFire = dut.io.resp.valid.peek().litValue == 1
                if (outputFire) {
                    println(s"cycle $i")
                    val bf16Shorts = Array.tabulate(16) { k => dut.io.resp.bits.result(k).peek().litValue.toShort}

                    printBF16Array(sqrtData, false)
                    printBF16Array(bf16Shorts, false)
                    printBF16Array(expectedSqrt, true)
                    checkVectorTolerance(bf16Shorts, expectedSqrt, true)
                    println("")
                }
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
            }


            // Poking the inputs
            dut.reset.poke(true.B)
            dut.clock.step(10)
            dut.reset.poke(false.B)
            dut.clock.step(1)

            // Poke inputs for rcp
            dut.io.req.valid.poke(true.B)
            dut.io.req.bits.tag.poke(0.U)
            dut.io.req.bits.isSqrt.poke(false.B)
            dut.io.req.bits.laneMask.poke(0xFFFF.U)
            rcpData.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
            rcpData.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.bVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 7) {
                // Print values when output fires
                val outputFire = dut.io.resp.valid.peek().litValue == 1
                if (outputFire) {
                    println(s"cycle $i")
                    val bf16Shorts = Array.tabulate(16) { k => dut.io.resp.bits.result(k).peek().litValue.toShort}

                    printBF16Array(rcpData, false)
                    printBF16Array(bf16Shorts, false)
                    printBF16Array(expectedRcp, true)
                    checkVectorTolerance(bf16Shorts, expectedRcp, true) shouldBe true
                    println("")
                }
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
            }
        }
    }
}
