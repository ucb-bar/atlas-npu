// ============================================================================
// RVSqCbTest.scala — Legacy square/cube vector tests kept in commented form.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.RVSqCbTest
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
import VectorTestUtils._

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

class RVSqCbTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

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
            line = true, cond = true, branch = true, fsm = true, tgl = true, assert = true
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
        println("RVSqCbTest=FAILED")
        } else if (outcome.isSucceeded) {
        println("RVSqCbTest=PASSED")
        }
        outcome
    }
    //----------- CI/CD INCLUDE --------------

    val p = VpuParams()

    it should "Verify the correctness of square" in {
        makeSim("RVSqCbTest_square").simulate(new SquareCubeVec(BF16)) { module =>
            val dut = module.wrapped
            // Test data
            val dataA = randomBF16Vec()
            val dataB = randomBF16Vec()

            // Expected values from golden model
            val expectedSquare1 = goldenSquareCubeVector(dataA, false)
            val expectedSquare2 = goldenSquareCubeVector(dataB, false)

            // Poke inputs
            dut.io.req.valid.poke(true.B)
            dut.io.req.bits.isCube.poke(false.B)
            dut.io.req.bits.roundingMode.poke(0.U) 
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 5) {
                if (i == 1) {
                    dut.io.req.valid.poke(true.B)
                    dut.io.req.bits.isCube.poke(false.B)
                    dut.io.req.bits.roundingMode.poke(0.U) 
                    dut.io.req.bits.laneMask.poke(0xFFFF.U) 
                    dataB.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
                }

                val iv = dut.io.req.valid.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1
                val respFire = ov == 1

                // Print values at after two cycles
                if (respFire) {
                    println(s"Cycle = $i")
                    val actual: Array[Short] = (0 until 16).map { i => (dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF).toShort}.toArray
                    
                    if (i == 1) {
                        printBF16Array(dataA, false)
                        printBF16Array(actual, false)
                        printBF16Array(expectedSquare1, true)
                        checkVectorTolerance(actual, expectedSquare1, true)
                    } else if (i == 2) {
                        printBF16Array(dataB, false)
                        printBF16Array(actual, false)
                        printBF16Array(expectedSquare2, true)
                        checkVectorTolerance(actual, expectedSquare2, true)
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
            }
        }
    }

    it should "Verify the correctness of cube" in {
        makeSim("RVSqCbTest_cube").simulate(new SquareCubeVec(BF16)) { module =>
            val dut = module.wrapped
            // Test data
            val dataA = randomBF16Vec()
            val dataB = randomBF16Vec()
            val dataC = randomBF16Vec()

            // Expected values from golden model
            val expectedCube1 = goldenSquareCubeVector(dataA, true)
            val expectedCube2 = goldenSquareCubeVector(dataB, true)
            val expectedCube3 = goldenSquareCubeVector(dataC, true)

            // Poke inputs
            dut.io.req.valid.poke(true.B)
            dut.io.req.bits.isCube.poke(true.B)
            dut.io.req.bits.roundingMode.poke(0.U) 
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 5) {
                if (i == 1) {
                    dut.io.req.valid.poke(true.B)
                    dut.io.req.bits.isCube.poke(true.B)
                    dut.io.req.bits.roundingMode.poke(0.U) 
                    dut.io.req.bits.laneMask.poke(0xFFFF.U) 
                    dataB.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
                } else if (i == 2) {
                    dut.io.req.valid.poke(true.B)
                    dut.io.req.bits.isCube.poke(true.B)
                    dut.io.req.bits.roundingMode.poke(0.U) 
                    dut.io.req.bits.laneMask.poke(0xFFFF.U) 
                    dataC.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
                }
                val iv = dut.io.req.valid.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1
                val respFire = ov == 1

                // Print values at after two cycles
                if (respFire) {
                    println(s"Cycle = $i")
                    val actual: Array[Short] = (0 until 16).map { i => (dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF).toShort}.toArray
                    if (i == 1) {
                        printBF16Array(dataA, false)
                        printBF16Array(actual, false)
                        printBF16Array(expectedCube1, true)
                        checkVectorTolerance(actual, expectedCube1, true)
                    } else if (i == 2) {
                        printBF16Array(dataB, false)
                        printBF16Array(actual, false)
                        printBF16Array(expectedCube2, true)
                        checkVectorTolerance(actual, expectedCube2, true)
                    } else if (i == 3) {
                        printBF16Array(dataC, false)
                        printBF16Array(actual, false)
                        printBF16Array(expectedCube3, true)
                        checkVectorTolerance(actual, expectedCube3, true)
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
            }
        }
    }
}
