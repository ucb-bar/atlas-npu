package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import atlas.common.VpuParams
import sp26FPUnits.hardfloat._
import svsim.CommonCompilationSettings.Timescale.Unit.s
import fpex.FPType.BF16T
import scala.math._
import scala.util.Random
import VectorTestUtils._
import scala.collection.immutable.LazyList.cons
import org.scalatest.Outcome 

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsVETUnitTestingSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "VETUnitTesting"

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

class VETUnitTesting extends AnyFlatSpec with Matchers with PeekPokeAPI {

    //----------- CI/CD INCLUDE --------------
    override def withFixture(test: NoArgTest): Outcome = {
        val outcome = super.withFixture(test)
        if (outcome.isFailed) {
        println("VETUnitTesting=FAILED")
        } else if (outcome.isSucceeded) {
        println("VETUnitTesting=PASSED")
        }
        outcome
    }
    //----------- CI/CD INCLUDE --------------

    val p = VpuParams()

    behavior of s"VET Testing"
    it should "VET Correctness" in {
        PersistentVcsVETUnitTestingSimulator.simulate(new VectorEngine(p)) { module =>
            val dut = module.wrapped
            // Reset VET
            dut.reset.poke(true.B)
            dut.clock.step(1)
            dut.reset.poke(false.B)
            dut.clock.step(1)

            // Use for col reduce max
            val r50 = randomBF16Matrix()
            val r51 = randomBF16Matrix()
            val formatedr50 = packMatrix(r50)
            val formatedr51 = packMatrix(r51)
            val expectedr50 = goldenColMaxMatrix(r50)
            val expectedr51 = goldenColMaxMatrix(r51)
            val finalVecMax = goldenPairwiseMaxVector(expectedr50(0), expectedr51(0))
            val expected50n51 = Array.fill(32)(finalVecMax)

            // Use for col reduce min
            val r52 = randomBF16Matrix()
            val r53 = randomBF16Matrix()
            val formatedr52 = packMatrix(r52)
            val formatedr53 = packMatrix(r53)
            val expectedr52 = goldenColMinMatrix(r52)
            val expectedr53 = goldenColMinMatrix(r53)
            val finalVecMin = goldenPairwiseMinVector(expectedr52(0), expectedr53(0))
            val expected52n53 = Array.fill(32)(finalVecMin)

            // Use for col reduce sum
            val r54 = randomBF16Matrix(useInt = false)   // To avoid overflow in sum, use smaller values for r54
            val r55 = randomBF16Matrix(useInt = false)   // To avoid overflow in sum, use smaller values for r55
            val formatedr54 = packMatrix(r54)
            val formatedr55 = packMatrix(r55)
            val expected54n55 = goldenColSumMatrix(r54 ++ r55)

            for (i <- 1200 until 1600) {
                if (i == 1200) {
                    // Poking for ColwiseMax
                    dut.io.inst.valid.poke(true.B)
                    dut.io.inst.bits.instType.poke(VPUOp.cmax)
                    // Test different read banks
                    dut.io.inst.bits.instReadBank1.poke(50.U)
                    dut.io.inst.bits.instReadBank2.poke(50.U)    
                    dut.io.inst.bits.instWriteBank.poke(50.U)
                } else if (i == 1201) {
                    // Poking for ColwiseMin
                    dut.io.inst.valid.poke(true.B)
                    dut.io.inst.bits.instType.poke(VPUOp.cmin)
                    dut.io.inst.bits.instReadBank1.poke(52.U)
                    dut.io.inst.bits.instReadBank2.poke(52.U)
                    dut.io.inst.bits.instWriteBank.poke(52.U)
                } else if (i == 1400) {
                    // Poking for ColwiseSum
                    dut.io.inst.valid.poke(true.B)
                    dut.io.inst.bits.instType.poke(VPUOp.csum)
                    dut.io.inst.bits.instReadBank1.poke(54.U)
                    dut.io.inst.bits.instReadBank2.poke(54.U)
                    dut.io.inst.bits.instWriteBank.poke(54.U)
                } else {
                    dut.io.inst.valid.poke(false.B)
                }

                val readValid1: Boolean = dut.io.read1.valid.peek().litToBoolean
                val readBank1: Int = dut.io.read1.bits.bank.peek().litValue.toInt
                val readRow1: Int = dut.io.read1.bits.row.peek().litValue.toInt
                val rValid50 = (readValid1) && (readBank1 == 50)
                val rValid51 = (readValid1) && (readBank1 == 51)
                val rValid54 = (readValid1) && (readBank1 == 54)
                val rValid55 = (readValid1) && (readBank1 == 55)
                
                val readValid2: Boolean = dut.io.read2.valid.peek().litToBoolean
                val readBank2: Int = dut.io.read2.bits.bank.peek().litValue.toInt
                val readRow2: Int = dut.io.read2.bits.row.peek().litValue.toInt
                val rValid52 = (readValid2) && (readBank2 == 52)
                val rValid53 = (readValid2) && (readBank2 == 53)

                val writeValid1: Boolean = dut.io.write1.valid.peek().litToBoolean
                val writeBank1: Int = dut.io.write1.bits.bank.peek().litValue.toInt
                val writeRow1: Int = dut.io.write1.bits.row.peek().litValue.toInt
                val wValid50 = (writeValid1 && writeBank1 == 50)
                val wValid51 = (writeValid1 && writeBank1 == 51)
                val wValid54 = (writeValid1 && writeBank1 == 54)
                val wValid55 = (writeValid1 && writeBank1 == 55)

                val writeValid2: Boolean = dut.io.write2.valid.peek().litToBoolean
                val writeBank2: Int = dut.io.write2.bits.bank.peek().litValue.toInt
                val writeRow2: Int = dut.io.write2.bits.row.peek().litValue.toInt
                val wValid52 = (writeValid2 && writeBank2 == 52)
                val wValid53 = (writeValid2 && writeBank2 == 53)

                // Write to the register if writes are valid
                if (wValid50) {
                    r50(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
                } else if (wValid51) {
                    r51(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
                    if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Columnwise Max")
                    }
                } else if (wValid54) {
                    r54(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
                    println(s"Cycle: ${i} write1 valid row=${writeRow1}, bank=${writeBank1}")
                } else if (wValid55) {
                    r55(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
                    if (writeRow1 == 31) {
                        println(s"Cycle: ${i} => last row written for Columnwise Sum")
                        printLane(expected54n55, 30, expected = true)
                        printLane(r55, 30, expected = false)
                    }
                    println(s"Cycle: ${i} write1 valid row=${writeRow1}, bank=${writeBank1}")
                }

                if (wValid52) {
                    r52(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
                } else if (wValid53) {
                    r53(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
                    if (writeRow2 == 31) {println(s"Cycle: ${i} => last row written for Columnwise Min")
                    }
                }

                dut.clock.step(1)

                // Send data to the VET if reads were asserted last cycle
                if (rValid50) {
                    dut.io.data1.valid.poke(true.B)
                    dut.io.data1.bits.data.poke(formatedr50(readRow1).U)
                } else if (rValid51) {
                    dut.io.data1.valid.poke(true.B)
                    dut.io.data1.bits.data.poke(formatedr51(readRow1).U)
                } else if (rValid54) {
                    dut.io.data1.valid.poke(true.B)
                    dut.io.data1.bits.data.poke(formatedr54(readRow1).U)
                } else if (rValid55) {
                    dut.io.data1.valid.poke(true.B)
                    dut.io.data1.bits.data.poke(formatedr55(readRow1).U)
                } else {
                    dut.io.data1.valid.poke(false.B)
                } 

                if (rValid52) {
                    dut.io.data2.valid.poke(true.B)
                    dut.io.data2.bits.data.poke(formatedr52(readRow2).U)
                } else if (rValid53) {
                    dut.io.data2.valid.poke(true.B)
                    dut.io.data2.bits.data.poke(formatedr53(readRow2).U)
                } else {
                    dut.io.data2.valid.poke(false.B)
                }
            }

            // Exact checks: max and min select an operand, and the column sum is defined bit for bit.
            val results4 = Seq(
                ("r50", checkMatrixTolerance(r50, expected50n51, "r50", 0f, 0f)),
                ("r51", checkMatrixTolerance(r51, expected50n51, "r51", 0f, 0f)),
                ("r52", checkMatrixTolerance(r52, expected52n53, "r52", 0f, 0f)),
                ("r53", checkMatrixTolerance(r53, expected52n53, "r53", 0f, 0f)),
                ("r54", checkMatrixTolerance(r54, expected54n55, "r54", 0f, 0f)),
                ("r55", checkMatrixTolerance(r55, expected54n55, "r55", 0f, 0f))
            )

            println(s"Column reductions: passed ${results4.count(_._2)} / ${results4.length} tests")
            results4.filterNot(_._2).map(_._1) shouldBe empty
        }.result
    }
}