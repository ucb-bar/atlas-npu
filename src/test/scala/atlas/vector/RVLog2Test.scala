// ============================================================================
// RVLog2Test.scala — BF16 log2 vector tests for the RV-style vector datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.RVLog2Test
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
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsRVLog2Simulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "RVLog2Test"

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

class RVLog2Test extends AnyFlatSpec with Matchers with PeekPokeAPI {

    //----------- CI/CD INCLUDE --------------
    override def withFixture(test: NoArgTest): Outcome = {
        val outcome = super.withFixture(test)
        if (outcome.isFailed) {
        println("RVLog2Test=FAILED")
        } else if (outcome.isSucceeded) {
        println("RVLog2Test=PASSED")
        }
        outcome
    }
    //----------- CI/CD INCLUDE --------------

    val p = VpuParams()
    
    def fpTobf(f: Float): Int = {
        val x = java.lang.Float.floatToRawIntBits(f)
        val lsb = (x >>> 16) & 1
        val roundingBias = 0x7FFF + lsb
        ((x + roundingBias) >>> 16) & 0xFFFF
    }

    def bfTofp(b: Int): Float =
        java.lang.Float.intBitsToFloat((b & 0xFFFF) << 16)

    def goldenLog16(aVec: Seq[Float], bVec: Seq[Float]): Seq[Int] = {
        require(aVec.length == 16 && bVec.length == 16)

        aVec.zip(bVec).map { case (a, b) =>
            val aB = bfTofp(fpTobf(a))  
            val bB = bfTofp(fpTobf(b))  
            val y  = Math.log(aB) / Math.log(2)
            val yFloat = y.toFloat
            fpTobf(yFloat)                   // BF16 result bits as Int
        }
    }

    behavior of s"VectorEngine (BF16) Log2 lanes=${16}"
    it should "Verify the correctness of add" in {
        PersistentVcsRVLog2Simulator.simulate(new Log(BF16)) { module =>
            val dut = module.wrapped
            // Test data
            val dataA = Seq.fill(16)(scala.util.Random.nextFloat() * 1000)
            val dataB = Seq.fill(16)(scala.util.Random.nextFloat() * 1000)

            // Expected values from golden model
            val expected: Seq[Int] = goldenLog16(dataA, dataB)

            // Convert inputs to BF16 bits using the SAME converter as golden
            val dataA_bf16: Seq[Int] = dataA.map(fpTobf)
            val dataB_bf16: Seq[Int] = dataB.map(fpTobf)

            // Poke inputs
            dut.io.req.valid.poke(true.B)
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
            var failed = 0
            for (i <- 0 to 2) {
                val iv = dut.io.req.valid.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1
                val respFire = ov == 1
                println(f"[cycle=$i%2d] req: v=$iv%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d f=${if(respFire)1 else 0}%d")
                // Print values at after two cycles
                if (respFire && i >= 1) {
                    println(dataA)
                    val actual_arr = Array.tabulate(16) { k => dut.io.resp.bits.result(k).peek().litValue.toShort}
                    val expected_arr: Array[Short] = expected.map(_.toShort).toArray
                    if(!checkVectorTolerance(actual_arr, expected_arr, false)) failed += 1
                    val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toShort & 0xFFFF}
                    actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
                        val actF = bfTofp(act)
                        val expF = bfTofp(exp)
                        println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
            }
            failed shouldBe 0
        }
       
    }
}
