// ============================================================================
// RVTanhTest.scala — BF16 tanh vector tests for the RV-style vector datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.RVTanhTest
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

object PersistentVcsRVTanhSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "RVTanhTest"

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

class RVTanhTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

    //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("RVTanhTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("RVTanhTest=PASSED")
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

    def goldenTanh16(aVec: Seq[Float], bVec: Seq[Float]): Seq[Int] = {
        require(aVec.length == 16 && bVec.length == 16)

        aVec.zip(bVec).map { case (a, b) =>
            val aB = bfTofp(fpTobf(a))  
            val bB = bfTofp(fpTobf(b))  
            val y  = Math.tanh(aB)
            val yFloat = y.toFloat
            fpTobf(yFloat)                   // BF16 result bits as Int
        }
    }

    behavior of s"VectorEngine (BF16) TanhRec lanes=${16}"
    it should "Verify the correctness of add" in {
        PersistentVcsRVTanhSimulator.simulate(new TanhRec(BF16)) { module =>
            val dut = module.wrapped

            dut.reset.poke(true.B)
            dut.clock.step(3)
            dut.reset.poke(false.B)
            dut.clock.step(1)

            // Test data
            val dataA = Seq.fill(16)(scala.util.Random.nextFloat() * 2)
            val dataB = Seq.fill(16)(scala.util.Random.nextFloat() * 1000)

            // Expected values from golden model
            val expected: Seq[Int] = goldenTanh16(dataA, dataB)

            // Convert inputs to BF16 bits using the SAME converter as golden
            val dataA_bf16: Seq[Int] = dataA.map(fpTobf)
            val dataB_bf16: Seq[Int] = dataB.map(fpTobf)

            // Poke inputs
            dut.io.req.valid.poke(true.B)
            dut.io.req.bits.tag.poke(1.U) 
            dut.io.req.bits.whichBank.poke(2.U)
            dut.io.req.bits.wRow.poke(3.U)
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.xVec(i).poke(BigInt(bits & 0xFFFF)) }
            var failed = 0
            for (i <- 0 to 5) {
                val iv = dut.io.req.valid.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1
                val respFire = ov == 1
                val tag = dut.io.resp.bits.tag.peek().litValue.toInt
                val whichBank = dut.io.resp.bits.whichBank.peek().litValue.toInt
                val wRow = dut.io.resp.bits.wRow.peek().litValue.toInt
                println(f"[cycle=$i%2d] req: v=$iv%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d f=${if(respFire)1 else 0}%d")
                println(f"tag=$tag whichBank=$whichBank wRow=$wRow")
                // Print values at after two cycles
                if (respFire && i >= 1) {
                    println(dataA)
                    val actual_arr = Array.tabulate(16) { k => dut.io.resp.bits.result(k).peek().litValue.toShort}
                    val expected_arr: Array[Short] = expected.map(_.toShort).toArray
                    if(!checkVectorTolerance(actual_arr, expected_arr, false)) failed += 1
                    val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}
                    actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
                        val actF = bfTofp(act)
                        val expF = bfTofp(exp)
                        println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
                dut.io.req.bits.tag.poke(0.U) 
                dut.io.req.bits.whichBank.poke(0.U)
                dut.io.req.bits.wRow.poke(0.U)
            }
            failed shouldBe 0
        }
    }
}
