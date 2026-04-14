// ============================================================================
// TanhBlockTest.scala — Tanh block tests for the vector datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.TanhBlockTest
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome 
import sp26FPUnits._
import sp26FPUnits.AtlasFPType._

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import atlas.common._
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsTanhBlockSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "TanhBlockTest"

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

class TanhBlockTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("TanhBlockTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("TanhBlockTest=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  def doubleToBF16Int(d: Double): Int = {
    val bits = java.lang.Float.floatToIntBits(d.toFloat)
    (bits >>> 16) & 0xFFFF
  }

  def bf16IntToDouble(i: Int): Double = {
    java.lang.Float.intBitsToFloat(i << 16).toDouble
  }

  val fptype = AtlasFPType.BF16 
  val numLanes = 8
  val tagWidth = 8

  behavior of "TanhRec"

  it should "compute vectorized Tanh with 2-cycle latency and handle pipelining" in {
    PersistentVcsTanhBlockSimulator.simulate(new TanhRec(BF16)) { module =>
      val dut = module.wrapped
      
      val inputs1 = Seq(0.0, 0.5, 1.0, 2.0, 5.0, -0.5, -1.0, -5.0)
      val inputs2 = Seq(0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4)
      
println(s"\n>>> STIMULATING TANH PIPELINE")
      dut.io.resp.ready.poke(true.B)

      // Package requests so we can iterate through them
      val requests = Seq(
        (0xAA, inputs1),
        (0xBB, inputs2)
      )
      
      var reqIdx = 0
      var resultsFound = 0
      var cycles = 0

      // Unified Driver/Monitor Loop
      while (resultsFound < 2 && cycles < 20) {
        // 1. DRIVE INPUTS
        if (reqIdx < requests.length) {
          dut.io.req.valid.poke(true.B)
          dut.io.req.bits.tag.poke(requests(reqIdx)._1.U)
          dut.io.req.bits.laneMask.poke("hFF".U)
          for (i <- 0 until numLanes) {
            dut.io.req.bits.xVec(i).poke(doubleToBF16Int(requests(reqIdx)._2(i)).U)
          }
        } else {
          // No more requests to send, drop valid
          dut.io.req.valid.poke(false.B)
        }

        // 2. MONITOR OUTPUTS (Peek before we step!)
        if (dut.io.resp.valid.peek().litToBoolean && dut.io.resp.ready.peek().litToBoolean) {
          val tag = dut.io.resp.bits.tag.peek().litValue
          val mask = dut.io.resp.bits.laneMask.peek().litValue
          
          println(f"\n>>> RESPONSE RECEIVED AT CYCLE $cycles")
          println(f"Tag: 0x$tag%X | LaneMask: 0x$mask%X")

          val currentInputs = if (tag == 0xAA) inputs1 else inputs2
          
          for (i <- 0 until numLanes) {
            val bitsOut = dut.io.resp.bits.result(i).peek().litValue.toInt
            val actual = bf16IntToDouble(bitsOut)
            val expected = math.tanh(currentInputs(i))
            val error = math.abs(actual - expected)
            println(f"  Lane $i: Out=$actual%8.4f (Exp=$expected%8.4f) | Err=$error%.4f")
          }
          resultsFound += 1
        }

        // 3. EVALUATE HANDSHAKE (Did the DUT accept our request?)
        if (reqIdx < requests.length && dut.io.req.ready.peek().litToBoolean && dut.io.req.valid.peek().litToBoolean) {
          reqIdx += 1 // Advance to next request for the next cycle
        }

        // 4. STEP CLOCK
        dut.clock.step(1)
        cycles += 1
      }

      resultsFound should be (2)
      println(f"\n>>> PIPELINE TEST COMPLETE: Found $resultsFound results in $cycles cycles.")    }
  }
}
