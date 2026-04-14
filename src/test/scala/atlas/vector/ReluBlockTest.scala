// ============================================================================
// ReluBlockTest.scala — ReLU block tests for the vector datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.ReluBlockTest
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome 
import scala.util.Random
import sp26FPUnits._

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import atlas.common._
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsReluBlockTestSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "ReluBlockTest"

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

class ReluBlockTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("ReluBlockTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("ReluBlockTest=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  // BF16 Helpers
  def doubleToBF16Int(d: Double): Int = {
    val bits = java.lang.Float.floatToIntBits(d.toFloat)
    (bits >>> 16) & 0xFFFF
  }

  def bf16IntToDouble(i: Int): Double = {
    java.lang.Float.intBitsToFloat(i << 16).toDouble
  }

  // Expected ReLU logic
  def expectedRelu(d: Double): Double = if (d < 0.0) 0.0 else d

  val fptype = AtlasFPType.BF16
  val numLanes = 16
  val tagWidth = 8

  case class TestCase(tag: Int, mask: Int, inputs: Seq[Double])

  behavior of "Relu"

  it should "zero out negative values and preserve positive values across 16 lanes" in {
    PersistentVcsReluBlockTestSimulator.simulate(new Relu(fptype, numLanes, tagWidth)) { module =>
      val dut = module.wrapped
      
      val rng = new Random(334)
      
      val manualTests = Seq(
        // All positive
        TestCase(0x10, 0xFFFF, Seq.fill(16)(2.5)),
        // All negative
        TestCase(0x11, 0xFFFF, Seq.fill(16)(-1.0)),
        // Mixed
        TestCase(0x12, 0xFFFF, Seq.tabulate(16)(i => if(i % 2 == 0) (i + 1).toDouble else -(i + 1).toDouble)),
        // Small values and Zeros
        TestCase(0x13, 0xFFFF, Seq(0.0, -0.0, 0.001, -0.001) ++ Seq.fill(12)(0.0))
      )

      val randomTests = (0 until 4).map { i =>
        val randInputs = Seq.fill(16)((rng.nextDouble() * 40.0) - 20.0)
        TestCase(0x20 + i, 0xFFFF, randInputs)
      }

      val allTests = manualTests ++ randomTests
      
      // Initialize and Reset
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)
      dut.io.resp.ready.poke(true.B)

      var reqIdx = 0
      var resultsFound = 0
      var cycles = 0
      val timeout = allTests.length * 10 + 50

      println("\n=========================================================================================")
      println(f"STARTING RELU ACTIVATION TESTS (${allTests.length} Vectors)")
      println("=========================================================================================")

      while (resultsFound < allTests.length && cycles < timeout) {
        
        // Drive Inputs
        if (reqIdx < allTests.length) {
          dut.io.req.valid.poke(true.B)
          dut.io.req.bits.tag.poke(allTests(reqIdx).tag.U)
          dut.io.req.bits.laneMask.poke(allTests(reqIdx).mask.U)
          
          for (i <- 0 until numLanes) {
            dut.io.req.bits.aVec(i).poke(doubleToBF16Int(allTests(reqIdx).inputs(i)).U)
          }
        } else {
          dut.io.req.valid.poke(false.B)
        }

        // Monitor Outputs
        if (dut.io.resp.valid.peek().litToBoolean && dut.io.resp.ready.peek().litToBoolean) {
          val outTag = dut.io.resp.bits.tag.peek().litValue.toInt
          
          // Match output to the correct test case via Tag
          val tc = allTests.find(_.tag == outTag).get
          
          println(f"[Tag 0x${tc.tag}%02X] Verifying ReLU Lanes:")
          
          for (i <- 0 until numLanes) {
            val actualInt = dut.io.resp.bits.result(i).peek().litValue.toInt
            val actualVal = bf16IntToDouble(actualInt)
            
            // Normalize expectation: if it's -0.0, make it +0.0
            val rawExpectedVal = expectedRelu(tc.inputs(i))
            val expectedVal = if (rawExpectedVal == 0.0) 0.0 else rawExpectedVal
            
            val actualBits = actualInt & 0xFFFF
            val expectedBits = doubleToBF16Int(expectedVal) & 0xFFFF

            // Verification
            if (actualBits != expectedBits) {
              println(f"  FAILED Lane $i%2d:")
              println(f"    Input:    ${tc.inputs(i)}%10.4f (Raw: 0x${doubleToBF16Int(tc.inputs(i))}%04X)")
              println(f"    Expected: $expectedVal%10.4f (Bits: 0x$expectedBits%04X)")
              println(f"    Actual:   $actualVal%10.4f (Bits: 0x$actualBits%04X)")
            }

            actualBits should be (expectedBits)
          }
          
          println(f"  -> All 16 lanes correct for Tag 0x${tc.tag}%02X")
          resultsFound += 1
        }

        // Handshake logic: Advance reqIdx if request was accepted
        if (reqIdx < allTests.length && dut.io.req.ready.peek().litToBoolean && dut.io.req.valid.peek().litToBoolean) {
          reqIdx += 1 
        }

        dut.clock.step(1)
        cycles += 1
      }
      
      if (cycles >= timeout) fail("Test timed out!")
      
    }
  }
}
