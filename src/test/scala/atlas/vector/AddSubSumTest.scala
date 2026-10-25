// ============================================================================
// AddSubSumTest.scala — Functional tests for the flexible vector unit
// including Add, Sub, Sum Reduction.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.AddSubSumTest
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import sp26FPUnits._

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import atlas.common._
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsAddSubSumSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "AddSubSumTest"

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

class AddSubSumTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("AddSubSumTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("AddSubSumTest=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  // BF16 Helper: Double to 16-bit Int (Truncate bits)
  def doubleToBF16Int(d: Double): Int = {
    val bits = java.lang.Float.floatToIntBits(d.toFloat)
    (bits >>> 16) & 0xFFFF
  }

  // BF16 Helper: 16-bit Int to Double
  def bf16IntToDouble(i: Int): Double = {
    java.lang.Float.intBitsToFloat(i << 16).toDouble
  }

  // Configuration
  val fptype = AtlasFPType.BF16
  val numLanes = 16
  val tagWidth = 16 // Matches the default in your module definition

  behavior of "AddSubSumVec"

  it should "correctly compute pointwise Add/Sub, and Sum Reduction" in {
    PersistentVcsAddSubSumSimulator.simulate(new AddSubSumVec(fptype, numLanes, tagWidth)) { module =>
      val dut = module.wrapped

      // Generate Input Vectors
      // aVec: 1.0, 2.0, 3.0 ... 16.0
      // bVec: 1.5, 3.0, 4.5 ... 24.0 times 2^-8: the exact a + b and a - b fall between
      // BF16 values, and some on ties, so truncation and nearest even differ.
      val aVals = Seq.tabulate(numLanes)(i => (i + 1).toDouble)
      val bVals = Seq.tabulate(numLanes)(i => (i + 1) * 1.5 * math.pow(2, -8))

      // Calculate Expected Outputs: add and sub round the exact result to nearest even.
      def rne(d: Double): Double = bf16IntToDouble(VectorTestUtils.fpToBf16(d.toFloat) & 0xFFFF)
      val expectedAdd = aVals.zip(bVals).map { case (a, b) => rne(a + b) }
      val expectedSub = aVals.zip(bVals).map { case (a, b) => rne(a - b) }
      val expectedSum = Seq.fill(numLanes)(aVals.sum)

      // Test sequence: (Operation Name, isSub, isSum, Expected Results)
      // Note: We issue them in an order that won't cause fast-path/slow-path collisions
      val requests = Seq(
        ("ADD", false, false, false, expectedAdd),
        ("SUB", true,  false, false, expectedSub),
        ("SUM", false, true,  false, expectedSum)
      )

      // --- PIPELINE FLUSH ---
      dut.io.req.valid.poke(false.B)
      dut.clock.step(5) // Step 5 cycles to clear out any reset garbage
      
      println(s"\n>>> STIMULATING ADD/SUB/SUM PIPELINE")
      var reqIdx = 0
      var resultsFound = 0
      var cycles = 0

      // Unified Driver/Monitor Loop
      while (resultsFound < requests.length && cycles < 50) {
        
        // 1. DRIVE INPUTS
        if (reqIdx < requests.length) {
          val (_, isSub, isSum, _, _) = requests(reqIdx)
          
          dut.io.req.valid.poke(true.B)
          dut.io.req.bits.isSub.poke(isSub.B)
          dut.io.req.bits.isSum.poke(isSum.B)
          
          for (i <- 0 until numLanes) {
            dut.io.req.bits.aVec(i).poke(doubleToBF16Int(aVals(i)).U)
            dut.io.req.bits.bVec(i).poke(doubleToBF16Int(bVals(i)).U)
          }
        } else {
          dut.io.req.valid.poke(false.B)
        }

        // 2. MONITOR OUTPUTS
        if (dut.io.resp.valid.peek().litToBoolean) {
          val (reqName, _, isSumOp, _, expected) = requests(resultsFound)
          
          println(f"\n>>> RESPONSE RECEIVED FOR '$reqName' AT CYCLE $cycles")

          // Verification Loop
          for (i <- 0 until numLanes) {
            val resultBits = dut.io.resp.bits.result(i).peek().litValue.toInt
            val actual = bf16IntToDouble(resultBits)

            println(f"  Lane $i%2d | Actual: $actual%7.4f (Expected: ${expected(i)}%7.4f)")
            actual should be (expected(i))
          }
          resultsFound += 1
        }

        // 3. Advance once the request has been presented on the valid-only interface
        if (reqIdx < requests.length && dut.io.req.valid.peek().litToBoolean) {
          reqIdx += 1 
        }

        // 4. STEP CLOCK
        dut.clock.step(1)
        cycles += 1
      }

      resultsFound should be (requests.length)
      println(f"\n>>> TEST COMPLETE: Processed $resultsFound operations in $cycles cycles.")
    }
  }
}
