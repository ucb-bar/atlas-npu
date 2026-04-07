// ============================================================================
// ReduSumBlockTest.scala — Sum-reduction block tests for the vector datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.ReduSumBlockTest
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome 
import sp26FPUnits._
import scala.util.Random

class ReduSumBlockTest extends AnyFlatSpec with Matchers {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("ReduSumBlockTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("ReduSumBlockTest=PASSED")
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

  def expectedMaskedSum(inputs: Seq[Double], mask: Int): Double = {
    inputs.zipWithIndex.filter { case (_, i) => ((mask >> i) & 1) == 1 }.map(_._1).sum
  }

  val fptype = AtlasFPType.BF16
  val numLanes = 16
  val tagWidth = 8

  case class TestCase(tag: Int, mask: Int, inputs: Seq[Double])

  behavior of "ReduSumRec"

  it should "compute the masked sum and report relative error" in {
    simulate(new ReduSumRec(fptype, numLanes, tagWidth)) { dut =>
      
      val rng = new Random(42) // Fixed seed for reproducibility
      
      val manualTests = Seq(
        TestCase(0x10, 0xFFFF, Seq.fill(16)(2.0)),
        TestCase(0x11, 0x5555, Seq.tabulate(16)(i => (i + 1).toDouble)),
        TestCase(0x12, 0x0000, Seq.fill(16)(99.9)), // All disabled
        TestCase(0x13, 0x0001, Seq(42.0) ++ Seq.fill(15)(99.9)),
        TestCase(0x14, 0x8000, Seq.fill(15)(99.9) ++ Seq(12.5)),
        TestCase(0x15, 0xFFFF, Seq.fill(8)(5.0) ++ Seq.fill(8)(-5.0))
      )

      val randomTests = (0 until 4).map { i =>
        val randInputs = Seq.fill(16)((rng.nextDouble() * 20.0) - 10.0)
        val randMask = rng.nextInt(0x10000)
        TestCase(0x20 + i, randMask, randInputs)
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
      val timeout = allTests.length * 15 + 100

      println("\n=========================================================================================")
      println(f"STARTING REDUCTION SUM TESTS (${allTests.length} Vectors)")
      println("=========================================================================================")

      while (resultsFound < allTests.length && cycles < timeout) {
        
        // Drive Inputs
        if (reqIdx < allTests.length) {
          dut.io.req.valid.poke(true.B)
          dut.io.req.bits.tag.poke(allTests(reqIdx).tag.U)
          dut.io.req.bits.laneMask.poke(allTests(reqIdx).mask.U)
          dut.io.req.bits.roundingMode.poke(0.U)
          
          for (i <- 0 until numLanes) {
            dut.io.req.bits.aVec(i).poke(doubleToBF16Int(allTests(reqIdx).inputs(i)).U)
          }
        } else {
          dut.io.req.valid.poke(false.B)
        }

        // Monitor Outputs
        if (dut.io.resp.valid.peek().litToBoolean && dut.io.resp.ready.peek().litToBoolean) {
          val outTag = dut.io.resp.bits.tag.peek().litValue.toInt
          val resultBits = dut.io.resp.bits.result(0).peek().litValue.toInt
          val actualSum = bf16IntToDouble(resultBits)
          
          // Match output to the correct test case
          val tc = allTests.find(_.tag == outTag).get
          val expectedSum = expectedMaskedSum(tc.inputs, tc.mask)
          
          // Calculate Error
          val absErr = Math.abs(actualSum - expectedSum)
          val relErrPct = if (expectedSum != 0.0) (absErr / Math.abs(expectedSum)) * 100.0 else absErr * 100.0
          
          // Format the input array string for clean printing
          val inputStr = tc.inputs.take(4).map(v => f"$v%.1f").mkString("[", ", ", ", ...]")

          println(f"[Tag 0x${tc.tag}%02X | Mask 0x${tc.mask}%04X] Vector: $inputStr")
          println(f"    -> Expected: $expectedSum%8.4f  |  Actual: $actualSum%8.4f  |  Rel Error: $relErrPct%5.2f%%")

          // Verification (BF16 tree addition has a higher tolerance due to associative rounding differences)
          actualSum should be (expectedSum +- 0.5) 

          resultsFound += 1
        }

        // Handshake
        if (reqIdx < allTests.length && dut.io.req.ready.peek().litToBoolean && dut.io.req.valid.peek().litToBoolean) {
          reqIdx += 1 
        }

        dut.clock.step(1)
        cycles += 1
      }
      println("=========================================================================================\n")
    }
  }
}
