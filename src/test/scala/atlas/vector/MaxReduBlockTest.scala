package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome 
import sp26FPUnits._

class MaxReduBlockTest extends AnyFlatSpec with Matchers {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("MaxReduBlockTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("MaxReduBlockTest=PASSED")
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

  // Configuration - Adjusting to match your MaxRedu definition
  val fptype = AtlasFPType.BF16
  val numLanes = 16
  val tagWidth = 8

  behavior of "MaxRedu"

  it should "find the maximum BF16 value in a 16-lane vector with 1-cycle latency" in {
    simulate(new MaxRedu(fptype, numLanes, tagWidth)) { dut =>
      
      // Test Case 1: Mixed positive and negative
      val inputs1 = Seq(1.2, -5.0, 3.14, 0.0, 10.5, -20.0, 8.8, 0.1, 
                        -1.0, 2.2, 4.4, 6.6, -100.0, 0.5, 9.9, -0.2)
      
      // Test Case 2: All negative (Checking that it doesn't just return 0)
      val inputs2 = Seq(-1.5, -2.2, -0.5, -10.0, -8.8, -3.3, -4.4, -0.1,
                        -20.0, -15.0, -6.0, -7.0, -9.0, -12.0, -1.1, -0.05)
      
      println(s"\n>>> STIMULATING MAX REDUCTION PIPELINE")
      dut.io.resp.ready.poke(true.B)

      val requests = Seq(
        (0x11, inputs1),
        (0x22, inputs2)
      )
      
      var reqIdx = 0
      var resultsFound = 0
      var cycles = 0

      // Unified Driver/Monitor Loop
      while (resultsFound < requests.length && cycles < 30) {
        
        // 1. DRIVE INPUTS
        if (reqIdx < requests.length) {
          dut.io.req.valid.poke(true.B)
          dut.io.req.bits.tag.poke(requests(reqIdx)._1.U)
          for (i <- 0 until numLanes) {
            dut.io.req.bits.aVec(i).poke(doubleToBF16Int(requests(reqIdx)._2(i)).U)
          }
        } else {
          dut.io.req.valid.poke(false.B)
        }

        // 2. MONITOR OUTPUTS
        if (dut.io.resp.valid.peek().litToBoolean && dut.io.resp.ready.peek().litToBoolean) {
          val tag = dut.io.resp.bits.tag.peek().litValue
          val resultBits = dut.io.resp.bits.result.peek().litValue.toInt
          val actual = bf16IntToDouble(resultBits)
          
          val currentInputs = if (tag == 0x11) inputs1 else inputs2
          val expected = currentInputs.max
          
          println(f"\n>>> RESPONSE RECEIVED AT CYCLE $cycles")
          println(f"Tag: 0x$tag%X | Max Found: $actual%.4f (Expected: $expected%.4f)")

          // Verification
          actual should be (expected +- 0.01) // Using tolerance for float precision
          resultsFound += 1
        }

        // 3. EVALUATE HANDSHAKE
        if (reqIdx < requests.length && dut.io.req.ready.peek().litToBoolean && dut.io.req.valid.peek().litToBoolean) {
          reqIdx += 1 
        }

        // 4. STEP CLOCK
        dut.clock.step(1)
        cycles += 1
      }

      resultsFound should be (requests.length)
      println(f"\n>>> TEST COMPLETE: Found $resultsFound results in $cycles cycles.")
    }
  }
}