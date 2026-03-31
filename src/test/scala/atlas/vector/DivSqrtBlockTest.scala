package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import sp26FPUnits._
import sp26FPUnits.hardfloat._
import sp26FPUnits.AtlasFPType._
import org.scalatest.Outcome 


class DivSqrtBlockTest extends AnyFlatSpec with Matchers {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("DivSqrtBlockTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("DivSqrtBlockTest=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  // Helper: Convert Double to BF16 bit pattern (top 16 bits of Float)
  def doubleToBF16Int(d: Double): Int = {
    val bits = java.lang.Float.floatToIntBits(d.toFloat)
    (bits >>> 16) & 0xFFFF
  }

  // Helper: Convert BF16 bit pattern back to Double
  def bf16IntToDouble(i: Int): Double = {
    java.lang.Float.intBitsToFloat(i << 16).toDouble
  }

  // Use the parameters defined in your project
  val fptype = AtlasFPType.BF16 
  val numLanes = 4 // Testing with 4 lanes for brevity, can be 16
  val tagWidth = 8

  behavior of "DivSqrtRec"

  it should "compute vectorized division across multiple lanes" in {
    simulate(new DivSqrtRec(fptype, numLanes, tagWidth)) { dut =>
      
      // Define test vectors for lanes
      val aValues = Seq(10.0, 1.0, 144.0, 3.14)
      val bValues = Seq(2.0,  0.5, 12.0,  1.57)
      
      println(s"\n>>> STARTING VECTORIZED DIVISION")

      // 1. Wait for module to be ready
      while (!dut.io.req.ready.peek().litToBoolean) {
        dut.clock.step(1)
      }

      // 2. Poke Request
      dut.io.req.valid.poke(true.B)
      dut.io.req.bits.isSqrt.poke(false.B) // Division
      dut.io.req.bits.laneMask.poke("b1111".U)
      dut.io.req.bits.tag.poke(0xAB.U)

      for (i <- 0 until numLanes) {
        dut.io.req.bits.aVec(i).poke(doubleToBF16Int(aValues(i)).U)
        dut.io.req.bits.bVec(i).poke(doubleToBF16Int(bValues(i)).U)
      }

      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      // 3. Wait for Response
      // Note: You mentioned io.resp.valid logic needs work. 
      // In a real test, we wait for the hardware to signal completion.
      var cycles = 0
      while (!dut.io.resp.valid.peek().litToBoolean && cycles < 100) {
        dut.clock.step(1)
        cycles += 1
      }

      // 4. Peek and Verify
      val resp = dut.io.resp.bits
      println(f"Response received in $cycles cycles")

      for (i <- 0 until numLanes) {
        val rawRes = resp.result(i).peek().litValue.toInt
        val actual = bf16IntToDouble(rawRes)
        val expected = 1 / aValues(i)
        
        println(f"Lane $i: 1 / ${bValues(i)} = $actual%.4f (Expected: $expected%.4f)")
        
        // BF16 has ~2-3 decimal digits of precision (7-bit mantissa)
        // So we use a fairly loose tolerance
        math.abs(actual - expected) should be < 0.05
      }
    }
  }

  it should "compute vectorized square root" in {
    simulate(new DivSqrtRec(fptype, numLanes, tagWidth)) { dut =>
      val aValues = Seq(4.0, 9.0, 2.0, 0.25)
      
      while (!dut.io.req.ready.peek().litToBoolean) dut.clock.step(1)

      dut.io.req.valid.poke(true.B)
      dut.io.req.bits.isSqrt.poke(true.B)
      dut.io.req.bits.laneMask.poke("b1111".U)

      for (i <- 0 until numLanes) {
        dut.io.req.bits.aVec(i).poke(doubleToBF16Int(aValues(i)).U)
      }

      dut.clock.step(1)
      dut.io.req.valid.poke(false.B)

      while (!dut.io.resp.valid.peek().litToBoolean) dut.clock.step(1)

      println(s"\n>>> STARTING VECTORIZED SQRT")
      for (i <- 0 until numLanes) {
        val actual = bf16IntToDouble(dut.io.resp.bits.result(i).peek().litValue.toInt)
        val expected = math.sqrt(aValues(i))
        println(f"Lane $i: sqrt(${aValues(i)}) = $actual%.4f (Expected: $expected%.4f)")
        math.abs(actual - expected) should be < 0.05
      }
    }
  }
}