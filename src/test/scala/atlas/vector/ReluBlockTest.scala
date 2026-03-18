//package atlas.vector

//import chisel3._
//import chisel3.util._
//import chisel3.simulator.EphemeralSimulator._
//import org.scalatest.flatspec.AnyFlatSpec
//import org.scalatest.matchers.should.Matchers
//import scala.util.Random
//import fpex._

//class ReluBlockTest extends AnyFlatSpec with Matchers {

  //// BF16 Helpers
  //def doubleToBF16Int(d: Double): Int = {
    //val bits = java.lang.Float.floatToIntBits(d.toFloat)
    //(bits >>> 16) & 0xFFFF
  //}

  //def bf16IntToDouble(i: Int): Double = {
    //java.lang.Float.intBitsToFloat(i << 16).toDouble
  //}

  //// Expected ReLU logic
  //def expectedRelu(d: Double): Double = if (d < 0.0) 0.0 else d

  //val fptype = FPType.BF16T 
  //val numLanes = 16
  //val tagWidth = 8

  //case class TestCase(tag: Int, mask: Int, inputs: Seq[Double])

  //behavior of "Relu"

  //it should "zero out negative values and preserve positive values across 16 lanes" in {
    //simulate(new Relu(fptype, numLanes, tagWidth)) { dut =>
      
      //val rng = new Random(334)
      
      //val manualTests = Seq(
        //// All positive
        //TestCase(0x10, 0xFFFF, Seq.fill(16)(2.5)),
        //// All negative
        //TestCase(0x11, 0xFFFF, Seq.fill(16)(-1.0)),
        //// Mixed
        //TestCase(0x12, 0xFFFF, Seq.tabulate(16)(i => if(i % 2 == 0) (i + 1).toDouble else -(i + 1).toDouble)),
        //// Small values and Zeros
        //TestCase(0x13, 0xFFFF, Seq(0.0, -0.0, 0.001, -0.001) ++ Seq.fill(12)(0.0))
      //)

      //val randomTests = (0 until 4).map { i =>
        //val randInputs = Seq.fill(16)((rng.nextDouble() * 40.0) - 20.0)
        //TestCase(0x20 + i, 0xFFFF, randInputs)
      //}

      //val allTests = manualTests ++ randomTests
      
      //// Initialize and Reset
      //dut.reset.poke(true.B)
      //dut.clock.step(3)
      //dut.reset.poke(false.B)
      //dut.clock.step(1)
      //dut.io.resp.ready.poke(true.B)

      //var reqIdx = 0
      //var resultsFound = 0
      //var cycles = 0
      //val timeout = allTests.length * 10 + 50

      //println("\n=========================================================================================")
      //println(f"STARTING RELU ACTIVATION TESTS (${allTests.length} Vectors)")
      //println("=========================================================================================")

      //while (resultsFound < allTests.length && cycles < timeout) {
        
        //// Drive Inputs
        //if (reqIdx < allTests.length) {
          //dut.io.req.valid.poke(true.B)
          //dut.io.req.bits.tag.poke(allTests(reqIdx).tag.U)
          //dut.io.req.bits.laneMask.poke(allTests(reqIdx).mask.U)
          
          //for (i <- 0 until numLanes) {
            //dut.io.req.bits.aVec(i).poke(doubleToBF16Int(allTests(reqIdx).inputs(i)).U)
          //}
        //} else {
          //dut.io.req.valid.poke(false.B)
        //}

        //// Monitor Outputs
        //if (dut.io.resp.valid.peek().litToBoolean && dut.io.resp.ready.peek().litToBoolean) {
          //val outTag = dut.io.resp.bits.tag.peek().litValue.toInt
          
          //// Match output to the correct test case via Tag
          //val tc = allTests.find(_.tag == outTag).get
          
          //println(f"[Tag 0x${tc.tag}%02X] Verifying ReLU Lanes:")
          
          //for (i <- 0 until numLanes) {
            //val actualInt = dut.io.resp.bits.result(i).peek().litValue.toInt
            //val actualVal = bf16IntToDouble(actualInt)
            
            //// Normalize expectation: if it's -0.0, make it +0.0
            //val rawExpectedVal = expectedRelu(tc.inputs(i))
            //val expectedVal = if (rawExpectedVal == 0.0) 0.0 else rawExpectedVal
            
            //val actualBits = actualInt & 0xFFFF
            //val expectedBits = doubleToBF16Int(expectedVal) & 0xFFFF

            //// Verification
            //if (actualBits != expectedBits) {
              //println(f"  FAILED Lane $i%2d:")
              //println(f"    Input:    ${tc.inputs(i)}%10.4f (Raw: 0x${doubleToBF16Int(tc.inputs(i))}%04X)")
              //println(f"    Expected: $expectedVal%10.4f (Bits: 0x$expectedBits%04X)")
              //println(f"    Actual:   $actualVal%10.4f (Bits: 0x$actualBits%04X)")
            //}

            //actualBits should be (expectedBits)
          //}
          
          //println(f"  -> All 16 lanes correct for Tag 0x${tc.tag}%02X")
          //resultsFound += 1
        //}

        //// Handshake logic: Advance reqIdx if request was accepted
        //if (reqIdx < allTests.length && dut.io.req.ready.peek().litToBoolean && dut.io.req.valid.peek().litToBoolean) {
          //reqIdx += 1 
        //}

        //dut.clock.step(1)
        //cycles += 1
      //}
      
      //if (cycles >= timeout) fail("Test timed out!")
      
    //}
  //}
//}