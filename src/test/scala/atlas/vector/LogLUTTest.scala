//package atlas.vector

//import chisel3._
//import chisel3.simulator.EphemeralSimulator._
//import org.scalatest.flatspec.AnyFlatSpec
//import org.scalatest.matchers.should.Matchers
//import org.scalatest.Outcome
//import fpex._

//class Log2Spec extends AnyFlatSpec with Matchers {

  //def doubleToBF16Int(d: Double): Int = {
    //val bits = java.lang.Float.floatToIntBits(d.toFloat)
    //(bits >>> 16) & 0xFFFF
  //}

  //def bf16IntToDouble(i: Int): Double = {
    //java.lang.Float.intBitsToFloat(i << 16).toDouble
  //}

  //val fptype = FPType.BF16T 

  //behavior of "Log2"

  //it should "compute log2 and print internal state comparison" in {
    //simulate(new Log2(fptype)) { dut =>
      //// Start with a simple value: 2.0 (log2(2.0) should be exactly 1.0)
      //val testValues = Seq(1.0, 2.0, 4.0, 1.5)

      //testValues.foreach { v =>
        //val bitsIn = doubleToBF16Int(v)
        
        //println(s"\n>>> STIMULATING INPUT: $v (Bits: 0x${bitsIn.toHexString})")
        
        //dut.io.x.poke(bitsIn.U)
        //dut.io.valid.poke(true.B)
        
        //// Step 1: Trigger the logic
        //dut.clock.step(1)
        
        //// Capture outputs
        //val bitsOut = dut.io.result.peek().litValue.toInt
        //val valOut = bf16IntToDouble(bitsOut)
        //val expected = math.log(v) / math.log(2.0)

        //println(f"TESTBENCH RESULT: Val=$valOut%.4f (Bits=0x${bitsOut.toHexString})")
        //println(f"EXPECTED RESULT:  Val=$expected%.4f (Bits=0x${doubleToBF16Int(expected).toHexString})")
        
        //// Verification
        //if (math.abs(valOut - expected) > 0.1) {
          //println(s"!!! ERROR DETECTED for input $v")
        //}
      //}
    //}
  //}
//}