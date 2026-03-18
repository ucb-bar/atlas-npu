//package atlas.vector

//import chisel3._
//import chisel3.simulator.EphemeralSimulator._
//import org.scalatest.flatspec.AnyFlatSpec
//import org.scalatest.matchers.should.Matchers
//import atlas.common.VPUParams
//import sp26FPUnits.hardfloat._
//import svsim.CommonCompilationSettings.Timescale.Unit.s

//class SinCosLUTTest extends AnyFlatSpec with Matchers {
    //val ports = 1
    //val addrBits = 8
    //val m = 4
    //val n = 12
    //val entries = 1 << addrBits

    //// Golden Model: Replicates the EXACT math in your RTL
    //def goldenSinCosLUT1(i: Int, isCos: Boolean): BigInt = {
        //val max = math.Pi / 2
        //val min = 0.0
        //val r = min + i.toDouble * (max - min) / entries
        //val v = if (isCos) math.sin(max - r) else math.sin(r)
        //BigInt(math.round(v * (1 << n)))
    //}

    //// behavior of "SinCosLUT Test 1 (returning single sine value)"
    //// it should "Verify correctness of single sine value output" in {
    ////     simulate(new SinCosLUT(ports, addrBits, m, n)) { dut =>
    ////         var cycle = 0
    ////         // Sweep every address
    ////         for (i <- 0 until entries) {
    ////             dut.io.isCos(0).poke(false.B) // Set to sine mode
    ////             dut.io.ren(0).poke(true.B)
    ////             dut.io.raddr(0).poke(i.U)
                
    ////             dut.clock.step(1) // Wait for register update
    ////             cycle += 1
    ////             println(s"Forward 1 cycle")
                
    ////             val expectedY0 = goldenSinCosLUT1(i, false)
    ////             val nextIdx = if (i == entries - 1) i else i + 1
    ////             val expectedY1 = goldenSinCosLUT1(nextIdx, false)
    ////             println(s"Cycle: $cycle")
    ////             println(s"Expected Y0: $expectedY0, Expected Y1: $expectedY1")
    ////             println(s"Actual Y0: ${dut.io.rdata(0)(0).peek().litValue}, Actual Y1: ${dut.io.rdata(1)(0).peek().litValue}")
    ////             dut.io.rdata(0)(0).expect(expectedY0.U)
    ////             dut.io.rdata(1)(0).expect(expectedY1.U)
    ////         } 
    ////     }
    //// }

    //behavior of "SinCosLUT Test 2 (returning single cosine value)"
    //it should "Verify correctness of single cosine value output" in {
        //simulate(new SinCosLUT(ports, addrBits, m, n)) { dut =>
            //var cycle = 0
            //// Sweep every address
            //for (i <- 0 until entries) {
                //dut.io.isCos(0).poke(true.B) // Set to cosine mode
                //dut.io.ren(0).poke(true.B)
                //dut.io.raddr(0).poke(i.U)
                
                //dut.clock.step(1) // Wait for register update
                //cycle += 1
                
                //val nextIdx = if (i == entries - 1) i else i + 1
                //val expectedY0 = goldenSinCosLUT1(i, true)
                //val expectedY1 = goldenSinCosLUT1(nextIdx, true)
                //println(s"Cycle: $cycle")
                //println(s"Expected Y0: $expectedY0, Expected Y1: $expectedY1")
                //println(s"Actual Y0: ${dut.io.rdata(0)(0).peek().litValue}, Actual Y1: ${dut.io.rdata(1)(0).peek().litValue}")
                //println(s"Address: ${dut.io.raddr.peek().litValue}")
                //// println(s"Indecx: $i, Next Index: $nextIdx")
                //dut.io.rdata(0)(0).expect(expectedY0.U)
                //dut.io.rdata(1)(0).expect(expectedY1.U)
            //} 
        //}
    //}
//}