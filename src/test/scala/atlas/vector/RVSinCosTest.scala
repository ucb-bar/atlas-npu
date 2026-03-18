//package atlas.vector

//import chisel3._
//import chisel3.simulator.EphemeralSimulator._
//import org.scalatest.flatspec.AnyFlatSpec
//import org.scalatest.matchers.should.Matchers
//import atlas.common.VPUParams
//import svsim.CommonCompilationSettings.Timescale.Unit.s
//import sp26FPUnits.hardfloat._
//import fpex.FPType.BF16T


//class RVSinCosTest extends AnyFlatSpec with Matchers {
    //val p = VPUParams()
    
    //def fpTobf(f: Float): Int = {
        //val x = java.lang.Float.floatToRawIntBits(f)
        //val lsb = (x >>> 16) & 1
        //val roundingBias = 0x7FFF + lsb
        //((x + roundingBias) >>> 16) & 0xFFFF
    //}

    //def bfTofp(b: Int): Float =
        //java.lang.Float.intBitsToFloat((b & 0xFFFF) << 16)

    //def goldenSinCos16(aVec: Seq[Float], isCos: Boolean): Seq[Int] = {
        //require(aVec.length == 16)

        //aVec.map { a  =>
            //val aB = bfTofp(fpTobf(a))   
            //// Testing both exp and exp2
            //val y  = if (isCos) Math.cos(aB).toFloat else Math.sin(aB).toFloat
            //fpTobf(y)                   // BF16 result bits as Int
        //}
    //}

    //behavior of s"VectorEngine (BF16) SinCos lanes=${16}"
    //it should "Verify the correctness of add" in {
        //simulate(new SinCos(BF16T)) { dut =>
            //// Test data
            //val twoPi = (2 * Math.PI).toFloat
            //val threePi_over_2 = (1.5 * Math.PI).toFloat
            //val pi = (Math.PI).toFloat
            //val pi_over_2 = (0.5 * Math.PI).toFloat
            //val pi_over_4 = (0.25 * Math.PI).toFloat
            //val pi_over_6 = (Math.PI/ 6.0).toFloat
            //val pi_over_12 = (Math.PI/ 12.0).toFloat
            //val dataA = Seq.fill(16)((scala.util.Random.nextFloat() * twoPi).toFloat)
            //// val dataA = Seq.fill(16)(scala.util.Random.nextInt(2).toFloat)

            //// Expected values from golden model
            //val expectedSin: Seq[Int] = goldenSinCos16(dataA, false)
            //val expectedCos: Seq[Int] = goldenSinCos16(dataA, true)

            //// Convert inputs to BF16 bits using the SAME converter as golden
            //val dataA_bf16: Seq[Int] = dataA.map(fpTobf)

            //dut.reset.poke(true.B)
            //dut.clock.step(1)
            //dut.reset.poke(false.B)
            //dut.clock.step(1)

            //// Poke inputs for Sin
            //dut.io.req.valid.poke(true.B)
            //dut.io.resp.ready.poke(true.B)
            //dut.io.req.bits.roundingMode.poke(0.U) 
            //dut.io.req.bits.tag.poke(1.U) 
            //dut.io.req.bits.whichBank.poke(2.U)
            //dut.io.req.bits.wRow.poke(3.U)
            //dut.io.req.bits.cos.poke(false.B)
            //dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            //dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.xVec(i).poke(BigInt(bits & 0xFFFF)) }
            //for (i <- 0 to 10) {
                //println("Sin Test Begin")
                //val ir = dut.io.req.ready.peek().litValue
                //val iv = dut.io.req.valid.peek().litValue
                //val or = dut.io.resp.ready.peek().litValue
                //val ov = dut.io.resp.valid.peek().litValue
                //val reqFire  = iv == 1 && ir == 1
                //val respFire = ov == 1 && or == 1
                //val tag = dut.io.resp.bits.tag.peek().litValue.toInt
                //val whichBank = dut.io.resp.bits.whichBank.peek().litValue.toInt
                //val wRow = dut.io.resp.bits.wRow.peek().litValue.toInt
                //println(f"[cycle=$i%2d] req: v=$iv%d r=$ir%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d r=$or%d f=${if(respFire)1 else 0}%d")
                //println(f"tag=$tag whichBank=$whichBank wRow=$wRow")

                //// Print values at after two cycles
                //if (respFire && i >= 5) {
                    //println(dataA)
                    //val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}
                    //actual.zip(expectedSin).zipWithIndex.foreach { case ((act, exp), i) =>
                        //val actF = bfTofp(act)
                        //val expF = bfTofp(exp)
                        //println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    //}
                //}
                //// Step to next cycle and turn off input validity
                //dut.clock.step(1)
                //dut.io.req.valid.poke(false.B)
                //dut.io.req.bits.tag.poke(0.U) 
                //dut.io.req.bits.whichBank.poke(0.U)
                //dut.io.req.bits.wRow.poke(0.U)
            //}

            //// Poke inputs for Cos
            //dut.io.req.valid.poke(true.B)
            //dut.io.resp.ready.poke(true.B)
            //dut.io.req.bits.roundingMode.poke(0.U) 
            //dut.io.req.bits.tag.poke(4.U) 
            //dut.io.req.bits.whichBank.poke(5.U)
            //dut.io.req.bits.wRow.poke(6.U)
            //dut.io.req.bits.cos.poke(true.B)
            //dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            //dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.xVec(i).poke(BigInt(bits & 0xFFFF)) }
            //for (i <- 0 to 10) {
                //println("Cos Test Begin")
                //val ir = dut.io.req.ready.peek().litValue
                //val iv = dut.io.req.valid.peek().litValue
                //val or = dut.io.resp.ready.peek().litValue
                //val ov = dut.io.resp.valid.peek().litValue
                //val reqFire  = iv == 1 && ir == 1
                //val respFire = ov == 1 && or == 1
                //val tag = dut.io.resp.bits.tag.peek().litValue.toInt
                //val whichBank = dut.io.resp.bits.whichBank.peek().litValue.toInt
                //val wRow = dut.io.resp.bits.wRow.peek().litValue.toInt
                //println(f"[cycle=$i%2d] req: v=$iv%d r=$ir%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d r=$or%d f=${if(respFire)1 else 0}%d")
                //println(f"tag=$tag whichBank=$whichBank wRow=$wRow")

                //// Print values at after two cycles
                //if (respFire && i >= 5) {
                    //println(dataA)
                    //val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}
                    //actual.zip(expectedCos).zipWithIndex.foreach { case ((act, exp), i) =>
                        //val actF = bfTofp(act)
                        //val expF = bfTofp(exp)
                        //println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    //}
                //}
                //// Step to next cycle and turn off input validity
                //dut.clock.step(1)
                //dut.io.req.valid.poke(false.B)
                //dut.io.req.bits.tag.poke(0.U) 
                //dut.io.req.bits.whichBank.poke(0.U)
                //dut.io.req.bits.wRow.poke(0.U)
            //}
        //}
    //}
//}
