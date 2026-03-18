//package atlas.vector

//import chisel3._
//import chisel3.simulator.EphemeralSimulator._
//import org.scalatest.flatspec.AnyFlatSpec
//import org.scalatest.matchers.should.Matchers
//import atlas.common.VPUParams
//import sp26FPUnits.hardfloat._
//import svsim.CommonCompilationSettings.Timescale.Unit.s
//import fpex.FPType.BF16T
//import scala.math._

//class SinCosTest extends AnyFlatSpec with Matchers {
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

    //behavior of s"Sin/Cos Test (BF16) lanes=${16}"
    //it should "Verify the correctness of add" in {
        //simulate(new SinCos(BF16T, 16, 1)) { dut =>
            //// Test data
            //val twoPi = (2 * Math.PI).toFloat
            //val threePi_over_2 = (1.5 * Math.PI).toFloat
            //val pi = (Math.PI).toFloat
            //val pi_over_2 = (0.5 * Math.PI).toFloat
            //val pi_over_4 = (0.25 * Math.PI).toFloat
            //val pi_over_6 = (Math.PI/ 6.0).toFloat
            //val pi_over_12 = (Math.PI/ 12.0).toFloat
            //val dataA = Seq.fill(16)(scala.util.Random.nextFloat() * pi_over_2 + threePi_over_2)
            //// val dataA = Seq.fill(8)(pi_over_6 + pi_over_2) ++ Seq.fill(8)(pi_over_2 - pi_over_6)

            //// Expected values from golden model
            //val expected: Seq[Int] = goldenSinCos16(dataA, isCos = true)

            //// Convert inputs to BF16 bits using the SAME converter as golden
            //val dataA_bf16: Seq[Int] = dataA.map(fpTobf)
            //// val dataB_bf16: Seq[Int] = dataB.map(fpTobf)

            //// Poke inputs
            //dut.io.req.valid.poke(true.B)
            //dut.io.resp.ready.poke(true.B)
            //dut.io.req.bits.roundingMode.poke(0.U) 
            //dut.io.req.bits.tag.poke(0.U) 
            //dut.io.req.bits.cos.poke(true.B)
            //dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            //dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.xVec(i).poke(BigInt(bits & 0xFFFF)) }

            //// // Wait til valid
            //// while (!dut.io.resp.valid.peek().litToBoolean) {
            ////     dut.clock.step()    
                
            //// }
            //// dut.clock.step()
            //for (t <- 0 until 10) {


                //val reqValid = dut.io.req.valid.peek().litToBoolean
                //val reqReady = dut.io.req.ready.peek().litToBoolean
                //val reqFire  = reqValid && reqReady

                //val respValid = dut.io.resp.valid.peek().litToBoolean
                //val respReady = dut.io.resp.ready.peek().litToBoolean
                //val respFire  = respValid && respReady

                //val bits0 = (dut.io.resp.bits.result(0).peek().litValue & 0xFFFF).toInt
                //val f0    = bfTofp(bits0)

                //println(
                    //f"t=$t%3d | " +
                    //f"REQ: v=$reqValid r=$reqReady fire=$reqFire | " +
                    //f"RESP: v=$respValid r=$respReady fire=$respFire | " +
                    //f"lane0=0x$bits0%04x ($f0%f)"
                //)
                //// val k_d = dut.kval(1).peek().litValue
                //// val k_d8 = dut.kval(8).peek().litValue
                //// val addr = dut.addr(1).peek().litValue
                //// val addr8 = dut.addr(8).peek().litValue



                //// val stage3y0 = dut.d_stage3y0(0).peek().litValue
                //// val stage3y1 = dut.d_stage3y1(0).peek().litValue
                //// val stage3delta = dut.d_stage3delta(0).peek().litValue
                //// val stage3y08 = dut.d_stage3y0(8).peek().litValue
                //// val stage3y18 = dut.d_stage3y1(8).peek().litValue
                //// val stage3delta8 = dut.d_stage3delta(8).peek().litValue
                //// println(f"cycle=$t%2d, kval=$k_d%2d, addr=$addr%2d, stage3y0=$stage3y0%2d, stage3y1=$stage3y1%2d, stage3delta=$stage3delta%2d")
                //// println(f"cycle=$t%2d, kval=$k_d8%2d, addr8=$addr8%2d, stage3y0=$stage3y08%2d, stage3y1=$stage3y18%2d, stage3delta=$stage3delta8%2d")
                //dut.clock.step()
                //// dut.io.req.valid.poke(false.B)
            //}
            //// Print 
            //// for (i <- 0 until 16) {
            ////     val sign = dut.debug_rawSign(i).peek().litToBoolean
            ////     val exp  = dut.debug_rawExp(i).peek().litValue
            ////     val sig  = dut.debug_rawSig(i).peek().litValue

            ////     println(f"lane$i sign=$sign sExp=$exp sig=0x$sig%x")
            //// }

            //// now handshake occurs on this cycle
            //dut.clock.step()

            //// deassert valid so you don’t keep injecting the same op
            //dut.io.req.valid.poke(false.B)

            //// Read outputs as BF16 bits
            //val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}

            //// Check outputs (bit-exact BF16 compare)
            //actual.zip(expected).zipWithIndex.foreach { case ((act, exp), i) =>
                //val actF = bfTofp(act)
                //val expF = bfTofp(exp)
                //println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f), data=${dataA(i)}%f, data_in_bf_as_int=${dataA_bf16(i)}%f,")
            //}
            //// for (i <- 0 until 16) {
            ////     val k = dut.kval(i).peek().litValue
            ////     println(f"kval=$k%2d")
            //// }
        //}
    //}
//}