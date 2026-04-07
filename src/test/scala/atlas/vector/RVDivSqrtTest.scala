// ============================================================================
// RVDivSqrtTest.scala — BF16 divide/sqrt vector tests for the RV-style vector
// datapath.
//
// RUN: (from sp26-atlas-acc)
//    mill atlas.test.testOnly atlas.vector.RVDivSqrtTest
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome 
import atlas.common.VpuParams
import svsim.CommonCompilationSettings.Timescale.Unit.s
import sp26FPUnits.hardfloat._
import sp26FPUnits.AtlasFPType.BF16



class RVDivSqrtTest extends AnyFlatSpec with Matchers {

    //----------- CI/CD INCLUDE --------------
    override def withFixture(test: NoArgTest): Outcome = {
        val outcome = super.withFixture(test)
        if (outcome.isFailed) {
        println("RVDivSqrtTest=FAILED")
        } else if (outcome.isSucceeded) {
        println("RVDivSqrtTest=PASSED")
        }
        outcome
    }
   //----------- CI/CD INCLUDE --------------


    val p = VpuParams()
    
    def fpTobf(f: Float): Int = {
        val x = java.lang.Float.floatToRawIntBits(f)
        val lsb = (x >>> 16) & 1
        val roundingBias = 0x7FFF + lsb
        ((x + roundingBias) >>> 16) & 0xFFFF
    }

    def bfTofp(b: Int): Float =
        java.lang.Float.intBitsToFloat((b & 0xFFFF) << 16)

    def goldenDivSqrt16(aVec: Seq[Float], bVec: Seq[Float], isSqrt: Boolean): Seq[Int] = {
        require(aVec.length == 16 && bVec.length == 16)

        aVec.zip(bVec).map { case (a, b) =>
            val aB = bfTofp(fpTobf(a))  
            val bB = bfTofp(fpTobf(b))  
            val y  = if (isSqrt) Math.sqrt(aB) else (1 / aB)
            val y_float = y.toFloat
            fpTobf(y_float)                   // BF16 result bits as Int
        }
    }

    behavior of s"VectorEngine (BF16) Rcp/Sqrt lanes=${16}"
    it should "Verify the correctness of Rcp/Sqrt" in {
        simulate(new DivSqrtRec(BF16)) { dut =>
            // Test data
            val dataA = Seq.fill(16)(scala.util.Random.nextFloat() * 100)
            val dataB = Seq.fill(16)(scala.util.Random.nextFloat())

            // Expected values from golden model
            val expectedSqrt: Seq[Int] = goldenDivSqrt16(dataA, dataB, true)
            val expectedRcp: Seq[Int] = goldenDivSqrt16(dataB, dataB, false)

            // Convert inputs to BF16 bits using the SAME converter as golden
            val dataA_bf16: Seq[Int] = dataA.map(fpTobf)
            val dataB_bf16: Seq[Int] = dataB.map(fpTobf)

            // Poke inputs for sqrt
            dut.io.req.valid.poke(true.B)
            dut.io.resp.ready.poke(true.B)
            dut.io.req.bits.tag.poke(0.U) 
            dut.io.req.bits.isSqrt.poke(true.B)
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
            dataB_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.bVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 25) {
                val ir = dut.io.req.ready.peek().litValue
                val iv = dut.io.req.valid.peek().litValue
                val or = dut.io.resp.ready.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1 && ir == 1
                val respFire = ov == 1 && or == 1
                println(f"[cycle=$i%2d] req: v=$iv%d r=$ir%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d r=$or%d f=${if(respFire)1 else 0}%d")

                // Print values at after two cycles
                if (respFire) {
                    println(dataA)
                    val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}
                    actual.zip(expectedSqrt).zipWithIndex.foreach { case ((act, exp), i) =>
                        val actF = bfTofp(act)
                        val expF = bfTofp(exp)
                        println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                if (dut.io.req.ready.peek().litToBoolean) {
                    dut.io.req.valid.poke(true.B)
                } else {
                    dut.io.req.valid.poke(false.B)
                }
            }

            // Poke inputs for rcp
            dut.io.req.valid.poke(true.B)
            dut.io.resp.ready.poke(true.B)
            dut.io.req.bits.tag.poke(0.U) 
            dut.io.req.bits.isSqrt.poke(false.B)
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
            dataB_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 15) {
                val ir = dut.io.req.ready.peek().litValue
                val iv = dut.io.req.valid.peek().litValue
                val or = dut.io.resp.ready.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1 && ir == 1
                val respFire = ov == 1 && or == 1
                println(f"[cycle=$i%2d] req: v=$iv%d r=$ir%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d r=$or%d f=${if(respFire)1 else 0}%d")

                // Print values at after two cycles
                if (respFire) {
                    println(dataB)
                    val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}
                    actual.zip(expectedRcp).zipWithIndex.foreach { case ((act, exp), i) =>
                        val actF = bfTofp(act)
                        val expF = bfTofp(exp)
                        println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                if (dut.io.req.ready.peek().litToBoolean) {
                    dut.io.req.valid.poke(true.B)
                } else {
                    dut.io.req.valid.poke(false.B)
                }
            }
        }
    }
}
