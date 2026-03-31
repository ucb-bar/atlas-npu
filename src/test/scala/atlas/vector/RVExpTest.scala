package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome 
import atlas.common.VPUParams
import svsim.CommonCompilationSettings.Timescale.Unit.s
import sp26FPUnits.hardfloat._
import sp26FPUnits.AtlasFPType.BF16


class RVExpTest extends AnyFlatSpec with Matchers {

    //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("RVExpTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("RVExpTest=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

    val p = VPUParams()
    
    def fpTobf(f: Float): Int = {
        val x = java.lang.Float.floatToRawIntBits(f)
        val lsb = (x >>> 16) & 1
        val roundingBias = 0x7FFF + lsb
        ((x + roundingBias) >>> 16) & 0xFFFF
    }

    def bfTofp(b: Int): Float =
        java.lang.Float.intBitsToFloat((b & 0xFFFF) << 16)

    // def goldenExp16(aVec: Seq[Float], bVec: Seq[Float], isNeg: Boolean, isExp2: Boolean): Seq[Int] = {
    //     require(aVec.length == 16 && bVec.length == 16)

    //     aVec.zip(bVec).map { case (a, b) =>
    //         val aB = bfTofp(fpTobf(a))  
    //         val bB = bfTofp(fpTobf(b))  
    //         val y  = if (isExp2 && isNeg) Math.pow(2, -aB) 
    //                  else if (isExp2) Math.pow(2, aB) 
    //                  else if (!isExp2 && isNeg) Math.exp(-aB)
    //                  else Math.exp(aB)
    //         val yFloat = y.toFloat
    //         fpTobf(yFloat)                   // BF16 result bits as Int
    //     }
    // }

    def goldenExp16(aVec: Seq[Float], bVec: Seq[Float], isNeg: Boolean, isExp2: Boolean): Seq[Int] = {
        require(aVec.length == 16 && bVec.length == 16)

        aVec.zip(bVec).map { case (a, b) =>
            val aB = bfTofp(fpTobf(a))  
            val bB = bfTofp(fpTobf(b))  
            val y  = if (isExp2) Math.pow(2, aB) else Math.exp(aB)
            val yFloat = y.toFloat
            fpTobf(yFloat)                   // BF16 result bits as Int
        }
    }

    behavior of s"VectorEngine (BF16) Exp and Exp2 lanes=${16}"
    it should "Verify the correctness of add" in {
        simulate(new Exp(BF16)) { dut =>
            // Test data
            val dataA = Seq.fill(16)((scala.util.Random.nextFloat() - 0.5 ).toFloat)
            val dataB = Seq.fill(16)((scala.util.Random.nextFloat() - 0.5).toFloat)

            // Expected values from golden model
            val expectedNegExp2: Seq[Int] = goldenExp16(dataA, dataB, true, true)
            val expectedExp2: Seq[Int] = goldenExp16(dataA, dataB, false, true)
            val expectedNegExp: Seq[Int] = goldenExp16(dataA, dataB, true, false)
            val expectedExp: Seq[Int] = goldenExp16(dataA, dataB, false, false)

            // Convert inputs to BF16 bits using the SAME converter as golden
            val dataA_bf16: Seq[Int] = dataA.map(fpTobf)
            val dataB_bf16: Seq[Int] = dataB.map(fpTobf)

            dut.reset.poke(true.B)
            dut.clock.step(1)
            dut.reset.poke(false.B)
            dut.clock.step(1)

            // Poke inputs for negative exp2
            dut.io.req.valid.poke(true.B)
            dut.io.resp.ready.poke(true.B)
            dut.io.req.bits.roundingMode.poke(0.U) 
            dut.io.req.bits.tag.poke(1.U) 
            dut.io.req.bits.whichBank.poke(2.U)
            dut.io.req.bits.wRow.poke(3.U)
            dut.io.req.bits.neg.poke(true.B)
            dut.io.req.bits.isBase2.poke(true.B)
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.xVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 10) {
                println("Negative Exp2 Test Begin")
                val ir = dut.io.req.ready.peek().litValue
                val iv = dut.io.req.valid.peek().litValue
                val or = dut.io.resp.ready.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1 && ir == 1
                val respFire = ov == 1 && or == 1
                val tag = dut.io.resp.bits.tag.peek().litValue.toInt
                val whichBank = dut.io.resp.bits.whichBank.peek().litValue.toInt
                val wRow = dut.io.resp.bits.wRow.peek().litValue.toInt
                println(f"[cycle=$i%2d] req: v=$iv%d r=$ir%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d r=$or%d f=${if(respFire)1 else 0}%d")
                println(f"tag=$tag whichBank=$whichBank wRow=$wRow")
                // Print values at after two cycles
                if (respFire) {
                    
                    val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}
                    actual.zip(expectedNegExp2).zipWithIndex.foreach { case ((act, exp), i) =>
                        val actF = bfTofp(act)
                        val expF = bfTofp(exp)
                        println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
                dut.io.req.bits.tag.poke(0.U) 
                dut.io.req.bits.whichBank.poke(0.U)
                dut.io.req.bits.wRow.poke(0.U)
            }

            // Poke inputs for exp2
            dut.io.req.valid.poke(true.B)
            dut.io.resp.ready.poke(true.B)
            dut.io.req.bits.roundingMode.poke(0.U) 
            dut.io.req.bits.tag.poke(4.U) 
            dut.io.req.bits.whichBank.poke(5.U)
            dut.io.req.bits.wRow.poke(6.U)
            dut.io.req.bits.neg.poke(false.B)
            dut.io.req.bits.isBase2.poke(true.B)
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.xVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 10) {
                println("Positive Exp2 Test Begin")
                val ir = dut.io.req.ready.peek().litValue
                val iv = dut.io.req.valid.peek().litValue
                val or = dut.io.resp.ready.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1 && ir == 1
                val respFire = ov == 1 && or == 1
                val tag = dut.io.resp.bits.tag.peek().litValue.toInt
                val whichBank = dut.io.resp.bits.whichBank.peek().litValue.toInt
                val wRow = dut.io.resp.bits.wRow.peek().litValue.toInt
                println(f"[cycle=$i%2d] req: v=$iv%d r=$ir%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d r=$or%d f=${if(respFire)1 else 0}%d")
                println(f"tag=$tag whichBank=$whichBank wRow=$wRow")

                // Print values at after two cycles
                if (respFire) {
                    println(dataA)
                    val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}
                    actual.zip(expectedExp2).zipWithIndex.foreach { case ((act, exp), i) =>
                        val actF = bfTofp(act)
                        val expF = bfTofp(exp)
                        println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
                dut.io.req.bits.tag.poke(0.U) 
                dut.io.req.bits.whichBank.poke(0.U)
                dut.io.req.bits.wRow.poke(0.U)
            }


            // Poke inputs for negative exp 
            dut.io.req.valid.poke(true.B)
            dut.io.resp.ready.poke(true.B)
            dut.io.req.bits.roundingMode.poke(0.U) 
            dut.io.req.bits.tag.poke(7.U) 
            dut.io.req.bits.whichBank.poke(8.U)
            dut.io.req.bits.wRow.poke(9.U)
            dut.io.req.bits.neg.poke(true.B)
            dut.io.req.bits.isBase2.poke(false.B)
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.xVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 10) {
                println("Negative Exp Test Begin")
                val ir = dut.io.req.ready.peek().litValue
                val iv = dut.io.req.valid.peek().litValue
                val or = dut.io.resp.ready.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1 && ir == 1
                val respFire = ov == 1 && or == 1
                val tag = dut.io.resp.bits.tag.peek().litValue.toInt
                val whichBank = dut.io.resp.bits.whichBank.peek().litValue.toInt
                val wRow = dut.io.resp.bits.wRow.peek().litValue.toInt
                println(f"[cycle=$i%2d] req: v=$iv%d r=$ir%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d r=$or%d f=${if(respFire)1 else 0}%d")
                println(f"tag=$tag whichBank=$whichBank wRow=$wRow")

                // Print values at after two cycles
                if (respFire) {
                    println(dataA)
                    val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}
                    actual.zip(expectedNegExp).zipWithIndex.foreach { case ((act, exp), i) =>
                        val actF = bfTofp(act)
                        val expF = bfTofp(exp)
                        println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
                dut.io.req.bits.tag.poke(0.U) 
                dut.io.req.bits.whichBank.poke(0.U)
                dut.io.req.bits.wRow.poke(0.U)
            }

            // Poke inputs for negative exp 
            dut.io.req.valid.poke(true.B)
            dut.io.resp.ready.poke(true.B)
            dut.io.req.bits.roundingMode.poke(0.U) 
            dut.io.req.bits.tag.poke(10.U) 
            dut.io.req.bits.whichBank.poke(11.U)
            dut.io.req.bits.wRow.poke(12.U)
            dut.io.req.bits.neg.poke(false.B)
            dut.io.req.bits.isBase2.poke(false.B)
            dut.io.req.bits.laneMask.poke(0xFFFF.U) 
            dataA_bf16.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.xVec(i).poke(BigInt(bits & 0xFFFF)) }
            for (i <- 0 to 10) {
                println("Positive Exp Test Begin")
                val ir = dut.io.req.ready.peek().litValue
                val iv = dut.io.req.valid.peek().litValue
                val or = dut.io.resp.ready.peek().litValue
                val ov = dut.io.resp.valid.peek().litValue
                val reqFire  = iv == 1 && ir == 1
                val respFire = ov == 1 && or == 1
                val tag = dut.io.resp.bits.tag.peek().litValue.toInt
                val whichBank = dut.io.resp.bits.whichBank.peek().litValue.toInt
                val wRow = dut.io.resp.bits.wRow.peek().litValue.toInt
                println(f"[cycle=$i%2d] req: v=$iv%d r=$ir%d f=${if(reqFire)1 else 0}%d | resp: v=$ov%d r=$or%d f=${if(respFire)1 else 0}%d")
                println(f"tag=$tag whichBank=$whichBank wRow=$wRow")

                // Print values at after two cycles
                if (respFire && i >= 5) {
                    println(dataA)
                    val actual: Seq[Int] = (0 until 16).map { i => dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF}
                    actual.zip(expectedExp).zipWithIndex.foreach { case ((act, exp), i) =>
                        val actF = bfTofp(act)
                        val expF = bfTofp(exp)
                        println(f"lane=$i%2d  act=0x$act%04x ($actF%f)  exp=0x$exp%04x ($expF%f)")
                    }
                }
                // Step to next cycle and turn off input validity
                dut.clock.step(1)
                dut.io.req.valid.poke(false.B)
                dut.io.req.bits.tag.poke(0.U) 
                dut.io.req.bits.whichBank.poke(0.U)
                dut.io.req.bits.wRow.poke(0.U)
            }
        }
    }
}
