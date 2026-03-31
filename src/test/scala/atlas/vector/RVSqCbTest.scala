// package atlas.vector

// import chisel3._
// import chisel3.simulator.EphemeralSimulator._
// import org.scalatest.flatspec.AnyFlatSpec
// import org.scalatest.matchers.should.Matchers
// import org.scalatest.Outcome 
// import atlas.common.VPUParams
// import svsim.CommonCompilationSettings.Timescale.Unit.s
// import sp26FPUnits.hardfloat._
// import sp26FPUnits.AtlasFPType.BF16
// import VectorTestUtils._


// class RVSqCbTest extends AnyFlatSpec with Matchers {

//     //----------- CI/CD INCLUDE --------------
//     override def withFixture(test: NoArgTest): Outcome = {
//         val outcome = super.withFixture(test)
//         if (outcome.isFailed) {
//         println("RVSqCbTest=FAILED")
//         } else if (outcome.isSucceeded) {
//         println("RVSqCbTest=PASSED")
//         }
//         outcome
//     }
//     //----------- CI/CD INCLUDE --------------

//     val p = VPUParams()

//     it should "Verify the correctness of square" in {
//         simulate(new SquareCubeRec(BF16)) { dut =>
//             // Test data
//             val dataA = randomBF16Vec()
//             val dataB = randomBF16Vec()

//             // Expected values from golden model
//             val expectedSquare1 = goldenSquareCubeVector(dataA, false)
//             val expectedSquare2 = goldenSquareCubeVector(dataB, false)

//             // Poke inputs
//             dut.io.req.valid.poke(true.B)
//             dut.io.resp.ready.poke(true.B)
//             dut.io.req.bits.isCube.poke(false.B)
//             dut.io.req.bits.roundingMode.poke(0.U) 
//             dut.io.req.bits.laneMask.poke(0xFFFF.U) 
//             dataA.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
//             for (i <- 0 to 5) {
//                 if (i == 1) {
//                     dut.io.req.valid.poke(true.B)
//                     dut.io.resp.ready.poke(true.B)
//                     dut.io.req.bits.isCube.poke(false.B)
//                     dut.io.req.bits.roundingMode.poke(0.U) 
//                     dut.io.req.bits.laneMask.poke(0xFFFF.U) 
//                     dataB.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
//                 }

//                 val ir = dut.io.req.ready.peek().litValue
//                 val iv = dut.io.req.valid.peek().litValue
//                 val or = dut.io.resp.ready.peek().litValue
//                 val ov = dut.io.resp.valid.peek().litValue
//                 val reqFire  = iv == 1 && ir == 1
//                 val respFire = ov == 1 && or == 1

//                 // Print values at after two cycles
//                 if (respFire) {
//                     println(s"Cycle = $i")
//                     val actual: Array[Short] = (0 until 16).map { i => (dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF).toShort}.toArray
                    
//                     if (i == 1) {
//                         printBF16Array(dataA, false)
//                         printBF16Array(actual, false)
//                         printBF16Array(expectedSquare1, true)
//                     } else if (i == 2) {
//                         printBF16Array(dataB, false)
//                         printBF16Array(actual, false)
//                         printBF16Array(expectedSquare2, true)
//                     }
//                 }
//                 // Step to next cycle and turn off input validity
//                 dut.clock.step(1)
//                 dut.io.req.valid.poke(false.B)
//             }
//         }
//     }

//     it should "Verify the correctness of cube" in {
//         simulate(new SquareCubeRec(BF16)) { dut =>
//             // Test data
//             val dataA = randomBF16Vec()
//             val dataB = randomBF16Vec()
//             val dataC = randomBF16Vec()

//             // Expected values from golden model
//             val expectedCube1 = goldenSquareCubeVector(dataA, true)
//             val expectedCube2 = goldenSquareCubeVector(dataB, true)
//             val expectedCube3 = goldenSquareCubeVector(dataC, true)

//             // Poke inputs
//             dut.io.req.valid.poke(true.B)
//             dut.io.resp.ready.poke(true.B)
//             dut.io.req.bits.isCube.poke(true.B)
//             dut.io.req.bits.roundingMode.poke(0.U) 
//             dut.io.req.bits.laneMask.poke(0xFFFF.U) 
//             dataA.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
//             for (i <- 0 to 5) {
//                 if (i == 1) {
//                     dut.io.req.valid.poke(true.B)
//                     dut.io.resp.ready.poke(true.B)
//                     dut.io.req.bits.isCube.poke(true.B)
//                     dut.io.req.bits.roundingMode.poke(0.U) 
//                     dut.io.req.bits.laneMask.poke(0xFFFF.U) 
//                     dataB.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
//                 } else if (i == 2) {
//                     dut.io.req.valid.poke(true.B)
//                     dut.io.resp.ready.poke(true.B)
//                     dut.io.req.bits.isCube.poke(true.B)
//                     dut.io.req.bits.roundingMode.poke(0.U) 
//                     dut.io.req.bits.laneMask.poke(0xFFFF.U) 
//                     dataC.zipWithIndex.foreach { case (bits, i) => dut.io.req.bits.aVec(i).poke(BigInt(bits & 0xFFFF)) }
//                 }
//                 val ir = dut.io.req.ready.peek().litValue
//                 val iv = dut.io.req.valid.peek().litValue
//                 val or = dut.io.resp.ready.peek().litValue
//                 val ov = dut.io.resp.valid.peek().litValue
//                 val reqFire  = iv == 1 && ir == 1
//                 val respFire = ov == 1 && or == 1

//                 // Print values at after two cycles
//                 if (respFire) {
//                     println(s"Cycle = $i")
//                     val actual: Array[Short] = (0 until 16).map { i => (dut.io.resp.bits.result(i).peek().litValue.toInt & 0xFFFF).toShort}.toArray
//                     if (i == 2) {
//                         printBF16Array(dataA, false)
//                         printBF16Array(actual, false)
//                         printBF16Array(expectedCube1, true)
//                     } else if (i == 3) {
//                         printBF16Array(dataB, false)
//                         printBF16Array(actual, false)
//                         printBF16Array(expectedCube2, true)
//                     } else if (i == 4) {
//                         printBF16Array(dataC, false)
//                         printBF16Array(actual, false)
//                         printBF16Array(expectedCube3, true)
//                     }
//                 }
//                 // Step to next cycle and turn off input validity
//                 dut.clock.step(1)
//                 dut.io.req.valid.poke(false.B)
//             }
//         }
//     }
// }
