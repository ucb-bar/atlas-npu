package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import atlas.common.VpuParams
import sp26FPUnits.hardfloat._
import svsim.CommonCompilationSettings.Timescale.Unit.s
import fpex.FPType.BF16T
import scala.math._
import scala.util.Random
import VectorTestUtils._
import scala.collection.immutable.LazyList.cons

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsVETUnitTestingSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "VETUnitTesting"

  private val runDir: Path = {
    val rootDirStr = sys.env.getOrElse("MILL_WORKSPACE_ROOT", "/tmp")
    val baseDir = Paths.get(rootDirStr)
    val p = baseDir.resolve("tmp").resolve(test_name)
    Files.createDirectories(p)
    p.toAbsolutePath
  }

  override val backend: VcsBackend   = VcsBackend.initializeFromProcessEnvironment()
  override val tag: String           = test_name
  override val workspacePath: String = runDir.toString

  override val commonCompilationSettings: CommonCompilationSettings =
    CommonCompilationSettings(
      availableParallelism =
        CommonCompilationSettings.AvailableParallelism.UpTo(Runtime.getRuntime.availableProcessors())
    )

  override val backendSpecificCompilationSettings: Backend.CompilationSettings = {
    val cov = Backend.CoverageSettings(
      line = true, cond = true, branch = true, fsm = true, tgl = true
    )
    Backend.CompilationSettings(
      coverageSettings  = cov,
      coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
      simulationSettings = Backend.SimulationSettings(
        coverageSettings  = cov,
        coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
        coverageName      = Some(Backend.CoverageName(s"${test_name}_coverage"))
      )
    )
  }
}

class VETUnitTesting extends AnyFlatSpec with Matchers with PeekPokeAPI {
    val p = VpuParams()

    behavior of s"VET Testing"
    it should "VET Correctness" in {
        PersistentVcsVETUnitTestingSimulator.simulate(new VectorEngine(p)) { module =>
            val dut = module.wrapped
            // Random test data
            // Used for Exp
            // val r0 = randomBF16Matrix()                     
            // val r1 = randomBF16Matrix()   
            // val formatedr0 = packMatrix(r0) 
            // val formatedr1 = packMatrix(r1)   
            // val expectedr0 = goldenExpMatrix(r0, false) 
            // val expectedr1 = goldenExpMatrix(r1, false)               
            // // Used for Sin
            // val r2 = randomBF16Matrix(max = 6, min = 0) 
            // val r3 = randomBF16Matrix(max = 6, min = 0)  
            // val formatedr2 = packMatrix(r2)
            // val formatedr3 = packMatrix(r3)
            // val expectedr2 = goldenSinCosMatrix(r2, false)
            // val expectedr3 = goldenSinCosMatrix(r3, false)
            // // Use for Square
            // val r4 = randomBF16Matrix()
            // val r5 = randomBF16Matrix()  
            // val formatedr4 = packMatrix(r4)
            // val formatedr5 = packMatrix(r5)
            // val expectedr4 = goldenSquareCubeMatrix(r4, false)
            // val expectedr5 = goldenSquareCubeMatrix(r5, false)
            // // Use for Rcp
            // val r6 = randomBF16Matrix()
            // val r7 = randomBF16Matrix()
            // val formatedr6 = packMatrix(r6)
            // val formatedr7 = packMatrix(r7)
            // val expectedr6 = goldenRcpSqrtMatrix(r6, false)
            // val expectedr7 = goldenRcpSqrtMatrix(r7, false)
            // // Use for Log
            // val r8 = randomBF16Matrix(max = 16, min = 1)
            // val r9 = randomBF16Matrix(max = 16, min = 1)
            // val formatedr8 = packMatrix(r8)
            // val formatedr9 = packMatrix(r9)
            // val expectedr8 = goldenLogMatrix(r8)
            // val expectedr9 = goldenLogMatrix(r9)

            // // Used for Exp2
            // val r10 = randomBF16Matrix()                    
            // val r11 = randomBF16Matrix()       
            // val formatedr10 = packMatrix(r10)
            // val formatedr11 = packMatrix(r11)  
            // val expectedr10 = goldenExpMatrix(r10, true)
            // val expectedr11 = goldenExpMatrix(r11, true)           
            // // Used for Cos
            // val r12 = randomBF16Matrix(max = 6, min = 0)  
            // val r13 = randomBF16Matrix(max = 6, min = 0)  
            // val formatedr12 = packMatrix(r12)
            // val formatedr13 = packMatrix(r13)
            // val expectedr12 = goldenSinCosMatrix(r12, true)
            // val expectedr13 = goldenSinCosMatrix(r13, true)
            // // Use for Cube
            // val r14 = randomBF16Matrix()
            // val r15 = randomBF16Matrix()
            // val formatedr14 = packMatrix(r14)
            // val formatedr15 = packMatrix(r15)
            // val expectedr14 = goldenSquareCubeMatrix(r14, true)
            // val expectedr15 = goldenSquareCubeMatrix(r15, true)
            // // Use for Sqrt
            // val r16 = randomBF16Matrix(max = 1000, min = 1)
            // val r17 = randomBF16Matrix(max = 1000, min = 1)
            // val formatedr16 = packMatrix(r16)
            // val formatedr17 = packMatrix(r17)
            // val expectedr16 = goldenRcpSqrtMatrix(r16, true)
            // val expectedr17 = goldenRcpSqrtMatrix(r17, true)
            // // Used for Tanh
            // val r18 = randomBF16Matrix()
            // val r19 = randomBF16Matrix()
            // val formatedr18 = packMatrix(r18)
            // val formatedr19 = packMatrix(r19)
            // val expectedr18 = goldenTanhMatrix(r18)
            // val expectedr19 = goldenTanhMatrix(r19)

            // // Use for Max
            // val r20 = randomBF16Matrix()
            // val r21 = randomBF16Matrix()
            // val formatedr20 = packMatrix(r20)
            // val formatedr21 = packMatrix(r21)
            // val expectedr20 = goldenRowMaxMatrix(r20)
            // val expectedr21 = goldenRowMaxMatrix(r21)
            // // Use for ReduSum
            // val r22 = randomBF16Matrix()
            // val r23 = randomBF16Matrix()
            // val formatedr22 = packMatrix(r22)
            // val formatedr23 = packMatrix(r23)
            // val expectedr22 = goldenReduSumMatrix(r22)
            // val expectedr23 = goldenReduSumMatrix(r23)
            // // Use for Relu
            // val r24 = randomBF16Matrix()
            // val r25 = randomBF16Matrix()
            // val formatedr24 = packMatrix(r24)
            // val formatedr25 = packMatrix(r25)
            // val expectedr24 = goldenReluMatrix(r24)
            // val expectedr25 = goldenReluMatrix(r25)
            // // Use for RowMin
            // val r26 = randomBF16Matrix()
            // val r27 = randomBF16Matrix()
            // val formatedr26 = packMatrix(r26)
            // val formatedr27 = packMatrix(r27)
            // val expectedr26 = goldenRowMinMatrix(r26)
            // val expectedr27 = goldenRowMinMatrix(r27)

            // // // Use for FP8
            // // val r28 = randomBF16Matrix()
            // // val r29 = randomBF16Matrix()
            // // val formatedr28 = packMatrix(r28)
            // // val formatedr29 = packMatrix(r29)
            // // val expectedr28 = (r28)
            // // val expectedr29 = goldenRowMinMatrix(r29)

            // // Use for Add first input
            // val r30 = randomBF16Matrix()
            // val r31 = randomBF16Matrix()
            // val formatedr30 = packMatrix(r30)
            // val formatedr31 = packMatrix(r31)
            // // Use for Sub first input
            // val r32 = randomBF16Matrix()
            // val r33 = randomBF16Matrix()
            // val formatedr32 = packMatrix(r32)
            // val formatedr33 = packMatrix(r33)
            // // Use for Mul first input
            // val r34 = randomBF16Matrix()
            // val r35 = randomBF16Matrix()
            // val formatedr34 = packMatrix(r34)
            // val formatedr35 = packMatrix(r35)
            // // Use for PairMax first input
            // val r36 = randomBF16Matrix()
            // val r37 = randomBF16Matrix()
            // val formatedr36 = packMatrix(r36)
            // val formatedr37 = packMatrix(r37)
            // // Use for PairMin first input
            // val r38 = randomBF16Matrix()
            // val r39 = randomBF16Matrix()
            // val formatedr38 = packMatrix(r38)
            // val formatedr39 = packMatrix(r39)
            
            // // Use for Add second input
            // val r40 = randomBF16Matrix()
            // val r41 = randomBF16Matrix()
            // val formatedr40 = packMatrix(r40)
            // val formatedr41 = packMatrix(r41)
            // // Use for Sub second input
            // val r42 = randomBF16Matrix()
            // val r43 = randomBF16Matrix()
            // val formatedr42 = packMatrix(r42)
            // val formatedr43 = packMatrix(r43)
            // // Use for Mul second input
            // val r44 = randomBF16Matrix()
            // val r45 = randomBF16Matrix()
            // val formatedr44 = packMatrix(r44)
            // val formatedr45 = packMatrix(r45)
            // // Use for PairMax second input
            // val r46 = randomBF16Matrix()
            // val r47 = randomBF16Matrix()
            // val formatedr46 = packMatrix(r46)
            // val formatedr47 = packMatrix(r47)
            // // Use for PairMin second input
            // val r48 = randomBF16Matrix()
            // val r49 = randomBF16Matrix()
            // val formatedr48 = packMatrix(r48)
            // val formatedr49 = packMatrix(r49)

            // val expectedr30 = goldenAddSubMatrix(r30, r40, false)
            // val expectedr31 = goldenAddSubMatrix(r31, r41, false)
            // val expectedr32 = goldenAddSubMatrix(r32, r42, true)
            // val expectedr33 = goldenAddSubMatrix(r33, r43, true)
            // val expectedr34 = goldenMulMatrix(r34, r44)
            // val expectedr35 = goldenMulMatrix(r35, r45)
            // val expectedr36 = goldenPairwiseMaxMatrix(r36, r46)
            // val expectedr37 = goldenPairwiseMaxMatrix(r37, r47)
            // val expectedr38 = goldenPairwiseMinMatrix(r38, r48)
            // val expectedr39 = goldenPairwiseMinMatrix(r39, r49)

            // // Reset VET
            dut.reset.poke(true.B)
            dut.clock.step(1)
            dut.reset.poke(false.B)
            dut.clock.step(1)

            // // ---------------------------------------------------
            // // Unit Testing 1: Rcp, log, tanh, sin, cos, exp, exp2
            // // ---------------------------------------------------
            // for (i <- 0 until 500) {
            //     if (i == 0) {
            //         // Poking for exp
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.exp)
            //         dut.io.inst.bits.instReadBank1.poke(0.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(0.U)
            //     } else if (i == 1) {
            //         // Poking for cos
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.cos)
            //         dut.io.inst.bits.instReadBank1.poke(12.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(12.U)
            //     } else if (i == 100) {
            //         // Poking for sin
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.sin)
            //         dut.io.inst.bits.instReadBank1.poke(2.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(2.U)
            //     } else if (i == 101) {
            //         // Poking for cube
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.cube)
            //         dut.io.inst.bits.instReadBank1.poke(14.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(14.U)
            //     } else if (i == 200) {
            //         // Poking for square
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.square)
            //         dut.io.inst.bits.instReadBank1.poke(4.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(4.U)
            //     } else if (i == 201) {
            //         // Poking for sqrt
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.sqrt)
            //         dut.io.inst.bits.instReadBank1.poke(16.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(16.U)
            //     } else if (i == 300) {
            //         // Poking for rcp
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.rcp)
            //         dut.io.inst.bits.instReadBank1.poke(6.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(6.U)
            //     } else if (i == 301) {
            //         // Poking for tanh
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.tanh)
            //         dut.io.inst.bits.instReadBank1.poke(18.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(18.U)
            //     } else if (i == 400) {
            //         // Poking for log
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.log)
            //         dut.io.inst.bits.instReadBank1.poke(8.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(8.U)
            //     } else if (i == 401) {
            //         // Poking for exp2
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.exp2)
            //         dut.io.inst.bits.instReadBank1.poke(10.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(10.U)
            //     } else {
            //         dut.io.inst.valid.poke(false.B)
            //     }

            //     val readValid1: Boolean = dut.io.read1.valid.peek().litToBoolean
            //     val readBank1: Int = dut.io.read1.bits.bank.peek().litValue.toInt
            //     val readRow1: Int = dut.io.read1.bits.row.peek().litValue.toInt
            //     val rValid0 = (readValid1) && (readBank1 == 0)
            //     val rValid1 = (readValid1) && (readBank1 == 1)
            //     val rValid2 = (readValid1) && (readBank1 == 2)
            //     val rValid3 = (readValid1) && (readBank1 == 3)
            //     val rValid4 = (readValid1) && (readBank1 == 4)
            //     val rValid5 = (readValid1) && (readBank1 == 5)
            //     val rValid6 = (readValid1) && (readBank1 == 6)
            //     val rValid7 = (readValid1) && (readBank1 == 7)
            //     val rValid8 = (readValid1) && (readBank1 == 8)
            //     val rValid9 = (readValid1) && (readBank1 == 9)

            //     val readValid2: Boolean = dut.io.read2.valid.peek().litToBoolean
            //     val readBank2: Int = dut.io.read2.bits.bank.peek().litValue.toInt
            //     val readRow2: Int = dut.io.read2.bits.row.peek().litValue.toInt
            //     val rValid10 = (readValid2) && (readBank2 == 10)
            //     val rValid11 = (readValid2) && (readBank2 == 11)
            //     val rValid12 = (readValid2) && (readBank2 == 12)
            //     val rValid13 = (readValid2) && (readBank2 == 13)
            //     val rValid14 = (readValid2) && (readBank2 == 14)
            //     val rValid15 = (readValid2) && (readBank2 == 15)
            //     val rValid16 = (readValid2) && (readBank2 == 16)
            //     val rValid17 = (readValid2) && (readBank2 == 17)
            //     val rValid18 = (readValid2) && (readBank2 == 18)
            //     val rValid19 = (readValid2) && (readBank2 == 19)
                

            //     val writeValid1: Boolean = dut.io.write1.valid.peek().litToBoolean
            //     val writeBank1: Int = dut.io.write1.bits.bank.peek().litValue.toInt
            //     val writeRow1: Int = dut.io.write1.bits.row.peek().litValue.toInt
            //     val wValid0 = (writeValid1 && writeBank1 == 0)
            //     val wValid1 = (writeValid1 && writeBank1 == 1)
            //     val wValid2 = (writeValid1 && writeBank1 == 2)
            //     val wValid3 = (writeValid1 && writeBank1 == 3)
            //     val wValid4 = (writeValid1 && writeBank1 == 4)
            //     val wValid5 = (writeValid1 && writeBank1 == 5)
            //     val wValid6 = (writeValid1 && writeBank1 == 6)
            //     val wValid7 = (writeValid1 && writeBank1 == 7)
            //     val wValid8 = (writeValid1 && writeBank1 == 8)
            //     val wValid9 = (writeValid1 && writeBank1 == 9)

            //     val writeValid2: Boolean = dut.io.write2.valid.peek().litToBoolean
            //     val writeBank2: Int = dut.io.write2.bits.bank.peek().litValue.toInt
            //     val writeRow2: Int = dut.io.write2.bits.row.peek().litValue.toInt
            //     val wValid10 = (writeValid2 && writeBank2 == 10)
            //     val wValid11 = (writeValid2 && writeBank2 == 11)
            //     val wValid12 = (writeValid2 && writeBank2 == 12)
            //     val wValid13 = (writeValid2 && writeBank2 == 13)
            //     val wValid14 = (writeValid2 && writeBank2 == 14)
            //     val wValid15 = (writeValid2 && writeBank2 == 15)
            //     val wValid16 = (writeValid2 && writeBank2 == 16)
            //     val wValid17 = (writeValid2 && writeBank2 == 17)
            //     val wValid18 = (writeValid2 && writeBank2 == 18)
            //     val wValid19 = (writeValid2 && writeBank2 == 19)

            //     // Write to the register if writes are valid
            //     if (wValid0) {
            //         r0(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         //println(s"Cycle: ${i} write1 valid")
            //     } else if (wValid1) {
            //         r1(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle:  ${i} => last row written for Exp")}
            //     } else if (wValid2) {
            //         r2(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //     } else if (wValid3) {
            //         r3(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Sin")}
            //     } else if (wValid4) {
            //         r4(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //     } else if (wValid5) {
            //         r5(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Square")}
            //     } else if (wValid6) {
            //         r6(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //     } else if (wValid7) {
            //         r7(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Rcp")}
            //     } else if (wValid8) {
            //         r8(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //     } else if (wValid9) {
            //         r9(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Log")}
            //     }

            //     if (wValid10) {
            //         r10(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //         //println(s"Cycle: ${i} write1 valid")
            //     } else if (wValid11) {
            //         r11(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //         if (writeRow2 == 31) {println(s"Cycle: ${i} => last row written for Exp2")}
            //     } else if (wValid12) {
            //         r12(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //     } else if (wValid13) {
            //         r13(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //         if (writeRow2 == 31) {println(s"Cycle:  ${i} => last row written for Cos")}
            //     } else if (wValid14) {
            //         r14(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //     } else if (wValid15) {
            //         r15(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //         if (writeRow2 == 31) {println(s"Cycle: ${i} => last row written for Cube")}
            //     } else if (wValid16) {
            //         r16(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //     } else if (wValid17) {
            //         r17(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //         if (writeRow2 == 31) {println(s"Cycle: ${i} => last row written for Sqrt")}
            //     } else if (wValid18) {
            //         r18(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //     } else if (wValid19) {
            //         r19(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //         if (writeRow2 == 31) {println(s"Cycle: ${i} => last row written for Tanh")}
            //     }


            //     dut.clock.step(1)

            //     // Send data to the VET if reads were asserted last cycle
            //     if (rValid0) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr0(readRow1).U)
            //     } else if (rValid1) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr1(readRow1).U)
            //     } else if (rValid2) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr2(readRow1).U)
            //     } else if (rValid3) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr3(readRow1).U)
            //     } else if (rValid4) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr4(readRow1).U)
            //     } else if (rValid5) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr5(readRow1).U)
            //     } else if (rValid6) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr6(readRow1).U)
            //     } else if (rValid7) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr7(readRow1).U)
            //     } else if (rValid8) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr8(readRow1).U)
            //     } else if (rValid9) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr9(readRow1).U)
            //     } else {
            //         dut.io.data1.valid.poke(false.B)
            //     }

            //     if (rValid10) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr10(readRow2).U)
            //     } else if (rValid11) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr11(readRow2).U)
            //     } else if (rValid12) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr12(readRow2).U)
            //     } else if (rValid13) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr13(readRow2).U)
            //     } else if (rValid14) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr14(readRow2).U)
            //     } else if (rValid15) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr15(readRow2).U)
            //     } else if (rValid16) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr16(readRow2).U)
            //     } else if (rValid17) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr17(readRow2).U)
            //     } else if (rValid18) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr18(readRow2).U)
            //     } else if (rValid19) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr19(readRow2).U)
            //     } else {
            //         dut.io.data2.valid.poke(false.B)
            //     }
            // }

            // // ---------------------------------------------------
            // // Unit Testing 2: Rcp, log, tanh, sin, cos, exp, exp2
            // // ---------------------------------------------------
            // for (i <- 500 until 700) {
            //     if (i == 500) {
            //         // Poking for max
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.rmax)
            //         dut.io.inst.bits.instReadBank1.poke(20.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(20.U)
            //     } else if (i == 501) {
            //         // Poking for reduSum
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.reduSum)
            //         dut.io.inst.bits.instReadBank1.poke(22.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(22.U)
            //     } else if (i == 600) {
            //         // Poking for relu
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.relu)
            //         dut.io.inst.bits.instReadBank1.poke(24.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(24.U)
            //     } else if (i == 601) {
            //         // Poking for RowMin
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.rmin)
            //         dut.io.inst.bits.instReadBank1.poke(26.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(26.U)
            //     } else if (i == 1000) {
            //         // Poking for FP8
            //         dut.io.inst.valid.poke(false.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.fp8)
            //         dut.io.inst.bits.instReadBank1.poke(26.U)
            //         dut.io.inst.bits.instReadBank2.poke(0.U)    // Does not matter for single input
            //         dut.io.inst.bits.instWriteBank.poke(26.U)
            //     } else {
            //         dut.io.inst.valid.poke(false.B)
            //     }

            //     val readValid1: Boolean = dut.io.read1.valid.peek().litToBoolean
            //     val readBank1: Int = dut.io.read1.bits.bank.peek().litValue.toInt
            //     val readRow1: Int = dut.io.read1.bits.row.peek().litValue.toInt
            //     val rValid20 = (readValid1) && (readBank1 == 20)
            //     val rValid21 = (readValid1) && (readBank1 == 21)
            //     val rValid24 = (readValid1) && (readBank1 == 24)
            //     val rValid25 = (readValid1) && (readBank1 == 25)


            //     val readValid2: Boolean = dut.io.read2.valid.peek().litToBoolean
            //     val readBank2: Int = dut.io.read2.bits.bank.peek().litValue.toInt
            //     val readRow2: Int = dut.io.read2.bits.row.peek().litValue.toInt
            //     val rValid22 = (readValid2) && (readBank2 == 22)
            //     val rValid23 = (readValid2) && (readBank2 == 23)
            //     val rValid26 = (readValid2) && (readBank2 == 26)
            //     val rValid27 = (readValid2) && (readBank2 == 27)
                

            //     val writeValid1: Boolean = dut.io.write1.valid.peek().litToBoolean
            //     val writeBank1: Int = dut.io.write1.bits.bank.peek().litValue.toInt
            //     val writeRow1: Int = dut.io.write1.bits.row.peek().litValue.toInt
            //     val wValid20 = (writeValid1 && writeBank1 == 20)
            //     val wValid21 = (writeValid1 && writeBank1 == 21)
            //     val wValid24 = (writeValid1 && writeBank1 == 24)
            //     val wValid25 = (writeValid1 && writeBank1 == 25)

            //     val writeValid2: Boolean = dut.io.write2.valid.peek().litToBoolean
            //     val writeBank2: Int = dut.io.write2.bits.bank.peek().litValue.toInt
            //     val writeRow2: Int = dut.io.write2.bits.row.peek().litValue.toInt
            //     val wValid22 = (writeValid2 && writeBank2 == 22)
            //     val wValid23 = (writeValid2 && writeBank2 == 23)
            //     val wValid26 = (writeValid2 && writeBank2 == 26)
            //     val wValid27 = (writeValid2 && writeBank2 == 27)

            //     // // println(s"Cycle=$i w2=$wValid1 wRow1=$writeRow1")
            //     // val r1v = dut.io.read1.valid.peek().litToBoolean
            //     // val r1b = dut.io.read1.bits.bank.peek().litValue.toInt
            //     // val r1r = dut.io.read1.bits.row.peek().litValue.toInt

            //     // val r2v = dut.io.read2.valid.peek().litToBoolean
            //     // val r2b = dut.io.read2.bits.bank.peek().litValue.toInt
            //     // val r2r = dut.io.read2.bits.row.peek().litValue.toInt

            //     // val w1v = dut.io.write1.valid.peek().litToBoolean
            //     // val w1b = dut.io.write1.bits.bank.peek().litValue.toInt
            //     // val w1r = dut.io.write1.bits.row.peek().litValue.toInt

            //     // val w2v = dut.io.write2.valid.peek().litToBoolean
            //     // val w2b = dut.io.write2.bits.bank.peek().litValue.toInt
            //     // val w2r = dut.io.write2.bits.row.peek().litValue.toInt

            //     // val ready = dut.io.inst.ready.peek().litToBoolean
            //     // val valid = dut.io.inst.valid.peek().litToBoolean

            //     // println(s"Cycle=$i | R1 v=$r1v b=$r1b row=$r1r | R2 v=$r2v b=$r2b row=$r2r | W1 v=$w1v b=$w1b row=$w1r | W2 v=$w2v b=$w2b row=$w2r, ready=$ready valid=$valid")

            //     // Write to the register if writes are valid
            //     if (wValid20) {
            //         r20(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         //println(s"Cycle: ${i} write1 valid")
            //     } else if (wValid21) {
            //         r21(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Max")}
            //     } else if (wValid24) {
            //         r24(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //     } else if (wValid25) {
            //         r25(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Relu")}
            //     }

            //     if (wValid22) {
            //         r22(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //         //println(s"Cycle: ${i} write1 valid")
            //     } else if (wValid23) {
            //         r23(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //         if (writeRow2 == 31) {println(s"Cycle: ${i} => last row written for ReduSum")}
            //     } else if (wValid26) {
            //         r26(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //     } else if (wValid27) {
            //         r27(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
            //         if (writeRow2 == 31) {println(s"Cycle: ${i} => last row written for FP8")}
            //     }

            //     dut.clock.step(1)

            //     // Send data to the VET if reads were asserted last cycle
                
            //     if (rValid20) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr20(readRow1).U)
            //     } else if (rValid21) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr21(readRow1).U)
            //     } else if (rValid24) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr24(readRow1).U)
            //     } else if (rValid25) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr25(readRow1).U)
            //     } else {
            //         dut.io.data1.valid.poke(false.B)
            //     }

            //     if (rValid22) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr22(readRow2).U)
            //     } else if (rValid23) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr23(readRow2).U)
            //     } else if (rValid26) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr26(readRow2).U)
            //     } else if (rValid27) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr27(readRow2).U)
            //     } else {
            //         dut.io.data2.valid.poke(false.B)
            //     }
            // }
            
            // for (i <- 700 until 1200) {
            //     if (i == 700) {
            //         // Poking for sqrt
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.add)
            //         dut.io.inst.bits.instReadBank1.poke(30.U)
            //         dut.io.inst.bits.instReadBank2.poke(40.U)    
            //         dut.io.inst.bits.instWriteBank.poke(30.U)
            //     } else if (i == 800) {
            //         // Poking for sin
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.sub)
            //         dut.io.inst.bits.instReadBank1.poke(32.U)
            //         dut.io.inst.bits.instReadBank2.poke(42.U)    
            //         dut.io.inst.bits.instWriteBank.poke(32.U)
            //     } else if (i == 900) {
            //         // Poking for sin
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.mul)
            //         dut.io.inst.bits.instReadBank1.poke(34.U)
            //         dut.io.inst.bits.instReadBank2.poke(44.U)    
            //         dut.io.inst.bits.instWriteBank.poke(34.U)
            //     } else if (i == 1000) {
            //         // Poking for PairwiseMax
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.pairmax)
            //         dut.io.inst.bits.instReadBank1.poke(36.U)
            //         dut.io.inst.bits.instReadBank2.poke(46.U)    
            //         dut.io.inst.bits.instWriteBank.poke(36.U)
            //     } else if (i == 1100) {
            //         // Poking for PairwiseMin
            //         dut.io.inst.valid.poke(true.B)
            //         dut.io.inst.bits.instType.poke(VPUOp.pairmin)
            //         dut.io.inst.bits.instReadBank1.poke(38.U)
            //         dut.io.inst.bits.instReadBank2.poke(48.U)    
            //         dut.io.inst.bits.instWriteBank.poke(38.U)
            //     } else {
            //         dut.io.inst.valid.poke(false.B)
            //     }

            //     val readValid1: Boolean = dut.io.read1.valid.peek().litToBoolean
            //     val readBank1: Int = dut.io.read1.bits.bank.peek().litValue.toInt
            //     val readRow1: Int = dut.io.read1.bits.row.peek().litValue.toInt
            //     val rValid30 = (readValid1) && (readBank1 == 30)
            //     val rValid31 = (readValid1) && (readBank1 == 31)
            //     val rValid32 = (readValid1) && (readBank1 == 32)
            //     val rValid33 = (readValid1) && (readBank1 == 33)
            //     val rValid34 = (readValid1) && (readBank1 == 34)
            //     val rValid35 = (readValid1) && (readBank1 == 35)
            //     val rValid36 = (readValid1) && (readBank1 == 36)
            //     val rValid37 = (readValid1) && (readBank1 == 37)
            //     val rValid38 = (readValid1) && (readBank1 == 38)
            //     val rValid39 = (readValid1) && (readBank1 == 39)

            //     val readValid2: Boolean = dut.io.read2.valid.peek().litToBoolean
            //     val readBank2: Int = dut.io.read2.bits.bank.peek().litValue.toInt
            //     val readRow2: Int = dut.io.read2.bits.row.peek().litValue.toInt
            //     val rValid40 = (readValid2) && (readBank2 == 40)
            //     val rValid41 = (readValid2) && (readBank2 == 41)
            //     val rValid42 = (readValid2) && (readBank2 == 42)
            //     val rValid43 = (readValid2) && (readBank2 == 43)
            //     val rValid44 = (readValid2) && (readBank2 == 44)
            //     val rValid45 = (readValid2) && (readBank2 == 45)
            //     val rValid46 = (readValid2) && (readBank2 == 46)
            //     val rValid47 = (readValid2) && (readBank2 == 47)
            //     val rValid48 = (readValid2) && (readBank2 == 48)
            //     val rValid49 = (readValid2) && (readBank2 == 49)

            //     val writeValid1: Boolean = dut.io.write1.valid.peek().litToBoolean
            //     val writeBank1: Int = dut.io.write1.bits.bank.peek().litValue.toInt
            //     val writeRow1: Int = dut.io.write1.bits.row.peek().litValue.toInt
            //     val wValid30 = (writeValid1 && writeBank1 == 30)
            //     val wValid31 = (writeValid1 && writeBank1 == 31)
            //     val wValid32 = (writeValid1 && writeBank1 == 32)
            //     val wValid33 = (writeValid1 && writeBank1 == 33)
            //     val wValid34 = (writeValid1 && writeBank1 == 34)
            //     val wValid35 = (writeValid1 && writeBank1 == 35)
            //     val wValid36 = (writeValid1 && writeBank1 == 36)
            //     val wValid37 = (writeValid1 && writeBank1 == 37)
            //     val wValid38 = (writeValid1 && writeBank1 == 38)
            //     val wValid39 = (writeValid1 && writeBank1 == 39)

            //     val writeValid2: Boolean = dut.io.write2.valid.peek().litToBoolean
            //     val writeBank2: Int = dut.io.write2.bits.bank.peek().litValue.toInt
            //     val writeRow2: Int = dut.io.write2.bits.row.peek().litValue.toInt

            //     // Write to the register if writes are valid
            //     if (wValid30) {
            //         r30(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         // println(s"Cycle: ${i} write1 valid")
            //     } else if (wValid31) {
            //         r31(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Add")}
            //     } else if (wValid32) {
            //         r32(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //     } else if (wValid33) {
            //         r33(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Sub")}
            //     } else if (wValid34) {
            //         r34(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //     } else if (wValid35) {
            //         r35(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Mul")}
            //     } else if (wValid36) {
            //         r36(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //     } else if (wValid37) {
            //         r37(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for PairwiseMax")}
            //     } else if (wValid38) {
            //         r38(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //     } else if (wValid39) {
            //         r39(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
            //         if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for PairwiseMin")}
            //     }

            //     dut.clock.step(1)

            //     // Send data to the VET if reads were asserted last cycle
            //     if (rValid30) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr30(readRow1).U)
            //     } else if (rValid31) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr31(readRow1).U)
            //     } else if (rValid32) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr32(readRow1).U)
            //     } else if (rValid33) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr33(readRow1).U)
            //     } else if (rValid34) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr34(readRow1).U)
            //     } else if (rValid35) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr35(readRow1).U)
            //     } else if (rValid36) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr36(readRow1).U)
            //     } else if (rValid37) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr37(readRow1).U)
            //     } else if (rValid38) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr38(readRow1).U)
            //     } else if (rValid39) {
            //         dut.io.data1.valid.poke(true.B)
            //         dut.io.data1.bits.data.poke(formatedr39(readRow1).U)
            //     } else {
            //         dut.io.data1.valid.poke(false.B)
            //     }
                
            //     if (rValid40) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr40(readRow2).U)
            //     } else if (rValid41) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr41(readRow2).U)
            //     } else if (rValid42) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr42(readRow2).U)
            //     } else if (rValid43) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr43(readRow2).U)
            //     } else if (rValid44) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr44(readRow2).U)
            //     } else if (rValid45) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr45(readRow2).U)
            //     } else if (rValid46) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr46(readRow2).U)
            //     } else if (rValid47) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr47(readRow2).U)
            //     } else if (rValid48) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr48(readRow2).U)
            //     } else if (rValid49) {
            //         dut.io.data2.valid.poke(true.B)
            //         dut.io.data2.bits.data.poke(formatedr49(readRow2).U)
            //     } else {
            //         dut.io.data2.valid.poke(false.B)
            //     }
            // }

            // Fourth Test
            // Use for col reduce max
            val r50 = randomBF16Matrix()
            val r51 = randomBF16Matrix()
            val formatedr50 = packMatrix(r50)
            val formatedr51 = packMatrix(r51)
            val expectedr50 = goldenColMaxMatrix(r50)
            val expectedr51 = goldenColMaxMatrix(r51)
            val finalVecMax = goldenPairwiseMaxVector(expectedr50(0), expectedr51(0))
            val expected50n51 = Array.fill(32)(finalVecMax)

            // Use for col reduce min
            val r52 = randomBF16Matrix()
            val r53 = randomBF16Matrix()
            val formatedr52 = packMatrix(r52)
            val formatedr53 = packMatrix(r53)
            val expectedr52 = goldenColMinMatrix(r52)
            val expectedr53 = goldenColMinMatrix(r53)
            val finalVecMin = goldenPairwiseMinVector(expectedr52(0), expectedr53(0))
            val expected52n53 = Array.fill(32)(finalVecMin)

            // Use for col reduce sum
            val r54 = randomBF16Matrix(useInt = false)   // To avoid overflow in sum, use smaller values for r54
            val r55 = randomBF16Matrix(useInt = false)   // To avoid overflow in sum, use smaller values for r55
            val formatedr54 = packMatrix(r54)
            val formatedr55 = packMatrix(r55)
            val expectedr54 = goldenColSumMatrix(r54) 
            val expectedr55 = goldenColSumMatrix(r55)
            val finalVecSum = goldenAddSubVector(expectedr54(0), expectedr55(0), false)
            val expected54n55 = Array.fill(32)(finalVecSum)


            for (i <- 1200 until 1600) {
                if (i == 1200) {
                    // Poking for ColwiseMax
                    dut.io.inst.valid.poke(true.B)
                    dut.io.inst.bits.instType.poke(VPUOp.cmax)
                    // Test different read banks
                    dut.io.inst.bits.instReadBank1.poke(50.U)
                    dut.io.inst.bits.instReadBank2.poke(50.U)    
                    dut.io.inst.bits.instWriteBank.poke(50.U)
                } else if (i == 1201) {
                    // Poking for ColwiseMin
                    dut.io.inst.valid.poke(true.B)
                    dut.io.inst.bits.instType.poke(VPUOp.cmin)
                    dut.io.inst.bits.instReadBank1.poke(52.U)
                    dut.io.inst.bits.instReadBank2.poke(52.U)
                    dut.io.inst.bits.instWriteBank.poke(52.U)
                } else if (i == 1400) {
                    // Poking for ColwiseSum
                    dut.io.inst.valid.poke(true.B)
                    dut.io.inst.bits.instType.poke(VPUOp.csum)
                    dut.io.inst.bits.instReadBank1.poke(54.U)
                    dut.io.inst.bits.instReadBank2.poke(54.U)
                    dut.io.inst.bits.instWriteBank.poke(54.U)
                } else {
                    dut.io.inst.valid.poke(false.B)
                }

                val readValid1: Boolean = dut.io.read1.valid.peek().litToBoolean
                val readBank1: Int = dut.io.read1.bits.bank.peek().litValue.toInt
                val readRow1: Int = dut.io.read1.bits.row.peek().litValue.toInt
                val rValid50 = (readValid1) && (readBank1 == 50)
                val rValid51 = (readValid1) && (readBank1 == 51)
                val rValid54 = (readValid1) && (readBank1 == 54)
                val rValid55 = (readValid1) && (readBank1 == 55)
                
                val readValid2: Boolean = dut.io.read2.valid.peek().litToBoolean
                val readBank2: Int = dut.io.read2.bits.bank.peek().litValue.toInt
                val readRow2: Int = dut.io.read2.bits.row.peek().litValue.toInt
                val rValid52 = (readValid2) && (readBank2 == 52)
                val rValid53 = (readValid2) && (readBank2 == 53)

                val writeValid1: Boolean = dut.io.write1.valid.peek().litToBoolean
                val writeBank1: Int = dut.io.write1.bits.bank.peek().litValue.toInt
                val writeRow1: Int = dut.io.write1.bits.row.peek().litValue.toInt
                val wValid50 = (writeValid1 && writeBank1 == 50)
                val wValid51 = (writeValid1 && writeBank1 == 51)
                val wValid54 = (writeValid1 && writeBank1 == 54)
                val wValid55 = (writeValid1 && writeBank1 == 55)

                val writeValid2: Boolean = dut.io.write2.valid.peek().litToBoolean
                val writeBank2: Int = dut.io.write2.bits.bank.peek().litValue.toInt
                val writeRow2: Int = dut.io.write2.bits.row.peek().litValue.toInt
                val wValid52 = (writeValid2 && writeBank2 == 52)
                val wValid53 = (writeValid2 && writeBank2 == 53)

                // Write to the register if writes are valid
                if (wValid50) {
                    r50(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
                } else if (wValid51) {
                    r51(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
                    if (writeRow1 == 31) {println(s"Cycle: ${i} => last row written for Columnwise Max")
                    }
                } else if (wValid54) {
                    r54(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
                    println(s"Cycle: ${i} write1 valid row=${writeRow1}, bank=${writeBank1}")
                } else if (wValid55) {
                    r55(writeRow1) = unpackBF16Array(dut.io.write1.bits.data.peek().litValue)
                    if (writeRow1 == 31) {
                        println(s"Cycle: ${i} => last row written for Columnwise Sum")
                        printLane(expected54n55, 30, expected = true)
                        printLane(r55, 30, expected = false)
                    }
                    println(s"Cycle: ${i} write1 valid row=${writeRow1}, bank=${writeBank1}")
                }

                if (wValid52) {
                    r52(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
                } else if (wValid53) {
                    r53(writeRow2) = unpackBF16Array(dut.io.write2.bits.data.peek().litValue)
                    if (writeRow2 == 31) {println(s"Cycle: ${i} => last row written for Columnwise Min")
                    }
                }

                dut.clock.step(1)

                // Send data to the VET if reads were asserted last cycle
                if (rValid50) {
                    dut.io.data1.valid.poke(true.B)
                    dut.io.data1.bits.data.poke(formatedr50(readRow1).U)
                } else if (rValid51) {
                    dut.io.data1.valid.poke(true.B)
                    dut.io.data1.bits.data.poke(formatedr51(readRow1).U)
                } else if (rValid54) {
                    dut.io.data1.valid.poke(true.B)
                    dut.io.data1.bits.data.poke(formatedr54(readRow1).U)
                } else if (rValid55) {
                    dut.io.data1.valid.poke(true.B)
                    dut.io.data1.bits.data.poke(formatedr55(readRow1).U)
                } else {
                    dut.io.data1.valid.poke(false.B)
                } 

                if (rValid52) {
                    dut.io.data2.valid.poke(true.B)
                    dut.io.data2.bits.data.poke(formatedr52(readRow2).U)
                } else if (rValid53) {
                    dut.io.data2.valid.poke(true.B)
                    dut.io.data2.bits.data.poke(formatedr53(readRow2).U)
                } else {
                    dut.io.data2.valid.poke(false.B)
                }
                
                // if (rValid40) {
                //     dut.io.data2.valid.poke(true.B)
                //     dut.io.data2.bits.data.poke(formatedr40(readRow2).U)
                // } else {
                //     dut.io.data2.valid.poke(false.B)
                // }
            }



            // // Checking the output matrices with the expected matrices
            // val results1 = Seq(
            //     ("r0", checkMatrixTolerance(r0, expectedr0, "r0")),
            //     ("r1", checkMatrixTolerance(r1, expectedr1, "r1")),
            //     ("r2", checkMatrixTolerance(r2, expectedr2, "r2")),
            //     ("r3", checkMatrixTolerance(r3, expectedr3, "r3")),
            //     ("r4", checkMatrixTolerance(r4, expectedr4, "r4")),
            //     ("r5", checkMatrixTolerance(r5, expectedr5, "r5")),
            //     ("r6", checkMatrixTolerance(r6, expectedr6, "r6")),
            //     ("r7", checkMatrixTolerance(r7, expectedr7, "r7")),
            //     ("r8", checkMatrixTolerance(r8, expectedr8, "r8")),
            //     ("r9", checkMatrixTolerance(r9, expectedr9, "r9")),
            //     ("r10", checkMatrixTolerance(r10, expectedr10, "r10")),
            //     ("r11", checkMatrixTolerance(r11, expectedr11, "r11")),
            //     ("r12", checkMatrixTolerance(r12, expectedr12, "r12")),
            //     ("r13", checkMatrixTolerance(r13, expectedr13, "r13")),
            //     ("r14", checkMatrixTolerance(r14, expectedr14, "r14")),
            //     ("r15", checkMatrixTolerance(r15, expectedr15, "r15")),
            //     ("r16", checkMatrixTolerance(r16, expectedr16, "r16")),
            //     ("r17", checkMatrixTolerance(r17, expectedr17, "r17")),
            //     ("r18", checkMatrixTolerance(r18, expectedr18, "r18")),
            //     ("r19", checkMatrixTolerance(r19, expectedr19, "r19"))
            // )

            // // Count passes
            // val numPass1 = results1.count(_._2)
            // val total1   = results1.length

            // // Checking the output matrices with the expected matrices
            // val results2 = Seq(
            //     ("r20", checkMatrixTolerance(r20, expectedr20, "r20")),
            //     ("r21", checkMatrixTolerance(r21, expectedr21, "r21")),
            //     ("r22", checkMatrixTolerance(r22, expectedr22, "r22")),
            //     ("r23", checkMatrixTolerance(r23, expectedr23, "r23")),
            //     ("r24", checkMatrixTolerance(r24, expectedr24, "r24")),
            //     ("r25", checkMatrixTolerance(r25, expectedr25, "r25")),
            //     ("r26", checkMatrixTolerance(r26, expectedr26, "r26")),
            //     ("r27", checkMatrixTolerance(r27, expectedr27, "r27"))
            // )

            // // Count passes
            // val numPass2 = results2.count(_._2)
            // val total2   = results2.length

            // // Checking the output matrices with the expected matrices
            // val results3 = Seq(
            //     ("r30", checkMatrixTolerance(r30, expectedr30, "r30")),
            //     ("r31", checkMatrixTolerance(r31, expectedr31, "r31")),
            //     ("r32", checkMatrixTolerance(r32, expectedr32, "r32")),
            //     ("r33", checkMatrixTolerance(r33, expectedr33, "r33")),
            //     ("r34", checkMatrixTolerance(r34, expectedr34, "r34")),
            //     ("r35", checkMatrixTolerance(r35, expectedr35, "r35")),
            //     ("r36", checkMatrixTolerance(r36, expectedr36, "r36")),
            //     ("r37", checkMatrixTolerance(r37, expectedr37, "r37")),
            //     ("r38", checkMatrixTolerance(r38, expectedr38, "r38")),
            //     ("r39", checkMatrixTolerance(r39, expectedr39, "r39"))
            // )

            // // Count passes
            // val numPass3 = results3.count(_._2)
            // val total3   = results3.length

            val results4 = Seq(
                ("r50", checkMatrixTolerance(r50, expected50n51, "r50")),
                ("r51", checkMatrixTolerance(r51, expected50n51, "r51")),
                ("r52", checkMatrixTolerance(r52, expected52n53, "r52")),
                ("r53", checkMatrixTolerance(r53, expected52n53, "r53")),
                ("r54", checkMatrixTolerance(r54, expected54n55, "r54")),
                ("r55", checkMatrixTolerance(r55, expected54n55, "r55"))
            )

            // println(s"First  Round of Tests: Passed $numPass1 / $total1 tests")
            // println(s"Second Round of Tests: Passed $numPass2 / $total2 tests")
            // println(s"Third  Round of Tests: Passed $numPass3 / $total3 tests")
            println(s"Fourth Round of Tests: Passed ${results4.count(_._2)} / ${results4.length} tests")
            // Use assertion to check which matrix fail
            // results3.foreach { case (name, pass) =>
            //     assert(pass, s"$name failed")
            // }

            // val allResults = results1 ++ results2 ++ results3 ++ results4
            // allResults.foreach { case (name, pass) =>if (!pass) println(s"$name failed") }
            // assert(allResults.forall(_._2), "One or more matrices failed")
        }
    }
}