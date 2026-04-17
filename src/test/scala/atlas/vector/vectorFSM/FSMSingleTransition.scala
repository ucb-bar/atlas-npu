package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import atlas.common.VpuParams
import svsim.CommonCompilationSettings.Timescale.Unit.s
import sp26FPUnits.hardfloat._
import sp26FPUnits.AtlasFPType.BF16

import org.scalatest.Outcome 

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsFSMSingleTransitionSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "FSMSingleTransition"

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
      line = true, cond = true, branch = true, fsm = true, tgl = true, assert = true
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

class FSMSingleTransition extends AnyFlatSpec with Matchers with PeekPokeAPI {

    //----------- CI/CD INCLUDE --------------
    override def withFixture(test: NoArgTest): Outcome = {
        val outcome = super.withFixture(test)
        if (outcome.isFailed) {
        println("FSMSingleTransition=FAILED")
        } else if (outcome.isSucceeded) {
        println("FSMSingleTransition=PASSED")
        }
        outcome
    }
    //----------- CI/CD INCLUDE --------------
    
    val p = VpuParams()

    def vpuOpName(value: BigInt): String = {
        VPUOp.all.find(_.litValue == value).map(_.toString).getOrElse(s"Unknown($value)")
    }

    behavior of s"VectorFSM Testing"
    // it should "Verify the transition from single to double" in {
    //     simulate(new VectorFSM(p)) { dut =>
    //         // Reset the circuit
    //         dut.reset.poke(true.B)
    //         dut.clock.step(1)
    //         dut.reset.poke(false.B)
    //         dut.clock.step(1)

    //         // Testing
    //         for (i <- 0 to 135) {
    //             // Input fires at the beginning
    //             if (i == 0) {
    //                 dut.io.in.instFire.poke(true.B)
    //                 dut.io.in.instType.poke(VPUOp.fp8)
    //                 dut.io.in.instReadBank1.poke(1.U)
    //                 dut.io.in.instReadBank2.poke(2.U)
    //                 dut.io.in.instWriteBank.poke(3.U)
    //             // Input fires at the 65th cycle again which is when the first instruction is done
    //             } else if (i == 65) {
    //                 dut.io.in.instFire.poke(true.B)
    //                 dut.io.in.instType.poke(VPUOp.mul)
    //                 dut.io.in.instReadBank1.poke(11.U)
    //                 dut.io.in.instReadBank2.poke(12.U)
    //                 dut.io.in.instWriteBank.poke(13.U)
    //             }
    //             else {
    //                 dut.io.in.instFire.poke(false.B)
    //                 dut.io.in.instType.poke(VPUOp.sub)
    //                 dut.io.in.instReadBank1.poke(4.U)
    //                 dut.io.in.instReadBank2.poke(5.U)
    //                 dut.io.in.instWriteBank.poke(6.U)
    //             }

    //             // Cycle 1 to 64 should be when input handshake of compute units happens
    //             if (i >=1 && i <= 64) {
    //                 dut.io.in.dataInFire1.poke(true.B)
    //                 dut.io.in.dataInFire2.poke(true.B)
    //             // Cycle 66 to 129 should be when input handshake of compute units happens for second instruction fired at cycle 65
    //             } else if (i >=66 && i <= 129) {
    //                 dut.io.in.dataInFire1.poke(true.B)
    //                 dut.io.in.dataInFire2.poke(true.B)
    //             } else {
    //                 dut.io.in.dataInFire1.poke(false.B)
    //                 dut.io.in.dataInFire2.poke(false.B)
    //             }

    //             // Cycle 2 to 65 should be when output handshake of compute units happens
    //             if (i >= 2 && i <= 65) {
    //                 dut.io.in.dataOutFire1.poke(true.B)
    //             // Cycle 2 to 65 should be when output handshake of compute units happens for second instruciton fired at cycle 65
    //             } else if (i >= 67 && i <= 130) {
    //                 dut.io.in.dataOutFire1.poke(true.B)
    //             } else {
    //                 dut.io.in.dataOutFire1.poke(false.B)
    //                 dut.io.in.dataOutFire2.poke(false.B)
    //             }
                
    //             // State value
    //             val inst1 = dut.io.out.inst1.peek().litValue
    //             val inst2 = dut.io.out.inst2.peek().litValue

    //             println(
    //             s"""
    //                 |[Cycle Debug]
    //                 |cycle:  i=$i
    //                 |inst1:  a=${vpuOpName(inst1)} ($inst1)
    //                 |inst2:  b=${vpuOpName(inst2)} ($inst2)
    //                 |read1:  v=${dut.io.out.readValid1.peek().litToBoolean} bank=${dut.io.out.readBank1.peek().litValue} row=${dut.io.out.readRow1.peek().litValue} d=${dut.io.out.readDone1.peek().litToBoolean}
    //                 |read2:  v=${dut.io.out.readValid2.peek().litToBoolean} bank=${dut.io.out.readBank2.peek().litValue} row=${dut.io.out.readRow2.peek().litValue} d=${dut.io.out.readDone2.peek().litToBoolean}
    //                 |write1: v=${dut.io.out.writeValid1.peek().litToBoolean} bank=${dut.io.out.writeBank1.peek().litValue} row=${dut.io.out.writeRow1.peek().litValue} d=${dut.io.out.writeDone1.peek().litToBoolean}
    //                 |write2: v=${dut.io.out.writeValid2.peek().litToBoolean} bank=${dut.io.out.writeBank2.peek().litValue} row=${dut.io.out.writeRow2.peek().litValue} d=${dut.io.out.writeDone2.peek().litToBoolean}
    //                 |done:   d1=${dut.io.out.done1.peek().litToBoolean} d2=${dut.io.out.done2.peek().litToBoolean} ready=${dut.io.out.VEReady.peek().litToBoolean}
    //                 |state:  ${dut.io.out.state.peek().litValue}
    //                 |""".stripMargin
    //             )
    //             dut.clock.step(1)
    //         }
    //     }
    // }

    it should "Verify the transition from 2 single to double" in {
        PersistentVcsFSMSingleTransitionSimulator.simulate(new VectorFSM(p)) { module =>
            val dut = module.wrapped
            // Reset the circuit
            dut.reset.poke(true.B)
            dut.clock.step(1)
            dut.reset.poke(false.B)
            dut.clock.step(1)

            // Testing
            for (i <- 0 to 135) {
                // Input fires at the beginning
                if (i == 0) {
                    dut.io.in.instFire.poke(true.B)
                    dut.io.in.instType.poke(VPUOp.exp)
                    dut.io.in.instReadBank1.poke(1.U)
                    dut.io.in.instReadBank2.poke(2.U)
                    dut.io.in.instWriteBank.poke(3.U)
                // Input fires at the 65th cycle again which is when the first instruction is done
                } else if (i == 1) {
                    dut.io.in.instFire.poke(true.B)
                    dut.io.in.instType.poke(VPUOp.exp2)
                    dut.io.in.instReadBank1.poke(4.U)
                    dut.io.in.instReadBank2.poke(5.U)
                    dut.io.in.instWriteBank.poke(6.U)
                } else if (i == 66) {
                    dut.io.in.instFire.poke(true.B)
                    dut.io.in.instType.poke(VPUOp.mul)
                    dut.io.in.instReadBank1.poke(11.U)
                    dut.io.in.instReadBank2.poke(12.U)
                    dut.io.in.instWriteBank.poke(13.U)
                }
                else {
                    dut.io.in.instFire.poke(false.B)
                    dut.io.in.instType.poke(VPUOp.sub)
                    dut.io.in.instReadBank1.poke(7.U)
                    dut.io.in.instReadBank2.poke(8.U)
                    dut.io.in.instWriteBank.poke(9.U)
                }

                // Cycle 1 to 64 should be when input handshake of compute units happens
                if (i >= 1 && i <= 64) {
                    dut.io.in.dataInFire1.poke(true.B)
                // Cycle 67 to 130 should be when input handshake of the third compute units (double input) happens
                } else if (i >= 67 && i <= 130) {
                    dut.io.in.dataInFire1.poke(true.B)
                } else {
                    dut.io.in.dataInFire1.poke(false.B)
                }

                // Cycle 2 to 65 should be when output handshake of compute units happens
                if (i >= 2 && i <= 65) {
                    dut.io.in.dataOutFire1.poke(true.B)
                } else if (i >= 68 && i <= 131) {
                    dut.io.in.dataOutFire1.poke(true.B)
                } else {
                    dut.io.in.dataOutFire1.poke(false.B)
                }

                // Cycle 2 to 65 should be when input handshake of the second compute units happens
                if (i >= 2 && i <= 65) {
                    dut.io.in.dataInFire2.poke(true.B)
                // Cycle 67 to 130 should be when input handshake of the third compute units (double input) happens
                } else if (i >= 67 && i <= 130) {
                    dut.io.in.dataInFire2.poke(true.B)
                } else {
                    dut.io.in.dataInFire2.poke(false.B)
                }

                // Cycle 3 to 66 should be when output handshake of the second compute units happens
                if (i >= 3 && i <= 66) {
                    dut.io.in.dataOutFire2.poke(true.B)
                } else {
                    dut.io.in.dataOutFire2.poke(false.B)
                }
                
                // State value
                val inst1 = dut.io.out.inst1.peek().litValue
                val inst2 = dut.io.out.inst2.peek().litValue

                println(
                s"""
                    |[Cycle Debug]
                    |cycle:  i=$i
                    |inst1:  a=${vpuOpName(inst1)} ($inst1)
                    |inst2:  b=${vpuOpName(inst2)} ($inst2)
                    |read1:  v=${dut.io.out.readValid1.peek().litToBoolean} bank=${dut.io.out.readBank1.peek().litValue} row=${dut.io.out.readRow1.peek().litValue} d=${dut.io.out.readDone1.peek().litToBoolean}
                    |read2:  v=${dut.io.out.readValid2.peek().litToBoolean} bank=${dut.io.out.readBank2.peek().litValue} row=${dut.io.out.readRow2.peek().litValue} d=${dut.io.out.readDone2.peek().litToBoolean}
                    |write1: v=${dut.io.out.writeValid1.peek().litToBoolean} bank=${dut.io.out.writeBank1.peek().litValue} row=${dut.io.out.writeRow1.peek().litValue} d=${dut.io.out.writeDone1.peek().litToBoolean}
                    |write2: v=${dut.io.out.writeValid2.peek().litToBoolean} bank=${dut.io.out.writeBank2.peek().litValue} row=${dut.io.out.writeRow2.peek().litValue} d=${dut.io.out.writeDone2.peek().litToBoolean}
                    |done:   d1=${dut.io.out.done1.peek().litToBoolean} d2=${dut.io.out.done2.peek().litToBoolean} ready=${dut.io.out.VEReady.peek().litToBoolean}
                    |state:  ${dut.io.out.state.peek().litValue}
                    |""".stripMargin
                )
                dut.clock.step(1)
            }
        }
    }
}
