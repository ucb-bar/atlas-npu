// ============================================================================
// VectorEngineRawCsumSmokeTest.scala
//
// Single hand-written csum case against the raw VectorEngine (no MREG
// wrapper). Preserves a regression at the raw-engine IO after the old
// file-driven VectorEngineTestCsum.scala was retired in favor of the
// family-driven VectorEngineTopColReduceVectorTest (which exercises
// VectorEngineTop + MREG instead).
//
// Input : 32 rows, lane 0 = 1.0 (0x3F80), lanes 1..15 = 0. Other lanes
//         are irrelevant because csum reduces over column 0.
// Expect : sum over col 0 = 32 * 1.0 = 32.0 (0x4200) in the first output
//          row's lane 0.
//
// RUN:
//    mill atlas.test.testOnly atlas.vector.VectorEngineRawCsumSmokeTest
// ============================================================================
package atlas.vector

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common.VpuParams

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import atlas.common._
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsVectorEngineRawCsumSmokeSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "VectorEngineRawCsumSmokeTest"

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

class VectorEngineRawCsumSmokeTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed)         println("VectorEngineRawCsumSmokeTest=FAILED")
    else if (outcome.isSucceeded) println("VectorEngineRawCsumSmokeTest=PASSED")
    outcome
  }

  behavior of "raw VectorEngine"

  it should "column-sum 32 copies of 1.0 to 32.0 (smoke)" in {
    val p         = VpuParams()
    val numRows   = 32
    val bitsPerLn = 16
    val bitMask   = 0xFFFF

    // lane 0 = 0x3F80 (1.0), lanes 1..15 = 0
    val oneBits  = 0x3F80
    val rowsA: Seq[BigInt] = Seq.fill(numRows) {
      val lanes = Seq(oneBits) ++ Seq.fill(p.numLanes - 1)(0)
      lanes.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
        acc | (BigInt(v & bitMask) << (i * bitsPerLn))
      }
    }

    PersistentVcsVectorEngineRawCsumSmokeSimulator.simulate(new VectorEngine(p)) { module =>
      val dut = module.wrapped
      // ── reset ──
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      // ── idle ──
      dut.io.inst.valid.poke(false.B)
      dut.io.inst.bits.instType.poke(VPUOp.add)
      dut.io.inst.bits.instReadBank1.poke(0.U)
      dut.io.inst.bits.instReadBank2.poke(0.U)
      dut.io.inst.bits.instWriteBank.poke(0.U)
      dut.io.inst.bits.packScaleE8M0.poke(127.U)
      dut.io.inst.bits.unpackScaleE8M0.poke(127.U)
      dut.io.data1.valid.poke(false.B)
      dut.io.data2.valid.poke(false.B)
      dut.io.data1.bits.data.poke(0.U)
      dut.io.data2.bits.data.poke(0.U)

      // ── dispatch csum ──
      dut.io.inst.bits.instType.poke(VPUOp.csum)
      dut.io.inst.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.inst.valid.poke(false.B)

      // ── feed rowsA as read1 requests fire; collect write1 outputs ──
      val actualOutputs = scala.collection.mutable.ArrayBuffer[BigInt]()
      var pendingRead1 = false
      var readRow1Idx  = 0
      var cycles       = 0
      val timeoutLimit = 1000
      val expectedNum  = 32

      while (actualOutputs.length < expectedNum && cycles < timeoutLimit) {
        if (dut.io.write1.valid.peek().litToBoolean) {
          actualOutputs += dut.io.write1.bits.data.peek().litValue
        }

        dut.io.data1.valid.poke(pendingRead1.B)
        if (pendingRead1) {
          val outData = if (readRow1Idx < rowsA.length) rowsA(readRow1Idx) else BigInt(0)
          dut.io.data1.bits.data.poke(outData.U)
          readRow1Idx += 1
        } else {
          dut.io.data1.bits.data.poke(0.U)
        }

        dut.io.data2.valid.poke(false.B)
        dut.io.data2.bits.data.poke(0.U)

        pendingRead1 = dut.io.read1.valid.peek().litToBoolean

        dut.clock.step(1)
        cycles += 1
      }

      dut.io.data1.valid.poke(false.B)
      dut.io.data2.valid.poke(false.B)

      assert(
        actualOutputs.length == expectedNum,
        s"csum timed out: got ${actualOutputs.length} output rows, expected $expectedNum"
      )

      // Row 0 lane 0 holds the sum for column 0.
      val row0Lane0 = (actualOutputs.head & BigInt(bitMask)).toInt
      val expected  = 0x4200 // 32.0 in BF16

      row0Lane0 shouldBe expected
    }
  }
}
