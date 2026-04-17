// ============================================================================
// VectorEngineTest.scala — Unit tests for the Vector Processing Unit.
//
// Exercises the VectorEngineTop + MREG through a shared harness,
// covering basic vector arithmetic (add, sub) through the MREG file.
//
// mill atlas.test.testOnly atlas.vector.VectorEngineTopTest
//
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator._
import _root_.circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common._
import atlas.mreg.MregFile
import sp26FPUnits._
import atlas.vector._
import atlas.scalar.VpuCmd

import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import java.nio.file.{Files, Path, Paths}

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsVectorEngineTopSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val test_name = "VectorEngineTopTest"

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

// ============================================================================
// Test harness — VPU + MREG with backdoor read/write
// ============================================================================

class VectorEngineTopUnitHarness(
    p:     VpuParams  = VpuParams(),
    mregP: MregParams = MregParams()
) extends Module {

  val io = IO(new Bundle {
    val cmd         = Flipped(Valid(new VpuCmd()))
    val busy        = Output(Bool())
    
    // Backdoor MREG test ports
    val testWrite   = Flipped(Valid(new MregWriteReq(mregP)))
    val testRead    = Flipped(Valid(new MregReadReq(mregP)))
    val testReadOut = Valid(UInt(mregP.mregRowBits.W))
  })

  val vpu  = Module(new VectorEngineTop(p, mregP))
  val mreg = Module(new MregFile(mregP))

  // ── Command interface ──
  vpu.io.cmd <> io.cmd
  io.busy    := vpu.io.busy

  // ── VPU ↔ MREG (VPU ports) ──
  mreg.io.vpuReadReq0  <> vpu.io.mregReadReq0
  mreg.io.vpuReadReq1  <> vpu.io.mregReadReq1
  vpu.io.mregReadResp0 := mreg.io.vpuReadResp0
  vpu.io.mregReadResp1 := mreg.io.vpuReadResp1
  mreg.io.vpuWriteReq0 <> vpu.io.mregWriteReq0
  mreg.io.vpuWriteReq1 <> vpu.io.mregWriteReq1

  // ── Backdoor test ports (via LSU ports on MREG) ──
  mreg.io.lsuWriteReq <> io.testWrite
  mreg.io.lsuReadReq  <> io.testRead
  io.testReadOut      := mreg.io.lsuReadResp

  // ── Tie off unused MREG ports (MXU, XLU) ──
  mreg.io.mxu0ReadReq0.valid  := false.B
  mreg.io.mxu0ReadReq0.bits   := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.mxu0ReadReq1.valid  := false.B
  mreg.io.mxu0ReadReq1.bits   := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.mxu0WriteReq0.valid := false.B
  mreg.io.mxu0WriteReq0.bits  := 0.U.asTypeOf(new MregWriteReq(mregP))
  mreg.io.mxu0WriteReq1.valid := false.B
  mreg.io.mxu0WriteReq1.bits  := 0.U.asTypeOf(new MregWriteReq(mregP))

  mreg.io.mxu1ReadReq0.valid  := false.B
  mreg.io.mxu1ReadReq0.bits   := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.mxu1ReadReq1.valid  := false.B
  mreg.io.mxu1ReadReq1.bits   := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.mxu1WriteReq0.valid := false.B
  mreg.io.mxu1WriteReq0.bits  := 0.U.asTypeOf(new MregWriteReq(mregP))
  mreg.io.mxu1WriteReq1.valid := false.B
  mreg.io.mxu1WriteReq1.bits  := 0.U.asTypeOf(new MregWriteReq(mregP))

  mreg.io.xluReadReq.valid    := false.B
  mreg.io.xluReadReq.bits     := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.xluWriteReq.valid   := false.B
  mreg.io.xluWriteReq.bits    := 0.U.asTypeOf(new MregWriteReq(mregP))
}

// ============================================================================
// Tests
// ============================================================================

class VectorEngineTopTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed)         println("VectorEngineTopTest=FAILED")
    else if (outcome.isSucceeded) println("VectorEngineTopTest=PASSED")
    outcome
  }

  // ── BF16 constants
  val BF16_0   = 0x0000  // 0.0
  val BF16_1   = 0x3F80  // 1.0
  val BF16_2   = 0x4000  // 2.0
  val BF16_3   = 0x4040  // 3.0
  val BF16_4   = 0x4080  // 4.0
  val BF16_48  = 0x4240  // 48.0

  // ── MREG bank assignments ──
  val SRC1_BANK = 0
  val SRC2_BANK = 1
  val DEST1_BANK = 2
  val DEST2_BANK = 3

  // ── Numeric helpers ──
  
  /** Pack a sequence of elements into a single BigInt, each `elemWidth` bits. */
  def packElems(elems: Seq[Int], elemWidth: Int): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
      acc | (BigInt(v & ((1 << elemWidth) - 1)) << (i * elemWidth))
    }

  def runElaborationSmoke(p: VpuParams): Unit = {
    ChiselStage.emitCHIRRTL(new VectorEngineTopUnitHarness(p, MregParams())).nonEmpty shouldBe true
  }

  // ── Harness helpers ──

  def idle(dut: VectorEngineTopUnitHarness): Unit = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.cmd.bits.op.poke(0.U)
    dut.io.cmd.bits.vs1.poke(0.U)
    dut.io.cmd.bits.vs2.poke(0.U)
    dut.io.cmd.bits.vd.poke(0.U)
    dut.io.cmd.bits.scaleE8M0.poke(127.U)
    dut.io.cmd.bits.imm.poke(0.U)

    dut.io.testWrite.valid.poke(false.B)
    dut.io.testWrite.bits.mregId.poke(0.U)
    dut.io.testWrite.bits.row.poke(0.U)
    dut.io.testWrite.bits.data.poke(0.U)

    dut.io.testRead.valid.poke(false.B)
    dut.io.testRead.bits.mregId.poke(0.U)
    dut.io.testRead.bits.row.poke(0.U)
  }

  def trfWriteRow(dut: VectorEngineTopUnitHarness, bank: Int, row: Int, data: BigInt): Unit = {
    dut.io.testWrite.valid.poke(true.B)
    dut.io.testWrite.bits.mregId.poke(bank.U)
    dut.io.testWrite.bits.row.poke(row.U)
    dut.io.testWrite.bits.data.poke(data.U)
    dut.clock.step()
    dut.io.testWrite.valid.poke(false.B)
  }

  def trfReadRow(dut: VectorEngineTopUnitHarness, bank: Int, row: Int): BigInt = {
    dut.io.testRead.valid.poke(true.B)
    dut.io.testRead.bits.mregId.poke(bank.U)
    dut.io.testRead.bits.row.poke(row.U)
    dut.clock.step()
    dut.io.testRead.valid.poke(false.B)
    dut.io.testReadOut.valid.expect(true.B)
    dut.io.testReadOut.bits.peek().litValue
  }

  /** Issue a VPU command via the VpuCmd interface.
   *  VectorEngineTop handles the VpuCmd → VectorInput translation internally.
   */
  def sendVpuCmd(
    dut:      VectorEngineTopUnitHarness,
    op:       VPUOp.Type, 
    src1Bank: Int,
    src2Bank: Int,
    destBank: Int,
    rows:     Int = 1
  ): Unit = {
    dut.io.cmd.valid.poke(true.B)
    dut.io.cmd.bits.op.poke((op.litValue + 1).U)
    dut.io.cmd.bits.vs1.poke(src1Bank.U)
    dut.io.cmd.bits.vs2.poke(src2Bank.U)
    dut.io.cmd.bits.vd.poke(destBank.U)
    dut.io.cmd.bits.scaleE8M0.poke(127.U)
    dut.io.cmd.bits.imm.poke(0.U)

    dut.clock.step()
    dut.io.cmd.valid.poke(false.B)
  }

  def waitVpuIdle(dut: VectorEngineTopUnitHarness, max: Int = 500): Unit = {
    var i = 0
    while (i < max && dut.io.busy.peek().litToBoolean) { dut.clock.step(); i += 1 }
    require(i < max, "VPU execution timed out")
  }

  def loadUniformVector(dut: VectorEngineTopUnitHarness, p: VpuParams, bank: Int, row: Int, value: Int): Unit = {
    val rowData = packElems(Seq.fill(p.numLanes)(value), 16) // Assumes BF16 (16-bit) lanes
    trfWriteRow(dut, bank, row, rowData)
  }

  def readBF16Results(dut: VectorEngineTopUnitHarness, p: VpuParams, bank: Int, row: Int): Seq[Int] = {
    val resultStr = trfReadRow(dut, bank, row)
    (0 until p.numLanes).map(i => ((resultStr >> (i * 16)) & 0xFFFF).toInt)
  }

  // ── Test suite ──

  def runAllChecks(p: VpuParams): Unit = {
    val mregP = MregParams()

    PersistentVcsVectorEngineTopSimulator.simulate(new VectorEngineTopUnitHarness(p, mregP)) { module =>
      val dut = module.wrapped
      idle(dut)
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)
      idle(dut)

      // ── Test 1: VPU RMax across a BF16 register pair ──
      println("  Test 1: Vector Rmax across bank pair")
      loadUniformVector(dut, p, SRC1_BANK, row = 0, BF16_1)
      loadUniformVector(dut, p, SRC1_BANK + 1, row = 0, BF16_2)

      sendVpuCmd(dut, op = VPUOp.rmax, src1Bank = SRC1_BANK, src2Bank = SRC2_BANK, destBank = DEST1_BANK)

      waitVpuIdle(dut)
      dut.clock.step(2)

      readBF16Results(dut, p, DEST1_BANK, row = 0).foreach(_ shouldBe BF16_2)
      readBF16Results(dut, p, DEST1_BANK + 1, row = 0).foreach(_ shouldBe BF16_2)

      // ── Test 2: VPU Rmin across a BF16 register pair ──
      println("  Test 2: Vector Rmin across bank pair")
      idle(dut)
      loadUniformVector(dut, p, SRC1_BANK, row = 0, BF16_3)
      loadUniformVector(dut, p, SRC1_BANK + 1, row = 0, BF16_2)

      sendVpuCmd(dut, op = VPUOp.rmin, src1Bank = SRC1_BANK, src2Bank = SRC2_BANK, destBank = DEST1_BANK)

      waitVpuIdle(dut)
      dut.clock.step(2)

      readBF16Results(dut, p, DEST1_BANK, row = 0).foreach(_ shouldBe BF16_2)
      readBF16Results(dut, p, DEST1_BANK + 1, row = 0).foreach(_ shouldBe BF16_2)

      // ── Test 3: VPU RSum across a BF16 register pair ──
      println("  Test 3: Vector Rsum across bank pair")
      idle(dut)
      loadUniformVector(dut, p, SRC1_BANK, row = 0, BF16_1)
      loadUniformVector(dut, p, SRC1_BANK + 1, row = 0, BF16_2)

      sendVpuCmd(dut, op = VPUOp.rsum, src1Bank = SRC1_BANK, src2Bank = SRC2_BANK, destBank = DEST1_BANK)

      waitVpuIdle(dut, max = 800)
      dut.clock.step(2)

      readBF16Results(dut, p, DEST1_BANK, row = 0).foreach(_ shouldBe BF16_48)
      readBF16Results(dut, p, DEST1_BANK + 1, row = 0).foreach(_ shouldBe BF16_48)

      def runAliasedBinaryOp(
        label: String,
        op: VPUOp.Type,
        expectedLo: Int,
        expectedHi: Int
      ): Unit = {
        println(s"  $label")
        idle(dut)
        loadUniformVector(dut, p, SRC1_BANK, row = 0, BF16_1)
        loadUniformVector(dut, p, SRC1_BANK + 1, row = 0, BF16_2)

        sendVpuCmd(dut, op = op, src1Bank = SRC1_BANK, src2Bank = SRC1_BANK, destBank = DEST1_BANK)

        waitVpuIdle(dut)
        dut.clock.step(2)

        readBF16Results(dut, p, DEST1_BANK, row = 0).foreach(_ shouldBe expectedLo)
        readBF16Results(dut, p, DEST1_BANK + 1, row = 0).foreach(_ shouldBe expectedHi)
      }

      runAliasedBinaryOp("Test 4: Vector add with aliased source pair", VPUOp.add, BF16_2, BF16_4)
      runAliasedBinaryOp("Test 5: Vector sub with aliased source pair", VPUOp.sub, BF16_0, BF16_0)
      runAliasedBinaryOp("Test 6: Vector mul with aliased source pair", VPUOp.mul, BF16_1, BF16_4)
      runAliasedBinaryOp("Test 7: Vector min with aliased source pair", VPUOp.pairmin, BF16_1, BF16_2)
      runAliasedBinaryOp("Test 8: Vector max with aliased source pair", VPUOp.pairmax, BF16_1, BF16_2)
    }
  }

  "VectorEngineTop" should "pass all functional checks" in {
    // Instantiate with your actual default params here
    runAllChecks(VpuParams()) 
  }

  it should "elaborate" in {
    runElaborationSmoke(VpuParams())
  }
}
