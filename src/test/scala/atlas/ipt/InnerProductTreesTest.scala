// ============================================================================
// InnerProductTreesTest.scala — Unit tests for the IPT MXU.
//
// Exercises the InnerProductTrees datapath + MREG through a shared harness,
// covering zero/identity/negative/sparse inputs, weight & accumulator
// double-buffering, bias addition, and all-row correctness.
//
// RUN: (from sp26-atlas-acc) 
//    mill atlas.test.testOnly atlas.ipt.InnerProductTreesTest 
// ============================================================================

package atlas.ipt

import chisel3._
import chisel3.util._
import chisel3.simulator._
import _root_.circt.stage.ChiselStage
import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common._
import atlas.mxu._
import java.nio.file.{Files, Path, Paths}
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import scala.util.Try
import java.io.PrintWriter
import atlas.mreg.MregFile

// ============================================================================
// VCS simulator — persistent workspace with coverage
// ============================================================================

class PersistentVcsBasicSimulator(testSuffix: String) extends Simulator[VcsBackend] with PeekPokeAPI {

  private val runDir: Path = {
    val rootDirStr = sys.env.getOrElse("MILL_WORKSPACE_ROOT", "/tmp")
    val baseDir = Paths.get(rootDirStr)
    val p = baseDir.resolve("tmp").resolve(s"InnerProductTreesTest_$testSuffix")
    Files.createDirectories(p)
    p.toAbsolutePath
  }

  override val backend: VcsBackend   = VcsBackend.initializeFromProcessEnvironment()
  override val tag: String           = s"InnerProductTreesTest_$testSuffix"
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
        coverageName      = Some(Backend.CoverageName("InnerProductTreesTest_coverage"))
      )
    )
  }
}

// ============================================================================
// Test harness — IPT + MREG with backdoor read/write
// ============================================================================

class InnerProductTreesUnitHarness(
    p:     InnerProductTreeParams = InnerProductTreeParams(),
    mregP: MregParams             = MregParams()
) extends Module {

  val io = IO(new Bundle {
    val cmd         = Flipped(Valid(new MxuCmd(mregP.mregIdBits)))
    val dataBusy    = Output(Bool())
    val computeBusy = Output(Bool())
    val testWrite   = Flipped(Valid(new MregWriteReq(mregP)))
    val testRead    = Flipped(Valid(new MregReadReq(mregP)))
    val testReadOut = Valid(UInt(mregP.mregRowBits.W))
  })

  val ipt  = Module(new InnerProductTreesTop(p, mregP))
  val mreg = Module(new MregFile(mregP))

  // ── Command interface ──
  ipt.io.cmd     <> io.cmd
  io.dataBusy    := ipt.io.dataBusy
  io.computeBusy := ipt.io.computeBusy

  // ── IPT ↔ MREG (MXU-1 ports) ──
  mreg.io.mxu1ReadReq0  <> ipt.io.mregReadReq0
  mreg.io.mxu1ReadReq1  <> ipt.io.mregReadReq1
  ipt.io.mregReadResp0  := mreg.io.mxu1ReadResp0
  ipt.io.mregReadResp1  := mreg.io.mxu1ReadResp1
  mreg.io.mxu1WriteReq0 <> ipt.io.mregWriteReq0
  mreg.io.mxu1WriteReq1 <> ipt.io.mregWriteReq1

  // ── Backdoor test ports (via LSU ports on MREG) ──
  mreg.io.lsuWriteReq <> io.testWrite
  mreg.io.lsuReadReq  <> io.testRead
  io.testReadOut       := mreg.io.lsuReadResp

  // ── Tie off unused MREG ports ──
  mreg.io.mxu0ReadReq0.valid  := false.B
  mreg.io.mxu0ReadReq0.bits   := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.mxu0ReadReq1.valid  := false.B
  mreg.io.mxu0ReadReq1.bits   := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.mxu0WriteReq0.valid := false.B
  mreg.io.mxu0WriteReq0.bits  := 0.U.asTypeOf(new MregWriteReq(mregP))
  mreg.io.mxu0WriteReq1.valid := false.B
  mreg.io.mxu0WriteReq1.bits  := 0.U.asTypeOf(new MregWriteReq(mregP))
  mreg.io.vpuReadReq0.valid   := false.B
  mreg.io.vpuReadReq0.bits    := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.vpuReadReq1.valid   := false.B
  mreg.io.vpuReadReq1.bits    := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.vpuWriteReq0.valid  := false.B
  mreg.io.vpuWriteReq0.bits   := 0.U.asTypeOf(new MregWriteReq(mregP))
  mreg.io.vpuWriteReq1.valid  := false.B
  mreg.io.vpuWriteReq1.bits   := 0.U.asTypeOf(new MregWriteReq(mregP))
  mreg.io.xluReadReq.valid    := false.B
  mreg.io.xluReadReq.bits     := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.xluWriteReq.valid   := false.B
  mreg.io.xluWriteReq.bits    := 0.U.asTypeOf(new MregWriteReq(mregP))
}

// ============================================================================
// Tests
// ============================================================================

class InnerProductTreesTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed)         println("InnerProductTreesTest=FAILED")
    else if (outcome.isSucceeded) println("InnerProductTreesTest=PASSED")
    outcome
  }

  // ── FP8 E4M3 constants ──
  val E4M3_0  = 0x00  // 0.0
  val E4M3_1  = 0x38  // 1.0
  val E4M3_2  = 0x40  // 2.0
  val E4M3_N1 = 0xB8  // -1.0

  // ── BF16 constants (expected outputs) ──
  val BF16_0   = 0x0000  // 0.0
  val BF16_1   = 0x3F80  // 1.0
  val BF16_32  = 0x4200  // 32.0  (1×1 dot over 32 lanes)
  val BF16_33  = 0x4204  // 33.0  (32 + 1 bias)
  val BF16_N32 = 0xC200  // -32.0
  val BF16_4   = 0x4080  // 4.0   (sparse: 2×2 = 4)
  val BF16_64  = 0x4280  // 64.0  (2×1 dot over 32 lanes)

  // ── MREG bank assignments ──
  val WGT_BANK  = 0
  val ACT_BANK  = 2
  val BIAS_BANK = 4
  val OUT_BANK_LO = 6

  // ── Numeric helpers ──

  /** Pack a sequence of small integers into a single BigInt, each `elemWidth` bits. */
  def packElems(elems: Seq[Int], elemWidth: Int): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
      acc | (BigInt(v & ((1 << elemWidth) - 1)) << (i * elemWidth))
    }

  def runElaborationSmoke(p: InnerProductTreeParams): Unit = {
    ChiselStage.emitCHIRRTL(new InnerProductTreesUnitHarness(p, MregParams())).nonEmpty shouldBe true
  }

  // ── Harness helpers ──

  /** Drive all command and backdoor ports to idle / zero. */
  def idle(dut: InnerProductTreesUnitHarness): Unit = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.cmd.bits.op.poke(MxuOp.PushWeight)
    dut.io.cmd.bits.mregId.poke(0.U)
    dut.io.cmd.bits.accSel.poke(false.B)
    dut.io.cmd.bits.weightSlot.poke(false.B)
    dut.io.cmd.bits.scaleE8M0.poke(127.U)

    dut.io.testWrite.valid.poke(false.B)
    dut.io.testWrite.bits.mregId.poke(0.U)
    dut.io.testWrite.bits.row.poke(0.U)
    dut.io.testWrite.bits.data.poke(0.U)

    dut.io.testRead.valid.poke(false.B)
    dut.io.testRead.bits.mregId.poke(0.U)
    dut.io.testRead.bits.row.poke(0.U)
  }

  /** Backdoor write one MREG row. */
  def trfWriteRow(dut: InnerProductTreesUnitHarness, bank: Int, row: Int, data: BigInt): Unit = {
    dut.io.testWrite.valid.poke(true.B)
    dut.io.testWrite.bits.mregId.poke(bank.U)
    dut.io.testWrite.bits.row.poke(row.U)
    dut.io.testWrite.bits.data.poke(data.U)
    dut.clock.step()
    dut.io.testWrite.valid.poke(false.B)
  }

  /** Backdoor read one MREG row. */
  def trfReadRow(dut: InnerProductTreesUnitHarness, bank: Int, row: Int): BigInt = {
    dut.io.testRead.valid.poke(true.B)
    dut.io.testRead.bits.mregId.poke(bank.U)
    dut.io.testRead.bits.row.poke(row.U)
    dut.clock.step()
    dut.io.testRead.valid.poke(false.B)
    dut.io.testReadOut.valid.expect(true.B)
    dut.io.testReadOut.bits.peek().litValue
  }

  /** Issue a single-cycle MXU command. */
  def sendCmd(
    dut:        InnerProductTreesUnitHarness,
    op:         MxuOp.Type,
    mregId:     Int     = 0,
    accSel:     Boolean = false,
    weightSlot: Boolean = false,
    scaleE8M0:  Int     = 127
  ): Unit = {
    dut.io.cmd.valid.poke(true.B)
    dut.io.cmd.bits.op.poke(op)
    dut.io.cmd.bits.mregId.poke(mregId.U)
    dut.io.cmd.bits.accSel.poke(accSel.B)
    dut.io.cmd.bits.weightSlot.poke(weightSlot.B)
    dut.io.cmd.bits.scaleE8M0.poke(scaleE8M0.U)
    dut.clock.step()
    dut.io.cmd.valid.poke(false.B)
  }

  /** Wait until dataBusy deasserts (or timeout). */
  def waitDataIdle(dut: InnerProductTreesUnitHarness, max: Int = 500): Unit = {
    var i = 0
    while (i < max && dut.io.dataBusy.peek().litToBoolean) { dut.clock.step(); i += 1 }
    require(i < max, "Data FSM timed out")
  }

  /** Wait until computeBusy deasserts (or timeout). */
  def waitComputeIdle(dut: InnerProductTreesUnitHarness, max: Int = 500): Unit = {
    var i = 0
    while (i < max && dut.io.computeBusy.peek().litToBoolean) { dut.clock.step(); i += 1 }
    require(i < max, "Compute FSM timed out")
  }

  /** Fill an entire MREG bank with a uniform FP8 value. */
  def loadUniformTile(dut: InnerProductTreesUnitHarness, p: InnerProductTreeParams, bank: Int, value: Int): Unit = {
    val rowData = packElems(Seq.fill(p.vecLen)(value), 8)
    for (row <- 0 until p.tileRows)
      trfWriteRow(dut, bank, row, rowData)
  }

  /**
   * Read one row of BF16 results from the output bank pair (lo, lo+1).
   * Returns a 32-element Seq of 16-bit BF16 values.
   */
  def readBF16Results(dut: InnerProductTreesUnitHarness, p: InnerProductTreeParams, bankLo: Int, row: Int): Seq[Int] = {
    val lo = trfReadRow(dut, bankLo, row)
    val hi = trfReadRow(dut, bankLo + 1, row)
    (0 until 16).map(i => ((lo >> (i * 16)) & 0xFFFF).toInt) ++
    (0 until 16).map(i => ((hi >> (i * 16)) & 0xFFFF).toInt)
  }

  // ── Test suite ──

  /**
   * Run the full battery of functional checks for a given pipeline
   * configuration.  Each test is self-contained: idle → load → compute → read.
   */
  def runAllChecks(p: InnerProductTreeParams, testName: String): Unit = {
    val mregP = MregParams()
    val simulator = new PersistentVcsBasicSimulator(testName)

    
    simulator.simulate(new InnerProductTreesUnitHarness(p, mregP)) { module =>
      val dut = module.wrapped
      idle(dut)
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(1)
      idle(dut)

      // ── Test 1: zero × zero = zero ──
      println("  Test 1: zero × zero = zero")
      loadUniformTile(dut, p, WGT_BANK, E4M3_0)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_0)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK)
      waitComputeIdle(dut)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_0)

      // ── Test 2: all-ones dot product = 32 ──
      println("  Test 2: 1×1 dot = 32")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK)
      waitComputeIdle(dut)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_32)

      // ── Test 3: matmul + FP8 bias = 33 ──
      println("  Test 3: matmul + bias = 33")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      loadUniformTile(dut, p, BIAS_BANK, E4M3_1)
      sendCmd(dut, MxuOp.PushAccFP8, mregId = BIAS_BANK)
      waitDataIdle(dut)
      sendCmd(dut, MxuOp.MatmulAcc, mregId = ACT_BANK)
      waitComputeIdle(dut)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_33)

      // ── Test 4: negative weights → -32 ──
      println("  Test 4: negative weights = -32")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_N1)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK)
      waitComputeIdle(dut)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_N32)

      // ── Test 5: weight double-buffer (slot 0 = 1.0, slot 1 = 2.0) ──
      println("  Test 5: weight double-buffer")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK, weightSlot = false)
      waitDataIdle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_2)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK, weightSlot = true)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)

      // Compute with slot 0 → expect 32
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK, weightSlot = false)
      waitComputeIdle(dut)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_32

      // Compute with slot 1 → expect 64
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK, weightSlot = true)
      waitComputeIdle(dut)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_64

      // ── Test 6: sparse vector (only element 0 is non-zero) → 2*2 = 4 ──
      println("  Test 6: sparse = 4")
      idle(dut)
      val sparseRow = packElems(Seq(E4M3_2) ++ Seq.fill(p.vecLen - 1)(E4M3_0), 8)
      for (row <- 0 until p.tileRows) {
        trfWriteRow(dut, WGT_BANK, row, sparseRow)
        trfWriteRow(dut, ACT_BANK, row, sparseRow)
      }
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK)
      waitComputeIdle(dut)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_4)

      // ── Test 7: accumulator double-buffer ──
      println("  Test 7: acc double-buffer")
      idle(dut)

      // Acc buffer 0: 1×1 dot = 32
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK, accSel = false)
      waitComputeIdle(dut)

      // Acc buffer 1: 2×1 dot = 64
      loadUniformTile(dut, p, WGT_BANK, E4M3_2)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK, accSel = true)
      waitComputeIdle(dut)

      // Read back acc 0 → 32
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO, accSel = false)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_32

      // Read back acc 1 → 64
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO, accSel = true)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_64

      // ── Test 8: verify all rows produce the correct result ──
      println("  Test 8: all rows correct")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK)
      waitComputeIdle(dut)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      for (row <- 0 until p.tileRows) {
        readBF16Results(dut, p, OUT_BANK_LO, row).foreach { v =>
          assert(v == BF16_32, s"Row $row: expected 0x4200, got 0x${v.toHexString}")
        }
      }
    }
  }

  // Run the suite for several pipeline depths.
  "InnerProductTrees (combinational)" should "pass all checks" in {
    runAllChecks(InnerProductTreeParams.withPipelineDepth(1), "comb")
  }
  "InnerProductTrees (2-stage)" should "pass all checks" in {
    runAllChecks(InnerProductTreeParams.withPipelineDepth(2), "2stage")
  }
  "InnerProductTrees (3-stage)" should "pass all checks" in {
    runAllChecks(InnerProductTreeParams.withPipelineDepth(3), "3stage")
  }
  "InnerProductTrees (4-stage)" should "pass all checks" in {
    runAllChecks(InnerProductTreeParams.withPipelineDepth(4), "4stage")
  }

  it should "elaborate rectangular geometries" in {
    runElaborationSmoke(InnerProductTreeParams(mxu = MxuParams(
      arrayRows = 32, arrayCols = 24, accumBufferRows = 8)))
    runElaborationSmoke(InnerProductTreeParams(mxu = MxuParams(
      arrayRows = 32, arrayCols = 8, accumBufferRows = 20)))
  }
}
