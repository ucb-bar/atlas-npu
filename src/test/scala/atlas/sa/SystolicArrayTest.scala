// ============================================================================
// SystolicArrayTest.scala — Unit tests for the systolic-array MXU.
//
// Exercises the SystolicArray datapath + MREG through a shared harness,
// covering zero/identity/negative/sparse inputs, weight & accumulator
// double-buffering, bias addition, and all-row correctness.
//
// RUN: (from sp26-atlas-acc) 
//    mill atlas.test.testOnly atlas.sa.SystolicArrayTest 
// ============================================================================

package atlas.sa

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common._
import atlas.mxu.{MxuOp, MxuCmd}
import atlas.mreg.MregFile

// ============================================================================
// Test harness — SA + MREG with backdoor read/write
// ============================================================================

class SystolicArrayUnitHarness(
    p:     SystolicArrayParams = SystolicArrayParams(),
    mregP: MregParams          = MregParams()
) extends Module {

  val io = IO(new Bundle {
    val cmd         = Flipped(Valid(new MxuCmd(mregP.mregIdBits)))
    val dataBusy    = Output(Bool())
    val computeBusy = Output(Bool())
    val testWrite   = Flipped(Valid(new MregWriteReq(mregP)))
    val testRead    = Flipped(Valid(new MregReadReq(mregP)))
    val testReadOut = Valid(UInt(mregP.mregRowBits.W))
  })

  val sa   = Module(new SystolicArrayTop(p, mregP))
  val mreg = Module(new MregFile(mregP))

  // ── Command interface ──
  sa.io.cmd      <> io.cmd
  io.dataBusy    := sa.io.dataBusy
  io.computeBusy := sa.io.computeBusy

  // ── SA ↔ MREG (MXU-0 ports) ──
  mreg.io.mxu0ReadReq0  <> sa.io.mregReadReq0
  mreg.io.mxu0ReadReq1  <> sa.io.mregReadReq1
  sa.io.mregReadResp0    := mreg.io.mxu0ReadResp0
  sa.io.mregReadResp1    := mreg.io.mxu0ReadResp1
  mreg.io.mxu0WriteReq0  <> sa.io.mregWriteReq0
  mreg.io.mxu0WriteReq1  <> sa.io.mregWriteReq1

  // ── Backdoor test ports (via LSU ports on MREG) ──
  mreg.io.lsuWriteReq <> io.testWrite
  mreg.io.lsuReadReq  <> io.testRead
  io.testReadOut       := mreg.io.lsuReadResp

  // ── Tie off unused MREG ports ──
  mreg.io.mxu1ReadReq0.valid  := false.B
  mreg.io.mxu1ReadReq0.bits   := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.mxu1ReadReq1.valid  := false.B
  mreg.io.mxu1ReadReq1.bits   := 0.U.asTypeOf(new MregReadReq(mregP))
  mreg.io.mxu1WriteReq0.valid := false.B
  mreg.io.mxu1WriteReq0.bits  := 0.U.asTypeOf(new MregWriteReq(mregP))
  mreg.io.mxu1WriteReq1.valid := false.B
  mreg.io.mxu1WriteReq1.bits  := 0.U.asTypeOf(new MregWriteReq(mregP))
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

class SystolicArrayTest extends AnyFlatSpec with Matchers {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed)         println("SystolicArrayTest=FAILED")
    else if (outcome.isSucceeded) println("SystolicArrayTest=PASSED")
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
  val BF16_32  = 0x4200  // 32.0
  val BF16_33  = 0x4204  // 33.0
  val BF16_N32 = 0xC200  // -32.0
  val BF16_4   = 0x4080  // 4.0
  val BF16_64  = 0x4280  // 64.0

  // ── MREG bank assignments ──
  val WGT_BANK    = 0
  val ACT_BANK    = 2
  val BIAS_BANK   = 4
  val OUT_BANK_LO = 6

  // ── Numeric helpers ──

  def packElems(elems: Seq[Int], elemWidth: Int): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
      acc | (BigInt(v & ((1 << elemWidth) - 1)) << (i * elemWidth))
    }

  /** Total latency for data to drain through the systolic array. */
  private def saLatency(p: SystolicArrayParams): Int = p.rows + p.cols - 1

  // ── Harness helpers ──

  def idle(dut: SystolicArrayUnitHarness): Unit = {
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

  def trfWriteRow(dut: SystolicArrayUnitHarness, bank: Int, row: Int, data: BigInt): Unit = {
    dut.io.testWrite.valid.poke(true.B)
    dut.io.testWrite.bits.mregId.poke(bank.U)
    dut.io.testWrite.bits.row.poke(row.U)
    dut.io.testWrite.bits.data.poke(data.U)
    dut.clock.step()
    dut.io.testWrite.valid.poke(false.B)
  }

  def trfReadRow(dut: SystolicArrayUnitHarness, bank: Int, row: Int): BigInt = {
    dut.io.testRead.valid.poke(true.B)
    dut.io.testRead.bits.mregId.poke(bank.U)
    dut.io.testRead.bits.row.poke(row.U)
    dut.clock.step()
    dut.io.testRead.valid.poke(false.B)
    dut.io.testReadOut.valid.expect(true.B)
    dut.io.testReadOut.bits.peek().litValue
  }

  def sendCmd(
    dut:        SystolicArrayUnitHarness,
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

  def waitDataIdle(dut: SystolicArrayUnitHarness, max: Int = 500): Unit = {
    var i = 0
    while (i < max && dut.io.dataBusy.peek().litToBoolean) { dut.clock.step(); i += 1 }
    require(i < max, "Data FSM timed out")
  }

  def waitComputeIdle(dut: SystolicArrayUnitHarness, max: Int = 500): Unit = {
    var i = 0
    while (i < max && dut.io.computeBusy.peek().litToBoolean) { dut.clock.step(); i += 1 }
    require(i < max, "Compute FSM timed out")
  }

  /** Fill an entire MREG bank with a uniform FP8 value. */
  def loadUniformTile(dut: SystolicArrayUnitHarness, p: SystolicArrayParams, bank: Int, value: Int): Unit = {
    val rowData = packElems(Seq.fill(p.rows)(value), 8)
    for (row <- 0 until p.accSize)
      trfWriteRow(dut, bank, row, rowData)
  }

  /** Read one row of BF16 results (lo + hi banks → 32 lanes). */
  def readBF16Results(dut: SystolicArrayUnitHarness, p: SystolicArrayParams, bankLo: Int, row: Int): Seq[Int] = {
    val lo = trfReadRow(dut, bankLo, row)
    val hi = trfReadRow(dut, bankLo + 1, row)
    (0 until 16).map(i => ((lo >> (i * 16)) & 0xFFFF).toInt) ++
    (0 until 16).map(i => ((hi >> (i * 16)) & 0xFFFF).toInt)
  }

  // ── Test suite ──

  def runAllChecks(p: SystolicArrayParams): Unit = {
    val mregP       = MregParams()
    val compTimeout = p.accSize + saLatency(p) + 200

    simulate(new SystolicArrayUnitHarness(p, mregP)) { dut =>
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
      waitComputeIdle(dut, compTimeout)
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
      waitComputeIdle(dut, compTimeout)
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
      waitComputeIdle(dut, compTimeout)
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
      waitComputeIdle(dut, compTimeout)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_N32)

      // ── Test 5: weight double-buffer ──
      println("  Test 5: weight double-buffer")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK, weightSlot = false)
      waitDataIdle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_2)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK, weightSlot = true)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)

      // Slot 0 → 32
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK, weightSlot = false)
      waitComputeIdle(dut, compTimeout)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_32

      // Slot 1 → 64
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK, weightSlot = true)
      waitComputeIdle(dut, compTimeout)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_64

      // ── Test 6: sparse → 2*2 = 4 ──
      println("  Test 6: sparse = 4")
      idle(dut)
      val sparseRow = packElems(Seq(E4M3_2) ++ Seq.fill(p.rows - 1)(E4M3_0), 8)
      for (row <- 0 until p.accSize) {
        trfWriteRow(dut, WGT_BANK, row, sparseRow)
        trfWriteRow(dut, ACT_BANK, row, sparseRow)
      }
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK)
      waitComputeIdle(dut, compTimeout)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_4)

      // ── Test 7: accumulator double-buffer ──
      println("  Test 7: acc double-buffer")
      idle(dut)

      // Acc 0: 1×1 dot = 32
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK, accSel = false)
      waitComputeIdle(dut, compTimeout)

      // Acc 1: 2×1 dot = 64
      loadUniformTile(dut, p, WGT_BANK, E4M3_2)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK, accSel = true)
      waitComputeIdle(dut, compTimeout)

      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO, accSel = false)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_32

      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO, accSel = true)
      waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_64

      // ── Test 8: all rows correct ──
      println("  Test 8: all rows correct")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.PushWeight, mregId = WGT_BANK)
      waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, MxuOp.Matmul, mregId = ACT_BANK)
      waitComputeIdle(dut, compTimeout)
      sendCmd(dut, MxuOp.PopAccBF16, mregId = OUT_BANK_LO)
      waitDataIdle(dut)
      for (row <- 0 until p.accSize) {
        readBF16Results(dut, p, OUT_BANK_LO, row).foreach { v =>
          assert(v == BF16_32, s"Row $row: expected 0x4200, got 0x${v.toHexString}")
        }
      }
    }
  }

  "SystolicArray (HardFloatFMA)" should "pass all checks" in {
    runAllChecks(SystolicArrayParams())
  }
}
