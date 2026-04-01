package atlas.ipt

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common._

class InnerProductTreesUnitHarness(
  p: InnerProductTreeParams = InnerProductTreeParams(),
  rfP: RegFileParams = RegFileParams()
) extends Module {
  val io = IO(new Bundle {
    val cmd         = Flipped(Decoupled(new Mxu0Cmd(rfP)))
    val dataBusy    = Output(Bool())
    val computeBusy = Output(Bool())
    val dmaWriteIn  = Flipped(Valid(new RegFileWriteInput(rfP)))
    val dmaReadIn   = Flipped(Valid(new RegFileReadInput(rfP)))
    val dmaReadOut  = Output(Valid(UInt(rfP.SRAM_WIDTH.W)))
  })

  val seq = Module(new InnerProductTreesSequencer(p, rfP))
  val trf = Module(new TensorRegFile(rfP))

  seq.io.cmd      <> io.cmd
  io.dataBusy     := seq.io.dataBusy
  io.computeBusy  := seq.io.computeBusy

  trf.io.mxu0readinput0 <> seq.io.trfReadPort0In
  trf.io.mxu0readinput1 <> seq.io.trfReadPort1In
  seq.io.trfReadPort0Out := trf.io.mxu0readoutput0
  seq.io.trfReadPort1Out := trf.io.mxu0readoutput1
  trf.io.mxu0writeinput0 <> seq.io.trfWritePort0
  trf.io.mxu0writeinput1 <> seq.io.trfWritePort1

  // Tie off MXU1 write ports (both)
  trf.io.mxu1writeinput0.valid := false.B; trf.io.mxu1writeinput0.bits := 0.U.asTypeOf(new RegFileWriteInput(rfP))
  trf.io.mxu1writeinput1.valid := false.B; trf.io.mxu1writeinput1.bits := 0.U.asTypeOf(new RegFileWriteInput(rfP))

  trf.io.dmawriteinput <> io.dmaWriteIn
  trf.io.dmareadinput  <> io.dmaReadIn
  io.dmaReadOut        := trf.io.dmareadoutput

  // Tie off unused
  trf.io.mxu1readinput0.valid := false.B; trf.io.mxu1readinput0.bits := 0.U.asTypeOf(new RegFileReadInput(rfP))
  trf.io.mxu1readinput1.valid := false.B; trf.io.mxu1readinput1.bits := 0.U.asTypeOf(new RegFileReadInput(rfP))
  trf.io.vpureadinput0.valid  := false.B; trf.io.vpureadinput0.bits  := 0.U.asTypeOf(new RegFileReadInput(rfP))
  trf.io.vpureadinput1.valid  := false.B; trf.io.vpureadinput1.bits  := 0.U.asTypeOf(new RegFileReadInput(rfP))
  trf.io.xlureadinput.valid   := false.B; trf.io.xlureadinput.bits   := 0.U.asTypeOf(new RegFileReadInput(rfP))
  trf.io.vpuwriteinput0.valid := false.B; trf.io.vpuwriteinput0.bits := 0.U.asTypeOf(new RegFileWriteInput(rfP))
  trf.io.vpuwriteinput1.valid := false.B; trf.io.vpuwriteinput1.bits := 0.U.asTypeOf(new RegFileWriteInput(rfP))
  trf.io.xluwriteinput.valid  := false.B; trf.io.xluwriteinput.bits  := 0.U.asTypeOf(new RegFileWriteInput(rfP))
}

class InnerProductTreesTest extends AnyFlatSpec with Matchers {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) println("InnerProductTreesTest=FAILED")
    else if (outcome.isSucceeded) println("InnerProductTreesTest=PASSED")
    outcome
  }

  val E4M3_0 = 0x00; val E4M3_1 = 0x38; val E4M3_2 = 0x40; val E4M3_N1 = 0xB8
  val BF16_0 = 0x0000; val BF16_1 = 0x3F80; val BF16_32 = 0x4200; val BF16_33 = 0x4204
  val BF16_N32 = 0xC200; val BF16_4 = 0x4080; val BF16_64 = 0x4280
  val WGT_BANK = 0; val ACT_BANK = 2; val BIAS_BANK = 4; val OUT_BANK_LO = 6

  def packElems(elems: Seq[Int], elemWidth: Int): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
      acc | (BigInt(v & ((1 << elemWidth) - 1)) << (i * elemWidth)) }

  def idle(dut: InnerProductTreesUnitHarness): Unit = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.cmd.bits.op.poke(Mxu0Op.PushWeight)
    dut.io.cmd.bits.trfBank.poke(0.U); dut.io.cmd.bits.accSel.poke(false.B)
    dut.io.cmd.bits.weightSlot.poke(false.B); dut.io.cmd.bits.scaleE8M0.poke(127.U)
    dut.io.dmaWriteIn.valid.poke(false.B); dut.io.dmaWriteIn.bits.whichBank.poke(0.U)
    dut.io.dmaWriteIn.bits.wRow.poke(0.U); dut.io.dmaWriteIn.bits.wData.poke(0.U)
    dut.io.dmaReadIn.valid.poke(false.B); dut.io.dmaReadIn.bits.whichBank.poke(0.U)
    dut.io.dmaReadIn.bits.rRow.poke(0.U)
  }

  def trfWriteRow(dut: InnerProductTreesUnitHarness, bank: Int, row: Int, data: BigInt): Unit = {
    dut.io.dmaWriteIn.valid.poke(true.B); dut.io.dmaWriteIn.bits.whichBank.poke(bank.U)
    dut.io.dmaWriteIn.bits.wRow.poke(row.U); dut.io.dmaWriteIn.bits.wData.poke(data.U)
    dut.clock.step(); dut.io.dmaWriteIn.valid.poke(false.B)
  }

  def trfReadRow(dut: InnerProductTreesUnitHarness, bank: Int, row: Int): BigInt = {
    dut.io.dmaReadIn.valid.poke(true.B); dut.io.dmaReadIn.bits.whichBank.poke(bank.U)
    dut.io.dmaReadIn.bits.rRow.poke(row.U); dut.clock.step()
    dut.io.dmaReadIn.valid.poke(false.B)
    dut.io.dmaReadOut.valid.expect(true.B); dut.io.dmaReadOut.bits.peek().litValue
  }

  def sendCmd(dut: InnerProductTreesUnitHarness, op: Mxu0Op.Type,
              trfBank: Int = 0, accSel: Boolean = false,
              weightSlot: Boolean = false, scaleE8M0: Int = 127): Unit = {
    dut.io.cmd.valid.poke(true.B); dut.io.cmd.bits.op.poke(op)
    dut.io.cmd.bits.trfBank.poke(trfBank.U); dut.io.cmd.bits.accSel.poke(accSel.B)
    dut.io.cmd.bits.weightSlot.poke(weightSlot.B); dut.io.cmd.bits.scaleE8M0.poke(scaleE8M0.U)
    dut.clock.step(); dut.io.cmd.valid.poke(false.B)
  }

  def waitDataIdle(dut: InnerProductTreesUnitHarness, max: Int = 500): Unit = {
    var i = 0; while (i < max && dut.io.dataBusy.peek().litToBoolean) { dut.clock.step(); i += 1 }
    require(i < max, "Data FSM timed out") }
  def waitComputeIdle(dut: InnerProductTreesUnitHarness, max: Int = 500): Unit = {
    var i = 0; while (i < max && dut.io.computeBusy.peek().litToBoolean) { dut.clock.step(); i += 1 }
    require(i < max, "Compute FSM timed out") }

  def loadUniformTile(dut: InnerProductTreesUnitHarness, p: InnerProductTreeParams, bank: Int, value: Int): Unit = {
    val rowData = packElems(Seq.fill(p.vecLen)(value), 8)
    for (row <- 0 until p.tileRows) trfWriteRow(dut, bank, row, rowData)
  }

  def readBF16Results(dut: InnerProductTreesUnitHarness, p: InnerProductTreeParams, bankLo: Int, row: Int): Seq[Int] = {
    val lo = trfReadRow(dut, bankLo, row)
    val hi = trfReadRow(dut, bankLo + 1, row)
    (0 until 16).map(i => ((lo >> (i * 16)) & 0xFFFF).toInt) ++
    (0 until 16).map(i => ((hi >> (i * 16)) & 0xFFFF).toInt)
  }

  def runAllChecks(p: InnerProductTreeParams): Unit = {
    val rfP = RegFileParams()
    simulate(new InnerProductTreesUnitHarness(p, rfP)) { dut =>
      idle(dut); dut.reset.poke(true.B); dut.clock.step(5)
      dut.reset.poke(false.B); dut.clock.step(1); idle(dut)

      // 1: zero × zero = zero
      println("  Test 1: zero × zero = zero")
      loadUniformTile(dut, p, WGT_BANK, E4M3_0)
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK); waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_0)
      sendCmd(dut, Mxu0Op.Matmul, trfBank = ACT_BANK); waitComputeIdle(dut)
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO); waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_0)

      // 2: 1×1 = 32
      println("  Test 2: 1×1 dot = 32")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK); waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.Matmul, trfBank = ACT_BANK); waitComputeIdle(dut)
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO); waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_32)

      // 3: matmul + bias = 33
      println("  Test 3: matmul + bias = 33")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK); waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      loadUniformTile(dut, p, BIAS_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.PushAccFP8, trfBank = BIAS_BANK); waitDataIdle(dut)
      sendCmd(dut, Mxu0Op.MatmulAcc, trfBank = ACT_BANK); waitComputeIdle(dut)
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO); waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_33)

      // 4: negative = -32
      println("  Test 4: negative weights = -32")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_N1)
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK); waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.Matmul, trfBank = ACT_BANK); waitComputeIdle(dut)
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO); waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_N32)

      // 5: double-buffer weights
      println("  Test 5: weight double-buffer")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK, weightSlot = false); waitDataIdle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_2)
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK, weightSlot = true); waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.Matmul, trfBank = ACT_BANK, weightSlot = false); waitComputeIdle(dut)
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO); waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_32
      sendCmd(dut, Mxu0Op.Matmul, trfBank = ACT_BANK, weightSlot = true); waitComputeIdle(dut)
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO); waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_64

      // 6: sparse = 4
      println("  Test 6: sparse = 4")
      idle(dut)
      val sparseRow = packElems(Seq(E4M3_2) ++ Seq.fill(p.vecLen - 1)(E4M3_0), 8)
      for (row <- 0 until p.tileRows) { trfWriteRow(dut, WGT_BANK, row, sparseRow); trfWriteRow(dut, ACT_BANK, row, sparseRow) }
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK); waitDataIdle(dut)
      sendCmd(dut, Mxu0Op.Matmul, trfBank = ACT_BANK); waitComputeIdle(dut)
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO); waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).foreach(_ shouldBe BF16_4)

      // 7: acc double-buffer
      println("  Test 7: acc double-buffer")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK); waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.Matmul, trfBank = ACT_BANK, accSel = false); waitComputeIdle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_2)
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK); waitDataIdle(dut)
      sendCmd(dut, Mxu0Op.Matmul, trfBank = ACT_BANK, accSel = true); waitComputeIdle(dut)
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO, accSel = false); waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_32
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO, accSel = true); waitDataIdle(dut)
      readBF16Results(dut, p, OUT_BANK_LO, 0).head shouldBe BF16_64

      // 8: all rows
      println("  Test 8: all rows correct")
      idle(dut)
      loadUniformTile(dut, p, WGT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.PushWeight, trfBank = WGT_BANK); waitDataIdle(dut)
      loadUniformTile(dut, p, ACT_BANK, E4M3_1)
      sendCmd(dut, Mxu0Op.Matmul, trfBank = ACT_BANK); waitComputeIdle(dut)
      sendCmd(dut, Mxu0Op.PopAccBF16, trfBank = OUT_BANK_LO); waitDataIdle(dut)
      for (row <- 0 until p.tileRows) {
        readBF16Results(dut, p, OUT_BANK_LO, row).foreach(v =>
          assert(v == BF16_32, s"Row $row: expected 0x4200, got 0x${v.toHexString}"))
      }

      // 9: readiness
      println("  Test 9: readiness")
      idle(dut); dut.clock.step()
      dut.io.cmd.ready.expect(true.B)
      dut.io.dataBusy.expect(false.B); dut.io.computeBusy.expect(false.B)
    }
  }

  "InnerProductTrees (combinational)" should "pass all checks" in { runAllChecks(InnerProductTreeParams()) }
  "InnerProductTrees (2-stage)" should "pass all checks" in { runAllChecks(InnerProductTreeParams.withPipelineDepth(2)) }
  "InnerProductTrees (3-stage)" should "pass all checks" in { runAllChecks(InnerProductTreeParams.withPipelineDepth(3)) }
  "InnerProductTrees (4-stage)" should "pass all checks" in { runAllChecks(InnerProductTreeParams.withPipelineDepth(4)) }
}
