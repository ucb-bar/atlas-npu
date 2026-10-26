// ============================================================================
// LsuRowsTest.scala — vload and vstore at the row counts M = 1, 2, 3, 6, 7, 31, 32.
//
// vload moves M VMEM lines into rows 0..M-1 of m[vd], vstore rows 0..M-1 into M
// lines; the other rows and lines keep their values. Each path is busy for
// M + 2 cycles. vload.scale and vstore.scale move 4 lines whatever M is.
//
// RUN: mill atlas.test.testOnly atlas.lsu.LsuRowsTest   (Verilator)
// ============================================================================
package atlas.lsu

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common._
import atlas.mreg.{ScaleLineReq, ScaleLineWrite}
import atlas.scalar.LsuCmd
import atlas.scalar.ScalarISA.{LSU_VLOAD, LSU_VSTORE, LSU_VLOAD_SCALE, LSU_VSTORE_SCALE}
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

/** The LSU's vector paths with VMEM, MREG and scale ports that the test serves: a read returns its
  * line or row in the next cycle.
  */
class LsuPorts extends Module {
  private val p = MregParams()
  private val v = VmemParams()
  val io = IO(new Bundle {
    val cmd        = Flipped(Valid(new LsuCmd(v)))
    val vread      = Valid(new VmemLineReadPort(v))
    val vreadData  = Input(UInt(v.lineWidthBits.W))
    val vwrite     = Valid(new VmemLineWritePort(v))
    val mread      = Valid(new MregReadReq(p))
    val mreadData  = Input(UInt(p.mregRowBits.W))
    val mwrite     = Valid(new MregWriteReq(p))
    val scaleRead  = Valid(new ScaleLineReq(p))
    val scaleWrite = Valid(new ScaleLineWrite(p))
    val vloadBusy  = Output(Bool())
    val vstoreBusy = Output(Bool())
  })
  private val lsu = Module(new LSU(v, p)).io
  lsu.cmd                      := io.cmd
  lsu.scalarCmd.valid          := false.B
  lsu.scalarCmd.bits           := 0.U.asTypeOf(lsu.scalarCmd.bits)
  lsu.vmemScalarReadData.valid := false.B
  lsu.vmemScalarReadData.bits  := 0.U
  io.vread                     := lsu.vmemVecRead
  lsu.vmemVecReadData.valid    := RegNext(io.vread.valid, false.B)
  lsu.vmemVecReadData.bits     := RegNext(io.vreadData)
  io.vwrite                    := lsu.vmemVecWrite
  io.mread                     := lsu.mregReadReq
  lsu.mregReadResp.valid       := RegNext(io.mread.valid, false.B)
  lsu.mregReadResp.bits        := RegNext(io.mreadData)
  io.mwrite                    := lsu.mregWriteReq
  io.scaleRead                 := lsu.scaleLineRead
  lsu.scaleLineResp            := RegNext(io.scaleRead.bits.line)
  io.scaleWrite                := lsu.scaleWrite
  io.vloadBusy                 := lsu.vloadBusy
  io.vstoreBusy                := lsu.vstoreBusy
}

class LsuRowsTest extends AnyFlatSpec {
  private val counts = Seq(1, 2, 3, 6, 7, 31, 32)
  private val v      = VmemParams()

  /** VMEM lines by line address, the m-registers, and the lines and rows each transfer moved. */
  private class Mem(seed: Int) {
    private val rng = new Random(seed)
    val lines  = Array.fill(64)(BigInt(256, rng))
    val mreg   = Array.fill(4, 32)(BigInt(256, rng))
    var scales = 0
  }

  /** Issues one transfer in cycle 0, serves its reads and applies its writes until the path is idle.
    * Returns the cycles in which the path was busy.
    */
  private def run(dut: LsuPorts, mem: Mem, op: UInt, reg: Int, line: Int, rows: Int): Int = {
    dut.io.cmd.valid.poke(false.B)
    dut.reset.poke(true.B)
    dut.clock.step()
    dut.reset.poke(false.B)
    var busy  = 0
    var cycle = 0
    while (cycle == 0 || dut.io.vloadBusy.peek().litToBoolean || dut.io.vstoreBusy.peek().litToBoolean) {
      dut.io.cmd.valid.poke((cycle == 0).B)
      dut.io.cmd.bits.op.poke(op)
      dut.io.cmd.bits.mregBank.poke(reg.U)
      dut.io.cmd.bits.vmemLineAddr.poke(line.U)
      dut.io.cmd.bits.rows.poke((rows % 32).U)
      if (dut.io.vloadBusy.peek().litToBoolean || dut.io.vstoreBusy.peek().litToBoolean) busy += 1
      val r = dut.io.vread
      if (r.valid.peek().litToBoolean)
        dut.io.vreadData.poke(mem.lines(r.bits.bankIdx.peek().litValue.toInt * v.linesPerBank + r.bits.bankAddr.peek().litValue.toInt).U)
      val m = dut.io.mread
      if (m.valid.peek().litToBoolean)
        dut.io.mreadData.poke(mem.mreg(m.bits.mregId.peek().litValue.toInt)(m.bits.row.peek().litValue.toInt).U)
      val w = dut.io.vwrite
      if (w.valid.peek().litToBoolean)
        mem.lines(w.bits.bankIdx.peek().litValue.toInt * v.linesPerBank + w.bits.bankAddr.peek().litValue.toInt) =
          w.bits.data.peek().litValue
      val mw = dut.io.mwrite
      if (mw.valid.peek().litToBoolean)
        mem.mreg(mw.bits.mregId.peek().litValue.toInt)(mw.bits.row.peek().litValue.toInt) = mw.bits.data.peek().litValue
      if (dut.io.scaleWrite.valid.peek().litToBoolean || dut.io.scaleRead.valid.peek().litToBoolean) mem.scales += 1
      dut.clock.step()
      cycle += 1
      assert(cycle < 100, "the LSU did not finish")
    }
    busy
  }

  "The LSU" should "move rows 0..M-1 in M + 2 cycles, and 4 scale lines whatever M is" in {
    simulate(new LsuPorts) { dut =>
      for (m <- counts) {
        val load = new Mem(m)
        val before = load.mreg(1).clone
        assert(run(dut, load, LSU_VLOAD, 1, 32, m) == m + 2, s"vload at M $m")
        for (i <- 0 until 32) assert(load.mreg(1)(i) == (if (i < m) load.lines(32 + i) else before(i)), s"vload M $m row $i")

        val store = new Mem(m + 100)
        val lines = store.lines.clone
        assert(run(dut, store, LSU_VSTORE, 2, 0, m) == m + 2, s"vstore at M $m")
        for (i <- 0 until 64) assert(store.lines(i) == (if (i < m) store.mreg(2)(i) else lines(i)), s"vstore M $m line $i")

        for (op <- Seq(LSU_VLOAD_SCALE, LSU_VSTORE_SCALE)) {
          val scale = new Mem(m + 200)
          assert(run(dut, scale, op, 3, 4, m) == 6 && scale.scales == 4, s"scale transfer at M $m")
        }
      }
    }
  }
}
