// ============================================================================
// Nvfp4RowsTest.scala — the NVFP4 MXU at the row counts M = 1, 2, 3, 6, 7, 31, 32.
//
// As Mxu1RowsTest: a command of M rows moves rows 0..M-1 of its m-registers and
// accumulator, and the other rows keep their values (a weight push moves all
// rows). Each scenario runs from the same MREG and scales with every command at
// 32 rows and with the commands under test at M. A matmul on an accumulator
// issues only after the previous one's row 0 lands, 8 cycles after that one
// issues: matmuls on one accumulator issue every max(M, 8) cycles.
//
// RUN: mill atlas.test.testOnly atlas.nvfp4.Nvfp4RowsTest   (Verilator)
// ============================================================================
package atlas.nvfp4

import chisel3._
import chisel3.util._
import chisel3.simulator.{PeekPokeAPI, SimulatorAPI}
import chisel3.testing.HasTestingDirectory
import atlas.common.{MregParams, MregReadReq, MregWriteReq, Nvfp4MxuParams}
import org.scalatest.flatspec.AnyFlatSpec
import svsim.{Backend, BackendSettingsModifications}
import scala.util.Random

/** Nvfp4Mxu with MREG and scale ports that the test serves: a read returns its row in the next cycle,
  * as MregFile and MregScaleFile do.
  */
class Nvfp4Ports extends Module {
  private val p = MregParams()
  val io = IO(new Bundle {
    val cmd        = Flipped(Valid(new NvCmd(p.mregIdBits)))
    val cmdReady   = Output(Bool())
    val readData   = Input(Vec(2, UInt(p.mregRowBits.W)))
    val scaleData  = Input(Vec(2, UInt(32.W)))
    val reads      = Vec(2, Valid(new MregReadReq(p)))
    val scaleReads = Vec(2, Valid(new MregReadReq(p)))
    val writes     = Vec(2, Valid(new MregWriteReq(p)))
  })
  private val u = Module(new Nvfp4Mxu(Nvfp4MxuParams(), p)).io
  u.cmd       := io.cmd
  io.cmdReady := u.cmdReady
  for ((req, resp, write, i) <- Seq((u.mregReadReq0, u.mregReadResp0, u.mregWriteReq0, 0),
                                    (u.mregReadReq1, u.mregReadResp1, u.mregWriteReq1, 1))) {
    io.reads(i)      := req
    io.writes(i)     := write
    resp.valid       := RegNext(req.valid, false.B)
    resp.bits        := RegNext(io.readData(i))
    io.scaleReads(i) := u.scaleReadReq(i)
    u.scaleReadResp(i) := RegNext(io.scaleData(i))
  }
}

class Nvfp4RowsTest extends AnyFlatSpec with SimulatorAPI with PeekPokeAPI {
  import NvOp._
  // Verilator warns about the fp4_fedp sources (ucb-bar/nvfp4); standalone/sim/build.sh waives the
  // warnings too.
  private implicit val waiveWarnings: BackendSettingsModifications = new BackendSettingsModifications {
    def apply(s: Backend.Settings): Backend.Settings = s match {
      case v: svsim.verilator.Backend.CompilationSettings => v.copy(disableFatalExitOnWarnings = true)
      case other => other
    }
  }
  private implicit val directory: HasTestingDirectory = HasTestingDirectory.temporary()
  private val counts = Seq(1, 2, 3, 6, 7, 31, 32)

  /** A command; `m` says whether it runs at the row count under test (else at 32 rows). */
  private case class Cmd(op: NvOp.Type, reg: Int, acc: Int, m: Boolean, w: Int = 0, e: Int = 127)

  private type Mreg = Array[Array[BigInt]]

  /** Issues each command as soon as the unit takes it, serves the MREG and scale reads and applies the
    * writes. Returns the issue cycles.
    */
  private def run(dut: Nvfp4Ports, mreg: Mreg, scales: Array[Array[Long]], cmds: Seq[Cmd], rows: Int): Seq[Int] = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.readData.foreach(_.poke(0.U))
    dut.io.scaleData.foreach(_.poke(0.U))
    dut.reset.poke(true.B)
    dut.clock.step()
    dut.reset.poke(false.B)
    var queue  = cmds.toList
    val issued = Seq.newBuilder[Int]
    var cycle  = 0
    var quiet  = 0
    while (quiet < 80) {
      queue.headOption.foreach { c =>
        dut.io.cmd.bits.op.poke(c.op)
        dut.io.cmd.bits.mregId.poke(c.reg.U)
        dut.io.cmd.bits.accSel.poke((c.acc == 1).B)
        dut.io.cmd.bits.weightSlot.poke((c.w == 1).B)
        dut.io.cmd.bits.scaleE8M0.poke(c.e.U)
        dut.io.cmd.bits.rows.poke((if (c.m) rows % 32 else 0).U)
      }
      val issue = queue.nonEmpty && dut.io.cmdReady.peek().litToBoolean
      dut.io.cmd.valid.poke(issue.B)
      if (issue) { issued += cycle; queue = queue.tail }
      for (i <- 0 until 2) {
        val r = dut.io.reads(i)
        if (r.valid.peek().litToBoolean)
          dut.io.readData(i).poke(mreg(r.bits.mregId.peek().litValue.toInt)(r.bits.row.peek().litValue.toInt).U)
        val s = dut.io.scaleReads(i)
        if (s.valid.peek().litToBoolean)
          dut.io.scaleData(i).poke(scales(s.bits.mregId.peek().litValue.toInt)(s.bits.row.peek().litValue.toInt).U)
      }
      for (i <- 0 until 2) {
        val w = dut.io.writes(i)
        if (w.valid.peek().litToBoolean)
          mreg(w.bits.mregId.peek().litValue.toInt)(w.bits.row.peek().litValue.toInt) = w.bits.data.peek().litValue
      }
      dut.clock.step()
      cycle += 1
      if (queue.isEmpty) quiet += 1
    }
    issued.result()
  }

  /** Runs `cmds` at 32 rows and at each M from random MREG contents and scales. Rows 0..M-1 of `out`
    * must equal the 32-row run; rows M..31 must equal `old(reg, row)` of the MREG before the run.
    * Returns the issue cycles of each M.
    */
  private def check(dut: Nvfp4Ports, seed: Int, cmds: Seq[Cmd], out: Seq[Int])(old: (Mreg, Int, Int) => BigInt): Map[Int, Seq[Int]] = {
    val rng    = new Random(seed)
    val start  = Array.fill(64, 32)(BigInt(256, rng))
    val scales = Array.fill(64, 32)(rng.nextLong() & 0xffffffffL)
    val full   = start.map(_.clone)
    run(dut, full, scales, cmds, 32)
    counts.map { m =>
      val mreg  = start.map(_.clone)
      val times = run(dut, mreg, scales, cmds, m)
      for (reg <- out; row <- 0 until 32) {
        val want = if (row < m) full(reg)(row) else old(start, reg, row)
        assert(mreg(reg)(row) == want, s"M $m: m$reg row $row")
      }
      m -> times
    }.toMap
  }

  private def spacing(times: Seq[Int]): Seq[Int] = times.zip(times.tail).map { case (a, b) => b - a }

  "The NVFP4 MXU" should "feed M rows into rows 0..M-1 of the accumulator, every max(M, 8) cycles on one accumulator" in {
    simulate(new Nvfp4Ports) { dut =>
      val cmds = Seq(Cmd(PushAccFP32, 4, 0, m = false), Cmd(PushWeight, 8, 0, m = true),
        Cmd(Matmul, 9, 0, m = true), Cmd(MatmulAcc, 10, 0, m = true), Cmd(MatmulAcc, 11, 0, m = true),
        Cmd(PopFP32, 16, 0, m = false))
      val times = check(dut, 1, cmds, 16 until 20)((start, reg, row) => start(reg - 12)(row))
      for ((m, t) <- times) assert(spacing(t.slice(2, 5)) == Seq.fill(2)(m.max(8)), s"M $m: matmuls issue at $t")
    }
  }

  it should "issue matmuls on alternating accumulators every M cycles" in {
    simulate(new Nvfp4Ports) { dut =>
      val cmds = Seq(Cmd(PushAccFP32, 4, 0, m = false), Cmd(PushAccFP32, 12, 1, m = false),
        Cmd(PushWeight, 8, 1, m = false, w = 1),
        Cmd(Matmul, 9, 0, m = true, w = 1), Cmd(Matmul, 10, 1, m = true, w = 1),
        Cmd(MatmulAcc, 11, 0, m = true, w = 1), Cmd(MatmulAcc, 9, 1, m = true, w = 1),
        Cmd(PopFP32, 16, 0, m = false), Cmd(PopFP32, 20, 1, m = false))
      val times = check(dut, 2, cmds, 16 until 24)((start, reg, row) => start(if (reg < 20) reg - 12 else reg - 8)(row))
      for ((m, t) <- times) {
        val Seq(t0, t1, t2, t3) = t.slice(3, 7)
        assert(t1 - t0 == m && t2 == (t1 + m).max(t0 + 8) && t3 == (t2 + m).max(t1 + 8), s"M $m: matmuls issue at $t")
      }
    }
  }

  it should "pop rows 0..M-1 of the accumulator into rows 0..M-1 of the m-registers" in {
    simulate(new Nvfp4Ports) { dut =>
      // acc0 = a 32-row matmul; the three pops at M; then a matmul on acc0 waits for the FP32 pop.
      val cmds = Seq(Cmd(PushWeight, 8, 0, m = false), Cmd(Matmul, 9, 0, m = false),
        Cmd(PopBF16, 24, 0, m = true), Cmd(PopBF16Scaled, 28, 0, m = true, e = 131),
        Cmd(PopFP32, 32, 0, m = true), Cmd(Matmul, 10, 0, m = true))
      val times = check(dut, 3, cmds, Seq(24, 25, 28, 29) ++ (32 to 35))((start, reg, row) => start(reg)(row))
      for ((m, t) <- times) assert(spacing(t.drop(2)) == Seq(m, m, 2 * m), s"M $m: pops issue at $t")
    }
  }

  it should "push rows 0..M-1 of the m-registers into rows 0..M-1 of the accumulator" in {
    simulate(new Nvfp4Ports) { dut =>
      // Each push at M lands on the FP32 tile m4..m7 and is popped with all 32 rows; the pop waits
      // until the push is done (M or 2M cycles) and has written row 0 for the last time (T+1, or
      // T+M+1 for an FP32 push): it never reads a row in the cycle that the push writes it.
      val cmds = Seq(
        Cmd(PushAccFP32, 4, 1, m = false), Cmd(PushAccBF16, 36, 1, m = true), Cmd(PopFP32, 40, 1, m = false),
        Cmd(PushAccFP32, 4, 0, m = false), Cmd(PushAccFP32, 48, 0, m = true), Cmd(PopFP32, 52, 0, m = false))
      val times = check(dut, 4, cmds, (40 until 44) ++ (52 until 56))((start, reg, row) => start(4 + reg % 4)(row))
      for ((m, t) <- times) {
        val gaps = spacing(t)
        assert(Seq(gaps(1), gaps(4)) == Seq(m.max(2), (2 * m).max(m + 2)), s"M $m: pushes and pops issue at $t")
      }
    }
  }

  it should "issue vmatmul.acc after an accumulator push only once the push has written row 0" in {
    simulate(new Nvfp4Ports) { dut =>
      // A BF16 and an FP32 push at M, each followed by vmatmul.acc at M and an FP32 pop of all rows.
      val cmds = Seq(Cmd(PushWeight, 8, 0, m = false),
        Cmd(PushAccFP32, 4, 0, m = false), Cmd(PushAccBF16, 36, 0, m = true), Cmd(MatmulAcc, 9, 0, m = true),
        Cmd(PopFP32, 16, 0, m = false),
        Cmd(PushAccFP32, 12, 1, m = false), Cmd(PushAccFP32, 48, 1, m = true), Cmd(MatmulAcc, 10, 1, m = true),
        Cmd(PopFP32, 20, 1, m = false))
      val times = check(dut, 5, cmds, 16 until 24)((start, reg, row) => start(if (reg < 20) reg - 12 else reg - 8)(row))
      for ((m, t) <- times) {
        val gaps = spacing(t)
        assert(Seq(gaps(2), gaps(6)) == Seq(m.max(2), (2 * m).max(m + 2)), s"M $m: pushes and matmuls issue at $t")
      }
    }
  }
}
