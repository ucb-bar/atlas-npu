// ============================================================================
// Mxu1RowsTest.scala — mxu1 at the row counts M = 1, 2, 3, 6, 7, 31, 32 (vsetm).
//
// A command of M rows moves rows 0..M-1 of its m-registers and accumulator; the
// other rows keep their values. A weight push moves all rows. Each scenario
// runs from the same MREG twice: with every command at 32 rows, and with the
// commands under test at M. Rows 0..M-1 must equal the 32-row run, the other
// rows their old values. A matmul feeds M rows, one per cycle; a matmul on an
// accumulator issues only after the previous one's row 0 lands, 4 cycles after
// that one issues. So matmuls on one accumulator issue every max(M, 4) cycles.
//
// RUN: mill atlas.test.testOnly atlas.ipt.Mxu1RowsTest   (Verilator)
// ============================================================================
package atlas.ipt

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import atlas.mxu.MxuOp
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class Mxu1RowsTest extends AnyFlatSpec {
  private val counts = Seq(1, 2, 3, 6, 7, 31, 32)

  /** A command; `m` says whether it runs at the row count under test (else at 32 rows). */
  private case class Cmd(op: MxuOp.Type, reg: Int, acc: Int, m: Boolean, w: Int = 0, e: Int = 127)

  private type Mreg = Array[Array[BigInt]]

  /** Issues each command as soon as the unit takes it, serves the MREG reads and applies the writes.
    * Returns the issue cycles.
    */
  private def run(dut: Mxu1Ports, mreg: Mreg, cmds: Seq[Cmd], rows: Int): Seq[Int] = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.readData.foreach(_.poke(0.U))
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

  /** Random MREG contents: E4M3 codes, BF16 and FP32 bit patterns are all bits. */
  private def randomMreg(seed: Int): Mreg = {
    val rng = new Random(seed)
    Array.fill(64, 32)(BigInt(256, rng))
  }

  /** Runs `cmds` at 32 rows and at each M. Rows 0..M-1 of `out` must equal the 32-row run; rows M..31
    * must equal `old(reg, row)` of the MREG before the run. Returns the issue cycles of each M.
    */
  private def check(dut: Mxu1Ports, seed: Int, cmds: Seq[Cmd], out: Seq[Int])(old: (Mreg, Int, Int) => BigInt): Map[Int, Seq[Int]] = {
    val full = randomMreg(seed)
    run(dut, full, cmds, 32)
    counts.map { m =>
      val start = randomMreg(seed)
      val mreg  = start.map(_.clone)
      val times = run(dut, mreg, cmds, m)
      for (reg <- out; row <- 0 until 32) {
        val want = if (row < m) full(reg)(row) else old(start, reg, row)
        assert(mreg(reg)(row) == want, s"M $m: m$reg row $row")
      }
      m -> times
    }.toMap
  }

  private def spacing(times: Seq[Int]): Seq[Int] = times.zip(times.tail).map { case (a, b) => b - a }

  "mxu1" should "feed M rows into rows 0..M-1 of the accumulator, every max(M, 4) cycles on one accumulator" in {
    simulate(new Mxu1Ports) { dut =>
      // acc0 = the FP32 tile m4..m7; matmuls at M; then all 32 rows of acc0 to m16..m19.
      val cmds = Seq(Cmd(MxuOp.PushAccFP32, 4, 0, m = false), Cmd(MxuOp.PushWeight, 8, 0, m = true),
        Cmd(MxuOp.Matmul, 9, 0, m = true), Cmd(MxuOp.MatmulAcc, 10, 0, m = true),
        Cmd(MxuOp.MatmulAcc, 11, 0, m = true), Cmd(MxuOp.PopAccFP32, 16, 0, m = false))
      val times = check(dut, 1, cmds, 16 until 20)((start, reg, row) => start(reg - 12)(row))
      for ((m, t) <- times) assert(spacing(t.slice(2, 5)) == Seq.fill(2)(m.max(4)), s"M $m: matmuls issue at $t")
    }
  }

  it should "issue matmuls on alternating accumulators every M cycles" in {
    simulate(new Mxu1Ports) { dut =>
      val cmds = Seq(Cmd(MxuOp.PushAccFP32, 4, 0, m = false), Cmd(MxuOp.PushAccFP32, 12, 1, m = false),
        Cmd(MxuOp.PushWeight, 8, 1, m = false, w = 1),
        Cmd(MxuOp.Matmul, 9, 0, m = true, w = 1), Cmd(MxuOp.Matmul, 10, 1, m = true, w = 1),
        Cmd(MxuOp.MatmulAcc, 11, 0, m = true, w = 1), Cmd(MxuOp.MatmulAcc, 9, 1, m = true, w = 1),
        Cmd(MxuOp.PopAccFP32, 16, 0, m = false), Cmd(MxuOp.PopAccFP32, 20, 1, m = false))
      val times = check(dut, 2, cmds, 16 until 24)((start, reg, row) => start(if (reg < 20) reg - 12 else reg - 8)(row))
      for ((m, t) <- times) {
        val Seq(t0, t1, t2, t3) = t.slice(3, 7)
        assert(t1 - t0 == m && t2 == (t1 + m).max(t0 + 4) && t3 == (t2 + m).max(t1 + 4), s"M $m: matmuls issue at $t")
      }
    }
  }

  it should "pop rows 0..M-1 of the accumulator into rows 0..M-1 of the m-registers" in {
    simulate(new Mxu1Ports) { dut =>
      // acc0 = a 32-row matmul; then the four pops at M, back to back.
      val cmds = Seq(Cmd(MxuOp.PushWeight, 8, 0, m = false), Cmd(MxuOp.Matmul, 9, 0, m = false),
        Cmd(MxuOp.PopAccBF16, 24, 0, m = true), Cmd(MxuOp.PopAccFP8, 26, 0, m = true, e = 120),
        Cmd(MxuOp.PopAccBF16Scaled, 28, 0, m = true, e = 131), Cmd(MxuOp.PopAccFP32, 32, 0, m = true))
      val times = check(dut, 3, cmds, (24 to 26) ++ (28 to 29) ++ (32 to 35))((start, reg, row) => start(reg)(row))
      for ((m, t) <- times) assert(spacing(t.drop(2)) == Seq(m, m, m), s"M $m: pops issue at $t")
    }
  }

  it should "push rows 0..M-1 of the m-registers into rows 0..M-1 of the accumulator" in {
    simulate(new Mxu1Ports) { dut =>
      // Each push at M lands on the FP32 tile m4..m7 and is popped with all 32 rows. A pop reads
      // row 0 after the push's last write of it: 2 cycles after a BF16 or FP8 push, M + 2 after an
      // FP32 push. Last, an FP32 pop at M, and a matmul on its accumulator, which waits for the
      // pop's first phase (M + 1 cycles).
      val cmds = Seq(
        Cmd(MxuOp.PushAccFP32, 4, 1, m = false), Cmd(MxuOp.PushAccBF16, 36, 1, m = true),
        Cmd(MxuOp.PopAccFP32, 40, 1, m = false),
        Cmd(MxuOp.PushAccFP32, 4, 1, m = false), Cmd(MxuOp.PushAccFP8, 38, 1, m = true),
        Cmd(MxuOp.PopAccFP32, 44, 1, m = false),
        Cmd(MxuOp.PushAccFP32, 4, 0, m = false), Cmd(MxuOp.PushAccFP32, 48, 0, m = true),
        Cmd(MxuOp.PopAccFP32, 52, 0, m = false), Cmd(MxuOp.PopAccFP32, 56, 0, m = true),
        Cmd(MxuOp.Matmul, 9, 0, m = true))
      val times = check(dut, 4, cmds, (40 until 48) ++ (52 until 60))((start, reg, row) =>
        if (reg < 56) start(4 + reg % 4)(row) else start(reg)(row))
      for ((m, t) <- times) {
        val gaps = spacing(t)
        assert(Seq(gaps(1), gaps(4), gaps(7), gaps(9)) == Seq(2, 2, m + 2, m + 1), s"M $m: pushes and pops issue at $t")
      }
    }
  }
}
