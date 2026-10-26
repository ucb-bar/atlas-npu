package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Random

/** Every VPU instruction at the row counts M = 1, 2, 3, 6, 7, 31, 32 (vsetm). An instruction of M rows
  * reads and writes rows 0..M-1 of each register of its tiles (vli.col and vli.one: of m[vd]); the
  * other rows keep their values. So rows 0..M-1 of the result equal those of the 32-row instruction,
  * and a column reduction equals the 32-row one with rows M..31 of its source set to the identity of
  * the reduction. vpack and vunpack take all rows whatever M is. A pair stream takes 2M cycles: the
  * last write comes 2 (32 - M) cycles earlier than at 32 rows, 32 - M for a row reduction or a
  * one-register vli, 4 (32 - M) for a column reduction, which reads the pair twice.
  */
class VectorEngineRowsTest extends AnyFlatSpec {
  private val (pair, xreg, row0) = (0, 1, 2)   // ScalarISA.VSRC_*
  private val counts = Seq(1, 2, 3, 6, 7, 31, 32)
  private val specials = Seq(0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x3F80,
    0xBF80, 0x7F7F, 0xFF7F, 0x0080, 0x4049)
  private val (csum, cmax, cmin) = (15, 22, 23)
  private val (pack, unpack) = (17, 18)
  private val (vliOne, vliCol) = (27, 28)

  /** op (ScalarISA VPU_*), form, and the factor of 32 - M by which its last write comes earlier. */
  private case class Op(name: String, op: Int, src: Int, steps: Int)
  private val ops =
    Seq("vadd" -> 1, "vsub" -> 2, "vmul" -> 3, "vminimum" -> 25, "vmaximum" -> 24).flatMap { case (n, c) =>
      Seq(Op(n, c, pair, 2), Op(n + ".vx", c, xreg, 2), Op(n + ".vr", c, row0, 2)) } ++
    Seq("vrecip" -> 4, "vsqrt" -> 5, "vsin" -> 6, "vcos" -> 7, "vtanh" -> 8, "vlog2" -> 9, "vexp" -> 10,
      "vexp2" -> 11, "vsquare" -> 12, "vcube" -> 13, "vrelu" -> 19, "vmov" -> 26, "vgelu" -> 31,
      "vli.row" -> 29, "vli.all" -> 30).map { case (n, c) => Op(n, c, pair, 2) } ++
    Seq("vredsum.row" -> 14, "vredmax.row" -> 20, "vredmin.row" -> 21, "vli.one" -> vliOne, "vli.col" -> vliCol)
      .map { case (n, c) => Op(n, c, pair, 1) } ++
    Seq("vredsum" -> csum, "vredmax" -> cmax, "vredmin" -> cmin).map { case (n, c) => Op(n, c, pair, 4) } ++
    Seq("vpack" -> pack, "vunpack" -> unpack).map { case (n, c) => Op(n, c, pair, 0) }

  /** An instruction: T(vd) = op(T(vs1), T(vs2)) at `rows` rows (vpack and vunpack read T(vs2)). */
  private case class Cmd(op: Op, vd: Int, vs1: Int, vs2: Int, rows: Int)

  type Mem = Array[Array[BigInt]]

  private def row(lanes: Seq[Int]): BigInt = lanes.zipWithIndex.map { case (v, i) => BigInt(v) << (16 * i) }.sum

  private def randomMem(seed: Int): Mem = {
    val rng = new Random(seed)
    Array.tabulate(64, 32) { (_, _) =>
      row(Seq.fill(16)(if (rng.nextInt(4) == 0) specials(rng.nextInt(specials.size)) else rng.nextInt(1 << 16)))
    }
  }

  /** Issues each command as soon as it is not issue-busy and runs until the VPU is idle. Returns the
    * issue cycles and the cycle of the last write.
    */
  private def run(dut: VpuBroadcastHarness, mem: Mem, cmds: Seq[Cmd]): (Seq[Int], Int) = {
    dut.io.cmd.valid.poke(false.B)
    dut.reset.poke(true.B)
    dut.clock.step(2)
    dut.reset.poke(false.B)
    var queue  = cmds.toList
    val issued = Seq.newBuilder[Int]
    var cycle  = 0
    var last   = -1
    while (queue.nonEmpty || dut.io.busy.peek().litToBoolean) {
      queue.headOption.foreach { c =>
        dut.io.cmd.bits.op.poke(c.op.op.U)
        dut.io.cmd.bits.src.poke(c.op.src.U)
        dut.io.cmd.bits.vd.poke(c.vd.U)
        dut.io.cmd.bits.vs1.poke(c.vs1.U)
        dut.io.cmd.bits.vs2.poke(c.vs2.U)
        dut.io.cmd.bits.imm.poke((if (c.op.src == xreg) 0x3FC0 else 0x4049).U)
        dut.io.cmd.bits.scaleE8M0.poke(127.U)
        dut.io.cmd.bits.rows.poke((c.rows % 32).U)
      }
      val issue = queue.headOption.exists(c => !dut.io.issueBusy(c.op.src).peek().litValue.testBit(c.op.op))
      dut.io.cmd.valid.poke(issue.B)
      if (issue) { issued += cycle; queue = queue.tail }
      for (port <- 0 until 2) {
        val r = dut.io.reads(port)
        val data = if (r.valid.peek().litToBoolean) mem(r.bits.mregId.peek().litValue.toInt)(r.bits.row.peek().litValue.toInt)
                   else BigInt(0)
        dut.io.readData(port).poke(data.U(256.W))
      }
      for (port <- 0 until 2) {
        val w = dut.io.writes(port)
        if (w.valid.peek().litToBoolean) {
          mem(w.bits.mregId.peek().litValue.toInt)(w.bits.row.peek().litValue.toInt) = w.bits.data.peek().litValue
          last = cycle
        }
      }
      dut.clock.step()
      cycle += 1
      assert(cycle < 600, "the VPU did not finish")
    }
    (issued.result(), last)
  }

  /** The registers that an instruction writes. */
  private def dests(c: Cmd): Seq[Int] = if (Seq(pack, vliOne, vliCol).contains(c.op.op)) Seq(c.vd) else Seq(c.vd, c.vd + 1)

  /** The 32-row instruction that gives rows 0..M-1 of c: a column reduction reads rows M..31 of its
    * source pair as the identity of the reduction.
    */
  private def reference(dut: VpuBroadcastHarness, mem: Mem, c: Cmd): (Mem, Int) = {
    val m = mem.map(_.clone)
    val identity = Map(csum -> 0x8000, cmax -> 0xFFFF, cmin -> 0x7FFF)
    for (id <- identity.get(c.op.op); r <- Seq(c.vs1, c.vs1 + 1); i <- c.rows until 32) m(r)(i) = row(Seq.fill(16)(id))
    val (_, last) = run(dut, m, Seq(c.copy(rows = 32)))
    (m, last)
  }

  /** Rows 0..M-1 of each destination equal the reference, the other rows their values in `before`. */
  private def expect(c: Cmd, got: Mem, want: Mem, before: Mem): Unit = {
    val rows = if (c.op.steps == 0) 32 else c.rows
    for (r <- dests(c); i <- 0 until 32)
      assert(got(r)(i) == (if (i < rows) want(r)(i) else before(r)(i)), s"${c.op.name} at M ${c.rows}: m$r row $i")
  }

  "VectorEngineTop" should "run each instruction on rows 0..M-1 in 2M cycles per BF16 tile" in {
    simulate(new VpuBroadcastHarness) { dut =>
      for ((op, k) <- ops.zipWithIndex; m <- counts) {
        val before = randomMem(k)
        val c = Cmd(op, 8, 0, 2, m)
        val (want, last32) = reference(dut, before, c)
        val mem = before.map(_.clone)
        val (_, last) = run(dut, mem, Seq(c))
        expect(c, mem, want, before)
        assert(last == last32 - op.steps * (32 - m), s"${op.name} at M $m: last write at $last, at 32 rows $last32")
      }
    }
  }

  it should "keep the row count of the instruction in each slot" in {
    simulate(new VpuBroadcastHarness) { dut =>
      val op = ops.map(o => o.name -> o).toMap
      // Single-input instructions of other lane boxes overlap in the two slots, each with its own M.
      val cmds = Seq(Cmd(op("vexp"), 8, 0, 2, 3), Cmd(op("vtanh"), 12, 4, 2, 7), Cmd(op("vrecip"), 16, 6, 2, 1),
        Cmd(op("vredsum"), 20, 10, 2, 2), Cmd(op("vsqrt"), 24, 14, 2, 31), Cmd(op("vli.col"), 26, 0, 0, 5))
      val before = randomMem(99)
      val mem = before.map(_.clone)
      val (issued, _) = run(dut, mem, cmds)
      assert(issued(1) == issued(0) + 1, s"vtanh overlaps vexp: issued at $issued")
      for (c <- cmds) expect(c, mem, reference(dut, before, c)._1, before)
    }
  }
}
