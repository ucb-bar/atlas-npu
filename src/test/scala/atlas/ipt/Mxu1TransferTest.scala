// ============================================================================
// Mxu1TransferTest.scala — mxu1 accumulator pushes and pops, issued on cmdReady.
//
// Scaled BF16 pops: BF16_RNE(acc x 2^(e - 127)) rounds once. With E4M3 values
// in the accumulator this is the RTX 5090's vunpack conversion
// (baremetal/generators/fp8_silicon/vunpack.cvt.gz) of every code at every e.
// With FP32 values pushed by the FP32 push, the plain pop gives the RTX 5090's
// cvt.rn.bf16.f32 at both ends of every run of bf16_runs.cvt.gz, and the
// scaled pop the npu_model values of tests/test_mxu_pops.py.
// The FP32 push and pop copy every bit pattern, NaN payloads and -0 included.
//
// RUN: mill atlas.test.testOnly atlas.ipt.Mxu1TransferTest   (Verilator)
// ============================================================================
package atlas.ipt

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common.{InnerProductTreeParams, MregParams, MregReadReq, MregWriteReq}
import atlas.mxu.{MxuCmd, MxuOp}
import atlas.vector.CvtVectors
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ArrayBuffer

/** InnerProductTreesTop with MREG ports that the test serves: a read returns its row in the next
  * cycle, as MregFile does.
  */
class Mxu1Ports extends Module {
  private val p = MregParams()
  val io = IO(new Bundle {
    val cmd      = Flipped(Valid(new MxuCmd(p.mregIdBits)))
    val cmdReady = Output(Bool())
    val readData = Input(Vec(2, UInt(p.mregRowBits.W)))
    val reads    = Vec(2, Valid(new MregReadReq(p)))
    val writes   = Vec(2, Valid(new MregWriteReq(p)))
  })
  private val u = Module(new InnerProductTreesTop(InnerProductTreeParams(), p)).io
  u.cmd       := io.cmd
  io.cmdReady := u.cmdReady
  for ((req, resp, write, i) <- Seq((u.mregReadReq0, u.mregReadResp0, u.mregWriteReq0, 0),
                                    (u.mregReadReq1, u.mregReadResp1, u.mregWriteReq1, 1))) {
    io.reads(i)  := req
    io.writes(i) := write
    resp.valid   := RegNext(req.valid, false.B)
    resp.bits    := RegNext(io.readData(i))
  }
}

class Mxu1TransferTest extends AnyFlatSpec {
  private case class Cmd(op: MxuOp.Type, reg: Int, acc: Int, e: Int = 0)

  /** 64 m-registers x 32 rows of 256 bits. */
  private type Mreg = Array[Array[BigInt]]

  private def field(row: BigInt, bits: Int, j: Int): Long = ((row >> (bits * j)) & ((BigInt(1) << bits) - 1)).toLong
  private def row(values: Seq[Long], bits: Int): BigInt =
    values.zipWithIndex.map { case (v, j) => BigInt(v) << (bits * j) }.sum

  /** Issues each command as soon as the unit takes it (cmdReady), serves the MREG reads and
    * applies the writes, and runs 70 cycles past the last issue.
    */
  private def run(dut: Mxu1Ports, mreg: Mreg, cmds: Seq[Cmd]): Unit = {
    var queue = cmds.toList
    var quiet = 0
    while (quiet < 70) {
      queue.headOption.foreach { c =>
        dut.io.cmd.bits.op.poke(c.op)
        dut.io.cmd.bits.mregId.poke(c.reg.U)
        dut.io.cmd.bits.accSel.poke((c.acc == 1).B)
        dut.io.cmd.bits.weightSlot.poke(false.B)
        dut.io.cmd.bits.scaleE8M0.poke(c.e.U)
      }
      val issue = queue.nonEmpty && dut.io.cmdReady.peek().litToBoolean
      dut.io.cmd.valid.poke(issue.B)
      if (issue) queue = queue.tail
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
      if (queue.isEmpty) quiet += 1
    }
  }

  private def start(dut: Mxu1Ports): Mreg = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.readData.foreach(_.poke(0.U))
    dut.reset.poke(true.B)
    dut.clock.step()
    dut.reset.poke(false.B)
    Array.fill(64, 32)(BigInt(0))
  }

  private class Mismatches {
    var count = 0
    val first = ArrayBuffer[String]()
    def check(what: => String, got: Long, want: Long): Unit =
      if (got != want) {
        count += 1
        if (first.size < 8) first += f"$what: RTL 0x$got%x, expected 0x$want%x"
      }
  }

  /** The BF16 element j of row r of the pair {m[reg], m[reg + 1]}. */
  private def bf16(mreg: Mreg, reg: Int, r: Int, j: Int): Long = field(mreg(reg + j / 16)(r), 16, j % 16)

  behavior of "the mxu1 BF16 pops"

  it should "give the RTX 5090 BF16 of every E4M3 value at every e" in {
    val vunpack = CvtVectors.read("vunpack")("rtx5090") // (e << 8) | code -> BF16
    def code(r: Int, j: Int): Int = (32 * r + j) % 256   // rows 0-7 hold every code
    val bad = new Mismatches
    simulate(new Mxu1Ports) { dut =>
      val mreg = start(dut)
      for (r <- 0 until 32) mreg(0)(r) = row((0 until 32).map(j => code(r, j).toLong), 8)
      run(dut, mreg, Seq(Cmd(MxuOp.PushAccFP8, 0, 1)))
      // Back-to-back pops into the pairs m8 .. m62. The plain pop is e = 127 whatever the scale field.
      val pops = (0 until 256).map(e => Cmd(MxuOp.PopAccBF16Scaled, 0, 1, e)) :+ Cmd(MxuOp.PopAccBF16, 0, 1, 0)
      for (batch <- pops.grouped(28)) {
        val cmds = batch.zipWithIndex.map { case (c, k) => c.copy(reg = 8 + 2 * k) }
        run(dut, mreg, cmds)
        for (c <- cmds; e = if (c.op == MxuOp.PopAccBF16) 127 else c.e; r <- 0 until 32; j <- 0 until 32)
          bad.check(s"${c.op} e $e row $r column $j", bf16(mreg, c.reg, r, j), vunpack((e << 8) | code(r, j)))
      }
    }
    assert(bad.count == 0, s"${bad.count} mismatches, first: ${bad.first.mkString("; ")}")
  }

  /** The FP32 tile in m[reg..reg+3]: register q holds columns 8q..8q+7. */
  private def putFp32(mreg: Mreg, reg: Int, tile: Int => Long): Unit =
    for (q <- 0 until 4; r <- 0 until 32)
      mreg(reg + q)(r) = row((0 until 8).map(j => tile(32 * r + 8 * q + j)), 32)
  private def fp32(mreg: Mreg, reg: Int, r: Int, j: Int): Long = field(mreg(reg + j / 8)(r), 32, j % 8)

  it should "round FP32 accumulators as the RTX 5090 and npu_model do" in {
    val runs = CvtVectors.read("bf16_runs")
    val ends = runs("x").indices.flatMap(k =>                // both ends of each run
      Seq(runs("x")(k), if (k + 1 < runs("x").length) runs("x")(k + 1) - 1 else 0xffffffffL))
    val want = runs("rtx5090").flatMap(b => Seq(b, b))
    // (x, e, BF16_RNE(x * 2^(e - 127))): tests/test_mxu_pops.py SCALED, worked by hand.
    val scaled = Seq(
      (0x3f800000L, 0, 0x0040L), (0x3f800000L, 1, 0x0080L), (0x3f800000L, 126, 0x3f00L),
      (0x3f800000L, 127, 0x3f80L), (0x3f800000L, 128, 0x4000L), (0x3f800000L, 254, 0x7f00L),
      (0x3f800000L, 255, 0x7f80L), (0xbfc00000L, 254, 0xff40L), (0xbfc00000L, 255, 0xff80L),
      (0x7f7fffffL, 126, 0x7f00L), (0x7f7fffffL, 127, 0x7f80L),
      (0x3f808000L, 126, 0x3f00L), (0x3f818000L, 128, 0x4002L), (0x00000001L, 255, 0x3500L),
      (0x007fffffL, 255, 0x4080L), (0x00400000L, 127, 0x0040L), (0x00400000L, 0, 0x0000L), (0x80000001L, 127, 0x8000L),
      (0x00600000L, 121, 0x0002L), (0x00a00000L, 121, 0x0002L), (0x3cbfffc0L, 0, 0x0001L),
      (0x7f800000L, 0, 0x7f80L), (0xff800000L, 0, 0xff80L), (0x00000000L, 255, 0x0000L), (0x80000000L, 255, 0x8000L),
      (0x7fc00000L, 0, 0x7fffL), (0xffc00001L, 255, 0x7fffL), (0x7f800001L, 127, 0x7fffL))
    val bad = new Mismatches
    simulate(new Mxu1Ports) { dut =>
      val mreg = start(dut)
      // Each FP32 push is followed at once by its pop: the pop waits for the push's second phase.
      for (t <- 0 until (ends.length + 1023) / 1024) {
        putFp32(mreg, 4, i => ends((1024 * t + i).min(ends.length - 1)))
        run(dut, mreg, Seq(Cmd(MxuOp.PushAccFP32, 4, 0), Cmd(MxuOp.PopAccBF16, 8, 0)))
        for (i <- 0 until 1024 if 1024 * t + i < ends.length)
          bad.check(f"x 0x${ends(1024 * t + i)}%08x", bf16(mreg, 8, i / 32, i % 32), want(1024 * t + i))
      }
      putFp32(mreg, 4, i => scaled(i % scaled.length)._1)
      val es   = scaled.map(_._2).distinct
      val pops = es.zipWithIndex.map { case (e, k) => Cmd(MxuOp.PopAccBF16Scaled, 8 + 2 * k, 1, e) }
      run(dut, mreg, Cmd(MxuOp.PushAccFP32, 4, 1) +: pops)
      for (c <- pops; i <- 0 until 1024; (x, e, w) = scaled(i % scaled.length) if e == c.e)
        bad.check(f"x 0x$x%08x e $e", bf16(mreg, c.reg, i / 32, i % 32), w)
    }
    assert(bad.count == 0, s"${bad.count} mismatches, first: ${bad.first.mkString("; ")}")
  }

  behavior of "the mxu1 FP32 push and pop"

  it should "copy every FP32 bit pattern and keep a spilled accumulator" in {
    val special = Seq(0x00000000L, 0x80000000L, 0x00000001L, 0x80000001L, 0x007fffffL, 0x807fffffL, 0x00800000L,
      0x80800000L, 0x3f800000L, 0xbf800000L, 0x7f7fffffL, 0xff7fffffL, 0x7f800000L, 0xff800000L, 0x7fc00000L,
      0xffc00000L, 0x7fc12345L, 0xffffffffL, 0x7f800001L, 0xff800001L, 0x7fbfffffL, 0xffbfffffL)
    val rng  = new scala.util.Random(77)
    val tile = Array.tabulate(1024)(i => if (i < special.length) special(i) else rng.nextLong() & 0xffffffffL)
    val bf16Bits = Array.tabulate(1024)(_ => rng.nextInt(1 << 16).toLong)
    val bad = new Mismatches
    simulate(new Mxu1Ports) { dut =>
      val mreg = start(dut)
      putFp32(mreg, 4, tile)
      for (r <- 0 until 32; h <- 0 until 2)
        mreg(12 + h)(r) = row((0 until 16).map(j => bf16Bits(32 * r + 16 * h + j)), 16)
      // acc1 = tile; spill it to m16; overwrite acc1 with the BF16 tile at once (the push waits
      // for the pop's second phase); spill that to m20; restore the first spill into acc0, pop it to m24.
      run(dut, mreg, Seq(Cmd(MxuOp.PushAccFP32, 4, 1), Cmd(MxuOp.PopAccFP32, 16, 1), Cmd(MxuOp.PushAccBF16, 12, 1),
                         Cmd(MxuOp.PopAccFP32, 20, 1), Cmd(MxuOp.PushAccFP32, 16, 0), Cmd(MxuOp.PopAccFP32, 24, 0)))
      for (i <- 0 until 1024; r = i / 32; j = i % 32) {
        bad.check(s"spill row $r column $j", fp32(mreg, 16, r, j), tile(i))
        bad.check(s"BF16 push row $r column $j", fp32(mreg, 20, r, j), bf16Bits(i) << 16)
        bad.check(s"restore row $r column $j", fp32(mreg, 24, r, j), tile(i))
      }
    }
    assert(bad.count == 0, s"${bad.count} mismatches, first: ${bad.first.mkString("; ")}")
  }
}
