// ============================================================================
// Mxu1TransferTest.scala — mxu1 accumulator pops, issued on cmdReady.
//
// Scaled BF16 pops: BF16_RNE(acc x 2^(e - 127)) rounds once. With E4M3 values
// in the accumulator this is the RTX 5090's vunpack conversion
// (baremetal/generators/fp8_silicon/vunpack.cvt.gz) of every code at every e.
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

  behavior of "the mxu1 scaled BF16 pop"

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
}
