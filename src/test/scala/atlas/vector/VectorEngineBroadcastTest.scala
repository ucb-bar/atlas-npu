package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator.EphemeralSimulator._
import atlas.common.{MregParams, MregReadReq, MregWriteReq, VpuParams}
import atlas.scalar.VpuCmd
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Random

/** VectorEngineTop with an MREG outside it that answers a read in the next cycle. */
class VpuBroadcastHarness extends Module {
  val p = MregParams()
  val io = IO(new Bundle {
    val cmd       = Flipped(Valid(new VpuCmd))
    val readData  = Input(Vec(2, UInt(256.W)))
    val reads     = Vec(2, Valid(new MregReadReq(p)))
    val writes    = Vec(2, Valid(new MregWriteReq(p)))
    val issueBusy = Output(Vec(3, UInt((VPUOp.all.size + 1).W)))
    val busy      = Output(Bool())
  })
  val vpu = Module(new VectorEngineTop(VpuParams(), p))
  vpu.io.cmd <> io.cmd
  io.reads(0)  := vpu.io.mregReadReq0
  io.reads(1)  := vpu.io.mregReadReq1
  io.writes(0) := vpu.io.mregWriteReq0
  io.writes(1) := vpu.io.mregWriteReq1
  vpu.io.mregReadResp0.valid := RegNext(io.reads(0).valid, false.B)
  vpu.io.mregReadResp1.valid := RegNext(io.reads(1).valid, false.B)
  vpu.io.mregReadResp0.bits  := RegNext(io.readData(0))
  vpu.io.mregReadResp1.bits  := RegNext(io.readData(1))
  io.issueBusy := vpu.io.issueBusy
  io.busy      := vpu.io.busy
}

/** Runs commands on a VpuBroadcastHarness with an MREG of 64 registers of 32 rows (Mem). */
object VpuBroadcastHarness {
  val (pair, xreg, row0) = (0, 1, 2)   // ScalarISA.VSRC_*

  case class Cmd(op: Int, vd: Int, vs1: Int, vs2: Int = 0, src: Int = pair, imm: Int = 0)

  type Mem = Array[Array[BigInt]]

  private def reset(dut: VpuBroadcastHarness): Unit = {
    dut.io.cmd.valid.poke(false.B)
    dut.reset.poke(true.B)
    dut.clock.step(2)
    dut.reset.poke(false.B)
  }

  private def poke(dut: VpuBroadcastHarness, c: Cmd): Unit = {
    dut.io.cmd.bits.op.poke(c.op.U)
    dut.io.cmd.bits.src.poke(c.src.U)
    dut.io.cmd.bits.vd.poke(c.vd.U)
    dut.io.cmd.bits.vs1.poke(c.vs1.U)
    dut.io.cmd.bits.vs2.poke(c.vs2.U)
    dut.io.cmd.bits.imm.poke(c.imm.U)
    dut.io.cmd.bits.scaleE8M0.poke(127.U)
  }

  /** Issue each command at its cycle (it must not be issue-busy) and run until the VPU is idle. */
  def drive(dut: VpuBroadcastHarness, mem: Mem, cmds: Map[Int, Cmd]): Unit = {
    reset(dut)
    var cycle = 0
    while (cycle <= cmds.keys.max || dut.io.busy.peek().litToBoolean) {
      dut.io.cmd.valid.poke(cmds.contains(cycle).B)
      for (c <- cmds.get(cycle)) {
        assert(!dut.io.issueBusy(c.src).peek().litValue.testBit(c.op), s"cycle $cycle: $c is issue-busy")
        poke(dut, c)
      }
      for (port <- 0 until 2) {
        val r = dut.io.reads(port)
        val data = if (r.valid.peek().litToBoolean) mem(r.bits.mregId.peek().litValue.toInt)(r.bits.row.peek().litValue.toInt)
                   else BigInt(0)
        dut.io.readData(port).poke(data.U(256.W))
      }
      for (port <- 0 until 2) {
        val w = dut.io.writes(port)
        if (w.valid.peek().litToBoolean)
          mem(w.bits.mregId.peek().litValue.toInt)(w.bits.row.peek().litValue.toInt) = w.bits.data.peek().litValue
      }
      dut.clock.step()
      cycle += 1
      assert(cycle < 400, "the VPU did not finish")
    }
  }

  /** The first cycle in which `next` is not issue-busy, with `first` issued in cycle 0. */
  def firstIssue(dut: VpuBroadcastHarness, first: Cmd, next: Cmd): Int = {
    reset(dut)
    dut.io.cmd.valid.poke(true.B)
    poke(dut, first)
    dut.clock.step()
    dut.io.cmd.valid.poke(false.B)
    var cycle = 1
    while (dut.io.issueBusy(next.src).peek().litValue.testBit(next.op)) { dut.clock.step(); cycle += 1 }
    dut.clock.step(140)
    cycle
  }

  /** The tile T(r) as 32 rows of 32 lanes. */
  def tile(mem: Mem, r: Int): Seq[BigInt] = (0 until 32).flatMap(i => Seq(mem(r)(i), mem(r + 1)(i)))
}

/** The .vx and .vr forms of vadd, vsub, vmul, vminimum and vmaximum give, bit for bit, what the
  * .bf16 form gives with the broadcast operand as its second tile: x[rs2][31:16] in every element,
  * or row 0 of T(vs2) in every row. Each form runs alone, next to another op in the VPU's other
  * slot, with vd = vs2, and right after a .vr in the other slot (the row-0 registers change hands).
  */
class VectorEngineBroadcastTest extends AnyFlatSpec {
  import VpuBroadcastHarness._

  private val ops = Seq("vadd" -> 1, "vsub" -> 2, "vmul" -> 3, "vminimum" -> 25, "vmaximum" -> 24)
  private val (tanh, mov) = (8, 26)
  private val specials = Seq(0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x3F80,
    0xBF80, 0x7F7F, 0xFF7F, 0x0080, 0x4049)

  private def row(lanes: Seq[Int]): BigInt = lanes.zipWithIndex.map { case (v, i) => BigInt(v) << (16 * i) }.sum

  private def randomMem(seed: Int): Mem = {
    val rng = new Random(seed)
    Array.tabulate(64, 32) { (_, _) =>
      row(Seq.fill(16)(if (rng.nextInt(4) == 0) specials(rng.nextInt(specials.size)) else rng.nextInt(1 << 16)))
    }
  }

  /** What the .bf16 form writes to T(vd) with `second` as T(vs2), run alone on a copy of `mem`. */
  private def reference(dut: VpuBroadcastHarness, mem: Mem, op: Int, vs1: Int, second: (Int, Int) => BigInt): Seq[BigInt] = {
    val m = mem.map(_.clone)
    for (r <- 0 until 2; i <- 0 until 32) m(60 + r)(i) = second(r, i)
    drive(dut, m, Map(0 -> Cmd(op, 56, vs1, 60)))
    tile(m, 56)
  }

  private def scalarRow(s: Int): BigInt = row(Seq.fill(16)(s))

  private def expect(got: Seq[BigInt], want: Seq[BigInt], what: String): Unit = {
    val differ = got.indices.filter(i => got(i) != want(i))
    assert(differ.isEmpty, s"$what: rows (2 i + register) that differ")
  }

  "VectorEngineTop" should "give the .bf16 result for each .vx and .vr form" in {
    simulate(new VpuBroadcastHarness) { dut =>
      for (((name, op), k) <- ops.zipWithIndex; s <- Seq(0xC0A0, specials(k), specials(k + 5))) {
        val mem = randomMem(k)
        val want = reference(dut, mem, op, 0, (_, _) => scalarRow(s))
        drive(dut, mem, Map(0 -> Cmd(op, 8, 0, src = xreg, imm = s)))
        expect(tile(mem, 8), want, f"$name.vx 0x$s%04x")
      }
      for ((name, op) <- ops) {
        val mem = randomMem(op)
        val want = reference(dut, mem, op, 0, (r, _) => mem(2 + r)(0))
        drive(dut, mem, Map(0 -> Cmd(op, 8, 0, 2, src = row0)))
        expect(tile(mem, 8), want, s"$name.vr")
        val same = randomMem(op)
        drive(dut, same, Map(0 -> Cmd(op, 2, 0, 2, src = row0)))
        expect(tile(same, 2), want, s"$name.vr with vd = vs2")
      }
    }
  }

  it should "run a form in one slot next to another op" in {
    simulate(new VpuBroadcastHarness) { dut =>
      val mem = randomMem(7)
      val add = reference(dut, mem, 1, 0, (_, _) => scalarRow(0x3FC0))
      val mul = reference(dut, mem, 3, 4, (r, _) => mem(2 + r)(0))
      val alone = mem.map(_.clone)
      drive(dut, alone, Map(0 -> Cmd(tanh, 12, 10), 1 -> Cmd(mov, 14, 16)))
      // .vx in slot 1 and vtanh in slot 2, issued in back-to-back cycles
      val m1 = mem.map(_.clone)
      drive(dut, m1, Map(0 -> Cmd(1, 8, 0, src = xreg, imm = 0x3FC0), 1 -> Cmd(tanh, 12, 10)))
      expect(tile(m1, 8), add, "vadd.bf16.vx next to vtanh")
      expect(tile(m1, 12), tile(alone, 12), "vtanh next to vadd.bf16.vx")
      // vmov in slot 1, then .vr in slot 2 as soon as vmov has read its last row
      val wait = firstIssue(dut, Cmd(mov, 14, 16), Cmd(3, 8, 4, 2, src = row0))
      assert(wait == 64, s"vmul.bf16.vr waits for the other slot's reads: issue at $wait")
      val m2 = mem.map(_.clone)
      drive(dut, m2, Map(0 -> Cmd(mov, 14, 16), wait -> Cmd(3, 8, 4, 2, src = row0)))
      expect(tile(m2, 8), mul, "vmul.bf16.vr next to vmov")
      expect(tile(m2, 14), tile(alone, 14), "vmov next to vmul.bf16.vr")
      // .vr in slot 1, then .vx in slot 2, which needs no second read port
      val m3 = mem.map(_.clone)
      drive(dut, m3, Map(0 -> Cmd(3, 8, 4, 2, src = row0), 1 -> Cmd(1, 20, 0, src = xreg, imm = 0x3FC0)))
      expect(tile(m3, 8), mul, "vmul.bf16.vr next to vadd.bf16.vx")
      expect(tile(m3, 20), add, "vadd.bf16.vx next to vmul.bf16.vr")
      // .vr in slot 1, then .vr in slot 2: the row-0 registers change hands
      val sub = reference(dut, mem, 2, 0, (r, _) => mem(22 + r)(0))
      val next = firstIssue(dut, Cmd(3, 8, 4, 2, src = row0), Cmd(2, 20, 0, 22, src = row0))
      assert(next == 65, s"a .vr after a .vr: issue at $next")
      val m4 = mem.map(_.clone)
      drive(dut, m4, Map(0 -> Cmd(3, 8, 4, 2, src = row0), next -> Cmd(2, 20, 0, 22, src = row0)))
      expect(tile(m4, 8), mul, "vmul.bf16.vr, then vsub.bf16.vr")
      expect(tile(m4, 20), sub, "vsub.bf16.vr after vmul.bf16.vr")
    }
  }
}
