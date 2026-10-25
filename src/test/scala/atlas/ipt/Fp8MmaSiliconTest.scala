// ============================================================================
// Fp8MmaSiliconTest.scala — mxu1 against the RTX 5090 FP8 MMA
// (baremetal/generators/fp8_silicon, README.md).
//
// Lane: every vector of every *.vec.gz file goes through AnchorAccumulationTree
// (default pipeline), 16 lanes per cycle; d must equal the rtx5090 column.
// Chains: each tile of chain_s8.tiles.gz runs through InnerProductTreesTop as
// vmatpush.acc.bf16 C, then per K = 32 step vmatpush.weight and vmatmul.acc;
// the FP32 rows each matmul writes into the accumulator must equal the GPU's
// D after that step. All comparisons are bit for bit.
//
// RUN: mill atlas.test.testOnly atlas.ipt.Fp8MmaSiliconTest   (Verilator)
// ============================================================================
package atlas.ipt

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import chisel3.util.experimental.BoringUtils
import atlas.common.InnerProductTreeParams
import atlas.mxu.MxuOp
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

/** `n` IPT lanes, each with its own a (32 E4M3 codes, element k in byte k), w and FP32 c. */
class Fp8MmaLanes(n: Int) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt((256 * n).W))
    val w = Input(UInt((256 * n).W))
    val c = Input(UInt((32 * n).W))
    val d = Output(UInt((32 * n).W))
  })
  val p = InnerProductTreeParams()
  io.d := VecInit((0 until n).map { i =>
    val lane = Module(new AnchorAccumulationTree(p))
    lane.io.act        := io.a(256 * i + 255, 256 * i).asTypeOf(lane.io.act)
    lane.io.weightBuf0 := io.w(256 * i + 255, 256 * i).asTypeOf(lane.io.weightBuf0)
    lane.io.weightBuf1 := 0.U.asTypeOf(lane.io.weightBuf1)
    lane.io.bufReadSel := false.B
    lane.io.psum       := io.c(32 * i + 31, 32 * i)
    lane.io.out
  }).asUInt
}

/** The IPT unit harness, plus the rows its matmuls write into the accumulators. */
class Fp8MmaTileHarness extends InnerProductTreesUnitHarness {
  val accWrite = IO(new Bundle {
    val valid = Output(Bool())
    val row   = Output(UInt(5.W))
    val data  = Output(UInt(1024.W))
  })
  private val write = BoringUtils.tapAndRead(ipt.accBuf.io.computeWriteReq)
  accWrite.valid := write.valid
  accWrite.row   := write.bits.rowIdx
  accWrite.data  := write.bits.data.asUInt
}

class Fp8MmaSiliconTest extends AnyFlatSpec {
  private val root  = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT", sys.props("user.dir")))
  private val dir   = root.resolve("baremetal/generators/fp8_silicon")
  private val lanes = 16

  /** An fp8-vectors-v1 file: its name, the 68-byte records and the rtx5090 column. */
  private def readVec(path: Path): (String, Array[Byte], Array[Long]) = {
    val raw  = new GZIPInputStream(Files.newInputStream(path)).readAllBytes()
    val nl   = raw.indexOf('\n'.toByte)
    val hdr  = ujson.read(new String(raw, 0, nl, "UTF-8"))
    val n    = hdr("count").num.toInt
    val col  = hdr("columns").arr.map(_.str).indexOf("rtx5090")
    val recs = raw.slice(nl + 1, nl + 1 + 68 * n)
    val out  = ByteBuffer.wrap(raw, nl + 1 + 68 * n + 4 * n * col, 4 * n).order(ByteOrder.LITTLE_ENDIAN)
    (hdr("name").str, recs, Array.fill(n)(out.getInt() & 0xffffffffL))
  }

  /** Little-endian bytes [from, from + len) of `b` as an unsigned number. */
  private def le(b: Array[Byte], from: Int, len: Int): BigInt =
    BigInt(1, b.slice(from, from + len).reverse)

  behavior of "the IPT lane"

  it should "give the RTX 5090 result for every stored FP8 MMA vector" in {
    val files = Files.list(dir).iterator.asScala.filter(_.toString.endsWith(".vec.gz")).toSeq.sortBy(_.toString)
    assert(files.nonEmpty, s"no vectors in $dir")
    val bad = ArrayBuffer[String]()
    var total = 0L
    simulate(new Fp8MmaLanes(lanes)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      for (file <- files) {
        val (name, recs, want) = readVec(file)
        val n = want.length
        var wrong = 0
        // The lanes have a latency of two cycles: after the step that follows the poke of
        // batch b + 1, d holds the results of batch b.
        for (b <- 0 to (n + lanes - 1) / lanes) {
          if (b < (n + lanes - 1) / lanes) {
            val idx = (0 until lanes).map(l => (b * lanes + l).min(n - 1))
            dut.io.a.poke(idx.zipWithIndex.map { case (i, l) => le(recs, 68 * i, 32) << (256 * l) }.sum.U)
            dut.io.w.poke(idx.zipWithIndex.map { case (i, l) => le(recs, 68 * i + 32, 32) << (256 * l) }.sum.U)
            dut.io.c.poke(idx.zipWithIndex.map { case (i, l) => le(recs, 68 * i + 64, 4) << (32 * l) }.sum.U)
          }
          dut.clock.step()
          if (b > 0) {
            val d = dut.io.d.peek().litValue
            for (l <- 0 until lanes; i = (b - 1) * lanes + l if i < n) {
              val got = ((d >> (32 * l)) & 0xffffffffL).toLong
              if (got != want(i)) {
                wrong += 1
                if (bad.size < 8) bad += f"$name vector $i: RTL 0x$got%08x, RTX 5090 0x${want(i)}%08x"
              }
            }
          }
        }
        println(f"$name%-14s $n%7d vectors, $wrong%d differ")
        total += wrong
      }
    }
    assert(total == 0, s"$total vectors differ, first: ${bad.mkString("; ")}")
  }

  behavior of "the mxu1 FP32 accumulators"

  it should "chain K = 32 steps as the RTX 5090 does" in {
    val raw   = new GZIPInputStream(Files.newInputStream(dir.resolve("chain_s8.tiles.gz"))).readAllBytes()
    val nl    = raw.indexOf('\n'.toByte)
    val hdr   = ujson.read(new String(raw, 0, nl, "UTF-8"))
    val (tiles, steps) = (hdr("count").num.toInt, hdr("steps").num.toInt)
    val rec   = 4096 + 2048 * steps
    val gpu   = nl + 1 + tiles * rec  // the rtx5090 column: D[tile][step][row][column]
    def c(t: Int, r: Int, j: Int): BigInt = le(raw, nl + 1 + t * rec + 128 * r + 4 * j, 4)
    def a(t: Int, s: Int, r: Int): BigInt = le(raw, nl + 1 + t * rec + 4096 + 1024 * s + 32 * r, 32)
    def w(t: Int, s: Int, r: Int): BigInt = le(raw, nl + 1 + t * rec + 4096 + 1024 * (steps + s) + 32 * r, 32)
    def d(t: Int, s: Int, r: Int): BigInt = le(raw, gpu + 4096 * (steps * t + s) + 128 * r, 128)
    val (wgt, act, cLo) = (0, 2, 4)
    val bad = ArrayBuffer[String]()
    var wrong = 0
    simulate(new Fp8MmaTileHarness) { dut =>
      def idle(): Unit = {
        dut.io.cmd.valid.poke(false.B)
        dut.io.testWrite.valid.poke(false.B)
        dut.io.testRead.valid.poke(false.B)
      }
      def write(bank: Int, row: Int, data: BigInt): Unit = {
        dut.io.testWrite.valid.poke(true.B)
        dut.io.testWrite.bits.mregId.poke(bank.U)
        dut.io.testWrite.bits.row.poke(row.U)
        dut.io.testWrite.bits.data.poke(data.U)
        dut.clock.step()
        dut.io.testWrite.valid.poke(false.B)
      }
      def command(op: MxuOp.Type, bank: Int): Unit = {
        dut.io.cmd.valid.poke(true.B)
        dut.io.cmd.bits.op.poke(op)
        dut.io.cmd.bits.mregId.poke(bank.U)
        dut.io.cmd.bits.accSel.poke(false.B)
        dut.io.cmd.bits.weightSlot.poke(false.B)
        dut.io.cmd.bits.scaleE8M0.poke(127.U)
        dut.clock.step()
        dut.io.cmd.valid.poke(false.B)
      }
      /** Steps until the unit is idle; returns the accumulator rows written meanwhile. */
      def drain(): Map[Int, BigInt] = {
        val rows = scala.collection.mutable.Map[Int, BigInt]()
        var n = 0
        while (n < 200 && (dut.io.dataBusy.peek().litToBoolean || dut.io.computeBusy.peek().litToBoolean)) {
          if (dut.accWrite.valid.peek().litToBoolean)
            rows(dut.accWrite.row.peek().litValue.toInt) = dut.accWrite.data.peek().litValue
          dut.clock.step()
          n += 1
        }
        require(n < 200, "mxu1 did not finish")
        rows.toMap
      }
      dut.reset.poke(true.B)
      idle()
      dut.clock.step(2)
      dut.reset.poke(false.B)
      for (t <- 0 until tiles) {
        // C as BF16 (its low 16 bits are 0): columns 0-15 in m[cLo], 16-31 in m[cLo + 1].
        for (r <- 0 until 32; half <- 0 until 2)
          write(cLo + half, r, (0 until 16).map(j => (c(t, r, 16 * half + j) >> 16) << (16 * j)).sum)
        command(MxuOp.PushAccBF16, cLo)
        drain()
        for (s <- 0 until steps) {
          for (r <- 0 until 32) write(wgt, r, w(t, s, r))
          command(MxuOp.PushWeight, wgt)
          drain()
          for (r <- 0 until 32) write(act, r, a(t, s, r))
          command(MxuOp.MatmulAcc, act)
          val rows = drain()
          for (r <- 0 until 32; j <- 0 until 32) {
            val got  = rows.get(r).map(x => (x >> (32 * j)) & 0xffffffffL)
            val want = (d(t, s, r) >> (32 * j)) & 0xffffffffL
            if (!got.contains(want)) {
              wrong += 1
              if (bad.size < 8) bad += f"tile $t step $s [$r][$j]: RTL ${got.fold("none")(_.toString(16))}, RTX 5090 ${want.toString(16)}"
            }
          }
        }
      }
    }
    println(s"${hdr("name").str}: $tiles tiles x $steps steps x 1024 outputs, $wrong differ")
    assert(wrong == 0, s"$wrong outputs differ, first: ${bad.mkString("; ")}")
  }
}
