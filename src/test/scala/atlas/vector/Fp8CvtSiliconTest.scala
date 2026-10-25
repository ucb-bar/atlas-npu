// ============================================================================
// Fp8CvtSiliconTest.scala — FP8Pack, FP8Unpack and Cvt against the RTX 5090
// conversion vectors (baremetal/generators/fp8_silicon/*.cvt.gz, README.md).
//
// FP8Pack: every BF16 x at every e (2^24). FP8Unpack: every E4M3 x at every e.
// Cvt with FP32 inputs: the pop set, both ends of every run of the E4M3 and
// BF16 conversions of all 2^32 FP32 x, and E4M3 to FP32 of every code.
//
// RUN: mill atlas.test.testOnly atlas.vector.Fp8CvtSiliconTest   (Verilator)
// ============================================================================
package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import atlas.common.Cvt
import org.scalatest.flatspec.AnyFlatSpec
import sp26FPUnits.AtlasFPType.BF16
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream
import scala.collection.mutable.ArrayBuffer

/** FP8Pack with its 16 lanes in one word, so that one poke drives a row. */
class FP8PackRows extends Module {
  val io = IO(new Bundle {
    val valid = Input(Bool())
    val scale = Input(UInt(8.W))
    val x     = Input(UInt(256.W))
    val out   = Output(UInt(256.W))
  })
  val pack = Module(new FP8Pack(BF16))
  pack.io.req.valid      := io.valid
  pack.io.req.bits.scale := io.scale
  pack.io.req.bits.xVec  := io.x.asTypeOf(pack.io.req.bits.xVec)
  io.out := pack.io.resp.bits.result.asUInt
}

/** FP8Unpack with its 16 lanes in one word. */
class FP8UnpackRows extends Module {
  val io = IO(new Bundle {
    val valid    = Input(Bool())
    val scale    = Input(UInt(8.W))
    val x        = Input(UInt(256.W))
    val outValid = Output(Bool())
    val out      = Output(UInt(256.W))
  })
  val unpack = Module(new FP8Unpack(BF16))
  unpack.io.req.valid      := io.valid
  unpack.io.req.bits.scale := io.scale
  unpack.io.req.bits.xVec  := io.x.asTypeOf(unpack.io.req.bits.xVec)
  io.outValid := unpack.io.resp.valid
  io.out      := unpack.io.resp.bits.result.asUInt
}

/** The Cvt functions on 16 FP32 lanes, each with its own e. */
class CvtLanes extends Module {
  val io = IO(new Bundle {
    val x    = Input(Vec(16, UInt(32.W)))
    val e    = Input(Vec(16, UInt(8.W)))
    val e4m3 = Output(Vec(16, UInt(8.W)))
    val bf16 = Output(Vec(16, UInt(16.W)))
    val f32  = Output(Vec(16, UInt(32.W)))
  })
  io.e4m3 := VecInit(io.x.zip(io.e).map { case (x, e) => Cvt.e4m3RnSatfinite(x, e) })
  io.bf16 := VecInit(io.x.zip(io.e).map { case (x, e) => Cvt.bf16Rne(x, e) })
  io.f32  := VecInit(io.x.map(x => Cvt.e4m3ToF32(x(7, 0))))
}

class Fp8CvtSiliconTest extends AnyFlatSpec {
  private val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT", sys.props("user.dir")))

  /** The columns of a cvt-vectors-v1 file, each as little-endian unsigned values. */
  private def readCvt(name: String): Map[String, Array[Long]] = {
    val path = root.resolve(s"baremetal/generators/fp8_silicon/$name.cvt.gz")
    val raw  = new GZIPInputStream(Files.newInputStream(path)).readAllBytes()
    val nl   = raw.indexOf('\n'.toByte)
    val hdr  = ujson.read(new String(raw, 0, nl, "UTF-8"))
    val n    = hdr("count").num.toInt
    val buf  = ByteBuffer.wrap(raw, nl + 1, raw.length - nl - 1).slice().order(ByteOrder.LITTLE_ENDIAN)
    hdr("columns").arr.map { c =>
      val get: () => Long = c(1).str.last match {
        case '1' => () => buf.get() & 0xffL
        case '2' => () => buf.getShort() & 0xffffL
        case '4' => () => buf.getInt() & 0xffffffffL
      }
      c(0).str -> Array.fill(n)(get())
    }.toMap
  }

  private class Mismatches(what: String) {
    private var count = 0L
    private val first = ArrayBuffer[String]()
    def check(label: => String, got: Long, want: Long): Unit =
      if (got != want) {
        count += 1
        if (first.size < 8) first += f"$label: RTL 0x$got%x, RTX 5090 0x$want%x"
      }
    def assertNone(): Unit =
      assert(count == 0, s"$what: $count mismatches, first: ${first.mkString("; ")}")
  }

  private def reset(dut: Module): Unit = {
    dut.reset.poke(true.B)
    dut.clock.step()
    dut.reset.poke(false.B)
  }

  private def word(values: Seq[Long], bits: Int): BigInt =
    values.zipWithIndex.foldLeft(BigInt(0)) { case (w, (v, i)) => w | (BigInt(v) << (bits * i)) }

  private def field(w: BigInt, bits: Int, i: Int): Long = ((w >> (bits * i)) & ((BigInt(1) << bits) - 1)).toLong

  behavior of "FP8 conversions"

  it should "give the RTX 5090 result in FP8Pack for every BF16 x at every e" in {
    val want = readCvt("vpack")("rtx5090")
    val bad  = new Mismatches("vpack")
    simulate(new FP8PackRows) { dut =>
      reset(dut)
      dut.io.valid.poke(true.B)
      for (e <- 0 until 256) {
        dut.io.scale.poke(e.U)
        for (base <- 0 until 65536 by 32) {
          dut.io.x.poke(word((base until base + 16).map(_.toLong), 16).U)
          dut.clock.step()
          dut.io.x.poke(word((base + 16 until base + 32).map(_.toLong), 16).U)
          dut.clock.step()
          val out = dut.io.out.peek().litValue
          for (k <- 0 until 32)
            bad.check(f"x 0x${base + k}%04x e $e", field(out, 8, k), want((e << 16) | (base + k)))
        }
      }
    }
    bad.assertNone()
  }

  it should "give the RTX 5090 result in FP8Unpack for every E4M3 x at every e" in {
    val want = readCvt("vunpack")("rtx5090")
    val bad  = new Mismatches("vunpack")
    simulate(new FP8UnpackRows) { dut =>
      reset(dut)
      for (e <- 0 until 256; base <- 0 until 256 by 32) {
        dut.io.valid.poke(true.B)
        dut.io.scale.poke(e.U)
        dut.io.x.poke(word((base until base + 32).map(_.toLong), 8).U)
        dut.clock.step()
        dut.io.valid.poke(false.B)
        val beats = ArrayBuffer[BigInt]()
        while (beats.size < 2) {
          if (dut.io.outValid.peek().litToBoolean) beats += dut.io.out.peek().litValue
          dut.clock.step()
        }
        for (k <- 0 until 32)
          bad.check(f"x 0x${base + k}%02x e $e", field(beats(k / 16), 16, k % 16), want((e << 8) | (base + k)))
      }
    }
    bad.assertNone()
  }

  it should "give the RTX 5090 results in Cvt for FP32 inputs" in {
    val pop  = readCvt("pop")
    val push = readCvt("push")("rtx5090")
    val bad  = new Mismatches("Cvt")
    // Both ends of each run, at e = 127; the last run ends at 0xFFFFFFFF.
    def ends(name: String): IndexedSeq[(Long, Long)] = {
      val runs = readCvt(name)
      val x    = runs("x")
      x.indices.flatMap { k =>
        val last = if (k + 1 < x.length) x(k + 1) - 1 else 0xffffffffL
        Seq(x(k), last).map(_ -> runs("rtx5090")(k))
      }
    }
    val e4m3 = ends("e4m3_runs")
    val bf16 = ends("bf16_runs")
    simulate(new CvtLanes) { dut =>
      // Drives n inputs (x(i), e(i)), 16 at a time; check(i, lane) reads the outputs of input i.
      def sweep(n: Int, x: Int => Long, e: Int => Long)(check: (Int, Int) => Unit): Unit =
        for (base <- 0 until n by 16) {
          for (lane <- 0 until 16) {
            val i = (base + lane).min(n - 1)
            dut.io.x(lane).poke(x(i).U(32.W))
            dut.io.e(lane).poke(e(i).U(8.W))
          }
          dut.clock.step()
          for (lane <- 0 until 16 if base + lane < n) check(base + lane, lane)
        }
      def peek(port: UInt): Long = port.peek().litValue.toLong
      sweep(pop("x").length, pop("x")(_), pop("e")(_)) { (i, lane) =>
        bad.check(f"pop x 0x${pop("x")(i)}%08x e ${pop("e")(i)}", peek(dut.io.e4m3(lane)), pop("rtx5090")(i))
      }
      sweep(e4m3.length, e4m3(_)._1, _ => 127) { (i, lane) =>
        bad.check(f"e4m3 x 0x${e4m3(i)._1}%08x", peek(dut.io.e4m3(lane)), e4m3(i)._2)
      }
      sweep(bf16.length, bf16(_)._1, _ => 127) { (i, lane) =>
        bad.check(f"bf16 x 0x${bf16(i)._1}%08x", peek(dut.io.bf16(lane)), bf16(i)._2)
      }
      sweep(256, _.toLong, _ => 127) { (i, lane) =>
        bad.check(f"push x 0x$i%02x", peek(dut.io.f32(lane)), push(i))
      }
    }
    bad.assertNone()
  }
}
