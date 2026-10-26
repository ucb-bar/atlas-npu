package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Random

/** vredsum.row.bf16 adds the 32 elements of each row of {m[vs1], m[vs1+1]} in a balanced tree:
  * element j is lane j of m[vs1] for j < 16 and lane j - 16 of m[vs1+1] for j >= 16, the tree adds
  * elements 2i and 2i+1, then their sums in the same way. Each sum is an IEEE FP32 addition, nearest
  * even (Inf + -Inf is NaN, a finite sum above the FP32 range is Inf), and the root rounds once to
  * BF16, nearest even. NaN results are 0x7FC0. Rows with special values, and random rows.
  */
class VectorEngineRowSumTest extends AnyFlatSpec {
  import VpuBroadcastHarness._

  private val rsum = 14
  private val (inf, ninf, max, nmax) = (0x7F80, 0xFF80, 0x7F7F, 0xFF7F)
  private val specials = Seq(0x0000, 0x8000, 0x0001, 0x8001, inf, ninf, 0x7FC0, 0xFFC1, 0x7F81, 0x3F80,
    0xBF80, max, nmax, 0x0080, 0x4B80, 0xCB80)

  private def float(bf16: Int): Float = java.lang.Float.intBitsToFloat(bf16 << 16)

  private def bf16(f: Float): Int =
    if (f.isNaN) 0x7FC0
    else {
      val b = java.lang.Float.floatToRawIntBits(f)
      ((b + 0x7FFF + ((b >>> 16) & 1)) >>> 16) & 0xFFFF
    }

  private def tree(xs: Seq[Float]): Float = if (xs.size == 1) xs.head else tree(xs.grouped(2).map(_.reduce(_ + _)).toSeq)

  /** Row elements (32) and, for some, the sum they must give. */
  private val directed: Seq[(Seq[Int], Option[Int])] = {
    def at(values: (Int, Int)*): Seq[Int] = Seq.tabulate(32)(j => values.toMap.getOrElse(j, 0))
    Seq(
      at(0 -> inf, 1 -> nmax, 2 -> nmax, 3 -> nmax) -> Some(0x7FC0),  // Inf + (-max + -max = -Inf)
      at(0 -> max, 1 -> max, 2 -> max, 3 -> max) -> Some(inf),        // finite sums overflow to Inf
      at(0 -> max, 1 -> max, 2 -> nmax, 3 -> nmax) -> Some(0x7FC0),   // Inf + -Inf from finite sums
      at(0 -> max, 1 -> nmax, 2 -> max, 3 -> nmax) -> Some(0x0000),
      at(0 -> max, 16 -> max) -> Some(inf),                           // the root overflows
      at(0 -> inf, 31 -> ninf) -> Some(0x7FC0),
      at(0 -> inf, 1 -> ninf) -> Some(0x7FC0),
      at(5 -> inf, 9 -> inf, 20 -> max) -> Some(inf),
      at(7 -> 0xFFC1) -> Some(0x7FC0),                                // a NaN payload is not kept
      at(7 -> 0x7F81) -> Some(0x7FC0),                                // signaling NaN
      Seq.fill(32)(0x8000) -> Some(0x8000),
      (Seq.fill(31)(0x8000) :+ 0x0000) -> Some(0x0000),
      Seq.fill(32)(0x0001) -> Some(0x0020),                           // 32 x 2^-133 = 2^-128
      at(0 -> 0x0001, 1 -> 0x8001) -> Some(0x0000),
      // 2^24 + 1 rounds to 2^24 in FP32, -2^24 + 1 is exact: the tree gives 1, left to right gives 0.
      at(0 -> 0x4B80, 1 -> 0x3F80, 2 -> 0xCB80, 3 -> 0x3F80) -> Some(0x3F80),
      // 1 + 2^-8 is exact in FP32 and a BF16 tie: nearest even gives 1.
      at(0 -> 0x3F80, 1 -> 0x3B80) -> Some(0x3F80),
      at(0 -> 0x3F80, 1 -> 0x3B80, 2 -> 0x3780) -> Some(0x3F81)       // above the tie
    )
  }

  "vredsum.row.bf16" should "add each row in the tree order in FP32 and round to BF16 once" in {
    for ((row, want) <- directed; w <- want)
      assert(bf16(tree(row.map(float))) == w, f"the reference gives the sum 0x$w%04x of ${row.map(v => f"$v%04x")}")
    val rng = new Random(11)
    val rows = directed.map(_._1) ++ Seq.fill(64 - directed.size)(Seq.fill(32)(
      if (rng.nextInt(3) == 0) specials(rng.nextInt(specials.size)) else rng.nextInt(1 << 16)))
    simulate(new VpuBroadcastHarness) { dut =>
      for (part <- rows.grouped(32)) {
        val mem: Mem = Array.tabulate(64, 32) { (r, i) =>
          if (r > 1 || i >= part.size) BigInt(0)
          else (0 until 16).map(lane => BigInt(part(i)(16 * r + lane)) << (16 * lane)).sum
        }
        drive(dut, mem, Map(0 -> Cmd(rsum, 4, 0)))
        for ((row, i) <- part.zipWithIndex) {
          val want = bf16(tree(row.map(float)))
          val got = (0 until 32).map(j => ((mem(4 + j / 16)(i) >> (16 * (j % 16))) & 0xFFFF).toInt)
          assert(got.forall(_ == want), f"row ${row.map(v => f"$v%04x")}: 0x$want%04x, RTL ${got.map(v => f"$v%04x")}")
        }
      }
    }
  }
}
