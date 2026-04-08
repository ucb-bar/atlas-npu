// ============================================================================
// InnerProductTreesFullMatmulTest.scala — Multi-tile matmul tests for the
// inner product trees MXU under VCS with full code-coverage collection.
//
// RUN: (from sp26-atlas-acc) 
//    mill atlas.test.testOnly atlas.ipt.InnerProductTreesFullMatmulTest 
// ============================================================================

package atlas.ipt

import chisel3._
import chisel3.util._
import chisel3.simulator._
import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common._
import atlas.mxu.MxuOp
import java.nio.file.{Files, Path, Paths}
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import scala.util.Try
import java.io.PrintWriter

// ============================================================================
// VCS simulator for full-matmul tests — persistent workspace under
// test_run_dir/full_matmul_vcs with full code-coverage collection.
// ============================================================================

object PersistentVcsFullMatmulSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val runDir: Path = {
    val p = Paths.get("test_run_dir", "full_matmul_vcs")
    Files.createDirectories(p)
    p.toAbsolutePath
  }

  override val backend: VcsBackend = VcsBackend.initializeFromProcessEnvironment()
  override val tag: String         = "full_matmul_vcs"
  override val workspacePath: String = runDir.toString

  override val commonCompilationSettings: CommonCompilationSettings =
    CommonCompilationSettings(
      availableParallelism =
        CommonCompilationSettings.AvailableParallelism.UpTo(Runtime.getRuntime.availableProcessors())
    )

  override val backendSpecificCompilationSettings: Backend.CompilationSettings = {
    val cov = Backend.CoverageSettings(
      line = true, cond = true, branch = true, fsm = true, tgl = true
    )
    Backend.CompilationSettings(
      coverageSettings  = cov,
      coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
      simulationSettings = Backend.SimulationSettings(
        coverageSettings  = cov,
        coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
        coverageName      = Some(Backend.CoverageName("full_matmul_test_coverage"))
      )
    )
  }
}

// ============================================================================
// Full-matmul test — drives multi-tile matmul sequences through the IPT +
// MREG harness under VCS and compares BF16 outputs to Python golden vectors.
// ============================================================================

class InnerProductTreesFullMatmulTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

  override def withFixture(test: NoArgTest): Outcome = {
    val o = super.withFixture(test)
    if (o.isFailed)    println("InnerProductTreesFullMatmulTest=FAILED")
    else if (o.isSucceeded) println("InnerProductTreesFullMatmulTest=PASSED")
    o
  }

  // ── Resource & tolerance configuration ──

  val vectorResource = "/mxu_test_vectors/ipt_mxu_full_matmul_vectors.txt"

  /** Absolute tolerance: BF16 magnitudes at or below this are treated as zero. */
  val absTol = 0x0040

  /** Relative tolerance: maximum allowed relative error for non-tiny values. */
  val relTol = 0.01

  // MREG bank IDs used for weights, activations, and output halves.
  val WGT    = 0
  val ACT    = 2
  val OUT_LO = 8
  val OUT_HI = 9

  // ── Data types ──

  /** Full-matmul vector: one test case with multiple tiles. */
  case class FMV(
    id:  Int,
    nr:  Int,                    // number of rows (should match tileRows)
    nt:  Int,                    // number of tiles to accumulate
    wgt: Seq[Seq[Seq[Int]]],    // wgt(tile)(lane)(elem)
    act: Seq[Seq[Seq[Int]]],    // act(tile)(row)(elem)
    exp: Seq[Seq[Int]]          // exp(row)(lane) — expected BF16 outputs
  )

  /** Error metrics for a single lane comparison. */
  case class Errs(absErr: Double, relErr: Double)

  // ── Numeric helpers ──

  /** Parse an unsigned hex string to Int. */
  def h2i(h: String): Int = Integer.parseUnsignedInt(h, 16)

  /** Reinterpret a 16-bit BF16 pattern as a Float. */
  def bf16f(b: Int): Float = java.lang.Float.intBitsToFloat((b & 0xFFFF) << 16)

  /** Smallest positive normal float — used as a floor to avoid division by zero. */
  private val relEps = java.lang.Float.MIN_NORMAL.toDouble

  /**
   * Compare actual vs expected BF16 values with combined absolute + relative
   * tolerance.  Returns (pass, Errs).
   */
  def tolWithErr(a: Int, e: Int): (Boolean, Errs) = {
    val am = a & 0x7FFF
    val em = e & 0x7FFF

    // Both values tiny → pass unconditionally.
    if (am <= absTol && em <= absTol) {
      (true, Errs(0.0, 0.0))
    } else {
      val af     = bf16f(a).toDouble
      val ef     = bf16f(e).toDouble
      val absErr = math.abs(af - ef)
      val relErr = absErr / math.max(math.abs(ef), relEps)
      (relErr < relTol, Errs(absErr, relErr))
    }
  }

  def tol(a: Int, e: Int): Boolean = tolWithErr(a, e)._1

  /** Pack a vector of small integers into a single BigInt, each `w` bits wide. */
  def pack(es: Seq[Int], w: Int): BigInt =
    es.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
      acc | (BigInt(v & ((1 << w) - 1)) << (i * w))
    }

  // ── Vector file parser ──

  /**
   * Load full-matmul test vectors from a resource file.
   *
   * File format (whitespace-separated):
   *   # <id>                          — start a new test case
   *   num_rows <n>
   *   num_tiles <n>
   *   wgt <tile> <lane> <hex>...      — weight row for (tile, lane)
   *   act <tile> <row>  <hex>...      — activation row for (tile, row)
   *   exp <row> <hex>...              — expected BF16 outputs for row
   *   <blank line>                    — flush current case
   */
  def loadVectors(rp: String): Seq[FMV] = {
    val s  = Source.fromInputStream(getClass.getResourceAsStream(rp))
    val vs = ArrayBuffer[FMV]()

    var id = 0
    var nr = 32
    var nt = 1
    var wm = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Seq[Int]]]()
    var am = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Seq[Int]]]()
    var em = scala.collection.mutable.Map[Int, Seq[Int]]()
    var in = false

    def flush(): Unit = {
      if (!in) return
      val w = (0 until nt).map { t =>
        val m = wm.getOrElse(t, scala.collection.mutable.Map.empty)
        (0 until m.size).map(r => m(r))
      }
      val a = (0 until nt).map { t =>
        val m = am.getOrElse(t, scala.collection.mutable.Map.empty)
        (0 until m.size).map(i => m(i))
      }
      val e = (0 until em.size).map(i => em(i))
      vs += FMV(id, nr, nt, w, a, e)

      wm = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Seq[Int]]]()
      am = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Seq[Int]]]()
      em = scala.collection.mutable.Map[Int, Seq[Int]]()
      in = false
    }

    try {
      for (raw <- s.getLines()) {
        val l = raw.trim
        if (l.isEmpty) {
          flush()
        } else if (l.startsWith("#")) {
          flush()
          val p = l.drop(1).trim.split("\\s+")
          id = p(0).toInt
          in = true
        } else {
          val p = l.split("\\s+")
          p(0) match {
            case "num_rows"  => nr = p(1).toInt
            case "num_tiles" => nt = p(1).toInt
            case "wgt" =>
              wm.getOrElseUpdate(p(1).toInt, scala.collection.mutable.Map.empty)(p(2).toInt) =
                p.drop(3).map(h2i).toSeq
            case "act" =>
              am.getOrElseUpdate(p(1).toInt, scala.collection.mutable.Map.empty)(p(2).toInt) =
                p.drop(3).map(h2i).toSeq
            case "exp" =>
              em(p(1).toInt) = p.drop(2).map(h2i).toSeq
            case _ => // ignore unknown keys
          }
        }
      }
      flush()
    } finally {
      s.close()
    }
    vs.toSeq
  }

  // ── DUT helpers ──

  /** Drive all command and backdoor ports to idle / zero. */
  def idle(dut: InnerProductTreesUnitHarness): Unit = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.cmd.bits.op.poke(MxuOp.PushWeight)
    dut.io.cmd.bits.mregId.poke(0.U)
    dut.io.cmd.bits.accSel.poke(false.B)
    dut.io.cmd.bits.weightSlot.poke(false.B)
    dut.io.cmd.bits.scaleE8M0.poke(127.U)

    dut.io.testWrite.valid.poke(false.B)
    dut.io.testWrite.bits.mregId.poke(0.U)
    dut.io.testWrite.bits.row.poke(0.U)
    dut.io.testWrite.bits.data.poke(0.U)

    dut.io.testRead.valid.poke(false.B)
    dut.io.testRead.bits.mregId.poke(0.U)
    dut.io.testRead.bits.row.poke(0.U)
  }

  /** Spin until `c` is true or `m` cycles elapse; return final value of `c`. */
  def w8(dut: InnerProductTreesUnitHarness, c: => Boolean, m: Int): Boolean = {
    var i = 0
    while (i < m && !c) { dut.clock.step(); i += 1 }
    c
  }

  /** Backdoor write one MREG row. */
  def wr(dut: InnerProductTreesUnitHarness, bank: Int, row: Int, data: BigInt): Unit = {
    dut.io.testWrite.valid.poke(true.B)
    dut.io.testWrite.bits.mregId.poke(bank.U)
    dut.io.testWrite.bits.row.poke(row.U)
    dut.io.testWrite.bits.data.poke(data.U)
    dut.clock.step()
    dut.io.testWrite.valid.poke(false.B)
  }

  /** Backdoor read one MREG row (returns the full row as BigInt). */
  def rd(dut: InnerProductTreesUnitHarness, bank: Int, row: Int): BigInt = {
    dut.io.testRead.valid.poke(true.B)
    dut.io.testRead.bits.mregId.poke(bank.U)
    dut.io.testRead.bits.row.poke(row.U)
    dut.clock.step()
    dut.io.testRead.valid.poke(false.B)
    require(dut.io.testReadOut.valid.peek().litToBoolean)
    dut.io.testReadOut.bits.peek().litValue
  }

  /** Issue a single-cycle MXU command. */
  def cmd(
    dut: InnerProductTreesUnitHarness,
    op:  MxuOp.Type,
    tb:  Int     = 0,
    as:  Boolean = false,
    ws:  Boolean = false,
    sc:  Int     = 127
  ): Unit = {
    dut.io.cmd.valid.poke(true.B)
    dut.io.cmd.bits.op.poke(op)
    dut.io.cmd.bits.mregId.poke(tb.U)
    dut.io.cmd.bits.accSel.poke(as.B)
    dut.io.cmd.bits.weightSlot.poke(ws.B)
    dut.io.cmd.bits.scaleE8M0.poke(sc.U)
    dut.clock.step()
    dut.io.cmd.valid.poke(false.B)
  }

  // ── Main test ──

  "InnerProductTrees sequencer+MREG (VCS full matmul)" should "compute full matmul correctly" in {
    val p       = InnerProductTreeParams()
    val mregP   = MregParams()
    val vectors = loadVectors(vectorResource)
    require(vectors.nonEmpty)

    var totalPassed = 0
    var totalFailed = 0

    val outFile = new PrintWriter(
      "../../../../../src/test/resources/mxu_test_vectors/ipt_rtl_full_matmul_outputs.txt"
    )
    outFile.println(
      "# case_id row lane actual_hex expected_hex actual_float expected_float abs_err rel_err match"
    )

    try {
      PersistentVcsFullMatmulSimulator.simulate(new InnerProductTreesUnitHarness(p, mregP)) { module =>
        val dut = module.wrapped

        // Reset sequence
        dut.reset.poke(true.B)
        dut.clock.step(5)
        dut.reset.poke(false.B)
        dut.clock.step(1)
        idle(dut)
        dut.clock.step(1)

        for (tv <- vectors) {
          require(tv.nr == p.tileRows)
          println(f"═══ Case ${tv.id}: ${tv.nr} rows × ${tv.nt} tile(s) ═══")

          // ── Feed each tile: push weights, load activations, matmul ──
          for (t <- 0 until tv.nt) {
            idle(dut)

            // Load weight tile into MREG, then push into the weight buffer.
            for (lane <- 0 until p.numLanes)
              wr(dut, WGT, lane, pack(tv.wgt(t)(lane), 8))
            cmd(dut, MxuOp.PushWeight, tb = WGT)
            require(w8(dut, !dut.io.dataBusy.peek().litToBoolean, 200))

            // Load activation tile.
            for (r <- 0 until tv.nr)
              wr(dut, ACT, r, pack(tv.act(t)(r), 8))

            // First tile uses Matmul (clear acc); subsequent tiles accumulate.
            val op = if (t == 0) MxuOp.Matmul else MxuOp.MatmulAcc
            cmd(dut, op, tb = ACT)
            require(w8(dut, !dut.io.computeBusy.peek().litToBoolean, p.tileRows + p.latency + 200))
          }

          // ── Pop BF16 results and compare ──
          cmd(dut, MxuOp.PopAccBF16, tb = OUT_LO)
          require(w8(dut, !dut.io.dataBusy.peek().litToBoolean, 200))

          var cp = 0
          var cf = 0

          for (r <- 0 until tv.nr) {
            val ol = rd(dut, OUT_LO, r)
            val oh = rd(dut, OUT_HI, r)

            for (l <- 0 until p.numLanes) {
              // Extract BF16 value from the low or high half of the row.
              val a = if (l < 16) ((ol >> (l * 16)) & 0xFFFF).toInt
                      else        ((oh >> ((l - 16) * 16)) & 0xFFFF).toInt
              val e = tv.exp(r)(l) & 0xFFFF

              val af          = bf16f(a).toDouble
              val ef          = bf16f(e).toDouble
              val (ok, errs)  = tolWithErr(a, e)

              outFile.println(
                f"${tv.id}%8d $r%4d $l%4d 0x${a}%04x 0x${e}%04x " +
                f"${af}%14.5f ${ef}%14.5f ${errs.absErr}%10.5f ${errs.relErr}%10.5f " +
                f"${if (ok) "PASS" else "FAIL"}"
              )

              if (ok) {
                cp += 1
              } else {
                cf += 1
                println(
                  f"  FAIL r$r l$l: 0x${a}%04x(${af}%.6g) vs 0x${e}%04x(${ef}%.6g), " +
                  f"rel_err=${errs.relErr}%.6e"
                )
              }
            }
          }

          outFile.flush()
          println(
            f"  Result: $cp/${tv.nr * p.numLanes} passed" +
            (if (cf > 0) f", $cf FAILED" else "")
          )

          totalPassed += cp
          totalFailed += cf
          idle(dut)
          dut.clock.step(2)
        }
      }
    } finally {
      outFile.close()
    }

    println(s"\nTotal: $totalPassed passed, $totalFailed failed")
    totalFailed shouldBe 0
  }
}
