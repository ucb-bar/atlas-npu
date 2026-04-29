// ============================================================================
// SystolicArrayFullMatmulTest.scala — Multi-tile matmul tests for the
// systolic-array MXU under VCS with full code-coverage collection.
//
// RUN: (from sp26-atlas-acc) 
//    mill atlas.test.testOnly atlas.sa.SystolicArrayFullMatmulTest 
// ============================================================================

package atlas.sa

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
// VCS simulator — persistent workspace with coverage
// ============================================================================

object PersistentVcsSAFullMatmulSimulator extends Simulator[VcsBackend] with PeekPokeAPI {

  private val runDir: Path = {
    val rootDirStr = sys.env.getOrElse("MILL_WORKSPACE_ROOT", "/tmp")
    val baseDir = Paths.get(rootDirStr)
    val p = baseDir.resolve("tmp").resolve("SystolicArrayFullMatmulTest")
    Files.createDirectories(p)
    p.toAbsolutePath
  }

  override val backend: VcsBackend   = VcsBackend.initializeFromProcessEnvironment()
  override val tag: String           = "SystolicArrayFullMatmulTest"
  override val workspacePath: String = runDir.toString

  override val commonCompilationSettings: CommonCompilationSettings =
    CommonCompilationSettings(
      availableParallelism =
        CommonCompilationSettings.AvailableParallelism.UpTo(Runtime.getRuntime.availableProcessors())
    )

  override val backendSpecificCompilationSettings: Backend.CompilationSettings = {
    val cov = Backend.CoverageSettings(
      line = true, cond = true, branch = true, fsm = true, tgl = true, assert = true
    )
    Backend.CompilationSettings(
      coverageSettings  = cov,
      coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
      simulationSettings = Backend.SimulationSettings(
        coverageSettings  = cov,
        coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
        coverageName      = Some(Backend.CoverageName("SystolicArrayFullMatmulTest_coverage"))
      )
    )
  }
}

// ============================================================================
// Full-matmul test
// ============================================================================

class SystolicArrayFullMatmulTest extends AnyFlatSpec with Matchers with PeekPokeAPI {

  override def withFixture(test: NoArgTest): Outcome = {
    val o = super.withFixture(test)
    if (o.isFailed)         println("SystolicArrayFullMatmulTest=FAILED")
    else if (o.isSucceeded) println("SystolicArrayFullMatmulTest=PASSED")
    o
  }

  // ── Resource & tolerance configuration ──

  val vectorResource = "/mxu_test_vectors/sa_mxu_full_matmul_vectors.txt"

  val absTol = 0x0040
  val relTol = 0.01

  // ── MREG bank IDs ──

  val WGT    = 0
  val ACT    = 2
  val OUT_LO = 8
  val OUT_HI = 9

  // ── Data types ──

  case class FMV(
    id:  Int,
    nr:  Int,
    nt:  Int,
    wgt: Seq[Seq[Seq[Int]]],
    act: Seq[Seq[Seq[Int]]],
    exp: Seq[Seq[Int]]
  )

  case class Errs(absErr: Double, relErr: Double)

  // ── Numeric helpers ──

  def h2i(h: String): Int = Integer.parseUnsignedInt(h, 16)

  def bf16f(b: Int): Float = java.lang.Float.intBitsToFloat((b & 0xFFFF) << 16)

  private val relEps = java.lang.Float.MIN_NORMAL.toDouble

  def tolWithErr(a: Int, e: Int): (Boolean, Errs) = {
    val am = a & 0x7FFF
    val em = e & 0x7FFF
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

  def pack(es: Seq[Int], w: Int): BigInt =
    es.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
      acc | (BigInt(v & ((1 << w) - 1)) << (i * w))
    }

  // ── Vector file parser ──

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
            case _ =>
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

  def idle(dut: SystolicArrayUnitHarness): Unit = {
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

  def w8(dut: SystolicArrayUnitHarness, c: => Boolean, m: Int): Boolean = {
    var i = 0
    while (i < m && !c) { dut.clock.step(); i += 1 }
    c
  }

  def wr(dut: SystolicArrayUnitHarness, bank: Int, row: Int, data: BigInt): Unit = {
    dut.io.testWrite.valid.poke(true.B)
    dut.io.testWrite.bits.mregId.poke(bank.U)
    dut.io.testWrite.bits.row.poke(row.U)
    dut.io.testWrite.bits.data.poke(data.U)
    dut.clock.step()
    dut.io.testWrite.valid.poke(false.B)
  }

  def rd(dut: SystolicArrayUnitHarness, bank: Int, row: Int): BigInt = {
    dut.io.testRead.valid.poke(true.B)
    dut.io.testRead.bits.mregId.poke(bank.U)
    dut.io.testRead.bits.row.poke(row.U)
    dut.clock.step()
    dut.io.testRead.valid.poke(false.B)
    require(dut.io.testReadOut.valid.peek().litToBoolean)
    dut.io.testReadOut.bits.peek().litValue
  }

  def cmd(
    dut: SystolicArrayUnitHarness,
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

  "SystolicArray sequencer+MREG (VCS full matmul)" should "compute full matmul correctly" in {
    val p       = SystolicArrayParams()
    val mregP   = MregParams()
    val saLat   = p.rows + p.cols - 2   // systolic drain latency
    val vectors = loadVectors(vectorResource)
    require(vectors.nonEmpty)

    var totalPassed = 0
    var totalFailed = 0

    val outFile = new PrintWriter(
      "../../../../../src/test/resources/mxu_test_vectors/sa_rtl_full_matmul_outputs.txt"
    )
    outFile.println(
      "# case_id row lane actual_hex expected_hex actual_float expected_float abs_err rel_err match"
    )

    try {
      PersistentVcsSAFullMatmulSimulator.simulate(new SystolicArrayUnitHarness(p, mregP)) { module =>
        val dut = module.wrapped

        // Reset sequence
        dut.reset.poke(true.B)
        dut.clock.step(5)
        dut.reset.poke(false.B)
        dut.clock.step(1)
        idle(dut)
        dut.clock.step(1)

        for (tv <- vectors) {
          require(tv.nr == p.accSize)
          println(f"═══ Case ${tv.id}: ${tv.nr} rows × ${tv.nt} tile(s) ═══")

          // ── Feed each tile ──
          for (t <- 0 until tv.nt) {
            idle(dut)

            for (lane <- 0 until p.cols)
              wr(dut, WGT, lane, pack(tv.wgt(t)(lane), 8))
            cmd(dut, MxuOp.PushWeight, tb = WGT)
            require(w8(dut, !dut.io.dataBusy.peek().litToBoolean, 200))

            for (r <- 0 until tv.nr)
              wr(dut, ACT, r, pack(tv.act(t)(r), 8))

            val op = if (t == 0) MxuOp.Matmul else MxuOp.MatmulAcc
            cmd(dut, op, tb = ACT)
            require(w8(dut, !dut.io.computeBusy.peek().litToBoolean, p.accSize + saLat + 200))
          }

          // ── Pop BF16 results and compare ──
          cmd(dut, MxuOp.PopAccBF16, tb = OUT_LO)
          require(w8(dut, !dut.io.dataBusy.peek().litToBoolean, 200))

          var cp = 0
          var cf = 0

          for (r <- 0 until tv.nr) {
            val ol = rd(dut, OUT_LO, r)
            val oh = rd(dut, OUT_HI, r)

            for (l <- 0 until p.cols) {
              val a = if (l < 16) ((ol >> (l * 16)) & 0xFFFF).toInt
                      else        ((oh >> ((l - 16) * 16)) & 0xFFFF).toInt
              val e = tv.exp(r)(l) & 0xFFFF

              val af         = bf16f(a).toDouble
              val ef         = bf16f(e).toDouble
              val (ok, errs) = tolWithErr(a, e)

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
            f"  Result: $cp/${tv.nr * p.cols} passed" +
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
