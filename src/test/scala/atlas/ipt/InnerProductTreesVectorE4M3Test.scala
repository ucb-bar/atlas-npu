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
import java.nio.file.{Files, Path, Paths}
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import java.io.PrintWriter

object PersistentVcsVectorE4M3Simulator extends Simulator[VcsBackend] with PeekPokeAPI {
  private val runDir: Path = {
    val p = Paths.get("test_run_dir", "vector_e4m3_vcs")
    Files.createDirectories(p)
    p.toAbsolutePath
  }

  override val backend: VcsBackend = VcsBackend.initializeFromProcessEnvironment()
  override val tag: String = "vector_e4m3_vcs"
  override val workspacePath: String = runDir.toString

  override val commonCompilationSettings: CommonCompilationSettings =
    CommonCompilationSettings(
      availableParallelism =
        CommonCompilationSettings.AvailableParallelism.UpTo(Runtime.getRuntime.availableProcessors())
    )

  override val backendSpecificCompilationSettings: Backend.CompilationSettings = {
    val cov = Backend.CoverageSettings(line = true, cond = true, branch = true, fsm = true, tgl = true)
    Backend.CompilationSettings(
      coverageSettings = cov,
      coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
      simulationSettings = Backend.SimulationSettings(
        coverageSettings = cov,
        coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
        coverageName = Some(Backend.CoverageName("vector_e4m3_test_coverage"))
      )
    )
  }
}

class InnerProductTreesVectorE4M3Test extends AnyFlatSpec with Matchers with PeekPokeAPI {
  override def withFixture(test: NoArgTest): Outcome = {
    val o = super.withFixture(test)
    if (o.isFailed) println("InnerProductTreesVectorE4M3Test=FAILED")
    else if (o.isSucceeded) println("InnerProductTreesVectorE4M3Test=PASSED")
    o
  }

  val vectorResource = "/ipt_test_vectors/mxu_vectors_e4m3.txt"
  val WGT = 0
  val ACT = 2
  val BIAS = 4
  val PSUM_LO = 8
  val PSUM_HI = 9
  val OUT = 10

  case class TV(
    id: Int,
    ct: String,
    sel: Int,
    act: Seq[Int],
    wgt: Seq[Seq[Int]],
    bias: Seq[Int],
    psum: Seq[Int],
    scaleExp: Int,
    expE4M3: Seq[Int],
    reconBf16: Seq[Int]
  )

  def h2i(h: String): Int = Integer.parseUnsignedInt(h, 16)

  def s8h(h: String): Int = {
    val u = Integer.parseUnsignedInt(h, 16) & 0xFF
    if ((u & 0x80) != 0) u - 256 else u
  }

  def loadVectors(rp: String): Seq[TV] = {
    val s = Source.fromInputStream(getClass.getResourceAsStream(rp))
    val vs = ArrayBuffer[TV]()

    var id = 0
    var ct = ""
    var sel = 0
    var act = Seq.empty[Int]
    var wgt = ArrayBuffer[Seq[Int]]()
    var bias = Seq.empty[Int]
    var psum = Seq.empty[Int]
    var se = 0
    var ee = Seq.empty[Int]
    var rb = Seq.empty[Int]
    var in = false

    def f(): Unit = {
      if (in) {
        vs += TV(id, ct, sel, act, wgt.toSeq, bias, psum, se, ee, rb)
        wgt = ArrayBuffer[Seq[Int]]()
        act = Seq.empty
        bias = Seq.empty
        psum = Seq.empty
        se = 0
        ee = Seq.empty
        rb = Seq.empty
        in = false
      }
    }

    try {
      for (raw <- s.getLines()) {
        val l = raw.trim
        if (l.isEmpty) f()
        else if (l.startsWith("#")) {
          f()
          val p = l.drop(1).trim.split("\\s+")
          id = p(0).toInt
          ct = if (p.length > 1) p(1) else "?"
          in = true
        } else {
          val p = l.split("\\s+")
          p(0) match {
            case "sel"       => sel = p(1).toInt
            case "act"       => act = p.drop(1).map(h2i).toSeq
            case "wgt"       => wgt += p.drop(2).map(h2i).toSeq
            case "bias"      => bias = p.drop(1).map(h2i).toSeq
            case "psum"      => psum = p.drop(1).map(h2i).toSeq
            case "scale_exp" => se = s8h(p(1))
            case "exp_e4m3"  => ee = p.drop(1).map(h2i).toSeq
            case "recon_bf16"=> rb = p.drop(1).map(h2i).toSeq
            case _           =>
          }
        }
      }
      f()
    } finally s.close()

    vs.toSeq
  }

  def pack(es: Seq[Int], w: Int): BigInt =
    es.zipWithIndex.foldLeft(BigInt(0)) { case (a, (v, i)) =>
      a | (BigInt(v & ((1 << w) - 1)) << (i * w))
    }

  def idle(dut: InnerProductTreesUnitHarness): Unit = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.cmd.bits.op.poke(Mxu0Op.PushWeight)
    dut.io.cmd.bits.trfBank.poke(0.U)
    dut.io.cmd.bits.accSel.poke(false.B)
    dut.io.cmd.bits.weightSlot.poke(false.B)
    dut.io.cmd.bits.scaleE8M0.poke(127.U)

    dut.io.dmaWriteIn.valid.poke(false.B)
    dut.io.dmaWriteIn.bits.whichBank.poke(0.U)
    dut.io.dmaWriteIn.bits.wRow.poke(0.U)
    dut.io.dmaWriteIn.bits.wData.poke(0.U)

    dut.io.dmaReadIn.valid.poke(false.B)
    dut.io.dmaReadIn.bits.whichBank.poke(0.U)
    dut.io.dmaReadIn.bits.rRow.poke(0.U)
  }

  def w8(dut: InnerProductTreesUnitHarness, c: => Boolean, m: Int): Boolean = {
    var i = 0
    while (i < m && !c) {
      dut.clock.step()
      i += 1
    }
    c
  }

  def wr(dut: InnerProductTreesUnitHarness, b: Int, r: Int, d: BigInt): Unit = {
    dut.io.dmaWriteIn.valid.poke(true.B)
    dut.io.dmaWriteIn.bits.whichBank.poke(b.U)
    dut.io.dmaWriteIn.bits.wRow.poke(r.U)
    dut.io.dmaWriteIn.bits.wData.poke(d.U)
    dut.clock.step()
    dut.io.dmaWriteIn.valid.poke(false.B)
  }

  def rd(dut: InnerProductTreesUnitHarness, b: Int, r: Int): BigInt = {
    dut.io.dmaReadIn.valid.poke(true.B)
    dut.io.dmaReadIn.bits.whichBank.poke(b.U)
    dut.io.dmaReadIn.bits.rRow.poke(r.U)
    dut.clock.step()
    dut.io.dmaReadIn.valid.poke(false.B)
    require(dut.io.dmaReadOut.valid.peek().litToBoolean)
    dut.io.dmaReadOut.bits.peek().litValue
  }

  def cmd(
    dut: InnerProductTreesUnitHarness,
    op: Mxu0Op.Type,
    tb: Int = 0,
    as: Boolean = false,
    ws: Boolean = false,
    sc: Int = 127
  ): Unit = {
    dut.io.cmd.valid.poke(true.B)
    dut.io.cmd.bits.op.poke(op)
    dut.io.cmd.bits.trfBank.poke(tb.U)
    dut.io.cmd.bits.accSel.poke(as.B)
    dut.io.cmd.bits.weightSlot.poke(ws.B)
    dut.io.cmd.bits.scaleE8M0.poke(sc.U)
    dut.clock.step()
    dut.io.cmd.valid.poke(false.B)
  }

  "InnerProductTrees sequencer+TRF (VCS vectors, E4M3)" should "match E4M3-scaled Python ground truth" in {
    val p = InnerProductTreeParams()
    val rfP = RegFileParams()
    val vectors = loadVectors(vectorResource)
    require(vectors.nonEmpty)

    var passed = 0
    var failed = 0

    val outFile = new PrintWriter("../../../../../src/test/resources/ipt_test_vectors/rtl_vector_e4m3_outputs.txt")
    outFile.println("# case_id case_type row lane actual_hex expected_hex match")

    try {
      PersistentVcsVectorE4M3Simulator.simulate(new InnerProductTreesUnitHarness(p, rfP)) { module =>
        val dut = module.wrapped
        dut.reset.poke(true.B)
        dut.clock.step(5)
        dut.reset.poke(false.B)
        dut.clock.step(1)
        idle(dut)
        dut.clock.step(1)

        for (tv <- vectors) {
          require(tv.act.length == p.vecLen)
          require(tv.wgt.length == p.numLanes)
          idle(dut)

          for (lane <- 0 until p.numLanes)
            wr(dut, WGT, lane, pack(tv.wgt(lane), 8))

          cmd(dut, Mxu0Op.PushWeight, tb = WGT)
          require(w8(dut, !dut.io.dataBusy.peek().litToBoolean, 200))

          val ap = pack(tv.act, 8)
          for (r <- 0 until p.tileRows)
            wr(dut, ACT, r, ap)

          if (tv.sel == 1) {
            val bp = pack(tv.bias, 8)
            for (r <- 0 until p.tileRows)
              wr(dut, BIAS, r, bp)

            cmd(dut, Mxu0Op.PushAccFP8, tb = BIAS)
            require(w8(dut, !dut.io.dataBusy.peek().litToBoolean, 200))
          } else if (tv.sel == 2) {
            val pl = pack(tv.psum.take(16), 16)
            val ph = pack(tv.psum.drop(16), 16)
            for (r <- 0 until p.tileRows) {
              wr(dut, PSUM_LO, r, pl)
              wr(dut, PSUM_HI, r, ph)
            }

            cmd(dut, Mxu0Op.PushAccBF16, tb = PSUM_LO)
            require(w8(dut, !dut.io.dataBusy.peek().litToBoolean, 200))
          }

          val mop = if (tv.sel == 0) Mxu0Op.Matmul else Mxu0Op.MatmulAcc
          cmd(dut, mop, tb = ACT)
          require(w8(dut, !dut.io.computeBusy.peek().litToBoolean, p.tileRows + p.latency + 200))

          val e8m0 = (tv.scaleExp + 127) & 0xFF
          cmd(dut, Mxu0Op.PopAccFP8, tb = OUT, sc = e8m0)
          require(w8(dut, !dut.io.dataBusy.peek().litToBoolean, 200))

          var ok = true
          for (r <- 0 until p.tileRows) {
            val op = rd(dut, OUT, r)
            for (l <- 0 until p.numLanes) {
              val a = ((op >> (l * 8)) & 0xFF).toInt
              val e = tv.expE4M3(l) & 0xFF

              outFile.println(
                f"${tv.id}%8d ${tv.ct}%-16s $r%4d $l%4d 0x${a}%02x 0x${e}%02x ${if (a == e) "PASS" else "FAIL"}"
              )

              if (a != e) {
                ok = false
                println(f"FAIL case ${tv.id} [${tv.ct}] r$r l$l: e4m3=0x${a}%02x vs 0x${e}%02x")
              }
            }
          }

          outFile.flush()

          if (ok) passed += 1 else failed += 1
          idle(dut)
          dut.clock.step(2)
        }
      }
    } finally {
      outFile.close()
    }

    println(s"E4M3 vector: $passed passed, $failed failed out of ${vectors.length}")
    failed shouldBe 0
  }
}
