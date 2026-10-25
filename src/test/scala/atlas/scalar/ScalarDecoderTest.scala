package atlas.scalar

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import chisel3.util.BitPat
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._
import scala.util.Random

/** The decoder agrees with the ISA table (Instructions.all, generated from atlas_isa.toml). Words
  * with random operands decode as their instruction, legal and with its unit. A word one bit away
  * from an instruction's match word decodes as the table says: a word that sets a bit that the
  * table fixes, for example a reserved operand bit, is illegal unless it is another instruction.
  * The decoder's x-register fields (rd_x, rs1_ren, rs2_ren) are the table's x operands, and a .vx
  * or .vr form has the vpu_cmd of its .bf16 form.
  *
  * Each instruction's control vector equals its line in the golden table
  * src/test/resources/scalar_decode.txt. After a change of IDecode, write the table again and
  * review its diff:
  *   ATLAS_DECODE_GOLDEN=write ./mill atlas.test.testOnly atlas.scalar.ScalarDecoderTest
  */
class ScalarDecoderTest extends AnyFlatSpec {
  import ScalarISA._

  private val golden = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT", sys.props("user.dir")))
    .resolve("src/test/resources/scalar_decode.txt")
  private val header = Seq(
    "# The ScalarDecoder control vector of each ISA table entry, checked by ScalarDecoderTest.",
    "# A line has the entry's name and each field that is not 0, by its ScalarISA name (ALU_SUB:",
    "# alu=SUB); rd_wen, jal, jalr, rd_x, rs1_ren and rs2_ren if set; mxu1= or nv= and the MXU",
    "# command (funct7[3:0]).",
    "# Write it again after a change of IDecode, and review the diff:",
    "#   ATLAS_DECODE_GOLDEN=write ./mill atlas.test.testOnly atlas.scalar.ScalarDecoderTest")

  private def entry(word: BigInt): Option[(String, String)] =
    Instructions.all.collectFirst { case (name, p, unit) if (word & p.mask) == p.value => (name, unit) }

  /** The unit that the decoded word issues to, or None if it is illegal. */
  private def decodedUnit(d: DecodedInstr): Option[String] = {
    val units = Seq(
      "mxu1"   -> (d.is_mxu.peek().litToBoolean && d.mxu_sel.peek().litValue == 1),
      "nv"     -> (d.is_mxu.peek().litToBoolean && d.mxu_sel.peek().litValue == 0),
      "vpu"    -> (d.vpu_cmd.peek().litValue != 0),
      "xlu"    -> (d.xlu_cmd.peek().litValue != 0),
      "dma"    -> (d.dma_cmd.peek().litValue != 0),
      "lsu"    -> (d.mem_cmd.peek().litValue != 0 || d.lsu_cmd.peek().litValue != 0)
    ).collect { case (u, true) => u }
    assert(units.size <= 1, s"several units: $units")
    if (!d.legal.peek().litToBoolean) { assert(units.isEmpty); None }
    else Some(units.headOption.getOrElse("scalar"))
  }

  /** ScalarISA's names of the values of a field, from the constants with the prefix. */
  private def names(prefix: String): Map[BigInt, String] =
    ScalarISA.getClass.getMethods.toSeq
      .filter(m => m.getName.startsWith(prefix) && m.getReturnType == classOf[UInt])
      .filterNot(m => m.getName.endsWith("_X") || m.getName.endsWith("_NONE"))
      .map(m => m.invoke(ScalarISA).asInstanceOf[UInt].litValue -> m.getName.stripPrefix(prefix))
      .toMap

  private val fields = Seq[(String, String, DecodedInstr => UInt)](
    ("alu", "ALU_", _.alu_fn), ("br", "BR_", _.br_type), ("op1", "OP1_", _.op1_sel),
    ("op2", "OP2_", _.op2_sel), ("imm", "IMM_", _.imm_sel), ("csr", "CSR_", _.csr_cmd),
    ("dma", "DMA_", _.dma_cmd), ("vpu", "VPU_", _.vpu_cmd), ("src", "VSRC_", _.vpu_src),
    ("xlu", "XLU_", _.xlu_cmd), ("mem", "MEM_", _.mem_cmd), ("lsu", "LSU_", _.lsu_cmd)
  ).map { case (key, prefix, field) => (key, names(prefix), field) }
  private val mxuNames = names("MXU_")

  /** The decoded control vector, as a golden-table line without the name. */
  private def control(d: DecodedInstr): String = {
    def set(b: Bool) = b.peek().litToBoolean
    val values = fields.collect {
      case (key, n, field) if field(d).peek().litValue != 0 =>
        val v = field(d).peek().litValue
        s"$key=${n.getOrElse(v, v.toString)}"
    }
    val flags = Seq("rd_wen" -> d.rd_wen, "jal" -> d.is_jal, "jalr" -> d.is_jalr, "rd_x" -> d.rd_x,
      "rs1_ren" -> d.rs1_ren, "rs2_ren" -> d.rs2_ren)
      .collect { case (f, b) if set(b) => f }
    val mxu = Option.when(set(d.is_mxu)) {
      val cmd = d.mxu_cmd.peek().litValue
      s"${if (d.mxu_sel.peek().litValue == 1) "mxu1" else "nv"}=${mxuNames.getOrElse(cmd, cmd.toString)}"
    }
    val sc = Seq("sc=SELI" -> d.is_seli, "sc=DELAY" -> d.is_delay).collect { case (f, b) if set(b) => f }
    (values ++ flags ++ mxu ++ sc).mkString(" ")
  }

  "ScalarDecoder" should "decode every word as the ISA table does" in {
    val rng = new Random(1)
    val lines = Seq.newBuilder[String]
    simulate(new ScalarDecoder) { dut =>
      def check(word: BigInt): Unit = {
        dut.io.instr.poke(word.U(32.W))
        assert(decodedUnit(dut.io.decoded) == entry(word).map(_._2), f"word 0x$word%08x, table ${entry(word)}")
      }
      var illegal = 0
      for ((name, p, unit) <- Instructions.all) {
        val free = ~p.mask & BigInt(0xFFFFFFFFL)
        val vectors = for (_ <- 0 until 16) yield {
          val word = p.value | (BigInt(32, rng) & free)
          assert(entry(word).map(_._1).contains(name))
          check(word)
          control(dut.io.decoded)
        }
        assert(vectors.distinct.size == 1, s"$name: the control vector depends on the operands")
        lines += s"$name ${vectors.head}".trim
        for (b <- 0 until 32 if p.mask.testBit(b)) {
          val word = p.value.flipBit(b)
          if (entry(word).isEmpty) illegal += 1
          check(word)
        }
      }
      assert(illegal > 0)
    }
    val table = lines.result()
    if (sys.env.get("ATLAS_DECODE_GOLDEN").contains("write")) {
      Files.write(golden, (header ++ table).asJava)
      info(s"wrote $golden")
    } else {
      val want = Files.readAllLines(golden).asScala.toSeq.filterNot(_.startsWith("#"))
      val diff = (want.diff(table).map("- " + _) ++ table.diff(want).map("+ " + _)).mkString("\n")
      if (table != want) fail(s"the decoder differs from $golden (- golden, + decoder):\n$diff")
    }
  }

  it should "name the x-registers that the ISA table gives each instruction" in {
    simulate(new ScalarDecoder) { dut =>
      val d = dut.io.decoded
      for ((name, p, _) <- Instructions.all) {
        dut.io.instr.poke(p.value.U(32.W))
        val fields = Seq("rd" -> d.rd_x, "rs1" -> d.rs1_ren, "rs2" -> d.rs2_ren)
          .collect { case (f, b) if b.peek().litToBoolean => f }.toSet
        assert(fields == Instructions.xRegs.getOrElse(name, Set.empty[String]), name)
      }
    }
  }

  it should "decode a .vx or .vr form as its .bf16 op with another second operand" in {
    simulate(new ScalarDecoder) { dut =>
      def decode(p: BitPat): (BigInt, BigInt) = {
        dut.io.instr.poke(p.value.U(32.W))
        (dut.io.decoded.vpu_cmd.peek().litValue, dut.io.decoded.vpu_src.peek().litValue)
      }
      val pats = Instructions.all.map { case (name, p, _) => name -> p }.toMap
      for ((name, p, _) <- Instructions.all; (suffix, src) <- Seq("_VX" -> VSRC_XREG, "_VR" -> VSRC_ROW0)
           if name.endsWith(suffix)) {
        val (cmd, pair) = decode(pats(name.stripSuffix(suffix)))
        assert(cmd != 0 && pair == VSRC_PAIR.litValue && decode(p) == (cmd, src.litValue), name)
      }
    }
  }
}
