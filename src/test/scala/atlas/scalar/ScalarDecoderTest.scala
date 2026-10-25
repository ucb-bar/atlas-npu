package atlas.scalar

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Random

/** The decoder agrees with the ISA table (Instructions.all, generated from atlas_isa.toml). Words
  * with random operands decode as their instruction, legal and with its unit. A word one bit away
  * from an instruction's match word decodes as the table says: a word that sets a bit that the
  * table fixes, for example a reserved operand bit, is illegal unless it is another instruction.
  * The decoder's x-register fields (rd_x, rs1_ren, rs2_ren) are the table's x operands.
  */
class ScalarDecoderTest extends AnyFlatSpec {
  import ScalarISA._

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

  "ScalarDecoder" should "decode every word as the ISA table does" in {
    val rng = new Random(1)
    simulate(new ScalarDecoder) { dut =>
      def check(word: BigInt): Unit = {
        dut.io.instr.poke(word.U(32.W))
        assert(decodedUnit(dut.io.decoded) == entry(word).map(_._2), f"word 0x$word%08x, table ${entry(word)}")
      }
      var illegal = 0
      for ((name, p, unit) <- Instructions.all) {
        val free = ~p.mask & BigInt(0xFFFFFFFFL)
        for (_ <- 0 until 16) {
          val word = p.value | (BigInt(32, rng) & free)
          assert(entry(word).map(_._1).contains(name))
          check(word)
        }
        for (b <- 0 until 32 if p.mask.testBit(b)) {
          val word = p.value.flipBit(b)
          if (entry(word).isEmpty) illegal += 1
          check(word)
        }
      }
      assert(illegal > 0)
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
}
