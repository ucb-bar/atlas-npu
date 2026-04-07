// ============================================================================
// Assembler.scala — Tiny assembler for diplomatic Atlas tile tests.
//
// Parses a small assembly subset, resolves local labels, expands `LI`, and
// lowers mnemonics into instruction words using [[RV32Encode]].
// ============================================================================

package atlas.tile

import scala.collection.mutable.ArrayBuffer
import RV32Encode._

/** Minimal assembler used by the diplomatic tile test framework. */
object Assembler {
  private def parseReg(s: String): Int = {
    val n = s.stripPrefix("x").stripSuffix(",").trim.toInt
    require(n >= 0 && n <= 31, s"Invalid register: $s"); n
  }
  private def parseImm(s: String): Long = {
    val t = s.trim.stripSuffix(",")
    if (t.startsWith("-0x") || t.startsWith("-0X")) -java.lang.Long.parseUnsignedLong(t.drop(3), 16)
    else if (t.startsWith("0x") || t.startsWith("0X")) java.lang.Long.parseUnsignedLong(t.drop(2), 16)
    else t.toLong
  }
  private def expandLI(rd: Int, value: Long): Seq[Int] = {
    val v = value.toInt
    if (v >= -2048 && v <= 2047) Seq(ADDI(rd, 0, v & 0xFFF))
    else {
      val lo12 = (v << 20) >> 20; val hi20 = ((v - lo12) >> 12) & 0xFFFFF
      if (lo12 == 0) Seq(LUI(rd, hi20)) else Seq(LUI(rd, hi20), ADDI(rd, rd, lo12 & 0xFFF))
    }
  }
  def assemble(source: String): Seq[Int] = {
    def stripComment(line: String): String = line.takeWhile(_ != '#').trim
    def toks(line: String): Array[String] = stripComment(line).split("[\\s,]+").filter(_.nonEmpty)
    val lines = source.linesIterator.toIndexedSeq.map(stripComment).filter(_.nonEmpty)
    val labels = scala.collection.mutable.Map[String, Int]()
    var addr = 0
    for (line <- lines) {
      if (line.endsWith(":")) { labels(line.dropRight(1).trim) = addr }
      else { val parts = toks(line)
        if (parts.nonEmpty) addr += (if (parts(0).equalsIgnoreCase("LI")) expandLI(0, parseImm(parts(2))).length else 1)
      }
    }
    val code = ArrayBuffer[Int](); var pc = 0
    def resolve(s: String): Int = if (labels.contains(s)) labels(s) - pc else parseImm(s).toInt
    for (line <- lines) {
      if (!line.endsWith(":")) {
        val p = toks(line)
        if (p.nonEmpty) {
          p(0).toUpperCase match {
            case "NOP"    => code += NOP; pc += 1
            case "LI"     => val e = expandLI(parseReg(p(1)), parseImm(p(2))); code ++= e; pc += e.length
            case "ADD"    => code += ADD(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "SUB"    => code += SUB(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "AND"    => code += AND(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "OR"     => code += OR(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "XOR"    => code += XOR(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "SLL"    => code += SLL(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "SRL"    => code += SRL(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "SRA"    => code += SRA(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "SLT"    => code += SLT(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "SLTU"   => code += SLTU(parseReg(p(1)), parseReg(p(2)), parseReg(p(3))); pc += 1
            case "ADDI"   => code += ADDI(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "ANDI"   => code += ANDI(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "ORI"    => code += ORI(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "XORI"   => code += XORI(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SLTI"   => code += SLTI(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SLTIU"  => code += SLTIU(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SLLI"   => code += SLLI(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SRLI"   => code += SRLI(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SRAI"   => code += SRAI(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "LUI"    => code += LUI(parseReg(p(1)), parseImm(p(2)).toInt); pc += 1
            case "AUIPC"  => code += AUIPC(parseReg(p(1)), parseImm(p(2)).toInt); pc += 1
            case "BEQ"    => code += BEQ(parseReg(p(1)), parseReg(p(2)), resolve(p(3))); pc += 1
            case "BNE"    => code += BNE(parseReg(p(1)), parseReg(p(2)), resolve(p(3))); pc += 1
            case "BLT"    => code += BLT(parseReg(p(1)), parseReg(p(2)), resolve(p(3))); pc += 1
            case "BGE"    => code += BGE(parseReg(p(1)), parseReg(p(2)), resolve(p(3))); pc += 1
            case "BLTU"   => code += BLTU(parseReg(p(1)), parseReg(p(2)), resolve(p(3))); pc += 1
            case "BGEU"   => code += BGEU(parseReg(p(1)), parseReg(p(2)), resolve(p(3))); pc += 1
            case "JAL"    => code += JAL(parseReg(p(1)), resolve(p(2))); pc += 1
            case "JALR"   => code += JALR(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "FENCE"  => code += FENCE; pc += 1
            case "ECALL"  => code += ECALL; pc += 1
            case "EBREAK" => code += EBREAK; pc += 1
            case "DELAY"  => code += DELAY(parseImm(p(1)).toInt); pc += 1
            case "CSRW"   => code += CSRRW(0, parseImm(p(2)).toInt, parseReg(p(1))); pc += 1
            case "CSRR"   => code += CSRRS(parseReg(p(1)), parseImm(p(2)).toInt, 0); pc += 1
            case "CSRRW"  => code += CSRRW(parseReg(p(1)), parseImm(p(2)).toInt, parseReg(p(3))); pc += 1
            case "CSRRS"  => code += CSRRS(parseReg(p(1)), parseImm(p(2)).toInt, parseReg(p(3))); pc += 1
            case "CSRRC"  => code += CSRRC(parseReg(p(1)), parseImm(p(2)).toInt, parseReg(p(3))); pc += 1
            case "CSRRWI" => code += CSRRWI(parseReg(p(1)), parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "CSRRSI" => code += CSRRSI(parseReg(p(1)), parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "CSRRCI" => code += CSRRCI(parseReg(p(1)), parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            // ── Scalar loads/stores ──────────────────────
            case "LB"     => code += LB(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "LH"     => code += LH(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "LW"     => code += LW(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "LBU"    => code += LBU(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "LHU"    => code += LHU(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SELD"   => code += SELD(parseImm(p(1)).toInt, parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SB"     => code += SB(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SH"     => code += SH(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SW"     => code += SW(parseReg(p(1)), parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            // ── Tensor load/store ────────────────────────
            case "VLOAD"  => code += VLOAD(parseImm(p(1)).toInt, parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "VSTORE" => code += VSTORE(parseImm(p(1)).toInt, parseReg(p(2)), parseImm(p(3)).toInt); pc += 1
            case "SELI"   => code += SELI(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            // ── DMA ──────────────────────────────────────
            // DMA.LOAD  rd(vmem), rs1(dram_off), rs2(size), channel
            // DMA.STORE rd(dram_off), rs1(vmem), rs2(size), channel
            case "DMA.LOAD"   => code += DMA_LOAD(parseReg(p(1)), parseReg(p(2)), parseReg(p(3)), parseImm(p(4)).toInt); pc += 1
            case "DMA.STORE"  => code += DMA_STORE(parseReg(p(1)), parseReg(p(2)), parseReg(p(3)), parseImm(p(4)).toInt); pc += 1
            case "DMA.CONFIG" => code += DMA_CONFIG(parseReg(p(1)), parseImm(p(2)).toInt); pc += 1
            case "DMA.WAIT"   => code += DMA_WAIT(parseImm(p(1)).toInt); pc += 1
            // ── MXU ──────────────────────────────────────
            case "VMATPUSH.W.MXU0"         => code += VMATPUSH_W_MXU0(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VMATPUSH.W.MXU1"         => code += VMATPUSH_W_MXU1(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VMATPUSH.ACC.FP8.MXU0"   => code += VMATPUSH_AFP8_MXU0(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VMATPUSH.ACC.FP8.MXU1"   => code += VMATPUSH_AFP8_MXU1(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VMATPUSH.ACC.BF16.MXU0"  => code += VMATPUSH_ABF16_MXU0(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VMATPUSH.ACC.BF16.MXU1"  => code += VMATPUSH_ABF16_MXU1(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VMATPOP.FP8.MXU0"        => code += VMATPOP_FP8_MXU0(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VMATPOP.FP8.MXU1"        => code += VMATPOP_FP8_MXU1(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VMATPOP.BF16.MXU0"       => code += VMATPOP_BF16_MXU0(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VMATPOP.BF16.MXU1"       => code += VMATPOP_BF16_MXU1(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VMATMUL.MXU0"            => code += VMATMUL_MXU0(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VMATMUL.MXU1"            => code += VMATMUL_MXU1(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VMATMUL.ACC.MXU0"        => code += VMATMUL_ACC_MXU0(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VMATMUL.ACC.MXU1"        => code += VMATMUL_ACC_MXU1(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            // ── VPU / XLU ────────────────────────────────
            case "VADD.BF16"    => code += VADD_BF16(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VSUB.BF16"    => code += VSUB_BF16(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VMUL.BF16"    => code += VMUL_BF16(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VMIN.BF16"    => code += VMIN_BF16(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VMAX.BF16"    => code += VMAX_BF16(parseImm(p(1)).toInt, parseImm(p(2)).toInt, parseImm(p(3)).toInt); pc += 1
            case "VREDSUM.BF16" => code += VREDSUM_BF16(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VMOV"         => code += VMOV(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VRECIP.BF16"  => code += VRECIP_BF16(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VEXP"         => code += VEXP(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VRELU"        => code += VRELU(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VLI.ALL"      => code += VLI_ALL(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VLI.ROW"      => code += VLI_ROW(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VLI.COL"      => code += VLI_COL(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VLI.ONE"      => code += VLI_ONE(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VTRPOSE.XLU"  => code += VTRPOSE_XLU(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VREDMAX.XLU"  => code += VREDMAX_XLU(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case "VREDSUM.XLU"  => code += VREDSUM_XLU(parseImm(p(1)).toInt, parseImm(p(2)).toInt); pc += 1
            case other => throw new IllegalArgumentException(s"Unknown mnemonic: $other (line: $line)")
          }
        }
      }
    }
    code.toSeq
  }
}
