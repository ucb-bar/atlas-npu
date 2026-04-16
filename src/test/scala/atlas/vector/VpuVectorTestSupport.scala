// ============================================================================
// VpuVectorTestSupport.scala — shared helpers for the per-family
// VectorEngineTop<X>VectorTest suites.
//
// Splits out the parsing, tolerance, and harness-drive code that used to
// live inline in VectorEngineTopAllOpsTest.scala. Each family test class
// mixes in this trait and calls `runVectorFamilyTest(<family>)`.
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import chisel3.simulator._
import org.scalatest.matchers.should.Matchers
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import atlas.common._
import sp26FPUnits._
import atlas.scalar.VpuCmd

import svsim.vcs.{Backend => VcsBackend}

trait VpuVectorTestSupport extends Matchers with PeekPokeAPI {

  // ── MREG bank assignments shared across all family tests ──
  val SRC1_BANK  = 0
  val SRC2_BANK  = 1
  val DEST1_BANK = 2
  val DEST2_BANK = 3

  // ── Test-vector case class (shape comes from vpu_vector_file.py blocks) ──
  case class TestVector(
    id:        Int,
    desc:      String,
    vpuOp:     String,
    numLanes:  Int,
    vecA:      Seq[Int],
    vecB:      Seq[Int],
    scaleExp:  Int,
    leftAlign: Boolean,
    expected:  Seq[Int]
  )

  // ── Small numeric helpers ──
  def hexToInt(h: String): Int = Integer.parseUnsignedInt(h, 16)

  def packElems(elems: Seq[Int], elemWidth: Int): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
      acc | (BigInt(v & ((1 << elemWidth) - 1)) << (i * elemWidth))
    }

  def bf16BitsToFloat(bits: Int): Float = {
    val fp32Bits = (bits & 0xFFFF) << 16
    java.lang.Float.intBitsToFloat(fp32Bits)
  }

  def resultBitsToFloat(bits: Int, op: String, leftAlign: Boolean = false): Float =
    // fp8pack / fp8unpack are not in any of the six families, so this path
    // is BF16-only. Keep the `op` arg for forward compat with the monolith.
    bf16BitsToFloat(bits & 0xFFFF)

  /** Per-op tolerance buckets preserved from VectorEngineTopAllOpsTest.scala. */
  def checkTolerance(
    actual:    BigInt,
    expected:  Int,
    op:        String,
    leftAlign: Boolean = false
  ): (Boolean, Float, Float) = {
    val aFloat = resultBitsToFloat(actual.toInt, op, leftAlign)
    val eFloat = resultBitsToFloat(expected, op, leftAlign)

    if (aFloat.isNaN && eFloat.isNaN) return (true, 0.0f, 0.0f)
    if (aFloat.isPosInfinity && eFloat.isPosInfinity) return (true, 0.0f, 0.0f)
    if (aFloat.isNegInfinity && eFloat.isNegInfinity) return (true, 0.0f, 0.0f)

    val absError = math.abs(aFloat - eFloat)
    val relError =
      if (eFloat == 0.0f) { if (aFloat == 0.0f) 0.0f else Float.PositiveInfinity }
      else absError / math.abs(eFloat)

    val (maxRelError, maxAbsError) = op match {
      case "sin" | "cos" | "tanh" | "log" | "exp" | "exp2" | "sqrt" => (0.05f, 0.05f)
      case "cube" | "square"                                         => (0.02f, 0.02f)
      case _                                                         => (0.01f, 0.01f)
    }

    val ok = (absError <= maxAbsError) || (relError <= maxRelError)
    (ok, absError, relError)
  }

  def vpuOpToChiselEnum(vpuOp: String): VPUOp.Type = vpuOp match {
    case "add"     => VPUOp.add
    case "sub"     => VPUOp.sub
    case "mul"     => VPUOp.mul
    case "rcp"     => VPUOp.rcp
    case "sqrt"    => VPUOp.sqrt
    case "sin"     => VPUOp.sin
    case "cos"     => VPUOp.cos
    case "tanh"    => VPUOp.tanh
    case "log"     => VPUOp.log
    case "exp"     => VPUOp.exp
    case "exp2"    => VPUOp.exp2
    case "square"  => VPUOp.square
    case "cube"    => VPUOp.cube
    case "rsum"    => VPUOp.rsum
    case "relu"    => VPUOp.relu
    case "rmax"    => VPUOp.rmax
    case "rmin"    => VPUOp.rmin
    case "cmax"    => VPUOp.cmax
    case "cmin"    => VPUOp.cmin
    case "csum"    => VPUOp.csum
    case "pairmax" => VPUOp.pairmax
    case "pairmin" => VPUOp.pairmin
    case "mov"     => VPUOp.mov
    case "vliOne"  => VPUOp.vliOne
    case "vliCol"  => VPUOp.vliCol
    case "vliRow"  => VPUOp.vliRow
    case "vliAll"  => VPUOp.vliAll
    case other     => throw new IllegalArgumentException(s"Unsupported vpuOp string: '$other'")
  }

  // ── Resource parser for the VPU block format defined in vpu_vector_file.py ──
  def loadVectors(resourcePath: String): Seq[TestVector] = {
    val stream = getClass.getResourceAsStream(resourcePath)
    require(
      stream != null,
      s"Resource '$resourcePath' not found on classpath. Regenerate with: " +
        "python3 scripts/gen_vpu_test_vectors.py"
    )
    val src = Source.fromInputStream(stream)
    val vectors = ArrayBuffer[TestVector]()

    var id = 0
    var desc = ""
    var vpuOp = ""
    var numLanes = 0
    var vecA: Seq[Int] = Seq.empty
    var vecB: Seq[Int] = Seq.empty
    var expected: Seq[Int] = Seq.empty
    var scaleExp = 127
    var leftAlign = false
    var inCase = false

    def flush(): Unit = {
      if (inCase) {
        vectors += TestVector(id, desc, vpuOp, numLanes, vecA, vecB, scaleExp, leftAlign, expected)
        inCase = false
      }
    }

    try {
      for (raw <- src.getLines()) {
        val line = raw.trim
        if (line.isEmpty) {
          if (inCase && vpuOp.nonEmpty) flush()
        } else if (line.startsWith("#")) {
          if (inCase && vpuOp.nonEmpty) flush()
          val headerPat = """#\s+(\d+)\s+-\s+"(.*)"""".r
          line match {
            case headerPat(idStr, descStr) =>
              id = idStr.toInt
              desc = descStr
            case _ =>
              throw new RuntimeException(s"Invalid header format: $line")
          }
          vpuOp = ""
          numLanes = 0
          vecA = Seq.empty
          vecB = Seq.empty
          expected = Seq.empty
          scaleExp = 127
          leftAlign = false
          inCase = true
        } else {
          val parts = line.split("\\s+")
          parts(0) match {
            case "vpuOp"     => vpuOp = parts(1)
            case "numLanes"  => numLanes = parts(1).toInt
            case "vecA"      => vecA = parts.drop(1).map(hexToInt).toSeq
            case "vecB"      => vecB = parts.drop(1).map(hexToInt).toSeq
            case "exp"       => expected = parts.drop(1).map(hexToInt).toSeq
            case "scaleExp"  => scaleExp = hexToInt(parts(1))
            case "leftAlign" => leftAlign = (parts(1).toInt == 1)
            case other       => throw new RuntimeException(s"Unknown directive: $other")
          }
        }
      }
      flush()
    } finally {
      src.close()
    }
    vectors.toSeq
  }

  // ── Harness helpers (reuse VectorEngineTopUnitHarness from VectorEngineTopTest) ──
  def idle(dut: VectorEngineTopUnitHarness): Unit = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.cmd.bits.op.poke(0.U)
    dut.io.cmd.bits.vs1.poke(0.U)
    dut.io.cmd.bits.vs2.poke(0.U)
    dut.io.cmd.bits.vd.poke(0.U)
    dut.io.cmd.bits.scaleE8M0.poke(127.U)
    dut.io.cmd.bits.imm.poke(0.U)

    dut.io.testWrite.valid.poke(false.B)
    dut.io.testWrite.bits.mregId.poke(0.U)
    dut.io.testWrite.bits.row.poke(0.U)
    dut.io.testWrite.bits.data.poke(0.U)

    dut.io.testRead.valid.poke(false.B)
    dut.io.testRead.bits.mregId.poke(0.U)
    dut.io.testRead.bits.row.poke(0.U)
  }

  def trfWriteRow(dut: VectorEngineTopUnitHarness, bank: Int, row: Int, data: BigInt): Unit = {
    dut.io.testWrite.valid.poke(true.B)
    dut.io.testWrite.bits.mregId.poke(bank.U)
    dut.io.testWrite.bits.row.poke(row.U)
    dut.io.testWrite.bits.data.poke(data.U)
    dut.clock.step()
    dut.io.testWrite.valid.poke(false.B)
  }

  /** Col write: place vecA element i in lane 0 of row i, zero the rest. */
  def trfWriteCol(dut: VectorEngineTopUnitHarness, bank: Int, data: Seq[Int]): Unit = {
    for (i <- 0 until 32) {
      val dataAsInt       = data(i) & 0xFFFF
      val rowElems        = Seq(dataAsInt) ++ Seq.fill(15)(0)
      val rowDataAsBigInt = packElems(rowElems, 16)
      dut.io.testWrite.valid.poke(true.B)
      dut.io.testWrite.bits.mregId.poke(bank.U)
      dut.io.testWrite.bits.row.poke(i.U)
      dut.io.testWrite.bits.data.poke(rowDataAsBigInt.U)
      dut.clock.step()
      dut.io.testWrite.valid.poke(false.B)
    }
  }

  def clearBank(dut: VectorEngineTopUnitHarness, bank: Int): Unit = {
    for (i <- 0 until 32) {
      dut.io.testWrite.valid.poke(true.B)
      dut.io.testWrite.bits.mregId.poke(bank.U)
      dut.io.testWrite.bits.row.poke(i.U)
      dut.io.testWrite.bits.data.poke(0.U)
      dut.clock.step()
    }
    dut.io.testWrite.valid.poke(false.B)
  }

  def trfReadRow(dut: VectorEngineTopUnitHarness, bank: Int, row: Int): BigInt = {
    dut.io.testRead.valid.poke(true.B)
    dut.io.testRead.bits.mregId.poke(bank.U)
    dut.io.testRead.bits.row.poke(row.U)
    dut.clock.step()
    dut.io.testRead.valid.poke(false.B)
    dut.io.testReadOut.valid.expect(true.B)
    dut.io.testReadOut.bits.peek().litValue
  }

  /** Issue a VpuCmd. `imm` is used by the vli ops; other ops leave it zero.
   *
   * Note on opcode encoding: VectorEngineTop subtracts 1 from `cmd.bits.op`
   * before casting to VPUOp (ScalarISA VPU_* values are 1-indexed with
   * VPU_NONE=0, while VPUOp is 0-indexed). Tests drive the ScalarISA side,
   * so we poke `litValue + 1` here.
   */
  def sendVpuCmd(
    dut:      VectorEngineTopUnitHarness,
    op:       VPUOp.Type,
    src1Bank: Int,
    src2Bank: Int,
    destBank: Int,
    imm:      Int = 0
  ): Unit = {
    dut.io.cmd.valid.poke(true.B)
    dut.io.cmd.bits.op.poke((op.litValue + 1).U)
    dut.io.cmd.bits.vs1.poke(src1Bank.U)
    dut.io.cmd.bits.vs2.poke(src2Bank.U)
    dut.io.cmd.bits.vd.poke(destBank.U)
    dut.io.cmd.bits.scaleE8M0.poke(127.U)
    dut.io.cmd.bits.imm.poke((imm & 0xFFFF).U)
    dut.clock.step()
    dut.io.cmd.valid.poke(false.B)
  }

  def waitVpuIdle(dut: VectorEngineTopUnitHarness, max: Int = 500): Unit = {
    var i = 0
    while (i < max && dut.io.busy.peek().litToBoolean) { dut.clock.step(); i += 1 }
    require(i < max, "VPU execution timed out")
  }

  def readBF16Results(
    dut:  VectorEngineTopUnitHarness,
    p:    VpuParams,
    bank: Int,
    row:  Int
  ): Seq[Int] = {
    val rowStr = trfReadRow(dut, bank, row)
    (0 until p.numLanes).map(i => ((rowStr >> (i * 16)) & 0xFFFF).toInt)
  }

  // ── The unified per-family driver ─────────────────────────────────────
  /**
   * Load `/vpu_test_vectors/vpu_<family>_vectors.txt` from the classpath,
   * run each case through a fresh `VectorEngineTopUnitHarness`, write an
   * aligned results table to `vpu_<family>_rtl_outputs.txt` next to the
   * golden, and assert zero failures.
   */
  def runVectorFamilyTest(family: String, sim: Simulator[VcsBackend]): Unit = {
    val vectorResource = s"/vpu_test_vectors/vpu_${family}_vectors.txt"
    val rtlOutRelative =
      s"../../../../../src/test/resources/vpu_test_vectors/vpu_${family}_rtl_outputs.txt"

    val vectors = loadVectors(vectorResource)
    require(
      vectors.nonEmpty,
      s"No test vectors found in $vectorResource. Regenerate with: " +
        "python3 scripts/gen_vpu_test_vectors.py"
    )

    val p     = VpuParams()
    val mregP = MregParams()

    sim.simulate(new VectorEngineTopUnitHarness(p, mregP)) { module =>
      val dut = module.wrapped
      val outFile = new java.io.PrintWriter(rtlOutRelative)
      var passed  = 0
      var failed  = 0

      outFile.println(
        f"# ${"id"}%6s  ${"case_desc"}%-32s ${"lane"}%4s ${"vpuOp"}%-8s " +
        f"${"actual_hex"}%12s ${"expected_hex"}%12s " +
        f"${"actual_float"}%14s ${"expected_float"}%14s ${"RelErr%"}%8s ${"match"}%-6s"
      )

      for (tv <- vectors) {
        dut.reset.poke(true.B)
        dut.clock.step(5)
        dut.reset.poke(false.B)
        dut.clock.step(1)
        idle(dut)

        val vecAasBigInt: BigInt = packElems(tv.vecA, p.numLanes)
        val vecBasBigInt: BigInt = packElems(tv.vecB, p.numLanes)
        val opEnum               = vpuOpToChiselEnum(tv.vpuOp)

        val isCol =
          opEnum == VPUOp.cmin || opEnum == VPUOp.cmax || opEnum == VPUOp.csum
        val isVli =
          opEnum == VPUOp.vliOne || opEnum == VPUOp.vliCol ||
            opEnum == VPUOp.vliRow || opEnum == VPUOp.vliAll
        val isVliPair = opEnum == VPUOp.vliAll || opEnum == VPUOp.vliRow
        val isVliSingle = opEnum == VPUOp.vliCol || opEnum == VPUOp.vliOne
        val destBank =
          if (isVliSingle) DEST2_BANK else DEST1_BANK

        if (isCol) {
          clearBank(dut, SRC2_BANK)
          trfWriteCol(dut, SRC1_BANK, tv.vecA)
        } else if (!isVli) {
          trfWriteRow(dut, SRC1_BANK, 0, vecAasBigInt)
          trfWriteRow(dut, SRC2_BANK, 0, vecBasBigInt)
        }

        if (isVli) {
          sendVpuCmd(
            dut,
            op       = opEnum,
            src1Bank = 0,
            src2Bank = 0,
            destBank = destBank,
            imm      = tv.vecA(0)
          )
        } else {
          sendVpuCmd(
            dut,
            op       = opEnum,
            src1Bank = SRC1_BANK,
            src2Bank = SRC2_BANK,
            destBank = destBank
          )
        }

        println(f"[DBG] case ${tv.id}%3d op=${tv.vpuOp}")
        waitVpuIdle(dut)
        dut.clock.step(2)

        var caseOk    = true
        val actualSeq = readBF16Results(dut, p, destBank, row = 0)

        def logCompare(actual: Int, expected: Int, index: Int, label: String): Unit = {
          val aFloat                   = bf16BitsToFloat(actual)
          val eFloat                   = bf16BitsToFloat(expected)
          val (ok, absError, relError) = checkTolerance(actual, expected, tv.vpuOp, tv.leftAlign)
          val relErrorPct              = relError * 100.0f
          val aHexStr                  = f"0x${actual & 0xFFFF}%04x"
          val eHexStr                  = f"0x${expected & 0xFFFF}%04x"
          val matchStr                 = if (ok) "PASS" else "FAIL"

          outFile.println(
            f"  ${tv.id}%6d  ${"\"" + tv.desc + "\""}%-32s $index%4d ${tv.vpuOp}%-8s " +
              f"$aHexStr%12s $eHexStr%12s " +
              f"$aFloat%14.4f $eFloat%14.4f ${f"$relErrorPct%.2f%%"}%8s $matchStr%-6s"
          )

          if (!ok) {
            caseOk = false
            println(
              f"  FAIL case ${tv.id}%3d [${tv.desc}] $label: " +
                f"got 0x${actual & 0xFFFF}%04x ($aFloat%.4f), " +
                f"expected 0x${expected & 0xFFFF}%04x ($eFloat%.4f) " +
                f"[RelErr $relErrorPct%.2f%%, AbsErr $absError%.4f]"
            )
          }
        }

        val wideRead = isCol
        if (isVliPair) {
          for (bank <- Seq(destBank, destBank + 1)) {
            for (row <- 0 until mregP.mregRows) {
              val rowData = trfReadRow(dut, bank, row)
              val expected = tv.expected(row) & 0xFFFF
              for (lane <- 0 until p.numLanes) {
                val actual = ((rowData >> (16 * lane)) & 0xFFFF).toInt
                logCompare(actual, expected, row, s"bank $bank row $row slot $lane")
              }
            }
          }
        } else if (isVliSingle) {
          val untouchedBank = DEST1_BANK
          for (row <- 0 until mregP.mregRows) {
            val rowData = trfReadRow(dut, destBank, row)
            val expected = tv.expected(row) & 0xFFFF
            val actualHead = (rowData & 0xFFFF).toInt
            logCompare(actualHead, expected, row, s"bank $destBank row $row slot 0")

            for (lane <- 1 until p.numLanes) {
              val actual = ((rowData >> (16 * lane)) & 0xFFFF).toInt
              if (actual != 0) {
                caseOk = false
                println(
                  f"  FAIL case ${tv.id}%3d [${tv.desc}] bank $destBank row $row%2d slot $lane%2d: " +
                    f"got non-zero 0x${actual & 0xFFFF}%04x, expected 0x0000"
                )
              }
            }

            val untouchedRow = trfReadRow(dut, untouchedBank, row)
            for (lane <- 0 until p.numLanes) {
              val actual = ((untouchedRow >> (16 * lane)) & 0xFFFF).toInt
              if (actual != 0) {
                caseOk = false
                println(
                  f"  FAIL case ${tv.id}%3d [${tv.desc}] untouched bank $untouchedBank row $row%2d slot $lane%2d: " +
                    f"got non-zero 0x${actual & 0xFFFF}%04x, expected 0x0000"
                )
              }
            }
          }
        } else if (wideRead) {
          for (i <- 0 until 32) {
            val rowData: BigInt = trfReadRow(dut, DEST1_BANK, i)
            for (j <- 0 until 1) {
              val actual                       = ((rowData >> (16 * j)) & 0xFFFF).toInt
              val expected                     = tv.expected(i) & 0xFFFF
              logCompare(actual, expected, i, s"row $i slot $j")
            }
          }
        } else {
          for (r <- 0 until p.numLanes) {
            val actual                       = actualSeq(r)
            val expected                     = tv.expected(r)
            logCompare(actual, expected, r, s"lane $r")
          }
        }

        if (caseOk) passed += 1 else failed += 1
        dut.clock.step(1)
        outFile.println()
      }
      outFile.close()
      println(
        s"\nWrote sp26-atlas-acc/src/test/resources/vpu_test_vectors/vpu_${family}_rtl_outputs.txt"
      )
      println(s"Family '$family': $passed passed, $failed failed out of ${vectors.length}")
      failed shouldBe 0
    }
  }
}
