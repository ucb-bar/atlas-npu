/*
SystolicArrayVectorTest.scala: data-driven tests from Python-generated vectors

Reads src/test/resources/mxu_vectors.txt (generate with gen_mxu_vectors.py).
Drives SystolicArray with fractional FP values. Collects abs, ULP, rel error;
pass/fail based on relative error only (1% or both near zero).

Generate:  python3 scripts/gen_mxu_vectors.py --out src/test/resources/mxu_vectors.txt
Run:       mill atlas.test.testOnly atlas.sa.SystolicArrayVectorTest
*/

package atlas.sa

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common.SystolicArrayParams
import scala.io.Source
import scala.collection.mutable.ArrayBuffer

class SystolicArrayVectorTest extends AnyFlatSpec with Matchers {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("SystolicArrayVectorTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("SystolicArrayVectorTest=PASSED")
    }
    outcome
  }

  val vectorResource = "/mxu_vectors.txt"

  // Systolic array latency: rows + cols - 1
  def latency(p: SystolicArrayParams): Int = p.rows + p.cols - 1

  // Test vector data class (same format as IPT: weights are numLanes x vecLen = cols x rows)
  case class TestVector(
    id:       Int,
    caseType: String,
    sel:      Int,
    act:      Seq[Int],
    weights:  Seq[Seq[Int]], // cols x rows (IPT: numLanes x vecLen)
    bias:     Seq[Int],
    psum:     Seq[Int],
    expected: Seq[Int],
  )

  // Parser for the flat hex format (same as InnerProductTreesVectorTest)
  //
  //   # <id> <type>
  //   sel <n>
  //   act <hex> <hex> ...
  //   wgt <r> <hex> <hex> ...    (repeated per lane/column)
  //   bias <hex> <hex> ...
  //   psum <hex> <hex> ...
  //   exp <hex> <hex> ...

  def hexToInt(h: String): Int = Integer.parseUnsignedInt(h, 16)

  def loadVectors(resourcePath: String): Seq[TestVector] = {
    val stream = getClass.getResourceAsStream(resourcePath)
    require(
      stream != null,
      s"Resource '$resourcePath' not found. Run: python3 scripts/gen_mxu_vectors.py --out src/test/resources${resourcePath}",
    )
    val src = Source.fromInputStream(stream)
    val vectors = ArrayBuffer[TestVector]()

    var id       = 0
    var caseType = ""
    var sel      = 0
    var act      = Seq.empty[Int]
    var weights  = ArrayBuffer[Seq[Int]]()
    var bias     = Seq.empty[Int]
    var psum     = Seq.empty[Int]
    var expected = Seq.empty[Int]
    var inCase   = false

    def flush(): Unit = {
      if (inCase) {
        vectors += TestVector(id, caseType, sel, act, weights.toSeq, bias, psum, expected)
        weights = ArrayBuffer[Seq[Int]]()
        inCase = false
      }
    }

    try {
      for (raw <- src.getLines()) {
        val line = raw.trim
        if (line.isEmpty) {
          flush()
        } else if (line.startsWith("#")) {
          flush()
          val parts = line.drop(1).trim.split("\\s+")
          id = parts(0).toInt
          caseType = if (parts.length > 1) parts(1) else "unknown"
          inCase = true
        } else {
          val parts = line.split("\\s+")
          parts(0) match {
            case "sel"  => sel = parts(1).toInt
            case "act"  => act = parts.drop(1).map(hexToInt).toSeq
            case "wgt"  => weights += parts.drop(2).map(hexToInt).toSeq
            case "bias" => bias = parts.drop(1).map(hexToInt).toSeq
            case "psum" => psum = parts.drop(1).map(hexToInt).toSeq
            case "exp"  => expected = parts.drop(1).map(hexToInt).toSeq
            case _      =>
          }
        }
      }
      flush()
    } finally {
      src.close()
    }

    vectors.toSeq
  }

  def bf16BitsToFloat(bits: Int): Float = Bf16Compare.bf16BitsToFloat(bits)

  def selToEnum(sel: Int): AddendSel.Type = {
    sel match {
      case 0 => AddendSel.UseZero
      case 1 => AddendSel.UseBias
      case 2 => AddendSel.UsePsum
      case _ => AddendSel.UseZero
    }
  }

  // Test helpers
  def idle(dut: SystolicArray, p: SystolicArrayParams): Unit = {
    dut.io.computeReq.valid.poke(false.B)
    dut.io.weightLoadReq.valid.poke(false.B)
    for (i <- 0 until p.rows) dut.io.computeReq.bits.activationRow(i).poke(0.U)
    for (j <- 0 until p.cols) {
      dut.io.computeReq.bits.bias(j).poke(0.U)
      dut.io.computeReq.bits.psum(j).poke(0.U)
    }
    dut.io.computeReq.bits.addendSel.poke(AddendSel.UseZero)
    dut.io.computeReq.bits.weightBufReadSel.poke(false.B)
    dut.io.weightLoadReq.bits.weightSel.poke(WeightSel.UseTensorReg)
    dut.io.weightLoadReq.bits.weightBufWriteSel.poke(0.U)
    dut.io.weightLoadReq.bits.colIdx.poke(0.U)
    for (i <- 0 until p.rows) {
      dut.io.weightLoadReq.bits.weightsDirectMem(i).poke(0.U)
      dut.io.weightLoadReq.bits.weightsTensorReg(i).poke(0.U)
    }
  }

  def waitForResult(dut: SystolicArray, p: SystolicArrayParams): Unit = {
    val lat = latency(p)
    if (lat > 1) dut.clock.step(lat - 1)
  }

  // Test body
  "SystolicArray" should "match Python ground truth for fractional FP vectors" in {
    val p = SystolicArrayParams()
    require(p.rows == 32 && p.cols == 16,
      s"Vector test expects 32x16 array; got ${p.rows}x${p.cols}. Use default SystolicArrayParams().")

    val vectors = loadVectors(vectorResource)
    require(vectors.nonEmpty, "No test vectors found. Run gen_mxu_vectors.py first.")

    simulate(new SystolicArray(p)) { dut =>
      var passed = 0
      var failed = 0

      val outFile =
        new java.io.PrintWriter("../../../../../src/test/resources/systolic_array_vector_outputs.txt")
      outFile.println("# case_id case_type        col  actual_hex expected_hex  actual_float expected_float abs_err rel_err ulp match")

      for (tv <- vectors) {
        idle(dut, p)

        // Load weights one column at a time via weightsTensorReg
        // IPT weights[lane][col] -> SA col j gets weights(j) (column j = output j)
        dut.io.weightLoadReq.valid.poke(true.B)
        dut.io.weightLoadReq.bits.weightSel.poke(WeightSel.UseTensorReg)
        dut.io.weightLoadReq.bits.weightBufWriteSel.poke(0.U)
        for (j <- 0 until p.cols) {
          dut.io.weightLoadReq.bits.colIdx.poke(j.U)
          for (i <- 0 until p.rows) {
            dut.io.weightLoadReq.bits.weightsTensorReg(i).poke(tv.weights(j)(i).U)
            dut.io.weightLoadReq.bits.weightsDirectMem(i).poke(0.U)
          }
          dut.clock.step()
        }
        dut.io.weightLoadReq.valid.poke(false.B)

        // Fire compute
        dut.io.computeReq.valid.poke(true.B)
        for (i <- 0 until p.rows)
          dut.io.computeReq.bits.activationRow(i).poke(tv.act(i).U)
        for (j <- 0 until p.cols) {
          dut.io.computeReq.bits.bias(j).poke(tv.bias(j).U)
          dut.io.computeReq.bits.psum(j).poke(tv.psum(j).U)
        }
        dut.io.computeReq.bits.addendSel.poke(selToEnum(tv.sel))
        dut.io.computeReq.bits.weightBufReadSel.poke(false.B)
        dut.clock.step()
        dut.io.computeReq.valid.poke(false.B)

        waitForResult(dut, p)

        // Check outputs
        dut.io.outputRow.valid.expect(true.B)
        var caseOk = true
        for (j <- 0 until p.cols) {
          val actual   = dut.io.outputRow.bits(j).peek().litValue
          val expected = tv.expected(j)

          val a16 = actual.toInt & 0xFFFF
          val e16 = expected & 0xFFFF

          val aFloat = bf16BitsToFloat(a16)
          val eFloat = bf16BitsToFloat(e16)

          val (ok, errs) = Bf16Compare.compare(a16, e16)

          outFile.println(
            f"${tv.id}%8d ${tv.caseType}%-16s $j%4d   0x${a16}%04x       0x${e16}%04x " +
              f"$aFloat%14.6g $eFloat%14.6g ${errs.absErr}%10.6g ${errs.relErr}%10.6g ${errs.ulpDiff}%4d ${if (ok) "PASS" else "FAIL"}"
          )

          if (!ok) {
            caseOk = false
            println(
              f"  FAIL case ${tv.id} col $j: got 0x${a16}%04x ($aFloat%.6g) " +
                f"exp 0x${e16}%04x ($eFloat%.6g) " +
                f"[absErr=${errs.absErr}%.6g relErr=${errs.relErr}%.6g ulp=${errs.ulpDiff}] " +
                f"(pass: rel<=1%% or both near zero)"
            )
          }
        }

        if (caseOk) passed += 1 else failed += 1
      }

      outFile.close()
      println(s"\nWrote sp26-atlas-acc/src/test/resources/systolic_array_vector_outputs.txt")
      println(s"Vector test results: $passed passed, $failed failed out of ${vectors.length}")
      failed shouldBe 0
    }
  }
}