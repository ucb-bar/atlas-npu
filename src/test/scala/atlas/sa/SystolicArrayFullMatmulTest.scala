/*
SystolicArrayFullMatmulTest.scala: full matmul test (64 rows) for SystolicArray

Reads src/test/resources/mxu_full_matmul_vectors.txt. Per case:
  For each tile: load 32×16 weights, stream 64 activation rows back-to-back,
  collect outputs as psum for next tile. Compare final [64×16] against Python expected.

Architecture: [1×32] act × [32×16] wgt = [1×16] per cycle. Latency = rows + cols - 1.

Run:  mill atlas.test.testOnly atlas.sa.SystolicArrayFullMatmulTest
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

class SystolicArrayFullMatmulTest extends AnyFlatSpec with Matchers {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("SystolicArrayFullMatmulTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("SystolicArrayFullMatmulTest=PASSED")
    }
    outcome
  }

  val vectorResource = "/mxu_full_matmul_vectors.txt"

  // Systolic array latency: rows + cols - 1
  def latency(p: SystolicArrayParams): Int = p.rows + p.cols - 1

  // Test vector data

  case class FullMatmulVector(
    id:       Int,
    numRows:  Int,
    numTiles: Int,
    weights:  Seq[Seq[Seq[Int]]], // [tile][col][row]
    act:      Seq[Seq[Seq[Int]]], // [tile][row][col]
    expected: Seq[Seq[Int]]       // [row][col] (final result after all tiles)
  )

  // Parser

  def hexToInt(h: String): Int = Integer.parseUnsignedInt(h, 16)

  def loadVectors(resourcePath: String): Seq[FullMatmulVector] = {
    val stream = getClass.getResourceAsStream(resourcePath)
    require(stream != null,
      s"Resource '$resourcePath' not found. Run the Python generator first.")
    val src = Source.fromInputStream(stream)
    val vectors = ArrayBuffer[FullMatmulVector]()

    var id       = 0
    var numRows  = 64
    var numTiles = 1

    var wgtMap = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Seq[Int]]]()
    var actMap = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Seq[Int]]]()
    var expMap = scala.collection.mutable.Map[Int, Seq[Int]]()
    var inCase = false

    def flush(): Unit = {
      if (inCase) {
        val weights = (0 until numTiles).map { t =>
          val tileMap = wgtMap.getOrElse(t, scala.collection.mutable.Map.empty)
          (0 until tileMap.size).map(c => tileMap(c))
        }
        val act = (0 until numTiles).map { t =>
          val tileMap = actMap.getOrElse(t, scala.collection.mutable.Map.empty)
          (0 until tileMap.size).map(i => tileMap(i))
        }
        val expected = (0 until expMap.size).map(i => expMap(i))

        vectors += FullMatmulVector(id, numRows, numTiles, weights, act, expected)

        wgtMap = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Seq[Int]]]()
        actMap = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int, Seq[Int]]]()
        expMap = scala.collection.mutable.Map[Int, Seq[Int]]()
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
          inCase = true
        } else {
          val parts = line.split("\\s+")
          parts(0) match {
            case "num_rows"  => numRows  = parts(1).toInt
            case "num_tiles" => numTiles = parts(1).toInt
            case "wgt" =>
              val tile = parts(1).toInt
              val lane = parts(2).toInt
              val data = parts.drop(3).map(hexToInt).toSeq
              wgtMap.getOrElseUpdate(tile, scala.collection.mutable.Map.empty)(lane) = data
            case "act" =>
              val tile = parts(1).toInt
              val row  = parts(2).toInt
              val data = parts.drop(3).map(hexToInt).toSeq
              actMap.getOrElseUpdate(tile, scala.collection.mutable.Map.empty)(row) = data
            case "exp" =>
              val row  = parts(1).toInt
              val data = parts.drop(2).map(hexToInt).toSeq
              expMap(row) = data
            case _ =>
          }
        }
      }
      flush()
    } finally {
      src.close()
    }

    vectors.toSeq
  }

  def bf16BitsToFloat(bits16: Int): Float = Bf16Compare.bf16BitsToFloat(bits16)

  // DUT helpers

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

  def loadWeights(
    dut: SystolicArray,
    p: SystolicArrayParams,
    weights: Seq[Seq[Int]]
  ): Unit = {
    dut.io.weightLoadReq.valid.poke(true.B)
    dut.io.weightLoadReq.bits.weightSel.poke(WeightSel.UseTensorReg)
    dut.io.weightLoadReq.bits.weightBufWriteSel.poke(0.U)
    for (j <- 0 until p.cols) {
      dut.io.weightLoadReq.bits.colIdx.poke(j.U)
      for (i <- 0 until p.rows) {
        dut.io.weightLoadReq.bits.weightsTensorReg(i).poke(weights(j)(i).U)
        dut.io.weightLoadReq.bits.weightsDirectMem(i).poke(0.U)
      }
      dut.clock.step()
    }
    dut.io.weightLoadReq.valid.poke(false.B)
  }

  def selToEnum(sel: Int): AddendSel.Type = sel match {
    case 0 => AddendSel.UseZero
    case 1 => AddendSel.UseBias
    case 2 => AddendSel.UsePsum
    case _ => AddendSel.UseZero
  }

  // Test body

  "SystolicArray" should "compute full tensor-register matmul correctly" in {
    val p = SystolicArrayParams()
    val lat = latency(p)
    val vectors = loadVectors(vectorResource)
    require(vectors.nonEmpty, s"No test vectors found at $vectorResource")

    simulate(new SystolicArray(p)) { dut =>
      var totalPassed = 0
      var totalFailed = 0

      val outFile =
        new java.io.PrintWriter("../../../../../src/test/resources/rtl_sa_full_matmul_outputs.txt")
      outFile.println(
        "# case_id  tile  row  col  actual_hex  expected_hex  actual_float  expected_float  abs_err  rel_err  ulp  match"
      )

      for (tv <- vectors) {
        println(f"═══ Case ${tv.id}: ${tv.numRows} rows × ${tv.numTiles} tile(s) ═══")

        val psumBuf = Array.fill(tv.numRows, p.cols)(0)

        for (t <- 0 until tv.numTiles) {
          val sel = if (t == 0) 0 else 2

          idle(dut, p)
          loadWeights(dut, p, tv.weights(t))

          val results = Array.fill(tv.numRows, p.cols)(0)
          var resultsCollected = 0
          val totalCycles = tv.numRows + lat

          for (cycle <- 0 until totalCycles) {
            if (dut.io.outputRow.valid.peek().litToBoolean) {
              require(
                resultsCollected < tv.numRows,
                s"Case ${tv.id} tile $t: got more outputs than expected"
              )
              for (j <- 0 until p.cols) {
                results(resultsCollected)(j) = dut.io.outputRow.bits(j).peek().litValue.toInt
              }
              resultsCollected += 1
            }

            if (cycle < tv.numRows) {
              dut.io.computeReq.valid.poke(true.B)
              for (i <- 0 until p.rows)
                dut.io.computeReq.bits.activationRow(i).poke(tv.act(t)(cycle)(i).U)
              for (j <- 0 until p.cols) {
                dut.io.computeReq.bits.bias(j).poke(0.U)
                dut.io.computeReq.bits.psum(j).poke(psumBuf(cycle)(j).U)
              }
              dut.io.computeReq.bits.addendSel.poke(selToEnum(sel))
              dut.io.computeReq.bits.weightBufReadSel.poke(false.B)
            } else {
              dut.io.computeReq.valid.poke(false.B)
            }

            dut.clock.step()
          }

          if (dut.io.outputRow.valid.peek().litToBoolean && resultsCollected < tv.numRows) {
            for (j <- 0 until p.cols)
              results(resultsCollected)(j) = dut.io.outputRow.bits(j).peek().litValue.toInt
            resultsCollected += 1
          }

          require(
            resultsCollected == tv.numRows,
            s"Case ${tv.id} tile $t: expected ${tv.numRows} results, got $resultsCollected"
          )

          for (i <- 0 until tv.numRows; j <- 0 until p.cols)
            psumBuf(i)(j) = results(i)(j)
        }

        var casePassed = 0
        var caseFailed = 0

        for (i <- 0 until tv.numRows) {
          for (j <- 0 until p.cols) {
            val actual   = psumBuf(i)(j) & 0xFFFF
            val expected = tv.expected(i)(j) & 0xFFFF
            val aFloat = bf16BitsToFloat(actual)
            val eFloat = bf16BitsToFloat(expected)

            val (ok, errs) = Bf16Compare.compare(actual, expected)

            outFile.println(
              f"${tv.id}%8d ${tv.numTiles - 1}%4d $i%4d $j%4d   0x$actual%04x       0x$expected%04x" +
                f" ${aFloat}%14.4g ${eFloat}%14.4g ${errs.absErr}%10.6g ${errs.relErr}%10.6g ${errs.ulpDiff}%4d ${if (ok) "PASS" else "FAIL"}"
            )

            if (ok) {
              casePassed += 1
            } else {
              caseFailed += 1
              println(
                f"  FAIL row $i col $j: got 0x$actual%04x ($aFloat%.6g) " +
                  f"exp 0x$expected%04x ($eFloat%.6g) " +
                  f"[absErr=${errs.absErr}%.6g relErr=${errs.relErr}%.6g ulp=${errs.ulpDiff}]"
              )
            }
          }
        }

        val totalColChecks = tv.numRows * p.cols
        println(
          f"  Result: $casePassed/$totalColChecks passed" +
            (if (caseFailed > 0) f", $caseFailed FAILED" else "")
        )
        totalPassed += casePassed
        totalFailed += caseFailed
      }

      outFile.close()
      println(s"\nTotal: $totalPassed passed, $totalFailed failed")
      totalFailed shouldBe 0
    }
  }
}
