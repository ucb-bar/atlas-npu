/*
InnerProductTreesFullMatmulTest.scala
Full tensor-register matrix multiply test for weight-stationary MXU.

Architecture under test:
  - Weight tile:  32×16 (vecLen × numLanes), loaded once per tile
  - Activation:   64×32 tensor register, streamed one row per cycle
  - Per cycle:    [1×32] act × [32×16] wgt = [1×16] BF16 partial sums
  - Full op:      [64×K] × [K×16] = [64×16]  (K = inner_dim, tiled in chunks of 32)

Test flow per case:
  For each tile t in 0..<numTiles:
    1. Load 32×16 weights (16 lanes × 32 elements each)
    2. Stream all 64 activation rows back-to-back (pipelined)
    3. Collect 64 output rows; store as psum for next tile
  After last tile, compare final outputs against Python expected values.

Run:
  mill atlas.test.testOnly atlas.ipt.InnerProductTreesFullMatmulTest
*/

package atlas.ipt

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common.InnerProductTreeParams
import scala.io.Source
import scala.collection.mutable.ArrayBuffer

class InnerProductTreesFullMatmulTest extends AnyFlatSpec with Matchers {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("InnerProductTreesFullMatmulTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("InnerProductTreesFullMatmulTest=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  // Configuration

  val vectorResource = "/mxu_full_matmul_vectors.txt"

  // Tolerances:
  // - absTolerance: treat both |value| <= absTolerance (in BF16 magnitude bits) as equal
  // - relTolerance: non-symmetric relative error threshold: |a-e| / max(|e|, eps)
  val absTolerance = 0x0040
  val relTolerance = 0.01

  // Test vector data

  case class FullMatmulVector(
    id:       Int,
    numRows:  Int,
    numTiles: Int,
    weights:  Seq[Seq[Seq[Int]]], // [tile][lane][col]
    act:      Seq[Seq[Seq[Int]]], // [tile][row][col]
    expected: Seq[Seq[Int]]       // [row][lane] (final result after all tiles)
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
          (0 until tileMap.size).map(r => tileMap(r))
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

  // BF16 helpers + error computations

  def bf16BitsToFloat(bits16: Int): Float = {
    java.lang.Float.intBitsToFloat((bits16 & 0xFFFF) << 16)
  }

  // Non-symmetric relative error (relative-to-expected):
  //   relErr = |a - e| / max(|e|, eps)
  // eps avoids division by zero; choose a tiny float-scale value.
  private val relEps: Double = java.lang.Float.MIN_NORMAL.toDouble // ~1.175e-38

  // Pass condition:
  //  1) absTiny: both BF16 magnitudes <= absTolerance (bit-magnitude, sign stripped)
  //  2) rel:     |a-e| / max(|e|, eps) < relTolerance
  def bf16WithinTolerance(actual16: Int, expected16: Int, absTol: Int, relTol: Double): Boolean = {
    val a16 = actual16   & 0xFFFF
    val e16 = expected16 & 0xFFFF

    val aMag = a16 & 0x7FFF
    val eMag = e16 & 0x7FFF

    // Treat tiny magnitudes as equal (raw BF16 magnitude bits)
    if (aMag <= absTol && eMag <= absTol) return true

    // Relative error check (float space, relative-to-expected)
    val aF = bf16BitsToFloat(a16).toDouble
    val eF = bf16BitsToFloat(e16).toDouble
    val absErr = math.abs(aF - eF)
    val denom  = math.max(math.abs(eF), relEps)
    val relErr = absErr / denom

    relErr < relTol
  }

  // Metrics for printing/debugging:
  // - absErr: |aF - eF| (float-space)
  // - relErr: absErr / max(|eF|, eps) (float-space, non-symmetric)
  case class Bf16Errors(absErr: Double, relErr: Double)

  def bf16Errors(actual16: Int, expected16: Int): Bf16Errors = {
    val a16 = actual16   & 0xFFFF
    val e16 = expected16 & 0xFFFF

    val aF = bf16BitsToFloat(a16).toDouble
    val eF = bf16BitsToFloat(e16).toDouble
    val absErr = math.abs(aF - eF)

    // Make printing sane for exact matches (including 0 vs 0)
    if (absErr == 0.0) return Bf16Errors(0.0, 0.0)

    val denom  = math.max(math.abs(eF), relEps)
    val relErr = absErr / denom
    Bf16Errors(absErr, relErr)
  }

  // DUT helpers

  def idle(dut: InnerProductTrees, p: InnerProductTreeParams): Unit = {
    dut.io.compute.valid.poke(false.B)
    dut.io.weightLoad.valid.poke(false.B)
    for (j <- 0 until p.vecLen) dut.io.compute.bits.act(j).poke(0.U)
    for (r <- 0 until p.numLanes) {
      dut.io.compute.bits.bias(r).poke(0.U)
      dut.io.compute.bits.psum(r).poke(0.U)
    }
    dut.io.compute.bits.addendSel.poke(AddendSel.UseAct)
    dut.io.weightLoad.bits.weightSel.poke(WeightSel.UseTr)
    dut.io.weightLoad.bits.laneIdx.poke(0.U)
    dut.io.weightLoad.bits.last.poke(false.B)
    for (c <- 0 until p.vecLen) {
      dut.io.weightLoad.bits.weightsDma(c).poke(0.U)
      dut.io.weightLoad.bits.weightsTr(c).poke(0.U)
    }
  }

  def loadWeights(
    dut: InnerProductTrees,
    p: InnerProductTreeParams,
    weights: Seq[Seq[Int]]
  ): Unit = {
    dut.io.weightLoad.valid.poke(true.B)
    dut.io.weightLoad.bits.weightSel.poke(WeightSel.UseTr)
    for (r <- 0 until p.numLanes) {
      dut.io.weightLoad.bits.laneIdx.poke(r.U)
      dut.io.weightLoad.bits.last.poke((r == p.numLanes - 1).B)
      for (c <- 0 until p.vecLen) {
        dut.io.weightLoad.bits.weightsTr(c).poke(weights(r)(c).U)
        dut.io.weightLoad.bits.weightsDma(c).poke(0.U)
      }
      dut.clock.step()
    }
    dut.io.weightLoad.valid.poke(false.B)
  }

  def selToEnum(sel: Int): AddendSel.Type = sel match {
    case 0 => AddendSel.UseAct
    case 1 => AddendSel.UseBias
    case 2 => AddendSel.UsePsum
    case _ => AddendSel.UseAct
  }

  // Test body

  "InnerProductTrees" should "compute full tensor-register matmul correctly" in {
    val p = InnerProductTreeParams()
    val vectors = loadVectors(vectorResource)
    require(vectors.nonEmpty, s"No test vectors found at $vectorResource")

    simulate(new InnerProductTrees(p)) { dut =>
      var totalPassed = 0
      var totalFailed = 0

      val outFile =
        new java.io.PrintWriter("../../../../../src/test/resources/rtl_full_matmul_outputs.txt")
      outFile.println(
        "# case_id  tile  row  lane  actual_hex  expected_hex  actual_float  expected_float  match"
      )

      for (tv <- vectors) {
        println(f"═══ Case ${tv.id}: ${tv.numRows} rows × ${tv.numTiles} tile(s) ═══")

        // psumBuf holds intermediate results between tiles: [row][lane]
        val psumBuf = Array.fill(tv.numRows, p.numLanes)(0)

        for (t <- 0 until tv.numTiles) {
          val sel = if (t == 0) 0 else 2

          // 1. Load weights for this tile
          idle(dut, p)
          loadWeights(dut, p, tv.weights(t))

          // 2. Stream all activation rows and collect results (pipelined)
          val results = Array.fill(tv.numRows, p.numLanes)(0)
          var resultsCollected = 0
          val totalCycles = tv.numRows + p.latency

          for (cycle <- 0 until totalCycles) {
            // Collect output if valid
            if (dut.io.out.valid.peek().litToBoolean) {
              require(
                resultsCollected < tv.numRows,
                s"Case ${tv.id} tile $t: got more outputs than expected"
              )
              for (r <- 0 until p.numLanes) {
                results(resultsCollected)(r) = dut.io.out.bits(r).peek().litValue.toInt
              }
              resultsCollected += 1
            }

            // Fire compute for this row
            if (cycle < tv.numRows) {
              dut.io.compute.valid.poke(true.B)
              for (j <- 0 until p.vecLen)
                dut.io.compute.bits.act(j).poke(tv.act(t)(cycle)(j).U)
              for (r <- 0 until p.numLanes) {
                dut.io.compute.bits.bias(r).poke(0.U)
                dut.io.compute.bits.psum(r).poke(psumBuf(cycle)(r).U)
              }
              dut.io.compute.bits.addendSel.poke(selToEnum(sel))
            } else {
              dut.io.compute.valid.poke(false.B)
            }

            dut.clock.step()
          }

          // Drain any remaining result
          if (dut.io.out.valid.peek().litToBoolean && resultsCollected < tv.numRows) {
            for (r <- 0 until p.numLanes)
              results(resultsCollected)(r) = dut.io.out.bits(r).peek().litValue.toInt
            resultsCollected += 1
          }

          require(
            resultsCollected == tv.numRows,
            s"Case ${tv.id} tile $t: expected ${tv.numRows} results, got $resultsCollected"
          )

          // Store results as psum for next tile
          for (i <- 0 until tv.numRows; r <- 0 until p.numLanes)
            psumBuf(i)(r) = results(i)(r)
        }

        // 3. Compare final results against expected
        var casePassed = 0
        var caseFailed = 0

        for (i <- 0 until tv.numRows) {
          for (r <- 0 until p.numLanes) {
            val actual   = psumBuf(i)(r)
            val expected = tv.expected(i)(r)

            val a16 = actual   & 0xFFFF
            val e16 = expected & 0xFFFF
            val aFloat = bf16BitsToFloat(a16)
            val eFloat = bf16BitsToFloat(e16)

            val ok = bf16WithinTolerance(a16, e16, absTolerance, relTolerance)

            outFile.println(
              f"${tv.id}%8d ${tv.numTiles - 1}%4d $i%4d $r%4d   0x${a16}%04x       0x${e16}%04x" +
                f" ${aFloat}%14.4g ${eFloat}%14.4g ${if (ok) "PASS" else "FAIL"}"
            )

            if (ok) {
              casePassed += 1
            } else {
              caseFailed += 1
              val errs = bf16Errors(a16, e16)

              println(
                f"  FAIL row $i lane $r: got 0x${a16}%04x ($aFloat%.6g) " +
                  f"exp 0x${e16}%04x ($eFloat%.6g) " +
                  f"[absErr=${errs.absErr}%.6g relErr=${errs.relErr}%.6g] " +
                  f"(tol: abs<=0x${absTolerance}%04x rel<${relTolerance}%.5f)"
              )
            }
          }
        }

        val totalLaneChecks = tv.numRows * p.numLanes
        println(
          f"  Result: $casePassed/$totalLaneChecks passed" +
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
