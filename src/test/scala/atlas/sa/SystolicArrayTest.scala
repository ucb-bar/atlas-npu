/*
SystolicArrayTest.scala: basic unit tests for SystolicArray (parameterized MXU)

Uses: Chisel 7 EphemeralSimulator.
Run:  mill atlas.test.testOnly atlas.sa.SystolicArrayTest

Covers: zero/ones/dot-product, bias/psum addends, double-buffer, sparse weights,
ready signals, latency, back-to-back computes. Complements SystolicArrayComprehensiveTest.

E4M3: 0.0=0x00  1.0=0x38  2.0=0x40  -1.0=0xB8
BF16: 0.0=0x0000  1.0=0x3F80  32.0=0x4200  33.0=0x4204  -32.0=0xC200  4.0=0x4080  64.0=0x4280
*/

package atlas.sa

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import atlas.common.SystolicArrayParams

class SystolicArrayTest extends AnyFlatSpec with Matchers {

  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("SystolicArrayTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("SystolicArrayTest=PASSED")
    }
    outcome
  }

  // encoding constants
  val E4M3_0   = 0x00
  val E4M3_1   = 0x38
  val E4M3_2   = 0x40
  val E4M3_N1  = 0xB8

  val BF16_0   = 0x0000
  val BF16_1   = 0x3F80
  val BF16_8   = 0x4100
  val BF16_9   = 0x4110
  val BF16_16  = 0x4180
  val BF16_32  = 0x4200
  val BF16_33  = 0x4204
  val BF16_N8  = 0xC100
  val BF16_N32 = 0xC200
  val BF16_4   = 0x4080
  val BF16_64  = 0x4280

  // BF16 encoding for value n (for dot product expected values)
  def bf16For(n: Int): Int = n match {
    case 0  => BF16_0
    case 1  => BF16_1
    case 4  => BF16_4
    case 8  => BF16_8
    case 9  => BF16_9
    case 16 => BF16_16
    case 32 => BF16_32
    case 33 => BF16_33
    case 64 => BF16_64
    case _  => throw new IllegalArgumentException(s"No BF16 constant for $n")
  }

  // Systolic array latency: rows + cols - 1 (from activation skew + PE pipes + output deskew)
  def latency(p: SystolicArrayParams): Int = p.rows + p.cols - 1

  // helper functions

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

  // Load uniform weights into all columns, one column at a time.
  def loadWeights(
    dut: SystolicArray,
    p: SystolicArrayParams,
    v: Int,
    dma: Boolean = false,
    bufSel: Int = 0,
  ): Unit = {
    dut.io.weightLoadReq.valid.poke(true.B)
    if (dma) dut.io.weightLoadReq.bits.weightSel.poke(WeightSel.UseDma)
    else dut.io.weightLoadReq.bits.weightSel.poke(WeightSel.UseTensorReg)
    dut.io.weightLoadReq.bits.weightBufWriteSel.poke(bufSel.U)

    for (i <- 0 until p.rows) {
      if (dma) {
        dut.io.weightLoadReq.bits.weightsDirectMem(i).poke(v.U)
        dut.io.weightLoadReq.bits.weightsTensorReg(i).poke(0.U)
      } else {
        dut.io.weightLoadReq.bits.weightsTensorReg(i).poke(v.U)
        dut.io.weightLoadReq.bits.weightsDirectMem(i).poke(0.U)
      }
    }
    for (j <- 0 until p.cols) {
      dut.io.weightLoadReq.bits.colIdx.poke(j.U)
      dut.clock.step()
    }
    dut.io.weightLoadReq.valid.poke(false.B)
  }

  // Load one column's weight vector (per-element values).
  def loadWeightCol(
    dut: SystolicArray,
    p: SystolicArrayParams,
    colIdx: Int,
    vals: Seq[Int],
    dma: Boolean = false,
    bufSel: Int = 0,
  ): Unit = {
    dut.io.weightLoadReq.valid.poke(true.B)
    if (dma) dut.io.weightLoadReq.bits.weightSel.poke(WeightSel.UseDma)
    else dut.io.weightLoadReq.bits.weightSel.poke(WeightSel.UseTensorReg)
    dut.io.weightLoadReq.bits.weightBufWriteSel.poke(bufSel.U)
    dut.io.weightLoadReq.bits.colIdx.poke(colIdx.U)

    for (i <- 0 until p.rows) {
      if (dma) {
        dut.io.weightLoadReq.bits.weightsDirectMem(i).poke(vals(i).U)
        dut.io.weightLoadReq.bits.weightsTensorReg(i).poke(0.U)
      } else {
        dut.io.weightLoadReq.bits.weightsTensorReg(i).poke(vals(i).U)
        dut.io.weightLoadReq.bits.weightsDirectMem(i).poke(0.U)
      }
    }
    dut.clock.step()
    dut.io.weightLoadReq.valid.poke(false.B)
  }

  // fire compute, step 1 cycle, deassert. does NOT wait for result
  def compute(
    dut: SystolicArray,
    p: SystolicArrayParams,
    act: Int,
    bias: Int = E4M3_0,
    psum: Int = BF16_0,
    sel: AddendSel.Type = AddendSel.UseZero,
    readBuf: Boolean = false,
  ): Unit = {
    dut.io.computeReq.valid.poke(true.B)
    for (i <- 0 until p.rows) dut.io.computeReq.bits.activationRow(i).poke(act.U)
    for (j <- 0 until p.cols) {
      dut.io.computeReq.bits.bias(j).poke(bias.U)
      dut.io.computeReq.bits.psum(j).poke(psum.U)
    }
    dut.io.computeReq.bits.addendSel.poke(sel)
    dut.io.computeReq.bits.weightBufReadSel.poke(readBuf.B)
    dut.clock.step()
    dut.io.computeReq.valid.poke(false.B)
  }

  // wait for the pipeline to deliver the result after compute()
  def waitForResult(dut: SystolicArray, p: SystolicArrayParams): Unit = {
    val lat = latency(p)
    if (lat > 1) dut.clock.step(lat - 1)
  }

  // all checks for one config in a single simulate() call

  def runAllChecks(p: SystolicArrayParams): Unit = {
    simulate(new SystolicArray(p)) { dut =>

      val lat = latency(p)

      // functional checks

      val dotProd = p.rows  // inner product dimension
      val dotPlus1 = bf16For(dotProd + 1)
      val dotNeg = if (dotProd == 32) BF16_N32 else BF16_N8
      val dotVal = bf16For(dotProd)
      val dotVal2 = bf16For(dotProd * 2)

      // 1. zero x zero = zero
      idle(dut, p)
      loadWeights(dut, p, E4M3_0)
      compute(dut, p, E4M3_0)
      waitForResult(dut, p)
      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(bf16For(0).U(16.W))

      // 2. all-ones weights x all-ones activations = dotProd
      idle(dut, p)
      loadWeights(dut, p, E4M3_1)
      compute(dut, p, E4M3_1)
      waitForResult(dut, p)
      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(dotVal.U(16.W))

      // 3. dot product + bias: dotProd*1 + 1 = dotProd+1
      idle(dut, p)
      loadWeights(dut, p, E4M3_1)
      compute(dut, p, E4M3_1, bias = E4M3_1, sel = AddendSel.UseBias)
      waitForResult(dut, p)
      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(dotPlus1.U(16.W))

      // 4. dot product + psum: dotProd*1 + 1 = dotProd+1
      idle(dut, p)
      loadWeights(dut, p, E4M3_1)
      compute(dut, p, E4M3_1, psum = BF16_1, sel = AddendSel.UsePsum)
      waitForResult(dut, p)
      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(dotPlus1.U(16.W))

      // 5. negative weights: dotProd*(-1) = -dotProd
      idle(dut, p)
      loadWeights(dut, p, E4M3_N1)
      compute(dut, p, E4M3_1)
      waitForResult(dut, p)
      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(dotNeg.U(16.W))

      // 6. double-buffer: TR load to buf 0, output = dotVal. then DMA load to buf 1, output = dotVal2
      idle(dut, p)
      loadWeights(dut, p, E4M3_1, dma = false, bufSel = 0)
      compute(dut, p, E4M3_1, readBuf = false)
      waitForResult(dut, p)
      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(dotVal.U(16.W))

      loadWeights(dut, p, E4M3_2, dma = true, bufSel = 1)
      compute(dut, p, E4M3_1, readBuf = true)
      waitForResult(dut, p)
      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(dotVal2.U(16.W))

      // 7. sparse: w[col0][row0] = 2, a[row0] = 2. expected output = 4 for col 0, 0 for others
      idle(dut, p)
      val sparseCol = Seq(E4M3_2) ++ Seq.fill(p.rows - 1)(E4M3_0)
      val zeroCol = Seq.fill(p.rows)(E4M3_0)
      loadWeightCol(dut, p, 0, sparseCol)
      for (j <- 1 until p.cols) {
        loadWeightCol(dut, p, j, zeroCol)
      }

      dut.io.computeReq.valid.poke(true.B)
      for (i <- 0 until p.rows)
        dut.io.computeReq.bits.activationRow(i).poke((if (i == 0) E4M3_2 else E4M3_0).U)
      for (j <- 0 until p.cols) {
        dut.io.computeReq.bits.bias(j).poke(0.U)
        dut.io.computeReq.bits.psum(j).poke(0.U)
      }
      dut.io.computeReq.bits.addendSel.poke(AddendSel.UseZero)
      dut.io.computeReq.bits.weightBufReadSel.poke(false.B)
      dut.clock.step()
      dut.io.computeReq.valid.poke(false.B)
      waitForResult(dut, p)

      dut.io.outputRow.valid.expect(true.B)
      dut.io.outputRow.bits(0).expect(BF16_4.U(16.W))
      for (j <- 1 until p.cols) dut.io.outputRow.bits(j).expect(bf16For(0).U(16.W))

      // 8. ready signals always high
      idle(dut, p)
      dut.io.computeReq.ready.expect(true.B)
      dut.io.weightLoadReq.ready.expect(true.B)
      dut.clock.step(5)
      dut.io.computeReq.ready.expect(true.B)
      dut.io.weightLoadReq.ready.expect(true.B)

      // 9. outputRow.valid pulse: idle, then expect valid = false. compute, then expect valid = true. next then expect valid = false
      idle(dut, p)
      dut.clock.step()
      dut.io.outputRow.valid.expect(false.B)

      loadWeights(dut, p, E4M3_0)
      compute(dut, p, E4M3_0)
      waitForResult(dut, p)
      dut.io.outputRow.valid.expect(true.B)

      dut.clock.step()
      dut.io.outputRow.valid.expect(false.B)

      // pipeline-specific checks (latency = rows + cols - 1)

      // 10. exact latency: valid LOW for (latency−1) cycles, HIGH for 1, LOW again
      idle(dut, p)
      loadWeights(dut, p, E4M3_1)

      dut.io.computeReq.valid.poke(true.B)
      for (i <- 0 until p.rows) dut.io.computeReq.bits.activationRow(i).poke(E4M3_1.U)
      for (j <- 0 until p.cols) {
        dut.io.computeReq.bits.bias(j).poke(0.U)
        dut.io.computeReq.bits.psum(j).poke(0.U)
      }
      dut.io.computeReq.bits.addendSel.poke(AddendSel.UseZero)
      dut.io.computeReq.bits.weightBufReadSel.poke(false.B)
      dut.clock.step()
      dut.io.computeReq.valid.poke(false.B)

      for (_ <- 0 until lat - 1) {
        dut.io.outputRow.valid.expect(false.B)
        dut.clock.step()
      }
      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(dotVal.U(16.W))

      dut.clock.step()
      dut.io.outputRow.valid.expect(false.B)

      // 11. back-to-back: two consecutive fires, then expect two consecutive valid results
      idle(dut, p)
      loadWeights(dut, p, E4M3_1)

      // fire 1: act=1, expect dotVal
      dut.io.computeReq.valid.poke(true.B)
      for (i <- 0 until p.rows) dut.io.computeReq.bits.activationRow(i).poke(E4M3_1.U)
      for (j <- 0 until p.cols) {
        dut.io.computeReq.bits.bias(j).poke(0.U)
        dut.io.computeReq.bits.psum(j).poke(0.U)
      }
      dut.io.computeReq.bits.addendSel.poke(AddendSel.UseZero)
      dut.io.computeReq.bits.weightBufReadSel.poke(false.B)
      dut.clock.step()

      // fire 2: act=2, expect dotVal2 (valid still asserted)
      for (i <- 0 until p.rows) dut.io.computeReq.bits.activationRow(i).poke(E4M3_2.U)
      dut.clock.step()
      dut.io.computeReq.valid.poke(false.B)

      // drain pipeline to first result
      if (lat > 2) dut.clock.step(lat - 2)

      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(dotVal.U(16.W))

      dut.clock.step()
      dut.io.outputRow.valid.expect(true.B)
      for (j <- 0 until p.cols) dut.io.outputRow.bits(j).expect(dotVal2.U(16.W))

      dut.clock.step()
      dut.io.outputRow.valid.expect(false.B)
    }
  }

  // test cases: one Verilator compilation each

  "SystolicArray (32x16)" should "pass all checks" in {
    runAllChecks(SystolicArrayParams())
  }

  "SystolicArray (8x4)" should "pass all checks" in {
    runAllChecks(SystolicArrayParams(rows = 8, cols = 4))
  }
}