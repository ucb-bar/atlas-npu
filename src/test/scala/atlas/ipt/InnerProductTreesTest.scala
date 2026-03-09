/*
InnerProductTreesTest.scala: unit tests for the parameterized MXU

Uses: Chisel 7 built-in EphemeralSimulator.
Run:  mill atlas.test.testOnly atlas.ipt.InnerProductTreesTest

Basic Encodings (Int to FP mappings):
  E4M3 (bias=7):     0.0 = 0x00     1.0 = 0x38      2.0 = 0x40     -1.0 = 0xB8
  BF16 (bias=127):   0.0 = 0x0000   1.0 = 0x3F80   32.0 = 0x4200   33.0 = 0x4204
                   -32.0 = 0xC200   4.0 = 0x4080   64.0 = 0x4280
*/

package atlas.ipt

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome // <---------- INCLUDE THIS FOR CI/CD
import atlas.common.InnerProductTreeParams

class InnerProductTreesTest extends AnyFlatSpec with Matchers {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("InnerProductTreesTest=FAILED")
    } else if (outcome.isSucceeded) {
      println("InnerProductTreesTest=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  // encoding constants
  val E4M3_0   = 0x00
  val E4M3_1   = 0x38
  val E4M3_2   = 0x40
  val E4M3_N1  = 0xB8

  val BF16_0   = 0x0000
  val BF16_1   = 0x3F80
  val BF16_32  = 0x4200
  val BF16_33  = 0x4204
  val BF16_N32 = 0xC200
  val BF16_4   = 0x4080
  val BF16_64  = 0x4280

  // helper functions

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

  // Load uniform weights into all lanes, one vector at a time.
  def loadWeights(dut: InnerProductTrees, p: InnerProductTreeParams, v: Int, dma: Boolean = false): Unit = {
    dut.io.weightLoad.valid.poke(true.B)
    if (dma) dut.io.weightLoad.bits.weightSel.poke(WeightSel.UseDma)
    else dut.io.weightLoad.bits.weightSel.poke(WeightSel.UseTr)

    for (c <- 0 until p.vecLen) {
      if (dma) {
        dut.io.weightLoad.bits.weightsDma(c).poke(v.U)
        dut.io.weightLoad.bits.weightsTr(c).poke(0.U)
      } else {
        dut.io.weightLoad.bits.weightsTr(c).poke(v.U)
        dut.io.weightLoad.bits.weightsDma(c).poke(0.U)
      }
    }
    for (r <- 0 until p.numLanes) {
      dut.io.weightLoad.bits.laneIdx.poke(r.U)
      dut.io.weightLoad.bits.last.poke((r == p.numLanes - 1).B)
      dut.clock.step()
    }
    dut.io.weightLoad.valid.poke(false.B)
  }

  // Load one lane's weight vector (per-element values).
  def loadWeightVec(
    dut: InnerProductTrees,
    p: InnerProductTreeParams,
    laneIdx: Int,
    vals: Seq[Int],
    dma: Boolean = false,
    last: Boolean = false,
  ): Unit = {
    dut.io.weightLoad.valid.poke(true.B)
    if (dma) dut.io.weightLoad.bits.weightSel.poke(WeightSel.UseDma)
    else dut.io.weightLoad.bits.weightSel.poke(WeightSel.UseTr)

    dut.io.weightLoad.bits.laneIdx.poke(laneIdx.U)
    dut.io.weightLoad.bits.last.poke(last.B)
    for (c <- 0 until p.vecLen) {
      if (dma) {
        dut.io.weightLoad.bits.weightsDma(c).poke(vals(c).U)
        dut.io.weightLoad.bits.weightsTr(c).poke(0.U)
      } else {
        dut.io.weightLoad.bits.weightsTr(c).poke(vals(c).U)
        dut.io.weightLoad.bits.weightsDma(c).poke(0.U)
      }
    }
    dut.clock.step()
    dut.io.weightLoad.valid.poke(false.B)
  }

  // fire compute, step 1 cycle, deassert. does NOT wait for result
  def compute(
    dut: InnerProductTrees,
    p: InnerProductTreeParams,
    act: Int,
    bias: Int = E4M3_0,
    psum: Int = BF16_0,
    sel: AddendSel.Type = AddendSel.UseAct,
  ): Unit = {
    dut.io.compute.valid.poke(true.B)
    for (j <- 0 until p.vecLen) dut.io.compute.bits.act(j).poke(act.U)
    for (r <- 0 until p.numLanes) {
      dut.io.compute.bits.bias(r).poke(bias.U)
      dut.io.compute.bits.psum(r).poke(psum.U)
    }
    dut.io.compute.bits.addendSel.poke(sel)
    dut.clock.step()
    dut.io.compute.valid.poke(false.B)
  }

  // wait for the pipeline to deliver the result after compute()
  def waitForResult(dut: InnerProductTrees, p: InnerProductTreeParams): Unit = {
    if (p.latency > 1) dut.clock.step(p.latency - 1)
  }

  // all checks for one config in a single simulate() call

  def runAllChecks(p: InnerProductTreeParams): Unit = {
    simulate(new InnerProductTrees(p)) { dut =>

      // functional checks

      // 1. zero x zero = zero
      idle(dut, p)
      loadWeights(dut, p, E4M3_0)
      compute(dut, p, E4M3_0)
      waitForResult(dut, p)
      dut.io.out.valid.expect(true.B)
      for (r <- 0 until p.numLanes) dut.io.out.bits(r).expect(BF16_0.U)

      // 2. all-ones weights x all-ones activations = 32
      idle(dut, p)
      loadWeights(dut, p, E4M3_1)
      compute(dut, p, E4M3_1)
      waitForResult(dut, p)
      dut.io.out.valid.expect(true.B)
      for (r <- 0 until p.numLanes) dut.io.out.bits(r).expect(BF16_32.U)

      // 3. dot product + bias: 32x1 + 1 = 33
      idle(dut, p)
      loadWeights(dut, p, E4M3_1)
      compute(dut, p, E4M3_1, bias = E4M3_1, sel = AddendSel.UseBias)
      waitForResult(dut, p)
      dut.io.out.valid.expect(true.B)
      for (r <- 0 until p.numLanes) dut.io.out.bits(r).expect(BF16_33.U)

      // 4. dot product + psum: 32x1 + 1 = 33
      idle(dut, p)
      loadWeights(dut, p, E4M3_1)
      compute(dut, p, E4M3_1, psum = BF16_1, sel = AddendSel.UsePsum)
      waitForResult(dut, p)
      dut.io.out.valid.expect(true.B)
      for (r <- 0 until p.numLanes) dut.io.out.bits(r).expect(BF16_33.U)

      // 5. negative weights: 32x−1 = −32
      idle(dut, p)
      loadWeights(dut, p, E4M3_N1)
      compute(dut, p, E4M3_1)
      waitForResult(dut, p)
      dut.io.out.valid.expect(true.B)
      for (r <- 0 until p.numLanes) dut.io.out.bits(r).expect(BF16_N32.U)

      // 6. double-buffer: TR load, output = 32. then DMA load, output = 64
      idle(dut, p)
      loadWeights(dut, p, E4M3_1, dma = false)
      compute(dut, p, E4M3_1)
      waitForResult(dut, p)
      dut.io.out.bits(0).expect(BF16_32.U)

      loadWeights(dut, p, E4M3_2, dma = true)
      compute(dut, p, E4M3_1)
      waitForResult(dut, p)
      dut.io.out.bits(0).expect(BF16_64.U)

      // 7. sparse: w[r][0] = 2, a[0] = 2. expected output = 4
      idle(dut, p)
      val sparseVec = Seq(E4M3_2) ++ Seq.fill(p.vecLen - 1)(E4M3_0)
      for (r <- 0 until p.numLanes) {
        loadWeightVec(dut, p, r, sparseVec, dma = false, last = r == p.numLanes - 1)
      }

      dut.io.compute.valid.poke(true.B)
      for (j <- 0 until p.vecLen)
        dut.io.compute.bits.act(j).poke((if (j == 0) E4M3_2 else E4M3_0).U)
      for (r <- 0 until p.numLanes) {
        dut.io.compute.bits.bias(r).poke(0.U)
        dut.io.compute.bits.psum(r).poke(0.U)
      }
      dut.io.compute.bits.addendSel.poke(AddendSel.UseAct)
      dut.clock.step()
      dut.io.compute.valid.poke(false.B)
      waitForResult(dut, p)

      dut.io.out.valid.expect(true.B)
      for (r <- 0 until p.numLanes) dut.io.out.bits(r).expect(BF16_4.U)

      // 8. ready signals always high
      idle(dut, p)
      dut.io.compute.ready.expect(true.B)
      dut.io.weightLoad.ready.expect(true.B)
      dut.clock.step(5)
      dut.io.compute.ready.expect(true.B)
      dut.io.weightLoad.ready.expect(true.B)

      // 9. out.valid pulse: idle, then expect valid = false. compute, then expect valid = true. next then expect valid = false
      idle(dut, p)
      dut.clock.step()
      dut.io.out.valid.expect(false.B)

      loadWeights(dut, p, E4M3_0)
      compute(dut, p, E4M3_0)
      waitForResult(dut, p)
      dut.io.out.valid.expect(true.B)

      dut.clock.step()
      dut.io.out.valid.expect(false.B)

      // pipeline-specific checks

      if (p.numPipeCuts > 0) {

        // 10. exact latency: valid LOW for (latency−1) cycles, HIGH for 1, LOW again
        idle(dut, p)
        loadWeights(dut, p, E4M3_1)

        dut.io.compute.valid.poke(true.B)
        for (j <- 0 until p.vecLen) dut.io.compute.bits.act(j).poke(E4M3_1.U)
        for (r <- 0 until p.numLanes) {
          dut.io.compute.bits.bias(r).poke(0.U)
          dut.io.compute.bits.psum(r).poke(0.U)
        }
        dut.io.compute.bits.addendSel.poke(AddendSel.UseAct)
        dut.clock.step()
        dut.io.compute.valid.poke(false.B)

        for (_ <- 0 until p.latency - 1) {
          dut.io.out.valid.expect(false.B)
          dut.clock.step()
        }
        dut.io.out.valid.expect(true.B)
        for (r <- 0 until p.numLanes) dut.io.out.bits(r).expect(BF16_32.U)

        dut.clock.step()
        dut.io.out.valid.expect(false.B)

        // 11. back-to-back: two consecutive fires, then expect two consecutive valid results
        idle(dut, p)
        loadWeights(dut, p, E4M3_1)

        // fire 1: act=1, expect 32
        dut.io.compute.valid.poke(true.B)
        for (j <- 0 until p.vecLen) dut.io.compute.bits.act(j).poke(E4M3_1.U)
        for (r <- 0 until p.numLanes) {
          dut.io.compute.bits.bias(r).poke(0.U)
          dut.io.compute.bits.psum(r).poke(0.U)
        }
        dut.io.compute.bits.addendSel.poke(AddendSel.UseAct)
        dut.clock.step()

        // fire 2: act=2, expect 64 (valid still asserted)
        for (j <- 0 until p.vecLen) dut.io.compute.bits.act(j).poke(E4M3_2.U)
        dut.clock.step()
        dut.io.compute.valid.poke(false.B)

        // drain pipeline to first result
        if (p.latency > 2) dut.clock.step(p.latency - 2)

        dut.io.out.valid.expect(true.B)
        for (r <- 0 until p.numLanes) dut.io.out.bits(r).expect(BF16_32.U)

        dut.clock.step()
        dut.io.out.valid.expect(true.B)
        for (r <- 0 until p.numLanes) dut.io.out.bits(r).expect(BF16_64.U)

        dut.clock.step()
        dut.io.out.valid.expect(false.B)
      }
    }
  }

  // test cases: one Verilator compilation each

  "InnerProductTrees (combinational)" should "pass all checks" in {
    runAllChecks(InnerProductTreeParams())
  }

  "InnerProductTrees (2-stage)" should "pass all checks" in {
    runAllChecks(InnerProductTreeParams.withPipelineDepth(2))
  }

  "InnerProductTrees (3-stage)" should "pass all checks" in {
    runAllChecks(InnerProductTreeParams.withPipelineDepth(3))
  }

  "InnerProductTrees (4-stage)" should "pass all checks" in {
    runAllChecks(InnerProductTreeParams.withPipelineDepth(4))
  }
}