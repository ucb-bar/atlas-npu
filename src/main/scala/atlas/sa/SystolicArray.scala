package atlas.sa

import chisel3._
import chisel3.util._

import sp26FPUnits._
import sp26FPUnits.hardfloat._
import atlas.common.SystolicArrayParams
import sp26FPUnits.OutputFmtSel

// This code follows the lowRISC Style Guide: https://github.com/lowRISC/style-guides

// ChiselEnums
object AddendSel extends ChiselEnum {
  val UseBias, UsePsum, UseZero = Value
}
object WeightSel extends ChiselEnum {
  val UseDma, UseTensorReg = Value
}

// IO bundles
/**
  * Decoupled bundle of signals describing a compute request.
  * 
  * - activationRow: 
  * - bias: IEEE/standard width of type inT.
  * - psum: Partial sum, IEEE/standard width of type outT.
  * - addendSel: Select signal for bias, psum, or 0 for the b in y = x(A^T) + b.
  * - weightBufReadSel: Select signal for weight buffer to read from; false.B is Buffer 0 and true.B is Buffer 1
  * - scalingFactor: TBD, will be used to downscale output to fp8
  */
class ComputeRequest(p: SystolicArrayParams) extends Bundle {
  val activationRow = Vec(p.rows, UInt(p.inT.ieeeWidth.W))
  val bias = Vec(p.cols, UInt(p.inT.ieeeWidth.W))
  val psum = Vec(p.cols, UInt(p.outT.ieeeWidth.W))
  val addendSel = AddendSel()
  val weightBufReadSel = Bool()
  val scaleExp  = SInt(8.W) // scalar for whole row
  val outFmtSel = OutputFmtSel()
}

/**
  * Decoupled bundle of signals describing a weight load request.
  * 
  * - weightsDirectMem: Weight vector from DMA.
  * - weightsTensorReg: Weight vector from Tensor Register.
  * - weightSel: Chooses between weight vector from DMA and Tensor Register.
  * - weightBufWriteSel: Which double buffer to write to; false.B is Buffer 0 and true.B is Buffer 1
  */
class WeightLoadReq(p: SystolicArrayParams) extends Bundle {
  val weightsDirectMem = Vec(p.rows, UInt(p.inT.ieeeWidth.W))
  val weightsTensorReg = Vec(p.rows, UInt(p.inT.ieeeWidth.W))
  val weightSel = WeightSel()
  val weightBufWriteSel = UInt(1.W)
  val colIdx = UInt(log2Ceil(p.cols).W)
}

/**
  * Computes the equivalent of PyTorch's torch.nn.functional.linear,
  * y = x(A^T) + b.
  * 
  * https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
  * 
  * The dimensions of this systolic array are customizable in SystolicArrayParams.
  * 
  * - y: outputRow
  * - x: activationRow. Comes from the Tensor Registers.
  * - A^T: A matrix, transposed (composed of column vectors), used for weights.
  *   Passed in column by column as weight columns in weightLoadReq. Can either
  *   come from DMA or the Tensor Registers.
  * - b: bias
  * 
  * All inputs should be in standard/IEEE width of type inT (see 
  * SystolicArrayParams), apart from the partial sum psum. The psum 
  * is also of standard/IEEE width, but of type outT (see SystolicArrayParams).
  * 
  * To perform computations, all inputs are first converted to Berkeley
  * HardFloat recoded width of type outT. Then, fused multiply-add modules
  * are chained together to calculate y = x(A^T) + b.
  * 
  * The computed result is converted back into standard/IEEE format and
  * outputted as outputRow.
  */

class SystolicArray(p: SystolicArrayParams) extends Module {

  val io = IO(new Bundle {
    val computeReq = Flipped(Decoupled(new ComputeRequest(p)))
    val weightLoadReq = Flipped(Decoupled(new WeightLoadReq(p)))
    val outputRow = Valid(Vec(p.cols, UInt(p.outT.ieeeWidth.W)))
  })

  // Recodes IEEE/standard numbers to Berkeley HardFloat format.
  private def recode(bits: UInt, t: AtlasFPType): UInt = {
    if (t.ieeeWidth == 8) {
      // TODO: hardcoded altfmt to false.B since we're using E4M3. need to change this later
      AtlasFPType.E5M3.recode(fp8ToE5M3(bits, false.B))
    } else {
      t.recode(bits)
    }
  }
  private def recode(data: Vec[UInt], t: AtlasFPType): Vec[UInt] = {
    VecInit.tabulate(data.length)(i => recode(data(i), t))
  }

  // Decodes Berkeley HardFloat format back into IEEE/standard numbers. Assume t is not E4M3/E5M2.
  private def decode(bits: UInt, t: AtlasFPType): UInt = {
    fNFromRecFN(t.expWidth, t.sigWidth, bits)
  }
  private def decode(data: Vec[UInt], t: AtlasFPType): Vec[UInt] = {
    VecInit.tabulate(data.length)(i => decode(data(i), t))
  }

  // Converts from a Berkeley HardFloat format to another.
  private def convert(bits: UInt, inT: AtlasFPType, outT: AtlasFPType): UInt = {
    val converter: RecFNToRecFN = {
      if (inT.ieeeWidth == 8) {
        val e5m3 = AtlasFPType.E5M3
        Module(new RecFNToRecFN(e5m3.expWidth, e5m3.sigWidth, outT.expWidth, outT.sigWidth))
      } else {
        Module(new RecFNToRecFN(inT.expWidth, inT.sigWidth, outT.expWidth, outT.sigWidth))
      }
    }
    converter.io.in := bits
    converter.io.roundingMode := consts.round_near_even
    converter.io.detectTininess := consts.tininess_afterRounding
    converter.io.exceptionFlags := DontCare
    converter.io.out
  }
  private def convert(data: Vec[UInt], inT: AtlasFPType, outT: AtlasFPType): Vec[UInt] = {
    VecInit.tabulate(data.length)(i => convert(data(i), inT, outT))
  }

  // Always ready to read activation and weights.
  io.computeReq.ready := true.B
  io.weightLoadReq.ready := true.B
  val computeReq = io.computeReq.bits
  val weightLoadReq = io.weightLoadReq.bits

  // Create an array of PEs and connect them to each other.
  val pes = for (i <- 0 until p.rows) yield {
    for (j <- 0 until p.cols) yield {
      Module(new ProcessingElement(p.outT, p.useFP32Accumulation, p.useE4M3FMA)).suggestName(s"pe_${i}_${j}")
    }
  }
  for (i <- 0 until p.rows) {
    for (j <- 0 until (p.cols - 1)) {
      pes(i)(j + 1).io.act <> pes(i)(j).io.actQ
      pes(i)(j + 1).io.weightBufReadSel := pes(i)(j).io.weightBufReadSelQ
    }
  }
  for (i <- 0 until (p.rows - 1)) {
    for (j <- 0 until p.cols) {
      pes(i + 1)(j).io.addend <> pes(i)(j).io.macQ
    }
  }

  // Recode and convert addends (biases, psums, and zeroes). With FP32 accumulation, convert to FP32; otherwise use outT.
  // With useE4M3FMA, addends are BF16 IEEE (16b); otherwise recoded.
  val addendT: AtlasFPType = {
    if (p.useFP32Accumulation) {
      AtlasFPType.FP32
    } else {
      p.outT
    }
  }
  val peAccumWidth = if (p.useE4M3FMA) 16 else addendT.recodedWidth
  val biasRec = convert(recode(computeReq.bias, p.inT), p.inT, addendT)
  val psumRec = convert(recode(computeReq.psum, p.outT), p.outT, addendT)
  val zeroRec = convert(recode(VecInit(Seq.fill(p.cols)(0.U(p.outT.ieeeWidth.W))), p.outT), p.outT, addendT)

  val biasIEEE = decode(biasRec, p.outT)
  val psumIEEE = computeReq.psum
  val zeroIEEE = VecInit(Seq.fill(p.cols)(0.U(p.outT.ieeeWidth.W)))

  // Mux between bias, psum, and zero. useE4M3FMA uses BF16 IEEE; otherwise recoded.
  val addendsRec = MuxCase(zeroRec, Array(
    (computeReq.addendSel === AddendSel.UseBias) -> biasRec,
    (computeReq.addendSel === AddendSel.UsePsum) -> psumRec,
    (computeReq.addendSel === AddendSel.UseZero) -> zeroRec
  ))
  val addendsIEEE = MuxCase(zeroIEEE, Array(
    (computeReq.addendSel === AddendSel.UseBias) -> biasIEEE,
    (computeReq.addendSel === AddendSel.UsePsum) -> psumIEEE,
    (computeReq.addendSel === AddendSel.UseZero) -> zeroIEEE
  ))
  val addends = VecInit.tabulate(p.cols) { j =>
    if (p.useE4M3FMA) {
      Cat(0.U((addendT.recodedWidth - 16).W), addendsIEEE(j))
    } else {
      addendsRec(j)
    }
  }
  for (j <- 0 until p.cols) {
    val addendsPipe = Pipe(io.computeReq.valid, addends(j), j)
    pes(0)(j).io.addend.valid := addendsPipe.valid
    pes(0)(j).io.addend.bits := addendsPipe.bits(peAccumWidth - 1, 0)
  }

  // Skew the activations, recode and convert to type outT, and connect to left column of PEs.
  // Also skew the weightBufReadSel and connect to left column of PEs.
  // When useE4M3FMA, pass raw E4M3 in lower 8 bits.
  for (i <- 0 until p.rows) {
    val actPipe = Pipe(io.computeReq.valid, computeReq.activationRow(i), i)
    val actData =
      if (p.useE4M3FMA) {
        Cat(0.U((p.outT.recodedWidth - p.inT.ieeeWidth).W), actPipe.bits)
      } else {
        convert(recode(actPipe.bits, p.inT), p.inT, p.outT)
      }
    pes(i)(0).io.act.bits := actData
    pes(i)(0).io.act.valid := actPipe.valid

    pes(i)(0).io.weightBufReadSel := ShiftRegister(computeReq.weightBufReadSel, i)
  }

  // Depending on the outFmtSel, output raw results in BF16 or scale back down into FP8, pad with zeroes and output.
  val outFmtSelPipe = Pipe(io.computeReq.valid, computeReq.outFmtSel, p.rows + p.cols - 1)
  val scaleExpPipe = Pipe(io.computeReq.valid, computeReq.scaleExp, p.rows + p.cols - 1)


  val outConverters = for (j <- 0 until p.cols) yield { Module(new OutputConvStage()) }

  // Decode and deskew output of PEs, and connect to output port.
  // TODO: Decode then deskew, or deskew then decode?
  // When useE4M3FMA, macQ is already BF16 IEEE; otherwise convert and decode. 
  val outputPipes = for (j <- 0 until p.cols) yield {
    val outputDec =
      if (p.useE4M3FMA) {
        pes(p.rows - 1)(j).io.macQ.bits
      } else {
        val outputData = convert(pes(p.rows - 1)(j).io.macQ.bits, addendT, p.outT)
        decode(outputData, p.outT)
      }
    val pipe = Pipe(pes(p.rows - 1)(j).io.macQ.valid, outputDec, (p.cols - 1) - j)

    outConverters(j).io.bf16In := pipe.bits 
    outConverters(j).io.scaleExp := scaleExpPipe.bits
    outConverters(j).io.outFmtSel := outFmtSelPipe.bits
    io.outputRow.bits(j) := outConverters(j).io.out
    pipe
  }
  val outputValids = VecInit.tabulate(p.cols)(i => outputPipes(i).valid)
  io.outputRow.valid := outputValids.reduce(_ & _)


  // Select, recode, and convert weights to type outT. When useE4M3FMA, store raw E4M3 in lower 8 bits.
  val weights = Mux(weightLoadReq.weightSel === WeightSel.UseDma, weightLoadReq.weightsDirectMem, weightLoadReq.weightsTensorReg)
  val weightsRec = convert(recode(weights, p.inT), p.inT, p.outT)

  // Double buffering of weights. [w0 or w1][column index][row index]
  // TODO: Store weight buffers differently for useE4M3FMA (only 8 bits needed since we don't need to convert for hardfloat)
  val weightBufs = Reg(Vec(2, Vec(p.cols, Vec(p.rows, UInt(p.outT.recodedWidth.W)))))
  val weightsStored = VecInit.tabulate(p.rows) { r =>
    if (p.useE4M3FMA) {
      Cat(0.U((p.outT.recodedWidth - p.inT.ieeeWidth).W), weights(r))
    } else {
      weightsRec(r)
    }
  }
  when (io.weightLoadReq.fire) {
    weightBufs(weightLoadReq.weightBufWriteSel)(weightLoadReq.colIdx) := weightsStored
  }

  // Connect weight buffers to each PE.
  for (i <- 0 until p.rows) {
    for (j <- 0 until p.cols) {
      pes(i)(j).io.weight0 := weightBufs(0)(j)(i)
      pes(i)(j).io.weight1 := weightBufs(1)(j)(i)
    }
  }


}