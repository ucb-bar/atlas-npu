package atlas.sa

import chisel3._
import chisel3.util._

import sp26FPUnits._
import sp26FPUnits.hardfloat._

/**
  * One processing element in systolic array. Computes the
  * fused multiply-add by doing (act * weight) + addend with
  * the floating point format T. The format T should be
  * supported by Berkeley HardFloat and have recoded width.
  *
  * When useFP32Accumulation is true: multiply in T (e.g. BF16), accumulate in FP32;
  * addend and macQ use FP32 recoded width. Otherwise addend and macQ use T.
  *
  * When useE4M3FMA is true: use custom E4M3Mul + E4M3ProdAddBF16; act/weight are raw E4M3 (8b),
  * addend/macQ are BF16 IEEE (16b). Mutually exclusive with useFP32Accumulation.
  *
  * Inputs:
  *   - act (an activation element of type T with recodedWidth, or raw E4M3 in lower 8b when useE4M3FMA)
  *   - weight0 (a weight element of type T with recodedWidth, or raw E4M3 in lower 8b when useE4M3FMA)
  *   - weight1 (a weight element of type T with recodedWidth, or raw E4M3 in lower 8b when useE4M3FMA)
  *   - weightBufReadSel (a Bool to select between weight0 and weight1)
  *   - addend (accumulation type: FP32 recoded when useFP32Accumulation, BF16 IEEE when useE4M3FMA, else T)
  *
  * Outputs:
  *   - actQ (type T with recodedWidth, or raw E4M3 in lower 8b when useE4M3FMA)
  *   - macQ (accumulation type: FP32 recoded when useFP32Accumulation, BF16 IEEE when useE4M3FMA, else T)
  *
  * By default, T = BF16.
  * Type T should be supported by Berkeley HardFloat.
  */

class ProcessingElement(T: AtlasFPType, useFP32Accumulation: Boolean, useE4M3FMA: Boolean = false) extends Module {
  // When using FP32 accumulation, addend and macQ are FP32 recoded; when useE4M3FMA, BF16 IEEE (16b); otherwise T.
  val accumT: AtlasFPType = if (useFP32Accumulation) AtlasFPType.FP32 else T
  val accumWidth = if (useE4M3FMA) 16 else accumT.recodedWidth

  val io = IO(new Bundle {
    val act    = Flipped(Valid(UInt(T.recodedWidth.W)))
    val weight0 = Input(UInt(T.recodedWidth.W))
    val weight1 = Input(UInt(T.recodedWidth.W))
    val weightBufReadSel = Input(Bool())
    val addend = Flipped(Valid(UInt(accumWidth.W)))

    val actQ = Valid(UInt(T.recodedWidth.W))
    val macQ = Valid(UInt(accumWidth.W))
    val weightBufReadSelQ = Output(Bool())
	})

  val weight = Mux(io.weightBufReadSel, io.weight1, io.weight0)

  if (useE4M3FMA) {
    val fma = Module(new E4M3FMA)
    fma.io.a := io.act.bits(7, 0)
    fma.io.b := weight(7, 0)
    fma.io.addend16 := io.addend.bits

    io.actQ := Pipe(io.act.valid, io.act.bits, 1)
    io.macQ := Pipe(io.act.valid && io.addend.valid, fma.io.out16, 1)
    io.weightBufReadSelQ := RegNext(io.weightBufReadSel)
  } else if (useFP32Accumulation) {
    val fp32 = AtlasFPType.FP32
    val mulT = T

    // Multiply in BF16
    val mul = Module(new MulRecFN(mulT.expWidth, mulT.sigWidth))
    mul.io.a := io.act.bits
    mul.io.b := weight
    mul.io.roundingMode := consts.round_near_even
    mul.io.detectTininess := consts.tininess_afterRounding

    // Widen product BF16 -> FP32
    val widen = Module(new RecFNToRecFN(mulT.expWidth, mulT.sigWidth, fp32.expWidth, fp32.sigWidth))
    widen.io.in := mul.io.out
    widen.io.roundingMode := consts.round_near_even
    widen.io.detectTininess := consts.tininess_afterRounding
    widen.io.exceptionFlags := DontCare

    // Add in FP32
    val add = Module(new AddRecFN(fp32.expWidth, fp32.sigWidth))
    add.io.subOp := false.B
    add.io.a := widen.io.out
    add.io.b := io.addend.bits
    add.io.roundingMode := consts.round_near_even
    add.io.detectTininess := consts.tininess_afterRounding

    io.actQ := Pipe(io.act.valid, io.act.bits, 1)
    io.macQ := Pipe(io.act.valid && io.addend.valid, add.io.out, 1)
    io.weightBufReadSelQ := RegNext(io.weightBufReadSel)
  } else {
    // Fused multiply-add/multiply-accumulate (act * weight) + addend.
    val mac = Module(new MulAddRecFN(T.expWidth, T.sigWidth))
    mac.io.op := 0.U(2.W) // Bits to specify fused multiply-add operation.
    mac.io.a := io.act.bits
    mac.io.b := weight
    mac.io.c := io.addend.bits
    mac.io.roundingMode := consts.round_near_even
    mac.io.detectTininess := consts.tininess_afterRounding

    io.actQ := Pipe(io.act.valid, io.act.bits, 1)
    io.macQ := Pipe(io.act.valid && io.addend.valid, mac.io.out, 1)
    io.weightBufReadSelQ := RegNext(io.weightBufReadSel)
  }
}
