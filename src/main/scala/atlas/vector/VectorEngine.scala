// ============================================================================
// VectorEngine.scala — Standalone vector-engine wrapper.
//
// Dispatches one decoded vector operation to the appropriate lane-box module
// and forwards the resulting vector back through small Valid-only interfaces.
//
// Software-scheduled model:
//   The scalar core / compiler guarantees that commands are only issued
//   when the selected op's issue-busy bit is low.
//   `io.busy` reports only that some VPU work is still in flight.
//   No hardware backpressure is provided anywhere in the vector path.
//   Assertions fire on violations to aid debugging.
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import atlas.common.VpuParams
import sp26FPUnits._
import sp26FPUnits.hardfloat._

class VectorEngine(val p: VpuParams) extends Module with VectorParam {
	val io = IO(new Bundle {
			val inst   = Flipped(Valid(new VectorInput(p)))
			val data1  = Flipped(Valid(new RegData()))
			val data2  = Flipped(Valid(new RegData()))
			val read1  = Output(Valid(new RegFileReadReq()))
			val read2  = Output(Valid(new RegFileReadReq()))
			val write1 = Output(Valid(new RegFileWriteReq()))
			val write2 = Output(Valid(new RegFileWriteReq()))
      val activeReads  = Output(Vec(4, Valid(UInt(6.W))))
      val activeWrites = Output(Vec(4, Valid(UInt(6.W))))
      val issueBusy = Output(UInt((VPUOp.all.size + 1).W))
      val busy   = Output(Bool())
  })

  private def opUsesDoubleRead(op: VPUOp.Type): Bool =
    (op === VPUOp.add) || (op === VPUOp.sub) || (op === VPUOp.mul) ||
    (op === VPUOp.pairmax) || (op === VPUOp.pairmin) ||
    (op === VPUOp.rsum) || (op === VPUOp.rmax) || (op === VPUOp.rmin)

  // Internal Interface Signals
  val dataInFire1  = WireDefault(false.B)
  val dataInFire2  = WireDefault(false.B)
  val dataOutFire1 = WireDefault(false.B)
  val dataOutFire2 = WireDefault(false.B)
  val divSqrtReady = WireDefault(false.B)
  val writeData1   = Wire(UInt(256.W))
  val writeData2   = Wire(UInt(256.W))

  // Convert raw E8M0 into a clamped signed scale exponent.
  private def e8m0ToScaleExpClamped(scaleE8M0: UInt): SInt = {
    val scaleExpWide = scaleE8M0.zext -& 127.S(9.W)
    val scaleExp     = Wire(SInt(8.W))
    when(scaleExpWide > 127.S)           { scaleExp := 127.S }
      .elsewhen(scaleExpWide < (-128).S) { scaleExp := (-128).S }
      .otherwise                         { scaleExp := scaleExpWide(7, 0).asSInt }
    scaleExp
  }

	// FSM
	val fsm = Module(new VectorFSM(p))

  // ── Software-scheduled acceptance ──
  // Commands launch directly; illegal timing is caught by the assert below.
  fsm.io.in.instFire               := io.inst.valid
  fsm.io.in.instType               := io.inst.bits.instType
  fsm.io.in.instReadBank1          := io.inst.bits.instReadBank1
  fsm.io.in.instReadBank2          := io.inst.bits.instReadBank2
  fsm.io.in.instWriteBank          := io.inst.bits.instWriteBank
  fsm.io.in.instPackScaleE8M0      := io.inst.bits.packScaleE8M0
  fsm.io.in.instUnpackScaleE8M0    := io.inst.bits.unpackScaleE8M0
  fsm.io.in.dataInFire1            := dataInFire1
  fsm.io.in.dataInFire2            := dataInFire2
  fsm.io.in.dataOutFire1           := dataOutFire1
  fsm.io.in.dataOutFire2           := dataOutFire2
  fsm.io.in.divSqrtReady           := divSqrtReady
  fsm.io.in.instImm                := io.inst.bits.imm

	// Instruction Decodes
	val isAdd1     = (fsm.io.out.inst1 === VPUOp.add     )
  val isSub1     = (fsm.io.out.inst1 === VPUOp.sub     )
  val isMul1     = (fsm.io.out.inst1 === VPUOp.mul     )
  val isRcp1     = (fsm.io.out.inst1 === VPUOp.rcp     )
  val isSqrt1    = (fsm.io.out.inst1 === VPUOp.sqrt    )
  val isExp1     = (fsm.io.out.inst1 === VPUOp.exp     )
  val isExp21    = (fsm.io.out.inst1 === VPUOp.exp2    )
  val isSin1     = (fsm.io.out.inst1 === VPUOp.sin     )
  val isCos1     = (fsm.io.out.inst1 === VPUOp.cos     )
  val isTanh1    = (fsm.io.out.inst1 === VPUOp.tanh    )
  val isLog1     = (fsm.io.out.inst1 === VPUOp.log     )
  val isRMax1    = (fsm.io.out.inst1 === VPUOp.rmax    )
  val isRMin1    = (fsm.io.out.inst1 === VPUOp.rmin    )
  val isCMax1    = (fsm.io.out.inst1 === VPUOp.cmax    )
  val isCMin1    = (fsm.io.out.inst1 === VPUOp.cmin    )
  val isPairMax1 = (fsm.io.out.inst1 === VPUOp.pairmax )
  val isPairMin1 = (fsm.io.out.inst1 === VPUOp.pairmin )
  val isRSum1    = (fsm.io.out.inst1 === VPUOp.rsum    )
  val isCSum1    = (fsm.io.out.inst1 === VPUOp.csum    )
  val isFp8pack1   = (fsm.io.out.inst1 === VPUOp.fp8pack  )
  val isFp8unpack1 = (fsm.io.out.inst1 === VPUOp.fp8unpack)
  val isSquare1  = (fsm.io.out.inst1 === VPUOp.square  )
  val isCube1    = (fsm.io.out.inst1 === VPUOp.cube    )
  val isRelu1    = (fsm.io.out.inst1 === VPUOp.relu    )
  val isVliAll1  = (fsm.io.out.inst1 === VPUOp.vliAll  )
  val isVliRow1  = (fsm.io.out.inst1 === VPUOp.vliRow  )
  val isVliCol1  = (fsm.io.out.inst1 === VPUOp.vliCol  )
  val isVliOne1  = (fsm.io.out.inst1 === VPUOp.vliOne  )
  val isVli1 = isVliAll1 || isVliRow1 || isVliCol1 || isVliOne1
  val isMov1 = (fsm.io.out.inst1 === VPUOp.mov)

  val isAdd2     = (fsm.io.out.inst2 === VPUOp.add     )
  val isSub2     = (fsm.io.out.inst2 === VPUOp.sub     )
  val isMul2     = (fsm.io.out.inst2 === VPUOp.mul     )
  val isRcp2     = (fsm.io.out.inst2 === VPUOp.rcp     )
  val isSqrt2    = (fsm.io.out.inst2 === VPUOp.sqrt    )
  val isExp2     = (fsm.io.out.inst2 === VPUOp.exp     )
  val isExp22    = (fsm.io.out.inst2 === VPUOp.exp2    )
  val isSin2     = (fsm.io.out.inst2 === VPUOp.sin     )
  val isCos2     = (fsm.io.out.inst2 === VPUOp.cos     )
  val isTanh2    = (fsm.io.out.inst2 === VPUOp.tanh    )
  val isLog2     = (fsm.io.out.inst2 === VPUOp.log     )
  val isCMax2    = (fsm.io.out.inst2 === VPUOp.cmax    )
  val isCMin2    = (fsm.io.out.inst2 === VPUOp.cmin    )
  val isPairMax2 = (fsm.io.out.inst2 === VPUOp.pairmax )
  val isPairMin2 = (fsm.io.out.inst2 === VPUOp.pairmin )
  val isCSum2    = (fsm.io.out.inst2 === VPUOp.csum    )
  val isFp8pack2   = (fsm.io.out.inst2 === VPUOp.fp8pack  )
  val isFp8unpack2 = (fsm.io.out.inst2 === VPUOp.fp8unpack)
  val isSquare2  = (fsm.io.out.inst2 === VPUOp.square  )
  val isCube2    = (fsm.io.out.inst2 === VPUOp.cube    )
  val isRelu2    = (fsm.io.out.inst2 === VPUOp.relu    )
  val isVliAll2  = (fsm.io.out.inst2 === VPUOp.vliAll  )
  val isVliRow2  = (fsm.io.out.inst2 === VPUOp.vliRow  )
  val isVliCol2  = (fsm.io.out.inst2 === VPUOp.vliCol  )
  val isVliOne2  = (fsm.io.out.inst2 === VPUOp.vliOne  )
  val isMov2 = (fsm.io.out.inst2 === VPUOp.mov)

  val isAddValid     = (isAdd1     && io.data1.valid) || (isAdd2     && io.data2.valid)
  val isSubValid     = (isSub1     && io.data1.valid) || (isSub2     && io.data2.valid)
  val isMulValid     = (isMul1     && io.data1.valid) || (isMul2     && io.data2.valid)
  val isRcpValid     = (isRcp1     && io.data1.valid) || (isRcp2     && io.data2.valid)
  val isSqrtValid    = (isSqrt1    && io.data1.valid) || (isSqrt2    && io.data2.valid)
  val isExpValid     = (isExp1     && io.data1.valid) || (isExp2     && io.data2.valid)
  val isExp2Valid    = (isExp21    && io.data1.valid) || (isExp22    && io.data2.valid)
  val isSinValid     = (isSin1     && io.data1.valid) || (isSin2     && io.data2.valid)
  val isCosValid     = (isCos1     && io.data1.valid) || (isCos2     && io.data2.valid)
  val isTanhValid    = (isTanh1    && io.data1.valid) || (isTanh2    && io.data2.valid)
  val isLogValid     = (isLog1     && io.data1.valid) || (isLog2     && io.data2.valid)
  val isCMaxValid    = (isCMax1    && io.data1.valid) || (isCMax2    && io.data2.valid)
  val isCMinValid    = (isCMin1    && io.data1.valid) || (isCMin2    && io.data2.valid)
  val isPairMaxValid = (isPairMax1 && io.data1.valid) || (isPairMax2 && io.data2.valid)
  val isPairMinValid = (isPairMin1 && io.data1.valid) || (isPairMin2 && io.data2.valid)
  val isCSumValid    = (isCSum1    && io.data1.valid) || (isCSum2    && io.data2.valid)
  val isFp8packValid   = (isFp8pack1   && io.data1.valid) || (isFp8pack2   && io.data2.valid)
  val isFp8unpackValid = (isFp8unpack1 && io.data1.valid) || (isFp8unpack2 && io.data2.valid)
  val isSquareValid  = (isSquare1  && io.data1.valid) || (isSquare2  && io.data2.valid)
  val isCubeValid    = (isCube1    && io.data1.valid) || (isCube2    && io.data2.valid)
  val isReluValid    = (isRelu1    && io.data1.valid) || (isRelu2    && io.data2.valid)
  val isVli2 = isVliAll2 || isVliRow2 || isVliCol2 || isVliOne2
  val isVliValid = isVli1 || isVli2
  val isMovValid = isMov1 || isMov2
  val isRMaxPairValid = isRMax1 && io.data1.valid && io.data2.valid
  val isRMinPairValid = isRMin1 && io.data1.valid && io.data2.valid
  val isRSumPairValid = isRSum1 && io.data1.valid && io.data2.valid

	// Converting Register Data to Vector
	val vector1 = io.data1.bits.data.asTypeOf(Vec(p.numLanes, UInt(p.wordWidth.W)))
	val vector2 = io.data2.bits.data.asTypeOf(Vec(p.numLanes, UInt(p.wordWidth.W)))
  val pairRowReduceInput = VecInit(vector1 ++ vector2)

  // -------------------------------------------------------------------------
  // Functional Units
  // -------------------------------------------------------------------------
  val zeroVec = VecInit.fill(p.numLanes)(0.U(p.wordWidth.W))

  // ── Add/Sub (rsum handled separately as a 32-element BF16 pair reduction) ──
  val addSubSum = Module(new AddSubSumVec(p.BF16))
  addSubSum.io.req.valid                     := (isAddValid || isSubValid)
  addSubSum.io.req.bits.isDoneReadingColSum   := fsm.io.out.isDoneReadingColSum
  addSubSum.io.req.bits.isSub                := isSubValid
  addSubSum.io.req.bits.isSum                := false.B
  addSubSum.io.req.bits.aVec                 := Mux(isAddValid || isSubValid, vector1, vector2)
  addSubSum.io.req.bits.bVec                 := MuxCase(addSubSum.io.resp.bits.result,
                                                  Seq(
                                                    (isAddValid || isSubValid) -> vector2
                                                  ))

  val mul = Module(new MulRec(p.BF16))
  mul.io.req.valid             := isMulValid
  mul.io.req.bits.roundingMode := 0.U
  mul.io.req.bits.tag          := 0.U
  mul.io.req.bits.whichBank    := 0.U
  mul.io.req.bits.wRow         := 0.U
  mul.io.req.bits.laneMask     := 0xFFFF.U(16.W)
  mul.io.req.bits.aVec         := vector1
  mul.io.req.bits.bVec         := vector2

  val rcp = Module(new Rcp(p.BF16))
  rcp.io.req.valid             := isRcpValid
  rcp.io.req.bits.roundingMode := 0.U
  rcp.io.req.bits.laneMask     := 0xFFFF.U(16.W)
  rcp.io.req.bits.aVec         := Mux(isRcp1, vector1, vector2)

  val sqrt = Module(new Sqrt(p.BF16))
  sqrt.io.req.valid             := isSqrtValid
  sqrt.io.req.bits.roundingMode := 0.U
  sqrt.io.req.bits.laneMask     := 0xFFFF.U(16.W)
  sqrt.io.req.bits.aVec         := Mux(isSqrt1, vector1, vector2)

  val exp = Module(new Exp(p.BF16))
  exp.io.req.valid         := (isExpValid || isExp2Valid)
  exp.io.req.bits.isBase2  := isExp2Valid
  exp.io.req.bits.laneMask := 0xFFFF.U(16.W)
  exp.io.req.bits.xVec     := Mux(isExp1 || isExp21, vector1, vector2)

  val sinCos = Module(new SinCosVec(p.BF16))
  sinCos.io.req.valid         := (isSinValid || isCosValid)
  sinCos.io.req.bits.cos      := isCosValid
  sinCos.io.req.bits.laneMask := 0xFFFF.U(16.W)
  sinCos.io.req.bits.xVec     := Mux(isSin1 || isCos1, vector1, vector2)

  val log2 = Module(new Log(p.BF16))
  log2.io.req.valid         := isLogValid
  log2.io.req.bits.laneMask := 0xFFFF.U(16.W)
  log2.io.req.bits.aVec     := Mux(isLog1, vector1, vector2)

  val tanh = Module(new TanhRec(p.BF16))
  tanh.io.req.valid         := isTanhValid
  tanh.io.req.bits.tag      := 0.U
  tanh.io.req.bits.whichBank := 0.U
  tanh.io.req.bits.wRow     := 0.U
  tanh.io.req.bits.laneMask := 0xFFFF.U(16.W)
  tanh.io.req.bits.xVec     := Mux(isTanh1, vector1, vector2)

  val sqcb = Module(new SquareCubeVec(p.BF16))
  sqcb.io.req.valid             := (isSquareValid || isCubeValid)
  sqcb.io.req.bits.roundingMode := 0.U
  sqcb.io.req.bits.laneMask     := 0xFFFF.U(16.W)
  sqcb.io.req.bits.aVec         := Mux(isSquare1 || isCube1, vector1, vector2)
  sqcb.io.req.bits.isCube       := isCubeValid

  // ── FP8 pack/unpack ──
  val fp8packScaleE8M0Sel   = Mux(isFp8pack1,   fsm.io.out.packScale1,   fsm.io.out.packScale2)
  val fp8unpackScaleE8M0Sel = Mux(isFp8unpack1, fsm.io.out.unpackScale1, fsm.io.out.unpackScale2)
  val fp8packScaleExpSel    = e8m0ToScaleExpClamped(fp8packScaleE8M0Sel)
  val fp8unpackScaleExpSel  = e8m0ToScaleExpClamped(fp8unpackScaleE8M0Sel)

  val fp8pack = Module(new FP8Pack(p.BF16))
  fp8pack.io.req.valid         := isFp8packValid
  fp8pack.io.req.bits.xVec     := Mux(isFp8pack1, vector1, vector2)
  fp8pack.io.req.bits.expShift := fp8packScaleExpSel

  val fp8unpack = Module(new FP8Unpack(p.BF16))
  fp8unpack.io.req.valid         := isFp8unpackValid
  fp8unpack.io.req.bits.xVec     := Mux(isFp8unpack1, vector1, vector2)
  fp8unpack.io.req.bits.expShift := fp8unpackScaleExpSel

  val relu = Module(new Relu(p.BF16))
  relu.io.req.valid         := isReluValid
  relu.io.req.bits.tag      := 0.U
  relu.io.req.bits.whichBank := 0.U
  relu.io.req.bits.wRow     := 0.U
  relu.io.req.bits.laneMask := 0xFFFF.U(16.W)
  relu.io.req.bits.aVec     := Mux(isRelu1, vector1, vector2)

  val rowMaxLo = Module(new RowMax(p.BF16))
  rowMaxLo.io.req.valid     := isRMaxPairValid
  rowMaxLo.io.req.bits.aVec := vector1

  val rowMaxHi = Module(new RowMax(p.BF16))
  rowMaxHi.io.req.valid     := isRMaxPairValid
  rowMaxHi.io.req.bits.aVec := vector2

  val rowMaxPairScalar = compareReturnMax(rowMaxLo.io.resp.bits.result(0), rowMaxHi.io.resp.bits.result(0))
  val rowMaxPairValid = rowMaxLo.io.resp.valid && rowMaxHi.io.resp.valid
  val rowMaxPairResult = VecInit.fill(p.numLanes)(rowMaxPairScalar)

  val rowMinLo = Module(new RowMin(p.BF16))
  rowMinLo.io.req.valid     := isRMinPairValid
  rowMinLo.io.req.bits.aVec := vector1

  val rowMinHi = Module(new RowMin(p.BF16))
  rowMinHi.io.req.valid     := isRMinPairValid
  rowMinHi.io.req.bits.aVec := vector2

  val rowMinPairScalar = compareReturnMin(rowMinLo.io.resp.bits.result(0), rowMinHi.io.resp.bits.result(0))
  val rowMinPairValid = rowMinLo.io.resp.valid && rowMinHi.io.resp.valid
  val rowMinPairResult = VecInit.fill(p.numLanes)(rowMinPairScalar)

  val rowSumPair = Module(new ReduSumRec(p.BF16, numLanes = 2 * p.numLanes, tagWidth = 1))
  rowSumPair.io.req.valid := isRSumPairValid
  rowSumPair.io.req.bits.tag := 0.U
  rowSumPair.io.req.bits.roundingMode := 0.U
  rowSumPair.io.req.bits.laneMask := Fill(2 * p.numLanes, 1.U(1.W))
  rowSumPair.io.req.bits.whichBank := 0.U
  rowSumPair.io.req.bits.wRow := 0.U
  rowSumPair.io.req.bits.aVec := pairRowReduceInput
  val rowSumPairResult = VecInit(rowSumPair.io.resp.bits.result.take(p.numLanes))

  val pairMax = Module(new PairWiseMax(p.BF16))
  pairMax.io.req.valid                       := isPairMaxValid || isCMaxValid
  pairMax.io.req.bits.isDoneReadingColMax     := fsm.io.out.isDoneReadingColMax
  pairMax.io.req.bits.laneMask               := 0xFFFF.U(16.W)
  pairMax.io.req.bits.aVec                   := Mux(isCMax1 || isPairMaxValid, vector1, vector2)
  pairMax.io.req.bits.bVec                   := MuxCase(pairMax.io.resp.bits.result,
                                                  Seq(
                                                    (isCMax1 && (fsm.io.out.readRow1 === 1.U) && !fsm.io.out.readBank1(0)) -> vector1,
                                                    (isCMax2 && (fsm.io.out.readRow2 === 1.U) && !fsm.io.out.readBank2(0)) -> vector2,
                                                    (isPairMaxValid) -> vector2
                                                  ))

  val pairMin = Module(new PairWiseMin(p.BF16))
  pairMin.io.req.valid                       := isPairMinValid || isCMinValid
  pairMin.io.req.bits.isDoneReadingColMin     := fsm.io.out.isDoneReadingColMin
  pairMin.io.req.bits.laneMask               := 0xFFFF.U(16.W)
  pairMin.io.req.bits.aVec                   := Mux(isCMin1 || isPairMinValid, vector1, vector2)
  pairMin.io.req.bits.bVec                   := MuxCase(pairMin.io.resp.bits.result,
                                                  Seq(
                                                    (isCMin1 && (fsm.io.out.readRow1 === 1.U) && !fsm.io.out.readBank1(0)) -> vector1,
                                                    (isCMin2 && (fsm.io.out.readRow2 === 1.U) && !fsm.io.out.readBank2(0)) -> vector2,
                                                    (isPairMinValid) -> vector2
                                                  ))

  val vli = Module(new VectorLoadImm(p.BF16, p.numLanes, 5))
  vli.io.req.valid       := isVliValid
  vli.io.req.bits.op     := MuxCase(VPUOp.vliAll, Seq(
    isVliAll1 -> VPUOp.vliAll, isVliRow1 -> VPUOp.vliRow,
    isVliCol1 -> VPUOp.vliCol, isVliOne1 -> VPUOp.vliOne,
    isVliAll2 -> VPUOp.vliAll, isVliRow2 -> VPUOp.vliRow,
    isVliCol2 -> VPUOp.vliCol, isVliOne2 -> VPUOp.vliOne
  ))
  vli.io.req.bits.imm    := Mux(isVli1, fsm.io.out.imm1, fsm.io.out.imm2)
  vli.io.req.bits.rowIdx := Mux(isVli1, fsm.io.out.writeRow1, fsm.io.out.writeRow2)

  val mov = Module(new Mov(p.BF16.wordWidth, p.numLanes))
  // `VMOV` still needs the source row to be present; driving the lane box
  // for the whole lifetime of the instruction causes zero / stale rewrites.
  mov.io.req.valid         := (isMov1 && io.data1.valid) || (isMov2 && io.data2.valid)
  mov.io.req.bits.laneMask := 0xFFFF.U(16.W)
  mov.io.req.bits.aVec     := Mux(isMov1, vector1, vector2)

  // ── Column-sum (dedicated recoded-FP accumulator) ──
  val csum = Module(new ColAddVec(p.BF16))
  val zeroRecoded = recFNFromFN(p.BF16.expWidth, p.BF16.sigWidth + 16, 0.U(32.W))
  val zeroVecWide = VecInit.fill(p.numLanes)(zeroRecoded)
  csum.io.req.valid                        := isCSumValid
  csum.io.req.bits.isDoneReadingColSum     := fsm.io.out.isDoneReadingColSum
  csum.io.req.bits.aVec                    := Mux(isCSum1, vector1, vector2)
  csum.io.req.bits.bVec                    := Mux(fsm.io.out.isFirstEntryColSum, zeroVecWide, csum.io.resp.bits.result)
  val finalResultBF16 = VecInit(csum.io.resp.bits.result.map(res =>
    fNFromRecFN(p.BF16.expWidth, p.BF16.sigWidth + 16, res)(31, 16)
  ))

  // -------------------------------------------------------------------------
  // Internal Interface Signals Assignments
  // -------------------------------------------------------------------------
  dataInFire1 := MuxCase(false.B, Seq(
    ((isAdd1 || isSub1)                      && io.data1.valid) -> addSubSum.io.req.valid,
    (isRSum1                                 && io.data1.valid) -> rowSumPair.io.req.valid,
    (isMul1                                  && io.data1.valid) -> mul.io.req.valid,
    (isRcp1                                  && io.data1.valid) -> rcp.io.req.valid,
    (isSqrt1                                 && io.data1.valid) -> sqrt.io.req.valid,
    ((isExp1 || isExp21)                     && io.data1.valid) -> exp.io.req.valid,
    ((isSin1 || isCos1)                      && io.data1.valid) -> sinCos.io.req.valid,
    (isLog1                                  && io.data1.valid) -> log2.io.req.valid,
    (isTanh1                                 && io.data1.valid) -> tanh.io.req.valid,
    (isRMax1                                 && io.data1.valid) -> rowMaxLo.io.req.valid,
    (isRMin1                                 && io.data1.valid) -> rowMinLo.io.req.valid,
    ((isPairMax1 || isCMax1)                 && io.data1.valid) -> pairMax.io.req.valid,
    ((isPairMin1 || isCMin1)                 && io.data1.valid) -> pairMin.io.req.valid,
    ((isSquare1 || isCube1)                  && io.data1.valid) -> sqcb.io.req.valid,
    (isFp8pack1                              && io.data1.valid) -> fp8pack.io.req.valid,
    (isFp8unpack1                            && io.data1.valid) -> fp8unpack.io.req.valid,
    (isRelu1                                 && io.data1.valid) -> relu.io.req.valid,
    (isVli1                                  && io.data1.valid) -> vli.io.req.valid,
    (isMov1                                  && io.data1.valid) -> mov.io.req.valid,
    (isCSum1                                 && io.data1.valid) -> csum.io.req.valid
  ))

  dataInFire2 := MuxCase(false.B, Seq(
    ((isAdd2 || isSub2)                      && io.data2.valid) -> addSubSum.io.req.valid,
    (isRSum1                                 && io.data2.valid) -> rowSumPair.io.req.valid,
    (isMul2                                  && io.data2.valid) -> mul.io.req.valid,
    (isRcp2                                  && io.data2.valid) -> rcp.io.req.valid,
    (isSqrt2                                 && io.data2.valid) -> sqrt.io.req.valid,
    ((isExp2 || isExp22)                     && io.data2.valid) -> exp.io.req.valid,
    ((isSin2 || isCos2)                      && io.data2.valid) -> sinCos.io.req.valid,
    (isLog2                                  && io.data2.valid) -> log2.io.req.valid,
    (isTanh2                                 && io.data2.valid) -> tanh.io.req.valid,
    (isRMax1                                 && io.data2.valid) -> rowMaxHi.io.req.valid,
    (isRMin1                                 && io.data2.valid) -> rowMinHi.io.req.valid,
    ((isPairMax2 || isCMax2)                 && io.data2.valid) -> pairMax.io.req.valid,
    ((isPairMin2 || isCMin2)                 && io.data2.valid) -> pairMin.io.req.valid,
    ((isSquare2 || isCube2)                  && io.data2.valid) -> sqcb.io.req.valid,
    (isFp8pack2                              && io.data2.valid) -> fp8pack.io.req.valid,
    (isFp8unpack2                            && io.data2.valid) -> fp8unpack.io.req.valid,
    (isRelu2                                 && io.data2.valid) -> relu.io.req.valid,
    (isVli2                                  && io.data2.valid) -> vli.io.req.valid,
    (isMov2                                  && io.data2.valid) -> mov.io.req.valid,
    (isCSum2                                 && io.data2.valid) -> csum.io.req.valid
  ))

  dataOutFire1 := MuxCase(false.B, Seq(
    (isAdd1 || isSub1)            -> addSubSum.io.resp.valid,
    isRSum1                       -> rowSumPair.io.resp.valid,
    isMul1                        -> mul.io.resp.valid,
    isRcp1                        -> rcp.io.resp.valid,
    isSqrt1                       -> sqrt.io.resp.valid,
    (isExp1 || isExp21)           -> exp.io.resp.valid,
    (isSin1 || isCos1)            -> sinCos.io.resp.valid,
    isLog1                        -> log2.io.resp.valid,
    isTanh1                       -> tanh.io.resp.valid,
    isRMax1                       -> rowMaxPairValid,
    isRMin1                       -> rowMinPairValid,
    (isPairMax1 || isCMax1)       -> pairMax.io.resp.valid,
    (isPairMin1 || isCMin1)       -> pairMin.io.resp.valid,
    (isSquare1 || isCube1)        -> sqcb.io.resp.valid,
    isFp8pack1                    -> fp8pack.io.resp.valid,
    isFp8unpack1                  -> fp8unpack.io.resp.valid,
    isRelu1                       -> relu.io.resp.valid,
    isVli1                        -> vli.io.resp.valid,
    isMov1                        -> mov.io.resp.valid,
    isCSum1                       -> csum.io.resp.valid
  ))

  dataOutFire2 := MuxCase(false.B, Seq(
    // Two-input ops (add/sub/mul/pairmax/pairmin) never fire data2 out
    isRSum1                -> rowSumPair.io.resp.valid,
    isRcp2                 -> rcp.io.resp.valid,
    isSqrt2                -> sqrt.io.resp.valid,
    (isExp2 || isExp22)    -> exp.io.resp.valid,
    (isSin2 || isCos2)     -> sinCos.io.resp.valid,
    isLog2                 -> log2.io.resp.valid,
    isTanh2                -> tanh.io.resp.valid,
    isRMax1                -> rowMaxPairValid,
    isRMin1                -> rowMinPairValid,
    isCMax2                -> pairMax.io.resp.valid,
    isCMin2                -> pairMin.io.resp.valid,
    (isSquare2 || isCube2) -> sqcb.io.resp.valid,
    isFp8pack2             -> fp8pack.io.resp.valid,
    isFp8unpack2           -> fp8unpack.io.resp.valid,
    isRelu2                -> relu.io.resp.valid,
    isVli2                 -> vli.io.resp.valid,
    isMov2                 -> mov.io.resp.valid,
    isCSum2                -> csum.io.resp.valid
  ))

  writeData1 := MuxCase(0.U, Seq(
    (isAdd1 || isSub1)            -> addSubSum.io.resp.bits.result.asUInt,
    isRSum1                       -> rowSumPairResult.asUInt,
    isMul1                        -> mul.io.resp.bits.result.asUInt,
    isRcp1                        -> rcp.io.resp.bits.result.asUInt,
    isSqrt1                       -> sqrt.io.resp.bits.result.asUInt,
    (isExp1 || isExp21)           -> exp.io.resp.bits.result.asUInt,
    (isSin1 || isCos1)            -> sinCos.io.resp.bits.result.asUInt,
    isLog1                        -> log2.io.resp.bits.result.asUInt,
    isTanh1                       -> tanh.io.resp.bits.result.asUInt,
    isRMax1                       -> rowMaxPairResult.asUInt,
    isRMin1                       -> rowMinPairResult.asUInt,
    (isPairMax1 || isCMax1)       -> pairMax.io.resp.bits.result.asUInt,
    (isPairMin1 || isCMin1)       -> pairMin.io.resp.bits.result.asUInt,
    (isSquare1 || isCube1)        -> sqcb.io.resp.bits.result.asUInt,
    isFp8pack1                    -> fp8pack.io.resp.bits.result.asUInt,
    isFp8unpack1                  -> fp8unpack.io.resp.bits.result.asUInt,
    isRelu1                       -> relu.io.resp.bits.result.asUInt,
    isVli1                        -> vli.io.resp.bits.result.asUInt,
    isMov1                        -> mov.io.resp.bits.result.asUInt,
    isCSum1                       -> finalResultBF16.asUInt
  ))

  writeData2 := MuxCase(0.U, Seq(
    isRSum1                       -> rowSumPairResult.asUInt,
    isMul2                        -> mul.io.resp.bits.result.asUInt,
    isRcp2                        -> rcp.io.resp.bits.result.asUInt,
    isSqrt2                       -> sqrt.io.resp.bits.result.asUInt,
    (isExp2 || isExp22)           -> exp.io.resp.bits.result.asUInt,
    (isSin2 || isCos2)            -> sinCos.io.resp.bits.result.asUInt,
    isLog2                        -> log2.io.resp.bits.result.asUInt,
    isTanh2                       -> tanh.io.resp.bits.result.asUInt,
    isRMax1                       -> rowMaxPairResult.asUInt,
    isRMin1                       -> rowMinPairResult.asUInt,
    (isPairMax2 || isCMax2)       -> pairMax.io.resp.bits.result.asUInt,
    (isPairMin2 || isCMin2)       -> pairMin.io.resp.bits.result.asUInt,
    (isSquare2 || isCube2)        -> sqcb.io.resp.bits.result.asUInt,
    isFp8pack2                    -> fp8pack.io.resp.bits.result.asUInt,
    isFp8unpack2                  -> fp8unpack.io.resp.bits.result.asUInt,
    isRelu2                       -> relu.io.resp.bits.result.asUInt,
    isVli2                        -> vli.io.resp.bits.result.asUInt,
    isMov2                        -> mov.io.resp.bits.result.asUInt,
    isCSum2                       -> finalResultBF16.asUInt
  ))

  // Software-scheduled contract: commands must only be issued when the
  // sequencer is ready.
  assert(!io.inst.valid || fsm.io.out.VEReady,
    "VPU: command issued while sequencer is busy (software-scheduling contract violated)")
  assert(!(fsm.io.out.state === 2.U && opUsesDoubleRead(fsm.io.out.inst1) &&
           (io.data1.valid =/= io.data2.valid)),
    "VPU: dual-read instructions must receive both source rows in the same cycle")

  // Outputs
  io.read1.valid      := fsm.io.out.readValid1
  io.read1.bits.bank  := fsm.io.out.readBank1
  io.read1.bits.row   := fsm.io.out.readRow1

  io.read2.valid      := fsm.io.out.readValid2
  io.read2.bits.bank  := fsm.io.out.readBank2
  io.read2.bits.row   := fsm.io.out.readRow2

  io.write1.valid     := fsm.io.out.writeValid1
  io.write1.bits.bank := fsm.io.out.writeBank1
  io.write1.bits.row  := fsm.io.out.writeRow1
  io.write1.bits.data := writeData1

  io.write2.valid     := fsm.io.out.writeValid2
  io.write2.bits.bank := fsm.io.out.writeBank2
  io.write2.bits.row  := fsm.io.out.writeRow2
  io.write2.bits.data := writeData2

  io.activeReads  := fsm.io.out.activeReads
  io.activeWrites := fsm.io.out.activeWrites
  io.issueBusy := fsm.io.out.issueBusy
  io.busy := fsm.io.out.busy
}
