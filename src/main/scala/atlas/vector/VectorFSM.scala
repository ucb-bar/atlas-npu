package atlas.vector

import chisel3._
import chisel3.util._
import atlas.common.VpuParams
import sp26FPUnits._
import sp26FPUnits.hardfloat._

class FSMIn extends Bundle {
    val instFire      = Bool()
    val instType      = VPUOp()
    val instReadBank1 = UInt(6.W)
    val instReadBank2 = UInt(6.W)
    val instWriteBank = UInt(6.W)
    val instImm       = SInt(16.W)
    val instPackScaleE8M0   = UInt(8.W)
    val instUnpackScaleE8M0 = UInt(8.W)
    val dataInFire1  = Bool()
    val dataInFire2  = Bool()
    val dataOutFire1 = Bool()
    val dataOutFire2 = Bool()
    val divSqrtReady = Bool()
}

class FSMOut extends Bundle {
    val inst1 = VPUOp()
    val inst2 = VPUOp()
    val imm1  = SInt(16.W)
    val imm2  = SInt(16.W)
    val packScale1   = UInt(8.W)
    val packScale2   = UInt(8.W)
    val unpackScale1 = UInt(8.W)
    val unpackScale2 = UInt(8.W)
    val readValid1 = Bool()
    val readBank1  = UInt(6.W)
    val readRow1   = UInt(5.W)
    val readValid2 = Bool()
    val readBank2  = UInt(6.W)
    val readRow2   = UInt(5.W)
    val writeValid1 = Bool()
    val writeBank1  = UInt(6.W)
    val writeRow1   = UInt(5.W)
    val writeCount1 = UInt(6.W)
    val writeValid2 = Bool()
    val writeBank2  = UInt(6.W)
    val writeRow2   = UInt(5.W)
    val writeCount2 = UInt(6.W)
    val VEReady = Bool()
    val isDoneReadingColMax = Bool()
    val isDoneReadingColMin = Bool()
    val isDoneReadingColSum = Bool()
    val isFirstEntryColSum = Bool()
    val state      = UInt(2.W)
    val done1      = Bool()
    val done2      = Bool()
    val readDone1  = Bool()
    val readDone2  = Bool()
    val writeDone1 = Bool()
    val writeDone2 = Bool()
    val activeReads  = Vec(4, Valid(UInt(6.W)))
    val activeWrites = Vec(4, Valid(UInt(6.W)))
    val issueBusy = UInt((VPUOp.all.size + 1).W)
    val busy = Bool()
}

class VectorFSM(val p: VpuParams) extends Module {
    val io = IO(new Bundle {
        val in = Input(new FSMIn())
        val out = Output(new FSMOut())
    })

    val inst1         = RegInit(VPUOp.add)
    val readDone1     = RegInit(true.B)
    val readBank1     = RegInit(0.U(6.W))
    val readCounter1  = RegInit(0.U(7.W))
    val writeDone1    = RegInit(true.B)
    val writeBank1    = RegInit(0.U(6.W))
    val writeCounter1 = RegInit(0.U(7.W))
    val nextReadBank1 = readBank1 + 1.U
    val nextWriteBank1 = writeBank1 + 1.U
    val isNextReadBank1 = readCounter1(5)
    val isNextWriteBank1 = writeCounter1(5)
    val readCounter1FiveBits = readCounter1(4,0)
    val writeCounter1FiveBits = writeCounter1(4,0)

    val inst2         = RegInit(VPUOp.add)
    val readDone2     = RegInit(true.B)
    val readBank2     = RegInit(0.U(6.W))
    val readCounter2  = RegInit(0.U(7.W))
    val writeDone2    = RegInit(true.B)
    val writeBank2    = RegInit(0.U(6.W))
    val writeCounter2 = RegInit(0.U(7.W))
    val nextReadBank2 = readBank2 + 1.U
    val nextWriteBank2 = writeBank2 + 1.U
    val isNextReadBank2 = readCounter2(5)
    val isNextWriteBank2 = writeCounter2(5)
    val readCounter2FiveBits = readCounter2(4,0)
    val writeCounter2FiveBits = writeCounter2(4,0)

    val packScale1Reg   = RegInit(127.U(8.W))
    val packScale2Reg   = RegInit(127.U(8.W))
    val unpackScale1Reg = RegInit(127.U(8.W))
    val unpackScale2Reg = RegInit(127.U(8.W))
    val imm1Reg = RegInit(0.S(16.W))
    val imm2Reg = RegInit(0.S(16.W))

    val slot1NewInst = WireDefault(false.B)
    val slot2NewInst = WireDefault(false.B)

    private def opIsVli(op: VPUOp.Type): Bool =
        (op === VPUOp.vliAll) || (op === VPUOp.vliRow) ||
        (op === VPUOp.vliCol) || (op === VPUOp.vliOne)

    private def opIsVliColOrOne(op: VPUOp.Type): Bool =
        (op === VPUOp.vliCol) || (op === VPUOp.vliOne)

    private def opIsTwoInput(op: VPUOp.Type): Bool =
        (op === VPUOp.add) || (op === VPUOp.sub) || (op === VPUOp.mul) ||
        (op === VPUOp.pairmax) || (op === VPUOp.pairmin)

    private def opIsRowReduce(op: VPUOp.Type): Bool =
        (op === VPUOp.rsum) || (op === VPUOp.rmax) || (op === VPUOp.rmin)

    private def opUsesDoublePorts(op: VPUOp.Type): Bool =
        opIsTwoInput(op) || opIsRowReduce(op)

    private def opUsesDualWritePorts(op: VPUOp.Type): Bool =
        opIsRowReduce(op)

    private def opReadsPrimaryPair(op: VPUOp.Type): Bool =
        !opIsVli(op) && (op =/= VPUOp.fp8unpack)

    private def opReadsSecondaryPair(op: VPUOp.Type): Bool =
        opIsTwoInput(op)

    private def opWritesPair(op: VPUOp.Type): Bool =
        (op =/= VPUOp.fp8pack) && !opIsVliColOrOne(op)

    private def opUsesAddSubSum(op: VPUOp.Type): Bool =
        (op === VPUOp.add) || (op === VPUOp.sub) || (op === VPUOp.rsum)

    private def opUsesExp(op: VPUOp.Type): Bool =
        (op === VPUOp.exp) || (op === VPUOp.exp2)

    private def opUsesSinCos(op: VPUOp.Type): Bool =
        (op === VPUOp.sin) || (op === VPUOp.cos)

    private def opUsesSquareCube(op: VPUOp.Type): Bool =
        (op === VPUOp.square) || (op === VPUOp.cube)

    private def opUsesPairMax(op: VPUOp.Type): Bool =
        (op === VPUOp.pairmax) || (op === VPUOp.cmax)

    private def opUsesPairMin(op: VPUOp.Type): Bool =
        (op === VPUOp.pairmin) || (op === VPUOp.cmin)

    private def opUsesVli(op: VPUOp.Type): Bool =
        (op === VPUOp.vliAll) || (op === VPUOp.vliRow) ||
        (op === VPUOp.vliCol) || (op === VPUOp.vliOne)

    private def opsShareLogic(a: VPUOp.Type, b: VPUOp.Type): Bool =
        (a === b) ||
        (opUsesAddSubSum(a) && opUsesAddSubSum(b)) ||
        (opUsesExp(a) && opUsesExp(b)) ||
        (opUsesSinCos(a) && opUsesSinCos(b)) ||
        (opUsesSquareCube(a) && opUsesSquareCube(b)) ||
        (opUsesPairMax(a) && opUsesPairMax(b)) ||
        (opUsesPairMin(a) && opUsesPairMin(b)) ||
        (opUsesVli(a) && opUsesVli(b))

    when (slot1NewInst) {
        packScale1Reg   := io.in.instPackScaleE8M0
        unpackScale1Reg := io.in.instUnpackScaleE8M0
        imm1Reg         := io.in.instImm
    }
    when (slot2NewInst) {
        packScale2Reg   := io.in.instPackScaleE8M0
        unpackScale2Reg := io.in.instUnpackScaleE8M0
        imm2Reg         := io.in.instImm
    }

    val isColReduce1  = (inst1 === VPUOp.cmax) || (inst1 === VPUOp.cmin) || (inst1 === VPUOp.csum)
    val isColReduce2  = (inst2 === VPUOp.cmax) || (inst2 === VPUOp.cmin) || (inst2 === VPUOp.csum)
    val isFp8pack1    = (inst1 === VPUOp.fp8pack)
    val isFp8pack2    = (inst2 === VPUOp.fp8pack)
    val isFp8unpack1  = (inst1 === VPUOp.fp8unpack)
    val isFp8unpack2  = (inst2 === VPUOp.fp8unpack)
    val isVliAllOrRow1 = (inst1 === VPUOp.vliAll) || (inst1 === VPUOp.vliRow)
    val isVliAllOrRow2 = (inst2 === VPUOp.vliAll) || (inst2 === VPUOp.vliRow)
    val isVliColOrOne1 = (inst1 === VPUOp.vliCol) || (inst1 === VPUOp.vliOne)
    val isVliColOrOne2 = (inst2 === VPUOp.vliCol) || (inst2 === VPUOp.vliOne)
    val isVli1        = isVliAllOrRow1 || isVliColOrOne1
    val isVli2        = isVliAllOrRow2 || isVliColOrOne2

    // Asymmetric thresholds:
    //   - fp8pack writes 1 mreg (32 rows)
    //   - fp8unpack reads 1 mreg (32 rows)
    //   - vliCol / vliOne write exactly 1 mreg (32 rows)
    //   - vliAll / vliRow write a full BF16 tensor pair (64 rows)
    val readLim1  = Mux(isColReduce1, 127.U, Mux(isFp8unpack1, 31.U, 63.U))
    val doubleIsRowReduce = opIsRowReduce(inst1)
    val readLim2  = Mux(doubleIsRowReduce, 31.U, Mux(isColReduce2, 127.U, Mux(isFp8unpack2, 31.U, 63.U)))
    val doubleUsesDualWritePorts = opUsesDualWritePorts(inst1)
    val writeLim1 = Mux(doubleUsesDualWritePorts, 31.U, Mux(isColReduce1, 127.U, Mux(isFp8pack1 || isVliColOrOne1, 31.U, 63.U)))
    val writeLim2 = Mux(doubleUsesDualWritePorts, 31.U, Mux(isColReduce2, 127.U, Mux(isFp8pack2 || isVliColOrOne2, 31.U, 63.U)))

    val isWriteOnly = opIsVli(io.in.instType)
    val inputIsRowReduce = opIsRowReduce(io.in.instType)
    val inputUsesDoublePorts = opUsesDoublePorts(io.in.instType)
    val inputUsesDualWritePorts = opUsesDualWritePorts(io.in.instType)
    val inputReadBank2 = Mux(inputIsRowReduce, io.in.instReadBank1 + 1.U, io.in.instReadBank2)
    val inputWriteBank2 = io.in.instWriteBank + 1.U
    val done1 = writeDone1 || (io.in.dataOutFire1 && writeCounter1 === writeLim1)
    val done2 = writeDone2 || (io.in.dataOutFire2 && writeCounter2 === writeLim2)
    val doubleDone = Mux(doubleUsesDualWritePorts, done1 && done2, done1)
    val doubleReadLim = Mux(doubleIsRowReduce, 31.U, 63.U)

    val idle :: single :: double :: Nil = Enum(3)
    val state = RegInit(idle)
    val nextState = WireDefault(state)
    state := nextState

    private def canIssueOp(op: VPUOp.Type): Bool = {
        // The overlap path only has room for a second single-input op; anything
        // that needs both read slots must wait until the engine is otherwise free.
        val opNeedsDoublePorts = opUsesDoublePorts(op)
        MuxLookup(state, false.B)(Seq(
            idle   -> true.B,
            single -> Mux(done1 && done2, true.B,
                      Mux(done1, !opNeedsDoublePorts && !opsShareLogic(op, inst2),
                      Mux(done2, !opNeedsDoublePorts && !opsShareLogic(op, inst1),
                          false.B))),
            double -> doubleDone
        ))
    }

    val VEReady = canIssueOp(io.in.instType)

    switch(state) {
        is(idle) {
            when(io.in.instFire && inputUsesDoublePorts)      { nextState := double }
            .elsewhen(io.in.instFire && !inputUsesDoublePorts) { nextState := single }
            .otherwise                                { nextState := idle }
        }
        is(single) {
            when(io.in.instFire && (done1 && done2))      { nextState := Mux(inputUsesDoublePorts, double, single) }
            .elsewhen(!io.in.instFire && (done1 && done2)) { nextState := idle }
            .otherwise                                      { nextState := single }
        }
        is(double) {
            when(io.in.instFire && inputUsesDoublePorts)       { nextState := double }
            .elsewhen(io.in.instFire && !inputUsesDoublePorts)  { nextState := single }
            .elsewhen(!io.in.instFire && doubleDone)            { nextState := idle }
            .otherwise                                  { nextState := double }
        }
    }

    val readValid1 = WireDefault(false.B)
    val readValid2 = WireDefault(false.B)
    val isIDLE   = nextState === idle
    val isSINGLE = nextState === single
    val isDOUBLE = nextState === double
    val newInputToSINGLE = io.in.instFire && isSINGLE
    val newInputToDOUBLE = io.in.instFire && isDOUBLE
    val readEarly1 = newInputToDOUBLE || (newInputToSINGLE && done1)
    val readEarly2 = newInputToDOUBLE || (newInputToSINGLE && !done1)

    switch(state) {
        is(idle) {
            when(newInputToDOUBLE) {
                inst1 := io.in.instType; slot1NewInst := true.B
                readDone1 := false.B; readBank1 := io.in.instReadBank1; readCounter1 := 1.U
                writeDone1 := false.B; writeBank1 := io.in.instWriteBank; writeCounter1 := 0.U
                inst2 := VPUOp.add
                readDone2 := false.B; readBank2 := inputReadBank2; readCounter2 := 1.U
                writeDone2 := !inputUsesDualWritePorts
                writeBank2 := Mux(inputUsesDualWritePorts, inputWriteBank2, 0.U)
                writeCounter2 := 0.U
                readValid1 := true.B; readValid2 := true.B
            }.elsewhen(newInputToSINGLE) {
                inst1 := io.in.instType; slot1NewInst := true.B
                readDone1 := isWriteOnly; readBank1 := io.in.instReadBank1; readCounter1 := Mux(isWriteOnly, 0.U, 1.U)
                writeDone1 := false.B; writeBank1 := io.in.instWriteBank; writeCounter1 := 0.U
                inst2 := VPUOp.add; readDone2 := true.B; readBank2 := 0.U; readCounter2 := 0.U
                writeDone2 := true.B; writeBank2 := 0.U; writeCounter2 := 0.U
                readValid1 := !isWriteOnly; readValid2 := false.B
            }.otherwise {
                readValid1 := false.B; readValid2 := false.B
            }
        }
        is(single) {
            when(newInputToDOUBLE) {
                inst1 := io.in.instType; slot1NewInst := true.B
                readDone1 := false.B; readBank1 := io.in.instReadBank1; readCounter1 := 1.U
                writeDone1 := false.B; writeBank1 := io.in.instWriteBank; writeCounter1 := 0.U
                inst2 := VPUOp.add; readDone2 := false.B; readBank2 := inputReadBank2; readCounter2 := 1.U
                writeDone2 := !inputUsesDualWritePorts
                writeBank2 := Mux(inputUsesDualWritePorts, inputWriteBank2, 0.U)
                writeCounter2 := 0.U
                readValid1 := true.B; readValid2 := true.B
            }.elsewhen(newInputToSINGLE) {
                when(done1 && done2) {
                    inst1 := io.in.instType; slot1NewInst := true.B
                    readDone1 := isWriteOnly; readBank1 := io.in.instReadBank1; readCounter1 := Mux(isWriteOnly, 0.U, 1.U)
                    writeDone1 := false.B; writeBank1 := io.in.instWriteBank; writeCounter1 := 0.U
                    inst2 := VPUOp.add; readDone2 := true.B; readBank2 := 0.U; readCounter2 := 0.U
                    writeDone2 := true.B; writeBank2 := 0.U; writeCounter2 := 0.U
                    readValid1 := !isWriteOnly; readValid2 := false.B
                }.elsewhen(done1) {
                    inst1 := io.in.instType; slot1NewInst := true.B
                    readDone1 := isWriteOnly; readBank1 := io.in.instReadBank1; readCounter1 := Mux(isWriteOnly, 0.U, 1.U)
                    writeDone1 := false.B; writeBank1 := io.in.instWriteBank; writeCounter1 := 0.U
                    inst2 := inst2
                    readDone2 := Mux(readValid2 && io.in.dataInFire2 && readCounter2 === readLim2, true.B, readDone2)
                    readBank2 := readBank2; readCounter2 := Mux(readValid2 && io.in.dataInFire2, readCounter2 + 1.U, readCounter2)
                    writeDone2 := Mux(io.in.dataOutFire2 && writeCounter2 === writeLim2, true.B, writeDone2)
                    writeCounter2 := Mux(io.in.dataOutFire2, writeCounter2 + 1.U, writeCounter2)
                    readValid1 := !isWriteOnly; readValid2 := !readDone2
                }.elsewhen(done2) {
                    inst1 := inst1
                    readDone1 := Mux(io.in.dataInFire1 && readCounter1 === readLim1, true.B, readDone1)
                    readBank1 := readBank1; readCounter1 := Mux(readValid1 && io.in.dataInFire1, readCounter1 + 1.U, readCounter1)
                    writeDone1 := Mux(io.in.dataOutFire1 && writeCounter1 === writeLim1, true.B, writeDone1)
                    writeBank1 := writeBank1; writeCounter1 := Mux(io.in.dataOutFire1, writeCounter1 + 1.U, writeCounter1)
                    inst2 := io.in.instType; slot2NewInst := true.B
                    readDone2 := isWriteOnly; readBank2 := io.in.instReadBank1; readCounter2 := Mux(isWriteOnly, 0.U, 1.U)
                    writeDone2 := false.B; writeBank2 := io.in.instWriteBank; writeCounter2 := 0.U
                    readValid1 := !readDone1; readValid2 := !isWriteOnly
                }
            }.elsewhen(isIDLE) {
                inst1 := VPUOp.add; readDone1 := true.B; readBank1 := 0.U; readCounter1 := 0.U
                writeDone1 := true.B; writeBank1 := 0.U; writeCounter1 := 0.U
                inst2 := VPUOp.add; readDone2 := true.B; readBank2 := 0.U; readCounter2 := 0.U
                writeDone2 := true.B; writeBank2 := 0.U; writeCounter2 := 0.U
                readValid1 := false.B; readValid2 := false.B
            }.otherwise {
                inst1 := inst1
                readDone1 := Mux(io.in.dataInFire1 && readCounter1 === readLim1, true.B, readDone1)
                readBank1 := readBank1; readCounter1 := Mux(readValid1 && io.in.dataInFire1, readCounter1 + 1.U, readCounter1)
                writeDone1 := Mux(io.in.dataOutFire1 && writeCounter1 === writeLim1, true.B, writeDone1)
                writeBank1 := writeBank1; writeCounter1 := Mux(io.in.dataOutFire1, writeCounter1 + 1.U, writeCounter1)
                inst2 := inst2
                readDone2 := Mux(readValid2 && io.in.dataInFire2 && readCounter2 === readLim2, true.B, readDone2)
                readBank2 := readBank2; readCounter2 := Mux(readValid2 && io.in.dataInFire2, readCounter2 + 1.U, readCounter2)
                writeDone2 := Mux(io.in.dataOutFire2 && writeCounter2 === writeLim2, true.B, writeDone2)
                writeBank2 := writeBank2; writeCounter2 := Mux(io.in.dataOutFire2, writeCounter2 + 1.U, writeCounter2)
                readValid1 := !readDone1; readValid2 := !readDone2
            }
        }
        is(double) {
            // fp8pack/unpack never enter double state (single-input ops)
            when(newInputToDOUBLE) {
                inst1 := io.in.instType; slot1NewInst := true.B
                readDone1 := false.B; readBank1 := io.in.instReadBank1; readCounter1 := 1.U
                writeDone1 := false.B; writeBank1 := io.in.instWriteBank; writeCounter1 := 0.U
                inst2 := VPUOp.add; readDone2 := false.B; readBank2 := inputReadBank2; readCounter2 := 1.U
                writeDone2 := !inputUsesDualWritePorts
                writeBank2 := Mux(inputUsesDualWritePorts, inputWriteBank2, 0.U)
                writeCounter2 := 0.U
                readValid1 := true.B; readValid2 := true.B
            }.elsewhen(newInputToSINGLE) {
                inst1 := io.in.instType; slot1NewInst := true.B
                readDone1 := isWriteOnly; readBank1 := io.in.instReadBank1; readCounter1 := Mux(isWriteOnly, 0.U, 1.U)
                writeDone1 := false.B; writeBank1 := io.in.instWriteBank; writeCounter1 := 0.U
                inst2 := VPUOp.add; readDone2 := true.B; readBank2 := 0.U; readCounter2 := 0.U
                writeDone2 := true.B; writeBank2 := 0.U; writeCounter2 := 0.U
                readValid1 := !isWriteOnly; readValid2 := false.B
            }.elsewhen(isIDLE) {
                inst1 := VPUOp.add; readDone1 := true.B; readBank1 := 0.U; readCounter1 := 0.U
                writeDone1 := true.B; writeBank1 := 0.U; writeCounter1 := 0.U
                inst2 := VPUOp.add; readDone2 := true.B; readBank2 := 0.U; readCounter2 := 0.U
                writeDone2 := true.B; writeBank2 := 0.U; writeCounter2 := 0.U
                readValid1 := false.B; readValid2 := false.B
            }.otherwise {
                inst1 := inst1
                readDone1 := Mux(readValid1 && io.in.dataInFire1 && readCounter1 === doubleReadLim, true.B, readDone1)
                readBank1 := readBank1; readCounter1 := Mux(readValid1 && io.in.dataInFire1, readCounter1 + 1.U, readCounter1)
                writeDone1 := Mux(io.in.dataOutFire1 && writeCounter1 === writeLim1, true.B, writeDone1)
                writeBank1 := writeBank1
                writeCounter1 := Mux(io.in.dataOutFire1, writeCounter1 + 1.U, writeCounter1)
                inst2 := inst2
                readDone2 := Mux(readValid1 && io.in.dataInFire1 && readCounter2 === doubleReadLim, true.B, readDone2)
                readBank2 := readBank2; readCounter2 := Mux(readValid1 && io.in.dataInFire1, readCounter2 + 1.U, readCounter2)
                writeDone2 := Mux(doubleUsesDualWritePorts && io.in.dataOutFire2 && writeCounter2 === writeLim2, true.B, writeDone2)
                writeBank2 := writeBank2
                writeCounter2 := Mux(doubleUsesDualWritePorts && io.in.dataOutFire2, writeCounter2 + 1.U, writeCounter2)
                readValid1 := !readDone1; readValid2 := !readDone2
            }
        }
    }

    io.out.inst1 := inst1; io.out.inst2 := inst2
    io.out.imm1 := imm1Reg; io.out.imm2 := imm2Reg
    io.out.packScale1 := packScale1Reg; io.out.packScale2 := packScale2Reg
    io.out.unpackScale1 := unpackScale1Reg; io.out.unpackScale2 := unpackScale2Reg
    io.out.readValid1 := readValid1
    io.out.readBank1 := Mux(
        readEarly1,
        io.in.instReadBank1,
        Mux((state === double) && doubleIsRowReduce, readBank1, Mux(isNextReadBank1, nextReadBank1, readBank1))
    )
    io.out.readRow1 := readCounter1FiveBits
    io.out.readValid2 := readValid2
    io.out.readBank2 := Mux(
        readEarly2,
        Mux(newInputToDOUBLE, inputReadBank2, io.in.instReadBank1),
        Mux((state === double) && doubleIsRowReduce, readBank2, Mux(isNextReadBank2, nextReadBank2, readBank2))
    )
    io.out.readRow2 := readCounter2FiveBits
    io.out.writeValid1 := Mux(
        isColReduce1,
        !writeDone1 && io.in.dataOutFire1 && writeCounter1 >= 64.U,
        !writeDone1 && io.in.dataOutFire1
    )
    io.out.writeBank1 := Mux(
        (state === double) && doubleUsesDualWritePorts,
        writeBank1,
        Mux(isNextWriteBank1, nextWriteBank1, writeBank1)
    )
    io.out.writeRow1 := writeCounter1FiveBits
    io.out.writeCount1 := writeCounter1(5, 0)
    io.out.writeValid2 := Mux(
        isColReduce2,
        !writeDone2 && io.in.dataOutFire2 && writeCounter2 >= 64.U,
        !writeDone2 && io.in.dataOutFire2
    )
    io.out.writeBank2 := Mux(
        (state === double) && doubleUsesDualWritePorts,
        writeBank2,
        Mux(isNextWriteBank2, nextWriteBank2, writeBank2)
    )
    io.out.writeRow2 := writeCounter2FiveBits
    io.out.writeCount2 := writeCounter2(5, 0)
    io.out.done1 := done1; io.out.done2 := done2; io.out.VEReady := VEReady
    io.out.isDoneReadingColMax := ((readCounter1 >= 65.U || readDone1) && inst1 === VPUOp.cmax) || ((readCounter2 >= 65.U || readDone2) && inst2 === VPUOp.cmax)
    io.out.isDoneReadingColMin := ((readCounter1 >= 65.U || readDone1) && inst1 === VPUOp.cmin) || ((readCounter2 >= 65.U || readDone2) && inst2 === VPUOp.cmin)
    io.out.isDoneReadingColSum := ((readCounter1 >= 65.U || readDone1) && inst1 === VPUOp.csum) || ((readCounter2 >= 65.U || readDone2) && inst2 === VPUOp.csum)
    io.out.isFirstEntryColSum := (readCounter1 === 1.U && inst1 === VPUOp.csum) || (readCounter2 === 1.U && inst2 === VPUOp.csum)
    io.out.state := state; io.out.readDone1 := readDone1; io.out.readDone2 := readDone2
    io.out.writeDone1 := writeDone1; io.out.writeDone2 := writeDone2

    val emptyBankPort = 0.U.asTypeOf(Valid(UInt(6.W)))
    val activeReads = WireInit(VecInit(Seq.fill(4)(emptyBankPort)))
    val activeWrites = WireInit(VecInit(Seq.fill(4)(emptyBankPort)))
    val slot1Active = !readDone1 || !writeDone1
    val slot2Active = !readDone2 || !writeDone2
    val slot1ReadActive = slot1Active && !readDone1 && !opIsVli(inst1)
    val slot1WriteActive = slot1Active && !writeDone1
    val slot2ReadActiveSingle = (state === single) && slot2Active && !readDone2 && !opIsVli(inst2)
    val slot2ReadActiveDouble = (state === double) && slot2Active && !readDone2
    val mirroredDoubleRead = slot2ReadActiveDouble && opReadsSecondaryPair(inst1) && (readBank1 === readBank2)
    val slot2WriteActive =
        ((state === single) && slot2Active && !writeDone2) ||
        ((state === double) && doubleUsesDualWritePorts && slot2Active && !writeDone2)

    assert(!(mirroredDoubleRead && (readCounter1 =/= readCounter2)),
        "VPU FSM: aliased two-input operand reads must keep both counters aligned")
    assert(!(mirroredDoubleRead && (readDone1 =/= readDone2)),
        "VPU FSM: aliased two-input operand reads must complete together")

    activeReads(0).valid := slot1ReadActive
    activeReads(0).bits  := readBank1
    activeReads(1).valid := slot1ReadActive && opReadsPrimaryPair(inst1) && !((state === double) && doubleIsRowReduce)
    activeReads(1).bits  := readBank1 + 1.U
    activeReads(2).valid := slot2ReadActiveSingle || (slot2ReadActiveDouble && !mirroredDoubleRead)
    activeReads(2).bits  := readBank2
    activeReads(3).valid := Mux(state === double,
        slot2ReadActiveDouble && opReadsSecondaryPair(inst1) && !mirroredDoubleRead,
        slot2ReadActiveSingle && opReadsPrimaryPair(inst2)
    )
    activeReads(3).bits := readBank2 + 1.U

    activeWrites(0).valid := slot1WriteActive
    activeWrites(0).bits  := writeBank1
    activeWrites(1).valid := slot1WriteActive && opWritesPair(inst1) && !((state === double) && doubleUsesDualWritePorts)
    activeWrites(1).bits  := writeBank1 + 1.U
    activeWrites(2).valid := slot2WriteActive
    activeWrites(2).bits  := writeBank2
    activeWrites(3).valid := (state === single) && slot2WriteActive && opWritesPair(inst2)
    activeWrites(3).bits  := writeBank2 + 1.U

    io.out.activeReads := activeReads
    io.out.activeWrites := activeWrites
    io.out.issueBusy := VecInit(Seq(false.B) ++ VPUOp.all.toSeq.map(op => !canIssueOp(op))).asUInt
    io.out.busy := slot1Active || slot2Active
}
