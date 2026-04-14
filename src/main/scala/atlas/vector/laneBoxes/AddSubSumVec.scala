package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._


// Input bundles
class AddSubSumReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val isSub = Bool()  // 1 = sub, 0 = add
    val isSum = Bool()  // 1 = sumReduce, 0 = add/sub
    val isDoneReadingColSum = Bool() // For column reduction instructions, indicates when done reading all elements for a lane
    val aVec = Vec(numLanes, UInt(wordWidth.W))
    val bVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class AddSubSumResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class AddSubSumVec(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
    val io = IO(new Bundle {
        val req = Flipped(Valid(new AddSubSumReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Valid(new AddSubSumResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    // Adders
    val computeSigWidth = BF16T.sigWidth + 16
    val adders = Seq.fill(numLanes) {Module(new VectorAddRecFN(BF16T.expWidth, computeSigWidth))}
    
    // Stage 0: FN -> RecFN -> Rawfloat -> Add/Sub
    
    val aRecVecWiden = VecInit(io.req.bits.aVec.map(a => Cat(a, 0.U(16.W))))
    val bRecVecWiden = VecInit(io.req.bits.bVec.map(b => Cat(b, 0.U(16.W))))
    val aRecVec = VecInit(aRecVecWiden.map(a => recFNFromFN(BF16T.expWidth, computeSigWidth, a)))
    val bRecVec = VecInit(bRecVecWiden.map(b => recFNFromFN(BF16T.expWidth, computeSigWidth, b)))
    val addSubResult = Wire(Vec(numLanes, UInt((BF16T.expWidth + computeSigWidth + 1).W)))
    val reduceSumStage0 = Wire(Vec(8, UInt((BF16T.expWidth + computeSigWidth + 1).W)))
    for (i <- 0 until 8) {
        adders(i).io.subOp := io.req.bits.isSub && !io.req.bits.isSum
        // adders(i).io.subOp := false.B
        // aRecVec(0-7)
        adders(i).io.a := aRecVec(i)
        // bRecVec(0-7) or aRecVec(8-15)
        adders(i).io.b := Mux(io.req.valid && io.req.bits.isSum, aRecVec(i+8), bRecVec(i))
        adders(i).io.roundingMode := 0.U(3.W)             // May change this later
        adders(i).io.detectTininess := false.B
        addSubResult(i) := adders(i).io.out
        reduceSumStage0(i) := adders(i).io.out
    }

    // Stage 1
    val isValidStage1 = RegNext(io.req.valid, false.B)
    val isSubStage1 = RegNext(io.req.bits.isSub)
    val isSumStage1 = RegNext(io.req.bits.isSum)
    val reduceSumStage0Next = RegNext(reduceSumStage0)
    val reduceSumStage1 = Wire(Vec(4, UInt((BF16T.expWidth + computeSigWidth + 1).W)))
    for (i <- 0 until 4) {
        adders(i+8).io.subOp := io.req.bits.isSub && !isSumStage1
        // adders(i+8).io.subOp := false.B
        // aRecVec(8-11) or reduceSumStage0Next(0-3)
        adders(i+8).io.a := Mux(isSumStage1 && isValidStage1, reduceSumStage0Next(i), aRecVec(i+8))
        // bRecVec(8-11) or reduceSumStage0Next(4-7)
        adders(i+8).io.b := Mux(isSumStage1 && isValidStage1, reduceSumStage0Next(i+4), bRecVec(i+8))
        adders(i+8).io.roundingMode := 0.U(3.W)             // May change this later
        adders(i+8).io.detectTininess := false.B
        addSubResult(i+8) := adders(i+8).io.out
        reduceSumStage1(i) := adders(i+8).io.out
    }

    // Stage 2
    val isValidStage2 = RegNext(isValidStage1)
    val isSubStage2 = RegNext(isSubStage1)
    val isSumStage2 = RegNext(isSumStage1)
    val reduceSumStage1Next = RegNext(reduceSumStage1)
    val reduceSumStage2 = Wire(Vec(2, UInt((BF16T.expWidth + computeSigWidth + 1).W)))
    for (i <- 0 until 2) {
        // adders(i+12).io.subOp := io.req.bits.isSub && !isSumStage2   
        adders(i+12).io.subOp := io.req.bits.isSub && !isSumStage2
        // aRecVec(12-13) or reduceSumStage1Next(0-1)
        adders(i+12).io.a := Mux(isSumStage2 && isValidStage2, reduceSumStage1Next(i), aRecVec(i+12))
        // bRecVec(12-13) or reduceSumStage1Next(2-3)
        adders(i+12).io.b := Mux(isSumStage2 && isValidStage2, reduceSumStage1Next(i+2), bRecVec(i+12))
        adders(i+12).io.roundingMode := 0.U(3.W)             // May change this later
        adders(i+12).io.detectTininess := false.B
        addSubResult(i+12) := adders(i+12).io.out
        reduceSumStage2(i) := adders(i+12).io.out
    }

    // Stage 3
    val isValidStage3 = RegNext(isValidStage2)
    val isSubStage3 = RegNext(isSubStage2)
    val isSumStage3 = RegNext(isSumStage2)
    val reduceSumStage2Next = RegNext(reduceSumStage2)
    val reduceSumStage3 = Wire(UInt((BF16T.expWidth + computeSigWidth + 1).W))
    for (i <- 0 until 1) {
        // adders(i+14).io.subOp := io.req.bits.isSub && !isSumStage3
        adders(i+14).io.subOp := io.req.bits.isSub && !isSumStage3
        // aRecVec(14) or reduceSumStage2Next(0)
        adders(i+14).io.a := Mux(isSumStage3 && isValidStage3, reduceSumStage2Next(i), aRecVec(i+14))
        // bRecVec(14) or reduceSumStage2Next(1)
        adders(i+14).io.b := Mux(isSumStage3 && isValidStage3, reduceSumStage2Next(i+1), bRecVec(i+14))
        adders(i+14).io.roundingMode := 0.U(3.W)             // May change this later
        adders(i+14).io.detectTininess := false.B
        addSubResult(i+14) := adders(i+14).io.out
        reduceSumStage3 := adders(i+14).io.out
    }

    // Last element for normal Add/Sub
    // adders(15).io.subOp := io.req.bits.isSub
    adders(15).io.subOp := io.req.bits.isSub && !isSumStage3
    adders(15).io.a := aRecVec(15)
    adders(15).io.b := bRecVec(15)
    adders(15).io.roundingMode := 0.U(3.W)             // May change this later
    adders(15).io.detectTininess := false.B
    addSubResult(15) := adders(15).io.out

    // Stage 4: Output and Covnersion
    val isValidStage4 = RegNext(isValidStage3)
    val isSubStage4 = RegNext(isSubStage3)
    val isSumStage4 = RegNext(isSumStage3)
    val reduceSumStage3Next = RegNext(reduceSumStage3)
    val addSubResultNext = RegInit(0.U.asTypeOf(chiselTypeOf(addSubResult)))
    when (io.req.valid && !io.req.bits.isDoneReadingColSum) {addSubResultNext := addSubResult}

    // Outputs
    val isAddSubValid = (isValidStage1 && !isSumStage1)
    val isReduSumValid = (isValidStage4 && isSumStage4)
    val resValid = isAddSubValid || isReduSumValid
    val addSubOut = VecInit(addSubResultNext.map(v => fNFromRecFN(BF16T.expWidth, computeSigWidth, v)(31,16)))
    val reduSumOut = VecInit(Seq.fill(numLanes)(fNFromRecFN(BF16T.expWidth, computeSigWidth, reduceSumStage3Next)(31,16))) 

    // Assigning outputs
    io.resp.valid := resValid
    io.resp.bits.result := Mux(isReduSumValid, reduSumOut, addSubOut)
}
