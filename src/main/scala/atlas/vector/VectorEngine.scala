package atlas.vector

import chisel3._
import chisel3.util._
import atlas.common.VPUParams
import fpex._
import sp26FPUnits._

class VectorEngine(val p: VPUParams) extends Module {
  val io = IO(new Bundle {
    val in1   = Flipped(Decoupled(new VPUInput(p)))
    val in2   = Flipped(Decoupled(new VPUInput(p)))
    val out1  = Decoupled(new VPUOutput(p))
    val out2  = Decoupled(new VPUOutput(p))
  })

  // Recoded width is one bit wider
  val recW = p.expWidth + p.sigWidth + 1

  // Instruction Decodes
  val isAdd     = io.in1.bits.instType === VPUOp.add
  val isSub     = io.in1.bits.instType === VPUOp.sub
  val isMul     = io.in1.bits.instType === VPUOp.mul
  val isRcp     = io.in1.bits.instType === VPUOp.rcp
  val isSqrt    = io.in1.bits.instType === VPUOp.sqrt
  val isExp     = io.in1.bits.instType === VPUOp.exp
  val isExp2    = io.in1.bits.instType === VPUOp.exp2
  val isSin     = io.in1.bits.instType === VPUOp.sin
  val isCos     = io.in1.bits.instType === VPUOp.cos
  val isTanh    = io.in1.bits.instType === VPUOp.tanh
  val isLog     = io.in1.bits.instType === VPUOp.log
  val isMax     = io.in1.bits.instType === VPUOp.max
  val isReduSum = io.in1.bits.instType === VPUOp.reduSum
  val isFp8     = io.in1.bits.instType === VPUOp.fp8
  val isSquare  = io.in1.bits.instType === VPUOp.square
  val isCube    = io.in1.bits.instType === VPUOp.cube

  // -------------------------------------------------------------------------
  // Functional Units
  // -------------------------------------------------------------------------

  val addSub = Module(new AddSubRec(p.BF16T))
  addSub.io.req.valid := io.in1.valid && io.in2.valid && (isAdd || isSub)
  addSub.io.req.bits.roundingMode := 0.U         
  addSub.io.req.bits.tag := 0.U                  
  addSub.io.req.bits.whichBank := 0.U            
  addSub.io.req.bits.wRow := 0.U                 
  addSub.io.req.bits.sub := isSub
  addSub.io.req.bits.laneMask := 0xFFFF.U(16.W)  
  addSub.io.req.bits.aVec := io.in1.bits.vector
  addSub.io.req.bits.bVec := io.in2.bits.vector

  val mul = Module(new MulRec(p.BF16T))
  mul.io.req.valid := io.in1.valid && io.in2.valid && isMul
  mul.io.req.bits.roundingMode := 0.U            
  mul.io.req.bits.tag := 0.U                     
  mul.io.req.bits.whichBank := 0.U               
  mul.io.req.bits.wRow := 0.U                    
  mul.io.req.bits.laneMask := 0xFFFF.U(16.W)     
  mul.io.req.bits.aVec := io.in1.bits.vector
  mul.io.req.bits.bVec := io.in2.bits.vector

  val divSqrt = Module(new DivSqrtRec(p.BF16T))
  divSqrt.io.req.valid := io.in1.valid && io.in2.valid  && (isRcp || isSqrt)
  divSqrt.io.req.bits.roundingMode := 0.U        
  divSqrt.io.req.bits.tag := 0.U                 
  divSqrt.io.req.bits.whichBank := 0.U           
  divSqrt.io.req.bits.wRow := 0.U                
  divSqrt.io.req.bits.isSqrt := isSqrt
  divSqrt.io.req.bits.laneMask := 0xFFFF.U(16.W) 
  divSqrt.io.req.bits.aVec := io.in1.bits.vector
  divSqrt.io.req.bits.bVec := io.in2.bits.vector

  val exp = Module(new Exp(p.BF16T))
  exp.io.req.valid := io.in1.valid && (isExp || isExp2) 
  exp.io.req.bits.roundingMode := 0.U            
  exp.io.req.bits.tag := 0.U                     
  exp.io.req.bits.whichBank := 0.U               
  exp.io.req.bits.wRow := 0.U                    
  exp.io.req.bits.neg := io.in1.bits.isExpNeg
  exp.io.req.bits.isBase2 := isExp2
  exp.io.req.bits.laneMask := 0xFFFF.U(16.W)     
  exp.io.req.bits.xVec := io.in1.bits.vector

  val sinCos = Module(new SinCos(p.BF16T))
  sinCos.io.req.valid := io.in1.valid && (isSin || isCos)
  sinCos.io.req.bits.roundingMode := 0.U         
  sinCos.io.req.bits.tag := 0.U                  
  sinCos.io.req.bits.whichBank := 0.U            
  sinCos.io.req.bits.wRow := 0.U                 
  sinCos.io.req.bits.cos := isCos
  sinCos.io.req.bits.laneMask := 0xFFFF.U(16.W)  
  sinCos.io.req.bits.xVec := io.in1.bits.vector

  val log2 = Module(new Log2Rec(p.BF16T))
  log2.io.req.valid := io.in1.valid && isLog
  log2.io.req.bits.tag := 0.U                    
  log2.io.req.bits.whichBank := 0.U              
  log2.io.req.bits.wRow := 0.U                   
  log2.io.req.bits.laneMask := 0xFFFF.U(16.W)    
  log2.io.req.bits.xVec := io.in1.bits.vector

  val tanh = Module(new TanhRec(p.BF16T))
  tanh.io.req.valid := io.in1.valid && isTanh
  tanh.io.req.bits.tag := 0.U                    
  tanh.io.req.bits.whichBank := 0.U              
  tanh.io.req.bits.wRow := 0.U                   
  tanh.io.req.bits.laneMask := 0xFFFF.U(16.W)    
  tanh.io.req.bits.xVec := io.in1.bits.vector

  val max = Module(new MaxRedu(p.BF16T))
  max.io.req.valid := io.in1.valid && isMax
  max.io.req.bits.tag := 0.U                    
  max.io.req.bits.whichBank := 0.U               
  max.io.req.bits.wRow := 0.U                    
  max.io.req.bits.aVec := io.in1.bits.vector

  val reduSum = Module(new ReduSumRec(p.BF16T))
  reduSum.io.req.valid := io.in1.valid && isReduSum
  reduSum.io.req.bits.tag := 0.U                   
  reduSum.io.req.bits.whichBank := 0.U 
  reduSum.io.req.bits.wRow := 0.U      
  reduSum.io.req.bits.roundingMode := 0.U          
  reduSum.io.req.bits.laneMask := 0xFFFF.U(16.W)   
  reduSum.io.req.bits.aVec := io.in1.bits.vector

  val sqcb = Module(new SquareCubeRec(p.BF16T))
  sqcb.io.req.valid := io.in1.valid && (isSquare || isCube)
  sqcb.io.req.bits.roundingMode := 0.U            
  sqcb.io.req.bits.tag := 0.U                     
  sqcb.io.req.bits.whichBank := 0.U               
  sqcb.io.req.bits.wRow := 0.U                    
  sqcb.io.req.bits.laneMask := 0xFFFF.U(16.W)     
  sqcb.io.req.bits.aVec := io.in1.bits.vector   
  sqcb.io.req.bits.isCube := isCube  
  
  val fp8 = Module(new FP8(p.BF16T))
  fp8.io.req.valid := io.in1.valid && isFp8
  fp8.io.req.bits.tag := 0.U 
  fp8.io.req.bits.whichBank := 0.U
  fp8.io.req.bits.wRow := 0.U
  fp8.io.req.bits.laneMask := 0xFFFF.U(16.W)
  fp8.io.req.bits.xVec := io.in1.bits.vector
  fp8.io.req.bits.expShift := 0.S
  fp8.io.req.bits.leftAlign := false.B

  // -------------------------------------------------------------------------
  // Mux Routing Interface
  // -------------------------------------------------------------------------

  // Tie all unit ready signals directly to the output. If an idle unit burps 
  // garbage data, this ensures it instantly flushes out instead of clogging.
  addSub.io.resp.ready  := io.out1.ready
  mul.io.resp.ready     := io.out1.ready
  divSqrt.io.resp.ready := io.out1.ready
  exp.io.resp.ready     := io.out1.ready
  sinCos.io.resp.ready  := io.out1.ready
  log2.io.resp.ready    := io.out1.ready
  tanh.io.resp.ready    := io.out1.ready
  max.io.resp.ready     := io.out1.ready
  reduSum.io.resp.ready := io.out1.ready
  sqcb.io.resp.ready    := io.out1.ready
  fp8.io.resp.ready     := io.out1.ready

  // Route Input Ready
  io.in1.ready := MuxCase(false.B, Seq(
    (isAdd || isSub)     -> addSub.io.req.ready,
    isMul                -> mul.io.req.ready,
    (isRcp || isSqrt)    -> divSqrt.io.req.ready,
    (isExp || isExp2)    -> exp.io.req.ready,
    (isSin || isCos)     -> sinCos.io.req.ready,
    isTanh               -> tanh.io.req.ready,
    isLog                -> log2.io.req.ready,
    isMax                -> max.io.req.ready,
    isReduSum            -> reduSum.io.req.ready,
    (isSquare || isCube) -> sqcb.io.req.ready,
    isFp8                -> fp8.io.req.ready
  ))
  io.in2.ready := io.in1.ready

  io.out2.valid := false.B
  io.out2.bits  := DontCare

  // Route Output Valid (Crucial fix to prevent early garbage reads!)
  io.out1.valid := MuxCase(false.B, Seq(
    (isAdd || isSub)     -> addSub.io.resp.valid,
    isMul                -> mul.io.resp.valid,
    (isRcp || isSqrt)    -> divSqrt.io.resp.valid,
    (isExp || isExp2)    -> exp.io.resp.valid,
    (isSin || isCos)     -> sinCos.io.resp.valid,
    isTanh               -> tanh.io.resp.valid,
    isLog                -> log2.io.resp.valid,
    isMax                -> max.io.resp.valid,
    isReduSum            -> reduSum.io.resp.valid,
    (isSquare || isCube) -> sqcb.io.resp.valid,
    isFp8                -> fp8.io.resp.valid
  ))

  // Route Output Data
  for (i <- 0 until p.numLanes) {
    io.out1.bits.vectorOutputData(i) := MuxCase(0.U, Seq(
      (isAdd || isSub)     -> addSub.io.resp.bits.result(i),
      isMul                -> mul.io.resp.bits.result(i),
      (isRcp || isSqrt)    -> divSqrt.io.resp.bits.result(i),
      (isExp || isExp2)    -> exp.io.resp.bits.result(i),
      (isSin || isCos)     -> sinCos.io.resp.bits.result(i),
      isTanh               -> tanh.io.resp.bits.result(i),
      isLog                -> log2.io.resp.bits.result(i),
      isMax                -> max.io.resp.bits.result(i),
      isReduSum            -> reduSum.io.resp.bits.result(i),
      (isSquare || isCube) -> sqcb.io.resp.bits.result(i),
      isFp8                -> fp8.io.resp.bits.result(i)
    ))
  }
}