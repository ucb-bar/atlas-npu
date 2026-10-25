// ============================================================================
// GeluVec.scala — Vector GELU lane box (vgelu.bf16).
//
// GELU_tanh(x) as the eight BF16 instructions of its definition, through the
// lane boxes that execute them:
//   t = vcube(x); t = vmul(t, c1); t = vadd(t, x); t = vmul(t, c2);
//   t = vtanh(t); t = vadd(t, 1); y = vmul(x, t); y = vmul(y, 0.5)
// Each step rounds as its instruction does, so vgelu gives the result of the
// sequence bit for bit. One row per cycle. A result comes 10 cycles after its
// row: one cycle in each lane box, and a register in front of each adder, so
// that no stage is deeper than the input stage of vadd.
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._

class GeluReq(wordWidth: Int, numLanes: Int) extends Bundle {
    val aVec = Vec(numLanes, UInt(wordWidth.W))
}

class GeluResp(wordWidth: Int, numLanes: Int) extends Bundle {
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class GeluVec(BF16T: AtlasFPType, numLanes: Int = 16) extends Module {
    val io = IO(new Bundle {
        val req = Flipped(Valid(new GeluReq(BF16T.wordWidth, numLanes)))
        val resp = Valid(new GeluResp(BF16T.wordWidth, numLanes))
    })

    // BF16 bits of 0.044715 and sqrt(2/pi), nearest even (ditflow kernels.py GELU_K1, GELU_K2)
    val (c1, c2, one, half) = (0x3D37, 0x3F4C, 0x3F80, 0x3F00)
    val allLanes = Fill(numLanes, 1.U(1.W))

    private def splat(bits: Int): Vec[UInt] = VecInit.fill(numLanes)(bits.U(BF16T.wordWidth.W))

    private def response(valid: Bool, result: Vec[UInt]): Valid[Vec[UInt]] = {
        val r = Wire(Valid(Vec(numLanes, UInt(BF16T.wordWidth.W))))
        r.valid := valid
        r.bits  := result
        r
    }

    private def cube(valid: Bool, a: Vec[UInt]): Valid[Vec[UInt]] = {
        val box = Module(new SquareCubeVec(BF16T, numLanes))
        box.io.req.valid             := valid
        box.io.req.bits.roundingMode := 0.U
        box.io.req.bits.isCube       := true.B
        box.io.req.bits.laneMask     := allLanes
        box.io.req.bits.aVec         := a
        response(box.io.resp.valid, box.io.resp.bits.result)
    }

    private def mul(valid: Bool, a: Vec[UInt], b: Vec[UInt]): Valid[Vec[UInt]] = {
        val box = Module(new MulRec(BF16T, numLanes))
        box.io.req.valid             := valid
        box.io.req.bits.roundingMode := 0.U
        box.io.req.bits.tag          := 0.U
        box.io.req.bits.whichBank    := 0.U
        box.io.req.bits.wRow         := 0.U
        box.io.req.bits.laneMask     := allLanes
        box.io.req.bits.aVec         := a
        box.io.req.bits.bVec         := b
        response(box.io.resp.valid, box.io.resp.bits.result)
    }

    private def add(valid: Bool, a: Vec[UInt], b: Vec[UInt]): Valid[Vec[UInt]] = {
        val box = Module(new AddSubSumVec(BF16T, numLanes))
        box.io.req.valid                    := valid
        box.io.req.bits.isSub               := false.B
        box.io.req.bits.isDoneReadingColSum := false.B
        box.io.req.bits.aVec                := a
        box.io.req.bits.bVec                := b
        response(box.io.resp.valid, box.io.resp.bits.result)
    }

    private def tanh(valid: Bool, a: Vec[UInt]): Valid[Vec[UInt]] = {
        val box = Module(new TanhRec(BF16T, numLanes))
        box.io.req.valid          := valid
        box.io.req.bits.tag       := 0.U
        box.io.req.bits.whichBank := 0.U
        box.io.req.bits.wRow      := 0.U
        box.io.req.bits.laneMask  := allLanes
        box.io.req.bits.xVec      := a
        response(box.io.resp.valid, box.io.resp.bits.result)
    }

    // x for the two steps that read it again: 3 and 8 cycles after its row.
    val x  = io.req.bits.aVec
    val x3 = ShiftRegister(x, 3)
    val x8 = ShiftRegister(x3, 5)

    val t0 = cube(io.req.valid, x)
    val t1 = Pipe(mul(t0.valid, t0.bits, splat(c1)))
    val t2 = add(t1.valid, t1.bits, x3)
    val t3 = mul(t2.valid, t2.bits, splat(c2))
    val t4 = Pipe(tanh(t3.valid, t3.bits))
    val t5 = add(t4.valid, t4.bits, splat(one))
    val y1 = mul(t5.valid, x8, t5.bits)
    val y2 = mul(y1.valid, y1.bits, splat(half))

    io.resp.valid       := y2.valid
    io.resp.bits.result := y2.bits
}
