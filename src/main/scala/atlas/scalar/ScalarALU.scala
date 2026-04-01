package atlas.scalar

import chisel3._
import chisel3.util._

class ScalarALU extends Module {
  val io = IO(new Bundle {
    val op  = Input(UInt(4.W))
    val a   = Input(UInt(32.W))
    val b   = Input(UInt(32.W))
    val out = Output(UInt(32.W))
  })

  val shamt = io.b(4, 0)   // RV32I: 5-bit shift amount

  io.out := MuxLookup(io.op, 0.U)(Seq(
    ScalarISA.ALU_ADD    -> (io.a + io.b),
    ScalarISA.ALU_SUB    -> (io.a - io.b),
    ScalarISA.ALU_AND    -> (io.a & io.b),
    ScalarISA.ALU_OR     -> (io.a | io.b),
    ScalarISA.ALU_XOR    -> (io.a ^ io.b),
    ScalarISA.ALU_SLL    -> (io.a << shamt)(31, 0),
    ScalarISA.ALU_SRL    -> (io.a >> shamt),
    ScalarISA.ALU_SRA    -> (io.a.asSInt >> shamt).asUInt,
    ScalarISA.ALU_SLT    -> (io.a.asSInt < io.b.asSInt).asUInt,
    ScalarISA.ALU_SLTU   -> (io.a < io.b).asUInt,
    ScalarISA.ALU_PASS_B -> io.b
  ))
}
