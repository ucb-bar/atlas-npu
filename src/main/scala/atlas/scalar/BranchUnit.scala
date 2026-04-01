package atlas.scalar

import chisel3._
import chisel3.util._

class BranchUnit extends Module {
  val io = IO(new Bundle {
    val branch_type = Input(UInt(3.W))
    val rs1         = Input(UInt(32.W))
    val rs2         = Input(UInt(32.W))
    val taken       = Output(Bool())
  })

  val eq  = io.rs1 === io.rs2
  val lt  = io.rs1.asSInt < io.rs2.asSInt
  val ltu = io.rs1 < io.rs2

  io.taken := MuxLookup(io.branch_type, false.B)(Seq(
    ScalarISA.BR_NONE -> false.B,
    ScalarISA.BR_EQ   -> eq,
    ScalarISA.BR_NE   -> !eq,
    ScalarISA.BR_LT   -> lt,
    ScalarISA.BR_GE   -> !lt,
    ScalarISA.BR_LTU  -> ltu,
    ScalarISA.BR_GEU  -> !ltu
  ))
}
