/*
PcControl.scala
Program counter control for the scalar pipeline frontend.

The PC is a word index into IMEM starting at 0; the architectural PC is the
byte address, 4 x the index. Increments by 1 per cycle.

No delay slot. A taken branch or jump drops the instruction fetched after it:
  Cycle N  : branch/jump evaluated in execute (at branch_pc).
             Instruction at branch_pc+1 is in-flight; it is dropped.
             Fetch redirects to the target for the next cycle.
  Cycle N+1: bubble. Target instruction is in-flight.
  Cycle N+2: target instruction executes.
*/

package atlas.scalar

import chisel3._
import chisel3.util._

/** Program-counter controller.
  *
  * @param resetPC  Initial IMEM word index after reset.
  */
class PcControl(resetPC: Int = 0) extends Module {
  val io = IO(new Bundle {
    val redirect        = Input(Bool())
    val redirect_target = Input(UInt(32.W))   // word index
    val stall           = Input(Bool())
    val halted          = Input(Bool())
    val restart         = Input(Bool())
    val pc              = Output(UInt(32.W))   // word index for IMEM fetch
    val s1_pc           = Output(UInt(32.W))   // word index of the instruction in execute
    val s1_valid        = Output(Bool())
  })

  val pc_reg       = RegInit(resetPC.U(32.W))
  val fetch_pc_reg = RegInit(resetPC.U(32.W))
  val s1_valid_reg = RegInit(false.B)

  when(io.restart) {
    pc_reg        := resetPC.U
    fetch_pc_reg  := resetPC.U
    s1_valid_reg  := false.B
  }.elsewhen(io.halted) {
    s1_valid_reg := false.B
  }.elsewhen(io.stall) {
    // All registers hold their values.  Do NOT overwrite fetch_pc_reg
    // with pc_reg here: by the time a stall takes effect, pc_reg has
    // already advanced past the stalled instruction.  Copying it would
    // corrupt s1_pc, breaking PC-relative branches after DELAY stalls.
  }.otherwise {
    // A redirect drops the in-flight fetch (pc_reg) and fetches the target.
    s1_valid_reg := !io.redirect
    fetch_pc_reg := pc_reg
    pc_reg       := Mux(io.redirect, io.redirect_target, pc_reg + 1.U)
  }

  io.pc       := pc_reg
  io.s1_pc    := fetch_pc_reg
  io.s1_valid := s1_valid_reg
}
