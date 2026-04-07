/*
PcControl.scala
Program counter control for the scalar pipeline frontend.

The PC is a word index (not byte address) starting at 0.
It directly addresses IMEM words. Increments by 1 per cycle.

Implements 2 architectural delay slots:
  Cycle N  : branch/jump evaluated in execute (at branch_pc).
             Instruction at branch_pc+1 is in-flight (delay slot 1).
  Cycle N+1: delay slot 1 executes. Fetch PC advances to branch_pc+2
             (delay slot 2).
  Cycle N+2: delay slot 2 executes. Fetch redirects to target.
  Cycle N+3: target instruction executes.
*/

package atlas.scalar

import chisel3._
import chisel3.util._

/** Program-counter controller with architectural delay-slot handling.
  *
  * @param resetPC  Initial IMEM word index after reset.
  */
class PcControl(resetPC: Int = 0) extends Module {
  val io = IO(new Bundle {
    val redirect        = Input(Bool())
    val redirect_target = Input(UInt(32.W))
    val stall           = Input(Bool())
    val halted          = Input(Bool())
    val softReset       = Input(Bool())
    val pc              = Output(UInt(32.W))   // word index for IMEM fetch
    val s1_pc           = Output(UInt(32.W))   // PC of instruction in execute
    val s1_valid        = Output(Bool())
    val in_delay_slot   = Output(Bool())
  })

  val pc_reg       = RegInit(resetPC.U(32.W))
  val fetch_pc_reg = RegInit(resetPC.U(32.W))
  val s1_valid_reg = RegInit(false.B)

  // 2-deep delay slot counter: 0=normal, 2=first slot pending, 1=second slot pending
  val delay_count  = RegInit(0.U(2.W))
  val saved_target = Reg(UInt(32.W))

  when(io.softReset) {
    pc_reg        := resetPC.U
    fetch_pc_reg  := resetPC.U
    s1_valid_reg  := false.B
    delay_count   := 0.U
  }.elsewhen(io.halted) {
    s1_valid_reg := false.B
  }.elsewhen(io.stall) {
    fetch_pc_reg := pc_reg
  }.otherwise {
    s1_valid_reg := true.B
    fetch_pc_reg := pc_reg

    when(io.redirect && delay_count === 0.U) {
      // Branch evaluated. The in-flight fetch (pc_reg) is delay slot 1.
      // Advance to fetch delay slot 2.
      saved_target := io.redirect_target
      delay_count  := 2.U
      pc_reg       := pc_reg + 1.U
    }.elsewhen(delay_count === 2.U) {
      // Delay slot 1 now in execute. Delay slot 2 in-flight (at pc_reg).
      // Redirect fetch to target so it arrives after delay slot 2.
      delay_count := 1.U
      pc_reg      := saved_target
    }.elsewhen(delay_count === 1.U) {
      // Delay slot 2 now in execute. Target in-flight (at saved_target).
      // Resume normal sequential execution from target.
      delay_count := 0.U
      pc_reg      := pc_reg + 1.U
    }.otherwise {
      pc_reg := pc_reg + 1.U
    }
  }

  io.pc            := pc_reg
  io.s1_pc         := fetch_pc_reg
  io.s1_valid      := s1_valid_reg
  io.in_delay_slot := delay_count =/= 0.U
}
