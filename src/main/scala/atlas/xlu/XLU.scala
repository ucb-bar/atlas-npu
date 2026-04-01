/*
XLU.scala — Cross-lane Transpose Unit (XLU)
Updated for 32x32 fp8 tensor registers: n=32, w=8.
Each tensor register stores 32 rows x 32 fp8 values = 32 x 256 bits.
*/
package atlas.xlu

import atlas.common._
import chisel3._
import chisel3.util._

class pe(w: Int) extends Module {
  val io = IO(new Bundle {
    val c_in = Input(Bool()); val c_out = Output(Bool())
    val x_in = Input(UInt(w.W)); val x_not_empty = Input(Bool())
    val y_in = Input(UInt(w.W)); val y_not_empty = Input(Bool())
    val top_out = Output(UInt(w.W)); val top_not_empty = Output(Bool())
    val right_out = Output(UInt(w.W)); val right_not_empty = Output(Bool())
  })
  io.top_out       := RegNext(Mux(io.c_in, io.x_in, io.y_in), 0.U)
  io.top_not_empty := RegNext(Mux(io.c_in, io.x_not_empty, io.y_not_empty), false.B)
  io.right_out       := RegNext(Mux(io.c_in, io.y_in, io.x_in), 0.U)
  io.right_not_empty := RegNext(Mux(io.c_in, io.y_not_empty, io.x_not_empty), false.B)
  io.c_out := RegNext(io.c_in, false.B)
}

class SystolicTranspose(n: Int, w: Int) extends Module {
  val io = IO(new Bundle {
    val clear = Input(Bool())
    val c_in = Input(Vec(n, Bool())); val x_in = Input(Vec(n, UInt(w.W)))
    val x_not_empty = Input(Vec(n, Bool()))
    val col_out = Output(Vec(n, UInt(w.W))); val col_not_empty = Output(Vec(n, Bool()))
    val resultMatrix = Output(Vec(n, Vec(n, UInt(w.W))))
  })
  val pes = Seq.fill(n)(Seq.fill(n)(Module(new pe(w))))
  for (r <- 0 until n; c <- 0 until n) {
    if (c == 0) {
      pes(r)(c).io.c_in := RegNext(io.c_in(r), false.B)
      pes(r)(c).io.x_in := io.x_in(r); pes(r)(c).io.x_not_empty := io.x_not_empty(r)
    } else {
      pes(r)(c).io.c_in := RegNext(pes(r)(c-1).io.c_out, false.B)
      pes(r)(c).io.x_in := pes(r)(c-1).io.right_out; pes(r)(c).io.x_not_empty := pes(r)(c-1).io.right_not_empty
    }
    if (r == n-1) { pes(r)(c).io.y_in := 0.U; pes(r)(c).io.y_not_empty := false.B }
    else { pes(r)(c).io.y_in := pes(r+1)(c).io.top_out; pes(r)(c).io.y_not_empty := pes(r+1)(c).io.top_not_empty }
  }
  val result = RegInit(VecInit(Seq.fill(n)(VecInit(Seq.fill(n)(0.U(w.W))))))
  val element_count = RegInit(VecInit(Seq.fill(n)(0.U(log2Ceil(n+1).W))))
  for (c <- 0 until n) { io.col_out(c) := pes(0)(c).io.top_out; io.col_not_empty(c) := pes(0)(c).io.top_not_empty }
  when(io.clear) {
    for (j <- 0 until n) { element_count(j) := 0.U; for (i <- 0 until n) result(j)(i) := 0.U }
  }.otherwise {
    for (j <- 0 until n) when(io.col_not_empty(j)) {
      result(j)(element_count(j)(log2Ceil(n)-1, 0)) := io.col_out(j); element_count(j) := element_count(j) + 1.U
    }
  }
  io.resultMatrix := result
}

class xlu_instruction extends Bundle {
  val op = UInt(2.W)
  val whichBank_source = UInt(6.W)
  val whichBank_destination = UInt(6.W)
}

class xlu_engine(p: RegFileParams) extends Module {
  val n = 32   // 32x32 fp8 tensor registers
  val w = 8    // 8-bit FP8 elements

  val io = IO(new Bundle {
    val instruction_received = Flipped(Valid(new xlu_instruction))
    val busy = Output(Bool())
    val xluReadToMreg = Valid(new RegFileReadInput(p))
    val xluReadFromMreg = Flipped(Valid(UInt(p.SRAM_WIDTH.W)))
    val xluWriteToMreg = Valid(new RegFileWriteInput(p))
  })

  object xlu_states extends ChiselEnum {
    val IDLE, getting_data_from_mreg, array_start, array_in_progress, wait_for_result, send_toMREG = Value
  }
  import xlu_states._

  val state = RegInit(IDLE)
  val active_instruction = Reg(new xlu_instruction)
  val transit_spot = Reg(Vec(n, Vec(n, UInt(w.W))))

  private val nBits = log2Ceil(n) // 5
  val request_counter = RegInit(0.U(6.W))
  val respond_counter = RegInit(0.U(6.W))
  val which_column_entering = RegInit(0.U(6.W))
  val outputCounter = RegInit(0.U(7.W))
  val destination_row_counter = RegInit(0.U(6.W))

  val sys = Module(new SystolicTranspose(n, w))
  sys.io.clear := false.B
  for (r <- 0 until n) { sys.io.c_in(r) := false.B; sys.io.x_in(r) := 0.U; sys.io.x_not_empty(r) := false.B }

  io.xluReadToMreg.valid := false.B; io.xluReadToMreg.bits := 0.U.asTypeOf(io.xluReadToMreg.bits)
  io.xluWriteToMreg.valid := false.B; io.xluWriteToMreg.bits := 0.U.asTypeOf(io.xluWriteToMreg.bits)
  io.busy := state =/= IDLE

  switch(state) {
    is(IDLE) { when(io.instruction_received.valid) {
      active_instruction := io.instruction_received.bits
      state := getting_data_from_mreg
      request_counter := 0.U; respond_counter := 0.U
    }}
    is(getting_data_from_mreg) {
      when(request_counter < n.U) {
        io.xluReadToMreg.valid := true.B
        io.xluReadToMreg.bits.whichBank := active_instruction.whichBank_source
        io.xluReadToMreg.bits.rRow := request_counter(nBits-1, 0)
        request_counter := request_counter + 1.U
      }
      when(io.xluReadFromMreg.valid) {
        for (i <- 0 until n) transit_spot(respond_counter(nBits-1, 0))(i) := io.xluReadFromMreg.bits((i+1)*w-1, i*w)
        respond_counter := respond_counter + 1.U
      }
      when(io.xluReadFromMreg.valid && respond_counter === (n-1).U) { state := array_start }
    }
    is(array_start) {
      sys.io.clear := true.B
      for (r <- 0 until n) sys.io.c_in(r) := true.B
      which_column_entering := 0.U; state := array_in_progress
    }
    is(array_in_progress) {
      for (r <- 0 until n) {
        sys.io.x_in(r) := transit_spot(r)(which_column_entering(nBits-1, 0))
        sys.io.x_not_empty(r) := true.B
      }
      which_column_entering := which_column_entering + 1.U
      when(which_column_entering === (n-1).U) { state := wait_for_result; outputCounter := 0.U }
    }
    is(wait_for_result) {
      outputCounter := outputCounter + 1.U
      when(outputCounter === (2*n).U) { state := send_toMREG; destination_row_counter := 0.U }
    }
    is(send_toMREG) {
      io.xluWriteToMreg.valid := true.B
      io.xluWriteToMreg.bits.whichBank := active_instruction.whichBank_destination
      io.xluWriteToMreg.bits.wRow := destination_row_counter(nBits-1, 0)
      io.xluWriteToMreg.bits.wData := sys.io.resultMatrix(destination_row_counter(nBits-1, 0)).asUInt
      destination_row_counter := destination_row_counter + 1.U
      when(destination_row_counter === (n-1).U) { state := IDLE }
    }
  }
}
