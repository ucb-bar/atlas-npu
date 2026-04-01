package atlas.scalar

import chisel3._
import chisel3.util._

class ScalarRegFile extends Module {
  val io = IO(new Bundle {
    val rs1_addr = Input(UInt(5.W))
    val rs2_addr = Input(UInt(5.W))
    val rd_addr  = Input(UInt(5.W))   // third read port for DMA x[rd]
    val rs1_data = Output(UInt(32.W))
    val rs2_data = Output(UInt(32.W))
    val rd_data  = Output(UInt(32.W))
    val wr_en    = Input(Bool())
    val wr_addr  = Input(UInt(5.W))
    val wr_data  = Input(UInt(32.W))
  })

  val regs = Reg(Vec(32, UInt(32.W)))

  io.rs1_data := Mux(io.rs1_addr === 0.U, 0.U, regs(io.rs1_addr))
  io.rs2_data := Mux(io.rs2_addr === 0.U, 0.U, regs(io.rs2_addr))
  io.rd_data  := Mux(io.rd_addr  === 0.U, 0.U, regs(io.rd_addr))

  when(io.wr_en && io.wr_addr =/= 0.U) {
    regs(io.wr_addr) := io.wr_data
  }
}
