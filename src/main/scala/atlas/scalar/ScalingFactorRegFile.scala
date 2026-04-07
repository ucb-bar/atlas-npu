/*
ScalingFactorRegFile.scala
Scalar-side register file for FP8 block-scaling factors.

Stores the small bank of E8M0 scaling bytes that scalar software can load
before issuing quantized MXU pop operations.
*/

package atlas.scalar

import chisel3._
import chisel3.util._
import ScalarISA._

/** Register file holding the software-visible FP8 scaling factors. */
class ScalingFactorRegFile extends Module {
  val io = IO(new Bundle {
    val writeEn   = Input(Bool())
    val writeIdx  = Input(UInt(5.W))
    val writeData = Input(UInt(8.W))
    val regs      = Output(Vec(NUM_SCALE_REGS, UInt(8.W)))
  })

  val reg = RegInit(VecInit(Seq.fill(NUM_SCALE_REGS)(0.U(8.W))))
  when(io.writeEn) { reg(io.writeIdx) := io.writeData }
  io.regs := reg
}