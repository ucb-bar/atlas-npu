// ============================================================================
// ScalarCoreBundles.scala — Shared scalar-core bundles and command formats.
// ============================================================================

package atlas.scalar

import chisel3._
import chisel3.util._
import atlas.common.VmemParams
import atlas.lsu.LsuScalarCmd
import ScalarISA._

class CSRInternalPort extends Bundle {
  val addr        = Input(UInt(12.W))
  val op          = Input(UInt(3.W))
  val wdata       = Input(UInt(32.W))
  val rdata       = Output(UInt(32.W))
  val valid       = Input(Bool())
  val halted      = Input(Bool())
  val set_illegal = Input(Bool())
  val set_ecall   = Input(Bool())
  val set_ebreak  = Input(Bool())
  val illegal_pc  = Input(UInt(32.W))
  val inst_retire = Input(Bool())
}

class ImemFetchPort extends Bundle {
  val addr  = Output(UInt(AtlasMemMap.IMEM_ADDR_BITS.W))
  val rdata = Input(UInt(32.W))
}

class EngineStatus extends Bundle {
  val busy  = Bool()
  val done  = Bool()
  val error = Bool()
}

class VpuStatus extends Bundle {
  val busy      = Bool()
  val done      = Bool()
  val error     = Bool()
  // Bit i matches ScalarISA.VPU_* encoding i; bit 0 corresponds to VPU_NONE.
  val issueBusy = UInt(NUM_VPU_STATUS_OPS.W)
}

class DmaCmd extends Bundle {
  val op       = UInt(3.W)
  val vmemAddr = UInt(32.W)
  val addr     = UInt(32.W)
  val size     = UInt(32.W)
  val channel  = UInt(3.W)
}

class MxuCmd extends Bundle {
  val op         = UInt(4.W)
  val mregBank   = UInt(6.W)
  val accSel     = Bool()
  val weightSlot = Bool()
  val scaleE8M0  = UInt(8.W)
}

class VpuCmd extends Bundle {
  val op        = UInt(5.W)
  val vd        = UInt(6.W)
  val vs1       = UInt(6.W)
  val vs2       = UInt(6.W)
  val imm       = UInt(16.W)
  val scaleE8M0 = UInt(8.W)
}

class XluCmd extends Bundle {
  val op      = UInt(2.W)
  val dstBank = UInt(6.W)
  val srcBank = UInt(6.W)
}

class LsuCmd extends Bundle {
  val op           = UInt(2.W)
  val mregBank     = UInt(6.W)
  val vmemLineAddr = UInt(13.W)
}

class ScalarCoreIO(spP: VmemParams) extends Bundle {
  val imemFetch     = new ImemFetchPort
  val dmaCmd        = Valid(new DmaCmd)
  val mxu0Cmd       = Valid(new MxuCmd)
  val mxu1Cmd       = Valid(new MxuCmd)
  val vpuCmd        = Valid(new VpuCmd)
  val xluCmd        = Valid(new XluCmd)
  val lsuCmd        = Valid(new LsuCmd)

  val scalarMemCmd  = Valid(new LsuScalarCmd(spP))
  val scalarMemResp = Flipped(Valid(UInt(32.W)))

  val dma_busy      = Input(Vec(8, Bool()))
  val vpu_status    = Input(new VpuStatus)

  // Split LSU busy: scalar path (load pending) and vector path (vload/vstore active).
  val lsu_scalar_busy = Input(Bool())
  val lsu_vec_busy    = Input(Bool())

  val mregReadBusy   = Input(UInt(64.W))
  val mregWriteBusy  = Input(UInt(64.W))

  val csrPort       = Flipped(new CSRInternalPort)
  val scaleRegs     = Output(Vec(NUM_SCALE_REGS, UInt(8.W)))
  val dmaBaseReg    = Output(UInt(32.W))
  val halted        = Output(Bool())
  val execRun       = Input(Bool())
  val execRunWrite  = Input(Bool())
}
