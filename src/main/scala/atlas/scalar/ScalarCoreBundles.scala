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
  val addr     = UInt(64.W)
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

class LsuCmd(vmemP: VmemParams) extends Bundle {
  val op           = UInt(3.W)
  val mregBank     = UInt(6.W)
  val vmemLineAddr = UInt(vmemP.lineAddrBits.W)
}

class ScalarCoreIO(spP: VmemParams) extends Bundle {
  val imemFetch     = new ImemFetchPort
  val dmaCmd        = Valid(new DmaCmd)
  val mxu0Cmd       = Valid(new MxuCmd)
  val mxu1Cmd       = Valid(new MxuCmd)
  val nvCmd         = Valid(new MxuCmd)   // NVFP4 MXU (MXU_NV_* in op)
  val vpuCmd        = Valid(new VpuCmd)
  val xluCmd        = Valid(new XluCmd)
  val lsuCmd        = Valid(new LsuCmd(spP))

  val scalarMemCmd  = Valid(new LsuScalarCmd(spP))
  val scalarMemResp = Flipped(Valid(UInt(32.W)))

  val dma_busy      = Input(Vec(8, Bool()))
  val vpu_status    = Input(new VpuStatus)

  // Split LSU busy: scalar load path plus independent VLOAD / VSTORE engines.
  val lsu_scalar_busy = Input(Bool())
  val lsu_vload_busy  = Input(Bool())
  val lsu_vstore_busy = Input(Bool())

  val mregReadBusy   = Input(UInt(64.W))
  val mregWriteBusy  = Input(UInt(64.W))

  // Interlock: each unit's ability to take the command decoded in S1.
  val mxu0Ready       = Input(Bool())   // MXU0 would accept mxu0Cmd.bits this cycle
  val mxu1Ready       = Input(Bool())   // MXU1 would accept mxu1Cmd.bits this cycle
  val nvReady         = Input(Bool())   // the NVFP4 MXU would accept nvCmd.bits this cycle
  val xlu_busy        = Input(Bool())
  val dma_can_enqueue = Input(Bool())
  val lsu_vload_vmem_bank  = Flipped(Valid(UInt(spP.bankIdBits.W)))
  val lsu_vstore_vmem_bank = Flipped(Valid(UInt(spP.bankIdBits.W)))

  val csrPort       = Flipped(new CSRInternalPort)
  val scaleRegs     = Output(Vec(NUM_SCALE_REGS, UInt(8.W)))
  val dmaBaseReg    = Output(UInt(32.W))
  val halted        = Output(Bool())
  val execRun       = Input(Bool())
  val execRunWrite  = Input(Bool())
  val perf          = Output(new ScalarPerfEvents)
}

/** Per-cycle front-end events for the performance counters (CSRFile). */
class ScalarPerfEvents extends Bundle {
  val running    = Bool()   // the core is not halted
  val issue      = Bool()   // an instruction issued from S1
  val stallUnit  = Bool()   // S1 holds an instruction: its unit cannot take the command (structural)
  val stallData  = Bool()   // S1 holds an instruction for a data hazard: MREG, VMEM bank, scalar load, scale
  val stallDma   = Bool()   // dma.wait on a busy channel
  val stallDelay = Bool()   // a delay instruction counts down
}
