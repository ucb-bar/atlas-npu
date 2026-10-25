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
  val write       = Input(Bool())    // the instruction writes the CSR (csrrs / csrrc with x0 do not)
  val illegal     = Output(Bool())   // addr is not mapped, or write is set and the CSR is read-only
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
  val mxu1Cmd       = Valid(new MxuCmd)
  val nvCmd         = Valid(new MxuCmd)   // NVFP4 MXU (MXU slot 0)
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
  // The causes of the stallUnit and stallData cycles; a cycle sets every cause that holds.
  val stallUnitMxu0    = Bool()   // MXU slot 0 (the NVFP4 MXU) cannot take the command
  val stallUnitMxu1    = Bool()   // MXU1 cannot take the command
  val stallUnitVpu     = Bool()   // the VPU cannot take the command
  val stallUnitXlu     = Bool()   // the XLU is busy
  val stallUnitLsu     = Bool()   // the vload or vstore path is busy, or a scalar load is pending
  val stallUnitDma     = Bool()   // the DMA channel or the next command slot is busy
  val stallDataMreg    = Bool()   // MREG reservation or physical port
  val stallDataVmem    = Bool()   // an LSU path holds the VMEM bank
  val stallDataDmaVmem = Bool()   // a DMA transfer would overtake an LSU transfer on its banks
  val stallDataXreg    = Bool()   // a scalar load result or the register write port is pending
  val stallDataScale   = Bool()   // a seld result or the scale register write port is pending
}
