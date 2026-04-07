/*
ScalarCoreBundles.scala
Shared scalar-core bundles and command formats.

Collects the small protocol bundles used between ScalarCore and neighboring
blocks such as the DMA engine, LSU, MXUs, VPU, XLU, and CSR path.
*/

package atlas.scalar

import chisel3._
import chisel3.util._
import atlas.common.VmemParams
import atlas.lsu.LsuScalarCmd
import ScalarISA._

// ================================================================
// Shared bundles used by ScalarCore and neighboring engine modules
// ================================================================

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

class DmaCmd extends Bundle {
  val op       = UInt(3.W)
  val vmemAddr = UInt(32.W)
  val addr     = UInt(32.W)
  val size     = UInt(32.W)
  val channel  = UInt(3.W)
}

class MxuCmd extends Bundle {
  val op         = UInt(4.W)
  val mregBank    = UInt(6.W)
  val accSel     = Bool()
  val weightSlot = Bool()
  val scaleE8M0  = UInt(8.W)
}

class VpuCmd extends Bundle {
  val op  = UInt(5.W)
  val vd  = UInt(6.W)
  val vs1 = UInt(6.W)
  val vs2 = UInt(6.W)
  val imm = UInt(16.W)
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

/** Top-level ScalarCore IO bundle.
  *
  * @param spP  Shared VMEM / scalar-memory geometry parameters.
  */
class ScalarCoreIO(spP: VmemParams) extends Bundle {
  val imemFetch     = new ImemFetchPort
  val dmaCmd        = Valid(new DmaCmd)
  val mxu0Cmd       = Valid(new MxuCmd)
  val mxu1Cmd       = Valid(new MxuCmd)
  val vpuCmd        = Valid(new VpuCmd)
  val xluCmd        = Valid(new XluCmd)
  val lsuCmd        = Valid(new LsuCmd)

  // Scalar mem via LSU
  val scalarMemCmd  = Valid(new LsuScalarCmd(spP))
  val scalarMemResp = Flipped(Valid(UInt(32.W)))

  val dma_busy      = Input(Vec(8, Bool()))
  val vpu_status    = Input(new EngineStatus)
  val lsu_busy      = Input(Bool())

  // Engine command backpressure: true when the engine can accept a new command
  val mxu0_cmd_ready = Input(Bool())
  val mxu1_cmd_ready = Input(Bool())
  val xlu_cmd_ready  = Input(Bool())

  // Direction-aware MREG bank busy bitvectors from the bank tracker
  val mregReadBusy   = Input(UInt(64.W))
  val mregWriteBusy  = Input(UInt(64.W))

  val csrPort       = Flipped(new CSRInternalPort)
  val scaleRegs     = Output(Vec(NUM_SCALE_REGS, UInt(8.W)))
  val dmaBaseReg    = Output(UInt(32.W))
  val halted        = Output(Bool())
  val softReset     = Input(Bool())
}
