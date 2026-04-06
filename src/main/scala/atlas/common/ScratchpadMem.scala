// /*
// ScratchpadMem.scala — VMEM (Vector Memory / Scratchpad).

// Storage: SyncReadMem with UInt(lineWidth.W) elements — no Vec, no masks.
// All writes are full-line (256-bit). The LSU performs read-modify-write
// for scalar byte/half/word stores.

// Read arbitration:  LSU > DMA
// Write arbitration: LSU > DMA
// */

// package atlas.common

// import chisel3._
// import chisel3.util._

// // ── Port bundles ─────────────────────────────────────────────────────

// class ScratchpadLineReadPort(p: ScratchpadParams) extends Bundle {
//   val addr = UInt(p.lineAddrBits.W)
// }

// class ScratchpadLineWritePort(p: ScratchpadParams) extends Bundle {
//   val addr = UInt(p.lineAddrBits.W)
//   val data = UInt(p.lineWidth.W)
// }

// // ── ScratchpadMem ────────────────────────────────────────────────────

// class ScratchpadMem(p: ScratchpadParams) extends Module {

//   val io = IO(new Bundle {
//     val dmaRead     = Flipped(Valid(new ScratchpadLineReadPort(p)))
//     val dmaReadData = Valid(UInt(p.lineWidth.W))
//     val dmaWrite    = Flipped(Valid(new ScratchpadLineWritePort(p)))

//     val lsuRead     = Flipped(Valid(new ScratchpadLineReadPort(p)))
//     val lsuReadData = Valid(UInt(p.lineWidth.W))
//     val lsuWrite    = Flipped(Valid(new ScratchpadLineWritePort(p)))
//   })

//   val mem = SyncReadMem(p.numLines, UInt(p.lineWidth.W))

//   // ── Read arbitration (LSU > DMA) ─────────────────────────────────
//   val readAddr = Wire(UInt(p.lineAddrBits.W))
//   val readEn   = Wire(Bool())
//   val readSel  = Wire(Bool())   // false=LSU, true=DMA

//   when(io.lsuRead.valid) {
//     readAddr := io.lsuRead.bits.addr
//     readEn   := true.B
//     readSel  := false.B
//   }.elsewhen(io.dmaRead.valid) {
//     readAddr := io.dmaRead.bits.addr
//     readEn   := true.B
//     readSel  := true.B
//   }.otherwise {
//     readAddr := 0.U
//     readEn   := false.B
//     readSel  := false.B
//   }

//   val rdData     = mem.read(readAddr, readEn)
//   val r1_readSel = RegNext(readSel)
//   val r1_readEn  = RegNext(readEn, false.B)

//   io.lsuReadData.valid := r1_readEn && !r1_readSel
//   io.lsuReadData.bits  := rdData

//   io.dmaReadData.valid := r1_readEn && r1_readSel
//   io.dmaReadData.bits  := rdData

//   // ── Write arbitration (LSU > DMA) ───────────────────────────────
//   when(io.lsuWrite.valid) {
//     mem.write(io.lsuWrite.bits.addr, io.lsuWrite.bits.data)
//   }.elsewhen(io.dmaWrite.valid) {
//     mem.write(io.dmaWrite.bits.addr, io.dmaWrite.bits.data)
//   }
// }

package atlas.common

import chisel3._
import chisel3.util._
import freechips.rocketchip.subsystem.{BaseSubsystem,TLBusWrapperLocation, SBUS}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.resources.{DiplomacyUtils}
import org.chipsalliance.cde.config.{Field, Config, Parameters}
import freechips.rocketchip.prci.{ClockSinkDomain, ClockSinkParameters}
import chisel3.Data.DataEquality

// ── ScratchpadMem ────────────────────────────────────────────────────

// class ScratchpadMem(sp: ScratchpadParams, beatBytes: Int)(implicit p: Parameters) extends LazyModule {

//   val xbar = TLXbar()
  
//   (0 until sp.subBanks).map { sb =>
//     val ram = LazyModule(new TLRAM(
//       address = AddressSet(sp.base, sp.sizeBytes - 1),
//       beatBytes = beatBytes
//     ){
//         override lazy val desiredName = s"TLRAM_Scratchpad_$sb"
//     })

//     ram.node := TLFragmenter(beatBytes, sp.lineBytes, nameSuffix = Some("ScratchpadBank")) := xbar // := TLBuffer()
//   }

//   lazy val module = new LazyModuleImp(this) {}

//   override lazy val desiredName = "ScratchpadBank"
// }

// ── Port bundles ─────────────────────────────────────────────────────

class ScratchpadLineReadPort(p: ScratchpadParams) extends Bundle {
  val addr = UInt(p.lineAddrBits.W)
}

class ScratchpadLineWritePort(p: ScratchpadParams) extends Bundle {
  val addr = UInt(p.lineAddrBits.W)
  val data = UInt(p.lineWidth.W)
}

// ── ScratchpadMem ────────────────────────────────────────────────────


// class ScratchpadMem(sp: ScratchpadParams, beatBytes: Int)(implicit p: Parameters) extends LazyModule {

//   val node = TLManagerNode(Seq(TLSlavePortParameters.v1(
//     managers = Seq(TLSlaveParameters.v1(
//       address            = Seq(AddressSet(sp.base, sp.sizeBytes - 1)),
//       regionType         = RegionType.IDEMPOTENT,
//       supportsGet        = TransferSizes(beatBytes, sp.lineBytes),
//       supportsPutFull    = TransferSizes(beatBytes, sp.lineBytes),
//       supportsPutPartial = TransferSizes(beatBytes, sp.lineBytes)
//     )),
//     beatBytes = beatBytes
//   )))

//   override lazy val desiredName = "ScratchpadMem"

//   lazy val module = new Impl
//   class Impl extends LazyRawModuleImp(this) {
//     // slave node for memory access from Saturn Core and DMA, variable latency
//     val (tl, edge) = node.in.head

//     // sram for lsu access, fixed latency
//     val mem = SyncReadMem(sp.numLines, UInt(sp.lineWidth.W))

//     // IO for DMA and LSU
//     val io = IO(new Bundle {
//       val dmaRead     = Flipped(Valid(new ScratchpadLineReadPort(sp)))
//       val dmaReadData = Valid(UInt(sp.lineWidth.W))
//       val dmaWrite    = Flipped(Valid(new ScratchpadLineWritePort(sp)))

//       val lsuRead     = Flipped(Valid(new ScratchpadLineReadPort(sp)))
//       val lsuReadData = Valid(UInt(sp.lineWidth.W))
//       val lsuWrite    = Flipped(Valid(new ScratchpadLineWritePort(sp)))
//     })

//     // ── TL ready - valid  ─────────────────────────────────
//     tl.a.ready := !io.lsuRead.valid && !io.lsuWrite.valid && !io.dmaRead.valid && !io.dmaWrite.valid
    
//     // ── TL address conversion ─────────────────────────────────
//     val tlAddr   = tl.a.bits.address(sp.byteAddrBits - 1, log2Ceil(sp.lineBytes))

//     // ── Read arbitration (LSU > DMA > SBUS) ─────────────────────────────────
//     val readAddr = Wire(UInt(sp.lineAddrBits.W))
//     val readEn   = Wire(Bool())
//     val readSel  = Wire(UInt(2.W))   // 0=None, 1=LSU, 2=DMA, 3=TL

//     val tlReadValid = tl.a.valid && tl.a.bits.opcode === TLMessages.Get && tl.a.ready

//     when(io.lsuRead.valid) {
//       readAddr := io.lsuRead.bits.addr
//       readEn   := true.B
//       readSel  := 1.U
//     }.elsewhen(io.dmaRead.valid) {
//       readAddr := io.dmaRead.bits.addr
//       readEn   := true.B
//       readSel  := 2.U
//     }.elsewhen(tlReadValid) {
//       readAddr := tlAddr
//       readEn   := true.B
//       readSel  := 3.U
//     }.otherwise {
//       readAddr := 0.U
//       readEn   := false.B
//       readSel  := 0.U
//     }

//     val rdData     = mem.read(readAddr, readEn)
//     val r1_readSel = RegNext(readSel)
//     val r1_readEn  = RegNext(readEn, false.B)
//     val r1_tlRdValid = RegNext(tlReadValid && readSel === 3.U, false.B)

//     val r1_tlSource = RegNext(tl.a.bits.source)
//     val r1_tlSize   = RegNext(tl.a.bits.size)

//     io.lsuReadData.valid := r1_readEn && r1_readSel === 1.U
//     io.lsuReadData.bits  := rdData

//     io.dmaReadData.valid := r1_readEn && r1_readSel === 2.U
//     io.dmaReadData.bits  := rdData

//     // ── TODO Write arbitration (LSU > DMA > SBUS) ─────────────────────────────── 

//     val tlWriteValid = tl.a.valid && tl.a.ready && (tl.a.bits.opcode === TLMessages.PutFullData || tl.a.bits.opcode === TLMessages.PutPartialData)
//     val r1_tlWrValid = RegNext(tlWriteValid, false.B)
   

//     when(io.lsuWrite.valid) {
//       mem.write(io.lsuWrite.bits.addr, io.lsuWrite.bits.data)
//     }.elsewhen(io.dmaWrite.valid) {
//       mem.write(io.dmaWrite.bits.addr, io.dmaWrite.bits.data)
//     }.elsewhen(tlWriteValid) {
//       mem.write(tlAddr, tl.a.bits.data)
//     }

//     tl.d.valid := r1_tlRdValid || r1_tlWrValid
//     tl.d.bits  := Mux(
//       r1_tlRdValid,
//       edge.AccessAck(r1_tlSource, r1_tlSize, rdData),
//       edge.AccessAck(r1_tlSource, r1_tlSize)
//     )

//   }
// }

class ScratchpadMem(sp: ScratchpadParams, bundle: TLBundleParameters) extends Module{

    // sram for lsu access, fixed latency
    val mem = SyncReadMem(sp.numLines, UInt(sp.lineWidth.W))

    // IO for DMA and LSU
    val io = IO(new Bundle {
      val tl = Flipped(new TLBundle(bundle))

      val dmaRead     = Flipped(Valid(new ScratchpadLineReadPort(sp)))
      val dmaReadData = Valid(UInt(sp.lineWidth.W))
      val dmaWrite    = Flipped(Valid(new ScratchpadLineWritePort(sp)))

      val lsuRead     = Flipped(Valid(new ScratchpadLineReadPort(sp)))
      val lsuReadData = Valid(UInt(sp.lineWidth.W))
      val lsuWrite    = Flipped(Valid(new ScratchpadLineWritePort(sp)))
    })

    val tl = io.tl

    // ── TL ready - valid  ─────────────────────────────────
    tl.a.ready := !io.lsuRead.valid && !io.lsuWrite.valid && !io.dmaRead.valid && !io.dmaWrite.valid
    
    // ── TL address conversion ─────────────────────────────────
    val tlAddr   = tl.a.bits.address(sp.byteAddrBits - 1, log2Ceil(sp.lineBytes))

    // ── Read arbitration (LSU > DMA > SBUS) ─────────────────────────────────
    val readAddr = Wire(UInt(sp.lineAddrBits.W))
    val readEn   = Wire(Bool())
    val readSel  = Wire(UInt(2.W))   // 0=None, 1=LSU, 2=DMA, 3=TL

    val tlReadValid = tl.a.valid && tl.a.bits.opcode === TLMessages.Get && tl.a.ready

    when(io.lsuRead.valid) {
      readAddr := io.lsuRead.bits.addr
      readEn   := true.B
      readSel  := 1.U
    }.elsewhen(io.dmaRead.valid) {
      readAddr := io.dmaRead.bits.addr
      readEn   := true.B
      readSel  := 2.U
    }.elsewhen(tlReadValid) {
      readAddr := tlAddr
      readEn   := true.B
      readSel  := 3.U
    }.otherwise {
      readAddr := 0.U
      readEn   := false.B
      readSel  := 0.U
    }

    val rdData     = mem.read(readAddr, readEn)
    val r1_readSel = RegNext(readSel)
    val r1_readEn  = RegNext(readEn, false.B)
    val r1_tlRdValid = RegNext(tlReadValid && readSel === 3.U, false.B)

    val r1_tlSource = RegNext(tl.a.bits.source)
    val r1_tlSize   = RegNext(tl.a.bits.size)

    io.lsuReadData.valid := r1_readEn && r1_readSel === 1.U
    io.lsuReadData.bits  := rdData

    io.dmaReadData.valid := r1_readEn && r1_readSel === 2.U
    io.dmaReadData.bits  := rdData

    // ── TODO Write arbitration (LSU > DMA > SBUS) ─────────────────────────────── 

    val tlWriteValid = tl.a.valid && tl.a.ready && (tl.a.bits.opcode === TLMessages.PutFullData || tl.a.bits.opcode === TLMessages.PutPartialData)
    val r1_tlWrValid = RegNext(tlWriteValid, false.B)
   

    when(io.lsuWrite.valid) {
      mem.write(io.lsuWrite.bits.addr, io.lsuWrite.bits.data)
    }.elsewhen(io.dmaWrite.valid) {
      mem.write(io.dmaWrite.bits.addr, io.dmaWrite.bits.data)
    }.elsewhen(tlWriteValid) {
      mem.write(tlAddr, tl.a.bits.data)
    }


    // D channel response
    tl.d.valid        := r1_tlRdValid || r1_tlWrValid
    tl.d.bits.opcode  := Mux(r1_tlRdValid, TLMessages.AccessAckData, TLMessages.AccessAck)
    tl.d.bits.param   := 0.U
    tl.d.bits.size    := r1_tlSize
    tl.d.bits.source  := r1_tlSource
    tl.d.bits.sink    := 0.U
    tl.d.bits.denied  := false.B
    tl.d.bits.data    := Mux(r1_tlRdValid, rdData, 0.U)
    tl.d.bits.corrupt := false.B

  }


// case object ScratchpadKey extends Field[Seq[ScratchpadParams]](Nil)

// object ScratchpadInjector extends SubsystemInjector((p, baseSubsystem) => {
//   implicit val implicitP: Parameters = p

//   p(ScratchpadKey).zipWithIndex.foreach { case (params, si) =>
//     val bus  = baseSubsystem.locateTLBusWrapper(params.busWhere)
//     val name = params.name

//     require(params.subBanks >= 1)

//     (0 until params.banks).map { b =>
//       val bank = LazyModule(new ScratchpadMem(params, bus.beatBytes))
//       // bank.clockNode := bus.fixedClockNode   // drive clock from SBUS
//       bus.coupleTo(s"$name-$si-$b") { bank.node :=  TLBuffer() := _ }
//     }
//   }
// })


