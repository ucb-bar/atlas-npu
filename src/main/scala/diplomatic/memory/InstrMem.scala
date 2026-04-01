/*
InstrMem.scala — SRAM-based instruction memory.
Single-bank, 32-bit wide, 16384 deep (64 KB).
TileLink manager port for program loading. Fetch port for pipeline.
*/

package atlas.scalar

import chisel3._
import chisel3.util._
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}

class InstrMem(tlBundleParams: TLBundleParameters) extends Module {
  val io = IO(new Bundle {
    val fetch = Flipped(new ImemFetchPort)
    val tl    = Flipped(new TLBundle(tlBundleParams))
  })

  val mem = SyncReadMem(AtlasMemMap.IMEM_WORDS, UInt(32.W))

  io.fetch.rdata := mem.read(io.fetch.addr)

  val sIdle :: sResp :: Nil = Enum(2)
  val state = RegInit(sIdle)
  val respIsGet  = Reg(Bool())
  val respSource = Reg(UInt(tlBundleParams.sourceBits.W))
  val respSize   = Reg(UInt(tlBundleParams.sizeBits.W))

  private def tlToWordAddr(addr: UInt): UInt =
    (addr - AtlasMemMap.IMEM_BASE.U(tlBundleParams.addressBits.W))(
      AtlasMemMap.IMEM_ADDR_BITS + 1, 2)

  val tlReadAddr = Wire(UInt(AtlasMemMap.IMEM_ADDR_BITS.W))
  val tlReadEn   = Wire(Bool())
  tlReadAddr := 0.U; tlReadEn := false.B
  val tlReadData = mem.read(tlReadAddr, tlReadEn)

  io.tl.a.ready := (state === sIdle)

  io.tl.d.valid       := (state === sResp)
  io.tl.d.bits.opcode := Mux(respIsGet, TLMessages.AccessAckData, TLMessages.AccessAck)
  io.tl.d.bits.param  := 0.U
  io.tl.d.bits.size   := respSize
  io.tl.d.bits.source := respSource
  io.tl.d.bits.sink   := 0.U
  io.tl.d.bits.denied := false.B
  io.tl.d.bits.data   := Mux(respIsGet, tlReadData, 0.U)
  io.tl.d.bits.corrupt := false.B

  switch(state) {
    is(sIdle) {
      when(io.tl.a.fire) {
        val a = io.tl.a.bits
        respSource := a.source; respSize := a.size
        val wordAddr = tlToWordAddr(a.address)
        when(a.opcode === TLMessages.Get) {
          tlReadAddr := wordAddr; tlReadEn := true.B; respIsGet := true.B
        }.otherwise {
          mem.write(wordAddr, a.data); respIsGet := false.B
        }
        state := sResp
      }
    }
    is(sResp) { when(io.tl.d.fire) { state := sIdle } }
  }
}
