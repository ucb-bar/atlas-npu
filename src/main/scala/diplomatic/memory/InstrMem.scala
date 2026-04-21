/*
InstrMem.scala — SRAM-based instruction memory.
Single-bank, 32-bit wide, sized by `AtlasMemMap.IMEM_WORDS`
(default Atlas map: 32768 words = 128 KiB).
TileLink manager port for program loading. Fetch port for pipeline.

Read-port arbitration: the SRAM has one read and one write port (1R1W).
Pipeline fetch has priority over TileLink Get, so runtime host reads are
backpressured until fetch is inactive and can never stall the accelerator.
*/

package atlas.scalar

import chisel3._
import chisel3.util._
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}

/** SRAM-backed instruction memory for the IMEM address window
  * `[AtlasMemMap.IMEM_BASE, AtlasMemMap.IMEM_BASE + AtlasMemMap.IMEM_SIZE)`,
  * with TileLink load and scalar fetch ports.
  *
  * @param tlBundleParams  TileLink bundle parameters for the program-load port.
  */
class InstrMem(tlBundleParams: TLBundleParameters) extends Module {
  val io = IO(new Bundle {
    val fetch       = Flipped(new ImemFetchPort)
    val fetchActive = Input(Bool())
    val tl          = Flipped(new TLBundle(tlBundleParams))
  })

  val mem = SyncReadMem(AtlasMemMap.IMEM_WORDS, UInt(32.W))

  // ── TileLink state machine ────────────────────────────────────────────
  val sIdle :: sResp :: Nil = Enum(2)
  val state = RegInit(sIdle)
  val respIsGet  = Reg(Bool())
  val respSource = Reg(UInt(tlBundleParams.sourceBits.W))
  val respSize   = Reg(UInt(tlBundleParams.sizeBits.W))

  private def tlToWordAddr(addr: UInt): UInt =
    (addr - AtlasMemMap.IMEM_BASE.U(tlBundleParams.addressBits.W))(
      AtlasMemMap.IMEM_ADDR_BITS + 1, 2)

  // ── Read-port arbitration ─────────────────────────────────────────────
  // Fetch owns the single read port while the core is running. TileLink Get
  // is only accepted once fetch goes inactive (e.g. core halted).
  val tlGetFire = io.tl.a.fire && io.tl.a.bits.opcode === TLMessages.Get
  val tlWordAddr = tlToWordAddr(io.tl.a.bits.address)

  val readAddr = Mux(io.fetchActive, io.fetch.addr, tlWordAddr)
  val readEn   = io.fetchActive || tlGetFire
  val readData = mem.read(readAddr, readEn)

  // Fetch is never displaced by TileLink Get, so its data is always valid
  // whenever the core is actively fetching.
  io.fetch.rdata := readData

  // ── TileLink write path (uses the dedicated write port) ───────────────
  // Writes go through the SRAM write port, so they never conflict with reads.

  // ── TileLink A channel ────────────────────────────────────────────────
  val tlGetBlocked = io.fetchActive && io.tl.a.valid &&
    io.tl.a.bits.opcode === TLMessages.Get
  io.tl.a.ready := (state === sIdle) && !tlGetBlocked

  // ── TileLink D channel ────────────────────────────────────────────────
  io.tl.d.valid       := (state === sResp)
  io.tl.d.bits.opcode := Mux(respIsGet, TLMessages.AccessAckData, TLMessages.AccessAck)
  io.tl.d.bits.param  := 0.U
  io.tl.d.bits.size   := respSize
  io.tl.d.bits.source := respSource
  io.tl.d.bits.sink   := 0.U
  io.tl.d.bits.denied := false.B
  io.tl.d.bits.data   := Mux(respIsGet, readData, 0.U)
  io.tl.d.bits.corrupt := false.B

  // ── State machine ────────────────────────────────────────────────────
  switch(state) {
    is(sIdle) {
      when(io.tl.a.fire) {
        val a = io.tl.a.bits
        respSource := a.source
        respSize   := a.size
        when(a.opcode === TLMessages.Get) {
          respIsGet := true.B
        }.otherwise {
          mem.write(tlToWordAddr(a.address), a.data)
          respIsGet := false.B
        }
        state := sResp
      }
    }
    is(sResp) {
      when(io.tl.d.fire) { state := sIdle }
    }
  }
}
