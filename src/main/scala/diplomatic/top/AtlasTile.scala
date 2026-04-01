/*
AtlasTile.scala — Diplomatic TileLink wrapper for AtlasCore.
*/

package atlas.tile

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._

import atlas.common.AtlasParams
import atlas.scalar.{AtlasMemMap, ScalarISA}

class AtlasTile(
  tp: AtlasParams = AtlasParams()
)(implicit p: Parameters) extends LazyModule {

  private val dmaP = tp.dma

  // ── Manager nodes (IMEM, CSR) ─────────────────────────────────────
  val imemNode = TLManagerNode(Seq(TLSlavePortParameters.v1(
    managers = Seq(TLSlaveParameters.v1(
      address            = Seq(AddressSet(AtlasMemMap.IMEM_BASE, AtlasMemMap.IMEM_SIZE - 1)),
      regionType         = RegionType.IDEMPOTENT,
      supportsGet        = TransferSizes(4, 4),
      supportsPutFull    = TransferSizes(4, 4),
      supportsPutPartial = TransferSizes(4, 4)
    )),
    beatBytes = 4
  )))

  val csrNode = TLManagerNode(Seq(TLSlavePortParameters.v1(
    managers = Seq(TLSlaveParameters.v1(
      address            = Seq(AddressSet(AtlasMemMap.CSR_BASE, AtlasMemMap.CSR_SIZE - 1)),
      regionType         = RegionType.IDEMPOTENT,
      supportsGet        = TransferSizes(4, 4),
      supportsPutFull    = TransferSizes(4, 4),
      supportsPutPartial = TransferSizes(4, 4)
    )),
    beatBytes = 4
  )))

  // ── DMA master node with internal width adaptation ──────────────
  // The DMA module operates at `widthBytes` (default 32 = 256 bits)
  // per beat.  A TLWidthWidget inside the tile adapts this to
  // whatever beat width the system bus negotiates.
  //
  // Diplomatic graph inside the tile:
  //   dmaClientNode (32-byte beats) → dmaWidthAdapter → dmaNode (bus beats)
  //
  private val dmaClientNode = TLClientNode(Seq(TLMasterPortParameters.v1(
    clients = Seq(TLMasterParameters.v1(
      name     = "atlas-dma",
      sourceId = IdRange(0, dmaP.maxInFlight)
    ))
  )))

  private val dmaWidthAdapter = LazyModule(
    new TLWidthWidget(dmaP.widthBytes))

  /** External-facing DMA node.  Connect this to the system bus. */
  val dmaNode: TLOutwardNode = dmaWidthAdapter.node

  // Wire internal client through the adapter
  dmaWidthAdapter.node := dmaClientNode

  // ── Module implementation ──────────────────────────────────────
  lazy val module = new LazyModuleImp(this) {
    val (imemBundle, imemEdge) = imemNode.in.head
    val (csrBundle,  csrEdge)  = csrNode.in.head

    // dmaClientNode gives us the 32-byte-wide bundle
    val (dmaBundle,  dmaEdge)  = dmaClientNode.out.head

    val core = Module(new AtlasCore(
      tp     = tp,
      imemBP = imemEdge.bundle,
      csrBP  = csrEdge.bundle,
      dmaBP  = dmaEdge.bundle
    ))

    core.io.imemTL <> imemBundle
    core.io.csrTL  <> csrBundle
    dmaBundle      <> core.io.dmaTL

    val halted = IO(Output(Bool()))
    halted := core.io.halted

    val dbg = IO(Output(chiselTypeOf(core.io.dbg)))
    dbg := core.io.dbg
  }
}
