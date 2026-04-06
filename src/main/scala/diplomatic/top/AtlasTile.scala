/*
AtlasTile.scala — Diplomatic TileLink wrapper for AtlasCore.
*/

package atlas.tile

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.{Field, Config, Parameters}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.subsystem.{BaseSubsystem,TLBusWrapperLocation, SBUS}
import atlas.common.AtlasParams
import testchipip.soc.{SubsystemInjectorKey, SubsystemInjector}

import atlas.scalar.{AtlasMemMap, ScalarISA}

class AtlasTile(
  tp: AtlasParams = AtlasParams()
)(implicit p: Parameters) extends LazyModule {

  private val dmaP = tp.dma
  private val sp  = tp.scratchpad

  // ── Manager nodes (IMEM, CSR) ─────────────────────────────────────
  val imemNode = TLManagerNode(Seq(TLSlavePortParameters.v1(
    managers = Seq(TLSlaveParameters.v1(
      address            = Seq(AddressSet(AtlasMemMap.IMEM_BASE, AtlasMemMap.IMEM_SIZE - 1)),
      regionType         = RegionType.IDEMPOTENT,
      supportsGet        = TransferSizes(1, 4),
      supportsPutFull    = TransferSizes(1, 4),
      supportsPutPartial = TransferSizes(1, 4),
      fifoId             = Some(0)
    )),
    beatBytes = 4
  )))

  val csrNode = TLManagerNode(Seq(TLSlavePortParameters.v1(
    managers = Seq(TLSlaveParameters.v1(
      address            = Seq(AddressSet(AtlasMemMap.CSR_BASE, 0xFFF)),
      regionType         = RegionType.IDEMPOTENT,
      supportsGet        = TransferSizes(1, 4),
      supportsPutFull    = TransferSizes(1, 4),
      supportsPutPartial = TransferSizes(1, 4),
      fifoId             = Some(0)
    )),
    beatBytes = 4
  )))

  val vmemNode = TLManagerNode(Seq(TLSlavePortParameters.v1(
    managers = Seq(TLSlaveParameters.v1(
      address            = Seq(AddressSet(sp.base, sp.sizeBytes - 1)),
      regionType         = RegionType.IDEMPOTENT,
      supportsGet        = TransferSizes(1, sp.beatBytes),
      supportsPutFull    = TransferSizes(1, sp.beatBytes),
      supportsPutPartial = TransferSizes(1, sp.beatBytes),
      fifoId             = Some(0)
    )),
    beatBytes = sp.beatBytes
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
  dmaWidthAdapter.node := dmaClientNode

  private val dmaBuffer = LazyModule(
    new TLBuffer())
  dmaBuffer.node := dmaWidthAdapter.node

  /** External-facing DMA node.  Connect this to the system bus. */
  val dmaNode: TLOutwardNode = dmaBuffer.node

  // ── Module implementation ──────────────────────────────────────
  lazy val module = new LazyModuleImp(this) {
    val (imemBundle, imemEdge) = imemNode.in.head
    val (csrBundle,  csrEdge)  = csrNode.in.head
    val (vmemBundle, vmemEdge)  = vmemNode.in.head

    // dmaClientNode gives us the 32-byte-wide bundle
    val (dmaBundle,  dmaEdge)  = dmaClientNode.out.head

    val core = Module(new AtlasCore(
      tp     = tp,
      imemBP = imemEdge.bundle,
      csrBP  = csrEdge.bundle,
      dmaBP  = dmaEdge.bundle, 
      vmemBP = vmemEdge.bundle
    ))

    core.io.imemTL <> imemBundle
    core.io.csrTL  <> csrBundle
    dmaBundle      <> core.io.dmaTL
    vmemBundle     <> core.io.vmemTL

    val halted = IO(Output(Bool()))
    halted := core.io.halted

    val dbg = IO(Output(chiselTypeOf(core.io.dbg)))
    dbg := core.io.dbg
  }
}


case class AtlasTileParams(
  base: BigInt,
  size: BigInt,
  busWhere: TLBusWrapperLocation = SBUS,
  name: String = "atlas-dmaedge",
  disableMonitors: Boolean = false,
  atlasParams: AtlasParams = AtlasParams(),
)

case object AtlasTileKey extends Field[Seq[AtlasTileParams]](Nil)

case object AtlasTileInjector extends SubsystemInjector((p, baseSubsystem) => {
  implicit val implicitP: Parameters = p

  p(AtlasTileKey).zipWithIndex.foreach { case (params, si) =>
    val bus = baseSubsystem.locateTLBusWrapper(params.busWhere)
    val dmaRange = AddressSet(params.base, params.size - 1)

    val sp = params.atlasParams.scratchpad

    val atlasDomain = bus.generateSynchronousDomain(s"atlas_domain_$si")
    val atlasTile = atlasDomain { LazyModule(new AtlasTile(params.atlasParams)) }

    // connect DMA, IMEM, and CSR to SBUS
    def connect(): Unit = {
      bus.coupleFrom(s"${params.name}-$si-dma") {
        _ := TLBuffer() := atlasTile.dmaNode
      }

      bus.coupleTo(s"${params.name}-$si-imem") {
        atlasTile.imemNode :=
          TLFragmenter(4, bus.beatBytes) :=
          TLBuffer() :=
          TLWidthWidget(bus.beatBytes) := _
      }

      bus.coupleTo(s"${params.name}-$si-csr") {
        atlasTile.csrNode :=
          TLFragmenter(4, bus.beatBytes) :=
          TLBuffer() :=
          TLWidthWidget(bus.beatBytes) := _
      }

      bus.coupleTo(s"${params.name}-$si-spad") {
        atlasTile.vmemNode :=
          TLFragmenter(sp.beatBytes, bus.beatBytes) :=
          TLBuffer() :=
          TLWidthWidget(bus.beatBytes) := _
      }

    }

    if (params.disableMonitors)
      DisableMonitors({ implicit p => connect() })(p)
    else
      connect()
  }
})
