// ============================================================================
// AtlasTile.scala — Diplomatic TileLink wrapper for AtlasCore.
//
// Wraps the non-diplomatic AtlasCore with four TileLink nodes:
//   • imemNode  (manager / slave)  – Host writes instructions here.
//   • csrNode   (manager / slave)  – Host reads/writes control registers.
//   • vmemNode  (manager / slave)  – Host direct access to VMEM.
//   • dmaNode   (client  / master) – DMA engine issues loads & stores.
//
// A TLWidthWidget inside the tile adapts the DMA port from the engine's
// native beat width (default 32 bytes = 256 bits) down to whatever the
// system bus negotiates.
// ============================================================================

package atlas.tile

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.{Field, Config, Parameters}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.subsystem.{BaseSubsystem, TLBusWrapperLocation, SBUS, PBUS}
import atlas.common.AtlasParams
import testchipip.soc.{SubsystemInjectorKey, SubsystemInjector}

import atlas.scalar.{AtlasMemMap, ScalarISA}

private object AtlasTile {
  def contiguousAddressSets(base: BigInt, size: BigInt): Seq[AddressSet] = {
    require(size > 0, s"VMEM size must be positive, got $size")

    val sets = scala.collection.mutable.ArrayBuffer.empty[AddressSet]
    var nextBase = base
    var remaining = size

    while (remaining > 0) {
      var chunk = BigInt(1) << (remaining.bitLength - 1)
      while ((nextBase & (chunk - 1)) != 0) {
        chunk = chunk >> 1
      }
      sets += AddressSet(nextBase, chunk - 1)
      nextBase += chunk
      remaining -= chunk
    }

    sets.toSeq
  }
}

/** Diplomatic shell around [[AtlasCore]].
  *
  * @param tp  Full set of Atlas design parameters.
  * @param p   Implicit Rocket-Chip configuration parameters.
  */
class AtlasTile(
    tp: AtlasParams = AtlasParams()
)(implicit p: Parameters) extends LazyModule {

  private val dmaP  = tp.dma
  private val vmemP = tp.vmem

  // ==========================================================================
  // Manager (slave) nodes — IMEM, CSR, VMEM
  // ==========================================================================

  /** Instruction-memory port.  The host writes the program here before launch. */
  val imemNode = TLManagerNode(Seq(TLSlavePortParameters.v1(
    managers = Seq(TLSlaveParameters.v1(
      address            = Seq(AddressSet(AtlasMemMap.IMEM_BASE,
                                          AtlasMemMap.IMEM_SIZE - 1)),
      regionType         = RegionType.IDEMPOTENT,
      supportsGet        = TransferSizes(1, 4),
      supportsPutFull    = TransferSizes(1, 4),
      supportsPutPartial = TransferSizes(1, 4),
      fifoId             = Some(0)
    )),
    beatBytes = 4
  )))

  /** Control/status register port.  The host uses this to start/stop the core
    * and to read back status and debug information.
    */
  val csrNode = TLManagerNode(Seq(TLSlavePortParameters.v1(
    managers = Seq(TLSlaveParameters.v1(
      address            = Seq(AddressSet(AtlasMemMap.CSR_BASE,
                                          AtlasMemMap.CSR_WINDOW_SIZE - 1)),
      regionType         = RegionType.IDEMPOTENT,
      supportsGet        = TransferSizes(1, 4),
      supportsPutFull    = TransferSizes(1, 4),
      supportsPutPartial = TransferSizes(1, 4),
      fifoId             = Some(0)
    )),
    beatBytes = 4
  )))

  /** VMEM direct-access port.  The host can read/write VMEM over the bus. */
  val vmemNode = TLManagerNode(Seq(TLSlavePortParameters.v1(
    managers = Seq(TLSlaveParameters.v1(
      address            = AtlasTile.contiguousAddressSets(vmemP.base, BigInt(vmemP.sizeBytes)),
      regionType         = RegionType.IDEMPOTENT,
      supportsGet        = TransferSizes(1, vmemP.beatBytes),
      supportsPutFull    = TransferSizes(1, vmemP.beatBytes),
      supportsPutPartial = TransferSizes(1, vmemP.beatBytes),
      fifoId             = Some(0)
    )),
    beatBytes = vmemP.beatBytes
  )))

  // ── DMA master node with internal width adaptation ──────────────
  // The DMA module operates at `beatBytes` (default 32 = 256 bits)
  // per beat.  A TLWidthWidget inside the tile adapts this to
  // whatever beat width the system bus negotiates.
  //
  // Diplomatic graph inside the tile:
  //   dmaClientNode (32-byte beats) → dmaWidthAdapter → dmaNode (bus beats)

  /** Internal DMA client at the engine's native beat width. */
  private val dmaClientNode = TLClientNode(Seq(TLMasterPortParameters.v1(
    clients = Seq(TLMasterParameters.v1(
      name     = "atlas-dma",
      sourceId = IdRange(0, dmaP.maxInFlight)
    ))
  )))

  /** Width adapter: converts native DMA beats to bus-width beats. */
  private val dmaWidthAdapter = LazyModule(
    new TLWidthWidget(dmaP.beatBytes))
  dmaWidthAdapter.node := dmaClientNode

  private val dmaBuffer = LazyModule(
    new TLBuffer())
  dmaBuffer.node := dmaWidthAdapter.node

  /** External-facing DMA node.  Connect this to the system bus. */
  val dmaNode: TLOutwardNode = dmaBuffer.node

  // ==========================================================================
  // Module implementation
  // ==========================================================================

  lazy val module = new LazyModuleImp(this) {

    // Resolve diplomatic bundles.
    val (imemBundle, imemEdge) = imemNode.in.head
    val (csrBundle,  csrEdge)  = csrNode.in.head
    val (vmemBundle, vmemEdge) = vmemNode.in.head
    val (dmaBundle,  dmaEdge)  = dmaClientNode.out.head

    // Instantiate the non-diplomatic core.
    val core = Module(new AtlasCore(
      tp     = tp,
      imemBP = imemEdge.bundle,
      csrBP  = csrEdge.bundle,
      dmaBP  = dmaEdge.bundle,
      vmemBP = vmemEdge.bundle
    ))

    // Connect TileLink ports.
    core.io.imemTL <> imemBundle
    core.io.csrTL  <> csrBundle
    dmaBundle      <> core.io.dmaTL
    vmemBundle     <> core.io.vmemTL

    // Expose core status to the SoC level.
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
  name: String = "atlas",
  disableMonitors: Boolean = false,
  atlasParams: AtlasParams = AtlasParams(),
)

case object AtlasTileKey extends Field[Seq[AtlasTileParams]](Nil)

case object AtlasTileInjector extends SubsystemInjector((p, baseSubsystem) => {
  implicit val implicitP: Parameters = p

  p(AtlasTileKey).zipWithIndex.foreach { case (params, si) =>
    val bus = baseSubsystem.locateTLBusWrapper(params.busWhere)
    val pbus = baseSubsystem.locateTLBusWrapper(PBUS)
    val dmaRange = AddressSet(params.base, params.size - 1)

    val vmemP = params.atlasParams.vmem

    val atlasDomain = bus.generateSynchronousDomain(s"atlas_domain_$si")
    val atlasTile = atlasDomain { LazyModule(new AtlasTile(params.atlasParams)) }
    val csrMaxTransferBytes = math.min(pbus.blockBytes, AtlasMemMap.CSR_WINDOW_SIZE)

    // connect DMA, IMEM, CSR, and VMEM to SBUS
    def connect(): Unit = {
      bus.coupleFrom(s"${params.name}-$si-dma") {
        _ := TLBuffer() := atlasTile.dmaNode
      }

      pbus.coupleTo(s"${params.name}-$si-imem") {
        atlasTile.imemNode :=
          TLFragmenter(4, pbus.blockBytes) :=
          TLWidthWidget(pbus.beatBytes) :=
          TLBuffer() := _
      }

      pbus.coupleTo(s"${params.name}-$si-csr") {
        atlasTile.csrNode :=
          TLFragmenter(4, csrMaxTransferBytes) :=
          TLWidthWidget(pbus.beatBytes) :=
          TLBuffer() := _
      }

      bus.coupleTo(s"${params.name}-$si-vmem") {
        atlasTile.vmemNode :=
          TLFragmenter(vmemP.beatBytes, bus.beatBytes) :=
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
