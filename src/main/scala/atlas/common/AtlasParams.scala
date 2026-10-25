package atlas.common

/** NVFP4 MXU geometry (see docs/nvfp4-mxu.md).
  *
  * One instruction multiplies a 32 x 64 NVFP4 activation tile (one m-register plus its
  * scale sidecar) by a 64 x 32 weight tile, accumulating into 32 x 32 FP32. Each of the
  * `lanes` fp4_fedp units computes one output column, one activation row per cycle.
  */
case class Nvfp4MxuParams(
  lanes:       Int = 32,  // output columns = fp4_fedp units
  rows:        Int = 32,  // activation rows per tile (m-register rows)
  laneLatency: Int = 5    // fp4_fedp: d is valid 5 cycles after in_vld
) {
  val k:          Int = 64                 // E2M1 elements per m-register row (256 bits)
  val scaleBytes: Int = k / 16             // one UE4M3 scale per 16 elements
  val accBits:    Int = 32                 // FP32 accumulators
}

/** Atlas configuration. MXU slot 1 is the inner-product-tree MXU; with `nvfp4`, slot 0 is
  * the NVFP4 MXU, else slot 0 is empty.
  */
case class AtlasParams(
  ipt:   InnerProductTreeParams  = InnerProductTreeParams(),
  vpu:   VpuParams               = VpuParams(),
  dma:   DmaParams               = DmaParams(),
  mreg:  MregParams              = MregParams(),
  vmem:  VmemParams              = VmemParams(),
  nvfp4: Option[Nvfp4MxuParams]  = None
) {
  def hasNvfp4: Boolean = nvfp4.nonEmpty
}

/** Named configurations. */
object AtlasConfigs {
  /** Inner-product-tree MXU only. */
  val default = AtlasParams()
  /** Inner-product-tree MXU + the NVFP4 MXU. */
  val iptNvfp4 = AtlasParams(nvfp4 = Some(Nvfp4MxuParams()))

  val byName: Map[String, AtlasParams] =
    Map("default" -> default, "ipt-nvfp4" -> iptNvfp4)
}
