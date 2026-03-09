package atlas.common

case class AtlasParams(
  ipt: InnerProductTreeParams = InnerProductTreeParams(),
  sa:  SystolicArrayParams    = SystolicArrayParams(),
  vpu: VPUParams              = VPUParams(),
  dma: DMAParams              = DMAParams()
)
