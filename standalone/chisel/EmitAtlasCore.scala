// ============================================================================
// EmitAtlasCore.scala — SystemVerilog for AtlasCore with fixed TL-UL ports.
//
//   ./mill standalone.runMain atlas.standalone.EmitAtlasCore <dir> [--assertions] [--config=NAME]
//
// NAME is one of atlas.common.AtlasConfigs.byName: default (systolic array +
// inner-product trees), ipt (inner-product trees only), ipt-nvfp4 (inner-product
// trees + the NVFP4 MXU).
//
// The standalone SoC (standalone/rtl) talks to these ports directly:
//   imemTL, csrTL  32-bit data, 32-bit address        (managers)
//   vmemTL         256-bit data, 32-bit address       (manager)
//   dmaTL          256-bit data, 64-bit address       (client, one beat per request)
// Without --assertions the Verification layer is dropped (FPGA build); with it
// the assertions are inlined (simulation).
// ============================================================================

package atlas.standalone

import circt.stage.ChiselStage
import freechips.rocketchip.tilelink.TLBundleParameters
import atlas.common.{AtlasConfigs, AtlasParams}
import atlas.tile.AtlasCore

object EmitAtlasCore extends App {
  val dir        = args.headOption.getOrElse("build/rtl")
  val assertions = args.contains("--assertions")
  val config     = args.collectFirst { case a if a.startsWith("--config=") => a.stripPrefix("--config=") }
                       .getOrElse("default")
  val tp: AtlasParams = AtlasConfigs.byName.getOrElse(config,
    sys.error(s"unknown config $config; choose one of ${AtlasConfigs.byName.keys.mkString(", ")}"))

  private def port(addressBits: Int, dataBits: Int, sourceBits: Int) =
    TLBundleParameters(addressBits = addressBits, dataBits = dataBits,
                       sourceBits = sourceBits, sinkBits = 1, sizeBits = 4)

  ChiselStage.emitSystemVerilogFile(
    new AtlasCore(
      tp     = tp,
      imemBP = port(32, 32, 1),
      csrBP  = port(32, 32, 1),
      dmaBP  = port(64, tp.dma.beatBytes * 8, tp.dma.tagBits),
      vmemBP = port(32, tp.vmem.beatBytes * 8, 1)
    ),
    args = Array("--target-dir", dir),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--split-verilog", "-o", dir,
      s"--default-layer-specialization=${if (assertions) "enable" else "disable"}"
    )
  )
}
