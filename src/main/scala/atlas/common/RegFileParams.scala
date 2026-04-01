package atlas.common

import chisel3.util.log2Ceil

case class RegFileParams(
  SRAM_WIDTH:       Int = 256,  // 32 fp8 values per row
  SRAM_DEPTH:       Int = 32,   // 32 rows per tensor register
  NUM_BANKS:        Int = 64,   // 64 tensor registers
  NUM_READ_PORTS:   Int = 8,
  NUM_WRITE_PORTS:  Int = 6
) {
  val SRAM_DEPTH_BITS: Int = log2Ceil(SRAM_DEPTH)
  val NUM_BANKS_BITS:  Int = log2Ceil(NUM_BANKS)
}
