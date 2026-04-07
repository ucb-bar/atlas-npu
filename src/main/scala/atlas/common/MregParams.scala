// ============================================================================
// MregParams.scala — Parameters and port bundles for the matrix register file.
//
// The spec defines matrix registers as "m registers" (mreg):
//   NUM_MREG       = 64   matrix registers
//   MREG_ROWS      = 32   rows per register
//   MREG_ROW_BYTES = 32   bytes per row  (256 bits = 32 × fp8)
//   MREG_BYTES     = 1024 bytes per register
//
// Multiple functional units (MXU-0, MXU-1, VPU, XLU, LSU/DMA) access the
// register file through dedicated read and write ports.
// ============================================================================

package atlas.common

import chisel3._
import chisel3.util._

/** Design-time parameters for the matrix register file, named to match the
  * architectural spec.
  *
  * @param mregRowBytes  Bytes per row (spec: `MREG_ROW_BYTES`, default 32).
  * @param mregRows      Rows per matrix register (spec: `MREG_ROWS`, default 32).
  * @param numMreg       Number of matrix registers (spec: `NUM_MREG`, default 64).
  * @param numReadPorts  Number of independent read ports.
  * @param numWritePorts Number of independent write ports.
  */
case class MregParams(
    mregRowBytes: Int = 32, // MREG_ROW_BYTES — 32 bytes = 256 bits per row.
    mregRows:     Int = 32, // MREG_ROWS      — 32 rows per matrix register.
    numMreg:      Int = 64, // NUM_MREG       — 64 matrix registers.
    numReadPorts: Int = 8,
    numWritePorts: Int = 6
) {
  /** Bit-width of one SRAM row (MREG_ROW_BYTES × 8). */
  val mregRowBits: Int = mregRowBytes * 8

  /** Bits needed to address a row within a single register. */
  val mregRowAddrBits: Int = log2Ceil(mregRows)

  /** Bits needed to select a matrix register. */
  val mregIdBits: Int = log2Ceil(numMreg)

  /** Total bytes per matrix register (MREG_BYTES). */
  val mregBytes: Int = mregRowBytes * mregRows
}

// ============================================================================
// Port bundles
// ============================================================================

/** Read request: selects one row from one matrix register.
  *
  * Wrapped in `Valid(...)` — when valid is asserted the SRAM performs a
  * synchronous read; the result appears one cycle later.
  */
class MregReadReq(p: MregParams) extends Bundle {
  val mregId = UInt(p.mregIdBits.W)       // Which matrix register to read.
  val row    = UInt(p.mregRowAddrBits.W)  // Which row within that register.
}

/** Write request: addresses one row and carries the write payload. */
class MregWriteReq(p: MregParams) extends Bundle {
  val mregId = UInt(p.mregIdBits.W)       // Target matrix register.
  val row    = UInt(p.mregRowAddrBits.W)  // Target row within the register.
  val data   = UInt(p.mregRowBits.W)      // Data to write.
}
