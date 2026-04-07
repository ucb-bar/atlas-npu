// ============================================================================
// Vmem.scala — VMEM parameters and line-level port bundles.
//
// VMEM is the on-chip vector scratchpad memory.  It is addressed at
// line granularity for bulk transfers (DMA, tensor moves) and at byte
// granularity for scalar loads/stores (via LSU read-modify-write).
//
// Spec reference:
//   VMEM_BASE          = 0x2000_0000
//   VMEM_SIZE          = 1 MiB           (capacity; default here is 256 KiB)
//   VMEM_TENSOR_ALIGN  = 32 bytes        (= one line)
//   DMA_ALIGN          = 32 bytes        (= one line)
//   MREG_ROW_BYTES     = 32 bytes        (line width matches mreg row width)
// ============================================================================

package atlas.common

import chisel3._
import chisel3.util._

/** Design-time parameters for the vector scratchpad memory.
  *
  * A VMEM "line" is the atomic unit of bulk access.  Its width matches
  * the mreg row width (MREG_ROW_BYTES × 8 = 256 bits = 32 bytes) so
  * that one line read/write moves exactly one tensor-register row.
  *
  * @param lineWidthBits  Bits per line (default 256 = MREG_ROW_BYTES × 8).
  * @param capacityBytes  Total VMEM capacity in bytes (default 256 KiB).
  * @param base           TileLink base address visible on the system bus.
  * @param beatBytes      TileLink beat width in bytes for the slave port.
  */
case class VmemParams(
  lineWidthBits: Int    = 256,
  capacityBytes: Int    = 256 * 1024,
  base:          BigInt = 0x2000_0000L,
  beatBytes:     Int    = 32
) {
  /** Bytes per line. */
  val lineBytes: Int = lineWidthBits / 8               // 32.

  /** Total number of lines in VMEM. */
  val numLines: Int = capacityBytes / lineBytes         // 8192.

  /** Bits needed to address a line. */
  val lineAddrBits: Int = log2Ceil(numLines)            // 13.

  /** Scalar word width in bits. */
  val wordWidth: Int = 32

  /** 32-bit words per line. */
  val wordsPerLine: Int = lineWidthBits / wordWidth     // 8.

  /** Bits needed to select a word within a line. */
  val wordOffBits: Int = log2Ceil(wordsPerLine)         // 3.

  /** Total number of 32-bit words in VMEM. */
  val numWords: Int = numLines * wordsPerLine           // 65536.

  /** Bits needed to address a word. */
  val wordAddrBits: Int = log2Ceil(numWords)            // 16.

  /** Bits needed for a byte address across all of VMEM. */
  val byteAddrBits: Int = log2Ceil(capacityBytes)       // 18.

  // ── Backward-compatible aliases ──
  def lineWidth: Int = lineWidthBits
  def sizeBytes: Int = capacityBytes
}

// ============================================================================
// Port bundles — line-granularity read and write
// ============================================================================

/** Line-level read request: addresses one VMEM line.
  *
  * Wrapped in `Valid(...)` — when valid is asserted, the VMEM performs
  * a synchronous read and the data appears one cycle later.
  */
class VmemLineReadPort(p: VmemParams) extends Bundle {
  val addr = UInt(p.lineAddrBits.W)   // Line address within VMEM.
}

/** Line-level write request: addresses one line and carries the full payload.
  *
  * All writes are full-line (no byte-masking).  The LSU performs
  * read-modify-write for sub-line scalar stores.
  */
class VmemLineWritePort(p: VmemParams) extends Bundle {
  val addr = UInt(p.lineAddrBits.W)   // Target line address.
  val data = UInt(p.lineWidthBits.W)  // Full line payload (256 bits).
}
