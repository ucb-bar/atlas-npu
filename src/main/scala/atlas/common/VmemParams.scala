// ============================================================================
// VmemParams.scala — VMEM parameters and port bundles (architecture).
//
// 8-bank interleaved VMEM.  Banking on low bits of line address:
//   lineAddr[bankIdBits-1 : 0]             → bank index
//   lineAddr[lineAddrBits-1 : bankIdBits]  → line within bank
// ============================================================================

package atlas.common

import chisel3._
import chisel3.util._

case class VmemParams(
  lineWidthBits: Int    = 256,
  capacityBytes: Int    = 256 * 1024,
  base:          BigInt = 0x2000_0000L,
  beatBytes:     Int    = 32,
  numBanks:      Int    = 8
) {
  val lineBytes: Int        = lineWidthBits / 8               // 32
  val numLines: Int         = capacityBytes / lineBytes       // 8192
  val lineAddrBits: Int     = log2Ceil(numLines)              // 13
  val wordWidth: Int        = 32
  val wordsPerLine: Int     = lineWidthBits / wordWidth       // 8
  val wordOffBits: Int      = log2Ceil(wordsPerLine)          // 3
  val numWords: Int         = numLines * wordsPerLine         // 65536
  val wordAddrBits: Int     = log2Ceil(numWords)              // 16
  val byteAddrBits: Int     = log2Ceil(capacityBytes)         // 18

  // ── Banking ──
  val bankIdBits: Int       = log2Ceil(numBanks)              // 3
  val linesPerBank: Int     = numLines / numBanks             // 1024
  val bankLineAddrBits: Int = log2Ceil(linesPerBank)          // 10
  val lineOffBits: Int      = log2Ceil(lineBytes)             // 5

  // ── Aliases ──
  def lineWidth: Int = lineWidthBits
  def sizeBytes: Int = capacityBytes

  require(numLines % numBanks == 0)

  /** Extract bank index from a line address. */
  def getBankIdx(lineAddr: UInt): UInt = lineAddr(bankIdBits - 1, 0)
  /** Extract bank-local line address from a line address. */
  def getBankAddr(lineAddr: UInt): UInt = lineAddr(lineAddrBits - 1, bankIdBits)
}

// ============================================================================
// Port bundles
// ============================================================================

/** Read request with pre-decomposed bank address. */
class VmemLineReadPort(p: VmemParams) extends Bundle {
  val bankIdx  = UInt(p.bankIdBits.W)
  val bankAddr = UInt(p.bankLineAddrBits.W)
}

/** Full-line write (no masking). Used by DMA and VSTORE. */
class VmemLineWritePort(p: VmemParams) extends Bundle {
  val bankIdx  = UInt(p.bankIdBits.W)
  val bankAddr = UInt(p.bankLineAddrBits.W)
  val data     = UInt(p.lineWidthBits.W)
}

/** Byte-masked write. Used by LSU scalar stores. */
class MaskedVmemLineWritePort(p: VmemParams) extends Bundle {
  val bankIdx  = UInt(p.bankIdBits.W)
  val bankAddr = UInt(p.bankLineAddrBits.W)
  val data     = UInt(p.lineWidthBits.W)
  val mask     = Vec(p.lineBytes, Bool())
}
