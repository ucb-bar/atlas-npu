// ============================================================================
// VmemParams.scala — VMEM parameters and port bundles (architecture).
//
// Atlas uses a block-banked VMEM. Each bank owns one contiguous `bankBytes`
// range, and the default Atlas memory map places this window at
// `AtlasMemMap.VMEM_BASE` with size `AtlasMemMap.VMEM_SIZE`.
//
// Banking on high bits of line address:
//   lineAddr[lineAddrBits-1 : bankLineAddrBits]  → bank index
//   lineAddr[bankLineAddrBits-1 : 0]             → line within bank
// ============================================================================

package atlas.common

import chisel3._
import chisel3.util._
import atlas.scalar.AtlasMemMap

case class VmemParams(
  lineWidthBits: Int    = 256,
  capacityBytes: Int    = AtlasMemMap.VMEM_SIZE,
  base:          BigInt = BigInt(AtlasMemMap.VMEM_BASE),
  beatBytes:     Int    = 32,
  numBanks:      Int    = 8
) {
  val lineBytes: Int        = lineWidthBits / 8
  val numLines: Int         = capacityBytes / lineBytes
  val lineAddrBits: Int     = log2Ceil(numLines)
  val wordWidth: Int        = 32
  val wordsPerLine: Int     = lineWidthBits / wordWidth
  val wordOffBits: Int      = log2Ceil(wordsPerLine)
  val numWords: Int         = numLines * wordsPerLine
  val wordAddrBits: Int     = log2Ceil(numWords)
  val byteAddrBits: Int     = log2Ceil(capacityBytes)

  // ── Banking ──
  val bankIdBits: Int       = log2Ceil(numBanks)
  val bankBytes: Int        = capacityBytes / numBanks
  val linesPerBank: Int     = numLines / numBanks
  val bankLineAddrBits: Int = log2Ceil(linesPerBank)
  val lineOffBits: Int      = log2Ceil(lineBytes)

  // ── Aliases ──
  def lineWidth: Int = lineWidthBits
  def sizeBytes: Int = capacityBytes

  private def isPowerOfTwo(x: Int): Boolean = x > 0 && (x & (x - 1)) == 0

  require(lineWidthBits % 8 == 0)
  require(capacityBytes % lineBytes == 0)
  require(isPowerOfTwo(capacityBytes))
  require(isPowerOfTwo(numBanks))
  require(numLines % numBanks == 0)
  require(lineAddrBits == bankLineAddrBits + bankIdBits)

  /** Extract bank index from a line address. */
  def getBankIdx(lineAddr: UInt): UInt = lineAddr(lineAddrBits - 1, bankLineAddrBits)
  /** Extract bank-local line address from a line address. */
  def getBankAddr(lineAddr: UInt): UInt = lineAddr(bankLineAddrBits - 1, 0)
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
