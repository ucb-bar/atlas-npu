package atlas.common

import chisel3.util.log2Ceil

case class ScratchpadParams(
  lineWidth:   Int = 256,             // bits per line (matches TRF row width)
  sizeBytes:   Int = 256 * 1024       // 256 KB default
) {
  val lineBytes:    Int = lineWidth / 8            // 32
  val numLines:     Int = sizeBytes / lineBytes    // 8192
  val lineAddrBits: Int = log2Ceil(numLines)       // 13
  val wordWidth:    Int = 32
  val wordsPerLine: Int = lineWidth / wordWidth    // 8
  val wordOffBits:  Int = log2Ceil(wordsPerLine)   // 3
  val numWords:     Int = numLines * wordsPerLine  // 65536
  val wordAddrBits: Int = log2Ceil(numWords)       // 16
  val byteAddrBits: Int = log2Ceil(sizeBytes)      // 18
}