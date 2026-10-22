// ============================================================================
// TileLink.scala — stand-in for the rocket-chip TileLink types AtlasCore uses.
//
// AtlasCore and its memories (InstrMem, CSRFile, Vmem, DmaEngine) touch only
// TLBundle / TLBundleParameters / TLMessages, and only TL-UL channels A and D.
// This file provides those three names with rocket-chip's field names, widths
// and opcode values, so the standalone build elaborates AtlasCore without
// rocket-chip. The Chipyard build never sees this file.
// ============================================================================

package freechips.rocketchip.tilelink

import chisel3._
import chisel3.util._

case class TLBundleParameters(
  addressBits: Int,
  dataBits:    Int,
  sourceBits:  Int,
  sinkBits:    Int,
  sizeBits:    Int
)

object TLMessages {
  def PutFullData    = 0.U(3.W)
  def PutPartialData = 1.U(3.W)
  def ArithmeticData = 2.U(3.W)
  def LogicalData    = 3.U(3.W)
  def Get            = 4.U(3.W)
  def Hint           = 5.U(3.W)
  def AccessAck      = 0.U(3.W)
  def AccessAckData  = 1.U(3.W)
  def HintAck        = 2.U(3.W)
}

class TLBundleA(p: TLBundleParameters) extends Bundle {
  val opcode  = UInt(3.W)
  val param   = UInt(3.W)
  val size    = UInt(p.sizeBits.W)
  val source  = UInt(p.sourceBits.W)
  val address = UInt(p.addressBits.W)
  val mask    = UInt((p.dataBits / 8).W)
  val data    = UInt(p.dataBits.W)
  val corrupt = Bool()
}

class TLBundleD(p: TLBundleParameters) extends Bundle {
  val opcode  = UInt(3.W)
  val param   = UInt(2.W)
  val size    = UInt(p.sizeBits.W)
  val source  = UInt(p.sourceBits.W)
  val sink    = UInt(p.sinkBits.W)
  val denied  = Bool()
  val data    = UInt(p.dataBits.W)
  val corrupt = Bool()
}

/** Client-side view: A is an output, D an input (as in rocket-chip). */
class TLBundle(val params: TLBundleParameters) extends Bundle {
  val a = Decoupled(new TLBundleA(params))
  val d = Flipped(Decoupled(new TLBundleD(params)))
}
