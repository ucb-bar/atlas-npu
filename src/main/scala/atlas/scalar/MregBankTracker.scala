/*
MregBankTracker.scala
Purely combinational aggregator for MREG bank activity across all engines.

Collects per-bank read/write reports from MXU0, MXU1, XLU, VPU, LSU, and
any pending (not-yet-accepted) commands, then produces two bitvectors:
  - readBusy:  OR of all banks currently being read by any engine.
  - writeBusy: OR of all banks currently being written (or pending write)
               by any engine.

The scalar core uses these bitvectors for direction-aware hazard detection:
  - Instruction reads  bank X → stall only if writeBusy(X)          (RAW)
  - Instruction writes bank X → stall if readBusy(X) | writeBusy(X) (WAR/WAW)
*/
package atlas.scalar

import chisel3._
import chisel3.util._

/** Aggregates per-engine MREG read/write reports into busy bitvectors.
  *
  * @param numBanks    Total number of matrix-register banks tracked.
  * @param numReaders  Number of read-report ports presented to the tracker.
  * @param numWriters  Number of write-report ports presented to the tracker.
  */
class MregBankTracker(
    numBanks:   Int = 64,
    numReaders: Int = 10,
    numWriters: Int = 10
) extends Module {

  private val bankBits = log2Ceil(numBanks)

  val io = IO(new Bundle {
    val readers  = Input(Vec(numReaders, Valid(UInt(bankBits.W))))
    val writers  = Input(Vec(numWriters, Valid(UInt(bankBits.W))))
    val readBusy  = Output(UInt(numBanks.W))
    val writeBusy = Output(UInt(numBanks.W))
  })

  private def bankMask(port: Valid[UInt]): UInt =
    Mux(port.valid, 1.U(numBanks.W) << port.bits, 0.U(numBanks.W))

  io.readBusy  := io.readers.map(bankMask).reduce(_ | _)
  io.writeBusy := io.writers.map(bankMask).reduce(_ | _)
}
