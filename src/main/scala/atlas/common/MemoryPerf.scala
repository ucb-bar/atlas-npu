package atlas.common

import chisel3._
import chisel3.util._

/** Event increments only: the tile PMU owns accumulation, enable and snapshots. */
class ImemPerfEvents extends Bundle {
  val fetchCycles = Bool()
  val hostWriteWords = Bool()
  val hostWriteWordsWhileRunning = Bool()
  val hostWriteStallCycles = Bool()
  val hostReadRequests = Bool()
  val hostReadBlockedByFetchCycles = Bool()
  val responseBackpressureCycles = Bool()
  val hostBusyCycles = Bool()
  val halfFetchCycles = Vec(2, Bool())
  val halfHostWriteWords = Vec(2, Bool())
  val halfHostWriteWordsWhileRunning = Vec(2, Bool())
}

/** No state and no influence on SRAM or TileLink handshakes. */
class ImemPerfMonitor extends Module {
  val io = IO(new Bundle {
    val fetchActive = Input(Bool())
    val fetchHalf = Input(Bool())
    val hostHalf = Input(Bool())
    val hostIsRead = Input(Bool())
    val hostIsWrite = Input(Bool())
    val requestValid = Input(Bool())
    val requestReady = Input(Bool())
    val responseValid = Input(Bool())
    val responseReady = Input(Bool())
    val busy = Input(Bool())
    val events = Output(new ImemPerfEvents)
  })
  val write = io.requestValid && io.requestReady && io.hostIsWrite
  io.events.fetchCycles := io.fetchActive
  io.events.hostWriteWords := write
  io.events.hostWriteWordsWhileRunning := write && io.fetchActive
  io.events.hostWriteStallCycles := io.requestValid && !io.requestReady && io.hostIsWrite
  io.events.hostReadRequests := io.requestValid && io.requestReady && io.hostIsRead
  io.events.hostReadBlockedByFetchCycles := io.requestValid && io.hostIsRead && io.fetchActive
  io.events.responseBackpressureCycles := io.responseValid && !io.responseReady
  io.events.hostBusyCycles := io.busy || io.requestValid
  for (half <- 0 until 2) {
    io.events.halfFetchCycles(half) := io.fetchActive && io.fetchHalf === (half == 1).B
    io.events.halfHostWriteWords(half) := write && io.hostHalf === (half == 1).B
    io.events.halfHostWriteWordsWhileRunning(half) :=
      write && io.fetchActive && io.hostHalf === (half == 1).B
  }
}

object DmaLatencyBuckets {
  // Exclusive upper bounds. Final bucket has no upper bound.
  val upperBounds = Seq(16, 32, 64, 128, 256, 512, 1024)
  val names = Seq("0_15", "16_31", "32_63", "64_127", "128_255", "256_511", "512_1023", "1024_plus")
}

class DmaTransactionPerf(timestampBits: Int) extends Bundle {
  val issued = Bool()
  val completed = Bool()
  val latencyCycles = UInt(timestampBits.W)
  val responseLatencyCycles = UInt(timestampBits.W)
  val responseBackpressureCycles = UInt(timestampBits.W)
  val latencyBuckets = Vec(DmaLatencyBuckets.names.size, Bool())
}

/** Native DMA transactions (one beat each). Timestamp state follows source IDs,
  * independently of the PMU enable/clear window. Latencies are emitted on completion.
  * Durations must be shorter than 2^timestampBits cycles; subtraction is modular.
  */
class DmaLatencyTracker(numSources: Int, timestampBits: Int = 32) extends Module {
  require(numSources > 0 && isPow2(numSources))
  require(timestampBits >= 11)
  private val sourceBits = math.max(1, log2Ceil(numSources))
  val io = IO(new Bundle {
    val issue = Flipped(Valid(new Bundle {
      val source = UInt(sourceBits.W)
      val isRead = Bool()
    }))
    val response = Flipped(Valid(UInt(sourceBits.W)))
    val responseAccepted = Input(Bool())
    val reads = Output(new DmaTransactionPerf(timestampBits))
    val writes = Output(new DmaTransactionPerf(timestampBits))
  })
  val cycle = RegInit(0.U(timestampBits.W))
  cycle := cycle + 1.U
  val issuedAt = Reg(Vec(numSources, UInt(timestampBits.W)))
  val firstResponseAge = Reg(Vec(numSources, UInt(timestampBits.W)))
  val isRead = Reg(Vec(numSources, Bool()))
  val active = RegInit(VecInit(Seq.fill(numSources)(false.B)))
  val responseSeen = RegInit(VecInit(Seq.fill(numSources)(false.B)))
  val source = io.response.bits
  val completed = io.response.valid && io.responseAccepted
  // Also handle a zero-cycle responder. Reuse on completion observes OLD metadata.
  val bypassIssue = !active(source) && io.issue.valid && io.issue.bits.source === source
  val responseIsRead = Mux(bypassIssue, io.issue.bits.isRead, isRead(source))
  val age = Mux(bypassIssue, 0.U, cycle - issuedAt(source))
  val arrivalAge = Mux(!bypassIssue && responseSeen(source), firstResponseAge(source), age)
  when(io.issue.valid) {
    assert(!active(io.issue.bits.source) || (completed && source === io.issue.bits.source),
      "DMA performance tracker: source reused before completion")
  }
  when(io.response.valid) {
    assert(active(source) || bypassIssue, "DMA performance tracker: response has no issue")
  }
  for (id <- 0 until numSources) {
    val issue = io.issue.valid && io.issue.bits.source === id.U
    val response = io.response.valid && source === id.U
    when(issue) {
      issuedAt(id) := cycle
      isRead(id) := io.issue.bits.isRead
      responseSeen(id) := false.B
      active(id) := !(response && io.responseAccepted && !active(id))
    }.elsewhen(response) {
      when(!responseSeen(id)) {
        firstResponseAge(id) := age
        responseSeen(id) := true.B
      }
      when(io.responseAccepted) { active(id) := false.B }
    }
    // A same-cycle first response can remain stalled after a zero-cycle issue.
    when(issue && response && !active(id) && !io.responseAccepted) {
      firstResponseAge(id) := 0.U
      responseSeen(id) := true.B
    }
  }
  for ((events, read) <- Seq((io.reads, true), (io.writes, false))) {
    val done = completed && responseIsRead === read.B
    events.issued := io.issue.valid && io.issue.bits.isRead === read.B
    events.completed := done
    events.latencyCycles := Mux(done, age, 0.U)
    events.responseLatencyCycles := Mux(done, arrivalAge, 0.U)
    events.responseBackpressureCycles := Mux(done, age - arrivalAge, 0.U)
    for (bucket <- DmaLatencyBuckets.names.indices) {
      val lower = if (bucket == 0) true.B else age >= DmaLatencyBuckets.upperBounds(bucket - 1).U
      val upper = if (bucket == DmaLatencyBuckets.upperBounds.size) true.B
                  else age < DmaLatencyBuckets.upperBounds(bucket).U
      events.latencyBuckets(bucket) := done && lower && upper
    }
  }
}
