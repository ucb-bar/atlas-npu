package atlas.perf

import chisel3._
import chisel3.util._

final case class AtlasPerfCounter(name: String, offset: Int)

object AtlasPerfCounterMap {
  val ControlOffset     = 0x20
  val CounterBaseOffset = 0x24
  val EnableBit         = 0
  val ClearBit          = 1

  private val baseCounterNames = Seq(
    "run_cycles",
    "dma_active_cycles",
    "dma_commands",
    "dma_beats_issued",
    "dma_beats_completed",
    "dma_vmem_stall_cycles",
    "dma_saturated_cycles",
    "dma_inflight_limit_blocked_cycles",
    "dma_outstanding_cycles",
    "dma_response_wait_beat_cycles",
    "dma_fence_wait_cycles",
    "vpu_active_cycles",
    "vpu_commands",
    "mxu0_compute_cycles",
    "mxu0_data_cycles",
    "mxu0_commands",
    "mxu1_compute_cycles",
    "mxu1_data_cycles",
    "mxu1_commands"
  )

  def counters(mregBanks: Int, vmemBanks: Int): Seq[AtlasPerfCounter] = {
    require(mregBanks > 0, "At least one MREG bank is required")
    require(vmemBanks > 0, "At least one VMEM bank is required")
    val bankCounterNames =
      (0 until mregBanks).flatMap { bank =>
        Seq(
          s"mreg_bank_${bank}_read_cycles",
          s"mreg_bank_${bank}_write_cycles",
          s"mreg_bank_${bank}_read_contended_cycles",
          s"mreg_bank_${bank}_write_contended_cycles"
        )
      } ++ (0 until vmemBanks).flatMap { bank =>
        Seq(
          s"vmem_bank_${bank}_read_cycles",
          s"vmem_bank_${bank}_write_cycles",
          s"vmem_bank_${bank}_contended_cycles"
        )
      }

    (baseCounterNames ++ bankCounterNames).zipWithIndex.map { case (name, index) =>
      AtlasPerfCounter(name, CounterBaseOffset + 4 * index)
    }
  }

  val DefaultMregBanks = 32
  val DefaultVmemBanks = 6
  val Counters = counters(DefaultMregBanks, DefaultVmemBanks)

  def counterEndOffset(mregBanks: Int, vmemBanks: Int): Int =
    CounterBaseOffset + 4 * counters(mregBanks, vmemBanks).size

  val CounterEndOffset = counterEndOffset(DefaultMregBanks, DefaultVmemBanks)
}

class AtlasPerfEvents(
  mregBanks: Int = AtlasPerfCounterMap.DefaultMregBanks,
  vmemBanks: Int = AtlasPerfCounterMap.DefaultVmemBanks
) extends Bundle {
  val runActive                    = Bool()
  val dmaActive                    = Bool()
  val dmaCommand                   = Bool()
  val dmaIssuedBeat                = Bool()
  val dmaCompletedBeat             = Bool()
  val dmaVmemStall                 = Bool()
  val dmaSaturated                 = Bool()
  val dmaLimitBlocked              = Bool()
  val dmaOutstanding               = Bool()
  val dmaResponseWaitBeatCycles    = UInt(32.W)
  val dmaFenceWait                 = Bool()
  val vpuActive                    = Bool()
  val vpuCommand                   = Bool()
  val mxu0Compute                  = Bool()
  val mxu0Data                     = Bool()
  val mxu0Command                  = Bool()
  val mxu1Compute                  = Bool()
  val mxu1Data                     = Bool()
  val mxu1Command                  = Bool()
  val mregRead                     = Vec(mregBanks, Bool())
  val mregWrite                    = Vec(mregBanks, Bool())
  val mregReadContended            = Vec(mregBanks, Bool())
  val mregWriteContended           = Vec(mregBanks, Bool())
  val vmemRead                     = Vec(vmemBanks, Bool())
  val vmemWrite                    = Vec(vmemBanks, Bool())
  val vmemContended                = Vec(vmemBanks, Bool())
}

class AtlasPerfCounterBank(
  mregBanks: Int = AtlasPerfCounterMap.DefaultMregBanks,
  vmemBanks: Int = AtlasPerfCounterMap.DefaultVmemBanks
) extends Module {
  private val wordIndexBits = 10
  private val counterMap    = AtlasPerfCounterMap.counters(mregBanks, vmemBanks)
  private val counterCount  = counterMap.size

  val io = IO(new Bundle {
    val enable    = Input(Bool())
    val clear     = Input(Bool())
    val events    = Input(new AtlasPerfEvents(mregBanks, vmemBanks))
    val wordIndex = Input(UInt(wordIndexBits.W))
    val readData  = Output(UInt(32.W))
  })

  val counters = RegInit(VecInit(Seq.fill(counterCount)(0.U(32.W))))
  val increments: Seq[UInt] = Seq(
    io.events.runActive.asUInt,
    io.events.dmaActive.asUInt,
    io.events.dmaCommand.asUInt,
    io.events.dmaIssuedBeat.asUInt,
    io.events.dmaCompletedBeat.asUInt,
    io.events.dmaVmemStall.asUInt,
    io.events.dmaSaturated.asUInt,
    io.events.dmaLimitBlocked.asUInt,
    io.events.dmaOutstanding.asUInt,
    io.events.dmaResponseWaitBeatCycles,
    io.events.dmaFenceWait.asUInt,
    io.events.vpuActive.asUInt,
    io.events.vpuCommand.asUInt,
    io.events.mxu0Compute.asUInt,
    io.events.mxu0Data.asUInt,
    io.events.mxu0Command.asUInt,
    io.events.mxu1Compute.asUInt,
    io.events.mxu1Data.asUInt,
    io.events.mxu1Command.asUInt
  ) ++ (0 until mregBanks).flatMap { bank =>
    Seq(
      io.events.mregRead(bank).asUInt,
      io.events.mregWrite(bank).asUInt,
      io.events.mregReadContended(bank).asUInt,
      io.events.mregWriteContended(bank).asUInt
    )
  } ++ (0 until vmemBanks).flatMap { bank =>
    Seq(
      io.events.vmemRead(bank).asUInt,
      io.events.vmemWrite(bank).asUInt,
      io.events.vmemContended(bank).asUInt
    )
  }

  require(increments.size == counterCount,
    "Performance event order must match the counter map")

  when(io.clear) {
    counters.foreach(_ := 0.U)
  }.elsewhen(io.enable) {
    for (i <- 0 until counterCount) {
      when(increments(i).orR) { counters(i) := counters(i) + increments(i) }
    }
  }

  private val readWords = counterMap.zipWithIndex.flatMap {
    case (counter, index) =>
      val wordIndex = counter.offset / 4
      Seq(wordIndex.U -> counters(index))
  }

  io.readData := MuxLookup(io.wordIndex, 0.U(32.W))(readWords)
}