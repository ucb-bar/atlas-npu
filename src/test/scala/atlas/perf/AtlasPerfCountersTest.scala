package atlas.perf

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import svsim.CommonCompilationSettings
import svsim.vcs.{Backend => VcsBackend}
import svsim.vcs.Backend
import java.nio.file.{Files, Paths}

object AtlasPerfCounterSimulator extends Simulator[VcsBackend] with PeekPokeAPI {
  private val runDir = {
    val root = Paths.get(sys.env.getOrElse("MILL_WORKSPACE_ROOT", "/tmp"))
    val path = root.resolve("tmp").resolve("AtlasPerfCountersTest")
    Files.createDirectories(path)
    path.toAbsolutePath
  }

  override val backend: VcsBackend   = VcsBackend.initializeFromProcessEnvironment()
  override val tag: String           = "AtlasPerfCountersTest"
  override val workspacePath: String = runDir.toString
  override val commonCompilationSettings: CommonCompilationSettings =
    CommonCompilationSettings(
      availableParallelism =
        CommonCompilationSettings.AvailableParallelism.UpTo(Runtime.getRuntime.availableProcessors())
    )
  override val backendSpecificCompilationSettings: Backend.CompilationSettings = {
    val coverage = Backend.CoverageSettings(
      line = true, cond = true, branch = true, fsm = true, tgl = true, assert = true
    )
    Backend.CompilationSettings(
      coverageSettings  = coverage,
      coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
      simulationSettings = Backend.SimulationSettings(
        coverageSettings  = coverage,
        coverageDirectory = Some(Backend.CoverageDirectory("coverage.vdb")),
        coverageName      = Some(Backend.CoverageName("AtlasPerfCountersTest_coverage"))
      )
    )
  }
}

class AtlasPerfCountersTest extends AnyFlatSpec with Matchers with PeekPokeAPI {
  private def clearEvents(dut: AtlasPerfCounterBank): Unit = {
    dut.io.events.runActive.poke(false.B)
    dut.io.events.dmaActive.poke(false.B)
    dut.io.events.dmaCommand.poke(false.B)
    dut.io.events.dmaIssuedBeat.poke(false.B)
    dut.io.events.dmaCompletedBeat.poke(false.B)
    dut.io.events.dmaVmemStall.poke(false.B)
    dut.io.events.dmaSaturated.poke(false.B)
    dut.io.events.dmaLimitBlocked.poke(false.B)
    dut.io.events.dmaOutstanding.poke(false.B)
    dut.io.events.dmaResponseWaitBeatCycles.poke(0.U)
    dut.io.events.dmaFenceWait.poke(false.B)
    dut.io.events.vpuActive.poke(false.B)
    dut.io.events.vpuCommand.poke(false.B)
    dut.io.events.mxu0Compute.poke(false.B)
    dut.io.events.mxu0Data.poke(false.B)
    dut.io.events.mxu0Command.poke(false.B)
    dut.io.events.mxu1Compute.poke(false.B)
    dut.io.events.mxu1Data.poke(false.B)
    dut.io.events.mxu1Command.poke(false.B)
    dut.io.events.mregRead.foreach(_.poke(false.B))
    dut.io.events.mregWrite.foreach(_.poke(false.B))
    dut.io.events.mregReadContended.foreach(_.poke(false.B))
    dut.io.events.mregWriteContended.foreach(_.poke(false.B))
    dut.io.events.vmemRead.foreach(_.poke(false.B))
    dut.io.events.vmemWrite.foreach(_.poke(false.B))
    dut.io.events.vmemContended.foreach(_.poke(false.B))
  }

  private def expectCounter(dut: AtlasPerfCounterBank, name: String, value: BigInt): Unit = {
    val counter = AtlasPerfCounterMap.Counters.find(_.name == name).get
    dut.io.wordIndex.poke((counter.offset / 4).U)
    dut.io.readData.expect(value.U)
  }

  "AtlasPerfCounterBank" should "count enabled events in mapped 32-bit words" in {
    AtlasPerfCounterSimulator.simulate(new AtlasPerfCounterBank) { module =>
      val dut = module.wrapped

      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)

      dut.io.enable.poke(true.B)
      dut.io.clear.poke(false.B)
      dut.io.wordIndex.poke((AtlasPerfCounterMap.Counters.head.offset / 4).U)
      clearEvents(dut)
      dut.io.events.runActive.poke(true.B)
      dut.io.events.dmaIssuedBeat.poke(true.B)
      dut.io.events.mxu0Compute.poke(true.B)
      dut.clock.step(3)

      dut.io.readData.expect(3.U)
      expectCounter(dut, "dma_beats_issued", 3)
      expectCounter(dut, "mxu0_compute_cycles", 3)
      dut.io.wordIndex.poke((AtlasPerfCounterMap.CounterEndOffset / 4).U)
      dut.io.readData.expect(0.U)

      dut.io.enable.poke(false.B)
      dut.clock.step(1)
      dut.io.wordIndex.poke((AtlasPerfCounterMap.Counters.head.offset / 4).U)
      dut.io.readData.expect(3.U)

      dut.io.clear.poke(true.B)
      dut.clock.step(1)
      dut.io.clear.poke(false.B)
      dut.io.readData.expect(0.U)
    }
  }

  it should "count every event independently and preserve sequential and overlapping phases" in {
    AtlasPerfCounterSimulator.simulate(new AtlasPerfCounterBank) { module =>
      val dut = module.wrapped

      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      dut.io.enable.poke(true.B)
      dut.io.clear.poke(false.B)
      dut.io.wordIndex.poke(0.U)
      clearEvents(dut)
      dut.clock.step(1)

      val eventPins = Seq(
        dut.io.events.runActive,
        dut.io.events.dmaActive,
        dut.io.events.dmaCommand,
        dut.io.events.dmaIssuedBeat,
        dut.io.events.dmaCompletedBeat,
        dut.io.events.dmaVmemStall,
        dut.io.events.vpuActive,
        dut.io.events.vpuCommand,
        dut.io.events.mxu0Compute,
        dut.io.events.mxu0Data,
        dut.io.events.mxu0Command,
        dut.io.events.mxu1Compute,
        dut.io.events.mxu1Data,
        dut.io.events.mxu1Command
      )
      val eventNames = Seq(
        "run_cycles",
        "dma_active_cycles",
        "dma_commands",
        "dma_beats_issued",
        "dma_beats_completed",
        "dma_vmem_stall_cycles",
        "vpu_active_cycles",
        "vpu_commands",
        "mxu0_compute_cycles",
        "mxu0_data_cycles",
        "mxu0_commands",
        "mxu1_compute_cycles",
        "mxu1_data_cycles",
        "mxu1_commands"
      )

      eventPins.zipWithIndex.foreach { case (event, index) =>
        clearEvents(dut)
        event.poke(true.B)
        dut.clock.step(1)
        eventNames.zipWithIndex.foreach { case (name, nameIndex) =>
          expectCounter(dut, name, if (nameIndex <= index) 1 else 0)
        }
        AtlasPerfCounterMap.Counters.drop(eventNames.size).foreach { counter =>
          expectCounter(dut, counter.name, 0)
        }
      }

      clearEvents(dut)
      dut.io.events.dmaActive.poke(true.B)
      dut.io.events.mxu0Compute.poke(true.B)
      dut.io.events.vpuActive.poke(true.B)
      dut.clock.step(2)

      eventNames.foreach { name =>
        val count = (if (Seq("dma_active_cycles", "mxu0_compute_cycles", "vpu_active_cycles").contains(name)) 2 else 1)
        expectCounter(dut, name, count)
      }
      AtlasPerfCounterMap.Counters.drop(eventNames.size).foreach { counter =>
        expectCounter(dut, counter.name, 0)
      }
    }
  }

  it should "count DMA pressure, fence waits, and per-bank use and contention" in {
    AtlasPerfCounterSimulator.simulate(new AtlasPerfCounterBank) { module =>
      val dut = module.wrapped

      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      clearEvents(dut)
      dut.io.enable.poke(true.B)
      dut.io.clear.poke(false.B)

      dut.io.events.dmaSaturated.poke(true.B)
      dut.io.events.dmaLimitBlocked.poke(true.B)
      dut.io.events.dmaOutstanding.poke(true.B)
      dut.io.events.dmaResponseWaitBeatCycles.poke(3.U)
      dut.io.events.dmaFenceWait.poke(true.B)
      dut.io.events.mregRead(2).poke(true.B)
      dut.io.events.mregReadContended(2).poke(true.B)
      dut.io.events.mregWrite(5).poke(true.B)
      dut.io.events.mregWriteContended(5).poke(true.B)
      dut.io.events.vmemRead(1).poke(true.B)
      dut.io.events.vmemContended(1).poke(true.B)
      dut.io.events.vmemWrite(4).poke(true.B)
      dut.clock.step(2)

      expectCounter(dut, "dma_saturated_cycles", 2)
      expectCounter(dut, "dma_inflight_limit_blocked_cycles", 2)
      expectCounter(dut, "dma_outstanding_cycles", 2)
      expectCounter(dut, "dma_response_wait_beat_cycles", 6)
      expectCounter(dut, "dma_fence_wait_cycles", 2)
      expectCounter(dut, "mreg_bank_2_read_cycles", 2)
      expectCounter(dut, "mreg_bank_2_read_contended_cycles", 2)
      expectCounter(dut, "mreg_bank_5_write_cycles", 2)
      expectCounter(dut, "mreg_bank_5_write_contended_cycles", 2)
      expectCounter(dut, "vmem_bank_1_read_cycles", 2)
      expectCounter(dut, "vmem_bank_1_contended_cycles", 2)
      expectCounter(dut, "vmem_bank_4_write_cycles", 2)
    }
  }
}