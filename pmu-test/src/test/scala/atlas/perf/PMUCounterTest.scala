package atlas.perf

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import pmu.{PMURegistry, TilePMU}

class PMUCounterHarness extends Module {
  val io = IO(new Bundle {
    val runCycle = Input(Bool())
    val workItems = Input(UInt(3.W))
    val enable = Input(Bool())
    val clear = Input(Bool())
    val snapshot = Input(Bool())
    val counters = Output(Vec(2, UInt(64.W)))
    val snapshotCounters = Output(Vec(2, UInt(64.W)))
    val snapshotValid = Output(Bool())
  })

  val registry = new PMURegistry("test_tile")
  registry.register("work.items", io.workItems, unit = "items")
  registry.register("core.run_cycles", io.runCycle, unit = "cycles")
  val pmu = TilePMU.build(registry.build())

  pmu.enable := io.enable
  pmu.clear := io.clear
  pmu.snapshot := io.snapshot
  io.counters := pmu.values
  io.snapshotCounters := pmu.snapshot_values
  io.snapshotValid := pmu.snapshot_valid
}

class PMUCounterTest extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  private def initialize(dut: PMUCounterHarness): Unit = {
    dut.io.runCycle.poke(false.B)
    dut.io.workItems.poke(0.U)
    dut.io.enable.poke(false.B)
    dut.io.clear.poke(false.B)
    dut.io.snapshot.poke(false.B)
  }

  "The PMU registry" should "sort event IDs independently of registration order" in {
    test(new PMUCounterHarness) { dut =>
      dut.pmu.manifest.events.map(_.name) shouldBe Vector("core.run_cycles", "work.items")
      dut.pmu.manifest.events.map(_.id) shouldBe Vector(0, 1)
    }
  }

  "The PMU counters" should "count enabled events, ignore disabled events, and clear" in {
    test(new PMUCounterHarness) { dut =>
      initialize(dut)
      dut.io.runCycle.poke(true.B)
      dut.io.workItems.poke(3.U)
      dut.clock.step(3)
      dut.io.counters(0).expect(0.U)
      dut.io.counters(1).expect(0.U)

      dut.io.enable.poke(true.B)
      dut.clock.step(4)
      dut.io.counters(0).expect(4.U)
      dut.io.counters(1).expect(12.U)

      dut.io.enable.poke(false.B)
      dut.io.workItems.poke(7.U)
      dut.clock.step(2)
      dut.io.counters(0).expect(4.U)
      dut.io.counters(1).expect(12.U)

      dut.io.enable.poke(true.B)
      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.counters(0).expect(0.U)
      dut.io.counters(1).expect(0.U)
      dut.io.clear.poke(false.B)
      dut.clock.step(2)
      dut.io.counters(0).expect(2.U)
      dut.io.counters(1).expect(14.U)
    }
  }

  it should "snapshot the same-edge increment and remain stable as live counters change" in {
    test(new PMUCounterHarness) { dut =>
      initialize(dut)
      dut.io.enable.poke(true.B)
      dut.io.runCycle.poke(true.B)
      dut.io.workItems.poke(2.U)
      dut.clock.step(3)
      dut.io.counters(0).expect(3.U)
      dut.io.counters(1).expect(6.U)

      dut.io.snapshot.poke(true.B)
      dut.clock.step()
      dut.io.snapshot.poke(false.B)
      dut.io.snapshotValid.expect(true.B)
      dut.io.counters(0).expect(4.U)
      dut.io.counters(1).expect(8.U)
      dut.io.snapshotCounters(0).expect(4.U)
      dut.io.snapshotCounters(1).expect(8.U)

      dut.io.runCycle.poke(false.B)
      dut.io.workItems.poke(5.U)
      dut.clock.step(2)
      dut.io.counters(0).expect(4.U)
      dut.io.counters(1).expect(18.U)
      dut.io.snapshotCounters(0).expect(4.U)
      dut.io.snapshotCounters(1).expect(8.U)
    }
  }

  it should "capture the pre-clear next value when snapshot and clear coincide" in {
    test(new PMUCounterHarness) { dut =>
      initialize(dut)
      dut.io.enable.poke(true.B)
      dut.io.runCycle.poke(true.B)
      dut.io.workItems.poke(1.U)
      dut.clock.step(3)

      dut.io.workItems.poke(4.U)
      dut.io.clear.poke(true.B)
      dut.io.snapshot.poke(true.B)
      dut.clock.step()
      dut.io.counters(0).expect(0.U)
      dut.io.counters(1).expect(0.U)
      dut.io.snapshotCounters(0).expect(4.U)
      dut.io.snapshotCounters(1).expect(7.U)

      dut.io.clear.poke(false.B)
      dut.io.snapshot.poke(false.B)
      dut.io.enable.poke(false.B)
      dut.clock.step()
      dut.io.snapshotCounters(0).expect(4.U)
      dut.io.snapshotCounters(1).expect(7.U)
    }
  }

  it should "reset live and captured values and invalidate the snapshot" in {
    test(new PMUCounterHarness) { dut =>
      initialize(dut)
      dut.io.enable.poke(true.B)
      dut.io.runCycle.poke(true.B)
      dut.io.workItems.poke(2.U)
      dut.clock.step(2)
      dut.io.snapshot.poke(true.B)
      dut.clock.step()
      dut.io.snapshotValid.expect(true.B)

      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.io.counters(0).expect(0.U)
      dut.io.counters(1).expect(0.U)
      dut.io.snapshotCounters(0).expect(0.U)
      dut.io.snapshotCounters(1).expect(0.U)
      dut.io.snapshotValid.expect(false.B)
    }
  }
}
