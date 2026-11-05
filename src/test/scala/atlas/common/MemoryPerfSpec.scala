package atlas.common

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable
import scala.util.Random

class MemoryPerfSpec extends AnyFlatSpec with PeekPokeAPI {
  "IMEM performance events" should "distinguish accepted writes, stalls, reads and both halves" in {
    EphemeralSimulator.simulate(new ImemPerfMonitor) { dut =>
      // Exhaust all control combinations, including halted loading and concurrent fetch/write.
      for (bits <- 0 until 512) {
        def flag(bit: Int): Boolean = (bits & (1 << bit)) != 0
        val fetch = flag(0); val fetchHalf = flag(1); val hostHalf = flag(2)
        val read = flag(3); val write = !read; val valid = flag(4); val ready = flag(5)
        val dValid = flag(6); val dReady = flag(7); val busy = flag(8)
        dut.io.fetchActive.poke(fetch.B); dut.io.fetchHalf.poke(fetchHalf.B)
        dut.io.hostHalf.poke(hostHalf.B); dut.io.hostIsRead.poke(read.B)
        dut.io.hostIsWrite.poke(write.B); dut.io.requestValid.poke(valid.B)
        dut.io.requestReady.poke(ready.B); dut.io.responseValid.poke(dValid.B)
        dut.io.responseReady.poke(dReady.B); dut.io.busy.poke(busy.B)
        val accepted = valid && ready && write
        val e = dut.io.events
        e.fetchCycles.expect(fetch.B)
        e.hostWriteWords.expect(accepted.B)
        e.hostWriteWordsWhileRunning.expect((accepted && fetch).B)
        e.hostWriteStallCycles.expect((valid && !ready && write).B)
        e.hostReadRequests.expect((valid && ready && read).B)
        e.hostReadBlockedByFetchCycles.expect((valid && read && fetch).B)
        e.responseBackpressureCycles.expect((dValid && !dReady).B)
        e.hostBusyCycles.expect((busy || valid).B)
        for (half <- 0 until 2) {
          e.halfFetchCycles(half).expect((fetch && fetchHalf == (half == 1)).B)
          e.halfHostWriteWords(half).expect((accepted && hostHalf == (half == 1)).B)
          e.halfHostWriteWordsWhileRunning(half).expect((accepted && fetch && hostHalf == (half == 1)).B)
        }
      }
    }
  }

  "DMA latency tracking" should "match source IDs through reordering, stalls, reuse, histogram boundaries and wrap" in {
    // An 11-bit timestamp makes repeated rollover practical in a short simulation.
    EphemeralSimulator.simulate(new DmaLatencyTracker(4, timestampBits = 11)) { dut =>
      dut.io.issue.valid.poke(false.B); dut.io.issue.bits.source.poke(0.U)
      dut.io.issue.bits.isRead.poke(false.B); dut.io.response.valid.poke(false.B)
      dut.io.response.bits.poke(0.U); dut.io.responseAccepted.poke(false.B)
      dut.reset.poke(true.B); dut.clock.step(); dut.reset.poke(false.B)
      case class Pending(start: Int, read: Boolean, var arrival: Option[Int] = None)
      val pending = mutable.Map.empty[Int, Pending]
      var cycle = 0
      def step(issue: Option[(Int, Boolean)] = None, response: Option[(Int, Boolean)] = None): Unit = {
        dut.io.issue.valid.poke(issue.nonEmpty.B)
        dut.io.issue.bits.source.poke(issue.map(_._1).getOrElse(0).U)
        dut.io.issue.bits.isRead.poke(issue.exists(_._2).B)
        dut.io.response.valid.poke(response.nonEmpty.B)
        dut.io.response.bits.poke(response.map(_._1).getOrElse(0).U)
        dut.io.responseAccepted.poke(response.exists(_._2).B)
        val info = response.map { case (id, _) =>
          pending.getOrElse(id, Pending(cycle, issue.get._2))
        }
        val age = info.map(p => (cycle - p.start) & 2047).getOrElse(0)
        val arrival = info.flatMap(_.arrival).getOrElse(age)
        for ((e, read) <- Seq((dut.io.reads, true), (dut.io.writes, false))) {
          val done = response.exists(_._2) && info.exists(_.read == read)
          e.issued.expect(issue.exists(_._2 == read).B)
          e.completed.expect(done.B)
          e.latencyCycles.expect((if (done) age else 0).U)
          e.responseLatencyCycles.expect((if (done) arrival else 0).U)
          e.responseBackpressureCycles.expect((if (done) age - arrival else 0).U)
          val bucket = DmaLatencyBuckets.upperBounds.count(age >= _)
          for (i <- DmaLatencyBuckets.names.indices) e.latencyBuckets(i).expect((done && i == bucket).B)
        }
        response.foreach { case (id, accepted) =>
          if (accepted) pending.remove(id)
          else if (pending.contains(id) && pending(id).arrival.isEmpty) pending(id).arrival = Some(age)
        }
        issue.foreach { case (id, read) =>
          if (!response.contains((id, true)) || info.exists(_.start != cycle)) {
            val p = Pending(cycle, read)
            if (response.exists(r => r._1 == id && !r._2) && !pending.contains(id)) p.arrival = Some(0)
            pending(id) = p
          }
        }
        dut.clock.step(); cycle += 1
      }
      step(Some((0, true))); step(Some((1, false)))
      step(response = Some((1, true))) // Out-of-order write completion.
      step(response = Some((0, false))); step(response = Some((0, false)))
      step(response = Some((0, true))) // First D-valid counted once despite stalls.
      step(Some((0, false))); step(Some((2, true)))
      step(Some((3, true)), Some((0, true))) // Different-source issue and completion.
      step(response = Some((3, true))); step(response = Some((2, true)))
      step(Some((0, true)), Some((0, true))) // Zero-cycle completion.
      step(Some((0, false)), Some((0, false))); step(response = Some((0, true)))
      step(Some((0, true))); step(Some((0, false)), Some((0, true)))
      step(response = Some((0, true))) // Reuse observes the new transaction's direction.
      for (read <- Seq(true, false); latency <- Seq(0, 1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 1200)) {
        if (latency == 0) step(Some((0, read)), Some((0, true)))
        else { step(Some((0, read))); for (_ <- 1 until latency) step(); step(response = Some((0, true))) }
      }
      val random = new Random(194)
      var held: Option[Int] = None
      for (_ <- 0 until 6000) {
        val free = (0 until 4).filterNot(pending.contains)
        val issue = if (free.nonEmpty && random.nextBoolean()) Some((free(random.nextInt(free.size)), random.nextBoolean())) else None
        val source = held.orElse(if (pending.nonEmpty && random.nextBoolean()) Some(pending.keys.toSeq.sorted.apply(random.nextInt(pending.size))) else None)
        val response = source.map(id => (id, random.nextInt(3) != 0))
        held = response.filterNot(_._2).map(_._1)
        step(issue, response)
      }
      while (pending.nonEmpty) {
        val id = held.filter(pending.contains).getOrElse(pending.keys.min)
        step(response = Some((id, true))); held = None
      }
    }
  }
}
