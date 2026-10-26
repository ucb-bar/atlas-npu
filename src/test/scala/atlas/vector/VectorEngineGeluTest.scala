package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec

/** vgelu.bf16 gives, bit for bit, what the eight .bf16 instructions of its definition give on
  * VectorEngineTop: vcube; vmul.vx c1; vadd x; vmul.vx c2; vtanh; vadd.vx 1; vmul x; vmul.vx 0.5.
  * Every BF16 input, and vgelu in either VPU slot next to another op.
  */
class VectorEngineGeluTest extends AnyFlatSpec {
  import VpuBroadcastHarness._

  private val (mul, add, tanh, cube, mov, gelu) = (3, 1, 8, 13, 26, 31)
  private val (c1, c2, one, half) = (0x3D37, 0x3F4C, 0x3F80, 0x3F00)

  /** The eight instructions from T(vs1) to T(vd), one after the other. */
  private def sequence(dut: VpuBroadcastHarness, mem: Mem, vd: Int, vs1: Int): Unit =
    Seq(Cmd(cube, vd, vs1), Cmd(mul, vd, vd, src = xreg, imm = c1), Cmd(add, vd, vd, vs1),
        Cmd(mul, vd, vd, src = xreg, imm = c2), Cmd(tanh, vd, vd), Cmd(add, vd, vd, src = xreg, imm = one),
        Cmd(mul, vd, vs1, vd), Cmd(mul, vd, vd, src = xreg, imm = half))
      .foreach(c => drive(dut, mem, Map(0 -> c)))

  "vgelu.bf16" should "give the eight .bf16 instructions' result for every BF16 input" in {
    simulate(new VpuBroadcastHarness) { dut =>
      for (chunk <- 0 until 64) {
        val mem: Mem = Array.tabulate(64, 32) { (bank, row) =>
          if (bank > 1) BigInt(0)
          else (0 until 16).map(lane => BigInt(chunk * 1024 + bank * 512 + row * 16 + lane) << (16 * lane)).sum
        }
        drive(dut, mem, Map(0 -> Cmd(gelu, 4, 0)))
        sequence(dut, mem, 8, 0)
        val (got, want) = (tile(mem, 4), tile(mem, 8))
        val differ = got.indices.filter(i => got(i) != want(i))
        assert(differ.isEmpty, f"inputs 0x${chunk * 1024}%04x and up: rows (2 i + register) that differ")
      }
    }
  }

  it should "run in either VPU slot next to another op" in {
    simulate(new VpuBroadcastHarness) { dut =>
      val rng = new scala.util.Random(3)
      val mem: Mem = Array.fill(64, 32)((0 until 16).map(lane => BigInt(rng.nextInt(1 << 16)) << (16 * lane)).sum)
      val alone = mem.map(_.clone)
      drive(dut, alone, Map(0 -> Cmd(gelu, 4, 0)))
      drive(dut, alone, Map(0 -> Cmd(tanh, 8, 2)))
      drive(dut, alone, Map(0 -> Cmd(mul, 12, 6, src = xreg, imm = c2)))
      // vtanh in slot 1, vgelu in slot 2
      val m1 = mem.map(_.clone)
      drive(dut, m1, Map(0 -> Cmd(tanh, 8, 2), 1 -> Cmd(gelu, 4, 0)))
      // vgelu in slot 1, vmul.vx in slot 2
      val m2 = mem.map(_.clone)
      drive(dut, m2, Map(0 -> Cmd(gelu, 4, 0), 1 -> Cmd(mul, 12, 6, src = xreg, imm = c2)))
      // A second vgelu and a two-input .bf16 op wait for vgelu's last write; vtanh does not.
      assert(firstIssue(dut, Cmd(gelu, 4, 0), Cmd(gelu, 8, 2)) == 74)
      assert(firstIssue(dut, Cmd(gelu, 4, 0), Cmd(add, 8, 2, 6)) == 74)
      assert(firstIssue(dut, Cmd(gelu, 4, 0), Cmd(tanh, 8, 2)) == 1)
      for ((m, other) <- Seq((m1, 8), (m2, 12))) {
        assert(tile(m, 4) == tile(alone, 4), s"vgelu next to the op that writes m$other")
        assert(tile(m, other) == tile(alone, other), s"the op that writes m$other next to vgelu")
      }
    }
  }
}
