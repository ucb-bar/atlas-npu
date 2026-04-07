// ============================================================================
// AssemblyTestRunner.scala — Reusable framework for running assembly tests.
//
// Convention:
//   • DBG0 = 0x01 → PASS
//   • DBG0 = any other non-zero → FAIL (value = failing test case number)
//   • Test terminates via ECALL or EBREAK after writing DBG0.
//     The core drains all async work (DMA, store buffer) before halting.
//   • Legacy tests using JAL x0, 0 self-loops are also supported:
//     the runner detects halted OR dbg0 != 0, whichever comes first.
//
// Supports directives embedded in assembly comments:
//   # @TIMEOUT <cycles>
//   # @DRAM <word_offset> <hex_data>
//   # @CHECK_DRAM <word_offset> <expected_hex>
//   # @DRAM_BASE <hex_address>
//   # @PYTHON_GEN <script_path> [args...]
//   # @BF16_REL_TOL <float>
//   # @BF16_ABS_TOL <float>
//
// If @BF16_REL_TOL is present, DRAM checks are interpreted as packed BF16
// lanes inside each SRAM beat and compared element-wise with tolerance.
// Otherwise checks are exact integer equality.
//
// After loading the program, the runner pulses softReset via the harness
// to start the core (which initialises in the halted state).
// ============================================================================

package atlas.tile.assembly

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import atlas.common._
import atlas.tile.{AtlasTileTestHarness, Assembler}

import scala.io.Source
import scala.sys.process._

case class DramPreload(wordOffset: Int, data: BigInt)
case class DramCheck(wordOffset: Int, expected: BigInt)

case class TestDirectives(
    timeout:      Int              = 50000,
    dramBase:     Long             = 0x60000000L,
    dramPreloads: Seq[DramPreload] = Seq.empty,
    dramChecks:   Seq[DramCheck]   = Seq.empty,
    bf16RelTol:   Option[Double]   = None,
    bf16AbsTol:   Double           = 1e-2
)

object AssemblyTestRunner {

  private val PASS_VALUE = 1
  private val DBG0_CSR   = 0xC10

  // ==========================================================================
  // Python generator support
  // ==========================================================================

  private def resolveGeneratorsDir(): String = {
    val url = getClass.getResource("/assembly/generators")
    if (url != null && url.getProtocol == "file") {
      new java.io.File(url.toURI).getAbsolutePath
    } else {
      val envDir = sys.env.getOrElse(
        "ATLAS_GENERATORS_DIR",
        "generators/sp26-atlas-acc/src/test/resources/assembly/generators"
      )
      new java.io.File(envDir).getAbsolutePath
    }
  }

  private def runPythonGenerator(
      scriptPath: String,
      args:       Seq[String] = Seq.empty,
      verbose:    Boolean     = false
  ): (Seq[DramPreload], Seq[DramCheck], Option[Int]) = {
    val genDir   = resolveGeneratorsDir()
    val fullPath = if (scriptPath.startsWith("/")) scriptPath else s"$genDir/$scriptPath"
    val cmd      = Seq("python3", fullPath) ++ args

    if (verbose) println(s"  [python-gen] Running: ${cmd.mkString(" ")}")

    val stdout   = new StringBuilder
    val stderr   = new StringBuilder
    val exitCode = cmd ! ProcessLogger(
      stdout.append(_).append('\n'),
      stderr.append(_).append('\n')
    )

    require(exitCode == 0,
      s"Python generator failed (exit $exitCode):\n" +
      s"  cmd: ${cmd.mkString(" ")}\n  stderr: ${stderr.toString.trim}")

    parseGeneratorJson(stdout.toString.trim)
  }

  private def parseGeneratorJson(
      json: String
  ): (Seq[DramPreload], Seq[DramCheck], Option[Int]) = {
    val preloads = scala.collection.mutable.ArrayBuffer[DramPreload]()
    val checks   = scala.collection.mutable.ArrayBuffer[DramCheck]()
    var timeout: Option[Int] = None

    val preloadPat = """"word_offset"\s*:\s*(\d+)\s*,\s*"data"\s*:\s*"(0x[0-9a-fA-F]+)"""".r
    for (m <- preloadPat.findAllMatchIn(json)) {
      preloads += DramPreload(m.group(1).toInt, BigInt(m.group(2).stripPrefix("0x"), 16))
    }

    val checkPat = """"word_offset"\s*:\s*(\d+)\s*,\s*"expected"\s*:\s*"(0x[0-9a-fA-F]+)"""".r
    for (m <- checkPat.findAllMatchIn(json)) {
      checks += DramCheck(m.group(1).toInt, BigInt(m.group(2).stripPrefix("0x"), 16))
    }

    val timeoutPat = """"timeout"\s*:\s*(\d+)""".r
    timeoutPat.findFirstMatchIn(json).foreach(m => timeout = Some(m.group(1).toInt))

    (preloads.toSeq, checks.toSeq, timeout)
  }

  // ==========================================================================
  // Directive parsing
  // ==========================================================================

  def parseDirectives(source: String, verbose: Boolean = false): TestDirectives = {
    var timeout    = 50000
    var dramBase   = 0x60000000L
    var bf16RelTol = Option.empty[Double]
    var bf16AbsTol = 1e-2
    val preloads   = scala.collection.mutable.ArrayBuffer[DramPreload]()
    val checks     = scala.collection.mutable.ArrayBuffer[DramCheck]()

    for (line <- source.linesIterator) {
      val t = line.trim

      if (t.contains("@TIMEOUT"))
        """@TIMEOUT\s+(\d+)""".r.findFirstMatchIn(t).foreach(m => timeout = m.group(1).toInt)

      if (t.contains("@DRAM_BASE"))
        """@DRAM_BASE\s+(0x[0-9a-fA-F]+|\d+)""".r.findFirstMatchIn(t)
          .foreach(m => dramBase = java.lang.Long.decode(m.group(1)))

      if (t.contains("@BF16_REL_TOL"))
        """@BF16_REL_TOL\s+([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)""".r
          .findFirstMatchIn(t).foreach(m => bf16RelTol = Some(m.group(1).toDouble))

      if (t.contains("@BF16_ABS_TOL"))
        """@BF16_ABS_TOL\s+([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)""".r
          .findFirstMatchIn(t).foreach(m => bf16AbsTol = m.group(1).toDouble)

      if (t.contains("@DRAM "))
        """@DRAM\s+(\d+)\s+(0x[0-9a-fA-F]+|\d+)""".r.findFirstMatchIn(t).foreach { m =>
          preloads += DramPreload(m.group(1).toInt,
            BigInt(m.group(2).stripPrefix("0x").stripPrefix("0X"), 16))
        }

      if (t.contains("@CHECK_DRAM"))
        """@CHECK_DRAM\s+(\d+)\s+(0x[0-9a-fA-F]+|\d+)""".r.findFirstMatchIn(t).foreach { m =>
          checks += DramCheck(m.group(1).toInt,
            BigInt(m.group(2).stripPrefix("0x").stripPrefix("0X"), 16))
        }

      if (t.contains("@PYTHON_GEN"))
        """@PYTHON_GEN\s+(\S+)(.*)""".r.findFirstMatchIn(t).foreach { m =>
          val script = m.group(1)
          val args   = m.group(2).trim.split("\\s+").filter(_.nonEmpty).toSeq
          val (pyPre, pyChk, pyTo) = runPythonGenerator(script, args, verbose)
          preloads ++= pyPre; checks ++= pyChk
          pyTo.foreach(to => timeout = to)
        }
    }

    TestDirectives(timeout, dramBase, preloads.toSeq, checks.toSeq, bf16RelTol, bf16AbsTol)
  }

  // ==========================================================================
  // BF16 comparison helpers
  // ==========================================================================

  private def bf16ToFloat(bits: Int): Float =
    java.lang.Float.intBitsToFloat((bits & 0xFFFF) << 16)

  private def unpackBf16Lanes(word: BigInt, beatBits: Int): Seq[Int] =
    (0 until beatBits / 16).map(i => ((word >> (16 * i)) & 0xFFFF).toInt)

  private def compareBf16Beat(
      actual: BigInt, expected: BigInt, beatBits: Int,
      relTol: Double, absTol: Double
  ): Seq[String] = {
    val actLanes = unpackBf16Lanes(actual, beatBits)
    val expLanes = unpackBf16Lanes(expected, beatBits)

    actLanes.zip(expLanes).zipWithIndex.flatMap { case ((aBits, eBits), lane) =>
      val a      = bf16ToFloat(aBits).toDouble
      val e      = bf16ToFloat(eBits).toDouble
      val absErr = math.abs(a - e)
      val relErr = if (math.abs(e) < 1e-12) Double.PositiveInfinity else absErr / math.abs(e)
      if (absErr <= absTol || relErr <= relTol) Nil
      else Seq(f"lane[$lane]: got=$a%.6g exp=$e%.6g abs_err=$absErr%.6g rel_err=$relErr%.6g")
    }
  }

  // ==========================================================================
  // Source loading
  // ==========================================================================

  def loadSource(resourcePath: String): String = {
    val stream = getClass.getResourceAsStream(resourcePath)
    require(stream != null, s"Resource not found: $resourcePath")
    val s = Source.fromInputStream(stream)
    try s.mkString finally s.close()
  }

  // ==========================================================================
  // Test execution
  // ==========================================================================

  def runTest(
      tp:       AtlasParams,
      source:   String,
      testName: String,
      verbose:  Boolean = false
  ): (Boolean, String) = {
    val directives    = parseDirectives(source, verbose)
    val program       = Assembler.assemble(source)
    val dramSizeBytes = 4 * 1024 * 1024
    val beatBits      = tp.dma.beatBytes * 8

    if (verbose) {
      println(
        s"[$testName] Assembled ${program.length} instructions, " +
        s"timeout=${directives.timeout}, " +
        s"preloads=${directives.dramPreloads.length}, " +
        s"checks=${directives.dramChecks.length}, " +
        s"bf16RelTol=${directives.bf16RelTol.getOrElse("exact")}, " +
        s"bf16AbsTol=${directives.bf16AbsTol}")
    }

    var result: (Boolean, String) = (false, "Did not complete")

    simulate(new AtlasTileTestHarness(
      tp, dramBase = directives.dramBase, dramSizeBytes = dramSizeBytes
    )) { dut =>
      // ── Reset ──
      dut.reset.poke(true.B)
      dut.io.imemWrite.valid.poke(false.B)
      dut.io.softReset.poke(false.B)
      dut.io.ramWriteValid.poke(false.B)
      dut.io.ramWriteIdx.poke(0.U)
      dut.io.ramWriteData.poke(0.U)
      dut.io.ramReadIdx.poke(0.U)
      dut.clock.step(2)

      // ── DRAM preloads ──
      for (pl <- directives.dramPreloads) {
        dut.io.ramWriteValid.poke(true.B)
        dut.io.ramWriteIdx.poke(pl.wordOffset.U)
        dut.io.ramWriteData.poke(pl.data.U)
        dut.clock.step()
      }
      dut.io.ramWriteValid.poke(false.B)

      // ── IMEM program load ──
      for ((instr, idx) <- program.zipWithIndex) {
        dut.io.imemWrite.valid.poke(true.B)
        dut.io.imemWrite.bits.addr.poke(idx.U)
        dut.io.imemWrite.bits.data.poke((instr.toLong & 0xFFFFFFFFL).U)
        dut.clock.step()
      }
      dut.io.imemWrite.valid.poke(false.B)
      dut.clock.step()

      // ── Start core via softReset ──
      // The core initialises halted; pulse softReset through the CSR TL
      // interface to clear the halted flag and begin execution.
      dut.reset.poke(false.B)
      dut.io.softReset.poke(true.B)
      dut.clock.step(2) // TL A handshake + D response
      dut.io.softReset.poke(false.B)
      dut.clock.step()

      // ── Run ──
      var cycles = 0; var done = false

      while (cycles < directives.timeout && !done) {
        dut.clock.step(); cycles += 1
        if (dut.io.halted.peek().litToBoolean) done = true
      }

      // ── Check results ──
      if (!done) {
        result = (false, s"TIMEOUT after ${directives.timeout} cycles")
      } else {
        val dbg0 = dut.io.dbg.dbg0.peek().litValue.toLong
        if (dbg0 == PASS_VALUE) {
          var dramOk = true
          val failures = scala.collection.mutable.ArrayBuffer[String]()

          for (chk <- directives.dramChecks) {
            dut.io.ramReadIdx.poke(chk.wordOffset.U)
            dut.clock.step()
            val actual = dut.io.ramReadData.peek().litValue

            directives.bf16RelTol match {
              case Some(relTol) =>
                val laneErrs = compareBf16Beat(actual, chk.expected, beatBits, relTol, directives.bf16AbsTol)
                if (laneErrs.nonEmpty) {
                  dramOk = false
                  failures += s"  word[${chk.wordOffset}]: got 0x${actual.toString(16)}, expected 0x${chk.expected.toString(16)}"
                  laneErrs.foreach(msg => failures += s"    $msg")
                }
              case None =>
                if (actual != chk.expected) {
                  failures += f"  word[${chk.wordOffset}]: got 0x${actual}%x, expected 0x${chk.expected}%x"
                  dramOk = false
                }
            }
          }

          if (dramOk) {
            val mode = directives.bf16RelTol match {
              case Some(rt) => f"BF16 relTol=$rt%.4g absTol=${directives.bf16AbsTol}%.4g"
              case None     => s"${directives.dramChecks.length} DRAM checks OK"
            }
            result = (true, s"PASS ($cycles cycles, $mode)")
          } else {
            result = (false, s"DRAM check(s) failed at cycle $cycles:\n${failures.mkString("\n")}")
          }
        } else if (dbg0 == 0) {
          result = (false, s"SALU halted unexpectedly at cycle $cycles (illegal instruction?)")
        } else {
          result = (false, s"FAIL: DBG0 = $dbg0 (test case $dbg0 failed) at cycle $cycles")
        }
      }
    }

    if (verbose) println(s"[$testName] ${result._2}")
    result
  }
}
