package atlas.vector

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Outcome
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import atlas.common.VPUParams

class VectorEngineTestDriver extends AnyFlatSpec with Matchers {

  //----------- CI/CD INCLUDE --------------
  override def withFixture(test: NoArgTest): Outcome = {
    val outcome = super.withFixture(test)
    if (outcome.isFailed) {
      println("VectorEngineTestDriver=FAILED")
    } else if (outcome.isSucceeded) {
      println("VectorEngineTestDriver=PASSED")
    }
    outcome
  }
  //----------- CI/CD INCLUDE --------------

  // Configuration
  val vectorResource = "/vpu_vectors.txt"  
  val ulpTolerance   = 1
  val absTolerance   = 0x0040 // Values smaller than this BF16 are considered 0

  // Test vector data class
  case class TestVector(
    id:       Int,      
    desc:     String,    
    vpuOp:    String,
    numLanes: Int,
    vecA:     Seq[Int],
    vecB:     Seq[Int],
    expected: Seq[Int]
  ) {
    
    // Returns the vector block as a formatted string
    def toFlatFormat: String = {
      val hexA   = vecA.map(v => f"$v%04X").mkString(" ")
      val hexB   = vecB.map(v => f"$v%04X").mkString(" ")
      val hexExp = expected.map(v => f"$v%04X").mkString(" ")

      s"""# $id - "$desc"
         |vpuOp $vpuOp
         |numLanes $numLanes
         |vecA $hexA
         |vecB $hexB
         |exp  $hexExp""".stripMargin
    }

    def print(): Unit = {
      println(toFlatFormat)
      println() 
    }
  }

  def hexToInt(h: String): Int = Integer.parseUnsignedInt(h, 16)

  def loadVectors(resourcePath: String): Seq[TestVector] = {
    val stream = getClass.getResourceAsStream(resourcePath)

    require(
      stream != null,
      s"Resource '$resourcePath' not found on classpath. Run: python3 scripts/gen_vpu_vectors.py --out src/test/resources${resourcePath}"
    )

    val src = Source.fromInputStream(stream)
    val vectors = ArrayBuffer[TestVector]()

    var id = 0
    var desc = ""
    var vpuOp = ""
    var numLanes = 0
    var vecA = Seq.empty[Int]
    var vecB = Seq.empty[Int]
    var expected = Seq.empty[Int]
    var inCase = false

    def flush(): Unit = {
      if (inCase) {
        vectors += TestVector(
          id = id,
          desc = desc,
          vpuOp = vpuOp,
          numLanes = numLanes,
          vecA = vecA,
          vecB = vecB,
          expected = expected
        )
        inCase = false
      }
    }

    try {
      for (raw <- src.getLines()) {
        val line = raw.trim

        if (line.isEmpty) {
          if (inCase && vpuOp.nonEmpty) flush()
        } else if (line.startsWith("#")) {
          if (inCase && vpuOp.nonEmpty) flush()

          val pattern = """#\s+(\d+)\s+-\s+"(.*)"""".r
          line match {
            case pattern(idStr, descStr) =>
              id = idStr.toInt
              desc = descStr
            case _ =>
              throw new RuntimeException(s"Invalid header format: $line")
          }

          vpuOp = ""
          numLanes = 0
          vecA = Seq.empty
          vecB = Seq.empty
          expected = Seq.empty
          inCase = true
        } else {
          val parts = line.split("\\s+")
          parts(0) match {
            case "vpuOp"    => vpuOp = parts(1)
            case "numLanes" => numLanes = parts(1).toInt
            case "vecA"     => vecA = parts.drop(1).map(hexToInt).toSeq
            case "vecB"     => vecB = parts.drop(1).map(hexToInt).toSeq
            case "exp"      => expected = parts.drop(1).map(hexToInt).toSeq
            case other      =>
              throw new RuntimeException(s"Unknown directive: $other")
          }
        }
      }
      flush()
    } finally {
      src.close()
    }

    vectors.toSeq
  }

  def bf16BitsToFloat(bits: Int): Float = {
    val fp32Bits = (bits & 0xFFFF) << 16
    java.lang.Float.intBitsToFloat(fp32Bits)
  }

  def checkTolerance(actual: BigInt, expected: Int, op: String): (Boolean, Float, Float) = {
    val aFloat = bf16BitsToFloat(actual.toInt & 0xFFFF)
    val eFloat = bf16BitsToFloat(expected & 0xFFFF)
    
    if (aFloat.isNaN && eFloat.isNaN) return (true, 0.0f, 0.0f)
    if (aFloat.isPosInfinity && eFloat.isPosInfinity) return (true, 0.0f, 0.0f)
    if (aFloat.isNegInfinity && eFloat.isNegInfinity) return (true, 0.0f, 0.0f)
    
    val absError = math.abs(aFloat - eFloat)
    val relError = if (eFloat == 0.0f) {
      if (aFloat == 0.0f) 0.0f else Float.PositiveInfinity
    } else {
      absError / math.abs(eFloat)
    }

    val (maxRelError, maxAbsError) = op match {
      case "sin" | "cos" | "tanh" | "log" | "exp" | "exp2" | "sqrt" => 
        (0.05f, 0.05f) 
      case "cube" | "square" => 
        (0.02f, 0.02f) 
      case _ => 
        (0.015f, 0.01f) 
    }

    val ok = (absError <= maxAbsError) || (relError <= maxRelError)
    (ok, absError, relError)
  }

  def vpuOpToChiselEnum(vpuOp: String): VPUOp.Type = {
    vpuOp match {
      case "add"    => VPUOp.add
      case "sub"    => VPUOp.sub
      case "mul"    => VPUOp.mul
      case "rcp"    => VPUOp.rcp
      case "sqrt"   => VPUOp.sqrt
      case "sin"    => VPUOp.sin
      case "cos"    => VPUOp.cos
      case "tanh"   => VPUOp.tanh
      case "log"    => VPUOp.log
      case "exp"    => VPUOp.exp
      case "exp2"   => VPUOp.exp2
      case "square" => VPUOp.square
      case "cube"   => VPUOp.cube
      case "max"    => VPUOp.max
      case "reduSum"=> VPUOp.reduSum
      case "fp8"    => VPUOp.fp8
      case other    => 
        throw new IllegalArgumentException(s"Unsupported or unexpected vpuOp string found: '$other'")
    }
  }

  // -------------------------------------------------------------------------
  // Updated Helper Functions for New IO
  // -------------------------------------------------------------------------

  def initializeInterface(dut: VectorEngine, drainCycles: Int = 4): Unit = {
    dut.io.in1.valid.poke(false.B)
    dut.io.in2.valid.poke(false.B)
    
    // Always ready to accept output on our unified streams
    dut.io.out1.ready.poke(true.B)
    dut.io.out2.ready.poke(true.B)

    for (i <- 0 until dut.p.numLanes) {
      // Changed vectorInputData to vector based on new VPUInput definition
      dut.io.in1.bits.vector(i).poke(0.U)
      dut.io.in2.bits.vector(i).poke(0.U)
    }
    
    if (drainCycles > 0) dut.clock.step(drainCycles)
  }

  // Test body
  "Vector Engine" should "match Python ground truth for fractional FP vectors" in {
    val p = VPUParams()
    val vectors = loadVectors(vectorResource)
    require(vectors.nonEmpty, "No test vectors found. Run gen_vpu_vectors.py first.")

    simulate(new VectorEngine(p)) { dut =>
      var passed = 0
      var failed = 0
      
      val outFile = new java.io.PrintWriter("../../../../../src/test/resources/rtl_actual_outputs.txt")
      outFile.println(
        f"# ${"id"}%6s  ${"case_desc"}%-32s ${"lane"}%4s ${"vpuOp"}%-8s " +
        f"${"actual_hex"}%12s ${"expected_hex"}%12s " +
        f"${"actual_float"}%14s ${"expected_float"}%14s ${"RelErr%"}%8s ${"match"}%-6s " +
        f"| ${"inA_hex"}%8s ${"inA_float"}%10s ${"inB_hex"}%10s ${"inB_float"}%10s"
      )

      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)
      
      for (tv <- vectors) {

        // 0. Initialize Interface
        initializeInterface(dut, drainCycles=0)
        
        // 1. Setup and Fire Input
        dut.io.in1.valid.poke(true.B)
        dut.io.in2.valid.poke(true.B) // Drive both streams so twoOpReady can resolve
        
        val opEnum = vpuOpToChiselEnum(tv.vpuOp)
        dut.io.in1.bits.instType.poke(opEnum)
        dut.io.in2.bits.instType.poke(opEnum)
        
        // Defaulting isExpNeg to false unless you parse it from python
        dut.io.in1.bits.isExpNeg.poke(false.B)
        dut.io.in2.bits.isExpNeg.poke(false.B)

        // Vector A to in1, Vector B to in2
        for (i <- 0 until dut.p.numLanes) {
          // Changed vectorInputData to vector based on new VPUInput definition
          dut.io.in1.bits.vector(i).poke(tv.vecA(i).U)
          dut.io.in2.bits.vector(i).poke(tv.vecB(i).U)
        }

        // Wait for in1.ready
        var cycles = 0
        while (!dut.io.in1.ready.peek().litToBoolean && cycles < 64) {
          dut.clock.step(1)
          cycles += 1
        }
        dut.io.in1.ready.expect(true.B, s"Timed out waiting for in1.ready on test vector: # ${tv.id}\nVector Details:\n${tv.toFlatFormat}\n")
        dut.clock.step(1) 
        
        // Drop valid so we don't double-fire
        dut.io.in1.valid.poke(false.B) 
        dut.io.in2.valid.poke(false.B) 

        // 2. Wait for Output Valid on out1 (now driven by the RRArbiter)
        cycles = 0
        val timeoutLimit = 1000
        while (!dut.io.out1.valid.peek().litToBoolean && cycles < timeoutLimit) {
          dut.clock.step(1)
          cycles += 1
        }
        
        assert(dut.io.out1.valid.peek().litToBoolean, s"Timed out waiting for out1.valid on unit resolving ${tv.vpuOp} (case id: ${tv.id})")
        
        // 3. Check Outputs
        var caseOk = true

        for (r <- 0 until p.numLanes) {
          
          // Read from our arbitrated out1 stream
          val actual = dut.io.out1.bits.vectorOutputData(r).peek().litValue
          val expected = tv.expected(r)
          
          val aFloat = bf16BitsToFloat(actual.toInt & 0xFFFF)
          val eFloat = bf16BitsToFloat(expected & 0xFFFF)
          
          val (ok, absError, relError) = checkTolerance(actual, expected, tv.vpuOp)
          val relErrorPercent = relError * 100.0f

          val aHexStr = f"0x${actual.toInt & 0xFFFF}%04x"
          val eHexStr = f"0x${expected & 0xFFFF}%04x"
          val matchStr = if (ok) "PASS" else "FAIL"

          val failContext = if (!ok) { 
            val inAHex = tv.vecA(r) & 0xFFFF
            val inBHex = tv.vecB(r) & 0xFFFF
            val inAFloat = bf16BitsToFloat(inAHex)
            val inBFloat = bf16BitsToFloat(inBHex)
            
            val inAHexStr = f"0x$inAHex%04x"
            val inBHexStr = f"0x$inBHex%04x"
            
            f"| $inAHexStr%8s $inAFloat%10.4f $inBHexStr%10s $inBFloat%10.4f"
          } else {
            "" 
          }

          outFile.println(
            f"  ${tv.id}%6d  ${"\"" + tv.desc + "\""}%-32s $r%4d ${tv.vpuOp}%-8s " +
            f"$aHexStr%12s $eHexStr%12s " +
            f"$aFloat%14.4f $eFloat%14.4f ${f"$relErrorPercent%.2f%%"}%8s $matchStr%-6s " + failContext
          )

          if (!ok) {
            caseOk = false
            println(f"  FAIL case ${tv.id} [${tv.desc}] lane $r: got 0x${actual.toInt & 0xFFFF}%04x ($aFloat%.4f), expected 0x${expected & 0xFFFF}%04x ($eFloat%.4f) [RelErr: $relErrorPercent%.2f%%, AbsErr: $absError%.4f]")
          }
        }

        if (caseOk) passed += 1 else failed += 1

        // 4. Complete Output Handshake (Arbiter completes pop)
        dut.clock.step(1)
        outFile.println()
      }

      outFile.close()
      println(s"\nWrote sp26-atlas-acc/src/test/resources/rtl_actual_outputs.txt")
      println(s"Vector test results: $passed passed, $failed failed out of ${vectors.length}")

      failed shouldBe 0
    }
  }
}