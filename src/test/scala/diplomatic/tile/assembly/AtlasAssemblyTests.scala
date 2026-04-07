/*
AtlasAssemblyTests.scala
ScalaTest suite that discovers and runs all assembly .S test programs.

Run all: (from the chipyard root)
  SBT_OPTS="-Xmx8G -Xss4M -XX:+UseG1GC" sbt "project sp26atlas" "testOnly atlas.tile.assembly.AtlasAssemblyTests"
*/

package atlas.tile.assembly

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import atlas.common.AtlasParams

class AtlasAssemblyTests extends AnyFlatSpec with Matchers {

  private val tp = AtlasParams()

  private val testFiles: Seq[String] = {
    val index = AssemblyTestRunner.loadSource("/assembly/index.txt")
    index.linesIterator
      .map(_.trim)
      .filter(l => l.nonEmpty && !l.startsWith("#"))
      .toSeq
  }

  private def category(file: String): String = {
    val name = file.stripSuffix(".S")
    if      (name.startsWith("mxu"))     "MXU"
    else if (name.startsWith("xlu_"))    "XLU"
    else if (name.startsWith("dma_"))    "DMA"
    else if (name.startsWith("alu_"))    "Scalar"
    else if (name.startsWith("csr_"))    "Scalar"
    else if (name.startsWith("branch_")) "Scalar"
    else if (name.startsWith("jal_"))    "Scalar"
    else if (name.startsWith("wkld_"))   "Workload"
    else if (name.startsWith("pipe_"))   "Pipeline"
    else                                 "General"
  }

  for (file <- testFiles) {
    val testName = file.stripSuffix(".S")
    val cat      = category(file)

    s"[$cat] $testName" should "pass" in {
      val source        = AssemblyTestRunner.loadSource(s"/assembly/$file")
      val (passed, msg) = AssemblyTestRunner.runTest(tp, source, testName, verbose = true)
      assert(passed, msg)
    }
  }
}
