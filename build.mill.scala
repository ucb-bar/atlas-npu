import mill._
import mill.scalalib._
import mill.scalalib.scalafmt.ScalafmtModule
import mill.scalalib.TestModule.ScalaTest

object atlas extends ScalaModule with ScalafmtModule {
  def scalaVersion = "2.13.16"
  def scalacOptions = Seq(
    "-deprecation",
    "-feature",
    "-unchecked",
    "-language:reflectiveCalls"
  )
  
  // Override to use standard src/main/scala layout
  override def sources = Task.Sources(millSourcePath / os.up / "src" / "main" / "scala")
  
  // Chisel + plugin
  def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:7.0.0-RC4"
  )
  def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:7.0.0-RC4"
  )
  
  // ScalaTest-based test module (Mill 0.13 requires this)
  object test extends ScalaTests with ScalaTest with ScalafmtModule {
    def ivyDeps = super.ivyDeps() ++ Agg(
      ivy"org.scalatest::scalatest:3.2.19"
    )
    
    // Override to use standard src/test/scala layout
    override def sources = Task.Sources(millSourcePath / os.up / os.up / "src" / "test" / "scala")
  }
}
