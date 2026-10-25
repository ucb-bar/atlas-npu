package sp26FPUnits

import chisel3._
import chisel3.util._
import sp26FPUnits.hardfloat._
import sp26FPUnits.hardfloat.consts._

trait VectorParam {
    /** x^2 or x^3 of BF16 bits x, both exact, rounded once to BF16 as vmul rounds: nearest
      * even, with subnormals, NaN gives 0x7FC0. */
    def squareCube(x: UInt, isCube: Bool): UInt = {
        val raw  = rawFloatFromFN(8, 8, x)   // sig = 01.fffffff; exponent sExp - 256
        val m    = raw.sig(7, 0)
        // m^3, or m^2 * 2^8: 24 bits with the leading one in bit 21, 22 or 23.
        val prod = Mux(isCube, m * m * m, Cat(m * m, 0.U(8.W)))
        val low  = !prod(23, 22).orR
        // The exact power as a RawFloat(10, 23): sig / 2^22 * 2^(sExp - 1024).
        val exact = Wire(new RawFloat(10, 23))
        exact.isNaN  := raw.isNaN
        exact.isInf  := raw.isInf
        exact.isZero := raw.isZero
        exact.sign   := raw.sign && isCube
        exact.sExp   := (Mux(isCube, raw.sExp * 3.S + 257.S, raw.sExp * 2.S + 512.S) - low.zext)(11, 0).asSInt
        exact.sig    := Mux(low, prod << 1, prod)(23, 0)
        val round = Module(new RoundAnyRawFNToRecFN(10, 23, 8, 8, 0))
        round.io.invalidExc     := isSigNaNRawFloat(raw)
        round.io.infiniteExc    := false.B
        round.io.in             := exact
        round.io.roundingMode   := round_near_even
        round.io.detectTininess := tininess_afterRounding
        fNFromRecFN(8, 8, round.io.out)
    }

    def compareReturnMin(a: UInt, b: UInt): UInt = {
        // Returns a if a < b, else returns b. 
        val aOrdered = Mux(a(15) === 1.U, ~a, a ^ "h8000".U)
        val bOrdered = Mux(b(15) === 1.U, ~b, b ^ "h8000".U)
        Mux(aOrdered < bOrdered, a, b)
    }

    def compareReturnMax(a: UInt, b: UInt): UInt = {
        // Returns a if a > b, else returns b. 
        val aOrdered = Mux(a(15) === 1.U, ~a, a ^ "h8000".U)
        val bOrdered = Mux(b(15) === 1.U, ~b, b ^ "h8000".U)
        Mux(aOrdered > bOrdered, a, b)
    }
}
