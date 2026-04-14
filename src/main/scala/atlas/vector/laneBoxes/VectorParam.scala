package sp26FPUnits

import chisel3._
import chisel3.util._

trait VectorParam {
    def squareCube(sign: UInt, exp: UInt, fra: UInt, isCube: Bool): UInt = {
        val expW  = 8
        val fracW = 7
        val bias  = 127.S
        val squareFracWidth = (fracW + 1)* 2    // 16
        val cubeFracWidth = (fracW + 1) * 3     // 24

        // Real exponent and mantissa for the input number, square, and cube
        val realExp = exp.pad(10).asSInt - bias    // Need 2 extra bits to guard overflow
        val squareExp = realExp << 1
        val cubeExp = squareExp + realExp 

        val mantissa = Cat(1.U(1.W), fra)   // Assume normal, subnorm will be flushed later
        val squareMant = mantissa * mantissa
        val cubeMant = squareMant * mantissa

        // Extra exp for square and cube
        val squareExtraExp = Wire(SInt(10.W))
        when (squareMant(squareFracWidth-1)) { squareExtraExp := 1.S }
        .otherwise(squareExtraExp := 0.S)
        val cubeExtraExp = Wire(SInt(10.W))
        when (cubeMant(cubeFracWidth-1)) { cubeExtraExp := 2.S }
        .elsewhen (cubeMant(cubeFracWidth-2)) { cubeExtraExp := 1.S }
        .otherwise { cubeExtraExp := 0.S }

        // Adjusted exponent after adding extra exp and bias
        val adjustedSquareExp = squareExp + squareExtraExp + bias
        val adjustedCubeExp = cubeExp + cubeExtraExp + bias

        // Highest 1 bit index for square and cube mantissa to determine normalization shift
        val squareHighestIdx = Wire(UInt(log2Ceil(squareFracWidth).W))
        val cubeHighestIdx = Wire(UInt(log2Ceil(cubeFracWidth).W))
        squareHighestIdx := 0.U
        cubeHighestIdx := 0.U
        for (i <- 0 until squareFracWidth) {
            when(squareMant(i)) { squareHighestIdx := i.U }
        }
        for (i <- 0 until cubeFracWidth) {
            when(cubeMant(i)) { cubeHighestIdx := i.U }
        }

        // Shift so leading 1 lands in bit 7 of an 8-bit mantissa: [1].[7 frac bits]
        val squareShiftAmt = Wire(UInt(log2Ceil(squareFracWidth+1).W))
        val cubeShiftAmt = Wire(UInt(log2Ceil(cubeFracWidth+1).W))
        squareShiftAmt := Mux(squareHighestIdx > 7.U, squareHighestIdx - 7.U, 0.U)
        cubeShiftAmt := Mux(cubeHighestIdx > 7.U, cubeHighestIdx - 7.U, 0.U)

        // Get mantissa after shifting
        val squareMantPre = squareMant >> squareShiftAmt
        val cubeMantPre = cubeMant >> cubeShiftAmt
        val squareFrac = squareMantPre(6, 0)
        val cubeFrac = cubeMantPre(6, 0)

        // Check for special cases in the square results
        val isSquareResultZero = (adjustedSquareExp <= 0.S)
        val isSquareResultInf = (adjustedSquareExp >= 255.S)

        // Check for special cases in the cube results
        val isCubeResultZero = (adjustedCubeExp <= 0.S)
        val isCubeResultInf = (adjustedCubeExp >= 255.S)

        // Check for special cases in the inputs
        val isInputZero = ((exp === 0.U) && (fra === 0.U))
        val isInputSubnormal = (exp === 0.U) && (fra =/= 0.U)
        val isInputInf = (exp === 255.U) && (fra === 0.U)
        val isInputNaN = (exp === 255.U) && (fra =/= 0.U)

        val squareResult = Wire(UInt(16.W))
        val cubeResult = Wire(UInt(16.W))

        // Sign for square is always positive, sign for cube is same as input
        when (isInputZero || isInputSubnormal || isInputNaN || isSquareResultZero) {
          squareResult := Cat(0.U, 0.U(expW.W), 0.U(fracW.W)) 
        } .elsewhen(isInputInf || isSquareResultInf) {
          squareResult := Cat(0.U, Fill(expW, 1.U(1.W)), 0.U(fracW.W)) 
        } .otherwise {
          squareResult := Cat(0.U, adjustedSquareExp(expW-1,0), squareFrac)
        }

        when (isInputZero || isInputSubnormal || isInputNaN || isCubeResultZero) {
          cubeResult := Cat(sign, 0.U(expW.W), 0.U(fracW.W)) 
        } .elsewhen(isInputInf || isCubeResultInf) {
          cubeResult := Cat(sign, Fill(expW, 1.U(1.W)), 0.U(fracW.W)) 
        } .otherwise {
          cubeResult := Cat(sign, adjustedCubeExp(expW-1,0), cubeFrac)
        }

        val result = Mux(isCube, cubeResult, squareResult)
        result
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
