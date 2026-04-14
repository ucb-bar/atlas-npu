package atlas.vector

import chisel3._
import atlas.common.VpuParams
import sp26FPUnits.hardfloat._
import fpex.FPType.BF16T
import scala.math._
import scala.util.Random

object VectorTestUtils {
    // Float -> BF16 as Short (16 bits)
    def fpToBf16(f: Float): Short = {
        val x = java.lang.Float.floatToRawIntBits(f)
        val lsb = (x >>> 16) & 1
        val roundingBias = 0x7FFF + lsb
        val bf = ((x + roundingBias) >>> 16) & 0xFFFF
        bf.toShort   // cast down to 16-bit
    }

    // BF16 as Short (16 bits) to Float
    def bf16ToFp(b: Short): Float = {
        val intVal = (b & 0xFFFF) << 16   // avoid sign extension
        java.lang.Float.intBitsToFloat(intVal)
    }

    // Generate an array of 16 BF16 of the same value as Short (16 bits)
    def constantBF16Vec(n: Int = 16, value: Float = 0.0f): Array[Short] = {
        val bf16Val = fpToBf16(value)
        Array.fill(n)(bf16Val)
    }

    // Generate a matrix of 32 * 16 BF16 of the same value as Short (16 bits)
    def constantBF16Matrix(lanes: Int = 32, laneWidth: Int = 16, value: Float = 0.0f): Array[Array[Short]] = {
        Array.fill(lanes)(constantBF16Vec(laneWidth, value))
    }

    // Generate an array of 16 BF16 as Short (16 bits) range in [min, max)
    def randomBF16Vec(n: Int = 16, max: Int = 10, min: Int = -10, useInt: Boolean = false): Array[Short] = {
        Array.fill(n) {val f = if (useInt) {(min + Random.nextInt(max - min)).toFloat} else {min + Random.nextFloat() * (max - min)}
            fpToBf16(f)
        }
    }

    // Generate a matrix of 32 * 16 BF16 of the random values rannge in [min, max)
    def randomBF16Matrix(lanes: Int = 32, laneWidth: Int = 16, max: Int = 10, min: Int = -10, useInt: Boolean = false): Array[Array[Short]] = {
        Array.fill(lanes) {
            randomBF16Vec(n = laneWidth, max = max, min = min, useInt = useInt)
        }
    }

    // Conver a vector of (16) of BF 16 to UInt(256.W)
    // Example: Seq(A, B, C, D) => [D][C][B][A]
    def packBF16Array(xs: Array[Short]): BigInt = {
        require(xs.length == 16)
        xs.zipWithIndex.foldLeft(BigInt(0)) {
            case (acc, (x, i)) =>
                acc | (BigInt(x & 0xFFFF) << (i * 16))
        }
    }

    // Convert UInt(256.W) to a vector (16) of BF16
    // Example: [A][B][C][D] => Seq(A, B, C, D)
    def unpackBF16Array(x: BigInt): Array[Short] = {
        Array.tabulate(16) { i =>
            val shift = i * 16
            ((x >> shift) & 0xFFFF).toShort
        }
    }

    // Pack the BF16 matrix to appropriate format for VET
    def packMatrix(mat: Array[Array[Short]]): Array[BigInt] = {
        require(mat.length == 32, s"Matrix must have 32 lanes, got ${mat.length}")

        mat.zipWithIndex.foreach { case (row, i) =>
            require(row.length == 16, s"Lane $i must have 16 elements, got ${row.length}")
        }

        mat.map(packBF16Array)
    }

    // Unpack the BF16 matrix to float
    def unpackMatrix(packed: Array[BigInt]): Array[Array[Short]] = {
        require(packed.length == 32, s"Matrix must have 32 rows, got ${packed.length}")
        packed.map(unpackBF16Array)
    }

    // Print a vector of BF16
    def printBF16Array(arr: Array[Short], expected: Boolean): Unit = {
        val prefix = if (expected) "Expected" else "Actual  "
        val floatRow = arr.map(bf16ToFp)
        val formattedRow = floatRow.map(x => f"$x%8.4f").mkString(", ")     // Change the printing format, adjust if necessary
        println(s"$prefix: $formattedRow") 
    }

    // Print a row of BF16 within a matrix
    def printLane(mat: Array[Array[Short]], laneIdx: Int, expected: Boolean): Unit = {
        require(mat.length == 32, s"Matrix must have 32 lanes, got ${mat.length}")
        require(laneIdx >= 0 && laneIdx < 32, s"Invalid lane index: $laneIdx")

        val row = mat(laneIdx)
        require(row.length == 16, s"Lane $laneIdx must have 16 elements, got ${row.length}")

        val prefix = if (expected) "Expected" else "Actual  "
        val floatRow = row.map(bf16ToFp)
        val formattedRow = floatRow.map(x => f"$x%8.4f").mkString(", ")     // Change the printing format, adjust if necessary
        println(s"$prefix Lane $laneIdx: $formattedRow")    
    }

    // Print the value of the whole matrix (32 * 16 BF16)
    def printMatrixAsFloat(mat: Array[Array[Short]], expected: Boolean): Unit = {
        require(mat.length == 32, s"Matrix must have 32 lanes, got ${mat.length}")

        if (expected) {
            println("The expected values are: ")
        } else {
            println("The actual   values are: ")
        }

        mat.zipWithIndex.foreach { case (row, i) =>
            require(row.length == 16, s"Lane $i must have 16 elements, got ${row.length}")

            val floatRow = row.map(bf16ToFp)
            println(s"Lane $i: " + floatRow.mkString(", "))
        }
    }
    
    /* Golden functions operate on individual variable:
     *  0. Note that BF16 is interpreted as Short
     *  1. Convert BF16 to Float
     *  2. Do operation in Float
     *  3. Convert Float to BF16
     */
    def goldenAddSub(a: Short, b: Short, isSub: Boolean): Short = {
        val aFP = bf16ToFp(a)
        val bFP = bf16ToFp(b)
        val y = if (isSub) aFP - bFP else aFP + bFP
        fpToBf16(y)
    }   

    def goldenMul(a: Short, b: Short): Short = {
        val aFP = bf16ToFp(a)
        val bFP = bf16ToFp(b)
        val y  = aFP * bFP
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    def goldenRcpSqrt(a: Short, isSqrt: Boolean): Short = {
        val aFP = bf16ToFp(a)
        val y  = if (isSqrt) Math.sqrt(aFP) else (1 / aFP)
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    def goldenSinCos(a: Short, isCos: Boolean): Short = {
        val aFP = bf16ToFp(a)
        val y  = if (isCos) Math.cos(aFP) else Math.sin(aFP)
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    def goldenTanh(a: Short): Short = {
        val aFP = bf16ToFp(a)
        val y  = Math.tanh(aFP)
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    def goldenLog(a: Short): Short = {
        val aFP = bf16ToFp(a)
        val y  = Math.log(aFP) / Math.log(2)
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    def goldenExp(a: Short, isExp2: Boolean): Short = {
        val aFP = bf16ToFp(a)
        val y  = if (isExp2) Math.pow(2, aFP) else Math.exp(aFP)
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    def goldenSqureCube(a: Short, isCube: Boolean): Short = {
        val aFP = bf16ToFp(a)
        val y  = if (isCube) aFP * aFP * aFP else aFP * aFP 
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    def goldenFP8(a: Short, expShift: Int): Short = {
        def packE4M3(sign: Int, exp: Int, mant: Int): Short = {
            (((sign & 1) << 7) | ((exp & 0xF) << 3) | (mant & 0x7)).toShort
        }

        val sign = (a >>> 15) & 1
        val expBF = (a >>> 7) & 0xFF
        val mantBF = a & 0x7F

        val isZero = (expBF == 0) && (mantBF == 0)
        if (isZero) return 0.toShort

        // Subtract expShift before converting the BF16 exponent into E4M3.
        val expAdj = expBF - 120 - expShift
        val expClamped = if (expAdj <= 0) 0.toShort
                         else if (expAdj >= 15) 15.toShort
                         else expAdj

        val mantFP8 = (mantBF >>> 4) & 0x7
        packE4M3(sign, expClamped, mantFP8)
    }

    def goldenRelu(x: Short): Short = {
        val xFP = bf16ToFp(x)
        val y  = if (xFP < 0.0) 0.0 else xFP
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    def goldenPairwiseMax(a: Short, b: Short): Short = {
        val aFP = bf16ToFp(a)
        val bFP = bf16ToFp(b)
        val y  = math.max(aFP, bFP)
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    def goldenPairwiseMin(a: Short, b: Short): Short = {
        val aFP = bf16ToFp(a)
        val bFP = bf16ToFp(b)
        val y  = math.min(aFP, bFP)
        val yFloat = y.toFloat
        fpToBf16(yFloat)
    }

    // ---------------------------------------------------------------------------------------
    // Golden functions operate on a vector (16) of variables (BF16 interpreted as Short)
    // ---------------------------------------------------------------------------------------
    def goldenAddSubVector(aVec: Array[Short], bVec: Array[Short], isSub: Boolean): Array[Short] = {
        require(aVec.length == 16 && bVec.length == 16)
        Array.tabulate(16) { i => goldenAddSub(aVec(i), bVec(i), isSub) }
    }

    def goldenMulVector(aVec: Array[Short], bVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16 && bVec.length == 16)
        Array.tabulate(16) { i => goldenMul(aVec(i), bVec(i)) }
    }

    def goldenRcpSqrtVector(aVec: Array[Short], isSqrt: Boolean): Array[Short] = {
        require(aVec.length == 16)
        Array.tabulate(16) { i => goldenRcpSqrt(aVec(i), isSqrt) }
    }

    def goldenSinCosVector(aVec: Array[Short], isCos: Boolean): Array[Short] = {
        require(aVec.length == 16)
        Array.tabulate(16) { i => goldenSinCos(aVec(i), isCos) }
    }

    def goldenTanhVector(aVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16)
        Array.tabulate(16) { i => goldenTanh(aVec(i)) }
    }

    def goldenLogVector(aVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16)
        Array.tabulate(16) { i => goldenLog(aVec(i)) }
    }

    def goldenExpVector(aVec: Array[Short], isExp2: Boolean): Array[Short] = {
        require(aVec.length == 16)
        Array.tabulate(16) { i => goldenExp(aVec(i), isExp2) }
    }

    def goldenSquareCubeVector(aVec: Array[Short], isCube: Boolean): Array[Short] = {
        require(aVec.length == 16)
        Array.tabulate(16) { i => goldenSqureCube(aVec(i), isCube) }
    }

    def goldenMaxVector(aVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16)

        // Find the maximum value (convert to FP for correct comparison)
        val maxVal = aVec.map(bf16ToFp).max
        val maxBf16 = fpToBf16(maxVal)

        // Output: first element = max, rest = 0
        Array.tabulate(16) { i =>
            if (i == 0) maxBf16 else 0.toShort
        }
    }

    def goldenReduSumVector(aVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16)

        val sumFP = aVec.map(bf16ToFp).sum
        val sumBF16 = fpToBf16(sumFP.toFloat)

        Array.tabulate(16) { i =>
            if (i == 0) sumBF16 else 0.toShort
        }
    }

    def goldenFP8Vector(aVec: Array[Short], expShift: Int): Array[Short] = {
        require(aVec.length == 16)
        Array.tabulate(16) { i => goldenFP8(aVec(i), expShift) }
    }

    def goldenReluVector(aVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16)
        Array.tabulate(16) { i => goldenRelu(aVec(i)) }
    }

    def goldenPairwiseMaxVector(aVec: Array[Short], bVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16 && bVec.length == 16)
        Array.tabulate(16) { i => goldenPairwiseMax(aVec(i), bVec(i)) }
    }

    def goldenPairwiseMinVector(aVec: Array[Short], bVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16 && bVec.length == 16)
        Array.tabulate(16) { i => goldenPairwiseMin(aVec(i), bVec(i)) }
    }

    def goldenRowMaxVector(aVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16)

        // Find the maximum value (convert to FP for correct comparison)
        val maxVal = aVec.map(bf16ToFp).max
        val maxBf16 = fpToBf16(maxVal)

        // Output: all elements = max
        Array.fill(16)(maxBf16)
    }

    def goldenRowMinVector(aVec: Array[Short]): Array[Short] = {
        require(aVec.length == 16)

        // Find the minimum value (convert to FP for correct comparison)
        val minVal = aVec.map(bf16ToFp).min
        val minBf16 = fpToBf16(minVal)

        // Output: all elements = min
        Array.fill(16)(minBf16)
    }

    // ---------------------------------------------------------------------------------------
    // Golden functions operate on a matrix (32 * 16) of variables (BF16 interpreted as Short)
    // ---------------------------------------------------------------------------------------
    def goldenAddSubMatrix(aMat: Array[Array[Short]], bMat: Array[Array[Short]], isSub: Boolean): Array[Array[Short]] = {
        require(aMat.length == bMat.length)
        aMat.zip(bMat).map { case (aLane, bLane) =>
            require(aLane.length == 16 && bLane.length == 16)
            goldenAddSubVector(aLane, bLane, isSub)
        }
    }

    def goldenMulMatrix(aMat: Array[Array[Short]], bMat: Array[Array[Short]]): Array[Array[Short]] = {
        require(aMat.length == bMat.length)
        aMat.zip(bMat).map { case (aLane, bLane) =>
            require(aLane.length == 16 && bLane.length == 16)
            goldenMulVector(aLane, bLane)
        }
    }

    def goldenRcpSqrtMatrix(aMat: Array[Array[Short]], isSqrt: Boolean): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenRcpSqrtVector(lane, isSqrt)
        }
    }

    def goldenSinCosMatrix(aMat: Array[Array[Short]], isCos: Boolean): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenSinCosVector(lane, isCos)
        }
    }

    def goldenTanhMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenTanhVector(lane)
        }
    }

    def goldenLogMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenLogVector(lane)
        }
    }

    def goldenExpMatrix(aMat: Array[Array[Short]], isExp2: Boolean): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenExpVector(lane, isExp2)
        }
    }

    def goldenSquareCubeMatrix(aMat: Array[Array[Short]], isCube: Boolean): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenSquareCubeVector(lane, isCube)
        }
    }

    def goldenMaxMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenMaxVector(lane)
        }
    }

    def goldenReduSumMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenReduSumVector(lane)
        }
    }

    def goldenFP8Matrix(aMat: Array[Array[Short]], expShift: Int): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenFP8Vector(lane, expShift)
        }
    }

    def goldenReluMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenReluVector(lane)
        }
    }

    def goldenPairwiseMaxMatrix(aMat: Array[Array[Short]], bMat: Array[Array[Short]]): Array[Array[Short]] = {
        require(aMat.length == bMat.length)
        aMat.zip(bMat).map { case (aLane, bLane) =>
            require(aLane.length == 16 && bLane.length == 16)
            goldenPairwiseMaxVector(aLane, bLane)
        }
    }

    def goldenPairwiseMinMatrix(aMat: Array[Array[Short]], bMat: Array[Array[Short]]): Array[Array[Short]] = {
        require(aMat.length == bMat.length)
        aMat.zip(bMat).map { case (aLane, bLane) =>
            require(aLane.length == 16 && bLane.length == 16)
            goldenPairwiseMinVector(aLane, bLane)
        }
    }

    def goldenRowMaxMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenRowMaxVector(lane)
        }
    }

    def goldenRowMinMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
        aMat.map { lane =>
            require(lane.length == 16)
            goldenRowMinVector(lane)
        }
    }

    def goldenColMaxMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
        require(aMat.length == 32, s"Matrix must have 32 lanes, got ${aMat.length}")
        aMat.zipWithIndex.foreach { case (row, i) =>
            require(row.length == 16, s"Lane $i must have 16 elements, got ${row.length}")
        }

        // Step 1: find max of each column
        val colMaxVec: Array[Short] = Array.tabulate(16) { col =>
            val maxVal = aMat.map(row => bf16ToFp(row(col))).max
            fpToBf16(maxVal)
        }

        // Step 2: replicate that vector to all rows
        Array.fill(32)(colMaxVec.clone())
    }

    def goldenColMinMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
    require(aMat.length == 32, s"Matrix must have 32 lanes, got ${aMat.length}")
    aMat.zipWithIndex.foreach { case (row, i) =>
        require(row.length == 16, s"Lane $i must have 16 elements, got ${row.length}")
    }

    // Step 1: find min of each column
    val colMinVec: Array[Short] = Array.tabulate(16) { col =>
        val minVal = aMat.map(row => bf16ToFp(row(col))).min
        fpToBf16(minVal)
    }

    // Step 2: replicate that vector to all rows
    Array.fill(32)(colMinVec.clone())
}

    def goldenColSumMatrix(aMat: Array[Array[Short]]): Array[Array[Short]] = {
        require(aMat.length == 32, s"Matrix must have 32 lanes, got ${aMat.length}")
        
        // 1. Initialize an accumulator for the 16 columns
        val colSums = Array.fill(16)(0.0f)

        // 2. Sum across the 32 rows for each column
        for (row <- 0 until 32) {
            for (col <- 0 until 16) {
            val bf16Bits = aMat(row)(col)
            // Convert BF16 bits (Short) to Float
            // BF16 is just the upper 16 bits of a 32-bit Float
            val f32 = java.lang.Float.intBitsToFloat(bf16Bits << 16)
            colSums(col) += f32
            }
        }

        // 3. Convert sums back to BF16 bits
        val resultRow = colSums.map { f =>
            val bits = java.lang.Float.floatToIntBits(f)
            (bits >> 16).toShort
        }

        // 4. Broadcast: Fill all 32 rows with the resulting sum vector
        Array.fill(32)(resultRow)
    }


    // ---------------------------------------------------------------------------------------
    // Functions checking the tolerance on variable, vector (16) and matrix(32 * 16)
    // ---------------------------------------------------------------------------------------
    /* Single Tolerance Checker: 
     *  1. Convert BF16 (Interpreted as Short) of actual and expected valeus to Float
     *  2. Check for special case of Float
     *  3. Determine absolute and relative error
     *  4. Returns (pass, absError, relError)
     */
    def checkTolerance(
        actual: Short,
        expected: Short,
        absTolerance: Float = 0.01f,    // Default Tolerance for now, adjust later
        relTolerance: Float = 0.01f     // Default Tolerance for now, adjust later
    ): (Boolean, Float, Float) = {
        val aFloat = bf16ToFp(actual)
        val eFloat = bf16ToFp(expected)

        if (aFloat.isNaN && eFloat.isNaN) return (true, 0.0f, 0.0f)
        if (aFloat.isPosInfinity && eFloat.isPosInfinity) return (true, 0.0f, 0.0f)
        if (aFloat.isNegInfinity && eFloat.isNegInfinity) return (true, 0.0f, 0.0f)

        val absError = math.abs(aFloat - eFloat).toFloat
        val relError =
            if (eFloat == 0.0f) {
                if (aFloat == 0.0f) 0.0f else Float.PositiveInfinity
            } else {
                (absError / math.abs(eFloat)).toFloat
            }

        val pass = (absError <= absTolerance) || (relError <= relTolerance)
        (pass, absError, relError)
    }

    /* Vector Tolerance Checker: 
     *  1. Check the length of the actual and expcted arrays match
     *  2. Call checkTolerance for each pair of element in the arrays
     *  3. Print message of any of the pairs fail
     *  4. Print the expected and the actual vector if not all values pass tolerance check
     *  5. Returns allPass
     */
    def checkVectorTolerance(
        actual: Array[Short],
        expected: Array[Short],
        printPass: Boolean = false,
        printPrecision: Boolean = false,
        label: String = "Vector",
        absTolerance: Float = 0.01f,    // Default Tolerance for now, adjust later
        relTolerance: Float = 0.01f     // Default Tolerance for now, adjust later
    ): Boolean = {
        require(
            actual.length == expected.length,
            s"$label length mismatch: actual=${actual.length}, expected=${expected.length}"
        )

        var allPass = true

        for (i <- actual.indices) {
            val (pass, absError, relError) =
                checkTolerance(
                    actual(i),
                    expected(i),
                    absTolerance,
                    relTolerance
                )

            if (!pass) {
                val aFloat = bf16ToFp(actual(i))
                val eFloat = bf16ToFp(expected(i))

                println(
                    f"[$label] MISMATCH at idx=$i: expected=$eFloat%.4f actual=$aFloat%.4f " +
                    f"absError=$absError%.4f relError=$relError%.4f " +
                    f"absTol=$absTolerance%.4f relTol=$relTolerance%.4f"
                )
                allPass = false
            }

            if (printPrecision) {
                val aFloat = bf16ToFp(actual(i))
                val eFloat = bf16ToFp(expected(i))

                println(
                    f"[$label] Precision at idx=$i: expected=$eFloat%.4f actual=$aFloat%.4f " +
                    f"absError=$absError%.4f relError=$relError%.4f " +
                    f"absTol=$absTolerance%.4f relTol=$relTolerance%.4f"
                )
            }
        }

        if (!allPass) {
            val expectedStr = expected.map(bf16ToFp).map(x => f"$x%8.4f")mkString(", ") // Change the printing format, adjust if necessary
            val actualStr   = actual.map(bf16ToFp).map(x => f"$x%8.4f")mkString(", ")   // Change the printing format, adjust if necessary

            println(s"[$label] Expected: $expectedStr")
            println(s"[$label] Actual  : $actualStr")
        }
        if (allPass && printPass) {
            println(s"[$label] PASSED tolerance check")
        }

        allPass
    }

    /* Matrix Tolerance Checker: 
     *  1. Check the size of the actual and expcted matrix match
     *  2. Call checkTolerance for each pair of arrays in the matrix
     *  3. Print message of any of the pairs fail (Optional: only if printOnFailOnly set to true)
     *  4. Print fail/pass message at the end
     *  5. Returns allPass
     */
    def checkMatrixTolerance(
        actual: Array[Array[Short]],
        expected: Array[Array[Short]],
        label: String = "Matrix",
        absTolerance: Float = 0.01f,    // Default Tolerance for now, adjust later
        relTolerance: Float = 0.01f,    // Default Tolerance for now, adjust later
        printOnFailOnly: Boolean = true
    ): Boolean = {

        require(actual.length == expected.length,
            s"$label lane count mismatch: actual=${actual.length}, expected=${expected.length}")

        var allPass = true

        for (lane <- actual.indices) {
            val pass = checkVectorTolerance(
                actual = actual(lane),
                expected = expected(lane),
                label = s"$label lane $lane",
                absTolerance = absTolerance,
                relTolerance = relTolerance
            )

            if (!pass) {
                allPass = false

                // Only print full lane if requested
                // if (printOnFailOnly) {
                //     printLane(expected, lane, true)
                //     printLane(actual, lane, false)
                //     println("")
                // }
            }
        }

        if (allPass) {
            println(s"$label PASSED tolerance check")
        } else {
            println(s"$label FAILED tolerance check")
        }

        allPass
    }

}
