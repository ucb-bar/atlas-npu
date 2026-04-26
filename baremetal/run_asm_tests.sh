#!/usr/bin/env bash
set -u
set -o pipefail

# Usage:
#   ./run_asm_tests.sh test1 test2 test3
#   ./run_asm_tests.sh --all
#   ./run_asm_tests.sh --parallel test1 test2 test3   # parallelize VCS sims
#   ./run_asm_tests.sh -jN  test1 test2 test3         # cap concurrency at N
#
# Warm-up: if the VCS simv isn't built yet (fresh clone, branch switch,
# stale generated-src), run ONE test serially first to build simv, then
# -jN the rest. Parallel sweeps race on simv regen and corrupt each other.
#
# Behavior:
#   - If generators/gen_<test>.py exists, the script generates a golden JSON
#     and assembles with --golden-json.
#   - Otherwise, it assembles without golden checks.
#   - Build phase (golden + assemble + CMake + build) runs serially once for
#     all tests. Simulations run either serially (default) or in parallel.
#   - A test is marked failed if:
#       * any build step exits nonzero, or
#       * the simulator log contains explicit failure signatures even if
#         make/sim exits with code 0.
#
# Assumes this script lives at:
#   chipyard/generators/sp26-atlas-acc/baremetal/run_asm_tests.sh

RUN_ALL=0
PARALLEL=0
MAX_JOBS=0   # 0 = unbounded (one job per test)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      RUN_ALL=1
      shift
      ;;
    --parallel)
      PARALLEL=1
      shift
      ;;
    -j*)
      MAX_JOBS="${1#-j}"
      PARALLEL=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

BAREMETAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHIPYARD_ROOT="$(cd "$BAREMETAL_DIR/../../.." && pwd)"

TESTS_DIR="$BAREMETAL_DIR/tests"
ASSEMBLY_DIR="$BAREMETAL_DIR/assembly"
GENERATORS_DIR="$BAREMETAL_DIR/generators"
BUILD_DIR="$TESTS_DIR/build"
VCS_DIR="$CHIPYARD_ROOT/sims/vcs"
CONFIG="EE290SimConfig"

if [[ ! -d "$VCS_DIR" ]]; then
  echo "Error: could not find Chipyard VCS directory at:"
  echo "  $VCS_DIR"
  exit 1
fi

TEST_NAMES=()

if [[ $RUN_ALL -eq 1 ]]; then
  while IFS= read -r file; do
    name="$(basename "$file" .S)"
    TEST_NAMES+=("$name")
  done < <(find "$ASSEMBLY_DIR" -maxdepth 1 -name "*.S" | sort)
else
  if [[ $# -eq 0 ]]; then
    echo "Error: provide test names or use --all"
    exit 1
  fi
  TEST_NAMES=("$@")
fi

if [[ ${#TEST_NAMES[@]} -eq 0 ]]; then
  echo "No tests found."
  exit 1
fi

echo "Baremetal dir: $BAREMETAL_DIR"
echo "Chipyard root: $CHIPYARD_ROOT"
echo "Config: $CONFIG"
echo "Tests: ${TEST_NAMES[*]}"
if [[ $PARALLEL -eq 1 ]]; then
  if [[ $MAX_JOBS -gt 0 ]]; then
    echo "Sim mode: parallel (max $MAX_JOBS concurrent)"
  else
    echo "Sim mode: parallel (all concurrent)"
  fi
else
  echo "Sim mode: serial"
fi
echo

# Holds the test-name / build-status pair.
# build_status[i]=0 means ok-to-sim, 1 means build failed (sim skipped).
declare -a BUILD_STATUS=()

# ── Phase 1: golden + assemble per test (serial) ──────────────────────
for test_name in "${TEST_NAMES[@]}"; do
  asm_file="$ASSEMBLY_DIR/${test_name}.S"
  gen_py="$GENERATORS_DIR/gen_${test_name}.py"

  echo "============================================================"
  echo "Build prep: $test_name"
  echo "============================================================"

  rc=0
  if [[ ! -f "$asm_file" ]]; then
    echo "  Missing assembly file: $asm_file"
    rc=1
  elif [[ -f "$gen_py" ]]; then
    echo "  Generating golden JSON"
    (
      cd "$BAREMETAL_DIR" &&
      python3 "$gen_py" > "$GENERATORS_DIR/${test_name}.json"
    ) || rc=1

    if [[ $rc -eq 0 ]]; then
      echo "  Assembling with golden JSON"
      (
        cd "$BAREMETAL_DIR" &&
        python3 assembler.py "assembly/${test_name}.S" \
          --out-c "tests/atlas_${test_name}.c" \
          --golden-json "generators/${test_name}.json"
      ) || rc=1
    fi
  else
    echo "  Assembling (no golden generator)"
    (
      cd "$BAREMETAL_DIR" &&
      python3 assembler.py "assembly/${test_name}.S" \
        --out-c "tests/atlas_${test_name}.c"
    ) || rc=1
  fi

  BUILD_STATUS+=("$rc")
done

# ── Phase 2: CMake configure + build ───────────────────────────────────
# Build only the atlas_<name> targets the user asked for (default `all`
# globs every assembly/*.S — 85+ binaries + objdump dumps — which is
# ~85× wasted work when the user only wants one test). With --all we
# still build everything. Parallelism follows the user's -jN cap when
# given, otherwise nproc — so -jN means "at most N concurrent in both
# build and sim."
echo "============================================================"
echo "CMake configure + build (shared)"
echo "============================================================"
cmake -S "$TESTS_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug || exit 1

if [[ ${MAX_JOBS:-0} -gt 0 ]]; then
  BUILD_PAR="$MAX_JOBS"
else
  BUILD_PAR="$(nproc)"
fi

if [[ $RUN_ALL -eq 1 ]]; then
  cmake --build "$BUILD_DIR" -j"$BUILD_PAR" || exit 1
else
  build_targets=()
  for i in "${!TEST_NAMES[@]}"; do
    [[ "${BUILD_STATUS[$i]}" -eq 0 ]] || continue
    build_targets+=("atlas_${TEST_NAMES[$i]}")
  done
  if [[ ${#build_targets[@]} -gt 0 ]]; then
    cmake --build "$BUILD_DIR" --target "${build_targets[@]}" -j"$BUILD_PAR" || exit 1
  fi
fi

# Verify binaries exist for each successfully-prepped test.
for i in "${!TEST_NAMES[@]}"; do
  [[ "${BUILD_STATUS[$i]}" -eq 0 ]] || continue
  bin="$BUILD_DIR/atlas_${TEST_NAMES[$i]}.riscv"
  if [[ ! -f "$bin" ]]; then
    echo "  Missing binary after build: $bin"
    BUILD_STATUS[$i]=1
  fi
done

# ── Phase 3: simulations ───────────────────────────────────────────────
# Each sim writes its own log. We check pass/fail from the log after the
# sim exits (matching the original grep-based signature detection).
SIM_LOG_DIR="$(mktemp -d)"
declare -a SIM_LOG=()
declare -a SIM_PID=()
declare -a SIM_EXIT=()

FAIL_RE='DRAM MISMATCH|FAIL:|Assertion failed: \*\*\* FAILED \*\*\*|Fatal:|FAILED \(\*+\)|\*\*\* FAILED \*\*\*'
PASS_RE='^PASS:|\*\*\* PASSED \*\*\*'

sim_one_test() {
  local test_name="$1"
  local bin="$BUILD_DIR/atlas_${test_name}.riscv"
  local log="$2"

  (
    cd "$VCS_DIR" &&
    make run-binary GEN_COVERAGE=1 \
      BINARY="$bin" \
      CONFIG="$CONFIG" \
      LOADMEM=1
  ) > "$log" 2>&1
}

# ── Throttle helper: block until outstanding PIDs drop below MAX_JOBS ──
# Per-PID wait + explicit exit-code capture. Works correctly even if a child
# fails — the Codex review flagged `wait -n ... || wait` as miscounting pending.
await_slot() {
  local -n _pids=$1
  local cap=$2
  while [[ ${#_pids[@]} -ge $cap ]]; do
    # wait for the oldest outstanding pid (FIFO drain; simple and correct).
    local pid=${_pids[0]}
    wait "$pid"
    EXIT_OF_PID[$pid]=$?
    _pids=("${_pids[@]:1}")
  done
}

declare -A EXIT_OF_PID

# Launch per test: background with throttle if --parallel and -jN; otherwise
# serial (start → wait → record exit).
running_pids=()
for i in "${!TEST_NAMES[@]}"; do
  tn="${TEST_NAMES[$i]}"
  if [[ "${BUILD_STATUS[$i]}" -ne 0 ]]; then
    SIM_LOG+=("")
    SIM_PID+=("0")
    SIM_EXIT+=("1")
    continue
  fi

  log="$SIM_LOG_DIR/${tn}.simlog"
  SIM_LOG+=("$log")

  if [[ $PARALLEL -eq 1 ]]; then
    if [[ $MAX_JOBS -gt 0 ]]; then
      await_slot running_pids "$MAX_JOBS"
    fi
    echo "  [launch] $tn → $log"
    sim_one_test "$tn" "$log" &
    pid=$!
    SIM_PID+=("$pid")
    SIM_EXIT+=("0")   # overwritten after wait
    running_pids+=("$pid")
  else
    echo "  [serial] $tn → $log"
    sim_one_test "$tn" "$log"
    SIM_PID+=("0")
    SIM_EXIT+=("$?")
  fi
done

# Drain remaining parallel PIDs and capture each exit code explicitly.
if [[ $PARALLEL -eq 1 ]]; then
  echo "  Waiting for all sims to complete..."
  for pid in "${running_pids[@]}"; do
    wait "$pid"
    EXIT_OF_PID[$pid]=$?
  done
  # Back-fill SIM_EXIT for parallel tests from EXIT_OF_PID.
  for i in "${!TEST_NAMES[@]}"; do
    [[ "${BUILD_STATUS[$i]}" -eq 0 ]] || continue
    p="${SIM_PID[$i]}"
    if [[ "$p" != "0" && -n "${EXIT_OF_PID[$p]:-}" ]]; then
      SIM_EXIT[$i]="${EXIT_OF_PID[$p]}"
    fi
  done
fi

# ── Phase 4: tally ─────────────────────────────────────────────────────
# A test PASSES only when:
#   1) build succeeded (BUILD_STATUS[i] == 0111
#   2) sim exit status was 0
#   3) log contains a PASS marker ("PASS: ..." or "*** PASSED ***")
#   4) log contains no FAIL_RE match
# Any other outcome is a FAIL.
PASSED=()
FAILED=()

for i in "${!TEST_NAMES[@]}"; do
  tn="${TEST_NAMES[$i]}"
  if [[ "${BUILD_STATUS[$i]}" -ne 0 ]]; then
    FAILED+=("$tn (build)")
    continue
  fi

  log="${SIM_LOG[$i]}"
  if [[ -z "$log" || ! -f "$log" ]]; then
    FAILED+=("$tn (no log)")
    continue
  fi

  # Show the per-test tail to keep diagnostic output on screen.
  echo "============================================================"
  echo "Sim log tail: $tn"
  echo "============================================================"
  tail -30 "$log"
  echo

  if [[ "${SIM_EXIT[$i]}" -ne 0 ]]; then
    FAILED+=("$tn (sim exit ${SIM_EXIT[$i]})")
    continue
  fi

  if grep -Eq "$FAIL_RE" "$log"; then
    FAILED+=("$tn (failure signature)")
    continue
  fi

  if ! grep -Eq "$PASS_RE" "$log"; then
    FAILED+=("$tn (no PASS marker)")
    continue
  fi

  PASSED+=("$tn")
done

rm -rf "$SIM_LOG_DIR"

echo "==================== SUMMARY ===================="
echo "Passed: ${#PASSED[@]}"
for t in "${PASSED[@]}"; do
  echo "  PASS  $t"
done

echo "Failed: ${#FAILED[@]}"
for t in "${FAILED[@]}"; do
  echo "  FAIL  $t"
done

if [[ ${#FAILED[@]} -ne 0 ]]; then
  exit 1
fi
