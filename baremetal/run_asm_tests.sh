#!/usr/bin/env bash
set -u
set -o pipefail

# Usage:
#   ./run_asm_tests.sh test1 test2 test3
#   ./run_asm_tests.sh --all
#
# Behavior:
#   - If generators/gen_<test>.py exists, the script generates a golden JSON
#     and assembles with --golden-json.
#   - Otherwise, it assembles without golden checks.
#   - After generating the C source, it re-runs CMake so newly created tests
#     are visible to the build system.
#   - It builds the whole build directory instead of one target.
#   - A test is marked failed if:
#       * any build step exits nonzero, or
#       * the simulator log contains explicit failure signatures even if
#         make/sim exits with code 0.
#
# Assumes this script lives at:
#   chipyard/generators/sp26-atlas-acc/baremetal/run_asm_tests.sh

RUN_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      RUN_ALL=1
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
echo

PASSED=()
FAILED=()

run_one_test() {
  local test_name="$1"

  local asm_file="$ASSEMBLY_DIR/${test_name}.S"
  local gen_py="$GENERATORS_DIR/gen_${test_name}.py"
  local golden_json="$GENERATORS_DIR/${test_name}.json"
  local binary="$BUILD_DIR/atlas_${test_name}.riscv"
  local sim_log
  local sim_rc

  echo "============================================================"
  echo "Running test: $test_name"
  echo "============================================================"

  if [[ ! -f "$asm_file" ]]; then
    echo "Missing assembly file: $asm_file"
    return 1
  fi

  if [[ -f "$gen_py" ]]; then
    echo "[1/5] Generating golden JSON"
    (
      cd "$BAREMETAL_DIR" &&
      python3 "$gen_py" > "$golden_json"
    ) || return 1

    echo "[2/5] Assembling with golden JSON"
    (
      cd "$BAREMETAL_DIR" &&
      python3 assembler.py "assembly/${test_name}.S" \
        --out-c "tests/atlas_${test_name}.c" \
        --golden-json "generators/${test_name}.json"
    ) || return 1
  else
    echo "[1/4] Assembling (no golden generator)"
    (
      cd "$BAREMETAL_DIR" &&
      python3 assembler.py "assembly/${test_name}.S" \
        --out-c "tests/atlas_${test_name}.c"
    ) || return 1
  fi

  if [[ -f "$gen_py" ]]; then
    echo "[3/5] Reconfiguring CMake"
  else
    echo "[2/4] Reconfiguring CMake"
  fi
  cmake -S "$TESTS_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug || return 1

  if [[ -f "$gen_py" ]]; then
    echo "[4/5] Building tests"
  else
    echo "[3/4] Building tests"
  fi
  cmake --build "$BUILD_DIR" || return 1

  if [[ ! -f "$binary" ]]; then
    echo "Missing binary: $binary"
    return 1
  fi

  if [[ -f "$gen_py" ]]; then
    echo "[5/5] Running simulation"
  else
    echo "[4/4] Running simulation"
  fi

  sim_log="$(mktemp)"
  (
    cd "$VCS_DIR" &&
    make run-binary GEN_COVERAGE=1 \
      BINARY="$binary" \
      CONFIG="$CONFIG" \
      LOADMEM=1
  ) 2>&1 | tee "$sim_log"
  sim_rc=${PIPESTATUS[0]}

  if [[ $sim_rc -ne 0 ]]; then
    echo "Simulation command failed with exit code $sim_rc"
    rm -f "$sim_log"
    return 1
  fi

  if grep -Eq \
    'DRAM MISMATCH|FAIL:|Assertion failed: \*\*\* FAILED \*\*\*|Fatal:|FAILED \(\*+\)|\*\*\* FAILED \*\*\*' \
    "$sim_log"; then
    echo "Simulation reported a test failure"
    rm -f "$sim_log"
    return 1
  fi

  rm -f "$sim_log"
  return 0
}

for test_name in "${TEST_NAMES[@]}"; do
  if run_one_test "$test_name"; then
    PASSED+=("$test_name")
  else
    FAILED+=("$test_name")
  fi
  echo
done

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
