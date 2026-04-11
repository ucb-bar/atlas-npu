#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_permute_tests.sh — Generate random Atlas tests and run them in baremetal.
#
# Writes .S directly to baremetal/assembly/, gen_*.py to baremetal/generators/,
# then calls run_asm_tests.sh with the generated test names.
#
# Usage:
#   ./run_permute_tests.sh --num 20 --count 5
#   ./run_permute_tests.sh --num 30 --count 10 --families mxu
#   ./run_permute_tests.sh --dry-run --num 10
#   ./run_permute_tests.sh --gen-only --num 20 --count 5
#   ./run_permute_tests.sh --clean
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAREMETAL_DIR="${SCRIPT_DIR}/.."
RANDOM_GEN="${SCRIPT_DIR}/random_gen.py"
RUN_ASM_TESTS="${BAREMETAL_DIR}/run_asm_tests.sh"
ASSEMBLY_DIR="${BAREMETAL_DIR}/assembly"
GENERATORS_DIR="${BAREMETAL_DIR}/generators"
PYTHON="${PYTHON:-python3}"

red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
cyan()  { printf '\033[1;36m%s\033[0m\n' "$*"; }
die()   { red "ERROR: $*" >&2; exit 1; }

DRY_RUN=0
GEN_ONLY=0
DO_CLEAN=0
GEN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --gen-only)  GEN_ONLY=1; shift ;;
        --clean)     DO_CLEAN=1; shift ;;
        *)           GEN_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${DO_CLEAN} -eq 1 ]]; then
    cyan "Cleaning generated random_* files..."
    rm -f "${ASSEMBLY_DIR}"/random_*.S
    rm -f "${GENERATORS_DIR}"/gen_random_*.py
    rm -f "${GENERATORS_DIR}"/random_*.json
    green "Clean."
    exit 0
fi

[[ -f "${RANDOM_GEN}" ]] || die "random_gen.py not found at ${RANDOM_GEN}"

if [[ ${DRY_RUN} -eq 1 ]]; then
    ${PYTHON} "${RANDOM_GEN}" --dry-run "${GEN_ARGS[@]}"
    exit 0
fi

# Generate directly into assembly/ and generators/
cyan "Generating random tests..."
mkdir -p "${ASSEMBLY_DIR}" "${GENERATORS_DIR}"

OUTPUT=$(${PYTHON} "${RANDOM_GEN}" \
    --asm-dir "${ASSEMBLY_DIR}" \
    --gen-dir "${GENERATORS_DIR}" \
    "${GEN_ARGS[@]}")
echo "${OUTPUT}"

# Extract test names from generated files
TEST_NAMES=()
for f in "${ASSEMBLY_DIR}"/random_*.S; do
    [[ -f "${f}" ]] || continue
    TEST_NAMES+=("$(basename "${f}" .S)")
done

if [[ ${#TEST_NAMES[@]} -eq 0 ]]; then
    die "No tests generated"
fi

green "Generated ${#TEST_NAMES[@]} tests."

if [[ ${GEN_ONLY} -eq 1 ]]; then
    echo ""
    echo "To run them:"
    echo "  cd ${BAREMETAL_DIR} && ./run_asm_tests.sh ${TEST_NAMES[*]}"
    exit 0
fi

[[ -f "${RUN_ASM_TESTS}" ]] || die "run_asm_tests.sh not found at ${RUN_ASM_TESTS}"

cyan "Running tests via run_asm_tests.sh..."
echo ""

cd "${BAREMETAL_DIR}"
exec bash "${RUN_ASM_TESTS}" "${TEST_NAMES[@]}"
