# Atlas Baremetal Tests

This directory is the source of truth for Atlas baremetal tests.

## Quick Commands (Combo Suite)

From `generators/sp26-atlas-acc/baremetal`:

```bash
# Generate combo suite (assembly + C + manifest + coverage)
make gen

# List short test names
make list-tests

# Build all combo binaries
make build-all

# Run smoke tier
make run-smoke CONFIG=EE290SimConfig

# Run all tiers
make run-all CONFIG=EE290SimConfig

# Run one test by short name
make run-test TEST=smk_salu_dma_05 CONFIG=EE290SimConfig
```

Generated combo-suite artifacts:

- Assembly: `assembly/generated/<tier>/*.S`
- C tests: `tests/atlas_<short_name>.c`
- Results:
  - `results/atlas_combo_suite_manifest.json`
  - `results/atlas_combo_coverage_report.json`
  - `results/atlas_combo_run_results.json`
  - `results/atlas_combo_run_summary.md`
  - `results/logs/*.log`

## Manual Single-Test Flow (Handwritten Assembly)

1) Optional DRAM golden JSON (from `baremetal/`):

```bash
python3 generators/gen_<test_name>.py > generators/<test_name>.json
```

2) Assemble to C:

```bash
python3 assembler.py assembly/<test_name>.S \
  --out-c tests/atlas_<test_name>.c \
  --golden-json generators/<test_name>.json
```

3) Build:

```bash
cmake -S tests -B tests/build -DCMAKE_BUILD_TYPE=Debug
cmake --build tests/build --target atlas_<test_name>
```

4) Run:

```bash
cd $CHIPYARD/sims/vcs
make run-binary \
  BINARY=../../generators/sp26-atlas-acc/baremetal/tests/build/atlas_<test_name>.riscv \
  CONFIG=EE290SimConfig LOADMEM=1
```

## Single-Command Handwritten Assembly Flow
From the `sp26-atlas-acc/baremetal` directory, to run select baremetal tests (e.g., `test1`, `test2`, and `test3`):
```
./run_asm_tests.sh test1 test2 test3
```

To run all handwritten baremetal tests (i.e., those in `sp26-atlas-acc/baremetal/assembly`):
```
./run_asm_tests.sh --all
```

## Randomized Instructions for Scalar, DMA, LSU, MXU0, and MXU1
```
cd baremetal/random_permutations
./run_permute_tests.sh --num 20 --count 5       # detailed usage instructions are at the top of run_permute_tests.sh
```
