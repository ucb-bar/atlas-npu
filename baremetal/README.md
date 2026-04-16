# Atlas Baremetal Tests

This directory is the source of truth for Atlas baremetal tests.

## Manual Single-Test Flow

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
  GEN_COVERAGE=1 \
  BINARY=../../generators/sp26-atlas-acc/baremetal/tests/build/atlas_<test_name>.riscv \
  CONFIG=EE290SimConfig LOADMEM=1
```

## RUN THIS: Single-Command for Testing Flow
From the `sp26-atlas-acc/baremetal` directory, to run select baremetal tests (e.g., `test1`, `test2`, and `test3`):

```
./run_asm_tests.sh test1 test2 test3
```

To run all handwritten baremetal tests (i.e., those in `sp26-atlas-acc/baremetal/assembly`):

```
./run_asm_tests.sh --all
```
