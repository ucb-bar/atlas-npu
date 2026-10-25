# Atlas Baremetal Tests

This directory is the source of truth for Atlas baremetal tests. From the Atlas root, run `git submodule update --init npu-model`; then, from this directory, run `uv sync --locked --project ../npu-model --python 3.14` once to create its Python environment. Make defaults `NPU_MODEL_DIR` to `../npu-model`.

## Manual Single-Test Flow

1) Optional DRAM golden JSON (from `baremetal/`):

```bash
python3 generators/gen_<test_name>.py > generators/<test_name>.json
```

2) Assemble to C (Make is preferred for ordinary tests):

```bash
PYTHONPATH=../npu-model uv run --locked --project ../npu-model --python 3.14 python atlas_asm.py assembly/<test_name>.S \
  --out-c tests/atlas_<test_name>.c \
  --golden-json generators/<test_name>.json
```

The assembler also accepts `--out-hex`, `--out-bin`, `--dram-base`, and `--beat-bytes`. The DRAM base comes from a JSON entry when present, otherwise `@DRAM_BASE`, otherwise the CLI default. JSON preload/check data is used when supplied; inline `@DRAM` and `@CHECK_DRAM` provide the fallback when no JSON golden is supplied.

`atlas_asm.py --disasm FILE` prints the assembly of a program: hex text (one word per line) or, for
a `.bin` file, raw binary. The disassembler (`npu_model.isa.disasm`) decodes the words with the ISA
table and puts labels at branch and jump targets. Its output assembles to the same words.

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
./agentic/run_asm_tests.sh test1 test2 test3
```

To run all handwritten baremetal tests (i.e., those in `sp26-atlas-acc/baremetal/assembly`):

```
./agentic/run_asm_tests.sh --all
```
