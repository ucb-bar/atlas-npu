# Atlas baremetal tests

## With DRAM golden checks

**1. Generate golden JSON** (from `baremetal/`):

```bash
cd generators/sp26-atlas-acc/baremetal
python3 generators/gen_<test_name>.py > generators/<test_name>.json
```

**2. Assemble and emit C**:

```bash
python3 assembler.py assembly/<test_name>.S \
  --out-c tests/atlas_<test_name>.c \
  --golden-json generators/<test_name>.json
```

**3. Build** (from `baremetal/tests/`):

```bash
cd tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target atlas_<test_name>
```

**4. Run in simulation**

```bash
cd $CHIPYARD/sims/vcs
make run-binary \
  BINARY=../../generators/sp26-atlas-acc/baremetal/tests/build/atlas_<test_name>.riscv \
  CONFIG=AtlasShuttleVectorConfig
```

---

## Without DRAM golden checks

**1.** `cd generators/sp26-atlas-acc/baremetal`

**2. Assemble only:**

```bash
python3 assembler.py assembly/<test_name>.S --out-c tests/atlas_<test_name>.c
```

**3. Build** (from `baremetal/tests/`):

```bash
cd tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target atlas_<test_name>
```

**4. Simulation** (same as above):

```bash
cd $CHIPYARD/sims/vcs
make run-binary \
  BINARY=../../generators/sp26-atlas-acc/baremetal/tests/build/atlas_<test_name>.riscv \
  CONFIG=AtlasShuttleVectorConfig
```

