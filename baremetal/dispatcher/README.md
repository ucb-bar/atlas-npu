# Atlas IMEM dispatcher

The dispatcher lets the host load the next Atlas kernel while another runs,
without restarting Atlas between kernels. It requires no RTL changes.

## How it works

The host C code runs on Shuttle, with Saturn executing RVV instructions.
Shuttle loads Atlas IMEM through scalar MMIO stores; `dispatcher.S` runs on Atlas.

1. The host installs the dispatcher at IMEM word 0 and starts Atlas once.
2. The host loads a kernel into a free slot, fences, and writes an execution
   command to DBG1 identifying the slot and a new sequence number.
3. Atlas polls DBG1, acknowledges STARTED through DBG0, and jumps to the slot.
4. The kernel returns to the dispatcher, which publishes DONE and polls again.
   If no new command is available, Atlas spins in the dispatcher.

```text
Host:   load A → submit A → load B / submit B while A runs → reuse released A
Atlas:           dispatcher → A → dispatcher → B → dispatcher → ...
```

Loading code alone never launches it. There can be one running and one pending
command. The host helpers prevent overwriting active or queued slots. A newer
acknowledgment also proves older invocations completed, so missing a brief DONE
value is safe. Data-buffer ownership is the application's responsibility.

| Region | IMEM word indices | Host byte addresses |
| --- | --- | --- |
| Dispatcher (1 KiB reserved) | 0x0000–0x00ff | 0x20000–0x203ff |
| Slot A / 0 (63 KiB) | 0x0100–0x3fff | 0x20400–0x2ffff |
| Slot B / 1 (63 KiB) | 0x4100–0x7fff | 0x30400–0x3ffff |

Words 0x4000–0x40ff are unused. Each kernel enters at the first word of its slot.

The host owns DBG1 (`0x40014`, Atlas CSR `0xC11`); Atlas owns DBG0 (`0x40010`,
CSR `0xC10`) after initialization. Command encoding is `(sequence << 1) | slot`;
STARTED is `command << 1`, DONE is `(command << 1) | 1`, and initial READY is 1.
The helpers manage sequences, starting at 1; they must not wrap past 0x3fffffff.

## How to use it

Include [atlas_dispatcher.h](atlas_dispatcher.h) in your host C program; the existing test builds already configure its location. Call the dispatcher helpers from a single host thread. Atlas can execute concurrently while that thread loads the next kernel.

| Step | Host API | Effect |
| --- | --- | --- |
| Initialize once | `atlas_dispatch_init(&d, poll_limit)` | Load the dispatcher, start at PC 0, wait for READY. Atlas must already be halted with prior engine work complete. |
| Load a free slot | `atlas_dispatch_stage(&d, slot, words, count)` | Copy assembled kernel instructions into IMEM. |
| Launch or queue | `atlas_dispatch_submit(&d, slot, &command)` | Publish a new execution command after ordering code/input writes. |
| Wait for results | `atlas_dispatch_wait(command, poll_limit)` | Wait for completion before consuming outputs or reusing the slot. |

Check every return value. `BUSY` means retry later; a timeout does not cancel work
or release its slot. Poll limits count MMIO reads, not cycles. After a command is
accepted, the host can queue the next one. Already-loaded code can be submitted
again without reloading once its slot is free. See [test_dispatcher.c](../tests/dispatcher/test_dispatcher.c)
for executable usage.

Do not read IMEM while Atlas runs: those reads block. Do not write another start
while the dispatcher runs: it resets PC to 0. To stop, wait for the last submitted
command to complete, then write EXEC_CONTROL=0 (`0x40018`).

**Kernel calling convention:** reserve `x28–x31`, leave DBG0/DBG1 to the dispatcher,
initialize other registers as needed, and use position-independent code or relocate
absolute targets for the selected slot. Finish required compute, DMA, and stores
before returning; the dispatcher does not drain engines automatically. Use the
appropriate scheduling/delays and `DMA.WAIT`, then end with:

```asm
    JALR x0, x31, 1
    NOP
```

Atlas JALR targets are word indices. The return offset 1 skips the call's delay
slot. Existing kernels must replace their successful ECALL exit and remove any
conflicting register/DBG usage before they can run under the dispatcher.

Runtime files stay in `baremetal/dispatcher/`; the test driver and kernel live in
`baremetal/tests/dispatcher/`. Generated headers and executables live only in
build directories. The supporting runtime files are:

- `dispatcher.S`: Atlas's execution loop.
- `atlas_dispatcher.h`: inline host C helpers.
- `generate.py`: uses `../assembler.py` to produce the instruction headers at build
  time. Python does not run on the target.
- `dispatcher.cmake`: integrates image generation into both baremetal suites.

The generated `dispatcher_image.h` contains the instructions the host copies into
IMEM. Existing test builds generate it and configure its include path. For a
separate application, run `generate.py --out-dir <build-directory>` and add that
directory and `baremetal/dispatcher/` to the C include path.

## What the tests cover

The tests use **the same small [test_kernel.S](../tests/dispatcher/test_kernel.S) in both slots**,
not kernels from `baremetal/assembly/`. It runs a busy loop, increments VMEM word 0,
and returns. Integration into the suites adds test targets; it does not convert
existing compute kernels.

[test_dispatcher.c](../tests/dispatcher/test_dispatcher.c) installs the dispatcher, verifies it spins
while A is only partially loaded, fully loads and launches A, then stages and
queues B while A runs. It checks slot protection, completion acknowledgments,
and that an unchanged command does not execute twice. It then reuses B without
reloading, reloads A, executes A, and stops Atlas.

The [Saturn variant](../../../../tests/saturn_atlas_dispatcher.c) adds an RVV
load/add/store transformation of 32 VMEM words between the second and third Atlas
invocations, while Atlas spins. Expected VMEM[0] values are:

- Scalar test: `0 → 1 → 2 → 3 → 4`.
- Saturn test: `0 → 1 → 2 → 102 → 103 → 104`.

Both variants passed full-system EE290SimConfig VCS simulation in the original
`chipyard` checkout on 2026-09-28.
These tests verify control
and data handoffs, not MXU/VPU computation, inference performance, or simultaneous
Saturn/Atlas compute. Post-gate discovery was checked by dry run only.

## Build and run

Run from the **Chipyard root** with the Chipyard RISC-V toolchain, Python 3, and
VCS environment available. Building an ELF does not run it.

```sh
# Atlas: build only, then build/run/report through the existing runner.
make -C generators/atlas-npu/baremetal dispatcher
make -C generators/atlas-npu/baremetal test dispatcher GEN_COVERAGE=0

# Saturn/Atlas: build and run the RVV variant.
cmake -S tests -B tests/build
cmake --build tests/build --target saturn_atlas_dispatcher
make -C sims/vcs run-binary CONFIG=EE290SimConfig LOADMEM=1 GEN_COVERAGE=0 \
  BINARY="$PWD/tests/build/saturn_atlas_dispatcher.riscv"
```

Both tests print `*** PASSED ***` on success. Logs are under
`sims/vcs/output/chipyard.harness.TestHarness.EE290SimConfig/`, named
`atlas_dispatcher.log` and `saturn_atlas_dispatcher.log`. `LOADMEM=1` preloads the
host ELF into system memory; the running host still loads Atlas IMEM through MMIO.
CMake automatically assembles instruction headers into each build directory.

The baremetal runner defaults to `EE290SimConfig`. Tacit, Rocket Chip, and
Shuttle now match the regular Chipyard revisions. Full dispatcher simulation in
this checkout remains unverified because VCS could not connect to the license
server. VCS must be on PATH with a reachable Synopsys license server.
CI/CD and post-gate automation are not maintained here; use the local build
targets above.
