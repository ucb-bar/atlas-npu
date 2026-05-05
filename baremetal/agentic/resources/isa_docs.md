This page documents the assembly surface accepted by chipyard/generators/sp26-atlas-acc/baremetal/assembler.py and the software-scheduling contract enforced by the current Atlas hardware.
It is intentionally tied to the current Chisel implementation, not generic RV32I behavior.

Quick model

Atlas scalar control flow uses IMEM word indices, not byte addresses.
Numeric branch and JAL offsets are in words (instructions), not bytes.
Branches and jumps have two architectural delay slots.
The scalar pipeline only stalls for DELAY imm and DMA.WAIT ch.
All other accelerator instructions are fire-and-forget. If software issues them too early or with conflicting banks/resources, current hardware either asserts or ignores/drops the new command.


Default machine geometry



Resource
Current default




Scalar registers
x0..x31


Scale registers
32 entries, 8 bits each, indexed as plain integers 0..31



MREG banks
64 banks, each 32 rows x 32 B = 1024 B


VMEM
256 KiB total, 8 banks, 32-byte lines, 32 KiB per bank


DMA beat size
32 B


DMA channels
8


Max single DMA transfer
4096 B




Operand conventions


rd, rs1, rs2 are scalar integer registers written as xN.

vd, vs1, vs2 in MXU/VPU/XLU instructions are MREG bank numbers written as plain integers 0..63, not xN.

scaleReg operands are plain integer scale-register indices 0..31.

accSel and wSlot are currently 1-bit selectors in hardware, so software should treat them as 0 or 1.

channel is the 3-bit DMA channel field, so software should treat it as 0..7.


Addressing conventions
Atlas uses several different address units. This is the most important thing to get right in software.



Instruction family
Address unit seen by software
Effective address




LB/LH/LW/LBU/LHU/SB/SH/SW/SELD
local VMEM byte address
x[rs1] + imm



VLOAD/VSTORE base register
local VMEM word address

x[rs1] + imm12 * 32 words



DMA.LOAD VMEM operand
local VMEM word address
x[rd]



DMA.STORE VMEM operand
local VMEM word address
x[rs1]



DMA.LOAD DRAM operand
DRAM byte offset from DMA.CONFIG base
dmaBase + x[rs1]



DMA.STORE DRAM operand
DRAM byte offset from DMA.CONFIG base
dmaBase + x[rd]




Scalar VMEM accesses

Scalar loads/stores use byte addressing into local VMEM.
Example: LW x9, x8, 4 reads the 32-bit word at local VMEM byte address x8 + 4.


VLOAD / VSTORE
VLOAD and VSTORE use the custom VLS immediate format.


rs1 holds a local VMEM word address.


The encoded imm12 is sign-extended and shifted left by 5 before addition.


Effective VMEM word address:
effWordAddr = x[rs1] + imm12 * 32


Hardware then drops the low 3 word bits to convert that word address into a 32-byte VMEM line address.


One MREG is 1024 B = 256 words = 32 VMEM lines.


Because imm12 steps in chunks of 32 words = 128 B, adjacent 1 KiB tensor slots are typically selected with immediates 0, 8, 16, ...


Example:


VSTORE 4, x8, 0 with x8 = 1024 stores bank m4 at VMEM word 1024


VSTORE 5, x8, 8 stores bank m5 at VMEM word 1280



DMA


DMA.CONFIG rs1, ch loads the global DMA base register from x[rs1].
Current hardware ignores the encoded ch field on DMA.CONFIG; the base register is global, not per-channel.

DMA.LOAD copies DRAM to VMEM.

DMA.STORE copies VMEM to DRAM.
VMEM addresses supplied to DMA are local VMEM word addresses.
DRAM addresses supplied to DMA are byte offsets added to the configured DMA base.
Software should keep DRAM addresses, VMEM addresses, and transfer sizes aligned to the 32-byte DMA beat.


Control-flow semantics


The scalar PC is an IMEM word index.


JAL and branches therefore use word offsets.


JALR also works in word units:
target = x[rs1] + imm


JAL / JALR write pc + 1 to the link register, again in word units.


Atlas has two architectural branch delay slots:

the instruction after the branch/jump always executes
the second instruction after the branch/jump also always executes



Placing another branch or jump in either delay slot is illegal and halts the core.

Assembler-facing label rule:

- labels are accepted directly as targets of `BEQ/BNE/BLT/BGE/BLTU/BGEU` and `JAL`
- labels are **not** accepted as immediates to `LI`, `LUI`, `AUIPC`, `ADDI`, `JALR`, DMA
  instructions, or other numeric-operand instructions

Preferred helper/subroutine pattern:

- call a local helper with `JAL x1, helper_label`
- fill both delay slots with safe work or `NOP`
- return with `JALR x0, x1, 0`
- fill the return's two delay slots with safe work or `NOP`

Do not synthesize function pointers from labels unless the assembler explicitly gains
relocation support.



Instruction formats


R: [31:25] funct7 | [24:20] rs2 | [19:15] rs1 | [14:12] funct3 | [11:7] rd | [6:0] opcode


I: [31:20] imm12 | [19:15] rs1 | [14:12] funct3 | [11:7] rd | [6:0] opcode


S: [31:25] imm[11:5] | [24:20] rs2 | [19:15] rs1 | [14:12] funct3 | [11:7] imm[4:0] | [6:0] opcode


B: standard RV32I split-branch immediate

U: [31:12] imm20 | [11:7] rd | [6:0] opcode


J: standard RV32I split-jump immediate

VR: [31:25] funct7 | [24:19] vs2 | [18:13] vs1 | [12:7] vd | [6:0] opcode


VLS: [31:20] imm12 | [19:15] rs1 | [14:13] f2 | [12:7] vd | [6:0] opcode


VI: [31:16] imm16 | [15:13] f3 | [12:7] vd | [6:0] opcode



Instruction reference

Pseudo / system



Mnemonic
Operands
Encoding
Notes




NOP
-
pseudo
expands to ADDI x0, x0, 0



LI
rd, imm
pseudo
expands to either ADDI or LUI + ADDI

`imm` must be a numeric literal accepted by `assembler.py` (`123`, `-7`, `0x00000001`, etc.).
The current assembler does not relocate symbolic labels through `LI`, so `LI x26, helper`
is illegal even if `helper:` exists later in the file.



DELAY
imm
I, opcode 0x67, funct3 001

stalls the scalar frontend for imm cycles


FENCE
-
I, opcode 0x0F, funct3 000

standard fence encoding


ECALL
-
fixed 0x00000073

halts current test/program flow


EBREAK
-
fixed 0x00100073

halts current test/program flow




RV32I integer ALU

R-type



Mnemonic
Operands
Opcode / funct
Notes




ADD
rd, rs1, rs2

0x33, f3=0, f7=0x00

standard RV32I


SUB
rd, rs1, rs2

0x33, f3=0, f7=0x20

standard RV32I


SLL
rd, rs1, rs2

0x33, f3=1, f7=0x00

standard RV32I


SLT
rd, rs1, rs2

0x33, f3=2, f7=0x00

standard RV32I


SLTU
rd, rs1, rs2

0x33, f3=3, f7=0x00

standard RV32I


XOR
rd, rs1, rs2

0x33, f3=4, f7=0x00

standard RV32I


SRL
rd, rs1, rs2

0x33, f3=5, f7=0x00

standard RV32I


SRA
rd, rs1, rs2

0x33, f3=5, f7=0x20

standard RV32I


OR
rd, rs1, rs2

0x33, f3=6, f7=0x00

standard RV32I


AND
rd, rs1, rs2

0x33, f3=7, f7=0x00

standard RV32I




I-type / U-type



Mnemonic
Operands
Opcode / funct
Notes




ADDI
rd, rs1, imm

0x13, f3=0

standard RV32I


SLTI
rd, rs1, imm

0x13, f3=2

standard RV32I


SLTIU
rd, rs1, imm

0x13, f3=3

standard RV32I


XORI
rd, rs1, imm

0x13, f3=4

standard RV32I


ORI
rd, rs1, imm

0x13, f3=6

standard RV32I


ANDI
rd, rs1, imm

0x13, f3=7

standard RV32I


SLLI
rd, rs1, shamt

0x13, f3=1, upper imm 0x00

standard RV32I


SRLI
rd, rs1, shamt

0x13, f3=5, upper imm 0x00

standard RV32I


SRAI
rd, rs1, shamt

0x13, f3=5, upper imm 0x20

standard RV32I


LUI
rd, imm20
0x37
standard RV32I


AUIPC
rd, imm20
0x17
standard RV32I




Branches and jumps



Mnemonic
Operands
Opcode / funct
Notes




BEQ
rs1, rs2, off

0x63, f3=0


off is a word offset


BNE
rs1, rs2, off

0x63, f3=1


off is a word offset


BLT
rs1, rs2, off

0x63, f3=4


off is a word offset


BGE
rs1, rs2, off

0x63, f3=5


off is a word offset


BLTU
rs1, rs2, off

0x63, f3=6


off is a word offset


BGEU
rs1, rs2, off

0x63, f3=7


off is a word offset


JAL
rd, off
0x6F

off is a word offset


JALR
rd, rs1, off

0x67, f3=0


target = x[rs1] + off in word units

`off` is a numeric word offset, not a symbolic label. The current assembler only resolves
labels in branch instructions and `JAL`; it does not resolve labels placed into `JALR`
immediates or into scalar registers with `LI`.




CSR and scale-register instructions



Mnemonic
Operands
Opcode / funct
Notes




CSRRW
rd, csr, rs1

0x73, f3=1

standard CSR write/read


CSRRS
rd, csr, rs1

0x73, f3=2

standard CSR set/read


CSRRC
rd, csr, rs1

0x73, f3=3

standard CSR clear/read


CSRRWI
rd, csr, zimm

0x73, f3=5

standard CSR immediate


CSRRSI
rd, csr, zimm

0x73, f3=6

standard CSR immediate


CSRRCI
rd, csr, zimm

0x73, f3=7

standard CSR immediate


CSRW
rs, csr
pseudo
encodes as CSRRW x0, csr, rs



CSRR
rd, csr
pseudo
encodes as CSRRS rd, csr, x0



SELD
sd, rs1, imm

0x03, f3=6

loads an 8-bit scale value via scalar VMEM load path into scale reg sd



SELI
sd, imm

0x03, f3=7

writes low 8 bits of imm into scale reg sd




Scale registers are also mapped in CSR space at 0x860..0x87F, but SELI / SELD are the assembler-visible direct helpers.

Scalar VMEM loads and stores



Mnemonic
Operands
Opcode / funct
Notes




LB
rd, rs1, imm

0x03, f3=0

scalar VMEM byte load


LH
rd, rs1, imm

0x03, f3=1

scalar VMEM halfword load


LW
rd, rs1, imm

0x03, f3=2

scalar VMEM word load


LBU
rd, rs1, imm

0x03, f3=4

scalar VMEM unsigned byte load


LHU
rd, rs1, imm

0x03, f3=5

scalar VMEM unsigned halfword load


SB
rs2, rs1, imm

0x23, f3=0

scalar VMEM byte store


SH
rs2, rs1, imm

0x23, f3=1

scalar VMEM halfword store


SW
rs2, rs1, imm

0x23, f3=2

scalar VMEM word store




Tensor LSU instructions



Mnemonic
Operands
Opcode / funct
Notes




VLOAD
vd, rs1, imm12
VLS, opcode 0x07, f2=00

loads one full MREG from VMEM into bank vd



VSTORE
vs, rs1, imm12
VLS, opcode 0x07, f2=01

stores one full MREG from bank vs to VMEM



For both instructions:


rs1 is a VMEM word-address base
effective VMEM word address is x[rs1] + imm12 * 32

one instruction transfers exactly one MREG = 1024 B = 32 VMEM lines


DMA instructions



Mnemonic
Operands
Opcode / funct
Notes




DMA.LOAD
rd, rs1, rs2, ch
R, opcode 0x7B, f7=0x00, f3=ch


rd = VMEM word-address reg, rs1 = DRAM byte-offset reg, rs2 = size-bytes reg


DMA.STORE
rd, rs1, rs2, ch
R, opcode 0x7B, f7=0x01, f3=ch


rd = DRAM byte-offset reg, rs1 = VMEM word-address reg, rs2 = size-bytes reg


DMA.CONFIG
rs1, ch
I, opcode 0x7F, imm=0x000, f3=ch

loads global DMA base from x[rs1]; channel field is currently ignored


DMA.WAIT
ch
I, opcode 0x7F, imm=0x020, f3=ch

stalls until channel ch is no longer busy




MXU instructions
MXU0 and MXU1 use the same operand conventions and only differ in funct7.



Mnemonic
Operands
Opcode / funct7
Notes




VMATPUSH.W.MXU0
wSlot, vs
VR, 0x77, f7=0x00

pushes one FP8 MREG into MXU0 weight slot


VMATPUSH.W.MXU1
wSlot, vs
VR, 0x77, f7=0x01

pushes one FP8 MREG into MXU1 weight slot


VMATPUSH.ACC.FP8.MXU0
accSel, vs
VR, 0x77, f7=0x02

pushes one FP8 MREG into MXU0 accum buffer


VMATPUSH.ACC.FP8.MXU1
accSel, vs
VR, 0x77, f7=0x03

pushes one FP8 MREG into MXU1 accum buffer


VMATPUSH.ACC.BF16.MXU0
accSel, vs
VR, 0x77, f7=0x04

reads BF16 banks vs and vs+1 and seeds MXU0 accum buffer


VMATPUSH.ACC.BF16.MXU1
accSel, vs
VR, 0x77, f7=0x05

reads BF16 banks vs and vs+1 and seeds MXU1 accum buffer


VMATPOP.FP8.MXU0
vd, scaleReg, accSel
VR, 0x77, f7=0x06

pops one FP8 MREG from MXU0 accum buffer using scale register


VMATPOP.FP8.MXU1
vd, scaleReg, accSel
VR, 0x77, f7=0x07

pops one FP8 MREG from MXU1 accum buffer using scale register


VMATPOP.BF16.MXU0
vd, accSel
VR, 0x77, f7=0x08

writes BF16 banks vd and vd+1 from MXU0 accum buffer


VMATPOP.BF16.MXU1
vd, accSel
VR, 0x77, f7=0x09

writes BF16 banks vd and vd+1 from MXU1 accum buffer


VMATMUL.MXU0
accSel, vs1, wSlot
VR, 0x77, f7=0x0A

MXU0 matmul without accumulation


VMATMUL.MXU1
accSel, vs1, wSlot
VR, 0x77, f7=0x0B

MXU1 matmul without accumulation


VMATMUL.ACC.MXU0
accSel, vs1, wSlot
VR, 0x77, f7=0x0C

MXU0 matmul with accumulation


VMATMUL.ACC.MXU1
accSel, vs1, wSlot
VR, 0x77, f7=0x0D

MXU1 matmul with accumulation




VPU instructions
Unless noted otherwise, BF16 VPU instructions operate on MREG bank pairs:

primary pair read starts at vs1

secondary pair read starts at vs2

pair write starts at vd


For those pair operands, the starting bank must be even.

Binary BF16 pair ops



Mnemonic
Operands
Opcode / funct7
Notes




VADD.BF16
vd, vs1, vs2
VR, 0x57, f7=0x00

pair + pair -> pair


VSUB.BF16
vd, vs1, vs2
VR, 0x57, f7=0x02

pair + pair -> pair


VMUL.BF16
vd, vs1, vs2
VR, 0x57, f7=0x03

pair + pair -> pair


VMIN.BF16
vd, vs1, vs2
VR, 0x57, f7=0x04

pair + pair -> pair


VMAX.BF16
vd, vs1, vs2
VR, 0x57, f7=0x06

pair + pair -> pair




Reductions



Mnemonic
Operands
Opcode / funct7
Notes




VREDSUM.BF16
vd, vs1
VR, 0x57, f7=0x01

column reduction, long-latency path


VREDMIN.BF16
vd, vs1
VR, 0x57, f7=0x05

column reduction, long-latency path


VREDMAX.BF16
vd, vs1
VR, 0x57, f7=0x07

column reduction, long-latency path


VREDSUM.ROW.BF16
vd, vs1
VR, 0x57, f7=0x21

row reduction


VREDMIN.ROW.BF16
vd, vs1
VR, 0x57, f7=0x24

row reduction


VREDMAX.ROW.BF16
vd, vs1
VR, 0x57, f7=0x26

row reduction




Unary / pack / unpack



Mnemonic
Operands
Opcode / funct7
Notes




VMOV
vd, vs
VR, 0x57, f7=0x40

pair -> pair


VRECIP.BF16
vd, vs
VR, 0x57, f7=0x41

pair -> pair


VEXP
vd, vs
VR, 0x57, f7=0x42

pair -> pair


VEXP2
vd, vs
VR, 0x57, f7=0x43

pair -> pair


VFP8PACK
vd, vs2, scaleReg
VR, 0x57, f7=0x44

reads BF16 pair at vs2, writes one FP8 bank vd



VFP8UNPACK
vd, vs, scaleReg
VR, 0x57, f7=0x45

reads one FP8 bank vs, writes BF16 pair vd, vd+1



VSQUARE.BF16
vd, vs
VR, 0x57, f7=0x46

pair -> pair


VCUBE.BF16
vd, vs
VR, 0x57, f7=0x47

pair -> pair


VRELU
vd, vs
VR, 0x57, f7=0x48

pair -> pair


VSIN
vd, vs
VR, 0x57, f7=0x49

pair -> pair


VCOS
vd, vs
VR, 0x57, f7=0x4A

pair -> pair


VTANH
vd, vs
VR, 0x57, f7=0x4B

pair -> pair


VLOG2
vd, vs
VR, 0x57, f7=0x4C

pair -> pair


VSQRT
vd, vs
VR, 0x57, f7=0x4D

pair -> pair




VLI immediate fills



Mnemonic
Operands
Opcode / funct
Notes




VLI.ALL
vd, imm16
VI, opcode 0x5F, f3=0

writes pair vd, vd+1; vd must be even


VLI.ROW
vd, imm16
VI, opcode 0x5F, f3=1

writes pair vd, vd+1; vd must be even


VLI.COL
vd, imm16
VI, opcode 0x5F, f3=2

writes one bank vd



VLI.ONE
vd, imm16
VI, opcode 0x5F, f3=3

writes one bank vd





XLU instruction



Mnemonic
Operands
Opcode / funct7
Notes




VTRPOSE.XLU
vd, vs1
VR, opcode 0x6B, f7=0x00

transposes one 32 x 32 MREG




Software-scheduling contract
Atlas is software scheduled. The rules below come directly from the current hardware assertions and command interfaces.



Area
Rule software must obey
Practical guidance




Control flow
Branches and jumps have two delay slots. A branch/jump inside a delay slot is illegal.
Put two NOPs or two safe non-branch instructions after every branch/jump unless you are intentionally filling the slots.


Scalar loads
Scalar load latency is fixed at 2 cycles, and issuing another scalar load while one is pending asserts.
If you need the loaded scalar result immediately, insert DELAY 1.


Scalar load vs writeback
A returning scalar load cannot collide with another scalar register writeback, and a returning SELD cannot collide with SELI.

DELAY 1 after a scalar load/SELD avoids this in simple code.


VMEM LSU bank conflicts
VMEM is 1R1W per bank. Current hardware asserts on same-port conflicts: scalar load vs VLOAD to the same VMEM bank, or scalar store vs VSTORE to the same VMEM bank.
Keep scalar-load and vector-load traffic on different VMEM banks in the same cycle, and likewise for scalar-store and vector-store traffic.


VLOAD / VSTORE issue
Issuing VLOAD while the VLOAD path is busy asserts. Issuing VSTORE while the VSTORE path is busy asserts.
Do not back-to-back another instruction of the same kind until the previous one has completed.


VLOAD / VSTORE alignment

VLOAD and VSTORE bases must be 1 KiB aligned for block banking. Hardware asserts if the effective line address is not aligned.
Use VMEM word bases that are multiples of 256 words, or equivalently effective line bases that are multiples of 32 lines.


DMA address / size granularity
DMA internally counts whole 32-byte beats.
Keep DRAM address, VMEM word address, and transfer size aligned to 32 bytes, and keep size <= 4096.


DMA channel reuse

DMA.LOAD / DMA.STORE launch is not stalled on a busy channel. The engine is intended to be synchronized with DMA.WAIT ch.
Reuse a DMA channel only after DMA.WAIT ch. Treat DMA as one outstanding command per channel.


DMA queue depth
The engine has 8 command slots and no launch-side full check.
Keep total in-flight DMA commands within the 8-channel budget.


MREG bank conflicts
MREG is 1R1W per bank. The scalar core asserts if an instruction reads a bank with a pending write, or writes a bank with a pending read/write.
Do not overlap producer/consumer commands on the same MREG bank unless the engine-specific overlap rule explicitly allows it.


VPU pair-bank rules
Current hardware asserts if a VPU pair-read primary bank, pair-read secondary bank, or pair-write destination bank is odd.
For pair-based BF16 VPU ops, use even vs1, even vs2, and even vd.


VPU issue timing
VPU commands must only issue when the selected issueBusy bit is clear.
If you are not intentionally exploiting VPU overlap, insert a conservative DELAY after each VPU command.


VPU dual-read aliasing
Two VPU reads may alias the same bank only if they request the same row.
Do not try to hand-schedule same-bank dual reads except the natural mirrored case.


MXU slot timing
MXU sequencers assert if compute/push/pop issue into busy slots or if weight-slot / accum-buffer / drain conflicts occur.
Unless you are intentionally exploiting overlap, wait between MXU commands so Slot A, Slot B, and Slot C are all clear.


MXU BF16 push / pop footprints

VMATPUSH.ACC.BF16 reads vs and vs+1. VMATPOP.BF16 writes vd and vd+1.
Reserve both banks when scheduling around them.


XLU issue timing
XLU only accepts a command in Idle; there is no backpressure or assert on a new command while busy.
Always insert a DELAY large enough for the transpose to finish before issuing another VTRPOSE.XLU or before reading its destination bank.




Delay guidance
There are two kinds of timing numbers below:

exact fixed latencies, where the hardware has a simple deterministic formula
conservative delays, which are the rounded-up values used by the checked-in baremetal tests

If you are writing simple code and not intentionally overlapping engines, the conservative values are the easiest safe starting point.



Sequence
Fixed hardware latency / rule
Conservative value used in tests




Scalar load -> immediate scalar consumer
scalar load path is 2 cycles
DELAY 1



VLOAD -> read bank via another engine

mregRows + 1 cycles, 33 by default
DELAY 40



VSTORE -> read back same VMEM region or issue another VSTORE on same path

mregRows + 1 cycles, 33 by default
DELAY 40



VTRPOSE.XLU -> consume destination bank

2 * mregRows cycles, 64 by default
DELAY 80


Standalone MXU push / pop in simple tests
engine-specific slot service, deterministic but multi-cycle
typically DELAY 40



Standalone 32x32 MXU matmul before pop/readback
depends on engine issue + drain timing
typically DELAY 120




VMATPUSH.ACC.BF16 immediately followed by VMATPOP.FP8 in fused code
deterministic but multi-stage
often DELAY 80



Standalone VPU unary / binary / row-reduction / VLI op
no simple software-visible ready signal beyond issueBusy

typically DELAY 150



VPU column reduction (VREDSUM.BF16, VREDMIN.BF16, VREDMAX.BF16)
long reduction path
typically DELAY 250



Two intentionally overlapped VPU single-input ops issued back-to-back
current overlap suites wait after the pair
typically DELAY 300



DMA completion
use DMA.WAIT ch, not a fixed DELAY

DMA.WAIT ch




Legal overlap guide
Here, "legal" means the current hardware will accept the issue timing without asserting. It does not mean two commands are numerically sensible if they touch the same VMEM region or the same logical tensor contents.
The broad cross-engine rule is:

a new instruction that reads MREG bank mN is legal only if no engine is currently writing mN

a new instruction that writes MREG bank mN is legal only if no engine is currently reading or writing mN

for pair-based BF16 ops, reserve both banks in the pair


Scalar / LSU overlap



Pattern
Legal?
Conditions




scalar store -> scalar store
yes
stores are single-cycle masked VMEM writes


scalar store -> scalar load
yes
normal scalar register dependency rules still apply


scalar load -> unrelated scalar store
yes
the store must not need the load result immediately


scalar load -> scalar load
no
the first scalar load response is still pending


scalar load -> immediate consumer of loaded rd

no
use DELAY 1




VLOAD + VSTORE in flight together
yes
they use independent LSU FSMs; still avoid same-data races


scalar load + VSTORE

yes
structurally allowed because VMEM uses separate read and write ports


scalar store + VLOAD

yes
structurally allowed because VMEM uses separate write and read ports


scalar load + VLOAD to the same VMEM bank in the same cycle
no
asserted in VMEM.scala



scalar store + VSTORE to the same VMEM bank in the same cycle
no
asserted in VMEM.scala




VLOAD -> any instruction that reads or writes the destination MREG bank
no
the destination bank is write-busy until the load finishes



VSTORE -> any instruction that reads the source MREG bank
no
the source bank is read-busy until the store finishes



Practical examples:


SW ... followed immediately by VLOAD ... is legal, even on the same VMEM bank.

LW ... followed immediately by VSTORE ... is legal, even on the same VMEM bank.

LW ... followed immediately by another LW ... is illegal.

VLOAD 4, ... followed immediately by VTRPOSE.XLU ..., 4 is illegal until the VLOAD write into bank 4 completes.


VPU overlap
The current VPU has exactly one useful overlap mode: a second command may issue behind a still-running first command only when the second command is a single-input op whose issueBusy bit is clear.
The single-input ops that can participate in this overlap path are:

VMOV
VRECIP.BF16
VEXP
VEXP2
VFP8PACK
VFP8UNPACK
VSQUARE.BF16
VCUBE.BF16
VRELU
VSIN
VCOS
VTANH
VLOG2
VSQRT
VREDSUM.BF16
VREDMIN.BF16
VREDMAX.BF16
VLI.ALL
VLI.ROW
VLI.COL
VLI.ONE

The following VPU ops do not use the overlap path and should be treated as non-overlapping issue points:

VADD.BF16
VSUB.BF16
VMUL.BF16
VMIN.BF16
VMAX.BF16
VREDSUM.ROW.BF16
VREDMIN.ROW.BF16
VREDMAX.ROW.BF16

Even within the single-input set, the back-to-back pair is illegal if the new op shares the same logic family as the still-running op. The important cases are:

same op after itself, for example VMOV -> VMOV


VEXP with VEXP2


VSIN with VCOS


VSQUARE.BF16 with VCUBE.BF16

any VLI.* after another VLI.*


Practical examples:

legal: VEXP -> VSIN

legal: VMOV -> VFP8PACK

legal: VREDMAX.BF16 -> VLOG2

illegal: VEXP -> VEXP2

illegal: VSIN -> VCOS

illegal: VLI.ALL -> VLI.ONE

illegal: VADD.BF16 -> anything intended to use the overlap slot

All normal MREG rules still apply on top of issueBusy. For example, VMOV 4, 0 followed immediately by VEXP 6, 4 is still illegal because the second command tries to read a bank that the first command is still writing.

MXU overlap
Each MXU (MXU0 and MXU1) has three independently scheduled resources:

Slot A: compute or a single-port push
Slot B: a single-port push only
Slot C: pop only

Legal overlap patterns are therefore the combinations that land on different slots and avoid the explicit hardware conflicts on weightSlot, accSel, drain state, and MREG banks.



Pattern
Legal?
Conditions





VMATPUSH.W + VMATPUSH.ACC.FP8

yes
one uses Slot A and the other uses Slot B



VMATMUL{.ACC} + VMATPUSH.W

yes
the push uses the other read slot and weightSlot differs from the active compute



VMATMUL{.ACC} + VMATPUSH.ACC.FP8

yes

accSel differs from the active compute / drain target



VMATPUSH.{W or ACC.FP8} + VMATPOP.{FP8 or BF16}

yes
pop uses Slot C, pop destination MREG banks are not being read by the push, and accSel does not conflict



VMATMUL{.ACC} + VMATPOP.{FP8 or BF16}

yes
pop uses Slot C, pop accSel differs from the active compute / drain target, and destination MREG banks are free



VMATMUL{.ACC} + VMATPUSH.* + VMATPOP.*

yes
one command occupies each of A/B/C and all weightSlot / accSel / MREG-bank rules above are satisfied



VMATPUSH.ACC.BF16 overlapping with other pushes or compute
no
BF16 acc push needs both read slots and reads vs plus vs+1



new compute while Slot A is busy with push or active compute
no
only issue when Slot A is idle; do not rely on hidden backpressure


push into a weightSlot already used by active compute
no
asserted in both MXU sequencers


compute using a weightSlot currently being pushed
no
asserted in both MXU sequencers


push / pop / compute targeting the same accSel as an active compute drain
no
asserted in both MXU sequencers


pop writing an MREG bank currently being read by push/compute
no
asserted in both MXU sequencers


push reading an MREG bank currently being written by pop
no
asserted in both MXU sequencers



Practical examples used by checked-in tests:


VMATPUSH.W.MXU0 0, 5 -> VMATPUSH.ACC.FP8.MXU0 1, 12


VMATMUL.ACC.MXU0 0, 0, 0 -> VMATPUSH.W.MXU0 1, 7


VMATPUSH.ACC.FP8.MXU0 1, 12 -> VMATPOP.FP8.MXU0 8, 0, 0


VMATMUL.ACC.MXU0 0, 0, 0 -> VMATPUSH.W.MXU0 1, 7 -> VMATPOP.FP8.MXU0 8, 0, 1



DMA and XLU overlap



Pattern
Legal?
Conditions





DMA.LOAD / DMA.STORE on different DMA channels
yes
software must still keep total in-flight commands within the 8-slot engine budget


reuse of the same DMA channel before DMA.WAIT ch

not a safe programming model
there is no launch-side full/busy check; treat it as one outstanding command per channel


DMA overlapping LSU traffic
yes
LSU has priority on each VMEM port; DMA may make slower progress when it loses arbitration



VTRPOSE.XLU overlapping another VTRPOSE.XLU

no
XLU only accepts a new command in Idle




VTRPOSE.XLU overlapping other engines that touch different MREG banks
yes
normal MREG read/write hazard rules still apply



VTRPOSE.XLU while source or destination bank is busy in another engine
no
blocked by the scalar core's MREG hazard tracking



If you are not intentionally hand-scheduling one of the legal patterns above, the conservative delays in the previous table are the safer programming model.