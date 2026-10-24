# TSI over UDP

UART TSI runs at 921,600 baud, about 90 KB/s. This rate limits how fast a host loads kernels and
data into the standalone Atlas SoC (standalone/README.md). The VCU118 build thus has a second host
link on the board's 1000BASE-T port: TSI over UDP. The host sends plain UDP datagrams. Each datagram
carries TSI commands in the UART TSI byte format (PyUARTSI, Chipyard UART-TSI) behind a short
sequence header. The FPGA runs each datagram once, in sequence order, and sends one reply datagram.

UART TSI stays as the fallback and debug link. The two links share the SoC's fabric through
`tl_arbiter`, so a host can use both at the same time.

## Protocol

The FPGA listens on UDP port 21587 (`0x5453`, "TS"). All fields are little-endian 32-bit words, as
in UART TSI. A UDP payload is a whole number of words.

Request (host to FPGA):

```text
word 0   seq     sequence number (u32, wraps)
word 1   flags   [0] SYNC: open a session. The other bits are 0.
word 2+  TSI commands, whole commands only:
         cmd (0 = read, 1 = write), addr[31:0], addr[63:32], len[31:0], len[63:32]  (len = words - 1),
         then, for a write, len + 1 data words
```

Reply (FPGA to the source IP address and UDP port of the request):

```text
word 0   seq     the request's seq
word 1   flags   [0] SYNC: the session is open
                 [1] DUP: this seq ran before and did not run again
                 [2] NOSYNC: no session for this host (send SYNC)
                 [3] DROP: did not run, because an earlier datagram is missing
word 2   expect  the next seq that the FPGA runs
word 3   status  [15:0] commands run
                 [16] CUT: the datagram ended inside a command
                 [17] BADCMD: a cmd other than 0 or 1 (the FPGA skips the rest of the datagram)
                 [18] OVERFLOW: the reads asked for more than 364 words (the FPGA skips the excess)
                 [19] DENIED: a fabric response in this datagram was denied
word 4+  read data: every word that the datagram's reads returned, in order, at most 364 words
```

The FPGA sends the reply only after every command of the datagram is complete, including the write
responses. A reply thus also means "all writes are done". A SYNC, DUP, NOSYNC or DROP reply has no
read data.

### Sessions

`eth_tsi` keeps one session. A session belongs to one peer: the IP address and UDP port that sent
the SYNC that opened it. `eth_tsi` also keeps the seq of that SYNC, the opening seq. The expect
value is the next seq that the session runs.

| The FPGA receives | The FPGA does |
|---|---|
| SYNC from the session's peer, with the opening seq | Replies SYNC with the current expect. The session does not change. This SYNC is a copy of the one that opened the session. |
| Any other SYNC | Opens a new session for the sender, with expect = seq + 1, and replies SYNC. The old session ends. A host restarts this way. |
| A datagram without SYNC, and no session is open | Replies NOSYNC. Runs nothing. |
| A datagram without SYNC from a peer other than the session's | Replies NOSYNC. Runs nothing. The session does not change. |
| seq = expect | Runs the commands, replies, and increments expect. |
| seq older than expect ((expect - seq) as signed 32-bit > 0) | Replies DUP. Runs nothing. The host lost the reply, not the request. |
| seq newer than expect | Replies DROP. Runs nothing. An earlier datagram was lost. |

A SYNC carries no commands: `eth_tsi` ignores the words after word 1. A datagram runs only when its
seq equals expect, so each seq runs at most once. A retransmitted write never runs twice, for
example the EXEC_CONTROL write that starts a kernel. A late copy of the opening SYNC does not reset
expect, so it cannot make a seq run again.

### Short, malformed and cut datagrams

`eth_tsi` drops a payload that is shorter than 8 bytes, or not a whole number of words, without a
reply. ETH_DROP counts it.

A reset of the GMII side (link loss, PCS/PMA reset) can cut a datagram while it crosses to the core
clock. `eth_udp_stack` then ends that datagram with an extra word of stale data, marked with
`rx_abort`. A datagram that ends before its UDP length is also a cut datagram. `eth_tsi` never uses
the last word of a cut datagram:

- Cut before the first command (in seq or flags): `eth_tsi` drops the datagram without a reply, and
  ETH_DROP counts it. The session does not change.
- Cut inside the commands: the commands received whole run. The whole words of a partly received
  write line are written too. The reply has CUT set, and expect increments. The host must not send
  that seq again, because writes before the cut ran. EthLink raises LinkError.

`eth_tsi` ignores words past the UDP length.

### Commands and addresses

The decoder works like `uart_tsi`, but on the 32-byte beat fabric. It takes one word per cycle while
the fabric keeps up.

- VMEM and DRAM (from `0x2000_0000`): the write words of one 32-byte line merge into one masked Put.
  The Put goes out while the next line fills. A read Gets each line once.
- sys_regs, IMEM and CSR (below `0x2000_0000`): one transaction per word, as from `tl_widen`.
- Addresses at or above 4 GiB are not on the bus. Writes there are dropped and reads return 0, as in
  `uart_tsi`.

### Denied accesses

The SoC answers these accesses at once with a denied response (status bit DENIED, read data 0):

- An unmapped address (`tl_xbar`).
- An IMEM read while the scalar core runs. IMEM serves the instruction fetch then. IMEM writes work
  while the core runs.
- On the board, a DRAM access before the DDR4 is calibrated (sys_regs FLAGS bit 3 is 0).

A request that a device cannot serve thus never holds the fabric, and the other link keeps working.
sys_regs DENIED counts the denied responses of both links.

### Size limits

- UDP payload: at most 1472 bytes, one 1500-byte IPv4 packet. The FPGA drops IP fragments.
- Read words per datagram: at most 364 (REPLY_WORDS). A full reply is 16 + 4 x 364 = 1472 bytes.
- One write command: at most (1472 - 8 - 20) / 4 = 361 words. Several commands can share a datagram.

## RTL structure

```text
host NIC
   │  1000BASE-T, RJ45
DP83867 PHY                PHY address 3, configured over MDIO by vcu118_sgmii
   │  SGMII over LVDS, 6-wire: the PHY drives a 625 MHz clock
vcu118_sgmii               PCS/PMA IP gig_ethernet_pcs_pma_0, PHY reset and MDIO bring-up
   │  GMII, 8 bits at 125 MHz (gmii_clk)
eth_udp_stack              MAC, ARP, IPv4, UDP (verilog-ethernet), datagram FIFO,
   │                       async FIFOs to the core clock
   │  datagram words (32 bits, core clock), rx_abort
eth_tsi                    sessions, TSI decoder, reply buffer (364 words)
   │  beat fabric (32-byte TL-UL), one transaction in flight
tl_arbiter ─── tl_widen ─── uart_tsi ─── UART TSI
   │
tl_xbar ─── sys_regs, IMEM, CSR, VMEM, DRAM
```

| Module | File | Function |
|---|---|---|
| `vcu118_sgmii` | `standalone/rtl/eth/vcu118_sgmii.sv` | PCS/PMA IP (`standalone/fpga/vcu118_sgmii_pcs.tcl`) and the DP83867 bring-up, after verilog-ethernet's `example/VCU118/fpga_1g` |
| `eth_udp_stack` | `standalone/rtl/eth/eth_udp_stack.sv` | UDP/IPv4 endpoint on the vendored verilog-ethernet (`standalone/rtl/eth/vendor`, upstream 77320a9, MIT) |
| `eth_tsi` | `standalone/rtl/eth/eth_tsi.sv` | the protocol above: sessions, decoder, reply |
| `tl_arbiter` | `standalone/rtl/tsi/tl_arbiter.sv` | two hosts on the beat fabric: host 0 is UART TSI, host 1 is Ethernet TSI (from `ucb-bar/nvfp4@0c611e1`) |
| `atlas_tsi_soc` | `standalone/rtl/atlas_tsi_soc.sv` | `uart_tsi`, `eth_tsi` and `atlas_soc` |
| `vcu118_atlas_top` | `standalone/rtl/vcu118_atlas_top.sv` | board top: `vcu118_sgmii`, `eth_udp_stack`, `atlas_tsi_soc`, DDR4. Parameters ETH_IP and ETH_MAC. |

`vcu118_sgmii` brings up the PHY in the core clock after each reset:

1. It holds `phy_reset_n` low for 1 ms, then waits 2 ms.
2. It reads PHYIDR1 and PHYIDR2 at PHY address 3. `phy_id_ok` is 1 when they are `0x2000` and
   `0xA23x` (DP83867, x is the silicon revision).
3. It writes CFG4 = `0x0070`, SGMIICTL1 = `0x4000` (6-wire SGMII: the PHY drives the 625 MHz clock)
   and 10M_SGMII_CFG = `0x0015` through REGCR and ADDAR.
4. 1 ms after the last write, it releases the PCS/PMA reset. `phy_ready` becomes 1.

The PCS/PMA clocks, `gmii_clk` among them, run only after the PHY drives the 625 MHz clock. The MDIO
clock is at most 2.5 MHz.

`eth_udp_stack` receives UDP datagrams to ETH_IP and port 21587 that are not IP fragments. It puts a
prefix in front of each payload (peer IP, payload length, peer port) and keeps whole datagrams in a
32 KiB FIFO. A datagram that does not fit is dropped. Receive never waits on the core, so ARP keeps
working while `eth_tsi` is busy or the core is in reset. To send, `eth_udp_stack` takes the prefix
of the reply as the UDP header (destination IP and port, source port 21587, TTL 64) and computes the
UDP checksum. The subnet mask is 0, so the FPGA sends an ARP request for each peer directly, with no
gateway. ICMP and other IP protocols are discarded.

## Clock domains

| Clock | Source | Logic |
|---|---|---|
| core clock `clk` | MMCM from the 250 MHz system clock, CLK_MHZ (default 50 MHz) | `atlas_soc`, `uart_tsi`, `eth_tsi`, sys_regs, the bring-up and MDIO of `vcu118_sgmii`, the core side of the `eth_udp_stack` FIFOs |
| GMII clock `gmii_clk` | PCS/PMA `clk125_out`, 125 MHz, from the PHY's 625 MHz clock | MAC, ARP, IPv4, UDP, receive filter, datagram FIFO |
| SGMII clocks | the PHY's 625 MHz clock and the PCS/PMA's 312.5 MHz clock | PCS/PMA LVDS transceiver |
| MIG user clock | 200 MHz from the DDR4 MIG | the AXI side of `atlas_ddr4_dram` |

These signals cross between the core clock and the GMII clock:

- Datagrams: `axis_async_fifo_adapter` in each direction, 8 to 32 bits on receive and 32 to 8 bits
  on transmit. The receive FIFO's bad-frame mark becomes `rx_abort`.
- ETH_IP to the GMII clock, and the receive counters (ETH_RXERR, ETH_RXFRAMES) to the core clock:
  `eth_cdc_word`, a request/acknowledge handshake. The destination takes the data 2 cycles after the
  synchronized request edge.
- PCS/PMA status: a two-flop synchronizer per bit in `atlas_tsi_soc`.
- `gmii_alive`: `vcu118_sgmii` synchronizes a divided `gmii_clk` into the core clock. `gmii_alive`
  is 0 when that signal does not toggle for 4 us, or while the PCS/PMA is in reset.
- GMII reset: `eth_udp_stack` stretches `rst_125_out` to at least 8 core-clock cycles (sized from
  CLK_HZ). Both sides of each async FIFO thus see the reset.

`standalone/fpga/build.tcl` puts the core clock, the MIG user clock and the PCS/PMA clocks in
asynchronous clock groups. It also sources verilog-ethernet's constraint scripts
(`axis_async_fifo.tcl`, `eth_mac_fifo.tcl`, `sync_reset.tcl`). The clock groups take precedence over
the max-delay constraints of those scripts, but the bus-skew constraints still apply. The crossings
thus rely on their design: gray-coded FIFO pointers, and the margin of the `eth_cdc_word` handshake.

## Board setup

Ethernet is in every board build of `standalone/fpga/build_bitstream.sh`. The pins are in
`standalone/fpga/vcu118_eth.xdc` (bank 64, 1.8 V). The SGMII lines are AC-coupled, so the receive
and 625 MHz clock inputs use the internal 100-ohm termination and DQS_BIAS.

1. Connect the VCU118's RJ45 port to a 1 Gb/s NIC of the host with an Ethernet cable.
2. Give the NIC an address on 172.20.0.0/16:

   ```bash
   sudo ip addr add 172.20.0.1/16 dev <nic>
   sudo ip link set <nic> up
   ```

3. Program the FPGA (standalone/README.md, "Build and run").
4. Wait about 3 s. The PHY negotiates the copper link again after each programming and after each
   CPU_RESET.
5. Check that the FPGA answers ARP: `arping -I <nic> -c 3 172.20.1.18`. The FPGA has no ICMP, so
   `ping` gets no answer.
6. Run the tests over Ethernet:

   ```bash
   python3 standalone/host/run_tests.py --eth --config ipt-nvfp4 --out build/board-eth build/bundle/nodelay/*.json
   ```

The FPGA's address is 172.20.1.18 by default. Any host address on 172.20.0.0/16 works, because the
FPGA replies to the source address of each request.

### Check the link

Read ETH_STATUS over UART TSI when the Ethernet link does not answer:

```python
import sys; sys.path.insert(0, "standalone/host")
from atlas_host import TsiLink, SYS_ETH_STATUS
print(hex(TsiLink().read_words(SYS_ETH_STATUS, 1)[0]))
```

Find the first bit in this table that is 0:

| ETH_STATUS bit | If the bit is 0 |
|---|---|
| [18] Ethernet present | The bitstream has no Ethernet. |
| [19] PHY ID matched | MDIO does not read a DP83867 at PHY address 3. |
| [16] PHY bring-up done | The MDIO sequence is not complete. It takes about 5 ms after reset. |
| [20] GMII clock running | The PHY does not drive the 625 MHz clock, or the PCS/PMA is in reset. |
| [1] link sync | The PCS/PMA has no SGMII sync with the PHY. |
| [7] PHY link status | The copper link is down. Check the cable and the host NIC. |
| [0] link status | SGMII autonegotiation between the PCS/PMA and the PHY is not complete. |
| [17] session open | No host has opened a session since reset. |

On a 1 Gb/s full-duplex link, bits [0], [1], [7] and [12] are 1, and [11:10] is 2 (`0b10`).
ETH_RXERR counts the frames that the FPGA dropped before `eth_tsi`, and the frames with a bad FCS.
If a datagram gets no reply, read ETH_RXFRAMES over UART TSI before and after you send it. Both
halves count all received frames, also frames that are not for the FPGA. If [15:0] does not
increase, the frame did not get to the GMII interface: it is lost or waits in the PHY, the PCS/PMA
or a switch. If [15:0] increases but [31:16] does not, the frame waits in the MAC, or the MAC
dropped it and ETH_RXERR increases.

### Change the FPGA's address

Write the new address to sys_regs ETH_IP (`0x030`) over either link. For example, over UART TSI:

```python
TsiLink().write_words(0x030, [0xAC14_0213])      # 172.20.2.19
```

Over Ethernet, the write runs, but its reply can come from the new address. The open EthLink does
not accept that reply and raises LinkError. Open a new EthLink at the new address. ETH_IP returns to
the build's value (parameter ETH_IP of `vcu118_atlas_top`) after CPU_RESET and after programming.

## System registers

| Offset | Name | |
|---|---|---|
| `0x030` | ETH_IP | the FPGA's IPv4 address, read/write. Reset value: parameter ETH_IP, default 172.20.1.18 (`0xAC14_0112`) |
| `0x034` | ETH_STATUS | [15:0] PCS/PMA status, 0 while the GMII clock is stopped ([0] link status, [1] link sync, [7] PHY link status, [11:10] speed, 2 = 1 Gb/s, [12] full duplex), [16] PHY bring-up done (MDIO writes sent), [17] session open, [18] Ethernet present in this build, [19] PHY ID matched (DP83867), [20] GMII clock running |
| `0x038` | ETH_RUN | datagrams run |
| `0x03C` | ETH_DUP | datagrams answered DUP (not run again) |
| `0x040` | ETH_DROP | datagrams dropped or not run: malformed, cut before the commands, NOSYNC, DROP |
| `0x044` | ETH_PORT | UDP port, 21587 (`0x5453`) |
| `0x048` | ETH_MAC_LO | MAC address [31:0] |
| `0x04C` | ETH_MAC_HI | MAC address [47:32]. The default MAC is `02:00:41:54:4c:53`. |
| `0x050` | ETH_RXERR | [15:0] frames dropped before `eth_tsi`, [31:16] frames with a bad FCS or a GMII error |
| `0x054` | ETH_RXFRAMES | [15:0] frames at the GMII input (rising edges of GMII rx_dv), [31:16] frames out of the MAC |

ETH_RUN, ETH_DUP and ETH_DROP reset with the SoC. The ETH_RXERR and ETH_RXFRAMES counters wrap at
16 bits and reset with the GMII side. ETH_RXERR [15:0] counts these frames: MAC FIFO full, bad
frames, IP header errors, UDP to ETH_IP on another port, IP fragments, datagram FIFO full, payloads
shorter than the UDP length, and datagrams cut by a core reset. In a build without Ethernet (the
word-level Verilator model), ETH_STATUS and the counters read 0.

## Host software

### EthLink

`standalone/host/atlas_host.py` has `EthLink`, with the same interface as `SimLink` and `TsiLink`
(`write_words`, `read_words`, `poll`, `close`):

```python
from atlas_host import Atlas, EthLink

link = EthLink()                                 # 172.20.1.18:21587; opens a session (SYNC)
atlas = Atlas(link)                              # checks the sys_regs ID
link.write_bulk(0x9000_0000, data)               # bytes, whole little-endian 32-bit words
outputs = link.read_words(0x9000_0800, 256)
print(link.stats)                                # datagrams, resends, dups, drops, timeouts, ...
link.close()
```

`EthLink(ip, port, window, timeout, retries, cosim, seq, nudge)` has the defaults
`ip="172.20.1.18"`, `port=21587`, `window=16`, `timeout=0.05` (seconds), `retries=50` and
`nudge=0.001` (seconds). `cosim=BINARY` starts the
Ethernet model and connects to it. `batch(ops)` runs a list of `("w", addr, words)` and `("r", addr,
n)` in order, pipelined, and returns the read words. `Atlas.run` loads DRAM with `write_bulk` when
the link has `bulk`.

EthLink follows these rules:

- It opens the session with a SYNC at a random seq. It sends the SYNC again for at least 6 s
  (ETH_OPEN_S) and at least `retries` x `timeout`, because the copper link needs about 3 s after
  programming or CPU_RESET. The wait for each SYNC doubles from `timeout` up to 10 x `timeout` (50
  ms to 500 ms with the defaults).
- Up to `window` datagrams are unacknowledged. After an RTO expiry or a DROP reply, EthLink sends
  again from the oldest unacknowledged datagram (go-back-N). It goes back at most once for each DROP
  expect, until the next RTO expiry.
- When nothing arrives for `nudge` seconds after the last datagram that it sent or received, EthLink
  sends the newest unacknowledged datagram once more. On the VCU118, a lone frame can wait in the
  receive path until the next frame arrives (see Results). The copy releases it, and the board
  answers the copy DUP. `nudge=None` turns this off; the model never uses it.
- The retransmission timeout (RTO) follows the measured round-trip time (RFC 6298). It doubles each
  time it expires, up to 16 x `timeout` on the board and 30 s on the model.
- EthLink raises LinkError when the board acknowledges nothing for `retries` x `timeout` (about 2.5
  s with the defaults) and the RTO expired at least 3 times. On the model, `timeout` is at least 3 s
  (ETH_SIM_TIMEOUT), so EthLink raises LinkError after about 150 s without progress.
- A DUP reply has no data. If the reply of a read datagram was lost, EthLink sends those reads again
  in a new datagram. A datagram with writes waits until every earlier read has its data, so a read
  sent again never sees a later write.
- Each call returns when the board has acknowledged all its datagrams. Then its writes are done.

EthLink raises LinkError in these cases:

- The board does not answer (no progress, as above). The error text gives the time waited and the
  number of tries.
- The board answers NOSYNC to a datagram of the session. The board was reset or programmed again, or
  another host opened a session.
- A DUP or DROP reply has an expect past every seq that the link sent. Another SYNC restarted the
  session, for example from a second host.
- The data of a read datagram was lost more than `retries` times.
- A reply has CUT, BADCMD or OVERFLOW, or fewer commands or read words than the datagram sent.
- A write is denied. A denied read returns 0 and increments `stats["denied"]`.

Use one EthLink at a time. A second EthLink, in the same program or in another, opens a new session,
and the first link then raises LinkError. UART TSI (`TsiLink`) works at the same time as EthLink.

### run_tests.py

```bash
python3 standalone/host/run_tests.py --eth --config ipt-nvfp4 --out build/board-eth build/bundle/nodelay/*.json
python3 standalone/host/run_tests.py --eth 172.20.2.19:21587 --out build/board-eth build/bundle/nodelay/*.json
python3 standalone/host/run_tests.py --eth-sim build/vl_eth/atlas_eth_sim --config ipt-nvfp4 -j 16 \
    --out build/sim-eth build/bundle/nodelay/*.json
```

`--eth [IP[:PORT]]` runs the tests one after another on the board (default 172.20.1.18:21587).
`--eth-sim BINARY` runs each test in a new Ethernet model on its own UDP port, in parallel with
`-j`. The completion budget is 20 s on the board and 600 s on the model (`--budget`). At the end,
`run_tests.py` prints the EthLink statistics.

### link_bench.py

`standalone/host/link_bench.py` measures the write and read throughput and the latency of
single-word transactions, for each link that you give:

```bash
python3 standalone/host/link_bench.py --eth --uart
python3 standalone/host/link_bench.py --eth-sim build/vl_eth/atlas_eth_sim --tsi-sim build/vl_tsi/atlas_tsi_sim
```

It writes random words into DRAM with `write_words` (and with `write_bulk` on Ethernet), reads them
back and compares. Then it times reads of sys_regs ID and writes of SCRATCH. The output is a table:
MiB, write MB/s, bulk MB/s, read MB/s, errors, and read and write latency (median / 99th percentile,
in us). For Ethernet it also prints the EthLink statistics and the board's counters. `--mb` sets the
data size (default 16 MiB on Ethernet, 256 KiB on the UART, 64 KiB on the Ethernet model, 1 KiB on
the UART model). `--small` sets the number of timed transactions, `--addr` the DRAM address (default
`0x9000_0000`). `--window` and `--timeout` go to EthLink.

### test_ethlink.py

`standalone/host/test_ethlink.py` tests EthLink against MockBoard, a UDP server that follows the
rules of `eth_tsi` over a model of the address map. Its network model drops, duplicates, reorders
and delays datagrams. The tests check that each write runs exactly once.

```bash
python3 standalone/host/test_ethlink.py -v
python3 standalone/host/test_ethlink.py -k test_heavy_loss
```

`--serve [PORT] [--loss P] [--seed N] [--dram-mb N] [--ready-first]` runs MockBoard as a stand-in
for the Ethernet model. It prints `PORT <n>` (for port 0) and `READY`, and stops on end of input or
SIGTERM. `--eth-sim` takes one executable, so put the command in a script:

```bash
printf '#!/bin/sh\nexec python3 %s/standalone/host/test_ethlink.py --serve "$@"\n' "$PWD" > build/mock_eth_sim
chmod +x build/mock_eth_sim
python3 standalone/host/link_bench.py --eth-sim build/mock_eth_sim
```

## Simulation

| Command | What it runs |
|---|---|
| `MODEL=eth standalone/sim/build.sh build/rtl-sim build/vl_eth` | builds `build/vl_eth/atlas_eth_sim`: `eth_udp_stack` and `atlas_tsi_soc` (`standalone/sim/atlas_eth_sim.sv`) behind a UDP socket (`standalone/sim/eth_cosim.cpp`) |
| `standalone/sim/tb_eth_tsi.sh [out_dir] [+n=N] [+seed=N]` | `eth_tsi` alone against a beat-fabric memory with random latency, and `tl_arbiter` with two random hosts |
| `standalone/sim/tb_eth_udp_stack.sh [out_dir] [seed] [clk_hz]` | `eth_udp_stack` alone on raw GMII frames, with a core clock within 4% of clk_hz (default 50000000) |
| `python3 standalone/host/test_ethlink.py` | EthLink against MockBoard (above) |

The Ethernet model plays the PHY and the host's network. It runs the core clock at 50 MHz and
`gmii_clk` at 125 MHz. It binds 127.0.0.1:PORT and prints `READY` when it listens. PORT is the first
argument or `ETH_SIM_PORT` (default 21587). Port 0 picks a free port and prints `PORT <n>`. It wraps
each datagram that it receives into an Ethernet/IPv4/UDP frame from the host, and answers the FPGA's
ARP requests. It checks each frame from the FPGA (FCS, checksums, lengths, addresses) and sends the
payload back. It stops on SIGTERM, on end of input, or on `Q`. `S` prints `S <cycles> <wall s>
<paused s> <datagrams in> <datagrams out> <queue drops>`. An RTL assertion prints `X <cycle>` and
stops the model.

| Variable | Effect |
|---|---|
| `ETH_SIM_LOSS=p` | In each direction, drops, duplicates or holds back each datagram, each with probability p (default 0). |
| `ETH_SIM_SEED=n` | Seed of the loss injection (default 1). |
| `ETH_SIM_HOST_IP` | The host's address on the simulated link (default 172.20.0.1). |
| `ETH_SIM_FPGA_PORT` | The FPGA's UDP port (default 21587). |
| `ETH_SIM_NOIDLE=1` | Never pause. By default the model pauses while the core is halted and the link was quiet for 50,000 cycles. |

The model sends frames to the FPGA at GMII line rate, which is far slower in simulation than the
host. Frames for the FPGA thus wait in a queue of at most 64 KiB, as in the buffer of a NIC or
switch. ARP replies always go in. The model drops the datagrams that do not fit and counts them as
queue drops: in the statistics at exit, and as the last field of the `S` line.

`tb_eth_tsi` covers SYNC, run in order, DUP, DROP, NOSYNC, CUT, BADCMD, OVERFLOW, DENIED, unaligned
and partial lines, the word path to sys_regs, IMEM and CSR, addresses at or above 4 GiB,
back-to-back datagrams and reply backpressure. It also covers cut datagrams: `rx_abort` in each
phase, and datagrams shorter or longer than their UDP length. For the session rules, it checks that
another peer gets NOSYNC, that a copy of the opening SYNC changes nothing, and that a SYNC from
another peer takes over. `tb_eth_udp_stack` checks ARP, the words on the core side, the reply frames
(headers, checksums, FCS), filtering and counters, line rate, and a change of ETH_IP. It also
applies GMII resets of 1 to 50 cycles: each datagram that the core gets is whole, or ends with one
word marked `rx_abort`, and nothing stalls. Run it with clk_hz 20000000 too: a slower core clock
needs a longer GMII reset stretch. Below about 43 MHz the test skips line rate, because the core
then takes less than line rate.

## Resources and timing

Vivado 2024.1, 50 MHz, all three configurations meet timing (`bitstreams/AtlasStandaloneEth`, with
ETH_RXFRAMES).
`ipt-nvfp4` uses `SSI_SpreadLogic_high` placement and `Explore` routing (build_bitstream.sh
arguments 5 and 6); the others use the defaults.

| Config | WNS / WHS (ns) | PCS/PMA 312.5 MHz WNS (ns) | LUTs |
|---|---|---|---|
| `default` | +0.223 / +0.010 | +0.384 | 624K (52.8%) |
| `ipt` | +0.101 / +0.010 | +0.101 | 362K (30.6%) |
| `ipt-nvfp4` | +0.130 / +0.010 | +0.664 | 543K (46.0%) |

| Module | LUTs | FFs | Block RAM |
|---|---|---|---|
| `eth_udp_stack` | 2,264 | 3,609 | 14 RAMB36 + 4 RAMB18 |
| `vcu118_sgmii` (with the PCS/PMA IP) | 945 | 1,247 | |
| `eth_tsi` | 1,085 | 1,403 | 1 RAMB18 |
| `tl_arbiter` | 189 | 4 | |

Together that is about 4,500 LUTs and 6,300 flip-flops.

## Results

VCU118, `ipt-nvfp4` bitstream, host NIC at 1 Gb/s through a switch (2026-10-23). The link comes up
with ETH_STATUS `0x1f388b`: PHY ID matched, SGMII sync, 1 Gb/s full duplex.

`link_bench.py --eth --uart` (16 MiB into DDR4 over Ethernet, 256 KiB over UART; 0 errors):

| Link | write_words | write_bulk | read_words | read latency (median / p99) |
|---|---|---|---|---|
| Ethernet | 62.5 MB/s | 98 MB/s | 38.6 MB/s | 59 us / 2.4 ms |
| UART TSI | 0.077 MB/s | | 0.091 MB/s | 449 us / 536 us |

| Run | UART TSI | Ethernet |
|---|---|---|
| 126 baremetal tests (`run_tests.py`, no-DELAY bundles) | 93 s | 1.0 s |
| specialist_dit policy (`ditflow.flow run`, 8.1 MiB image, 1.15 s kernel) | 110 s | 1.7 s |

The test results over Ethernet equal those over UART TSI in status, instret, DBG0, DBG1 and all
264,384 checked words. Only the cycle counts differ, by DDR4 timing (median 0.4%). The policy is
bit-exact with its golden in both runs.

A frame that arrives at the board right after the board sent a reply sometimes waits until the next
frame arrives: up to 1 s, the period of a broadcast on the test network. With requests sent less
than 20 us after the previous reply, 25 to 30% of them waited; with 50 us or more, none of 1,200
did. A lone datagram to another port never waited. ETH_RXFRAMES shows that the frame waits before
the GMII interface: in 98 of 100 held requests, the GMII count did not include the held frame (in
the other 2, a broadcast was counted in its place). So the frame waits in the switch, the DP83867
PHY or the PCS/PMA IP, not in the MAC, the UDP stack or eth_tsi. A direct cable from the host NIC to
the board, without the switch, would tell the switch from the PHY. EthLink's nudge (above) sends the
newest datagram again after 1 ms without a reply. With it, a 16 MiB `link_bench` run has 0 timeouts
(about 500 nudges, answered DUP); without it, 548 timeouts of 50 ms each.

## Limits

- The FPGA has no ICMP and no DHCP. `ping` gets no answer. Use `arping`, and give the host a static
  address.
- The FPGA answers only on UDP port 21587. Datagrams to other ports count in ETH_RXERR [15:0], and
  the host gets no "port unreachable".
- One session at a time. A SYNC from another host takes over the session.
- The host must be on the same Ethernet segment as the FPGA. The FPGA has no gateway: it sends an
  ARP request for each peer directly.
- CPU_RESET resets the PHY. The copper link is down for about 3 s.
- ETH_IP returns to the build's value after CPU_RESET.
- IMEM reads while the scalar core runs are denied and return 0. Stop the core first.
- On the board, DRAM accesses before the DDR4 is calibrated are denied.
- A frame that arrives right after the board sent a reply can wait until the next frame, before the
  FPGA (see Results). EthLink works around it; a host that does not use EthLink sees some replies
  late, up to the time to the next frame.
