// Verilator driver for atlas_soc_sim: the host side of the TSI link at the word level.
// host/atlas_host.py (SimLink) drives it over stdin/stdout, one command per line, hex values:
//
//   W <addr> <w0> <w1> ...   write consecutive words           -> "K" (or "E <addr>" if denied)
//   R <addr> <n>             read n consecutive words          -> "D <w0> <w1> ..."
//   P <addr> <max_cycles>    read addr until nonzero, the core halts (CSR STATUS reason set),
//                            or the budget runs out            -> "P <value> <cycles> <status>"
//   C <n>                    run n idle cycles                 -> "C <cycle>"
//   B <addr> <path>          load the file at DRAM address addr through a backdoor, in no
//                            simulated time (the READY line lists B when supported) -> "K"
//   Q                        exit                              -> "Q <cycle>"
//
// An assertion in the RTL (sim build keeps Atlas's Verification layer) stops the run: the
// driver prints "X <cycle>" and exits nonzero.
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Vatlas_soc_sim.h"
#include "Vatlas_soc_sim___024root.h"
#include "verilated.h"

static VerilatedContext* ctx;
static Vatlas_soc_sim* top;
static uint64_t cycle = 0;

static void check_finish() {
    if (ctx->gotFinish()) {
        printf("X %llu\n", (unsigned long long)cycle);
        fflush(stdout);
        top->final();
        exit(2);
    }
}

// One clock: inputs are set while clk is low; fire conditions are sampled before the edge.
static void tick() {
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    ++cycle;
    ctx->timeInc(1);
    check_finish();
}

static bool access(bool write, uint32_t addr, uint32_t data, uint32_t& rdata) {
    top->h_a_valid   = 1;
    top->h_a_write   = write;
    top->h_a_address = addr;
    top->h_a_data    = data;
    top->h_d_ready   = 0;
    for (;;) {
        top->clk = 0;
        top->eval();
        bool fire = top->h_a_ready;
        top->clk = 1;
        top->eval();
        ++cycle;
        ctx->timeInc(1);
        check_finish();
        if (fire) break;
    }
    top->h_a_valid = 0;
    top->h_d_ready = 1;
    for (;;) {
        top->clk = 0;
        top->eval();
        bool fire = top->h_d_valid;
        bool denied = top->h_d_denied;
        rdata = top->h_d_data;
        top->clk = 1;
        top->eval();
        ++cycle;
        ctx->timeInc(1);
        check_finish();
        if (fire) {
            top->h_d_ready = 0;
            return !denied;
        }
    }
}

int main(int argc, char** argv) {
    ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);
    top = new Vatlas_soc_sim{ctx};
    top->h_a_valid = 0;
    top->h_d_ready = 0;
    top->rst = 1;
    for (int i = 0; i < 20; ++i) tick();
    top->rst = 0;
    for (int i = 0; i < 20; ++i) tick();
    printf("READY %llu B\n", (unsigned long long)cycle);
    fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        if (!(in >> cmd)) continue;
        if (cmd == "W") {
            uint32_t addr, w, r;
            in >> std::hex >> addr;
            bool ok = true;
            uint32_t bad = 0;
            while (in >> std::hex >> w) {
                if (!access(true, addr, w, r) && ok) { ok = false; bad = addr; }
                addr += 4;
            }
            if (ok) printf("K\n");
            else printf("E %x\n", bad);
        } else if (cmd == "R") {
            uint32_t addr, n, r;
            in >> std::hex >> addr >> n;
            printf("D");
            for (uint32_t i = 0; i < n; ++i, addr += 4) {
                access(false, addr, 0, r);
                printf(" %x", r);
            }
            printf("\n");
        } else if (cmd == "P") {
            uint32_t addr, r = 0, status = 0;
            uint64_t budget, start = cycle;
            in >> std::hex >> addr >> budget;
            while (cycle - start < budget) {
                access(false, addr, 0, r);
                if (r) break;
                access(false, 0x40008, 0, status);  // CSR STATUS: [0] halted, [2:1] reason
                if ((status & 1) && (status & 6)) break;
                for (int i = 0; i < 32; ++i) tick();
            }
            printf("P %x %llx %x\n", r, (unsigned long long)(cycle - start), status);
        } else if (cmd == "B") {
            uint32_t addr;
            std::string path;
            in >> std::hex >> addr >> path;
            auto& mem = top->rootp->atlas_soc_sim__DOT__u_soc__DOT__g_onchip__DOT__u_dram__DOT__mem;
            const uint64_t lines = sizeof(mem.m_storage) / sizeof(mem.m_storage[0]);
            FILE* f = fopen(path.c_str(), "rb");
            if (!f) {
                printf("E %x\n", addr);
            } else {
                std::vector<uint8_t> buf;
                uint8_t chunk[1 << 16];
                size_t n;
                while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
                fclose(f);
                for (size_t i = 0; i < buf.size(); ++i) {
                    uint64_t a = (uint64_t)addr + i;            // the RTL's aliasing: address bits [LINE_W+4:5]
                    uint32_t& w = mem.m_storage[(a >> 5) & (lines - 1)].m_storage[(a & 31) >> 2];
                    int sh = 8 * (a & 3);
                    w = (w & ~(0xFFu << sh)) | ((uint32_t)buf[i] << sh);
                }
                printf("K\n");
            }
        } else if (cmd == "C") {
            uint64_t n;
            in >> std::hex >> n;
            for (uint64_t i = 0; i < n; ++i) tick();
            printf("C %llx\n", (unsigned long long)cycle);
        } else if (cmd == "Q") {
            printf("Q %llx\n", (unsigned long long)cycle);
            fflush(stdout);
            break;
        }
        fflush(stdout);
    }
    top->final();
    delete top;
    delete ctx;
    return 0;
}
