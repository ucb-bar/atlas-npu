// Verilator co-simulation of atlas_tsi_soc through its UART pins, driven by PyUARTSI
// (atlas_host.TsiLink(cosim=...)). Adapted from ucb-bar/nvfp4@0c611e1:fpga/sim/tsi_cosim.cpp.
//
// stdin, one command per line:
//   W <hex>  queue bytes for the target's RX pin
//   R <n>    run until n bytes have left the TX pin, then print "D <hex>" (or "T" on timeout)
//   Q        send the queued bytes, run 20 more byte times, and exit
// Both UART directions use CLKS_PER_BIT clock cycles per bit, 8N1, the RTL's own divisor.
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iostream>
#include <string>

#include "Vatlas_tsi_soc.h"
#include "verilated.h"

static VerilatedContext* ctx;
static Vatlas_tsi_soc* top;
static uint64_t cycle = 0;

struct Driver {  // host -> target RX pin
    std::deque<uint8_t> q;
    uint16_t frame = 0;
    int bits = 0, cnt = 0;
    int step() {
        if (bits == 0 && cnt == 0 && !q.empty()) {
            frame = uint16_t(0x200 | (q.front() << 1));  // stop, data LSB first, start
            q.pop_front();
            bits = 10;
        }
        if (bits == 0) return 1;
        int level = frame & 1;
        if (++cnt == CLKS_PER_BIT) {
            cnt = 0;
            frame >>= 1;
            --bits;
        }
        return level;
    }
    bool idle() const { return q.empty() && bits == 0; }
} drv;

struct Monitor {  // target TX pin -> host
    std::string got;
    int state = 0, cnt = 0, bit = 0;
    uint8_t byte = 0;
    void step(int txd) {
        switch (state) {
            case 0:
                if (!txd) { state = 1; cnt = CLKS_PER_BIT / 2; }
                break;
            case 1:  // middle of the start bit
                if (--cnt == 0) {
                    if (txd) { state = 0; break; }
                    state = 2; cnt = CLKS_PER_BIT; bit = 0; byte = 0;
                }
                break;
            case 2:
                if (--cnt == 0) {
                    byte |= uint8_t(txd << bit);
                    cnt = CLKS_PER_BIT;
                    if (++bit == 8) state = 3;
                }
                break;
            case 3:  // middle of the stop bit
                if (--cnt == 0) {
                    got.push_back(char(byte));
                    state = 0;
                }
                break;
        }
    }
} mon;

static void tick() {
    top->rxd = drv.step();
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    mon.step(top->txd);
    ++cycle;
    ctx->timeInc(1);
    if (ctx->gotFinish()) {
        printf("X %llu\n", (unsigned long long)cycle);
        fflush(stdout);
        exit(2);
    }
}

int main(int argc, char** argv) {
    ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);
    top = new Vatlas_tsi_soc{ctx};
    top->rxd = 1;
    top->rst = 1;
    for (int i = 0; i < 16; ++i) tick();
    top->rst = 0;
    for (int i = 0; i < 16; ++i) tick();

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line[0] == 'W') {
            for (size_t i = 2; i + 1 < line.size(); i += 2)
                drv.q.push_back(uint8_t(std::stoi(line.substr(i, 2), nullptr, 16)));
        } else if (line[0] == 'R') {
            size_t n = std::stoul(line.substr(2));
            uint64_t limit = cycle + 20ull * CLKS_PER_BIT * (n + drv.q.size() + 64) + 4000000;
            while (mon.got.size() < n && cycle < limit) tick();
            if (mon.got.size() < n) {
                printf("T\n");
            } else {
                printf("D ");
                for (size_t i = 0; i < n; ++i) printf("%02x", uint8_t(mon.got[i]));
                printf("\n");
                mon.got.erase(0, n);
            }
            fflush(stdout);
        } else if (line[0] == 'Q') {
            while (!drv.idle()) tick();
            for (int i = 0; i < 200 * CLKS_PER_BIT; ++i) tick();
            printf("Q %llu %zu\n", (unsigned long long)cycle, mon.got.size());
            fflush(stdout);
            break;
        }
    }
    top->final();
    return 0;
}
