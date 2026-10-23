// Verilator co-simulation of atlas_eth_sim (eth_udp_stack + atlas_tsi_soc) behind a real UDP
// socket. The driver plays the PHY and the host's network. A datagram that arrives on
// 127.0.0.1:<port> goes to the FPGA as an Ethernet/IPv4/UDP frame on GMII rx, from HOST_IP and
// the sender's UDP port. A UDP frame from GMII tx goes back to that sender. ARP requests for
// HOST_IP get a reply with a fake host MAC. atlas_host.EthLink talks to it as to the board.
//
//   atlas_eth_sim [port]   UDP port (or env ETH_SIM_PORT); default 21587, 0 picks a free port
//
// stdout: "PORT <n>", then "READY <n>" when the model is out of reset and the socket listens.
// "X <cycle>" when an RTL assertion stops the run (exit code 2). The model runs until SIGTERM,
// EOF on stdin, or "Q" on stdin (prints "Q <cycle>"). "S" on stdin prints
// "S <cycles> <wall s> <paused s> <datagrams in> <datagrams out>". Statistics go to stderr at exit.
//
// Environment:
//   ETH_SIM_LOSS=p      each datagram, in each direction, is dropped with probability p,
//                       duplicated with probability p, and held back behind the next one with
//                       probability p (default 0)
//   ETH_SIM_SEED=n      seed of the loss injection (default 1)
//   ETH_SIM_HOST_IP     the host's IPv4 address on the simulated link (default 172.20.0.1)
//   ETH_SIM_FPGA_PORT   the FPGA's UDP port (default 21587)
//   ETH_SIM_NOIDLE=1    never pause the model (it pauses while the core is halted and the link
//                       has been quiet for 50,000 cycles, until a datagram or stdin arrives)
//
// Clocks: clk 50 MHz, gmii_clk 125 MHz (5 GMII cycles per 2 core cycles). The FPGA's IP and MAC
// come from the model (local_ip is sys_regs ETH_IP). Frames go out at GMII line rate with a
// 12-byte gap; checks on frames from the FPGA (FCS, IPv4 header checksum, UDP checksum and
// lengths, addresses) print "eth_cosim: bad frame" on stderr and count in the statistics.
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "Vatlas_eth_sim.h"
#include "verilated.h"

using Bytes = std::vector<uint8_t>;

static VerilatedContext* ctx;
static Vatlas_eth_sim* top;
static uint64_t cycle = 0;  // core clock cycles
static volatile sig_atomic_t stop_req = 0;

static const uint8_t HOST_MAC[6] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x01};
static uint32_t host_ip = 0xAC140001;  // 172.20.0.1
static uint16_t fpga_port = 21587;
static uint16_t ip_id = 0;

static struct {
    uint64_t in = 0, in_bytes = 0, out = 0, out_bytes = 0, arp = 0, bad = 0, oversize = 0;
} st;

// ── Byte helpers, CRC-32, Internet checksum ─────────────────────────────────
static void put16(Bytes& b, uint32_t v) { b.push_back(uint8_t(v >> 8)); b.push_back(uint8_t(v)); }
static void put32(Bytes& b, uint32_t v) { put16(b, v >> 16); put16(b, v & 0xFFFF); }
static void put_mac(Bytes& b, const uint8_t* m) { b.insert(b.end(), m, m + 6); }
static uint32_t get16(const uint8_t* p) { return uint32_t(p[0]) << 8 | p[1]; }
static uint32_t get32(const uint8_t* p) { return get16(p) << 16 | get16(p + 2); }

static uint32_t crc32(const uint8_t* p, size_t n) {
    static uint32_t table[256];
    if (!table[1])
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return ~c;
}

static uint32_t csum_add(uint32_t s, const uint8_t* p, size_t n) {
    for (size_t i = 0; i + 1 < n; i += 2) s += get16(p + i);
    if (n & 1) s += uint32_t(p[n - 1]) << 8;
    return s;
}
static uint16_t csum_fold(uint32_t s) {
    while (s >> 16) s = (s & 0xFFFF) + (s >> 16);
    return uint16_t(~s);
}
static uint32_t udp_pseudo(uint32_t src, uint32_t dst, uint32_t len) {
    return (src >> 16) + (src & 0xFFFF) + (dst >> 16) + (dst & 0xFFFF) + 17 + len;
}

static void fpga_mac(uint8_t* m) {
    uint64_t v = top->local_mac;
    for (int i = 0; i < 6; ++i) m[i] = uint8_t(v >> (40 - 8 * i));
}

static std::string ip_str(uint32_t ip) {
    char s[16];
    snprintf(s, sizeof s, "%u.%u.%u.%u", ip >> 24, (ip >> 16) & 255, (ip >> 8) & 255, ip & 255);
    return s;
}

static void bad(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void bad(const char* fmt, ...) {
    ++st.bad;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "eth_cosim: bad frame at cycle %llu: ", (unsigned long long)cycle);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

// Preamble, SFD, padding to 60 bytes, FCS.
static Bytes gmii_frame(Bytes eth) {
    if (eth.size() < 60) eth.resize(60, 0);
    uint32_t fcs = crc32(eth.data(), eth.size());
    Bytes f(7, 0x55);
    f.push_back(0xD5);
    f.insert(f.end(), eth.begin(), eth.end());
    for (int i = 0; i < 4; ++i) f.push_back(uint8_t(fcs >> (8 * i)));
    return f;
}

static Bytes udp_to_fpga(uint16_t src_port, const Bytes& payload) {
    uint8_t mac[6];
    fpga_mac(mac);
    uint32_t dst_ip = top->local_ip, ulen = 8 + payload.size();
    Bytes e;
    put_mac(e, mac);
    put_mac(e, HOST_MAC);
    put16(e, 0x0800);
    size_t ip = e.size();
    put16(e, 0x4500);
    put16(e, 20 + ulen);
    put16(e, ip_id++);
    put16(e, 0x4000);  // DF
    put16(e, 64 << 8 | 17);
    put16(e, 0);
    put32(e, host_ip);
    put32(e, dst_ip);
    uint16_t hc = csum_fold(csum_add(0, &e[ip], 20));
    e[ip + 10] = uint8_t(hc >> 8);
    e[ip + 11] = uint8_t(hc);
    size_t udp = e.size();
    put16(e, src_port);
    put16(e, fpga_port);
    put16(e, ulen);
    put16(e, 0);
    e.insert(e.end(), payload.begin(), payload.end());
    uint16_t uc = csum_fold(csum_add(udp_pseudo(host_ip, dst_ip, ulen), &e[udp], ulen));
    if (uc == 0) uc = 0xFFFF;
    e[udp + 6] = uint8_t(uc >> 8);
    e[udp + 7] = uint8_t(uc);
    return gmii_frame(e);
}

static Bytes arp_reply(const uint8_t* sha, uint32_t spa) {
    Bytes e;
    put_mac(e, sha);
    put_mac(e, HOST_MAC);
    put16(e, 0x0806);
    put16(e, 1);
    put16(e, 0x0800);
    put16(e, 6 << 8 | 4);
    put16(e, 2);
    put_mac(e, HOST_MAC);
    put32(e, host_ip);
    put_mac(e, sha);
    put32(e, spa);
    return gmii_frame(e);
}

// ── GMII ────────────────────────────────────────────────────────────────────
struct RxDriver {  // frames -> gmii_rxd, one byte per GMII cycle, 12 idle bytes between frames
    std::deque<Bytes> q;
    size_t pos = 0;
    int gap = 0;
    void step() {  // while gmii_clk is low: the inputs for the next rising edge
        top->gmii_rx_er = 0;
        if (gap > 0 || q.empty()) {
            if (gap > 0) --gap;
            top->gmii_rx_dv = 0;
            top->gmii_rxd = 0;
            return;
        }
        top->gmii_rx_dv = 1;
        top->gmii_rxd = q.front()[pos];
        if (++pos == q.front().size()) {
            q.pop_front();
            pos = 0;
            gap = 12;
        }
    }
    void push_urgent(Bytes f) { q.insert(q.begin() + (pos ? 1 : 0), std::move(f)); }  // next frame out
    bool busy() const { return !q.empty() || gap > 0; }
} rx;

static void handle_tx_frame(const Bytes& g, bool er);

struct TxMonitor {  // gmii_txd -> frames
    Bytes cur;
    bool er = false;
    void step() {  // after a rising edge
        if (top->gmii_tx_en) {
            cur.push_back(top->gmii_txd);
            er |= top->gmii_tx_er;
        } else if (!cur.empty()) {
            handle_tx_frame(cur, er);
            cur.clear();
            er = false;
        }
    }
    bool busy() const { return !cur.empty(); }
} tx;

// ── Socket and loss injection ───────────────────────────────────────────────
static int sock = -1;
static std::map<uint16_t, sockaddr_in> peers;  // host-side UDP port -> sender address

struct Datagram {
    uint16_t port;  // the host-side UDP port
    Bytes data;
};

struct Lossy {  // drop / duplicate / hold back behind the next datagram, each with probability p
    const char* name;
    std::function<void(const Datagram&)> deliver;
    double p = 0;
    std::mt19937_64 rng;
    bool held = false;
    Datagram hold;
    uint64_t held_at = 0;
    uint64_t drops = 0, dups = 0, reorders = 0;
    void submit(Datagram d) {
        double r = p > 0 ? std::uniform_real_distribution<double>(0, 1)(rng) : 1;
        if (r < p) {
            ++drops;
        } else if (r < 2 * p) {
            ++dups;
            deliver(d);
            deliver(d);
        } else if (r < 3 * p && !held) {
            ++reorders;
            held = true;
            hold = std::move(d);
            held_at = cycle;
            return;
        } else {
            deliver(d);
        }
        release();
    }
    void release() {
        if (held) {
            held = false;
            deliver(hold);
        }
    }
    void timeout() {
        if (held && cycle - held_at > 2000) release();
    }
};

static Lossy to_fpga{"to FPGA", [](const Datagram& d) { rx.q.push_back(udp_to_fpga(d.port, d.data)); }};
static Lossy to_host{"to host", [](const Datagram& d) {
                         auto it = peers.find(d.port);
                         if (it == peers.end()) {
                             fprintf(stderr, "eth_cosim: reply to unknown port %u dropped\n", d.port);
                             return;
                         }
                         sendto(sock, d.data.data(), d.data.size(), 0, (const sockaddr*)&it->second,
                                sizeof it->second);
                         ++st.out;
                         st.out_bytes += d.data.size();
                     }};

static void socket_rx() {
    static uint8_t buf[65536];
    for (;;) {
        sockaddr_in from{};
        socklen_t fl = sizeof from;
        ssize_t n = recvfrom(sock, buf, sizeof buf, MSG_DONTWAIT, (sockaddr*)&from, &fl);
        if (n < 0) return;
        if (n > 1472) {  // would not fit a 1500-byte MTU; the FPGA does not reassemble fragments
            ++st.oversize;
            fprintf(stderr, "eth_cosim: %zd-byte datagram dropped (max 1472)\n", n);
            continue;
        }
        uint16_t port = ntohs(from.sin_port);
        peers[port] = from;
        ++st.in;
        st.in_bytes += n;
        to_fpga.submit({port, Bytes(buf, buf + n)});
    }
}

// ── Frames from the FPGA ────────────────────────────────────────────────────
static void handle_tx_frame(const Bytes& g, bool er) {
    if (er) return bad("gmii_tx_er during the frame");
    if (g.size() < 8 || g[7] != 0xD5 || std::any_of(g.begin(), g.begin() + 7, [](uint8_t b) { return b != 0x55; }))
        return bad("no preamble/SFD");
    const uint8_t* f = g.data() + 8;
    size_t n = g.size() - 8;
    if (n < 64) return bad("%zu-byte frame (min 64)", n);
    uint32_t fcs = f[n - 4] | f[n - 3] << 8 | f[n - 2] << 16 | uint32_t(f[n - 1]) << 24;
    if (crc32(f, n - 4) != fcs) return bad("FCS mismatch");
    n -= 4;
    uint8_t mac[6];
    fpga_mac(mac);
    if (memcmp(f + 6, mac, 6)) return bad("source MAC is not the FPGA's");
    uint32_t type = get16(f + 12);
    const uint8_t* p = f + 14;

    if (type == 0x0806) {
        if (get16(p) != 1 || get16(p + 2) != 0x0800 || get16(p + 4) != 0x0604) return bad("ARP header");
        uint32_t oper = get16(p + 6), spa = get32(p + 14), tpa = get32(p + 24);
        if (oper != 1) return;
        if (tpa == host_ip) {
            ++st.arp;
            rx.push_urgent(arp_reply(p + 8, spa));
        } else {
            fprintf(stderr, "eth_cosim: ARP request for %s (host is %s) not answered\n", ip_str(tpa).c_str(),
                    ip_str(host_ip).c_str());
        }
        return;
    }
    if (type != 0x0800) return bad("ethertype 0x%04x", type);
    if (memcmp(f, HOST_MAC, 6)) return bad("destination MAC is not the host's");
    if (n < 14 + 28) return bad("short IPv4 frame");
    uint32_t total = get16(p + 2);
    if (p[0] != 0x45) return bad("IPv4 version/IHL 0x%02x", p[0]);
    if (total < 28 || 14 + total > n) return bad("IPv4 length %u in a %zu-byte frame", total, n);
    if (csum_fold(csum_add(0, p, 20)) != 0) return bad("IPv4 header checksum");
    if (get16(p + 6) & 0x3FFF) return bad("IPv4 fragment");
    if (p[9] != 17) return bad("IPv4 protocol %u", p[9]);
    uint32_t src = get32(p + 12), dst = get32(p + 16);
    if (src != top->local_ip) return bad("source IP %s, not local_ip", ip_str(src).c_str());
    if (dst != host_ip) return bad("destination IP %s, not the host", ip_str(dst).c_str());
    const uint8_t* u = p + 20;
    uint32_t ulen = get16(u + 4), ucs = get16(u + 6);
    if (ulen != total - 20) return bad("UDP length %u, IPv4 payload %u", ulen, total - 20);
    if (ucs && csum_fold(csum_add(udp_pseudo(src, dst, ulen), u, ulen)) != 0) return bad("UDP checksum");
    if (get16(u) != fpga_port) return bad("UDP source port %u", get16(u));
    to_host.submit({uint16_t(get16(u + 2)), Bytes(u + 8, u + ulen)});
}

// ── Clocks ──────────────────────────────────────────────────────────────────
static const uint64_t CORE_HALF = 10, GMII_HALF = 4;  // ns: 50 MHz, 125 MHz
static uint64_t t_core = 0, t_gmii = 0;

static void check_finish() {
    if (ctx->gotFinish()) {
        printf("X %llu\n", (unsigned long long)cycle);
        fflush(stdout);
        top->final();
        exit(2);
    }
}

// Run to the next rising edge of clk, with every GMII edge before it.
static void core_cycle() {
    for (;;) {
        uint64_t t = std::min(t_core, t_gmii);
        bool core_rise = false, gmii_rise = false;
        if (t_core == t) {
            top->clk = !top->clk;
            core_rise = top->clk;
            t_core += CORE_HALF;
        }
        if (t_gmii == t) {
            top->gmii_clk = !top->gmii_clk;
            gmii_rise = top->gmii_clk;
            t_gmii += GMII_HALF;
            if (!gmii_rise) rx.step();
        }
        ctx->time(t);
        top->eval();
        if (gmii_rise) tx.step();
        check_finish();
        if (core_rise) {
            ++cycle;
            return;
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────
static double paused_s = 0;  // wall time spent paused (idle)
static double wall_s() {
    static auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void print_stats() {
    double w = wall_s(), run = w - paused_s;
    fprintf(stderr,
            "eth_cosim: %llu cycles in %.1f s, %.1f s of it paused (%.0f cycles/s running); datagrams in %llu "
            "(%llu B), out %llu (%llu B); ARP replies %llu; bad frames %llu, oversize %llu\n",
            (unsigned long long)cycle, w, paused_s, run > 0 ? cycle / run : 0.0, (unsigned long long)st.in,
            (unsigned long long)st.in_bytes, (unsigned long long)st.out, (unsigned long long)st.out_bytes,
            (unsigned long long)st.arp, (unsigned long long)st.bad, (unsigned long long)st.oversize);
    for (Lossy* l : {&to_fpga, &to_host})
        if (l->p > 0)
            fprintf(stderr, "eth_cosim: injected %s: %llu dropped, %llu duplicated, %llu reordered\n", l->name,
                    (unsigned long long)l->drops, (unsigned long long)l->dups, (unsigned long long)l->reorders);
}

static bool parse_ip(const char* s, uint32_t& ip) {
    in_addr a;
    if (inet_pton(AF_INET, s, &a) != 1) return false;
    ip = ntohl(a.s_addr);
    return true;
}

// stdin: EOF or "Q" stops, "S" prints statistics. Returns false to stop.
static int stdin_fd = 0;  // -1 when stdin is not open
static bool service_stdin() {
    static std::string buf;
    if (stdin_fd < 0) return true;
    pollfd pf{stdin_fd, POLLIN, 0};
    if (poll(&pf, 1, 0) <= 0) return true;
    if (pf.revents & POLLNVAL) {
        stdin_fd = -1;
        return true;
    }
    char tmp[4096];
    ssize_t n = read(stdin_fd, tmp, sizeof tmp);
    if (n <= 0) return false;
    buf.append(tmp, n);
    size_t nl;
    while ((nl = buf.find('\n')) != std::string::npos) {
        std::string line = buf.substr(0, nl);
        buf.erase(0, nl + 1);
        if (line.empty()) continue;
        if (line[0] == 'Q') return false;
        if (line[0] == 'S') {
            printf("S %llu %.3f %.3f %llu %llu\n", (unsigned long long)cycle, wall_s(), paused_s,
                   (unsigned long long)st.in, (unsigned long long)st.out);
            fflush(stdout);
        }
    }
    return true;
}

int main(int argc, char** argv) {
    ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);

    int port = 21587;
    if (const char* s = getenv("ETH_SIM_PORT")) port = atoi(s);
    for (int i = 1; i < argc; ++i)
        if (argv[i][0] != '+') port = atoi(argv[i]);
    if (const char* s = getenv("ETH_SIM_HOST_IP"))
        if (!parse_ip(s, host_ip)) {
            fprintf(stderr, "eth_cosim: bad ETH_SIM_HOST_IP %s\n", s);
            return 1;
        }
    if (const char* s = getenv("ETH_SIM_FPGA_PORT")) fpga_port = uint16_t(atoi(s));
    double loss = 0;
    if (const char* s = getenv("ETH_SIM_LOSS")) loss = atof(s);
    uint64_t seed = 1;
    if (const char* s = getenv("ETH_SIM_SEED")) seed = strtoull(s, nullptr, 0);
    bool idle_ok = !getenv("ETH_SIM_NOIDLE");
    to_fpga.p = to_host.p = std::min(loss, 1.0 / 3);
    to_fpga.rng.seed(seed);
    to_host.rng.seed(seed ^ 0x9E3779B97F4A7C15ull);

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    int bufsz = 4 << 20;
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &bufsz, sizeof bufsz);
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &bufsz, sizeof bufsz);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(uint16_t(port));
    if (sock < 0 || bind(sock, (sockaddr*)&addr, sizeof addr) < 0) {
        fprintf(stderr, "eth_cosim: cannot bind 127.0.0.1:%d: %s\n", port, strerror(errno));
        return 1;
    }
    socklen_t al = sizeof addr;
    getsockname(sock, (sockaddr*)&addr, &al);
    port = ntohs(addr.sin_port);

    struct sigaction sa {};
    sa.sa_handler = [](int) { stop_req = 1; };
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);

    top = new Vatlas_eth_sim{ctx};
    top->clk = 0;
    top->gmii_clk = 0;
    top->rst = 1;
    top->gmii_rst = 1;
    rx.step();
    for (int i = 0; i < 32; ++i) core_cycle();
    top->rst = 0;
    top->gmii_rst = 0;
    for (int i = 0; i < 256; ++i) core_cycle();
    wall_s();
    printf("PORT %d\nREADY %d\n", port, port);
    fflush(stdout);

    // Pause (block on the socket) while the core is halted and the link has been quiet for IDLE
    // cycles: nothing happens until the host sends something.
    const uint64_t IDLE = 50000;
    uint64_t last_busy = cycle;
    while (!stop_req) {
        core_cycle();
        if (cycle % 16) continue;
        socket_rx();
        to_fpga.timeout();
        to_host.timeout();
        if (rx.busy() || tx.busy() || to_fpga.held || to_host.held || !top->halted) last_busy = cycle;
        if (!service_stdin()) break;
        if (idle_ok && cycle - last_busy > IDLE) {
            pollfd pf[2] = {{sock, POLLIN, 0}, {stdin_fd, POLLIN, 0}};
            double t = wall_s();
            poll(pf, 2, 100);
            paused_s += wall_s() - t;
        }
    }
    printf("Q %llu\n", (unsigned long long)cycle);
    fflush(stdout);
    print_stats();
    top->final();
    delete top;
    delete ctx;
    return 0;
}
