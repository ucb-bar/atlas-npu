// eth_udp_stack self-test. Raw Ethernet frames (preamble, SFD, FCS) go in on GMII rx at 125 MHz;
// the core side runs within 4% of CLK_HZ (period and phase from the seed). Checks: ARP replies,
// the core-side words (prefix + payload), replies on GMII tx (ARP for the peer, Ethernet/IPv4/UDP
// headers, IPv4 header checksum, UDP length and checksum, FCS), filtering and counters, the tx
// length guards, line rate, ARP while the core does not take datagrams or is in reset, GMII
// resets of 1 to 50 cycles (rx_abort, no splice, no stall), and a change of local_ip.
// Build and run: standalone/sim/tb_eth_udp_stack.sh [out_dir] [seed] [clk_hz]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <random>
#include <vector>

#include "Veth_udp_stack.h"
#include "verilated.h"

#ifndef CLK_HZ
#define CLK_HZ 50000000  // the stack's CLK_HZ parameter
#endif

using Bytes = std::vector<uint8_t>;
using Words = std::vector<uint32_t>;

static const uint64_t LOCAL_MAC = 0x020041544C53ull;
static const uint16_t PORT = 0x5453;
static const uint64_t GH = 4000;  // GMII half period in ps (125 MHz)
static uint64_t CH = 10000;       // core half period in ps, CLK_HZ +-4%, set from the seed

static VerilatedContext* ctx;
static Veth_udp_stack* top;
static std::mt19937 rng;
static uint64_t t_ps = 0, tg = 0, tc = 0, gcyc = 0, ccyc = 0;
static int fails = 0;

#define CHECK(c, ...)                                                   \
    do {                                                                \
        if (!(c)) {                                                     \
            printf("FAIL line %d (gmii cycle %llu): ", __LINE__,        \
                   (unsigned long long)gcyc);                           \
            printf(__VA_ARGS__);                                        \
            printf("\n");                                               \
            if (++fails > 20) exit(1);                                  \
        }                                                               \
    } while (0)

static int rnd(int n) { return int(rng() % unsigned(n)); }

// ── Frames ──────────────────────────────────────────────────────────────────
static uint32_t crc32(const uint8_t* p, size_t n) {
    uint32_t c = 0xFFFFFFFF;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1)));
    }
    return ~c;
}

static void put(Bytes& b, uint64_t v, int n) {  // big-endian
    for (int i = n - 1; i >= 0; i--) b.push_back(uint8_t(v >> (8 * i)));
}

static uint64_t get(const Bytes& b, size_t off, int n) {
    uint64_t v = 0;
    for (int i = 0; i < n; i++) v = (v << 8) | b.at(off + i);
    return v;
}

static uint32_t sum16(const Bytes& b, size_t off, size_t n, uint32_t s = 0) {  // ones' complement
    for (size_t i = 0; i < n; i += 2) s += uint32_t(b[off + i] << 8) | (i + 1 < n ? b[off + i + 1] : 0);
    while (s >> 16) s = (s & 0xFFFF) + (s >> 16);
    return s;
}

static Bytes eth(uint64_t dst, uint64_t src, uint16_t type, const Bytes& payload) {
    Bytes b;
    put(b, dst, 6);
    put(b, src, 6);
    put(b, type, 2);
    b.insert(b.end(), payload.begin(), payload.end());
    return b;
}

static Bytes arp(uint16_t oper, uint64_t sha, uint32_t spa, uint64_t tha, uint32_t tpa) {
    Bytes b;
    put(b, 1, 2);
    put(b, 0x0800, 2);
    b.push_back(6);
    b.push_back(4);
    put(b, oper, 2);
    put(b, sha, 6);
    put(b, spa, 4);
    put(b, tha, 6);
    put(b, tpa, 4);
    return b;
}

static Bytes ipv4(uint32_t src, uint32_t dst, uint8_t proto, const Bytes& payload,
                  uint16_t flags_frag = 0x4000, bool bad_sum = false) {
    Bytes b;
    b.push_back(0x45);
    b.push_back(0);
    put(b, 20 + payload.size(), 2);
    put(b, 0x1234, 2);
    put(b, flags_frag, 2);
    b.push_back(64);
    b.push_back(proto);
    put(b, 0, 2);
    put(b, src, 4);
    put(b, dst, 4);
    uint16_t c = uint16_t(~sum16(b, 0, 20));
    if (bad_sum) c ^= 1;
    b[10] = uint8_t(c >> 8);
    b[11] = uint8_t(c);
    b.insert(b.end(), payload.begin(), payload.end());
    return b;
}

static uint32_t pseudo(uint32_t src, uint32_t dst, size_t udp_len) {
    Bytes ph;
    put(ph, src, 4);
    put(ph, dst, 4);
    put(ph, 17, 2);
    put(ph, udp_len, 2);
    return sum16(ph, 0, ph.size());
}

static Bytes udp(uint32_t src, uint32_t dst, uint16_t sport, uint16_t dport, const Bytes& data,
                 int len_adj = 0) {
    Bytes b;
    put(b, sport, 2);
    put(b, dport, 2);
    put(b, 8 + data.size() + len_adj, 2);
    put(b, 0, 2);
    b.insert(b.end(), data.begin(), data.end());
    uint16_t c = uint16_t(~sum16(b, 0, b.size(), pseudo(src, dst, b.size())));
    if (c == 0) c = 0xFFFF;
    b[6] = uint8_t(c >> 8);
    b[7] = uint8_t(c);
    return b;
}

static Bytes pattern(uint32_t tag, size_t n) {
    Bytes b(n);
    for (size_t i = 0; i < n; i++) b[i] = uint8_t(i < 4 ? tag >> (8 * i) : tag * 7 + i * 13 + (i >> 8));
    return b;
}

static Words le_words(const Bytes& b) {
    Words w((b.size() + 3) / 4, 0);
    for (size_t i = 0; i < b.size(); i++) w[i / 4] |= uint32_t(b[i]) << (8 * (i % 4));
    return w;
}

// ── GMII and core-side models ───────────────────────────────────────────────
struct GmiiDriver {  // frames into gmii_rxd
    struct Beat { uint8_t d, dv, er; int tag; };
    std::deque<Beat> q;
    std::vector<uint64_t> end_cyc = std::vector<uint64_t>(64, 0);
    unsigned sent = 0, good = 0;  // frames, and those without a bad FCS or GMII error
    void frame(Bytes f, bool bad_fcs = false, bool er = false, int tag = -1) {
        sent++;
        good += !bad_fcs && !er;
        while (f.size() < 60) f.push_back(0);
        uint32_t c = crc32(f.data(), f.size()) ^ (bad_fcs ? 1u : 0u);
        for (int i = 0; i < 4; i++) f.push_back(uint8_t(c >> (8 * i)));
        for (int i = 0; i < 7; i++) q.push_back({0x55, 1, 0, -1});
        q.push_back({0xD5, 1, 0, -1});
        for (size_t i = 0; i < f.size(); i++)
            q.push_back({f[i], 1, uint8_t(er && i == 20), i + 1 == f.size() ? tag : -1});
        for (int i = 0; i < 12; i++) q.push_back({0, 0, 0, -1});
    }
    void drive() {
        if (q.empty()) {
            top->gmii_rxd = 0;
            top->gmii_rx_dv = 0;
            top->gmii_rx_er = 0;
            return;
        }
        Beat b = q.front();
        q.pop_front();
        top->gmii_rxd = b.d;
        top->gmii_rx_dv = b.dv;
        top->gmii_rx_er = b.er;
        if (b.tag >= 0) end_cyc[b.tag] = gcyc;
    }
} drv;

struct GmiiMonitor {  // frames out of gmii_txd, FCS checked and stripped
    Bytes cur;
    uint64_t start = 0, rst_at = 0;
    int cut = 0;  // frames a GMII reset ended on the wire
    std::deque<Bytes> frames;
    std::deque<uint64_t> starts;
    void sample() {
        CHECK(!top->gmii_tx_er, "gmii_tx_er");
        if (top->gmii_rst) rst_at = gcyc;
        if (top->gmii_tx_en) {
            if (cur.empty()) start = gcyc;
            cur.push_back(top->gmii_txd);
        } else if (!cur.empty()) {
            bool pre = cur.size() >= 8 + 64;
            for (int i = 0; pre && i < 7; i++) pre = cur[i] == 0x55;
            bool ok = pre && cur[7] == 0xD5, fcs_ok = false;
            Bytes f;
            if (ok) {
                f.assign(cur.begin() + 8, cur.end());
                uint32_t fcs = uint32_t(f[f.size() - 4]) | uint32_t(f[f.size() - 3]) << 8 |
                               uint32_t(f[f.size() - 2]) << 16 | uint32_t(f[f.size() - 1]) << 24;
                f.resize(f.size() - 4);
                fcs_ok = crc32(f.data(), f.size()) == fcs;
            }
            if (ok && fcs_ok) {
                frames.push_back(f);
                starts.push_back(start);
            } else if (rst_at >= start && gcyc <= rst_at + 3) {
                cut++;
            } else {
                CHECK(ok, "bad preamble/SFD or short frame (%zu bytes)", cur.size());
                CHECK(fcs_ok, "bad FCS on tx frame");
            }
            cur.clear();
        }
    }
} mon;

struct Dgram {
    Words w;
    bool abort;  // ended by a word with rx_abort, which is not in w
};

struct CoreRx {  // rx_* words, one datagram per rx_last
    Words cur;
    std::deque<Dgram> dgrams;
    int ready_pct = 70;
    bool hold = false, abort_ok = false;
    void pre() {
        CHECK(!top->rx_abort || (top->rx_valid && top->rx_last), "rx_abort without rx_valid and rx_last");
        if (top->rx_valid && top->rx_ready) {
            CHECK(abort_ok || !top->rx_abort, "rx_abort without a GMII reset");
            if (!top->rx_abort) cur.push_back(top->rx_data);
            if (top->rx_last) {
                dgrams.push_back({cur, bool(top->rx_abort)});
                cur.clear();
            }
        }
    }
    void post() {
        if (top->rst) cur.clear();  // the core (eth_tsi) forgets a partial datagram
        top->rx_ready = !hold && !top->rst && rnd(100) < ready_pct;
    }
} core_rx;

struct CoreTx {  // tx_* words with random gaps
    std::deque<std::pair<uint32_t, bool>> q;
    int gap_pct = 30;
    bool valid = false;
    void send(const Words& w) {
        for (size_t i = 0; i < w.size(); i++) q.push_back({w[i], i + 1 == w.size()});
    }
    void pre() {
        if (valid && top->tx_ready) {
            q.pop_front();
            valid = false;
        }
    }
    void post() {  // a presented word stays until taken
        if (!valid) valid = !q.empty() && !top->rst && rnd(100) >= gap_pct;
        top->tx_valid = valid;
        top->tx_data = valid ? q.front().first : 0;
        top->tx_last = valid && q.front().second;
    }
} core_tx;

// ── Clocking ────────────────────────────────────────────────────────────────
static void step() {
    uint64_t tn = tg < tc ? tg : tc;
    bool g = tg == tn, c = tc == tn;
    bool g_rise = g && !top->gmii_clk, c_rise = c && !top->clk;
    if (g_rise) mon.sample();
    if (c_rise) {
        core_rx.pre();
        core_tx.pre();
    }
    ctx->timeInc(tn - t_ps);
    t_ps = tn;
    if (g) {
        top->gmii_clk = !top->gmii_clk;
        tg += GH;
    }
    if (c) {
        top->clk = !top->clk;
        tc += CH;
    }
    top->eval();
    if (g_rise) {
        gcyc++;
        drv.drive();
    }
    if (c_rise) {
        ccyc++;
        core_rx.post();
        core_tx.post();
    }
    if (g_rise || c_rise) top->eval();
}

static void run(uint64_t n) {  // n GMII cycles
    uint64_t end = gcyc + n;
    while (gcyc < end) step();
}

template <class F>
static bool wait_for(F f, uint64_t max_g) {
    uint64_t end = gcyc + max_g;
    while (!f()) {
        if (gcyc >= end) return false;
        step();
    }
    return true;
}

static Bytes next_tx(uint64_t max_g = 20000) {
    if (!wait_for([] { return !mon.frames.empty(); }, max_g)) {
        CHECK(false, "no tx frame within %llu cycles", (unsigned long long)max_g);
        return {};
    }
    Bytes f = mon.frames.front();
    mon.frames.pop_front();
    mon.starts.pop_front();
    return f;
}

static Words next_rx(uint64_t max_g = 20000) {
    if (!wait_for([] { return !core_rx.dgrams.empty(); }, max_g)) {
        CHECK(false, "no datagram on the core side within %llu cycles", (unsigned long long)max_g);
        return {};
    }
    Words w = core_rx.dgrams.front().w;
    core_rx.dgrams.pop_front();
    return w;
}

static void quiet(uint64_t n, const char* what) {  // nothing may come out on either side
    run(n);
    CHECK(mon.frames.empty(), "%s: unexpected tx frame", what);
    CHECK(core_rx.dgrams.empty(), "%s: unexpected datagram on the core side", what);
    mon.frames.clear();
    mon.starts.clear();
    core_rx.dgrams.clear();
}

// ── Peers and checks ────────────────────────────────────────────────────────
static uint32_t local_ip = 0xAC140112;                                // 172.20.1.18
static const uint64_t MAC_A = 0x02000000000Aull, MAC_B = 0x02000000000Bull;
static const uint32_t IP_A = 0xAC140001, IP_B = 0x0A090807;           // 172.20.0.1, 10.9.8.7

static void send_udp(uint64_t smac, uint32_t sip, uint16_t sport, uint32_t dip, uint16_t dport,
                     const Bytes& data, int tag = -1) {
    drv.frame(eth(LOCAL_MAC, smac, 0x0800, ipv4(sip, dip, 17, udp(sip, dip, sport, dport, data))),
              false, false, tag);
}

static void send_arp_request(uint64_t smac, uint32_t sip, uint32_t tip, int tag = -1) {
    drv.frame(eth(0xFFFFFFFFFFFFull, smac, 0x0806, arp(1, smac, sip, 0, tip)), false, false, tag);
}

static void check_arp(const Bytes& f, uint64_t dmac, uint16_t oper, uint64_t tha, uint32_t tpa) {
    CHECK(f.size() == 60, "ARP frame length %zu", f.size());
    if (f.size() < 42) return;
    CHECK(get(f, 0, 6) == dmac, "ARP dest MAC %012llx", (unsigned long long)get(f, 0, 6));
    CHECK(get(f, 6, 6) == LOCAL_MAC, "ARP src MAC");
    CHECK(get(f, 12, 2) == 0x0806, "ARP ethertype %04llx", (unsigned long long)get(f, 12, 2));
    CHECK(get(f, 14, 2) == 1 && get(f, 16, 2) == 0x0800 && f[18] == 6 && f[19] == 4, "ARP htype/ptype/len");
    CHECK(get(f, 20, 2) == oper, "ARP oper %llu", (unsigned long long)get(f, 20, 2));
    CHECK(get(f, 22, 6) == LOCAL_MAC && get(f, 28, 4) == local_ip, "ARP sender");
    CHECK(get(f, 32, 6) == tha && get(f, 38, 4) == tpa, "ARP target %012llx %08llx",
          (unsigned long long)get(f, 32, 6), (unsigned long long)get(f, 38, 4));
}

static void check_udp(const Bytes& f, uint64_t dmac, uint32_t dip, uint16_t dport, const Bytes& data) {
    size_t n = 14 + 20 + 8 + data.size();
    CHECK(f.size() == (n < 60 ? 60 : n), "UDP frame length %zu, want %zu", f.size(), n);
    if (f.size() < n) return;
    CHECK(get(f, 0, 6) == dmac, "dest MAC %012llx", (unsigned long long)get(f, 0, 6));
    CHECK(get(f, 6, 6) == LOCAL_MAC && get(f, 12, 2) == 0x0800, "src MAC / ethertype");
    CHECK(f[14] == 0x45 && f[15] == 0, "IP version/IHL/DSCP %02x %02x", f[14], f[15]);
    CHECK(get(f, 16, 2) == 28 + data.size(), "IP total length %llu", (unsigned long long)get(f, 16, 2));
    CHECK((get(f, 20, 2) & 0x3FFF) == 0, "IP fragment fields %04llx", (unsigned long long)get(f, 20, 2));
    CHECK(f[22] == 64 && f[23] == 17, "TTL %u protocol %u", f[22], f[23]);
    CHECK(sum16(f, 14, 20) == 0xFFFF, "IP header checksum");
    CHECK(get(f, 26, 4) == local_ip && get(f, 30, 4) == dip, "IP addresses %08llx -> %08llx",
          (unsigned long long)get(f, 26, 4), (unsigned long long)get(f, 30, 4));
    CHECK(get(f, 34, 2) == PORT && get(f, 36, 2) == dport, "UDP ports %llu -> %llu",
          (unsigned long long)get(f, 34, 2), (unsigned long long)get(f, 36, 2));
    CHECK(get(f, 38, 2) == 8 + data.size(), "UDP length %llu", (unsigned long long)get(f, 38, 2));
    // A computed checksum of 0 goes out as 0 ("none"; udp_checksum_gen does not send 0xFFFF).
    CHECK(sum16(f, 34, 8 + data.size(), pseudo(local_ip, dip, 8 + data.size())) == 0xFFFF,
          "UDP checksum %04llx", (unsigned long long)get(f, 40, 2));
    CHECK(Bytes(f.begin() + 42, f.begin() + n) == data, "UDP payload differs");
}

// A datagram on the core side: prefix (peer IP, {length, peer port}), then the payload.
static Words dgram_words(uint32_t sip, uint16_t sport, const Bytes& data) {
    Words w = {sip, uint32_t(data.size()) << 16 | sport};
    Words p = le_words(data);
    w.insert(w.end(), p.begin(), p.end());
    return w;
}

static void check_dgram(const Words& w, uint32_t sip, uint16_t sport, const Bytes& data) {
    Words want = dgram_words(sip, sport, data);
    CHECK(w.size() == want.size(), "datagram %zu words, want %zu", w.size(), want.size());
    if (w.size() != want.size()) return;
    CHECK(w[0] == want[0] && w[1] == want[1], "prefix %08x %08x, want %08x %08x", w[0], w[1], want[0], want[1]);
    size_t full = data.size() / 4;
    for (size_t i = 0; i < full; i++) CHECK(w[2 + i] == want[2 + i], "payload word %zu", i);
    if (data.size() % 4) {  // bytes past the end of the payload are don't-care
        uint32_t m = (1u << (8 * (data.size() % 4))) - 1;
        CHECK((w[2 + full] & m) == (want[2 + full] & m), "last partial word");
    }
}

static Words reply_words(uint32_t ip, uint16_t port, const Bytes& data) {
    Words w = {ip, port};
    Words p = le_words(data);
    w.insert(w.end(), p.begin(), p.end());
    return w;
}

static void settle() { run(200); }  // let local_ip and the counters cross

// rx_frames counts the frames at the GMII input ([15:0]) and out of the MAC ([31:16]). The MAC
// passes those without a bad FCS or GMII error: its FIFO never fills, as the stack drains it.
struct FrameCount { uint32_t rx; unsigned sent, good; };

static FrameCount frame_count() {  // once the frames queued so far are through
    wait_for([] { return drv.q.empty(); }, 1000000);
    run(5000);
    return {top->rx_frames, drv.sent, drv.good};
}

static void check_frames(const FrameCount& c, const char* what) {
    auto at = [&](int sh) { return uint16_t((top->rx_frames >> sh) - (c.rx >> sh)); };
    uint16_t g = uint16_t(drv.sent - c.sent), m = uint16_t(drv.good - c.good);
    wait_for([&] { return drv.q.empty() && at(0) == g && at(16) == m; }, 200000);
    run(3000);
    CHECK(at(0) == g && at(16) == m, "%s: rx_frames GMII +%u MAC +%u, want +%u +%u", what, at(0),
          at(16), g, m);
}

// ── Tests ───────────────────────────────────────────────────────────────────
static void test_arp() {
    send_arp_request(MAC_A, IP_A, local_ip);
    check_arp(next_tx(), MAC_A, 2, MAC_A, IP_A);
    send_arp_request(MAC_A, IP_A, local_ip + 1);
    quiet(3000, "ARP for another IP");
    printf("arp: ok\n");
}

static void test_echo_a() {
    Bytes d = pattern(1, 24);
    send_udp(MAC_A, IP_A, 40000, local_ip, PORT, d);
    check_dgram(next_rx(), IP_A, 40000, d);
    Bytes r = pattern(2, 20);
    core_tx.send(reply_words(IP_A, 40000, r));
    check_udp(next_tx(), MAC_A, IP_A, 40000, r);  // MAC learned from A's ARP request
    printf("echo A: ok\n");
}

static void test_arp_for_peer() {
    // B is on another subnet and never announced itself: the reply must ARP for B directly.
    Bytes d = pattern(3, 8);
    send_udp(MAC_B, IP_B, 5555, local_ip, PORT, d);
    check_dgram(next_rx(), IP_B, 5555, d);
    Bytes r = pattern(4, 36);
    core_tx.send(reply_words(IP_B, 5555, r));
    check_arp(next_tx(), 0xFFFFFFFFFFFFull, 1, 0, IP_B);
    drv.frame(eth(LOCAL_MAC, MAC_B, 0x0806, arp(2, MAC_B, IP_B, LOCAL_MAC, local_ip)));
    check_udp(next_tx(), MAC_B, IP_B, 5555, r);
    printf("ARP for peer: ok\n");
}

static void test_filter() {
    FrameCount fc = frame_count();
    unsigned drop0 = top->rx_dropped, bad0 = top->rx_bad;
    auto expect = [&](unsigned drop, unsigned bad, const char* what) {
        quiet(3000, what);
        CHECK(unsigned(top->rx_dropped - drop0) == drop && unsigned(top->rx_bad - bad0) == bad,
              "%s: rx_dropped +%u rx_bad +%u, want +%u +%u", what, unsigned(top->rx_dropped - drop0),
              unsigned(top->rx_bad - bad0), drop, bad);
        check_frames(fc, what);
    };
    Bytes d = pattern(5, 16);
    auto to_us = [&](const Bytes& u, uint16_t flags_frag = 0x4000, bool bad_sum = false) {
        return eth(LOCAL_MAC, MAC_A, 0x0800, ipv4(IP_A, local_ip, 17, u, flags_frag, bad_sum));
    };
    send_udp(MAC_A, IP_A, 40000, local_ip, 1234, d);
    expect(1, 0, "other port");
    send_udp(MAC_A, IP_A, 40000, local_ip + 7, PORT, d);
    expect(1, 0, "other IP");
    drv.frame(eth(LOCAL_MAC, MAC_A, 0x0800, ipv4(IP_A, local_ip, 1, Bytes{8, 0, 0xF7, 0xFF, 0, 0, 0, 0})));
    expect(1, 0, "ICMP echo");
    drv.frame(to_us(udp(IP_A, local_ip, 40000, PORT, d), 0x2000));
    expect(2, 0, "IP fragment (MF)");
    drv.frame(to_us(udp(IP_A, local_ip, 40000, PORT, d), 0x4000, true));
    expect(3, 0, "bad IP header checksum");
    drv.frame(to_us(udp(IP_A, local_ip, 40000, PORT, {})));
    expect(4, 0, "empty UDP payload");
    drv.frame(to_us(udp(IP_A, local_ip, 40000, PORT, d, 4)));
    expect(5, 0, "UDP length past the end of the packet");
    drv.frame(to_us(udp(IP_A, local_ip, 40000, PORT, d)), true);
    expect(6, 1, "bad FCS");
    drv.frame(to_us(udp(IP_A, local_ip, 40000, PORT, d)), false, true);
    expect(7, 2, "GMII rx_er");
    drv.frame(eth(LOCAL_MAC, MAC_A, 0x86DD, Bytes(80, 0x11)));
    expect(7, 2, "IPv6");
    top->gmii_clk_en = 0;  // the MAC does not see the frame; rx_frames [15:0] counts it
    send_udp(MAC_A, IP_A, 40000, local_ip, PORT, d);
    drv.good--;
    expect(7, 2, "gmii_clk_en low");
    top->gmii_clk_en = 1;
    send_udp(MAC_A, IP_A, 40000, local_ip, PORT, d);
    check_dgram(next_rx(), IP_A, 40000, d);
    check_frames(fc, "datagram");
    printf("filter and counters: ok\n");
}

static void test_sizes(int n) {
    // Random payload sizes (any byte count, up to 1472) in, random reply sizes out, both checked.
    for (int i = 0; i < n; i++) {
        size_t len = i == 0 ? 1472 : i == 1 ? 1 : size_t(1 + rnd(1472));
        Bytes d = pattern(100 + i, len);
        uint16_t sport = uint16_t(1024 + rnd(60000));
        send_udp(MAC_A, IP_A, sport, local_ip, PORT, d);
        check_dgram(next_rx(), IP_A, sport, d);
        size_t rlen = 4 * size_t(i == 0 ? 368 : 1 + rnd(368));
        Bytes r = pattern(200 + i, rlen);
        core_tx.send(reply_words(IP_A, sport, r));
        check_udp(next_tx(), MAC_A, IP_A, sport, r);
    }
    printf("sizes (%d datagrams and replies): ok\n", n);
}

static void test_tx_guards() {
    core_tx.send(reply_words(IP_A, 40000, pattern(6, 1476)));  // one word over the limit
    quiet(30000, "over-long reply");
    core_tx.send(Words{IP_A, 40000});                          // prefix only
    core_tx.send(Words{IP_A});                                 // cut prefix
    quiet(5000, "reply without payload");
    Bytes r = pattern(7, 12);
    core_tx.send(reply_words(IP_A, 40001, r));
    check_udp(next_tx(), MAC_A, IP_A, 40001, r);
    printf("tx guards: ok\n");
}

// The core takes nothing: datagrams fill the rx FIFO and the rest are dropped, while ARP requests
// in between are still answered at once. Then the core drains what was kept, in order and intact.
static void test_backpressure() {
    settle();
    unsigned drop0 = top->rx_dropped;
    core_rx.hold = true;
    const int N = 40;
    for (int i = 0; i < N; i++) {
        send_udp(MAC_A, IP_A, 7000, local_ip, PORT, pattern(1000 + i, 1472));
        if (i == 25) send_arp_request(MAC_A, IP_A, local_ip, 1);
    }
    send_arp_request(MAC_A, IP_A, local_ip, 2);
    int arps = 0;
    uint64_t worst = 0;
    wait_for([&] {
        while (!mon.frames.empty()) {
            Bytes f = mon.frames.front();
            uint64_t s = mon.starts.front();
            mon.frames.pop_front();
            mon.starts.pop_front();
            check_arp(f, MAC_A, 2, MAC_A, IP_A);
            uint64_t lat = s - drv.end_cyc[1 + arps];
            worst = lat > worst ? lat : worst;
            arps++;
        }
        return arps == 2;
    }, 200000);
    CHECK(arps == 2, "ARP replies while the core stalls: %d of 2", arps);
    // The MAC FIFO stores whole frames, so an ARP request waits for the datagram ahead of it
    // (1538 cycles at most), but no longer: the backlog in the rx FIFO does not delay it.
    CHECK(worst < 2000, "ARP reply latency %llu cycles while the core stalls", (unsigned long long)worst);
    run(2000);
    unsigned dropped = unsigned(top->rx_dropped - drop0);
    CHECK(core_rx.dgrams.empty(), "datagram delivered while rx_ready is low");
    core_rx.hold = false;
    wait_for([&] { return core_rx.dgrams.size() + dropped >= size_t(N); }, 400000);
    run(2000);
    size_t got = core_rx.dgrams.size();
    CHECK(got + dropped == N && got >= 20, "kept %zu, dropped %u, sent %d", got, dropped, N);
    uint32_t prev = 0;
    for (size_t k = 0; k < got; k++) {
        Words w = core_rx.dgrams.front().w;
        core_rx.dgrams.pop_front();
        uint32_t tag = w.size() > 2 ? w[2] : 0;
        CHECK(tag >= 1000 && tag < 1000 + N && (k == 0 || tag > prev), "order: tag %u after %u", tag, prev);
        check_dgram(w, IP_A, 7000, pattern(tag, 1472));
        prev = tag;
    }
    printf("backpressure: ok (kept %zu, dropped %u, ARP latency <= %llu cycles)\n", got, dropped,
           (unsigned long long)worst);
}

// The core goes into reset with datagrams in flight: ARP still works, everything the core gets is
// whole, and datagrams after the reset all arrive.
static void test_core_reset() {
    for (int i = 0; i < 6; i++) send_udp(MAC_A, IP_A, 7001, local_ip, PORT, pattern(2000 + i, 200 + 50 * i));
    run(1500);
    top->rst = 1;
    for (int i = 6; i < 12; i++) send_udp(MAC_A, IP_A, 7001, local_ip, PORT, pattern(2000 + i, 300));
    send_arp_request(MAC_A, IP_A, local_ip);
    check_arp(next_tx(), MAC_A, 2, MAC_A, IP_A);
    run(5000);
    top->rst = 0;
    run(2000);
    for (int i = 12; i < 18; i++) send_udp(MAC_A, IP_A, 7001, local_ip, PORT, pattern(2000 + i, 64 + 4 * i));
    run(20000);
    int late = 0;
    while (!core_rx.dgrams.empty()) {
        Words w = core_rx.dgrams.front().w;
        core_rx.dgrams.pop_front();
        uint32_t tag = w.size() > 2 ? w[2] : 0;
        CHECK(tag >= 2000 && tag < 2018, "tag %u", tag);
        if (tag < 2000 || tag >= 2018) continue;
        size_t len = tag < 2006 ? 200 + 50 * (tag - 2000) : tag < 2012 ? 300 : 64 + 4 * (tag - 2000);
        check_dgram(w, IP_A, 7001, pattern(tag, len));
        late += tag >= 2012;
    }
    CHECK(late == 6, "%d of 6 datagrams after the core reset", late);
    Bytes r = pattern(8, 40);
    core_tx.send(reply_words(IP_A, 7001, r));
    check_udp(next_tx(), MAC_A, IP_A, 7001, r);
    printf("core reset: ok\n");
}

// GMII resets of 1 to 50 cycles at random points while datagrams cross to the core (slow or fast)
// and replies cross to GMII. Every datagram the core gets is whole, or the start of one ended by
// a word with rx_abort (never spliced, nothing old again); rx_abort comes only there. GMII tx
// carries only whole replies the core sent, in order, and ARP for the peer (answered here). A
// datagram and a reply sent after each reset get through.
static void test_gmii_resets(int trials) {
    std::map<uint32_t, size_t> sent;  // datagram tag -> payload bytes
    std::map<uint32_t, Bytes> replies;  // reply tag -> payload
    uint32_t tag = 0x30000, rtag = 0x50000, last = 0, rlast = 0, want_rx = 0, want_tx = 0;
    bool got_rx = false, got_tx = false;
    int whole = 0, cut = 0, out = 0;
    auto new_dgram = [&](size_t len) {
        sent[tag] = len;
        send_udp(MAC_A, IP_A, 7004, local_ip, PORT, pattern(tag, len));
        return tag++;
    };
    auto new_reply = [&](size_t len) {
        replies[rtag] = pattern(rtag, len);
        core_tx.send(reply_words(IP_A, 7104, replies[rtag]));
        return rtag++;
    };
    auto drain = [&] {
        while (!mon.frames.empty()) {
            Bytes f = mon.frames.front();
            mon.frames.pop_front();
            mon.starts.pop_front();
            if (f.size() >= 42 && get(f, 12, 2) == 0x0806 && get(f, 20, 2) == 1) {
                check_arp(f, 0xFFFFFFFFFFFFull, 1, 0, IP_A);
                drv.frame(eth(LOCAL_MAC, MAC_A, 0x0806, arp(2, MAC_A, IP_A, LOCAL_MAC, local_ip)));
            } else if (f.size() >= 42 && get(f, 12, 2) == 0x0806) {
                check_arp(f, MAC_A, 2, MAC_A, IP_A);
            } else {
                uint32_t t = f.size() >= 46 ? le_words(Bytes(f.begin() + 42, f.begin() + 46))[0] : 0;
                auto it = replies.find(t);
                CHECK(it != replies.end() && t > rlast, "tx frame (%zu bytes) is not a new reply: tag %x after %x",
                      f.size(), t, rlast);
                if (it == replies.end()) continue;
                check_udp(f, MAC_A, IP_A, 7104, it->second);
                rlast = t;
                got_tx |= t == want_tx;
                out++;
            }
        }
        while (!core_rx.dgrams.empty()) {
            Dgram d = core_rx.dgrams.front();
            core_rx.dgrams.pop_front();
            const Words& w = d.w;
            if (d.abort && w.size() < 3) {  // cut inside the prefix
                CHECK(w.empty() || (w[0] == IP_A && (w.size() < 2 || (w[1] & 0xFFFF) == 7004)),
                      "cut datagram: prefix %08x %08x", w.size() > 0 ? w[0] : 0, w.size() > 1 ? w[1] : 0);
                cut++;
                continue;
            }
            uint32_t t = w.size() > 2 ? w[2] : 0;
            auto it = sent.find(t);
            CHECK(it != sent.end() && t > last, "%s datagram (%zu words) is not a new one: tag %u after %u",
                  d.abort ? "cut" : "whole", w.size(), t, last);
            if (it == sent.end()) continue;
            last = t;
            if (!d.abort) {
                check_dgram(w, IP_A, 7004, pattern(t, it->second));
                got_rx |= t == want_rx;
                whole++;
                continue;
            }
            Words want = dgram_words(IP_A, 7004, pattern(t, it->second));
            CHECK(w.size() < want.size(), "datagram %u: rx_abort after all %zu words", t, w.size());
            for (size_t i = 0; i < w.size() && i < want.size(); i++)
                CHECK(w[i] == want[i], "cut datagram %u word %zu: %08x, want %08x", t, i, w[i], want[i]);
            cut++;
        }
    };
    static const int ready[] = {2, 10, 30, 70, 100};
    core_rx.abort_ok = true;
    for (int k = 0; k < trials; k++) {
        core_rx.ready_pct = ready[rnd(5)];
        for (int i = rnd(4); i > 0; i--) new_dgram(rnd(4) ? size_t(4 + rnd(400)) : 1472);
        for (int i = rnd(3); i > 0; i--) new_reply(4 * size_t(1 + rnd(120)));
        run(rnd(3000));
        drain();
        int pulse = 1 + rnd(50);
        top->gmii_rst = 1;
        run(pulse);
        top->gmii_rst = 0;
        core_rx.ready_pct = 70;
        run(1000);  // arp_cache clears its 512 entries after reset and ignores writes meanwhile
        drain();
        send_arp_request(MAC_A, IP_A, local_ip);
        want_rx = new_dgram(4 + rnd(200));
        want_tx = new_reply(4 * size_t(1 + rnd(30)));
        got_rx = got_tx = false;
        bool ok = wait_for([&] {
            drain();
            return got_rx && got_tx;
        }, 100000);
        CHECK(ok, "reset %d (%d cycles): the datagram after it %s, the reply after it %s (tx_ready %d)", k, pulse,
              got_rx ? "arrived" : "did not arrive", got_tx ? "went out" : "did not go out", top->tx_ready);
        if (!ok) break;
    }
    core_rx.abort_ok = false;
    printf("GMII resets: ok (%d resets of 1-50 cycles; datagrams: %d whole, %d cut with rx_abort, %zu lost; "
           "%d of %zu replies out, %d cut on the wire)\n", trials, whole, cut,
           sent.size() - whole - cut, out, replies.size(), mon.cut);
}

// Back-to-back full-size datagrams at line rate, the core taking 70% of cycles: none dropped.
static void test_line_rate() {
    double core_mbs = 0.7 * 4 * 5e5 / double(CH);  // what the core takes, MB/s
    if (core_mbs < 125 * 1472 / 1538.0) {
        printf("line rate: skipped (the core takes %.0f MB/s)\n", core_mbs);
        return;
    }
    settle();
    unsigned drop0 = top->rx_dropped;
    const int N = 100;
    for (int i = 0; i < N; i++) send_udp(MAC_A, IP_A, 7003, local_ip, PORT, pattern(4000 + i, 1472));
    uint64_t t0 = gcyc;
    wait_for([] { return core_rx.dgrams.size() == N; }, 400000);
    uint64_t t = gcyc - t0;
    for (int i = 0; !core_rx.dgrams.empty(); i++) {
        check_dgram(core_rx.dgrams.front().w, IP_A, 7003, pattern(4000 + i, 1472));
        core_rx.dgrams.pop_front();
    }
    settle();
    CHECK(top->rx_dropped == drop0, "%u dropped at line rate", unsigned(top->rx_dropped - drop0));
    printf("line rate: ok (%d x 1472 bytes in %llu cycles = %.1f MB/s payload)\n", N,
           (unsigned long long)t, N * 1472.0 / (t * 8e-9) / 1e6);
}

static void test_ip_change() {
    uint32_t old = local_ip;
    local_ip = 0xAC140132;  // 172.20.1.50
    top->local_ip = local_ip;
    settle();
    send_arp_request(MAC_A, IP_A, old);
    quiet(3000, "ARP for the old IP");
    send_arp_request(MAC_A, IP_A, local_ip);
    check_arp(next_tx(), MAC_A, 2, MAC_A, IP_A);
    send_udp(MAC_A, IP_A, 40000, old, PORT, pattern(11, 16));
    quiet(3000, "datagram to the old IP");
    Bytes d = pattern(12, 16);
    send_udp(MAC_A, IP_A, 40000, local_ip, PORT, d);
    check_dgram(next_rx(), IP_A, 40000, d);
    Bytes r = pattern(13, 16);
    core_tx.send(reply_words(IP_A, 40000, r));
    check_udp(next_tx(), MAC_A, IP_A, 40000, r);
    printf("local_ip change: ok\n");
}

int main(int argc, char** argv) {
    ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);
    unsigned seed = 1;
    for (int i = 1; i < argc; i++)
        if (argv[i][0] != '+') seed = unsigned(strtoul(argv[i], nullptr, 0));
    rng.seed(seed);
    CH = uint64_t(5e11 / CLK_HZ * (9615 + rnd(800)) / 10000);
    tc = uint64_t(rnd(int(CH)));
    top = new Veth_udp_stack{ctx};
    top->gmii_clk_en = 1;
    top->gmii_rst = 1;
    top->rst = 1;
    top->local_ip = local_ip;
    run(50);
    top->gmii_rst = 0;
    top->rst = 0;
    settle();

    FrameCount fc = frame_count();
    test_arp();
    test_echo_a();
    test_arp_for_peer();
    test_filter();
    test_sizes(60);
    test_tx_guards();
    test_line_rate();
    test_backpressure();
    test_core_reset();
    check_frames(fc, "all tests before the GMII resets");
    test_gmii_resets(200);
    fc = frame_count();
    test_ip_change();
    check_frames(fc, "local_ip change");
    printf("rx_frames: ok\n");

    top->final();
    printf("%s: %d failures, seed %u, core clock %.2f MHz, %llu GMII cycles, %llu core cycles\n",
           fails ? "FAIL" : "PASS", fails, seed, 1e6 / (2.0 * CH), (unsigned long long)gcyc,
           (unsigned long long)ccyc);
    delete top;
    delete ctx;
    return fails ? 1 : 0;
}
