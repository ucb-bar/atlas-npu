// Verilator check of Atlas's fp4_fedp (src/main/resources/vsrc/nvfp4) against NVIDIA silicon.
//
//   build/fp4_fedp_vec/fp4_fedp_vec FILE.vec[.gz] ...      (built and run by fp4_fedp_vec.sh)
//
// Each file is an nvfp4-vectors-v1 set (README.md here), plain or gzip, kind 0 (NVFP4), so mode
// is 0 as in Nvfp4Mxu. One vector enters per cycle, and every d must leave exactly 5 cycles after
// its in_vld. d is compared with each silicon column (rtx5090, thor); any NaN equals any NaN.
// Disagreements on a vector of the known gap (an all-zero 16-element block sets the alignment, as
// in check_nvfp4_silicon.py) are listed as such; any other disagreement, or a latency other than
// 5, makes the exit status 1.
#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Vfp4_fedp.h"
#include "verilated.h"

static const char* SILICON[] = {"rtx5090", "thor"};
static const int LATENCY = 5;

struct VecFile {
    std::string name;
    int kind = -1;
    size_t count = 0;
    std::vector<std::string> columns;
    std::vector<uint64_t> indices;  // index in the full set (subsets), else empty
    std::string body;               // count 76-byte records, then one uint32 column per name
};

static std::string read_file(const char* path) {
    gzFile f = gzopen(path, "rb");  // reads plain files too
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    std::string s;
    char buf[1 << 16];
    int n;
    while ((n = gzread(f, buf, sizeof buf)) > 0) s.append(buf, n);
    gzclose(f);
    return s;
}

// Minimal reader for the flat keys of the JSON header line.
static size_t key(const std::string& h, const char* k) {
    size_t p = h.find(std::string("\"") + k + "\":");
    return p == std::string::npos ? p : p + strlen(k) + 3;
}

static VecFile load(const char* path) {
    VecFile v;
    std::string raw = read_file(path);
    size_t nl = raw.find('\n');
    std::string h = raw.substr(0, nl);
    size_t p;
    if ((p = key(h, "name")) != std::string::npos) {
        size_t q = h.find('"', p) + 1;
        v.name = h.substr(q, h.find('"', q) - q);
    }
    if ((p = key(h, "kind")) != std::string::npos) v.kind = atoi(h.c_str() + p);
    if ((p = key(h, "count")) != std::string::npos) v.count = strtoull(h.c_str() + p, nullptr, 10);
    if ((p = key(h, "columns")) != std::string::npos) {
        size_t end = h.find(']', p);
        for (size_t q = h.find('"', p); q < end; q = h.find('"', q + 1)) {
            size_t e = h.find('"', q + 1);
            v.columns.push_back(h.substr(q + 1, e - q - 1));
            q = e;
        }
    }
    if ((p = key(h, "indices")) != std::string::npos) {
        const char* s = h.c_str() + h.find('[', p) + 1;
        char* e;
        for (;;) {
            uint64_t x = strtoull(s, &e, 10);
            if (e == s) break;
            v.indices.push_back(x);
            s = e;
            while (*s == ',' || *s == ' ') ++s;
        }
    }
    v.body = raw.substr(nl + 1);
    if (v.body.size() < v.count * (76 + 4 * v.columns.size())) {
        fprintf(stderr, "%s: truncated\n", path);
        exit(2);
    }
    return v;
}

static uint32_t le32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24; }
static bool is_nan(uint32_t x) { return (x >> 23 & 255) == 255 && (x & 0x7FFFFF); }
static bool same(uint32_t x, uint32_t y) { return x == y || (is_nan(x) && is_nan(y)); }

// UE4M3 exponent after normalization; see check_nvfp4_silicon.zero_block_sets_alignment.
static int ue4m3_exp(uint8_t x) {
    x &= 127;
    return x >> 3 ? (x >> 3) - 7 : (x & 7 ? -6 : -126);
}
static bool zero_block_sets_alignment(const uint8_t* r) {
    const uint8_t *a = r, *b = r + 32, *sa = r + 64, *sb = r + 68;
    uint32_t c = le32(r + 72);
    for (int g = 0; g < 4; ++g)
        if ((sa[g] & 127) == 127 || (sb[g] & 127) == 127) return false;
    if ((c >> 23 & 255) == 255) return false;
    bool zero[4];
    int nz = 0, top = (c >> 23 & 255 ? int(c >> 23 & 255) : 1) - 127;
    for (int g = 0; g < 4; ++g) {
        zero[g] = true;
        for (int k = 16 * g; k < 16 * g + 16; ++k)
            if ((a[k / 2] >> 4 * (k % 2) & 7) && (b[k / 2] >> 4 * (k % 2) & 7)) zero[g] = false;
        nz += zero[g];
    }
    if (nz == 0 || nz == 4) return false;
    for (int g = 0; g < 4; ++g)
        if (!zero[g]) top = std::max(top, ue4m3_exp(sa[g]) + ue4m3_exp(sb[g]));
    for (int g = 0; g < 4; ++g)
        if (zero[g] && ue4m3_exp(sa[g]) + ue4m3_exp(sb[g]) > top) return true;
    return false;
}
static const char* gap_class(const uint8_t* r) {
    return zero_block_sets_alignment(r) ? "all-zero 16-element block sets the alignment" : nullptr;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vfp4_fedp top;
    auto tick = [&] {
        top.clk = 0;
        top.eval();
        top.clk = 1;
        top.eval();
    };
    top.in_vld = 0;
    top.mode = 0;  // NVFP4, as Nvfp4Mxu ties it
    top.rst = 1;
    for (int i = 0; i < 8; ++i) tick();
    top.rst = 0;
    tick();

    size_t comparisons = 0, n_known = 0, n_other = 0, bad_latency = 0;
    std::vector<std::string> known, other;
    for (int f = 1; f < argc; ++f) {
        if (argv[f][0] == '+') continue;  // Verilator plusargs
        VecFile v = load(argv[f]);
        if (v.kind != 0) {
            printf("%s: kind %d is not NVFP4, skipped\n", v.name.c_str(), v.kind);
            continue;
        }
        const uint8_t* recs = reinterpret_cast<const uint8_t*>(v.body.data());
        std::vector<uint32_t> d(v.count);
        std::vector<uint64_t> t_in(v.count);
        uint64_t cycle = 0;
        size_t fed = 0, got = 0;
        while (got < v.count) {
            if (fed < v.count) {
                const uint8_t* r = recs + 76 * fed;
                for (int w = 0; w < 8; ++w) {
                    top.a[w] = le32(r + 4 * w);
                    top.b[w] = le32(r + 32 + 4 * w);
                }
                top.sfa = le32(r + 64);
                top.sfb = le32(r + 68);
                top.c = le32(r + 72);
                top.in_vld = 1;
                t_in[fed++] = cycle;
            } else {
                top.in_vld = 0;
            }
            tick();
            ++cycle;
            if (top.out_vld) {
                if (cycle - t_in[got] != LATENCY) ++bad_latency;
                d[got++] = top.d;
            }
            if (cycle > v.count + 100) {
                fprintf(stderr, "%s: only %zu of %zu outputs\n", v.name.c_str(), got, v.count);
                return 1;
            }
        }

        printf("%-20s %7zu vectors", v.name.c_str(), v.count);
        std::vector<int> cols;
        for (const char* s : SILICON)
            for (size_t k = 0; k < v.columns.size(); ++k)
                if (v.columns[k] == s) cols.push_back(int(k));
        std::vector<size_t> ok(cols.size(), 0);
        for (size_t i = 0; i < v.count; ++i) {
            const uint8_t* r = recs + 76 * i;
            std::string outs, cls;
            bool off = false, unexplained = false;
            for (size_t j = 0; j < cols.size(); ++j) {
                uint32_t s = le32(recs + 76 * v.count + 4 * v.count * cols[j] + 4 * i);
                char buf[64];
                snprintf(buf, sizeof buf, " %s=%08x", v.columns[cols[j]].c_str(), s);
                outs += buf;
                if (same(d[i], s)) { ++ok[j]; continue; }
                off = true;
                const char* g = gap_class(r);
                if (!g || (!cls.empty() && cls != g)) unexplained = true;
                if (g) cls = g;
            }
            if (!off) continue;
            char line[512];
            snprintf(line, sizeof line, "  %s #%llu: rtl=%08x%s [%s]", v.name.c_str(),
                     (unsigned long long)(v.indices.empty() ? i : v.indices[i]), d[i], outs.c_str(),
                     unexplained ? "unexplained" : cls.c_str());
            (unexplained ? other : known).push_back(line);
        }
        for (size_t j = 0; j < cols.size(); ++j) printf("   RTL == %s %7zu", v.columns[cols[j]].c_str(), ok[j]);
        printf("\n");
        comparisons += v.count * cols.size();
    }
    n_known = known.size();
    n_other = other.size();
    printf("known gaps (documented): %zu vectors\n", n_known);
    for (auto& s : known) printf("%s\n", s.c_str());
    printf("other disagreements: %zu vectors\n", n_other);
    for (auto& s : other) printf("%s\n", s.c_str());
    printf("latency: %s\n", bad_latency ? "WRONG" : "5 cycles on every vector");
    bool pass = !n_other && !bad_latency;
    printf("%s: %zu comparisons; vectors off silicon: %zu known gap, %zu other\n", pass ? "PASS" : "FAIL",
           comparisons, n_known, n_other);
    top.final();
    return pass ? 0 : 1;
}
