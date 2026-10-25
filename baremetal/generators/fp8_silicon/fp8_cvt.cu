// Runs the FP8 and BF16 conversions of the Atlas FP8 path through the sm_120a conversion
// instructions (PTX ISA, "cvt"). x * G(e) = x * 2^(e - 127) is formed in FP32 with exact
// power-of-two multiplies (two for e = 255, because 2^128 is not an FP32 value).
//
//   fp8_cvt vpack <out.bin>          2^24 bytes; index e << 16 | BF16 code x:
//                                    cvt.rn.satfinite.e4m3x2.f32 of x * G(e)
//   fp8_cvt vunpack <out.bin>        2^16 uint16; index e << 8 | E4M3 code x:
//                                    cvt.rn.f16x2.e4m3x2, cvt.f32.f16, * G(e), cvt.rn.bf16.f32
//   fp8_cvt push <out.bin>           256 uint32; E4M3 code: cvt.rn.f16x2.e4m3x2, cvt.f32.f16
//   fp8_cvt pop <in.bin> <out.bin>   records {uint32 x, uint32 e} -> one byte:
//                                    cvt.rn.satfinite.e4m3x2.f32 of x * G(e)
//   fp8_cvt e4m3 <out.bin>           every FP32 x: cvt.rn.satfinite.e4m3x2.f32 of x, as runs
//   fp8_cvt bf16 <out.bin>           every FP32 x: cvt.rn.bf16.f32 of x, as runs
//   fp8_cvt bf16x2 <out.bin>         every FP32 x: cvt.rn.bf16x2.f32 of x, as runs
//
// A run is a record {uint32 x, uint32 d}: the conversion gives d from x up to the next run's x.
// Each x2 conversion converts the same value in both halves; a difference is reported.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__device__ unsigned int halves_differ;

__device__ float scale(float x, uint32_t e) {
    if (e == 255) return __fmul_rn(__fmul_rn(x, 0x1p127f), 2.0f);
    return __fmul_rn(x, __uint_as_float(e ? e << 23 : 0x0040'0000));  // G(0) = 2^-127, subnormal
}

__device__ uint32_t to_e4m3(float x) {
    uint16_t d;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(x), "f"(x));
    if ((d >> 8) != (d & 0xFF)) atomicOr(&halves_differ, 1);
    return d & 0xFF;
}

__device__ uint32_t to_bf16(float x) {
    uint16_t d;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(x));
    return d;
}

__device__ uint32_t to_bf16x2(float x) {
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(x), "f"(x));
    if ((d >> 16) != (d & 0xFFFF)) atomicOr(&halves_differ, 1);
    return d & 0xFFFF;
}

__device__ float from_e4m3(uint32_t x) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)(x << 8 | x)));
    if ((h2 >> 16) != (h2 & 0xFFFF)) atomicOr(&halves_differ, 1);
    float d;
    asm("cvt.f32.f16 %0, %1;" : "=f"(d) : "h"((uint16_t)h2));
    return d;
}

__global__ void vpack(uint8_t* out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = to_e4m3(scale(__uint_as_float((i & 0xFFFF) << 16), i >> 16));
}

__global__ void vunpack(uint16_t* out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = to_bf16(scale(from_e4m3(i & 0xFF), i >> 8));
}

__global__ void push(uint32_t* out) {
    out[threadIdx.x] = __float_as_uint(from_e4m3(threadIdx.x));
}

__global__ void pop(const uint32_t* in, uint8_t* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = to_e4m3(scale(__uint_as_float(in[2 * i]), in[2 * i + 1]));
}

// Each thread scans 2^16 consecutive inputs and appends a run where the output changes.
template <uint32_t (*CVT)(float)>
__global__ void runs(uint2* out, unsigned int* count, unsigned int cap) {
    uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) << 16;
    uint32_t prev = first ? CVT(__uint_as_float(first - 1)) : ~0u;
    for (uint32_t k = 0; k < 0x10000; k++) {
        uint32_t d = CVT(__uint_as_float(first + k));
        if (d != prev) {
            unsigned int slot = atomicAdd(count, 1);
            if (slot < cap) out[slot] = make_uint2(first + k, d);
        }
        prev = d;
    }
}

#define CK(x)                                                                         \
    do {                                                                              \
        cudaError_t e = (x);                                                          \
        if (e != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

template <typename T>
static T* dev(size_t n) {
    T* d;
    CK(cudaMalloc(&d, sizeof(T) * n));
    return d;
}

template <typename T>
static std::vector<T> host(const T* d, size_t n) {
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<T> out(n);
    CK(cudaMemcpy(out.data(), d, sizeof(T) * n, cudaMemcpyDeviceToHost));
    return out;
}

template <uint32_t (*CVT)(float)>
static std::vector<uint2> scan() {
    const unsigned int cap = 1 << 20;
    uint2* d = dev<uint2>(cap);
    unsigned int* count = dev<unsigned int>(1);
    CK(cudaMemset(count, 0, 4));
    runs<CVT><<<256, 256>>>(d, count, cap);
    unsigned int n = host(count, 1)[0];
    if (n > cap) fprintf(stderr, "%u runs, more than %u\n", n, cap), exit(1);
    std::vector<uint2> out = host(d, n);
    std::sort(out.begin(), out.end(), [](uint2 p, uint2 q) { return p.x < q.x; });
    return out;
}

template <typename T>
static void save(const char* path, const std::vector<T>& v) {
    FILE* f = fopen(path, "wb");
    if (!f || fwrite(v.data(), sizeof(T), v.size(), f) != v.size()) perror(path), exit(1);
    fclose(f);
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "";
    const char* out = argv[argc - 1];
    if (!strcmp(mode, "vpack") && argc == 3) {
        uint8_t* d = dev<uint8_t>(1 << 24);
        vpack<<<1 << 16, 256>>>(d);
        save(out, host(d, 1 << 24));
    } else if (!strcmp(mode, "vunpack") && argc == 3) {
        uint16_t* d = dev<uint16_t>(1 << 16);
        vunpack<<<256, 256>>>(d);
        save(out, host(d, 1 << 16));
    } else if (!strcmp(mode, "push") && argc == 3) {
        uint32_t* d = dev<uint32_t>(256);
        push<<<1, 256>>>(d);
        save(out, host(d, 256));
    } else if (!strcmp(mode, "pop") && argc == 4) {
        FILE* f = fopen(argv[2], "rb");
        if (!f) return perror(argv[2]), 1;
        std::vector<uint32_t> in;
        uint32_t w;
        while (fread(&w, 4, 1, f) == 1) in.push_back(w);
        fclose(f);
        uint32_t n = in.size() / 2;
        uint32_t* d_in = dev<uint32_t>(2 * n);
        uint8_t* d = dev<uint8_t>(n);
        CK(cudaMemcpy(d_in, in.data(), 8 * n, cudaMemcpyHostToDevice));
        pop<<<(n + 255) / 256, 256>>>(d_in, d, n);
        save(out, host(d, n));
    } else if (!strcmp(mode, "e4m3") && argc == 3) {
        save(out, scan<to_e4m3>());
    } else if (!strcmp(mode, "bf16") && argc == 3) {
        save(out, scan<to_bf16>());
    } else if (!strcmp(mode, "bf16x2") && argc == 3) {
        save(out, scan<to_bf16x2>());
    } else {
        fprintf(stderr, "usage: %s vpack|vunpack|push|e4m3|bf16|bf16x2 <out.bin>\n"
                        "       %s pop <in.bin> <out.bin>\n", argv[0], argv[0]);
        return 2;
    }
    unsigned int differ;
    CK(cudaMemcpyFromSymbol(&differ, halves_differ, 4));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    fprintf(stderr, "%s sm_%d%d: %s%s\n", p.name, p.major, p.minor, mode,
            differ ? ", the two halves of an x2 conversion differ" : "");
    return differ ? 1 : 0;
}
