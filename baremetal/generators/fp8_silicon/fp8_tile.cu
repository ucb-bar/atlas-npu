// K = 32 x STEPS chains of 32 x 32 FP8 tiles on the GPU, for the mxu1 accumulator check.
//
//   fp8_tile <in.bin> <out.bin>
//
// A record (fp8_vectors.py tiles) is {uint32 c[32][32]; uint8 a[STEPS][32][32], w[STEPS][32][32]}.
// Step s computes D_s = D_{s-1} + A_s W_s^T (D_{-1} = C; row j of W_s is output column j) as eight
// mma.sync m16n8k32 E4M3 x E4M3 + FP32, and writes D_s: STEPS x 32 x 32 uint32 per record.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int STEPS = 4;
struct Tile {
    uint32_t c[32][32];
    uint8_t a[STEPS][32][32], w[STEPS][32][32];
};

__device__ uint32_t ld32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

__global__ void run(const Tile* tiles, uint32_t* out, int n) {
    int t_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (t_idx >= n) return;
    const Tile& T = tiles[t_idx];
    int g = lane >> 2, t = lane & 3;
    for (int s = 0; s < STEPS; s++) {
        const uint32_t* C = s ? out + 1024 * (STEPS * t_idx + s - 1) : &T.c[0][0];
        uint32_t* D = out + 1024 * (STEPS * t_idx + s);
        for (int rb = 0; rb < 32; rb += 16)
            for (int cb = 0; cb < 32; cb += 8) {
                uint32_t a0 = ld32(&T.a[s][rb + g][4 * t]), a1 = ld32(&T.a[s][rb + g + 8][4 * t]);
                uint32_t a2 = ld32(&T.a[s][rb + g][16 + 4 * t]), a3 = ld32(&T.a[s][rb + g + 8][16 + 4 * t]);
                uint32_t b0 = ld32(&T.w[s][cb + g][4 * t]), b1 = ld32(&T.w[s][cb + g][16 + 4 * t]);
                int r0 = rb + g, r1 = rb + g + 8, j = cb + 2 * t;
                float c0 = __uint_as_float(C[32 * r0 + j]), c1 = __uint_as_float(C[32 * r0 + j + 1]);
                float c2 = __uint_as_float(C[32 * r1 + j]), c3 = __uint_as_float(C[32 * r1 + j + 1]);
                float d0, d1, d2, d3;
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
                D[32 * r0 + j] = __float_as_uint(d0);
                D[32 * r0 + j + 1] = __float_as_uint(d1);
                D[32 * r1 + j] = __float_as_uint(d2);
                D[32 * r1 + j + 1] = __float_as_uint(d3);
            }
        __syncwarp();
    }
}

int main(int argc, char** argv) {
    if (argc != 3) return fprintf(stderr, "usage: fp8_tile <in.bin> <out.bin>\n"), 2;
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    std::vector<Tile> tiles;
    Tile t;
    while (fread(&t, sizeof t, 1, f) == 1) tiles.push_back(t);
    fclose(f);
    int n = tiles.size();
    Tile* d_t;
    uint32_t* d_o;
    if (cudaMalloc(&d_t, sizeof(Tile) * n) || cudaMalloc(&d_o, 4096 * STEPS * n)) return 1;
    cudaMemcpy(d_t, tiles.data(), sizeof(Tile) * n, cudaMemcpyHostToDevice);
    run<<<(n * 32 + 255) / 256, 256>>>(d_t, d_o, n);
    if (cudaDeviceSynchronize() != cudaSuccess) return fprintf(stderr, "CUDA error\n"), 1;
    std::vector<uint32_t> o(1024 * STEPS * n);
    cudaMemcpy(o.data(), d_o, 4096 * STEPS * n, cudaMemcpyDeviceToHost);
    f = fopen(argv[2], "wb");
    fwrite(o.data(), 4, o.size(), f);
    fclose(f);
    fprintf(stderr, "%d tiles, %d steps\n", n, STEPS);
    return 0;
}
