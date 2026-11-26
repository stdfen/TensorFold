#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "native_moe.cu"

static void ck(cudaError_t e, const char* where) { if (e != cudaSuccess) { fprintf(stderr, "FAIL %s: %s\n", where, cudaGetErrorString(e)); exit(1); } }
template <typename T> static T* dev(const T* h, size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T)), "malloc"); if (h) ck(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice), "h2d"); return p; }

int main() {
    const int D = 32, I = 32, E = 2, S = 2;
    __nv_bfloat16 x[D]; for (auto& v : x) v = __float2bfloat16(1.0f);
    int pick[S] = {0, 1}; float route[S] = {0.25f, 0.75f}; uint8_t kinds[E] = {0, 1};
    uint8_t routed[16 * I], routed_s[I] = {}; uint8_t shared[D * I], shared_s[1] = {127};
    for (auto& v : routed) v = 0x22; for (auto& v : routed_s) v = 127; for (auto& v : shared) v = 0x38;
    auto dx = dev(x, D); auto dp = dev(pick, S); auto dr = dev(route, S); auto dk = dev(kinds, E);
    auto rw1 = dev(routed, 16 * I); auto rs1 = dev(routed_s, I); auto sw1 = dev(shared, D * I); auto ss1 = dev(shared_s, 1);
    uint64_t hw[E] = {(uint64_t)rw1, (uint64_t)sw1}, hs[E] = {(uint64_t)rs1, (uint64_t)ss1};
    auto w1 = dev(hw, E); auto s1 = dev(hs, E); auto w3 = dev(hw, E); auto s3 = dev(hs, E); auto w2 = dev(hw, E); auto s2 = dev(hs, E);
    auto act = dev<__nv_bfloat16>(nullptr, S * I); auto out = dev<float>(nullptr, D);
    tf_dsv41_native_moe_gateup<<<1,128>>>(dx, dp, w1, s1, w3, s3, dk, act, 1, S, E, D, I, 10.0f);
    tf_dsv41_native_moe_down<<<1,128>>>(act, dp, dr, w2, s2, dk, out, 1, S, E, D, I);
    float got[D]; ck(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost), "d2h");
    const float gate = 10.0f / (1.0f + expf(-10.0f)) * 10.0f;
    const float rounded = __bfloat162float(__float2bfloat16_rn(gate));
    // Routed MXFP4 consumes rounded BF16 directly (32 * 100); the shared MXFP8 down projection dynamically
    // requantizes 100 to E4M3 384 at scale 0.25, i.e. 96 (32 * 96). Route weights combine both paths.
    const float want = 0.25f * I * rounded + 0.75f * I * 96.0f;
    for (int i = 0; i < D; ++i) if (fabsf(got[i] - want) > 1e-3f * want) { fprintf(stderr, "FAIL out[%d] %.9g want %.9g\n", i, got[i], want); return 1; }
    ck(cudaDeviceSynchronize(), "sync"); puts("PASS native routed+shared MoE");
    return 0;
}
