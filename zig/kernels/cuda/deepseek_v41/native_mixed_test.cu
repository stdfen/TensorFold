// Device fixture for native_mixed.cu; compile as a standalone CUDA executable.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "native_mixed.cu"

static void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "FAIL %s: %s\n", where, cudaGetErrorString(e));
        exit(1);
    }
}

template <typename T>
static T* device(const T* host, size_t count) {
    T* p = nullptr;
    check(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc");
    if (host) check(cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return p;
}

static void expect(const char* name, const float* got, const float* want, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (fabsf(got[i] - want[i]) > 1e-5f) {
            fprintf(stderr, "FAIL %s[%zu]: got %.9g want %.9g\n", name, i, got[i], want[i]);
            exit(1);
        }
    }
    printf("PASS %s\n", name);
}

int main() {
    {
        const float xf[] = {1, 1, 1, 1};
        const float wf[] = {1, 2, 3, 4, -1, 0.5f, 0, 1};
        __nv_bfloat16 x[4], w[8];
        for (int i = 0; i < 4; ++i) x[i] = __float2bfloat16(xf[i]);
        for (int i = 0; i < 8; ++i) w[i] = __float2bfloat16(wf[i]);
        auto dx = device(x, 4); auto dw = device(w, 8); auto out = device<float>(nullptr, 2);
        tf_dsv41_native_bf16<<<1, 128>>>(dx, dw, nullptr, out, 1, 2, 4);
        float got[2]; check(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost), "bf16 D2H");
        const float want[] = {10, 0.5f}; expect("bf16", got, want, 2);
        auto out16 = device<__nv_bfloat16>(nullptr, 2);
        tf_dsv41_native_bf16_bf16<<<1, 128>>>(dx, dw, nullptr, out16, 1, 2, 4);
        __nv_bfloat16 got16_raw[2]; check(cudaMemcpy(got16_raw, out16, sizeof(got16_raw), cudaMemcpyDeviceToHost), "bf16 output D2H");
        const float got16[] = {__bfloat162float(got16_raw[0]), __bfloat162float(got16_raw[1])};
        expect("bf16 output", got16, want, 2);
        cudaFree(dx); cudaFree(dw); cudaFree(out); cudaFree(out16);
    }
    {
        __nv_bfloat16 x[32]; uint8_t w[64], scales[1] = {127};
        for (int i = 0; i < 32; ++i) x[i] = __float2bfloat16(1.0f);
        for (int i = 0; i < 64; ++i) w[i] = 0x38;
        auto dx = device(x, 32); auto dw = device(w, 64); auto ds = device(scales, 1); auto out = device<float>(nullptr, 2);
        tf_dsv41_native_mxfp8<<<1, 128>>>(dx, dw, ds, out, 1, 2, 32);
        float got[2]; check(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost), "mxfp8 D2H");
        const float want[] = {32, 32}; expect("mxfp8 dynamic", got, want, 2);
        cudaFree(dx); cudaFree(dw); cudaFree(ds); cudaFree(out);
    }
    {
        __nv_bfloat16 x[32]; uint8_t w[32], scales[2] = {127, 128};
        for (int i = 0; i < 32; ++i) x[i] = __float2bfloat16(1.0f);
        for (int i = 0; i < 16; ++i) w[i] = 0x22;
        for (int i = 16; i < 32; ++i) w[i] = 0xaa;
        auto dx = device(x, 32); auto dw = device(w, 32); auto ds = device(scales, 2); auto out = device<float>(nullptr, 2);
        tf_dsv41_native_mxfp4<<<1, 128>>>(dx, dw, ds, out, 1, 2, 32);
        float got[2]; check(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost), "mxfp4 D2H");
        const float want[] = {32, -64}; expect("mxfp4 group32", got, want, 2);
        cudaFree(dx); cudaFree(dw); cudaFree(ds); cudaFree(out);
    }
    {
        const float xf[] = {1, 10, 100, 1000};
        __nv_bfloat16 x[4]; for (int i = 0; i < 4; ++i) x[i] = __float2bfloat16(xf[i]);
        const uint8_t w[] = {0x38, 0x40, 0x44, 0x48}; const uint8_t scales[] = {127};
        auto dx = device(x, 4); auto dw = device(w, 4); auto ds = device(scales, 1); auto out = device<float>(nullptr, 2);
        tf_dsv41_native_wo_a<<<1, 128>>>(dx, dw, ds, out, 1, 2, 1, 2);
        float got[2]; check(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost), "wo_a D2H");
        const float want[] = {21, 4300}; expect("wo_a", got, want, 2);
        cudaFree(dx); cudaFree(dw); cudaFree(ds); cudaFree(out);
    }
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return 0;
}
