// Real official-safetensors slices against native_mixed.cu. The generated fixture header is supplied with -I.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "native_mixed.cu"
#include "native_mixed_official_fixture.h"

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
    if (host && count) check(cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return p;
}

int main() {
    for (size_t ci = 0; ci < tf_dsv41_official_case_count; ++ci) {
        const auto& c = tf_dsv41_official_cases[ci];
        __nv_bfloat16* input = (__nv_bfloat16*)malloc(c.k * sizeof(__nv_bfloat16));
        for (size_t i = 0; i < c.k; ++i) input[i] = __float2bfloat16(c.input[i]);
        auto dx = device(input, c.k);
        auto ds = c.scale_len ? device(c.scale, c.scale_len) : (uint8_t*)nullptr;
        auto out = device<float>(nullptr, c.n);
        if (c.kind == 0) {
            auto dw = device((const __nv_bfloat16*)c.weight, c.weight_len / 2);
            tf_dsv41_native_bf16<<<(c.n + 127) / 128, 128>>>(dx, dw, nullptr, out, 1, c.n, c.k);
            cudaFree(dw);
        } else {
            auto dw = device(c.weight, c.weight_len);
            if (c.kind == 1) tf_dsv41_native_mxfp8_plain<<<(c.n + 127) / 128, 128>>>(dx, dw, ds, out, 1, c.n, c.k);
            else if (c.kind == 2) tf_dsv41_native_mxfp4<<<(c.n + 127) / 128, 128>>>(dx, dw, ds, out, 1, c.n, c.k);
            else tf_dsv41_native_wo_a<<<(c.n + 127) / 128, 128>>>(dx, dw, ds, out, 1, 1, c.n, c.k);
            cudaFree(dw);
        }
        float* got = (float*)malloc(c.n * sizeof(float));
        check(cudaMemcpy(got, out, c.n * sizeof(float), cudaMemcpyDeviceToHost), "output D2H");
        for (size_t i = 0; i < c.n; ++i) {
            const float tolerance = 1e-5f + 1e-5f * fabsf(c.expected[i]);
            if (fabsf(got[i] - c.expected[i]) > tolerance) {
                fprintf(stderr, "FAIL %s[%zu]: got %.9g want %.9g tolerance %.9g\n", c.name, i, got[i], c.expected[i], tolerance);
                return 1;
            }
        }
        printf("PASS official %s\n", c.name);
        free(got); free(input); cudaFree(dx); cudaFree(ds); cudaFree(out);
    }
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return 0;
}
