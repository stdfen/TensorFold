// Correctness-first operators over DeepSeek-V4.1-Flash-1017's official row-major mixed weights.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace tf_dsv41_native {

__device__ __forceinline__ float ue8m0(uint8_t x) {
    return x == 255 ? nanf("") : ldexpf(1.0f, static_cast<int>(x) - 127);
}

__device__ __forceinline__ float e4m3(uint8_t x) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(x), __NV_E4M3)));
}

__device__ __forceinline__ float e2m1(uint8_t x) {
    const float sign = x & 8 ? -1.0f : 1.0f;
    const int exponent = (x >> 1) & 3;
    const int mantissa = x & 1;
    if (exponent == 0) return sign * (mantissa ? 0.5f : 0.0f);
    return sign * ldexpf(1.0f + 0.5f * mantissa, exponent - 1);
}

__device__ __forceinline__ float bf16_round(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float dynamic_e4m3(const __nv_bfloat16* x, int row, int col, int k) {
    const int first = col & ~31;
    const int last = min(first + 32, k);
    float amax = 1.0e-4f;
    for (int i = first; i < last; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(x[static_cast<long long>(row) * k + i])));
    const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
    const float scale = ldexpf(1.0f, max(-127, min(127, exponent)));
    const float value = __bfloat162float(x[static_cast<long long>(row) * k + col]);
    const uint8_t code = __nv_cvt_float_to_fp8(fminf(448.0f, fmaxf(-448.0f, value / scale)), __NV_SATFINITE, __NV_E4M3);
    return bf16_round(e4m3(code) * scale);
}

template <typename T>
__device__ __forceinline__ void store(T* out, long long at, float value) { out[at] = value; }

template <>
__device__ __forceinline__ void store(__nv_bfloat16* out, long long at, float value) { out[at] = __float2bfloat16_rn(value); }

template <bool DYNAMIC, typename Out>
__device__ void mxfp8_body(const __nv_bfloat16* x, const uint8_t* weight, const uint8_t* scales,
                           Out* out, int rows, int n, int k) {
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long count = static_cast<long long>(rows) * n;
    if (at >= count) return;
    const int batch = at / n;
    const int output = at % n;
    const int kb = (k + 31) / 32;
    float sum = 0.0f;
    for (int col = 0; col < k; ++col) {
        const float xv = DYNAMIC ? dynamic_e4m3(x, batch, col, k) : __bfloat162float(x[static_cast<long long>(batch) * k + col]);
        const float wv = e4m3(weight[static_cast<long long>(output) * k + col]) * ue8m0(scales[(output / 32) * kb + col / 32]);
        sum = fmaf(xv, wv, sum);
    }
    store(out, at, sum);
}

template <typename Out>
__device__ void mxfp4_body(const __nv_bfloat16* x, const uint8_t* weight, const uint8_t* scales,
                           Out* out, int rows, int n, int k) {
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long count = static_cast<long long>(rows) * n;
    if (at >= count) return;
    const int batch = at / n;
    const int output = at % n;
    float sum = 0.0f;
    for (int col = 0; col < k; ++col) {
        const uint8_t packed = weight[static_cast<long long>(output) * (k / 2) + col / 2];
        const uint8_t code = col & 1 ? packed >> 4 : packed & 15;
        const float wv = e2m1(code) * ue8m0(scales[static_cast<long long>(output) * (k / 32) + col / 32]);
        sum = fmaf(__bfloat162float(x[static_cast<long long>(batch) * k + col]), wv, sum);
    }
    store(out, at, sum);
}

template <typename Out>
__device__ void bf16_body(const __nv_bfloat16* x, const __nv_bfloat16* weight, Out* out, int rows, int n, int k) {
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long count = static_cast<long long>(rows) * n;
    if (at >= count) return;
    const int batch = at / n;
    const int output = at % n;
    float sum = 0.0f;
    for (int col = 0; col < k; ++col)
        sum = fmaf(__bfloat162float(x[static_cast<long long>(batch) * k + col]),
                   __bfloat162float(weight[static_cast<long long>(output) * k + col]), sum);
    store(out, at, sum);
}

template <typename Out>
__device__ void wo_a_body(const __nv_bfloat16* x, const uint8_t* weight, const uint8_t* scales,
                          Out* out, int rows, int groups, int rank, int k) {
    const int n = groups * rank;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long count = static_cast<long long>(rows) * n;
    if (at >= count) return;
    const int batch = at / n;
    const int output = at % n;
    const int group = output / rank;
    const int kb = (k + 31) / 32;
    float sum = 0.0f;
    for (int col = 0; col < k; ++col) {
        const float wv = bf16_round(e4m3(weight[static_cast<long long>(output) * k + col]) *
                                    ue8m0(scales[(output / 32) * kb + col / 32]));
        sum = fmaf(__bfloat162float(x[(static_cast<long long>(batch) * groups + group) * k + col]), wv, sum);
    }
    store(out, at, sum);
}

}  // namespace tf_dsv41_native

extern "C" __global__ void tf_dsv41_native_mxfp8(const __nv_bfloat16* x, const uint8_t* weight,
                                                   const uint8_t* scales, float* out, int rows, int n, int k) {
    tf_dsv41_native::mxfp8_body<true>(x, weight, scales, out, rows, n, k);
}

extern "C" __global__ void tf_dsv41_native_mxfp8_plain(const __nv_bfloat16* x, const uint8_t* weight,
                                                         const uint8_t* scales, float* out, int rows, int n, int k) {
    tf_dsv41_native::mxfp8_body<false>(x, weight, scales, out, rows, n, k);
}

extern "C" __global__ void tf_dsv41_native_mxfp8_bf16(const __nv_bfloat16* x, const uint8_t* weight,
                                                        const uint8_t* scales, __nv_bfloat16* out, int rows, int n, int k) {
    tf_dsv41_native::mxfp8_body<true>(x, weight, scales, out, rows, n, k);
}

extern "C" __global__ void tf_dsv41_native_mxfp8_plain_bf16(const __nv_bfloat16* x, const uint8_t* weight,
                                                              const uint8_t* scales, __nv_bfloat16* out, int rows, int n, int k) {
    tf_dsv41_native::mxfp8_body<false>(x, weight, scales, out, rows, n, k);
}

extern "C" __global__ void tf_dsv41_native_mxfp4(const __nv_bfloat16* x, const uint8_t* weight,
                                                   const uint8_t* scales, float* out, int rows, int n, int k) {
    tf_dsv41_native::mxfp4_body(x, weight, scales, out, rows, n, k);
}

extern "C" __global__ void tf_dsv41_native_mxfp4_bf16(const __nv_bfloat16* x, const uint8_t* weight,
                                                        const uint8_t* scales, __nv_bfloat16* out, int rows, int n, int k) {
    tf_dsv41_native::mxfp4_body(x, weight, scales, out, rows, n, k);
}

extern "C" __global__ void tf_dsv41_native_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                                  const uint8_t*, float* out, int rows, int n, int k) {
    tf_dsv41_native::bf16_body(x, weight, out, rows, n, k);
}

extern "C" __global__ void tf_dsv41_native_bf16_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                                       const uint8_t*, __nv_bfloat16* out, int rows, int n, int k) {
    tf_dsv41_native::bf16_body(x, weight, out, rows, n, k);
}

extern "C" __global__ void tf_dsv41_native_wo_a(const __nv_bfloat16* x, const uint8_t* weight,
                                                  const uint8_t* scales, float* out, int rows, int groups,
                                                  int rank, int k) {
    tf_dsv41_native::wo_a_body(x, weight, scales, out, rows, groups, rank, k);
}

extern "C" __global__ void tf_dsv41_native_wo_a_bf16(const __nv_bfloat16* x, const uint8_t* weight,
                                                       const uint8_t* scales, __nv_bfloat16* out, int rows, int groups,
                                                       int rank, int k) {
    tf_dsv41_native::wo_a_body(x, weight, scales, out, rows, groups, rank, k);
}
