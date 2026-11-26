// Correctness-first routed MoE over official MXFP4 experts plus the official MXFP8 shared expert.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace tf_dsv41_native_moe {

__device__ __forceinline__ float ue8m0(uint8_t x) { return x == 255 ? nanf("") : ldexpf(1.0f, int(x) - 127); }
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
__device__ __forceinline__ float dynamic_e4m3(const __nv_bfloat16* x, int col, int k) {
    const int first = col & ~31;
    const int last = min(first + 32, k);
    float amax = 1.0e-4f;
    for (int i = first; i < last; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(x[i])));
    const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
    const float scale = ldexpf(1.0f, max(-127, min(127, exponent)));
    const uint8_t code = __nv_cvt_float_to_fp8(fminf(448.0f, fmaxf(-448.0f, __bfloat162float(x[col]) / scale)), __NV_SATFINITE, __NV_E4M3);
    return __bfloat162float(__float2bfloat16_rn(e4m3(code) * scale));
}
__device__ __forceinline__ float dot(const __nv_bfloat16* x, const uint8_t* w, const uint8_t* s,
                                     int row, int k, bool fp8) {
    float sum = 0.0f;
    for (int col = 0; col < k; ++col) {
        float xv = __bfloat162float(x[col]);
        float wv;
        if (fp8) {
            xv = dynamic_e4m3(x, col, k);
            wv = e4m3(w[(long long)row * k + col]) * ue8m0(s[(row / 32) * ((k + 31) / 32) + col / 32]);
        } else {
            const uint8_t packed = w[(long long)row * (k / 2) + col / 2];
            wv = e2m1(col & 1 ? packed >> 4 : packed & 15) * ue8m0(s[(long long)row * (k / 32) + col / 32]);
        }
        sum = fmaf(xv, wv, sum);
    }
    return sum;
}
__device__ __forceinline__ const uint8_t* ptr(const uint64_t* table, int expert) {
    return reinterpret_cast<const uint8_t*>(table[expert]);
}

}  // namespace tf_dsv41_native_moe

extern "C" __global__ void tf_dsv41_native_moe_gateup(
    const __nv_bfloat16* x, const int* pick,
    const uint64_t* w1, const uint64_t* s1, const uint64_t* w3, const uint64_t* s3,
    const uint8_t* fp8, __nv_bfloat16* act,
    int rows, int slots, int experts, int hidden, int width, float limit) {
    const long long at = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = (long long)rows * slots * width;
    if (at >= count) return;
    const int i = at % width;
    const int pair = at / width;
    const int expert = pick[pair];
    if (expert < 0 || expert >= experts) { act[at] = __float2bfloat16(0.0f); return; }
    const __nv_bfloat16* row = x + (long long)(pair / slots) * hidden;
    const bool dense = fp8[expert] != 0;
    const float g = tf_dsv41_native_moe::dot(row, tf_dsv41_native_moe::ptr(w1, expert), tf_dsv41_native_moe::ptr(s1, expert), i, hidden, dense);
    const float u = tf_dsv41_native_moe::dot(row, tf_dsv41_native_moe::ptr(w3, expert), tf_dsv41_native_moe::ptr(s3, expert), i, hidden, dense);
    const float a = fminf(g, limit);
    const float silu = a / (1.0f + expf(-a));
    act[at] = __float2bfloat16_rn(silu * fminf(limit, fmaxf(-limit, u)));
}

extern "C" __global__ void tf_dsv41_native_moe_down(
    const __nv_bfloat16* act, const int* pick, const float* route,
    const uint64_t* w2, const uint64_t* s2, const uint8_t* fp8, float* out,
    int rows, int slots, int experts, int hidden, int width) {
    const long long at = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = (long long)rows * hidden;
    if (at >= count) return;
    const int row = at / hidden;
    const int output = at % hidden;
    float total = 0.0f;
    for (int slot = 0; slot < slots; ++slot) {
        const int pair = row * slots + slot;
        const int expert = pick[pair];
        if (expert < 0 || expert >= experts) continue;
        const bool dense = fp8[expert] != 0;
        const float value = tf_dsv41_native_moe::dot(act + (long long)pair * width,
            tf_dsv41_native_moe::ptr(w2, expert), tf_dsv41_native_moe::ptr(s2, expert), output, width, dense);
        total = fmaf(route[pair], value, total);
    }
    out[at] = total;
}
