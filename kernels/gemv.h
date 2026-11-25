// Kernel interfaces for the GEMV project. y = W x, W is M x K, row-major.
// The argument names are used by kernels/*.cfg (HBM mapping) and the host.
#pragma once

#include <ap_int.h>
#include <cstdint>

constexpr int GROUP = 64;      // weights per quantization group
constexpr int MAX_K = 4096;    // size of the on-chip copy of x

// L1: FP32 baseline.
//   W : M*K float  (m_axi bundle gmem0 -> HBM[0])
//   x : K float    (m_axi bundle gmem1 -> HBM[1])
//   y : M float    (m_axi bundle gmem1 -> HBM[1])
extern "C" void gemv_fp32(const float* W, const float* x, float* y, int M, int K);

// L2: group-wise INT8. One 512-bit word = one group of 64 int8 weights.
//   Wq : M*K/64 words      (gmem0 -> HBM[0])
//   s  : M*K/64 float      (gmem1 -> HBM[1])  scale per row and group
//   xq : K/64 words        (gmem2 -> HBM[2])
//   sx : activation scale  (scalar)
//   y  : M float           (gmem2 -> HBM[2])
extern "C" void gemv_int8(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                          float sx, float* y, int M, int K);

// L3a: same interface and HBM mapping as gemv_int8 (y on its own bundle gmem3,
// still HBM[2]). Dataflow: terms -> NPS = 8 FP32 partial sums -> tree -> y.
// Output order differs from gemv_int8: golden y_int8ps.f32.bin.
extern "C" void gemv_int8_ps(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                             float sx, float* y, int M, int K);

// L3b: as gemv_int8_ps, but one ap_fixed<48, 24> accumulator per row instead of
// FP32 partial sums. Golden: y_int8fx.f32.bin.
extern "C" void gemv_int8_fx(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                             float sx, float* y, int M, int K);

// L3c: symmetric INT4 weights, 128 per 512-bit word (2 groups of 64), one FP32
// scale per group; s = the float array s4[m][g] read as 64-bit pairs. K % 128 == 0.
// Golden: y_int4.f32.bin. Files: Wq4.i4.bin (packed), s4.f32.bin.
extern "C" void gemv_int4_ps(const ap_uint<512>* Wq, const ap_uint<64>* s, const ap_uint<512>* xq,
                             float sx, float* y, int M, int K);

// L3d: batch of nb <= BMAX (8) activation vectors per weight pass.
//   xq : nb*K/64 words (vector b first), sxv : nb floats, y : nb*M floats, y[b*M + m].
// Golden: y_int8psB.f32.bin ([8][M]), inputs xqB.i8.bin, sxB.f32.bin.
extern "C" void gemv_int8_ps_b(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                               const float* sxv, float* y, int M, int K, int nb);
