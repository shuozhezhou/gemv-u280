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
