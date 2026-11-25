// Front end shared by the L3 INT8 kernels (gemv_int8_ps, gemv_int8_fx).
//
// compute_terms() streams one FP32 term per (row, group):
//   t[m][g] = s[m][g] * float(P[m][g]),  P = exact int32 dot product of 64 int8 pairs,
// in row-major order. The (m, g) loops are flattened into one loop over all M*G
// words, so Wq and s are read as one long sequential burst each and the loop runs
// at II = 1 with no per-row pipeline drain (the per-row overhead of gemv_int8).
#pragma once

#include <hls_stream.h>

#include "gemv.h"

static int32_t group_dot(ap_uint<512> w, ap_uint<512> x) {
#pragma HLS INLINE
    int32_t P = 0;
    for (int j = 0; j < GROUP; j++) {
#pragma HLS UNROLL
        ap_int<8> a = w.range(8 * j + 7, 8 * j);
        ap_int<8> b = x.range(8 * j + 7, 8 * j);
        P += (int32_t)(a * b);
    }
    return P;
}

static void compute_terms(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                          hls::stream<float>& terms, int M, int K) {
    const int G = K / GROUP;
    ap_uint<512> xbuf[MAX_K / GROUP];

load_x:
    for (int g = 0; g < G; g++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
        xbuf[g] = xq[g];
    }

    int g = 0;
words:
    for (int i = 0; i < M * G; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 262144
        int32_t P = group_dot(Wq[i], xbuf[g]);
        terms.write(s[i] * (float)P);
        g = (g == G - 1) ? 0 : g + 1;
    }
}
