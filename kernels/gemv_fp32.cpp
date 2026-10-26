// L1: FP32 GEMV baseline, y = W x.
//
// x is copied on chip once, then each row is a dot product over k with a
// pipelined loop. acc depends on its own previous value, so a new add can only
// start when the previous FP add has finished: the loop II equals the FP add
// latency instead of 1. This is the expected bottleneck of L1 (fixed in L3 with
// partial sums). The order of operations matches the golden model in
// scripts/gen_data.py (k = 0..K-1, multiply then add), so the output is bit-exact.
#include "gemv.h"

extern "C" void gemv_fp32(const float* W, const float* x, float* y, int M, int K) {
#pragma HLS INTERFACE m_axi port = W bundle = gmem0 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = x bundle = gmem1
#pragma HLS INTERFACE m_axi port = y bundle = gmem1

    float xbuf[MAX_K];

load_x:
    for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 4096
        xbuf[k] = x[k];
    }

rows:
    for (int m = 0; m < M; m++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 4096
        float acc = 0.0f;
    dot:
        for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 4096
            acc += W[(long)m * K + k] * xbuf[k];
        }
        y[m] = acc;
    }
}
