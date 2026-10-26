// L2: group-wise INT8 GEMV, y = sx * sum_g s[m][g] * P[m][g].
//
// One 512-bit word of Wq holds one group: 64 int8 weights, byte j = weight j
// (little-endian, the same layout as the int8 array written by gen_data.py).
// Per group: 64 int8 x int8 products summed exactly in an int32 adder tree
// (|P| <= 64 * 127 * 127 < 2^20), then one FP32 multiply by the group scale and
// one FP32 add into the row accumulator. The FP32 add still carries a loop
// dependence, so the group loop runs at II = fadd latency, but each iteration
// now consumes 64 weights instead of 1. The op order matches golden_int8() in
// scripts/gen_data.py, so the output is bit-exact.
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

extern "C" void gemv_int8(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                          float sx, float* y, int M, int K) {
#pragma HLS INTERFACE m_axi port = Wq bundle = gmem0 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = s bundle = gmem1
#pragma HLS INTERFACE m_axi port = xq bundle = gmem2
#pragma HLS INTERFACE m_axi port = y bundle = gmem2

    const int G = K / GROUP;
    ap_uint<512> xbuf[MAX_K / GROUP];

load_x:
    for (int g = 0; g < G; g++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
        xbuf[g] = xq[g];
    }

rows:
    for (int m = 0; m < M; m++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 4096
        float acc = 0.0f;
    groups:
        for (int g = 0; g < G; g++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
            int32_t P = group_dot(Wq[(long)m * G + g], xbuf[g]);
            float term = s[(long)m * G + g] * (float)P;
            acc += term;
        }
        y[m] = sx * acc;
    }
}
