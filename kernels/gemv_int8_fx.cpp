// L3b: group-wise INT8 GEMV with a fixed-point row accumulator, group loop at II = 1.
//
// Instead of FP32 partial sums (gemv_int8_ps), every term t = s*P is converted to
// ap_fixed<48, 24> (24 integer bits incl. sign, 24 fraction bits; default AP_TRN =
// floor, AP_WRAP) and added into one fixed-point accumulator. A 48-bit integer add
// finishes in one cycle, so the single accumulator needs no partial sums, and the
// sum is exact (no rounding, independent of the order). At the end of the row the
// accumulator is converted to FP32 (round to nearest even) and scaled:
//   y[m] = sx * (float(acc) * 2^-24),
// which is golden_int8_fx() in scripts/gen_data.py (golden file y_int8fx.f32.bin).
// Range: the fixed-point range is +-2^23; in all our test cases |t| and every
// running sum stay below 407 (< 2^9), and gen_data.py asserts that no case
// overflows. Truncating a term costs less than 2^-24, i.e. < 64 x 2^-24 = 3.8e-6
// per row at K = 4096 (before the sx scale), far below the INT8 quantization error.
//
// Dataflow: compute_terms (int8_terms.h) -> accumulate_fx -> scale_store.
#include <ap_fixed.h>

#include "int8_terms.h"

constexpr int FX_BITS = 48, FX_FRAC = 24;
typedef ap_fixed<FX_BITS, FX_BITS - FX_FRAC> acc_t;

static void accumulate_fx(hls::stream<float>& terms, hls::stream<acc_t>& sums, int M, int K) {
    const int G = K / GROUP;
    acc_t acc = 0;
    int g = 0;
sums:
    for (int i = 0; i < M * G; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 262144
        acc_t t = terms.read();             // float -> fixed, floor
        acc_t a = (g == 0) ? acc_t(0) : acc;
        a += t;                              // exact
        acc = a;
        if (g == G - 1) sums.write(a);
        g = (g == G - 1) ? 0 : g + 1;
    }
}

static void scale_store(hls::stream<acc_t>& sums, float sx, float* y, int M) {
rows:
    for (int m = 0; m < M; m++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
        acc_t a = sums.read();
        ap_int<FX_BITS> raw = a.range(FX_BITS - 1, 0);    // acc * 2^24 as an integer
        float f = (float)(long long)raw * 5.9604644775390625e-08f;  // * 2^-24, exact
        y[m] = sx * f;
    }
}

extern "C" void gemv_int8_fx(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                             float sx, float* y, int M, int K) {
#pragma HLS INTERFACE m_axi port = Wq bundle = gmem0 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = s bundle = gmem1 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = xq bundle = gmem2
#pragma HLS INTERFACE m_axi port = y bundle = gmem3
#pragma HLS DATAFLOW

    hls::stream<float> terms("terms");
    hls::stream<acc_t> sums("sums");
#pragma HLS STREAM variable = terms depth = 64
#pragma HLS STREAM variable = sums depth = 8

    compute_terms(Wq, s, xq, terms, M, K);
    accumulate_fx(terms, sums, M, K);
    scale_store(sums, sx, y, M);
}
