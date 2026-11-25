// L3a: group-wise INT8 GEMV with NPS = 8 FP32 partial sums, group loop at II = 1.
//
// In gemv_int8 the row sum acc += s*P is one FP32 add chain, so a new group can
// only start every 7 cycles (fadd latency at 300 MHz). Here group g of a row is
// added into partial sum a[g % 8]: a given partial sum is updated only every 8th
// group, which leaves the adder 8 >= 7 cycles, so a new group starts every cycle.
// At the end of the row the 8 partial sums are added in a fixed tree,
//   ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)),   y[m] = sx * sum,
// which is golden_int8_ps() in scripts/gen_data.py (golden file y_int8ps.f32.bin).
// The summation order differs from gemv_int8, so the output is bit-exact against
// y_int8ps, not y_int8 (both are within ~1e-7 x max|y| of each other).
//
// Three dataflow processes, connected by FIFOs:
//   compute_terms (int8_terms.h)  one term s*P per word, II = 1, all M*G words
//   accumulate                    8 partial sums, 8 groups per 8 cycles, emits 8 values per row
//   reduce_store                  tree + scale, one row every 8 cycles
#include "int8_terms.h"

constexpr int NPS = 8;  // partial sums; must be >= the FP32 add latency (7)

// Adjacent pairs, level by level (same order as tree_sum() in gen_data.py).
// In place: step j writes v[j] and reads v[2j], v[2j+1], which are not yet written.
static float tree_sum(float v[NPS]) {
#pragma HLS INLINE
    for (int w = NPS / 2; w >= 1; w /= 2) {
#pragma HLS UNROLL
        for (int j = 0; j < w; j++) {
#pragma HLS UNROLL
            v[j] = v[2 * j] + v[2 * j + 1];
        }
    }
    return v[0];
}

// Each row is processed in C = ceil(G / NPS) chunks of NPS groups; group
// g = c * NPS + p goes to partial sum acc[p]. One chunk is one pipeline iteration
// at II = NPS = 8 with the NPS adds unrolled: the terms are still read one per
// cycle, and each acc[p] is a plain register updated once per 8 cycles, so its
// 7-cycle FP32 add fits without any dependence pragma (a first version with one
// group per iteration and acc[g % 8] got II = 4: HLS cannot see the distance-8
// dependence through the dynamic index). A partial sum restarts from 0.0f in the
// first chunk of a row; groups g >= G of the last chunk (padding) consume no term
// and leave it unchanged. The last chunk of a row emits acc[0..7] in order.
static void accumulate(hls::stream<float>& terms, hls::stream<float>& partials, int M, int K) {
    const int G = K / GROUP;
    const int C = (G + NPS - 1) / NPS;
    float acc[NPS];
#pragma HLS ARRAY_PARTITION variable = acc complete

    int c = 0;
chunks:
    for (int i = 0; i < M * C; i++) {
#pragma HLS PIPELINE II = 8
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32768
        for (int p = 0; p < NPS; p++) {
#pragma HLS UNROLL
            float a = (c == 0) ? 0.0f : acc[p];
            if (c * NPS + p < G) a = a + terms.read();
            acc[p] = a;
            if (c == C - 1) partials.write(a);
        }
        c = (c == C - 1) ? 0 : c + 1;
    }
}

static void reduce_store(hls::stream<float>& partials, float sx, float* y, int M) {
rows:
    for (int m = 0; m < M; m++) {
#pragma HLS PIPELINE II = 8
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
        float v[NPS];
#pragma HLS ARRAY_PARTITION variable = v complete
        for (int p = 0; p < NPS; p++) {
#pragma HLS UNROLL
            v[p] = partials.read();
        }
        y[m] = sx * tree_sum(v);
    }
}

extern "C" void gemv_int8_ps(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                             float sx, float* y, int M, int K) {
#pragma HLS INTERFACE m_axi port = Wq bundle = gmem0 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = s bundle = gmem1 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = xq bundle = gmem2
#pragma HLS INTERFACE m_axi port = y bundle = gmem3
#pragma HLS DATAFLOW
    static_assert(NPS == 8, "the accumulate and reduce II are written as 8");

    hls::stream<float> terms("terms");
    hls::stream<float> partials("partials");
#pragma HLS STREAM variable = terms depth = 64
#pragma HLS STREAM variable = partials depth = 32

    compute_terms(Wq, s, xq, terms, M, K);
    accumulate(terms, partials, M, K);
    reduce_store(partials, sx, y, M);
}
