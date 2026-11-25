// L3c: group-wise INT4 GEMV, one 512-bit word (128 weights = 2 groups) per cycle.
//
// Weights: symmetric int4 in [-7, 7], one FP32 scale per group of 64 (s4), packed
// two per byte: weight e of a word is bits [4e+3 : 4e] (gen_data.py pack_int4).
// Activations stay int8 (xq, sx), so one weight word meets two 512-bit x words.
// Per word j of a row: P0, P1 = exact int32 dot products of groups 2j and 2j+1,
//   w = s4[2j] * float(P0) + s4[2j+1] * float(P1)     (two fmul, then one fadd),
// then the same partial-sum scheme as gemv_int8_ps over words: a[j % 8] += w,
// tree ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)), y = sx * sum.
// This is golden_int4_ps() in scripts/gen_data.py (golden file y_int4.f32.bin).
// Same structure as gemv_dse.cpp with WBITS = 4, GS = 64, NPS = 8 (csynth:
// 131,248 cycles at 4096 x 4096 = half of the INT8 kernel).
//
// The s argument is the float array s4[m][g]; it is read as 64-bit words, word j
// = {s4[2j+1], s4[2j]} (high, low), so both scales of a weight word arrive in one
// cycle. Requires K to be a multiple of 128.
#include <hls_stream.h>

#include "gemv.h"

constexpr int NPS = 8;        // partial sums (>= FP32 add latency 7)
constexpr int WPW = 128;      // int4 weights per 512-bit word
constexpr int GPW = 2;        // groups per word

static float as_float(ap_uint<32> b) {
#pragma HLS INLINE
    union {
        uint32_t u;
        float f;
    } c;
    c.u = b;
    return c.f;
}

static void compute_terms(const ap_uint<512>* Wq, const ap_uint<64>* s, const ap_uint<512>* xq,
                          hls::stream<float>& terms, int M, int K) {
    const int NX = K / GROUP;  // x words
    const int NW = K / WPW;    // weight words per row
    ap_uint<512> xbuf[MAX_K / GROUP];
#pragma HLS ARRAY_PARTITION variable = xbuf cyclic factor = 2

load_x:
    for (int g = 0; g < NX; g++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 64
        xbuf[g] = xq[g];
    }

    int w = 0;
words:
    for (int i = 0; i < M * NW; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 131072
        ap_uint<512> wd = Wq[i];
        ap_uint<64> sw = s[i];
        float t[GPW];
        for (int h = 0; h < GPW; h++) {
#pragma HLS UNROLL
            ap_uint<512> xw = xbuf[GPW * w + h];
            int32_t P = 0;
            for (int j = 0; j < GROUP; j++) {
#pragma HLS UNROLL
                const int e = h * GROUP + j;  // weight index in the word
                ap_int<4> a = wd.range(4 * e + 3, 4 * e);
                ap_int<8> b = xw.range(8 * j + 7, 8 * j);
                P += (int32_t)(a * b);
            }
            t[h] = as_float(sw.range(32 * h + 31, 32 * h)) * (float)P;
        }
        terms.write(t[0] + t[1]);
        w = (w == NW - 1) ? 0 : w + 1;
    }
}

// Same as accumulate() in gemv_int8_ps.cpp, over words instead of groups.
static void accumulate(hls::stream<float>& terms, hls::stream<float>& partials, int M, int K) {
    const int NW = K / WPW;
    const int C = (NW + NPS - 1) / NPS;
    float acc[NPS];
#pragma HLS ARRAY_PARTITION variable = acc complete

    int c = 0;
chunks:
    for (int i = 0; i < M * C; i++) {
#pragma HLS PIPELINE II = 8
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16384
        for (int p = 0; p < NPS; p++) {
#pragma HLS UNROLL
            float a = (c == 0) ? 0.0f : acc[p];
            if (c * NPS + p < NW) a = a + terms.read();
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
        for (int w = NPS / 2; w >= 1; w /= 2) {
#pragma HLS UNROLL
            for (int j = 0; j < w; j++) {
#pragma HLS UNROLL
                v[j] = v[2 * j] + v[2 * j + 1];
            }
        }
        y[m] = sx * v[0];
    }
}

extern "C" void gemv_int4_ps(const ap_uint<512>* Wq, const ap_uint<64>* s, const ap_uint<512>* xq,
                             float sx, float* y, int M, int K) {
#pragma HLS INTERFACE m_axi port = Wq bundle = gmem0 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = s bundle = gmem1 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = xq bundle = gmem2
#pragma HLS INTERFACE m_axi port = y bundle = gmem3
#pragma HLS DATAFLOW

    hls::stream<float> terms("terms");
    hls::stream<float> partials("partials");
#pragma HLS STREAM variable = terms depth = 64
#pragma HLS STREAM variable = partials depth = 32

    compute_terms(Wq, s, xq, terms, M, K);
    accumulate(terms, partials, M, K);
    reduce_store(partials, sx, y, M);
}
