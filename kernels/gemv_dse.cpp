// Bit-width design-space exploration (C synthesis only, no test data):
// gemv_int8_ps generalized to WBITS-bit weights, used by scripts/dse_bits.sh.
//
//   WBITS  weight bits, 8 or 4    -> WPW = 512 / WBITS weights per 512-bit word
//   GS     group size, 64 or 128  -> GPW = WPW / GS groups (scales) per word
//   NPS    FP32 partial sums: 8 (II = 1) or 1 (one add chain, II = fadd latency)
//
// Activations stay int8, so one weight word needs WPW / 64 words of x. Layout
// (INT4): weight e of a word is bits [4e+3 : 4e], two's complement; scales of
// the GPW groups of a word are packed into one sword_t, low group first.
// WBITS = 8, GS = 64, NPS = 8 is the same design as gemv_int8_ps.
#include <hls_stream.h>

#include "gemv.h"

#ifndef WBITS
#define WBITS 8
#endif
#ifndef GS
#define GS 64
#endif
#ifndef NPS
#define NPS 8
#endif
// Loop trip count at 4096 x 4096 (weight words), for the csynth latency estimate.
#if WBITS == 8
#define TC_WORDS 262144
#define XW 1  // 512-bit x words per weight word (64 int8 activations each)
#elif WBITS == 4
#define TC_WORDS 131072
#define XW 2
#else
#error "WBITS must be 8 or 4"
#endif
#if NPS == 8
#define TC_CHUNKS (TC_WORDS / 8)
#elif NPS == 1
#define TC_CHUNKS TC_WORDS
#else
#error "NPS must be 8 or 1"
#endif

constexpr int WPW = 512 / WBITS;  // weights per word
constexpr int GPW = WPW / GS;     // groups per word
static_assert(XW == WPW / GROUP && WPW % GS == 0 && GS % GROUP == 0, "unsupported WBITS / GS");
typedef ap_uint<32 * GPW> sword_t;

static float as_float(ap_uint<32> b) {
#pragma HLS INLINE
    union {
        uint32_t u;
        float f;
    } c;
    c.u = b;
    return c.f;
}

static void compute_terms(const ap_uint<512>* Wq, const sword_t* s, const ap_uint<512>* xq,
                          hls::stream<float>& terms, int M, int K) {
    const int NX = K / GROUP;  // x words
    const int NW = K / WPW;    // weight words per row
    ap_uint<512> xbuf[MAX_K / GROUP];
#if XW > 1
#pragma HLS ARRAY_PARTITION variable = xbuf cyclic factor = XW
#endif

load_x:
    for (int g = 0; g < NX; g++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
        xbuf[g] = xq[g];
    }

    int w = 0;
words:
    for (int i = 0; i < M * NW; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = TC_WORDS
        ap_uint<512> wd = Wq[i];
        sword_t sw = s[i];
        float t[GPW];
        for (int gi = 0; gi < GPW; gi++) {
#pragma HLS UNROLL
            int32_t P = 0;
            for (int j = 0; j < GS; j++) {
#pragma HLS UNROLL
                const int e = gi * GS + j;  // weight index in the word
                ap_int<WBITS> a = wd.range(WBITS * e + WBITS - 1, WBITS * e);
                ap_int<8> b = xbuf[w * XW + e / GROUP].range(8 * (e % GROUP) + 7, 8 * (e % GROUP));
                P += (int32_t)(a * b);
            }
            t[gi] = as_float(sw.range(32 * gi + 31, 32 * gi)) * (float)P;
        }
        float term = t[0];
        for (int gi = 1; gi < GPW; gi++) term = term + t[gi];
        terms.write(term);
        w = (w == NW - 1) ? 0 : w + 1;
    }
}

static void accumulate(hls::stream<float>& terms, hls::stream<float>& partials, int M, int K) {
    const int NW = K / WPW;
    const int C = (NW + NPS - 1) / NPS;  // chunks of NPS words per row
    float acc[NPS];
#pragma HLS ARRAY_PARTITION variable = acc complete

    int c = 0;
chunks:
    for (int i = 0; i < M * C; i++) {
#pragma HLS PIPELINE II = NPS
#pragma HLS LOOP_TRIPCOUNT min = 1 max = TC_CHUNKS
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
#pragma HLS PIPELINE II = NPS
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

extern "C" void gemv_dse(const ap_uint<512>* Wq, const sword_t* s, const ap_uint<512>* xq,
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
