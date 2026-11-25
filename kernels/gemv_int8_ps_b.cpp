// L3d: batched INT8 GEMV, Y[b] = W x_b for b < nb <= BMAX, one weight pass.
//
// Each 512-bit weight word (one group of 64 int8 weights, as in gemv_int8_ps) is
// read from HBM ONCE and multiplied with the matching group of all BMAX activation
// vectors in the same cycle (BMAX x 64 int8 multipliers), so one pass over W
// produces nb outputs. Per vector the math and the summation order are exactly
// those of gemv_int8_ps: t_b = s * float(P_b), 8 partial sums a_b[g % 8], the tree
// ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)), y_b = sx_b * sum. So Y[b] is bit-exact to
// golden_int8_ps(Wq, s, xq_b, sx_b) (gen_data.py: y_int8psB.f32.bin, [BATCH][M]).
//
// Arguments:
//   Wq  : M*K/64 words            (gmem0)  as gemv_int8_ps
//   s   : M*K/64 floats           (gmem1)  as gemv_int8_ps
//   xq  : nb*K/64 words           (gmem2)  vector b = words b*K/64 .. (b+1)*K/64-1
//   sxv : nb floats               (gmem3)  activation scale of each vector
//   y   : nb*M floats             (gmem3)  y[b*M + m]
//   M, K, nb (1 <= nb <= BMAX; vectors b >= nb are not computed into y)
// BMAX is a compile-time constant (-DBMAX=4|8, default 8).
#include "int8_terms.h"

#ifndef BMAX
#define BMAX 8
#endif

constexpr int NPS = 8;  // partial sums per vector (>= FP32 add latency 7)

struct fvec {
    float v[BMAX];
};

static int clamp_nb(int nb) {
#pragma HLS INLINE
    return nb < 1 ? 1 : (nb > BMAX ? BMAX : nb);
}

static void compute_terms_b(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                            hls::stream<fvec>& terms, int M, int K, int nb_arg) {
    const int G = K / GROUP;
    const int nb = clamp_nb(nb_arg);
    ap_uint<512> xbuf[BMAX][MAX_K / GROUP];
#pragma HLS ARRAY_PARTITION variable = xbuf dim = 1 complete

    int b = 0, g = 0;
load_x:
    for (int i = 0; i < nb * G; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
        xbuf[b][g] = xq[i];
        if (g == G - 1) {
            g = 0;
            b++;
        } else {
            g++;
        }
    }

    g = 0;
words:
    for (int i = 0; i < M * G; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 262144
        ap_uint<512> w = Wq[i];
        float sc = s[i];
        fvec t;
        for (int v = 0; v < BMAX; v++) {
#pragma HLS UNROLL
            t.v[v] = sc * (float)group_dot(w, xbuf[v][g]);
        }
        terms.write(t);
        g = (g == G - 1) ? 0 : g + 1;
    }
}

// accumulate() of gemv_int8_ps, for BMAX vectors side by side.
static void accumulate_b(hls::stream<fvec>& terms, hls::stream<fvec>& partials, int M, int K) {
    const int G = K / GROUP;
    const int C = (G + NPS - 1) / NPS;
    float acc[BMAX][NPS];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0

    int c = 0;
chunks:
    for (int i = 0; i < M * C; i++) {
#pragma HLS PIPELINE II = 8
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32768
        for (int p = 0; p < NPS; p++) {
#pragma HLS UNROLL
            const bool take = c * NPS + p < G;
            fvec t, o;
            if (take) t = terms.read();
            for (int v = 0; v < BMAX; v++) {
#pragma HLS UNROLL
                float a = (c == 0) ? 0.0f : acc[v][p];
                if (take) a = a + t.v[v];
                acc[v][p] = a;
                o.v[v] = a;
            }
            if (c == C - 1) partials.write(o);
        }
        c = (c == C - 1) ? 0 : c + 1;
    }
}

static void reduce_store_b(hls::stream<fvec>& partials, const float* sxv, float* y, int M,
                           int nb_arg) {
    const int nb = clamp_nb(nb_arg);
    float sx[BMAX];
#pragma HLS ARRAY_PARTITION variable = sx complete
load_sx:
    for (int v = 0; v < BMAX; v++) {
#pragma HLS PIPELINE II = 1
        sx[v] = (v < nb) ? sxv[v] : 0.0f;
    }

rows:
    for (int m = 0; m < M; m++) {
#pragma HLS PIPELINE II = 8
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
        fvec part[NPS];
#pragma HLS ARRAY_PARTITION variable = part complete dim = 0
        for (int p = 0; p < NPS; p++) {
#pragma HLS UNROLL
            part[p] = partials.read();
        }
        for (int v = 0; v < BMAX; v++) {
#pragma HLS UNROLL
            float u[NPS];
#pragma HLS ARRAY_PARTITION variable = u complete
            for (int p = 0; p < NPS; p++) {
#pragma HLS UNROLL
                u[p] = part[p].v[v];
            }
            for (int w = NPS / 2; w >= 1; w /= 2) {
#pragma HLS UNROLL
                for (int j = 0; j < w; j++) {
#pragma HLS UNROLL
                    u[j] = u[2 * j] + u[2 * j + 1];
                }
            }
            if (v < nb) y[(long)v * M + m] = sx[v] * u[0];
        }
    }
}

extern "C" void gemv_int8_ps_b(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                               const float* sxv, float* y, int M, int K, int nb) {
#pragma HLS INTERFACE m_axi port = Wq bundle = gmem0 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = s bundle = gmem1 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = xq bundle = gmem2
#pragma HLS INTERFACE m_axi port = sxv bundle = gmem3
#pragma HLS INTERFACE m_axi port = y bundle = gmem3 num_write_outstanding = 32
#pragma HLS DATAFLOW

    hls::stream<fvec> terms("terms");
    hls::stream<fvec> partials("partials");
#pragma HLS STREAM variable = terms depth = 64
#pragma HLS STREAM variable = partials depth = 32

    compute_terms_b(Wq, s, xq, terms, M, K, nb);
    accumulate_b(terms, partials, M, K);
    reduce_store_b(partials, sxv, y, M, nb);
}
