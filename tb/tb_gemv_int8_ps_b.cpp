// Vitis HLS C testbench for gemv_int8_ps_b (batched INT8 GEMV).
// Usage: tb_gemv_int8_ps_b <case_dir> [<case_dir> ...]   (passed by scripts/hls.tcl)
// Each case is run with nb = BMAX, nb = 3 and nb = 1 activation vectors (xqB.i8.bin,
// sxB.f32.bin); every output vector y[b*M .. b*M+M-1] must be bit-exact to row b of
// y_int8psB.f32.bin (= golden_int8_ps of vector b), and nothing past nb*M is written.
#include <cstdio>
#include <string>
#include <vector>

#include "binio.h"
#include "gemv.h"

#ifndef BMAX
#define BMAX 8
#endif
constexpr int BATCH = 8;  // vectors stored per case by gen_data.py

static std::vector<ap_uint<512>> pack(const int8_t* v, size_t n) {
    std::vector<ap_uint<512>> words(n / GROUP);
    for (size_t w = 0; w < words.size(); w++)
        for (int j = 0; j < GROUP; j++)
            words[w].range(8 * j + 7, 8 * j) = (uint8_t)v[w * GROUP + j];
    return words;
}

static int run_case(const std::string& dir) {
    CaseMeta meta = read_meta(dir);
    const int M = meta.M, K = meta.K, G = K / GROUP;
    std::printf("case %s (M=%d, K=%d, BMAX=%d)\n", dir.c_str(), M, K, BMAX);
    if (K > MAX_K || K % GROUP != 0) {
        std::printf("  unsupported K=%d\n", K);
        return 1;
    }
    std::vector<int8_t> Wq = read_bin<int8_t>(dir + "/Wq.i8.bin", (size_t)M * K);
    std::vector<float> s = read_bin<float>(dir + "/s.f32.bin", (size_t)M * G);
    std::vector<int8_t> xqB = read_bin<int8_t>(dir + "/xqB.i8.bin", (size_t)BATCH * K);
    std::vector<float> sxB = read_bin<float>(dir + "/sxB.f32.bin", BATCH);
    std::vector<float> golden = read_bin<float>(dir + "/y_int8psB.f32.bin", (size_t)BATCH * M);

    std::vector<ap_uint<512>> Wq_w = pack(Wq.data(), Wq.size());
    std::vector<ap_uint<512>> xq_w = pack(xqB.data(), xqB.size());  // all 8 vectors, b-major
    int bad = 0;
    for (int nb : {BMAX, 3, 1}) {
        if (nb > BMAX) continue;
        std::vector<float> y((size_t)(BMAX + 1) * M, -1.0f);  // one extra slice: must stay -1
        gemv_int8_ps_b(Wq_w.data(), s.data(), xq_w.data(), sxB.data(), y.data(), M, K, nb);
        std::printf(" nb=%d\n", nb);
        for (int b = 0; b < nb; b++)
            bad += compare_f32(&y[(size_t)b * M], &golden[(size_t)b * M], M, 0.0f) != 0;
        for (size_t i = (size_t)nb * M; i < y.size(); i++)
            if (y[i] != -1.0f) {
                std::printf("  wrote past nb*M at %zu\n", i);
                bad++;
                break;
            }
    }
    return bad;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <case_dir> [...]\n", argv[0]);
        return 1;
    }
    int failed = 0;
    for (int i = 1; i < argc; i++) failed += run_case(argv[i]) != 0;
    std::printf(failed ? "TEST FAILED (%d case(s))\n" : "TEST PASSED\n", failed);
    return failed ? 1 : 0;
}
