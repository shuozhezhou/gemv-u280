// Vitis HLS C testbench for gemv_int8.
// Usage: tb_gemv_int8 <case_dir> [<case_dir> ...]   (passed by scripts/hls.tcl)
// Each case: pack the int8 files into 512-bit words, run the kernel, and require
// a bit-exact match with y_int8.f32.bin.
#include <cstdio>
#include <string>
#include <vector>

#include "binio.h"
#include "gemv.h"

// Packs n int8 values (n a multiple of 64) into 512-bit words, byte j of a word
// = element j, the same layout the host gets by copying the int8 array.
static std::vector<ap_uint<512>> pack(const std::vector<int8_t>& v) {
    std::vector<ap_uint<512>> words(v.size() / GROUP);
    for (size_t w = 0; w < words.size(); w++)
        for (int j = 0; j < GROUP; j++)
            words[w].range(8 * j + 7, 8 * j) = (uint8_t)v[w * GROUP + j];
    return words;
}

static int run_case(const std::string& dir) {
    CaseMeta meta = read_meta(dir);
    const int M = meta.M, K = meta.K, G = K / GROUP;
    std::printf("case %s (M=%d, K=%d)\n", dir.c_str(), M, K);
    if (K > MAX_K || K % GROUP != 0) {
        std::printf("  unsupported K=%d\n", K);
        return 1;
    }
    std::vector<int8_t> Wq = read_bin<int8_t>(dir + "/Wq.i8.bin", (size_t)M * K);
    std::vector<float> s = read_bin<float>(dir + "/s.f32.bin", (size_t)M * G);
    std::vector<int8_t> xq = read_bin<int8_t>(dir + "/xq.i8.bin", K);
    float sx = read_bin<float>(dir + "/sx.f32.bin", 1)[0];
    std::vector<float> golden = read_bin<float>(dir + "/y_int8.f32.bin", M);
    std::vector<float> y(M, -1.0f);

    std::vector<ap_uint<512>> Wq_w = pack(Wq), xq_w = pack(xq);
    gemv_int8(Wq_w.data(), s.data(), xq_w.data(), sx, y.data(), M, K);

    return compare_f32(y.data(), golden.data(), M, 0.0f);
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
