// Vitis HLS C testbench for gemv_fp32.
// Usage: tb_gemv_fp32 <case_dir> [<case_dir> ...]   (passed by scripts/hls.tcl)
// Each case: run the kernel, compare with y_fp32.f32.bin.
// Pass rule: |y - golden| <= 1e-6 * max|golden|; the number of bit-exact values
// is printed as well (all of them are expected to be, since the op order matches).
#include <cstdio>
#include <string>
#include <vector>

#include "binio.h"
#include "gemv.h"

static int run_case(const std::string& dir) {
    CaseMeta meta = read_meta(dir);
    const int M = meta.M, K = meta.K;
    std::printf("case %s (M=%d, K=%d)\n", dir.c_str(), M, K);
    if (K > MAX_K) {
        std::printf("  K=%d exceeds MAX_K=%d\n", K, MAX_K);
        return 1;
    }
    std::vector<float> W = read_bin<float>(dir + "/W.f32.bin", (size_t)M * K);
    std::vector<float> x = read_bin<float>(dir + "/x.f32.bin", K);
    std::vector<float> golden = read_bin<float>(dir + "/y_fp32.f32.bin", M);
    std::vector<float> y(M, -1.0f);

    gemv_fp32(W.data(), x.data(), y.data(), M, K);

    return compare_f32(y.data(), golden.data(), M, 1e-6f);
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
