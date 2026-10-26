// XRT host program for the GEMV kernels.
//
//   gemv_host <xclbin> <gemv_fp32|gemv_int8> <case_dir> [iters]
//
// 1. Loads the test case (data/FORMAT.md) and the xclbin.
// 2. Allocates one buffer per kernel argument in the HBM bank the argument is
//    connected to (kernel.group_id), copies the inputs over PCIe.
// 3. Runs the kernel once as a warm-up, then `iters` times, timing each run
//    (kernel time only, no PCIe copies).
// 4. Reads y back, compares it with the golden output and writes it to
//    build/y_out.f32.bin (for scripts/check.py).
// 5. Runs the same computation in plain C++ on the host CPU (single thread, same
//    order of operations) as the software version and CPU baseline.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "binio.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------- CPU versions

static void cpu_fp32(const std::vector<float>& W, const std::vector<float>& x,
                     std::vector<float>& y, int M, int K) {
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            volatile float prod = W[(size_t)m * K + k] * x[k];  // no FMA contraction
            acc = acc + prod;
        }
        y[m] = acc;
    }
}

static void cpu_int8(const std::vector<int8_t>& Wq, const std::vector<float>& s,
                     const std::vector<int8_t>& xq, float sx, std::vector<float>& y, int M,
                     int K) {
    const int G = K / 64;
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        for (int g = 0; g < G; g++) {
            int32_t P = 0;
            for (int j = 0; j < 64; j++)
                P += (int32_t)Wq[(size_t)m * K + g * 64 + j] * (int32_t)xq[g * 64 + j];
            volatile float term = s[(size_t)m * G + g] * (float)P;
            acc = acc + term;
        }
        y[m] = sx * acc;
    }
}

// ---------------------------------------------------------------- timing helper

struct Timing {
    double avg_ms, min_ms, max_ms;
};

template <typename F>
static Timing time_runs(F run_once, int iters) {
    run_once();  // warm-up
    std::vector<double> t;
    for (int i = 0; i < iters; i++) {
        auto t0 = Clock::now();
        run_once();
        t.push_back(ms_since(t0));
    }
    double sum = 0;
    for (double v : t) sum += v;
    return {sum / iters, *std::min_element(t.begin(), t.end()),
            *std::max_element(t.begin(), t.end())};
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: %s <xclbin> <gemv_fp32|gemv_int8> <case_dir> [iters]\n", argv[0]);
        return 1;
    }
    const std::string xclbin = argv[1], kname = argv[2], dir = argv[3];
    const int iters = argc > 4 ? std::max(1, std::atoi(argv[4])) : 1;
    const bool is_int8 = kname == "gemv_int8";
    if (!is_int8 && kname != "gemv_fp32") {
        std::printf("unknown kernel %s\n", kname.c_str());
        return 1;
    }

    CaseMeta meta = read_meta(dir);
    const int M = meta.M, K = meta.K, G = K / 64;
    std::printf("case %s: M=%d K=%d kind=%s seed=%d, kernel %s, %d timed run(s)\n",
                dir.c_str(), M, K, meta.kind.c_str(), meta.seed, kname.c_str(), iters);

    std::vector<float> golden =
        read_bin<float>(dir + (is_int8 ? "/y_int8.f32.bin" : "/y_fp32.f32.bin"), M);
    std::vector<float> y(M, 0.0f), y_cpu(M, 0.0f);

    // ---- device setup
    auto t0 = Clock::now();
    xrt::device device(0);
    auto uuid = device.load_xclbin(xclbin);
    xrt::kernel krnl(device, uuid, kname);
    std::printf("device: %s, xclbin loaded in %.1f ms\n",
                device.get_info<xrt::info::device::name>().c_str(), ms_since(t0));

    double bytes_read = 0, h2d_ms = 0, d2h_ms = 0, cpu_ms = 0;
    Timing kt{};

    if (!is_int8) {
        std::vector<float> W = read_bin<float>(dir + "/W.f32.bin", (size_t)M * K);
        std::vector<float> x = read_bin<float>(dir + "/x.f32.bin", K);

        xrt::bo bo_W(device, (size_t)M * K * sizeof(float), krnl.group_id(0));
        xrt::bo bo_x(device, (size_t)K * sizeof(float), krnl.group_id(1));
        xrt::bo bo_y(device, (size_t)M * sizeof(float), krnl.group_id(2));

        t0 = Clock::now();
        bo_W.write(W.data());
        bo_x.write(x.data());
        bo_W.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        h2d_ms = ms_since(t0);

        kt = time_runs([&] { krnl(bo_W, bo_x, bo_y, M, K).wait(); }, iters);

        t0 = Clock::now();
        bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_y.read(y.data());
        d2h_ms = ms_since(t0);

        bytes_read = (double)M * K * 4 + K * 4;
        t0 = Clock::now();
        cpu_fp32(W, x, y_cpu, M, K);
        cpu_ms = ms_since(t0);
    } else {
        std::vector<int8_t> Wq = read_bin<int8_t>(dir + "/Wq.i8.bin", (size_t)M * K);
        std::vector<float> s = read_bin<float>(dir + "/s.f32.bin", (size_t)M * G);
        std::vector<int8_t> xq = read_bin<int8_t>(dir + "/xq.i8.bin", K);
        float sx = read_bin<float>(dir + "/sx.f32.bin", 1)[0];

        xrt::bo bo_Wq(device, (size_t)M * K, krnl.group_id(0));
        xrt::bo bo_s(device, (size_t)M * G * sizeof(float), krnl.group_id(1));
        xrt::bo bo_xq(device, (size_t)K, krnl.group_id(2));
        xrt::bo bo_y(device, (size_t)M * sizeof(float), krnl.group_id(4));

        t0 = Clock::now();
        bo_Wq.write(Wq.data());
        bo_s.write(s.data());
        bo_xq.write(xq.data());
        bo_Wq.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_s.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_xq.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        h2d_ms = ms_since(t0);

        kt = time_runs([&] { krnl(bo_Wq, bo_s, bo_xq, sx, bo_y, M, K).wait(); }, iters);

        t0 = Clock::now();
        bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_y.read(y.data());
        d2h_ms = ms_since(t0);

        bytes_read = (double)M * K + (double)M * G * 4 + K;
        t0 = Clock::now();
        cpu_int8(Wq, s, xq, sx, y_cpu, M, K);
        cpu_ms = ms_since(t0);
    }

    write_bin("build/y_out.f32.bin", y.data(), M);

    // ---- results
    std::printf("\nkernel time   avg %.3f ms  min %.3f ms  max %.3f ms  (%d runs)\n", kt.avg_ms,
                kt.min_ms, kt.max_ms, iters);
    std::printf("bytes read    %.1f MiB\n", bytes_read / (1 << 20));
    std::printf("bandwidth     %.2f GB/s (bytes read / avg kernel time)\n",
                bytes_read / (kt.avg_ms * 1e-3) / 1e9);
    std::printf("PCIe          host->device %.2f ms, device->host %.3f ms\n", h2d_ms, d2h_ms);
    std::printf("CPU (1 thread, same op order) %.3f ms\n", cpu_ms);

    const float tol = is_int8 ? 0.0f : 1e-6f;
    std::printf("\nFPGA vs golden:\n");
    int bad = compare_f32(y.data(), golden.data(), M, tol);
    std::printf("CPU  vs golden:\n");
    int bad_cpu = compare_f32(y_cpu.data(), golden.data(), M, tol);

    std::printf(bad == 0 ? "TEST PASSED\n" : "TEST FAILED\n");
    if (bad_cpu) std::printf("note: CPU version does not match golden either\n");
    return bad == 0 ? 0 : 1;
}
