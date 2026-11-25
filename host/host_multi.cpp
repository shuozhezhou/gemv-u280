// XRT host for N compute units of one GEMV kernel (multi-HBM scaling).
//
//   gemv_host_multi <xclbin> <gemv_int8_ps|gemv_int4_ps|gemv_int8_ps_b> <N> <case_dir>
//                   [iters] [nb]
//
// The xclbin holds N CUs <kernel>_1 .. <kernel>_N (kernels/<kernel>_x<N>.cfg), each
// with its own HBM pseudo-channel for Wq. The M rows are split into N contiguous
// slices (the first M % N slices get one row more). CU i gets its own Wq, s, xq
// and y buffers, allocated in the banks its ports are connected to (group_id).
// Every timed iteration starts all N runs, then waits for all of them, so the
// time is the wall time of one full GEMV on N CUs. Rows are independent, so the
// concatenated y must match the single-CU golden (y_int8ps / y_int4) bit for bit.
//
// Batch mode (gemv_int8_ps_b, nb vectors, default 8): every CU gets all nb vectors
// (xqB.i8.bin, sxB.f32.bin) and writes y[b * rows + r]; the host gathers Y[b][m] and
// checks each vector against row b of y_int8psB.f32.bin. Reports time per batch and
// per vector; weights are read once per batch.
#include <algorithm>
#include <chrono>
#include <cstdio>
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

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("usage: %s <xclbin> <gemv_int8_ps|gemv_int4_ps|gemv_int8_ps_b> <N> <case_dir> "
                    "[iters] [nb]\n",
                    argv[0]);
        return 1;
    }
    const std::string xclbin = argv[1], kname = argv[2], dir = argv[4];
    const int N = std::atoi(argv[3]);
    const int iters = argc > 5 ? std::max(1, std::atoi(argv[5])) : 1;
    const bool int4 = kname == "gemv_int4_ps";
    const bool batch = kname == "gemv_int8_ps_b";
    const int BATCH = 8;  // vectors stored per case (gen_data.py)
    const int nb = batch ? std::min(BATCH, std::max(1, argc > 6 ? std::atoi(argv[6]) : BATCH)) : 1;
    if (!int4 && !batch && kname != "gemv_int8_ps") {
        std::printf("unknown kernel %s\n", kname.c_str());
        return 1;
    }

    CaseMeta meta = read_meta(dir);
    const int M = meta.M, K = meta.K, G = K / 64;
    if (N < 1 || N > M) {
        std::printf("need 1 <= N <= M\n");
        return 1;
    }
    const size_t row_bytes = int4 ? K / 2 : K;  // weight bytes per row
    std::printf("case %s: M=%d K=%d kind=%s seed=%d, kernel %s x %d CU(s), %d timed run(s)%s\n",
                dir.c_str(), M, K, meta.kind.c_str(), meta.seed, kname.c_str(), N, iters,
                batch ? (", batch nb = " + std::to_string(nb)).c_str() : "");

    std::vector<int8_t> Wq =
        read_bin<int8_t>(dir + (int4 ? "/Wq4.i4.bin" : "/Wq.i8.bin"), (size_t)M * row_bytes);
    std::vector<float> s = read_bin<float>(dir + (int4 ? "/s4.f32.bin" : "/s.f32.bin"), (size_t)M * G);
    std::vector<int8_t> xq = batch ? read_bin<int8_t>(dir + "/xqB.i8.bin", (size_t)BATCH * K)
                                   : read_bin<int8_t>(dir + "/xq.i8.bin", K);
    std::vector<float> sxv = batch ? read_bin<float>(dir + "/sxB.f32.bin", BATCH)
                                   : read_bin<float>(dir + "/sx.f32.bin", 1);
    const float sx = sxv[0];
    const char* gold = batch ? "/y_int8psB.f32.bin" : int4 ? "/y_int4.f32.bin" : "/y_int8ps.f32.bin";
    std::vector<float> golden = read_bin<float>(dir + gold, (size_t)(batch ? BATCH : 1) * M);
    std::vector<float> y((size_t)nb * M, 0.0f);

    // ---- device setup
    auto t0 = Clock::now();
    xrt::device device(0);
    auto uuid = device.load_xclbin(xclbin);
    std::printf("device: %s, xclbin loaded in %.1f ms\n",
                device.get_info<xrt::info::device::name>().c_str(), ms_since(t0));

    std::vector<int> rows(N), row0(N);
    for (int i = 0, r = 0; i < N; i++) {
        rows[i] = M / N + (i < M % N ? 1 : 0);
        row0[i] = r;
        r += rows[i];
    }

    std::vector<xrt::kernel> krnl;
    std::vector<xrt::bo> bo_Wq, bo_s, bo_xq, bo_sx, bo_y;
    // argument indices: batch kernel (Wq, s, xq, sxv, y, M, K, nb), others (Wq, s, xq, sx, y, M, K)
    for (int i = 0; i < N; i++) {
        const std::string cu = kname + ":{" + kname + "_" + std::to_string(i + 1) + "}";
        krnl.emplace_back(device, uuid, cu);
        xrt::kernel& k = krnl.back();
        bo_Wq.emplace_back(device, rows[i] * row_bytes, k.group_id(0));
        bo_s.emplace_back(device, (size_t)rows[i] * G * sizeof(float), k.group_id(1));
        bo_xq.emplace_back(device, (size_t)nb * K, k.group_id(2));
        if (batch) bo_sx.emplace_back(device, (size_t)BATCH * sizeof(float), k.group_id(3));
        bo_y.emplace_back(device, (size_t)nb * rows[i] * sizeof(float), k.group_id(4));
        std::printf("CU %d: rows %d..%d, banks Wq %d, s %d, xq %d, y %d\n", i + 1, row0[i],
                    row0[i] + rows[i] - 1, k.group_id(0), k.group_id(1), k.group_id(2),
                    k.group_id(4));
    }

    t0 = Clock::now();
    for (int i = 0; i < N; i++) {
        bo_Wq[i].write(Wq.data() + (size_t)row0[i] * row_bytes);
        bo_s[i].write(s.data() + (size_t)row0[i] * G);
        bo_xq[i].write(xq.data());  // batch: the first nb vectors (b-major)
        if (batch) {
            bo_sx[i].write(sxv.data());
            bo_sx[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        bo_Wq[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_s[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_xq[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    const double h2d_ms = ms_since(t0);

    auto run_all = [&] {
        std::vector<xrt::run> runs;
        for (int i = 0; i < N; i++)
            runs.push_back(batch ? krnl[i](bo_Wq[i], bo_s[i], bo_xq[i], bo_sx[i], bo_y[i], rows[i],
                                           K, nb)
                                 : krnl[i](bo_Wq[i], bo_s[i], bo_xq[i], sx, bo_y[i], rows[i], K));
        for (auto& r : runs) r.wait();
    };
    run_all();  // warm-up
    std::vector<double> t;
    for (int it = 0; it < iters; it++) {
        auto t1 = Clock::now();
        run_all();
        t.push_back(ms_since(t1));
    }
    double avg = 0;
    for (double v : t) avg += v;
    avg /= iters;

    for (int i = 0; i < N; i++) {
        bo_y[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::vector<float> part((size_t)nb * rows[i]);
        bo_y[i].read(part.data());
        for (int b = 0; b < nb; b++)  // CU layout y[b * rows + r] -> Y[b][row0 + r]
            std::copy(part.begin() + (size_t)b * rows[i], part.begin() + (size_t)(b + 1) * rows[i],
                      y.begin() + (size_t)b * M + row0[i]);
    }
    write_bin("build/y_out.f32.bin", y.data(), (size_t)nb * M);

    const double bytes_read = (double)M * row_bytes + (double)M * G * 4 + (double)N * nb * K;
    std::printf("\nkernel time   avg %.3f ms  min %.3f ms  max %.3f ms  (%d runs, %d CUs)\n", avg,
                *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()),
                iters, N);
    std::printf("bytes read    %.1f MiB\n", bytes_read / (1 << 20));
    std::printf("bandwidth     %.2f GB/s aggregate (bytes read / avg time), %.2f GB/s per CU\n",
                bytes_read / (avg * 1e-3) / 1e9, bytes_read / (avg * 1e-3) / 1e9 / N);
    std::printf("PCIe          host->device %.2f ms\n", h2d_ms);
    if (batch)
        std::printf("batch         %d vectors: %.3f ms per batch, %.3f ms per vector, "
                    "%.2f GMAC/s\n",
                    nb, avg, avg / nb, (double)nb * M * K / (avg * 1e-3) / 1e9);

    std::printf("\nFPGA vs golden:\n");
    int bad = 0;
    for (int b = 0; b < nb; b++)
        bad += compare_f32(&y[(size_t)b * M], &golden[(size_t)b * M], M, 0.0f);
    std::printf(bad == 0 ? "TEST PASSED\n" : "TEST FAILED\n");
    return bad == 0 ? 0 : 1;
}
