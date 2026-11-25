// CPU baseline for the same GEMV, multi-threaded (OpenMP), with Intel RAPL energy.
// Fair-comparison helper for the U280 results: same data files, same INT8 math as the kernel.
//
// build: g++ -O3 -march=native -fopenmp -o cpu_bench host/cpu_bench.cpp
// run  : sudo ./cpu_bench <case_dir> <fp32|int8> <threads> <iters> [cold]
//   cold = rotate through enough copies of W (>= 512 MiB) so every GEMV streams W from DRAM,
//          like one LLM decode step where the weights do not fit in cache.
// RAPL : sums /sys/class/powercap/intel-rapl:{0,1} (package) and :{0,1}:0 (dram); needs root.
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const int G = 64;

template <class T>
static std::vector<T> load(const std::string& p, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
  return v;
}

static double rapl_joules() {  // package + dram, both sockets
  const char* z[] = {"intel-rapl:0", "intel-rapl:1", "intel-rapl:0/intel-rapl:0:0", "intel-rapl:1/intel-rapl:1:0"};
  double s = 0;
  for (auto d : z) {
    std::ifstream f(std::string("/sys/class/powercap/") + d + "/energy_uj");
    unsigned long long u = 0;
    if (f >> u) s += u * 1e-6;
  }
  return s;
}

static void gemv_fp32(const float* W, const float* x, float* y, int M, int K) {
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; m++) {
    const float* w = W + (size_t)m * K;
    float acc = 0.f;
#pragma omp simd reduction(+ : acc)
    for (int k = 0; k < K; k++) acc += w[k] * x[k];
    y[m] = acc;
  }
}

// same math as the INT8 kernel: exact int32 sum per 64-group, then acc += s[m,g] * P in group order
static void gemv_int8(const int8_t* Wq, const float* s, const int8_t* xq, float sx, float* y, int M, int K) {
  const int NG = K / G;
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; m++) {
    const int8_t* w = Wq + (size_t)m * K;
    float acc = 0.f;
    for (int g = 0; g < NG; g++) {
      int32_t p = 0;
#pragma omp simd reduction(+ : p)
      for (int i = 0; i < G; i++) p += (int32_t)w[g * G + i] * (int32_t)xq[g * G + i];
      acc += s[(size_t)m * NG + g] * (float)p;
    }
    y[m] = sx * acc;
  }
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s case_dir fp32|int8 threads iters [cold]\n", argv[0]); return 1; }
  std::string dir = argv[1], kind = argv[2];
  int th = atoi(argv[3]), iters = atoi(argv[4]);
  bool cold = argc > 5 && std::string(argv[5]) == "cold";
  int M = 0, K = 0;
  { std::ifstream f(dir + "/meta.txt"); std::string k; int v; while (f >> k >> v) { if (k == "M") M = v; if (k == "K") K = v; if (k == "kind") break; } }
  omp_set_num_threads(th);
  const bool i8 = kind == "int8";
  const size_t wbytes = (size_t)M * K * (i8 ? 1 : 4);
  const int copies = cold ? (int)std::max<size_t>(1, (512ull << 20) / wbytes + 1) : 1;

  auto Wf = i8 ? std::vector<float>() : load<float>(dir + "/W.f32.bin", (size_t)M * K);
  auto x = load<float>(dir + "/x.f32.bin", K);
  auto Wq = i8 ? load<int8_t>(dir + "/Wq.i8.bin", (size_t)M * K) : std::vector<int8_t>();
  auto s = i8 ? load<float>(dir + "/s.f32.bin", (size_t)M * K / G) : std::vector<float>();
  auto xq = i8 ? load<int8_t>(dir + "/xq.i8.bin", K) : std::vector<int8_t>();
  float sx = i8 ? load<float>(dir + "/sx.f32.bin", 1)[0] : 0.f;
  auto gold = load<float>(dir + (i8 ? "/y_int8.f32.bin" : "/y_fp32.f32.bin"), M);
  auto yref = load<double>(dir + "/y_ref.f64.bin", M);

  // copies are first-touched inside the parallel region so pages spread over both NUMA nodes
  std::vector<char*> buf(copies);
  for (int c = 0; c < copies; c++) {
    buf[c] = static_cast<char*>(aligned_alloc(64, (wbytes + 63) / 64 * 64));
    const char* src = i8 ? reinterpret_cast<const char*>(Wq.data()) : reinterpret_cast<const char*>(Wf.data());
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
      size_t row = wbytes / M;
      memcpy(buf[c] + m * row, src + m * row, row);
    }
  }
  std::vector<float> y(M);
  auto run = [&](int c) {
    if (i8) gemv_int8(reinterpret_cast<int8_t*>(buf[c]), s.data(), xq.data(), sx, y.data(), M, K);
    else gemv_fp32(reinterpret_cast<float*>(buf[c]), x.data(), y.data(), M, K);
  };
  for (int c = 0; c < copies; c++) run(c);  // warm-up (page faults, thread pool)

  double e0 = rapl_joules();
  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; it++) run(it % copies);
  auto t1 = std::chrono::steady_clock::now();
  double e1 = rapl_joules();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  double joule = (e1 - e0) / iters;

  double maxd = 0, sc = 0, num = 0, den = 0;
  for (int m = 0; m < M; m++) sc = std::max(sc, std::fabs((double)gold[m]));
  for (int m = 0; m < M; m++) {
    maxd = std::max(maxd, std::fabs((double)y[m] - gold[m]));
    num += (y[m] - yref[m]) * (y[m] - yref[m]);
    den += yref[m] * yref[m];
  }
  printf("cpu,%s,%s,M=%d,K=%d,threads=%d,iters=%d,%s,copies=%d,avg_ms=%.4f,GBs=%.2f,rapl_J_per_gemv=%.5f,rapl_W=%.1f,"
         "maxdiff_vs_golden=%.3g,relL2_vs_f64=%.4g\n",
         dir.c_str(), kind.c_str(), M, K, th, iters, cold ? "cold" : "warm", copies, ms, wbytes / ms / 1e6, joule,
         joule / (ms / 1e3), maxd / (sc > 0 ? sc : 1), std::sqrt(num / (den > 0 ? den : 1)));
  return 0;
}
