// Shared I/O helpers for the HLS testbenches and the XRT host.
// File format: see data/FORMAT.md.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct CaseMeta {
    int M = 0;
    int K = 0;
    int group = 64;
    int seed = 0;
    std::string kind;
};

inline CaseMeta read_meta(const std::string& dir) {
    std::ifstream f(dir + "/meta.txt");
    if (!f) {
        std::fprintf(stderr, "cannot open %s/meta.txt\n", dir.c_str());
        std::exit(1);
    }
    std::map<std::string, std::string> kv;
    std::string key, val;
    while (f >> key >> val) kv[key] = val;
    CaseMeta m;
    m.M = std::stoi(kv["M"]);
    m.K = std::stoi(kv["K"]);
    if (kv.count("group")) m.group = std::stoi(kv["group"]);
    if (kv.count("seed")) m.seed = std::stoi(kv["seed"]);
    if (kv.count("kind")) m.kind = kv["kind"];
    return m;
}

// Reads exactly n elements of T; exits on size mismatch.
template <typename T>
std::vector<T> read_bin(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    size_t bytes = static_cast<size_t>(f.tellg());
    if (bytes != n * sizeof(T)) {
        std::fprintf(stderr, "%s: expected %zu bytes, found %zu\n", path.c_str(),
                     n * sizeof(T), bytes);
        std::exit(1);
    }
    std::vector<T> v(n);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

template <typename T>
void write_bin(const std::string& path, const T* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), n * sizeof(T));
}

// Compares out against golden. tol = 0 means bit-exact.
// Prints the first few mismatches and returns their count.
inline int compare_f32(const float* out, const float* golden, int n, float tol,
                       int max_print = 10) {
    int bad = 0;
    float worst = 0.0f;
    for (int i = 0; i < n; i++) {
        bool ok;
        float rel = 0.0f;
        if (tol == 0.0f) {
            ok = std::memcmp(&out[i], &golden[i], sizeof(float)) == 0;
        } else {
            float denom = std::fmax(std::fabs(golden[i]), 1e-30f);
            rel = std::fabs(out[i] - golden[i]) / denom;
            ok = rel <= tol || std::fabs(out[i] - golden[i]) <= 1e-30f;
        }
        if (rel > worst) worst = rel;
        if (!ok) {
            if (bad < max_print)
                std::printf("  mismatch y[%d]: got %.9g, expected %.9g\n", i, out[i],
                            golden[i]);
            bad++;
        }
    }
    std::printf("  compared %d values, %d mismatches, max rel err %.3g (tol %g)\n", n,
                bad, worst, tol);
    return bad;
}
