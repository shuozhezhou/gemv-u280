#!/usr/bin/env python3
"""Generate GEMV test cases: inputs, quantized inputs and golden outputs.

File format and formulas: data/FORMAT.md.

    python3 scripts/gen_data.py                 # default cases (small ones + 4096x4096)
    python3 scripts/gen_data.py --no-large      # skip the 4096x4096 cases
    python3 scripts/gen_data.py --small 100     # also 100 random small cases (data/small/)

Golden outputs are computed in float32 in the same order as the kernels:
  FP32 kernel:  acc = 0; for k: acc = acc + W[m,k]*x[k]          (mul, then add; no FMA)
  INT8 kernel:  P[m,g] exact int32; acc = 0; for g: acc = acc + s[m,g]*float(P[m,g]);
                y[m] = sx * acc
"""
import argparse
import os
import shutil

import numpy as np

GROUP = 64
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


# ---------------------------------------------------------------- inputs

def rand_weights(rng, M, K, sigma=0.02, outlier_frac=0.001, outlier_mult=10.0):
    """Gaussian weights plus a few large outliers, so group scales differ."""
    W = rng.normal(0.0, sigma, size=(M, K))
    n_out = max(1, int(outlier_frac * M * K))
    idx = rng.choice(M * K, size=n_out, replace=False)
    W.flat[idx] *= outlier_mult
    return W.astype(np.float32)


def rand_activation(rng, K):
    return rng.normal(0.0, 1.0, size=K).astype(np.float32)


def case_rand(rng, M, K):
    return rand_weights(rng, M, K), rand_activation(rng, K)


def case_onehot(rng, M, K):
    """Each row has a single 1.0, so y[m] = x[col[m]] (checkable by hand)."""
    W = np.zeros((M, K), np.float32)
    W[np.arange(M), rng.integers(0, K, size=M)] = 1.0
    return W, rand_activation(rng, K)


def case_pm127(rng, M, K):
    """|W| and |x| constant, so every quantized value is +-127.
    Row 0 has the same signs as x: every group sum is the maximum 64*127*127."""
    xs = rng.choice([-1.0, 1.0], size=K)
    Ws = rng.choice([-1.0, 1.0], size=(M, K))
    Ws[0] = xs
    return (0.05 * Ws).astype(np.float32), (0.5 * xs).astype(np.float32)


def case_zero(rng, M, K):
    """Random weights with all-zero groups (every 3rd group), an all-zero row
    and an all-zero column of groups; checks the s = 1 rule for empty groups."""
    W, x = case_rand(rng, M, K)
    G = K // GROUP
    Wg = W.reshape(M, G, GROUP)
    for m in range(M):
        Wg[m, (m % 3)::3, :] = 0.0
    Wg[:, 0, :] = 0.0
    W[M // 2, :] = 0.0
    return W, x


KINDS = {"rand": case_rand, "onehot": case_onehot, "pm127": case_pm127, "zero": case_zero}


# ---------------------------------------------------------------- quantization

def quantize_weights(W):
    M, K = W.shape
    Wg = W.reshape(M, K // GROUP, GROUP)
    amax = np.abs(Wg).max(axis=2)                                   # float32
    s = np.where(amax > 0, amax / np.float32(127), np.float32(1)).astype(np.float32)
    q = np.rint(Wg / s[:, :, None])                                 # half to even
    q = np.clip(q, -127, 127).astype(np.int8).reshape(M, K)
    return q, s


def quantize_activation(x):
    amax = np.float32(np.abs(x).max())
    sx = np.float32(amax / np.float32(127)) if amax > 0 else np.float32(1)
    xq = np.clip(np.rint(x / sx), -127, 127).astype(np.int8)
    return xq, sx


# ---------------------------------------------------------------- golden models

def golden_fp32(W, x):
    """Sequential float32 dot product per row, k = 0..K-1."""
    M, K = W.shape
    acc = np.zeros(M, np.float32)
    for k in range(K):
        prod = (W[:, k] * x[k]).astype(np.float32)   # rounded product
        acc = (acc + prod).astype(np.float32)         # rounded sum
    return acc


def reference_f64(W, x):
    """W @ x in float64, summed sequentially over k. Not W.astype(f64) @ x: that
    calls BLAS, whose summation order (and so the last bits) differs between
    machines, which would break the checksums."""
    M, K = W.shape
    W64, x64 = W.astype(np.float64), x.astype(np.float64)
    acc = np.zeros(M, np.float64)
    for k in range(K):
        acc += W64[:, k] * x64[k]
    return acc


def group_sums(Wq, xq):
    M, K = Wq.shape
    G = K // GROUP
    P = (Wq.reshape(M, G, GROUP).astype(np.int64) * xq.reshape(1, G, GROUP)).sum(axis=2)
    assert np.abs(P).max() <= GROUP * 127 * 127 < 2**20
    return P.astype(np.int32)


def golden_int8(Wq, s, xq, sx):
    P = group_sums(Wq, xq)
    M, G = P.shape
    acc = np.zeros(M, np.float32)
    for g in range(G):
        term = (s[:, g] * P[:, g].astype(np.float32)).astype(np.float32)
        acc = (acc + term).astype(np.float32)
    return (np.float32(sx) * acc).astype(np.float32)


# ---------------------------------------------------------------- writing

def write_case(name, M, K, kind, seed, quiet=False):
    assert K % GROUP == 0, "K must be a multiple of 64"
    rng = np.random.default_rng(seed)
    W, x = KINDS[kind](rng, M, K)

    Wq, s = quantize_weights(W)
    xq, sx = quantize_activation(x)
    y_fp32 = golden_fp32(W, x)
    y_int8 = golden_int8(Wq, s, xq, sx)
    y_ref = reference_f64(W, x)

    d = os.path.join(ROOT, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    with open(os.path.join(d, "meta.txt"), "w", newline="\n") as f:
        f.write(f"M {M}\nK {K}\ngroup {GROUP}\nseed {seed}\nkind {kind}\n")
    files = {
        "W.f32.bin": W, "x.f32.bin": x, "y_fp32.f32.bin": y_fp32,
        "Wq.i8.bin": Wq, "s.f32.bin": s, "xq.i8.bin": xq,
        "sx.f32.bin": np.array([sx], np.float32), "y_int8.f32.bin": y_int8,
        "y_ref.f64.bin": y_ref,
    }
    for fname, arr in files.items():
        np.ascontiguousarray(arr).tofile(os.path.join(d, fname))

    if not quiet:
        scale = np.abs(y_ref).max() or 1.0
        err_fp32 = np.abs(y_fp32 - y_ref).max() / scale
        err_int8 = np.abs(y_int8 - y_ref).max() / scale
        print(f"{name:22s} M={M:5d} K={K:5d}  fp32 err {err_fp32:.2e}  "
              f"int8 err {err_int8:.2e}  (max |y-y_ref| / max|y_ref|)")


def self_test():
    """Cross-check the golden models against independent computations."""
    rng = np.random.default_rng(0)
    W, x = case_rand(rng, 32, 256)
    # Sequential float32 vs float64: must be close.
    y64 = W.astype(np.float64) @ x.astype(np.float64)
    assert np.allclose(golden_fp32(W, x), y64, rtol=0, atol=1e-5 * np.abs(y64).max())
    # Quantized group sums vs a plain Python loop.
    Wq, s = quantize_weights(W)
    xq, sx = quantize_activation(x)
    P = group_sums(Wq, xq)
    for m in (0, 31):
        for g in (0, 3):
            ref = sum(int(Wq[m, k]) * int(xq[k]) for k in range(g * GROUP, (g + 1) * GROUP))
            assert P[m, g] == ref
    # Quantization round trip error is at most half a step per element.
    deq = Wq.reshape(32, 4, GROUP) * s[:, :, None]
    assert np.all(np.abs(deq - W.reshape(32, 4, GROUP)) <= 0.5 * s[:, :, None] * (1 + 1e-6))
    # pm127: all quantized values are +-127 and row 0 reaches the largest group sum.
    W, x = case_pm127(rng, 4, 128)
    Wq, s = quantize_weights(W)
    xq, sx = quantize_activation(x)
    assert np.all(np.abs(Wq) == 127) and np.all(np.abs(xq) == 127)
    assert np.all(group_sums(Wq, xq)[0] == GROUP * 127 * 127)
    print("self test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-large", action="store_true", help="skip the 4096x4096 cases")
    ap.add_argument("--small", type=int, default=0, help="number of random small cases")
    args = ap.parse_args()

    self_test()
    # Committed test cases (small).
    write_case("n64_rand_s1", 64, 64, "rand", 1)
    write_case("n64_onehot_s1", 64, 64, "onehot", 1)
    write_case("n64_pm127_s1", 64, 64, "pm127", 1)
    write_case("n64x128_zero_s1", 64, 128, "zero", 1)
    write_case("n128_rand_s1", 128, 128, "rand", 1)
    write_case("n100x256_rand_s1", 100, 256, "rand", 1)   # M not a power of two
    # Full size (not committed; regenerated from the seeds).
    if not args.no_large:
        for seed in (1, 2, 3):
            write_case(f"n4096_rand_s{seed}", 4096, 4096, "rand", seed)
    # Random small shapes for the sw_emu sweep (not committed).
    if args.small:
        rng = np.random.default_rng(12345)
        for i in range(args.small):
            M = int(rng.integers(1, 97))
            K = GROUP * int(rng.integers(1, 5))
            write_case(f"small/r{i:03d}", M, K, "rand", 1000 + i, quiet=True)
        print(f"wrote {args.small} random small cases to data/small/")


if __name__ == "__main__":
    main()
