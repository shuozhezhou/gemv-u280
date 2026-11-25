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
  INT8 PS:      same terms t = s[m,g]*float(P[m,g]), but NPS = 8 partial sums:
                a[g % 8] = a[g % 8] + t (g in order), then the fixed tree
                ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)), y[m] = sx * sum
  INT4 PS:      symmetric int4 weights (s4 = max|W|/7, q in [-7, 7]), 2 groups per
                512-bit word: word term w = t[2j] + t[2j+1], then 8 partial sums
                a[j % 8] over words j and the same tree as INT8 PS
  INT8 PS, batch (gemv_int8_ps_b): BATCH activation vectors x_b, each quantized on
                its own (xq_b, sx_b); y_b = golden_int8_ps(Wq, s, xq_b, sx_b). Vector 0
                is the case's x, so y_int8psB[0] == y_int8ps.
  INT8 FX:      same terms, each converted to fixed point with FX_FRAC fraction bits
                (floor, = ap_fixed AP_TRN), summed exactly as integers,
                y[m] = sx * (float(acc) * 2^-FX_FRAC)
"""
import argparse
import os
import shutil

import numpy as np

GROUP = 64
NPS = 8                 # partial sums of the gemv_int8_ps kernel (>= FP32 add latency 7)
BATCH = 8               # activation vectors per case for gemv_int8_ps_b (BMAX)
FX_BITS, FX_FRAC = 48, 24   # gemv_int8_fx accumulator: ap_fixed<48, 24>
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


def quantize_weights4(W):
    """Symmetric INT4 with one FP32 scale per group of 64: q in [-7, 7]."""
    M, K = W.shape
    Wg = W.reshape(M, K // GROUP, GROUP)
    amax = np.abs(Wg).max(axis=2)
    s = np.where(amax > 0, amax / np.float32(7), np.float32(1)).astype(np.float32)
    q = np.clip(np.rint(Wg / s[:, :, None]), -7, 7).astype(np.int8).reshape(M, K)
    return q, s


def pack_int4(q):
    """(M, K) int4 values -> (M, K/2) bytes; byte b = weight 2b (low nibble) |
    weight 2b+1 (high nibble), two's complement. A 512-bit word = 128 weights,
    weight e in bits [4e+3 : 4e]."""
    u = (q.astype(np.int16) & 0xF).astype(np.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def unpack_int4(p):
    lo = (p & 0xF).astype(np.int8)
    hi = (p >> 4).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo)
    hi = np.where(hi > 7, hi - 16, hi)
    q = np.empty((p.shape[0], 2 * p.shape[1]), np.int8)
    q[:, 0::2], q[:, 1::2] = lo, hi
    return q


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


def group_terms(Wq, s, xq):
    """t[m,g] = s[m,g] * float(P[m,g]), one rounded float32 multiply."""
    P = group_sums(Wq, xq)
    return (s * P.astype(np.float32)).astype(np.float32)


def golden_int8(Wq, s, xq, sx):
    t = group_terms(Wq, s, xq)
    M, G = t.shape
    acc = np.zeros(M, np.float32)
    for g in range(G):
        acc = (acc + t[:, g]).astype(np.float32)
    return (np.float32(sx) * acc).astype(np.float32)


def tree_sum(v):
    """Pairwise tree over a list of float32 arrays, adjacent pairs level by level:
    8 inputs -> ((v0+v1)+(v2+v3))+((v4+v5)+(v6+v7)). Same order as tree_sum() in
    kernels/gemv_int8_ps.cpp."""
    v = list(v)
    while len(v) > 1:
        v = [(v[2 * j] + v[2 * j + 1]).astype(np.float32) for j in range(len(v) // 2)]
    return v[0]


def golden_int8_ps(Wq, s, xq, sx, nps=NPS):
    """gemv_int8_ps: partial sum a[p] collects the groups g with g % nps == p, in
    order of g, starting from 0.0f (a partial sum with no group stays 0.0f)."""
    t = group_terms(Wq, s, xq)
    M, G = t.shape
    a = np.zeros((nps, M), np.float32)
    for g in range(G):
        a[g % nps] = (a[g % nps] + t[:, g]).astype(np.float32)
    total = tree_sum([a[p] for p in range(nps)])
    return (np.float32(sx) * total).astype(np.float32)


def golden_int4_ps(q4, s4, xq, sx, nps=NPS):
    """gemv_int4_ps: one word = groups 2j (low half) and 2j+1; word term
    w = t[2j] + t[2j+1] (float32), then partial sums a[j % nps] over words j in
    order, the same tree as golden_int8_ps, y = sx * sum."""
    t = group_terms(q4, s4, xq)
    M, G = t.shape
    assert G % 2 == 0, "INT4 needs K to be a multiple of 128"
    w = (t[:, 0::2] + t[:, 1::2]).astype(np.float32)
    a = np.zeros((nps, M), np.float32)
    for j in range(G // 2):
        a[j % nps] = (a[j % nps] + w[:, j]).astype(np.float32)
    total = tree_sum([a[p] for p in range(nps)])
    return (np.float32(sx) * total).astype(np.float32)


def batch_activations(x, seed, K):
    """BATCH activation vectors: x itself, then BATCH-1 vectors from their own seeds
    (so the vectors do not depend on how many other files a case writes)."""
    xs = [x] + [rand_activation(np.random.default_rng(seed * 1000 + b), K) for b in range(1, BATCH)]
    q = [quantize_activation(v) for v in xs]
    return np.stack([a for a, _ in q]), np.array([b for _, b in q], np.float32)


def golden_int8_ps_batch(Wq, s, xqB, sxB):
    """[BATCH][M]: golden_int8_ps for every vector (the batch kernel keeps the
    per-vector op order of gemv_int8_ps)."""
    return np.stack([golden_int8_ps(Wq, s, xqB[b], sxB[b]) for b in range(len(sxB))])


def golden_int8_fx(Wq, s, xq, sx, bits=FX_BITS, frac=FX_FRAC):
    """gemv_int8_fx: each term is converted to ap_fixed<bits, bits-frac> with the
    default AP_TRN (floor) mode, the fixed-point sum is exact (int64 here), and
    y = sx * (float(acc) * 2^-frac). float32 * 2^frac is exact in float64, so the
    floor is exact; int64 -> float32 rounds to nearest even like the hardware."""
    t = group_terms(Wq, s, xq)
    q = np.floor(t.astype(np.float64) * 2.0 ** frac).astype(np.int64)
    acc = q.sum(axis=1)                       # exact: no rounding in fixed point
    lim = 2 ** (bits - 1)
    assert np.abs(q).max(initial=0) < lim and np.abs(np.cumsum(q, axis=1)).max(initial=0) < lim,         "fixed-point accumulator would overflow"
    yf = (acc.astype(np.float32) * np.float32(2.0 ** -frac)).astype(np.float32)
    return (np.float32(sx) * yf).astype(np.float32)


# ---------------------------------------------------------------- writing

def write_case(name, M, K, kind, seed, quiet=False):
    assert K % GROUP == 0, "K must be a multiple of 64"
    rng = np.random.default_rng(seed)
    W, x = KINDS[kind](rng, M, K)

    Wq, s = quantize_weights(W)
    xq, sx = quantize_activation(x)
    y_fp32 = golden_fp32(W, x)
    y_int8 = golden_int8(Wq, s, xq, sx)
    y_int8ps = golden_int8_ps(Wq, s, xq, sx)
    y_int8fx = golden_int8_fx(Wq, s, xq, sx)
    y_ref = reference_f64(W, x)
    xqB, sxB = batch_activations(x, seed, K)
    y_int8psB = golden_int8_ps_batch(Wq, s, xqB, sxB)
    assert np.array_equal(y_int8psB[0].view(np.uint32), y_int8ps.view(np.uint32))
    has_int4 = K % (2 * GROUP) == 0          # INT4: 2 groups per 512-bit word
    if has_int4:
        q4, s4 = quantize_weights4(W)
        y_int4 = golden_int4_ps(q4, s4, xq, sx)

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
        "y_int8ps.f32.bin": y_int8ps, "y_int8fx.f32.bin": y_int8fx,
        "y_ref.f64.bin": y_ref,
        "xqB.i8.bin": xqB, "sxB.f32.bin": sxB, "y_int8psB.f32.bin": y_int8psB,
    }
    if has_int4:
        files.update({"Wq4.i4.bin": pack_int4(q4), "s4.f32.bin": s4, "y_int4.f32.bin": y_int4})
    for fname, arr in files.items():
        np.ascontiguousarray(arr).tofile(os.path.join(d, fname))

    if not quiet:
        scale = np.abs(y_ref).max() or 1.0
        err_fp32 = np.abs(y_fp32 - y_ref).max() / scale
        err = {k: np.abs(v - y_ref).max() / scale for k, v in
               (("int8", y_int8), ("ps", y_int8ps), ("fx", y_int8fx))}
        print(f"{name:22s} M={M:5d} K={K:5d}  fp32 err {err_fp32:.2e}  "
              f"int8 err {err['int8']:.2e} (ps {err['ps']:.2e}, fx {err['fx']:.2e})"
              + (f"  int4 err {np.abs(y_int4 - y_ref).max() / scale:.2e}" if has_int4 else "")
              + "  (max |y-y_ref| / max|y_ref|)")


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
    # Partial-sum and fixed-point models vs plain Python loops (G = 21: not a
    # multiple of 8, so some partial sums get 3 groups and some 2).
    W, x = case_rand(rng, 6, 21 * GROUP)
    Wq, s = quantize_weights(W)
    xq, sx = quantize_activation(x)
    t = group_terms(Wq, s, xq)
    y_ps, y_fx = golden_int8_ps(Wq, s, xq, sx), golden_int8_fx(Wq, s, xq, sx)
    f32 = np.float32
    for m in range(6):
        a = [f32(0)] * NPS
        for g in range(21):
            a[g % NPS] = f32(a[g % NPS] + t[m, g])
        tot = f32(f32(f32(a[0] + a[1]) + f32(a[2] + a[3])) + f32(f32(a[4] + a[5]) + f32(a[6] + a[7])))
        assert f32(sx * tot) == y_ps[m]
        acc = sum(int(np.floor(float(t[m, g]) * 2 ** FX_FRAC)) for g in range(21))
        assert f32(sx * f32(f32(acc) * f32(2.0 ** -FX_FRAC))) == y_fx[m]
    # Both are close to the sequential INT8 result.
    y8 = golden_int8(Wq, s, xq, sx)
    assert np.allclose(y_ps, y8, rtol=0, atol=1e-5 * np.abs(y8).max())
    assert np.allclose(y_fx, y8, rtol=0, atol=1e-5 * np.abs(y8).max())
    # INT4: pack/unpack round trip, range, and the word-pair golden vs a loop.
    W, x = case_rand(rng, 5, 21 * 2 * GROUP)
    q4, s4 = quantize_weights4(W)
    xq, sx = quantize_activation(x)
    assert np.abs(q4).max() <= 7 and np.array_equal(unpack_int4(pack_int4(q4)), q4)
    t = group_terms(q4, s4, xq)
    y4 = golden_int4_ps(q4, s4, xq, sx)
    for m in range(5):
        a = [f32(0)] * NPS
        for j in range(21):
            a[j % NPS] = f32(a[j % NPS] + f32(t[m, 2 * j] + t[m, 2 * j + 1]))
        tot = f32(f32(f32(a[0] + a[1]) + f32(a[2] + a[3])) + f32(f32(a[4] + a[5]) + f32(a[6] + a[7])))
        assert f32(sx * tot) == y4[m]
    print("self test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-large", action="store_true", help="skip the 4096x4096 cases")
    ap.add_argument("--small", type=int, default=0, help="number of random small cases")
    ap.add_argument("--llama", action="store_true",
                    help="also llama7b_ffn_11008x4096 (seed 1, same as scripts/gen_sweep.py)")
    args = ap.parse_args()

    self_test()
    # Committed test cases (small).
    write_case("n64_rand_s1", 64, 64, "rand", 1)
    write_case("n64_onehot_s1", 64, 64, "onehot", 1)
    write_case("n64_pm127_s1", 64, 64, "pm127", 1)
    write_case("n64x128_zero_s1", 64, 128, "zero", 1)
    write_case("n128_rand_s1", 128, 128, "rand", 1)
    write_case("n100x256_rand_s1", 100, 256, "rand", 1)   # M not a power of two
    write_case("n40x1344_rand_s1", 40, 1344, "rand", 1)   # 21 groups: > 8 partial sums
    write_case("n24x2688_rand_s1", 24, 2688, "rand", 1)   # INT4: 21 words: > 8 partial sums
    # Full size (not committed; regenerated from the seeds).
    if not args.no_large:
        for seed in (1, 2, 3):
            write_case(f"n4096_rand_s{seed}", 4096, 4096, "rand", seed)
    if args.llama:
        write_case("llama7b_ffn_11008x4096", 11008, 4096, "rand", 1)
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
