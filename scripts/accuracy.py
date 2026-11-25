#!/usr/bin/env python3
"""Accuracy of the GEMV outputs and a software bit-width sweep.

    python3 scripts/accuracy.py                  # all cases + sweep on n4096_rand_s1
    python3 scripts/accuracy.py --no-sweep       # accuracy table only

Part 1, accuracy. For every test case, compare the golden outputs y_fp32.f32.bin
and y_int8.f32.bin with y_ref.f64.bin (W x in float64 with the original FP32
weights, i.e. the true result). The U280 outputs are bit-identical to these golden
files (results/hw/*.log), so these numbers are also the accuracy of the hardware.

Part 2, bit-width design-space exploration (software only). Quantize the weights of
one 4096 x 4096 case to b bits with one FP32 scale per group of G weights
(symmetric, qmax = 2^(b-1) - 1, round half to even, same rules as data/FORMAT.md)
and the activation to a bits (one scale for the whole vector), then compute
y = sum_g s_w * s_x * (Wq . xq) in float64 and compare with y_ref.
Storage cost: bytes/weight = b/8 + 4/G (the FP32 scale is shared by G weights).

Metrics (y = output, r = y_ref):
  rel_l2   = ||y - r||_2 / ||r||_2
  max_rel  = max|y - r| / max|r|
  cosine   = (y . r) / (||y|| ||r||)
  snr_db   = 20 log10(||r|| / ||y - r||)

Writes results/accuracy/accuracy_metrics.txt and results/dse/bitwidth_sweep.csv.
"""
import argparse
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")

SMALL_CASES = ["n64_rand_s1", "n64_onehot_s1", "n64_pm127_s1", "n64x128_zero_s1",
               "n128_rand_s1", "n100x256_rand_s1"]
LARGE_CASES = ["n4096_rand_s1", "n4096_rand_s2", "n4096_rand_s3"]

W_BITS = [8, 6, 4, 3, 2]
GROUPS = [32, 64, 128, 4096]          # 4096 = one scale per row (K = 4096)
A_BITS = [8, 6, 4]                    # activation sweep at 8-bit weights, group 64


# ---------------------------------------------------------------- helpers

def read_meta(d):
    meta = {}
    with open(os.path.join(d, "meta.txt")) as f:
        for line in f:
            k, v = line.split()
            meta[k] = v
    return int(meta["M"]), int(meta["K"])


def load(d, name, dtype):
    return np.fromfile(os.path.join(d, name), dtype=dtype)


def metrics(y, r):
    y = y.astype(np.float64)
    r = r.astype(np.float64)
    e = y - r
    ne, nr, ny = np.linalg.norm(e), np.linalg.norm(r), np.linalg.norm(y)
    rel_l2 = ne / nr
    max_rel = np.abs(e).max() / np.abs(r).max()
    cosine = float(y @ r) / (ny * nr) if ny > 0 else 0.0
    snr_db = 20.0 * np.log10(nr / ne) if ne > 0 else float("inf")
    return rel_l2, max_rel, cosine, snr_db


# ---------------------------------------------------------------- part 1

def accuracy_table(lines):
    lines.append("Part 1: accuracy of the golden outputs (= U280 hardware outputs, which are")
    lines.append("bit-identical to them) against y_ref.f64 (float64, original weights)")
    lines.append("")
    hdr = f"{'case':20s} {'kernel':6s} {'rel_l2':>10s} {'max_rel':>10s} {'cosine':>12s} {'snr_db':>8s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name in LARGE_CASES + SMALL_CASES:
        d = os.path.join(DATA, name)
        if not os.path.isfile(os.path.join(d, "y_ref.f64.bin")):
            lines.append(f"{name:20s} missing (run scripts/gen_data.py)")
            continue
        r = load(d, "y_ref.f64.bin", np.float64)
        for kern, fname in (("fp32", "y_fp32.f32.bin"), ("int8", "y_int8.f32.bin")):
            y = load(d, fname, np.float32)
            rel_l2, max_rel, cos, snr = metrics(y, r)
            lines.append(f"{name:20s} {kern:6s} {rel_l2:10.3e} {max_rel:10.3e} "
                         f"{cos:12.8f} {snr:8.2f}")
    lines.append("")


# ---------------------------------------------------------------- part 2

def quantize_groups(W, bits, G):
    """Symmetric per-group quantization; returns dequantized weights (float64)."""
    M, K = W.shape
    qmax = 2 ** (bits - 1) - 1
    Wg = W.reshape(M, K // G, G).astype(np.float32)
    amax = np.abs(Wg).max(axis=2)
    s = np.where(amax > 0, amax / np.float32(qmax), np.float32(1)).astype(np.float32)
    q = np.clip(np.rint(Wg / s[:, :, None]), -qmax, qmax)
    return (q * s[:, :, None].astype(np.float64)).reshape(M, K)


def quantize_vector(x, bits):
    qmax = 2 ** (bits - 1) - 1
    amax = np.float32(np.abs(x).max())
    sx = np.float32(amax / np.float32(qmax)) if amax > 0 else np.float32(1)
    xq = np.clip(np.rint(x / sx), -qmax, qmax)
    return xq * np.float64(sx)


def bitwidth_sweep(case, lines, csv_path):
    d = os.path.join(DATA, case)
    M, K = read_meta(d)
    W = load(d, "W.f32.bin", np.float32).reshape(M, K)
    x = load(d, "x.f32.bin", np.float32)
    r = load(d, "y_ref.f64.bin", np.float64)
    x64 = x.astype(np.float64)

    rows = []
    for wb in W_BITS:
        for G in GROUPS:
            Wd = quantize_groups(W, wb, G)
            for ab in (8, 32):              # 32 = activation kept in FP32
                xd = quantize_vector(x, ab) if ab < 32 else x64
                y = Wd @ xd
                rows.append((wb, G, ab, wb / 8 + 4 / G) + metrics(y, r))
    Wd = quantize_groups(W, 8, 64)
    for ab in A_BITS[1:]:
        y = Wd @ quantize_vector(x, ab)
        rows.append((8, 64, ab, 1 + 4 / 64) + metrics(y, r))

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="\n") as f:
        f.write("w_bits,group,a_bits,bytes_per_weight,rel_l2,max_rel,cosine,snr_db\n")
        for wb, G, ab, bpw, rel_l2, max_rel, cos, snr in rows:
            f.write(f"{wb},{G},{ab},{bpw:.4f},{rel_l2:.4e},{max_rel:.4e},{cos:.8f},{snr:.2f}\n")

    lines.append(f"Part 2: bit-width sweep on {case} (software, float64 accumulation)")
    lines.append("a_bits 32 = activation not quantized (FP32); group 4096 = one scale per row")
    lines.append("")
    hdr = (f"{'w_bits':>6s} {'group':>6s} {'a_bits':>6s} {'B/weight':>9s} "
           f"{'rel_l2':>10s} {'max_rel':>10s} {'cosine':>12s} {'snr_db':>8s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for wb, G, ab, bpw, rel_l2, max_rel, cos, snr in rows:
        lines.append(f"{wb:6d} {G:6d} {ab:6d} {bpw:9.4f} {rel_l2:10.3e} {max_rel:10.3e} "
                     f"{cos:12.8f} {snr:8.2f}")
    lines.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-sweep", action="store_true", help="skip the bit-width sweep")
    ap.add_argument("--case", default="n4096_rand_s1", help="case for the sweep")
    args = ap.parse_args()

    lines = []
    accuracy_table(lines)
    if not args.no_sweep:
        bitwidth_sweep(args.case, lines, os.path.join(ROOT, "results", "dse", "bitwidth_sweep.csv"))

    out = os.path.join(ROOT, "results", "accuracy", "accuracy_metrics.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="\n") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
