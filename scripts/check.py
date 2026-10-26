#!/usr/bin/env python3
"""Compare a kernel output file against the golden output of a test case.

    python3 scripts/check.py data/n64_rand_s1 y_out.f32.bin --kernel fp32
    python3 scripts/check.py data/n64_rand_s1 y_out.f32.bin --kernel int8   # bit-exact
    python3 scripts/check.py --verify-checksums                            # data/CHECKSUMS.md5

Same rule as common/binio.h: tol = 0 means bit-exact, otherwise
|out - golden| <= tol * max|golden|. Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def read_meta(case):
    meta = {}
    with open(os.path.join(case, "meta.txt")) as f:
        for line in f:
            k, v = line.split()
            meta[k] = v
    return int(meta["M"]), int(meta["K"])


def compare(out, golden, tol):
    scale = float(np.abs(golden).max()) or 1.0
    err = np.abs(out.astype(np.float64) - golden.astype(np.float64)) / scale
    same = out.view(np.uint32) == golden.view(np.uint32)
    bad = ~same if tol == 0 else ~(err <= tol)
    for i in np.flatnonzero(bad)[:10]:
        print(f"  mismatch y[{i}]: got {out[i]:.9g}, expected {golden[i]:.9g}")
    print(f"  {len(out)} values: {same.sum()} bit-exact, {bad.sum()} mismatches, "
          f"max err {err.max():.3g} x max|y| (tol {tol:g})")
    return int(bad.sum())


def verify_checksums():
    path = os.path.join(DATA, "CHECKSUMS.md5")
    ok = True
    with open(path) as f:
        for line in f:
            digest, name = line.split()
            name = name.lstrip("*")
            p = os.path.join(DATA, name)
            if not os.path.exists(p):
                print(f"  missing  {name}")
                ok = False
                continue
            with open(p, "rb") as g:
                got = hashlib.md5(g.read()).hexdigest()
            status = "ok" if got == digest else "DIFFERENT"
            ok &= got == digest
            print(f"  {status:9s} {name}")
    print("CHECKSUMS PASSED" if ok else "CHECKSUMS FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--kernel", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--tol", type=float, default=None,
                    help="default: 0 (bit-exact) for int8, 1e-6 for fp32")
    ap.add_argument("--verify-checksums", action="store_true")
    args = ap.parse_args()

    if args.verify_checksums:
        sys.exit(verify_checksums())
    if not (args.case and args.output):
        ap.error("case and output are required")

    M, _ = read_meta(args.case)
    golden = np.fromfile(os.path.join(args.case, f"y_{args.kernel}.f32.bin"), np.float32)
    out = np.fromfile(args.output, np.float32)
    if out.size != M:
        print(f"output has {out.size} values, expected {M}")
        sys.exit(1)
    tol = args.tol if args.tol is not None else (0.0 if args.kernel == "int8" else 1e-6)
    bad = compare(out, golden, tol)
    print("TEST PASSED" if bad == 0 else "TEST FAILED")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
