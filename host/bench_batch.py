#!/usr/bin/env python3
"""Batched GEMV kernel (gemv_int8_ps_b) on the U280: time per call and bit-exactness for nb = 1..8.

usage: python3 host/bench_batch.py <xclbin> [iters] [kernel: gemv_int8_ps_b (default) | gemv_int8_ps_bw]
Every result is compared with gen_data.golden_int8_ps for each vector of the batch.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_fpga import FPGABatch, Linear, gen_data  # noqa: E402

xclbin = sys.argv[1]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
kname = sys.argv[3] if len(sys.argv) > 3 else "gemv_int8_ps_b"
f = FPGABatch(xclbin, kname)
rng = np.random.default_rng(1)
shapes = [("gpt2_attn_768x768", 768, 768), ("gpt2_mlp_3072x768", 3072, 768),
          ("sq4096", 4096, 4096), ("llama7b_ffn_11008x4096", 11008, 4096)]
print("shape,M,K,nb,avg_ms,min_ms,ms_per_vector,eff_GBps,bit_exact")
for name, M, K in shapes:
    lin = Linear(name, rng.standard_normal((M, K)).astype(np.float32) * 0.05, np.zeros(M, np.float32))
    f.upload(lin)
    for nb in range(1, 9):
        xs = [gen_data.quantize_activation(rng.standard_normal(K).astype(np.float32)) for _ in range(nb)]
        xq = [a for a, _ in xs]; sx = [b for _, b in xs]
        Y = f.gemv_batch(lin, xq, sx)
        ok = all(np.array_equal(Y[b].view(np.uint32), gen_data.golden_int8_ps(lin.Wq, lin.s, xq[b], sx[b]).view(np.uint32))
                 for b in range(nb))
        ts = []
        for _ in range(iters if M * K <= 4096 * 4096 else iters // 2):
            t0 = f.kernel_s
            f.gemv_batch(lin, xq, sx)
            ts.append(f.kernel_s - t0)
        ts = np.array(ts) * 1e3
        print(f"{name},{M},{K},{nb},{ts.mean():.4f},{ts.min():.4f},{ts.mean()/nb:.4f},"
              f"{lin.Wq.nbytes/ts.mean()/1e6:.2f},{'PASS' if ok else 'FAIL'}", flush=True)
print("BENCH DONE")
