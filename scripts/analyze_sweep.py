"""Analyze the U280 size sweep (results/hw_sweep/logs).

Inputs : logs/<kernel>_<case>.log  (host output), logs/runs.csv (start/end of every run),
         logs/power.csv (board power from `xbutil examine -r electrical`, ~1 sample/s)
Outputs: results/hw_sweep/sweep_summary.csv, power_summary.txt, llama7b_token_estimate.txt,
         results/hw_sweep/figures/*.png
Usage  : python scripts/analyze_sweep.py   (run from the repo root)
"""
import csv, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join("results", "hw_sweep")
L = os.path.join(D, "logs")
F = os.path.join(D, "figures")
os.makedirs(F, exist_ok=True)
CLK = 300e6
C_INT8, C_FP32 = "#2a78d6", "#eb6834"


def parse_log(path):
    t = open(path).read()
    g = lambda p, f=float: f(re.search(p, t).group(1))
    return dict(
        M=g(r"M=(\d+)", int), K=g(r"K=(\d+)", int),
        avg=g(r"kernel time\s+avg ([\d.]+) ms"), mn=g(r"min ([\d.]+) ms"), mx=g(r"max ([\d.]+) ms"),
        runs=g(r"\((\d+) runs\)", int), load=g(r"xclbin loaded in ([\d.]+) ms"),
        h2d=g(r"host->device ([\d.]+) ms"), cpu=g(r"CPU \(1 thread, same op order\) ([\d.]+) ms"),
        gbs=g(r"bandwidth\s+([\d.]+) GB/s"), passed="TEST PASSED" in t)


runs = list(csv.DictReader(open(os.path.join(L, "runs.csv"))))
pw = np.array([(float(a), float(b)) for a, b in list(csv.reader(open(os.path.join(L, "power.csv"))))[1:] if b])
t_first, t_last = float(runs[0]["start"]), float(runs[-1]["end"])
idle_noxcl = pw[pw[:, 0] < t_first, 1]           # before any xclbin of this sweep was loaded
idle_loaded = pw[pw[:, 0] > t_last + 1.0, 1]      # xclbin still loaded, no kernel running


def window_power(r, s):
    """Mean board power while the timed kernel loop runs (skip load + H2D, stop before CPU check)."""
    t0 = float(r["start"]) + s["load"] / 1e3 + s["h2d"] / 1e3 + 0.5
    t1 = float(r["start"]) + s["load"] / 1e3 + s["runs"] * s["avg"] / 1e3
    w = pw[(pw[:, 0] >= t0) & (pw[:, 0] <= t1), 1]
    return (w.mean(), len(w)) if len(w) >= 3 else (float("nan"), len(w))


rows = []
for r in runs:
    k, c = r["kernel"], r["case"]
    s = parse_log(os.path.join(L, f"{k}_{c}.log"))
    if int(r["iters"]) != s["runs"]:
        continue
    p, n = window_power(r, s)
    rows.append(dict(kernel=k, case=c, **s, power_w=p, power_samples=n, run_start=float(r["start"])))

# keep the longest run per (kernel, case) for power; timing rows are identical anyway
best = {}
for x in rows:
    key = (x["kernel"], x["case"])
    if key not in best or x["runs"] > best[key]["runs"]:
        best[key] = x

P0 = float(np.mean(idle_loaded))
out = []
for (k, c), x in best.items():
    MK = x["M"] * x["K"]
    gemv_s = x["avg"] / 1e3
    out.append(dict(
        kernel=k, case=c, M=x["M"], K=x["K"], runs=x["runs"], avg_ms=x["avg"], min_ms=x["mn"], max_ms=x["mx"],
        cycles=round(gemv_s * CLK), cycles_per_64w=round(gemv_s * CLK / (MK / 64), 2),
        gbs=x["gbs"], gmacs=round(MK / gemv_s / 1e9, 3), cpu1t_ms=x["cpu"], speedup_vs_cpu1t=round(x["cpu"] / x["avg"], 2),
        h2d_ms=x["h2d"], power_w=round(x["power_w"], 2), power_samples=x["power_samples"],
        energy_mj=round(x["power_w"] * x["avg"], 2) if x["power_samples"] >= 3 else "",
        dyn_energy_mj=round((x["power_w"] - P0) * x["avg"], 3) if x["power_samples"] >= 3 else "",
        passed=x["passed"]))
order = ["sq256", "sq512", "gpt2_attn_768x768", "sq1024", "gpt2_mlp_3072x768", "sq2048", "sq4096",
         "llama7b_attn_4096x4096", "llama7b_ffn_11008x4096"]
out.sort(key=lambda r: (r["kernel"], order.index(r["case"])))
with open(os.path.join(D, "sweep_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

T = {(r["kernel"], r["case"]): r for r in out}

# ---- power summary --------------------------------------------------------------------------------
with open(os.path.join(D, "power_summary.txt"), "w") as f:
    f.write("U280 board power, xbutil examine -r electrical, ~1 sample/s (node pc157)\n")
    f.write(f"idle, before this sweep's xclbin loads : {idle_noxcl.mean():.2f} W  (n={len(idle_noxcl)})\n")
    f.write(f"idle, user xclbin loaded, no kernel    : {P0:.2f} W  (n={len(idle_loaded)})\n")
    for r in out:
        if r["energy_mj"] != "":
            f.write(f"{r['kernel']:10s} {r['case']:24s} {r['power_w']:6.2f} W  (+{r['power_w']-P0:5.2f} W over loaded idle, "
                    f"n={r['power_samples']:2d})  energy/GEMV {r['energy_mj']:9.2f} mJ board, {r['dyn_energy_mj']:8.3f} mJ dynamic\n")
    a, b = T[("gemv_int8", "sq4096")], T[("gemv_fp32", "sq4096")]
    f.write(f"\n4096x4096: energy per GEMV FP32 {b['energy_mj']:.0f} mJ vs INT8 {a['energy_mj']:.1f} mJ "
            f"-> {b['energy_mj']/a['energy_mj']:.1f}x less energy with INT8\n")

# ---- Llama-2-7B one-token decode estimate ----------------------------------------------------------
# per layer: q,k,v,o (4096x4096) + gate, up (11008x4096) + down (4096x11008, same MAC count; K>4096 not
# supported by the current kernel, so its time is taken equal to gate/up). 32 layers. lm_head not counted.
with open(os.path.join(D, "llama7b_token_estimate.txt"), "w") as f:
    f.write("Llama-2-7B, one decode token = 32 layers x (4 x 4096x4096 + 3 x 11008x4096) GEMVs, measured kernel times\n")
    for k in ("gemv_fp32", "gemv_int8"):
        ta, tf = T[(k, "llama7b_attn_4096x4096")], T[(k, "llama7b_ffn_11008x4096")]
        layer = 4 * ta["avg_ms"] + 3 * tf["avg_ms"]
        tok = 32 * layer / 1e3
        e = 32 * (4 * (ta["energy_mj"] or ta["power_w"] * ta["avg_ms"]) + 3 * tf["energy_mj"]) / 1e3 if tf["energy_mj"] != "" else float("nan")
        cpu = 32 * (4 * ta["cpu1t_ms"] + 3 * tf["cpu1t_ms"]) / 1e3
        f.write(f"{k:10s} layer {layer:9.2f} ms   token {tok:8.2f} s  ({1/tok:6.3f} tok/s)   "
                f"board energy/token ~{e:8.1f} J   CPU 1-thread same op order {cpu:6.2f} s/token\n")

# ---- cycle model: cycles = c0 + c_row * M + c_grp * (M*K/64), least squares over all measured shapes ----
with open(os.path.join(D, "cycle_model.txt"), "w") as f:
    f.write("Least-squares fit of measured kernel cycles (300 MHz): cycles = c0 + c_row*M + c_grp*(M*K/64)\n")
    for k in ("gemv_fp32", "gemv_int8"):
        R = [r for r in out if r["kernel"] == k]
        M = np.array([r["M"] for r in R], float); K = np.array([r["K"] for r in R], float)
        cyc = np.array([r["avg_ms"] * 1e-3 * CLK for r in R])
        A = np.c_[np.ones_like(M), M, M * K / 64]
        sol = np.linalg.lstsq(A, cyc, rcond=None)[0]
        err = np.abs(A @ sol - cyc) / cyc
        f.write(f"{k:10s} c0 = {sol[0]:8.0f} cyc ({sol[0]/CLK*1e6:5.1f} us)   c_row = {sol[1]:7.1f} cyc   "
                f"c_grp = {sol[2]:8.3f} cyc per 64 weights ({sol[2]/64:.3f} per weight)   max fit error {100*err.max():.2f} %\n")
    f.write("\nReading: FP32 = 7.00 cycles per weight (fadd latency, II=7). INT8 = 6.99 cycles per 64-weight group\n"
            "(same II=7, now per group) + ~100 cycles per row + ~37 us per call. With II=1 per group (L3) the group\n"
            "term for 4096x4096 drops from 1.835 M to 0.262 M cycles, so the per-row term (~0.41 M) would dominate\n"
            "unless rows are overlapped.\n")

# ---- figures ---------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.6})

sq = ["sq256", "sq512", "sq1024", "sq2048", "sq4096"]
n = [int(c[2:]) for c in sq]
fig, ax = plt.subplots(figsize=(6.4, 4))
for k, col, lab in (("gemv_fp32", C_FP32, "L1 FP32"), ("gemv_int8", C_INT8, "L2 INT8")):
    y = [T[(k, c)]["avg_ms"] for c in sq]
    ax.plot(n, y, "-o", color=col, lw=2, ms=6, label=lab)
    ax.annotate(f"{y[-1]:.1f} ms" if y[-1] < 100 else f"{y[-1]:.0f} ms", (n[-1], y[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9)
y = [T[("gemv_int8", c)]["cpu1t_ms"] for c in sq]
ax.plot(n, y, "--s", color="#7a7a74", lw=1.5, ms=5, label="CPU 1 thread (INT8 math)")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(n, [str(v) for v in n])
ax.set_xlabel("matrix size N (N x N)"); ax.set_ylabel("time per GEMV (ms, log)")
ax.set_title("U280 kernel time vs size (measured)", loc="left")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(F, "time_vs_size.png"), dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4))
for k, col, lab in (("gemv_fp32", C_FP32, "L1 FP32"), ("gemv_int8", C_INT8, "L2 INT8")):
    y = [T[(k, c)]["gbs"] for c in sq]
    ax.plot(n, y, "-o", color=col, lw=2, ms=6, label=lab)
ax.axhline(14.4, color="#7a7a74", lw=1, ls="--")
ax.text(n[0], 14.4, " one HBM pseudo-channel peak ~14.4 GB/s", va="bottom", fontsize=9, color="#555550")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(n, [str(v) for v in n])
ax.set_xlabel("matrix size N (N x N)"); ax.set_ylabel("achieved bandwidth (GB/s, log)")
ax.set_title("Achieved memory bandwidth: far below one HBM channel", loc="left")
ax.legend(frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(F, "bandwidth_vs_size.png"), dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(pw[:, 0] - t_first, pw[:, 1], color="#3a3a36", lw=1.2)
for r in runs:
    t0, t1 = float(r["start"]) - t_first, float(r["end"]) - t_first
    ax.axvspan(t0, t1, color=C_INT8 if r["kernel"] == "gemv_int8" else C_FP32, alpha=0.12, lw=0)
ax.axhline(P0, color="#7a7a74", lw=1, ls="--")
ax.set_xlabel("time since first run (s)  [blue band = INT8 run, orange band = FP32 run; gaps = xclbin load]")
ax.set_ylabel("board power (W)")
ax.set_title("U280 board power during the sweep", loc="left")
ax.set_ylim(18.5, 24)
fig.tight_layout(); fig.savefig(os.path.join(F, "power_timeline.png"), dpi=180); plt.close(fig)

# ---- CPU baseline (host/cpu_bench.cpp, RAPL package+DRAM of both sockets) vs the U280 -------------
cpu_path = os.path.join(L, "cpu_bench.out")
if os.path.exists(cpu_path):
    cpu = []
    for line in open(cpu_path):
        if not line.startswith("cpu,"):
            continue
        p = line.strip().split(",")
        kv = dict(x.split("=", 1) for x in p if "=" in x)
        cpu.append(dict(case=os.path.basename(p[1]), kind=p[2], mode=p[7], threads=int(kv["threads"]),
                        ms=float(kv["avg_ms"]), gbs=float(kv["GBs"]), j=float(kv["rapl_J_per_gemv"]), w=float(kv["rapl_W"])))
    with open(os.path.join(D, "cpu_vs_fpga.txt"), "w") as f:
        f.write("CPU: 2x Xeon Gold 6226R (32 cores / 64 threads), g++ -O3 -march=native -fopenmp, AVX-512.\n"
                "Energy = RAPL package + DRAM of both sockets (includes CPU idle ~" +
                open(os.path.join(L, "cpu_idle.txt")).read().split(":")[-1].strip()[:5] + " W).\n"
                "cold = W rotated over >= 512 MiB of copies so each GEMV streams from DRAM. FPGA energy = U280 board power x time.\n\n")
        for c in ("sq4096", "llama7b_ffn_11008x4096"):
            f.write(f"{c}\n  {'impl':34s} {'ms/GEMV':>9s} {'GB/s':>7s} {'W':>6s} {'mJ/GEMV':>9s}\n")
            for k, lab in (("gemv_fp32", "U280 L1 FP32"), ("gemv_int8", "U280 L2 INT8")):
                r = T[(k, c)]
                f.write(f"  {lab:34s} {r['avg_ms']:9.3f} {r['gbs']:7.2f} {r['power_w']:6.1f} {r['energy_mj']:9.1f}\n")
            for x in cpu:
                if x["case"] == c and x["mode"] == "cold":
                    f.write(f"  {'CPU ' + x['kind'] + ' ' + str(x['threads']) + ' thr (cold)':34s} {x['ms']:9.3f} {x['gbs']:7.1f} {x['w']:6.0f} {x['j']*1e3:9.1f}\n")
            best = min((x for x in cpu if x["case"] == c and x["mode"] == "cold" and x["kind"] == "int8"), key=lambda x: x["j"])
            pf = T[("gemv_int8", c)]["power_w"]
            f.write(f"  -> best CPU INT8 energy {best['j']*1e3:.1f} mJ ({best['threads']} thr). At {pf:.1f} W the U280 matches it when a\n"
                    f"     GEMV takes <= {best['j']/pf*1e3:.2f} ms (energy crossover); L2 takes {T[('gemv_int8', c)]['avg_ms']:.2f} ms.\n\n")

    c = "sq4096"
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for kind, mk in (("int8", "o"), ("fp32", "s")):
        pts = sorted((x for x in cpu if x["case"] == c and x["mode"] == "cold" and x["kind"] == kind), key=lambda x: x["threads"])
        ax.plot([x["ms"] for x in pts], [x["j"] * 1e3 for x in pts], "-" + mk, color="#7a7a74", lw=1.2, ms=6,
                mfc="white" if kind == "fp32" else "#7a7a74", label=f"CPU {kind.upper()} 1-64 threads")
        for x in pts:
            if x["threads"] in (1, 32):
                ax.annotate(f"{x['threads']}t", (x["ms"], x["j"] * 1e3), xytext=(5, 4), textcoords="offset points", fontsize=8, color="#555550")
    for k, col, lab in (("gemv_fp32", C_FP32, "U280 L1 FP32"), ("gemv_int8", C_INT8, "U280 L2 INT8")):
        r = T[(k, c)]
        ax.plot([r["avg_ms"]], [r["energy_mj"]], "D", color=col, ms=8, label=lab)
    pf = T[("gemv_int8", c)]["power_w"]
    xs = np.logspace(-2, 3, 50)
    ax.plot(xs, pf * xs, ":", color=C_INT8, lw=1)
    ax.text(25, pf * 25 * 0.45, f"U280 at {pf:.1f} W:\nenergy = P x t", fontsize=8, color="#555550")
    bj = min(x["j"] for x in cpu if x["case"] == c and x["mode"] == "cold" and x["kind"] == "int8") * 1e3
    tx = bj / pf
    ax.plot([tx], [bj], "o", mfc="white", mec=C_INT8, ms=9, mew=1.5)
    ax.annotate(f"energy crossover\n{tx:.2f} ms (L3 target)", (tx, bj), xytext=(12, -22), textcoords="offset points",
                fontsize=8, color="#3a3a36")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.05, 1000); ax.set_ylim(5, 2e4)
    ax.set_xlabel("time per 4096x4096 GEMV (ms, log)"); ax.set_ylabel("energy per GEMV (mJ, log)")
    ax.set_title("Time vs energy: U280 kernels and a 2-socket Xeon (measured)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(F, "energy_vs_time_cpu_fpga.png"), dpi=180); plt.close(fig)
    print(open(os.path.join(D, "cpu_vs_fpga.txt")).read())

print(open(os.path.join(D, "power_summary.txt")).read())
print(open(os.path.join(D, "llama7b_token_estimate.txt")).read())
for r in out:
    print(f"{r['kernel']:10s} {r['case']:24s} {r['avg_ms']:9.3f} ms  {r['cycles_per_64w']:6.2f} cyc/64w  "
          f"{r['gbs']:6.2f} GB/s  {r['gmacs']:7.3f} GMAC/s  x{r['speedup_vs_cpu1t']:6.2f} vs CPU1t")
