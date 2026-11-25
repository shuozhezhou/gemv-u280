#!/usr/bin/env python3
"""Figures for section 10 / report section 4.4.

  results/l3/figures/energy_vs_time_all.png   time and energy of one 4096 x 4096 GEMV, every design
  results/spec_decode/batch_time_vs_nb.png    batched kernel: time per call vs vectors per call
usage: python3 scripts/plot_nov25.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FPGA, CPU, MUTED = "#2a6fdb", "#d9822b", "#6b7280"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444"})

# ---------------------------------------------------------------- figure 3
# (label, ms, mJ, device) measured: sections 9 and 10 of the README
pts = [("L1 FP32", 391.669, 8918, "U280"), ("L2 INT8", 7.500, 174.6, "U280"),
       ("L3a INT8", 1.311, 32.0, "U280"), ("L3c INT4", 0.677, 16.7, "U280"),
       ("4 CUs", 0.371, 11.6, "U280"), ("8 CUs", 0.240, 8.4, "U280"),
       ("CPU INT8 1 thr", 2.532, 250.4, "CPU"), ("CPU INT8 8 thr", 0.346, 55.6, "CPU"),
       ("CPU INT8 32 thr", 0.102, 35.0, "CPU"), ("CPU FP32 32 thr", 0.300, 93.5, "CPU")]
off = {"L1 FP32": (-8, 6, "right"), "L2 INT8": (6, 4, "left"), "L3a INT8": (6, 4, "left"),
       "L3c INT4": (6, 4, "left"), "4 CUs": (6, -2, "left"), "8 CUs": (6, -4, "left"),
       "CPU INT8 1 thr": (6, 2, "left"), "CPU INT8 8 thr": (6, 3, "left"),
       "CPU INT8 32 thr": (6, 5, "left"), "CPU FP32 32 thr": (6, 3, "left")}
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
for dev, col, mk in (("U280", FPGA, "o"), ("CPU", CPU, "s")):
    xs = [p[1] for p in pts if p[3] == dev]; ys = [p[2] for p in pts if p[3] == dev]
    ax.scatter(xs, ys, s=34, color=col, marker=mk, edgecolor="white", linewidth=1.2, zorder=3,
               label="U280 (board power)" if dev == "U280" else "2 x Xeon 6226R (RAPL)")
for lab, ms, mj, dev in pts:
    dx, dy, ha = off[lab]
    ax.annotate(lab, (ms, mj), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8, color="#222")
ax.axhline(34.2, color=MUTED, lw=1, ls=":", zorder=1)
ax.text(300, 37, "best CPU energy, 34.2 mJ", color=MUTED, fontsize=7.5, ha="right")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.06, 800)
ax.set_xlabel("time per 4096 x 4096 GEMV (ms, log)"); ax.set_ylabel("energy per GEMV (mJ, log)")
ax.grid(True, which="major", color="#e5e7eb", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
os.makedirs(os.path.join(R, "l3", "figures"), exist_ok=True)
fig.savefig(os.path.join(R, "l3", "figures", "energy_vs_time_all.png"))
print("wrote energy_vs_time_all.png")


# ---------------------------------------------------------------- figure 4
def load(path):
    rows = [r for r in csv.reader(open(path)) if r and r[0] and not r[0].startswith(("[", "shape", "BENCH"))]
    return [dict(shape=r[0], nb=int(r[3]), ms=float(r[4]), ok=r[8]) for r in rows]


old = load(os.path.join(R, "spec_decode", "kernel_b_strided", "logs_spec", "bench_batch.csv"))
new = load(os.path.join(R, "spec_decode", "kernel_bw", "logs_spec", "bench_batch_bw.csv"))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), dpi=200)
for ax, shape, title in ((axs[0], "gpt2_mlp_3072x768", "3072 x 768 (GPT-2 MLP)"),
                         (axs[1], "sq4096", "4096 x 4096")):
    for data, col, lab, mk in ((old, MUTED, "strided store (b)", "s"), (new, FPGA, "burst store (bw)", "o")):
        d = [r for r in data if r["shape"] == shape]
        ax.plot([r["nb"] for r in d], [r["ms"] for r in d], color=col, lw=2, marker=mk, ms=4, label=lab)
    ax.set_title(title, fontsize=9); ax.set_xlabel("vectors per call (nb)")
    ax.set_xticks(range(1, 9)); ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", color="#e5e7eb", lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel("kernel time per call (ms)")
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(R, "spec_decode", "batch_time_vs_nb.png"))
print("wrote batch_time_vs_nb.png")
