#!/usr/bin/env python3
"""Summarize speculative-decoding runs (scripts/run_spec_board.sh logs) into a CSV.

usage: python3 scripts/summarize_spec.py <logs_spec dir> <tag> [out.csv]
Per run: tokens, passes over the weights, decode tokens/s, decode kernel ms/token, whether the
output equals the token-by-token run, and the mean board power while the run was active
(power_<tag>.csv sampled every ~1 s, runs_<tag>.csv start/end times) -> energy per new token
over the generation phase (mean power x decode wall time / tokens).
"""
import csv, glob, os, re, sys

d, tag = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(d, f"spec_summary_{tag}.csv")
power = [(float(r["time"]), float(r["watts"])) for r in csv.DictReader(open(os.path.join(d, f"power_{tag}.csv")))
         if r["watts"]]
runs = {r["name"]: (float(r["start"]), float(r["end"])) for r in csv.DictReader(open(os.path.join(d, f"runs_{tag}.csv")))}
prompts = {"p1": "text", "p2": "code", "p3": "qa"}
rows = []
for f in sorted(glob.glob(os.path.join(d, f"gen_p*_{tag}.log"))):
    name = os.path.basename(f)[: -len(f"_{tag}.log")]
    res = [l for l in open(f, encoding="utf-8", errors="replace") if l.startswith("RESULT gen")]
    if not res:
        continue
    s = res[-1]
    g = lambda k, typ=float: typ(re.search(rf"\b{k}=([0-9.]+)", s).group(1)) if re.search(rf"\b{k}=([0-9.]+)", s) else None
    m = re.match(r"gen_(p\d)_(\w+)", name)
    mode = m.group(2)
    t0, t1 = runs.get(name, (0, 0))
    w = [p for t, p in power if t0 <= t <= t1]
    pw = sum(w) / len(w) if w else float("nan")
    ntok = g("new_tokens", int)
    dwall = g("decode_wall")
    rows.append(dict(
        prompt=prompts.get(m.group(1), m.group(1)), mode=mode, prompt_tokens=g("prompt_tokens", int), new_tokens=ntok,
        spec_k=g("spec_k", int) or 0, verify_passes=g("verify_passes", int), accepted=g("accepted", int),
        drafts=g("drafts", int), decode_passes=g("decode_passes", int),
        decode_tok_per_s=g("decode_tok_per_s"), decode_kernel_ms_per_tok=None, new_tok_per_s=g("new_tok_per_s"),
        identical=("identical_to_ref=True" in s) if "identical_to_ref" in s else "",
        board_w=round(pw, 2), samples=len(w),
        mj_per_token=round(pw * dwall / ntok * 1e3, 1) if dwall and w else None))
    km = re.search(r"decode_kernel=[0-9.]+s \(([0-9.]+) ms/token\)", s)
    rows[-1]["decode_kernel_ms_per_tok"] = float(km.group(1)) if km else None
with open(out, "w", newline="") as fo:
    wr = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)
for r in rows:
    print(r)
print("wrote", out)
