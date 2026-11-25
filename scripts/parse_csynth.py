#!/usr/bin/env python3
"""Extract the numbers we report from a Vitis HLS csynth.rpt (2023.1 text format).

    python3 scripts/parse_csynth.py build/hls_gemv_int8_ps/sol/syn/report/csynth.rpt
    python3 scripts/parse_csynth.py --csv --label w4_g64_p8 <csynth.rpt> [--header]

Reads the "Performance & Resource Estimates" table: the top module row gives
latency, interval and resources (BRAM_18K, DSP, FF, LUT, URAM; whole kernel incl.
AXI adapters); every pipelined loop row ('o name', Pipelined = yes) gives its II.
"""
import argparse
import re
import sys

FIELDS = ["label", "latency_cycles", "latency_ms_300MHz", "interval", "slack_ns",
          "loop_II", "BRAM_18K", "DSP", "FF", "LUT", "URAM"]


def num(cell):
    """'70 (1%)' -> 70, '-' -> 0."""
    m = re.match(r"\s*(-?[\d.]+)", cell)
    return float(m.group(1)) if m else 0


def parse(path):
    rows = []
    with open(path) as f:
        in_table = False
        for line in f:
            if "Performance & Resource Estimates" in line:
                in_table = True
                continue
            if in_table and line.startswith("== "):
                break
            if in_table and line.strip().startswith("|") and ("+ " in line or "o " in line):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) < 14 or cells[0].startswith("Modules"):
                    continue
                rows.append(cells)
    if not rows:
        sys.exit(f"{path}: no performance table found")
    top = rows[0]
    # columns: name, issue, slack, latency(cycles), latency(ns), iteration latency,
    #          interval, trip count, pipelined, BRAM, DSP, FF, LUT, URAM
    loops = []
    for c in rows:
        if c[0].startswith("o ") and c[8] == "yes":
            loops.append(f"{c[0][2:].strip()}={c[6]}")
    lat = int(num(top[3]))
    return {
        "latency_cycles": lat,
        "latency_ms_300MHz": f"{lat * 3.33e-6:.3f}",
        "interval": top[6],
        "slack_ns": top[2],
        "loop_II": " ".join(loops),
        "BRAM_18K": int(num(top[9])),
        "DSP": int(num(top[10])),
        "FF": int(num(top[11])),
        "LUT": int(num(top[12])),
        "URAM": int(num(top[13])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--csv", action="store_true", help="one CSV row")
    ap.add_argument("--header", action="store_true", help="print the CSV header first")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    r = parse(args.report)
    r["label"] = args.label
    if args.csv:
        if args.header:
            print(",".join(FIELDS))
        print(",".join(f'"{r[k]}"' if k == "loop_II" else str(r[k]) for k in FIELDS))
    else:
        for k in FIELDS[1:]:
            print(f"{k:18s} {r[k]}")


if __name__ == "__main__":
    main()
