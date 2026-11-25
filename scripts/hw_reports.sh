#!/usr/bin/env bash
# Post-route summary of a finished hw build, from the repository root:
#   bash scripts/hw_reports.sh gemv_int8_ps [gemv_int8_ps_x4 ...]
# Copies the routed reports to results/hw_build/<name>_*.rpt and prints WNS/TNS,
# the kernel clocks stored in the xclbin, and the kernel utilisation table.
cd "$(dirname "$0")/.."
mkdir -p results/hw_build
for n in "$@"; do
    d=build/$n.hw
    x=$d/$n.xclbin
    echo "================ $n"
    [ -f "$x" ] || { echo "no $x"; continue; }
    ts=$(find $d -name "*timing_summary_routed.rpt" | grep -v "_x/" | head -1)
    [ -z "$ts" ] && ts=$(find $d -name "*hw_bb_locked_timing_summary_routed.rpt" | head -1)
    ku=$(find $d -name "*kernel_util_routed.rpt" | head -1)
    fu=$(find $d -name "*full_util_routed.rpt" | head -1)
    au=$(find $d -name "automation_summary*.txt" | head -1)
    [ -n "$ts" ] && cp "$ts" results/hw_build/${n}_timing_summary_routed.rpt
    [ -n "$ku" ] && cp "$ku" results/hw_build/${n}_kernel_util_routed.rpt
    [ -n "$fu" ] && cp "$fu" results/hw_build/${n}_full_util_routed.rpt
    [ -n "$au" ] && cp "$au" results/hw_build/${n}_automation_summary.txt
    echo "xclbin: $PWD/$x ($(du -h $x | cut -f1))"
    if [ -n "$ts" ]; then
        echo "timing ($ts):"
        grep -A2 "    WNS(ns)" "$ts" | head -3
    fi
    echo "kernel clocks in the xclbin:"
    xclbinutil --info --input "$x" 2>/dev/null | grep -iE -A3 "^ *Name: *(DATA_CLK|KERNEL_CLK)|Frequency" | grep -iE "name|frequency" | head -6
    grep -iE "clock.*(scal|achiev)|frequency.*scal" $d/logs/link/v++.log 2>/dev/null | head -3
    [ -n "$ku" ] && grep -E '^\| (Name|Platform|gemv)|Used Resources' "$ku"
done
