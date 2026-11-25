#!/usr/bin/env bash
# Bit-width DSE, C synthesis only (build VM, ~2 min per point):
#   source env.sh && bash scripts/dse_bits.sh [WBITS:GS:NPS ...]
#
# Synthesizes kernels/gemv_dse.cpp once per point (one HLS project per point in
# build/hls_dse_w<WBITS>_g<GS>_p<NPS>) and collects II, latency and resources at
# 4096 x 4096 into results/dse/csynth_bits.csv. The csynth reports of the real
# kernels (gemv_int8, gemv_int8_ps, gemv_int8_fx), if built, are added as rows.
set -u
cd "$(dirname "$0")/.."

POINTS=${*:-"8:64:1 8:64:8 4:64:8 4:128:8 4:128:1"}
OUT=results/dse
mkdir -p "$OUT"
CSV=$OUT/csynth_bits.csv
first=1

add_row() {  # <label> <csynth.rpt>
    if [ $first = 1 ]; then
        python3 scripts/parse_csynth.py --csv --header --label "$1" "$2" > "$CSV" && first=0
    else
        python3 scripts/parse_csynth.py --csv --label "$1" "$2" >> "$CSV"
    fi
}

for p in $POINTS; do
    IFS=: read -r wb gs nps <<< "$p"
    name=w${wb}_g${gs}_p${nps}
    echo "== csynth gemv_dse $name"
    KERNEL=gemv_dse STEP=csynth PROJ=hls_dse_$name HLS_CFLAGS="-DWBITS=$wb -DGS=$gs -DNPS=$nps" \
        vitis_hls -f scripts/hls.tcl > "$OUT/csynth_$name.log" 2>&1
    rpt=build/hls_dse_$name/sol/syn/report/csynth.rpt
    if [ -f "$rpt" ]; then
        cp "$rpt" "$OUT/csynth_$name.rpt"
        add_row "dse_$name" "$rpt"
    else
        echo "   FAILED, see $OUT/csynth_$name.log"
    fi
done

for k in gemv_int8 gemv_int8_ps gemv_int8_fx; do
    rpt=build/hls_$k/sol/syn/report/csynth.rpt
    [ -f "$rpt" ] && add_row "$k" "$rpt"
done

echo "== $CSV"
cat "$CSV"
