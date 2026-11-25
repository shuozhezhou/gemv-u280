#!/bin/bash
# Hardware size sweep + card power on the U280 node. Run from /mydata/sweep.
# Writes logs/<kernel>_<case>.log, logs/runs.csv (start/end time of every run) and logs/power.csv.
source /opt/xilinx/xrt/setup.sh > /dev/null
unset XCL_EMULATION_MODE
BDF=$(xbutil examine 2>/dev/null | grep -oE '\[[0-9a-f:.]+\]' | head -1 | tr -d '[]')
mkdir -p logs
echo "time,watts" > logs/power.csv
echo "kernel,case,iters,start,end" > logs/runs.csv

# sample card power once per second in the background
(
  while true; do
    w=$(xbutil examine -d "$BDF" -r electrical 2>/dev/null | grep -m1 -iE '^\s*Power\s*:' | grep -oE '[0-9]+(\.[0-9]+)?')
    echo "$(date +%s.%N),${w}" >> logs/power.csv
    sleep 1
  done
) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

sleep 15   # idle baseline

run() {  # kernel case iters
  local k=$1 c=$2 n=$3 x
  x=build/gemv_fp32.hw/gemv_fp32.xclbin
  [ "$k" = gemv_int8 ] && x=build/gemv_int8.hw/gemv_int8.xclbin
  local t0=$(date +%s.%N)
  ./build/gemv_host "$x" "$k" "data/$c" "$n" > "logs/${k}_${c}.log" 2>&1
  local t1=$(date +%s.%N)
  echo "$k,$c,$n,$t0,$t1" >> logs/runs.csv
  echo "$(date +%T) $k $c: $(grep -E 'kernel time' logs/${k}_${c}.log | head -1) | $(tail -1 logs/${k}_${c}.log)"
}

for c in sq256 sq512 sq1024 sq2048 gpt2_attn_768x768 gpt2_mlp_3072x768; do
  run gemv_int8 "$c" 1000
  run gemv_fp32 "$c" 200
done
for c in sq4096 llama7b_attn_4096x4096; do
  run gemv_int8 "$c" 500
  run gemv_fp32 "$c" 50
done
run gemv_int8 llama7b_ffn_11008x4096 300
run gemv_fp32 llama7b_ffn_11008x4096 30

# long steady runs for a clean power reading (about 20 s each)
run gemv_int8 sq4096 2700
run gemv_fp32 sq4096 50

sleep 10   # idle tail
echo "SWEEP DONE"
