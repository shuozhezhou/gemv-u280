#!/bin/bash
# CPU baseline sweep on the U280 host (2x Xeon Gold 6226R). Run from the sweep dir that holds data/
# (copy host/cpu_bench.cpp next to this script first).
# Output: logs/cpu_bench.csv-style lines in logs/cpu_bench.out, idle RAPL power in logs/cpu_idle.txt
set -e
g++ -O3 -march=native -fopenmp -o cpu_bench cpu_bench.cpp
mkdir -p logs
read_j() { sudo sh -c 'cat /sys/class/powercap/intel-rapl:0/energy_uj /sys/class/powercap/intel-rapl:1/energy_uj /sys/class/powercap/intel-rapl:0/intel-rapl:0:0/energy_uj /sys/class/powercap/intel-rapl:1/intel-rapl:1:0/energy_uj' | awk '{s+=$1} END {printf "%.6f", s/1e6}'; }
j0=$(read_j); sleep 10; j1=$(read_j)
echo "idle CPU package+dram power over 10 s: $(echo "($j1-$j0)/10" | bc -l) W" | tee logs/cpu_idle.txt
: > logs/cpu_bench.out
for c in sq4096 llama7b_ffn_11008x4096; do
  for t in 1 4 8 16 32 64; do
    for mode in warm cold; do
      n=200; [ $t -le 4 ] && n=40
      sudo ./cpu_bench data/$c int8 $t $n $mode | tee -a logs/cpu_bench.out
      n=100; [ $t -le 4 ] && n=20
      sudo ./cpu_bench data/$c fp32 $t $n $mode | tee -a logs/cpu_bench.out
    done
  done
done
echo CPU BENCH DONE
