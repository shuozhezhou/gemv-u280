#!/usr/bin/env bash
# L3 runs on a U280 node (XRT only), from a copy of the repository that contains
# build/gemv_host and build/<kernel>.hw/<kernel>.xclbin:
#
#   bash scripts/run_l3_board.sh [kernel ...]      # default: gemv_int8_ps gemv_int8_fx
#
# Every committed case + the three 4096 x 4096 cases, 100 timed runs each;
# logs in results/hw/<kernel>_<case>.log (same names as the L1/L2 runs).
set -u
cd "$(dirname "$0")/.."
KERNELS=${*:-"gemv_int8_ps gemv_int8_fx"}
CASES="data/n64_rand_s1 data/n64_onehot_s1 data/n64_pm127_s1 data/n64x128_zero_s1
       data/n128_rand_s1 data/n100x256_rand_s1 data/n40x1344_rand_s1
       data/n4096_rand_s1 data/n4096_rand_s2 data/n4096_rand_s3"

python3 -c "import numpy" 2>/dev/null || sudo apt-get install -y python3-numpy
[ -d data/n4096_rand_s3 ] || python3 scripts/gen_data.py --small 0
python3 scripts/check.py --verify-checksums | tail -1
source /opt/xilinx/xrt/setup.sh > /dev/null
unset XCL_EMULATION_MODE
mkdir -p results/hw

for k in $KERNELS; do
    xclbin=build/$k.hw/$k.xclbin
    [ -f "$xclbin" ] || { echo "missing $xclbin"; continue; }
    for c in $CASES; do
        log=results/hw/${k}_$(basename $c).log
        ./build/gemv_host "$xclbin" "$k" "$c" 100 > "$log" 2>&1
        printf "%-14s %-20s %s  %s\n" "$k" "$(basename $c)" \
            "$(grep -o 'TEST [A-Z]*' "$log" | tail -1)" "$(grep '^kernel time' "$log")"
    done
done
