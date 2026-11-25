#!/usr/bin/env bash
# L3 flow on the oct-build VM (Vitis 2023.1), from the repository root:
#
#   bash scripts/run_l3_vm.sh                 # all stages below, in this order
#   bash scripts/run_l3_vm.sh csim csynth     # selected stages
#   KERNELS=gemv_int8_ps bash scripts/run_l3_vm.sh hw
#
# Stages
#   data    test data (needs python3-numpy) + checksum check
#   csim    HLS C simulation: all committed cases, then 4096 x 4096 (bit-exact)
#   csynth  HLS synthesis; report copied to results/csynth/, summary printed
#   hw      starts `make xclbin TARGET=hw` in the background (detached, ~75-100 min
#           per kernel, kernels one after another); log: build/hw_<kernel>.log
#   sw_emu  host + sw_emu on the committed cases, 4096 x 4096 and data/small/*
#   hw_emu  RTL emulation on n128_rand_s1 and n40x1344_rand_s1
#   dse     bit-width sweep, scripts/dse_bits.sh -> results/dse/csynth_bits.csv
#
# Logs go to results/<stage>/<kernel>_*.log like the L1/L2 results.
set -u
cd "$(dirname "$0")/.."
source env.sh

KERNELS=${KERNELS:-"gemv_int8_ps gemv_int8_fx"}
STAGES=${*:-"data csim csynth hw sw_emu hw_emu dse"}
CASES="data/n64_rand_s1 data/n64_onehot_s1 data/n64_pm127_s1 data/n64x128_zero_s1
       data/n128_rand_s1 data/n100x256_rand_s1 data/n40x1344_rand_s1"
HWEMU_CASES="data/n128_rand_s1 data/n40x1344_rand_s1"
mkdir -p results/csim results/csynth results/sw_emu results/hw_emu build

verdict() {  # <log>: prints PASSED / FAILED from the last TEST line
    if grep -q "TEST PASSED" "$1"; then echo PASSED; else echo "FAILED ($1)"; fi
}

for stage in $STAGES; do
    echo "================ $stage"
    case $stage in
    data)
        python3 -c "import numpy" 2>/dev/null || sudo apt-get install -y python3-numpy
        python3 scripts/gen_data.py --small 100 && python3 scripts/check.py --verify-checksums | tail -1
        ;;
    csim)
        for k in $KERNELS; do
            make csim KERNEL=$k > results/csim/${k}_cases.log 2>&1
            echo "$k csim small cases: $(verdict results/csim/${k}_cases.log)"
            make csim KERNEL=$k CSIM_CASES=data/n4096_rand_s1 > results/csim/${k}_n4096.log 2>&1
            echo "$k csim 4096x4096:   $(verdict results/csim/${k}_n4096.log)"
        done
        ;;
    csynth)
        for k in $KERNELS; do
            make csynth KERNEL=$k > build/csynth_$k.log 2>&1
            rpt=build/hls_$k/sol/syn/report/csynth.rpt
            if [ -f "$rpt" ]; then
                cp "$rpt" results/csynth/${k}_csynth.rpt
                echo "-- $k"
                python3 scripts/parse_csynth.py "$rpt"
            else
                echo "$k csynth FAILED, see build/csynth_$k.log"
            fi
        done
        ;;
    hw)
        cmd=""
        for k in $KERNELS; do
            cmd="$cmd make xclbin KERNEL=$k TARGET=hw > build/hw_$k.log 2>&1;"
        done
        setsid nohup bash -lc "cd $PWD && source env.sh > /dev/null; make host > build/host.log 2>&1; $cmd" \
            > build/hw_chain.log 2>&1 < /dev/null &
        echo "hw build started in the background (pid $!): $KERNELS"
        echo "follow with: tail -f build/hw_<kernel>.log; result build/<kernel>.hw/<kernel>.xclbin"
        ;;
    sw_emu)
        make host > build/host.log 2>&1 || { echo "host build FAILED"; cat build/host.log; exit 1; }
        for k in $KERNELS; do
            for c in $CASES data/n4096_rand_s1; do
                log=results/sw_emu/${k}_$(basename $c).log
                make run KERNEL=$k TARGET=sw_emu CASE=$c > "$log" 2>&1
                echo "$k sw_emu $(basename $c): $(verdict $log)"
            done
            pass=0 fail=0
            for c in data/small/r*; do
                if make run KERNEL=$k TARGET=sw_emu CASE=$c > build/sweep.log 2>&1 &&
                   grep -q "TEST PASSED" build/sweep.log; then
                    pass=$((pass + 1))
                else
                    fail=$((fail + 1)); echo "  sweep FAILED: $c"
                fi
            done
            echo "sw_emu random sweep (data/small, $((pass + fail)) cases): $pass passed, $fail failed" |
                tee results/sw_emu/${k}_random_sweep.txt
        done
        ;;
    hw_emu)
        make host > build/host.log 2>&1
        for k in $KERNELS; do
            for c in $HWEMU_CASES; do
                log=results/hw_emu/${k}_$(basename $c).log
                make run KERNEL=$k TARGET=hw_emu CASE=$c > "$log" 2>&1
                echo "$k hw_emu $(basename $c): $(verdict $log)"
            done
        done
        ;;
    dse)
        bash scripts/dse_bits.sh
        ;;
    *)
        echo "unknown stage $stage"
        exit 1
        ;;
    esac
done
