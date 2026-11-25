#!/bin/bash
# Batched kernel + prompt-lookup speculative decoding on the U280, with board power sampling.
# usage: [KERNEL=gemv_int8_ps_bw] [PPLCHECK=] [SKIPPPL=1] [SKIPCHECK=1] bash scripts/run_spec_board.sh <xclbin> <model_dir> <tag> [tokens]
X=$1; MODEL=$2; T=$3; N=${4:-128}
source /opt/xilinx/xrt/setup.sh > /dev/null; unset XCL_EMULATION_MODE
D=logs_spec; mkdir -p $D
BDF=$(xbutil examine 2>/dev/null | grep -oE '\[[0-9a-f:.]+\]' | head -1 | tr -d '[]')
P=$D/power_$T.csv; echo "time,watts" > $P
( while true; do w=$(xbutil examine -d "$BDF" -r electrical 2>/dev/null | grep -m1 -iE '^\s*Power\s*:' | grep -oE '[0-9]+(\.[0-9]+)?'); echo "$(date +%s.%N),$w" >> $P; sleep 1; done ) &
S=$!; trap 'kill $S 2>/dev/null' EXIT
R=$D/runs_$T.csv; echo "name,start,end" > $R
C="python3 host/llm_fpga.py --model $MODEL --xclbin $X --kernel ${KERNEL:-gemv_int8_ps_b} --backend fpga --smooth 0.5 --calib calib_text.txt"
go() { n=$1; shift; t0=$(date +%s.%N); $C "$@" < /dev/null > $D/${n}_$T.log 2>&1; echo "$n,$t0,$(date +%s.%N)" >> $R; grep -h "RESULT\|TEXT\|Traceback\|Error" $D/${n}_$T.log; }
# perplexity, 8 tokens per weight pass (must equal the token-by-token gemv_int8_ps result)
[ -z "$SKIPPPL" ] && go ppl_b8 ${PPLCHECK---check} --ppl ppl_text.txt --max-ppl-tokens 256 --batch 8
i=0
while IFS= read -r PR; do
  i=$((i+1))
  go gen_p${i}_base   --prompt "$PR" --tokens $N --save $D/ref_p${i}_$T.npz
  go gen_p${i}_spec7  --prompt "$PR" --tokens $N --spec 7 --ref $D/ref_p${i}_$T.npz
  go gen_p${i}_spec7n2 --prompt "$PR" --tokens $N --spec 7 --min-ngram 2 --ref $D/ref_p${i}_$T.npz
  go gen_p${i}_spec3n2 --prompt "$PR" --tokens $N --spec 3 --min-ngram 2 --ref $D/ref_p${i}_$T.npz
done < prompts_spec.txt
[ -z "$SKIPCHECK" ] && go gen_p1_spec7_check --prompt "$(head -1 prompts_spec.txt)" --tokens $N --spec 7 --check --ref $D/ref_p1_$T.npz
echo SPEC DONE $T
