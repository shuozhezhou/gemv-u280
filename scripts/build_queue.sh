#!/usr/bin/env bash
# Starts hw builds one after another, keeping at most MAXJ v++ links running
# (build VM: 8 vCPU / 62 GB -> 3 links at JOBS=4). Run detached from the repo root:
#   setsid nohup bash scripts/build_queue.sh "KERNEL NCU" ... > build/queue.log 2>&1 &
cd "$(dirname "$0")/.."
MAXJ=${MAXJ:-3}
for item in "$@"; do
    read -r k n <<< "$item"
    while [ "$(pgrep -fc '[u]nwrapped/lnx64.o/v\+\+ -l')" -ge "$MAXJ" ]; do sleep 60; done
    s=$([ "$n" = 1 ] && echo "" || echo "_x$n")
    echo "$(date +%T) start $k NCU=$n"
    # -o: never rebuild an existing .xo (a header copied later is newer than it)
    xo=build/$k.hw/$k.xo
    oflag=$([ -f "$xo" ] && echo "-o $xo")
    setsid nohup bash -lc "source env.sh >/dev/null; make xclbin KERNEL=$k TARGET=hw NCU=$n JOBS=4 $oflag" \
        > build/hw_$k$s.log 2>&1 < /dev/null &
    sleep 300   # let the new link reach v++ -l before counting again
done
echo "$(date +%T) queue empty"
