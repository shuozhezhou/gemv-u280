# Vitis HLS C simulation / C synthesis for one kernel.
# Usage (from the repo root):
#   KERNEL=gemv_fp32 STEP=csim  CASE="data/n64_rand_s1 data/n128_rand_s1" vitis_hls -f scripts/hls.tcl
#   KERNEL=gemv_fp32 STEP=csynth                     vitis_hls -f scripts/hls.tcl
# STEP: csim | csynth | cosim | all   (default all = csim + csynth)
# Optional: HLS_CFLAGS (extra compiler flags, e.g. -DWBITS=4) and PROJ (project
# name, default hls_$KERNEL), used by scripts/dse_bits.sh for one project per point.

set kernel $::env(KERNEL)
set step   [expr {[info exists ::env(STEP)] ? $::env(STEP) : "all"}]
set case   [expr {[info exists ::env(CASE)] ? $::env(CASE) : "data/n64_rand_s1"}]
set root   [pwd]
set cflags [expr {[info exists ::env(HLS_CFLAGS)] ? $::env(HLS_CFLAGS) : ""}]
set proj   [expr {[info exists ::env(PROJ)] ? $::env(PROJ) : "hls_$kernel"}]
set tbargs {}
foreach c $case { lappend tbargs "$root/$c" }
set tbargs [join $tbargs " "]

# HLS project names may not contain '/', so create the project inside build/.
file mkdir build
cd build
open_project -reset $proj
set_top $kernel
add_files $root/kernels/$kernel.cpp -cflags "-I$root/kernels -I$root/common $cflags"
if {[file exists $root/tb/tb_$kernel.cpp]} {
    add_files -tb $root/tb/tb_$kernel.cpp -cflags "-I$root/kernels -I$root/common $cflags"
}

open_solution -reset sol -flow_target vitis
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.33 -name default   ;# 300 MHz

if {$step eq "csim" || $step eq "all"} {
    csim_design -argv $tbargs
}
if {$step eq "csynth" || $step eq "cosim" || $step eq "all"} {
    csynth_design
}
if {$step eq "cosim"} {
    cosim_design -argv $tbargs
}
exit
