# Vitis HLS C simulation / C synthesis for one kernel.
# Usage (from the repo root):
#   KERNEL=gemv_fp32 STEP=csim  CASE=data/n64_rand_s1 vitis_hls -f scripts/hls.tcl
#   KERNEL=gemv_fp32 STEP=csynth                     vitis_hls -f scripts/hls.tcl
# STEP: csim | csynth | cosim | all   (default all = csim + csynth)

set kernel $::env(KERNEL)
set step   [expr {[info exists ::env(STEP)] ? $::env(STEP) : "all"}]
set case   [expr {[info exists ::env(CASE)] ? $::env(CASE) : "data/n64_rand_s1"}]
set root   [pwd]

open_project -reset build/hls_$kernel
set_top $kernel
add_files kernels/$kernel.cpp -cflags "-I$root/kernels -I$root/common"
add_files -tb tb/tb_$kernel.cpp -cflags "-I$root/kernels -I$root/common"

open_solution -reset sol -flow_target vitis
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.33 -name default   ;# 300 MHz

if {$step eq "csim" || $step eq "all"} {
    csim_design -argv "$root/$case"
}
if {$step eq "csynth" || $step eq "cosim" || $step eq "all"} {
    csynth_design
}
if {$step eq "cosim"} {
    cosim_design -argv "$root/$case"
}
exit
