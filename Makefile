# GEMV on Alveo U280 — Vitis 2023.1 build flow.
#
#   source env.sh
#   make data                                   # generate test inputs + golden outputs
#   make csim   KERNEL=gemv_fp32                # HLS C simulation, all small cases
#   make csynth KERNEL=gemv_fp32                # HLS report: II, latency, resources
#   make run    KERNEL=gemv_fp32 TARGET=sw_emu  CASE=data/n64_rand_s1
#   make run    KERNEL=gemv_fp32 TARGET=hw_emu  CASE=data/n128_rand_s1
#   make run    KERNEL=gemv_fp32 TARGET=hw      CASE=data/n4096_rand_s1
#   make xclbin KERNEL=gemv_int8_ps TARGET=hw NCU=8 TAG=_aligned  # kernels/gemv_int8_ps_x8_aligned.cfg
#   make host_multi                             # host for NCU > 1 (host/host_multi.cpp)
#
# TARGET: sw_emu | hw_emu | hw
# KERNEL: gemv_fp32 | gemv_int8 | gemv_int8_ps (L3a) | gemv_int8_fx (L3b) | gemv_int4_ps (L3c)

TARGET   ?= sw_emu
KERNEL   ?= gemv_fp32
PLATFORM ?= xilinx_u280_gen3x16_xdma_1_202211_1
CASE     ?= data/n64_rand_s1
ITERS    ?= 100
# parallel Vivado jobs per hw link (2 builds x 4 fit in 8 vCPU / 64 GB)
JOBS     ?= 4
# All committed small cases (csim runs every one of them).
CSIM_CASES ?= data/n64_rand_s1 data/n64_onehot_s1 data/n64_pm127_s1 data/n64x128_zero_s1               data/n128_rand_s1 data/n100x256_rand_s1 data/n40x1344_rand_s1 data/n24x2688_rand_s1
PYTHON   ?= python3

# Compute units: NCU = 1 uses kernels/<kernel>.cfg; NCU = n > 1 uses
# kernels/<kernel>_x<n>.cfg and links into build/<kernel>_x<n>.<target>/, reusing
# the single-CU .xo of build/<kernel>.<target>/.
# TAG selects a config variant, e.g. NCU=8 TAG=_aligned -> kernels/<kernel>_x8_aligned.cfg.
NCU    ?= 1
TAG    ?=
SUFFIX := $(if $(filter-out 1,$(NCU)),_x$(NCU))$(TAG)
XODIR  := build/$(KERNEL).$(TARGET)
BUILD  := build/$(KERNEL)$(SUFFIX).$(TARGET)
XO     := $(XODIR)/$(KERNEL).xo
XCLBIN := $(BUILD)/$(KERNEL)$(SUFFIX).xclbin
CFG    := kernels/$(KERNEL)$(SUFFIX).cfg
HOST   := build/gemv_host
HOST_MULTI := build/gemv_host_multi

VPP_FLAGS  := -t $(TARGET) --platform $(PLATFORM) --save-temps
XO_FLAGS   := --temp_dir $(XODIR)/_x --log_dir $(XODIR)/logs --report_dir $(XODIR)/reports
LINK_FLAGS := --temp_dir $(BUILD)/_x --log_dir $(BUILD)/logs --report_dir $(BUILD)/reports
CXXFLAGS  := -std=c++17 -O2 -Wall -Icommon -I$(XILINX_XRT)/include
LDFLAGS   := -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread

.PHONY: all data csim csynth cosim xo xclbin host host_multi run emconfig clean cleanall

all: xclbin host

data:
	$(PYTHON) scripts/gen_data.py --small 100
	$(PYTHON) scripts/check.py --verify-checksums

# ---- HLS (C simulation / synthesis) -----------------------------------------
csim:
	KERNEL=$(KERNEL) STEP=csim CASE="$(CSIM_CASES)" vitis_hls -f scripts/hls.tcl
csynth:
	KERNEL=$(KERNEL) STEP=csynth vitis_hls -f scripts/hls.tcl
cosim:
	KERNEL=$(KERNEL) STEP=cosim CASE=$(CASE) vitis_hls -f scripts/hls.tcl

# ---- v++ compile and link ---------------------------------------------------
xo: $(XO)
$(XO): kernels/$(KERNEL).cpp $(wildcard kernels/*.h)
	mkdir -p $(XODIR)
	v++ -c $(VPP_FLAGS) $(XO_FLAGS) -k $(KERNEL) -I kernels -o $@ $<

xclbin: $(XCLBIN)
$(XCLBIN): $(XO) $(CFG)
	mkdir -p $(BUILD)
	v++ -l $(VPP_FLAGS) $(LINK_FLAGS) --jobs $(JOBS) --config $(CFG) -o $@ $<

# ---- host -------------------------------------------------------------------
host: $(HOST)
$(HOST): host/host.cpp common/binio.h
	mkdir -p build
	g++ $(CXXFLAGS) $< -o $@ $(LDFLAGS)
host_multi: $(HOST_MULTI)
$(HOST_MULTI): host/host_multi.cpp common/binio.h
	mkdir -p build
	g++ $(CXXFLAGS) $< -o $@ $(LDFLAGS)

emconfig: emconfig.json
emconfig.json:
	emconfigutil --platform $(PLATFORM) --od .

# ---- run --------------------------------------------------------------------
# For hw the xclbin is an order-only prerequisite: it is built if missing, but a
# newer timestamp on a source file (e.g. after copying the repo) does not start
# a 2-hour rebuild. Run `make xclbin TARGET=hw` explicitly to rebuild.
# XRT looks for emconfig.json next to the host binary or in $EMCONFIG_PATH.
ifeq ($(TARGET),hw)
run: $(HOST) | $(XCLBIN)
	unset XCL_EMULATION_MODE; ./$(HOST) $(XCLBIN) $(KERNEL) $(CASE) $(ITERS)
else
run: $(XCLBIN) $(HOST) emconfig.json
	XCL_EMULATION_MODE=$(TARGET) EMCONFIG_PATH=$(CURDIR) ./$(HOST) $(XCLBIN) $(KERNEL) $(CASE) 1
endif

clean:
	rm -rf $(BUILD) emconfig.json .Xil *.log *.jou xrt.run_summary
cleanall:
	rm -rf build emconfig.json .Xil _x *.log *.jou xrt.run_summary
