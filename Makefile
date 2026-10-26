# GEMV on Alveo U280 — Vitis 2023.1 build flow.
#
#   source env.sh
#   make data                                   # generate test inputs + golden outputs
#   make csim   KERNEL=gemv_fp32                # HLS C simulation (64x64)
#   make csynth KERNEL=gemv_fp32                # HLS report: II, latency, resources
#   make run    KERNEL=gemv_fp32 TARGET=sw_emu  CASE=data/n64_rand_s1
#   make run    KERNEL=gemv_fp32 TARGET=hw_emu  CASE=data/n128_rand_s1
#   make run    KERNEL=gemv_fp32 TARGET=hw      CASE=data/n4096_rand_s1
#
# TARGET: sw_emu | hw_emu | hw        KERNEL: gemv_fp32 | gemv_int8

TARGET   ?= sw_emu
KERNEL   ?= gemv_fp32
PLATFORM ?= xilinx_u280_gen3x16_xdma_1_202211_1
CASE     ?= data/n64_rand_s1
ITERS    ?= 100
PYTHON   ?= python3

BUILD  := build/$(KERNEL).$(TARGET)
XO     := $(BUILD)/$(KERNEL).xo
XCLBIN := $(BUILD)/$(KERNEL).xclbin
HOST   := build/gemv_host

VPP_FLAGS := -t $(TARGET) --platform $(PLATFORM) --save-temps --temp_dir $(BUILD)/_x \
             --log_dir $(BUILD)/logs --report_dir $(BUILD)/reports
CXXFLAGS  := -std=c++17 -O2 -Wall -Icommon -I$(XILINX_XRT)/include
LDFLAGS   := -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread

.PHONY: all data csim csynth cosim xo xclbin host run emconfig clean cleanall

all: xclbin host

data:
	$(PYTHON) scripts/gen_data.py

# ---- HLS (C simulation / synthesis) -----------------------------------------
csim:
	KERNEL=$(KERNEL) STEP=csim CASE=$(CASE) vitis_hls -f scripts/hls.tcl
csynth:
	KERNEL=$(KERNEL) STEP=csynth vitis_hls -f scripts/hls.tcl
cosim:
	KERNEL=$(KERNEL) STEP=cosim CASE=$(CASE) vitis_hls -f scripts/hls.tcl

# ---- v++ compile and link ---------------------------------------------------
xo: $(XO)
$(XO): kernels/$(KERNEL).cpp kernels/gemv.h
	mkdir -p $(BUILD)
	v++ -c $(VPP_FLAGS) -k $(KERNEL) -I kernels -o $@ $<

xclbin: $(XCLBIN)
$(XCLBIN): $(XO) kernels/$(KERNEL).cfg
	v++ -l $(VPP_FLAGS) --config kernels/$(KERNEL).cfg -o $@ $<

# ---- host -------------------------------------------------------------------
host: $(HOST)
$(HOST): host/host.cpp common/binio.h
	mkdir -p build
	g++ $(CXXFLAGS) $< -o $@ $(LDFLAGS)

emconfig: emconfig.json
emconfig.json:
	emconfigutil --platform $(PLATFORM) --od .

# ---- run --------------------------------------------------------------------
ifeq ($(TARGET),hw)
run: $(XCLBIN) $(HOST)
	unset XCL_EMULATION_MODE; ./$(HOST) $(XCLBIN) $(KERNEL) $(CASE) $(ITERS)
else
run: $(XCLBIN) $(HOST) emconfig.json
	XCL_EMULATION_MODE=$(TARGET) ./$(HOST) $(XCLBIN) $(KERNEL) $(CASE) 1
endif

clean:
	rm -rf build/$(KERNEL).$(TARGET) emconfig.json .Xil *.log *.jou xrt.run_summary
cleanall:
	rm -rf build emconfig.json .Xil _x *.log *.jou xrt.run_summary
