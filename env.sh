# Source this on an OCT / CloudLab node:  source env.sh
#
# Where things run on OCT (as of Oct 2026):
#   oct-build VM  : Vitis + Vivado (csim, csynth, sw_emu, hw_emu, v++, host compile)
#   oct-u280 node : XRT runtime only, runs the host binary with the built xclbin
VITIS_VER=${VITIS_VER:-2023.1}

VITIS_SETTINGS=""
for d in /fpga/Xilinx/Vitis /proj/octfpga-PG0/tools/Xilinx/Vitis /tools/Xilinx/Vitis; do
    if [ -f "$d/$VITIS_VER/settings64.sh" ]; then
        VITIS_SETTINGS="$d/$VITIS_VER/settings64.sh"
        break
    fi
done
if [ -n "$VITIS_SETTINGS" ]; then
    source "$VITIS_SETTINGS"
else
    echo "env.sh: Vitis $VITIS_VER not found (fine on a U280 node: run-only)" >&2
fi

if [ -f /opt/xilinx/xrt/setup.sh ]; then
    source /opt/xilinx/xrt/setup.sh > /dev/null
else
    echo "env.sh: XRT not found at /opt/xilinx/xrt" >&2
fi

export PLATFORM=${PLATFORM:-xilinx_u280_gen3x16_xdma_1_202211_1}
echo "Vitis: $(command -v v++)   XRT: ${XILINX_XRT:-missing}   platform: $PLATFORM"
