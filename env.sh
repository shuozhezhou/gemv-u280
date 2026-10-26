# Source this on an OCT / CloudLab node:  source env.sh
# On oct-u280 nodes the tools are already exported by /etc/profile (login shell);
# this file makes non-login shells and the build machine work the same way.
VITIS_VER=${VITIS_VER:-2023.1}
TOOLS=/proj/octfpga-PG0/tools/Xilinx

if [ -f "$TOOLS/Vitis/$VITIS_VER/settings64.sh" ]; then
    source "$TOOLS/Vitis/$VITIS_VER/settings64.sh"
elif [ -f "/tools/Xilinx/Vitis/$VITIS_VER/settings64.sh" ]; then
    source "/tools/Xilinx/Vitis/$VITIS_VER/settings64.sh"
else
    echo "env.sh: Vitis $VITIS_VER settings64.sh not found" >&2
fi

if [ -f /opt/xilinx/xrt/setup.sh ]; then
    source /opt/xilinx/xrt/setup.sh > /dev/null
else
    echo "env.sh: XRT not found at /opt/xilinx/xrt" >&2
fi

export PLATFORM=${PLATFORM:-xilinx_u280_gen3x16_xdma_1_202211_1}
echo "Vitis: $(command -v v++)   XRT: ${XILINX_XRT:-missing}   platform: $PLATFORM"
