# Quantized GEMV for LLM Inference on the Alveo U280

EECE 5698 — FPGAs in the Cloud, Fall 2026
Shuozhe Zhou, Yifan Zhu

> **Project Update 1** — graded snapshot: fork / tag `project-update-1` (frozen 2026-10-29).
> _This section is filled in before submission._

## 1. Project summary

When a large language model generates text, every new token multiplies each weight
matrix by a single activation vector (GEMV). Each weight is read once and used for one
multiply–add, so the speed is set by memory bandwidth, not compute. We build a GEMV
accelerator on the Alveo U280 with Vitis HLS, in layers:

| Layer | Design | Status |
|---|---|---|
| L0 | Repository, file format, NumPy golden model, XRT host | in progress |
| L1 | FP32 GEMV, one HBM channel, pipelined dot product | not started |
| L2 | Group-wise INT8 weights (64 per group, FP32 scale), bit-exact | not started |
| L3 | Partial sums (II = 1), dataflow, several HBM channels | planned |
| L4 | INT4, real Llama-2-7B weights, GPU comparison | optional |

Problem size: W is 4096 × 4096 (one attention projection of a 7B model).

## 2. What is implemented for this update

_TBD_

## 3. Repository layout

```
kernels/   HLS kernels (gemv_fp32.cpp, gemv_int8.cpp), interface gemv.h, v++ link configs (*.cfg)
host/      XRT host program: loads a test case, runs the kernel, times it, checks the output
tb/        Vitis HLS C testbenches (one per kernel)
common/    binio.h — file I/O and output comparison shared by host and testbenches
scripts/   gen_data.py (inputs + golden outputs), hls.tcl (csim/csynth)
data/      small test cases with golden outputs; FORMAT.md describes the files
results/   logs and reports from our runs
Makefile   build / run for sw_emu, hw_emu, hw
env.sh     tool setup on OCT nodes
```

## 4. Kernel interface

```cpp
// y = W x, W is M x K row-major
extern "C" void gemv_fp32(const float* W, const float* x, float* y, int M, int K);
extern "C" void gemv_int8(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                          float sx, float* y, int M, int K);
```

HBM mapping is in `kernels/*.cfg`. Data file format and quantization formulas: `data/FORMAT.md`.

## 5. How to run

### 5.1 Environment

- Open Cloud Testbed / CloudLab, profile `oct-u280`, **Tool Version 2023.1**, Ubuntu 22.04
  (hardware runs). HLS-only steps also work on the OCT build machine.
- Platform `xilinx_u280_gen3x16_xdma_1_202211_1`.

```bash
git clone <repo-url> && cd gemv-u280
source env.sh
```

### 5.2 Steps

_TBD — commands per step (data, csim, csynth, sw_emu, hw_emu, hw)._

## 6. Results

_TBD_

## 7. Next steps

_TBD_
