# Quantized GEMV for LLM Inference on the Alveo U280

EECE 5698 — FPGAs in the Cloud, Fall 2026

Shuozhe Zhou, Yifan Zhu

> **Project Update 2** (due 2026-11-24). The graded snapshot is branch
> [`project-update2`](https://github.com/shuozhezhou/gemv-u280/tree/project-update2);
> it does not change after the deadline. New in this update: accuracy metrics against
> the true result (section 7), a bit-width design-space exploration (section 8), where
> the input data comes from (section 3.1), who does what (section 10) and the code we
> started from (section 11). `main` keeps moving.
> The report for this update is
> [`reports/ProjectUpdate2-ShuozheZhou-YifanZhu.pdf`](reports/ProjectUpdate2-ShuozheZhou-YifanZhu.pdf).
>
> Project Update 1 is frozen as branch and release
> [`project-update-1`](https://github.com/shuozhezhou/gemv-u280/tree/project-update-1).

## 1. Project summary

When a large language model generates text, every new token multiplies each weight
matrix by a single activation vector (GEMV, `y = W x`). Each weight is read once and
used for one multiply–add, so on a well-designed accelerator the speed is set by memory
bandwidth. Storing the weights as **INT8 with one FP32 scale per group of 64 weights**
cuts the bytes read by about 4× compared with FP32, at a small accuracy cost.

We are building a GEMV accelerator for this format on the **AMD Alveo U280** with
**Vitis HLS 2023.1**, on the Open Cloud Testbed (OCT). The problem size is
W = 4096 × 4096, one attention projection of a 7B model such as Llama 2.
Every design is checked against a software golden model **bit for bit**.

The work is split into layers; each starts from a design that passes all tests.

| Layer | Design | Status |
|---|---|---|
| L0 | Repository, file format, NumPy golden model, XRT host, tests | **done** |
| L1 | FP32 GEMV, one HBM channel, pipelined dot product | **done — bit-exact on the U280** |
| L2 | Group-wise INT8 weights (64 per group, FP32 scale) | **done — bit-exact on the U280, 52× faster than L1** |
| L3 | Partial sums for II = 1, dataflow, several HBM channels | next |
| L4 | INT4, real Llama-2-7B weights, GPU comparison | optional |

## 2. What is implemented

Update 1 (L0 to L2):

- **Golden model and test data** (`scripts/gen_data.py`): fixed-seed inputs, the
  quantization, and golden outputs computed in float32 in exactly the order the
  kernels use, so results can be compared bit for bit. Corner cases (one-hot rows,
  all values ±127 = largest group sum, all-zero groups and rows), random cases,
  100 random small shapes, and three full 4096 × 4096 cases. MD5 checksums prove the
  large files regenerate identically on any machine.
- **L1 FP32 kernel** (`kernels/gemv_fp32.cpp`): x is copied into on-chip memory, then
  each row is a pipelined dot product. Passes csim, sw_emu, hw_emu and runs
  **bit-exact on the U280 at 4096 × 4096** (3 seeds), 300 MHz, 391.7 ms.
- **L2 INT8 kernel** (`kernels/gemv_int8.cpp`): one 512-bit HBM word = one group of
  64 int8 weights; 64 int8 multiplies + an int32 adder tree per cycle of the group
  loop, then one FP32 scale and accumulate. Passes csim, sw_emu (incl. 100 random
  shapes) and hw_emu, and runs **bit-exact on the U280 at 4096 × 4096** (3 seeds),
  300 MHz, 7.50 ms — **52× faster than L1**.
- **HLS C testbenches** (`tb/`), one per kernel, running all test cases.
- **XRT host program** (`host/host.cpp`): loads a test case, runs the kernel 100 times
  after a warm-up, reports kernel time, bandwidth and PCIe time, checks the output
  against the golden file, and runs the same computation on the CPU (single thread,
  same order of operations) as the software version / CPU baseline.
- **Makefile** for csim, csynth, sw_emu, hw_emu and hw, plus all logs and reports of
  our runs in `results/`.

Update 2 (no new kernel; measurements and analysis on the Update 1 designs):

- **Accuracy metrics** (`scripts/accuracy.py`): relative L2 error, max error, cosine
  similarity and SNR of the FP32 and INT8 outputs against the true result
  (float64, original weights), for all test cases (section 7).
- **Bit-width design-space exploration** (software): weights 8/6/4/3/2 bits ×
  group size 32/64/128/per-row, activations 8/6/4 bits (section 8).
- **Input data documented** (section 3.1) and **contributions / code sources**
  stated (sections 10 and 11).

## 3. Files

```
kernels/gemv.h            kernel interfaces (shared by kernels and testbenches)
kernels/gemv_fp32.cpp     L1 FP32 kernel
kernels/gemv_int8.cpp     L2 INT8 kernel
kernels/gemv_fp32.cfg     v++ link config: W -> HBM[0], x,y -> HBM[1]
kernels/gemv_int8.cfg     v++ link config: Wq -> HBM[0], s -> HBM[1], xq,y -> HBM[2]
tb/tb_gemv_fp32.cpp       HLS C testbench, FP32 (tolerance 1e-6 x max|y|; all bit-exact)
tb/tb_gemv_int8.cpp       HLS C testbench, INT8 (bit-exact required)
host/host.cpp             XRT host program + CPU version
common/binio.h            file I/O and output comparison (host and testbenches)
scripts/gen_data.py       test inputs, quantization, golden outputs
scripts/check.py          compare an output file with the golden output; verify checksums
scripts/accuracy.py       accuracy against the float64 reference; bit-width sweep
scripts/hls.tcl           Vitis HLS csim / csynth / cosim script
data/FORMAT.md            data file format and quantization formulas
data/<case>/              committed small test cases: inputs + golden outputs
data/CHECKSUMS.md5        checksums of the generated 4096 x 4096 cases
results/                  our logs and reports (csim, csynth, sw_emu, hw_emu, hw_build, hw,
                          accuracy, dse)
Makefile, env.sh          build/run commands and tool setup
```

**Inputs and golden outputs.** Each test case is a directory under `data/` with
`meta.txt` (M, K, seed) and raw little-endian binaries; `y_fp32.f32.bin` and
`y_int8.f32.bin` are the golden outputs. The six small cases are committed; the three
4096 × 4096 cases (82 MB each) and the 100 random small shapes are regenerated with
`make data`, which also verifies the checksums. See `data/FORMAT.md`.

### 3.1 Input data: where it comes from

All inputs are **synthetic** and made by `scripts/gen_data.py` with fixed seeds, so
anyone can regenerate exactly the same files:

- Weights: Gaussian with σ = 0.02 (the typical scale of LLM weight matrices), and
  0.1% of the entries multiplied by 10 as outliers. Real LLM weights have such
  outliers; they make the scales of different groups differ, which is the case
  group-wise quantization is designed for.
- Activation x: standard normal N(0, 1).
- Corner cases: one-hot rows (y[m] = x[col], checkable by hand), all values ±127
  after quantization (largest possible group sum), all-zero groups and rows.
- 100 random small shapes (`data/small/`, `--small 100`).
- Three 4096 × 4096 cases (seeds 1, 2, 3): the size of one attention projection
  of a 7B model such as Llama 2.

The large files are not committed; `make data` regenerates them and checks their
MD5 checksums (`data/CHECKSUMS.md5`). The checksums are identical on Windows with
NumPy 2.3.3 and on Ubuntu with NumPy 1.21.5. Real Llama-2-7B weights and recorded
activations are planned for the final phase.

## 4. Kernel interface

```cpp
// y = W x, W is M x K row-major, K a multiple of 64, K <= 4096
extern "C" void gemv_fp32(const float* W, const float* x, float* y, int M, int K);
extern "C" void gemv_int8(const ap_uint<512>* Wq, const float* s, const ap_uint<512>* xq,
                          float sx, float* y, int M, int K);
```

INT8 formulas (row m, group g of 64 columns):
`s = max|W|/127`, `Wq = clip(round_half_even(W/s), -127, 127)`,
`P[m][g] = Σ Wq·xq` (exact int32), `y[m] = sx · Σ_g s[m][g] · P[m][g]` (float32, g in order).

## 5. How to run

### 5.0 Tools and settings

| Item | Value |
|---|---|
| Target card | AMD Alveo U280 on the Open Cloud Testbed (OCT) |
| Platform | `xilinx_u280_gen3x16_xdma_1_202211_1` |
| Part | `xcu280-fsvh2892-2L-e` |
| Kernel clock | 300 MHz (3.33 ns): HLS clock in `scripts/hls.tcl`; v++ uses the platform's default 300 MHz kernel clock |
| Tools | Vitis / Vitis HLS 2023.1, XRT 2.15 on the card host |
| Host compiler | g++ (C++17) from the build VM |
| Python | Python 3 with NumPy (data generation, checks, accuracy) |

On OCT the tools and the FPGAs are on different machines:

| Machine | CloudLab profile | Used for |
|---|---|---|
| Build VM | `oct-build`, Tool Version **2023.1**, RAM 64 GB, 8 CPUs, no remote desktop | csim, csynth, sw_emu, hw_emu, hardware build, host compile |
| U280 node | `oct-u280`, Workflow **Vitis**, Tool Version **2023.1**, **Ubuntu 22.04**, no remote desktop; name a free node (see cluster status) | running on the card |

(The `oct-u280` nodes only have the XRT runtime; Vitis is not available there.)

### 5.1 Setup (build VM)

```bash
git clone https://github.com/shuozhezhou/gemv-u280.git && cd gemv-u280
sudo apt-get install -y python3-numpy      # NumPy is not preinstalled
source env.sh                              # Vitis 2023.1 + XRT; prints the platform
make data                                  # all test cases; ends with "CHECKSUMS PASSED"
```

### 5.2 C simulation and synthesis (build VM, about 1 minute each)

```bash
make csim   KERNEL=gemv_fp32      # all 6 small cases, expect "TEST PASSED"
make csim   KERNEL=gemv_int8
make csim   KERNEL=gemv_fp32 CSIM_CASES=data/n4096_rand_s1    # full size
make csynth KERNEL=gemv_fp32      # report: build/hls_gemv_fp32/sol/syn/report/csynth.rpt
make csynth KERNEL=gemv_int8
```

### 5.3 Software and hardware emulation (build VM)

```bash
make run KERNEL=gemv_fp32 TARGET=sw_emu CASE=data/n64_rand_s1      # ~1 min build
make run KERNEL=gemv_int8 TARGET=sw_emu CASE=data/n64_pm127_s1
make run KERNEL=gemv_fp32 TARGET=hw_emu CASE=data/n128_rand_s1     # ~9 min build, ~1.5 min run
```

Every run ends with `TEST PASSED` and prints how many outputs are bit-exact.

### 5.4 Accuracy and bit-width sweep (any machine with Python 3 + NumPy, a few seconds)

```bash
python3 scripts/gen_data.py          # if data/n4096_rand_s* are missing
python3 scripts/accuracy.py          # -> results/accuracy/accuracy_metrics.txt,
                                     #    results/dse/bitwidth_sweep.csv
```

### 5.5 Hardware

Build on the build VM (about 100 minutes):

```bash
make host
make xclbin KERNEL=gemv_fp32 TARGET=hw     # -> build/gemv_fp32.hw/gemv_fp32.xclbin
make xclbin KERNEL=gemv_int8 TARGET=hw     # -> build/gemv_int8.hw/gemv_int8.xclbin (~75 min)
```

Copy the repository including `build/gemv_host` and the `.xclbin` to the U280 node, then:

```bash
sudo apt-get install -y python3-numpy
python3 scripts/gen_data.py && python3 scripts/check.py --verify-checksums
source /opt/xilinx/xrt/setup.sh && unset XCL_EMULATION_MODE
xbutil examine                              # card must show "Yes" under Ready
./build/gemv_host build/gemv_fp32.hw/gemv_fp32.xclbin gemv_fp32 data/n4096_rand_s1 100
./build/gemv_host build/gemv_int8.hw/gemv_int8.xclbin gemv_int8 data/n4096_rand_s1 100
```

Arguments: `<xclbin> <gemv_fp32|gemv_int8> <case_dir> [timed runs]`. The host writes
the output to `build/y_out.f32.bin`; `python3 scripts/check.py <case> build/y_out.f32.bin --kernel fp32`
checks it independently.

## 6. Results

### 6.1 Correctness

| Test | FP32 (L1) | INT8 (L2) |
|---|---|---|
| csim, 6 small cases | bit-exact | bit-exact |
| csim, 4096 × 4096 | bit-exact (3 seeds) | bit-exact |
| sw_emu, 7 cases + 100 random shapes | all pass, bit-exact | all pass, bit-exact |
| hw_emu (RTL simulation) | 3/3 bit-exact | 3/3 bit-exact (incl. the ±127 case) |
| **U280 hardware** | **9/9 cases bit-exact, incl. 3 × 4096 × 4096** | **9/9 cases bit-exact, incl. 3 × 4096 × 4096** |

Error of the outputs against a float64 reference with the original weights
(max |y − y_ref| / max |y_ref|, 4096 × 4096): FP32 1.6–2.2 × 10⁻⁶, INT8 0.98–1.35 × 10⁻²
(the cost of quantization).

### 6.2 L1 FP32 on the U280 (4096 × 4096, 100 runs, kernel time only)

| | Value |
|---|---|
| Kernel time | **391.66 ms** (min 391.66, max 391.68) |
| csynth prediction | 117,489,813 cycles × 3.33 ns = 391.63 ms |
| Data read / bandwidth | 64 MiB / **0.17 GB/s** (≈1.2% of one HBM pseudo-channel) |
| Host → device copy (PCIe) | 16.8 ms for 64 MiB |
| Kernel clock | 300 MHz achieved (timing met, WNS +0.014 ns) |
| Kernel resources (post-route) | 3,514 LUT, 4,301 FF, 8 DSP, 2 BRAM, 1 URAM (≈0.3%) |
| CPU, 1 thread, same op order (Xeon Gold 6226R) | 54.5 ms |

**Why L1 is slow — the initiation interval.** The inner loop is
`acc += W[m][k] * x[k]`. Each addition needs the result of the previous one, and an FP32
adder at 300 MHz takes 7 cycles (`fadd_32ns_32ns_32_7` in the report). So a new
iteration can only start every 7 cycles: **II = 7** (HLS tried II = 1…6 and reported a
carried-dependence violation for each). One row takes 4096 × 7 + 14 cycles, the whole
matrix 117.5 M cycles, which matches the measured time to 0.01%. The kernel reads only
4 bytes every 7 cycles, so it is limited by this dependence chain, not by memory, and is
about 7× slower than one CPU core. L2 and L3 address exactly this.

### 6.3 L2 INT8 on the U280 (4096 × 4096, 100 runs, kernel time only)

| | Value |
|---|---|
| Correctness | **bit-exact** (tolerance 0) for 3 seeds at 4096 × 4096 and all 6 small cases |
| Kernel time | **7.50 ms** (min 7.50, max 7.52) |
| csynth prediction | 2,244,753 cycles × 3.33 ns = 7.48 ms |
| **Speed-up over L1** | **52×** (391.66 ms → 7.50 ms; proposal target ≥ 3×) |
| Data read / bandwidth | 17 MiB (16 MiB weights + 1 MiB scales + 4 KiB x) / **2.38 GB/s** |
| Host → device copy (PCIe) | 4.3 ms for 17 MiB |
| Kernel clock | 300 MHz achieved (timing met, WNS +0.016 ns) |
| Kernel resources (post-route) | 6,371 LUT, 8,597 FF, 71 DSP, 38 BRAM (≈0.5% LUT) |
| CPU, 1 thread, same op order (Xeon Gold 6226R) | 23.7 ms → the FPGA kernel is 3.2× faster |

**Why 52× with the same II.** The group loop still has II = 7, because the FP32
accumulate `acc += s·P` still depends on itself. But one iteration now consumes a whole
512-bit word, 64 weights, through 64 int8 multipliers and an int32 adder tree, instead of
one weight. 4096 × 64 groups × 7 cycles ≈ 1.8 M cycles plus per-row overhead ≈ 2.24 M
cycles, against 117.5 M for L1. At 2.38 GB/s the kernel still uses only about 16% of one
HBM pseudo-channel, so it is still limited by the FP32 add chain — the target of L3.

## 7. Accuracy against the true result

Bit-exactness says the hardware computes exactly what the golden model computes. It
does not say how close that is to the correct answer. `scripts/accuracy.py` compares
the outputs with `y_ref.f64.bin`, the product W x computed in float64 from the
original (unquantized) weights. The U280 outputs are bit-identical to the golden files
(section 6.1), so **these are also the accuracy numbers of the hardware**.

Metrics: rel. L2 = ‖y − y_ref‖ / ‖y_ref‖; max rel. = max |y − y_ref| / max |y_ref|;
cosine similarity; SNR = 20 log10(‖y_ref‖ / ‖y − y_ref‖).

| Case (4096 × 4096) | Kernel | rel. L2 | max rel. | cosine | SNR (dB) |
|---|---|---|---|---|---|
| seed 1 | FP32 | 1.145e-06 | 1.610e-06 | 1.00000000 | 118.83 |
| seed 1 | INT8 | 1.139e-02 | 9.807e-03 | 0.99993526 | 38.87 |
| seed 2 | FP32 | 1.140e-06 | 1.932e-06 | 1.00000000 | 118.86 |
| seed 2 | INT8 | 1.219e-02 | 1.173e-02 | 0.99992574 | 38.28 |
| seed 3 | FP32 | 1.173e-06 | 2.245e-06 | 1.00000000 | 118.62 |
| seed 3 | INT8 | 1.227e-02 | 1.350e-02 | 0.99992467 | 38.22 |

The small cases are in `results/accuracy/accuracy_metrics.txt` (INT8 rel. L2
6.2e-03 to 1.3e-02; the ±127 case is exact up to float32 rounding because every value
is representable). FP32 error is float32 rounding only (about 1e-6). INT8 costs about
1% relative error, SNR about 38 to 39 dB, cosine 0.99993.

## 8. Design-space exploration: bit widths

Software sweep on `n4096_rand_s1` (`scripts/accuracy.py`, full table in
`results/dse/bitwidth_sweep.csv`). Weights are quantized to b bits with one FP32 scale
per group (symmetric, qmax = 2^(b−1) − 1, round half to even), the activation to 8 bits
as in the kernel. Bytes per weight = b/8 + 4/group; this is what a bandwidth-bound
kernel has to read, so it sets the best possible kernel time.

Relative L2 error (activation INT8):

| Weight bits | group 32 | group 64 | group 128 | per row (4096) |
|---|---|---|---|---|
| 8 | 1.05e-02 | **1.14e-02** (our L2) | 1.27e-02 | 3.48e-02 |
| 6 | 2.68e-02 | 3.29e-02 | 4.00e-02 | 1.37e-01 |
| 4 | 1.12e-01 | 1.38e-01 | 1.68e-01 | 5.49e-01 |
| 3 | 2.37e-01 | 2.76e-01 | 3.29e-01 | 8.47e-01 |
| 2 | 6.20e-01 | 6.88e-01 | 7.48e-01 | 9.50e-01 |

Bytes per weight: 8 bit 1.125 / 1.0625 / 1.031 / 1.001; 6 bit 0.875 / 0.8125 / 0.781 /
0.751; 4 bit 0.625 / 0.5625 / 0.531 / 0.501 (group 32 / 64 / 128 / per row).

Activation bits (weights 8 bit, group 64): 8 bit 1.14e-02, 6 bit 3.54e-02,
4 bit 1.54e-01. With the activation kept in FP32 the 8-bit g64 error drops to 7.6e-03.

What this shows:

- Per-group scales matter more and more as the bit width drops. At 8 bits, per-row
  scales cost 3× more error than groups of 64; at 4 bits per-row quantization is
  unusable (55% error) while groups of 32 to 64 give 11 to 14%.
- At 8-bit weights the 8-bit activation sets an error floor around 1e-2: smaller
  groups help little (1.05e-02 at g32 vs 1.14e-02 at g64).
- Plain round-to-nearest at 4 bits or below loses a lot of accuracy on this data.
  Real INT4 schemes (GPTQ, AWQ) add error compensation; we will test them on real
  Llama-2 weights. 6-bit g64 (0.81 B/weight, 24% fewer bytes than 8-bit g64, 3.3e-02
  error) is a possible middle point.
- This sweep is software only. Hardware runs at other bit widths are planned (L4).

## 9. Next steps

- L3: remove the II = 7 FP32 accumulation chain, either with several partial sums or
  by accumulating in fixed point (`ap_fixed`), overlap load / compute / store with
  dataflow, and spread rows over several HBM pseudo-channels; target ≥ 70% of the peak
  bandwidth of the channels used.
- Hardware runs at 256 × 256 and 1024 × 1024 in addition to 4096 × 4096.
- Other bit widths in hardware (INT4 / `ap_int`), guided by section 8.
- Real Llama-2-7B weights and recorded activations as inputs; CPU and GPU baselines.

## 10. Contributions

Shuozhe Zhou wrote all code in this repository so far and ran all experiments
(golden model, kernels, testbenches, host program, Makefile, builds and hardware
runs on OCT, accuracy script and bit-width sweep, README and reports).

Yifan Zhu's planned parts for the final phase, committed under his own GitHub account:
extracting real Llama-2-7B weights and activations as test inputs, the CPU/GPU
baselines, and the accuracy analysis on real data. Shuozhe Zhou continues with the
L3 kernel and the hardware runs.

## 11. Code sources

- `host/host.cpp` started from the XRT host program of the `hello_world_xrt` example
  in AMD/Xilinx Vitis_Accel_Examples (`host_xrt/hello_world_xrt`,
  https://github.com/Xilinx/Vitis_Accel_Examples), which we used in HW2. We kept its
  structure (open device, load xclbin, create buffers with `kernel.group_id`, run,
  sync); the test-case loading, timing, checking and CPU version are ours.
- Everything else (kernels, testbenches, scripts, Makefile, data format) was written
  by us. We use the Vitis HLS header `ap_int.h` and the XRT C++ API.
