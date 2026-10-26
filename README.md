# Quantized GEMV for LLM Inference on the Alveo U280

EECE 5698 — FPGAs in the Cloud, Fall 2026
Shuozhe Zhou, Yifan Zhu

> **Project Update 1** (due 2026-10-29). The graded snapshot is frozen as branch
> [`project-update-1`](https://github.com/shuozhezhou/gemv-u280/tree/project-update-1)
> and release/tag [`project-update-1`](https://github.com/shuozhezhou/gemv-u280/releases/tag/project-update-1)
> (GitHub does not allow forking a repository into the same account). Neither changes
> after the deadline; `main` keeps moving.

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

## 2. What is implemented for this update

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
scripts/hls.tcl           Vitis HLS csim / csynth / cosim script
data/FORMAT.md            data file format and quantization formulas
data/<case>/              committed small test cases: inputs + golden outputs
data/CHECKSUMS.md5        checksums of the generated 4096 x 4096 cases
results/                  our logs and reports (csim, csynth, sw_emu, hw_emu, hw_build, hw)
Makefile, env.sh          build/run commands and tool setup
```

**Inputs and golden outputs.** Each test case is a directory under `data/` with
`meta.txt` (M, K, seed) and raw little-endian binaries; `y_fp32.f32.bin` and
`y_int8.f32.bin` are the golden outputs. The six small cases are committed; the three
4096 × 4096 cases (82 MB each) and the 100 random small shapes are regenerated with
`make data`, which also verifies the checksums. See `data/FORMAT.md`.

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

### 5.4 Hardware

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

## 7. Next steps

- L3: split each row's FP32 sum into several partial sums so the group loop reaches
  II = 1, overlap load / compute / store with dataflow, and spread rows over several HBM
  pseudo-channels; target ≥ 70% of the peak bandwidth of the channels used.
- Real Llama-2-7B weights and recorded activations as inputs (L4).
