# L3 lab notes (developed on branch `l3-dev`, merged into `main` on Nov 25)

Goal: new hardware-backed results after Project Update 2 (supplement) and for the
final report (Dec 18). Only measured numbers go here; each entry says where it
came from.

## 2026-11-24 — code ready (local, before the VM)

### What was added

| File | What |
|---|---|
| `kernels/int8_terms.h` | shared front end: x -> BRAM, then ONE flattened loop over all M*G words (II = 1): `t = s * float(P)` streamed out. Flattening removes the per-row pipeline drain + AXI read latency of L2 (L2: 548 cycles per row for 64 groups). |
| `kernels/gemv_int8_ps.cpp/.cfg` | L3a: dataflow `compute_terms -> accumulate -> reduce_store`. 8 FP32 partial sums `a[g % 8]`, `DEPENDENCE ... inter RAW distance = 8 true` (true distance is exactly 8: a slot is read only if written 8 iterations earlier in the same row; rows padded to a multiple of 8 iterations, pad iterations consume nothing). Tree `((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))`, `y = sx * sum`. y on its own AXI bundle gmem3 (still HBM[2]). |
| `kernels/gemv_int8_fx.cpp/.cfg` | L3b: same front end, one `ap_fixed<48,24>` accumulator per row (exact integer adds, II = 1 without partial sums), `y = sx * (float(acc_int) * 2^-24)`. |
| `kernels/gemv_dse.cpp` | csynth-only DSE kernel: `-DWBITS=8/4 -DGS=64/128 -DNPS=8/1`. INT4 = 128 weights per 512-bit word. |
| `tb/tb_gemv_int8_ps.cpp`, `tb/tb_gemv_int8_fx.cpp` | HLS testbenches, bit-exact vs `y_int8ps` / `y_int8fx`. |
| `scripts/gen_data.py` | `golden_int8_ps`, `golden_int8_fx`, self tests vs plain-Python loops; new committed case `n40x1344_rand_s1` (G = 21 > 8, not a multiple of 8). |
| `scripts/parse_csynth.py`, `scripts/dse_bits.sh` | csynth.rpt -> numbers / `results/dse/csynth_bits.csv`. |
| `scripts/run_l3_vm.sh` | VM flow: data, csim, csynth, hw (background), sw_emu, hw_emu, dse. |
| `host/host.cpp` | kernel names `gemv_int8_ps`, `gemv_int8_fx` (same args), golden `y_int8ps` / `y_int8fx`, CPU versions in the same op order. |
| `scripts/check.py` | `--kernel int8ps|int8fx`. |
| `scripts/hls.tcl` | optional `HLS_CFLAGS`, `PROJ`; testbench only if it exists. |
| `Makefile` | `n40x1344_rand_s1` in `CSIM_CASES`; xo depends on `kernels/*.h`. |
| `data/FORMAT.md`, `data/CHECKSUMS.md5` | new golden files documented; 6 new checksums (n4096 s1-s3 y_int8ps / y_int8fx). |

### Local checks (Windows, Python 3, NumPy 2.3.3)

- `py -3 scripts/gen_data.py --small 100`: self test passed; all previously
  committed files regenerate **byte-identical** (git shows no change) and the
  old 27 checksums pass.
- New golden vs L2 golden (`y_int8`), bit-different outputs:
  n4096 s1 3285/4096 (ps), 3191/4096 (fx); s2 3374 / 3261; s3 3372 / 3264;
  n40x1344 27/40 (ps), 26/40 (fx); n64_rand 0/64 (ps, G = 1 so same order), 1/64 (fx).
  Max |ps - int8| / max|y|: 2.47e-7 (s1), 2.64e-7 (s2), 3.80e-7 (s3);
  max |fx - int8| / max|y|: 2.05e-7, 2.64e-7, 2.85e-7.
  => the bit-exact test against the new files is a real test of the new order.
- Error vs float64 reference (max |y - y_ref| / max|y_ref|), 4096 x 4096:
  s1 int8 9.8069e-3, ps 9.8069e-3, fx 9.8069e-3 (same to 5 digits; s2 1.1726e-2, s3 1.3496e-2 for all three).
- Fixed-point range over all 110 cases: max |term| = max |running sum| = 406.40 (< 2^9; limit 2^23).
- Python transliteration of the kernel loops (accumulate with padding/emit order,
  reduce tree, fx path) vs golden files: 108 cases (committed + n4096_s1 + 100
  small), **0 mismatches**.
- No C++ compiler locally: kernels/host not compiled yet.

### Expected (to be checked by csynth, not measured)

- L2 per row: 548 cycles (64 groups x II 7 + overhead). L3a ideal: 1 word/cycle
  -> 4096*64 = 262,144 cycles = 0.87 ms at 300 MHz, which needs 19.2 GB/s from
  HBM[0]; one pseudo-channel gives ~14.4 GB/s, so expect ~1.2 ms on the board.

## 2026-11-24 23:10 — waiting for a build VM

- First VM (gemv-build2, fpga-build1:27410) failed to boot; polling
  stopped. U280 node for later board runs: pc157 (the size sweep runs in ~/sweep;
  L3 files go to /mydata/l3).
- Added `scripts/run_l3_board.sh` (U280 node: all cases incl. 3 x 4096, 100 runs,
  logs to results/hw/).

## 2026-11-24 evening — build VM gemv-build3 (fpga-build2:27410), Vitis HLS 2023.1

Setup notes: the VM needed `sudo apt-get install -y libtinfo5` (vitis_hls failed
with `libtinfo.so.5: cannot open shared object file`) and `python3-numpy`
(waits for the boot-time XRT install to release the dpkg lock).
`python3 scripts/gen_data.py --small 100` on the VM: **CHECKSUMS PASSED** (incl. the
6 new y_int8ps / y_int8fx checksums -> the new goldens reproduce across machines).

### csim (Vitis HLS 2023.1, results/csim/)

| Kernel | 7 committed cases | 4096 x 4096 (s1) |
|---|---|---|
| gemv_int8_ps (final version) | all bit-exact, TEST PASSED | 4096/4096 bit-exact, TEST PASSED |
| gemv_int8_fx | all bit-exact, TEST PASSED | 4096/4096 bit-exact, TEST PASSED |

(The first ps version, below, also passed csim bit-exact on all of these.)

### csynth at 4096 x 4096, 300 MHz target (results/csynth/gemv_int8_{ps,fx}_csynth.rpt)

| | L2 gemv_int8 | L3a ps v1 (II=1 + DEPENDENCE) | **L3a gemv_int8_ps** | **L3b gemv_int8_fx** |
|---|---|---|---|---|
| Latency (cycles) | 2,244,753 | 1,048,681 | **262,313** | **262,313** |
| Latency at 300 MHz | 7.475 ms | 3.492 ms | **0.874 ms** | **0.874 ms** |
| Loop II | groups = 7 | words 1, sums **4**, rows 8 | load_x 1, words 1, chunks 8 (8 groups), rows 8 | load_x 1, words 1, sums 1, rows 1 |
| Est. slack (ns) | -0.00 | -0.87 | -0.24 (est. Fmax 374 MHz) | -0.00 |
| BRAM_18K | 70 | 72 | 72 | 72 |
| DSP | 39 | 49 | 51 | 50 |
| FF | 12,774 | 17,456 | 17,537 | 16,984 |
| LUT | 13,505 | 15,719 | 16,092 | 15,392 |

- 262,313 cycles = 262,144 groups (1 per cycle) + 169 cycles total: **no per-row
  overhead** (L2: ~99 cycles/row measured on the board). Flattened M*G loop +
  dataflow does what the cycle model of section 9 asked for (ideal ~0.27 M; the
  "1 group/cycle + 100 cycles/row" alternative would be ~0.95 M).
- ps v1: one group per iteration, `acc[g % 8]` fully partitioned + `DEPENDENCE
  variable=acc inter RAW distance=8 true`. HLS ignored the distance for the
  partitioned registers: II violation "carried dependence ... distance = 1" on
  the partition elements, final II = 4, 1.05 M cycles, slack -0.87 ns (probably the dynamic-index 8:1 mux).
- ps final: chunks of 8 groups per pipeline iteration at II = 8, the 8 adds
  unrolled, so each acc[p] is a fixed register with a distance-1 dependence and
  8 cycles >= fadd latency 7. Still 1 term per cycle; same op order, same golden;
  no dependence pragma needed. HLS 200-871: estimated clock 2.569 ns vs effective
  budget 2.431 ns (3.33 - 0.899 uncertainty) -> est. Fmax 374 MHz.
- Same latency for ps and fx: both are bound by the 1-word/cycle front end
  (compute_terms, 262,313 cycles); at 64 B/cycle x 300 MHz = 19.2 GB/s this
  exceeds one HBM pseudo-channel (~14.4 GB/s), so the board time should be HBM-bound.

### hw builds (started 21:20 / 21:22 MST VM time, JOBS=4 each, in parallel)

- `make xclbin KERNEL=gemv_int8_fx TARGET=hw JOBS=4` -> build/hw_gemv_int8_fx.log
- `make xclbin KERNEL=gemv_int8_ps TARGET=hw JOBS=4` -> build/hw_gemv_int8_ps.log

### Bit-width DSE, csynth only, 4096 x 4096 (results/dse/csynth_bits.csv, kernels/gemv_dse.cpp)

| Point | Weight bits | Group | NPS | Latency (cycles) | ms @300 MHz | Loop II | BRAM_18K | DSP | FF | LUT |
|---|---|---|---|---|---|---|---|---|---|---|
| w8_g64_p1 | 8 | 64 | 1 | 1,048,682 | 3.492 | chunks = 4 (1 word/iter) | 72 | 47 | 16,502 | 14,681 |
| w8_g64_p8 (= gemv_int8_ps) | 8 | 64 | 8 | 262,313 | 0.874 | chunks = 8 (8 words/iter) | 72 | 51 | 17,489 | 16,060 |
| w4_g64_p8 | 4 | 64 | 8 | 131,248 | 0.437 | 8 | 66 | 93 | 21,509 | 19,608 |
| w4_g128_p8 | 4 | 128 | 8 | 131,243 | 0.437 | 8 | 64 | 84 | 22,097 | 20,091 |
| w4_g128_p1 | 4 | 128 | 1 | 524,396 | 1.746 | chunks = 4 | 64 | 80 | 21,111 | 18,716 |

- All `words` loops II = 1; INT4 packs 128 weights per word -> half the words ->
  half the cycles (131 k) at the same 64 B/cycle; the scale reads halve too for g128.
- NPS = 1 (single FP32 chain) gives II = 4 in this flattened structure (HLS picks a
  shorter fadd than L2's 7-cycle one; slack -0.04), i.e. 4x slower than NPS = 8.
- INT4 costs ~+42 / +33 DSP and ~+3.5-4 k LUT over INT8 (two x-words per weight
  word, 128 multipliers per cycle).
- gemv_int8_ps row and the w8_g64_p8 row agree (262,313 cycles; resources within 50 FF/LUT).

### sw_emu fix

First sw_emu attempt: every case FAILED at `v++ -l -t sw_emu`:
`ERROR: [v++ 17-1309] Gcc: /usr/include/features-time64.h:20:10: fatal error: bits/wordsize.h`
(Ubuntu 22.04.2 + Vitis 2023.1, multiarch include dir not searched). Fix in env.sh:
`CPATH=/usr/include/x86_64-linux-gnu`, `LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`
(only if that header exists). After the fix: gemv_int8_ps sw_emu n64_rand_s1:
FPGA 64/64 bit-exact, CPU 64/64 bit-exact, TEST PASSED. Rerun of the full sw_emu +
hw_emu stage started (build/l3_chain2.log), plus a csim re-check with the new env.

### sw_emu (after the env.sh fix; results/sw_emu/gemv_int8_{ps,fx}_*.log)

| | 7 committed cases | 4096 x 4096 s1 | 100 random shapes (data/small) |
|---|---|---|---|
| gemv_int8_ps | 7/7 PASSED (bit-exact) | PASSED, 4096/4096 bit-exact | 100 passed, 0 failed |
| gemv_int8_fx | 7/7 PASSED (bit-exact) | PASSED, 4096/4096 bit-exact | 100 passed, 0 failed |

The host's CPU versions (`cpu_int8_ps`, `cpu_int8_fx`) are also 4096/4096
bit-exact against their goldens, so the CPU baseline uses the same op order.

### hw_emu (RTL simulation, results/hw_emu/gemv_int8_{ps,fx}_*.log)

| | n128_rand_s1 | n40x1344_rand_s1 (21 groups, padded chunk) |
|---|---|---|
| gemv_int8_ps | PASSED, 128/128 bit-exact | PASSED, 40/40 bit-exact |
| gemv_int8_fx | PASSED, 128/128 bit-exact | PASSED, 40/40 bit-exact |

csim re-check after the env.sh (CPATH) change: gemv_int8_ps small cases TEST PASSED.

## 2026-11-24 22:00 MST — new work approved: multi-CU scaling (5) and INT4 in hardware (6)

- (5) `kernels/gemv_int8_ps_x{2,4,8}.cfg`: N CUs, CU i Wq -> HBM[2(i-1)] (own
  pseudo-channel and memory controller), s/xq/y -> HBM[2(i-1)+1], all `slr=...:SLR0`.
  Makefile `NCU=n` links `build/gemv_int8_ps_x<n>.hw/gemv_int8_ps_x<n>.xclbin` from
  the single-CU .xo. `host/host_multi.cpp` (`make host_multi`): contiguous row slices,
  per-CU buffers in each CU's banks, start all N runs then wait all, bit-exact check
  of the concatenated y vs y_int8ps (or y_int4).
- (6) `kernels/gemv_int4_ps.cpp` (+ .cfg, tb): symmetric int4 [-7, 7], s4 = max|W|/7
  per group of 64, 128 weights per word, 2 groups per cycle; s read as 64-bit pairs.
  gen_data.py: `quantize_weights4`, `pack_int4`, `golden_int4_ps`, files `Wq4.i4.bin`,
  `s4.f32.bin`, `y_int4.f32.bin` (only when K % 128 == 0), new committed case
  `n24x2688_rand_s1` (21 int4 words per row), `--llama` writes
  `llama7b_ffn_11008x4096` (seed 1, same as scripts/gen_sweep.py on main).
  CHECKSUMS.md5 now 42 lines (n4096 s1-s3 incl. INT4 files). Local regen: all
  older files byte-identical, CHECKSUMS PASSED. host.cpp: `gemv_int4_ps` + `cpu_int4_ps`.
- INT4 accuracy (max |y - y_ref| / max|y_ref|): n4096 s1 1.43e-1, s2 1.43e-1,
  s3 1.59e-1, llama7b_ffn_11008x4096 1.54e-1, n24x2688 1.22e-1 (INT8 on the same
  cases: 0.98-1.49e-2). Symmetric 4-bit with the outlier weights of our generator
  costs ~10x the INT8 error.
- 22:03 MST: `make xclbin KERNEL=gemv_int8_ps TARGET=hw NCU=4 JOBS=4` started
  (build/hw_gemv_int8_ps_x4.log), 3rd concurrent build.

### INT4 gemv_int4_ps — HLS (VM)

- VM data regen with `--small 100 --llama`: CHECKSUMS PASSED (42 files incl. INT4).
- csim (results/csim/gemv_int4_ps_*.log): n64x128_zero 64/64, n128_rand 128/128,
  n100x256 100/100, n24x2688 24/24 bit-exact (K = 64 / 1344 cases skipped: K % 128 != 0);
  n4096_rand_s1 4096/4096 bit-exact. TEST PASSED.
- csynth (results/csynth/gemv_int4_ps_csynth.rpt): 131,248 cycles = 0.437 ms at
  300 MHz; II: load_x 1, words 1 (1 word = 128 weights / cycle), chunks 8, rows 8;
  est. Fmax 374 MHz (slack -0.24 vs the 0.9 ns uncertainty margin);
  BRAM 66, DSP 93, FF 21,509, LUT 19,608 (= the DSE w4_g64_p8 point).

### Build queue (scripts/build_queue.sh, build/queue.log)

22:10 MST: queue started, max 3 concurrent `v++ -l` (JOBS=4 each): next
gemv_int4_ps (1 CU), then gemv_int8_ps NCU=8, then NCU=2 — each starts when a
running link finishes. Running at that time: gemv_int8_ps, gemv_int8_fx, gemv_int8_ps_x4.

### host_multi (VM, 22:08 MST)

- `make host host_multi` on the VM: both compile with -Wall, no warnings
  (binaries copied to the worktree's build/, which is git-ignored).
- Host logic check with multi-CU **sw_emu** links (`make xclbin KERNEL=gemv_int8_ps
  TARGET=sw_emu NCU=4|8`, same cfgs as hw): x4 and x8 on n100x256 (uneven slices,
  13/12 rows per CU at x8), n40x1344, n4096_rand_s1 — all bit-exact vs y_int8ps,
  TEST PASSED (results/sw_emu/gemv_int8_ps_x{4,8}_*.log). sw_emu reports bank 0 for
  every port (emulation); hw banks are checked on the board.
- pyxrt gotcha: float scalar kernel args are truncated to int;
  pass struct.unpack('<i', struct.pack('<f', sx))[0]. (C++ hosts here are unaffected.)

### hw builds done: gemv_int8_ps and gemv_int8_fx (23:21 MST, ~2 h each with 3 builds sharing 8 vCPU)

| | gemv_int8_ps | gemv_int8_fx | (L2 gemv_int8, for reference) |
|---|---|---|---|
| xclbin (VM) | ~/l3/build/gemv_int8_ps.hw/gemv_int8_ps.xclbin | ~/l3/build/gemv_int8_fx.hw/gemv_int8_fx.xclbin | |
| WNS / TNS | +0.016 ns / 0 | +0.016 ns / 0 | +0.016 ns |
| WHS | +0.006 ns | +0.006 ns | |
| Kernel clock (DATA_CLK) | 300 MHz (auto-scaling estimate 331.0 MHz, capped at 300) | 300 MHz (estimate 337.1 MHz) | 300 MHz |
| HBM clock | 450 MHz | 450 MHz | |
| Kernel LUT | 7,912 (0.67%) | 7,497 (0.64%) | 6,371 |
| LUTAsMem | 1,364 | 1,364 | 1,305 |
| REG | 9,899 | 9,274 | 8,597 |
| BRAM | 31 | 31 | 38 |
| DSP | 78 | 77 | 71 |

Reports: results/hw_build/gemv_int8_{ps,fx}_{timing_summary_routed,kernel_util_routed,full_util_routed}.rpt,
_automation_summary.txt. Build queue then started gemv_int4_ps (23:22:36).

## 2026-11-25 00:00 MST — board results (pc157) and HBM-switch alignment

Board (pc157): gemv_int8_ps 4096² 1.311 ms
(13.6 GB/s, 24.4 W, 32 mJ per GEMV), gemv_int8_fx 1.31 ms, all bit-exact; GPT-2 small
28 tok/s with 0 mismatches. x4 (even/odd mapping) on 11008x4096: 1/2/3 CUs ->
13.9 / 27.3 / 40.1 GB/s, but 4 CUs -> 27.7 GB/s. Pair test: CU3+CU4 0.71 ms vs 0.39 ms
alone, other pairs ~0.44 ms.

Cause (from build/gemv_int8_ps_x4.hw/reports/link/automation_summary.txt):
v++ connects CU i's m_axi_gmem0..3 to hmss_0 S(4i-3)..S(4i) in CU order, e.g.
gemv_int8_ps_3 gmem0..3 -> S09..S12, gemv_int8_ps_4 gmem0..3 -> S13..S16, regardless of
the bank. With the even/odd mapping CU3 used HBM[4],[5] and CU4 HBM[6],[7] — both in
the PC 4-7 switch — so both cross the lateral switch links into the same switch.

Actions (00:06 MST):
- Stopped the queue and killed the misaligned x8 / x2 links (incl. orphaned Vivado
  processes, found by their working directory); deleted their cfgs and build dirs.
- New `kernels/gemv_int8_ps_x{2,4,8}_aligned.cfg`: CU i Wq -> HBM[4(i-1)],
  s/xq/y -> HBM[4(i-1)+1], i.e. only banks of the switch its S ports sit on; all SLR0.
  Makefile `TAG=_aligned` -> build/gemv_int8_ps_x<n>_aligned.hw/gemv_int8_ps_x<n>_aligned.xclbin.
  (v++ 2023.1 `sp=` takes only the bank; we did not find a way to pin the hmss port,
  so the alignment relies on the observed in-order port assignment, verified below.)
- 00:07 started x8_aligned and x4_aligned (JOBS=4, reuse the ps .xo); gemv_int4_ps
  (started 23:22) keeps running -> 3 builds.
