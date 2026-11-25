# Data file format

Every test case is one directory under `data/`, e.g. `data/n64_rand_s1/`.
All arrays are raw little-endian binaries, row-major, no header.
`scripts/gen_data.py` writes them; `common/binio.h` reads them (HLS testbench and XRT host).

## meta.txt

Plain text, one `key value` pair per line:

```
M 64
K 64
group 64
seed 1
kind rand
```

`M` = rows of W (= length of y), `K` = columns of W (= length of x).
`K` must be a multiple of `group` (64).

## Files

| File | Type | Elements | Meaning |
|---|---|---|---|
| `W.f32.bin` | float32 | M·K | FP32 weights |
| `x.f32.bin` | float32 | K | FP32 activation |
| `y_fp32.f32.bin` | float32 | M | Golden output of the FP32 kernel (float32, k = 0..K-1 in order) |
| `Wq.i8.bin` | int8 | M·K | Quantized weights, in [-127, 127] |
| `s.f32.bin` | float32 | M·(K/64) | Weight scale per row and group, row-major `s[m][g]` |
| `xq.i8.bin` | int8 | K | Quantized activation |
| `sx.f32.bin` | float32 | 1 | Activation scale |
| `y_int8.f32.bin` | float32 | M | Golden output of the INT8 kernel `gemv_int8` (bit-exact target) |
| `y_int8ps.f32.bin` | float32 | M | Golden output of `gemv_int8_ps` (8 FP32 partial sums, see below) |
| `y_int8fx.f32.bin` | float32 | M | Golden output of `gemv_int8_fx` (fixed-point accumulator, see below) |
| `Wq4.i4.bin` | uint8 | M·K/2 | INT4 weights, two per byte: weight 2b in the low nibble of byte b, 2b+1 in the high nibble, two's complement in [-7, 7] (only if K % 128 == 0) |
| `s4.f32.bin` | float32 | M·(K/64) | INT4 scale per row and group, `s4 = max|W|/7` (only if K % 128 == 0) |
| `y_int4.f32.bin` | float32 | M | Golden output of `gemv_int4_ps` (only if K % 128 == 0) |
| `xqB.i8.bin` | int8 | 8·K | 8 quantized activation vectors for `gemv_int8_ps_b`, vector b at b·K; vector 0 = `xq` |
| `sxB.f32.bin` | float32 | 8 | their activation scales (sxB[0] = sx) |
| `y_int8psB.f32.bin` | float32 | 8·M | `[b][m]`: golden_int8_ps of vector b (row 0 = `y_int8ps`) |
| `y_ref.f64.bin` | float64 | M | `W x` in float64 from the original weights, summed sequentially over k (accuracy reference only; no BLAS, so it is identical on every machine) |

## Quantization

For row `m`, group `g` (columns `64g .. 64g+63`):

```
s[m][g]   = max |W[m][k]| / 127          (s = 1 if the group is all zero)
Wq[m][k]  = clip(round_half_even(W[m][k] / s[m][g]), -127, 127)
sx        = max |x| / 127
xq[k]     = clip(round_half_even(x[k] / sx), -127, 127)
P[m][g]   = sum_{k in g} Wq[m][k] * xq[k]          (exact, int32)
y_int8[m] = sx * ( sum_g  s[m][g] * float(P[m][g]) ) (float32, g = 0..K/64-1 in order)
```

The FP32 operations are done in float32 in exactly this order, so the
kernel output must match `y_int8.f32.bin` bit for bit.

## L3 variants (same inputs, same terms, different summation)

With `t[m][g] = s[m][g] * float(P[m][g])` (one float32 multiply, as above):

```
gemv_int8_ps (NPS = 8 partial sums, float32):
  a[p]       = 0                                   p = 0..7
  a[g % 8]   = a[g % 8] + t[m][g]                  g = 0..K/64-1 in order
  y_int8ps[m] = sx * ( ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)) )

gemv_int8_fx (ap_fixed<48, 24> accumulator, exact):
  q[m][g]    = floor(t[m][g] * 2^24)               (exact integer; = AP_TRN)
  acc[m]     = sum_g q[m][g]                       (exact, |acc| < 2^47)
  y_int8fx[m] = sx * ( float(acc[m]) * 2^-24 )     (int -> float rounds to nearest even)
```

Both differ from `y_int8` only in the last bits (≤ 4 × 10⁻⁷ × max|y| on the
4096 × 4096 cases); each kernel is checked bit for bit against its own file.

## INT4 (`gemv_int4_ps`)

```
s4[m][g]  = max |W[m][k]| / 7          (s4 = 1 if the group is all zero)
q4[m][k]  = clip(round_half_even(W[m][k] / s4[m][g]), -7, 7)
t[m][g]   = s4[m][g] * float(sum_{k in g} q4[m][k] * xq[k])       (float32)
w[m][j]   = t[m][2j] + t[m][2j+1]           one 512-bit word = 128 weights = groups 2j, 2j+1
a[j % 8]  = a[j % 8] + w[m][j]              j = 0..K/128-1 in order
y_int4[m] = sx * ( ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)) )
```

## Batch (`gemv_int8_ps_b`)

Vectors 1..7 are `rand` activations from seeds `seed*1000 + b`, each quantized on
its own (`xqB[b]`, `sxB[b]`); `y_int8psB[b] = golden_int8_ps(Wq, s, xqB[b], sxB[b])`.
