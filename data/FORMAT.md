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
| `y_int8.f32.bin` | float32 | M | Golden output of the INT8 kernel (bit-exact target) |
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
