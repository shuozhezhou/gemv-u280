#!/usr/bin/env python3
"""LLM text generation (GPT-2 small, Qwen3) where every linear layer runs as a GEMV on the U280.

The FPGA kernel (L2 gemv_int8, or an L3 kernel with the same argument list) computes all
matrix-vector products: per layer qkv (2304x768), attention out (768x768), MLP up (3072x768),
MLP down (768x3072), plus the LM head (50257x768). The host does embeddings, LayerNorm, GELU,
softmax/attention over the KV cache and the bias adds, all in NumPy float32.

Weights are quantized once with the same function as the test data (scripts/gen_data.py,
group of 64, symmetric INT8) and stay resident in HBM. For each GEMV the activation is
quantized on the host (gen_data.quantize_activation), sent over PCIe, and y is read back.

Backends:
  fp32    NumPy float32 reference (original weights)
  golden  NumPy bit-exact model of the INT8 kernel (gen_data.golden_int8 math, vectorized)
  fpga    the U280 through pyxrt; --check compares every GEMV with the golden model bit by bit

Batched kernel (gemv_int8_ps_b, --kernel gemv_int8_ps_b): one pass over the weights computes the
GEMV for up to 8 activation vectors. The model code yields (Linear, x) requests instead of calling
the GEMV directly, so several tokens can be advanced through the network in lock step and every
Linear is executed once for all of them. Each token sees exactly the same per-vector arithmetic as
in token-by-token decoding, so the logits are bit-identical. This is used for
  --spec K   prompt-lookup speculative decoding: guess up to K next tokens by matching the last
             n-gram earlier in the text, verify them all in one batched forward pass, keep the
             longest correct prefix (+1 token). Greedy output is unchanged, weight passes drop.
  --ppl with a batched kernel: teacher-forced perplexity 8 tokens per weight pass.

Usage (U280 node, after `source /opt/xilinx/xrt/setup.sh`):
  python3 host/gpt2_fpga.py --model gpt2 --xclbin build/gemv_int8.hw/gemv_int8.xclbin \
      --backend fpga --check --prompt "Field-programmable gate arrays" --tokens 40
  python3 host/gpt2_fpga.py --model gpt2 --backend fp32 --ppl text.txt   # perplexity
Model files (Hugging Face 'openai-community/gpt2'): model.safetensors, vocab.json, merges.txt.
"""
import argparse, json, os, re, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import gen_data  # noqa: E402  (quantize_weights, quantize_activation, GROUP)

G = gen_data.GROUP


# ------------------------------------------------------------------ model files
def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        raw = mm[base + a: base + b]
        if v["dtype"] == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif v["dtype"] == "BF16":   # bfloat16 = upper 16 bits of float32
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif v["dtype"] == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        else:
            raise ValueError((k, v["dtype"]))
        out[k] = arr.reshape(v["shape"])
    return out


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    # GPT-2 split pattern; \p{L} / \p{N} approximated with the std-lib `re` classes
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")

    # Qwen2/Qwen3 pre-tokenizer (single digits; a word may take one leading non-letter)
    PAT_QWEN = re.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\r\n\w]|_)?[^\W\d_]+|\d| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

    def __init__(self, d, qwen=False):
        if qwen:
            self.PAT = self.PAT_QWEN
        self.enc = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8"))
        self.dec = {v: k for k, v in self.enc.items()}
        merges = open(os.path.join(d, "merges.txt"), encoding="utf-8").read().split("\n")[1:-1]
        self.rank = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.cache = {}

    def bpe(self, tok):
        if tok in self.cache:
            return self.cache[tok]
        w = list(tok)
        while len(w) > 1:
            pairs = [(self.rank.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(w, w[1:]))]
            r, i = min(pairs)
            if r == 1 << 30:
                break
            a, b = w[i], w[i + 1]
            nw, j = [], 0
            while j < len(w):
                if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                    nw.append(a + b)
                    j += 2
                else:
                    nw.append(w[j])
                    j += 1
            w = nw
        self.cache[tok] = w
        return w

    def encode(self, text):
        ids = []
        for t in self.PAT.findall(text):
            t = "".join(self.b2u[b] for b in t.encode("utf-8"))
            ids += [self.enc[p] for p in self.bpe(t)]
        return ids

    def decode(self, ids):
        return bytearray(self.u2b[c] for c in "".join(self.dec[i] for i in ids)).decode("utf-8", errors="replace")


# ------------------------------------------------------------------ GEMV backends
class Linear:
    """y = W x + b with W stored as (M, K) = (out, in)."""

    def __init__(self, name, W, b):
        self.name, self.W, self.b = name, np.ascontiguousarray(W, np.float32), b
        self.M, self.K = self.W.shape
        assert self.K % G == 0 and self.K <= 4096, (name, self.W.shape)
        self.d = None                                           # per-input-channel smoothing factors
        self.Wq, self.s = gen_data.quantize_weights(self.W)   # same quantizer as the test data
        self.dev = None
        self.amax = np.zeros(self.K, np.float32)               # calibration: max |x_k| seen

    def smooth(self, alpha):
        """SmoothQuant-style migration: d_k = amax_k^a / wmax_k^(1-a); W' = W*d, x' = x/d (exact in real math)."""
        wmax = np.abs(self.W).max(axis=0)
        d = (np.maximum(self.amax, 1e-5) ** alpha / np.maximum(wmax, 1e-5) ** (1 - alpha)).astype(np.float32)
        self.d = d
        self.Wq, self.s = gen_data.quantize_weights((self.W * d[None, :]).astype(np.float32))


def golden_int8_fast(Wq, s, xq, sx):
    """Vectorized version of gen_data.golden_int8 (same float32 op order -> same bits)."""
    M, K = Wq.shape
    P = (Wq.reshape(M, K // G, G).astype(np.int32) * xq.reshape(K // G, G).astype(np.int32)).sum(axis=2)
    acc = np.zeros(M, np.float32)
    for g in range(K // G):
        acc = (acc + (s[:, g] * P[:, g].astype(np.float32)).astype(np.float32)).astype(np.float32)
    return (np.float32(sx) * acc).astype(np.float32)


def golden_for(kernel):
    """Bit-exact software model of each kernel (the summation order differs between kernels)."""
    if kernel.startswith("gemv_int8_ps"):
        return gen_data.golden_int8_ps            # 8 interleaved partial sums + adder tree (L3a)
    if kernel.startswith("gemv_int8_fx"):
        return gen_data.golden_int8_fx            # fixed-point accumulation (L3b)
    return golden_int8_fast                       # sequential group order (L2)


class FPGA:
    """N compute units (CUs) of the same kernel. Every matrix is split by rows into N contiguous slices;
    slice i lives in the HBM banks of CU i, and all N CUs run concurrently on one GEMV."""

    def __init__(self, xclbin, kname, ncu=1):
        import pyxrt
        self.x = pyxrt
        self.dev = pyxrt.device(0)
        t0 = time.time()
        self.uuid = self.dev.load_xclbin(pyxrt.xclbin(xclbin))
        names = [kname] if ncu == 1 else [f"{kname}:{{{kname}_{i+1}}}" for i in range(ncu)]
        self.ks = [pyxrt.kernel(self.dev, self.uuid, n, pyxrt.kernel.cu_access_mode.shared) for n in names]
        self.n = ncu
        print(f"[fpga] {kname} x{ncu} loaded from {xclbin} in {time.time()-t0:.1f} s", flush=True)
        self.TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.bo_x = [pyxrt.bo(self.dev, 4096, pyxrt.bo.normal, k.group_id(2)) for k in self.ks]
        self.bo_y = {}
        self.kernel_s = 0.0
        self.calls = 0
        self.bytes = 0

    def slices(self, M):
        b = [M * i // self.n for i in range(self.n + 1)]
        return [(b[i], b[i + 1]) for i in range(self.n)]

    def upload(self, lin):
        x = self.x
        parts = []
        for i, (r0, r1) in enumerate(self.slices(lin.M)):
            k = self.ks[i]
            Wq = np.ascontiguousarray(lin.Wq[r0:r1]); s = np.ascontiguousarray(lin.s[r0:r1])
            bw = x.bo(self.dev, max(Wq.nbytes, 64), x.bo.normal, k.group_id(0))
            bs = x.bo(self.dev, max(s.nbytes, 64), x.bo.normal, k.group_id(1))
            bw.write(Wq.reshape(-1).view(np.uint8), 0); bw.sync(self.TO, Wq.nbytes, 0)
            bs.write(s.reshape(-1).view(np.uint8), 0); bs.sync(self.TO, s.nbytes, 0)
            key = (i, r1 - r0)
            if key not in self.bo_y:
                self.bo_y[key] = x.bo(self.dev, max((r1 - r0) * 4, 64), x.bo.normal, k.group_id(4))
            parts.append((r0, r1, bw, bs, self.bo_y[key]))
        lin.dev = parts

    def gemv(self, lin, xq, sx):
        # pyxrt only takes int / bo arguments: pass the float32 scale as its raw 32-bit pattern
        sx_bits = struct.unpack("<i", struct.pack("<f", float(sx)))[0]
        for bx in self.bo_x:
            bx.write(xq.view(np.uint8), 0); bx.sync(self.TO, lin.K, 0)
        t0 = time.perf_counter()
        runs = [self.ks[i](bw, bs, self.bo_x[i], sx_bits, by, r1 - r0, lin.K)
                for i, (r0, r1, bw, bs, by) in enumerate(lin.dev)]
        for r in runs:
            r.wait()
        self.kernel_s += time.perf_counter() - t0
        self.calls += 1
        self.bytes += lin.Wq.nbytes
        y = np.empty(lin.M, np.float32)
        for r0, r1, bw, bs, by in lin.dev:
            by.sync(self.FROM, (r1 - r0) * 4, 0)
            y[r0:r1] = np.frombuffer(bytes(by.read((r1 - r0) * 4, 0)), dtype=np.float32)
        return y


class FPGABatch:
    """One CU of gemv_int8_ps_b / gemv_int8_ps_bw: y[b] = W x_b for b < nb <= 8 with a single pass over W.
    Arguments: (Wq, s, xq[nb*K bytes], sxv[nb floats], y, M, K, nb).
    gemv_int8_ps_b  writes y as nb*M floats, y[b*M + m];
    gemv_int8_ps_bw writes y as M rows of 8 floats, y[m*8 + b] (one 256-bit burst word per row)."""
    BMAX = 8

    def __init__(self, xclbin, kname):
        import pyxrt
        self.x = pyxrt
        self.dev = pyxrt.device(0)
        t0 = time.time()
        self.uuid = self.dev.load_xclbin(pyxrt.xclbin(xclbin))
        self.k = pyxrt.kernel(self.dev, self.uuid, kname, pyxrt.kernel.cu_access_mode.shared)
        self.interleaved = kname.startswith("gemv_int8_ps_bw")
        print(f"[fpga] {kname} (batch <= {self.BMAX}) loaded from {xclbin} in {time.time()-t0:.1f} s", flush=True)
        self.TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.bo_x = pyxrt.bo(self.dev, self.BMAX * 4096, pyxrt.bo.normal, self.k.group_id(2))
        self.bo_sx = pyxrt.bo(self.dev, 64, pyxrt.bo.normal, self.k.group_id(3))
        self.bo_y = {}
        self.kernel_s = 0.0
        self.calls = 0          # kernel launches = passes over the weights
        self.vecs = 0           # activation vectors processed
        self.bytes = 0
        self.hist = [0] * (self.BMAX + 1)

    def upload(self, lin):
        x, k = self.x, self.k
        bw = x.bo(self.dev, max(lin.Wq.nbytes, 64), x.bo.normal, k.group_id(0))
        bs = x.bo(self.dev, max(lin.s.nbytes, 64), x.bo.normal, k.group_id(1))
        bw.write(np.ascontiguousarray(lin.Wq).reshape(-1).view(np.uint8), 0); bw.sync(self.TO, lin.Wq.nbytes, 0)
        bs.write(np.ascontiguousarray(lin.s).reshape(-1).view(np.uint8), 0); bs.sync(self.TO, lin.s.nbytes, 0)
        if lin.M not in self.bo_y:
            self.bo_y[lin.M] = x.bo(self.dev, self.BMAX * lin.M * 4, x.bo.normal, k.group_id(4))
        lin.dev = (bw, bs, self.bo_y[lin.M])

    def gemv_batch(self, lin, xqs, sxs):
        nb = len(xqs)
        assert 1 <= nb <= self.BMAX
        bw, bs, by = lin.dev
        X = np.ascontiguousarray(np.stack(xqs)).reshape(-1)
        self.bo_x.write(X.view(np.uint8), 0); self.bo_x.sync(self.TO, nb * lin.K, 0)
        S = np.zeros(16, np.float32); S[:nb] = sxs
        self.bo_sx.write(S.view(np.uint8), 0); self.bo_sx.sync(self.TO, 64, 0)
        t0 = time.perf_counter()
        r = self.k(bw, bs, self.bo_x, self.bo_sx, by, lin.M, lin.K, nb)
        r.wait()
        self.kernel_s += time.perf_counter() - t0
        self.calls += 1; self.vecs += nb; self.hist[nb] += 1
        self.bytes += lin.Wq.nbytes
        if self.interleaved:
            by.sync(self.FROM, lin.M * 32, 0)
            Y = np.frombuffer(bytes(by.read(lin.M * 32, 0)), dtype=np.float32).reshape(lin.M, self.BMAX)
            return np.ascontiguousarray(Y[:, :nb].T)
        by.sync(self.FROM, nb * lin.M * 4, 0)
        return np.frombuffer(bytes(by.read(nb * lin.M * 4, 0)), dtype=np.float32).reshape(nb, lin.M)

    def gemv(self, lin, xq, sx):
        return self.gemv_batch(lin, [xq], [sx])[0]


# ------------------------------------------------------------------ GPT-2
class GPT2:
    def __init__(self, d, backend, fpga=None, check=False):
        w = load_safetensors(os.path.join(d, "model.safetensors"))
        p = "h." if "h.0.ln_1.weight" in w else "transformer.h."
        q = "" if "wte.weight" in w else "transformer."
        self.wte, self.wpe = w[q + "wte.weight"], w[q + "wpe.weight"]
        self.lnf = (w[q + "ln_f.weight"], w[q + "ln_f.bias"])
        self.L = sum(1 for k in w if k.endswith(".ln_1.weight"))
        self.H, self.D = 12, self.wte.shape[1]
        self.backend, self.fpga, self.check = backend, fpga, check
        self.layers = []
        t0 = time.time()
        for i in range(self.L):
            g = lambda n: w[f"{p}{i}.{n}"]
            # HF GPT-2 Conv1D stores (in, out): y = x @ W + b  ->  GEMV matrix is W.T (out, in)
            self.layers.append(dict(
                ln1=(g("ln_1.weight"), g("ln_1.bias")), ln2=(g("ln_2.weight"), g("ln_2.bias")),
                qkv=Linear(f"h{i}.qkv", g("attn.c_attn.weight").T, g("attn.c_attn.bias")),
                proj=Linear(f"h{i}.attn_out", g("attn.c_proj.weight").T, g("attn.c_proj.bias")),
                fc=Linear(f"h{i}.mlp_up", g("mlp.c_fc.weight").T, g("mlp.c_fc.bias")),
                fc2=Linear(f"h{i}.mlp_down", g("mlp.c_proj.weight").T, g("mlp.c_proj.bias"))))
        self.head = Linear("lm_head", self.wte, np.zeros(self.wte.shape[0], np.float32))
        self.lins = [l[k] for l in self.layers for k in ("qkv", "proj", "fc", "fc2")] + [self.head]
        self.calibrating = False
        self.mismatch = 0
        self.gemvs = 0
        print(f"[model] {self.L} layers, {sum(l.M*l.K for l in self.lins)/1e6:.1f} M linear weights, "
              f"quantized in {time.time()-t0:.1f} s", flush=True)
    def calibrate(self, ids, alpha):
        self.calibrating = True
        self.run(ids, 1)
        self.calibrating = False
        for l in self.lins:
            l.smooth(alpha)
        self.gemvs = 0
        print(f"[smooth] alpha={alpha}, calibrated on {len(ids)} tokens", flush=True)

    def to_device(self):
        fpga = self.fpga
        if fpga:
            t0 = time.time()
            for l in self.lins:
                fpga.upload(l)
            print(f"[fpga] {sum(l.Wq.nbytes for l in self.lins)/2**20:.1f} MiB INT8 weights resident in HBM "
                  f"({time.time()-t0:.1f} s)", flush=True)

    def linear(self, lin, x):
        self.gemvs += 1
        if self.calibrating:
            np.maximum(lin.amax, np.abs(x), out=lin.amax)
        if self.backend == "fp32" or self.calibrating:
            y = lin.W @ x
        else:
            xq, sx = self.quant(lin, x)
            if self.backend == "fpga":
                y = self.fpga.gemv(lin, xq, sx)
                if self.check:
                    ref = self.golden(lin.Wq, lin.s, xq, sx)
                    if not np.array_equal(y.view(np.uint32), ref.view(np.uint32)):
                        self.mismatch += 1
            else:
                y = self.golden(lin.Wq, lin.s, xq, sx)
        return (y + lin.b).astype(np.float32)

    @staticmethod
    def quant(lin, x):
        xs = x if lin.d is None else (x / lin.d)
        return gen_data.quantize_activation(xs.astype(np.float32))

    def linear_multi(self, lin, xs):
        """The same Linear for several vectors; one weight pass on a batched FPGA kernel."""
        if not (self.backend == "fpga" and hasattr(self.fpga, "gemv_batch")) or self.calibrating or len(xs) == 1:
            return [self.linear(lin, x) for x in xs]
        self.gemvs += len(xs)
        q = [self.quant(lin, x) for x in xs]
        Y = self.fpga.gemv_batch(lin, [a for a, _ in q], [b for _, b in q])
        if self.check:
            for (xq, sx), y in zip(q, Y):
                ref = self.golden(lin.Wq, lin.s, xq, sx)
                if not np.array_equal(y.view(np.uint32), ref.view(np.uint32)):
                    self.mismatch += 1
        return [(y + lin.b).astype(np.float32) for y in Y]

    def step(self, tok, pos, cache):
        """One token through the network (drives the _step generator with single GEMVs)."""
        g = self._step(tok, pos, cache)
        try:
            req = next(g)
            while True:
                req = g.send(self.linear(*req))
        except StopIteration as e:
            return e.value

    def step_multi(self, toks, pos0, cache):
        """Tokens toks[j] at positions pos0+j, advanced in lock step: every Linear runs once for all
        of them. Token j is resumed after token j-1, so its attention sees the K/V of tokens < j that
        were just appended to the cache -- exactly what sequential decoding sees."""
        gens = [self._step(tk, pos0 + j, cache) for j, tk in enumerate(toks)]
        reqs = [next(g) for g in gens]
        while True:
            lin = reqs[0][0]
            assert all(r[0] is lin for r in reqs)
            ys = self.linear_multi(lin, [r[1] for r in reqs])
            outs = []
            for j, g in enumerate(gens):
                try:
                    reqs[j] = g.send(ys[j])
                except StopIteration as e:
                    outs.append(e.value)
            if outs:
                assert len(outs) == len(gens)
                return outs

    def run_batched(self, ids, B=8):
        """Teacher forcing over ids, B tokens per weight pass. Returns logits for every position."""
        cache = [([], []) for _ in range(self.L)]
        logits = []
        for p in range(0, len(ids), B):
            logits += self.step_multi(ids[p:p + B], p, cache)
        return logits

    @staticmethod
    def lookup(out, k, nmax, nmin=1):
        """Prompt lookup: find the most recent earlier occurrence of the last n tokens (n = nmax..1)
        and propose the (up to k) tokens that followed it. Matches shorter than nmin are not used."""
        L = len(out)
        for n in range(min(nmax, L - 1), nmin - 1, -1):
            tail = out[L - n:]
            for i in range(L - n - 1, -1, -1):
                if out[i:i + n] == tail:
                    d = out[i + n:i + n + k]
                    if d:
                        return d
        return []

    def run_spec(self, ids, n_new, k=7, nmax=3, B=8, nmin=1):
        """Greedy decoding with prompt-lookup speculation. Returns (ids, stats)."""
        cache = [([], []) for _ in range(self.L)]
        out = list(ids)
        for p in range(0, len(ids) - 1, B):                       # prefill all but the last prompt token
            self.step_multi(ids[p:min(p + B, len(ids) - 1)], p, cache)
        self.mark()
        passes = proposed = accepted = 0
        while len(out) - len(ids) < n_new:
            p = len(out) - 1
            left = n_new - (len(out) - len(ids))
            draft = self.lookup(out, min(k, B - 1, left - 1), nmax, nmin) if left > 1 else []
            lgs = self.step_multi([out[-1]] + draft, p, cache)
            new = []
            for j in range(len(draft) + 1):
                g = int(np.argmax(lgs[j]))
                new.append(g)
                if j < len(draft) and g == draft[j]:
                    continue
                break
            a = len(new) - 1                                      # drafts accepted
            for c in cache:                                       # drop K/V of rejected drafts
                del c[0][p + a + 1:]; del c[1][p + a + 1:]
            out += new
            passes += 1; proposed += len(draft); accepted += a
        return out[:len(ids) + n_new], dict(passes=passes, proposed=proposed, accepted=accepted)

    @staticmethod
    def ln(x, wb):
        m = x.mean(); v = ((x - m) ** 2).mean()
        return ((x - m) / np.sqrt(v + 1e-5) * wb[0] + wb[1]).astype(np.float32)

    @staticmethod
    def gelu(x):
        return (0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))).astype(np.float32)

    def _step(self, tok, pos, cache):
        x = (self.wte[tok] + self.wpe[pos]).astype(np.float32)
        hd = self.D // self.H
        for i, l in enumerate(self.layers):
            qkv = yield l["qkv"], self.ln(x, l["ln1"])
            q, k, v = np.split(qkv, 3)
            cache[i][0].append(k.reshape(self.H, hd)); cache[i][1].append(v.reshape(self.H, hd))
            K = np.stack(cache[i][0], 1); V = np.stack(cache[i][1], 1)          # (H, T, hd)
            a = np.einsum("hd,htd->ht", q.reshape(self.H, hd), K) / np.sqrt(hd)
            a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
            o = np.einsum("ht,htd->hd", a, V).reshape(-1).astype(np.float32)
            x = x + (yield l["proj"], o)
            h = self.gelu((yield l["fc"], self.ln(x, l["ln2"])))
            x = x + (yield l["fc2"], h)
        return (yield self.head, self.ln(x, self.lnf))

    def mark(self):
        """Remember wall time and FPGA counters at the end of the prompt (prefill)."""
        f = self.fpga
        self.marked = (time.time(), f.kernel_s if f else 0.0, f.calls if f else 0)

    def run(self, ids, n_new, B=1):
        """Feed the prompt (B tokens per weight pass when B > 1), then greedily generate n_new tokens
        one at a time. Returns (all ids, logits list)."""
        cache = [([], []) for _ in range(self.L)]
        out, logits = list(ids), []
        p0 = 0
        if B > 1:
            for p in range(0, len(ids) - 1, B):
                logits += self.step_multi(ids[p:min(p + B, len(ids) - 1)], p, cache)
            p0 = len(ids) - 1
        self.mark()
        for pos in range(p0, len(ids) + n_new - 1):
            lg = self.step(out[pos], pos, cache)
            logits.append(lg)
            if pos >= len(ids) - 1:
                out.append(int(np.argmax(lg)))
        return out, logits


# ------------------------------------------------------------------ Qwen3
class Qwen3(GPT2):
    """Qwen3 decoder: RMSNorm, GQA with per-head q/k RMSNorm, RoPE, SwiGLU MLP, tied embeddings.
    q/k/v are fused into one GEMV and gate/up into one GEMV (they share the input vector)."""

    def __init__(self, d, backend, fpga=None, check=False):
        cfg = json.load(open(os.path.join(d, "config.json")))
        w = load_safetensors(os.path.join(d, "model.safetensors"))
        self.cfg = cfg
        self.L, self.D = cfg["num_hidden_layers"], cfg["hidden_size"]
        self.H, self.KV, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps, self.theta = cfg["rms_norm_eps"], cfg["rope_theta"]
        self.emb = w["model.embed_tokens.weight"]
        self.normf = w["model.norm.weight"]
        self.backend, self.fpga, self.check = backend, fpga, check
        self.layers = []
        t0 = time.time()
        z = lambda n: np.zeros(n, np.float32)
        for i in range(self.L):
            g = lambda n: w[f"model.layers.{i}.{n}"]
            qkv = np.concatenate([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")])
            gu = np.concatenate([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")])
            self.layers.append(dict(
                n1=g("input_layernorm.weight"), n2=g("post_attention_layernorm.weight"),
                qn=g("self_attn.q_norm.weight"), kn=g("self_attn.k_norm.weight"),
                qkv=Linear(f"l{i}.qkv", qkv, z(qkv.shape[0])),
                o=Linear(f"l{i}.o", g("self_attn.o_proj.weight"), z(self.D)),
                gu=Linear(f"l{i}.gate_up", gu, z(gu.shape[0])),
                down=Linear(f"l{i}.down", g("mlp.down_proj.weight"), z(self.D))))
        hw = w.get("lm_head.weight", self.emb)
        self.head = Linear("lm_head", hw, z(hw.shape[0]))
        self.lins = [l[k] for l in self.layers for k in ("qkv", "o", "gu", "down")] + [self.head]
        self.calibrating = False
        self.mismatch = 0
        self.gemvs = 0
        self.inv_freq = 1.0 / (self.theta ** (np.arange(0, self.hd, 2, dtype=np.float64) / self.hd))
        print(f"[model] Qwen3 {self.L} layers, {sum(l.M*l.K for l in self.lins)/1e6:.1f} M linear weights, "
              f"quantized in {time.time()-t0:.1f} s", flush=True)

    def rms(self, x, w):
        x = x.astype(np.float32)
        return (x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps) * w).astype(np.float32)

    def rope(self, x, pos):  # x: (heads, hd), HF rotate_half convention
        f = pos * self.inv_freq
        cos = np.concatenate([np.cos(f), np.cos(f)]).astype(np.float32)
        sin = np.concatenate([np.sin(f), np.sin(f)]).astype(np.float32)
        h = self.hd // 2
        rot = np.concatenate([-x[:, h:], x[:, :h]], axis=1)
        return (x * cos + rot * sin).astype(np.float32)

    def _step(self, tok, pos, cache):
        x = self.emb[tok].astype(np.float32)
        H, KV, hd = self.H, self.KV, self.hd
        for i, l in enumerate(self.layers):
            qkv = yield l["qkv"], self.rms(x, l["n1"])
            q = qkv[:H * hd].reshape(H, hd)
            k = qkv[H * hd:(H + KV) * hd].reshape(KV, hd)
            v = qkv[(H + KV) * hd:].reshape(KV, hd)
            q = self.rope(self.rms(q, l["qn"]), pos)
            k = self.rope(self.rms(k, l["kn"]), pos)
            cache[i][0].append(k); cache[i][1].append(v)
            K = np.stack(cache[i][0], 1); V = np.stack(cache[i][1], 1)          # (KV, T, hd)
            K = np.repeat(K, H // KV, axis=0); V = np.repeat(V, H // KV, axis=0)
            a = np.einsum("hd,htd->ht", q, K) / np.sqrt(hd)
            a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
            o = np.einsum("ht,htd->hd", a, V).reshape(-1).astype(np.float32)
            x = x + (yield l["o"], o)
            gu = yield l["gu"], self.rms(x, l["n2"])
            gt, up = np.split(gu, 2)
            h = (gt * (0.5 * (1 + np.tanh(0.5 * gt))) * up).astype(np.float32)   # silu, overflow-free sigmoid
            x = x + (yield l["down"], h)
        return (yield self.head, self.rms(x, self.normf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dir with model.safetensors, vocab.json, merges.txt")
    ap.add_argument("--backend", choices=["fp32", "golden", "fpga"], default="fpga")
    ap.add_argument("--xclbin", default="build/gemv_int8.hw/gemv_int8.xclbin")
    ap.add_argument("--kernel", default="gemv_int8")
    ap.add_argument("--cus", type=int, default=1, help="compute units in the xclbin (rows split across them)")
    ap.add_argument("--check", action="store_true", help="compare every FPGA GEMV with the golden model (bit-exact)")
    ap.add_argument("--prompt", default="Field-programmable gate arrays are")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--ppl", help="text file: report perplexity (teacher forcing) instead of generating")
    ap.add_argument("--max-ppl-tokens", type=int, default=256)
    ap.add_argument("--save", help="save generated ids / logits to this .npz")
    ap.add_argument("--smooth", type=float, default=None, help="SmoothQuant alpha (e.g. 0.5); needs --calib")
    ap.add_argument("--calib", help="calibration text file for --smooth (use different text than --ppl)")
    ap.add_argument("--spec", type=int, default=0, help="prompt-lookup speculative decoding, up to K draft tokens")
    ap.add_argument("--ngram", type=int, default=3, help="longest n-gram matched by the prompt lookup")
    ap.add_argument("--min-ngram", type=int, default=1, help="shortest n-gram match that may propose drafts")
    ap.add_argument("--batch", type=int, default=8, help="tokens per weight pass for batched ppl / prefill")
    ap.add_argument("--ref", help=".npz from a non-speculative run: check the generated ids are identical")
    a = ap.parse_args()

    arch = json.load(open(os.path.join(a.model, "config.json"))).get("model_type", "gpt2")
    tok = Tokenizer(a.model, qwen=arch.startswith("qwen"))
    batched = a.kernel.startswith("gemv_int8_ps_b")
    fpga = None
    if a.backend == "fpga":
        fpga = FPGABatch(a.xclbin, a.kernel) if batched else FPGA(a.xclbin, a.kernel, a.cus)
    m = (Qwen3 if arch.startswith("qwen") else GPT2)(a.model, a.backend, fpga, a.check)
    m.golden = golden_for(a.kernel)
    if a.smooth is not None:
        m.calibrate(tok.encode(open(a.calib, encoding="utf-8").read())[:256], a.smooth)
    m.to_device()

    if a.ppl:
        ids = tok.encode(open(a.ppl, encoding="utf-8").read())[: a.max_ppl_tokens]
        t0 = time.time()
        if batched or (a.backend != "fpga" and a.batch > 1):
            logits = m.run_batched(ids, a.batch)
        else:
            _, logits = m.run(ids, 1)
        nll = []
        for t in range(len(ids) - 1):
            lg = logits[t].astype(np.float64)
            lse = lg.max() + np.log(np.exp(lg - lg.max()).sum())
            nll.append(lse - lg[ids[t + 1]])
        print(f"RESULT ppl backend={a.backend} smooth={a.smooth} tokens={len(ids)} ppl={np.exp(np.mean(nll)):.3f} "
              f"time={time.time()-t0:.1f}s gemvs={m.gemvs} mismatches={m.mismatch}"
              + (f" weight_passes={fpga.calls} kernel={fpga.kernel_s:.2f}s nb_hist={fpga.hist}"
                 if fpga is not None and batched else ""))
        if a.save:
            np.savez_compressed(a.save, ids=np.array(ids), top1=np.array([int(np.argmax(l)) for l in logits]))
        return

    ids = tok.encode(a.prompt.replace('\\' + "n", '\n'))   # a literal backslash-n in --prompt is a newline
    t0 = time.time()
    spec = None
    if a.spec:
        out, spec = m.run_spec(ids, a.tokens, a.spec, a.ngram, a.batch, a.min_ngram)
        logits = []
    else:
        out, logits = m.run(ids, a.tokens, a.batch if batched else 1)
    t1 = time.time()
    wall = t1 - t0
    n_steps = len(ids) + a.tokens - 1
    print("TEXT >>> " + tok.decode(out).replace("\n", "\\n"))
    msg = (f"RESULT gen backend={a.backend} smooth={a.smooth} prompt_tokens={len(ids)} new_tokens={a.tokens} steps={n_steps} "
           f"wall={wall:.2f}s ({n_steps/wall:.2f} steps/s) gemvs={m.gemvs}")
    if fpga:
        msg += (f" fpga_kernel_time={fpga.kernel_s:.2f}s ({fpga.kernel_s/n_steps*1e3:.1f} ms/token, "
                f"{fpga.calls} calls, {fpga.bytes/fpga.kernel_s/1e9:.2f} GB/s of weights incl. call overhead)")
        if a.check:
            msg += f" bit_exact_mismatches={m.mismatch}/{m.gemvs}"
        if batched:
            msg += f" weight_passes={fpga.calls} nb_hist={fpga.hist}"
    if spec:
        msg += (f" spec_k={a.spec} ngram={a.min_ngram}..{a.ngram} verify_passes={spec['passes']} drafts={spec['proposed']} "
                f"accepted={spec['accepted']} tokens_per_pass={a.tokens/spec['passes']:.2f}")
    if a.ref:
        r = np.load(a.ref)["ids"]
        msg += f" identical_to_ref={bool(np.array_equal(r[:len(out)], np.array(out)))}"
    msg += f" new_tok_per_s={a.tokens/wall:.2f}"
    # generation phase only (after the prompt has been processed)
    tm, km, cm = m.marked
    msg += f" decode_wall={t1-tm:.2f}s decode_tok_per_s={a.tokens/(t1-tm):.2f}"
    if fpga:
        msg += (f" decode_kernel={fpga.kernel_s-km:.3f}s ({(fpga.kernel_s-km)/a.tokens*1e3:.2f} ms/token)"
                f" decode_passes={fpga.calls-cm} ({(fpga.calls-cm)/a.tokens:.2f} per token)")
    print(msg, flush=True)
    if a.save:
        np.savez_compressed(a.save, ids=np.array(out), prompt_len=len(ids),
                            top1=np.array([int(np.argmax(l)) for l in logits]))


if __name__ == "__main__":
    main()
