import sys, time, itertools, numpy as np
sys.path.insert(0, "/mydata/gpt2run/host"); sys.path.insert(0, "/mydata/gpt2run/scripts")
import llm_fpga as L
d = "/mydata/l3/data/sq4096"; M = K = 4096
class Lin: pass
lin = Lin(); lin.M, lin.K = M, K
lin.Wq = np.fromfile(d + "/Wq.i8.bin", np.int8).reshape(M, K); lin.s = np.fromfile(d + "/s.f32.bin", np.float32).reshape(M, K // 64)
xq = np.fromfile(d + "/xq.i8.bin", np.int8); sx = float(np.fromfile(d + "/sx.f32.bin", np.float32)[0])
f = L.FPGA("/mydata/l3/build/gemv_int8_ps_x4.hw/gemv_int8_ps_x4.xclbin", "gemv_int8_ps", 4)
f.upload(lin)
import struct
bits = struct.unpack("<i", struct.pack("<f", sx))[0]
for bx in f.bo_x: bx.write(xq.view(np.uint8), 0); bx.sync(f.TO, K, 0)
def t(cus, reps=200):
    best = []
    for _ in range(reps):
        t0 = time.perf_counter()
        runs = [f.ks[i](lin.dev[i][2], lin.dev[i][3], f.bo_x[i], bits, lin.dev[i][4], lin.dev[i][1]-lin.dev[i][0], K) for i in cus]
        for r in runs: r.wait()
        best.append(time.perf_counter() - t0)
    return np.median(best) * 1e3
for c in [(0,), (1,), (2,), (3,)]: print("CU", [i+1 for i in c], "%.3f ms" % t(c))
for c in itertools.combinations(range(4), 2): print("CUs", [i+1 for i in c], "%.3f ms" % t(c))
print("CUs [1,2,3,4] %.3f ms" % t((0,1,2,3)))
