# Findings preserved from the killed PLE-fix agent run (2026-09-11)

Extracted from its transcript: assistant notes and numeric tool results, in order. Code artifacts: repack.py, patch.py, norm_patch.py, test_exact.py, repack.log, verify_full.log, test_exact.json in this directory.

I'll start by reading the required context files.

---

[tool result] mx.metal.device_info is deprecated and will be removed in a future version. Use mx.device_info instead.
3.11.10 (main, Oct 16 2024, 08:56:36) [Clang 18.1.8 ]
mlx 0.32.2
np 2.3.5
phys GB 128.0
pagesize 16384
wired limit {'device_name': 'Apple M5 Max', 'max_recommended_working_set_size': 130566586368, 'memory_size': 137438953472, 'architecture': 'applegpu_g17s', 'max_buffer_length': 86586540032, 'resource_limit': 499000}

---

Now I have the full picture. Let me write the repacker.

---

[tool result] layer 1 rows 320001536 dims 160 bits 4 gs 32
 shard_offsets[:4] [0, 2500012, 5000024, 7500036] len 128
 shard rows uniq {2500012}
 wb 80 sb 10 stride 100
 total bytes 32.0001536 GB
 weight_scale 0.00019931793212890625
 prefix language_model.model.layers.1.ple.ple_embedding.ngram_embedding

---

[tool result] """Upper-bound estimate for a device-side gather from a packed resident PLE table.

The oQ4e PLE table is ~32 GB of 4-bit rows (160 dims -> 80 B/row + scales).
fuse_quantized_shards() would make the lookup one mx.take over that buffer.
Emulate exactly that access pattern: 32768 random rows out of a 32 GB buffer.
"""
import mlx.core as mx, time, statistics
GB = 1024**3
ROW = 96           # 80 B packed weight + 16 B scales/biases per 160-dim row
ROWS = int(32 * GB // ROW)
tbl = mx.zeros((ROWS, ROW), dtype=mx.uint8)
mx.eval(tbl)
print(f"table {tbl.nbytes/1e9:.1f} GB, {ROWS/1e6:.0f}M rows")
for n in (512, 2048, 4096):
    idx = mx.random.randint(0, ROWS, (n * 16,)).astype(mx.uint32)
    mx.eval(idx)
    ts = []
    for _ in range(5):
        mx.synchronize(); t0 = time.perf_counter()
        mx.eval(mx.take(tbl, idx, axis=0))
        mx.synchronize(); ts.append(time.perf_counter() - t0)
    t = min(ts)
    print(f"  {n:5d} tokens ({n*16} rows): {t*1e3:7.2f} ms  "
          f"({n*16*ROW/t/1e9:.1f} GB/s effective)")
import mlx.core as mx, time
from load_omlx import load
from profile import timeit
m,_ = load(ple_mode="resident"); lm=m.language_model
print("active %.1f GB" % (mx.get_active_memory()/1e9))
x = (mx.random.normal((1,2048,2560))*0.02).astype(mx.bfloat16); mx.eval(x)
mlp = lm.model.layers[0].mlp
G = next(l.linear_attn for l in lm.model.layers if l.is_linear)
hc = lm.model.layers[0].attn_hyper_connection
h = (mx.random.normal((1,2048,10240))*0.02).astype(mx.bfloat16); mx.eval(h)
print("moe block ms  %.2f (mmap: 10.69)" % (timeit(lambda: mx.eval(mlp(x)), reps=3)*1e3))
print("gdn layer ms  %.2f (mmap:  9.66)" % (timeit(lambda: mx.eval(G(x,None,None)), reps=3)*1e3))
print("hc block  ms  %.2f (mmap:  2.14)" % (timeit(lambda: mx.eval(hc(h)), reps=3)*1e3))

from profile import make_prompt, forward
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpPLELayer
ids = make_prompt(2048)
forward(lm, ids)
print("body           ms %.1f" % (timeit(lambda: forward(lm, ids), reps=3, cool=10)*1e3))
P = next(l.ple for l in lm.model.layers if "ple" in l)
h = (mx.random.normal((1,2048,10240))*0.02).astype(mx.bfloat16); mx.eval(h)
print("ple layer      ms %.1f (mmap warm: 27.8)" % (timeit(lambda: mx.eval(P(h, ids, None, None)), reps=3)*1e3))
orig = Qwen4ExpPLELayer.__call__
Qwen4ExpPLELayer.__call__ = lambda self, hs, ii, c, m, target_verify=False: mx.zeros_like(hs)
print("body no_ple    ms %.1f" % (timeit(lambda: forward(lm, ids), reps=3, cool=10)*1e3))
Qwen4ExpPLELayer.__c

---

[tool result] model    ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp
out      ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/ple-packed
layouts  rows, planar   planned 64.00 GB, free 642.81 GB
  layer 1: 320,001,536 rows x 160 dims, 4-bit group 32; row = 80+10+10 = 100 B
    shard   8/128  20,000,096 rows     4.0s   0.50 GB/s in
    shard  16/128  40,000,192 rows     8.1s   0.49 GB/s in
    shard  24/128  60,000,288 rows    12.3s   0.49 GB/s in
    shard  32/128  80,000,384 rows    16.4s   0.49 GB/s in
    shard  40/128  100,000,480 rows    20.5s   0.49 GB/s in
    shard  48/128  120,000,576 rows    24.3s   0.49 GB/s in
    shard  56/128  140,000,672 rows    28.0s   0.50 GB/s in
    shard  64/128  160,000,768 rows    31.8s   0.50 GB/s in
    shard  72/128  180,000,864 rows    35.5s   0.51 GB/s in
    shard  80/128  200,000,960 rows    39.3s   0.51 GB/s in
    shard  88/128  220,001,056 rows    43.1s   0.51 GB/s in
    shard  96/128  240,001,152 rows    46.8s   0.51 GB/s in
    shard 104/128  260,001,248 rows    50.6s   0.51 GB/s in
    shard 112/128  280,001,344 rows    55.5s   0.50 GB/s in
    shard 120/128  300,001,440 rows    60.6s   0.50 GB/s in
    shard 128/128  320,001,536 rows    65.5s   0.49 GB/s in
wrote ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/ple-packed/manifest.json

verifying 200,000 rows per layer against the source shards
{
 "1": {
  "rows_checked": 199933,
  "rows_layout_mismatched_rows": 0,
  "rows_layout_exact": true,
  "planar_mismatched_weight": 0,
  "planar_mismatched_scales": 0,
  "planar_mismatched_biases": 0,
  "planar_exact": true
 }
}
verify took 14.0s
BIT-EXACT
  layer1.rows.bin              32.00 GB
  layer1.weight.u32            25.60 GB
  layer1.scales.bf16           3.20 GB
  layer1.biases.bf16           3.20 GB
=== files ===
total 125038880
drwxr-xr-x@  8 maxshugar  staff          256 Sep 11 23:33 .
drwxr-xr-x@ 37 maxshugar  staff         1184 Sep 11 23:31 ..
-rw-r--r--@  1 maxshugar  staff   3200015360 Sep 11 23:33 layer1.biases.bf16
-rw-r--r--@  1 maxshugar  staff  32000153600 Sep 11 23:33 layer1.rows.bin
-rw-r--r--@  1 maxshugar  staff   3200015360 Sep 11 23:33 layer1.scales.bf16
-rw-r--r--@  1 maxshugar  staff  25600122880 Sep 11 23:33 layer1.weight.u32
-rw-r--r--@  1 maxshugar  staff         4469 Sep 11 23:33 manifest.json
-rw-r--r--@  1 maxshugar  staff          227 Sep 11 23:33 verify.json

---

Repack done in 65 s, bit-exact on 200k sampled rows. Let me add a full-table verification and run it in the background while I write the patch.

---

Now the runtime patch.

---

[tool result] model    ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp
out      ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/ple-packed
layouts  rows, planar   planned 64.00 GB, free 578.79 GB

full verification: every packed byte against its source shard
    full-verify shard  16/128  40,000,192 rows    16.1s
    full-verify shard  32/128  80,000,384 rows    31.9s
    full-verify shard  48/128  120,000,576 rows    44.1s
    full-verify shard  64/128  160,000,768 rows    60.9s
    full-verify shard  80/128  200,000,960 rows    77.8s
    full-verify shard  96/128  240,001,152 rows    94.5s
    full-verify shard 112/128  280,001,344 rows   111.5s
    full-verify shard 128/128  320,001,536 rows   128.5s
{
 "1": {
  "rows_checked": 320001536,
  "mismatched_rows": {
   "rows": 0,
   "weight": 0,
   "scales": 0,
   "biases": 0
  },
  "exact": true,
  "seconds": 128.5
 }
}
verify took 128.7s
BIT-EXACT
  layer1.rows.bin              32.00 GB
  layer1.weight.u32            25.60 GB
  layer1.scales.bf16           3.20 GB
  layer1.biases.bf16           3.20 GB
done

---

Both layouts verified byte-for-byte across all 320M rows. Now let's test the patch against the stock lookup.

---

[tool result] [transformers] You are using a model of type `qwen4_exp` to instantiate a model of type ``. This may be expected if you are loading a checkpoint that shares a subset of the architecture (e.g., loading a `sam2_video` checkpoint into `Sam2Model`), but is otherwise not supported and can yield errors. Please verify that the checkpoint is compatible with the model you are instantiating.
loading ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp (ple mode mmap)
  PLE runtime mode: mmap
  loaded in 35.2s, active 72.8 GB, peak 72.8 GB
n-gram ids (1, 2048, 16) -> 32,768 rows, range [8044, 320001333]

[1] stock SSD gather (reference)
  (1, 2048, 16, 160) mlx.core.bfloat16, 952.6 ms (cold)

[2] packed 'rows' mode
  patched 1 layer(s); resident bytes 0.00 GB; load 0.0s; mlx active 72.9 GB
  table gather bitwise identical to stock: True
  PLE layer output bitwise identical to stock: True
  stock restored and still bitwise identical: True
  rows-mode pages read 32,727 (536 MB), pread 416 ms

[2] packed 'resident' mode
  patched 1 layer(s); resident bytes 32.00 GB; load 42.4s; mlx active 105.0 GB
  table gather bitwise identical to stock: True
  PLE layer output bitwise identical to stock: True
  stock restored and still bitwise identical: True

[3] grouped RMSNorm: bf16 Metal kernel vs the fp32 round trip
  hc_norm                                identical 100.00%  max 1 ULP  mean 0.0000 ULP  max|abs| 1.562e-02  max rel 7.752e-03
  hc_norm (wide)                         identical 100.00%  max 1 ULP  mean 0.0000 ULP  max|abs| 1.562e-02  max rel 7.576e-03

  whole prefill_forward (norm + projections + gated mean):
  prefill_forward.mixed                  identical  99.95%  max 47744 ULP  mean 0.0173 ULP  max|abs| 1.562e-02  max rel 9.766e+26
  prefill_forward.hyper_input            identical 100.00%  max 0 ULP  mean 0.0000 ULP  max|abs| 0.000e+00  max rel 0.000e+00
  prefill_forward.injection              identical  99.99%  max 1 ULP  mean 0.0001 ULP  max|abs| 7.812e-03  max rel 7.634e-03

[4] whole-body forward, greedy-argmax agreement on the last position
  body hidden state                      identical   2.17%  max 49017 ULP  mean 1048.8069 ULP  max|abs| 2.800e+01  max rel 1.109e+30
  last-position logits                   identical   3.72%  max 34342 ULP  mean 468.6950 ULP  max|abs| 4.062e-01  max rel 4.096e+05
  greedy argmax agrees (last position): True
  greedy argmax agrees over all 2048 positions: 85.791%

wrote ~/inferenc

---

PLE lookup is bit-exact in both modes. The norm's body-level divergence needs a proper reference — let me fix the ULP metric and add an fp32 ground truth plus a real-text argmax check.

---

## Commands/files it ran or wrote (abridged)

- Bash: ls -la ~/inference-server/ 2>&1 | head -50; echo "---"; ls -la ~/inference-server/kernels/ 2>&1 | head -50; echo "---"; ls -la ~/inference-server/profile-flashn
- Bash: cat ~/inference-server/kernels/COMMON.md; echo "=== LOAD_OMLX ==="; cat ~/inference-server/profile-flashnext/load_omlx.py; echo "=== _env.sh ==="; cat ~/inferen
- Bash: cat ~/inference-server/profile-flashnext/REPORT.md
- Bash: cd ~/inference-server/profile-flashnext && wc -l profile.py && sed -n '1,420p' profile.py
- Bash: cd ~/inference-server/profile-flashnext && sed -n '420,886p' profile.py
- Bash: ls -la /Applications/oMLX.app/Contents/Resources/omlx/patches/ | head -60; echo "==="; ls -la /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwe
- Bash: cd /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat && cat __init__.py; echo "=== residency.py ==="; cat residency.py; echo "=== 
- Bash: cd /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp && ls -la && wc -l *.py
- Bash: cd /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp && grep -n "^class \|^def \|^_PLE\|^PLE" lang
- Bash: cd /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp && sed -n '1877,2400p' language.py
- Bash: cd /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp && sed -n '2394,2790p' language.py
- Bash: M=~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp; ls -la $M | head -30; echo "---df---"; df -h ~/Engineering/MLX/_models | tail -2; echo "---confi
- Bash: M=~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp; python3 - <<'EOF' import json,os,collections M=os.path.expanduser("~/Engineering/MLX/_models/Jun
- Bash: mkdir -p ~/inference-server/kernels/ple-fix && cd ~/inference-server/kernels/ple-fix && cp ~/inference-server/profile-flashnext/_env.sh . && . ./_env.sh && $PY 
- Write: ~/inference-server/kernels/ple-fix/repack.py
- Bash: cd ~/inference-server/kernels/ple-fix && . ./_env.sh && $PY - <<'EOF' import sys; sys.path.insert(0,'.') import repack, json model = repack.DEFAULT_MODEL info, 
- Edit: ~/inference-server/kernels/ple-fix/repack.py
- Bash: cd ~/inference-server/kernels/ple-fix && . ./_env.sh && $PY - <<'EOF' import sys; sys.path.insert(0,'.') import repack, json model = repack.DEFAULT_MODEL info, 
- Bash: cd ~/inference-server/kernels/ple-fix && . ./_env.sh && nohup $PY repack.py --verify-rows 200000 > repack.log 2>&1 & echo started; sleep 60; tail -5 repack.log;
- Bash: cd ~/inference-server/profile-flashnext && cat fused_gather_est.py resident_micro.py; echo "=== log ==="; tail -3 ~/inference-server/kernels/ple-fix/repack.log;
- Bash: cd ~/inference-server/kernels/ple-fix && cat repack.log; echo "=== files ==="; ls -la ~/Engineering/MLX/_models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/ple-packed/
- Edit: ~/inference-server/kernels/ple-fix/repack.py
- Bash: cd ~/inference-server/kernels/ple-fix && python3 - <<'PY' import re,io p='repack.py'; s=open(p).read() s=s.replace('    ap.add_argument("--verify-only", action=
- Bash: cd ~/inference-server/kernels/ple-fix && . ./_env.sh && $PY - <<'EOF' import mlx.core as mx, numpy as np a = mx.array(np.arange(10, dtype=np.uint16)) print("vie
- Write: ~/inference-server/kernels/ple-fix/patch.py
- Bash: cd ~/inference-server/kernels/ple-fix && . ./_env.sh && $PY -c " import mlx.nn as nn, inspect src = inspect.getsource(nn.Module.__setattr__) print(src) print('-
- Bash: cd ~/inference-server/kernels/ple-fix && python3 - <<'PY' p='patch.py'; s=open(p).read() s=s.replace('''    embedding._packed_prefetch = packed_prefetch.__get__
- Bash: cd ~/inference-server/kernels/ple-fix && sed -i '' 's|return (raw\[:, :wb\].view("<u4"),|return (raw[:, :wb].copy().view("<u4"),|' patch.py && grep -n 'copy().v
- Bash: pgrep -fl "repack.py" | head; echo "---"; cd /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp && 
- Bash: cd /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp && sed -n '1,120p' hc_fused.py
- Bash: cd /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp && sed -n '226,300p;300,460p' hc_fused.py
- Bash: cd /Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp && sed -n '1049,1095p' language.py; echo "===
- Write: ~/inference-server/kernels/ple-fix/norm_patch.py
- Bash: cd ~/inference-server/kernels/ple-fix && cat verify_full.log; pgrep -f repack.py >/dev/null && echo "STILL RUNNING" || echo "done"
- Write: ~/inference-server/kernels/ple-fix/test_exact.py
- Bash: cd ~/inference-server/kernels/ple-fix && . ./_env.sh && $PY test_exact.py 2>&1 | tail -60
- Bash: cd ~/inference-server/kernels/ple-fix && python3 - <<'PY' p='test_exact.py'; s=open(p).read()  s=s.replace('''def ulp_report(ref, got, name):     """Difference 
- Bash: cd ~/inference-server/kernels/ple-fix && . ./_env.sh && $PY -c "import mlx.core as mx; print(hasattr(mx,'median'))"
- Bash: cd ~/inference-server/kernels/ple-fix && python3 - <<'PY' p='test_exact.py'; s=open(p).read() s=s.replace('''        disagree = mx.argmax(rb_l, -1) != mx.argmax
- Bash: cd ~/inference-server/kernels/ple-fix && . ./_env.sh && $PY test_exact.py > test_exact.log 2>&1; echo "exit $?"; tail -50 test_exact.log
