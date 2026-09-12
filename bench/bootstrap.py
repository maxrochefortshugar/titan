# Launch oMLX's server in-process with the small-M patch applied (bypasses the shadowed sitecustomize).
import inspect, os, sys, runpy, atexit, threading, time
def _wants_model(fn):
    try:
        ps = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    return bool(ps) and ps[0].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
# Optional: swap the bundled mlx_vlm for another checkout (e.g. 0.7.0 staged in staging/mlxvlm070) before oMLX imports it
_alt = os.environ.get("OMLX_MLXVLM_PATH")
if _alt:
    sys.path.insert(0, os.path.expanduser(_alt))
    import mlx_vlm as _mv; sys.stderr.write(f"[staging] mlx_vlm from {_mv.__file__} version {getattr(_mv, '__version__', '?')}\n"); sys.stderr.flush()

if os.environ.get("OMLX_SMALLM_QMM") == "1":
    sys.path.insert(0, os.path.expanduser("~/inference-server/kernels/smallm"))
    import kernel as smallm
    ok = smallm.apply()
    calls = {"n": 0, "shapes": {}}
    _orig = smallm.qmm_smallm
    def counted(x, *a, **k):
        calls["n"] += 1; s = tuple(x.shape); calls["shapes"][s] = calls["shapes"].get(s, 0) + 1
        return _orig(x, *a, **k)
    smallm.qmm_smallm = counted
    # the routing wrapper may hold a direct reference; patch through the module attribute used by the router if present
    for name in dir(smallm):
        obj = getattr(smallm, name)
        if callable(obj) and getattr(obj, "__module__", "") == "kernel" and name.startswith("_route"):
            pass
    def reporter():
        while True:
            time.sleep(20); sys.stderr.write(f"[staging] smallm routed calls so far: {calls['n']} shapes={dict(list(calls['shapes'].items())[:6])}\n"); sys.stderr.flush()
    threading.Thread(target=reporter, daemon=True).start()
    sys.stderr.write(f"[staging] small-M qmm patch applied={ok}\n"); sys.stderr.flush()

# Experiment overrides (staging only)
_step = int(os.environ.get("OMLX_PREFILL_STEP", "0") or 0)
_blk = int(os.environ.get("OMLX_ARRAYS_CACHE_BLOCK", "0") or 0)
if _step or _blk:
    import omlx.scheduler as _sch
    if _step:
        _orig_base = _sch.Scheduler._base_prefill_step_size
        def _base(self, processed, remaining, _orig=_orig_base):
            v = _orig(self, processed, remaining)
            return max(v, _step) if v else _step
        _sch.Scheduler._base_prefill_step_size = _base
        sys.stderr.write(f"[staging] prefill step size override -> {_step}\n")
    if _blk:
        _sch.Scheduler._ARRAYS_CACHE_BLOCK_SIZE = _blk
        _sch.Scheduler._POOLING_ROTATING_BLOCK_SIZE = _blk
        sys.stderr.write(f"[staging] arrays-cache block size override -> {_blk}\n")
    sys.stderr.flush()

# INT8 MoE gather prefill kernel (workstream moe-int8), opt-in
if os.environ.get("OMLX_MOE_INT8_PREFILL") == "1":
    sys.path.insert(0, os.path.expanduser("~/inference-server/kernels/moe-int8"))
    import patch as _moe8
    _moe8.install()
    def _moe8_reporter():
        while True:
            time.sleep(20)
            try: sys.stderr.write(f"[staging] moe-int8 stats: {_moe8.stats()}\n"); sys.stderr.flush()
            except Exception as e: sys.stderr.write(f"[staging] moe-int8 stats error {e!r}\n")
    threading.Thread(target=_moe8_reporter, daemon=True).start()
    sys.stderr.write("[staging] moe-int8 prefill patch installed\n"); sys.stderr.flush()

# Qwen4-Exp prefill patches (workstream ple-fix), opt-in, applied post-load.
if os.environ.get("OMLX_PLE_PACKED") == "1" or os.environ.get("OMLX_QWEN4_BF16_NORM") == "1":
    import importlib.util as _ilu
    from omlx.engine.vlm import VLMBatchedEngine
    _vlm_start = VLMBatchedEngine.start
    def _load(name, path):
        spec = _ilu.spec_from_file_location(name, os.path.expanduser(path)); mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod); return mod
    async def _patched_start(self):
        await _vlm_start(self)
        model = getattr(self, "_vlm_model", None)
        if model is None:
            return
        try:
            _ple_packed = _load("ple_packed_patch", "~/inference-server/kernels/ple-fix/patch.py")
            n = _ple_packed.apply_ple_packed_patch(model, self._model_name)
            sys.stderr.write(f"[staging] PLE packed table: {n} layer(s) patched (mode={os.environ.get('OMLX_PLE_PACKED_MODE')})\n")
        except Exception as exc:
            sys.stderr.write(f"[staging] PLE packed patch failed (stock path kept): {exc!r}\n")
        try:
            _norm = _load("qwen4_norm_patch", "~/inference-server/kernels/ple-fix/norm_patch.py")
            ok = _norm.apply_bf16_norm_patch()
            sys.stderr.write(f"[staging] bf16 grouped norm: applied={ok}\n")
        except Exception as exc:
            sys.stderr.write(f"[staging] bf16 norm patch failed: {exc!r}\n")
        sys.stderr.flush()
    VLMBatchedEngine.start = _patched_start



# Round-2 import-time patches: OMLX_ROUND2_IMPORT_PATCHES="path/to/patch.py[:func],..." run before the server starts
for _i, _spec in enumerate([x for x in os.environ.get("OMLX_ROUND2_IMPORT_PATCHES", "").split(",") if x.strip()]):
    try:
        import importlib.util as _ilu3
        _path, _, _fn = _spec.strip().partition(":")
        _sp = _ilu3.spec_from_file_location(f"round2_import_{_i}", os.path.expanduser(_path)); _m = _ilu3.module_from_spec(_sp); _sp.loader.exec_module(_m)
        _ok = getattr(_m, _fn or "install")()
        sys.stderr.write(f"[staging] round2 import-time {_spec}: {_ok}\n")
    except Exception as _exc:
        sys.stderr.write(f"[staging] round2 import-time {_spec} FAILED: {_exc!r}\n")
    sys.stderr.flush()

# Round-2 patches: OMLX_ROUND2_PATCHES="path/to/patch.py,another/patch.py" (each exposes install()), applied after model load
_r2 = [x for x in os.environ.get("OMLX_ROUND2_PATCHES", "").split(",") if x.strip()]
if _r2:
    import importlib.util as _ilu2
    from omlx.engine.vlm import VLMBatchedEngine as _VBE2
    _prev_start2 = _VBE2.start
    async def _r2_start(self):
        await _prev_start2(self)
        model = getattr(self, "_vlm_model", None)
        for i, path in enumerate(_r2):
            try:
                _p, _, _fn = path.strip().partition(":")
                spec = _ilu2.spec_from_file_location(f"round2_patch_{i}", os.path.expanduser(_p)); mod = _ilu2.module_from_spec(spec); spec.loader.exec_module(mod)
                _f = getattr(mod, _fn or "install"); ok = _f(model) if _wants_model(_f) else _f()
                sys.stderr.write(f"[staging] round2 {path.strip()}: install()={ok}\n")
            except Exception as exc:
                sys.stderr.write(f"[staging] round2 {path.strip()} FAILED: {exc!r}\n")
        sys.stderr.flush()
    _VBE2.start = _r2_start

sys.argv = ["omlx"] + sys.argv[1:]
runpy.run_module("omlx.cli", run_name="__main__")
