# Production launcher: runs oMLX's CLI in-process with the Qwen4-Exp prefill/decode patches applied after model load.
# Patches live in ~/inference-server/kernels/ple-fix. Opt-in by env (set in run-omlx.sh); if a patch fails the stock path is kept.
import inspect, os, sys, runpy, importlib.util
def _wants_model(fn):
    try:
        ps = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    return bool(ps) and ps[0].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.expanduser(path)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def _log(msg): sys.stderr.write(f"[prod-patch] {msg}\n"); sys.stderr.flush()

# --- import-time patches (module-function swaps; must run before the server imports the engine) ---
# int8 MoE gather for prefill (kernels/moe-int8), gate_up only via OMLX_MOE_INT8_SKIP_DOWN=1
if os.environ.get("OMLX_MOE_INT8_PREFILL") == "1":
    try:
        _m8 = _load("moe_int8_patch", "~/inference-server/kernels/moe-int8/patch.py"); _m8.install(); _log("moe-int8 prefill patch installed")
    except Exception as exc:
        _log(f"moe-int8 install failed (stock path kept): {exc!r}")
# generic: OMLX_ROUND2_IMPORT_PATCHES="path/patch.py[:func],..."
for _i, _spec in enumerate([x for x in os.environ.get("OMLX_ROUND2_IMPORT_PATCHES", "").split(",") if x.strip()]):
    try:
        _path, _, _fn = _spec.strip().partition(":")
        _ok = getattr(_load(f"r2_import_{_i}", _path), _fn or "install")(); _log(f"import-time patch {_spec}: {_ok}")
    except Exception as exc:
        _log(f"import-time patch {_spec} FAILED: {exc!r}")

# --- post-load patches (instance rebinding; run after VLMBatchedEngine.start) ---
_R2 = [x.strip() for x in os.environ.get("OMLX_ROUND2_PATCHES", "").split(",") if x.strip()]
if os.environ.get("OMLX_PLE_PACKED") == "1" or os.environ.get("OMLX_QWEN4_BF16_NORM") == "1" or _R2:
    from omlx.engine.vlm import VLMBatchedEngine
    _orig_start = VLMBatchedEngine.start
    async def _patched_start(self):
        await _orig_start(self)
        model = getattr(self, "_vlm_model", None)
        if model is None:
            return
        if os.environ.get("OMLX_PLE_PACKED") == "1":
            try:
                n = _load("ple_packed_patch", "~/inference-server/kernels/ple-fix/patch.py").apply_ple_packed_patch(model, self._model_name)
                _log(f"PLE packed table: {n} layer(s) patched (mode={os.environ.get('OMLX_PLE_PACKED_MODE')})")
            except Exception as exc:
                _log(f"PLE packed patch failed (stock path kept): {exc!r}")
        if os.environ.get("OMLX_QWEN4_BF16_NORM") == "1":
            try:
                ok = _load("qwen4_norm_patch", "~/inference-server/kernels/ple-fix/norm_patch.py").apply_bf16_norm_patch()
                _log(f"bf16 grouped norm: applied={ok}")
            except Exception as exc:
                _log(f"bf16 norm patch failed: {exc!r}")
        for i, path in enumerate(_R2):
            try:
                _p, _, _fn = path.partition(":"); _pm = _load(f"r2_post_{i}", _p); _f = getattr(_pm, _fn or "install"); ok = _f(model) if _wants_model(_f) else _f(); _log(f"post-load patch {path}: install()={ok}")
            except Exception as exc:
                _log(f"post-load patch {path} FAILED: {exc!r}")
    VLMBatchedEngine.start = _patched_start
    _log("hooks installed")
sys.argv = ["omlx"] + sys.argv[1:]
runpy.run_module("omlx.cli", run_name="__main__")
