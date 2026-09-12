"""Test-side stand-in for ``mlx_vlm.models.qwen4_exp.language._PLE_IO_POOL``.

kdev has mlx but not mlx_vlm (the vendored copy lives inside oMLX.app, which is
read-only and pulls its own mlx).  The only thing ple-fix/patch.py imports from
it is the 48-worker IO pool declared at language.py:1919, so the tests register
a module with exactly that object.  Production imports the real one.
"""
import sys
import types
from concurrent.futures import ThreadPoolExecutor


def install_stub():
    if "mlx_vlm.models.qwen4_exp.language" in sys.modules:
        return sys.modules["mlx_vlm.models.qwen4_exp.language"]
    mod = types.ModuleType("mlx_vlm.models.qwen4_exp.language")
    mod._PLE_IO_POOL = ThreadPoolExecutor(max_workers=48, thread_name_prefix="ple-io")
    for name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.qwen4_exp"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
    sys.modules["mlx_vlm.models.qwen4_exp"].language = mod
    sys.modules["mlx_vlm.models.qwen4_exp.language"] = mod
    return mod
