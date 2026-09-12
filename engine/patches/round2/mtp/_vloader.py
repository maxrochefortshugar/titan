"""Load mlx-vlm 0.7.0's qwen3_5 speculative verifier kernels standalone.

The module only needs mlx plus a handful of mlx-vlm helpers that the
quantized-head code paths never touch, so the unused imports are stubbed
and the real file is executed from the unpacked 0.7.0 wheel.
"""
import sys, types, importlib.util
from pathlib import Path

W = Path.home() / "inference-server/kernels/round2/mtp/wheels/unpacked"


def load():
    if "_v070_specver" in sys.modules:
        return sys.modules["_v070_specver"]
    pkg = types.ModuleType("_v070")
    pkg.__path__ = []
    sys.modules.setdefault("_v070", pkg)

    # stub the sibling modules the verifier imports but the quantized-head
    # helpers do not use
    for name, attrs in (
        ("_v070.activations", ("swiglu",)),
        ("_v070.base", ("LanguageModelOutput", "kv_sequence_length",
                        "scaled_dot_product_attention", "slice_kv_sequence")),
        ("_v070.gated_delta", ("gated_delta_update_with_states",)),
    ):
        m = types.ModuleType(name)
        for a in attrs:
            setattr(m, a, None)
        sys.modules[name] = m

    # the exact_speculative_verify module is real and small
    spec = importlib.util.spec_from_file_location(
        "_v070.exact_speculative_verify",
        W / "mlx_vlm/models/exact_speculative_verify.py",
    )
    esv = importlib.util.module_from_spec(spec)
    sys.modules["_v070.exact_speculative_verify"] = esv
    spec.loader.exec_module(esv)

    src = (W / "mlx_vlm/models/qwen3_5/speculative_verifier.py").read_text()
    src = src.replace("from ..activations", "from _v070.activations")
    src = src.replace("from ..base", "from _v070.base")
    src = src.replace("from ..exact_speculative_verify", "from _v070.exact_speculative_verify")
    src = src.replace("from .gated_delta", "from _v070.gated_delta")
    mod = types.ModuleType("_v070_specver")
    mod.__file__ = str(W / "mlx_vlm/models/qwen3_5/speculative_verifier.py")
    sys.modules["_v070_specver"] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod
