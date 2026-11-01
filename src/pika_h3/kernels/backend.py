# SPDX-License-Identifier: Apache-2.0
"""Select the fused DiT kernel implementation: the CUDA extension (compiled on first use) or Triton."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

BACKENDS = ("auto", "cuda", "triton")
_EXTENSION_NAME = "pika_h3_fused_dit_v1"
_CSRC = Path(__file__).resolve().parent / "csrc"
_SOURCES = ("bindings.cpp", "qk_norm_rope.cu", "rmsnorm_modulate.cu")
_CACHE: dict[str, SimpleNamespace] = {}


def _load_cuda():
    # Compiled once per environment into the torch extensions cache (TORCH_EXTENSIONS_DIR); requires nvcc and ninja.
    from torch.utils.cpp_extension import load

    module = load(name=_EXTENSION_NAME, sources=[str(_CSRC / name) for name in _SOURCES],
                  extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-std=c++17"], verbose=False)
    return SimpleNamespace(name="cuda", qk_norm_rope=module.qk_norm_rope, rmsnorm_modulate=module.rmsnorm_modulate)


def _load_triton():
    from . import triton_kernels

    return SimpleNamespace(name=triton_kernels.NAME, qk_norm_rope=triton_kernels.qk_norm_rope,
                           rmsnorm_modulate=triton_kernels.rmsnorm_modulate)


def load_backend(preference: str = "auto") -> SimpleNamespace:
    """Return an object with ``name``, ``qk_norm_rope`` and ``rmsnorm_modulate``.

    ``auto`` tries the CUDA extension and falls back to Triton with a warning; ``cuda`` and ``triton`` are strict.
    """
    if preference not in BACKENDS:
        raise ValueError(f"unknown fused kernel backend {preference!r}; expected one of {BACKENDS}")
    if preference in _CACHE:
        return _CACHE[preference]
    if preference == "triton":
        backend = _load_triton()
    elif preference == "cuda":
        backend = _load_cuda()
    else:
        try:
            backend = _load_cuda()
        except Exception as error:  # missing nvcc/ninja/headers: same math is available through Triton
            warnings.warn(f"Fused DiT CUDA extension unavailable ({type(error).__name__}: {error}); "
                          "using the Triton kernels.", RuntimeWarning, stacklevel=2)
            backend = _load_triton()
    _CACHE[preference] = backend
    return backend
