# SPDX-License-Identifier: Apache-2.0
"""Opt-in fused kernels for the MiniMax-H3 DiT blocks.

Two fusions replace production elementwise chains inside every ``MiniMaxH3TransformerBlock``:

* per-head ``norm_q`` / ``norm_k`` RMSNorm(128) + partial RoPE (96 of 128 channels) -> one kernel for Q and K;
* ``norm1`` / ``norm2`` RMSNorm(5376) + AdaLN ``* (1 + scale) + shift`` -> one kernel.

The kernels reproduce the production BF16 rounding sequence of each chain; they differ from the default path only
where the order of the FP32 RMS reduction changes a rounding, so the output is very close to, but not guaranteed
bit-identical with, the default path. The GEMMs, attention, gating, residuals and the text refiner are unchanged. The hooks are installed on module instances from this package, so the patched Diffusers
sources (and their runtime checksums) stay untouched. The block forward below mirrors
``MiniMaxH3TransformerBlock.forward`` of the checksum-pinned ``transformer_minimax_h3.py``.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from .backend import load_backend

HIDDEN_SIZE = 5376
NUM_HEADS = 56
HEAD_DIM = 128
ROTARY_DIM = 96
_ALIGNMENT = 16
_FALLBACK_WARNED = set()


def _aligned(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> bool:
    return (tensor.device == device and tensor.dtype == dtype and tensor.is_contiguous()
            and tensor.data_ptr() % _ALIGNMENT == 0)


def _warn_fallback(kernel: str) -> None:
    if kernel not in _FALLBACK_WARNED:
        _FALLBACK_WARNED.add(kernel)
        warnings.warn(f"fused {kernel} kernel skipped for inputs outside its contract; using the production path",
                      RuntimeWarning, stacklevel=3)


def _norm_modulate(ops, norm: nn.RMSNorm, hidden_states, scale, shift, indices):
    """``norm(hidden_states) * (1 + scale[indices]) + shift[indices]``, fused when the inputs allow it.

    ``indices`` come from the model's own packed layout (``timestep_indices * 3 + token_tags``); unlike
    ``index_select``, the kernel does not range-check them.
    """
    device = hidden_states.device
    weight, scale_table, shift_table = norm.weight.contiguous(), scale.contiguous(), shift.contiguous()
    indices = indices.contiguous()
    eligible = (
        device.type == "cuda" and hidden_states.dim() == 3 and hidden_states.shape[0] == 1
        and hidden_states.shape[2] == HIDDEN_SIZE and _aligned(hidden_states, device, torch.bfloat16)
        and all(_aligned(t, device, torch.bfloat16) for t in (weight, scale_table, shift_table))
        and indices.device == device and indices.dtype == torch.int64 and indices.numel() == hidden_states.shape[1]
    )
    if not eligible:
        _warn_fallback("RMSNorm+modulate")
        return norm(hidden_states) * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    rows = hidden_states.shape[1]
    out = ops.rmsnorm_modulate(hidden_states.view(rows, HIDDEN_SIZE), weight, scale_table, shift_table, indices,
                               float(norm.eps))
    return out.view_as(hidden_states)


def _qk_norm_rope(ops, attn, query, key, cos, sin):
    """Fused ``norm_q``/``norm_k`` + RoPE; returns ``(q, k)`` as ``[1, S, heads, head_dim]`` or None if ineligible."""
    device = query.device
    seq_len = query.shape[1] if query.dim() == 3 else -1
    wq, wk = attn.norm_q.weight.contiguous(), attn.norm_k.weight.contiguous()
    eligible = (
        device.type == "cuda" and query.dim() == 3 and query.shape[0] == 1
        and query.shape[2] == NUM_HEADS * HEAD_DIM and key.shape == query.shape
        and all(_aligned(t, device, torch.bfloat16) for t in (query, key, wq, wk))
        and all(_aligned(t, device, torch.float32) for t in (cos, sin))
        and tuple(cos.shape) == (seq_len, ROTARY_DIM) and tuple(sin.shape) == (seq_len, ROTARY_DIM)
    )
    if not eligible:
        _warn_fallback("QK-norm+RoPE")
        return None
    q, k = ops.qk_norm_rope(query.view(seq_len, -1), key.view(seq_len, -1), wq, wk, cos, sin, float(attn.norm_q.eps))
    return q.view(1, seq_len, NUM_HEADS, HEAD_DIM), k.view(1, seq_len, NUM_HEADS, HEAD_DIM)


def _make_processor_class():
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor, _apply_rotary_emb

    class FusedQKNormRoPEAttnProcessor(MiniMaxH3AttnProcessor):
        """``MiniMaxH3AttnProcessor`` with the Q/K RMSNorm + RoPE chain replaced by one fused kernel."""

        def __init__(self, ops, previous=None):
            self._ops = ops
            self._attention_backend = getattr(previous, "_attention_backend", None)
            self._parallel_config = getattr(previous, "_parallel_config", None)

        def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
            if rotary_emb is None or attn.fused_projections:
                return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
            fused = _qk_norm_rope(self._ops, attn, query, key, *rotary_emb)
            if fused is None:
                query = _apply_rotary_emb(attn.norm_q(query.unflatten(-1, (attn.heads, -1))), *rotary_emb)
                key = _apply_rotary_emb(attn.norm_k(key.unflatten(-1, (attn.heads, -1))), *rotary_emb)
            else:
                query, key = fused
            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend, parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.flatten(2, 3).type_as(query)
            hidden_states = attn.to_out[0](hidden_states)
            return attn.to_out[1](hidden_states)

    return FusedQKNormRoPEAttnProcessor


def _make_block_forward(block, ops):
    def forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)

        residual = hidden_states
        norm_hidden_states = _norm_modulate(ops, block.norm1, hidden_states, scale_msa, shift_msa, adaln_indices)
        attn_output = block.attn(norm_hidden_states, rotary_emb, attention_mask)
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attn_output

        residual = hidden_states
        norm_hidden_states = _norm_modulate(ops, block.norm2, hidden_states, scale_mlp, shift_mlp, adaln_indices)
        ff_output = block.ff(norm_hidden_states)
        return residual + gate_mlp.index_select(0, adaln_indices) * ff_output

    return forward


def _check_block(block) -> None:
    attn = block.attn
    norms = (block.norm1, block.norm2, attn.norm_q, attn.norm_k)
    if any(norm.weight is None or norm.eps is None for norm in norms):
        raise ValueError("fused DiT kernels need affine RMSNorm modules with an explicit eps")
    if (tuple(block.norm1.weight.shape) != (HIDDEN_SIZE,) or tuple(block.norm2.weight.shape) != (HIDDEN_SIZE,)
            or attn.heads != NUM_HEADS or attn.head_dim != HEAD_DIM
            or tuple(attn.norm_q.weight.shape) != (HEAD_DIM,) or attn.norm_q.eps != attn.norm_k.eps):
        raise ValueError(f"fused DiT kernels support hidden size {HIDDEN_SIZE}, {NUM_HEADS} heads x {HEAD_DIM}")


def install_fused_dit_kernels(model: nn.Module, *, backend: str = "auto") -> str:
    """Route every ``MiniMaxH3TransformerBlock`` of ``model`` through the fused kernels and return the backend name.

    Install after loading weights/adapters and before ``enable_group_offload``, so offload hooks wrap the fused
    forward. Calls whose inputs do not match the kernels' contract (dtype, device, layout, batch 1) use the
    production computation.
    """
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

    blocks = [module for module in model.modules() if isinstance(module, MiniMaxH3TransformerBlock)]
    if not blocks:
        raise ValueError("model has no MiniMaxH3TransformerBlock to accelerate")
    for block in blocks:
        _check_block(block)
    ops = load_backend(backend)
    processor_class = _make_processor_class()
    for block in blocks:
        block.forward = _make_block_forward(block, ops)
        block.attn.set_processor(processor_class(ops, block.attn.processor))
    model._pika_fused_dit_kernels = ops.name
    return ops.name
