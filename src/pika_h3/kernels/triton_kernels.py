# SPDX-License-Identifier: Apache-2.0
"""Triton versions of the fused DiT kernels, used when the CUDA extension cannot be built.

Same math as ``csrc/``: the production BF16 rounding sequence of ``nn.RMSNorm`` followed by the rotary embedding
or the AdaLN modulation. Signatures match the extension: 2-D ``[rows, channels]`` BF16 inputs.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

NAME = "triton"
_HEADS_PER_PROGRAM = 16


@triton.jit
def _round_bf16(x):
    # Round-to-nearest-even to BF16, kept in FP32. Integer bit arithmetic, because a float down/up-cast pair can be
    # simplified away by the compiler, which would skip the production rounding.
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _qk_norm_rope_kernel(Q, K, WQ, WK, COS, SIN, QO, KO, eps,
                         HEADS: tl.constexpr, D: tl.constexpr, RD: tl.constexpr,
                         BH: tl.constexpr, GROUPS: tl.constexpr):
    pid = tl.program_id(0)
    tok = (pid // GROUPS).to(tl.int64)
    h = (pid % GROUPS) * BH + tl.arange(0, BH)
    head_mask = h < HEADS
    d = tl.arange(0, D)
    half = RD // 2
    rotary = d < RD
    low = d < half
    high = (d >= half) & rotary
    partner = (d + half) % RD
    c = _round_bf16(tl.load(COS + tok * RD + (d % RD), mask=rotary, other=0.0))
    s = _round_bf16(tl.load(SIN + tok * RD + (d % RD), mask=rotary, other=0.0))
    base = tok * (HEADS * D) + h[:, None] * D
    mask = head_mask[:, None]
    for i in tl.static_range(2):
        if i == 0:
            src = Q
            dst = QO
            w = tl.load(WQ + d).to(tl.float32)
            w_partner = tl.load(WQ + partner, mask=rotary, other=0.0).to(tl.float32)
        else:
            src = K
            dst = KO
            w = tl.load(WK + d).to(tl.float32)
            w_partner = tl.load(WK + partner, mask=rotary, other=0.0).to(tl.float32)
        x = tl.load(src + base + d[None, :], mask=mask, other=0.0).to(tl.float32)
        x_up = tl.load(src + base + half + d[None, :], mask=mask & low[None, :], other=0.0).to(tl.float32)
        x_down = tl.load(src + base - half + d[None, :], mask=mask & high[None, :], other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / D) + eps)[:, None]
        n = _round_bf16(x * rstd * w[None, :])                              # RMSNorm output
        n_partner = _round_bf16((x_up + x_down) * rstd * w_partner[None, :])  # n(c + 48) or n(c - 48)
        rotated = tl.where(low[None, :], -n_partner, n_partner)             # rotate_half
        y = tl.where(rotary[None, :], _round_bf16(n * c[None, :]) + _round_bf16(rotated * s[None, :]), n)
        tl.store(dst + base + d[None, :], y.to(tl.bfloat16), mask=mask)


@triton.jit
def _rmsnorm_modulate_kernel(X, W, SCALE, SHIFT, IDX, OUT, eps, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(X + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / H) + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    idx = tl.load(IDX + row)
    scale = tl.load(SCALE + idx * H + offs, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(SHIFT + idx * H + offs, mask=mask, other=0.0).to(tl.float32)
    n = _round_bf16(x * rstd * w)                      # RMSNorm output
    y = _round_bf16(n * _round_bf16(1.0 + scale)) + shift  # * (1 + scale) + shift, each op rounded as in BF16
    tl.store(OUT + row * H + offs, y.to(tl.bfloat16), mask=mask)


def qk_norm_rope(query, key, wq, wk, cos, sin, eps, heads=56):
    """``query``/``key`` ``[S, heads * 128]`` BF16, ``cos``/``sin`` ``[S, rotary_dim]`` FP32 -> ``[q, k]``."""
    seq_len, width = query.shape
    head_dim = width // heads
    q_out = torch.empty_like(query)
    k_out = torch.empty_like(key)
    groups = triton.cdiv(heads, _HEADS_PER_PROGRAM)
    with torch.cuda.device(query.device):
        _qk_norm_rope_kernel[(seq_len * groups,)](
            query, key, wq, wk, cos, sin, q_out, k_out, eps,
            HEADS=heads, D=head_dim, RD=cos.shape[-1], BH=_HEADS_PER_PROGRAM, GROUPS=groups, num_warps=4)
    return [q_out, k_out]


def rmsnorm_modulate(x, weight, scale_table, shift_table, indices, eps):
    """``x`` ``[rows, H]`` BF16, tables ``[entries, H]`` BF16, ``indices`` ``[rows]`` int64 -> ``[rows, H]``."""
    rows, hidden = x.shape
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _rmsnorm_modulate_kernel[(rows,)](
            x, weight, scale_table, shift_table, indices, out, eps,
            H=hidden, BLOCK=triton.next_power_of_2(hidden), num_warps=8)
    return out
