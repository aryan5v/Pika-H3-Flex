// SPDX-License-Identifier: Apache-2.0
// Fused per-head RMSNorm(128) + partial RoPE (96 of 128 channels) for the MiniMax-H3 DiT query and key.
//
// Each lane owns the rotate-half pair of 8-channel chunks (c, c + 48) of one head, so the RoPE partner is already in
// registers and every 16-byte chunk is read once. Eight lanes cover one head (RMS reduction = 3 shuffles), one warp
// covers four heads and walks all 56 heads of one token. blockIdx.y selects Q or K.
//
// The kernel reproduces the production rounding sequence of `nn.RMSNorm` followed by `_apply_rotary_emb` in BF16:
//   n = bf16((x * rstd) * w);  out = bf16(bf16(n * bf16(cos)) + bf16(rotate_half(n) * bf16(sin)))
// so it differs from the unfused path only where the FP32 sum of squares (reduction order) changes a rounding.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHeads = 56;
constexpr int kHeadDim = 128;
constexpr int kRotaryDim = 96;
constexpr int kIters = kHeads / 4;  // heads per lane group
constexpr int kPrefetch = 3;
constexpr int kMinBlocks = 24;

__device__ __forceinline__ uint4 load16(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
// Round two values to BF16 (nearest-even) with one paired conversion and return them as FP32.
__device__ __forceinline__ float2 round_bf16x2(float a, float b) { return __bfloat1622float2(__floats2bfloat162_rn(a, b)); }
__device__ __forceinline__ void store16(void* p, uint4 v) { *reinterpret_cast<uint4*>(p) = v; }

__global__ __launch_bounds__(32, kMinBlocks) void qk_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ wq, const __nv_bfloat16* __restrict__ wk,
    const float* __restrict__ cos, const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out, __nv_bfloat16* __restrict__ k_out, float eps, int64_t seq_len) {
  const int lane = threadIdx.x & 31;
  const int slot = lane & 7;
  const int head_in_warp = lane >> 3;
  const int c1 = (slot < 6) ? slot : (6 + slot);
  const int c2 = (slot < 6) ? (slot + 6) : (c1 + 2);
  const int64_t tok = blockIdx.x;
  if (tok >= seq_len) return;
  const bool is_q = (blockIdx.y == 0);
  const __nv_bfloat16* __restrict__ src = is_q ? q : k;
  __nv_bfloat16* __restrict__ dst = is_q ? q_out : k_out;
  const __nv_bfloat16* __restrict__ w = is_q ? wq : wk;

  const bool rotary = slot < 6;  // chunks 12..15 (channels 96..127) pass through
  float w1[8], w2[8], cos1[8], sin1[8], cos2[8], sin2[8];
  {
    const uint4 vw1 = load16(w + c1 * 8);
    const uint4 vw2 = load16(w + c2 * 8);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&vw1)[i]);
      w1[2 * i] = f.x; w1[2 * i + 1] = f.y;
      f = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&vw2)[i]);
      w2[2 * i] = f.x; w2[2 * i + 1] = f.y;
    }
    if (rotary) {
      const float* pc1 = cos + tok * kRotaryDim + c1 * 8;
      const float* ps1 = sin + tok * kRotaryDim + c1 * 8;
      const float* pc2 = cos + tok * kRotaryDim + c2 * 8;
      const float* ps2 = sin + tok * kRotaryDim + c2 * 8;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        cos1[i] = round_bf16(pc1[i]); sin1[i] = round_bf16(ps1[i]);
        cos2[i] = round_bf16(pc2[i]); sin2[i] = round_bf16(ps2[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) { cos1[i] = 1.f; sin1[i] = 0.f; cos2[i] = 1.f; sin2[i] = 0.f; }
    }
  }

  const int64_t base = tok * (int64_t)(kHeads * kHeadDim);
  const int off1 = c1 * 8, off2 = c2 * 8;
  uint4 buf1[kPrefetch], buf2[kPrefetch];
#pragma unroll
  for (int j = 0; j < kPrefetch; j++) {
    const int64_t off = base + (int64_t)(j * 4 + head_in_warp) * kHeadDim;
    buf1[j] = load16(src + off + off1);
    buf2[j] = load16(src + off + off2);
  }
#pragma unroll
  for (int i = 0; i < kIters; i++) {
    const int s = i % kPrefetch;
    const int64_t off = base + (int64_t)(i * 4 + head_in_warp) * kHeadDim;
    const uint4 v1 = buf1[s], v2 = buf2[s];
    float y1[8], y2[8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&v1)[j]);
      y1[2 * j] = f.x; y1[2 * j + 1] = f.y;
      f = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&v2)[j]);
      y2[2 * j] = f.x; y2[2 * j + 1] = f.y;
    }
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) { s1 = fmaf(y1[j], y1[j], s1); s2 = fmaf(y2[j], y2[j], s2); }
    float ss = s1 + s2;
#pragma unroll
    for (int m = 1; m < 8; m <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
    const float rstd = rsqrtf(ss * (1.0f / (float)kHeadDim) + eps);
    uint4 o1, o2;
    __nv_bfloat162* po1 = reinterpret_cast<__nv_bfloat162*>(&o1);
    __nv_bfloat162* po2 = reinterpret_cast<__nv_bfloat162*>(&o2);
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      // RMSNorm outputs of channels c (n1) and c + 48 (n2), two channels per paired conversion
      const float2 n1 = round_bf16x2(y1[j] * rstd * w1[j], y1[j + 1] * rstd * w1[j + 1]);
      const float2 n2 = round_bf16x2(y2[j] * rstd * w2[j], y2[j + 1] * rstd * w2[j + 1]);
      if (rotary) {
        // rotate_half: channel c pairs with -n(c + 48), channel c + 48 with n(c)
        const float2 a1 = round_bf16x2(n1.x * cos1[j], n1.y * cos1[j + 1]);
        const float2 b1 = round_bf16x2(n2.x * sin1[j], n2.y * sin1[j + 1]);
        const float2 a2 = round_bf16x2(n2.x * cos2[j], n2.y * cos2[j + 1]);
        const float2 b2 = round_bf16x2(n1.x * sin2[j], n1.y * sin2[j + 1]);
        po1[j / 2] = __floats2bfloat162_rn(a1.x - b1.x, a1.y - b1.y);
        po2[j / 2] = __floats2bfloat162_rn(a2.x + b2.x, a2.y + b2.y);
      } else {
        po1[j / 2] = __floats2bfloat162_rn(n1.x, n1.y);
        po2[j / 2] = __floats2bfloat162_rn(n2.x, n2.y);
      }
    }
    store16(dst + off + off1, o1);
    store16(dst + off + off2, o2);
    if (i + kPrefetch < kIters) {
      const int64_t next = base + (int64_t)((i + kPrefetch) * 4 + head_in_warp) * kHeadDim;
      buf1[s] = load16(src + next + off1);
      buf2[s] = load16(src + next + off2);
    }
  }
}

void check_bf16_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), name,
              " must be a contiguous CUDA bfloat16 tensor");
}

}  // namespace

std::vector<torch::Tensor> qk_norm_rope(torch::Tensor query, torch::Tensor key, torch::Tensor wq, torch::Tensor wk,
                                        torch::Tensor cos, torch::Tensor sin, double eps) {
  check_bf16_cuda(query, "query");
  check_bf16_cuda(key, "key");
  check_bf16_cuda(wq, "norm_q weight");
  check_bf16_cuda(wk, "norm_k weight");
  TORCH_CHECK(query.dim() == 2 && query.size(1) == kHeads * kHeadDim, "query must be [S, 7168]");
  TORCH_CHECK(key.sizes() == query.sizes(), "key must match query");
  TORCH_CHECK(wq.numel() == kHeadDim && wk.numel() == kHeadDim, "norm weights must have 128 elements");
  const int64_t seq_len = query.size(0);
  for (const auto& t : {cos, sin}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous() && t.dim() == 2 &&
                    t.size(0) == seq_len && t.size(1) == kRotaryDim,
                "cos/sin must be contiguous CUDA float32 [S, 96]");
  }
  const c10::cuda::CUDAGuard guard(query.device());
  auto q_out = torch::empty_like(query);
  auto k_out = torch::empty_like(key);
  if (seq_len == 0) return {q_out, k_out};
  const dim3 grid((unsigned int)seq_len, 2u);
  qk_norm_rope_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(wq.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(wk.data_ptr()),
      cos.data_ptr<float>(), sin.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr()),
      (float)eps, seq_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_out};
}
