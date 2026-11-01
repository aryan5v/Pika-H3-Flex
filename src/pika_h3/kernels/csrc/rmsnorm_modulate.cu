// SPDX-License-Identifier: Apache-2.0
// Fused RMSNorm(5376) + AdaLN shift/scale modulation for the MiniMax-H3 DiT blocks, reproducing the production BF16
// rounding sequence of `norm(x) * (1.0 + scale[idx]) + shift[idx]`:
//   n = bf16((x * rstd) * w);  t = bf16(1 + scale);  out = bf16(bf16(n * t) + shift)
// so it differs from the unfused path only where the FP32 sum of squares (reduction order) changes a rounding.
// One block of 96 threads handles kRows consecutive token rows. The weight, 1 + scale and shift rows of the current
// AdaLN index are staged in shared memory, so each row costs one streaming read and one streaming write.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHidden = 5376;
constexpr int kThreads = 96;
constexpr int kChunks = kHidden / 8;        // 16-byte chunks per row
constexpr int kPerThread = kChunks / kThreads;  // 7
constexpr int kRows = 16;                   // rows per block
static_assert(kChunks % kThreads == 0, "hidden size must split evenly across the block");

// Round two values to BF16 (nearest-even) with one paired conversion and return them as FP32.
__device__ __forceinline__ float2 round_bf16x2(float a, float b) { return __bfloat1622float2(__floats2bfloat162_rn(a, b)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int k = 16; k > 0; k >>= 1) v += __shfl_xor_sync(0xffffffffu, v, k);
  return v;
}

__global__ __launch_bounds__(kThreads) void rmsnorm_modulate_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scale, const __nv_bfloat16* __restrict__ shift,
    const int64_t* __restrict__ indices, __nv_bfloat16* __restrict__ out, float eps, int64_t num_rows) {
  extern __shared__ char smem[];
  __nv_bfloat16* ws = reinterpret_cast<__nv_bfloat16*>(smem);  // weight
  __nv_bfloat16* ts = ws + kHidden;                            // bf16(1 + scale[idx])
  __nv_bfloat16* sh = ts + kHidden;                            // shift[idx]
  __shared__ float partial[kThreads / 32];

  const int tid = threadIdx.x;
  for (int ci = tid; ci < kChunks; ci += kThreads) {
    *reinterpret_cast<uint4*>(ws + ci * 8) = *reinterpret_cast<const uint4*>(weight + ci * 8);
  }
  const int64_t row0 = (int64_t)blockIdx.x * kRows;
  int64_t current = -1;
  for (int r = 0; r < kRows; r++) {
    const int64_t row = row0 + r;
    if (row >= num_rows) return;  // uniform across the block
    const int64_t idx = indices[row];
    if (idx != current) {
      __syncthreads();
      for (int ci = tid; ci < kChunks; ci += kThreads) {
        const uint4 sv = *reinterpret_cast<const uint4*>(scale + idx * kHidden + ci * 8);
        const uint4 hv = *reinterpret_cast<const uint4*>(shift + idx * kHidden + ci * 8);
        const __nv_bfloat162* ps = reinterpret_cast<const __nv_bfloat162*>(&sv);
        uint4 tv;
        __nv_bfloat162* pt = reinterpret_cast<__nv_bfloat162*>(&tv);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const float2 g = __bfloat1622float2(ps[i]);
          pt[i] = __floats2bfloat162_rn(1.0f + g.x, 1.0f + g.y);
        }
        *reinterpret_cast<uint4*>(ts + ci * 8) = tv;
        *reinterpret_cast<uint4*>(sh + ci * 8) = hv;
      }
      __syncthreads();
      current = idx;
    }
    const int64_t base = row * kHidden + tid * 8;
    uint4 xv[kPerThread];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPerThread; i++) {
      xv[i] = *reinterpret_cast<const uint4*>(x + base + i * (kThreads * 8));
      const __nv_bfloat162* px = reinterpret_cast<const __nv_bfloat162*>(&xv[i]);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float2 f = __bfloat1622float2(px[k]);
        ss += f.x * f.x + f.y * f.y;
      }
    }
    ss = warp_sum(ss);
    if (tid % 32 == 0) partial[tid / 32] = ss;
    __syncthreads();
    const float rstd = rsqrtf((partial[0] + partial[1] + partial[2]) * (1.0f / (float)kHidden) + eps);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kPerThread; i++) {
      const int c8 = (tid + i * kThreads) * 8;
      const uint4 wv = *reinterpret_cast<const uint4*>(ws + c8);
      const uint4 tv = *reinterpret_cast<const uint4*>(ts + c8);
      const uint4 hv = *reinterpret_cast<const uint4*>(sh + c8);
      const __nv_bfloat162* pw = reinterpret_cast<const __nv_bfloat162*>(&wv);
      const __nv_bfloat162* pt = reinterpret_cast<const __nv_bfloat162*>(&tv);
      const __nv_bfloat162* ps = reinterpret_cast<const __nv_bfloat162*>(&hv);
      const __nv_bfloat162* px = reinterpret_cast<const __nv_bfloat162*>(&xv[i]);
      uint4 ov;
      __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float2 f = __bfloat1622float2(px[k]);
        const float2 w2 = __bfloat1622float2(pw[k]);
        const float2 t2 = __bfloat1622float2(pt[k]);
        const float2 g = __bfloat1622float2(ps[k]);
        const float2 n = round_bf16x2(f.x * rstd * w2.x, f.y * rstd * w2.y);
        const float2 m = round_bf16x2(n.x * t2.x, n.y * t2.y);
        po[k] = __floats2bfloat162_rn(m.x + g.x, m.y + g.y);
      }
      *reinterpret_cast<uint4*>(out + base + i * (kThreads * 8)) = ov;
    }
  }
}

void check_bf16_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), name,
              " must be a contiguous CUDA bfloat16 tensor");
}

}  // namespace

torch::Tensor rmsnorm_modulate(torch::Tensor x, torch::Tensor weight, torch::Tensor scale_table,
                               torch::Tensor shift_table, torch::Tensor indices, double eps) {
  check_bf16_cuda(x, "x");
  check_bf16_cuda(weight, "weight");
  check_bf16_cuda(scale_table, "scale table");
  check_bf16_cuda(shift_table, "shift table");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kHidden, "x must be [rows, 5376]");
  TORCH_CHECK(weight.numel() == kHidden, "weight must have 5376 elements");
  TORCH_CHECK(scale_table.dim() == 2 && scale_table.size(1) == kHidden && shift_table.sizes() == scale_table.sizes(),
              "scale/shift tables must be [entries, 5376]");
  TORCH_CHECK(indices.is_cuda() && indices.scalar_type() == torch::kLong && indices.is_contiguous() &&
                  indices.numel() == x.size(0),
              "indices must be a contiguous CUDA int64 tensor with one entry per row");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  const int64_t num_rows = x.size(0);
  if (num_rows == 0) return out;
  const int smem = 3 * kHidden * (int)sizeof(__nv_bfloat16);
  const int64_t grid = (num_rows + kRows - 1) / kRows;
  rmsnorm_modulate_kernel<<<(unsigned int)grid, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(scale_table.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(shift_table.data_ptr()),
      indices.data_ptr<int64_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps, num_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
