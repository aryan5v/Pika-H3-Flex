// SPDX-License-Identifier: Apache-2.0
// Python bindings for the fused MiniMax-H3 DiT kernels.
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> qk_norm_rope(torch::Tensor query, torch::Tensor key, torch::Tensor wq, torch::Tensor wk,
                                        torch::Tensor cos, torch::Tensor sin, double eps);
torch::Tensor rmsnorm_modulate(torch::Tensor x, torch::Tensor weight, torch::Tensor scale_table,
                               torch::Tensor shift_table, torch::Tensor indices, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_norm_rope", &qk_norm_rope, "Fused per-head RMSNorm(128) + partial RoPE(96) for Q and K");
  m.def("rmsnorm_modulate", &rmsnorm_modulate, "Fused RMSNorm(5376) + AdaLN shift/scale modulation");
}
