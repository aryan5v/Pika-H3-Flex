# SPDX-License-Identifier: Apache-2.0
"""Optional fused GPU kernels for the H3 DiT forward."""
from .dit import install_fused_dit_kernels

__all__ = ["install_fused_dit_kernels"]
