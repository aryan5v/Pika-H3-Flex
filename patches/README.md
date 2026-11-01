# Diffusers compatibility patches

These patches apply to the public Diffusers revision pinned in `requirements.txt`.

- `0001-h3-adaln-fp32.patch` evaluates AdaLN activations in FP32 before casting to each projection dtype.
- `0002-h3-nfe-count.patch` keeps the requested number of generator evaluations for the denoising schedule.
- `0003-h3-vae-fast-fp32-decode.patch` speeds up the FP32 video VAE decode. The decoder stays FP32; see below.

The manifest records the upstream revision, patch checksums, and expected source checksums. The installer rejects unexpected source files and verifies the patched files.

```bash
python tools/apply_diffusers_patches.py
python tools/apply_diffusers_patches.py --check
```

## FP32 video decode settings

`0003-h3-vae-fast-fp32-decode.patch` reads three environment variables when the video VAE is loaded or decodes:

| Variable | Default | Effect |
| --- | --- | --- |
| `H3_VAE_EMU_GEMM` | `1` | Runs the decoder's FP32 linear layers through cuBLAS `CUBLAS_COMPUTE_32F_EMULATED_16BFX9`. Each FP32 operand is split exactly into three BF16 parts and accumulated in FP32, so only the summation order changes. TF32 stays off. `0` uses native FP32 GEMMs. |
| `H3_VAE_TILE_BATCH` | `28` | Decodes up to this many equal-shape spatial tiles of a temporal clip in one decoder call. Tiles are independent, so the math is unchanged. `1` decodes tiles one at a time. |
| `H3_VAE_COMPILE` | `0` | `1` compiles the tile decoder with `torch.compile`. The first decode in each process pays the compile time, so it only pays off when one process decodes several videos. |

The emulated GEMM needs cuBLAS 12.9 or newer and has been measured on GB200 only. If cuBLAS does not provide it, or the compiled decoder fails, the decoder falls back to the native FP32 path and logs a warning.
