# SDXL Tile ControlNet (Detail enhance)

The xinsir SDXL tile ControlNet is a small utility dependency SceneWorks uses only for the Image Editor's **Detail** enhancer. **It does not generate images on its own, and there is nothing to prompt.** The Detail tool feeds the working image through this ControlNet on an SDXL/RealVisXL backbone to refine fine texture while keeping the original composition.

## Installation

The native worker (MLX on macOS, candle on Windows/CUDA) resolves this model from the shared Hugging Face cache and does **not** auto-download it. Install it once from the **Models** screen — or use the one-click **Download** button the Image Editor's Detail panel shows when it is missing. It downloads into the shared Hugging Face cache, so every SDXL detail-capable backbone reuses the same copy.

If Detail enhance reports "tile ControlNet weights not found (download xinsir/controlnet-tile-sdxl-1.0)," this is the model to install.

## Practical Notes

There is nothing to prompt here. The two Detail-panel sliders do the work:

- **Detail amount** — how much new fine texture the refine pass invents (higher = more).
- **Structure lock** — how strongly the tile ControlNet holds the result to the source composition (higher = closer to the original).

apache-2.0 licensed, commercial use OK, ungated. Roughly 2.5 GB on disk.
