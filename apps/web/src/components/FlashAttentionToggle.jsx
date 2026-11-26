import React from "react";

// SW-7276 (feature SW-F379): candle-gen routes every candle model's attention through one shared
// FlashAttention-2 dispatch behind a single per-job toggle, so the control applies to every
// candle-routed image and video model — not just SDXL. MLX (Mac) has no such kernel path, so it is
// hidden there; `backend` is the studio's `activeBackend` ("mlx" | "candle").
export const FLASH_ATTENTION_TOOLTIP =
  "FlashAttention-2 for this model's attention on CUDA (candle) — faster and lower VRAM. " +
  "Calls the kernel can't take (masked text encoders, f32 paths) stay on the standard path.";

export function flashAttentionAvailable(backend) {
  return backend === "candle";
}

export function FlashAttentionToggle({ backend, checked, onChange }) {
  if (!flashAttentionAvailable(backend)) return null;
  return (
    <label className="checkline flash-attn-toggle" title={FLASH_ATTENTION_TOOLTIP}>
      <input checked={checked} onChange={(event) => onChange(event.target.checked)} type="checkbox" />
      Flash attention
    </label>
  );
}
