// Pure mask helpers for the Image Editor's inpaint-mask + smart-select tool (sc-2436 /
// sc-3751 / sc-6105). Extracted verbatim from ImageEditor.jsx (sc-9752, F-052 follow-up)
// so the stateful `useMaskTool` hook can share them without importing back into
// ImageEditor.jsx (a cycle). ImageEditor.jsx re-exports every symbol here to keep its
// public surface — and its test imports — byte-for-byte unchanged. Pure / React-free.

// The `POST /api/v1/jobs` body for a smart-select image_segment job (sc-3751 / backend
// sc-6105). Same generic-jobs shape as upscale/detail; the worker (native-MLX SAM3) reads
// `sourceAssetId` + a `box` prompt `[x1,y1,x2,y2]` in source-image pixel coords and returns a
// binary mask asset. Pure for testing.
export function buildSegmentJobBody({ project, requestedGpu, sourceAssetId, box, displayName }) {
  return {
    type: "image_segment",
    projectId: project.id,
    projectName: project.name ?? null,
    requestedGpu,
    payload: {
      projectId: project.id,
      sourceAssetId,
      box,
      displayName,
    },
  };
}

// Convert an image-pixel rect `{x,y,width,height}` to a SAM3 box prompt `[x1,y1,x2,y2]`, ordered
// (positive width/height) and rounded to whole pixels. Pure for testing.
export function rectToSegmentBox(rect) {
  const x1 = Math.round(Math.min(rect.x, rect.x + rect.width));
  const y1 = Math.round(Math.min(rect.y, rect.y + rect.height));
  const x2 = Math.round(Math.max(rect.x, rect.x + rect.width));
  const y2 = Math.round(Math.max(rect.y, rect.y + rect.height));
  return [x1, y1, x2, y2];
}

// The smart-select mask preview tint: translucent pink, matching the brush-stroke color so the
// auto-mask and any brush refinements read as one selection.
export const MASK_PREVIEW_RGBA = [255, 40, 120, 128];

// Recolor a decoded white-on-black mask's RGBA buffer in place to pink-on-transparent for the
// on-canvas preview: foreground (luminance > 127) → translucent pink, background → transparent.
// Pure (operates on the pixel buffer) so it's unit-testable without a real canvas.
export function tintMaskRgbaInPlace(data) {
  const [r, g, b, a] = MASK_PREVIEW_RGBA;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 127) {
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = a;
    } else {
      data[i + 3] = 0;
    }
  }
  return data;
}

// Whether the brush strokes form an actual mask region (at least one non-erase
// stroke with a drawn segment). Erase-only strokes don't count. Pure.
export function maskHasContent(lines) {
  return (lines ?? []).some((line) => !line.erase && (line.points?.length ?? 0) >= 2);
}
