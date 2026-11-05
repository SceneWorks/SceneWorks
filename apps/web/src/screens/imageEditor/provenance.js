

// Filename for a Save / Download export (sc-2434): the source name with an
// "-edited" suffix before the extension, always .png — the working image is
// rasterized to PNG, so the original extension would be misleading. Pure.
export function editedFilename(source) {
  const base = (source?.name || "image").replace(/\.[^./\\]+$/, "").trim() || "image";
  return `${base}-edited.png`;
}

// Provenance for a saved edit, stored under the new asset's top-level `extra`
// (sc-2434): which source it was derived from + the ordered edit chain
// (crop/upscale/…) applied this session. Pure for unit testing.
export function buildSaveProvenance({ source, edits, width, height, layers }) {
  const provenance = {
    editor: "image_editor",
    source: source?.assetId
      ? { kind: "asset", assetId: source.assetId, name: source.name ?? null }
      : { kind: "upload", name: source?.name ?? null },
    edits: edits ?? [],
    width: width ?? null,
    height: height ?? null,
  };
  // Layer summary (sc-6121): record what the flattened asset was composited from —
  // bottom→top name / opacity / blend / visibility. Omitted for the degenerate
  // single-layer document (it adds nothing over the flat bitmap, and keeps a plain
  // non-layered save's provenance byte-for-byte as it was before layers).
  if (Array.isArray(layers) && layers.length > 1) {
    provenance.layers = layers.map((layer) => ({
      name: layer.name,
      opacity: layer.opacity,
      blendMode: layer.blendMode,
      visible: layer.visible,
    }));
  }
  return provenance;
}
