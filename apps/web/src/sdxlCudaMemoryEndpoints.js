const GIB = 1024 ** 3;

// Catalog guidance for one measured request coordinate. Runtime admission still checks the actual
// request, artifact bytes and available VRAM; these figures never replace the resident manifest row.
export function sdxlCudaMemoryEstimate(model, tier, backend, projection = import.meta.env.VITE_SDXL_CUDA_MEMORY_ENDPOINTS) {
  if (model?.id !== "sdxl" || tier !== "q4" || backend !== "candle") return null;
  const variant = model.variants?.find((variant) => variant.variant === "q4");
  if (Array.isArray(model.variants) && (!variant || variant.pendingArtifact || variant.installState === "pending")) return null;
  let data = projection;
  if (typeof data === "string") {
    try { data = JSON.parse(data); } catch { return null; }
  }
  const geometry = data?.geometry;
  const settings = data?.settings;
  const parameters = data?.optimized?.parameters;
  const validEndpoint = (endpoint) => Number.isSafeInteger(endpoint?.peakBytes) && endpoint.peakBytes > 0
    && /^sdxl-candle-sdxl-staged-decode-attention-v\d+$/.test(endpoint.calibrationFingerprint ?? "")
    && /^imc-[a-f0-9]{20}$/.test(endpoint.recordId ?? "") && /^[a-f0-9]{64}$/.test(endpoint.sourceSha256 ?? "");
  if (data?.schemaVersion !== 1 || data.modelId !== "sdxl" || data.backend !== "candle"
      || data.tier !== "q4" || data.mode !== "text_to_image"
      || geometry?.width !== 1024 || geometry.height !== 1024 || geometry.batch !== 1
      || geometry.frames !== 1 || geometry.referenceCount !== 0
      || settings?.steps !== 30 || settings.guidance !== 7 || settings.sampler !== "ddim" || settings.count !== 1
      || !validEndpoint(data.resident) || !validEndpoint(data.optimized)
      || data.resident.calibrationFingerprint !== data.optimized.calibrationFingerprint
      || data.resident.recordId === data.optimized.recordId || data.resident.peakBytes < data.optimized.peakBytes
      || parameters?.stageResidency !== true || parameters.decodeTileEdge !== 256
      || parameters.decodeOverlap !== 64 || parameters.attentionChunkSize !== 16777216
      || Object.keys(parameters).length !== 4
      || !Number.isFinite(data.recaptureSpread) || data.recaptureSpread < 0 || data.recaptureSpread >= 1
      || !Number.isFinite(data.reserveGb) || data.reserveGb <= 0
      || typeof data.physicalEightGiBDeviceValidated !== "boolean") return null;
  const requiredGb = (peakBytes) => Math.ceil(peakBytes * (1 + data.recaptureSpread)) / GIB + data.reserveGb;
  const optimizedPeakGb = data.optimized.peakBytes / GIB;
  const residentPeakGb = data.resident.peakBytes / GIB;
  const optimizedHostGb = requiredGb(data.optimized.peakBytes);
  const residentHostGb = requiredGb(data.resident.peakBytes);
  return {
    optimizedPeakGb, residentPeakGb, optimizedHostGb, residentHostGb,
    label: `Q4 optimized ~${optimizedHostGb.toFixed(1)} GiB`,
    description: `Stock SDXL Q4, 1024 × 1024, one image, default sampling (30 steps, guidance 7, DDIM), no adapters. `
      + `Measured CUDA peaks: optimized ${optimizedPeakGb.toFixed(2)} GiB; resident ${residentPeakGb.toFixed(2)} GiB. `
      + `Estimated VRAM budgets including ${data.recaptureSpread * 100}% allowance and ${data.reserveGb} GiB reserve: `
      + `optimized ${optimizedHostGb.toFixed(2)} GiB; resident ${residentHostGb.toFixed(2)} GiB. `
      + `Fit depends on available VRAM; the capture does not validate a physical 8 GiB card.`,
  };
}
