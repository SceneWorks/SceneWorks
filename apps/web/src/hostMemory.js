export const HOST_MEMORY_KINDS = Object.freeze({
  unified: "unified",
  dedicated: "dedicated",
});

function positiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function normalizedHostMemory(gb, kind, platform) {
  const value = positiveNumber(gb);
  if (value === null || !Object.values(HOST_MEMORY_KINDS).includes(kind)) {
    return null;
  }
  return {
    gb: value,
    kind,
    platform: typeof platform === "string" ? platform : null,
  };
}

// The runtime capability decides which backend is active; memory kind only decides whether this
// reading can safely budget that backend. A Mac in MLX observe mode, for example, can report unified
// memory while the active lane is still Candle. Treat that mismatch as unknown rather than selecting
// the wrong lane's evidence or pretending shared RAM is dedicated VRAM.
export function hostMemoryGbForBackend(hostMemory, backend) {
  const expectedKind =
    backend === "mlx" ? HOST_MEMORY_KINDS.unified : HOST_MEMORY_KINDS.dedicated;
  return hostMemory?.kind === expectedKind ? positiveNumber(hostMemory.gb) : null;
}

// Normalize the desktop Tauri probe without collapsing unified memory and dedicated VRAM into one
// untyped scalar. New desktop builds provide memoryKind/memoryMb; the named legacy fields keep the
// web compatible with an older sidecar during an app update.
export function hostMemoryFromGpuInfo(info) {
  if (!info || typeof info !== "object") {
    return null;
  }
  const memoryMb = positiveNumber(info.memoryMb);
  const normalized = normalizedHostMemory(
    memoryMb === null ? null : memoryMb / 1024,
    info.memoryKind,
    info.platform,
  );
  if (normalized) {
    return normalized;
  }
  const unified = positiveNumber(info.unifiedMemoryMb);
  if (unified !== null) {
    return normalizedHostMemory(unified / 1024, HOST_MEMORY_KINDS.unified, info.platform);
  }
  const dedicated = positiveNumber(info.gpuMemoryMb);
  return dedicated === null
    ? null
    : normalizedHostMemory(dedicated / 1024, HOST_MEMORY_KINDS.dedicated, info.platform);
}

// Normalize the remote REST probe. memoryKind/memoryGb is the authoritative pair; the old named
// fields remain a compatibility fallback and are selected by platform rather than null-coalescing
// across memory kinds.
export function hostMemoryFromCapabilities(caps) {
  if (!caps || typeof caps !== "object") {
    return null;
  }
  const normalized = normalizedHostMemory(
    caps.memoryGb,
    caps.memoryKind,
    caps.platform,
  );
  if (normalized) {
    return normalized;
  }
  if (caps.platform === "macos") {
    return normalizedHostMemory(
      caps.unifiedMemoryGb,
      HOST_MEMORY_KINDS.unified,
      caps.platform,
    );
  }
  return normalizedHostMemory(
    caps.gpuMemoryGb,
    HOST_MEMORY_KINDS.dedicated,
    caps.platform,
  );
}
