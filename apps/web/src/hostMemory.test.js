import { describe, expect, it } from "vitest";
import {
  hostMemoryFromCapabilities,
  hostMemoryFromGpuInfo,
  hostMemoryGbForBackend,
} from "./hostMemory.js";

describe("typed host memory", () => {
  it("keeps an 8 GB unified pool distinct from 8 GB dedicated VRAM", () => {
    expect(
      hostMemoryFromCapabilities({
        platform: "macos",
        memoryKind: "unified",
        memoryGb: 8,
      }),
    ).toEqual({ gb: 8, kind: "unified", platform: "macos" });
    expect(
      hostMemoryFromCapabilities({
        platform: "windows",
        memoryKind: "dedicated",
        memoryGb: 8,
      }),
    ).toEqual({ gb: 8, kind: "dedicated", platform: "windows" });
  });

  it("uses platform-specific legacy fields without cross-kind null coalescing", () => {
    expect(
      hostMemoryFromCapabilities({
        platform: "windows",
        unifiedMemoryGb: 128,
        gpuMemoryGb: 24,
      }),
    ).toEqual({ gb: 24, kind: "dedicated", platform: "windows" });
    expect(
      hostMemoryFromCapabilities({
        platform: "macos",
        unifiedMemoryGb: 64,
        gpuMemoryGb: 24,
      }),
    ).toEqual({ gb: 64, kind: "unified", platform: "macos" });
  });

  it("normalizes the desktop probe for both memory kinds", () => {
    expect(
      hostMemoryFromGpuInfo({
        platform: "macos",
        memoryKind: "unified",
        memoryMb: 8192,
      }),
    ).toMatchObject({ gb: 8, kind: "unified" });
    expect(
      hostMemoryFromGpuInfo({
        platform: "windows",
        gpuMemoryMb: 8192,
      }),
    ).toMatchObject({ gb: 8, kind: "dedicated" });
  });

  it("uses the active backend as authority and rejects a mismatched memory kind", () => {
    const unified = { gb: 8, kind: "unified", platform: "macos" };
    const dedicated = { gb: 8, kind: "dedicated", platform: "windows" };
    expect(hostMemoryGbForBackend(unified, "mlx")).toBe(8);
    expect(hostMemoryGbForBackend(unified, "candle")).toBeNull();
    expect(hostMemoryGbForBackend(dedicated, "candle")).toBe(8);
    expect(hostMemoryGbForBackend(dedicated, "mlx")).toBeNull();
  });
});
