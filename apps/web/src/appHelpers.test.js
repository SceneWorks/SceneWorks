import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  buildLocalJobStack,
  failedJobNotice,
  generatedResultAssetCount,
  hasCapability,
  isActiveWorker,
  isImageGenerationJob,
  isInterleaveJob,
  isPlaceholderOnlyGpuWorker,
  isSelectableGpuWorker,
  isVideoGenerationJob,
  jobFreshnessMs,
  mergeFreshJobs,
  noticeKindForJob,
  parseSseJson,
  readStoredAccent,
  readStoredTheme,
} from "./appHelpers.js";
import { DEFAULT_ACCENT } from "./accents.js";

// sc-8854 (F-052): behavior lock for the pure helpers extracted out of App.jsx.

describe("worker classification", () => {
  it("treats any non-offline worker as active", () => {
    expect(isActiveWorker({ status: "idle" })).toBe(true);
    expect(isActiveWorker({ status: "offline" })).toBe(false);
  });

  it("checks capabilities defensively", () => {
    expect(hasCapability({ capabilities: ["gpu"] }, "gpu")).toBe(true);
    expect(hasCapability({ capabilities: ["gpu"] }, "cpu")).toBe(false);
    expect(hasCapability({}, "gpu")).toBe(false);
  });

  it("flags placeholder-only GPU workers", () => {
    expect(isPlaceholderOnlyGpuWorker({ capabilities: ["gpu", "placeholder"] })).toBe(true);
    expect(isPlaceholderOnlyGpuWorker({ capabilities: ["gpu", "cuda"] })).toBe(false);
    expect(isPlaceholderOnlyGpuWorker({ capabilities: ["cpu"] })).toBe(false);
  });

  it("selects only real, non-placeholder GPU workers", () => {
    expect(isSelectableGpuWorker({ gpuId: "0", capabilities: ["gpu", "cuda"] })).toBe(true);
    expect(isSelectableGpuWorker({ gpuId: "cpu", capabilities: ["gpu"] })).toBe(false);
    expect(isSelectableGpuWorker({ gpuId: "0", capabilities: ["gpu", "placeholder"] })).toBe(false);
  });
});

describe("job classification + notices", () => {
  it("classifies image / video / interleave jobs", () => {
    expect(isImageGenerationJob({ type: "image_generate" })).toBe(true);
    expect(isImageGenerationJob({ type: "video_generate" })).toBe(false);
    expect(isVideoGenerationJob({ type: "video_bridge" })).toBe(true);
    expect(isInterleaveJob({ type: "image_interleave" })).toBe(true);
    expect(isInterleaveJob({ type: "image_generate" })).toBe(false);
  });

  it("builds a readable failure notice with a fallback detail", () => {
    expect(failedJobNotice({ type: "lora_train", error: "boom" })).toBe("lora train: boom");
    expect(failedJobNotice({ type: "image_generate" })).toBe(
      "image generate: Failed without additional worker detail.",
    );
  });

  it("maps lora import/train jobs to their own notice kind", () => {
    expect(noticeKindForJob({ type: "lora_import" })).toBe("lora-import");
    expect(noticeKindForJob({ type: "lora_train" })).toBe("lora-train");
    expect(noticeKindForJob({ type: "image_generate" })).toBe("general");
    expect(noticeKindForJob(null)).toBe("general");
  });

  it("counts generated result assets from either shape", () => {
    expect(generatedResultAssetCount({ result: { assetIds: ["a", "b"] } })).toBe(2);
    expect(generatedResultAssetCount({ result: { assets: [{}, {}, {}] } })).toBe(3);
    expect(generatedResultAssetCount({ result: {} })).toBe(0);
  });
});

describe("parseSseJson", () => {
  it("parses valid JSON payloads", () => {
    expect(parseSseJson({ data: '{"x":1}' }, "test")).toEqual({ x: 1 });
  });

  it("returns null and warns on malformed data", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(parseSseJson({ data: "not json" }, "test")).toBeNull();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

describe("job freshness + merge", () => {
  it("reads the freshest available timestamp, falling back to 0", () => {
    expect(jobFreshnessMs({ updatedAt: "2026-01-01T00:00:00Z" })).toBeGreaterThan(0);
    expect(jobFreshnessMs({})).toBe(0);
  });

  it("keeps the fresher of duplicate jobs and retains client-only entries", () => {
    const current = [
      { id: "a", status: "succeeded", createdAt: "2026-01-02T00:00:00Z", updatedAt: "2026-01-02T00:00:00Z" },
      { id: "client-only", status: "succeeded", createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z" },
    ];
    const server = [{ id: "a", status: "succeeded", createdAt: "2026-01-02T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z" }];
    const merged = mergeFreshJobs(current, server);
    const byId = Object.fromEntries(merged.map((job) => [job.id, job]));
    // The client's fresher "a" wins over the staler server copy.
    expect(byId.a.updatedAt).toBe("2026-01-02T00:00:00Z");
    // The client-only entry is retained.
    expect(byId["client-only"]).toBeTruthy();
  });
});

describe("buildLocalJobStack", () => {
  const isGen = (job) => job.type === "image_generate";

  it("includes remembered runs plus active project generation jobs, de-duped and oldest-first", () => {
    const jobs = [
      { id: "old", type: "image_generate", projectId: "p1", status: "succeeded", createdAt: "2026-01-01T00:00:00Z" },
      { id: "running", type: "image_generate", projectId: "p1", status: "running", createdAt: "2026-01-03T00:00:00Z" },
      { id: "queued", type: "image_generate", projectId: "p1", status: "queued", createdAt: "2026-01-04T00:00:00Z" },
      { id: "other", type: "image_generate", projectId: "p2", status: "running", createdAt: "2026-01-05T00:00:00Z" },
    ];
    const stack = buildLocalJobStack(["old"], jobs, "p1", isGen);
    const ids = stack.map((job) => job.id);
    // Remembered terminal "old" is kept; active p1 jobs added; p2 excluded; no dupes.
    expect(ids).toContain("old");
    expect(ids).toContain("running");
    expect(ids).toContain("queued");
    expect(ids).not.toContain("other");
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("returns an empty stack when there is no active project", () => {
    const jobs = [{ id: "a", type: "image_generate", projectId: "p1", status: "running", createdAt: "2026-01-01T00:00:00Z" }];
    expect(buildLocalJobStack([], jobs, null, isGen)).toEqual([]);
  });
});

describe("stored theme + accent readers", () => {
  const store = new Map();

  beforeEach(() => {
    store.clear();
    vi.stubGlobal("window", {
      localStorage: {
        getItem: (key) => (store.has(key) ? store.get(key) : null),
        setItem: (key, value) => store.set(key, value),
      },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reads a valid stored theme and defaults to light otherwise", () => {
    store.set("sceneworks-theme", "dark");
    expect(readStoredTheme()).toBe("dark");
    store.set("sceneworks-theme", "banana");
    expect(readStoredTheme()).toBe("light");
    store.delete("sceneworks-theme");
    expect(readStoredTheme()).toBe("light");
  });

  it("reads a valid stored accent and defaults otherwise", () => {
    store.set("sceneworks-accent", "not-an-accent");
    expect(readStoredAccent()).toBe(DEFAULT_ACCENT);
  });
});
