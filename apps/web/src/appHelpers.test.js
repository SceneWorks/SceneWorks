import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  buildLocalJobStack,
  failedJobNotice,
  generatedResultAssetCount,
  hasCapability,
  isActiveWorker,
  isAudioGenerationJob,
  isImageGenerationJob,
  isInterleaveJob,
  isPlaceholderOnlyGpuWorker,
  trainingRegistrationFailure,
  isSelectableGpuWorker,
  isVideoGenerationJob,
  jobFreshnessMs,
  mergeFreshJobs,
  noticeKindForJob,
  parseSseJson,
  readStoredAccent,
  readStoredTheme,
  reconcileAuthoritativeJobs,
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
  it("classifies image / video / audio / interleave jobs", () => {
    expect(isImageGenerationJob({ type: "image_generate" })).toBe(true);
    expect(isImageGenerationJob({ type: "video_generate" })).toBe(false);
    expect(isVideoGenerationJob({ type: "video_bridge" })).toBe(true);
    // SceneWorks Audio Studio (sc-13404): the audio-generation job type.
    expect(isAudioGenerationJob({ type: "audio_generate" })).toBe(true);
    expect(isAudioGenerationJob({ type: "video_generate" })).toBe(false);
    expect(isVideoGenerationJob({ type: "audio_generate" })).toBe(false);
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

  it("reconciles a stale active row to the authoritative terminal snapshot", () => {
    const current = [{
      id: "job-1",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:01:00Z",
    }];
    const server = [{
      id: "job-1",
      status: "completed",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:02:00Z",
    }];
    expect(reconcileAuthoritativeJobs(current, server)).toEqual(server);
  });

  it("retains a newer durable client row when snapshot timestamps tie", () => {
    const current = [{
      id: "job-1",
      status: "completed",
      revision: 8,
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:02:00Z",
    }];
    const staleServer = [{
      ...current[0],
      status: "running",
      revision: 7,
    }];
    expect(reconcileAuthoritativeJobs(current, staleServer)).toEqual(current);
  });

  it("drops an active row absent from the authoritative active set but retains capped terminal history", () => {
    const current = [
      { id: "stale-active", status: "running", createdAt: "2026-01-02T00:00:00Z" },
      { id: "local-history", status: "completed", createdAt: "2026-01-01T00:00:00Z" },
    ];
    expect(reconcileAuthoritativeJobs(current, [])).toEqual([current[1]]);
  });

  it("drops only explicitly cleared terminal history when the server snapshot is capped", () => {
    const retained = { id: "retained", status: "completed", createdAt: "2026-01-01T00:00:00Z" };
    const cleared = { id: "cleared", status: "failed", createdAt: "2025-12-31T00:00:00Z" };
    expect(reconcileAuthoritativeJobs([retained, cleared], [], ["cleared"])).toEqual([retained]);
  });

  it("lets a clear tombstone win when the same server snapshot also contains the row", () => {
    const cleared = {
      id: "cleared",
      status: "completed",
      revision: 4,
      createdAt: "2026-01-01T00:00:00Z",
    };
    expect(reconcileAuthoritativeJobs([cleared], [cleared], ["cleared"])).toEqual([]);
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

  it("builds the audio local-job lane from audio_generate jobs only (sc-13404)", () => {
    // The `audio` lane (rememberLocalGenerationJob('audio', job)) filters with isAudioGenerationJob,
    // so a co-resident image/video job in the same project never leaks into the audio stack.
    const jobs = [
      { id: "speak", type: "audio_generate", projectId: "p1", status: "running", createdAt: "2026-01-03T00:00:00Z" },
      { id: "pic", type: "image_generate", projectId: "p1", status: "running", createdAt: "2026-01-04T00:00:00Z" },
      { id: "clip", type: "video_generate", projectId: "p1", status: "running", createdAt: "2026-01-05T00:00:00Z" },
    ];
    const stack = buildLocalJobStack(["speak"], jobs, "p1", isAudioGenerationJob);
    const ids = stack.map((job) => job.id);
    expect(ids).toContain("speak");
    expect(ids).not.toContain("pic");
    expect(ids).not.toContain("clip");
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

describe("trainingRegistrationFailure", () => {
  // sc-15036: a `lora_train` job now produces EITHER an adapter or a FULL base checkpoint, and each
  // registrar reports under its own result key — only one of which is present on any given run.
  //
  // The bug this pins: the notice branch checked only `loraRegistered === false`, so a failed
  // BASE-CHECKPOINT registration (where `loraRegistered` is `undefined`) fell into the success
  // path. The user was told the run succeeded while the job's own result carried
  // `baseCheckpointRegistrationError` saying the ~8 GB checkpoint had not been registered — and the
  // catalog was refreshed as if something had landed.
  //
  // Discriminating: the adapter-failure and base-checkpoint-failure shapes differ ONLY in which key
  // carries the `false`, and both must surface; the two success shapes must both stay silent.
  it("surfaces a failed base-checkpoint registration, not just a failed LoRA one", () => {
    expect(trainingRegistrationFailure({ loraRegistered: false, loraRegistrationError: "no adapter found" })).toBe(
      "no adapter found",
    );
    expect(
      trainingRegistrationFailure({
        baseCheckpointRegistered: false,
        baseCheckpointRegistrationError: "no complete fine-tuned base checkpoint found",
      }),
    ).toBe("no complete fine-tuned base checkpoint found");
  });

  it("stays silent when the run's own registrar succeeded", () => {
    // An ABSENT key means "that registrar did not claim this job" — the normal case for the other
    // kind. It must never read as a failure, or every successful adapter run would show an error.
    expect(trainingRegistrationFailure({ loraRegistered: true, loraId: "lora_1" })).toBeNull();
    expect(trainingRegistrationFailure({ baseCheckpointRegistered: true, baseCheckpointModelId: "finetune_1" })).toBeNull();
    expect(trainingRegistrationFailure({})).toBeNull();
    expect(trainingRegistrationFailure(undefined)).toBeNull();
  });

  it("falls back to a generic message when the registrar reported failure without a reason", () => {
    expect(trainingRegistrationFailure({ baseCheckpointRegistered: false })).toBe(
      "Completed training but could not register the result.",
    );
  });
});
