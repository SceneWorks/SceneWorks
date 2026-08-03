import { describe, expect, it } from "vitest";

import {
  livePreviewPlaceholderAssets,
  livePreviewState,
  modelForJob,
  modelPreviewSupport,
  PRE_DENOISE_STAGES,
  previewBackendForWorker,
} from "./previewSupport.js";

// sc-16965 (epic 16948). These are the predicates the card reads; the point of pinning them here is
// that "unknown" must stay a distinct third answer everywhere. Folding it into `false` is the exact
// regression this story exists to prevent — it would make the UI assert "this route cannot live
// preview" about a route it has no measurement for.

const candleWorker = { id: "w1", gpuId: "gpu-0", capabilities: ["gpu", "image_generate", "candle"] };
const mlxWorker = { id: "w2", gpuId: "mlx", capabilities: ["gpu", "image_generate"] };
const cpuWorker = { id: "w3", gpuId: "cpu-0", capabilities: ["cpu"] };

describe("previewBackendForWorker", () => {
  it("reads the candle lane marker the routing gate already keys off", () => {
    expect(previewBackendForWorker(candleWorker, {})).toBe("candle");
  });

  it("recognises the macOS worker by its fixed mlx gpu id", () => {
    expect(previewBackendForWorker(mlxWorker, {})).toBe("mlx");
  });

  it("falls back to the job's stamped backend when no worker is resolved", () => {
    // A queue snapshot can outlive the worker row; image jobs stamp backend_label(gpu_id).
    expect(previewBackendForWorker(null, { backend: "mlx" })).toBe("mlx");
  });

  it("returns null for anything it cannot identify", () => {
    expect(previewBackendForWorker(cpuWorker, {})).toBeNull();
    expect(previewBackendForWorker(undefined, undefined)).toBeNull();
    // A CUDA gpu id is NOT evidence of the candle engine lane on its own — the marker is.
    expect(previewBackendForWorker({ gpuId: "gpu-0", capabilities: ["gpu"] }, { backend: "gpu-0" }))
      .toBeNull();
  });
});

describe("modelPreviewSupport", () => {
  const model = { id: "krea_2_turbo", preview: { byBackend: { candle: true, mlx: false } } };

  it("answers per backend", () => {
    expect(modelPreviewSupport(model, "candle")).toBe(true);
    expect(modelPreviewSupport(model, "mlx")).toBe(false);
  });

  it("returns null — not false — when the backend has no entry", () => {
    expect(modelPreviewSupport(model, "torch")).toBeNull();
  });

  it("returns null for a model with no preview block and for a null backend", () => {
    expect(modelPreviewSupport({ id: "external" }, "candle")).toBeNull();
    expect(modelPreviewSupport(model, null)).toBeNull();
    expect(modelPreviewSupport(null, "candle")).toBeNull();
  });

  it("ignores a non-boolean value rather than coercing it", () => {
    expect(modelPreviewSupport({ preview: { byBackend: { candle: "yes" } } }, "candle")).toBeNull();
  });
});

describe("modelForJob", () => {
  const models = [{ id: "sdxl" }, { id: "krea_2_turbo" }];

  it("resolves the job's generation model out of the catalog", () => {
    expect(modelForJob({ payload: { model: "sdxl" } }, models)).toBe(models[0]);
  });

  it("is null-safe on a missing catalog, payload, or id", () => {
    expect(modelForJob({ payload: { model: "sdxl" } }, null)).toBeNull();
    expect(modelForJob({}, models)).toBeNull();
    expect(modelForJob({ payload: { model: "nope" } }, models)).toBeNull();
  });
});

describe("livePreviewState", () => {
  const supporting = { id: "krea_2_turbo", preview: { byBackend: { candle: true } } };
  const inert = { id: "sdxl", preview: { byBackend: { candle: false } } };
  const running = (overrides) => ({
    type: "image_generate",
    status: "running",
    stage: "loading_model",
    ...overrides,
  });

  // A `model_download` payload also carries a `model` field, so without the job-type gate a download
  // of a preview-capable model would sit in `downloading` — a pre-denoise stage — and grow a
  // "preview starting…" cell for a job that never denoises anything.
  it("is `unknown` for a job type that never runs the denoise lane", () => {
    for (const type of ["model_download", "lora_train", "image_upscale", "video_generate"]) {
      expect(
        livePreviewState(running({ type, stage: "downloading" }), supporting, "candle"),
        type,
      ).toBe("unknown");
    }
  });

  it("covers the denoise job types", () => {
    expect(livePreviewState(running({ type: "image_edit" }), supporting, "candle")).toBe("pending");
  });

  it("is `live` while a streamed frame is on a running job", () => {
    const job = running({ result: { previewFrame: { dataUrl: "data:image/jpeg;base64,QQ==" } } });
    expect(livePreviewState(job, supporting, "candle")).toBe("live");
  });

  it("is `pending` for a supporting model still in its pre-denoise phase", () => {
    for (const stage of PRE_DENOISE_STAGES) {
      expect(livePreviewState(running({ stage }), supporting, "candle"), stage).toBe("pending");
    }
  });

  it("is `unsupported` for a wired route that does not preview", () => {
    expect(livePreviewState(running({}), inert, "candle")).toBe("unsupported");
  });

  it("is `unknown` when the catalog has no answer", () => {
    expect(livePreviewState(running({}), { id: "external" }, "candle")).toBe("unknown");
    expect(livePreviewState(running({}), supporting, null)).toBe("unknown");
  });

  // The first `generating` POST is emitted BEFORE frame 1 exists: the sampler calls `on_progress`
  // and only then the preview hook, both down one channel, and `GenEvent::Preview` deliberately
  // does not POST — the frame rides the NEXT Step update (`image_jobs/base.rs`). So step 1 is still
  // "no frame can have arrived", and dropping out of `pending` there blinks the cell away one step
  // before the frame restores it.
  it("holds `pending` across the first denoise step, which posts before frame 1 exists", () => {
    const job = running({ stage: "generating", message: "Image 1/2 — step 1/20." });
    expect(livePreviewState(job, supporting, "candle")).toBe("pending");
  });

  it("reads the step through an auto-tier disclosure prefix", () => {
    const job = running({
      stage: "generating",
      message: "Streaming blocks to keep this fit — Image 1/1 — step 1/8.",
    });
    expect(livePreviewState(job, supporting, "candle")).toBe("pending");
  });

  // The bound, one step wide: by step 2 an emitting route has posted a frame, so a supporting model
  // still showing nothing falls back to `supported` — no cell, i.e. today's behaviour — instead of
  // holding a placeholder for the whole render.
  it("leaves `pending` once the job reports a step past the first", () => {
    for (const message of ["Image 1/2 — step 2/20.", "Image 2/2 — step 17/20."]) {
      expect(
        livePreviewState(running({ stage: "generating", message }), supporting, "candle"),
        message,
      ).toBe("supported");
    }
  });

  it("falls back to `supported` when the message carries no step at all", () => {
    // Safe direction: an unparseable message degrades to the pre-sc-16965 card, never to a
    // placeholder stuck for the whole render.
    expect(livePreviewState(running({ stage: "generating" }), supporting, "candle")).toBe(
      "supported",
    );
    expect(
      livePreviewState(running({ stage: "generating", message: "Decoding." }), supporting, "candle"),
    ).toBe("supported");
  });

  it("prefers `live` over `pending` when step 1 already carries a frame", () => {
    const job = running({
      stage: "generating",
      message: "Image 1/1 — step 1/4.",
      result: { previewFrame: { dataUrl: "data:image/jpeg;base64,QQ==" } },
    });
    expect(livePreviewState(job, supporting, "candle")).toBe("live");
  });

  it("never shows the placeholder for a non-supporting route, whatever the step", () => {
    const job = running({ stage: "generating", message: "Image 1/1 — step 1/20." });
    expect(livePreviewState(job, inert, "candle")).toBe("unsupported");
    expect(livePreviewState(job, { id: "external" }, "candle")).toBe("unknown");
  });

  it("never reports `live` or `pending` on a terminal job", () => {
    for (const status of ["completed", "failed", "canceled"]) {
      // A stale frame can survive in the record after a failure between POSTs.
      const stale = running({
        status,
        stage: "queued",
        result: { previewFrame: { dataUrl: "data:image/jpeg;base64,QQ==" } },
      });
      expect(livePreviewState(stale, supporting, "candle"), status).toBe("supported");
      expect(livePreviewState(running({ status, stage: "queued" }), supporting, "candle"), status)
        .toBe("supported");
    }
  });
});

describe("livePreviewPlaceholderAssets", () => {
  it("emits exactly one interim cell for `pending`", () => {
    const assets = livePreviewPlaceholderAssets("pending");
    expect(assets).toHaveLength(1);
    expect(assets[0].__interim).toBe(true);
    expect(assets[0].__previewPlaceholder).toBe(true);
    // No url: the grid must not hand a urlless asset to AssetThumbnail.
    expect(assets[0].url).toBeUndefined();
  });

  it("emits nothing for every other state", () => {
    for (const state of ["live", "supported", "unsupported", "unknown"]) {
      expect(livePreviewPlaceholderAssets(state), state).toEqual([]);
    }
  });
});
