import { describe, it, expect } from "vitest";
import { resolveJobResultAssets, assetBatchIndex, jobAudioResultAssets } from "./jobResultAssets.js";

// Characterization tests (sc-8853): these lock in the CURRENT behavior of the
// four resolver copies this module replaces, proving the unification is
// behavior-preserving per call site. The option combos map to the old helpers:
//   ImageStudio.jobResultAssets      → { type:"image", sortByBatchIndex:true, mergeResultAssets:true }
//   VideoStudio.jobVideoResultAssets → { type:"video", mergeResultAssets:true }
//   QueueScreen.resolveJobAssets     → { mergeResultAssets:true } (type-agnostic)
//   characterPanels.jobImageAssets   → { type:"image", mergeResultAssets:false }

const imageOpts = { type: "image", sortByBatchIndex: true, mergeResultAssets: true };
const videoOpts = { type: "video", mergeResultAssets: true };
const queueOpts = { mergeResultAssets: true };
const characterOpts = { type: "image", mergeResultAssets: false };

function img(id, extra = {}) {
  return { id, type: "image", ...extra };
}
function vid(id, extra = {}) {
  return { id, type: "video", ...extra };
}
function aud(id, extra = {}) {
  return { id, type: "audio", ...extra };
}

describe("resolveJobResultAssets — assetIds branch", () => {
  it("preserves worker-emitted assetIds order (slot order), image lane", () => {
    const catalog = [img("b"), img("a"), img("c")];
    const job = { result: { assetIds: ["a", "b", "c"] } };
    expect(resolveJobResultAssets(job, catalog, imageOpts).map((a) => a.id)).toEqual(["a", "b", "c"]);
  });

  it("does NOT re-sort assetIds even when sortByBatchIndex is on", () => {
    // assetIds already carry slot order; a mismatched batchIndex must not reorder.
    const catalog = [img("a", { batchIndex: 5 }), img("b", { batchIndex: 0 })];
    const job = { result: { assetIds: ["a", "b"] } };
    expect(resolveJobResultAssets(job, catalog, imageOpts).map((a) => a.id)).toEqual(["a", "b"]);
  });

  it("filters assetIds to the requested type (image)", () => {
    const catalog = [img("a"), vid("v")];
    const job = { result: { assetIds: ["a", "v"] } };
    expect(resolveJobResultAssets(job, catalog, imageOpts).map((a) => a.id)).toEqual(["a"]);
  });

  it("video lane keeps only video assets", () => {
    const catalog = [img("a"), vid("v")];
    const job = { result: { assetIds: ["v", "a"] } };
    expect(resolveJobResultAssets(job, catalog, videoOpts).map((a) => a.id)).toEqual(["v"]);
  });

  it("queue lane keeps any truthy asset (type-agnostic)", () => {
    const catalog = [img("a"), vid("v")];
    const job = { result: { assetIds: ["a", "v"] } };
    expect(resolveJobResultAssets(job, catalog, queueOpts).map((a) => a.id)).toEqual(["a", "v"]);
  });

  it("merge lanes fall back to the embedded result record when not yet catalogued", () => {
    const catalog = []; // catalog has not caught up
    const job = { result: { assetIds: ["a"], assets: [img("a", { pending: true })] } };
    const out = resolveJobResultAssets(job, catalog, imageOpts);
    expect(out.map((a) => a.id)).toEqual(["a"]);
    expect(out[0].pending).toBe(true);
  });

  it("catalog record wins over the embedded record once known (merge lanes)", () => {
    const catalog = [img("a", { saved: true })];
    const job = { result: { assetIds: ["a"], assets: [img("a", { pending: true })] } };
    expect(resolveJobResultAssets(job, catalog, imageOpts)[0].saved).toBe(true);
  });

  it("character lane is catalog-only: a not-yet-catalogued id is dropped", () => {
    const catalog = []; // no embedded-record fallback for characterPanels
    const job = { result: { assetIds: ["a"], assets: [img("a")] } };
    expect(resolveJobResultAssets(job, catalog, characterOpts)).toEqual([]);
  });

  it("character lane resolves catalogued ids", () => {
    const catalog = [img("a"), vid("v")];
    const job = { result: { assetIds: ["a", "v"] } };
    expect(resolveJobResultAssets(job, catalog, characterOpts).map((a) => a.id)).toEqual(["a"]);
  });
});

describe("resolveJobResultAssets — result.assets branch (merge lanes)", () => {
  it("image lane maps embedded records through the catalog", () => {
    const catalog = [img("a", { saved: true })];
    const job = { result: { assets: [img("a"), img("b")] } };
    const out = resolveJobResultAssets(job, catalog, imageOpts);
    expect(out.map((a) => a.id)).toEqual(["a", "b"]);
    expect(out[0].saved).toBe(true); // catalog record substituted
  });

  it("queue lane keeps embedded records of any type", () => {
    const job = { result: { assets: [img("a"), vid("v")] } };
    expect(resolveJobResultAssets(job, [], queueOpts).map((a) => a.id)).toEqual(["a", "v"]);
  });

  it("character lane never consults result.assets (returns [])", () => {
    const job = { result: { assets: [img("a")] } };
    expect(resolveJobResultAssets(job, [], characterOpts)).toEqual([]);
  });
});

describe("resolveJobResultAssets — generationSetId branch", () => {
  it("image lane sorts the set by batch index", () => {
    const catalog = [
      img("late", { generationSetId: "g1", batchIndex: 2 }),
      img("first", { generationSetId: "g1", batchIndex: 0 }),
      img("mid", { generationSetId: "g1", batchIndex: 1 }),
      img("other", { generationSetId: "g2", batchIndex: 0 }),
    ];
    const job = { result: { generationSetId: "g1" } };
    expect(resolveJobResultAssets(job, catalog, imageOpts).map((a) => a.id)).toEqual(["first", "mid", "late"]);
  });

  it("video lane does NOT sort the set (catalog order preserved)", () => {
    const catalog = [
      vid("b", { generationSetId: "g1", batchIndex: 2 }),
      vid("a", { generationSetId: "g1", batchIndex: 0 }),
    ];
    const job = { result: { generationSetId: "g1" } };
    expect(resolveJobResultAssets(job, catalog, videoOpts).map((a) => a.id)).toEqual(["b", "a"]);
  });

  it("queue lane keeps any type in the set, unsorted", () => {
    const catalog = [
      img("i", { generationSetId: "g1" }),
      vid("v", { generationSetId: "g1" }),
      img("x", { generationSetId: "g2" }),
    ];
    const job = { result: { generationSetId: "g1" } };
    expect(resolveJobResultAssets(job, catalog, queueOpts).map((a) => a.id)).toEqual(["i", "v"]);
  });

  it("character lane filters the set to images, unsorted", () => {
    const catalog = [
      img("i", { generationSetId: "g1" }),
      vid("v", { generationSetId: "g1" }),
    ];
    const job = { result: { generationSetId: "g1" } };
    expect(resolveJobResultAssets(job, catalog, characterOpts).map((a) => a.id)).toEqual(["i"]);
  });
});

describe("resolveJobResultAssets — empty / guard cases", () => {
  it("returns [] when the job has no result", () => {
    expect(resolveJobResultAssets({}, [img("a")], imageOpts)).toEqual([]);
    expect(resolveJobResultAssets(null, [img("a")], queueOpts)).toEqual([]);
  });

  it("returns [] when a non-array catalog is passed", () => {
    const job = { result: { generationSetId: "g1" } };
    expect(resolveJobResultAssets(job, undefined, imageOpts)).toEqual([]);
  });
});

// jobAudioResultAssets (epic 13400 A5, sc-13405) is the audio twin of the video
// lane: it wraps resolveJobResultAssets(job, assets, { type: "audio" }), so the
// shared results zone (WorkerProgressCard audio-player) plays only audio outputs.
describe("jobAudioResultAssets", () => {
  it("resolves only audio assets from assetIds (slot order), dropping other types", () => {
    const catalog = [aud("a"), img("i"), vid("v")];
    const job = { result: { assetIds: ["v", "a", "i"] } };
    expect(jobAudioResultAssets(job, catalog).map((x) => x.id)).toEqual(["a"]);
  });

  it("falls back to the embedded audio result record before the catalog catches up", () => {
    const job = { result: { assets: [aud("a", { origin: "audio_studio" })] } };
    const out = jobAudioResultAssets(job, []);
    expect(out.map((x) => x.id)).toEqual(["a"]);
    expect(out[0].origin).toBe("audio_studio");
  });

  it("does NOT resolve a wrong-typed (image/video) job result as audio", () => {
    const catalog = [img("i"), vid("v")];
    const job = { result: { assetIds: ["i", "v"] } };
    expect(jobAudioResultAssets(job, catalog)).toEqual([]);
  });

  it("keeps the generationSetId audio members in catalog order (no batch sort — audio isn't image)", () => {
    const catalog = [
      aud("b", { generationSetId: "g1", batchIndex: 2 }),
      aud("a", { generationSetId: "g1", batchIndex: 0 }),
      vid("v", { generationSetId: "g1" }),
      img("x", { generationSetId: "g2" }),
    ];
    const job = { result: { generationSetId: "g1" } };
    expect(jobAudioResultAssets(job, catalog).map((x) => x.id)).toEqual(["b", "a"]);
  });

  it("returns [] when the job has no result", () => {
    expect(jobAudioResultAssets({}, [aud("a")])).toEqual([]);
  });
});

describe("assetBatchIndex", () => {
  it("prefers explicit batchIndex fields in order", () => {
    expect(assetBatchIndex({ batchIndex: 3 })).toBe(3);
    expect(assetBatchIndex({ recipe: { batchIndex: 4 } })).toBe(4);
    expect(assetBatchIndex({ recipe: { normalizedSettings: { batchIndex: 5 } } })).toBe(5);
    expect(assetBatchIndex({ lineage: { batchIndex: 6 } })).toBe(6);
  });

  it("falls back to a _NNNN filename suffix (0-based)", () => {
    expect(assetBatchIndex({ file: { path: "/x/img_0003.png" } })).toBe(2);
  });

  it("falls back to a trailing #N in the display name (0-based)", () => {
    expect(assetBatchIndex({ displayName: "Scene #4" })).toBe(3);
  });

  it("sorts unknown assets last", () => {
    expect(assetBatchIndex({})).toBe(Number.POSITIVE_INFINITY);
  });
});
