import { describe, expect, it } from "vitest";

import {
  catalogToWebPreviewSupport,
  derivePreviewSupport,
  parseEngineModelTable,
  PREVIEW_SUPPORT_GENERATOR,
} from "./previewSupportDerivation.js";
import previewSupport from "./previewSupport.json";
// Raw source text (Vite `?raw`) so the guard derives from the same bytes the generator reads. Both
// live outside the web root — see the server.fs.allow entries in vite.config.js (mirrors the
// style.txt / builtin.styles.jsonc pair).
import enginesSource from "../../../../crates/sceneworks-worker/src/engines.rs?raw";
import candleFactsRaw from "../../../../config/engine-capabilities/capabilities.candle.json?raw";
import previewSupportManifestRaw from "../../../../config/manifests/builtin.preview-support.jsonc?raw";
import { fallbackModels } from "../constants.js";

const factsFiles = [JSON.parse(candleFactsRaw)];
const derived = derivePreviewSupport(parseEngineModelTable(enginesSource), factsFiles);

// Guards sc-16965 (epic 16948). `config/manifests/builtin.preview-support.jsonc` — the block
// rust-api merges onto every /api/v1/models entry as `preview.byBackend` — must stay a mechanical
// derivation of (a) the stage-1 engine-capability dumps and (b) the MODEL_TABLE join in engines.rs.
// Never hand-edited. If either input moves — a family story flips a descriptor and the facts file is
// re-dumped, or a new MODEL_TABLE row lands — re-run `npm run gen:preview-support` (apps/web); this
// fails until both artifacts are regenerated.
//
// This is the half of the design that runs on EVERY PR. Stage 1 needs a linked engine registry and
// so can only run on macOS or a `backend-candle` lane (both dispatch-only), but it writes checked-in
// files, and everything below reads only those. That the stage-1 lane is dispatch-only is not an
// unguarded gap: sc-16951's `candle-gen-catalog` bidirectional test guards descriptor-level truth in
// the inference repo's CI continuously.
describe("preview-support catalog: the artifacts are derived, not authored", () => {
  it("re-deriving from engines.rs + the facts files reproduces builtin.preview-support.jsonc", () => {
    expect(JSON.parse(previewSupportManifestRaw)).toEqual(derived);
  });

  it("re-deriving reproduces the web fallback table previewSupport.json", () => {
    expect(previewSupport).toEqual(catalogToWebPreviewSupport(derived));
  });

  it("the two artifacts agree on every model (one derivation, two writes)", () => {
    const manifest = JSON.parse(previewSupportManifestRaw);
    expect(previewSupport.models).toEqual(manifest.models);
    expect(previewSupport.backends).toEqual(manifest.backends);
    expect(previewSupport.version).toBe(manifest.version);
  });

  it("names the generator so the only sanctioned update path is discoverable", () => {
    expect(JSON.parse(previewSupportManifestRaw).generatedBy).toBe(PREVIEW_SUPPORT_GENERATOR);
  });

  it("stamps the inference revision each facts file was dumped under", () => {
    const manifest = JSON.parse(previewSupportManifestRaw);
    for (const backend of manifest.backends) {
      expect(manifest.generatedFrom[backend].inferenceRevision).toMatch(/^[0-9a-f]{40}$/);
    }
  });
});

// The stage-1 facts file is a checked-in SOURCE, so it gets the same scrutiny style.txt gets: an
// empty or truncated one derives as "no route supports live preview", which is a confident wrong
// answer rather than a missing one. The Rust dumper refuses to write one; this refuses to read one.
describe("preview-support catalog: the stage-1 facts files are non-vacuous", () => {
  it("the candle dump carries the full registry, not an empty one", () => {
    const facts = JSON.parse(candleFactsRaw);
    expect(facts.backend).toBe("candle");
    expect(facts.engines.length).toBeGreaterThan(40);
    expect(facts.generatedFrom.inferenceRevision).toMatch(/^[0-9a-f]{40}$/);
  });

  it("an empty facts file is refused rather than derived from", () => {
    expect(() =>
      derivePreviewSupport(parseEngineModelTable(enginesSource), [
        { backend: "candle", engines: [] },
      ]),
    ).toThrow(/vacuous-green/);
  });

  it("deriving with no facts files at all is refused", () => {
    expect(() => derivePreviewSupport(parseEngineModelTable(enginesSource), [])).toThrow(
      /no stage-1 facts files/,
    );
  });
});

describe("preview-support catalog: the MODEL_TABLE join", () => {
  const rows = parseEngineModelTable(enginesSource);

  it("parses the full engines.rs model table", () => {
    expect(rows.length).toBeGreaterThan(40);
    expect(rows.every((row) => row.sceneworksId && row.engineId)).toBe(true);
  });

  it("is many-to-one: distinct model ids may share one engine id", () => {
    // z_image_edit runs the Turbo weights through the engine's img2img path, so both SceneWorks ids
    // resolve to the `z_image_turbo` engine — and therefore to the same preview answer.
    const shared = rows.filter((row) => row.engineId === "z_image_turbo").map((r) => r.sceneworksId);
    expect(shared).toEqual(expect.arrayContaining(["z_image_turbo", "z_image_edit"]));
    expect(previewSupport.models.z_image_turbo).toEqual(previewSupport.models.z_image_edit);
  });

  it("a MODEL_TABLE row with no matching engine facts is omitted, not defaulted to false", () => {
    // Absence must never be encoded as `false`: a backend that never registered the engine has no
    // opinion, and inventing one would let the UI claim "no live preview" about a route it cannot
    // see. Every emitted entry is backed by a real facts row.
    const factsIds = new Set(factsFiles[0].engines.map((engine) => engine.id));
    for (const [modelId, byBackend] of Object.entries(previewSupport.models)) {
      if (!("candle" in byBackend)) continue;
      const engineId = rows.find((row) => row.sceneworksId === modelId)?.engineId;
      expect(factsIds.has(engineId), `${modelId} → ${engineId}`).toBe(true);
    }
  });

  it("throws rather than silently emptying when MODEL_TABLE cannot be found", () => {
    expect(() => parseEngineModelTable("// no table here\n")).toThrow(/MODEL_TABLE/);
  });
});

// The current truth at inference pin d4802320: sc-16951 flipped the three Krea routes and sc-16952
// flipped Qwen-Image. The remaining family stories (sc-16953…sc-16960) each flip more; when one
// lands and the facts file is re-dumped, THIS list is what fails and tells the next author to
// regenerate — which is the whole point of landing sc-16965 before them rather than after.
describe("preview-support catalog: the shipped candle answers", () => {
  it("advertises live preview for exactly the wired candle routes", () => {
    const advertising = Object.entries(previewSupport.models)
      .filter(([, byBackend]) => byBackend.candle === true)
      .map(([id]) => id)
      .sort();
    expect(advertising).toEqual(["krea_2_raw", "krea_2_turbo", "qwen_image"]);
  });

  it("says false — not unknown — for a candle route that is wired but does not preview", () => {
    expect(previewSupport.models.sdxl).toEqual({ candle: false });
    expect(previewSupport.models.sensenova_u1_8b).toEqual({ candle: false });
  });

  it("is engine-KEYED: every entry is a per-backend map, never a bare boolean", () => {
    for (const [modelId, byBackend] of Object.entries(previewSupport.models)) {
      expect(typeof byBackend, modelId).toBe("object");
      for (const [backend, supported] of Object.entries(byBackend)) {
        expect(previewSupport.backends, `${modelId}.${backend}`).toContain(backend);
        expect(typeof supported, `${modelId}.${backend}`).toBe("boolean");
      }
    }
  });
});

// The offline fallback catalog (used when GET /api/v1/models is unreachable) must carry the same
// flag the served catalog does, or the card's three states collapse to two the moment the API
// blinks. constants.js applies the generated table programmatically rather than hand-listing ids —
// so this asserts the application, not a transcription.
describe("preview-support catalog: the offline fallback mirrors the served flag", () => {
  it("stamps preview.byBackend onto every fallback model the table knows", () => {
    for (const model of fallbackModels) {
      const expected = previewSupport.models[model.id];
      if (!expected) {
        expect(model.preview, `${model.id} is not in the generated table`).toBeUndefined();
        continue;
      }
      expect(model.preview?.byBackend, model.id).toEqual(expected);
    }
  });

  it("covers the candle routes that DO advertise preview", () => {
    // qwen_image (sc-16952) is the advertising route that is also in the offline seed — the Krea
    // ids are catalog-only, so this is the one the fallback path can actually exercise.
    const advertising = fallbackModels
      .filter((model) =>
        Object.values(model.preview?.byBackend ?? {}).some((supported) => supported === true),
      )
      .map((model) => model.id);
    expect(advertising).toContain("qwen_image");
    expect(fallbackModels.find((model) => model.id === "sdxl").preview.byBackend).toEqual({
      candle: false,
    });
  });
});
