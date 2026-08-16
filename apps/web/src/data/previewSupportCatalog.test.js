import { describe, expect, it } from "vitest";

import {
  assertBackendCoverage,
  catalogToWebPreviewSupport,
  derivePreviewSupport,
  parseBespokePreviewRoutes,
  parseEngineModelTable,
  parsePulidPreviewRoutes,
  parseSceneworksAudioBackends,
  parseSceneworksBackends,
  PREVIEW_SUPPORT_GENERATOR,
} from "./previewSupportDerivation.js";
import previewSupport from "./previewSupport.json";
// Raw source text (Vite `?raw`) so the guard derives from the same bytes the generator reads. Both
// live outside the web root — see the server.fs.allow entries in vite.config.js (mirrors the
// style.txt / builtin.styles.jsonc pair).
import enginesSource from "../../../../crates/sceneworks-worker/src/engines.rs?raw";
import qwenEditCandleSource from "../../../../crates/sceneworks-worker/src/image_jobs/qwen_edit_candle.rs?raw";
import pulidMlxSource from "../../../../crates/sceneworks-worker/src/image_jobs/pulid.rs?raw";
import pulidCandleSource from "../../../../crates/sceneworks-worker/src/image_jobs/pulid_candle.rs?raw";
import previewSupportManifestRaw from "../../../../config/manifests/builtin.preview-support.jsonc?raw";
// The inference pin itself. `verifyEngineCapabilityFacts` in scripts/bump-inference.mjs compares the
// facts files against it, but that script runs only when someone bumps THROUGH it — CI never invokes
// it. So a pin edited by hand would leave the whole catalog advertising the PREVIOUS revision's
// truth with nothing red. `check-license-coverage.mjs` had already learned this (it compares
// `audit.inferenceRevision` to the live pin set AND runs in the parity lane); this is the same
// assertion for the same class of artifact, on a guard that runs on every PR.
import workerCargoToml from "../../../../crates/sceneworks-worker/Cargo.toml?raw";
// The DECLARED backend set (sc-17119). Every other input below is discovered from the facts
// directory, and a directory listing cannot see a file that was never written — which is why
// `capabilities.mlx.json` was absent for four consecutive pins with the whole suite green.
import factsDeclarationSource from "../../../../crates/sceneworks-worker/src/engine_capability_facts.rs?raw";
import { fallbackModels } from "../constants.js";

// DISCOVERED, never listed. The generator globs this directory so a newly dumped backend is picked
// up with no edit; a guard that hardcodes `capabilities.candle.json` would then fail the moment the
// macOS lane lands `capabilities.mlx.json` — i.e. it would punish exactly the follow-up the design
// is waiting on. Everything below is derived per-backend from whatever is on disk.
const factsModules = import.meta.glob(
  "../../../../config/engine-capabilities/capabilities.*.json",
  { eager: true, query: "?raw", import: "default" },
);
const factsEntries = Object.entries(factsModules)
  .map(([path, raw]) => ({ name: path.slice(path.lastIndexOf("/") + 1), facts: JSON.parse(raw) }))
  .sort((left, right) => (left.name < right.name ? -1 : left.name > right.name ? 1 : 0));
const factsFiles = factsEntries.map((entry) => entry.facts);

// The AUDIO registry's dumps (sc-17593), one directory down. The glob above is single-level, so the
// two sets cannot bleed into each other — which matters because they share backend names ("candle"
// is the audio backend on every platform) while their engine-id namespaces are independent.
const audioFactsModules = import.meta.glob(
  "../../../../config/engine-capabilities/audio/capabilities.*.json",
  { eager: true, query: "?raw", import: "default" },
);
const audioFactsEntries = Object.entries(audioFactsModules)
  .map(([path, raw]) => ({ name: path.slice(path.lastIndexOf("/") + 1), facts: JSON.parse(raw) }))
  .sort((left, right) => (left.name < right.name ? -1 : left.name > right.name ? 1 : 0));
const audioFactsFiles = audioFactsEntries.map((entry) => entry.facts);

const rows = parseEngineModelTable(enginesSource);
const qwenPreviewRoutes = parseBespokePreviewRoutes(qwenEditCandleSource);
const pulidPreviewRoutes = parsePulidPreviewRoutes(pulidMlxSource, pulidCandleSource);
const bespokePreviewRoutes = [
  ...qwenPreviewRoutes,
  ...pulidPreviewRoutes,
];
const derived = derivePreviewSupport(rows, factsFiles, audioFactsFiles, bespokePreviewRoutes);
const bespokePreviewKeys = new Set(
  bespokePreviewRoutes.map((route) => `${route.modelId}\u0000${route.backend}`),
);

/** Every audio engine id, which for audio IS the SceneWorks model id it answers for. */
const audioEngineIds = new Set(
  audioFactsFiles.flatMap((facts) => facts.engines.map((engine) => engine.id)),
);

/** The single inference revision every Cargo manifest in the worker pins, read from the pin itself. */
const inferencePin = (() => {
  const revisions = new Set(
    [
      ...workerCargoToml.matchAll(
        /github\.com\/SceneWorks\/inference"[^}\n]*\brev\s*=\s*"([0-9a-f]{40})"/g,
      ),
    ].map((match) => match[1]),
  );
  if (revisions.size !== 1) {
    throw new Error(
      `expected exactly one inference revision in crates/sceneworks-worker/Cargo.toml, found: ${
        [...revisions].join(", ") || "none"
      }`,
    );
  }
  return [...revisions][0];
})();

// Guards sc-16965 (epic 16948). `config/manifests/builtin.preview-support.jsonc` — the block
// rust-api merges onto every /api/v1/models entry as `preview.byBackend` — must stay a mechanical
// derivation of (a) the stage-1 engine-capability dumps and (b) the MODEL_TABLE join in engines.rs.
// Never hand-edited. If either input moves — a family story flips a descriptor and the facts file is
// re-dumped, or a new MODEL_TABLE row lands — re-run `npm run gen:preview-support` (apps/web); this
// fails until both artifacts are regenerated.
//
// This is the half of the design that runs on EVERY PR *and on every platform*. Stage 1 needs a
// linked engine registry, so writing a facts file can only happen on macOS or a `backend-candle`
// lane — but it writes checked-in files, and everything below reads only those.
//
// Reading them is not taking them on trust. Both stage-1 lanes are themselves PR lanes that re-dump
// and diff their own file: `macos-mlx.yml` for `capabilities.mlx.json` (sc-17119) and the
// self-hosted CUDA `windows-candle.yml` for `capabilities.candle.json` (sc-17592). Without those,
// this suite would be satisfied by a restamp — it re-derives the catalog from the file's CONTENTS,
// so rewriting only the `inferenceRevision` line over a stale engine list keeps it green.
// Descriptor-level truth is guarded continuously upstream besides, by sc-16951's
// `candle-gen-catalog` bidirectional test in the inference repo's CI.
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

  it("covers every backend on disk — the artifacts list exactly the dumped facts files", () => {
    // The UNION over both registries (sc-17593). `backends` is what a consumer iterates to find a
    // key, so a backend that only the audio lane dumps must still appear — asserting the media set
    // alone would make that branch of the derivation permanently unreachable. The two sets coincide
    // today (audio is `candle`, which media also dumps); the union is what must hold when they stop
    // coinciding.
    const onDisk = [
      ...new Set(
        [...factsEntries, ...audioFactsEntries].map((entry) => entry.facts.backend),
      ),
    ].sort();
    expect(JSON.parse(previewSupportManifestRaw).backends).toEqual(onDisk);
    expect(previewSupport.backends).toEqual(onDisk);
  });
});

describe("preview-support catalog: bespoke Candle Qwen-Edit preview", () => {
  it("derives the two live catalog routes from the worker's exact sink-bearing declaration", () => {
    expect(qwenPreviewRoutes).toEqual([
      { modelId: "qwen_image_edit_2511", backend: "candle", supportsPreview: true },
      {
        modelId: "qwen_image_edit_2511_lightning",
        backend: "candle",
        supportsPreview: true,
      },
    ]);
    for (const route of qwenPreviewRoutes) {
      expect(previewSupport.models[route.modelId]?.[route.backend]).toBe(true);
    }
  });

  it("drops the claim loudly if the request stops carrying drive_gen_items' preview sink", () => {
    const withoutSink = qwenEditCandleSource.replace(
      /\n\s*preview,\n\s*\};/,
      "\n                        ..Default::default()\n                    };",
    );
    expect(withoutSink).not.toBe(qwenEditCandleSource);
    expect(() => parseBespokePreviewRoutes(withoutSink)).toThrow(/live preview sink/);
  });

  it("refuses a declared route that has no MODEL_TABLE authority", () => {
    expect(() =>
      derivePreviewSupport(rows, factsFiles, audioFactsFiles, [
        { modelId: "not_a_shipped_model", backend: "candle", supportsPreview: true },
      ]),
    ).toThrow(/unknown MODEL_TABLE id/);
  });
});

describe("preview-support catalog: direct-dispatch PuLID preview", () => {
  it("derives both native preview sinks from their exact worker requests", () => {
    expect(pulidPreviewRoutes).toEqual([
      {
        modelId: "pulid_flux_dev",
        backend: "mlx",
        supportsPreview: true,
        directDispatch: true,
      },
      {
        modelId: "pulid_flux_dev",
        backend: "candle",
        supportsPreview: true,
        directDispatch: true,
      },
    ]);
    expect(previewSupport.models.pulid_flux_dev).toEqual({ candle: true, mlx: true });
  });

  it("fails closed if either direct request drops its live sink", () => {
    const withoutMlxSink = pulidMlxSource.replace(/\n\s*preview,\n/, "\n");
    expect(() => parsePulidPreviewRoutes(withoutMlxSink, pulidCandleSource)).toThrow(
      /MLX GenerationRequest has no live preview sink/,
    );
    const withoutCandleSink = pulidCandleSource.replace(
      /\n\s*preview:\s*preview\.clone\(\),\n/,
      "\n",
    );
    expect(() => parsePulidPreviewRoutes(pulidMlxSource, withoutCandleSink)).toThrow(
      /Candle PulidFluxRequest has no live preview sink/,
    );
  });
});

// The stage-1 facts file is a checked-in SOURCE, so it gets the same scrutiny style.txt gets: an
// empty or truncated one derives as "no route supports live preview", which is a confident wrong
// answer rather than a missing one. The Rust dumper refuses to write one; this refuses to read one.
describe("preview-support catalog: the stage-1 facts files are non-vacuous", () => {
  it("finds at least one checked-in dump", () => {
    expect(factsEntries.length).toBeGreaterThan(0);
  });

  it.each(factsEntries.map((entry) => [entry.name, entry.facts]))(
    "%s carries a full registry, not an empty one",
    (name, facts) => {
      // The filename is the routing key — one file per backend so two lanes never rewrite the same
      // file — so a dump whose `backend` disagrees with its own name would silently take another
      // lane's slot.
      expect(`capabilities.${facts.backend}.json`).toBe(name);
      // Both shipped registries are large (candle 51; the MLX bundle registers 27 provider crates
      // and 38 preview-capable routes alone), so a dump under this floor is truncated, not small.
      expect(facts.engines.length).toBeGreaterThan(40);
      expect(facts.generatedFrom.inferenceRevision).toMatch(/^[0-9a-f]{40}$/);
    },
  );

  // The assertion `bump-inference.mjs` makes at bump time, made again where CI can actually see it.
  // Without this, a pin edited outside the sanctioned script leaves every facts file — and therefore
  // the entire served catalog — describing the PREVIOUS revision's descriptors, silently.
  it.each(factsEntries.map((entry) => [entry.name, entry.facts]))(
    "%s was dumped at the revision the worker currently pins",
    (name, facts) => {
      expect(facts.generatedFrom.inferenceRevision, `${name} vs crates/sceneworks-worker/Cargo.toml`)
        .toBe(inferencePin);
    },
  );

  it("the manifest stamps each backend's dump revision, and it is the current pin", () => {
    const manifest = JSON.parse(previewSupportManifestRaw);
    for (const backend of manifest.backends) {
      expect(manifest.generatedFrom[backend].inferenceRevision, backend).toBe(inferencePin);
    }
  });

  it("an empty facts file is refused rather than derived from", () => {
    expect(() => derivePreviewSupport(rows, [{ backend: "candle", engines: [] }])).toThrow(
      /vacuous-green/,
    );
  });

  it("deriving with no facts files at all is refused", () => {
    expect(() => derivePreviewSupport(rows, [])).toThrow(/no stage-1 facts files/);
  });
});

// sc-17119. The one assertion in this pipeline that can see a file which is NOT there.
//
// Everything above is scoped to what the facts directory contains — the generator globs it, this
// guard globs it, and `verifyEngineCapabilityFacts` validates only the files that exist. So a
// backend that was never dumped is not a failure anywhere: it is simply absent, and absence derives
// as "unknown", which renders exactly as it did before the feature shipped. `capabilities.mlx.json`
// was absent for four consecutive pins beginning with the one sc-16965 itself shipped on, with
// every one of those guards green.
//
// The required set comes from the Rust const, parsed — not from a JS copy, whose stale-failure mode
// is to stop requiring a backend, i.e. to silently re-open this hole.
describe("preview-support catalog: every backend SceneWorks can run is dumped", () => {
  const declared = parseSceneworksBackends(factsDeclarationSource);
  const onDisk = factsEntries.map((entry) => entry.facts.backend).sort();

  it("parses SCENEWORKS_BACKENDS out of engine_capability_facts.rs", () => {
    expect(declared.length).toBeGreaterThan(0);
    // Both shipped lanes, named: macOS links mlx, `--features backend-candle` links candle.
    expect(declared).toEqual(expect.arrayContaining(["candle", "mlx"]));
  });

  it("every declared backend has a checked-in facts file", () => {
    expect(() => assertBackendCoverage(declared, onDisk)).not.toThrow();
    for (const backend of declared) {
      expect(onDisk, `capabilities.${backend}.json was never dumped`).toContain(backend);
    }
  });

  it("the served catalog advertises every declared backend, not just the dumped ones", () => {
    // The `backends` array is what a consumer iterates. If it were ever a strict subset of the
    // declared set, the UI would have no key at all for the missing engine's routes. Declared over
    // BOTH registries, for the same reason the on-disk assertion above takes the union: an
    // audio-only backend is still a backend a consumer must find a key for.
    const declaredEverywhere = [
      ...new Set([...declared, ...parseSceneworksAudioBackends(factsDeclarationSource)]),
    ].sort();
    expect(JSON.parse(previewSupportManifestRaw).backends).toEqual(declaredEverywhere);
    expect(previewSupport.backends).toEqual(declaredEverywhere);
  });

  // Mutation-shaped: this is what the guard is FOR, so assert it actually fires. Deleting
  // capabilities.mlx.json reduces the on-disk set to ["candle"] and nothing else in this file
  // notices — every other assertion iterates `factsEntries` and would simply do one less round.
  it("a declared backend with no dump is a loud failure, naming it and how to dump it", () => {
    expect(() => assertBackendCoverage(["candle", "mlx"], ["candle"])).toThrow(/mlx/);
    expect(() => assertBackendCoverage(["candle", "mlx"], ["candle"])).toThrow(
      /dump-engine-capabilities/,
    );
    // ...and the reverse, so the failure cannot be "fixed" by dumping the wrong lane twice.
    expect(() => assertBackendCoverage(["candle", "mlx"], ["mlx"])).toThrow(/candle/);
  });

  it("a dump for a backend the Rust const does not declare is also a failure", () => {
    // A stale const requires nothing for the new backend, so deleting its file again would pass.
    expect(() => assertBackendCoverage(["candle"], ["candle", "mlx"])).toThrow(/mlx/);
  });

  it("refuses a renamed or restructured const rather than requiring nothing", () => {
    const renamed = factsDeclarationSource.replace(
      "const SCENEWORKS_BACKENDS: &[&str] = &[",
      "const SCENEWORKS_ENGINE_BACKENDS: &[&str] = &[",
    );
    expect(renamed).not.toBe(factsDeclarationSource);
    expect(() => parseSceneworksBackends(renamed)).toThrow(/SCENEWORKS_BACKENDS/);
  });

  it("refuses an entry it cannot read rather than dropping it", () => {
    // Dropping an entry SHRINKS the required set — the one failure direction that is invisible,
    // because a guard that asks for less always passes.
    //
    // Built synthetically rather than by `.replace()`-ing today's exact two-element literal: that
    // fixture would silently become a no-op the moment a third backend is declared, which is the
    // very change this list exists to accommodate.
    expect(() =>
      parseSceneworksBackends('pub const SCENEWORKS_BACKENDS: &[&str] = &["candle", "MLX2"];'),
    ).toThrow(/only 1 parsed/);
  });

  it("does not anchor on the const's own doc comment", () => {
    // The doc comment names both backends and `capabilities.mlx.json` in prose; a comment-blind
    // parse could match there and read an empty or wrong list.
    expect(parseSceneworksBackends(factsDeclarationSource)).toEqual(declared);
    expect(() =>
      parseSceneworksBackends("// const SCENEWORKS_BACKENDS: &[&str] = &[\"ghost\"];\n"),
    ).toThrow(/SCENEWORKS_BACKENDS/);
  });

  it("refuses a duplicated entry", () => {
    expect(() =>
      parseSceneworksBackends('pub const SCENEWORKS_BACKENDS: &[&str] = &["mlx", "mlx"];'),
    ).toThrow(/twice/);
  });

  it("refuses an empty list rather than requiring nothing", () => {
    expect(() => parseSceneworksBackends("pub const SCENEWORKS_BACKENDS: &[&str] = &[];")).toThrow(
      /empty/,
    );
  });

  it("refuses an empty DECLARED set, not just an empty parse", () => {
    // assertBackendCoverage is exported and called from three places, and not all of them get
    // their list from parseSceneworksBackends. With an empty declared set every check below it is
    // vacuously true — it would pass for any set of dumped files, including none at all.
    expect(() => assertBackendCoverage([], ["candle", "mlx"])).toThrow(/empty/);
    expect(() => assertBackendCoverage([], [])).toThrow(/empty/);
  });

  it("refuses a facts file with no readable backend rather than reporting a nameless one", () => {
    // Without this, `undefined` flows through and the operator is told that backend(s) `[]` are
    // undeclared — an empty bracket list that names neither the backend nor the file.
    expect(() => assertBackendCoverage(["candle", "mlx"], ["candle", undefined])).toThrow(
      /no readable `backend`/,
    );
    expect(() => assertBackendCoverage(["candle", "mlx"], ["candle", ""])).toThrow(
      /no readable `backend`/,
    );
  });
});

// sc-17593. The audio registry (`crate::inference_runtime::audio`) is a SEPARATE ProviderRegistry
// that nothing dumped until this story, so `supports_preview` was unknown for every audio route on
// every platform — reported exactly like "not wired".
//
// sc-17119's `SCENEWORKS_BACKENDS` guard could not see it, and that is the point of this block:
// `AUDIO_BACKEND` is `"candle"` on all three runtime bundles, and `candle` is already dumped from the
// MEDIA registry, so backend-level coverage was satisfied coincidentally while the audio engines were
// missing entirely. Coverage has to be asserted per registry, against its own declared const.
describe("preview-support catalog: the audio registry is dumped too", () => {
  const declaredAudio = parseSceneworksAudioBackends(factsDeclarationSource);
  const audioOnDisk = audioFactsEntries.map((entry) => entry.facts.backend).sort();

  /** A minimal well-formed dump of each kind, for assertions about the DERIVATION rather than about
   *  what happens to be checked in — a fixture built from the on-disk files goes empty, and takes
   *  its assertion with it, in precisely the never-dumped state this block guards against. */
  const syntheticAudioFacts = ({ id = "kokoro_82m", inferenceRevision = inferencePin } = {}) => ({
    registry: "audio",
    backend: "candle",
    generatedFrom: { inferenceRevision, dumper: "test" },
    engines: [{ id, modality: "audio", supportsPreview: false }],
  });
  const syntheticMediaFacts = () => ({
    backend: "candle",
    generatedFrom: { inferenceRevision: inferencePin, dumper: "test" },
    engines: [{ id: "krea_2_turbo", modality: "image", supportsPreview: true }],
  });

  it("parses SCENEWORKS_AUDIO_BACKENDS as its own const, not as SCENEWORKS_BACKENDS", () => {
    expect(declaredAudio).toEqual(["candle"]);
    // The two anchors must not match each other's declaration — `SCENEWORKS_AUDIO_BACKENDS` contains
    // the string `BACKENDS`, and a loose anchor would read one const as the other and quietly
    // require the wrong set.
    expect(parseSceneworksBackends(factsDeclarationSource)).not.toEqual(declaredAudio);
  });

  it("every declared audio backend has a checked-in dump", () => {
    expect(audioFactsEntries.length).toBeGreaterThan(0);
    expect(() => assertBackendCoverage(declaredAudio, audioOnDisk, "audio")).not.toThrow();
    for (const backend of declaredAudio) {
      expect(audioOnDisk, `audio/capabilities.${backend}.json was never dumped`).toContain(backend);
    }
  });

  // Mutation-shaped, and the one that matters most: this is the exact state the repo was in before
  // this story, and every other assertion in this file was green in it.
  it("a declared audio backend with no dump is a loud failure naming the audio lane", () => {
    expect(() => assertBackendCoverage(["candle"], [], "audio")).toThrow(/candle/);
    expect(() => assertBackendCoverage(["candle"], [], "audio")).toThrow(/audio/);
    expect(() => assertBackendCoverage(["candle"], [], "audio")).toThrow(
      /dump-engine-capabilities/,
    );
  });

  it("the media coverage check cannot stand in for it", () => {
    // Precisely the coincidence that hid this: the media dumps alone satisfy SCENEWORKS_BACKENDS,
    // audio dumped or not. If this ever starts throwing, the two checks have been conflated.
    expect(() =>
      assertBackendCoverage(parseSceneworksBackends(factsDeclarationSource), ["candle", "mlx"]),
    ).not.toThrow();
  });

  it.each(audioFactsEntries.map((entry) => [entry.name, entry.facts]))(
    "audio/%s is a real dump: discriminated, non-vacuous, at the current pin",
    (name, facts) => {
      expect(`capabilities.${facts.backend}.json`).toBe(name);
      // The discriminator is what stops a file in the wrong directory being read as the other kind:
      // a media and an audio dump can carry the SAME `backend`, so the name cannot separate them.
      expect(facts.registry).toBe("audio");
      // 14 generators at the current pin — 12 provider crates ride the audio lane, but four of them
      // register no generator at all (Chatterbox VE, OpenVoice, Whisper, CLAP) while Stable Audio 3
      // registers six and MMAudio two. The floor is set off the GENERATOR count, not the crate
      // count, so a dump in single digits is truncated rather than merely small.
      expect(facts.engines.length).toBeGreaterThan(10);
      expect(facts.engines.every((engine) => engine.modality === "audio")).toBe(true);
      expect(facts.generatedFrom.inferenceRevision).toBe(inferencePin);
    },
  );

  it("reads an audio dump as media (or vice versa) only over a loud failure", () => {
    // Both directions, because a copy into the wrong directory is the realistic mistake and the
    // silent outcome would be a whole registry answering for the other's routes.
    //
    // Synthetic fixtures, not slices of what is on disk: a fixture derived from `audioFactsFiles`
    // becomes an empty array — and this assertion a no-op — in exactly the state this whole block
    // exists to catch, the one where nothing dumped the audio registry.
    expect(() => derivePreviewSupport(rows, [syntheticAudioFacts()], [syntheticAudioFacts()])).toThrow(
      /"audio" dump but it was read as "media"/,
    );
    expect(() => derivePreviewSupport(rows, [syntheticMediaFacts()], [syntheticMediaFacts()])).toThrow(
      /"media" dump but it was read as "audio"/,
    );
  });

  it("answers false — not unknown — for every audio route the registry declares", () => {
    // The whole user-visible content of this story. Before it, every one of these was absent from
    // the table, which the catalog reports as `unknown` and the UI renders identically to "wired and
    // does not preview". These are the SceneWorks ids the audio jobs actually route on.
    for (const shipped of [
      "kokoro_82m",
      "chatterbox_tts",
      "acestep_v15_turbo",
      "moss_sfx_v2",
      "moss_tts_realtime",
      "moss_ttsd_v05",
    ]) {
      expect(audioEngineIds, `${shipped} is not in the audio dump`).toContain(shipped);
      expect(previewSupport.models[shipped], shipped).toEqual({ candle: false });
    }
  });

  it("no audio route advertises live preview at this pin", () => {
    // The measurement sc-17593 required before choosing a shape, asserted where CI can see it. The
    // Rust twin (`no_audio_generator_advertises_live_preview_at_this_pin`) checks the registry; this
    // checks what actually shipped in the artifact. If a pin lights one up, both go red — and
    // `previewSupport.js`'s PREVIEW_JOB_TYPES still gates the placeholder to image jobs, so the UI
    // question has to be answered before the flag means anything to a user.
    const advertising = [...audioEngineIds].filter((id) =>
      Object.values(previewSupport.models[id] ?? {}).some((supported) => supported === true),
    );
    expect(advertising).toEqual([]);
  });

  it("leaves the lane's non-generator providers unknown rather than inventing false", () => {
    // `supports_preview` lives on `Capabilities`, which only GENERATORS carry. openvoice_v2 is an
    // AudioTransform and chatterbox_ve a VoiceEmbedder — different descriptor types with no such
    // field — so the registry genuinely has no opinion and the catalog must not manufacture one.
    for (const nonGenerator of ["openvoice_v2", "chatterbox_ve"]) {
      expect(audioEngineIds, nonGenerator).not.toContain(nonGenerator);
      expect(previewSupport.models[nonGenerator], nonGenerator).toBeUndefined();
    }
  });

  it("refuses an id claimed by both registries instead of letting one win", () => {
    // Any MODEL_TABLE id collides, including one whose engine no facts file answers for: the clash
    // is between the two NAMESPACES, not between two answers. Assert both, so narrowing the check
    // to "ids that happen to be in the artifact" cannot pass.
    const answered = Object.keys(derived.models).find((id) =>
      rows.some((row) => row.sceneworksId === id),
    );
    const unanswered = rows.find((row) => !(row.sceneworksId in derived.models))?.sceneworksId;
    expect(answered).toBeTruthy();
    for (const collidingId of [answered, unanswered].filter(Boolean)) {
      expect(() =>
        derivePreviewSupport(rows, factsFiles, [syntheticAudioFacts({ id: collidingId })]),
      ).toThrow(/both a MODEL_TABLE model id and an audio engine id/);
    }
  });

  it("refuses a media and audio dump of one backend at different revisions", () => {
    // `candle` is dumped twice — once per registry — and the two must describe one checkout, or
    // `generatedFrom` claims a state that never existed.
    expect(() =>
      derivePreviewSupport(rows, factsFiles, [
        syntheticAudioFacts({ inferenceRevision: "0".repeat(40) }),
      ]),
    ).toThrow(/two different inference revisions/);
  });

  it("still derives when no audio dump is supplied, so the argument stays optional", () => {
    // The generator, the bump script and this guard all pass audio facts; the coverage assertion is
    // what makes their absence fail, NOT a crash inside the derivation. Keeping this total means the
    // failure the operator sees is the one with the remediation text.
    const withoutAudio = derivePreviewSupport(rows, factsFiles);
    expect(Object.keys(withoutAudio.models).length).toBeLessThan(
      Object.keys(derived.models).length,
    );
    for (const id of audioEngineIds) expect(withoutAudio.models[id], id).toBeUndefined();
  });
});

describe("preview-support catalog: the MODEL_TABLE join", () => {
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
    // see. Every emitted entry is backed by a real facts row, on every backend on disk.
    for (const { facts } of factsEntries) {
      const factsIds = new Set(facts.engines.map((engine) => engine.id));
      const audioIds = new Set(
        audioFactsFiles
          .filter((audio) => audio.backend === facts.backend)
          .flatMap((audio) => audio.engines.map((engine) => engine.id)),
      );
      for (const [modelId, byBackend] of Object.entries(previewSupport.models)) {
        if (!(facts.backend in byBackend)) continue;
        // Two namespaces answer under one backend key: MODEL_TABLE ids through the media facts, and
        // audio engine ids through the identity join. Bespoke direct-dispatch routes are backed by
        // the parsed worker sink above. An entry backed by NONE of those would be invented.
        if (audioIds.has(modelId)) continue;
        if (bespokePreviewKeys.has(`${modelId}\u0000${facts.backend}`)) continue;
        const engineId = rows.find((row) => row.sceneworksId === modelId)?.engineId;
        expect(factsIds.has(engineId), `${modelId} → ${engineId} @ ${facts.backend}`).toBe(true);
      }
    }
  });

  it("throws rather than silently emptying when MODEL_TABLE cannot be found", () => {
    expect(() => parseEngineModelTable("// no table here\n")).toThrow(/MODEL_TABLE/);
  });

  it("refuses a row it cannot read rather than dropping it", () => {
    // A row that grows a nested-brace field is invisible to the brace-free matcher. Dropping it
    // reads downstream as "unknown", which renders exactly as before — i.e. it looks fine. The
    // count assertion is what turns that into a loud failure.
    const withNestedField = enginesSource.replace(
      'sceneworks_id: "sdxl",',
      'limits: Limits { max: 4 },\n        sceneworks_id: "sdxl",',
    );
    expect(withNestedField).not.toBe(enginesSource);
    expect(() => parseEngineModelTable(withNestedField)).toThrow(
      /declares \d+ ModelRow entries but only \d+ parsed/,
    );
  });

  it("does not parse a commented-out row as a live one", () => {
    // Over-claiming a model that no longer exists is the dangerous direction here: the catalog would
    // advertise preview support for a route that is not wired at all.
    const withGhost = enginesSource.replace(
      "pub(crate) const MODEL_TABLE: &[ModelRow] = &[\n",
      "pub(crate) const MODEL_TABLE: &[ModelRow] = &[\n" +
        "    /*\n" +
        "    ModelRow {\n" +
        '        sceneworks_id: "ghost_model",\n' +
        '        engine_id: "ghost_engine",\n' +
        '        default_repo: "x/y",\n' +
        "        default_steps: 1,\n" +
        "        default_guidance: 1.0,\n" +
        '        adapter_label: "ghost",\n' +
        "    },\n" +
        "    */\n",
    );
    expect(withGhost).not.toBe(enginesSource);
    const ghosted = parseEngineModelTable(withGhost);
    expect(ghosted.map((row) => row.sceneworksId)).not.toContain("ghost_model");
    expect(ghosted.length).toBe(rows.length);
  });

  it("is not truncated by a `];` that only appears inside a comment", () => {
    // The old first-`\n];` terminator stopped there and dropped every row after it — silently,
    // because the survivors still cleared the row-count floor.
    for (const injected of ["    // the table ends with\n    // ];\n", "    /* like this:\n];\n    */\n"]) {
      const withStray = enginesSource.replace(
        "pub(crate) const MODEL_TABLE: &[ModelRow] = &[\n",
        `pub(crate) const MODEL_TABLE: &[ModelRow] = &[\n${injected}`,
      );
      expect(withStray).not.toBe(enginesSource);
      expect(parseEngineModelTable(withStray).length, injected).toBe(rows.length);
    }
  });
});

// What the shipped artifacts claim, cross-checked against the RAW facts rather than against a
// hand-written id list. A literal list would have to be re-typed by each remaining family story
// (sc-16953…sc-16960) and would fail outright the first time a second backend is dumped; this
// re-joins the facts independently of `derivePreviewSupport`, so it still catches a re-dump that
// nobody regenerated — which is the whole point of landing sc-16965 before those stories.
describe("preview-support catalog: the shipped answers match the dumped facts", () => {
  it.each(factsEntries.map((entry) => [entry.facts.backend, entry.facts]))(
    "advertises live preview on %s for exactly the descriptor or bespoke-worker routes that say so",
    (backend, facts) => {
      const advertisingEngines = new Set(
        facts.engines.filter((engine) => engine.supportsPreview).map((engine) => engine.id),
      );
      const expected = rows
        .filter((row) => advertisingEngines.has(row.engineId))
        .map((row) => row.sceneworksId)
        // The audio registry answers under the same backend key by the identity join, so its
        // advertising engines belong in the expected set too — otherwise the day an audio provider
        // gains preview this test fails as a phantom over-claim rather than reporting the change.
        .concat(
          audioFactsFiles
            .filter((audio) => audio.backend === backend)
            .flatMap((audio) =>
              audio.engines.filter((engine) => engine.supportsPreview).map((engine) => engine.id),
            ),
        )
        .concat(
          bespokePreviewRoutes
            .filter((route) => route.backend === backend && route.supportsPreview)
            .map((route) => route.modelId),
        )
        .sort();
      const actual = Object.entries(previewSupport.models)
        .filter(([, byBackend]) => byBackend[backend] === true)
        .map(([id]) => id)
        .sort();
      expect(actual).toEqual(expected);
      expect(actual.length, `${backend} advertises nothing — epic 16948 has wired routes`)
        .toBeGreaterThan(0);
    },
  );

  it.each(factsEntries.map((entry) => [entry.facts.backend, entry.facts]))(
    "says false — not unknown — for every %s route that is wired and does not preview",
    (backend, facts) => {
      const nonPreviewing = new Map(
        facts.engines.map((engine) => [engine.id, engine.supportsPreview]),
      );
      const wiredFalse = rows.filter((row) => nonPreviewing.get(row.engineId) === false);
      expect(wiredFalse.length).toBeGreaterThan(0);
      for (const row of wiredFalse) {
        expect(
          previewSupport.models[row.sceneworksId]?.[backend],
          `${row.sceneworksId} → ${row.engineId} @ ${backend}`,
        ).toBe(false);
      }
    },
  );

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

  it("seeds BOTH answers, so the offline path can exercise every state", () => {
    // Derived, not named: which ids are in the offline seed is a product decision that moves, and a
    // hardcoded pair breaks on a re-dump or a new backend. What must hold is that the seed carries
    // at least one supporting and one non-supporting route — otherwise a whole UI state is
    // unreachable when the API is down.
    const seeded = fallbackModels.filter((model) => previewSupport.models[model.id]);
    expect(seeded.length).toBeGreaterThan(0);
    const answers = seeded.flatMap((model) => Object.values(model.preview?.byBackend ?? {}));
    expect(answers, "no fallback model advertises live preview").toContain(true);
    expect(answers, "no fallback model reports a wired non-previewing route").toContain(false);
  });
});
