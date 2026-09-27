import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  YUE2_FIELD_KINDS,
  buildEditOperation,
  buildYue2JobRequest,
  defaultEditDraft,
  defaultYue2Settings,
  installJobStatusLabel,
  isCoverComponentDownload,
  midiNoteLabel,
  seedValueProblem,
  stripYue2ExportHeader,
  transcriptionSettingsProblems,
  yue2AbcExport,
  yue2CoverSetup,
  yue2FieldDisabledReason,
  restoreYue2Settings,
  usagePolicyChips,
  yue2ModelIdentity,
  yue2RequestProblems,
  yue2RunExport,
  yue2RunView,
  yue2TierRows,
  yue2TranscriptionView,
} from "./yue2Lab.js";

// sc-23000: the YuE2 Song Lab's pure half. The component tests assert the bodies the lab actually
// POSTs; these pin the per-kind field contract (a mirror of core `FIELD_KINDS`) and the run view.

const EVERY_CONTROL = {
  ...defaultYue2Settings(),
  style: "dream pop, airy female vocal",
  lyrics: "[verse]\nhello",
  planning: "melody",
  planSource: "sample",
  count: "3",
  tier: "q8",
  decoder: "legacy",
  seed: "42",
  cfgScale: "1.5",
  steps: "32",
  precision: "fp32",
  offloadPolicy: "sequential",
  stageResidency: "on",
  chunkAttention: "on",
  attentionChunkSize: "393216",
  tileVaeDecode: "on",
  decodeTileEdge: "256",
  scoreSampling: {
    temperature: "0.9",
    topP: "0.95",
    topK: "40",
    repetitionPenalty: "1.1",
    penaltyWindow: "32",
    minTokens: "64",
    maxTokens: "2048",
  },
  semanticSampling: {
    temperature: "1",
    topP: "0.9",
    topK: "50",
    repetitionPenalty: "1.2",
    penaltyWindow: "16",
    minTokens: "300",
    maxTokens: "8000",
  },
  restorePlanJobId: "job_plan",
  coverVersionId: "ver_1",
  coverMode: "melody",
  coverKeep: "vocal",
  coverTranslatedFrom: "[verse]\nhola",
  transcribeAssetId: "asset_take",
  transcribeMaxSeconds: "90",
  transcribeOverlapSeconds: "150",
  transcribeLookaheadSeconds: "50",
};

// The request contract shared with the server (sc-22988 review item 7): a fixture of the field
// table and of the bodies this builder produces, which the rust-api suite
// (`apps/rust-api/src/tests/yue2_jobs.rs`, `the_web_lab_request_bodies_…`) pins to core
// `FIELD_KINDS` and replays through `Yue2JobSpec` (deny_unknown_fields) + `validate_request`. A
// drift on either side reds one of the two suites; regenerate with UPDATE_YUE2_WEB_FIXTURE=1.
const FIXTURES = resolve(dirname(fileURLToPath(import.meta.url)), "../../../crates/sceneworks-core/src/yue2_score/fixtures");
const WEB_REQUESTS_PATH = resolve(FIXTURES, "web-job-requests.json");
const SCORE_ABC = readFileSync(resolve(FIXTURES, "score.abc"), "utf8");
const WEB_REQUESTS = JSON.parse(readFileSync(WEB_REQUESTS_PATH, "utf8"));
const CORE_FIELD_KINDS = WEB_REQUESTS.fieldKinds;

// Every body shape the lab submits, from its own settings.
function webJobRequests() {
  const scenarios = [
    ["create", EVERY_CONTROL, {}],
    ["create", { ...EVERY_CONTROL, planSource: "supplied", planning: "full", suppliedScore: SCORE_ABC }, {}],
    ["plan", EVERY_CONTROL, {}],
    ["fromPlan", { ...EVERY_CONTROL, planSource: "restore" }, { plan: { style: "folk", lyrics: "[verse]\nplanned" } }],
    ["cover", EVERY_CONTROL, {}],
    ["cover", { ...EVERY_CONTROL, coverSource: "inline", coverScore: SCORE_ABC, coverMode: "full" }, {}],
    ["renderVersion", EVERY_CONTROL, { versionId: "ver_1" }],
    ["decode", EVERY_CONTROL, { sourceJobId: "job_src", decoder: "standard" }],
    ["create", { ...defaultYue2Settings(), lyrics: "la la" }, {}],
    // sc-23002: a transcription with and without its window settings.
    ["transcribe", EVERY_CONTROL, { sourceAudioAssetId: "asset_take" }],
    ["transcribe", defaultYue2Settings(), { sourceAudioAssetId: "asset_take" }],
  ];
  return scenarios.map(([kind, settings, target]) => buildYue2JobRequest(kind, settings, target, "auto"));
}

describe("YuE2 request builder (sc-23000)", () => {
  it("mirrors the core FIELD_KINDS table exactly", () => {
    expect(YUE2_FIELD_KINDS).toEqual(CORE_FIELD_KINDS);
  });

  it("builds exactly the request bodies the server's contract test replays", () => {
    const current = { fieldKinds: YUE2_FIELD_KINDS, bodies: webJobRequests() };
    if (process.env.UPDATE_YUE2_WEB_FIXTURE === "1") {
      writeFileSync(WEB_REQUESTS_PATH, `${JSON.stringify(current, null, 2)}\n`);
    }
    // Mutation that reds this: any change to what `buildYue2JobRequest` sends for these settings.
    expect(JSON.parse(readFileSync(WEB_REQUESTS_PATH, "utf8"))).toEqual(current);
    expect(new Set(current.bodies.map((body) => body.kind))).toEqual(
      new Set(["create", "plan", "fromPlan", "cover", "renderVersion", "decode", "transcribe"]),
    );
  });

  it("never sends a field to a kind that does not read it", () => {
    for (const kind of ["create", "plan", "fromPlan", "cover", "renderVersion", "decode", "transcribe"]) {
      const body = buildYue2JobRequest(kind, EVERY_CONTROL, {
        versionId: "ver_1",
        sourceJobId: "job_src",
        sourceAudioAssetId: "asset_take",
      });
      for (const field of Object.keys(body)) {
        if (field === "kind" || field === "requestedGpu") continue;
        if (field === "memory") {
          const acoustic = ["stageResidency", "chunkAttention", "attentionChunkSize"].some((key) => key in body.memory);
          const decode = ["tileVaeDecode", "decodeTileEdge"].some((key) => key in body.memory);
          if (acoustic) expect(CORE_FIELD_KINDS["memory.acoustic"]).toContain(kind);
          if (decode) expect(CORE_FIELD_KINDS["memory.decode"]).toContain(kind);
          continue;
        }
        expect(CORE_FIELD_KINDS[field], `${field} on ${kind}`).toContain(kind);
      }
    }
  });

  it("maps a decode job to exactly its source, decoder, tier, precision and decode memory", () => {
    expect(buildYue2JobRequest("decode", EVERY_CONTROL, { sourceJobId: "job_src" }, "auto")).toEqual({
      kind: "decode",
      sourceJobId: "job_src",
      decoder: "legacy",
      tier: "q8",
      precision: "fp32",
      memory: { tileVaeDecode: true, decodeTileEdge: 256 },
      requestedGpu: "auto",
    });
  });

  it("omits every unset control so the model default applies", () => {
    const settings = { ...defaultYue2Settings(), lyrics: "la la" };
    expect(buildYue2JobRequest("create", settings)).toEqual({ kind: "create", lyrics: "la la", planning: "full" });
  });

  // Core `validate_common`: a chunk size is read only with chunked attention ON, a tile edge only
  // with tiled decode ON, and only a melody cover chooses what it keeps.
  it("sends a memory size and a cover keep only with the switch that makes them meaningful", () => {
    const off = { ...EVERY_CONTROL, chunkAttention: "off", tileVaeDecode: "off", coverMode: "full" };
    const body = buildYue2JobRequest("cover", off);
    expect(body.memory).toEqual({ stageResidency: true, chunkAttention: false, tileVaeDecode: false });
    expect(body.cover).toEqual({ versionId: "ver_1", mode: "full", translatedFrom: "[verse]\nhola" });
    expect(buildYue2JobRequest("cover", defaultYue2Settings()).cover).toEqual({ mode: "melody" });
  });

  it("sends a supplied score instead of the planning sampler", () => {
    const body = buildYue2JobRequest("create", { ...EVERY_CONTROL, planSource: "supplied", suppliedScore: "X:1\n" });
    expect(body.score).toBe("X:1");
    expect(body.scoreSampling).toBeUndefined();
  });

  it("names the problems of an incomplete request in user terms", () => {
    const settings = defaultYue2Settings();
    expect(yue2RequestProblems("create", settings)).toEqual(["Write the lyrics the song sings."]);
    expect(yue2RequestProblems("cover", { ...settings, lyrics: "x" })).toEqual([
      "Choose the reviewed score version the cover follows.",
    ]);
    expect(
      yue2RequestProblems("create", { ...settings, lyrics: "x", planSource: "supplied", planning: "off" }),
    ).toEqual([
      "Paste the ABC score to plan from, or choose to sample a new plan.",
      "A supplied score needs full or melody planning.",
    ]);
    expect(yue2RequestProblems("fromPlan", settings)).toEqual(["Choose the saved plan to restore."]);
    expect(yue2RequestProblems("plan", { ...settings, lyrics: "x", planning: "off" })).toEqual([
      "Plan only needs full or melody planning — planning off makes no score.",
    ]);
    expect(yue2RequestProblems("cover", { ...settings, lyrics: "x", coverVersionId: "v", coverMode: "" })).toEqual([
      "Choose the cover mode (melody or full).",
    ]);
    expect(
      yue2RequestProblems("create", { ...settings, lyrics: "x", chunkAttention: "on", attentionChunkSize: "512" }),
    ).toEqual(["The attention chunk size must be at least 393216 score elements."]);
    expect(
      yue2RequestProblems("decode", { ...settings, tileVaeDecode: "on", decodeTileEdge: "2048" }, { sourceJobId: "j" }),
    ).toEqual(["The decode tile must be between 1 and 1024 latent frames."]);
  });
});

describe("YuE2 settings restore (sc-23000)", () => {
  it("keeps known values and drops malformed ones", () => {
    const restored = restoreYue2Settings({
      lyrics: "kept",
      optIn: true,
      seed: 5, // wrong type: seed is a control string
      bogus: "dropped",
      scoreSampling: { topK: "40", junk: "x" },
      presets: [{ id: "p", name: "Warm", settings: {} }, { id: 1 }],
    });
    expect(restored.lyrics).toBe("kept");
    expect(restored.optIn).toBe(true);
    expect(restored.seed).toBe("");
    expect(restored).not.toHaveProperty("bogus");
    expect(restored.scoreSampling.topK).toBe("40");
    expect(restored.scoreSampling).not.toHaveProperty("junk");
    expect(restored.presets.map((preset) => preset.id)).toEqual(["p"]);
  });
});

const YUE2_ENTRY = {
  id: "yue2",
  name: "YuE2 Song Generation (Experimental, Noncommercial)",
  experimental: true,
  nonCommercial: true,
  licenseNotice: "…licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0): …",
  licenseUrl: "https://example.test/LICENSE",
  ui: { label: "YuE2 (Experimental)" },
  conditionalComponents: [
    {
      componentId: "yue2_sheetsage2",
      requiredFor: ["cover"],
      repo: "m-a-p/SheetSage2",
      revision: "eab522a8168e8b8b8c4856bf8609cd86198f01fe",
      estimatedSizeBytes: 228772419,
      license: "cc-by-nc-4.0",
      nonCommercial: true,
      licenseBasis: "Owner decision: CC BY-NC 4.0, noncommercial, attribution per NOTICE.",
      installState: "installed",
    },
    {
      componentId: "yue2_mert_v2_fullsong",
      requiredFor: ["cover"],
      repo: "m-a-p/MERT-v2-FullSong",
      revision: "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
      estimatedSizeBytes: 2529845146,
      license: "cc-by-nc-4.0",
      nonCommercial: true,
      licenseBasis: "Owner decision: CC BY-NC 4.0, noncommercial, attribution per NOTICE.",
      installState: "incomplete",
    },
  ],
  conditionalPurposes: { cover: { installState: "incomplete", blocked: false } },
  variants: [
    { variant: "bf16", installState: "installed" },
    { variant: "q8", installState: "derivationPending", derivationPending: true },
    { variant: "q4", installState: "pending", pendingArtifact: true },
  ],
};

describe("YuE2 model identity (sc-23000)", () => {
  it("reads the licence from the catalog's notice and carries the version", () => {
    expect(yue2ModelIdentity(YUE2_ENTRY)).toMatchObject({
      version: "YuE2 (v2)",
      license: "CC BY-NC 4.0",
      experimental: true,
      nonCommercial: true,
    });
    expect(usagePolicyChips(null, YUE2_ENTRY)).toEqual(["YuE2 (v2)", "Experimental", "Noncommercial", "CC BY-NC 4.0"]);
  });

  // sc-23002: the cover closure is installable (no block), per component and as a purpose.
  it("reads the cover closure's install state, sizes and licence from the catalog", () => {
    const setup = yue2CoverSetup(YUE2_ENTRY);
    // Mutation that reds this: reading `installed` from any single component instead of the purpose.
    expect(setup).toMatchObject({ declared: true, installState: "incomplete", installed: false, blocked: false });
    expect(setup.totalBytes).toBe(228772419 + 2529845146);
    expect(setup.components.map((row) => [row.componentId, row.installState, row.license, row.nonCommercial])).toEqual([
      ["yue2_sheetsage2", "installed", "CC BY-NC 4.0", true],
      ["yue2_mert_v2_fullsong", "incomplete", "CC BY-NC 4.0", true],
    ]);
    expect(yue2CoverSetup({ ...YUE2_ENTRY, conditionalPurposes: { cover: { installState: "installed" } } }).installed).toBe(true);
    // A block declared again is still reported, with its reason.
    const reblocked = {
      ...YUE2_ENTRY,
      conditionalComponents: [{ ...YUE2_ENTRY.conditionalComponents[0], blocked: { reason: "r", unblock: "u" } }],
    };
    expect(yue2CoverSetup(reblocked)).toMatchObject({ blocked: true, components: [{ blocked: { reason: "r", unblock: "u" } }] });
    // A cover-closure download is told apart from YuE2's own tier installs by its repo.
    expect(isCoverComponentDownload({ type: "model_download", payload: { modelId: "yue2", repo: "m-a-p/SheetSage2" } }, YUE2_ENTRY)).toBe(true);
    expect(isCoverComponentDownload({ type: "model_download", payload: { modelId: "yue2", repo: "m-a-p/YuE2-3B" } }, YUE2_ENTRY)).toBe(false);
  });

  it("treats a derived tier as installable and an unpublished one as not", () => {
    const rows = yue2TierRows(YUE2_ENTRY);
    expect(rows.find((row) => row.tier === "q8")).toMatchObject({ derived: true, installable: true });
    expect(rows.find((row) => row.tier === "q4")).toMatchObject({ installable: false, state: "not published yet" });
  });
});

describe("YuE2 run view (sc-23000)", () => {
  const POLICY = { schema: "sceneworks.usagePolicy.v1", nonCommercial: true, experimental: true, license: { license: null, notice: "CC BY-NC 4.0 …" } };

  it("reads the stage and progress of a running job", () => {
    const view = yue2RunView({
      id: "j",
      status: "running",
      progress: 0.42,
      message: "Generating semantic tokens: 120 of 9000.",
      payload: { yue2: { kind: "create" } },
    });
    expect(view).toMatchObject({ running: true, progress: 0.42, message: "Generating semantic tokens: 120 of 9000." });
  });

  it("reports truncation, warnings and side-effect errors truthfully", () => {
    const view = yue2RunView({
      id: "j",
      status: "completed",
      payload: { yue2: { kind: "create" } },
      result: {
        yue2: {
          truncated: { abc: false, semantic: true },
          warnings: [{ code: "semantic_truncated", message: "hit max tokens" }],
          sideEffectErrors: ["score version: boom"],
          usagePolicy: POLICY,
        },
      },
    });
    expect(view.truncations.map((item) => item.key)).toEqual(["semantic"]);
    expect(view.warnings).toEqual([{ code: "semantic_truncated", message: "hit max tokens" }]);
    expect(view.sideEffectErrors).toEqual(["score version: boom"]);
    expect(view.usagePolicy).toBe(POLICY);
  });

  it("reports unknown truncation as unknown, never as not truncated", () => {
    const view = yue2RunView({ status: "failed", payload: { yue2: { kind: "renderVersion" } }, result: { yue2: { truncated: null, error: "x" } } });
    expect(view.truncationUnknown).toBe(true);
    expect(view.truncations).toEqual([]);
    expect(yue2RunView({ status: "completed", payload: { yue2: {} }, result: { yue2: { truncated: { abc: false, semantic: false } } } }).truncationUnknown).toBe(false);
  });

  it("says a derived-tier install waits for an audio-lane worker", () => {
    expect(installJobStatusLabel({ status: "queued", payload: { localDerivation: { variant: "q8" } } })).toBe(
      "queued — waits for an audio-lane worker to derive it",
    );
    expect(installJobStatusLabel({ status: "running", payload: { localDerivation: { variant: "q8" } } })).toBe("deriving on this machine");
    expect(installJobStatusLabel({ status: "queued", payload: { variant: "bf16" } })).toBe("queued");
  });

  it("says when a truncated plan was not saved as a score version", () => {
    const view = yue2RunView({
      status: "completed",
      payload: { yue2: { kind: "plan" } },
      result: { yue2: { truncated: { abc: true }, scoreVersionSkipped: "abc_truncated", score: { abc: "X:1" } } },
    });
    expect(view.scoreVersionSkipped).toContain("not saved as a score version");
    expect(view.truncations.map((item) => item.key)).toEqual(["abc"]);
  });

  it("surfaces the worker's error for a failed job, never an empty state", () => {
    expect(
      yue2RunView({ status: "failed", payload: { yue2: {} }, result: { yue2: { error: "engine refused" } } }).error,
    ).toBe("engine refused");
    expect(yue2RunView({ status: "failed", payload: { yue2: {} }, error: "worker died" }).error).toBe("worker died");
    expect(yue2RunView({ status: "failed", payload: { yue2: {} } }).error).toBe(
      "The run failed without an error message.",
    );
  });

  it("carries the usage policy into the run export", () => {
    const job = {
      id: "j",
      status: "completed",
      payload: { yue2: { kind: "plan" }, usagePolicy: POLICY },
      result: { yue2: { effectiveSettings: { seed: 1 }, score: { abc: "X:1" }, run: { planIdentity: "ab" } } },
    };
    const exported = yue2RunExport(job, [{ id: "a1", extra: { usagePolicy: POLICY } }], "2026-01-01T00:00:00Z");
    expect(exported.usagePolicy).toBe(POLICY);
    expect(exported.effectiveSettings).toEqual({ seed: 1 });
    expect(exported.score).toEqual({ abc: "X:1" });
    expect(exported.audio[0].usagePolicy).toBe(POLICY);
  });
});

describe("YuE2 score edit operations (sc-23000)", () => {
  it("builds each bounded operation in the core's op-tagged shape", () => {
    const draft = defaultEditDraft();
    expect(
      buildEditOperation({ ...draft, op: "reharmonize", chordChanges: [{ bar: "2", onsetQuarters: "3/2", chord: "" }] }),
    ).toEqual({ operation: { op: "reharmonize", changes: [{ bar: 2, onsetQuarters: "3/2", chord: null }] } });
    expect(buildEditOperation({ ...draft, op: "strip_chords", keepVoice: "Vocal" })).toEqual({
      operation: { op: "strip_chords", keepVoice: "Vocal" },
    });
    expect(buildEditOperation({ ...draft, op: "set_tempo", bpm: "96" })).toEqual({ operation: { op: "set_tempo", bpm: 96 } });
    expect(buildEditOperation({ ...draft, op: "arrange_sections", sectionOrder: "0, 1, 1", lyrics: "x" })).toEqual({
      operation: { op: "arrange_sections", sectionOrder: [0, 1, 1], lyrics: "x" },
    });
    expect(buildEditOperation({ ...draft, op: "set_lyrics", lyrics: "new" })).toEqual({
      operation: { op: "set_lyrics", lyrics: "new" },
    });
    expect(buildEditOperation({ ...draft, op: "set_style", style: "jazz" })).toEqual({
      operation: { op: "set_style", style: "jazz" },
    });
    expect(
      buildEditOperation({
        ...draft,
        op: "replace_score",
        abc: "X:1",
        allowHarmony: true,
        allowMelody: true,
        melodyVoices: "both",
        melodyFromBar: "3",
        melodyToBar: "4",
        cot: "melody",
      }),
    ).toEqual({
      operation: {
        op: "replace_score",
        abc: "X:1",
        allow: { harmony: true, melody: { voices: ["Vocal", "Ins"], fromBar: 3, toBar: 4 } },
        cot: "melody",
      },
    });
  });

  it("names what is missing instead of sending an empty operation", () => {
    expect(buildEditOperation({ ...defaultEditDraft(), op: "set_tempo" }).error).toBe("Give the new tempo in BPM.");
  });
});

describe("YuE2 fix pass (sc-23000)", () => {
  it("exports a seed of 2^53 - 1 exactly", () => {
    const job = {
      id: "j",
      status: "completed",
      payload: { yue2: { kind: "create", seed: 9007199254740991 } },
      result: { yue2: { effectiveSettings: { seed: 9007199254740991 } } },
    };
    const exported = JSON.parse(JSON.stringify(yue2RunExport(job, [], "t")));
    expect(exported.effectiveSettings.seed).toBe(9007199254740991);
    expect(exported.submitted.seed).toBe(Number.MAX_SAFE_INTEGER);
  });

  it("refuses a typed seed JavaScript cannot hold exactly", () => {
    expect(seedValueProblem("9007199254740991")).toBeNull();
    expect(seedValueProblem("")).toBeNull();
    for (const bad of ["9007199254740992", "9007199254740993", "-1", "1.5", "1e3"]) {
      expect(seedValueProblem(bad), bad).toBe("The seed must be a whole number from 0 to 9007199254740991.");
    }
    expect(yue2RequestProblems("create", { ...defaultYue2Settings(), lyrics: "x", seed: "9007199254740993" })).toEqual([
      "The seed must be a whole number from 0 to 9007199254740991.",
    ]);
    // A kind that does not read the seed does not complain about it.
    expect(yue2RequestProblems("renderVersion", { ...defaultYue2Settings(), seed: "9007199254740993" }, { versionId: "v" })).toEqual([]);
  });

  it("says a workspace is needed", () => {
    expect(yue2RequestProblems("create", { ...defaultYue2Settings(), lyrics: "x" }, {}, { hasProject: false })).toEqual([
      "Open or create a workspace first.",
    ]);
  });

  it("gives core's reason for a control a kind does not read, and none for one it does", () => {
    expect(yue2FieldDisabledReason("count", "fromPlan")).toBe(
      "Not used by a From saved plan job: a restored plan, a score version and a cached decode render the same take every time.",
    );
    expect(yue2FieldDisabledReason("seed", "fromPlan")).toBe(
      "Not used by a From saved plan job: a saved plan fixes it (an edited plan is a new request).",
    );
    expect(yue2FieldDisabledReason("steps", "plan")).toBe(
      "Not used by a Plan only job: a plan-only job stops after planning the score.",
    );
    expect(yue2FieldDisabledReason("seed", "decode")).toBe(
      "Not used by a Cached decode job: a cached decode re-renders the source run's latents and generates nothing.",
    );
    expect(yue2FieldDisabledReason("seed", "create")).toBeNull();
    expect(yue2FieldDisabledReason("seed", null)).toBeNull();
  });

  it("round-trips an exported score through the header strip", () => {
    const policy = { nonCommercial: true, experimental: true, license: { notice: "(CC BY-NC 4.0)" } };
    const exported = yue2AbcExport("X:1\nT:\n", policy);
    expect(exported.split("\n")[0]).toBe(
      "% SceneWorks YuE2 export: weights licence CC BY-NC 4.0 · NONCOMMERCIAL USE ONLY · experimental model",
    );
    expect(stripYue2ExportHeader(exported)).toBe("X:1\nT:\n");
    expect(stripYue2ExportHeader("X:1\n% verse\n")).toBe("X:1\n% verse\n");
  });
});

describe("YuE2 cover from a recording (sc-23002)", () => {
  it("sends a transcription exactly its recording and window settings", () => {
    // Mutation that reds this: dropping `transcription` from YUE2_FIELD_KINDS (the block never sends).
    expect(buildYue2JobRequest("transcribe", EVERY_CONTROL, { sourceAudioAssetId: "asset_take" }, "auto")).toEqual({
      kind: "transcribe",
      sourceAudioAssetId: "asset_take",
      transcription: { maxSeconds: 90, overlapSeconds: 150, lookaheadSeconds: 50 },
      requestedGpu: "auto",
    });
    // Unset settings are omitted so the engine defaults apply.
    expect(buildYue2JobRequest("transcribe", defaultYue2Settings(), { sourceAudioAssetId: "asset_take" })).toEqual({
      kind: "transcribe",
      sourceAudioAssetId: "asset_take",
    });
  });

  it("never names a recording in a cover", () => {
    const body = buildYue2JobRequest("cover", EVERY_CONTROL, { sourceAudioAssetId: "asset_take" });
    // Mutation that reds this: adding sourceAudioAssetId to a cover (field table or cover block).
    expect(body).not.toHaveProperty("sourceAudioAssetId");
    expect(body.cover).not.toHaveProperty("sourceAudioAssetId");
    expect(body.cover.versionId).toBe("ver_1");
  });

  it("checks the window rule over the resolved values", () => {
    const at = (patch) => transcriptionSettingsProblems({ ...defaultYue2Settings(), ...patch });
    expect(at({})).toEqual([]);
    expect(at({ transcribeOverlapSeconds: "299.5", transcribeLookaheadSeconds: "299.5" })).toEqual([]);
    // Mutation that reds this: `overlap > 300` instead of `>= 300`.
    expect(at({ transcribeOverlapSeconds: "300" })).toEqual(["The window overlap must be at least 0 and under 300 seconds."]);
    // An overlap below the DEFAULT look-ahead (100 s) is refused — the rule reads resolved values.
    // Mutation that reds this: comparing only a typed look-ahead.
    expect(at({ transcribeOverlapSeconds: "50" })).toEqual(["The look-ahead must be between 0 and the window overlap (50 s)."]);
    expect(at({ transcribeLookaheadSeconds: "-1" })).toEqual(["The look-ahead must be between 0 and the window overlap (200 s)."]);
    expect(at({ transcribeMaxSeconds: "0" })).toEqual(["The transcription length limit must be more than 0 seconds."]);
    expect(yue2RequestProblems("transcribe", defaultYue2Settings(), {})).toEqual(["Choose the recording to transcribe."]);
  });

  it("covers a transcribed score only in its own mode", () => {
    const settings = { ...defaultYue2Settings(), lyrics: "x", coverVersionId: "ver_t", coverMode: "full" };
    const coverVersion = { id: "ver_t", cot: "melody", transcription: { transcriptionId: "t", mode: "melody" } };
    // Mutation that reds this: dropping the transcribed-mode check from yue2RequestProblems.
    expect(yue2RequestProblems("cover", settings, {}, { coverVersion })).toEqual([
      "This transcribed score is a melody-only score — cover it in melody mode.",
    ]);
    expect(yue2RequestProblems("cover", { ...settings, coverMode: "melody" }, {}, { coverVersion })).toEqual([]);
    // A version that did not come from a recording keeps the existing rules.
    expect(yue2RequestProblems("cover", settings, {}, { coverVersion: { id: "ver_t", cot: "melody" } })).toEqual([]);
  });

  it("reads the review, warnings, octave evidence, readiness and exports off the record", () => {
    const view = yue2TranscriptionView({
      id: "yue2t_1",
      device: "metal",
      sourceAudioAssetId: "asset_take",
      source: { name: "take.wav", duration_seconds: 31.5 },
      review: {
        voices: {
          vocal: { notes: 0, min_pitch: null, max_pitch: null, median_pitch: null },
          instrumental: { notes: 3, min_pitch: 48, max_pitch: 55, median_pitch: 50 },
        },
        distinct_chords: ["C:maj"],
        distinct_chord_roots: 1,
        keys: ["C:major"],
        sections: ["verse"],
        bars: 4,
        diagnostics: ["inferred 4/4"],
        warnings: [{ code: "ignored", message: "the record's warnings win" }],
      },
      warnings: [{ code: "empty_melody", message: "no melody notes" }],
      octaveEvidence: {
        method: "m",
        transcribed_midi_range: [60, 72],
        notes_checked: 4,
        notes_with_more_energy_at_f0_half: 3,
        fraction_f0_half_dominant: 0.75,
        notes: [{ start: 0.5, midi: 64 }],
      },
      readiness: { melody: { ready: false, reason: "the transcription has no melody notes" }, full: { ready: true } },
      versions: { melody: null, full: "ver_full" },
      versionErrors: { full: "unsupported notation" },
      abcErrors: { melody: "score syntax rejected", full: null },
      exports: [
        { path: "chord.lab", kind: "lab", sha256: "a" },
        { path: "melody.mid", kind: "midi", sha256: "b" },
        { path: "score.abc", kind: "abc", sha256: "c" },
        { path: "vocal.mid", kind: "midi", sha256: "d" },
      ],
    });
    expect(view.warnings).toEqual([{ code: "empty_melody", message: "no melody notes" }]);
    expect(view.readiness.melody).toEqual({ ready: false, reason: "the transcription has no melody notes" });
    expect(view.readiness.full).toEqual({ ready: true, reason: "" });
    expect(view.versionErrors).toEqual([{ mode: "full", message: "unsupported notation" }]);
    expect(view.abcErrors).toEqual({ melody: "score syntax rejected", full: null });
    expect(view.octave).toMatchObject({ checked: 4, f0HalfDominant: 3, fraction: 0.75, range: [60, 72] });
    // Mutation that reds this: grouping exports in manifest order instead of by kind.
    expect(view.exportGroups.map((group) => [group.kind, group.files.map((file) => file.path)])).toEqual([
      ["midi", ["melody.mid", "vocal.mid"]],
      ["lab", ["chord.lab"]],
      ["abc", ["score.abc"]],
    ]);
    expect(view.instrumental).toEqual({ notes: 3, minPitch: 48, maxPitch: 55, medianPitch: 50 });
    expect(view.device).toBe("metal");
    expect(midiNoteLabel(60)).toBe("C4 (60)");
    expect(midiNoteLabel(69.5)).toBe("A#4 (69.5)");
    expect(midiNoteLabel(null)).toBe("—");
  });
});
