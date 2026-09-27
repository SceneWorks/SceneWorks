import { describe, expect, it } from "vitest";
import {
  YUE2_FIELD_KINDS,
  blockedTranscription,
  buildEditOperation,
  buildYue2JobRequest,
  defaultEditDraft,
  defaultYue2Settings,
  restoreYue2Settings,
  usagePolicyChips,
  yue2ModelIdentity,
  yue2RequestProblems,
  yue2RunExport,
  yue2RunView,
  yue2TierRows,
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
};

// The core contract's own table (crates/sceneworks-core/src/yue2_score/jobs.rs FIELD_KINDS).
const CORE_FIELD_KINDS = {
  style: ["create", "plan", "fromPlan", "cover"],
  lyrics: ["create", "plan", "fromPlan", "cover"],
  seed: ["create", "plan", "cover"],
  cfgScale: ["create", "plan", "cover"],
  steps: ["create", "fromPlan", "cover", "renderVersion"],
  planning: ["create", "plan"],
  score: ["create", "plan"],
  scoreSampling: ["create", "plan"],
  semanticSampling: ["create", "fromPlan", "cover", "renderVersion"],
  decoder: ["create", "fromPlan", "cover", "renderVersion", "decode"],
  tier: ["create", "plan", "fromPlan", "cover", "renderVersion", "decode"],
  precision: ["create", "plan", "fromPlan", "cover", "renderVersion", "decode"],
  offloadPolicy: ["create", "fromPlan", "cover", "renderVersion"],
  "memory.acoustic": ["create", "fromPlan", "cover", "renderVersion"],
  "memory.decode": ["create", "fromPlan", "cover", "renderVersion", "decode"],
  planJobId: ["fromPlan"],
  sourceJobId: ["decode"],
  versionId: ["renderVersion"],
  cover: ["cover"],
  sourceAudioAssetId: ["transcribe"],
  count: ["create", "plan", "cover"],
};

describe("YuE2 request builder (sc-23000)", () => {
  it("mirrors the core FIELD_KINDS table exactly", () => {
    expect(YUE2_FIELD_KINDS).toEqual(CORE_FIELD_KINDS);
  });

  it("never sends a field to a kind that does not read it", () => {
    for (const kind of ["create", "plan", "fromPlan", "cover", "renderVersion", "decode"]) {
      const body = buildYue2JobRequest(kind, EVERY_CONTROL, { versionId: "ver_1", sourceJobId: "job_src" });
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
      blocked: { reason: "blocked: owner licensing decision", unblock: "the owner records a basis" },
    },
  ],
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

  it("reports the recorded transcription block", () => {
    expect(blockedTranscription(YUE2_ENTRY)).toEqual([
      {
        componentId: "yue2_sheetsage2",
        repo: undefined,
        reason: "blocked: owner licensing decision",
        unblock: "the owner records a basis",
      },
    ]);
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
