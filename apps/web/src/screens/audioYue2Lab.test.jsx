import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";

// sc-23000 — the YuE2 Song Lab inside Audio Studio. Every control-mapping assertion below reads the
// request body the lab actually handed to `apiFetch` (the network seam), never component state.

const { apiFetchMock, persistMock } = vi.hoisted(() => ({
  apiFetchMock: vi.fn(),
  persistMock: vi.fn(() => Promise.resolve()),
}));

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, apiFetch: (...args) => apiFetchMock(...args) };
});

vi.mock("../uiPreferences.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, persistNavigationPreferences: (...args) => persistMock(...args) };
});

import { ApiError } from "../api.js";
import { AppContext } from "../context/AppContext.js";
import { AudioStudio } from "./AudioStudio.jsx";
import { seedStudioSettingsFromServer } from "../hooks/useStudioSettings.js";

// ---- fixtures -------------------------------------------------------------------------------

const KOKORO = {
  id: "kokoro_82m",
  name: "Kokoro 82M (Speech)",
  type: "audio",
  audio: { voices: [{ id: "af_heart", label: "Heart" }], languages: ["en-US"], sampleRates: [24000], maxDurationSecs: 30 },
};
const ACESTEP = {
  id: "acestep_v15_turbo",
  name: "ACE-Step v1.5 XL Turbo (Music)",
  type: "audio",
  audio: { languages: ["en"], sampleRates: [48000], maxDurationSecs: 600, editModes: ["repaint"], conditioning: ["AudioEdit"] },
};
const MOSS = {
  id: "moss_sfx_v2",
  name: "MOSS SoundEffect v2 (SFX)",
  type: "audio",
  audio: { languages: ["en"], sampleRates: [48000], maxDurationSecs: 30 },
};
const OPENVOICE = {
  id: "openvoice_v2",
  name: "OpenVoice V2 (Voice Conversion)",
  type: "audio",
  audio: { sampleRates: [22050], conditioning: ["ReferenceAudio"] },
};
// A YuE1 row as the catalog names it for the commercial pointer (not installed here).
const YUE1 = { id: "yue_en_cot", name: "YuE English CoT (Lyrics to Song)", type: "audio", installState: "missing" };

const BLOCK_REASON = "blocked: owner licensing decision for SheetSage2/MERT port code.";
const BLOCK_UNBLOCK = "The owner records a basis for the port code.";

// Mirrors config/manifests/builtin.models.jsonc `yue2` (the fields the lab reads).
function yue2Entry(overrides = {}) {
  return {
    id: "yue2",
    name: "YuE2 Song Generation (Experimental, Noncommercial)",
    type: "audio",
    experimental: true,
    nonCommercial: true,
    requiresLicenseAcknowledgment: true,
    licenseNotice:
      "YuE2 is an EXPERIMENTAL model offered for NONCOMMERCIAL use only. Its weights are licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).",
    licenseUrl: "https://huggingface.co/m-a-p/YuE2-3B/blob/1a96eca/LICENSE",
    commercialUse: {
      eligible: false,
      reason: "YuE2's weights are licensed under CC BY-NC 4.0",
      alternativeNote: "YuE1 is the intended commercially licensed lyrics-to-song option.",
      alternatives: ["yue_en_cot"],
    },
    audio: { sampleRates: [48000], supportsGuidance: true, supportsSymbolicSong: true, supportsAudioArtifacts: true },
    conditionalComponents: [
      { componentId: "yue2_sheetsage2", requiredFor: ["cover"], blocked: { reason: BLOCK_REASON, unblock: BLOCK_UNBLOCK } },
      { componentId: "yue2_mert_v2_fullsong", requiredFor: ["cover"], blocked: { reason: BLOCK_REASON, unblock: BLOCK_UNBLOCK } },
    ],
    installState: "installed",
    hasVariantMatrix: true,
    variants: [
      { variant: "bf16", installState: "installed" },
      { variant: "q8", installState: "installed" },
      { variant: "q4", installState: "derivationPending", derivationPending: true },
    ],
    ui: {
      label: "YuE2 (Experimental)",
      promptGuide: { title: "YuE2 Song Generation Guide", path: "/prompt-guides/yue2.md" },
    },
    ...overrides,
  };
}

const STANDARD = [KOKORO, MOSS, ACESTEP, OPENVOICE];
const NOTICE = "Rendering a YuE2 score version regenerates the whole recording from its score, style and lyrics.";
const POLICY = {
  schema: "sceneworks.usagePolicy.v1",
  modelId: "yue2",
  experimental: true,
  nonCommercial: true,
  license: { license: null, notice: "… (CC BY-NC 4.0) …" },
};

const INSPECTION = {
  scope: "exact",
  sha256: "ab",
  bpm: 88,
  unitLength: "1/16",
  durationQuarters: "32",
  nominalDurationSeconds: 21.8,
  sections: [
    { index: 0, label: "verse", firstBar: 1, barCount: 4, startQuarters: "0", lengthQuarters: "16" },
    { index: 1, label: "chorus", firstBar: 5, barCount: 4, startQuarters: "16", lengthQuarters: "16" },
  ],
  voices: {
    Vocal: { soundingNotes: 48, measures: 8, chords: [{ onsetQuarters: "0", chord: "C" }, { onsetQuarters: "4", chord: "G" }] },
    Ins: { soundingNotes: 0, measures: 8, chords: [] },
  },
};

const VERSION_SUMMARY = {
  id: "ver_1",
  createdAt: "2026-09-01T00:00:00Z",
  parentVersionId: null,
  rootVersionId: "ver_1",
  origin: "plan",
  editOperation: null,
  editBrief: null,
  cot: "full",
  renderCount: 1,
};
const VERSION_2 = { ...VERSION_SUMMARY, id: "ver_2", parentVersionId: "ver_1", origin: "edit", editOperation: "set_tempo", editBrief: "slower", renderCount: 0 };
const VERSION_RECORD = {
  id: "ver_1",
  parentVersionId: null,
  rootVersionId: "ver_1",
  origin: "plan",
  request: { style: "dream pop", lyrics: "[verse]\nhello", cot: "full", seed: 831001 },
  score: { abc: "X:1\nK:C\n% verse\nV: Vocal\nC4|" },
  renderNotice: NOTICE,
};

function router(overrides = {}) {
  const state = { ack: false, ...overrides };
  return async (path, _token, options = {}) => {
    const method = options.method ?? "GET";
    if (overrides.handle) {
      const handled = await overrides.handle(path, method, options);
      if (handled !== undefined) return handled;
    }
    if (path === "/api/v1/models/yue2/license-acknowledgment") {
      if (method === "PUT") {
        state.ack = true;
      } else if (method === "DELETE") {
        state.ack = false;
      }
      return { modelId: "yue2", required: true, termsSha256: "t", acknowledged: state.ack };
    }
    if (path === "/api/v1/projects/project_1/yue2/score-versions") {
      return { items: [VERSION_SUMMARY, VERSION_2], unreadable: [], renderNotice: NOTICE };
    }
    if (path.startsWith("/api/v1/projects/project_1/yue2/score-versions/") && path.endsWith("/inspection")) {
      return INSPECTION;
    }
    if (path.startsWith("/api/v1/projects/project_1/yue2/score-versions/")) {
      const id = path.split("/").pop();
      return {
        version: { ...VERSION_RECORD, id },
        renders: id === "ver_1" ? [{ id: "rnd_1", status: "completed", audioAssetId: "asset_song", truncated: { abc: false, semantic: false } }] : [],
      };
    }
    if (path === "/api/v1/yue2/score/inspect") {
      return INSPECTION;
    }
    if (path === "/api/v1/projects/project_1/yue2/comparisons") {
      if (method === "POST") {
        state.comparisons = [{
          id: "cmp_1",
          createdAt: "2026-09-02T00:00:00Z",
          a: { versionId: "ver_1", origin: "plan", render: { audioAssetId: "asset_song", truncated: {} } },
          b: { versionId: "ver_2", origin: "edit", editOperation: "set_tempo", render: null },
          lineage: "b_derives_from_a",
          symbolicDifferences: ["tempo 88 → 72"],
          warnings: [],
          renderNotice: NOTICE,
        }];
        return state.comparisons[0];
      }
      return { items: state.comparisons ?? [], unreadable: [], renderNotice: NOTICE };
    }
    if (path === "/api/v1/projects/project_1/yue2/jobs") {
      const body = JSON.parse(options.body);
      return { jobs: [{ id: "job_new", status: "queued", projectId: "project_1", payload: { yue2: { kind: body.kind } } }], batchId: null };
    }
    return {};
  };
}

function context(overrides = {}) {
  const models = overrides.models ?? [...STANDARD, YUE1, yue2Entry()];
  return {
    token: "tok",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    models,
    audioModels: models.filter((model) => model.type === "audio" && model.installState !== "missing"),
    jobs: [],
    audioLocalJobs: [],
    recentAudioAssets: [],
    jobAction: vi.fn(),
    createAudioJob: vi.fn(async () => null),
    rememberLocalGenerationJob: vi.fn(),
    createModelDownloadJob: vi.fn(async () => ({ id: "dl" })),
    setActiveView: vi.fn(),
    requestedGpu: "auto",
    preferencesHydrated: true,
    ...overrides,
  };
}

// ---- helpers --------------------------------------------------------------------------------

const settle = async () => {
  await act(async () => {
    for (let index = 0; index < 10; index += 1) await Promise.resolve();
  });
};
const wait = async (ms) => {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, ms));
  });
};
const buttonWithText = (root, text) =>
  [...root.querySelectorAll("button")].find((button) => button.textContent.replace(/\s+/g, " ").trim() === text);
const buttonStarting = (root, text) =>
  [...root.querySelectorAll("button")].find((button) => button.textContent.replace(/\s+/g, " ").trim().startsWith(text));
const byLabel = (root, label) => root.querySelector(`[aria-label="${label}"]`);
async function typeText(element, value) {
  await act(async () => {
    const proto = element.tagName === "TEXTAREA" ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(element, value);
    element.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
}
async function type(element, value) {
  await act(async () => setInput(element, value));
}
async function choose(element, value) {
  await act(async () => setSelect(element, value));
}
function calls(pathSuffix, method = "POST") {
  return apiFetchMock.mock.calls.filter(([path, , options]) => path.endsWith(pathSuffix) && (options?.method ?? "GET") === method);
}
function readBlob(blob) {
  return new Promise((resolve, reject) => {
    const reader = new window.FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}
function lastJobBody() {
  const call = calls("/yue2/jobs").at(-1);
  return call ? JSON.parse(call[2].body) : null;
}

const ENABLED_SETTINGS = { optIn: true };

function seedLab(yue2lab = ENABLED_SETTINGS, audio = { songLab: true }) {
  seedStudioSettingsFromServer({ project_1: { audio, yue2lab } });
}

describe("YuE2 Song Lab (sc-23000)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    apiFetchMock.mockReset();
    persistMock.mockClear();
    apiFetchMock.mockImplementation(router());
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.restoreAllMocks();
  });

  async function render(ctx) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={ctx}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await settle();
  }
  const lab = () => container.querySelector('[data-testid="yue2-song-lab"]');

  async function openEnabledLab(ctx = context(), routerOptions = { ack: true }) {
    apiFetchMock.mockImplementation(router(routerOptions));
    seedLab();
    await render(ctx);
    expect(lab(), "the lab opens").toBeTruthy();
    expect(container.querySelector('[data-testid="yue2-gate"]')).toBeNull();
  }

  // ---- AC1: explicit selection, never displacing V1 / standard defaults ------------------------

  it("leaves the standard modes, their defaults and pickers untouched when YuE2 is in the catalog", async () => {
    await render(context());
    const tabs = [...container.querySelector(".mode-control").querySelectorAll(".mode-tab")].map((tab) =>
      tab.textContent.trim(),
    );
    expect(tabs).toEqual(["Speech", "Music", "Sound FX", "Voice Clone"]);
    expect(buttonWithText(container.querySelector(".mode-control"), "Speech").className).toContain("active");
    expect(container.querySelector(".settings-field-model select").value).toBe("kokoro_82m");
    // The lab is offered, separately marked, and NOT selected.
    const labTab = container.querySelector('[data-testid="yue2-lab-tab"]');
    expect(labTab.textContent).toContain("Experimental");
    expect(labTab.getAttribute("aria-selected")).toBe("false");
    expect(lab()).toBeNull();
    // No lab request is made until the user chooses the lab.
    expect(apiFetchMock.mock.calls.some(([path]) => path.includes("yue2"))).toBe(false);
    // YuE2 never enters a standard picker; Music keeps its own default.
    await click(buttonWithText(container.querySelector(".mode-control"), "Music"));
    const musicOptions = [...container.querySelector(".settings-field-model select").options].map((option) => option.value);
    expect(musicOptions).toEqual(["acestep_v15_turbo"]);
    expect(musicOptions).not.toContain("yue2");
  });

  it("opens only on an explicit click, behind the opt-in and the server licence acknowledgment", async () => {
    await render(context());
    await click(container.querySelector('[data-testid="yue2-lab-tab"]'));
    await settle();
    const gate = container.querySelector('[data-testid="yue2-gate"]');
    expect(gate).toBeTruthy();
    expect(container.querySelector("form.studio-shell").hidden).toBe(true);
    // Version, experimental and licence are visible before anything can be selected or downloaded.
    expect(gate.textContent).toContain("YuE2 (v2)");
    expect(gate.textContent).toContain("Experimental");
    expect(gate.textContent).toContain("Noncommercial");
    expect(gate.textContent).toContain("CC BY-NC 4.0");
    expect(container.querySelector('[data-testid="yue2-commercial-note"]').textContent).toContain(
      "YuE English CoT (Lyrics to Song)",
    );
    const enable = buttonWithText(gate, "Enable the experimental Song Lab");
    expect(enable.disabled).toBe(true);
    const [understand, accept] = gate.querySelectorAll('input[type="checkbox"]');
    await click(understand);
    expect(enable.disabled).toBe(true);
    await click(accept);
    expect(enable.disabled).toBe(false);
    await click(enable);
    await settle();
    expect(calls("/license-acknowledgment", "PUT")).toHaveLength(1);
    expect(container.querySelector('[data-testid="yue2-compose"]')).toBeTruthy();
    // Leaving the lab returns to the standard form exactly as it was.
    await click(buttonWithText(lab(), "Speech"));
    expect(container.querySelector("form.studio-shell").hidden).toBe(false);
    expect(container.querySelector(".settings-field-model select").value).toBe("kokoro_82m");
  });

  it("does not re-check the acknowledgment when a catalog refresh hands back an equal entry", async () => {
    await openEnabledLab();
    const reads = calls("/license-acknowledgment", "GET").length;
    // A catalog refresh: same entry, new object identity.
    await render(context({ models: [...STANDARD, YUE1, yue2Entry()] }));
    expect(calls("/license-acknowledgment", "GET").length).toBe(reads);
    expect(container.querySelector('[data-testid="yue2-compose"]')).toBeTruthy();
    // Changed licence terms DO re-read it.
    await render(context({ models: [...STANDARD, YUE1, yue2Entry({ licenseNotice: "New terms (CC BY-NC 4.0)." })] }));
    expect(calls("/license-acknowledgment", "GET").length).toBe(reads + 1);
  });

  it("renders the acknowledgment read failure instead of an empty gate", async () => {
    apiFetchMock.mockImplementation(async (path) => {
      if (path.endsWith("/license-acknowledgment")) throw new ApiError("catalog unavailable", { status: 503 });
      return {};
    });
    seedLab();
    await render(context());
    expect(container.querySelector('[data-testid="yue2-ack-error"]').textContent).toContain("catalog unavailable");
    expect(buttonWithText(container, "Enable the experimental Song Lab").disabled).toBe(true);
  });

  // ---- AC3: persistence across relaunch (server prefs) ----------------------------------------

  it("restores the lab after a relaunch from the server-seeded settings and acknowledgment", async () => {
    await openEnabledLab(context(), { ack: true });
    // The seeded snapshot is what a relaunch hands back (App seeds it from GET /ui-preferences).
    window.localStorage.clear();
    await act(async () => root.unmount());
    ({ container, root } = mountRoot());
    seedLab({ optIn: true, lyrics: "restored lyrics", style: "lofi", tier: "q8", advancedOpen: false });
    await render(context());
    expect(byLabel(lab(), "Lyrics").value).toBe("restored lyrics");
    expect(byLabel(lab(), "Style").value).toBe("lofi");
    expect(byLabel(lab(), "Tier").value).toBe("q8");
    expect(calls("/license-acknowledgment", "PUT")).toHaveLength(0);
  });

  it("writes the lab settings to the server preferences", async () => {
    await openEnabledLab();
    await typeText(byLabel(lab(), "Lyrics"), "[chorus]\nremember me");
    await wait(500);
    const map = persistMock.mock.calls.at(-1)?.[0]?.advancedStudio;
    expect(map?.project_1?.yue2lab?.lyrics).toBe("[chorus]\nremember me");
    expect(map?.project_1?.yue2lab?.optIn).toBe(true);
    expect(map?.project_1?.audio?.songLab).toBe(true);
  });

  // ---- AC2: every control reaches the request -------------------------------------------------

  async function setEveryAdvancedControl() {
    await click(buttonStarting(lab(), "Advanced"));
    await type(byLabel(lab(), "Seed"), "42");
    await type(byLabel(lab(), "Guidance"), "1.5");
    await type(byLabel(lab(), "ODE steps"), "32");
    await choose(byLabel(lab(), "Precision"), "fp32");
    await choose(byLabel(lab(), "Offload"), "sequential");
    const score = { Temperature: "0.9", "Top-p": "0.95", "Top-k": "40", "Repetition penalty": "1.1", "Penalty window": "32", "Min tokens": "64", "Max tokens": "2048" };
    const semantic = { Temperature: "1", "Top-p": "0.9", "Top-k": "50", "Repetition penalty": "1.2", "Penalty window": "16", "Min tokens": "300", "Max tokens": "8000" };
    for (const [label, value] of Object.entries(score)) await type(byLabel(lab(), `Score planning sampling ${label}`), value);
    for (const [label, value] of Object.entries(semantic)) await type(byLabel(lab(), `Semantic sampling ${label}`), value);
    await choose(byLabel(lab(), "Stage residency"), "on");
    await choose(byLabel(lab(), "Chunked attention"), "on");
    await type(byLabel(lab(), "Attention chunk size"), "393216");
    await choose(byLabel(lab(), "Tiled decode"), "on");
    await type(byLabel(lab(), "Decode tile"), "256");
    await choose(byLabel(lab(), "Tier"), "q8");
    await choose(byLabel(lab(), "Decoder"), "legacy");
  }
  const SCORE_SAMPLING = { temperature: 0.9, topP: 0.95, topK: 40, repetitionPenalty: 1.1, penaltyWindow: 32, minTokens: 64, maxTokens: 2048 };
  const SEMANTIC_SAMPLING = { temperature: 1, topP: 0.9, topK: 50, repetitionPenalty: 1.2, penaltyWindow: 16, minTokens: 300, maxTokens: 8000 };
  const MEMORY = { stageResidency: true, chunkAttention: true, attentionChunkSize: 393216, tileVaeDecode: true, decodeTileEdge: 256 };

  it("sends every compose and advanced control in the create request", async () => {
    await openEnabledLab();
    expect(lab().querySelector(".advanced-section.open")).toBeNull(); // collapsed by default
    await typeText(byLabel(lab(), "Style"), "dream pop, airy vocal");
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await click(buttonWithText(lab(), "Melody only"));
    await type(byLabel(lab(), "Takes"), "3");
    await setEveryAdvancedControl();
    await click(buttonWithText(lab(), "Generate song"));
    await settle();
    expect(lastJobBody()).toEqual({
      kind: "create",
      style: "dream pop, airy vocal",
      lyrics: "[verse]\nhello",
      seed: 42,
      cfgScale: 1.5,
      steps: 32,
      planning: "melody",
      scoreSampling: SCORE_SAMPLING,
      semanticSampling: SEMANTIC_SAMPLING,
      decoder: "legacy",
      tier: "q8",
      precision: "fp32",
      offloadPolicy: "sequential",
      count: 3,
      memory: MEMORY,
      requestedGpu: "auto",
    });
  });

  it("sends a plan-only request with only the fields a plan reads", async () => {
    await openEnabledLab();
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await click(lab().querySelector('[data-testid="yue2-compose"] input[type="checkbox"]'));
    await setEveryAdvancedControl();
    await click(buttonWithText(lab(), "Plan the score"));
    await settle();
    expect(lastJobBody()).toEqual({
      kind: "plan",
      lyrics: "[verse]\nhello",
      seed: 42,
      cfgScale: 1.5,
      planning: "full",
      scoreSampling: SCORE_SAMPLING,
      tier: "q8",
      precision: "fp32",
      requestedGpu: "auto",
    });
  });

  it("plans from a supplied ABC score after previewing it", async () => {
    await openEnabledLab();
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await click(buttonWithText(lab(), "Supply an ABC score"));
    // A supplied score cannot be planned with planning off.
    expect(buttonWithText(lab(), "Off (no score)").disabled).toBe(true);
    await typeText(byLabel(lab(), "Supplied ABC score"), "X:1\nK:C\nC4|");
    await click(buttonWithText(lab(), "Preview score"));
    await settle();
    expect(JSON.parse(calls("/yue2/score/inspect").at(-1)[2].body)).toEqual({ abc: "X:1\nK:C\nC4|" });
    expect(lab().querySelector('[data-testid="yue2-score-preview"]').textContent).toContain("88 BPM");
    await click(buttonWithText(lab(), "Generate song"));
    await settle();
    expect(lastJobBody()).toEqual({ kind: "create", lyrics: "[verse]\nhello", planning: "full", score: "X:1\nK:C\nC4|", requestedGpu: "auto" });
  });

  it("renders from a restored saved plan", async () => {
    const plan = {
      id: "job_plan",
      type: "audio_generate",
      status: "completed",
      projectId: "project_1",
      createdAt: "2026-09-01T00:00:00Z",
      payload: { yue2: { kind: "plan", style: "folk" }, usagePolicy: POLICY },
      result: { yue2: { run: { kind: "plan", dir: "yue2/runs/x", identity: "ab", planIdentity: "cd" }, usagePolicy: POLICY } },
    };
    await openEnabledLab(context({ jobs: [plan] }));
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nnew words");
    await click(buttonWithText(lab(), "Restore a saved plan"));
    await choose(byLabel(lab(), "Saved plan"), "job_plan");
    await setEveryAdvancedControl();
    await click(buttonWithText(lab(), "Render from the saved plan"));
    await settle();
    expect(lastJobBody()).toEqual({
      kind: "fromPlan",
      lyrics: "[verse]\nnew words",
      steps: 32,
      semanticSampling: SEMANTIC_SAMPLING,
      decoder: "legacy",
      tier: "q8",
      precision: "fp32",
      offloadPolicy: "sequential",
      planJobId: "job_plan",
      memory: MEMORY,
      requestedGpu: "auto",
    });
  });

  // ---- AC2: cover from a reviewed score; transcription visibly blocked -------------------------

  it("shows recording transcription as blocked with its reason and offers no working button", async () => {
    await openEnabledLab();
    await click(buttonWithText(lab(), "Cover"));
    const blocked = lab().querySelector('[data-testid="yue2-transcription-blocked"]');
    expect(blocked.textContent).toContain("Blocked");
    expect(blocked.textContent).toContain(BLOCK_REASON);
    expect(blocked.textContent).toContain(BLOCK_UNBLOCK);
    expect(blocked.textContent).toContain("yue2_sheetsage2");
    expect(blocked.querySelector("button")).toBeNull();
    expect(calls("/yue2/jobs").length).toBe(0);
  });

  it("covers a reviewed score version with every cover control", async () => {
    await openEnabledLab();
    await click(buttonWithText(lab(), "Cover"));
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await typeText(byLabel(lab(), "Style"), "acoustic");
    await choose(byLabel(lab(), "Cover score version"), "ver_1");
    await click(buttonWithText(lab(), "Review score"));
    await settle();
    expect(lab().querySelector('[data-testid="yue2-cover-review"]').textContent).toContain("verse");
    await choose(byLabel(lab(), "Cover mode"), "melody");
    await choose(byLabel(lab(), "Cover keep"), "vocal");
    await typeText(byLabel(lab(), "Translated from"), "[verse]\nhola");
    expect(lab().querySelector('[data-testid="yue2-regeneration-notice"]').textContent).toContain("regenerates the whole recording");
    await setEveryAdvancedControl();
    await click(buttonWithText(lab(), "Generate cover"));
    await settle();
    expect(lastJobBody()).toEqual({
      kind: "cover",
      style: "acoustic",
      lyrics: "[verse]\nhello",
      seed: 42,
      cfgScale: 1.5,
      steps: 32,
      semanticSampling: SEMANTIC_SAMPLING,
      decoder: "legacy",
      tier: "q8",
      precision: "fp32",
      offloadPolicy: "sequential",
      cover: { versionId: "ver_1", mode: "melody", keep: "vocal", translatedFrom: "[verse]\nhola" },
      memory: MEMORY,
      requestedGpu: "auto",
    });
  });

  it("covers a pasted reviewed ABC score", async () => {
    await openEnabledLab();
    await click(buttonWithText(lab(), "Cover"));
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await click(buttonWithText(lab(), "Paste ABC"));
    await typeText(byLabel(lab(), "Cover ABC score"), "X:1\nK:C\nC4|");
    await click(buttonWithText(lab(), "Generate cover"));
    await settle();
    expect(lastJobBody()).toEqual({ kind: "cover", lyrics: "[verse]\nhello", cover: { score: "X:1\nK:C\nC4|", mode: "melody" }, requestedGpu: "auto" });
  });

  // ---- AC2: score preview, bounded edits, renders, comparisons --------------------------------

  it("previews a score version, checks an edit's invariants and states that renders regenerate the recording", async () => {
    let editBody = null;
    await openEnabledLab(
      context(),
      {
        ack: true,
        handle: async (path, method, options) => {
          if (path.endsWith("/score-versions/ver_1/edits") && method === "POST") {
            editBody = JSON.parse(options.body);
            return {
              dryRun: true,
              renderNotice: NOTICE,
              version: {
                id: "ver_preview",
                edit: {
                  invariants: {
                    match: true,
                    checks: [
                      { name: "notes", status: "unchanged" },
                      { name: "tempo", status: "changedAsDeclared" },
                    ],
                    violations: [],
                  },
                },
              },
            };
          }
          return undefined;
        },
      },
    );
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    const rows = lab().querySelectorAll('[data-testid="yue2-version-row"]');
    expect(rows).toHaveLength(2);
    await click(rows[0]);
    await settle();
    expect(lab().querySelector('[data-testid="yue2-score-preview"]').textContent).toContain("chorus");
    expect(lab().querySelector('[data-testid="yue2-version-abc"]').textContent).toContain("X:1");
    expect(lab().querySelector('[data-testid="yue2-regeneration-notice"]').textContent).toContain(
      "regenerates the whole recording",
    );
    await choose(byLabel(lab(), "Edit operation"), "set_tempo");
    await type(lab().querySelector('[data-testid="yue2-edit-panel"] input[type="number"]'), "72");
    await type(byLabel(lab(), "Edit brief"), "slower verse");
    await click(buttonWithText(lab(), "Check edit"));
    await settle();
    expect(editBody).toEqual({
      operation: { op: "set_tempo", bpm: 72 },
      brief: "slower verse",
      provenance: { actor: "user", channel: "ui" },
      dryRun: true,
    });
    const report = lab().querySelector('[data-testid="yue2-invariant-report"]');
    expect(report.textContent).toContain("Invariants hold");
    expect(report.textContent).toContain("tempo — changedAsDeclared");
  });

  it("renders an invariant violation from the server as the edit's report", async () => {
    await openEnabledLab(context(), {
      ack: true,
      handle: async (path, method) => {
        if (path.endsWith("/edits") && method === "POST") {
          throw new ApiError("The edit changes what its contract fixes: melody changed in bar 3", {
            status: 422,
            code: "yue2_invariant_violation",
            context: { match: false, checks: [{ name: "melody", status: "violated" }], violations: ["melody changed in bar 3"] },
          });
        }
        return undefined;
      },
    });
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    await click(lab().querySelectorAll('[data-testid="yue2-version-row"]')[0]);
    await settle();
    await choose(byLabel(lab(), "Edit operation"), "set_style");
    await type(lab().querySelector('[data-testid="yue2-edit-panel"] .yue2-edit-fields input'), "jazz");
    await type(byLabel(lab(), "Edit brief"), "restyle");
    await click(buttonWithText(lab(), "Save as new version"));
    await settle();
    expect(lab().querySelector('[data-testid="yue2-edit-error"]').textContent).toContain("melody changed in bar 3");
    expect(lab().querySelector('[data-testid="yue2-invariant-report"]').textContent).toContain("melody — violated");
  });

  it("shows a failed render's unknown truncation as unknown", async () => {
    await openEnabledLab(context(), {
      ack: true,
      handle: async (path) => {
        if (path === "/api/v1/projects/project_1/yue2/score-versions/ver_2") {
          return {
            version: { ...VERSION_RECORD, id: "ver_2" },
            renders: [{ id: "rnd_f", status: "failed", truncated: null, error: "worker died" }],
          };
        }
        return undefined;
      },
    });
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    await click(lab().querySelectorAll('[data-testid="yue2-version-row"]')[1]);
    await settle();
    const renders = lab().querySelector('[data-testid="yue2-version-renders"]');
    expect(renders.querySelector('[data-testid="yue2-render-truncation-unknown"]')).toBeTruthy();
    expect(renders.textContent).toContain("worker died");
  });

  it("renders a score version with the synthesis controls", async () => {
    await openEnabledLab();
    await setEveryAdvancedControl();
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    await click(lab().querySelectorAll('[data-testid="yue2-version-row"]')[0]);
    await settle();
    await click(buttonWithText(lab(), "Render this version"));
    await settle();
    expect(lastJobBody()).toEqual({
      kind: "renderVersion",
      versionId: "ver_1",
      steps: 32,
      semanticSampling: SEMANTIC_SAMPLING,
      decoder: "legacy",
      tier: "q8",
      precision: "fp32",
      offloadPolicy: "sequential",
      memory: MEMORY,
      requestedGpu: "auto",
    });
  });

  it("creates an A/B listening comparison and plays both recordings", async () => {
    const song = { id: "asset_song", type: "audio", projectId: "project_1", displayName: "Song", file: { path: "a.wav" } };
    await openEnabledLab(context({ assets: [song] }));
    await click(buttonWithText(lab(), "Compare"));
    await settle();
    await choose(byLabel(lab(), "Version A"), "ver_1");
    await choose(byLabel(lab(), "Version B"), "ver_2");
    await settle();
    await click(buttonWithText(lab(), "Compare A / B"));
    await settle();
    expect(JSON.parse(calls("/yue2/comparisons").at(-1)[2].body)).toEqual({
      versionA: "ver_1",
      versionB: "ver_2",
      provenance: { actor: "user", channel: "ui" },
    });
    const result = lab().querySelector('[data-testid="yue2-comparison"]');
    expect(result.querySelector('[data-testid="yue2-compare-side-A"] audio')).toBeTruthy();
    expect(result.querySelector('[data-testid="yue2-compare-side-B"]').textContent).toContain("No render to listen to.");
    expect(result.textContent).toContain("tempo 88 → 72");
    expect(result.querySelector('[data-testid="yue2-regeneration-notice"]')).toBeTruthy();
  });

  // ---- AC3: live progress, cancel, truthful errors / truncation, artifacts ---------------------

  const RUNNING = {
    id: "job_run",
    type: "audio_generate",
    status: "running",
    projectId: "project_1",
    createdAt: "2026-09-03T00:00:00Z",
    progress: 0.4,
    message: "Generating semantic tokens: 120 of 9000.",
    payload: { yue2: { kind: "create", style: "synthwave" }, usagePolicy: POLICY },
  };
  const PLAN_RUNNING = { ...RUNNING, id: "job_planning", progress: 0.2, message: "Planning the score: token 40 of 4096.", payload: { yue2: { kind: "plan" }, usagePolicy: POLICY } };
  const TRUNCATED = {
    id: "job_trunc",
    type: "audio_generate",
    status: "completed",
    projectId: "project_1",
    createdAt: "2026-09-02T00:00:00Z",
    payload: { yue2: { kind: "create", style: "ballad" }, usagePolicy: POLICY },
    result: {
      assetIds: ["asset_song"],
      yue2: {
        run: { kind: "song", dir: "yue2/runs/r", identity: "ab", planIdentity: "cd" },
        truncated: { abc: false, semantic: true },
        warnings: [{ code: "semantic_truncated", message: "semantic phase hit max_tokens" }],
        effectiveSettings: { seed: 7, tier: "q8", decoder: "standard" },
        model: { id: "m-a-p/YuE2-3B", revision: "1a96", tier: "q8" },
        decoder: { id: "m-a-p/YuE2-Vae", revision: "9553" },
        score: { abc: "X:1\nK:C\nC4|", sha256: "ff" },
        scoreVersionId: "ver_1",
        usagePolicy: POLICY,
      },
    },
  };
  const FAILED = {
    id: "job_fail",
    type: "audio_generate",
    status: "failed",
    projectId: "project_1",
    createdAt: "2026-09-01T00:00:00Z",
    error: "yue2: the context budget exceeds 24576 tokens",
    payload: { yue2: { kind: "create" }, usagePolicy: POLICY },
    result: { yue2: { status: "failed", error: "yue2: the context budget exceeds 24576 tokens", usagePolicy: POLICY } },
  };
  const SONG_ASSET = {
    id: "asset_song",
    type: "audio",
    projectId: "project_1",
    displayName: "ballad (truncated)",
    file: { path: "assets/audios/g/song.wav" },
    extra: { usagePolicy: POLICY },
  };

  it("shows live stage progress for plan-only and audio runs, and cancels", async () => {
    const ctx = context({ jobs: [RUNNING, PLAN_RUNNING] });
    await openEnabledLab(ctx);
    const cards = [...container.querySelectorAll('[data-testid="yue2-run-card"]')];
    const song = cards.find((card) => card.dataset.jobId === "job_run");
    const plan = cards.find((card) => card.dataset.jobId === "job_planning");
    expect(song.querySelector('[data-testid="yue2-run-progress"]').textContent).toContain("Generating semantic tokens: 120 of 9000.");
    expect(song.querySelector(".progress-track span").style.width).toBe("40%");
    expect(plan.textContent).toContain("Plan only");
    expect(plan.querySelector('[data-testid="yue2-run-progress"]').textContent).toContain("Planning the score: token 40 of 4096.");
    // The run card carries the version and licence.
    expect(song.querySelector('[data-testid="yue2-policy-chips"]').textContent).toContain("Noncommercial");
    expect(song.querySelector('[data-testid="yue2-policy-chips"]').textContent).toContain("YuE2 (v2)");
    await click(buttonWithText(song, "Cancel"));
    expect(ctx.jobAction).toHaveBeenCalledWith(RUNNING, "cancel");
  });

  it("renders truncation, warnings, failures and the finished run's artifacts truthfully", async () => {
    await openEnabledLab(context({ jobs: [TRUNCATED, FAILED], assets: [SONG_ASSET] }));
    const cards = [...container.querySelectorAll('[data-testid="yue2-run-card"]')];
    const done = cards.find((card) => card.dataset.jobId === "job_trunc");
    const failed = cards.find((card) => card.dataset.jobId === "job_fail");
    expect(done.querySelector('[data-testid="yue2-run-truncated"]').textContent).toContain("recording ends before the lyrics do");
    expect(done.querySelector('[data-testid="yue2-run-warnings"]').textContent).toContain("semantic phase hit max_tokens");
    expect(done.querySelector("audio")).toBeTruthy();
    expect(done.querySelector('[data-testid="yue2-effective-settings"]').textContent).toContain('"tier": "q8"');
    expect(failed.querySelector('[data-testid="yue2-run-error"]').textContent).toContain("context budget exceeds 24576 tokens");
    expect(failed.querySelector('[data-testid="yue2-run-progress"]')).toBeNull();
  });

  it("says a truncated plan was not saved as a score version", async () => {
    const truncatedPlan = {
      ...TRUNCATED,
      id: "job_plan_trunc",
      payload: { yue2: { kind: "plan" }, usagePolicy: POLICY },
      result: { yue2: { truncated: { abc: true, semantic: false }, scoreVersionSkipped: "abc_truncated", score: { abc: "X:1" }, usagePolicy: POLICY } },
    };
    await openEnabledLab(context({ jobs: [truncatedPlan] }));
    const card = container.querySelector('[data-testid="yue2-run-card"]');
    expect(card.querySelector('[data-testid="yue2-run-version-skipped"]').textContent).toContain("not saved as a score version");
    expect(card.querySelector('[data-testid="yue2-run-truncated"]').textContent).toContain("incomplete plan");
    expect(buttonWithText(card, "Open score version")).toBeUndefined();
  });

  it("exports the run record and score with the usage policy carried into them", async () => {
    const blobs = [];
    const created = vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
      blobs.push(blob);
      return "blob:x";
    });
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const anchors = [];
    const realClick = window.HTMLAnchorElement.prototype.click;
    vi.spyOn(window.HTMLAnchorElement.prototype, "click").mockImplementation(function recordClick() {
      anchors.push(this.download);
      return realClick.call(this);
    });
    await openEnabledLab(context({ jobs: [TRUNCATED], assets: [SONG_ASSET] }));
    const card = container.querySelector('[data-testid="yue2-run-card"]');
    await click(buttonStarting(card, "Run record (.json)"));
    await click(buttonStarting(card, "Score (.abc)"));
    await settle();
    expect(created).toHaveBeenCalledTimes(2);
    const record = JSON.parse(await readBlob(blobs[0]));
    expect(record.usagePolicy).toEqual(POLICY);
    expect(record.effectiveSettings).toEqual({ seed: 7, tier: "q8", decoder: "standard" });
    expect(record.audio[0].usagePolicy).toEqual(POLICY);
    const abc = await readBlob(blobs[1]);
    // The exported score names its licence on its first line, then the score unchanged.
    expect(abc.split("\n")[0]).toBe(
      "% SceneWorks YuE2 export: weights licence CC BY-NC 4.0 · NONCOMMERCIAL USE ONLY · experimental model",
    );
    expect(abc.split("\n").slice(1).join("\n")).toBe("X:1\nK:C\nC4|");
    expect(anchors).toContain("yue2-run-job_trunc-noncommercial.json");
    expect(anchors).toContain("yue2-score-job_trunc-noncommercial.abc");
    // The take downloads under the licence-marked name, never its style text.
    const audioLink = [...card.querySelectorAll("a[download]")].map((anchor) => anchor.getAttribute("download"));
    expect(audioLink).toEqual(["yue2-song-asset_song-noncommercial.wav"]);
  });

  it("decodes a finished song again with the chosen decoder", async () => {
    await openEnabledLab(context({ jobs: [TRUNCATED], assets: [SONG_ASSET] }));
    await choose(byLabel(lab(), "Decoder"), "standard");
    await click(buttonWithText(container.querySelector('[data-testid="yue2-run-card"]'), "Decode again"));
    await settle();
    expect(lastJobBody()).toEqual({ kind: "decode", sourceJobId: "job_trunc", decoder: "standard", requestedGpu: "auto" });
  });

  it("renders a refused submission with the server's reason, and re-gates a lapsed licence", async () => {
    let refusal = new ApiError("YuE2 recording transcription is blocked: reason.", {
      status: 403,
      code: "component_blocked",
      context: { purpose: "cover", blocked: [{ componentId: "yue2_sheetsage2", reason: BLOCK_REASON, unblock: BLOCK_UNBLOCK }] },
    });
    await openEnabledLab(context(), {
      ack: true,
      handle: async (path, method) => {
        if (path.endsWith("/yue2/jobs") && method === "POST") throw refusal;
        return undefined;
      },
    });
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await click(buttonWithText(lab(), "Generate song"));
    await settle();
    const error = lab().querySelector('[data-testid="yue2-submit-error"]');
    expect(error.textContent).toContain("transcription is blocked");
    expect(error.textContent).toContain(BLOCK_UNBLOCK);
    refusal = new ApiError("Model 'yue2' requires accepting its license (current terms) before it runs.", {
      status: 403,
      code: "license_acknowledgment_required",
    });
    await click(buttonWithText(lab(), "Generate song"));
    await settle();
    expect(container.querySelector('[data-testid="yue2-gate"]')).toBeTruthy();
  });

  // ---- AC1: presets retain version and licence; install (derived tiers) ------------------------

  it("saves presets that visibly retain the model version and licence", async () => {
    await openEnabledLab();
    await typeText(byLabel(lab(), "Style"), "shoegaze");
    await choose(byLabel(lab(), "Tier"), "q8");
    await type(byLabel(lab(), "Preset name"), "Wall of sound");
    await click(buttonStarting(lab(), "Save preset"));
    const presets = lab().querySelector('[data-testid="yue2-presets"]');
    expect(presets.textContent).toContain("Wall of sound");
    expect(presets.textContent).toContain("YuE2 (v2) · CC BY-NC 4.0 · Noncommercial");
    await typeText(byLabel(lab(), "Style"), "changed");
    await choose(byLabel(lab(), "Tier"), "bf16");
    await click(buttonWithText(presets, "Wall of sound"));
    // A preset restores controls; free text (style, lyrics, scores) is never part of a preset.
    expect(byLabel(lab(), "Tier").value).toBe("q8");
    expect(byLabel(lab(), "Style").value).toBe("changed");
  });

  it("installs a derived tier through the deriver when YuE2 is not installed", async () => {
    const entry = yue2Entry({
      installState: "missing",
      variants: [
        { variant: "bf16", installState: "missing" },
        { variant: "q8", installState: "derivationPending", derivationPending: true },
        { variant: "q4", installState: "derivationPending", derivationPending: true },
      ],
    });
    const ctx = context({ models: [...STANDARD, YUE1, entry] });
    await openEnabledLab(ctx);
    const install = lab().querySelector('[data-testid="yue2-install"]');
    expect(install.textContent).toContain("derived on this machine");
    await click(buttonWithText(install, "Derive q8 here"));
    expect(ctx.createModelDownloadJob).toHaveBeenCalledWith(entry, { variant: "q8", choices: undefined });
    // Nothing can run until it is installed.
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    expect(buttonWithText(lab(), "Generate song").disabled).toBe(true);
  });
  // ---- fix pass: E2 downloads, seeds, disabled controls, ack mirror, badges, drafts ------------

  it("names a version's exported score with its licence on the first line", async () => {
    const blobs = [];
    vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
      blobs.push(blob);
      return "blob:x";
    });
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const names = [];
    const realClick = window.HTMLAnchorElement.prototype.click;
    vi.spyOn(window.HTMLAnchorElement.prototype, "click").mockImplementation(function recordClick() {
      names.push(this.download);
      return realClick.call(this);
    });
    await openEnabledLab();
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    await click(lab().querySelectorAll('[data-testid="yue2-version-row"]')[0]);
    await settle();
    await click(buttonStarting(lab().querySelector('[data-testid="yue2-score-workbench"]'), "Score (.abc)"));
    const abc = await readBlob(blobs.at(-1));
    expect(abc.split("\n")[0]).toContain("NONCOMMERCIAL USE ONLY");
    expect(abc.split("\n")[0]).toContain("CC BY-NC 4.0");
    expect(abc.split("\n")[1]).toBe("X:1");
    expect(names.at(-1)).toBe("yue2-score-ver_1-noncommercial.abc");
  });

  it("strips the export header when an exported score is pasted back in", async () => {
    await openEnabledLab();
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await click(buttonWithText(lab(), "Supply an ABC score"));
    await typeText(
      byLabel(lab(), "Supplied ABC score"),
      "% SceneWorks YuE2 export: weights licence CC BY-NC 4.0 · NONCOMMERCIAL USE ONLY\nX:1\nK:C\nC4|",
    );
    await click(buttonWithText(lab(), "Generate song"));
    await settle();
    expect(lastJobBody().score).toBe("X:1\nK:C\nC4|");
  });

  it("sends a seed of 2^53 - 1 exactly and refuses a larger one", async () => {
    await openEnabledLab();
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await click(buttonStarting(lab(), "Advanced"));
    await type(byLabel(lab(), "Seed"), "9007199254740991");
    await click(buttonWithText(lab(), "Generate song"));
    await settle();
    expect(calls("/yue2/jobs").at(-1)[2].body).toContain('"seed":9007199254740991');
    expect(lastJobBody().seed).toBe(Number.MAX_SAFE_INTEGER);
    const sent = calls("/yue2/jobs").length;
    await type(byLabel(lab(), "Seed"), "9007199254740993");
    const generate = buttonWithText(lab(), "Generate song");
    expect(generate.disabled).toBe(true);
    expect(lab().querySelector('[data-testid="yue2-compose"]').textContent).toContain(
      "The seed must be a whole number from 0 to 9007199254740991.",
    );
    expect(calls("/yue2/jobs").length).toBe(sent);
  });

  it("refuses an import seed beyond 2^53 - 1 with a sentence, sending nothing", async () => {
    await openEnabledLab();
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    await click(buttonStarting(lab(), "Import ABC"));
    await typeText(byLabel(lab(), "ABC score to import"), "X:1\nK:C\nC4|");
    await type(byLabel(lab(), "Import seed"), "9007199254740993");
    await click(buttonWithText(lab(), "Save version"));
    await settle();
    expect(lab().querySelector('[data-testid="yue2-import-error"]').textContent).toContain(
      "The seed must be a whole number from 0 to 9007199254740991.",
    );
    expect(calls("/yue2/score-versions", "POST")).toHaveLength(0);
  });

  it("disables the controls a restored-plan render does not read, with the reason", async () => {
    const plan = {
      id: "job_plan",
      type: "audio_generate",
      status: "completed",
      projectId: "project_1",
      createdAt: "2026-09-01T00:00:00Z",
      payload: { yue2: { kind: "plan", style: "folk" }, usagePolicy: POLICY },
      result: { yue2: { run: { kind: "plan", dir: "yue2/runs/x", identity: "ab", planIdentity: "cd" }, usagePolicy: POLICY } },
    };
    await openEnabledLab(context({ jobs: [plan] }));
    await click(buttonStarting(lab(), "Advanced"));
    await type(byLabel(lab(), "Takes"), "3");
    await type(byLabel(lab(), "Seed"), "42");
    expect(byLabel(lab(), "Takes").disabled).toBe(false);
    await click(buttonWithText(lab(), "Restore a saved plan"));
    for (const [label, reason] of [
      ["Takes", "Not used by a From saved plan job: a restored plan, a score version and a cached decode render the same take every time."],
      ["Seed", "Not used by a From saved plan job: a saved plan fixes it (an edited plan is a new request)."],
      ["Guidance", "Not used by a From saved plan job: a saved plan fixes it (an edited plan is a new request)."],
    ]) {
      expect(byLabel(lab(), label).disabled, label).toBe(true);
      expect(byLabel(lab(), label).title, label).toBe(reason);
    }
    // Plan-only: synthesis controls are disabled with the plan-only reason.
    await click(buttonWithText(lab(), "Sample a new plan"));
    await click(lab().querySelector('[data-testid="yue2-compose"] input[type="checkbox"]'));
    expect(byLabel(lab(), "ODE steps").disabled).toBe(true);
    expect(byLabel(lab(), "ODE steps").title).toBe(
      "Not used by a Plan only job: a plan-only job stops after planning the score.",
    );
    expect(byLabel(lab(), "Offload").disabled).toBe(true);
    expect(byLabel(lab(), "Stage residency").disabled).toBe(true);
    expect(byLabel(lab(), "Semantic sampling Temperature").closest("fieldset").disabled).toBe(true);
    expect(byLabel(lab(), "Seed").disabled).toBe(false);
  });

  it("clears the browser licence flag when the server says the acceptance lapsed", async () => {
    window.localStorage.setItem("sceneworks-license-ack:yue2", "true");
    apiFetchMock.mockImplementation(router({ ack: false }));
    seedLab();
    await render(context());
    expect(container.querySelector('[data-testid="yue2-gate"]')).toBeTruthy();
    expect(window.localStorage.getItem("sceneworks-license-ack:yue2")).toBeNull();
  });

  it("clears the browser licence flag on a license_acknowledgment_required refusal", async () => {
    await openEnabledLab(context(), {
      ack: true,
      handle: async (path, method) => {
        if (path.endsWith("/yue2/jobs") && method === "POST") {
          throw new ApiError("requires accepting its license", { status: 403, code: "license_acknowledgment_required" });
        }
        return undefined;
      },
    });
    expect(window.localStorage.getItem("sceneworks-license-ack:yue2")).toBe("true");
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    await click(buttonWithText(lab(), "Generate song"));
    await settle();
    expect(window.localStorage.getItem("sceneworks-license-ack:yue2")).toBeNull();
  });

  it("marks a completed but truncated run, and a truncated version render, as such", async () => {
    await openEnabledLab(context({ jobs: [TRUNCATED], assets: [SONG_ASSET] }));
    const badge = container.querySelector('[data-testid="yue2-run-status"]');
    expect(badge.textContent).toBe("completed · truncated");
    expect(badge.className).toBe("status-badge warning");
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    await click(lab().querySelectorAll('[data-testid="yue2-version-row"]')[0]);
    await settle();
    // ver_1's render in the router is not truncated: plain completed.
    expect(lab().querySelector('[data-testid="yue2-render-status"]').textContent).toBe("completed");
  });

  it("marks a completed run whose truncation is unknown", async () => {
    const unknown = { ...TRUNCATED, id: "job_unknown", result: { ...TRUNCATED.result, yue2: { ...TRUNCATED.result.yue2, truncated: null } } };
    await openEnabledLab(context({ jobs: [unknown], assets: [SONG_ASSET] }));
    const badge = container.querySelector('[data-testid="yue2-run-status"]');
    expect(badge.textContent).toBe("completed · truncation unknown");
  });

  it("keeps a supplied score too large to restore visibly session-only, with the durable way out", async () => {
    await openEnabledLab();
    await click(buttonWithText(lab(), "Supply an ABC score"));
    const huge = `X:1\n${"C4|".repeat(6000)}`;
    await typeText(byLabel(lab(), "Supplied ABC score"), huge);
    const note = lab().querySelector('[data-testid="yue2-session-only-note"]');
    expect(note.textContent).toContain("Kept for this session only");
    await click(buttonWithText(note, "Import it as a score version"));
    await settle();
    expect(byLabel(lab(), "ABC score to import").value).toBe(huge);
  });

  it("keeps an in-progress edit draft across a tab change", async () => {
    await openEnabledLab();
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    await click(lab().querySelectorAll('[data-testid="yue2-version-row"]')[0]);
    await settle();
    await choose(byLabel(lab(), "Edit operation"), "set_style");
    await type(byLabel(lab(), "Edit brief"), "moodier");
    await click(buttonWithText(lab(), "Compose"));
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    expect(byLabel(lab(), "Edit operation").value).toBe("set_style");
    expect(byLabel(lab(), "Edit brief").value).toBe("moodier");
    await wait(500);
    expect(persistMock.mock.calls.at(-1)[0].advancedStudio.project_1.yue2lab.editDraft.brief).toBe("moodier");
  });

  it("refuses to submit without a workspace, saying so", async () => {
    seedStudioSettingsFromServer({ default: { audio: { songLab: true }, yue2lab: ENABLED_SETTINGS } });
    apiFetchMock.mockImplementation(router({ ack: true }));
    await render(context({ activeProject: null }));
    await typeText(byLabel(lab(), "Lyrics"), "[verse]\nhello");
    expect(buttonWithText(lab(), "Generate song").disabled).toBe(true);
    expect(lab().querySelector('[data-testid="yue2-compose"]').textContent).toContain("Open or create a workspace first.");
  });

  it("sends strip_chords with the core's capitalised voice name", async () => {
    let editBody = null;
    await openEnabledLab(context(), {
      ack: true,
      handle: async (path, method, options) => {
        if (path.endsWith("/edits") && method === "POST") {
          editBody = JSON.parse(options.body);
          return { dryRun: true, renderNotice: NOTICE, version: { id: "v", edit: { invariants: { match: true, checks: [], violations: [] } } } };
        }
        return undefined;
      },
    });
    await click(buttonWithText(lab(), "Scores"));
    await settle();
    await click(lab().querySelectorAll('[data-testid="yue2-version-row"]')[0]);
    await settle();
    await choose(byLabel(lab(), "Edit operation"), "strip_chords");
    await choose(lab().querySelector('[data-testid="yue2-edit-panel"] .yue2-edit-fields select'), "Vocal");
    await type(byLabel(lab(), "Edit brief"), "melody only");
    await click(buttonWithText(lab(), "Check edit"));
    await settle();
    expect(editBody.operation).toEqual({ op: "strip_chords", keepVoice: "Vocal" });
  });

  it("refreshes the compare render picker when a render finishes", async () => {
    let renders = [];
    const ctx = context();
    await openEnabledLab(ctx, {
      ack: true,
      handle: async (path) => {
        if (path === "/api/v1/projects/project_1/yue2/score-versions/ver_2") {
          return { version: { ...VERSION_RECORD, id: "ver_2" }, renders };
        }
        return undefined;
      },
    });
    await click(buttonWithText(lab(), "Compare"));
    await settle();
    await choose(byLabel(lab(), "Version B"), "ver_2");
    await settle();
    expect([...byLabel(lab(), "Render B").options].map((option) => option.value)).toEqual([""]);
    renders = [{ id: "rnd_new", status: "completed", audioAssetId: "asset_song", truncated: { abc: false, semantic: false } }];
    const done = {
      id: "job_render_done",
      type: "audio_generate",
      status: "completed",
      projectId: "project_1",
      createdAt: "2026-09-04T00:00:00Z",
      payload: { yue2: { kind: "renderVersion", versionId: "ver_2" }, usagePolicy: POLICY },
      result: { yue2: { renderRecordId: "rnd_new", usagePolicy: POLICY } },
    };
    await render({ ...ctx, jobs: [done] });
    await settle();
    expect([...byLabel(lab(), "Render B").options].map((option) => option.value)).toEqual(["", "rnd_new"]);
  });

  it("offers the standard models' downloads on the standard surface when only YuE2 is installed", async () => {
    const offerable = { ...KOKORO, installState: "missing", recommended: true };
    const ctx = context({ models: [offerable, YUE1, yue2Entry()] });
    await render(ctx);
    // The studio opens (the Song Lab is reachable) …
    expect(container.querySelector('[data-testid="yue2-lab-tab"]')).toBeTruthy();
    // … and the standard surface still offers the recommended standard download.
    const form = container.querySelector("form.studio-shell");
    expect(form.hidden).toBe(false);
    expect(form.textContent).toContain("No standard audio model installed");
    await click(buttonWithText(form, "Download"));
    expect(ctx.createModelDownloadJob).toHaveBeenCalledWith(offerable);
  });
  it("labels a YuE2 take in the standard results with version and licence, and downloads it marked", async () => {
    const song = { ...SONG_ASSET, recipe: { model: "yue2", prompt: "ballad" }, extra: { yue2: {}, usagePolicy: POLICY } };
    await render(context({ recentAudioAssets: [song], assets: [song] }));
    const group = container.querySelector('[data-testid="audio-run-group"]');
    expect(group.textContent).toContain("YuE2 · Experimental");
    expect(group.textContent).toContain("YuE2 (v2)");
    expect(group.textContent).toContain("CC BY-NC 4.0");
    expect(group.querySelector('[data-testid="audio-take-card"] a[download]').getAttribute("download")).toBe(
      "yue2-song-asset_song-noncommercial.wav",
    );
  });
});
