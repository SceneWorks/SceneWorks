import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";

const { appConfirmMock } = vi.hoisted(() => ({ appConfirmMock: vi.fn(async () => true) }));
vi.mock("../appConfirm.jsx", async (importOriginal) => ({
  ...(await importOriginal()),
  appConfirm: appConfirmMock,
}));

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async () => ({})),
  };
});

import { AppContext } from "../context/AppContext.js";
import { AudioStudio } from "./AudioStudio.jsx";
import { KEEP_ALIVE_VIEWS, navSections, viewTitles } from "../App.jsx";
import { PROMPT_REFINE_MODEL_ID } from "../constants.js";

beforeEach(() => {
  appConfirmMock.mockReset();
  appConfirmMock.mockResolvedValue(true);
});

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

// Fixture audio models mirroring the seeded `type:"audio"` catalog entries (constants.js). Each
// carries only the `audio` sub-block the eligibility predicates + UI read — voices (speech),
// languages, maxDurationSecs, sampleRates, editModes (music), conditioning (voiceclone).
const KOKORO = {
  id: "kokoro_82m",
  name: "Kokoro 82M (Speech)",
  type: "audio",
  recommended: true,
  audio: {
    voices: [
      { id: "af_heart", label: "Heart" },
      { id: "am_michael", label: "Michael" },
      { id: "bf_emma", label: "Emma" },
    ],
    languages: ["en-US", "en-GB"],
    sampleRates: [24000],
    maxDurationSecs: 30,
  },
  ui: {
    label: "Kokoro 82M",
    promptGuide: { title: "Kokoro Speech Guide", path: "/prompt-guides/kokoro-82m.md" },
  },
};

const MOSS = {
  id: "moss_sfx_v2",
  name: "MOSS SoundEffect v2 (SFX)",
  type: "audio",
  audio: { languages: ["en", "zh"], sampleRates: [48000], maxDurationSecs: 30 },
  ui: {
    label: "MOSS SoundEffect v2",
    promptGuide: { title: "MOSS SoundEffect Guide", path: "/prompt-guides/moss-soundeffect-v2.md" },
  },
};

const ACESTEP = {
  id: "acestep_v15_turbo",
  name: "ACE-Step v1.5 XL Turbo (Music)",
  type: "audio",
  audio: {
    languages: ["en", "zh"],
    sampleRates: [48000],
    maxDurationSecs: 600,
    editModes: ["inpaint", "repaint", "extend", "cover"],
    conditioning: ["AudioEdit"],
  },
  ui: {
    label: "ACE-Step v1.5 XL Turbo",
    promptGuide: { title: "ACE-Step Music Guide", path: "/prompt-guides/acestep-v15-turbo.md" },
  },
};

const OPENVOICE = {
  id: "openvoice_v2",
  name: "OpenVoice V2 (Voice Conversion)",
  type: "audio",
  audio: { sampleRates: [22050], conditioning: ["ReferenceAudio"] },
  ui: {
    label: "OpenVoice V2",
    promptGuide: { title: "OpenVoice Guide", path: "/prompt-guides/openvoice-v2.md" },
  },
};

// Native clone-TTS generator (sc-13412): ReferenceAudio + VoiceEmbedding marks it as a single-call clone
// (isNativeCloneGenerator). Kept OUT of ALL_AUDIO so the existing converter-default tests stay unchanged;
// the sc-13412 tests add it explicitly to prove the picker prefers it and hides the OpenVoice τ.
const CHATTERBOX_TTS = {
  id: "chatterbox_tts",
  name: "Chatterbox (Cloned-Voice TTS)",
  type: "audio",
  audio: {
    languages: ["en", "en-US"],
    sampleRates: [24000],
    maxDurationSecs: 30,
    conditioning: ["VoiceEmbedding", "ReferenceAudio"],
  },
  ui: {
    label: "Chatterbox Clone-TTS",
    promptGuide: { title: "Chatterbox Guide", path: "/prompt-guides/chatterbox-tts.md" },
  },
};

// Streaming TTS (sc-13675): NO voice bank — serves Speech via audio.supportsStreaming. Kept OUT of
// ALL_AUDIO so the existing Speech tests keep Kokoro as the default; the streaming tests add it.
const MOSS_TTS_REALTIME = {
  id: "moss_tts_realtime",
  name: "MOSS-TTS-Realtime (Streaming Speech)",
  type: "audio",
  audio: {
    languages: ["en", "zh"],
    sampleRates: [24000],
    maxDurationSecs: 2400,
    supportsStreaming: true,
  },
  ui: {
    label: "MOSS-TTS-Realtime (Streaming)",
    promptGuide: { title: "MOSS Realtime Guide", path: "/prompt-guides/moss-tts-realtime.md" },
  },
};

// Multi-speaker dialogue TTS (sc-13676): NO voice bank — serves Speech via audio.supportsMultiSpeaker
// (+ maxSpeakers). Kept OUT of ALL_AUDIO so the existing Speech tests keep Kokoro as the default; the
// multi-speaker tests add it.
const MOSS_TTSD = {
  id: "moss_ttsd_v05",
  name: "MOSS-TTSD v0.5 (Multi-Speaker Dialogue)",
  type: "audio",
  audio: {
    languages: ["zh", "en"],
    sampleRates: [24000],
    maxDurationSecs: 300,
    supportsMultiSpeaker: true,
    maxSpeakers: 2,
  },
  ui: {
    label: "MOSS-TTSD v0.5 (Multi-Speaker)",
    promptGuide: { title: "MOSS Multi-Speaker Guide", path: "/prompt-guides/moss-ttsd-v05.md" },
  },
};

const ALL_AUDIO = [KOKORO, MOSS, ACESTEP, OPENVOICE];

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    audioModels: ALL_AUDIO,
    models: ALL_AUDIO,
    jobs: [],
    audioLocalJobs: [],
    jobAction: vi.fn(),
    createModelDownloadJob: vi.fn(),
    setActiveView: vi.fn(),
    setPreviewAsset: vi.fn(),
    macCapabilities: undefined,
    ...overrides,
  };
}

// A completed Speech run and its two takes (epic 14361 / sc-14364). The payload is what the
// studio actually posts, so the run header's chips and "Run again" are exercised against a
// real request rather than an invented shape.
const SPEECH_RUN_PAYLOAD = Object.freeze({
  model: "kokoro_82m",
  prompt: "She paused at the door, listening to the rain gather in the gutters.",
  voice: "af_heart",
  language: "en-US",
  targetDurationSecs: 12,
  seed: 4821,
});

const SPEECH_TAKES = [
  {
    id: "audio-asset-1",
    type: "audio",
    projectId: "project_1",
    displayName: "Take 1",
    file: { path: "assets/audios/genset_x/kokoro_take_1.wav", mimeType: "audio/wav", duration: 12 },
  },
  {
    id: "audio-asset-2",
    type: "audio",
    projectId: "project_1",
    displayName: "Take 2",
    file: { path: "assets/audios/genset_x/kokoro_take_2.wav", mimeType: "audio/wav", duration: 11 },
  },
];

const COMPLETED_SPEECH_JOB = {
  id: "audio-job-done",
  type: "audio_generate",
  status: "completed",
  createdAt: "2026-07-24T12:00:00Z",
  payload: SPEECH_RUN_PAYLOAD,
  result: { assetIds: ["audio-asset-1", "audio-asset-2"], expectedCount: 2 },
};

function takesContext(overrides = {}) {
  return baseContext({
    assets: SPEECH_TAKES,
    audioLocalJobs: [COMPLETED_SPEECH_JOB],
    createAudioJob: vi.fn(async () => null),
    rememberLocalGenerationJob: vi.fn(),
    ...overrides,
  });
}

const buttonWithText = (root, text) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent.trim() === text);
const modeTabs = (container) => container.querySelector(".mode-control");
const modeTab = (container, label) => buttonWithText(modeTabs(container), label);
const modelSelect = (container) => container.querySelector(".settings-field-model select");
const fieldByLabelStart = (container, label) =>
  [...container.querySelectorAll(".settings-bar label")].find((el) =>
    el.textContent.trim().startsWith(label),
  );
const setTextareaValue = async (el, value) => {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype,
      "value",
    ).set;
    setter.call(el, value);
    el.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
};
const settle = async () => {
  await act(async () => {
    for (let index = 0; index < 8; index += 1) {
      await Promise.resolve();
    }
  });
};

describe("AudioStudio shell (sc-13407)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("renders the four mode tabs in AUDIO_MODES order", async () => {
    await render(baseContext());
    const labels = [...modeTabs(container).querySelectorAll(".mode-tab")].map((b) => b.textContent.trim());
    expect(labels).toEqual(["Speech", "Music", "Sound FX", "Voice Clone"]);
  });

  it("shows an initially empty results zone", async () => {
    await render(baseContext());
    const results = container.querySelector(".studio-results");
    expect(results).toBeTruthy();
    expect(results.textContent).toContain("No audio yet");
    // No job cards until generation is wired (C1).
    expect(results.querySelector(".worker-progress-card")).toBeNull();
  });

  it("switches the active tab on click and snaps the model to one that serves it", async () => {
    await render(baseContext());

    // Opens on Speech (AUDIO_MODES[0]) — served by Kokoro.
    expect(modeTab(container, "Speech").className).toContain("active");
    expect(modelSelect(container).value).toBe("kokoro_82m");

    // Music is served only by ACE-Step, so switching snaps the model.
    await click(modeTab(container, "Music"));
    expect(modeTab(container, "Music").className).toContain("active");
    expect(modelSelect(container).value).toBe("acestep_v15_turbo");
  });

  it("drives the Speech settings from the selected model's audio Capabilities, not a hardcoded list", async () => {
    await render(baseContext());

    // Voice options come straight from KOKORO.audio.voices.
    const voiceSelect = fieldByLabelStart(container, "Voice").querySelector("select");
    expect([...voiceSelect.options].map((o) => o.value)).toEqual(["af_heart", "am_michael", "bf_emma"]);

    // Language options come from KOKORO.audio.languages.
    const langSelect = fieldByLabelStart(container, "Language").querySelector("select");
    expect([...langSelect.options].map((o) => o.value)).toEqual(["en-US", "en-GB"]);

    // Length is capped to KOKORO.audio.maxDurationSecs (30), never a hardcoded ceiling.
    const lengthInput = fieldByLabelStart(container, "Length").querySelector("input");
    expect(lengthInput.getAttribute("max")).toBe("30");
  });

  it("reflects a DIFFERENT model's capabilities — proving the fields aren't hardcoded", async () => {
    // A speech model whose voice bank + languages + cap differ from Kokoro's.
    const altSpeech = {
      id: "alt_speech",
      name: "Alt Speech",
      type: "audio",
      audio: {
        voices: [{ id: "nova", label: "Nova" }],
        languages: ["fr-FR"],
        sampleRates: [16000],
        maxDurationSecs: 12,
      },
      ui: { label: "Alt Speech" },
    };
    await render(baseContext({ audioModels: [altSpeech], models: [altSpeech] }));

    const voiceSelect = fieldByLabelStart(container, "Voice").querySelector("select");
    expect([...voiceSelect.options].map((o) => o.value)).toEqual(["nova"]);
    const langSelect = fieldByLabelStart(container, "Language").querySelector("select");
    expect([...langSelect.options].map((o) => o.value)).toEqual(["fr-FR"]);
    expect(fieldByLabelStart(container, "Length").querySelector("input").getAttribute("max")).toBe("12");
  });

  it("surfaces the Music edit ops from audio.editModes as a capability-driven scaffold", async () => {
    await render(baseContext());
    await click(modeTab(container, "Music"));

    const editChips = [...container.querySelectorAll(".settings-bar-styles .preset-chip")].map((b) =>
      b.textContent.trim(),
    );
    expect(editChips).toEqual(["inpaint", "repaint", "extend", "cover"]);
    // Music has no voice bank, so the Speech-only voice field is absent.
    expect(fieldByLabelStart(container, "Voice")).toBeFalsy();
  });

  it("surfaces the Voice Clone reference band + match strength from the converter (sc-13411)", async () => {
    await render(baseContext());
    await click(modeTab(container, "Voice Clone"));

    // Snaps to the CONVERTER (OpenVoice V2 — ReferenceAudio conditioning); a bare embedder never reaches
    // this tab. The reference-voice band + the match-strength control render for the real conversion.
    expect(modelSelect(container).value).toBe("openvoice_v2");
    const band = container.querySelector(".studio-source-band");
    expect(band).toBeTruthy();
    expect(band.textContent).toContain("Reference voice");
    expect(band.querySelector(".settings-field-match-strength input")).toBeTruthy();
  });

  it("filters a bare speaker embedder (Chatterbox-VE) out of the Voice Clone picker (sc-13411)", async () => {
    // Chatterbox-VE "serves" voiceclone conceptually (VoiceEmbedding) but cannot run the conversion, so
    // it must never appear in the generate picker — only converters (ReferenceAudio) do.
    const chatterbox = {
      id: "chatterbox_ve",
      name: "Chatterbox Voice Encoder",
      type: "audio",
      audio: { conditioning: ["VoiceEmbedding"] },
      ui: { label: "Chatterbox Voice Encoder" },
    };
    await render(
      baseContext({ audioModels: [...ALL_AUDIO, chatterbox], models: [...ALL_AUDIO, chatterbox] }),
    );
    await click(modeTab(container, "Voice Clone"));
    const options = [...modelSelect(container).querySelectorAll("option")].map((o) => o.value);
    expect(options).toContain("openvoice_v2");
    expect(options).not.toContain("chatterbox_ve");
  });

  it("does not treat a prompt-free voice encoder as a generator in a reduced catalog", async () => {
    const chatterbox = {
      id: "chatterbox_ve",
      name: "Chatterbox Voice Encoder",
      type: "audio",
      audio: { conditioning: ["VoiceEmbedding"] },
      ui: {
        label: "Chatterbox Voice Encoder",
        promptGuide: {
          title: "Chatterbox Voice Encoder Guide",
          path: "/prompt-guides/chatterbox-ve.md",
        },
      },
    };
    await render(baseContext({ audioModels: [chatterbox], models: [chatterbox] }));

    expect(container.querySelector(".model-availability-gate")).toBeTruthy();
    expect(container.querySelector(".audio-studio")).toBeNull();
    expect(buttonWithText(container, "Refine my prompt")).toBeUndefined();
  });

  it("renders the studio body when an audio model is installed (gate open)", async () => {
    await render(baseContext());
    expect(container.querySelector(".audio-studio")).toBeTruthy();
    expect(container.querySelector(".model-availability-gate")).toBeNull();
  });

  it("renders the ModelAvailabilityGate when no audio model is installed", async () => {
    // audioModels empty models the "catalog loaded, nothing installed" state (App.jsx fallback only
    // applies when the whole catalog is empty). The offers come from the full `models` catalog.
    await render(baseContext({ audioModels: [], models: ALL_AUDIO }));
    expect(container.querySelector(".model-availability-gate")).toBeTruthy();
    expect(container.querySelector(".audio-studio")).toBeNull();
    expect(container.textContent).toContain("Audio Studio needs an audio model");
  });
});

describe("AudioStudio prompt guides and refinement (sc-14353)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.restoreAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("opens the selected model's guide and switches from Kokoro to ACE-Step without stale content", async () => {
    const fetchGuide = vi.spyOn(globalThis, "fetch").mockImplementation(async (path) => ({
      ok: true,
      text: async () => `Guide for ${path}`,
    }));
    await render(baseContext());

    await click(buttonWithText(container, "Prompt guide"));
    await settle();
    expect(fetchGuide).toHaveBeenLastCalledWith("/prompt-guides/kokoro-82m.md");
    expect(document.querySelector("#prompt-guide-title").textContent).toBe("Kokoro Speech Guide");

    await click(buttonWithText(document, "Close"));
    await click(modeTab(container, "Music"));
    await click(buttonWithText(container, "Prompt guide"));
    await settle();
    expect(fetchGuide).toHaveBeenLastCalledWith("/prompt-guides/acestep-v15-turbo.md");
    expect(document.querySelector("#prompt-guide-title").textContent).toBe("ACE-Step Music Guide");
  });

  it("refines an ACE-Step music description with the audio workflow and applies only after review", async () => {
    const fetchGuide = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      text: async () => "# ACE-Step guide\n\nRefine only the Music Description.",
    });
    const refinePrompt = vi.fn(async () => "Dreamy synth-pop with warm analog pads and a rising final chorus.");
    await render(
      baseContext({
        refinePrompt,
        models: [
          ...ALL_AUDIO,
          { id: PROMPT_REFINE_MODEL_ID, name: "Prompt Refiner", installState: "installed" },
        ],
      }),
    );

    await click(modeTab(container, "Music"));
    const promptInput = container.querySelector('textarea[aria-label="Prompt"]');
    await setTextareaValue(promptInput, "dreamy pop");
    await click(buttonWithText(container, "Refine my prompt"));
    await settle();

    expect(fetchGuide).toHaveBeenCalledWith(
      "/prompt-guides/acestep-v15-turbo.md",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(refinePrompt).toHaveBeenCalledWith(expect.objectContaining({
      prompt: "dreamy pop",
      modelId: "acestep_v15_turbo",
      workflow: "audio",
      guide: "# ACE-Step guide\n\nRefine only the Music Description.",
      signal: expect.any(AbortSignal),
    }));
    expect(promptInput.value).toBe("dreamy pop");

    await click(buttonWithText(container, "Keep original"));
    expect(promptInput.value).toBe("dreamy pop");

    await click(buttonWithText(container, "Refine my prompt"));
    await settle();
    await click(buttonWithText(container, "Apply"));
    expect(promptInput.value).toBe(
      "Dreamy synth-pop with warm analog pads and a rising final chorus.",
    );
  });

  it("uses the generic audio guide for a third-party model without usable guide metadata", async () => {
    const thirdParty = {
      ...MOSS,
      id: "third_party_sfx",
      name: "Third-party SFX",
      ui: {
        label: "Third-party SFX",
        promptGuide: { title: " ", path: "" },
      },
    };
    const fetchGuide = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      text: async () => "# Generic audio guide",
    });
    await render(baseContext({ audioModels: [thirdParty], models: [thirdParty] }));

    await click(modeTab(container, "Sound FX"));
    await click(buttonWithText(container, "Prompt guide"));
    await settle();
    expect(fetchGuide).toHaveBeenLastCalledWith("/prompt-guides/generic-audio.md");
    expect(document.querySelector("#prompt-guide-title").textContent).toBe("Audio Prompt Guide");
  });

  it("aborts an in-flight refinement when the mode/model changes", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      text: async () => "# Speech guide",
    });
    let refineSignal;
    const refinePrompt = vi.fn(({ signal }) => {
      refineSignal = signal;
      return new Promise((resolve, reject) => {
        signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      });
    });
    await render(baseContext({ refinePrompt }));
    await setTextareaValue(container.querySelector('textarea[aria-label="Prompt"]'), "Read this aloud.");
    await click(buttonWithText(container, "Refine my prompt"));
    await settle();
    expect(refineSignal.aborted).toBe(false);

    await click(modeTab(container, "Music"));
    await settle();
    expect(refineSignal.aborted).toBe(true);
    expect(container.querySelector(".refine-review")).toBeNull();
    expect(container.querySelector(".refine-error")).toBeNull();
  });

  it("reuses the missing-refiner download affordance inside Audio Studio", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, text: async () => "# Guide" });
    const refinePrompt = vi.fn(async () => {
      throw new Error("prompt-refine model snapshot is not cached.");
    });
    const createModelDownloadJob = vi.fn(async () => ({ id: "download-refiner" }));
    await render(
      baseContext({
        refinePrompt,
        createModelDownloadJob,
        models: [
          ...ALL_AUDIO,
          { id: PROMPT_REFINE_MODEL_ID, name: "Prompt Refiner", installState: "missing" },
        ],
      }),
    );
    await setTextareaValue(container.querySelector('textarea[aria-label="Prompt"]'), "Read this.");
    await click(buttonWithText(container, "Refine my prompt"));
    await settle();

    await click(buttonWithText(container, "Download refinement model"));
    await settle();
    expect(createModelDownloadJob).toHaveBeenCalledWith(expect.objectContaining({
      id: PROMPT_REFINE_MODEL_ID,
    }));
    expect(container.querySelector(".refine-missing-model").textContent).toContain("Downloading");
  });

  it("guides and non-destructively refines each multi-speaker script turn", async () => {
    const fetchGuide = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      text: async () => "# Multi-speaker guide",
    });
    const refinePrompt = vi.fn(async () => "Hello there, and welcome.");
    await render(
      baseContext({
        audioModels: [MOSS_TTSD],
        models: [
          MOSS_TTSD,
          { id: PROMPT_REFINE_MODEL_ID, name: "Prompt Refiner", installState: "installed" },
        ],
        refinePrompt,
      }),
    );
    expect(container.querySelector('[data-testid="multi-speaker-script"]')).toBeTruthy();

    await click(buttonWithText(container, "Prompt guide"));
    await settle();
    expect(fetchGuide).toHaveBeenLastCalledWith("/prompt-guides/moss-ttsd-v05.md");
    expect(document.querySelector("#prompt-guide-title").textContent).toBe(
      "MOSS Multi-Speaker Guide",
    );
    await click(buttonWithText(document, "Close"));

    const firstTurn = container.querySelector('textarea[aria-label="Segment 1 text"]');
    const secondTurn = container.querySelector('textarea[aria-label="Segment 2 text"]');
    await setTextareaValue(firstTurn, "hello there and welcome");
    await setTextareaValue(secondTurn, "Keep my second turn.");
    const refineButtons = [...container.querySelectorAll("button")].filter((button) =>
      button.textContent.includes("Refine my prompt"),
    );
    expect(refineButtons).toHaveLength(2);
    await click(refineButtons[0]);
    await settle();

    expect(refinePrompt).toHaveBeenCalledWith(expect.objectContaining({
      prompt: "hello there and welcome",
      modelId: "moss_ttsd_v05",
      workflow: "audio",
      guide: "# Multi-speaker guide",
      signal: expect.any(AbortSignal),
    }));
    expect(firstTurn.value).toBe("hello there and welcome");
    expect(secondTurn.value).toBe("Keep my second turn.");

    await click(buttonWithText(container, "Apply"));
    expect(firstTurn.value).toBe("Hello there, and welcome.");
    expect(secondTurn.value).toBe("Keep my second turn.");
  });
});

// A Kokoro-shaped fixture whose voices carry the manifest's gender/accent so the grouped picker can
// be asserted (the sc-13407 KOKORO fixture ships a flat bank, exercising the ungrouped fallback).
const KOKORO_GROUPED = {
  id: "kokoro_82m",
  name: "Kokoro 82M (Speech)",
  type: "audio",
  recommended: true,
  audio: {
    voices: [
      { id: "af_heart", label: "Heart", gender: "female", accent: "american", language: "en-US" },
      { id: "am_michael", label: "Michael", gender: "male", accent: "american", language: "en-US" },
      { id: "bf_emma", label: "Emma", gender: "female", accent: "british", language: "en-GB" },
      { id: "bm_george", label: "George", gender: "male", accent: "british", language: "en-GB" },
    ],
    languages: ["en-US", "en-GB"],
    sampleRates: [24000],
    maxDurationSecs: 30,
  },
  ui: { label: "Kokoro 82M" },
};

describe("AudioStudio Speech generation (sc-13408)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const generateButton = (root) => buttonWithText(root, "Generate");
  const setTextarea = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const setSelect = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLSelectElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
  };
  const setNumber = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };

  it("Generate is disabled on an empty script and enabled once a script is typed", async () => {
    await render(baseContext({ createAudioJob: vi.fn(), rememberLocalGenerationJob: vi.fn() }));
    const button = generateButton(container);
    // Empty default script → the guard disables the CTA (never a silent no-op).
    expect(button.disabled).toBe(true);

    await setTextarea(container.querySelector(".prompt-input"), "The walking skeleton is alive.");
    expect(generateButton(container).disabled).toBe(false);
  });

  it("submitting the Speech form calls createAudioJob with args derived from the fields, then remembers the job in the audio lane", async () => {
    const job = { id: "audio-job-1", type: "audio_generate", status: "queued" };
    const createAudioJob = vi.fn(async () => job);
    const rememberLocalGenerationJob = vi.fn();
    await render(
      baseContext({
        audioModels: [KOKORO_GROUPED],
        models: [KOKORO_GROUPED],
        createAudioJob,
        rememberLocalGenerationJob,
      }),
    );

    await setTextarea(container.querySelector(".prompt-input"), "The walking skeleton is alive.");
    // Pick non-default values so the assertions discriminate (a hardcoded payload would not follow).
    await setSelect(fieldByLabelStart(container, "Voice").querySelector("select"), "bm_george");
    await setSelect(fieldByLabelStart(container, "Language").querySelector("select"), "en-GB");
    await setNumber(fieldByLabelStart(container, "Length").querySelector("input"), "6");
    // Advanced seed.
    await click(container.querySelector(".advanced-section-toggle"));
    await setNumber([...container.querySelectorAll(".advanced-panel input")][0], "123");

    await act(async () => {
      container
        .querySelector("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });

    expect(createAudioJob).toHaveBeenCalledTimes(1);
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.model).toBe("kokoro_82m");
    expect(payload.prompt).toBe("The walking skeleton is alive.");
    expect(payload.voice).toBe("bm_george");
    expect(payload.language).toBe("en-GB");
    expect(payload.targetDurationSecs).toBe(6);
    expect(payload.seed).toBe(123);
    // The returned job lands in the audio local-job lane so it stacks in the results zone.
    expect(rememberLocalGenerationJob).toHaveBeenCalledWith("audio", job);
  });

  it("clamps the requested length to the model's advertised cap", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "audio-job-clamp" }));
    await render(
      baseContext({
        audioModels: [KOKORO_GROUPED],
        models: [KOKORO_GROUPED],
        createAudioJob,
        rememberLocalGenerationJob: vi.fn(),
      }),
    );
    await setTextarea(container.querySelector(".prompt-input"), "Over the cap.");
    // The number input's max attribute won't stop a programmatic value; the submit clamps it.
    await setNumber(fieldByLabelStart(container, "Length").querySelector("input"), "999");
    await act(async () => {
      container
        .querySelector("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(createAudioJob.mock.calls[0][0].targetDurationSecs).toBe(30);
  });

  it("groups the voice options by accent + gender from the model's Capabilities", async () => {
    await render(
      baseContext({ audioModels: [KOKORO_GROUPED], models: [KOKORO_GROUPED], createAudioJob: vi.fn() }),
    );
    const voiceSelect = fieldByLabelStart(container, "Voice").querySelector("select");

    // The picker is structured into <optgroup>s whose labels come straight from accent + gender.
    const groups = [...voiceSelect.querySelectorAll("optgroup")];
    expect(groups.map((g) => g.getAttribute("label"))).toEqual([
      "American · Female",
      "American · Male",
      "British · Female",
      "British · Male",
    ]);
    // Each group holds exactly the voices with that accent+gender, in manifest order.
    const optionIds = (group) => [...group.querySelectorAll("option")].map((o) => o.value);
    expect(optionIds(groups[0])).toEqual(["af_heart"]);
    expect(optionIds(groups[1])).toEqual(["am_michael"]);
    expect(optionIds(groups[2])).toEqual(["bf_emma"]);
    expect(optionIds(groups[3])).toEqual(["bm_george"]);
    // And every advertised voice is still selectable (options flatten across optgroups).
    expect([...voiceSelect.options].map((o) => o.value)).toEqual([
      "af_heart",
      "am_michael",
      "bf_emma",
      "bm_george",
    ]);
  });

  it("groups a completed run into take cards instead of a worker card (sc-14364)", async () => {
    await render(takesContext());

    const results = container.querySelector(".studio-results");
    // The full worker card is the QUEUE's vocabulary now — a completed run never renders one here.
    expect(results.querySelector(".worker-progress-card")).toBeNull();
    expect(results.querySelector('[data-testid="audio-run-group"]')).toBeTruthy();
    expect(results.querySelectorAll('[data-testid="audio-take-card"]').length).toBe(2);
    expect(results.textContent).not.toContain("No audio yet");
    // The run header reads its chips off the job's OWN payload, never the live controls.
    const head = results.querySelector(".audio-run__head");
    expect([...head.querySelectorAll(".audio-run__chip")].map((el) => el.textContent)).toEqual([
      "af_heart",
      "en-US",
      "12 s",
      "seed 4821",
    ]);
    expect(head.textContent).toContain("Kokoro 82M (Speech)");
    // Nothing is loaded, so there is no deck and no <audio> element yet.
    expect(results.querySelector('[data-testid="audio-play-deck"]')).toBeNull();
  });

  it("playing a take mounts the deck on that clip and the × unloads it (sc-14364)", async () => {
    await render(takesContext());
    const results = container.querySelector(".studio-results");

    await click(results.querySelector('[aria-label="Play take 2"]'));

    const deck = results.querySelector('[data-testid="audio-play-deck"]');
    expect(deck).toBeTruthy();
    // The deck drives a REAL <audio> element pointed at the take that was played.
    const audioEl = deck.querySelector("audio");
    expect(audioEl.getAttribute("src")).toContain("kokoro_take_2.wav");
    expect(deck.textContent).toContain("Take 2");
    // …and that take's card carries the loaded ring, the first one does not.
    const cards = [...results.querySelectorAll('[data-testid="audio-take-card"]')];
    expect(cards[0].className).not.toContain("is-loaded");
    expect(cards[1].className).toContain("is-loaded");

    await click(deck.querySelector('[aria-label="Close player"]'));
    expect(results.querySelector('[data-testid="audio-play-deck"]')).toBeNull();
    expect(
      [...results.querySelectorAll('[data-testid="audio-take-card"]')].some((card) =>
        card.className.includes("is-loaded"),
      ),
    ).toBe(false);
  });

  it("Run again re-submits the run's OWN stored payload, not the current controls (sc-14364)", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "audio-job-2" }));
    const rememberLocalGenerationJob = vi.fn();
    await render(takesContext({ createAudioJob, rememberLocalGenerationJob }));

    await click(buttonWithText(container.querySelector(".audio-run__head"), "Run again"));

    expect(createAudioJob).toHaveBeenCalledTimes(1);
    expect(createAudioJob.mock.calls[0][0]).toEqual(SPEECH_RUN_PAYLOAD);
    expect(rememberLocalGenerationJob).toHaveBeenCalledWith("audio", { id: "audio-job-2" });
  });

  it("names the device the run is executing on in the strip's message line (screen 2a)", async () => {
    await render(
      baseContext({
        audioLocalJobs: [
          {
            id: "audio-job-running",
            type: "audio_generate",
            status: "running",
            progress: 0.44,
            message: "chunk 4 of 9 · ~6 s left",
            workerId: "worker-1",
            payload: { ...SPEECH_RUN_PAYLOAD },
            result: { expectedCount: 3 },
          },
        ],
        visibleWorkers: [{ id: "worker-1", capabilities: ["gpu"], gpuName: "Apple M3 Max (40-core)" }],
        createAudioJob: vi.fn(),
        rememberLocalGenerationJob: vi.fn(),
      }),
    );
    const strip = container.querySelector('[data-testid="audio-inflight-strip"]');
    expect(strip.textContent).toContain("chunk 4 of 9 · ~6 s left · Apple M3 Max (40-core)");
    // Still just the device — the hardware pills and GPU meters stay in the Queue.
    expect(strip.querySelector(".worker-progress-card__meters")).toBeNull();
  });

  it("renders a running run as the slim in-flight strip, not the full worker card (sc-14364)", async () => {
    const jobAction = vi.fn();
    await render(
      baseContext({
        audioLocalJobs: [
          {
            id: "audio-job-running",
            type: "audio_generate",
            status: "running",
            progress: 0.44,
            message: "chunk 4 of 9",
            payload: { ...SPEECH_RUN_PAYLOAD },
            result: { expectedCount: 3 },
          },
        ],
        jobAction,
        createAudioJob: vi.fn(),
        rememberLocalGenerationJob: vi.fn(),
      }),
    );

    const results = container.querySelector(".studio-results");
    const strip = results.querySelector('[data-testid="audio-inflight-strip"]');
    expect(strip).toBeTruthy();
    expect(strip.textContent).toContain("Speech · Kokoro 82M (Speech) · 3 takes");
    expect(strip.textContent).toContain("chunk 4 of 9");
    // `job.progress` is a 0..1 fraction, not a percentage — the strip must not render "44%"
    // for 0.44 (nor 4400% for 44).
    expect(strip.querySelector(".progress-track span").style.width).toBe("44%");
    // The GPU meters / job id / attempt counter stay in the Queue.
    expect(results.querySelector(".worker-progress-card")).toBeNull();

    await click(buttonWithText(strip, "Cancel"));
    expect(jobAction).toHaveBeenCalledWith(expect.objectContaining({ id: "audio-job-running" }), "cancel");
  });

  it("keeps the full worker card for a FAILED run so its error and retries stay reachable", async () => {
    await render(
      baseContext({
        audioLocalJobs: [
          {
            id: "audio-job-failed",
            type: "audio_generate",
            status: "failed",
            error: "the vocoder gave up",
            payload: { ...SPEECH_RUN_PAYLOAD },
          },
        ],
        createAudioJob: vi.fn(),
        rememberLocalGenerationJob: vi.fn(),
      }),
    );

    const results = container.querySelector(".studio-results");
    expect(results.querySelector(".worker-progress-card")).toBeTruthy();
    expect(results.textContent).toContain("the vocoder gave up");
  });

  it("Voice Clone needs a reference before Generate is enabled (sc-13411)", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "nope" }));
    await render(
      baseContext({ createAudioJob, rememberLocalGenerationJob: vi.fn() }),
    );
    // Voice Clone is served by OpenVoice in the base fixture; switch to it, type a script. With NO
    // reference selected the conversion has no target, so the CTA stays disabled and a direct submit is
    // a no-op — the guard is a missing reference now, not an unwired tab.
    await click(modeTab(container, "Voice Clone"));
    await setTextarea(container.querySelector(".prompt-input"), "reference this voice");
    expect(generateButton(container).disabled).toBe(true);
    await act(async () => {
      container
        .querySelector("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(createAudioJob).not.toHaveBeenCalled();
  });
});

describe("AudioStudio Sound FX generation (sc-13409)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const generateButton = (root) => buttonWithText(root, "Generate");
  const advancedFieldByLabel = (root, label) =>
    [...root.querySelectorAll(".advanced-panel label")].find((el) =>
      el.textContent.trim().startsWith(label),
    );
  const setTextarea = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const setSelect = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLSelectElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
  };
  const setNumber = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const submitForm = async () => {
    await act(async () => {
      container
        .querySelector("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
  };

  it("Generate is disabled on an empty SFX prompt and enabled once a description is typed", async () => {
    await render(baseContext({ createAudioJob: vi.fn(), rememberLocalGenerationJob: vi.fn() }));
    // Switch to Sound FX — the base fixture snaps the model to MOSS (the sole SFX model).
    await click(modeTab(container, "Sound FX"));
    expect(modelSelect(container).value).toBe("moss_sfx_v2");

    // Empty default prompt → the guard disables the CTA (never a silent no-op).
    expect(generateButton(container).disabled).toBe(true);
    await setTextarea(container.querySelector(".prompt-input"), "a heavy wooden door creaking open");
    expect(generateButton(container).disabled).toBe(false);
  });

  it("surfaces the CFG guidance + steps sampling knobs only on the Sound FX tab, not on Speech", async () => {
    await render(baseContext({ createAudioJob: vi.fn() }));
    // Speech (Kokoro) is not a diffusion model — no guidance/steps knobs.
    await click(container.querySelector(".advanced-section-toggle"));
    expect(advancedFieldByLabel(container, "Guidance")).toBeFalsy();
    expect(advancedFieldByLabel(container, "Steps")).toBeFalsy();

    // Sound FX (MOSS) exposes both — the diffusion sampling surface.
    await click(modeTab(container, "Sound FX"));
    expect(advancedFieldByLabel(container, "Guidance")).toBeTruthy();
    expect(advancedFieldByLabel(container, "Steps")).toBeTruthy();
    // MOSS ships no voice bank, so the Speech-only voice field never appears on SFX.
    expect(fieldByLabelStart(container, "Voice")).toBeFalsy();
  });

  it("submitting the Sound FX form calls createAudioJob with the SFX-derived payload, then remembers the job", async () => {
    const job = { id: "sfx-job-1", type: "audio_generate", status: "queued" };
    const createAudioJob = vi.fn(async () => job);
    const rememberLocalGenerationJob = vi.fn();
    await render(baseContext({ createAudioJob, rememberLocalGenerationJob }));

    await click(modeTab(container, "Sound FX"));
    await setTextarea(container.querySelector(".prompt-input"), "a heavy wooden door creaking open");
    // Non-default values so the assertions discriminate (a hardcoded payload would not follow them).
    await setSelect(fieldByLabelStart(container, "Language").querySelector("select"), "zh");
    await setNumber(fieldByLabelStart(container, "Length").querySelector("input"), "4");
    await click(container.querySelector(".advanced-section-toggle"));
    await setNumber(advancedFieldByLabel(container, "Seed").querySelector("input"), "11");
    await setNumber(advancedFieldByLabel(container, "Guidance").querySelector("input"), "6.5");
    await setNumber(advancedFieldByLabel(container, "Steps").querySelector("input"), "60");

    await submitForm();

    expect(createAudioJob).toHaveBeenCalledTimes(1);
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.model).toBe("moss_sfx_v2");
    expect(payload.prompt).toBe("a heavy wooden door creaking open");
    expect(payload.language).toBe("zh");
    expect(payload.targetDurationSecs).toBe(4);
    expect(payload.guidance).toBe(6.5);
    expect(payload.steps).toBe(60);
    expect(payload.seed).toBe(11);
    // MOSS advertises no voice surface, so the SFX payload never carries one.
    expect(payload.voice).toBeUndefined();
    // The returned job lands in the audio local-job lane so it stacks in the results zone.
    expect(rememberLocalGenerationJob).toHaveBeenCalledWith("audio", job);
  });

  it("omits guidance/steps when cleared so the model falls back to its own sampler default", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "sfx-job-default" }));
    await render(baseContext({ createAudioJob, rememberLocalGenerationJob: vi.fn() }));

    await click(modeTab(container, "Sound FX"));
    await setTextarea(container.querySelector(".prompt-input"), "gentle rain on a tin roof");
    // Leave guidance/steps untouched (empty) — they must not be sent.
    await submitForm();

    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.model).toBe("moss_sfx_v2");
    expect(payload.guidance).toBeUndefined();
    expect(payload.steps).toBeUndefined();
  });

  it("clamps the requested SFX length to the model's advertised cap", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "sfx-clamp" }));
    await render(baseContext({ createAudioJob, rememberLocalGenerationJob: vi.fn() }));
    await click(modeTab(container, "Sound FX"));
    await setTextarea(container.querySelector(".prompt-input"), "over the cap");
    await setNumber(fieldByLabelStart(container, "Length").querySelector("input"), "999");
    await submitForm();
    // MOSS advertises maxDurationSecs 30 — the submit clamps to it, never a hardcoded ceiling.
    expect(createAudioJob.mock.calls[0][0].targetDurationSecs).toBe(30);
  });
});

// A music model that (unlike the guidance-distilled ACE-Step turbo) DOES advertise CFG guidance +
// negative-prompt support — proves the advanced music knobs are capability-gated off the manifest
// flags, not hardcoded to the mode.
const MUSIC_WITH_GUIDANCE = {
  id: "music_guided",
  name: "Guided Music",
  type: "audio",
  audio: {
    languages: ["en"],
    sampleRates: [48000],
    maxDurationSecs: 120,
    editModes: ["inpaint", "repaint", "extend"],
    conditioning: ["AudioEdit"],
    supportsGuidance: true,
    supportsNegativePrompt: true,
  },
  ui: { label: "Guided Music" },
};

describe("AudioStudio Music generation (sc-13410)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const advancedFieldByLabel = (root, label) =>
    [...root.querySelectorAll(".advanced-panel label")].find((el) =>
      el.textContent.trim().startsWith(label),
    );
  const setTextarea = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const setNumber = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const setText = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const submitForm = async () => {
    await act(async () => {
      container
        .querySelector("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
  };

  it("switching to Music snaps to ACE-Step and reveals the describe-the-music fields + source band", async () => {
    await render(baseContext({ createAudioJob: vi.fn() }));
    await click(modeTab(container, "Music"));
    expect(modelSelect(container).value).toBe("acestep_v15_turbo");

    // The optional describe-the-music sub-block (BPM / key / lyrics) is present.
    expect(container.querySelector(".settings-field-bpm input")).toBeTruthy();
    expect(container.querySelector(".settings-field-key input")).toBeTruthy();
    expect(container.querySelector(".settings-field-lyrics textarea")).toBeTruthy();

    // The extend/edit source band is revealed (ACE-Step advertises editModes) with the three
    // advertised edit ops — capability-driven, never a hardcoded taxonomy.
    const band = container.querySelector(".studio-source-band");
    expect(band).toBeTruthy();
    expect(band.textContent).toContain("Source track");
    const editChips = [...band.querySelectorAll(".preset-chip")].map((b) => b.textContent.trim());
    expect(editChips).toEqual(["inpaint", "repaint", "extend", "cover"]);
  });

  it("hides the music fields + source band on Speech / Sound FX (no editModes advertised)", async () => {
    await render(baseContext({ createAudioJob: vi.fn() }));
    // Speech (Kokoro): no music sub-block, no source band.
    expect(container.querySelector(".settings-field-bpm input")).toBeNull();
    expect(container.querySelector(".studio-source-band")).toBeNull();
    // Sound FX (MOSS): still no editModes, so no source band.
    await click(modeTab(container, "Sound FX"));
    expect(container.querySelector(".settings-field-bpm input")).toBeNull();
    expect(container.querySelector(".studio-source-band")).toBeNull();
  });

  it("submitting Music maps the describe-the-music payload and omits guidance/negative for the distilled turbo", async () => {
    const job = { id: "music-job-1", type: "audio_generate", status: "queued" };
    const createAudioJob = vi.fn(async () => job);
    const rememberLocalGenerationJob = vi.fn();
    await render(baseContext({ createAudioJob, rememberLocalGenerationJob }));

    await click(modeTab(container, "Music"));
    await setTextarea(container.querySelector(".prompt-input"), "gentle lofi piano loop");
    // Non-default values so the assertions discriminate (a hardcoded payload would not follow them).
    await setNumber(container.querySelector(".settings-field-bpm input"), "92");
    await setText(container.querySelector(".settings-field-key input"), "C minor");
    await setTextarea(container.querySelector(".settings-field-lyrics textarea"), "[verse] la la la");
    await setNumber(fieldByLabelStart(container, "Length").querySelector("input"), "8");
    await click(container.querySelector(".advanced-section-toggle"));
    await setNumber(advancedFieldByLabel(container, "Seed").querySelector("input"), "77");
    await setNumber(advancedFieldByLabel(container, "Steps").querySelector("input"), "8");

    await submitForm();

    expect(createAudioJob).toHaveBeenCalledTimes(1);
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.model).toBe("acestep_v15_turbo");
    expect(payload.prompt).toBe("gentle lofi piano loop");
    expect(payload.bpm).toBe(92);
    expect(payload.musicalKey).toBe("C minor");
    expect(payload.lyrics).toBe("[verse] la la la");
    expect(payload.targetDurationSecs).toBe(8);
    expect(payload.steps).toBe(8);
    expect(payload.seed).toBe(77);
    // The guidance-distilled ACE-Step turbo advertises neither guidance nor negative-prompt support,
    // so the studio never sends them (the gen-core floor would reject them as typed Unsupported).
    expect(payload.guidance).toBeUndefined();
    expect(payload.negativePrompt).toBeUndefined();
    // Music carries no voice, and no source band was picked → plain text-to-music.
    expect(payload.voice).toBeUndefined();
    expect(payload.sourceAudioAssetId).toBeUndefined();
    expect(payload.editMode).toBeUndefined();
    expect(rememberLocalGenerationJob).toHaveBeenCalledWith("audio", job);
  });

  it("hides guidance + negative-prompt for ACE-Step but shows steps (capability-gated)", async () => {
    await render(baseContext({ createAudioJob: vi.fn() }));
    await click(modeTab(container, "Music"));
    await click(container.querySelector(".advanced-section-toggle"));
    // ACE-Step reads the top-level `steps`, so the solver-step count surfaces...
    expect(advancedFieldByLabel(container, "Steps")).toBeTruthy();
    // ...but the distilled turbo advertises no guidance / negative-prompt support, so neither shows.
    expect(advancedFieldByLabel(container, "Guidance")).toBeFalsy();
    expect(advancedFieldByLabel(container, "Negative prompt")).toBeFalsy();
  });

  it("shows AND sends guidance + negative for a music model that advertises them (not hardcoded)", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "guided-1" }));
    await render(
      baseContext({
        audioModels: [MUSIC_WITH_GUIDANCE],
        models: [MUSIC_WITH_GUIDANCE],
        createAudioJob,
        rememberLocalGenerationJob: vi.fn(),
      }),
    );
    await click(modeTab(container, "Music"));
    await setTextarea(container.querySelector(".prompt-input"), "orchestral swell");
    await click(container.querySelector(".advanced-section-toggle"));
    // This model advertises both — so the knobs are present (capability-gated off the manifest flags).
    expect(advancedFieldByLabel(container, "Guidance")).toBeTruthy();
    expect(advancedFieldByLabel(container, "Negative prompt")).toBeTruthy();
    await setNumber(advancedFieldByLabel(container, "Guidance").querySelector("input"), "4.5");
    await setTextarea(
      advancedFieldByLabel(container, "Negative prompt").querySelector("textarea"),
      "harsh distortion",
    );
    await submitForm();
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.model).toBe("music_guided");
    expect(payload.guidance).toBe(4.5);
    expect(payload.negativePrompt).toBe("harsh distortion");
  });

  it("rides an extend edit through as an AudioEdit: source + editMode=extend + editRegionEndSecs=length", async () => {
    // A source track is a persisted user selection, so it restores from the studio snapshot at mount
    // (like the Video Studio source band). Seed it directly, then pick the edit op.
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ sourceAudioAssetId: "audio-src-1" }),
    );
    const sourceAsset = { id: "audio-src-1", type: "audio", displayName: "Base loop" };
    const createAudioJob = vi.fn(async () => ({ id: "extend-1" }));
    await render(
      baseContext({
        assets: [sourceAsset],
        createAudioJob,
        rememberLocalGenerationJob: vi.fn(),
      }),
    );
    await click(modeTab(container, "Music"));
    await setTextarea(container.querySelector(".prompt-input"), "extend the outro");
    await setNumber(fieldByLabelStart(container, "Length").querySelector("input"), "20");
    // Pick the "extend" edit op from the advertised chips.
    const band = container.querySelector(".studio-source-band");
    const extendChip = [...band.querySelectorAll(".preset-chip")].find(
      (b) => b.textContent.trim() === "extend",
    );
    await click(extendChip);
    await submitForm();

    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.sourceAudioAssetId).toBe("audio-src-1");
    expect(payload.editMode).toBe("extend");
    // Extend reuses the Length field as the new TOTAL length (editRegionEndSecs); the worker begins the
    // appended tail at the source clip's own length. No interior region start/end is sent for extend.
    expect(payload.editRegionEndSecs).toBe(20);
    expect(payload.editRegionStartSecs).toBeUndefined();
  });

  it("rides an inpaint edit through with a bounded region + strength", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ sourceAudioAssetId: "audio-src-1" }),
    );
    const sourceAsset = { id: "audio-src-1", type: "audio", displayName: "Base loop" };
    const createAudioJob = vi.fn(async () => ({ id: "inpaint-1" }));
    await render(
      baseContext({
        assets: [sourceAsset],
        createAudioJob,
        rememberLocalGenerationJob: vi.fn(),
      }),
    );
    await click(modeTab(container, "Music"));
    await setTextarea(container.querySelector(".prompt-input"), "repaint the bridge");
    // editMode defaults to the first advertised op ("inpaint") — which reveals the region window.
    await setNumber(container.querySelector(".settings-field-region-start input"), "2");
    await setNumber(container.querySelector(".settings-field-region-end input"), "5");
    await setNumber(container.querySelector(".settings-field-edit-strength input"), "0.7");
    await submitForm();

    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.sourceAudioAssetId).toBe("audio-src-1");
    expect(payload.editMode).toBe("inpaint");
    expect(payload.editRegionStartSecs).toBe(2);
    expect(payload.editRegionEndSecs).toBe(5);
    expect(payload.editStrength).toBe(0.7);
  });

  it("rides a cover restyle through as a whole-clip AudioEdit: source + editMode=cover, no region (sc-13821)", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ sourceAudioAssetId: "audio-src-1" }),
    );
    const sourceAsset = { id: "audio-src-1", type: "audio", displayName: "Base loop" };
    const createAudioJob = vi.fn(async () => ({ id: "cover-1" }));
    await render(
      baseContext({
        assets: [sourceAsset],
        createAudioJob,
        rememberLocalGenerationJob: vi.fn(),
      }),
    );
    await click(modeTab(container, "Music"));
    await setTextarea(container.querySelector(".prompt-input"), "a brass ensemble of trumpets");
    // Default op is "inpaint", which reveals the region window: set a region FIRST, then switch to
    // cover. This is the mutation guard — without cover's region-less payload branch the stale 2..5
    // region would leak onto the whole-clip restyle.
    await setNumber(container.querySelector(".settings-field-region-start input"), "2");
    await setNumber(container.querySelector(".settings-field-region-end input"), "5");
    // Pick the "cover" edit op — the whole-clip restyle backed by the sft_cover coRequisite.
    const band = container.querySelector(".studio-source-band");
    const coverChip = [...band.querySelectorAll(".preset-chip")].find(
      (b) => b.textContent.trim() === "cover",
    );
    expect(coverChip).toBeTruthy();
    await click(coverChip);
    // Cover is whole-clip: the interior region window must NOT be shown (only inpaint/repaint reveal it).
    expect(container.querySelector(".settings-field-region-start")).toBeNull();
    expect(container.querySelector(".settings-field-region-end")).toBeNull();
    await submitForm();

    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.sourceAudioAssetId).toBe("audio-src-1");
    expect(payload.editMode).toBe("cover");
    // No region for a whole-clip restyle (the worker's audio_edit_region returns None for Cover).
    expect(payload.editRegionStartSecs).toBeUndefined();
    expect(payload.editRegionEndSecs).toBeUndefined();
  });
});

describe("AudioStudio Voice Clone generation (sc-13411)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const generateButton = (root) => buttonWithText(root, "Generate");
  const setTextarea = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const setNumber = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const submitForm = async () => {
    await act(async () => {
      container
        .querySelector("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
  };

  it("submits the Voice Clone chain: model=converter + referenceAudioAssetId + matchStrength + script, no voice", async () => {
    // The reference is a persisted user selection, so it restores from the studio snapshot at mount (like
    // the Music source band). Seed it, then the CTA is enabled and submit builds the conversion payload.
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ referenceAudioAssetId: "ref-voice-1" }),
    );
    const referenceAsset = { id: "ref-voice-1", type: "audio", displayName: "My reference voice" };
    const job = { id: "voiceclone-1", type: "audio_generate" };
    const createAudioJob = vi.fn(async () => job);
    const rememberLocalGenerationJob = vi.fn();
    await render(
      baseContext({
        assets: [referenceAsset],
        createAudioJob,
        rememberLocalGenerationJob,
      }),
    );
    await click(modeTab(container, "Voice Clone"));
    await setTextarea(
      container.querySelector(".prompt-input"),
      "Clone this into my reference voice.",
    );
    // A discriminating, non-default match strength so the test can't false-green on a default.
    await setNumber(container.querySelector(".settings-field-match-strength input"), "0.65");
    expect(generateButton(container).disabled).toBe(false);
    await submitForm();

    expect(createAudioJob).toHaveBeenCalledTimes(1);
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.model).toBe("openvoice_v2");
    expect(payload.prompt).toBe("Clone this into my reference voice.");
    expect(payload.referenceAudioAssetId).toBe("ref-voice-1");
    expect(payload.matchStrength).toBe(0.65);
    // No base voice is sent from the voiceclone tab (Kokoro's default reads the script); no music/edit knobs.
    expect(payload.voice).toBeUndefined();
    expect(payload.sourceAudioAssetId).toBeUndefined();
    expect(payload.bpm).toBeUndefined();
    // The run lands in the audio local-job lane so it stacks in the results zone.
    expect(rememberLocalGenerationJob).toHaveBeenCalledWith("audio", job);
  });

  it("omits matchStrength when cleared so the converter uses its own default", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ referenceAudioAssetId: "ref-voice-1" }),
    );
    const referenceAsset = { id: "ref-voice-1", type: "audio", displayName: "Ref" };
    const createAudioJob = vi.fn(async () => ({ id: "vc-default" }));
    await render(
      baseContext({ assets: [referenceAsset], createAudioJob, rememberLocalGenerationJob: vi.fn() }),
    );
    await click(modeTab(container, "Voice Clone"));
    await setTextarea(container.querySelector(".prompt-input"), "default strength");
    await submitForm();

    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.referenceAudioAssetId).toBe("ref-voice-1");
    expect(payload.matchStrength).toBeUndefined();
  });

  it("prefers the native clone-TTS generator and hides match strength when one is installed (sc-13412)", async () => {
    // With a native clone-TTS generator (ReferenceAudio + VoiceEmbedding) in the catalog, the Voice
    // Clone picker default snaps to it — a single-call clone — over the OpenVoice converter, and the
    // OpenVoice-only match-strength (τ) control is hidden (the native generator has no such knob).
    await render(
      baseContext({
        audioModels: [...ALL_AUDIO, CHATTERBOX_TTS],
        models: [...ALL_AUDIO, CHATTERBOX_TTS],
      }),
    );
    await click(modeTab(container, "Voice Clone"));

    expect(modelSelect(container).value).toBe("chatterbox_tts");
    const band = container.querySelector(".studio-source-band");
    expect(band).toBeTruthy();
    expect(band.textContent).toContain("Reference voice");
    expect(
      band.querySelector(".settings-field-match-strength"),
      "native clone has no OpenVoice τ, so match strength is hidden",
    ).toBeNull();
    // Both reference-consuming clone models are offered; the bare embedder still isn't.
    const options = [...modelSelect(container).querySelectorAll("option")].map((o) => o.value);
    expect(options).toContain("chatterbox_tts");
    expect(options).toContain("openvoice_v2");
  });

  it("submits the native clone in one step: model=chatterbox_tts + reference + script, no matchStrength (sc-13412)", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ referenceAudioAssetId: "ref-voice-1" }),
    );
    const referenceAsset = { id: "ref-voice-1", type: "audio", displayName: "My reference voice" };
    const job = { id: "native-clone-1", type: "audio_generate" };
    const createAudioJob = vi.fn(async () => job);
    const rememberLocalGenerationJob = vi.fn();
    await render(
      baseContext({
        assets: [referenceAsset],
        audioModels: [...ALL_AUDIO, CHATTERBOX_TTS],
        models: [...ALL_AUDIO, CHATTERBOX_TTS],
        createAudioJob,
        rememberLocalGenerationJob,
      }),
    );
    await click(modeTab(container, "Voice Clone"));
    expect(modelSelect(container).value).toBe("chatterbox_tts");
    await setTextarea(container.querySelector(".prompt-input"), "Render this in the cloned voice.");
    expect(generateButton(container).disabled).toBe(false);
    await submitForm();

    expect(createAudioJob).toHaveBeenCalledTimes(1);
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.model).toBe("chatterbox_tts");
    expect(payload.referenceAudioAssetId).toBe("ref-voice-1");
    expect(payload.prompt).toBe("Render this in the cloned voice.");
    // The native single-call clone has no OpenVoice τ — matchStrength is never sent, and no base voice.
    expect(payload.matchStrength).toBeUndefined();
    expect(payload.voice).toBeUndefined();
    expect(rememberLocalGenerationJob).toHaveBeenCalledWith("audio", job);
  });
});

describe("AudioStudio register-a-voice (sc-13517)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const setInput = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        "value",
      ).set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const nameInput = () => container.querySelector(".register-voice-row input");
  const saveButton = () => buttonWithText(container, "Save voice");

  // A reference clip is a persisted selection, restored from the studio snapshot at mount.
  const seedReference = () =>
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ referenceAudioAssetId: "ref-voice-1" }),
    );
  const referenceAsset = { id: "ref-voice-1", type: "audio", displayName: "My reference voice" };

  it("surfaces saved voices in the Voice Clone tab and selecting one sets the reference", async () => {
    const savedVoices = [
      { id: "voice_a", name: "Narrator", referenceAudioAssetId: "ref-voice-1" },
      { id: "voice_b", name: "Villain", referenceAudioAssetId: "ref-voice-2" },
    ];
    await render(baseContext({ assets: [referenceAsset], savedVoices, createSavedVoice: vi.fn() }));
    await click(modeTab(container, "Voice Clone"));

    const chips = [...container.querySelectorAll(".saved-voice-chip")];
    expect(chips.map((c) => c.querySelector(".saved-voice-select").textContent.trim())).toEqual([
      "Narrator",
      "Villain",
    ]);
    // None selected initially (no reference restored in this render).
    const narratorSelect = chips[0].querySelector(".saved-voice-select");
    await click(narratorSelect);
    expect(narratorSelect.getAttribute("aria-pressed")).toBe("true");
  });

  it("registers a voice and shows a near-duplicate WARNING when the backend flags one", async () => {
    seedReference();
    const createSavedVoice = vi.fn(async ({ name }) => ({
      id: "voice_new",
      name,
      referenceAudioAssetId: "ref-voice-1",
      nearDuplicate: { id: "voice_a", name: "Narrator", similarity: 0.97 },
    }));
    await render(baseContext({ assets: [referenceAsset], savedVoices: [], createSavedVoice }));
    await click(modeTab(container, "Voice Clone"));

    // Save is disabled until a name is typed.
    expect(saveButton().disabled).toBe(true);
    await setInput(nameInput(), "Narrator 2");
    expect(saveButton().disabled).toBe(false);
    await click(saveButton());

    expect(createSavedVoice).toHaveBeenCalledWith({
      name: "Narrator 2",
      referenceAudioAssetId: "ref-voice-1",
    });
    const warning = container.querySelector(".inline-warning");
    expect(warning).not.toBeNull();
    expect(warning.textContent).toContain("Narrator");
    expect(warning.textContent).toContain("97%");
  });

  it("registers a distinct voice with NO warning (info notice only)", async () => {
    seedReference();
    const createSavedVoice = vi.fn(async ({ name }) => ({
      id: "voice_new",
      name,
      referenceAudioAssetId: "ref-voice-1",
      nearDuplicate: null,
    }));
    await render(baseContext({ assets: [referenceAsset], savedVoices: [], createSavedVoice }));
    await click(modeTab(container, "Voice Clone"));
    await setInput(nameInput(), "Brand New Voice");
    await click(saveButton());

    expect(createSavedVoice).toHaveBeenCalledOnce();
    expect(container.querySelector(".inline-warning")).toBeNull();
    const notice = container.querySelector(".register-voice p[role='status']");
    expect(notice.textContent).toContain("Brand New Voice");
  });

  it("confirms before deleting a saved voice", async () => {
    const savedVoices = [{ id: "voice_a", name: "Narrator", referenceAudioAssetId: "ref-voice-1" }];
    const deleteSavedVoice = vi.fn(async () => ({ id: "voice_a", status: "deleted" }));
    await render(
      baseContext({
        assets: [referenceAsset],
        savedVoices,
        createSavedVoice: vi.fn(),
        deleteSavedVoice,
      }),
    );
    await click(modeTab(container, "Voice Clone"));
    await click(container.querySelector(".saved-voice-delete"));
    expect(appConfirmMock).toHaveBeenCalledWith(
      expect.objectContaining({ tone: "danger", confirmLabel: "Delete permanently" }),
    );
    expect(deleteSavedVoice).toHaveBeenCalledWith("voice_a");
  });

  it("keeps a saved voice when permanent deletion is canceled", async () => {
    appConfirmMock.mockResolvedValueOnce(false);
    const savedVoices = [{ id: "voice_a", name: "Narrator", referenceAudioAssetId: "ref-voice-1" }];
    const deleteSavedVoice = vi.fn();
    await render(
      baseContext({
        assets: [referenceAsset],
        savedVoices,
        createSavedVoice: vi.fn(),
        deleteSavedVoice,
      }),
    );
    await click(modeTab(container, "Voice Clone"));
    await click(container.querySelector(".saved-voice-delete"));

    expect(appConfirmMock).toHaveBeenCalledOnce();
    expect(deleteSavedVoice).not.toHaveBeenCalled();
  });

  it("guards a saved-voice delete while the mutation is in flight", async () => {
    const deletion = deferred();
    const savedVoices = [{ id: "voice_a", name: "Narrator", referenceAudioAssetId: "ref-voice-1" }];
    const deleteSavedVoice = vi.fn(() => deletion.promise);
    await render(
      baseContext({
        assets: [referenceAsset],
        savedVoices,
        createSavedVoice: vi.fn(),
        deleteSavedVoice,
      }),
    );
    await click(modeTab(container, "Voice Clone"));
    const deleteButton = container.querySelector(".saved-voice-delete");
    await click(deleteButton);

    expect(deleteSavedVoice).toHaveBeenCalledOnce();
    expect(deleteButton.disabled).toBe(true);
    deleteButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    expect(deleteSavedVoice).toHaveBeenCalledOnce();

    await act(async () => deletion.resolve({ id: "voice_a", status: "deleted" }));
  });
});

describe("AudioStudio streaming reveal (sc-13675)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const streamingBadge = () => container.querySelector('[data-testid="audio-streaming-badge"]');

  it("serves the Speech tab from supportsStreaming (no voice bank) and reveals the streaming badge", async () => {
    await render(baseContext({ audioModels: [MOSS_TTS_REALTIME], models: [MOSS_TTS_REALTIME] }));

    // Opens on Speech, with the streaming TTS selected — proving supportsStreaming serves Speech even
    // without a voice bank (capability-driven eligibility, sc-13675).
    expect(modeTab(container, "Speech").className).toContain("active");
    expect(modelSelect(container).value).toBe("moss_tts_realtime");

    // The results zone reveals the streaming affordance.
    expect(streamingBadge()).toBeTruthy();
    expect(streamingBadge().textContent).toContain("Streams incrementally");

    // It has NO voice bank, so the Speech voice picker is absent — but language + length still render
    // from its Capabilities (it is a real Speech model, not SFX).
    expect(fieldByLabelStart(container, "Voice")).toBeFalsy();
    expect(fieldByLabelStart(container, "Language").querySelector("select")).toBeTruthy();
    expect(fieldByLabelStart(container, "Length").querySelector("input").getAttribute("max")).toBe(
      "2400",
    );
  });

  it("does NOT reveal the streaming badge for a one-shot Speech model (Kokoro)", async () => {
    await render(baseContext({ audioModels: [KOKORO], models: [KOKORO] }));
    expect(modelSelect(container).value).toBe("kokoro_82m");
    // Kokoro does not advertise supportsStreaming → no streaming affordance (non-streaming unperturbed).
    expect(streamingBadge()).toBeNull();
  });

  it("toggles the badge with the selected model's capability, not the mode", async () => {
    // Both a streaming and a one-shot Speech model installed; the badge follows the SELECTED model.
    await render(
      baseContext({
        audioModels: [MOSS_TTS_REALTIME, KOKORO],
        models: [MOSS_TTS_REALTIME, KOKORO],
      }),
    );
    // Default selection is the first Speech model (the streaming one) → badge shown.
    expect(modelSelect(container).value).toBe("moss_tts_realtime");
    expect(streamingBadge()).toBeTruthy();

    // Switch to Kokoro (still Speech) → the badge disappears: capability-driven, not mode-driven.
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLSelectElement.prototype,
        "value",
      ).set;
      setter.call(modelSelect(container), "kokoro_82m");
      modelSelect(container).dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    expect(modelSelect(container).value).toBe("kokoro_82m");
    expect(streamingBadge()).toBeNull();
  });
});

describe("AudioStudio multi-speaker reveal (sc-13676)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const generateButton = (root) => buttonWithText(root, "Generate");
  const scriptEditor = () => container.querySelector('[data-testid="multi-speaker-script"]');
  const segmentTextareas = () => [...container.querySelectorAll(".script-segment-text")];
  const speakerSelects = () => [...container.querySelectorAll(".script-segment-speaker")];
  const setTextarea = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const setSelect = async (el, value) => {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value").set;
      setter.call(el, value);
      el.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
  };

  it("serves the Speech tab from supportsMultiSpeaker (no voice bank) and reveals the segmented-script editor", async () => {
    await render(baseContext({ audioModels: [MOSS_TTSD], models: [MOSS_TTSD] }));

    // Opens on Speech, with the multi-speaker model selected — proving supportsMultiSpeaker serves
    // Speech even without a voice bank (capability-driven eligibility, sc-13676).
    expect(modeTab(container, "Speech").className).toContain("active");
    expect(modelSelect(container).value).toBe("moss_ttsd_v05");

    // The segmented-script editor replaces the plain prompt textarea; a starter dialogue seeds one
    // turn per advertised speaker (maxSpeakers = 2), and each speaker select offers exactly 2 labels
    // (read off the model, never hardcoded).
    expect(scriptEditor()).toBeTruthy();
    expect(container.querySelector(".prompt-input:not(.multi-speaker-script)")).toBeNull();
    expect(segmentTextareas().length).toBe(2);
    expect([...speakerSelects()[0].querySelectorAll("option")].map((o) => o.value)).toEqual(["S1", "S2"]);

    // A voice picker never appears (multi-speaker models ship no fixed voice bank).
    expect(fieldByLabelStart(container, "Voice")).toBeFalsy();
  });

  it("does NOT reveal the script editor for a single-voice Speech model (Kokoro)", async () => {
    await render(baseContext({ audioModels: [KOKORO], models: [KOKORO] }));
    expect(modelSelect(container).value).toBe("kokoro_82m");
    // Kokoro is single-voice → the plain prompt textarea, no script editor (single-voice unperturbed).
    expect(scriptEditor()).toBeNull();
    expect(container.querySelector(".prompt-input")).toBeTruthy();
  });

  it("Generate is disabled until a script turn has text, then submits AudioParams.script", async () => {
    const job = { id: "audio-ms-1", type: "audio_generate", status: "queued" };
    const createAudioJob = vi.fn(async () => job);
    const rememberLocalGenerationJob = vi.fn();
    await render(
      baseContext({
        audioModels: [MOSS_TTSD],
        models: [MOSS_TTSD],
        createAudioJob,
        rememberLocalGenerationJob,
      }),
    );

    // Empty starter script → the guard disables the CTA (never a silent no-op).
    expect(generateButton(container).disabled).toBe(true);

    // Fill both turns and assign the second to Speaker 2.
    await setTextarea(segmentTextareas()[0], "Hello, how are you today?");
    await setTextarea(segmentTextareas()[1], "I'm doing great, thanks for asking!");
    await setSelect(speakerSelects()[1], "S2");
    expect(generateButton(container).disabled).toBe(false);

    await act(async () => {
      generateButton(container).dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });

    expect(createAudioJob).toHaveBeenCalledTimes(1);
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.model).toBe("moss_ttsd_v05");
    // The segmented dialogue submits as AudioParams.script — both turns, in order, with speakers.
    expect(payload.script).toEqual([
      { text: "Hello, how are you today?", speaker: "S1" },
      { text: "I'm doing great, thanks for asking!", speaker: "S2" },
    ]);
    expect(payload.prompt).toBe("");
    // A multi-speaker model ships no voice bank, so no `voice` is sent.
    expect(payload.voice).toBeUndefined();
    expect(rememberLocalGenerationJob).toHaveBeenCalledWith("audio", job);
  });

  it("adds and removes script turns (Add turn / remove), capped-speaker labels only", async () => {
    await render(baseContext({ audioModels: [MOSS_TTSD], models: [MOSS_TTSD] }));
    expect(segmentTextareas().length).toBe(2);

    // Add a third turn — dialogue length is unbounded (maxSpeakers caps DISTINCT speakers, not turns).
    await act(async () => {
      buttonWithText(container, "Add turn").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(segmentTextareas().length).toBe(3);
    // Every speaker select still offers only the 2 advertised labels.
    for (const select of speakerSelects()) {
      expect([...select.querySelectorAll("option")].map((o) => o.value)).toEqual(["S1", "S2"]);
    }

    // Remove one turn back to 2.
    await act(async () => {
      container
        .querySelector(".script-segment-remove:not([disabled])")
        .dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(segmentTextareas().length).toBe(2);
  });
});

describe("Audio nav registration (sc-13407)", () => {
  it("registers Audio in KEEP_ALIVE_VIEWS", () => {
    expect(KEEP_ALIVE_VIEWS.has("Audio")).toBe(true);
  });

  it("registers an Audio view title + blurb", () => {
    expect(viewTitles.Audio).toBeTruthy();
    expect(viewTitles.Audio.title).toBe("Audio Studio");
    expect(typeof viewTitles.Audio.blurb).toBe("string");
    expect(viewTitles.Audio.blurb.length).toBeGreaterThan(0);
  });

  it("registers Audio in the Workspace nav section with an icon", () => {
    const workspace = navSections.find((section) => section.label === "Workspace");
    expect(workspace).toBeTruthy();
    const audioItem = workspace.items.find((item) => item.id === "Audio");
    expect(audioItem).toBeTruthy();
    expect(audioItem.icon).toBeTruthy();
  });
});

// YuE lyrics-to-song fixtures (epic sc-19373, sc-19385) — mirror the seeded catalog entries: a
// segmented-lyrics music model with no edit surface; the ICL variant adds ReferenceAudio
// conditioning + a reference window. Tiers ride the /models variant matrix (bf16/q8/q4 downloads).
const yueVariants = (installed) =>
  ["q4", "q8", "bf16"].map((variant) => ({
    variant,
    installState: installed.includes(variant) ? "installed" : "missing",
  }));
const YUE_COT = {
  id: "yue_en_cot",
  name: "YuE English CoT (Lyrics-to-Song)",
  type: "audio",
  hasVariantMatrix: true,
  variants: yueVariants(["q4", "q8"]),
  audio: {
    languages: ["en"],
    sampleRates: [44100],
    supportsMultiSpeaker: false,
    supportsGuidance: true,
    supportsNegativePrompt: false,
    supportsSegmentedLyrics: true,
    supportsRepetitionPenalty: true,
    supportsReferenceRegion: false,
    supportsOutputLimiter: true,
  },
  ui: {
    label: "YuE English CoT",
    promptGuide: { title: "YuE Lyrics-to-Song Guide", path: "/prompt-guides/yue.md" },
  },
};
const YUE_ICL = {
  ...YUE_COT,
  id: "yue_en_icl",
  name: "YuE English ICL (Lyrics-to-Song)",
  audio: { ...YUE_COT.audio, conditioning: ["ReferenceAudio"], supportsReferenceRegion: true },
  ui: { ...YUE_COT.ui, label: "YuE English ICL" },
};

describe("AudioStudio YuE lyrics-to-song (sc-19385)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <AudioStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const withYue = (overrides = {}) =>
    baseContext({
      audioModels: [...ALL_AUDIO, YUE_COT, YUE_ICL],
      models: [...ALL_AUDIO, YUE_COT, YUE_ICL],
      createAudioJob: vi.fn(async () => ({ id: "yue-job" })),
      rememberLocalGenerationJob: vi.fn(),
      ...overrides,
    });
  const setValue = async (el, value) => {
    const proto =
      el.tagName === "TEXTAREA"
        ? window.HTMLTextAreaElement
        : el.tagName === "SELECT"
          ? window.HTMLSelectElement
          : window.HTMLInputElement;
    await act(async () => {
      Object.getOwnPropertyDescriptor(proto.prototype, "value").set.call(el, value);
      el.dispatchEvent(
        new window.Event(el.tagName === "SELECT" ? "change" : "input", { bubbles: true }),
      );
    });
  };
  const pressEnter = async (el) => {
    await act(async () => {
      el.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    });
  };
  const selectModel = async (id) => setValue(modelSelect(container), id);
  const submitForm = async () => {
    await act(async () => {
      container
        .querySelector("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
  };
  const caption = (label) => label.childNodes[0].textContent.trim();
  const settingsCaptions = () => [...container.querySelectorAll(".settings-bar label")].map(caption);
  const advancedCaptions = () => [...container.querySelectorAll(".advanced-panel label")].map(caption);
  const advancedField = (text) =>
    [...container.querySelectorAll(".advanced-panel label")].find((label) =>
      label.textContent.trim().startsWith(text),
    );
  const generateButton = () => buttonWithText(container, "Generate");
  const yueBand = () => container.querySelector('[data-testid="yue-reference-band"]');

  // Fill the minimum a YuE render needs: one lyric section + one genre tag.
  async function fillSong() {
    await setValue(
      container.querySelector('[aria-label="Section 1 lyrics"]'),
      "Staring at the sunset\ncolors paint the sky",
    );
    const tagInput = container.querySelector('[aria-label="Add genre tags"]');
    await setValue(tagInput, "uplifting, female");
    await pressEnter(tagInput);
  }

  it("AC1: a YuE model swaps in the section-labelled lyrics editor, genre tags and song knobs (Advanced collapsed)", async () => {
    await render(withYue());
    await click(modeTab(container, "Music"));
    await selectModel("yue_en_cot");

    // Section-labelled lyrics editor replaces the prompt; starter song = [verse] + [chorus].
    const editor = container.querySelector('[data-testid="yue-lyrics-editor"]');
    expect(editor).toBeTruthy();
    expect(container.querySelector('textarea[aria-label="Prompt"]')).toBeNull();
    const labelSelects = [...editor.querySelectorAll("select")];
    expect(labelSelects.map((select) => select.value)).toEqual(["verse", "chorus"]);
    expect([...labelSelects[0].options].map((option) => option.textContent)).toEqual([
      "[intro]",
      "[verse]",
      "[chorus]",
      "[bridge]",
      "[outro]",
    ]);
    await click(container.querySelector('[data-testid="yue-add-section"]'));
    expect(editor.querySelectorAll(".script-segment").length).toBe(3);

    // Genre tags: free text + upstream's top-200 list as suggestions, the browser collapsed.
    const tags = container.querySelector('[data-testid="yue-genre-tags"]');
    expect(tags).toBeTruthy();
    const datalist = container.querySelector("#yue-genre-tag-suggestions");
    const suggestions = [...datalist.querySelectorAll("option")].map((option) => option.value);
    expect(suggestions).toContain("Pop");
    expect(suggestions).toContain("airy vocal");
    // Upstream's case-variant duplicates ("Pop" / "pop") are folded.
    expect(suggestions).not.toContain("pop");
    expect(tags.querySelector("details").open).toBe(false);

    // Settings bar: ACE-Step's BPM / Key / Length / lyrics are gone; Sections + tier are in.
    expect(settingsCaptions()).toEqual(["Model", "Language", "Sections", "Quant tier"]);
    expect(container.querySelector(".settings-field-lyrics")).toBeNull();
    expect(container.querySelector(".studio-source-band:not(.yue-reference-band)")).toBeNull();
    const tierOptions = [...container.querySelector(".settings-field-tier select").options];
    expect(tierOptions.map((option) => [option.value, option.disabled])).toEqual([
      ["q4", false],
      ["q8", false],
      ["bf16", true],
    ]);

    // Advanced starts collapsed; opened it carries the R5 knobs (and no ACE-Step Steps).
    expect(container.querySelector(".advanced-panel")).toBeNull();
    await click(container.querySelector(".advanced-section-toggle"));
    expect(advancedCaptions()).toEqual([
      "Seed",
      "",
      "Guidance scale",
      "Max tokens per section",
      "Repetition penalty",
      "Output limiter",
      "Sample rate",
    ]);
    expect(advancedField("Guidance (CFG)").querySelector('input[type="checkbox"]').checked).toBe(true);
    // The engine's stage-1 context is 16384 positions and keeps `16384 - budget - 1` for the prompt,
    // so 16382 is the largest per-section budget it (and the API) accepts.
    expect(advancedField("Max tokens per section").querySelector("input").getAttribute("max")).toBe(
      "16382",
    );
  });

  it("AC1: submits the lyrics, genre tags and every R5 knob in the sc-19384 payload shape", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "yue-job" }));
    await render(withYue({ createAudioJob }));
    await click(modeTab(container, "Music"));
    await selectModel("yue_en_cot");
    expect(generateButton().disabled).toBe(true);

    await setValue(container.querySelector('[aria-label="Section 1 lyrics"]'), "Staring at the sunset");
    await setValue(container.querySelector('[aria-label="Section 2 label"]'), "bridge");
    await setValue(container.querySelector('[aria-label="Section 2 lyrics"]'), "  Every road you take  ");
    // Lyrics alone are not enough — the genre tags are the prompt.
    expect(generateButton().disabled).toBe(true);
    const tagInput = container.querySelector('[aria-label="Add genre tags"]');
    await setValue(tagInput, "inspiring, bright vocal");
    await pressEnter(tagInput);
    // A suggestion chip adds a tag too.
    await click(
      buttonWithText(container.querySelector('[data-testid="yue-tag-suggestions"]'), "Pop"),
    );
    expect(generateButton().disabled).toBe(false);

    await setValue(container.querySelector(".settings-field-segments input"), "3");
    await setValue(container.querySelector(".settings-field-tier select"), "q8");
    await click(container.querySelector(".advanced-section-toggle"));
    await setValue(advancedField("Seed").querySelector("input"), "11");
    await setValue(advancedField("Guidance scale").querySelector("input"), "1.75");
    await setValue(advancedField("Max tokens per section").querySelector("input"), "1500");
    await setValue(advancedField("Repetition penalty").querySelector("input"), "1.25");
    await setValue(advancedField("Output limiter").querySelector("select"), "rescale");

    await submitForm();
    expect(createAudioJob).toHaveBeenCalledTimes(1);
    expect(createAudioJob.mock.calls[0][0]).toEqual({
      model: "yue_en_cot",
      prompt: "inspiring bright vocal Pop",
      lyrics: "[verse]\nStaring at the sunset\n\n[bridge]\nEvery road you take",
      language: "en",
      targetDurationSecs: undefined,
      seed: 11,
      segments: 3,
      maxNewTokensPerSegment: 1500,
      repetitionPenalty: 1.25,
      guidanceEnabled: true,
      guidance: 1.75,
      quantTier: "q8",
      outputLimiter: "rescale",
    });
  });

  it("AC1: guidance off sends guidanceEnabled:false with no scale; cleared knobs are omitted", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "yue-job" }));
    await render(withYue({ createAudioJob }));
    await click(modeTab(container, "Music"));
    await selectModel("yue_en_cot");
    await fillSong();
    await click(container.querySelector(".advanced-section-toggle"));
    await setValue(advancedField("Guidance scale").querySelector("input"), "2");
    await click(advancedField("Guidance (CFG)").querySelector("input"));
    expect(advancedField("Guidance scale").querySelector("input").disabled).toBe(true);
    await submitForm();
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.prompt).toBe("uplifting female");
    expect(payload.guidanceEnabled).toBe(false);
    expect(payload.guidance).toBeUndefined();
    expect(payload.segments).toBeUndefined();
    expect(payload.maxNewTokensPerSegment).toBeUndefined();
    expect(payload.repetitionPenalty).toBeUndefined();
    expect(payload.outputLimiter).toBeUndefined();
    // No ACE-Step knob ever reaches a YuE job (the API refuses them).
    for (const key of ["bpm", "musicalKey", "steps", "editMode", "sourceAudioAssetId"]) {
      expect(payload[key]).toBeUndefined();
    }
  });

  it("AC1: the output limiter is gated on audio.supportsOutputLimiter", async () => {
    const noLimiter = { ...YUE_COT, audio: { ...YUE_COT.audio, supportsOutputLimiter: undefined } };
    await render(withYue({ audioModels: [ACESTEP, noLimiter], models: [ACESTEP, noLimiter] }));
    await click(modeTab(container, "Music"));
    await selectModel("yue_en_cot");
    await click(container.querySelector(".advanced-section-toggle"));
    expect(advancedField("Repetition penalty")).toBeTruthy();
    expect(advancedField("Output limiter")).toBeFalsy();
  });

  it("AC1: ACE-Step's Music controls are unchanged when YuE models are installed", async () => {
    // Explicit field list, identical with and without YuE in the catalog.
    const aceControls = async (context) => {
      await render(context);
      await click(modeTab(container, "Music"));
      await selectModel("acestep_v15_turbo");
      const bar = settingsCaptions();
      const prompt = Boolean(container.querySelector('textarea[aria-label="Prompt"]'));
      const band = container.querySelector(".studio-source-band")?.textContent.includes("Source track");
      const yue = [
        '[data-testid="yue-lyrics-editor"]',
        '[data-testid="yue-genre-tags"]',
        '[data-testid="yue-reference-band"]',
        ".settings-field-segments",
        ".settings-field-tier",
      ].some((selector) => container.querySelector(selector));
      await click(container.querySelector(".advanced-section-toggle"));
      const advanced = advancedCaptions();
      await act(async () => root.render(null));
      window.localStorage.clear();
      return { bar, prompt, band, yue, advanced };
    };
    const expected = {
      bar: ["Model", "Language", "Length (s)", "BPM", "Key", "Lyrics"],
      prompt: true,
      band: true,
      yue: false,
      advanced: ["Seed", "Steps", "Sample rate"],
    };
    expect(await aceControls(baseContext({ createAudioJob: vi.fn() }))).toEqual(expected);
    expect(await aceControls(withYue())).toEqual(expected);
  });

  it("AC1: an ACE-Step submit carries none of the YuE keys", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "ace-job" }));
    await render(withYue({ createAudioJob }));
    await click(modeTab(container, "Music"));
    await selectModel("acestep_v15_turbo");
    await setValue(container.querySelector('textarea[aria-label="Prompt"]'), "lofi piano");
    await submitForm();
    expect(Object.keys(createAudioJob.mock.calls[0][0]).sort()).toEqual(
      [
        "bpm",
        "language",
        "lyrics",
        "model",
        "musicalKey",
        "prompt",
        "seed",
        "steps",
        "targetDurationSecs",
      ].sort(),
    );
  });

  it("AC2: the ICL reference controls are disabled for a CoT model and never sent", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ iclReferenceAssetId: "ref-song", iclStartSecs: "5" }),
    );
    const createAudioJob = vi.fn(async () => ({ id: "yue-job" }));
    await render(
      withYue({ createAudioJob, assets: [{ id: "ref-song", type: "audio", displayName: "Ref" }] }),
    );
    await click(modeTab(container, "Music"));
    await selectModel("yue_en_cot");
    const band = yueBand();
    expect(band.disabled).toBe(true);
    const controls = [...band.querySelectorAll("button, input")];
    expect(controls.length).toBeGreaterThan(0);
    // A disabled <fieldset> disables every descendant control (buttons + the window inputs).
    expect(controls.every((control) => control.matches(":disabled"))).toBe(true);
    expect(band.querySelector(".settings-field-icl-start input")).toBeTruthy();
    await fillSong();
    await submitForm();
    const payload = createAudioJob.mock.calls[0][0];
    for (const key of [
      "iclMode",
      "iclReferenceAssetId",
      "iclVocalAssetId",
      "iclInstrumentalAssetId",
      "iclStartSecs",
      "iclEndSecs",
    ]) {
      expect(payload[key]).toBeUndefined();
    }
  });

  it("AC2: an ICL model enables the band, runs plain with no reference and sends single-track ICL", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "yue-job" }));
    await render(withYue({ createAudioJob }));
    await click(modeTab(container, "Music"));
    await selectModel("yue_en_icl");
    expect(yueBand().disabled).toBe(false);
    await fillSong();
    // No reference song picked → an `_icl` checkpoint runs as a plain prompt run (sc-19384 R1):
    // Generate is live and no ICL field is sent.
    expect(generateButton().disabled).toBe(false);
    await submitForm();
    expect(createAudioJob.mock.calls[0][0].iclMode).toBeUndefined();
    expect(createAudioJob.mock.calls[0][0].iclStartSecs).toBeUndefined();
    createAudioJob.mockClear();
    await act(async () => root.render(null));

    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({
        mode: "music",
        model: "yue_en_icl",
        iclReferenceAssetId: "ref-song",
        iclStartSecs: "5",
        iclEndSecs: "25",
      }),
    );
    await render(
      withYue({ createAudioJob, assets: [{ id: "ref-song", type: "audio", displayName: "Ref" }] }),
    );
    expect(modelSelect(container).value).toBe("yue_en_icl");
    await fillSong();
    expect(generateButton().disabled).toBe(false);
    await submitForm();
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.iclMode).toBe("single");
    expect(payload.iclReferenceAssetId).toBe("ref-song");
    expect(payload.iclVocalAssetId).toBeUndefined();
    expect(payload.iclStartSecs).toBe(5);
    expect(payload.iclEndSecs).toBe(25);
  });

  it("AC2: dual-track ICL needs BOTH the vocal and instrumental tracks", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ mode: "music", model: "yue_en_icl", iclVocalAssetId: "vox" }),
    );
    const createAudioJob = vi.fn(async () => ({ id: "yue-job" }));
    await render(withYue({ createAudioJob }));
    await fillSong();
    await click(buttonWithText(yueBand(), "Vocal + instrumental"));
    expect(yueBand().textContent).toContain("Instrumental track");
    expect(generateButton().disabled).toBe(true);
    await act(async () => root.render(null));

    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({
        mode: "music",
        model: "yue_en_icl",
        iclMode: "dual",
        iclVocalAssetId: "vox",
        iclInstrumentalAssetId: "inst",
      }),
    );
    await render(withYue({ createAudioJob }));
    await fillSong();
    expect(generateButton().disabled).toBe(false);
    await submitForm();
    const payload = createAudioJob.mock.calls[0][0];
    expect(payload.iclMode).toBe("dual");
    expect(payload.iclVocalAssetId).toBe("vox");
    expect(payload.iclInstrumentalAssetId).toBe("inst");
    expect(payload.iclReferenceAssetId).toBeUndefined();
  });

  it("AC1: a refined space-separated tag line splits into individual tag chips", async () => {
    const fetchGuide = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      text: async () => "# YuE guide",
    });
    const refinePrompt = vi.fn(async () => "uplifting female airy vocal");
    await render(
      withYue({
        refinePrompt,
        models: [
          ...ALL_AUDIO,
          YUE_COT,
          YUE_ICL,
          { id: PROMPT_REFINE_MODEL_ID, name: "Prompt Refiner", installState: "installed" },
        ],
      }),
    );
    await click(modeTab(container, "Music"));
    await selectModel("yue_en_cot");
    const tagInput = container.querySelector('[aria-label="Add genre tags"]');
    await setValue(tagInput, "pop");
    await pressEnter(tagInput);
    await click(buttonWithText(container, "Refine my prompt"));
    await settle();
    expect(refinePrompt).toHaveBeenCalledWith(expect.objectContaining({ prompt: "pop" }));
    await click(buttonWithText(container, "Apply"));
    const chips = [
      ...container.querySelectorAll('[data-testid="yue-genre-tags"] .yue-genre-tags__chosen .preset-chip > span'),
    ].map((chip) => chip.textContent);
    expect(chips).toEqual(["uplifting", "female", "airy vocal"]);
    // Each is its own removable tag, and the suggestion browser shows it selected.
    expect(container.querySelector('[aria-label="Remove tag female"]')).toBeTruthy();
    await click(buttonWithText(container.querySelector('[aria-label="Tag category"]'), "Vocal timbre"));
    expect(
      buttonWithText(container.querySelector('[data-testid="yue-tag-suggestions"]'), "airy vocal").getAttribute(
        "aria-pressed",
      ),
    ).toBe("true");
    fetchGuide.mockRestore();
  });

  it("AC1: the tag-category picker is a pressed-button group, not a partial ARIA tab pattern", async () => {
    await render(withYue());
    await click(modeTab(container, "Music"));
    await selectModel("yue_en_cot");
    const tags = container.querySelector('[data-testid="yue-genre-tags"]');
    expect(tags.querySelector('[role="tab"], [role="tablist"]')).toBeNull();
    const group = tags.querySelector('[role="group"][aria-label="Tag category"]');
    const buttons = [...group.querySelectorAll("button")];
    expect(buttons.length).toBe(5);
    expect(buttons.map((button) => button.getAttribute("aria-pressed"))).toEqual([
      "true",
      "false",
      "false",
      "false",
      "false",
    ]);
  });

  it("AC2: an ICL start at or past the effective end (empty end ⇒ 30 s) blocks Generate with a hint", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-audio-project_1",
      JSON.stringify({ mode: "music", model: "yue_en_icl", iclReferenceAssetId: "ref-song", iclStartSecs: "31" }),
    );
    await render(withYue({ assets: [{ id: "ref-song", type: "audio", displayName: "Ref" }] }));
    await fillSong();
    expect(generateButton().disabled).toBe(true);
    expect(container.querySelector('[data-testid="yue-icl-window-error"]').textContent).toContain("30 s");
    // An explicit end past the start clears it.
    await setValue(container.querySelector(".settings-field-icl-end input"), "40");
    expect(generateButton().disabled).toBe(false);
    expect(container.querySelector('[data-testid="yue-icl-window-error"]')).toBeNull();
  });

  it("AC3: the result card plays the mix and offers a download for each stem", async () => {
    const mix = {
      id: "yue-mix",
      projectId: "project_1",
      type: "audio",
      displayName: "Song",
      file: { path: "assets/audios/genset_y/yue_mix.wav", mimeType: "audio/wav", duration: 60 },
      extra: { audioStem: "mix", stemAssetIds: [{ stem: "vocals", assetId: "yue-vocals" }] },
    };
    const stem = (name) => ({
      id: `yue-${name}`,
      projectId: "project_1",
      type: "audio",
      displayName: `Song (${name})`,
      file: { path: `assets/audios/genset_y/yue_${name}.wav`, mimeType: "audio/wav", duration: 60 },
      extra: { audioStem: name, mixAssetId: "yue-mix" },
    });
    const job = {
      id: "yue-done",
      type: "audio_generate",
      status: "completed",
      createdAt: "2026-09-24T12:00:00Z",
      payload: { model: "yue_en_cot", prompt: "uplifting pop", lyrics: "[verse]\nla" },
      result: { assetIds: ["yue-mix", "yue-vocals", "yue-instrumental"] },
    };
    await render(
      withYue({
        assets: [mix, stem("vocals"), stem("instrumental")],
        // The stems also arrive in the project's recent clips — they must not re-list as takes.
        recentAudioAssets: [stem("instrumental"), stem("vocals"), mix],
        audioLocalJobs: [job],
      }),
    );
    const results = container.querySelector(".studio-results");
    const cards = [...results.querySelectorAll('[data-testid="audio-take-card"]')];
    expect(cards.length).toBe(1);
    const card = cards[0];
    const vocals = card.querySelector('[aria-label="Download vocals stem"]');
    const instrumental = card.querySelector('[aria-label="Download instrumental stem"]');
    expect(vocals.textContent).toContain("Vocals");
    expect(instrumental.textContent).toContain("Instrumental");
    // Each stem button downloads its OWN asset (the hidden anchor beside it).
    expect(vocals.previousElementSibling.getAttribute("href")).toContain("yue_vocals.wav");
    expect(instrumental.previousElementSibling.getAttribute("href")).toContain(
      "yue_instrumental.wav",
    );

    // Play loads the MIX; the deck offers the stems too.
    await click(card.querySelector('[aria-label="Play take 1"]'));
    const deck = results.querySelector('[data-testid="audio-play-deck"]');
    expect(deck.querySelector("audio").getAttribute("src")).toContain("yue_mix.wav");
    expect(deck.querySelector('[aria-label="Download vocals stem"]')).toBeTruthy();
    expect(deck.querySelector('[aria-label="Download instrumental stem"]')).toBeTruthy();
  });
});
