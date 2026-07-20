import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";

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
  ui: { label: "Kokoro 82M" },
};

const MOSS = {
  id: "moss_sfx_v2",
  name: "MOSS SoundEffect v2 (SFX)",
  type: "audio",
  audio: { languages: ["en", "zh"], sampleRates: [48000], maxDurationSecs: 30 },
  ui: { label: "MOSS SoundEffect v2" },
};

const ACESTEP = {
  id: "acestep_v15_turbo",
  name: "ACE-Step v1.5 XL Turbo (Music)",
  type: "audio",
  audio: {
    languages: ["en", "zh"],
    sampleRates: [48000],
    maxDurationSecs: 600,
    editModes: ["inpaint", "repaint", "extend"],
    conditioning: ["AudioEdit"],
  },
  ui: { label: "ACE-Step v1.5 XL Turbo" },
};

const OPENVOICE = {
  id: "openvoice_v2",
  name: "OpenVoice V2 (Voice Conversion)",
  type: "audio",
  audio: { sampleRates: [22050], conditioning: ["ReferenceAudio"] },
  ui: { label: "OpenVoice V2" },
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

const buttonWithText = (root, text) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent.trim() === text);
const modeTabs = (container) => container.querySelector(".mode-control");
const modeTab = (container, label) => buttonWithText(modeTabs(container), label);
const modelSelect = (container) => container.querySelector(".settings-field-model select");
const fieldByLabelStart = (container, label) =>
  [...container.querySelectorAll(".settings-bar label")].find((el) =>
    el.textContent.trim().startsWith(label),
  );

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
    expect(editChips).toEqual(["inpaint", "repaint", "extend"]);
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

  it("renders an audio-player results card for a completed audio job in the lane", async () => {
    const audioAsset = {
      id: "audio-asset-1",
      type: "audio",
      projectId: "project_1",
      displayName: "The walking skeleton is alive.",
      file: { path: "assets/audios/genset_x/2026-07-19_kokoro_walking.wav", mimeType: "audio/wav" },
    };
    const completedJob = {
      id: "audio-job-done",
      type: "audio_generate",
      status: "completed",
      result: { assetIds: ["audio-asset-1"], expectedCount: 1 },
    };
    await render(
      baseContext({
        assets: [audioAsset],
        audioLocalJobs: [completedJob],
        createAudioJob: vi.fn(),
        rememberLocalGenerationJob: vi.fn(),
      }),
    );

    const results = container.querySelector(".studio-results");
    expect(results.querySelector(".worker-progress-card")).toBeTruthy();
    // The completed clip surfaces through the shared audio-player card with a real <audio> src.
    const audioEl = results.querySelector("audio");
    expect(audioEl).toBeTruthy();
    expect(audioEl.getAttribute("src")).toContain(
      "assets/audios/genset_x/2026-07-19_kokoro_walking.wav",
    );
    expect(results.textContent).not.toContain("No audio yet");
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
    expect(editChips).toEqual(["inpaint", "repaint", "extend"]);
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
