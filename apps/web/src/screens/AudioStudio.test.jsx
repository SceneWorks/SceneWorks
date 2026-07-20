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

  it("surfaces the Voice Clone conditioning scaffold from audio.conditioning", async () => {
    await render(baseContext());
    await click(modeTab(container, "Voice Clone"));

    // Snaps to a conditioning model (OpenVoice), and shows the reference-voice scaffold copy.
    expect(modelSelect(container).value).toBe("openvoice_v2");
    expect(container.textContent).toContain("ReferenceAudio");
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

  it("does not submit on a still-scaffold tab (Music / Voice Clone stay inert)", async () => {
    const createAudioJob = vi.fn(async () => ({ id: "nope" }));
    await render(
      baseContext({ createAudioJob, rememberLocalGenerationJob: vi.fn() }),
    );
    // Music is served only by ACE-Step in the base fixture; switch to it, type a script. (Speech and
    // Sound FX are wired — C1/C2 — so Music/Voice Clone are the remaining scaffold tabs.)
    await click(modeTab(container, "Music"));
    await setTextarea(container.querySelector(".prompt-input"), "some music");
    // The CTA is disabled off a wired tab, and a direct form submit is a no-op there.
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
