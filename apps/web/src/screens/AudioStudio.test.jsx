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
