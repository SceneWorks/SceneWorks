import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { MAX_USER_JOB_LORAS } from "../presetUtils.js";
import { click, mountRoot, setFileInput, setInput, unmountRoot } from "../testUtils/dom.js";
import { resetStartupTimingForTests } from "../startupTiming.js";
import { promptWithKeyword } from "./SimpleLoraField.jsx";

// LoRA support in the Simple studios (epic 15404). Driven through the real shell and the
// real AppContext, so what's asserted is the payload the app would actually POST — the
// point of the feature is that a Simple run carries the SAME `loras` array the advanced
// studio sends, not a Simple-only shape.

const IMAGE_MODEL = {
  id: "z_image",
  name: "Z-Image",
  type: "image",
  family: "z-image",
  capabilities: ["text_to_image", "edit_image"],
  installState: "installed",
  limits: { resolutions: ["1024x1024"] },
};

const WAN_5B = {
  id: "wan_2_2",
  name: "Wan 2.2 (TI2V-5B)",
  type: "video",
  family: "wan-2-2",
  capabilities: ["text_to_video"],
  installState: "installed",
  limits: { resolutions: ["1280x720"], durations: [5] },
};

const WAN_A14B = { ...WAN_5B, id: "wan_2_2_t2v_14b", name: "Wan 2.2 14B (T2V)" };
const LTX = { ...WAN_5B, id: "ltx_2_3", name: "LTX-2.3", family: "ltx-video" };

const GRAIN = {
  id: "lora_grain",
  name: "Cinematic Grain 35mm",
  family: "z-image",
  scope: "global",
  installState: "installed",
  triggerWords: ["cinegrain"],
};
const INK = {
  id: "lora_ink",
  name: "Ink Wash Portrait",
  family: "z-image",
  scope: "project",
  installState: "installed",
  defaultWeight: 0.7,
};
// Each of these must be invisible to the picker, for a different reason.
const WRONG_FAMILY = { id: "lora_wan", name: "Slow Dolly", family: "wan-2-2", installState: "installed" };
const NOT_INSTALLED = { id: "lora_pending", name: "Still Downloading", family: "z-image", installState: "missing" };
const FAMILYLESS = { id: "lora_unknown", name: "Mystery Adapter", installState: "installed" };
const PRESET_MANAGED = {
  id: "lora_preset",
  name: "Preset Owned",
  family: "z-image",
  installState: "installed",
  presetManaged: true,
};

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-1", name: "Default" },
    assets: [],
    recentImageAssets: [],
    jobs: [],
    imageModels: [IMAGE_MODEL],
    videoModels: [WAN_5B, WAN_A14B],
    audioModels: [],
    models: [IMAGE_MODEL],
    loras: [GRAIN, INK, WRONG_FAMILY, NOT_INSTALLED, FAMILYLESS, PRESET_MANAGED],
    imageLocalJobs: [],
    videoLocalJobs: [],
    audioLocalJobs: [],
    visibleWorkers: [],
    macCapabilities: null,
    theme: "light",
    changeTheme: () => {},
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createVideoJob: vi.fn(async () => ({ id: "job-2" })),
    createAudioJob: vi.fn(async () => null),
    createModelDownloadJob: vi.fn(async () => null),
    createLoraDownloadJob: vi.fn(async () => null),
    createLoraImportJob: vi.fn(async () => ({
      id: "import-1",
      type: "lora_import",
      status: "queued",
      payload: { loraId: "analog_35mm", manifestEntry: { family: "z_image" } },
    })),
    jobAction: vi.fn(async () => {}),
    rememberLocalGenerationJob: vi.fn(),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    updateAssetStatus: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

function renderShell(root, context) {
  return act(async () => {
    root.render(
      <AppContext.Provider value={context}>
        <SimpleShell
          accent="teal"
          lockedToSimple={false}
          onAccentChange={() => {}}
          onModeChange={() => {}}
          onSimpleDefaultChange={() => {}}
          simpleDefault
        />
      </AppContext.Provider>,
    );
  });
}

function navButton(container, label) {
  return [...container.querySelectorAll(".su-nav-item")].find((node) => node.textContent.startsWith(label));
}

function pickRow(container, name) {
  return [...container.querySelectorAll(".su-lora-pick-row")].find((node) =>
    node.textContent.startsWith(name),
  );
}

function slotNames(container) {
  return [...container.querySelectorAll(".su-lora-slot .su-lora-slot-meta strong")].map((node) =>
    node.textContent.trim(),
  );
}

// The studio's Model control is a sheet-backed select; picking from it is two clicks.
async function selectModel(container, name) {
  await click(container.querySelector(".su-select"));
  await click(
    [...container.querySelectorAll(".su-option-row")].find((node) => node.textContent.includes(name)),
  );
}

async function openPicker(container) {
  await click(container.querySelector(".su-lora-add"));
}

async function addLora(container, name) {
  await openPicker(container);
  await click(pickRow(container, name));
}

async function typePrompt(container, value) {
  const textarea = container.querySelector("#su-image-prompt");
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
    setter.call(textarea, value);
    textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
}

describe("promptWithKeyword", () => {
  it("appends a keyword after a comma", () => {
    expect(promptWithKeyword("a lighthouse", "cinegrain")).toBe("a lighthouse, cinegrain");
  });

  it("is the keyword alone when the prompt is empty", () => {
    expect(promptWithKeyword("", "cinegrain")).toBe("cinegrain");
    expect(promptWithKeyword("   ", "cinegrain")).toBe("cinegrain");
  });

  // A second tap on the same chip must not double the keyword — the point of the chip is to
  // get the word into the prompt once, and a trailing "x, x" is noise the model reads.
  it("is a no-op when the keyword is already present, case-insensitively", () => {
    expect(promptWithKeyword("a shot, cinegrain", "cinegrain")).toBe("a shot, cinegrain");
    expect(promptWithKeyword("a shot, CineGrain", "cinegrain")).toBe("a shot, CineGrain");
  });

  it("does not leave a double comma behind a trailing one", () => {
    expect(promptWithKeyword("a lighthouse,", "cinegrain")).toBe("a lighthouse, cinegrain");
  });
});

describe("Simple studios — LoRA field", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    // The Simple studios now persist their settings (localStorage cache + the durable
    // ui-preferences copy), so without this a snapshot flushed by one case restores into
    // the next and the catalog-default assertions read the previous test's picks.
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    resetStartupTimingForTests();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders inside the settings bar as a peer of the other controls", async () => {
    await renderShell(root, baseContext());
    const bar = container.querySelector(".su-settings-bar");
    expect(bar.querySelector(".su-lora-field")).toBeTruthy();
    // Last child, per the design — under Variations, not above Model.
    expect(bar.lastElementChild.classList.contains("su-lora-field")).toBe(true);
    expect(container.querySelector(".su-lora-head > span").textContent).toBe("Installed and compatible");
  });

  // The eligibility rules are `presetUtils`' — this asserts Simple actually applies all four
  // of them rather than listing whatever is in the catalog.
  it("offers only installed, compatible, family-resolvable, non-preset LoRAs", async () => {
    await renderShell(root, baseContext());
    expect(container.querySelector(".su-lora-add-count").textContent).toBe("· 2 available");
    await openPicker(container);
    const offered = [...container.querySelectorAll(".su-lora-pick-row strong")].map((n) => n.textContent);
    expect(offered).toEqual(["Cinematic Grain 35mm", "Ink Wash Portrait"]);
  });

  it("adds a LoRA from the sheet, closes it, and updates the status line", async () => {
    await renderShell(root, baseContext());
    await addLora(container, "Cinematic Grain 35mm");
    expect(container.querySelector(".su-sheet")).toBeNull();
    expect(slotNames(container)).toEqual(["Cinematic Grain 35mm"]);
    expect(container.querySelector(".su-lora-head > span").textContent).toBe(
      "1 selected · installed and compatible",
    );
    expect(container.querySelector(".su-lora-add-count").textContent).toBe("· 1 available");
    // Weight defaults to the LoRA's own default (0.8 here, via `loraWeight`).
    expect(container.querySelector(".su-lora-weight-value").textContent).toBe("0.80");
  });

  it("seeds the slider from the LoRA's declared defaultWeight", async () => {
    await renderShell(root, baseContext());
    await addLora(container, "Ink Wash Portrait");
    expect(container.querySelector(".su-lora-weight-value").textContent).toBe("0.70");
  });

  it("keeps the bidirectional −2…2 weight range", async () => {
    await renderShell(root, baseContext());
    await addLora(container, "Cinematic Grain 35mm");
    const slider = container.querySelector(".su-lora-weight input[type=range]");
    expect(slider.min).toBe("-2");
    expect(slider.max).toBe("2");
    expect(slider.step).toBe("0.05");
  });

  it("removes a LoRA and forgets its weight override", async () => {
    await renderShell(root, baseContext());
    await addLora(container, "Cinematic Grain 35mm");
    await act(async () => setInput(container.querySelector(".su-lora-weight input[type=range]"), "1.5"));
    expect(container.querySelector(".su-lora-weight-value").textContent).toBe("1.50");
    await click(container.querySelector(".su-lora-remove"));
    expect(slotNames(container)).toEqual([]);
    // Re-adding starts from the LoRA's own default again, not the override we just cleared.
    await addLora(container, "Cinematic Grain 35mm");
    expect(container.querySelector(".su-lora-weight-value").textContent).toBe("0.80");
  });

  it("appends a trigger keyword to the prompt when its chip is clicked", async () => {
    await renderShell(root, baseContext());
    await typePrompt(container, "portrait of a lighthouse keeper");
    await addLora(container, "Cinematic Grain 35mm");
    const chip = container.querySelector(".su-lora-keyword");
    expect(chip.textContent).toBe("cinegrain");
    await click(chip);
    expect(container.querySelector("#su-image-prompt").value).toBe(
      "portrait of a lighthouse keeper, cinegrain",
    );
    await click(chip);
    expect(container.querySelector("#su-image-prompt").value).toBe(
      "portrait of a lighthouse keeper, cinegrain",
    );
  });

  it("caps user selections at MAX_USER_JOB_LORAS and says so", async () => {
    const many = Array.from({ length: MAX_USER_JOB_LORAS + 1 }, (_, index) => ({
      id: `lora_${index}`,
      name: `Adapter ${index}`,
      family: "z-image",
      scope: "global",
      installState: "installed",
    }));
    await renderShell(root, baseContext({ loras: many }));
    for (let index = 0; index < MAX_USER_JOB_LORAS; index += 1) {
      await addLora(container, `Adapter ${index}`);
    }
    expect(container.querySelector(".su-lora-note").textContent).toBe(
      `Limit reached — ${MAX_USER_JOB_LORAS} LoRAs per run.`,
    );
    // The Add row itself stays live (sc-15373) — the sheet is where the reason lives.
    expect(container.querySelector(".su-lora-add").disabled).toBe(false);
    await openPicker(container);
    const row = pickRow(container, `Adapter ${MAX_USER_JOB_LORAS}`);
    expect(row.disabled).toBe(true);
    expect(row.querySelector(".su-lora-pick-add").textContent).toBe("Limit reached");
  });

  it("sends the picked LoRAs on the generated job, in serializeLora shape", async () => {
    const context = baseContext();
    await renderShell(root, context);
    await typePrompt(container, "a lighthouse");
    await addLora(container, "Cinematic Grain 35mm");
    await act(async () => setInput(container.querySelector(".su-lora-weight input[type=range]"), "1.25"));
    await click([...container.querySelectorAll(".su-generate")][0]);
    const request = context.createImageJob.mock.calls[0][0];
    expect(request.loras).toEqual([
      expect.objectContaining({ id: "lora_grain", name: "Cinematic Grain 35mm", scope: "global", weight: 1.25 }),
    ]);
  });

  it("keeps a selection across a model change inside the same family", async () => {
    await renderShell(root, baseContext());
    await click(navButton(container, "Video"));
    await addLora(container, "Slow Dolly");
    expect(slotNames(container)).toEqual(["Slow Dolly"]);
    // Wan 5B → Wan 14B share the "wan-2-2" family, so the pick is still valid.
    await selectModel(container, "Wan 2.2 14B");
    expect(slotNames(container)).toEqual(["Slow Dolly"]);
  });

  it("drops a selection the new model's family can't take", async () => {
    await renderShell(root, baseContext({ videoModels: [WAN_5B, LTX] }));
    await click(navButton(container, "Video"));
    await addLora(container, "Slow Dolly");
    expect(slotNames(container)).toEqual(["Slow Dolly"]);
    await selectModel(container, "LTX-2.3");
    expect(slotNames(container)).toEqual([]);
    expect(container.querySelector(".su-lora-head > span").textContent).toBe("Installed and compatible");
    // Switching BACK must not resurrect it: the id has to be gone from the selection, not
    // merely filtered out of the render. Otherwise the picker would keep excluding a LoRA the
    // user can no longer see selected.
    await selectModel(container, "Wan 2.2 (TI2V-5B)");
    expect(slotNames(container)).toEqual([]);
    expect(container.querySelector(".su-lora-add-count").textContent).toBe("· 1 available");
  });

  // sc-11962's guard, ported: `compatibleLoras` is empty both when the catalog hasn't
  // resolved AND when nothing matches. Pruning during the former would permanently strip a
  // selection the user still has, so the prune only runs once both inputs are present.
  it("does not strip a selection while the LoRA catalog is unresolved", async () => {
    const context = baseContext();
    await renderShell(root, context);
    await addLora(container, "Cinematic Grain 35mm");
    expect(slotNames(container)).toEqual(["Cinematic Grain 35mm"]);
    await renderShell(root, baseContext({ loras: [] }));
    await renderShell(root, baseContext());
    expect(slotNames(container)).toEqual(["Cinematic Grain 35mm"]);
  });

  it("shows the A14B paired-expert hint only for the MoE models", async () => {
    await renderShell(root, baseContext());
    await click(navButton(container, "Video"));
    // Seeded on the dense 5B — single-file LoRAs, so no hint.
    expect(container.querySelector(".su-lora-hint")).toBeNull();
    await selectModel(container, "Wan 2.2 14B");
    expect(container.querySelector(".su-lora-hint").textContent).toContain(
      "high/low-noise pair",
    );
  });

  it("carries the picked LoRAs onto a video run", async () => {
    const context = baseContext();
    await renderShell(root, context);
    await click(navButton(container, "Video"));
    const textarea = container.querySelector("#su-video-prompt");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
      setter.call(textarea, "slow push-in");
      textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    await addLora(container, "Slow Dolly");
    await click(container.querySelector(".su-generate"));
    const request = context.createVideoJob.mock.calls[0][0];
    expect(request.loras).toEqual([expect.objectContaining({ id: "lora_wan", weight: 0.8 })]);
  });
});

describe("Simple studios — Add LoRA sheet import", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    // The Simple studios now persist their settings (localStorage cache + the durable
    // ui-preferences copy), so without this a snapshot flushed by one case restores into
    // the next and the catalog-default assertions read the previous test's picks.
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    resetStartupTimingForTests();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("reports why the list is empty and opens the import form in the same place", async () => {
    await renderShell(root, baseContext({ loras: [] }));
    await openPicker(container);
    expect(container.querySelector(".su-lora-pick-empty").textContent).toBe(
      "No installed LoRAs match Z-Image.",
    );
    // sc-15373: the fix is expanded, not one more tap away.
    expect(container.querySelector(".su-lora-import-panel")).toBeTruthy();
  });

  it("credits a still-downloading compatible LoRA in the empty message", async () => {
    await renderShell(root, baseContext({ loras: [NOT_INSTALLED] }));
    await openPicker(container);
    expect(container.querySelector(".su-lora-pick-empty").textContent).toBe(
      "No installed compatible LoRAs. Imports appear after the Queue completes.",
    );
  });

  it("stays collapsed when there is something to pick", async () => {
    await renderShell(root, baseContext());
    await openPicker(container);
    expect(container.querySelector(".su-lora-import-panel")).toBeNull();
    await click(container.querySelector(".su-lora-import-trigger"));
    expect(container.querySelector(".su-lora-import-panel")).toBeTruthy();
  });

  it("queues an import through the shared seam with an auto-detected family", async () => {
    const context = baseContext();
    await renderShell(root, context);
    await openPicker(container);
    await click(container.querySelector(".su-lora-import-trigger"));
    await act(async () => {
      setFileInput(container.querySelector(".su-lora-file-btn input"), [
        new window.File(["x"], "analog_35mm.safetensors"),
      ]);
    });
    expect(container.querySelector(".su-lora-file-name").textContent).toBe("analog_35mm.safetensors");
    const keywordInput = container.querySelector(".su-lora-kw input");
    await act(async () => setInput(keywordInput, "analog35"));
    await act(async () => {
      keywordInput.dispatchEvent(new window.KeyboardEvent("keydown", { bubbles: true, key: "Enter" }));
    });
    expect(container.querySelector(".su-lora-kw-chip").textContent).toContain("analog35");

    await click([...container.querySelectorAll(".su-accent-btn")].at(-1));
    const payload = context.createLoraImportJob.mock.calls[0][0];
    expect(payload.file.name).toBe("analog_35mm.safetensors");
    expect(payload.triggerWords).toEqual(["analog35"]);
    // Scope defaults to the open project; family is NEVER sent — the importer detects it.
    expect(payload.scope).toBe("project");
    expect(payload).not.toHaveProperty("family");
    // The detected family comes back normalized (z_image → z-image).
    expect(container.querySelector(".su-lora-import-msg--success").textContent).toBe(
      "LoRA import queued for analog_35mm. Detected family: z-image.",
    );
    // Form cleared, sheet still open.
    expect(container.querySelector(".su-lora-file-name").textContent).toBe("No file selected");
    expect(container.querySelector(".su-sheet")).toBeTruthy();
  });

  it("surfaces an import failure rather than swallowing it", async () => {
    const context = baseContext({
      createLoraImportJob: vi.fn(async () => {
        throw new Error("Uploaded LoRA file exceeds the 2GB limit");
      }),
    });
    await renderShell(root, context);
    await openPicker(container);
    await click(container.querySelector(".su-lora-import-trigger"));
    await act(async () => {
      setFileInput(container.querySelector(".su-lora-file-btn input"), [
        new window.File(["x"], "huge.safetensors"),
      ]);
    });
    await click([...container.querySelectorAll(".su-accent-btn")].at(-1));
    expect(container.querySelector(".su-lora-import-msg--error").textContent).toBe(
      "Uploaded LoRA file exceeds the 2GB limit",
    );
  });

  it("shows a pending slot driven by the live job feed, and clears it on completion", async () => {
    const runningJob = {
      id: "import-1",
      type: "lora_import",
      status: "queued",
      payload: { name: "analog_35mm.safetensors", manifestEntry: { family: "z_image" } },
    };
    const context = baseContext({ createLoraImportJob: vi.fn(async () => runningJob) });
    await renderShell(root, context);
    await openPicker(container);
    await click(container.querySelector(".su-lora-import-trigger"));
    await act(async () => {
      setFileInput(container.querySelector(".su-lora-file-btn input"), [
        new window.File(["x"], "analog_35mm.safetensors"),
      ]);
    });
    await click([...container.querySelectorAll(".su-accent-btn")].at(-1));

    // The job has to be in the feed for the slot to exist — nothing here is local state.
    await renderShell(root, baseContext({ ...context, jobs: [runningJob] }));
    const pending = container.querySelector(".su-lora-slot--pending");
    expect(pending).toBeTruthy();
    expect(pending.querySelector("strong").textContent).toBe("analog_35mm.safetensors");
    expect(pending.querySelector("small").textContent).toBe("importing · family detected: z-image");
    expect(pending.querySelector(".su-job-badge--running").textContent).toBe("Queued");
    expect(pending.querySelector(".su-bar--indeterminate")).toBeTruthy();

    // Terminal ⇒ gone, with no remount and no local bookkeeping to unwind.
    await renderShell(root, baseContext({ ...context, jobs: [{ ...runningJob, status: "completed" }] }));
    expect(container.querySelector(".su-lora-slot--pending")).toBeNull();
  });
});
