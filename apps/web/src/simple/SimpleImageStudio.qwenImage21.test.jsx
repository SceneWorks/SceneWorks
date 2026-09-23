// sc-24113 — Qwen-Image 2.1's controls in the SIMPLE image studio.
//
// Simple is an alternative shell, not a subset view. A control the model declares and only Advanced
// renders is a control every Simple user is missing, and this model's whole surface — a 32-px free
// size, 1–10 ordered references, a step floor of 2, seed, a negative prompt with true CFG, and a
// count ladder topping out at 8 — was invisible here. Simple's own contract is a reduced SURFACE,
// not a reduced payload (`simpleJobs.js` runs the same builder the full studio does), so the
// controls live behind ONE disclosure that starts COLLAPSED rather than eight more rows.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";

// `setInput` drives the HTMLInputElement value setter, which jsdom refuses on a textarea.
// Same trick, the matching prototype.
function setTextarea(element, value) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype,
    "value",
  ).set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async (path) =>
      path === "/api/v1/host-capabilities"
        ? { memoryGb: 128, memoryKind: "unified", platform: "macos" }
        : {},
    ),
  };
});

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../../config/manifests/builtin.models.jsonc");
const manifestModels = (() => {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  return Array.isArray(parsed) ? parsed : parsed.models;
})();

// The SHIPPED entry, so every bound below is the real declared one rather than a fixture's guess.
const QWEN_2_1 = {
  ...manifestModels.find((model) => model.id === "qwen_image_2_1"),
  installState: "installed",
  usable: true,
};

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-1", name: "Default" },
    assets: [],
    recentImageAssets: [],
    recentVideoAssets: [],
    jobs: [],
    imageModels: [QWEN_2_1],
    videoModels: [],
    audioModels: [],
    models: [QWEN_2_1],
    loras: [],
    imageLocalJobs: [],
    videoLocalJobs: [],
    audioLocalJobs: [],
    visibleWorkers: [],
    macCapabilities: null,
    theme: "light",
    changeTheme: () => {},
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createVideoJob: vi.fn(async () => null),
    createAudioJob: vi.fn(async () => null),
    createModelDownloadJob: vi.fn(async () => null),
    createLoraDownloadJob: vi.fn(async () => null),
    jobAction: vi.fn(async () => {}),
    rememberLocalGenerationJob: vi.fn(),
    refinePrompt: vi.fn(),
    qwenRewritePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    updateAssetStatus: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

describe("SimpleImageStudio with Qwen Image 2.1 (sc-24113)", () => {
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

  async function openImage(context = baseContext()) {
    await act(async () => {
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
    return context;
  }

  const advanced = () => container.querySelector("details.su-advanced");
  const fieldById = (id) => container.querySelector(`#${id}`);

  async function openAdvanced() {
    const fold = advanced();
    expect(fold, "the advanced disclosure renders").toBeTruthy();
    await act(async () => {
      fold.open = true;
      fold.dispatchEvent(new Event("toggle", { bubbles: false }));
    });
    return fold;
  }

  it("keeps the advanced controls COLLAPSED until asked", async () => {
    await openImage();
    // The busy-page rule: a reduced shell stays reduced for someone who never opens the fold.
    const fold = advanced();
    expect(fold.open).toBe(false);
    expect(fold.querySelector("summary").textContent).toBe("Advanced");

    // The controls live INSIDE the fold, which is what makes them collapsed. Asserted by
    // containment rather than absence: a closed `<details>` keeps its children in the DOM (the
    // browser hides them and skips them in focus order; jsdom models neither), so querying the
    // document would find them and prove nothing either way.
    expect(fold.contains(fieldById("su-image-steps"))).toBe(true);
    expect(fold.contains(fieldById("su-image-seed"))).toBe(true);
    expect(fold.contains(fieldById("su-image-width"))).toBe(true);

    // ... and the settings bar above it is untouched: Model, Resolution and Variations stay where
    // Simple's design puts them, so the fold ADDS a surface rather than moving one.
    expect(fold.contains(container.querySelector(".su-chips"))).toBe(false);
  });

  it("offers steps at the MODEL's floor, plus seed, negative prompt and true-CFG guidance", async () => {
    await openImage();
    await openAdvanced();

    const steps = fieldById("su-image-steps");
    expect(steps).toBeTruthy();
    // The declared floor, not a hardcoded 1 — below it the enqueue gate 400s.
    expect(steps.getAttribute("min")).toBe("2");
    expect(steps.getAttribute("placeholder")).toBe("40");

    expect(fieldById("su-image-seed")).toBeTruthy();
    // 2.1 declares no `image` block, and ABSENT MEANS TRUE for both axes, so both controls render.
    expect(fieldById("su-image-guidance")).toBeTruthy();
    expect(fieldById("su-image-negative")).toBeTruthy();
  });

  it("offers a free size on the model's own 32-px grid, not the blanket envelope", async () => {
    await openImage();
    await openAdvanced();

    const width = fieldById("su-image-width");
    expect(width).toBeTruthy();
    // The DECLARED envelope: 32–2752 in steps of 32. The blanket 256–4096/step-1 would be wrong at
    // both ends — it refuses every legal small size and admits sizes the engine cannot render.
    expect(width.getAttribute("min")).toBe("32");
    expect(width.getAttribute("max")).toBe("2752");
    expect(width.getAttribute("step")).toBe("32");
    expect(container.textContent).toContain("32–2752 px");
  });

  it("blocks Generate on an off-grid size and says why", async () => {
    await openImage();
    // A prompt, so the only thing that can disable Generate below is the geometry.
    await act(async () =>
      setTextarea(container.querySelector("#su-image-prompt"), "a lighthouse"),
    );
    await openAdvanced();

    await act(async () => setInput(fieldById("su-image-width"), "2050"));
    // The message names the grid, so the user is not left guessing which of the two boxes is wrong.
    expect(container.querySelector(".su-error")?.textContent).toContain("multiple of 32");

    const generate = container.querySelector(".su-generate");
    expect(generate.disabled, "an invalid size must not reach the enqueue gate").toBe(true);

    // ... and a legal size clears both.
    await act(async () => setInput(fieldById("su-image-width"), "2048"));
    expect(container.querySelector(".su-error")).toBeNull();
    expect(container.querySelector(".su-generate").disabled).toBe(false);
  });

  it("reads the variation ladder from the model instead of the hardcoded [1,2,4,6]", async () => {
    await openImage();
    const chips = [...container.querySelectorAll(".su-chips")]
      .find((node) => node.getAttribute("aria-label") === "Variations")
      ?.querySelectorAll(".su-chip");
    expect(chips).toBeTruthy();
    const labels = [...chips].map((chip) => chip.textContent.trim());
    // The model's own `limits.count`. The old hardcoded ladder offered a 6 nothing declares and
    // could not offer the 8 this engine takes.
    expect(labels).toEqual(["1", "2", "4", "8"]);
    expect(labels).toContain("8");
    expect(labels).not.toContain("6");
  });

  // sc-24114 — "preserve existing Qwen models as separate choices": the native-envelope surface is
  // 2.1's, so the older Qwen entries render in Simple exactly as before — the historical [1,2,4,6]
  // ladder and no Advanced fold — even though they declare `limits.count`.
  // *Mutation that reds this:* reading `limits.count` / rendering the fold for every model.
  it("leaves the existing Qwen models' Simple surface as it was", async () => {
    for (const id of ["qwen_image", "qwen_image_edit_2511"]) {
      const legacy = {
        ...manifestModels.find((model) => model.id === id),
        installState: "installed",
        usable: true,
      };
      window.localStorage.clear();
      await openImage(baseContext({ imageModels: [legacy], models: [legacy] }));
      const chips = [...container.querySelectorAll(".su-chips")]
        .find((node) => node.getAttribute("aria-label") === "Variations")
        ?.querySelectorAll(".su-chip");
      expect([...chips].map((chip) => chip.textContent.trim()), id).toEqual(["1", "2", "4", "6"]);
      expect(advanced(), id).toBeNull();
      await act(async () => root.render(null));
    }
  });

  it("sends the advanced knobs through the same builder the full studio uses", async () => {
    const context = await openImage();
    await openAdvanced();

    const prompt = container.querySelector("#su-image-prompt");
    await act(async () => setTextarea(prompt, "a lighthouse at dusk"));
    await act(async () => setInput(fieldById("su-image-steps"), "12"));
    await act(async () => setInput(fieldById("su-image-seed"), "4242"));
    await act(async () => setTextarea(fieldById("su-image-negative"), "watermark"));
    await act(async () => setInput(fieldById("su-image-guidance"), "4"));

    await click(container.querySelector(".su-generate"));
    expect(context.createImageJob).toHaveBeenCalled();
    const request = context.createImageJob.mock.calls[0][0];
    // Simple's contract: a reduced surface, the SAME payload. Each knob lands where the full
    // studio puts it, so a Simple run replays into Advanced unchanged.
    expect(request.seed).toBe(4242);
    expect(request.negativePrompt).toBe("watermark");
    expect(request.advanced.steps).toBe(12);
    expect(request.advanced.guidanceScale).toBe(4);
  });

  it("offers the transparency toggle with the prompt convention it needs", async () => {
    const context = await openImage();
    const toggle = container.querySelector(".su-transparency-toggle input");
    // Rendered because the model advertises alpha output; hidden for every model that does not.
    expect(toggle).toBeTruthy();

    await act(async () => setTextarea(container.querySelector("#su-image-prompt"), "a courier"));
    await click(toggle);

    // THE non-obvious half: there is no transparency MODE upstream, so the toggle alone yields an
    // opaque RGBA PNG. The affordance that fixes it must appear, and it must EDIT the visible
    // prompt rather than append anything silently.
    const hint = [...container.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Ask for transparency"),
    );
    expect(hint, "the toggle must be paired with the prompt convention").toBeTruthy();
    await click(hint);
    expect(container.querySelector("#su-image-prompt").value).toContain("transparent");

    // Offered once: the prompt now says it, so the button retires rather than accreting the
    // sentence on every press.
    expect(
      [...container.querySelectorAll("button")].some((button) =>
        button.textContent.includes("Ask for transparency"),
      ),
    ).toBe(false);

    await click(container.querySelector(".su-generate"));
    const request = context.createImageJob.mock.calls[0][0];
    expect(request.advanced.transparentBackground).toBe(true);
  });
});
