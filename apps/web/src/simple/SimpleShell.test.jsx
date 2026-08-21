import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";
import { resetStartupTimingForTests } from "../startupTiming.js";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async (path) =>
      path === "/api/v1/host-capabilities"
        ? { memoryGb: 80, memoryKind: "unified", platform: "macos" }
        : {},
    ),
  };
});

// The Simple shell mounts inside App's providers and reads everything through them, so a
// single legacy <AppContext.Provider> is enough to drive it in isolation (see AppContext's
// fallback contract). These tests cover the shell's own contract: navigation, the picker
// sheet, the toast, and that a Generate goes through the real job-creation action with the
// real payload — not a mock.

const IMAGE_MODEL = {
  id: "z_image",
  name: "Z-Image",
  type: "image",
  capabilities: ["text_to_image", "edit_image"],
  installState: "installed",
  limits: { resolutions: ["1024x1024", "1344x768"] },
  // Z-Image declares reference-guided generation in the real catalog, with this exact
  // strength config — so the Text tab's Reference tile is live for it.
  ui: { img2img: true, img2imgStrength: { default: 0.5, min: 0, max: 1, step: 0.05 } },
};

const TIERED_HIGHRES_MODEL = {
  ...IMAGE_MODEL,
  id: "tiered_highres",
  name: "Tiered Highres",
  defaults: { resolution: "1024x1024" },
  limits: { resolutions: ["1024x1024", "2048x2048"] },
  mlx: { minMemoryGb: 80 },
  hasVariantMatrix: true,
  variants: [
    {
      variant: "q4",
      installState: "installed",
      footprint: {
        diskSizeBytes: 8 * 1024 ** 3,
        peakMemoryBytes: 22 * 1024 ** 3,
        measuredPixels: 1024 * 1024,
      },
    },
  ],
};

const REFERENCE_ASSET = {
  id: "asset-9",
  type: "image",
  projectId: "project-1",
  displayName: "cliff_ref.png",
  url: "/media/cliff_ref.png",
};

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-1", name: "Default" },
    assets: [],
    recentImageAssets: [],
    jobs: [],
    imageModels: [IMAGE_MODEL],
    videoModels: [],
    audioModels: [],
    models: [IMAGE_MODEL],
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
    deleteAsset: vi.fn(),
    updateAssetStatus: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

function renderShell(root, context, props = {}) {
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
          {...props}
        />
      </AppContext.Provider>,
    );
  });
}

function buttonWithText(container, text) {
  return [...container.querySelectorAll("button")].find(
    (node) => node.textContent.trim() === text,
  );
}

// Nav items carry a count badge ("Queue1" when one job is pending), so they can't be
// matched by exact text.
function navButton(container, label) {
  return [...container.querySelectorAll(".su-nav-item")].find((node) =>
    node.textContent.startsWith(label),
  );
}

async function typePrompt(container, value) {
  const textarea = container.querySelector("#su-image-prompt");
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype,
      "value",
    ).set;
    setter.call(textarea, value);
    textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
}

describe("SimpleShell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    // The Simple studios now persist their settings (localStorage cache + the durable
    // ui-preferences copy), so without this a snapshot flushed by one case restores into
    // the next and the catalog-default assertions read the previous test's picks.
    window.localStorage.clear();
    // jsdom has no ResizeObserver; the shell's measurement hook degrades to the
    // unmeasured (desktop) band, which is what these tests exercise.
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    resetStartupTimingForTests();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("opens on the Image Studio with the persistent sidebar", async () => {
    await renderShell(root, baseContext());
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Image Studio");
    expect(container.querySelector(".su-nav")).toBeTruthy();
    // Desktop band ⇒ no hamburger.
    expect(container.querySelector(".su-hamburger")).toBeNull();
    // The Simple/Advanced switch lives in the sidebar footer.
    expect(container.querySelector(".su-nav-footer .su-switch")).toBeTruthy();
  });

  // sc-17162 — `ui.description` had exactly ONE reader in the app: the advanced Models screen's
  // card. Simple's only route there is "Manage", which switches the SHELL out from under the user,
  // so a Simple user identified every model by NAME ALONE. That is a reachability gap, not a
  // styling one: for MiniMax-H3 the string is the withheld-component disclosure, and a disclosure
  // that renders only on a screen the reduced shell hands off to is not discharged for that shell's
  // users.
  //
  // Driven through the REAL shell across all THREE studios, because the wiring is per-studio (three
  // separate call sites of `ModelDescription`) and a test that only covered Video — where
  // MiniMax-H3 lives — would have left Image and Audio silently unwired. Asserted against the
  // fixture's own string rather than a literal, so it is the model's declared copy that is proven
  // to reach the DOM.
  it("renders the selected model's description in every Simple studio (sc-17162)", async () => {
    // Eligibility is load-bearing here, not decoration: each studio picks its model out of the
    // catalog on its own terms, and a fixture that fails that filter selects NOTHING, so the studio
    // renders no model field and the assertion reads `undefined` for a reason that has nothing to
    // do with the wiring under test. Both legs failed exactly that way in draft — Video needs video
    // `capabilities`, and Audio resolves per MODE through `audioModelServesMode`, whose default
    // "music" mode requires a declared `audio.editModes`. That is the argument for driving all
    // three studios rather than extrapolating from whichever one happened to be green.
    const described = (type, description, extra) => ({
      ...IMAGE_MODEL,
      id: `${type}_described`,
      type,
      ui: { ...IMAGE_MODEL.ui, description },
      ...extra,
    });
    const image = described("image", "Image model. Declares what it does and does not do.", {
      capabilities: ["text_to_image"],
    });
    const video = described("video", "Video model. Not the hosted product; 2K is unreachable.", {
      capabilities: ["text_to_video", "image_to_video"],
    });
    const audio = described("audio", "Audio model. Mono only, and no voice cloning.", {
      capabilities: ["text_to_audio"],
      audio: { editModes: ["generate"], sampleRates: [44100] },
    });
    const context = baseContext({
      imageModels: [image],
      videoModels: [video],
      audioModels: [audio],
      models: [image, video, audio],
    });

    await renderShell(root, context);
    expect(container.querySelector(".su-model-desc")?.textContent).toBe(image.ui.description);

    await click(navButton(container, "Video"));
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Video Studio");
    expect(container.querySelector(".su-model-desc")?.textContent).toBe(video.ui.description);

    await click(navButton(container, "Audio"));
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Audio Studio");
    expect(container.querySelector(".su-model-desc")?.textContent).toBe(audio.ui.description);
  });

  it("renders nothing where a model declares no description (sc-17162)", async () => {
    // Rendering nothing is always correct — a model that declares no description owes none — and
    // this is the anti-vacuity guard for the assertion above: without it, a `ModelDescription` that
    // emitted an empty <p> for every model would still satisfy a querySelector-is-truthy check.
    // `IMAGE_MODEL` declares `ui` but no `description`, which is the common catalog shape.
    expect(IMAGE_MODEL.ui.description).toBeUndefined();
    await renderShell(root, baseContext());
    expect(container.querySelector(".su-model-desc")).toBeNull();
  });

  it("navigates between screens from the sidebar", async () => {
    await renderShell(root, baseContext());
    await click(navButton(container, "Queue"));
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Queue");
    await click(navButton(container, "Licenses"));
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Licenses");
  });

  it("records Assets ready only when the Simple Assets surface mounts", async () => {
    const mark = vi.fn();
    vi.stubGlobal("performance", {
      clearMarks: vi.fn(),
      clearMeasures: vi.fn(),
      mark,
      measure: vi.fn(),
      now: () => 1,
    });
    resetStartupTimingForTests();

    await renderShell(root, baseContext({ assetsReady: true }));
    expect(mark).not.toHaveBeenCalled();

    await click(navButton(container, "Assets"));
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Assets");
    expect(mark.mock.calls.map(([name]) => name)).toEqual([
      "sceneworks.assets-ready-render",
    ]);
  });

  it("shows the live queue count as a nav badge and drops it when nothing is pending", async () => {
    await renderShell(
      root,
      baseContext({
        jobs: [
          { id: "a", type: "image_generate", status: "running" },
          { id: "b", type: "image_generate", status: "queued" },
          { id: "c", type: "image_generate", status: "completed" },
        ],
      }),
    );
    expect(container.querySelector(".su-nav-badge").textContent).toBe("2");

    await renderShell(root, baseContext({ jobs: [{ id: "c", type: "image_generate", status: "completed" }] }));
    expect(container.querySelector(".su-nav-badge")).toBeNull();
  });

  it("opens the resolution picker as a sheet and applies the pick", async () => {
    await renderShell(root, baseContext());
    const resolutionButton = [...container.querySelectorAll(".su-select")].find((node) =>
      node.textContent.includes("1:1"),
    );
    expect(resolutionButton).toBeTruthy();
    await click(resolutionButton);

    const sheet = container.querySelector(".su-sheet");
    expect(sheet).toBeTruthy();
    expect(sheet.querySelector(".su-sheet-title").textContent).toBe("Size & aspect");

    const wide = [...sheet.querySelectorAll(".su-option-tile")].find((node) =>
      node.textContent.includes("16:9"),
    );
    await click(wide);
    // Sheet closes and the trigger reflects the new value.
    expect(container.querySelector(".su-sheet")).toBeNull();
    expect(
      [...container.querySelectorAll(".su-select")].some((node) => node.textContent.includes("16:9")),
    ).toBe(true);
  });

  it("gates resolutions with the same resolved tier Simple sends to generation", async () => {
    const context = baseContext({
      imageModels: [TIERED_HIGHRES_MODEL],
      models: [TIERED_HIGHRES_MODEL],
      macCapabilities: { macGatingActive: true },
    });
    await renderShell(root, context);
    await act(async () => {}); // host-memory promise + tier-dependent resolution memo

    const resolutionButton = [...container.querySelectorAll(".su-select")].find((node) =>
      node.textContent.includes("1:1"),
    );
    await click(resolutionButton);
    const options = [...container.querySelectorAll(".su-sheet .su-option-tile")];
    expect(options.some((node) => node.textContent.includes("2048"))).toBe(true);
  });

  it("opens the prompt guide overlay and closes it again", async () => {
    await renderShell(root, baseContext());
    await click(buttonWithText(container, "Prompt guide"));
    expect(container.textContent).toContain("Lead with the subject");
    await click(container.querySelector(".su-sheet-head .su-icon-btn"));
    expect(container.textContent).not.toContain("Lead with the subject");
  });

  it("submits a real image job through createImageJob and remembers it locally", async () => {
    const context = baseContext();
    await renderShell(root, context);

    const textarea = container.querySelector("#su-image-prompt");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value",
      ).set;
      setter.call(textarea, "a lighthouse at golden hour");
      textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
    });

    await click(container.querySelector(".su-generate"));

    expect(context.createImageJob).toHaveBeenCalledTimes(1);
    const payload = context.createImageJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      mode: "text_to_image",
      model: "z_image",
      prompt: "a lighthouse at golden hour",
      count: 1,
      width: 1024,
      height: 1024,
    });
    expect(context.rememberLocalGenerationJob).toHaveBeenCalledWith("image", { id: "job-1" });
  });

  // Regression for a shipped bug: the Reference tile armed, showed SET + a thumbnail and
  // toasted "Reference attached", but the id was routed to `sourceAssetId` — which
  // buildImageJobRequest discards outside edit mode — so the run silently ignored it. This
  // drives the REAL tile through the REAL picker sheet, which is what the payload-only
  // tests could not catch.
  it("attaches a Text-mode reference through the tile and sends it with a strength", async () => {
    const context = baseContext({ assets: [REFERENCE_ASSET], recentImageAssets: [REFERENCE_ASSET] });
    await renderShell(root, context);

    await typePrompt(container, "a lighthouse");
    await click(buttonWithText(container, "ReferenceTap to attach an image"));

    // The picker sheet lists the project's images; choose the one asset.
    const row = [...container.querySelectorAll(".su-option-row")].find((node) =>
      node.textContent.includes("cliff_ref.png"),
    );
    expect(row).toBeTruthy();
    await click(row);

    // Armed: the tile reports SET rather than the empty hint.
    expect(container.querySelector(".su-set-badge")).toBeTruthy();

    await click(container.querySelector(".su-generate"));

    const payload = context.createImageJob.mock.calls[0][0];
    expect(payload.referenceAssetId).toBe("asset-9");
    expect(payload.advanced.strength).toBe(0.5);
    expect(payload.sourceAssetId).toBeNull();
  });

  it("disables the Reference tile, with a reason, on a model that can't use one", async () => {
    const plainModel = { ...IMAGE_MODEL, id: "sdxl", name: "SDXL", ui: undefined };
    const context = baseContext({
      imageModels: [plainModel],
      models: [plainModel],
      assets: [REFERENCE_ASSET],
      recentImageAssets: [REFERENCE_ASSET],
    });
    await renderShell(root, context);

    const tile = [...container.querySelectorAll(".su-tile")].find((node) =>
      node.textContent.includes("Reference"),
    );
    expect(tile.disabled).toBe(true);
    expect(tile.textContent).toContain("SDXL can’t use a reference image");
  });

  // A Krea 2 edit run failed in the worker because Simple sent `loras: []`, and then the
  // failure was invisible: the studio rendered nothing and the reason was only in Logs.
  const KREA = {
    ...IMAGE_MODEL,
    id: "krea_2_turbo",
    name: "Krea 2 Turbo",
    family: "krea_2",
  };
  const KREA_LORA = {
    id: "krea2_identity_edit",
    name: "Krea 2 Identity Edit",
    conditioningRole: "image_edit",
    installState: "installed",
    compatibility: { families: ["krea_2"] },
    families: ["krea_2"],
  };

  it("auto-applies the managed edit LoRA on a Krea 2 edit run", async () => {
    const context = baseContext({
      imageModels: [KREA],
      models: [KREA],
      loras: [KREA_LORA],
      assets: [REFERENCE_ASSET],
      recentImageAssets: [REFERENCE_ASSET],
    });
    await renderShell(root, context);

    await click(buttonWithText(container, "Edit"));
    await typePrompt(container, "make it dusk");
    await click([...container.querySelectorAll(".su-tile")].find((n) => n.textContent.includes("Source image")));
    await click(
      [...container.querySelectorAll(".su-option-row")].find((n) => n.textContent.includes("cliff_ref.png")),
    );
    await click(container.querySelector(".su-generate"));

    const payload = context.createImageJob.mock.calls[0][0];
    expect(payload.mode).toBe("edit_image");
    expect(payload.sourceAssetId).toBe("asset-9");
    expect(payload.loras.map((l) => l.id)).toEqual(["krea2_identity_edit"]);
    expect(payload.loras[0].conditioningRole).toBe("image_edit");
  });

  it("blocks the run and offers the download when the edit LoRA isn't installed", async () => {
    const context = baseContext({
      imageModels: [KREA],
      models: [KREA],
      loras: [{ ...KREA_LORA, installState: "missing" }],
      assets: [REFERENCE_ASSET],
      recentImageAssets: [REFERENCE_ASSET],
    });
    await renderShell(root, context);
    await click(buttonWithText(container, "Edit"));
    await typePrompt(container, "make it dusk");

    expect(container.textContent).toContain("needs the Krea 2 Identity Edit LoRA");
    expect(container.querySelector(".su-generate").disabled).toBe(true);

    await click(buttonWithText(container, "Download it"));
    expect(context.createLoraDownloadJob).toHaveBeenCalledTimes(1);
    expect(context.createLoraDownloadJob.mock.calls[0][0].id).toBe("krea2_identity_edit");
  });

  it("surfaces a failed run's reason in the studio instead of rendering nothing", async () => {
    const failed = {
      id: "job-x",
      type: "image_generate",
      status: "failed",
      error: "Krea 2 edit requires the Krea 2 Identity Edit LoRA.",
      payload: {},
    };
    const context = baseContext({ jobs: [failed], imageLocalJobs: [failed] });
    await renderShell(root, context);

    expect(container.querySelector(".su-job-error").textContent).toBe(
      "Krea 2 edit requires the Krea 2 Identity Edit LoRA.",
    );
    expect(container.querySelector(".su-job-badge").textContent).toBe("Failed");
  });

  it("shows the failure reason on the Queue row too", async () => {
    const failed = {
      id: "job-x",
      type: "image_generate",
      status: "failed",
      error: "no worker supports image_generate",
      payload: { model: "z_image" },
    };
    await renderShell(root, baseContext({ jobs: [failed] }));
    await click(navButton(container, "Queue"));

    expect(container.querySelector(".su-job-error").textContent).toBe(
      "no worker supports image_generate",
    );
    expect(container.querySelector(".su-job-badge").textContent).toBe("Failed");
  });

  it("badges a canceled job as Canceled, not Failed", async () => {
    const canceled = {
      id: "job-c",
      type: "image_generate",
      status: "canceled",
      error: "Canceled before a worker started.",
      payload: { model: "z_image" },
    };
    await renderShell(root, baseContext({ jobs: [canceled] }));
    await click(navButton(container, "Queue"));
    expect(container.querySelector(".su-job-badge").textContent).toBe("Canceled");
  });

  // Simple had NO way to cancel a run and no progress readout anywhere — a submitted job
  // was a spinner with no end and no exit. The run strip under Generate is the same card
  // the Queue lists, so both surfaces get progress + Cancel from one component.
  const RUNNING_JOB = {
    id: "job-run",
    type: "image_generate",
    status: "running",
    progress: 0.42,
    payload: { model: "z_image", width: 1024, height: 1024 },
  };

  it("shows progress and a Cancel for the running job under Generate", async () => {
    const context = baseContext({ jobs: [RUNNING_JOB], imageLocalJobs: [RUNNING_JOB] });
    await renderShell(root, context);

    // The strip renders in the STUDIO, not only on the Queue screen.
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Image Studio");
    expect(container.querySelector(".su-job-badge").textContent).toBe("42%");
    expect(container.querySelector(".su-bar > span").style.width).toBe("42%");

    await click(buttonWithText(container, "Cancel"));
    expect(context.jobAction).toHaveBeenCalledTimes(1);
    expect(context.jobAction.mock.calls[0][0].id).toBe("job-run");
    expect(context.jobAction.mock.calls[0][1]).toBe("cancel");
  });

  it("offers Cancel on the Queue screen too", async () => {
    const context = baseContext({ jobs: [RUNNING_JOB] });
    await renderShell(root, context);
    await click(navButton(container, "Queue"));

    await click(buttonWithText(container, "Cancel"));
    expect(context.jobAction).toHaveBeenCalledWith(
      expect.objectContaining({ id: "job-run" }),
      "cancel",
    );
  });

  it("offers no Cancel once a run is terminal", async () => {
    const failed = { ...RUNNING_JOB, status: "failed", error: "out of memory" };
    await renderShell(root, baseContext({ jobs: [failed], imageLocalJobs: [failed] }));
    expect(container.querySelector(".su-job")).toBeTruthy();
    expect(buttonWithText(container, "Cancel")).toBeUndefined();
  });

  // The four terminal states are NOT interchangeable in a studio, which is exactly the
  // assumption worth pinning per-status rather than as one "terminal" case.
  //   completed → the results grid below IS the outcome
  //   canceled  → the user did it deliberately
  //   failed / interrupted → the card is the studio's ONLY carrier of the worker's reason
  const STUDIO_CARD_BY_STATUS = [
    ["completed", false],
    ["canceled", false],
    ["failed", true],
    ["interrupted", true],
  ];

  for (const [status, stays] of STUDIO_CARD_BY_STATUS) {
    it(`${stays ? "keeps" : "clears"} the studio run card for a ${status} run`, async () => {
      const job = { ...RUNNING_JOB, status, progress: 1, error: "out of memory" };
      await renderShell(root, baseContext({ jobs: [job], imageLocalJobs: [job] }));
      expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Image Studio");
      if (stays) {
        expect(container.querySelector(".su-job")).toBeTruthy();
        expect(container.querySelector(".su-job-error").textContent).toBe("out of memory");
      } else {
        expect(container.querySelector(".su-job")).toBeNull();
      }
    });

    it(`offers ${stays ? "a Dismiss" : "no Dismiss"} on the studio card for a ${status} run`, async () => {
      const job = { ...RUNNING_JOB, status, progress: 1, error: "out of memory" };
      await renderShell(root, baseContext({ jobs: [job], imageLocalJobs: [job] }));
      // Cancel and Dismiss are mutually exclusive — a terminal run can only be dismissed.
      expect(buttonWithText(container, "Cancel")).toBeUndefined();
      expect(Boolean(buttonWithText(container, "Dismiss"))).toBe(stays);
    });

    it(`always keeps the Queue row for a ${status} run`, async () => {
      const job = { ...RUNNING_JOB, status, progress: 1, error: "out of memory" };
      await renderShell(root, baseContext({ jobs: [job] }));
      await click(navButton(container, "Queue"));
      // The Queue is a history — it keeps every row regardless of outcome.
      expect(container.querySelector(".su-job")).toBeTruthy();
    });
  }

  it("dismisses a failed run's card from the studio without touching the Queue", async () => {
    const failed = { ...RUNNING_JOB, status: "failed", error: "out of memory" };
    await renderShell(root, baseContext({ jobs: [failed], imageLocalJobs: [failed] }));

    expect(container.querySelector(".su-job")).toBeTruthy();
    await click(buttonWithText(container, "Dismiss"));
    expect(container.querySelector(".su-job")).toBeNull();

    // The Queue is the history — dismissing in a studio must not remove it there.
    await click(navButton(container, "Queue"));
    expect(container.querySelector(".su-job")).toBeTruthy();
    expect(container.querySelector(".su-job-error").textContent).toBe("out of memory");
  });

  // The studios unmount on navigation, so a studio-local dismissal flag would resurrect the
  // card on the way back. This is why the dismissed set lives on the shell.
  it("keeps a dismissal after navigating away and back", async () => {
    const failed = { ...RUNNING_JOB, status: "failed", error: "out of memory" };
    await renderShell(root, baseContext({ jobs: [failed], imageLocalJobs: [failed] }));

    await click(buttonWithText(container, "Dismiss"));
    await click(navButton(container, "Queue"));
    await click(navButton(container, "Image"));

    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Image Studio");
    expect(container.querySelector(".su-job")).toBeNull();
  });

  it("sweeps an indeterminate bar for a claimed run with no percentage yet", async () => {
    const noProgress = { ...RUNNING_JOB, progress: 0 };
    await renderShell(root, baseContext({ jobs: [noProgress], imageLocalJobs: [noProgress] }));
    // A 0%-wide bar reads as "stuck"; the sweeping variant reads as "working".
    expect(container.querySelector(".su-bar").className).toContain("su-bar--indeterminate");
    expect(container.querySelector(".su-job-badge").textContent).toBe("Running");
  });

  // The audit's headline finding, pinned end-to-end: the studio must SEED from the model's
  // declared default, not the first allowed value. z_image declares 1024² while its limits
  // list leads with 768², so the wrong seed silently downgraded every Simple run of it.
  it("seeds the resolution from the model's declared default, not limits[0]", async () => {
    // Rendered as the FIRST model this shell ever sees. A prior render would leave a
    // still-valid resolution in state, which correctly sticks (switching model keeps a
    // selection the new model also allows — same as the advanced studio) and would mask
    // what the seed actually chose. That stickiness is exactly why a click-through in the
    // browser missed this bug.
    // Declared default, first-allowed, and the 1024² fallback are three DISTINCT values here,
    // so this can only pass by actually reading `defaults.resolution`. (A model declaring
    // 1024² — which every mismatching shipped model happens to do — would pass even with the
    // bug, because the fallback lands on the same answer.)
    const leadsLow = {
      ...IMAGE_MODEL,
      defaults: { resolution: "1344x768" },
      limits: { resolutions: ["768x768", "1024x1024", "1344x768"] },
    };
    const lowContext = baseContext({ imageModels: [leadsLow], models: [leadsLow] });
    await renderShell(root, lowContext);

    expect(
      [...container.querySelectorAll(".su-select")].some((n) => n.textContent.includes("1344×768")),
    ).toBe(true);

    await typePrompt(container, "a lighthouse");
    await click(container.querySelector(".su-generate"));
    const payload = lowContext.createImageJob.mock.calls[0][0];
    expect(payload.width).toBe(1344);
    expect(payload.height).toBe(768);
  });

  it("refuses to submit with an empty prompt", async () => {
    const context = baseContext();
    await renderShell(root, context);
    expect(container.querySelector(".su-generate").disabled).toBe(true);
    await click(container.querySelector(".su-generate"));
    expect(context.createImageJob).not.toHaveBeenCalled();
  });

  it("locks the mode switch and explains why when the viewport forces Simple", async () => {
    await renderShell(root, baseContext(), { lockedToSimple: true });
    const [simple, advanced] = container.querySelectorAll(".su-nav-footer .su-switch button");
    expect(simple.disabled).toBe(true);
    expect(advanced.disabled).toBe(true);
    expect(container.textContent).toContain("Phones always use the Simple UI.");
  });

  // "Manage" points the WORKSPACE at its Models screen — which is invisible while the
  // Simple shell is rendering. So it has to flip the shell too, or the button is inert.
  it("hands 'Manage' off to the full Models screen AND switches shells", async () => {
    const context = baseContext();
    const onModeChange = vi.fn();
    await renderShell(root, context, { onModeChange });
    await click(navButton(container, "Model Manager"));

    const manage = buttonWithText(container, "Manage");
    expect(manage).toBeTruthy();
    await click(manage);

    expect(context.setActiveView).toHaveBeenCalledWith("Models");
    expect(onModeChange).toHaveBeenCalledWith("advanced");
  });

  it("refuses the hand-off with a reason when the viewport locks Simple on", async () => {
    const context = baseContext();
    const onModeChange = vi.fn();
    await renderShell(root, context, { onModeChange, lockedToSimple: true });
    await click(navButton(container, "Model Manager"));
    await click(buttonWithText(container, "Manage"));

    expect(context.setActiveView).not.toHaveBeenCalled();
    expect(onModeChange).not.toHaveBeenCalled();
    expect(container.querySelector(".su-toast").textContent).toContain("Advanced workspace");
  });

  it("enqueues a real download for a model that isn't installed", async () => {
    const context = baseContext({
      models: [IMAGE_MODEL, { ...IMAGE_MODEL, id: "flux_dev", name: "FLUX.1 [dev]", installState: "missing" }],
    });
    await renderShell(root, context);
    await click(navButton(container, "Model Manager"));
    await click(buttonWithText(container, "Download"));

    expect(context.createModelDownloadJob).toHaveBeenCalledTimes(1);
    expect(context.createModelDownloadJob.mock.calls[0][0].id).toBe("flux_dev");
  });

  it("reports the mode switch to the host app", async () => {
    const onModeChange = vi.fn();
    await renderShell(root, baseContext(), { onModeChange });
    const advanced = [...container.querySelectorAll(".su-nav-footer .su-switch button")].find(
      (node) => node.textContent === "Advanced",
    );
    await click(advanced);
    expect(onModeChange).toHaveBeenCalledWith("advanced");
  });
});
