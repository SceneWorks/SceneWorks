import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { withImageStudioContext, FakeEventSource, response, settle } from "./main.testSupport.jsx";

// sc-18255: picking an img2img reference auto-presets the Aspect to the bucket closest to the
// reference's own aspect (sc-10195). That snap must fire ONCE per picked reference — the effect
// also depends on the asset list, whose array identity changes on every refresh (a generation
// completing, an SSE asset update, submitting a job). An unguarded re-run re-probed the reference
// and clobbered the Aspect the user had since chosen, so a Krea 2 Turbo job configured at
// 768x1024 was submitted at 1152x2048 (the 1080x1920 reference's closest bucket).
const REFERENCE_ASSET = {
  id: "asset-reference",
  projectId: "project-1",
  type: "image",
  displayName: "Portrait reference",
  file: { path: "assets/uploads/portrait.jpg", mimeType: "image/jpeg", width: 1080, height: 1920 },
  status: { favorite: false, rating: 0, rejected: false, trashed: false },
};

const KREA_MODEL = {
  id: "krea_2_turbo",
  name: "Krea 2 Turbo",
  type: "image",
  family: "krea_2",
  installed: true,
  defaults: { resolution: "1024x1024", steps: 8 },
  limits: { resolutions: ["1024x1024", "768x1024", "1152x2048"] },
  // Both capabilities, like the real entry — the mode round-trip below needs the model to keep
  // serving Edit mode so the option set stays stable across the trip.
  capabilities: ["text_to_image", "edit_image"],
  ui: { img2img: true, img2imgStrength: { default: 0.5, min: 0, max: 1, step: 0.05 } },
};

// A second img2img model with its OWN bucket list, for the model-switch case: the snap key is the
// option set, so switching here must re-snap the reference onto this model's closest bucket rather
// than leaving the model-defaults reset (1024x1024) standing against a 9:16 reference.
const OTHER_MODEL = {
  ...KREA_MODEL,
  id: "z_image_turbo",
  name: "Z-Image",
  family: "z-image",
  limits: { resolutions: ["1024x1024", "832x1216", "896x1600"] },
};

// A model with NO img2img panel. `img2imgReferenceAssetId` is never cleared on a model change, so a
// reference armed elsewhere must stop steering the Aspect the moment the panel is gone.
const NO_IMG2IMG_MODEL = {
  ...OTHER_MODEL,
  id: "flux2_klein_9b",
  name: "FLUX.2 klein",
  family: "flux2",
  ui: {},
};

// jsdom's Image never loads, so probeImageDimensions would never fire. Stand in a version that
// reports the reference's natural size on the next microtask, like a real decode would.
function stubImageDecode(naturalWidth, naturalHeight) {
  class ProbeImage {
    constructor() {
      this.naturalWidth = 0;
      this.naturalHeight = 0;
      this.onload = null;
      this.onerror = null;
    }
    set src(value) {
      this._src = value;
      if (!value) return;
      Promise.resolve().then(() => {
        this.naturalWidth = naturalWidth;
        this.naturalHeight = naturalHeight;
        this.onload?.();
      });
    }
    get src() {
      return this._src;
    }
  }
  global.Image = ProbeImage;
}

describe("Image Studio img2img reference aspect snap", () => {
  let container;
  let root;
  let createImageJob;
  const originalImage = global.Image;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    global.fetch = vi.fn(() => Promise.resolve(response([])));
    createImageJob = vi.fn();
    stubImageDecode(1080, 1920);
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    global.Image = originalImage;
    vi.restoreAllMocks();
  });

  // The prompt-tool tiles concatenate a title and a description into one button, so match on the
  // leading title rather than the whole node.
  const buttonByText = (text) =>
    [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.trim().startsWith(text),
    );
  const aspectSelect = () => document.body.querySelector(".settings-field-aspect select");
  const enterPrompt = async (label, value) => {
    await act(async () => {
      const field = document.querySelector(`textarea[aria-label="${label}"]`);
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set.call(field, value);
      field.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await settle();
  };

  async function pickReference() {
    await act(async () => {
      buttonByText("Select reference image").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".asset-picker-modal .asset-picker-card").click();
    });
    await settle();
    await act(async () => {
      buttonByText("Use Selection").click();
    });
    await settle();
  }

  function studio(assets, imageModels = [KREA_MODEL]) {
    return withImageStudioContext({
      activeProject: { id: "project-1", name: "Noir" },
      assets,
      characters: [],
      createImageJob,
      deleteAsset: () => {},
      purgeAsset: () => {},
      gpuOptions: ["auto"],
      imageModels,
      latestAssets: [],
      localJobs: [],
      loras: [],
      onPreview: () => {},
      requestedGpu: "auto",
      selectedAsset: null,
      setRequestedGpu: () => {},
      updateAssetStatus: () => {},
    });
  }

  it("snaps once on pick, then keeps the user's Aspect across asset-list refreshes (sc-18255)", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(studio([REFERENCE_ASSET]));
    });
    await settle();

    await act(async () => {
      buttonByText("Image reference").click();
    });
    await settle();
    await pickReference();

    // The 1080x1920 reference snaps the Aspect to its closest declared bucket.
    expect(aspectSelect().value).toBe("1152x2048");

    // The user then picks a different Aspect for this render.
    await act(async () => {
      const select = aspectSelect();
      select.value = "768x1024";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await settle();
    expect(aspectSelect().value).toBe("768x1024");

    // An asset-list refresh (new array identity, same reference still picked) must not re-snap.
    await act(async () => {
      root.render(studio([{ ...REFERENCE_ASSET }]));
    });
    await settle();
    expect(aspectSelect().value).toBe("768x1024");

    // Nor may any number of further refreshes.
    await act(async () => {
      root.render(studio([{ ...REFERENCE_ASSET }, { ...REFERENCE_ASSET, id: "asset-generated" }]));
    });
    await settle();
    expect(aspectSelect().value).toBe("768x1024");
  });

  it("names a deleted image reference before Generate and lets the user remove it", async () => {
    root = createRoot(container);
    await act(async () => { root.render(studio([REFERENCE_ASSET])); });
    await settle();
    await act(async () => { buttonByText("Image reference").click(); });
    await pickReference();
    await enterPrompt("Prompt", "A portrait");
    expect(buttonByText("Generate").disabled).toBe(false);

    await act(async () => { root.render(studio([])); });
    await settle();
    const summary = document.querySelector('[aria-label="Active image reference"]');
    expect(summary.textContent).toContain("Image reference: On");
    expect(summary.textContent).toContain("Portrait reference");
    expect(summary.textContent).toContain("assets/uploads/portrait.jpg");
    expect(summary.textContent).toContain("Remove it or select another reference image");
    expect(buttonByText("Generate").disabled).toBe(true);
    await act(async () => {
      document.querySelector("form.studio-shell").dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(createImageJob).not.toHaveBeenCalled();

    await act(async () => { buttonByText("Remove image reference").click(); });
    await settle();
    expect(document.querySelector('[aria-label="Active image reference"]')).toBeNull();
    expect(buttonByText("Generate").disabled).toBe(false);
  });

  it("keeps the reference visible in batch mode and blocks the batch when its file is missing", async () => {
    global.fetch = vi.fn((_url, options) => Promise.resolve(options?.method === "HEAD"
      ? { ok: false, status: 404 } : response([])));
    root = createRoot(container);
    await act(async () => { root.render(studio([REFERENCE_ASSET])); });
    await settle();
    await act(async () => { buttonByText("Image reference").click(); });
    await pickReference();
    await act(async () => { buttonByText("Batch").click(); });
    await enterPrompt("Batch prompts", "A portrait\nAnother portrait");
    const summary = document.querySelector('[aria-label="Active image reference"]');
    expect(summary.textContent).toContain("Portrait reference");
    expect(summary.textContent).toContain("original file");
    expect(buttonByText("Run batch").disabled).toBe(true);
    expect(global.fetch.mock.calls.some(([, options]) => options?.method === "HEAD")).toBe(true);

    // Repair is checked explicitly, and does not replace the chosen reference or batch.
    global.fetch = vi.fn(() => Promise.resolve(response([])));
    await act(async () => { buttonByText("Retry reference check").click(); });
    await settle();
    expect(buttonByText("Run batch").disabled).toBe(false);
    expect(document.querySelector('textarea[aria-label="Batch prompts"]').value).toBe("A portrait\nAnother portrait");
    await act(async () => { buttonByText("Remove image reference").click(); });
    await settle();
    expect(buttonByText("Run batch").disabled).toBe(false);
  });

  it.each(["trashed", "rejected"])("blocks a %s reference even while its catalog record exists", async (status) => {
    root = createRoot(container);
    await act(async () => { root.render(studio([REFERENCE_ASSET])); });
    await settle();
    await act(async () => { buttonByText("Image reference").click(); });
    await pickReference();
    await enterPrompt("Prompt", "A portrait");
    await act(async () => { root.render(studio([{ ...REFERENCE_ASSET, status: { [status]: true } }])); });
    await settle();
    expect(buttonByText("Generate").disabled).toBe(true);
    expect(document.querySelector('[aria-label="Active image reference"]').textContent).toContain("no longer available");
  });

  it("blocks while checking and ignores an old check after the reference is removed", async () => {
    let resolveHead;
    global.fetch = vi.fn((_url, options) => options?.method === "HEAD"
      ? new Promise((resolve) => { resolveHead = resolve; }) : Promise.resolve(response([])));
    root = createRoot(container);
    await act(async () => { root.render(studio([REFERENCE_ASSET])); });
    await settle();
    await act(async () => { buttonByText("Image reference").click(); });
    await pickReference();
    await enterPrompt("Prompt", "A portrait");
    expect(buttonByText("Generate").disabled).toBe(true);
    expect(document.querySelector('[aria-label="Active image reference"]').textContent).toContain("Checking the reference image");
    await act(async () => { buttonByText("Remove image reference").click(); });
    await settle();
    await act(async () => { resolveHead({ ok: false, status: 404 }); });
    await settle();
    expect(buttonByText("Generate").disabled).toBe(false);
    expect(document.querySelector('[aria-label="Active image reference"]')).toBeNull();
  });

  it("blocks a reference whose original file cannot decode", async () => {
    root = createRoot(container);
    await act(async () => { root.render(studio([REFERENCE_ASSET])); });
    await settle();
    global.Image = class {
      set src(value) { if (value) Promise.resolve().then(() => this.onerror?.()); }
    };
    await act(async () => { buttonByText("Image reference").click(); });
    await pickReference();
    await enterPrompt("Prompt", "A portrait");
    expect(buttonByText("Generate").disabled).toBe(true);
    expect(document.querySelector('[aria-label="Active image reference"]').textContent).toContain("original file for “Portrait reference");
  });

  it.each(["tiff", "heic", "x-exr"])("accepts a readable %s reference without requiring a browser codec", async (format) => {
    const asset = { ...REFERENCE_ASSET, file: { path: `assets/uploads/reference.${format}`, mimeType: `image/${format}` } };
    global.Image = class {
      set src(value) { if (value) Promise.resolve().then(() => this.onerror?.()); }
    };
    root = createRoot(container);
    await act(async () => { root.render(studio([asset])); });
    await settle();
    await act(async () => { buttonByText("Image reference").click(); });
    await pickReference();
    await enterPrompt("Prompt", "A portrait");
    expect(buttonByText("Generate").disabled).toBe(false);
    expect(document.querySelector('[aria-label="Active image reference"] [role="alert"]')).toBeNull();
  });

  it("re-snaps when the reference is cleared and re-picked (sc-18255)", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(studio([REFERENCE_ASSET]));
    });
    await settle();

    await act(async () => {
      buttonByText("Image reference").click();
    });
    await settle();
    await pickReference();
    expect(aspectSelect().value).toBe("1152x2048");

    await act(async () => {
      const select = aspectSelect();
      select.value = "768x1024";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await settle();

    await act(async () => {
      buttonByText("Remove").click();
    });
    await settle();
    await pickReference();

    // Re-picking is a deliberate user action, so the aspect preset applies again.
    expect(aspectSelect().value).toBe("1152x2048");
  });

  it("re-snaps when the resolution options themselves change (sc-18255)", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(studio([REFERENCE_ASSET]));
    });
    await settle();

    await act(async () => {
      buttonByText("Image reference").click();
    });
    await settle();
    await pickReference();
    expect(aspectSelect().value).toBe("1152x2048");

    // The reference is still armed, so the snap must run again on the NEW bucket list
    // when switching models, preserving the reference's 9:16 composition.
    const select = document.body.querySelector(".settings-field-model select");
    await act(async () => {
      root.render(studio([REFERENCE_ASSET], [KREA_MODEL, OTHER_MODEL]));
    });
    await settle();
    await act(async () => {
      select.value = "z_image_turbo";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await settle();

    expect(aspectSelect().value).toBe("896x1600");
  });

  it("stops steering the Aspect on a model with no img2img panel (sc-18255)", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(studio([REFERENCE_ASSET], [KREA_MODEL, NO_IMG2IMG_MODEL]));
    });
    await settle();

    await act(async () => {
      buttonByText("Image reference").click();
    });
    await settle();
    await pickReference();
    expect(aspectSelect().value).toBe("1152x2048");

    // The panel owning the reference is gone. Resolution continuity still preserves the
    // previous composition, but the invisible reference must not steer later manual choices.
    const select = document.body.querySelector(".settings-field-model select");
    await act(async () => {
      select.value = "flux2_klein_9b";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await settle();

    expect(aspectSelect().value).toBe("896x1600");
    await act(async () => {
      aspectSelect().value = "1024x1024";
      aspectSelect().dispatchEvent(new Event("change", { bubbles: true }));
    });
    await settle();
    expect(aspectSelect().value).toBe("1024x1024");
  });

  it("keeps the user's Aspect across a Text → Edit → Text round-trip (sc-18255)", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(studio([REFERENCE_ASSET]));
    });
    await settle();

    await act(async () => {
      buttonByText("Image reference").click();
    });
    await settle();
    await pickReference();
    expect(aspectSelect().value).toBe("1152x2048");

    await act(async () => {
      const select = aspectSelect();
      select.value = "768x1024";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await settle();

    // Krea 2 Turbo serves both text_to_image and edit_image, so this changes neither the model nor
    // the resolution options. The img2img panel is inoperative in Edit mode — but the reference is
    // still armed, and coming back must NOT re-snap over the Aspect the user chose.
    await act(async () => {
      buttonByText("Edit").click();
    });
    await settle();
    await act(async () => {
      buttonByText("Text").click();
    });
    await settle();

    expect(aspectSelect().value).toBe("768x1024");
  });
});
