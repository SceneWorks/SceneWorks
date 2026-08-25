// sc-17161 — MiniMax-H3 in the SIMPLE video studio.
//
// Simple is an alternative shell, not a subset view: it renders its own controls from the same
// catalog, so a licence obligation discharged only in Advanced is undischarged for every Simple
// user, and a duration label fixed only in Advanced still reads "5.1667s" on a chip here.
//
// What Simple deliberately does NOT get is the other ten video modes — it exposes Text → Video and
// Image → Video only, by design (`SimpleVideoStudio.jsx`), so Ref2VA stays an Advanced surface.
// That is asserted rather than assumed: it is the reason the reference pickers are absent here.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";
import { readSimpleStudioSnapshot } from "../simpleStudioStore.js";

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
const MINIMAX = {
  ...manifestModels.find((model) => model.id === "minimax_h3"),
  installState: "installed",
  usable: true,
};

/// The shipped 768p turbo entry as the catalog route serves it — the manifest entry (so the step
/// count and the declared `modelIds` partition are the real ones) plus the install state
/// `normalize_lora_entry` probes.
function turboFixture() {
  return {
    ...JSON5.parse(
      readFileSync(resolve(HERE, "../../../../config/manifests/builtin.loras.jsonc"), "utf8"),
    ).loras.find((lora) => lora.id === "minimax_h3_turbo_4step_768p"),
    scope: "builtin",
    installState: "installed",
    installedPath: "/data/loras/minimax_h3_turbo_4step_768p.safetensors",
  };
}

function baseContext() {
  return {
    activeProject: { id: "project-1", name: "Default" },
    assets: [],
    recentImageAssets: [],
    recentVideoAssets: [],
    jobs: [],
    imageModels: [],
    videoModels: [MINIMAX],
    audioModels: [],
    models: [MINIMAX],
    loras: [],
    imageLocalJobs: [],
    videoLocalJobs: [],
    audioLocalJobs: [],
    visibleWorkers: [],
    macCapabilities: null,
    theme: "light",
    changeTheme: () => {},
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createVideoJob: vi.fn(async () => ({ id: "job-1" })),
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
  };
}

describe("SimpleVideoStudio with MiniMax-H3 (sc-17161)", () => {
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

  async function openVideo(context) {
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
    const nav = [...container.querySelectorAll(".su-nav-item")].find((node) =>
      node.textContent.startsWith("Video"),
    );
    await click(nav);
  }

  it("carries the licence-required attribution", async () => {
    await openVideo(baseContext());
    const line = container.querySelector(".model-attribution");
    expect(line, "Simple is a shell, not a subset — §IV.2 applies here too").toBeTruthy();
    expect(line.textContent).toBe(MINIMAX.ui.attribution);
    expect(line.textContent).toContain("MiniMax H3");
  });

  it("labels the fourteen lattice rungs legibly and offers no 15s chip", async () => {
    await openVideo(baseContext());
    const chips = [...container.querySelectorAll("button")].filter((node) =>
      /^\d+(\.\d+)?s$/.test(node.textContent.trim()),
    );
    const labels = chips.map((chip) => chip.textContent.trim());
    expect(labels).toEqual([
      "5.17s",
      "5.88s",
      "6.58s",
      "7.29s",
      "8s",
      "8.71s",
      "9.42s",
      "10.13s",
      "10.83s",
      "11.54s",
      "12.25s",
      "12.96s",
      "13.67s",
      "14.38s",
    ]);
    // Raw values would read "5.1667s" — the label is rounded, the underlying value is not.
    expect(labels.some((label) => label.includes("1667"))).toBe(false);
    expect(labels).not.toContain("15s");
    expect(labels).toHaveLength(MINIMAX.limits.durations.length);
  });

  it("renders the model's declared duration hint, not just the chips (sc-17162)", async () => {
    // `ui.durationHint` had ONE reader (Advanced's helper copy), so the refusal it carries — "15s is
    // refused rather than shortened" — reached no Simple user at all. Read back out of the DOM and
    // compared against the MANIFEST string rather than a typed literal, so copy edits travel and a
    // hint that renders empty is red.
    await openVideo(baseContext());
    const hint = container.querySelector(".su-duration-hint");
    expect(hint, "Simple must carry the model's duration copy, not only its chips").toBeTruthy();
    expect(hint.textContent).toBe(MINIMAX.ui.durationHint);
    expect(hint.textContent).toContain("15s is refused");
  });

  it("renders the model's declared description, not just its name (sc-17162)", async () => {
    // `ui.description` had ONE reader in the whole app — the advanced Models screen's card. Simple's
    // only route there is "Manage", which switches the SHELL, so a Simple user identified the model
    // by NAME ALONE and never met the withheld-component disclosure the string carries: not the
    // hosted Hailuo product, 2K unreachable, dense attention. Same gap and same fix as the duration
    // hint above.
    //
    // Read back out of the DOM and compared against the MANIFEST string, not a typed literal, so
    // copy edits travel and a description that renders empty is red. The two extra assertions pin
    // the load-bearing halves of the disclosure specifically: a rewrite that keeps the field
    // non-empty but drops what the withheld components cost would otherwise stay green.
    await openVideo(baseContext());
    const description = container.querySelector(".su-model-desc");
    expect(description, "Simple must carry the model's description, not only its name").toBeTruthy();
    expect(description.textContent).toBe(MINIMAX.ui.description);
    expect(description.textContent).toContain("H3-Regenerate-2K");
    expect(description.textContent).toContain("not the hosted Hailuo product");
    // It must sit BELOW the licence attribution, which is an obligation to display prominently and
    // outranks descriptive copy. Asserted by document order rather than by CSS.
    const attribution = container.querySelector(".model-attribution");
    expect(attribution, "the licence attribution must still render").toBeTruthy();
    expect(
      attribution.compareDocumentPosition(description) & Node.DOCUMENT_POSITION_FOLLOWING,
      "the description must follow the licence attribution, not precede it",
    ).toBeTruthy();
  });

  it("sends the exact declared duration, not the rounded label", async () => {
    const context = baseContext();
    await openVideo(context);
    const textarea = container.querySelector("textarea");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value",
      ).set;
      setter.call(textarea, "a lighthouse in a storm");
      textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    const generate = [...container.querySelectorAll("button")].find((node) =>
      /generate|create|render/i.test(node.textContent.trim()),
    );
    await click(generate);
    expect(context.createVideoJob).toHaveBeenCalled();
    const payload = context.createVideoJob.mock.calls[0][0];
    // `defaults.duration`, byte for byte — the enqueue gate matches `limits.durations` exactly.
    expect(payload.duration).toBe(MINIMAX.defaults.duration);
    expect(MINIMAX.limits.durations).toContain(payload.duration);
  });

  // sc-18727 — the turbo DEFAULT reaches Simple, even though the turbo CONTROL deliberately does
  // not. Simple exposes no advanced sampler knobs, so a user here can never turn turbo on by hand;
  // if the seed did not run, every Simple MiniMax-H3 render would be the 2 h 25 m one that
  // sc-18729 measured, with nothing on screen explaining why.
  //
  // Asserted on the PAYLOAD, not on any control: there is nothing to click, so the only observable
  // difference between "seeded" and "not seeded" is the job that gets submitted.
  it("seeds the turbo adapter into the submitted job, with no control to set it", async () => {
    const turbo = turboFixture();
    // 4 against the model's own 50 — a non-default recipe, so this cannot pass against a shell that
    // simply forwards the model defaults.
    expect(turbo.sampling.steps).toBe(4);
    expect(MINIMAX.defaults.steps).toBe(50);

    const context = { ...baseContext(), loras: [turbo] };
    await openVideo(context);
    const textarea = container.querySelector("textarea");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value",
      ).set;
      setter.call(textarea, "a lighthouse in a storm");
      textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    await click(
      [...container.querySelectorAll("button")].find((node) =>
        /generate|create|render/i.test(node.textContent.trim()),
      ),
    );
    const payload = context.createVideoJob.mock.calls[0][0];
    expect((payload.loras ?? []).map((lora) => lora.id)).toContain(turbo.id);
    // …and it is not counted against the user's own LoRA budget: it is a builtin, which is the one
    // slot of headroom MAX_USER_JOB_LORAS (4) leaves inside MAX_JOB_LORAS_TOTAL (5).
    expect(payload.loras.find((lora) => lora.id === turbo.id).scope).toBe("builtin");
  });

  // sc-18727 — the other half of a default-on seed: it has to be turn-off-able.
  //
  // Simple ships NO turbo control, so deselecting the adapter in the LoRA picker is the only way a
  // Simple user can decline it. The seed effect re-runs on every `selectedLoraIds` change, so
  // without the one-shot marker (`turboSeededModels`) the removal would re-fire the seed and put
  // the adapter straight back — turbo would be literally un-turn-off-able from this shell, with no
  // error and no control to explain it. Nothing about that is visible in the seeding test above,
  // which is why it needs its own.
  //
  // Both halves of the marker are exercised: the IN-SESSION half (the effect must not re-add on the
  // very next render) and the DURABLE half (Simple's studio snapshot is mirrored to server
  // ui-preferences, so a relaunch that read back only `selectedLoraIds` would re-seed on every
  // launch and re-defeat the same "Off").
  it("does not re-add the turbo adapter after it is removed from the picker, across a relaunch", async () => {
    const turbo = turboFixture();
    const generate = async (context) => {
      const textarea = container.querySelector("textarea");
      await act(async () => {
        const setter = Object.getOwnPropertyDescriptor(
          window.HTMLTextAreaElement.prototype,
          "value",
        ).set;
        setter.call(textarea, "a lighthouse in a storm");
        textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
      });
      await click(
        [...container.querySelectorAll("button")].find((node) =>
          /generate|create|render/i.test(node.textContent.trim()),
        ),
      );
      return (context.createVideoJob.mock.calls.at(-1)[0].loras ?? []).map((lora) => lora.id);
    };

    const context = { ...baseContext(), loras: [turbo] };
    await openVideo(context);
    // Seeded, and shown as a normal selected LoRA — which is what makes the picker's own Remove
    // button the off switch.
    const remove = container.querySelector(`button[aria-label="Remove ${turbo.name}"]`);
    expect(remove, "the seeded adapter must be visible as a removable selection").toBeTruthy();
    await click(remove);
    expect(container.querySelector(`button[aria-label="Remove ${turbo.name}"]`)).toBeNull();
    expect(await generate(context)).not.toContain(turbo.id);

    // The snapshot the next launch reads. `writeSimpleStudioSnapshot` is debounced, so the unmount
    // flush is what makes the stored value observable (same rule as useStudioState.test.jsx).
    await unmountRoot(root, container);
    const stored = readSimpleStudioSnapshot("project-1");
    expect(stored.video.turboSeededModels).toContain(MINIMAX.id);
    expect(stored.video.selectedLoraIds ?? []).not.toContain(turbo.id);

    // Relaunch: a brand-new shell restoring that snapshot. The marker is what stops the seed from
    // firing again against an empty selection.
    ({ container, root } = mountRoot());
    const relaunched = { ...baseContext(), loras: [turbo] };
    await openVideo(relaunched);
    expect(container.querySelector(`button[aria-label="Remove ${turbo.name}"]`)).toBeNull();
    expect(await generate(relaunched)).not.toContain(turbo.id);
  });
});
