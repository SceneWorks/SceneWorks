// sc-25908 (epic 25899) — Qwen Image 2.1 Turbo on the catalog-driven surfaces outside the studios:
// the Model Manager tier panel and licence accept, the tier floors the Simple Model Manager quotes,
// the Training Studio target picker, and the rewriter gate's agreement with the worker.
//
// Every model object is projected from the SHIPPED manifest the way `GET /models` projects it, so
// the catalog is what is under test; the training targets are the Rust registry's checked-in
// contract fixture (`crates/sceneworks-core/tests/training_contracts.rs` pins it to the registry).

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import { createRoot } from "react-dom/client";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click } from "./testUtils/dom.js";
import { QWEN_IMAGE_2_1_MODEL_IDS, qwenRewriterAppliesTo } from "./constants.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../..");
const manifestModels = JSON5.parse(
  readFileSync(resolve(REPO, "config/manifests/builtin.models.jsonc"), "utf8"),
).models;
const entryOf = (id) => manifestModels.find((model) => model.id === id);
const TURBO_ENTRY = entryOf("qwen_image_2_1_turbo");
const BASE_ENTRY = entryOf("qwen_image_2_1");

// The `/models` projection of a manifest entry's tier matrix (apps/rust-api/src/models.rs): a
// `pendingArtifact` row is still a variant, in the API's `pending` install state.
function catalogModel(entry) {
  const { downloads, ...rest } = entry;
  return {
    ...rest,
    installState: "missing",
    installed: false,
    downloadable: true,
    hasVariantMatrix: true,
    variants: downloads
      .filter((download) => download.coRequisite !== true && download.variant)
      .map((download) => ({
        variant: download.variant,
        pendingArtifact: download.pendingArtifact === true,
        installState: download.pendingArtifact === true ? "pending" : "missing",
        cacheState: "missing",
        tierDeletable: download.pendingArtifact !== true,
        downloadSizeBytes: download.estimatedSizeBytes,
        footprint: download.footprint ?? null,
      })),
  };
}

describe("the Qwen rewriter gate agrees with the worker (sc-25908)", () => {
  // The worker serves a rewrite only for `sceneworks_core::jobs_store::QWEN_IMAGE_2_1_MODEL_IDS`;
  // the UI mirrors that id list. The two must name the same models, or the UI either hides a
  // working rewriter (Turbo, before this story) or offers one the worker refuses.
  // *Mutation that reds this:* dropping or adding an id on either side.
  it("offers the rewriter for exactly the models the worker accepts as a rewrite target", () => {
    const routing = readFileSync(resolve(REPO, "crates/sceneworks-core/src/jobs_store/routing.rs"), "utf8");
    const declared = /pub const QWEN_IMAGE_2_1_MODEL_IDS: &\[&str\] = &\[([^\]]*)\];/.exec(routing);
    expect(declared, "the Rust rewrite-target list is where this test reads it").toBeTruthy();
    const rustIds = [...declared[1].matchAll(/"([^"]+)"/g)].map((match) => match[1]).sort();
    expect([...QWEN_IMAGE_2_1_MODEL_IDS].sort()).toEqual(rustIds);
    const offered = manifestModels.filter(qwenRewriterAppliesTo).map((model) => model.id).sort();
    expect(offered).toEqual(rustIds);
    expect(offered).toContain("qwen_image_2_1_turbo");
    expect(qwenRewriterAppliesTo(entryOf("qwen_image"))).toBe(false);
    // An imported checkpoint on the same family is NOT a rewrite target: the worker refuses it.
    // *Mutation that reds this:* gating on `family === "qwen-image-2-1"`.
    expect(qwenRewriterAppliesTo({ id: "custom_qwen21_finetune", family: "qwen-image-2-1" })).toBe(false);
  });

  it("points Turbo at its own prompt guide, which ships", () => {
    const guide = TURBO_ENTRY.ui.promptGuide;
    expect(guide.path).toBe("/prompt-guides/qwen-image-2-1-turbo.md");
    expect(guide.path).not.toBe(BASE_ENTRY.ui.promptGuide.path);
    const text = readFileSync(resolve(REPO, "apps/web/public", guide.path.slice(1)), "utf8");
    expect(text).toContain("8-step schedule");
  });
});

describe("Model Manager with Qwen Image 2.1 Turbo (sc-25908)", () => {
  let container;
  let root;
  let ModelManagerScreen;
  let AppContext;
  let createModelDownloadJob;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    delete window.__TAURI__;
    createModelDownloadJob = vi.fn(async () => ({ id: "job", payload: {} }));
    vi.resetModules();
    vi.doMock("./api.js", () => ({
      apiFetch: vi.fn(async (path) =>
        path === "/api/v1/host-capabilities"
          ? { platform: "macos", memoryKind: "unified", memoryGb: 64 }
          : [],
      ),
      isAbortError: () => false,
      API_BASE_URL: "",
      eventUrl: () => "",
    }));
    ({ AppContext } = await import("./context/AppContext.js"));
    ({ ModelManagerScreen } = await import("./screens/ModelManagerScreen.jsx"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.doUnmock("./api.js");
    vi.restoreAllMocks();
  });

  async function render(models) {
    const value = {
      activeProject: null,
      jobs: [],
      loras: [],
      models,
      macCapabilities: { macGatingActive: true },
      presets: [],
      jobAction: () => {},
      setActiveView: () => {},
      deleteLora: () => {},
      deleteModel: () => {},
      deleteModelVariant: () => {},
      createModelDownloadJob,
      createModelConvertJob: () => {},
      createLoraImportJob: () => {},
      createModelImportJob: () => {},
    };
    await act(async () => {
      root.render(
        <AppContext.Provider value={value}>
          <ModelManagerScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const rowFor = (label) =>
    [...container.querySelectorAll(".model-tier-row")].find((row) =>
      row.querySelector(".model-tier-label").textContent.includes(label),
    );

  // q8/q4 are PUBLISHED (sc-25909): all three tiers install, each behind the same accept.
  // *Mutation that reds this:* dropping `requiresLicenseAcknowledgment` from the Turbo entry (a
  // checkbox is then live with no accept), or restoring `pendingArtifact` on q8/q4.
  it("lists three installable tiers behind Turbo's own licence accept", async () => {
    await render([catalogModel(TURBO_ENTRY)]);
    expect(container.querySelectorAll(".model-tier-row")).toHaveLength(3);
    const input = (label) => rowFor(label).querySelector("input");
    for (const label of ["bf16", "Q8", "Q4"]) {
      expect(rowFor(label).textContent).not.toContain("not published yet");
      expect(input(label).disabled, `${label} is blocked until the licence is accepted`).toBe(true);
    }
    expect(container.querySelector(".model-gated-notice")?.textContent).toContain("Qwen-Image 2.1 Turbo");

    const ack = container.querySelector(".model-license-ack input");
    expect(ack).toBeTruthy();
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "checked").set.call(ack, true);
      ack.dispatchEvent(new window.Event("click", { bubbles: true }));
    });
    for (const label of ["bf16", "Q8", "Q4"]) {
      expect(input(label).disabled).toBe(false);
      if (input(label).checked !== (label === "Q4")) await click(input(label));
    }
    await click(container.querySelector(".model-tier-actions button"));
    expect(createModelDownloadJob).toHaveBeenCalledTimes(1);
    expect(createModelDownloadJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: "qwen_image_2_1_turbo" }),
      { variant: "q4" },
    );
  });
});

describe("Simple Model Manager quotes Turbo's own tier floor (sc-25908)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  async function render(models, macGatingActive) {
    const [{ AppContext }, { SimpleModelManager }, { SimpleUiContext }] = await Promise.all([
      import("./context/AppContext.js"),
      import("./simple/SimpleModelManager.jsx"),
      import("./simple/SimpleUiContext.js"),
    ]);
    await act(async () => {
      root.render(
        <AppContext.Provider
          value={{
            models,
            loras: [],
            jobs: [],
            createModelDownloadJob: vi.fn(),
            createLoraDownloadJob: vi.fn(),
            macCapabilities: { macGatingActive },
          }}
        >
          <SimpleUiContext.Provider value={{ breakpoint: "desktop", toast: vi.fn(), openInAdvanced: vi.fn() }}>
            <SimpleModelManager />
          </SimpleUiContext.Provider>
        </AppContext.Provider>,
      );
    });
  }
  const meta = () => [...container.querySelectorAll(".su-row-meta")].map((node) => node.textContent).join(" ");

  // With q8/q4 published, the floor quoted is the lightest installable tier's (q4) on each
  // lane — read from Turbo's entry, and on MLX different from base's (20 vs 21), so a quote that
  // fell through to base's numbers would show.
  // *Mutation that reds this:* Turbo's `mlx.minMemoryGbByTier.q4` reading base's 21.
  it("quotes the q4 entry floor of Turbo's own rows", async () => {
    expect(TURBO_ENTRY.mlx.minMemoryGbByTier.q4).not.toBe(BASE_ENTRY.mlx.minMemoryGbByTier.q4);
    await render([catalogModel(TURBO_ENTRY)], true);
    expect(meta()).toContain(`needs ${TURBO_ENTRY.mlx.minMemoryGbByTier.q4} GB`);
    await render([catalogModel(TURBO_ENTRY)], false);
    expect(meta()).toContain(`needs ${TURBO_ENTRY.candle.minMemoryGbByTier.q4} GB`);
  });
});

describe("Training Studio target picker (sc-25908, E9)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  // Exclusion is REGISTRY-DRIVEN by design: TrainingStudio derives every base it offers only from
  // `trainingTargets` (the API's `GET /training/targets`), and there is no UI-side filter. So this
  // proves the shipped registry has no Turbo target and that the picker renders exactly that
  // registry — not that the UI would hide a Turbo target if the registry ever grew one (the API
  // refuses such a base: `untrainable_base_model_reason`).
  // *Mutation that reds this:* a registry target with `baseModel: "qwen_image_2_1_turbo"`.
  it("offers base Qwen Image 2.1 targets and nothing that trains on Turbo", async () => {
    const registry = JSON.parse(
      readFileSync(resolve(REPO, "tests/fixtures/rust_migration_contracts/training/target-registry.json"), "utf8"),
    );
    const { ConfigureJobPanel } = await import("./screens/training/ConfigureJobPanel.jsx");
    const { summarize } = await import("./validation/issues.js");
    const targets = registry.targets;
    const noop = () => {};
    await act(async () => {
      root.render(
        <ConfigureJobPanel
          active={{ id: "configure", title: "Configure training job" }}
          activeDataset={{ id: "ds1", name: "Set" }}
          configDraft={{}}
          configError=""
          configMessage=""
          configSnapshot={null}
          configValidity={summarize([])}
          datasetDoctor={{ report: null, loading: false }}
          datasets={[{ id: "ds1", name: "Set" }]}
          gpuOptions={["auto"]}
          isLokrNetwork={false}
          macLokrOnWanBlocked={false}
          macTargetBlocked={() => false}
          networkTypeOptions={[]}
          openDataset={noop}
          outputScopes={[]}
          qualityTiers={[]}
          resetConfigDefaults={noop}
          selectedPreset={null}
          selectedTarget={targets[0]}
          setActiveView={noop}
          setSelectedTargetId={noop}
          setShowAdvancedConfig={noop}
          showAdvancedConfig={false}
          showNetworkType={false}
          showTrainingAdapter={false}
          submitTrainingJob={noop}
          submittingJob={false}
          targetPresets={[]}
          trainingPresetsError=""
          trainingTargets={targets}
          trainingTargetsError=""
          updateConfigDraft={noop}
          updateQualityTier={noop}
          updateSelectedPreset={noop}
          visibleLrSchedulerOptions={[]}
          visibleOptimizerOptions={[]}
          visibleResolutionOptions={[]}
          visibleTrainingAdapterVersions={[]}
        />,
      );
    });
    const targetSelect = [...container.querySelectorAll("label")]
      .find((label) => label.firstChild?.textContent?.trim() === "Target")
      ?.querySelector("select");
    expect(targetSelect).toBeTruthy();
    const offered = [...targetSelect.options].map((option) => option.value);
    expect(offered).toHaveLength(targets.length);
    const baseOf = new Map(targets.map((target) => [target.id, target.baseModel]));
    expect(offered.filter((id) => baseOf.get(id) === "qwen_image_2_1").length).toBeGreaterThan(0);
    expect(offered.filter((id) => baseOf.get(id) === "qwen_image_2_1_turbo")).toEqual([]);
    expect([...targetSelect.options].some((option) => /turbo/i.test(option.textContent) && /qwen/i.test(option.textContent))).toBe(false);
  });
});
