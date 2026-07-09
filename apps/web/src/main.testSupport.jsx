// Shared test scaffolding for the App-shell test suite (sc-8943).
//
// main.test.jsx grew to ~11k lines because every App-wiring assertion had no
// per-domain home. That monolith was split into App.*.test.jsx files (one per
// screen/feature seam). This module holds the setup those files share — the
// context adapters, the FakeEventSource, the fetch-response builders, the DOM
// query helpers, and the Z-Image training fixtures — so the split files import
// it instead of duplicating it. It is a pure move: the helpers are byte-for-byte
// what main.test.jsx defined. It is named *.testSupport.jsx (not *.test.jsx) so
// Vitest's default include glob does not collect it as a test file.
import React, { act } from "react";
import { AppContext } from "./context/AppContext.js";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { ModelManagerScreen } from "./screens/ModelManagerScreen.jsx";
import { TrainingDataSetsLibrary, TrainingStudio } from "./screens/TrainingStudio.jsx";

// sc-1651 Phase B: screens converted to useAppContext() read their data from the
// provider instead of props. Tests wrap the screen in a provider carrying only the
// values that screen reads.
export function withAppContext(value, ui) {
  return <AppContext.Provider value={value}>{ui}</AppContext.Provider>;
}

// ModelManagerScreen (sc-1651 Phase B) reads primitives from context and derives
// its own on* callbacks. This adapter lets the existing tests keep their old
// prop-shaped objects (and their assertions on those fns) while feeding the
// screen via the provider.
// ImageStudio (sc-1651 Phase B) — same adapter idea as ModelManager: keep the
// old prop-shaped fixtures (and their assertions) and map onto the provider.
export function withImageStudioContext(p) {
  return withAppContext(
    {
      activeProject: p.activeProject,
      assets: p.assets,
      characters: p.characters,
      createImageJob: p.createImageJob,
      refinePrompt: p.refinePrompt,
      deleteAsset: p.deleteAsset,
      purgeAsset: p.purgeAsset,
      gpuOptions: p.gpuOptions,
      imageModels: p.imageModels,
      latestImageAssets: p.latestAssets,
      studioLaunch: p.launchRequest,
      imageLocalJobs: p.localJobs,
      loras: p.loras,
      presets: p.presets,
      requestedGpu: p.requestedGpu,
      selectedAsset: p.selectedAsset,
      setRequestedGpu: p.setRequestedGpu,
      updateAssetStatus: p.updateAssetStatus,
      setPreviewAsset: p.onPreview ?? (() => {}),
      jobAction: p.onCancelJob ? (job) => p.onCancelJob(job) : () => {},
      rememberLocalGenerationJob: p.onLocalJobCreated ? (_kind, job) => p.onLocalJobCreated(job) : () => {},
      setActiveView: (view) => {
        if (view === "Presets") p.onOpenPresets?.();
        else if (view === "Queue") p.onOpenQueue?.();
      },
    },
    <ImageStudio />,
  );
}

export function withModelManagerContext(p) {
  return withAppContext(
    {
      activeProject: p.activeProject,
      jobs: p.jobs,
      loras: p.loras,
      models: p.models,
      presets: p.recipePresets,
      jobAction: p.onCancelJob ? (job) => p.onCancelJob(job) : () => {},
      setActiveView: p.onOpenQueue ? () => p.onOpenQueue() : () => {},
      deleteLora: p.onDeleteLora,
      deleteModel: p.onDeleteModel,
      createModelDownloadJob: p.onDownloadModel,
      createModelConvertJob: p.onConvertModel,
      createLoraImportJob: p.onImportLora,
      createModelImportJob: p.onImportModel,
    },
    <ModelManagerScreen />,
  );
}

// TrainingStudio (sc-1651 Phase B) — maps the old prop-shaped fixtures onto the
// provider. The screen unwraps catalogs ({presets}/{targets}) and project-scopes
// datasets, so feed those raw, and route the derived callbacks back to the
// fixture's on* / wrapped fns.
export function withTrainingStudioContext(p) {
  return withAppContext(
    {
      activeProject: p.activeProject,
      authenticated: p.authenticated,
      assets: p.assets,
      gpuOptions: p.gpuOptions,
      jobs: p.jobs,
      setPreviewAsset: p.onPreview ?? (() => {}),
      importAsset: p.importAsset,
      trainingDatasets: p.datasets,
      trainingDatasetsProjectId: p.activeProject?.id,
      trainingDatasetsError: p.datasetsError,
      loadingTrainingDatasets: p.loadingDatasets,
      refreshTrainingDatasets: p.onRefreshDatasets ? () => p.onRefreshDatasets() : () => {},
      loadTrainingDataset: p.loadDataset,
      createTrainingDataset: p.createDataset,
      uploadTrainingDatasetItem: p.uploadDatasetItem,
      updateTrainingDataset: p.updateDataset,
      batchRenameTrainingDataset: p.batchRenameDataset,
      writeTrainingDatasetCaptionSidecars: p.writeCaptionSidecars,
      createTrainingDatasetCaptionJob: p.createCaptionJob,
      createTrainingJob: p.createTrainingJob,
      trainingPresets: { presets: p.trainingPresets },
      trainingPresetsError: p.trainingPresetsError,
      trainingTargets: { targets: p.trainingTargets },
      trainingTargetsError: p.trainingTargetsError,
    },
    <TrainingStudio />,
  );
}

export function withTrainingDataSetsLibraryContext(p) {
  return withAppContext(
    {
      activeProject: p.activeProject,
      authenticated: p.authenticated,
      assets: p.assets,
      characters: p.characters,
      gpuOptions: p.gpuOptions,
      jobs: p.jobs,
      setPreviewAsset: p.onPreview ?? (() => {}),
      importAsset: p.importAsset,
      trainingDatasets: p.datasets,
      trainingDatasetsProjectId: p.activeProject?.id,
      trainingDatasetsError: p.datasetsError,
      loadingTrainingDatasets: p.loadingDatasets,
      refreshTrainingDatasets: p.onRefreshDatasets ? () => p.onRefreshDatasets() : () => {},
      loadTrainingDataset: p.loadDataset,
      createTrainingDataset: p.createDataset,
      uploadTrainingDatasetItem: p.uploadDatasetItem,
      updateTrainingDataset: p.updateDataset,
      batchRenameTrainingDataset: p.batchRenameDataset,
      writeTrainingDatasetCaptionSidecars: p.writeCaptionSidecars,
      createTrainingDatasetCaptionJob: p.createCaptionJob,
      createTrainingJob: p.createTrainingJob,
      trainingPresets: { presets: p.trainingPresets },
      trainingPresetsError: p.trainingPresetsError,
      trainingTargets: { targets: p.trainingTargets },
      trainingTargetsError: p.trainingTargetsError,
      setActiveView: p.setActiveView,
    },
    <TrainingDataSetsLibrary />,
  );
}

// jsdom 27 omits Blob.text(); all real browsers implement it. Polyfill so the
// dataset import flow (which reads .txt caption sidecars) is exercisable here.
if (typeof Blob !== "undefined" && typeof Blob.prototype.text !== "function") {
  Blob.prototype.text = function text() {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result ?? ""));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(this);
    });
  };
}

export class FakeEventSource {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.listeners = {};
    FakeEventSource.instances.push(this);
  }

  addEventListener(event, handler) {
    this.listeners[event] = handler;
  }

  close() {}
}

export function response(payload) {
  return {
    ok: true,
    json: async () => payload,
  };
}

export function errorResponse(status, detail) {
  return {
    ok: false,
    status,
    json: async () => ({ detail }),
  };
}

export async function settle() {
  await act(async () => {
    for (let index = 0; index < 6; index += 1) {
      await Promise.resolve();
    }
  });
}

export function field(container, labelText) {
  const label = [...container.querySelectorAll("label")].find((item) => item.childNodes[0]?.textContent.trim() === labelText);
  return label?.querySelector("input, select, textarea");
}

export function loraPanel(container) {
  return container.querySelector("form[aria-label='Import LoRA']");
}

// The AdvancedSection disclosure unmounts its body when collapsed (sc-10474), so a
// test that touches an override field has to expand it first — unlike the `<details>`
// element it replaced, which kept its contents in the DOM.
export async function openAdvancedSection(scope = document.body) {
  const toggle = scope.querySelector(".advanced-section-toggle");
  if (toggle?.getAttribute("aria-expanded") === "false") {
    await act(async () => {
      toggle.click();
    });
  }
}

// The Model Manager is a tabbed interface (epic 10309): Image / Video / Utility / LoRAs.
// A model card or the LoRA import form is only mounted while its tab is active, so tests
// switch tabs after opening the Models page. The tab button text carries a trailing count
// badge, so match by substring.
export async function selectModelTab(label) {
  await act(async () => {
    [...document.body.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent.includes(label))?.click();
  });
  await settle();
}

// The family filter moved onto the tab bar (aria-label "Filter by family"), no longer a
// labelled control, so `field(container, "LoRA family")` no longer resolves it.
export function familyFilter(container) {
  return container.querySelector(".models-family-select");
}

export function modelImportPanel(container) {
  return container.querySelector("form[aria-label='Import model']");
}

export function buttonInside(scope, label) {
  return [...scope.querySelectorAll("button")].find((button) => button.textContent === label);
}

export function navLabels(container, sectionLabel) {
  const section = [...container.querySelectorAll(".sidebar-section")].find(
    (item) => item.querySelector(".sidebar-section-title")?.textContent === sectionLabel,
  );
  return [...(section?.querySelectorAll(".nav-label") ?? [])].map((item) => item.textContent);
}

export const zImageTrainingTarget = {
  id: "z_image_turbo_lora",
  name: "Z-Image-Turbo LoRA",
  modality: "image",
  outputKind: "lora",
  family: "z-image",
  baseModel: "z_image_turbo",
  kernel: "z_image_lora",
  defaults: {
    rank: 16,
    alpha: 16,
    learningRate: 0.0001,
    steps: 3000,
    batchSize: 1,
    gradientAccumulation: 1,
    resolution: 1024,
    saveEvery: 250,
    seed: 42,
    optimizer: "adamw8bit",
    advanced: {
      mixedPrecision: "bf16",
      cacheTextEmbeddings: true,
      gradientCheckpointing: true,
      timestepType: "sigmoid",
      timestepBias: "high_noise",
      lossType: "mse",
      weightDecay: 0.0001,
      sampleEvery: 250,
      sampleSteps: 8,
      sampleGuidanceScale: 1.0,
      qualityPreset: "balanced",
      outputScope: "project",
      requestedGpu: "auto",
    },
  },
  limits: {
    resolutions: [512, 768, 1024],
    optimizers: ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"],
    outputScopes: ["project", "global"],
  },
  ui: { label: "Z-Image-Turbo LoRA" },
};

export const zImageTrainingPresets = [
  {
    id: "z_image_turbo_lora.character.adamw8bit.balanced",
    version: 1,
    targetId: "z_image_turbo_lora",
    name: "Character balanced",
    recommendedFor: ["character"],
    optimizer: "adamw8bit",
    qualityPreset: "balanced",
    config: {
      ...zImageTrainingTarget.defaults,
      advanced: {
        ...zImageTrainingTarget.defaults.advanced,
        trainingAdapterRepo: "ostris/zimage_turbo_training_adapter",
        trainingAdapterVersion: "v2-default",
      },
    },
    ui: { default: true, order: 10 },
  },
  {
    id: "z_image_turbo_lora.character.adamw8bit.conservative",
    version: 1,
    targetId: "z_image_turbo_lora",
    name: "Character conservative",
    recommendedFor: ["character"],
    optimizer: "adamw8bit",
    qualityPreset: "conservative",
    config: {
      ...zImageTrainingTarget.defaults,
      rank: 8,
      alpha: 8,
      learningRate: 0.00005,
      advanced: {
        ...zImageTrainingTarget.defaults.advanced,
        qualityPreset: "conservative",
        trainingAdapterRepo: "ostris/zimage_turbo_training_adapter",
        trainingAdapterVersion: "v2-default",
      },
    },
    ui: { order: 20 },
  },
  {
    id: "z_image_turbo_lora.character.prodigyopt.balanced",
    version: 1,
    targetId: "z_image_turbo_lora",
    name: "Prodigy character (experimental)",
    recommendedFor: ["character"],
    optimizer: "prodigyopt",
    qualityPreset: "balanced",
    config: {
      ...zImageTrainingTarget.defaults,
      optimizer: "prodigyopt",
      learningRate: 1.0,
      steps: 1600,
      saveEvery: 200,
      advanced: {
        ...zImageTrainingTarget.defaults.advanced,
        sampleEvery: 200,
        trainingAdapterRepo: "ostris/zimage_turbo_training_adapter",
        trainingAdapterVersion: "v2-default",
      },
    },
    ui: { experimental: true, order: 40 },
  },
];

export async function changeField(input, value) {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(input.constructor.prototype, "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(new window.Event(input.tagName === "SELECT" ? "change" : "input", { bubbles: true }));
  });
}

export async function changeFile(input, file) {
  await act(async () => {
    Object.defineProperty(input, "files", { configurable: true, value: [file] });
    input.dispatchEvent(new window.Event("change", { bubbles: true }));
  });
}
