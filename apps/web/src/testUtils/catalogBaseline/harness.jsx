// PAT-8 before-side capture. Inputs and observations are checked in separately. Never regenerate
// the old observations from a candidate registry or derive the old population from candidate ids.
import React, { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { vi } from "vitest";
import { AppContext } from "../../context/AppContext.js";
import { ImageStudio } from "../../screens/ImageStudio.jsx";
import { VideoStudio } from "../../screens/VideoStudio.jsx";
import { AudioStudio } from "../../screens/AudioStudio.jsx";
import { VectorStudio } from "../../screens/VectorStudio.jsx";
import { DocumentStudio } from "../../screens/DocumentStudio.jsx";
import { TrainingStudio } from "../../screens/TrainingStudio.jsx";
import { CharacterStudio } from "../../screens/CharacterStudio.jsx";
import { SimpleImageStudio } from "../../simple/SimpleImageStudio.jsx";
import { SimpleVideoStudio } from "../../simple/SimpleVideoStudio.jsx";
import { SimpleAudioStudio } from "../../simple/SimpleAudioStudio.jsx";
import { SimpleSheet, SimpleSheetOptions } from "../../simple/SimpleSheet.jsx";
import { SimpleUiContext } from "../../simple/SimpleUiContext.js";
import { createStudioStateStore, loadSnapshotIntoStore, snapshotFromStore } from "../../simple/useStudioState.js";
import { AUDIO_MODES, VIDEO_MODES, audioModelServesMode } from "../../modelEligibility.js";

const imageModes = ["text_to_image", "edit_image", "character_image"];
const clean = (value) => String(value ?? "").replace(/\s+/g, " ").trim();
const screens = { image: ImageStudio, video: VideoStudio, audio: AudioStudio, vector: VectorStudio,
  document: DocumentStudio, training: TrainingStudio, character: CharacterStudio,
  simpleImage: SimpleImageStudio, simpleVideo: SimpleVideoStudio, simpleAudio: SimpleAudioStudio };
export const keyFor = (row) => `${row.platform}/${row.studio}/${row.model}/${row.mode}`;

export function population(catalog) {
  const rows = [];
  const add = (platform, studio, model, mode) => rows.push({ platform, studio, model, mode });
  for (const platform of Object.keys(catalog.platforms)) {
    for (const model of catalog.models) {
      const capabilities = model.capabilities ?? [];
      if (model.type === "image") {
        add(platform, "character", model.id, "test");
        for (const mode of imageModes.filter((mode) => capabilities.includes(mode) || mode === "edit_image" && capabilities.includes("image_edit"))) {
          add(platform, "image", model.id, mode);
          if (mode !== "character_image") add(platform, "simpleImage", model.id, mode);
        }
        if (capabilities.includes("interleave")) add(platform, "document", model.id, "interleave");
        if (model.ui?.viewAngles?.length) add(platform, "character", model.id, "angles");
        if (model.ui?.poseLibrary) add(platform, "character", model.id, "poses");
      }
      if (model.type === "video") for (const mode of VIDEO_MODES.filter((mode) => capabilities.includes(mode))) {
        add(platform, "video", model.id, mode);
        if (["text_to_video", "image_to_video"].includes(mode)) add(platform, "simpleVideo", model.id, mode);
      }
      if (model.type === "audio") {
        for (const mode of AUDIO_MODES.filter((mode) => audioModelServesMode(model, mode))) { add(platform, "audio", model.id, mode); if (mode !== "voiceclone") add(platform, "simpleAudio", model.id, mode); }
        if (model.audio?.supportsSymbolicSong) add(platform, "audio", model.id, "song_lab");
      }
      if (model.type === "vector") { for (const mode of capabilities) add(platform, "vector", model.id, mode); add(platform, "vector", model.id, "create_from_prompt"); }
    }
    for (const target of catalog.trainingTargets.targets) add(platform, "training", target.id, target.kernel);
  }
  return rows.sort((a, b) => keyFor(a).localeCompare(keyFor(b)));
}

export function unionPopulation(oldRows, candidateRows) {
  return [...new Map([...oldRows, ...candidateRows].map((row) => [keyFor(row), row])).values()];
}
export function requireOldPopulation(oldCatalog, candidateCatalog, oldRows = population(oldCatalog)) {
  const ids = new Set(candidateCatalog.models.map((model) => model.id));
  const missing = oldCatalog.models.filter((model) => !ids.has(model.id)).map((model) => model.id);
  if (missing.length) throw new Error(`Old public catalog models missing from candidate: ${missing.join(", ")}`);
  const candidates = new Set(population(candidateCatalog).map(keyFor));
  const missingRows = oldRows.map(keyFor).filter((key) => !candidates.has(key));
  if (missingRows.length) throw new Error(`Old public surfaces missing from candidate: ${missingRows.join(", ")}`);
}

export const reference = { id: "baseline-image", projectId: "baseline-project", type: "image", displayName: "Reference image", status: {}, width: 1024, height: 1024,
  file: { path: "assets/reference.png", mimeType: "image/png" } };
const clip = { ...reference, id: "baseline-video", type: "video", file: { path: "assets/reference.mp4", mimeType: "video/mp4", duration: 6, fps: 25, width: 1024, height: 1024 } };
const audio = { ...reference, id: "baseline-audio", type: "audio", file: { path: "assets/reference.wav", mimeType: "audio/wav", duration: 12 } };
function installed(model) {
  const variants = [...new Set((model.downloads ?? []).map((d) => d.variant).filter(Boolean))];
  return { ...model, installState: "installed", cacheState: "complete", usable: true,
    conditionalPurposes: Object.fromEntries((model.conditionalComponents ?? []).flatMap((component) => (component.requiredFor ?? []).map((purpose) => [purpose, { installState: "installed", blocked: false }]))),
    minQualityTier: model.mlx?.minQualityTier,
    hasVariantMatrix: variants.length > 1,
    variants: variants.map((variant) => ({ variant, installState: "installed" })),
  };
}
// Pinned API training.rs:206-243 projection. Test fixture only: the old API intentionally
// supplies different dense/checkpoint/preview defaults on Candle, before registry cutover.
function targetForPlatform(target, platform) {
  if (platform === "macos") return target;
  const next = structuredClone(target);
  const advanced = next.defaults.advanced;
  if (["kolors_lora", "sd3_lora", "mage_flow_lora", "anima_lora"].includes(target.kernel)) {
    advanced.gradientCheckpointing = false;
    advanced.sampleEvery = 0;
    if (target.kernel === "mage_flow_lora") advanced.fullFinetuneConfig = { mixedPrecision: "f32", gradientCheckpointing: false };
  }
  if (["wan_lora", "wan_moe_lora"].includes(target.kernel)) advanced.sampleEvery = 0;
  if (target.kernel === "wan_moe_lora") advanced.gradientCheckpointing = true;
  return next;
}
export function contextFor(catalog, row, overrides = {}) {
  const models = catalog.models.map(installed);
  const selected = models.find((model) => model.id === row.model);
  const calls = [];
  const submit = (kind) => vi.fn(async (payload) => { calls.push({ kind, payload: JSON.parse(JSON.stringify(payload)) }); return { id: `baseline-job-${calls.length}`, status: "queued" }; });
  const noop = vi.fn(async () => ({}));
  const dataset = { id: "baseline-dataset", name: "Baseline Dataset", version: 1, items: Array.from({ length: 8 }, (_, i) => ({ id: `item-${i}`, path: `images/${i}.png`, width: 1024, height: 1024, caption: { text: "A lighthouse in heavy fog" } })) };
  const character = { id: "baseline-character", name: "Baseline Character", type: "person", description: "A portrait", referenceAssetIds: [reference.id], references: [{ id: "baseline-reference", assetId: reference.id, asset: reference, approved: true }], status: {} };
  const context = {
    activeProject: { id: "baseline-project", name: "Baseline Project" }, token: "baseline-token",
    assets: [reference, clip, audio], characters: [character], selectedCharacter: character,
    selectedCharacterId: character.id, selectedAsset: null, selectedAssetId: "", jobs: [], loras: [], presets: [], promptBatches: [],
    gpuOptions: [], requestedGpu: "", models: selected ? [selected, ...models.filter((m) => selected.type === "vector" ? m.type !== "vector" : m.type === "utility")] : models,
    imageModels: selected?.type === "image" ? [selected] : [], videoModels: selected?.type === "video" ? [selected] : [], audioModels: selected?.type === "audio" ? [selected] : [],
    latestImageAssets: [], recentImageAssets: [], latestVideoAssets: [], recentVideoAssets: [], recentAudioAssets: [],
    imageLocalJobs: [], videoLocalJobs: [], audioLocalJobs: [], personTracks: [], personReadiness: {},
    macCapabilities: catalog.platforms[row.platform], macCapabilitiesAuthoritative: true, preferencesHydrated: true,
    createImageJob: submit("image"), createVideoJob: submit("video"), createAudioJob: submit("audio"), createVectorJob: submit("vector"),
    createCharacterTestJob: vi.fn(async (characterId, payload) => { calls.push({ kind: "characterTest", characterId, payload }); return { id: "baseline-character-job", status: "queued" }; }),
    createVectorPromptWorkflow: submit("vectorPrompt"), createInterleaveJob: submit("document"), createTrainingJob: submit("training"),
    importAsset: vi.fn(async () => reference),
    trainingTargets: { ...catalog.trainingTargets, targets: catalog.trainingTargets.targets.filter((target) => target.id === row.model).map((target) => targetForPlatform(target, row.platform)) },
    trainingPresets: { ...catalog.trainingPresets, presets: catalog.trainingPresets.presets.map((preset) => {
      const target = catalog.trainingTargets.targets.find((entry) => entry.id === preset.targetId);
      return target ? { ...preset, config: targetForPlatform({ ...target, defaults: preset.config }, row.platform).defaults } : preset;
    }) }, trainingDatasets: [dataset], loadTrainingDataset: vi.fn(async () => dataset), loadTrainingDatasetReadiness: vi.fn(async () => ({ status: "ready" })), trainingDatasetsProjectId: "baseline-project",
    refinePrompt: noop, deleteAsset: noop, purgeAsset: noop, jobAction: noop, rememberLocalGenerationJob: noop,
    setActiveView: noop, setPreviewAsset: noop, setRequestedGpu: noop, updateAssetStatus: noop, createPreset: noop,
    setSelectedAssetId: noop, setSelectedCharacterId: noop, createModelDownloadJob: noop, createLoraDownloadJob: noop,
    createPersonDetectionJob: noop, createPersonTrackJob: noop, saveTrackCorrections: noop, refreshTrainingDatasets: noop,
    ...overrides,
  };
  return { context, calls };
}
function labelFor(node) {
  if (!node) return "";
  return clean(node.getAttribute("aria-label") || node.labels?.[0]?.textContent || node.closest("label")?.textContent || node.getAttribute("placeholder") || node.textContent);
}
export function controls(container, persistenceKey) {
  return [...container.querySelectorAll("input,textarea,select,button,[role=tab],[role=slider]")].filter((node) => !node.closest("[hidden]")).map((node) => ({
    role: node.getAttribute("role") || (node.tagName === "SELECT" ? "combobox" : node.tagName === "TEXTAREA" ? "textbox" : node.tagName === "BUTTON" ? "button" : node.type),
    label: labelFor(node), optionId: node.getAttribute("value"), value: node.type === "checkbox" || node.type === "radio" ? node.checked : node.value ?? null,
    min: node.getAttribute("min"), max: node.getAttribute("max"), step: node.getAttribute("step"),
    available: true, disabled: Boolean(node.disabled), disabledReason: clean(node.title || node.closest("[title]")?.title),
    tooltip: clean(node.getAttribute("title") || (node.getAttribute("aria-describedby") ?? "").split(" ").map((id) => document.getElementById(id)?.textContent ?? "").join(" ")), persistenceKey,
    options: node.options ? [...node.options].map((option) => ({ id: option.value, label: clean(option.textContent), disabled: option.disabled })) : undefined,
    selected: node.getAttribute("aria-selected"), pressed: node.getAttribute("aria-pressed"), expanded: node.getAttribute("aria-expanded"),
  }));
}
export async function click(node) {
  if (!node) throw new Error("Missing baseline interaction target");
  await act(async () => node.click());
}
export async function change(node, value) {
  await act(async () => {
    const proto = node.tagName === "SELECT" ? HTMLSelectElement.prototype : node.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(node, value);
    node.dispatchEvent(new Event(node.tagName === "SELECT" ? "change" : "input", { bubbles: true }));
  });
}
export async function mount(catalog, row, saved = {}, overrides = {}) {
  // Seed identity generation, rather than dropping attachment ids from observations. Sequence
  // and identity relationships remain real and comparable within each mounted public workflow.
  let identity = 0;
  vi.spyOn(crypto, "randomUUID").mockImplementation(() => `00000000-0000-4000-8000-${String(++identity).padStart(12, "0")}`);
  localStorage.clear();
  const studio = { simpleImage: "image", simpleVideo: "video", simpleAudio: "audio" }[row.studio] ?? row.studio;
  const advancedKey = `sceneworks-studio-${studio}-baseline-project`;
  const simpleStudio = row.studio.startsWith("simple");
  const persistenceKey = simpleStudio ? "sceneworks-simple-studio" : advancedKey;
  localStorage.setItem(advancedKey, JSON.stringify({ model: row.model, mode: row.mode, seed: "42", count: 1,
    activeTab: row.mode, prompt: "A lighthouse in heavy fog", advancedOpen: true, multiPhaseOpen: true,
    sourceAssetId: reference.id, referenceAssetId: reference.id, referenceAssetIds: [reference.id],
    lastFrameAssetId: reference.id, sourceClipAssetId: clip.id, bridgeRightClipAssetId: clip.id,
    referenceAudioAssetId: audio.id, sourceAudioAssetId: audio.id, ...saved }));
  const { context, calls } = contextFor(catalog, row, overrides);
  const container = document.createElement("div"); document.body.append(container);
  const root = createRoot(container);
  const Screen = screens[row.studio];
  const studioState = createStudioStateStore();
  loadSnapshotIntoStore(studioState, { [studio]: JSON.parse(localStorage.getItem(advancedKey)) });
  const getPersisted = () => simpleStudio ? snapshotFromStore(studioState) : JSON.parse(localStorage.getItem(persistenceKey) ?? "null");
  function Shell() {
    const [sheet, setSheet] = useState(null);
    const simple = { breakpoint: "desktop", studioState, dismissedJobIds: new Set(), openSheet: setSheet, closeSheet: () => setSheet(null), openGuide: () => {}, toast: () => {}, openInAdvanced: () => {}, openPreview: () => {} };
    return <SimpleUiContext.Provider value={simple}><Screen />{sheet ? <SimpleSheet onClose={() => setSheet(null)} phone={false} title={sheet.title}>{sheet.body ?? <SimpleSheetOptions kind={sheet.kind} options={sheet.options} onSelect={(value) => { sheet.onSelect(value); setSheet(null); }} />}</SimpleSheet> : null}</SimpleUiContext.Provider>;
  }
  await act(async () => root.render(<AppContext.Provider value={context}><Shell /></AppContext.Provider>));
  const close = async () => { await act(async () => root.unmount()); container.remove(); };
  return { container, root, context, calls, persistenceKey, close, getPersisted };
}
export async function observe(catalog, row) {
  const mounted = await mount(catalog, row);
  const { container, persistenceKey, calls, close, getPersisted } = mounted;
  try {
    // Open native advanced disclosures instead of mounting mock controls.
    for (const toggle of [...container.querySelectorAll(".advanced-section-toggle")]) {
      if (toggle.getAttribute("aria-expanded") !== "true") await click(toggle);
    }
    if (row.studio === "character") await click([...container.querySelectorAll('[role="tab"]')].find((node) => clean(node.textContent).toLowerCase() === row.mode));
    if (row.mode === "song_lab") await click(container.querySelector('[data-testid="yue2-lab-tab"]'));
    if (row.studio === "vector" && row.mode === "create_from_prompt") {
      await click([...container.querySelectorAll('[role="tab"]')].find((node) => clean(node.textContent) === "Create from Prompt"));
      await change(container.querySelector('textarea[aria-label="Vector prompt"]'), "A lighthouse in heavy fog");
    }
    // Document's prompt starts empty; enter authored content through its real control.
    if (row.studio === "document") await change(container.querySelector("textarea"), "A lighthouse in heavy fog");
    if (row.studio === "training") {
      const dataset = [...container.querySelectorAll("label")].find((node) => node.textContent.startsWith("Dataset"))?.querySelector("select");
      if (dataset) await change(dataset, "baseline-dataset");
    }
    const submit = [...container.querySelectorAll("button")].filter((node) => !node.closest("[hidden]") && (node.type === "submit" || node.classList.contains("su-generate"))).find((node) => /^(Generate(?:\s|$)|Convert to SVG|Start training|Compose document|Create SVG|Test Character|Run test)/.test(clean(node.textContent)));
    const initial = controls(container, persistenceKey);
    if (submit && !submit.disabled) await click(submit);
    const defaults = JSON.parse(JSON.stringify(calls));
    const menus = [];
    for (const button of [...container.querySelectorAll(".su-select")]) {
      if (button.disabled) continue;
      await click(button);
      const dialog = container.querySelector('[role="dialog"]');
      if (!dialog) throw new Error("Simple picker did not open");
      const options = controls(dialog, persistenceKey);
      const selectedLabel = labelFor(dialog.querySelector('[aria-pressed="true"]'));
      const alternative = dialog.querySelector('[aria-pressed="false"]');
      const selected = alternative ? labelFor(alternative) : null;
      calls.length = 0;
      if (alternative) {
        await click(alternative);
        if (submit?.isConnected && !submit.disabled) await click(submit);
      } else await click(container.querySelector('[aria-label="Close"]'));
      menus.push({ label: labelFor(button), controls: options, changed: selected, calls: JSON.parse(JSON.stringify(calls)) });
      if (alternative && selectedLabel) {
        await click(button);
        const original = [...container.querySelectorAll('[role="dialog"] button')].find((node) => labelFor(node) === selectedLabel);
        if (original) await click(original); else await click(container.querySelector('[aria-label="Close"]'));
      }
    }
    const changes = [];
    // Round-trip each native scalar/select control. File imports and appearance navigation have
    // explicit interaction witnesses below; action buttons are never clicked indiscriminately.
    for (const node of [...container.querySelectorAll('input:not([type=file]):not([type=hidden]),select,textarea')]) {
      if (node.disabled || !node.isConnected || node.closest("[hidden]")) continue;
      const before = node.type === "checkbox" ? node.checked : node.value;
      let next;
      if (node.tagName === "SELECT") next = [...node.options].find((o) => !o.disabled && o.value !== before)?.value;
      else if (node.type === "checkbox") next = !before;
      else if (["number", "range"].includes(node.type)) {
        const step = Number(node.step) || 1, lo = node.min === "" ? 0 : Number(node.min), hi = node.max === "" ? Number(before || 0) + step : Number(node.max);
        next = String(Number(before || lo) + step <= hi ? Number(before || lo) + step : Math.max(lo, Number(before || lo) - step));
      } else if (node.type !== "radio" && node.type !== "color" && node.type !== "search") next = "Baseline authored value";
      if (next == null || next === before) continue;
      const beforeState = JSON.stringify(getPersisted());
      calls.length = 0;
      if (node.type === "checkbox") await click(node); else await change(node, next);
      if (submit?.isConnected && !submit.disabled) await click(submit);
      const beforePersisted = JSON.parse(beforeState ?? "null");
      const afterPersisted = getPersisted();
      const persisted = Object.fromEntries(Object.entries(afterPersisted ?? {}).filter(([key, value]) => JSON.stringify(value) !== JSON.stringify(beforePersisted?.[key])));
      changes.push({ control: initial.findIndex((c) => c.label === labelFor(node)), label: labelFor(node), value: next,
        calls: JSON.parse(JSON.stringify(calls)), persisted: Object.keys(persisted).length ? persisted : null,
        hints: [...container.querySelectorAll("[role=alert],.form-error,.error")].map((n) => clean(n.textContent)) });
      if (node.isConnected) { if (node.type === "checkbox") await click(node); else await change(node, before); }
    }
    for (const group of [...container.querySelectorAll(".su-chips")]) {
      const before = group.querySelector('[aria-checked="true"]');
      const alternative = group.querySelector('[aria-checked="false"]:not(:disabled)');
      if (!alternative) continue;
      calls.length = 0;
      await click(alternative);
      if (submit?.isConnected && !submit.disabled) await click(submit);
      changes.push({ control: initial.findIndex((control) => control.label === labelFor(alternative)), label: group.getAttribute("aria-label"), value: labelFor(alternative), calls: JSON.parse(JSON.stringify(calls)), persisted: getPersisted(), hints: [] });
      if (before?.isConnected) await click(before);
    }
    return { controls: initial, menus, availability: clean(container.querySelector(".model-availability-gate")?.textContent), submission: { available: Boolean(submit), disabled: submit?.disabled ?? null, calls: defaults }, changes,
      persisted: getPersisted(),
      hints: [...container.querySelectorAll(".field-hint,[role=alert],.form-error,.error")].map((node) => clean(node.textContent)),
    };
  } finally { await close(); }
}
