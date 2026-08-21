// sc-18727 — the MiniMax-H3 turbo control in the Advanced Video Studio.
//
// The defect class this file exists for is "declared but undrivable": a control that renders, holds
// state, and never reaches the job. sc-17160 shipped `referenceAudioAssetIds` with no control that
// could set it; sc-19502 found `limits.hardMinSteps` with no web reader at all. So nothing here
// asserts that the control EXISTS — every test follows the value into the payload
// `createVideoJob` actually receives, which is the last thing the web owns.
//
// The LoRA fixtures are the SHIPPED `builtin.loras.jsonc` entries, read at test time, for the same
// reason the model fixtures in the sibling file are: the whole point of the `sampling` block is
// that the four published variants disagree, and a retyped fixture would let the control and the
// catalog drift into agreeing about numbers neither of them holds.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, apiFetch: vi.fn(async () => ({})) };
});

import { AppContext } from "../context/AppContext.js";
import { VideoStudio } from "./VideoStudio.jsx";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFESTS = resolve(HERE, "../../../../config/manifests");
const readManifest = (file, key) => {
  const parsed = JSON5.parse(readFileSync(resolve(MANIFESTS, file), "utf8"));
  return Array.isArray(parsed) ? parsed : parsed[key];
};

const shippedModel = (id) => {
  const entry = readManifest("builtin.models.jsonc", "models").find((model) => model.id === id);
  if (!entry) throw new Error(`manifest has no model ${id}`);
  return { ...entry, installState: "installed", usable: true };
};
/// The shipped LoRA entry as the catalog route serves it: the manifest entry plus the install state
/// `normalize_lora_entry` probes. `installedPath` matters — an uninstalled adapter is filtered out
/// of `compatibleLoras`, which is exactly the state the "nothing installed" test asserts.
const shippedLora = (id) => {
  const entry = readManifest("builtin.loras.jsonc", "loras").find((lora) => lora.id === id);
  if (!entry) throw new Error(`manifest has no lora ${id}`);
  return {
    ...entry,
    scope: "builtin",
    installState: "installed",
    installedPath: `/data/loras/${id}.safetensors`,
  };
};

const MINIMAX = shippedModel("minimax_h3");
const MINIMAX_REF = shippedModel("minimax_h3_ref");
// The control leg for every "only where it does something" gate below. Bernini is a video model
// with no accelerator adapter at all, so a change that showed the turbo control globally rather
// than per-family fails here.
const BERNINI = shippedModel("bernini");

const TURBO_768P = shippedLora("minimax_h3_turbo_4step_768p");
const TURBO_8STEP = shippedLora("minimax_h3_turbo_8step");
const TURBO_4STEP_V01 = shippedLora("minimax_h3_turbo_4step_v01");
const TURBO_REF2V = shippedLora("minimax_h3_ref2v_turbo_4step");
const ALL_TURBO = [TURBO_768P, TURBO_8STEP, TURBO_4STEP_V01, TURBO_REF2V];
// A plain style LoRA on the same family — the "turbo is not just any LoRA" control.
const PLAIN_LORA = {
  id: "user_minimax_style",
  name: "A user style LoRA",
  family: "minimax-h3",
  compatibility: { families: ["minimax-h3"] },
  defaultWeight: 0.8,
  scope: "global",
  installState: "installed",
  installedPath: "/data/loras/user_minimax_style.safetensors",
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createPersonDetectionJob: vi.fn(),
    createPersonTrackJob: vi.fn(),
    createVideoJob: vi.fn(),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    importAsset: vi.fn(),
    latestVideoAssets: [],
    recentVideoAssets: [],
    studioLaunch: null,
    loras: ALL_TURBO,
    jobs: [],
    videoLocalJobs: [],
    jobAction: vi.fn(),
    rememberLocalGenerationJob: vi.fn(),
    setActiveView: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setPreviewAsset: vi.fn(),
    personTracks: [],
    personReadiness: {},
    presets: [],
    requestedGpu: "",
    saveTrackCorrections: vi.fn(),
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    updateAssetStatus: vi.fn(),
    models: [],
    videoModels: [MINIMAX],
    ...overrides,
  };
}

describe("MiniMax-H3 turbo control (sc-18726 / sc-18727)", () => {
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
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
  }

  const labelStartingWith = (text) =>
    [...container.querySelectorAll("label")].find((label) =>
      label.textContent.trim().startsWith(text),
    );
  const turboSelect = () => container.querySelector('select[aria-label="Turbo (step-distilled)"]');
  const openAdvanced = async () => {
    const toggle = container.querySelector(".advanced-section-toggle");
    if (toggle && container.querySelector(".advanced-panel") === null) await click(toggle);
  };
  const generate = async () => {
    await click(
      [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Render clip")),
    );
  };
  const lastJob = (context) => context.createVideoJob.mock.calls.at(-1)[0];
  const jobLoraIds = (context) => (lastJob(context).loras ?? []).map((lora) => lora.id);
  // The GENERIC picker's own controls — the second entry point this story has to keep in agreement
  // with the turbo control. Selecting goes through "Add LoRA" → a `.lora-pick-row`; deselecting
  // goes through the selected slot's own Remove button.
  const addFromPicker = async (name) => {
    await click(container.querySelector("button.lora-add"));
    const row = [...container.querySelectorAll("button.lora-pick-row")].find((node) =>
      node.textContent.includes(name),
    );
    if (!row) throw new Error(`the generic picker does not offer ${name}`);
    await click(row);
  };
  const removeFromPicker = async (name) => {
    const remove = container.querySelector(`button[aria-label="Remove ${name}"]`);
    if (!remove) throw new Error(`${name} is not shown as selected in the generic picker`);
    await click(remove);
  };

  // ------------------------------------------------------------------ the chain

  it("puts the selected variant's LoRA in the submitted payload", async () => {
    // THE reachability assertion. The recipe itself lives server-side (the worker resolves it from
    // the catalog id, so the web cannot forge a step count) — which makes "the id reaches `loras`"
    // exactly the whole of the web's contribution to the chain, and the only thing worth asserting
    // here. Anything less follows the value into React state and stops.
    const context = baseContext();
    await render(context);
    await openAdvanced();
    await act(async () => setSelect(turboSelect(), TURBO_8STEP.id));
    await generate();

    expect(jobLoraIds(context)).toContain(TURBO_8STEP.id);
    const submitted = lastJob(context).loras.find((lora) => lora.id === TURBO_8STEP.id);
    // 8 is NOT the model default (50), NOT the seeded 768p variant's 4, and NOT the other two
    // variants' 4 — so this cannot pass against a control wired to a constant.
    expect(TURBO_8STEP.sampling.steps).toBe(8);
    expect(submitted.role).toBe("accelerator");
  });

  it("defaults on with the validated 768p variant and can be turned off", async () => {
    // Default-on is the sc-18727 posture: sc-18729 measured 2.42 h against 12.6 min at the model's
    // own default canvas, so shipping default-off would make the out-of-the-box experience a
    // two-and-a-half-hour render.
    const context = baseContext();
    await render(context);
    await openAdvanced();
    expect(turboSelect().value).toBe(TURBO_768P.id);
    await generate();
    expect(jobLoraIds(context)).toContain(TURBO_768P.id);

    // …and Off is a real state, not a label: the adapter leaves the payload entirely, so the worker
    // resolves no recipe and the base 50-step schedule runs.
    await act(async () => setSelect(turboSelect(), ""));
    await generate();
    expect(jobLoraIds(context)).not.toContain(TURBO_768P.id);
    expect(lastJob(context).loras ?? []).toHaveLength(0);
  });

  it("pairs the reference partition with the ref2v adapter, not an fl2v one", async () => {
    // The four adapters are one family and the API's compatibility gate cannot tell them apart, so
    // nothing but this default stops a Ref2VA job from seeding the fl2v 768p file — shape-valid,
    // folds cleanly, and quietly wrong (a quality mismatch with no error).
    const context = baseContext({ videoModels: [MINIMAX_REF] });
    await render(context);
    await openAdvanced();
    expect(turboSelect().value).toBe(TURBO_REF2V.id);
    expect(turboSelect().value).not.toBe(TURBO_768P.id);
  });

  // ------------------------------------------------------- one selection, two views

  it("selecting a variant replaces any other accelerator but keeps plain LoRAs", async () => {
    // Two accelerators ask for two different schedules and the worker refuses the pair by name, so
    // a control that STACKED would offer a selection that can only 400. A plain style LoRA is
    // untouched — turbo is not a mode that evicts the picker.
    const context = baseContext({ loras: [...ALL_TURBO, PLAIN_LORA] });
    await render(context);
    await openAdvanced();
    await addFromPicker(PLAIN_LORA.name);

    await act(async () => setSelect(turboSelect(), TURBO_4STEP_V01.id));
    await generate();
    const ids = jobLoraIds(context);
    expect(ids).toContain(TURBO_4STEP_V01.id);
    expect(ids).not.toContain(TURBO_768P.id);
    expect(ids.filter((id) => ALL_TURBO.some((variant) => variant.id === id))).toHaveLength(1);
    expect(ids).toContain(PLAIN_LORA.id);
  });

  it("reflects a variant selected from the generic LoRA picker", async () => {
    // sc-18727's "the two entry points must not silently disagree". They cannot, because they are
    // one piece of state: the control READS `selectedLoraIds`. Driven from the picker side here —
    // deselecting the seeded variant there has to turn the control off, not leave it stale.
    const context = baseContext();
    await render(context);
    await openAdvanced();
    expect(turboSelect().value).toBe(TURBO_768P.id);

    // Deselect from the PICKER side: the control must follow.
    await removeFromPicker(TURBO_768P.name);
    expect(turboSelect().value).toBe("");
    await generate();
    expect(jobLoraIds(context)).not.toContain(TURBO_768P.id);

    // …and re-select a DIFFERENT variant from the picker side: the control must show that one, so
    // a user who never opens the turbo menu still gets the recipe their pick implies.
    await addFromPicker(TURBO_8STEP.name);
    expect(turboSelect().value).toBe(TURBO_8STEP.id);
    await generate();
    expect(jobLoraIds(context)).toContain(TURBO_8STEP.id);
  });

  // --------------------------------------------------------------- reachability edges

  it("offers only installed variants and says so when none are", async () => {
    // An uninstalled adapter would 400 at enqueue ("LoRA is not installed"), so offering it would be
    // a control whose only outcome is an error. With none installed the control still renders — the
    // alternative answers "why is this render two and a half hours?" with silence.
    const context = baseContext({ loras: [] });
    await render(context);
    await openAdvanced();
    expect(turboSelect()).toBeTruthy();
    expect(turboSelect().disabled).toBe(true);
    expect([...turboSelect().querySelectorAll("option")]).toHaveLength(1);
    expect(labelStartingWith("Turbo").parentElement.textContent).toContain(
      "No turbo adapter installed",
    );

    // One installed alongside three the catalog advertises but the user has NOT fetched: exactly
    // the installed one is offered. This is the arm that discriminates — a control built on the raw
    // catalog rather than the installed set would offer all four and every extra option would 400.
    const partial = baseContext({
      loras: [
        { ...TURBO_768P, installState: "missing", installedPath: null },
        TURBO_8STEP,
        { ...TURBO_4STEP_V01, installState: "missing", installedPath: null },
        { ...TURBO_REF2V, installState: "missing", installedPath: null },
      ],
    });
    await render(partial);
    await openAdvanced();
    expect([...turboSelect().querySelectorAll("option")].map((o) => o.value)).toEqual([
      "",
      TURBO_8STEP.id,
    ]);
    // …and the default-on seed lands on the one that is actually installed, rather than on the
    // preferred-but-absent 768p file.
    expect(turboSelect().value).toBe(TURBO_8STEP.id);
  });

  it("is absent on a model with no accelerator", async () => {
    const context = baseContext({ videoModels: [BERNINI], loras: ALL_TURBO });
    await render(context);
    await openAdvanced();
    expect(turboSelect()).toBeNull();
  });

  // ------------------------------------------------------------ the Off that has to survive

  it("does not re-seed turbo when the persisted snapshot already lists the model", async () => {
    // The DURABLE half of the one-shot marker. `turboSeededModels` is in the studio snapshot
    // (localStorage cache + the server ui-preferences copy `seedStudioSettingsFromServer` restores),
    // and dropping it from that object is invisible in-session: the in-memory marker still stops the
    // re-seed until the studio unmounts. It only bites on the NEXT mount — a nav away and back, or a
    // desktop relaunch — at which point a deliberate Off silently becomes On again, on a control
    // whose two states are a 12.6 min render and a 2 h 25 m one.
    //
    // `preferencesHydrated` is required, not decoration: `useStudioSettingsWriter`'s `ready` gate is
    // `preferencesHydrated && videoModels.length > 0`, so a context without it never writes at all
    // and this test would assert nothing (sc-15425's window).
    const context = baseContext({ preferencesHydrated: true });
    await render(context);
    await openAdvanced();
    expect(turboSelect().value).toBe(TURBO_768P.id);

    // A deliberate Off.
    await act(async () => setSelect(turboSelect(), ""));
    await generate();
    expect(jobLoraIds(context)).not.toContain(TURBO_768P.id);

    // What the next mount will read back. Both fields matter: the SELECTION alone cannot express
    // "already seeded", which is why the marker is persisted separately.
    const stored = JSON.parse(window.localStorage.getItem("sceneworks-studio-video-project_1"));
    expect(stored.turboSeededModels, "the marker must reach the persisted snapshot").toContain(
      MINIMAX.id,
    );
    expect(stored.selectedLoraIds ?? []).not.toContain(TURBO_768P.id);

    // Relaunch: a brand-new studio restoring that snapshot, catalog and all.
    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    const relaunched = baseContext({ preferencesHydrated: true });
    await render(relaunched);
    await openAdvanced();
    expect(turboSelect().value, "a stored Off must not be re-seeded on the next mount").toBe("");
    await generate();
    expect(jobLoraIds(relaunched)).not.toContain(TURBO_768P.id);
  });

  // ------------------------------------------------------------ turbo × the partition gate

  it("offers each partition only its own adapter once the declared-partition gate applies", async () => {
    // sc-19563 narrowed `loraMatchesModel` with the LoRAs' declared `modelIds`, and the turbo
    // control is built on `compatibleLoras` — so the two features compose rather than merely
    // coexisting: the fl2v variants become unofferable on the reference partition and the ref2v one
    // unofferable on the base partition, instead of being merely non-default. Asserted because
    // "the default is right" and "the wrong one cannot be chosen at all" are different claims, and
    // only the second closes the quality mismatch the ref2v test above describes.
    const base = baseContext({ videoModels: [MINIMAX], loras: ALL_TURBO });
    await render(base);
    await openAdvanced();
    expect([...turboSelect().querySelectorAll("option")].map((option) => option.value)).toEqual([
      "",
      TURBO_768P.id,
      TURBO_8STEP.id,
      TURBO_4STEP_V01.id,
    ]);

    const reference = baseContext({ videoModels: [MINIMAX_REF], loras: ALL_TURBO });
    await render(reference);
    await openAdvanced();
    expect([...turboSelect().querySelectorAll("option")].map((option) => option.value)).toEqual([
      "",
      TURBO_REF2V.id,
    ]);
    // The declarations these two lists are derived from — retyped here so a manifest edit that
    // erased `modelIds` (and re-collapsed both lists to all four) can't leave this test green.
    expect(TURBO_768P.modelIds).toEqual([MINIMAX.id]);
    expect(TURBO_REF2V.modelIds).toEqual([MINIMAX_REF.id]);
  });

  it("reports the recipe's step count while leaving Steps editable", async () => {
    // Upstream lists the 8-step adapter as "8 / 4", and the worker honours an explicit
    // `advanced.steps` over the variant's own — so the control REPORTS the count rather than
    // seizing the input, which is the Lightning precedent's opposite call and deliberately so.
    const context = baseContext();
    await render(context);
    await openAdvanced();
    await act(async () => setSelect(turboSelect(), TURBO_768P.id));
    const steps = labelStartingWith("Steps").querySelector("input");
    expect(steps.disabled).toBe(false);
    expect(steps.placeholder).toBe(`${TURBO_768P.sampling.steps} (Turbo)`);
    expect(steps.placeholder).not.toBe(String(MINIMAX.defaults.steps));

    await act(async () => setInput(steps, "6"));
    await generate();
    // 6 is neither the variant's 4 nor the model's default 50, so a payload carrying it proves the
    // override survived the turbo selection rather than being swallowed by it.
    expect(lastJob(context).advanced.steps).toBe(6);
    expect(jobLoraIds(context)).toContain(TURBO_768P.id);
  });
});
