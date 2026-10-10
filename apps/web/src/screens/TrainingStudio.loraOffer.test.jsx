import React, { act, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// sc-24815 — Training Studio offers to attach a finished run's adapter back to the
// character that owns its dataset. These are the story's three named cases (accept,
// decline, no character) plus the four ways the same completed job could otherwise attach
// twice: a re-render, a reload, a project round trip with a blocked storage record, and a
// second window answering the same run.
//
// The roster is held by the REAL useCharacters hook, not a stub, because "the
// character's LoRAs panel lists it without a manual refresh" is a claim about that
// hook's state update — a stubbed attach function could only prove the call happened.

const apiFetchMock = vi.fn();

vi.mock("../api.js", async () => {
  const actual = await vi.importActual("../api.js");
  return { ...actual, apiFetch: (...args) => apiFetchMock(...args) };
});

import { AppContext } from "../context/AppContext.js";
import { useCharacters } from "../hooks/useCharacters.js";
import { TrainingStudio } from "./TrainingStudio.jsx";

const PROJECT = { id: "project-a", name: "Project A" };
const PROJECT_B = { id: "project-b", name: "Project B" };
const RECENT = new Date().toISOString();
const DECISIONS_KEY = "sceneworks-trained-lora-offer:project-a";

// A blocked or quota-limited record: this feature's key reads as absent and refuses
// writes while the rest of storage keeps working. That is the case the in-memory
// per-project record exists for — without it, an accepted job is re-offered after a
// project round trip and a second accept prepends a duplicate link row.
//
// Replacing the accessor is the only thing that works: jsdom hands localStorage out
// through a getter, so spying on the instance leaves the module's own lookups untouched —
// an earlier version of this helper "passed" for exactly that reason and proved nothing.
let restoreBlockedStorage = null;
function blockOfferStorage() {
  const prefix = "sceneworks-trained-lora-offer:";
  const real = globalThis.localStorage;
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get() {
      return {
        getItem: (key) => (String(key).startsWith(prefix) ? null : real.getItem(key)),
        setItem: (key, value) => {
          if (String(key).startsWith(prefix)) {
            throw new Error("quota exceeded");
          }
          real.setItem(key, value);
        },
        removeItem: (key) => real.removeItem(key),
      };
    },
  });
  restoreBlockedStorage = () => {
    Object.defineProperty(globalThis, "localStorage", descriptor);
    restoreBlockedStorage = null;
  };
}

function character() {
  return { id: "char-mira", name: "Mira", loras: [] };
}

function dataset(overrides = {}) {
  return { id: "dataset-1", name: "Mira Set", version: 3, characterId: "char-mira", items: [], ...overrides };
}

// The PERSISTED job shape, deliberately. A lora_train payload carries only
// dryRun/outputName/plan/manifestEntry/baseModel (apps/rust-api/src/training.rs:1872) —
// no top-level datasetId — so the dataset association comes from the result
// (crates/sceneworks-worker/src/training_jobs.rs:2625) and manifestEntry.provenance
// (apps/rust-api/src/training.rs:1765). An earlier version of this fixture invented
// payload.datasetId and every test passed while production could never fire the offer.
function adapterJob(overrides = {}) {
  return {
    id: "job-train-1",
    type: "lora_train",
    status: "completed",
    projectId: "project-a",
    completedAt: RECENT,
    payload: {
      dryRun: false,
      outputName: "Mira v3",
      plan: { output: { loraId: "mira_v3", format: "safetensors" } },
      manifestEntry: {
        id: "mira_v3",
        name: "Mira v3",
        scope: "project",
        family: "z-image",
        triggerWords: ["mira"],
        source: { provider: "training", path: "loras/mira_v3" },
        provenance: { kind: "training", trainingJobId: "job-train-1", datasetId: "dataset-1" },
      },
      baseModel: "z-image-turbo",
    },
    result: { loraRegistered: true, loraId: "mira_v3", datasetId: "dataset-1" },
    ...overrides,
  };
}

function baseContext(overrides = {}) {
  return {
    activeProject: PROJECT,
    authenticated: true,
    assets: [],
    jobs: [adapterJob()],
    loras: [],
    models: [],
    trainingDatasets: [dataset()],
    trainingDatasetsProjectId: "project-a",
    loadingTrainingDatasets: false,
    refreshTrainingDatasets: () => {},
    // No targets → the Configure-job config effect early-returns, so the panel renders
    // in isolation without a training catalog.
    trainingPresets: { presets: [] },
    trainingTargets: { targets: [] },
    setActiveView: () => {},
    setPreviewAsset: () => {},
    ...overrides,
  };
}

// The real character data layer, plus a readout of exactly what Character Studio's
// LoRAs panel renders from: `characters[…].loras`.
function Harness({ children, contextOverrides, valueOverrides }) {
  const [error, setError] = useState("");
  const activeProjectRef = useRef(PROJECT);
  const charactersApi = useCharacters({
    token: "token-1",
    activeProject: PROJECT,
    activeProjectRef,
    setError,
    requestedGpu: "auto",
    setActiveView: () => {},
  });
  useEffect(() => {
    charactersApi.setCharacters([character()]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const value = { ...baseContext(contextOverrides), ...charactersApi, ...valueOverrides };
  return (
    <AppContext.Provider value={value}>
      {children}
      <ul aria-label="Character LoRAs panel">
        {(charactersApi.characters[0]?.loras ?? []).map((link) => (
          <li key={link.id}>{link.name}</li>
        ))}
      </ul>
      <p aria-label="App error">{error}</p>
    </AppContext.Provider>
  );
}

let container;
let root;

async function render(contextOverrides = {}, valueOverrides) {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(<Harness contextOverrides={contextOverrides} valueOverrides={valueOverrides}>{<TrainingStudio />}</Harness>);
  });
  await settle();
}

// Unmount and mount again with the same localStorage — the reload the story names.
async function reload(contextOverrides = {}) {
  await act(async () => {
    root.unmount();
  });
  container.remove();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(<Harness contextOverrides={contextOverrides}>{<TrainingStudio />}</Harness>);
  });
  await settle();
}

// Re-render in place — a project switch, NOT a remount. A remount would discard the
// in-memory per-project record and so could not prove it survives.
async function rerender(contextOverrides = {}) {
  await act(async () => {
    root.render(<Harness contextOverrides={contextOverrides}>{<TrainingStudio />}</Harness>);
  });
  await settle();
}

async function settle() {
  await act(async () => {
    for (let index = 0; index < 8; index += 1) {
      await Promise.resolve();
    }
  });
}

function buttonByText(label) {
  return [...container.querySelectorAll("button")].find((item) => item.textContent.trim() === label) ?? null;
}

async function click(node) {
  await act(async () => {
    node.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
  await settle();
}

// A second completed run of the same dataset, newer than adapterJob().
function newerAdapterJob() {
  const newer = adapterJob({
    id: "job-train-2",
    result: { loraRegistered: true, loraId: "mira_v4", datasetId: "dataset-1" },
  });
  newer.payload = {
    ...newer.payload,
    manifestEntry: { ...newer.payload.manifestEntry, id: "mira_v4", name: "Mira v4" },
  };
  return newer;
}

function offerHeading() {
  return offerPanel()?.querySelector("h3")?.textContent ?? "";
}

function offerPanel() {
  return container.querySelector('[aria-label="Attach trained LoRA"]');
}

function panelLoras() {
  return [...container.querySelectorAll('[aria-label="Character LoRAs panel"] li')].map((item) => item.textContent);
}

describe("TrainingStudio trained-LoRA offer (sc-24815)", () => {
  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    apiFetchMock.mockReset();
    apiFetchMock.mockImplementation(async (url, token, options) => {
      if (options?.method === "POST" && url.endsWith("/loras")) {
        const body = JSON.parse(options.body);
        return { ...character(), loras: [{ id: "character_lora_1", ...body }] };
      }
      throw new Error(`unexpected apiFetch ${options?.method ?? "GET"} ${url}`);
    });
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container?.remove();
    // Before clear(): the blocked stub has no clear(), and the real store must be back
    // for the next test to start empty.
    restoreBlockedStorage?.();
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it("offers to attach the trained LoRA to the character, naming both", async () => {
    await render();
    // Exact heading, not two substring hits on the same string: "Mira v3" contains
    // "Mira", so toContain twice proves only one name.
    expect(offerPanel()?.querySelector("h3")?.textContent).toBe("Attach “Mira v3” to Mira?");
    expect(buttonByText("Attach to Mira")).not.toBeNull();
  });

  it("attaches on accept and the character's LoRAs panel lists it without a refresh", async () => {
    await render();
    await click(buttonByText("Attach to Mira"));

    expect(apiFetchMock).toHaveBeenCalledTimes(1);
    const [url, , options] = apiFetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/projects/project-a/characters/char-mira/loras");
    expect(options.method).toBe("POST");
    // sourcePath is null here because the Train route never hydrates the loras domain
    // (apps/web/src/appHydration.js:38), so the only source is the run's manifest entry —
    // whose source.path is RELATIVE and is rejected by the store
    // (crates/sceneworks-core/src/character_store.rs:1351). null means "no file to copy"
    // (character_store.rs:1362) and the link still resolves by catalog id.
    expect(JSON.parse(options.body)).toEqual({
      loraId: "mira_v3",
      name: "Mira v3",
      sourcePath: null,
      triggerWords: ["mira"],
      defaultWeight: 1,
      compatibility: { families: ["z-image"] },
      scope: "project",
    });
    // The roster updated in place: no second request, and the panel already lists it.
    expect(panelLoras()).toEqual(["Mira v3"]);
    expect(offerPanel()).toBeNull();
    expect(window.localStorage.getItem("sceneworks-trained-lora-offer:project-a")).toContain("job-train-1");
  });

  it("leaves the character unchanged on decline and never re-asks that job id", async () => {
    await render();
    await click(buttonByText("Not now"));

    expect(apiFetchMock).not.toHaveBeenCalled();
    expect(offerPanel()).toBeNull();
    expect(panelLoras()).toEqual([]);

    // Re-render with a fresh context object, then a full reload.
    await render();
    expect(offerPanel()).toBeNull();
    await reload();
    expect(offerPanel()).toBeNull();
    expect(apiFetchMock).not.toHaveBeenCalled();
  });

  it("surfaces no offer for a dataset with no character", async () => {
    await render({ trainingDatasets: [dataset({ characterId: "" })] });
    expect(offerPanel()).toBeNull();
    expect(apiFetchMock).not.toHaveBeenCalled();
  });

  it("surfaces no offer for a base-checkpoint (non-adapter) run", async () => {
    await render({
      jobs: [adapterJob({ result: { baseCheckpointRegistered: true, baseCheckpointId: "mira_full" } })],
    });
    expect(offerPanel()).toBeNull();
    expect(apiFetchMock).not.toHaveBeenCalled();
  });

  it("keeps the offer retryable when the attach fails, and does not mark the job answered", async () => {
    apiFetchMock.mockRejectedValue(new Error("disk full"));
    await render();
    await click(buttonByText("Attach to Mira"));

    expect(offerPanel()).not.toBeNull();
    expect(container.querySelector('[aria-label="App error"]')?.textContent).toContain("disk full");
    expect(window.localStorage.getItem("sceneworks-trained-lora-offer:project-a")).toBeNull();

    // Declining after a failed attempt still retires the offer.
    await click(buttonByText("Not now"));
    expect(offerPanel()).toBeNull();
  });

  it("keeps an accepted job answered across a project round trip when storage is blocked", async () => {
    blockOfferStorage();
    await render();
    expect(offerPanel()).not.toBeNull();
    await click(buttonByText("Attach to Mira"));
    expect(apiFetchMock).toHaveBeenCalledTimes(1);
    expect(offerPanel()).toBeNull();

    // Away to another project and back. Nothing readable was persisted, so only the
    // in-memory per-project record can keep this answered — and a re-offer accepted again
    // would prepend a second character_lora row for the same adapter.
    await rerender({
      activeProject: PROJECT_B,
      trainingDatasetsProjectId: "project-b",
      trainingDatasets: [],
    });
    await rerender();

    expect(offerPanel()).toBeNull();
    expect(apiFetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not replace an unanswered offer when a second run finishes", async () => {
    await render();
    const heading = () => offerPanel()?.querySelector("h3")?.textContent ?? "";
    expect(heading()).toBe("Attach “Mira v3” to Mira?");

    // A newer completed run lands while the first question is still on screen, in the
    // SAME mounted component: the offer is state, not a derived value, so the question the
    // user is reading survives it. (render() would remount and lose that state, which is
    // exactly the difference this test is about.)
    const newer = adapterJob({
      id: "job-train-2",
      result: { loraRegistered: true, loraId: "mira_v4", datasetId: "dataset-1" },
    });
    newer.payload = {
      ...newer.payload,
      manifestEntry: { ...newer.payload.manifestEntry, id: "mira_v4", name: "Mira v4" },
    };
    await rerender({ jobs: [newer, adapterJob()] });

    expect(heading()).toBe("Attach “Mira v3” to Mira?");
    expect(apiFetchMock).not.toHaveBeenCalled();
  });

  it("sends exactly one attach when the accept button is clicked twice", async () => {
    await render();
    const button = buttonByText("Attach to Mira");
    // Two clicks inside one act() batch: the state flag has not committed yet, so a
    // state-only guard lets the second call through and the character gets two rows.
    await act(async () => {
      button.click();
      button.click();
      await Promise.resolve();
    });
    await settle();

    expect(apiFetchMock).toHaveBeenCalledTimes(1);
    expect(panelLoras()).toEqual(["Mira v3"]);
  });

  it("retires a displayed offer when another window answers the same run", async () => {
    await render();
    expect(offerPanel()).not.toBeNull();

    // What the other window's accept button writes, and the event the browser then fires
    // over — without a listener this window keeps offering a job that is already linked.
    window.localStorage.setItem(DECISIONS_KEY, JSON.stringify(["job-train-1"]));
    act(() => {
      window.dispatchEvent(new window.StorageEvent("storage", { key: DECISIONS_KEY }));
    });
    await settle();

    expect(offerPanel()).toBeNull();
    expect(apiFetchMock).not.toHaveBeenCalled();
  });

  // sc-24930: an answer lives in a ref and in storage, neither of which re-runs the
  // offer effect on its own. No re-render here on purpose — the older run has to appear
  // from the decline alone, not from some unrelated state change.
  it("offers the next undecided run as soon as the displayed one is declined", async () => {
    await render({ jobs: [newerAdapterJob(), adapterJob()] });
    expect(offerHeading()).toBe("Attach “Mira v4” to Mira?");

    await click(buttonByText("Not now"));

    expect(offerHeading()).toBe("Attach “Mira v3” to Mira?");
    expect(apiFetchMock).not.toHaveBeenCalled();
  });

  it("offers the next undecided run when another window answers the displayed one", async () => {
    await render({ jobs: [newerAdapterJob(), adapterJob()] });
    expect(offerHeading()).toBe("Attach “Mira v4” to Mira?");

    window.localStorage.setItem(DECISIONS_KEY, JSON.stringify(["job-train-2"]));
    act(() => {
      window.dispatchEvent(new window.StorageEvent("storage", { key: DECISIONS_KEY }));
    });
    await settle();

    expect(offerHeading()).toBe("Attach “Mira v3” to Mira?");
  });

  it("shows an attach failure only on the offer whose attach failed", async () => {
    apiFetchMock.mockRejectedValue(new Error("disk full"));
    await render({ jobs: [newerAdapterJob(), adapterJob()] });
    await click(buttonByText("Attach to Mira"));
    expect(offerPanel()?.textContent).toContain("Could not attach Mira v4");

    // Another window answers the failed run; the next offer must not inherit its error.
    window.localStorage.setItem(DECISIONS_KEY, JSON.stringify(["job-train-2"]));
    act(() => {
      window.dispatchEvent(new window.StorageEvent("storage", { key: DECISIONS_KEY }));
    });
    await settle();

    expect(offerHeading()).toBe("Attach “Mira v3” to Mira?");
    expect(offerPanel()?.textContent).not.toContain("Could not attach");
  });
});
