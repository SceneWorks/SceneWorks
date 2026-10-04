import React, { act, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// sc-24815 — Training Studio offers to attach a finished run's adapter back to the
// character that owns its dataset. These are the story's three named cases (accept,
// decline, no character) plus the two ways the same completed job could otherwise
// attach twice: a re-render and a reload.
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
const RECENT = new Date().toISOString();

function character() {
  return { id: "char-mira", name: "Mira", loras: [] };
}

function dataset(overrides = {}) {
  return { id: "dataset-1", name: "Mira Set", version: 3, characterId: "char-mira", items: [], ...overrides };
}

function adapterJob(overrides = {}) {
  return {
    id: "job-train-1",
    type: "lora_train",
    status: "completed",
    projectId: "project-a",
    completedAt: RECENT,
    payload: {
      datasetId: "dataset-1",
      outputName: "Mira v3",
      manifestEntry: {
        id: "mira_v3",
        name: "Mira v3",
        scope: "project",
        family: "z-image",
        triggerWords: ["mira"],
        source: { provider: "training", path: "loras/mira_v3" },
      },
    },
    result: { loraRegistered: true, loraId: "mira_v3" },
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
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it("offers to attach the trained LoRA to the character, naming both", async () => {
    await render();
    expect(offerPanel()?.textContent).toContain("Mira v3");
    expect(offerPanel()?.textContent).toContain("Mira");
  });

  it("attaches on accept and the character's LoRAs panel lists it without a refresh", async () => {
    await render();
    await click(buttonByText("Attach to Mira"));

    expect(apiFetchMock).toHaveBeenCalledTimes(1);
    const [url, , options] = apiFetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/projects/project-a/characters/char-mira/loras");
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual({
      loraId: "mira_v3",
      name: "Mira v3",
      sourcePath: "loras/mira_v3",
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
});
