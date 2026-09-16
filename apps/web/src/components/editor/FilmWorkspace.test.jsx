import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppContext } from "../../context/AppContext.js";

const { apiFetchMock, loadCredentialsMock, saveCredentialMock } = vi.hoisted(() => ({
  apiFetchMock: vi.fn(),
  loadCredentialsMock: vi.fn(),
  saveCredentialMock: vi.fn(),
}));
vi.mock("../../api.js", () => ({ apiFetch: apiFetchMock }));
vi.mock("../../credentials.js", async (importOriginal) => ({
  ...(await importOriginal()),
  loadCredentials: loadCredentialsMock,
  saveCredential: saveCredentialMock,
}));

import { FilmWorkspace } from "./FilmWorkspace.jsx";

function draft(overrides = {}) {
  return {
    schemaVersion: 1,
    id: "film_1",
    projectId: "project_1",
    revision: 1,
    title: "First film",
    originalScript: "",
    brief: "",
    structuredBrief: { synopsis: "", styleNotes: "", targetTotalSeconds: 30, beats: [], dialogue: [] },
    planning: { provider: "prompt_refiner", thinkingMode: "disabled", refinePrompts: false },
    productionPlan: {
      schemaVersion: 2,
      id: "film_1",
      version: 1,
      title: "First film",
      synopsis: "",
      model: { id: "minimax_h3", tier: "q4", fps: 24, resolution: "576x320" },
      limits: { maxRunSeconds: 3600, maxShotSeconds: 2700, maxAttemptsPerShot: 1, maxMemoryGb: 96 },
      sound: {},
      shots: [{
        id: "SH010", beat: "Opening shot", framing: "wide", prompt: "",
        targetDurationSeconds: 5.1667, startState: "Opening state", endState: "Closing state",
        conditioning: { mode: "text_to_video", referenceRoles: [] }, continuityRoles: [],
      }],
    },
    referencePack: { schemaVersion: 1, id: "film_1-references", version: 1, description: "", references: [], sound: [] },
    reviewPlan: { schemaVersion: 1, questions: [] },
    createdAt: "2026-09-16T00:00:00Z",
    updatedAt: "2026-09-16T00:00:00Z",
    ...overrides,
  };
}

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  apiFetchMock.mockReset();
  loadCredentialsMock.mockReset();
  loadCredentialsMock.mockResolvedValue([]);
  saveCredentialMock.mockReset();
  saveCredentialMock.mockResolvedValue([]);
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
});

async function renderWorkspace() {
  root = createRoot(container);
  await act(async () => {
    root.render(
      <AppContext.Provider value={{
        activeProject: { id: "project_1", name: "Project" }, token: "",
        refreshTimelines: vi.fn(), setSelectedTimelineId: vi.fn(),
      }}>
        <FilmWorkspace />
      </AppContext.Provider>,
    );
    await Promise.resolve();
  });
}

function changeValue(element, value) {
  const prototype = element.tagName === "TEXTAREA"
    ? window.HTMLTextAreaElement.prototype
    : element.tagName === "SELECT"
      ? window.HTMLSelectElement.prototype
      : window.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value").set.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

describe("FilmWorkspace", () => {
  it("creates, edits, saves, and reopens a project film draft without JSON authoring", async () => {
    const created = draft();
    const saved = draft({ revision: 2, title: "Workshop delivery" });
    let lists = 0;
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films" && options.method === "POST") return Promise.resolve(created);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve(lists++ === 0 ? [] : [saved]);
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.endsWith("/films/film_1") && options.method === "PUT") return Promise.resolve(saved);
      throw new Error(`Unexpected request ${path}`);
    });

    await renderWorkspace();
    const newButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "New film draft");
    await act(async () => { newButton.click(); await Promise.resolve(); });

    const title = container.querySelector('input[value="First film"]');
    await act(async () => {
      changeValue(title, "Workshop delivery");
      const prompt = container.querySelector('textarea[aria-label="Shot SH010 prompt"]');
      changeValue(prompt, "A courier enters a workshop carrying a red parcel.");
    });
    const save = [...container.querySelectorAll("button")].find((button) => button.textContent === "Save draft");
    await act(async () => { save.click(); await Promise.resolve(); });
    expect(apiFetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/project_1/films/film_1",
      "",
      expect.objectContaining({ method: "PUT" }),
    );
    const saveCall = apiFetchMock.mock.calls.find(([path, , options]) => path.endsWith("/films/film_1") && options?.method === "PUT");
    const body = JSON.parse(saveCall[2].body);
    expect(body.title).toBe("Workshop delivery");
    expect(body.productionPlan.shots[0].prompt).toContain("red parcel");

    act(() => root.unmount());
    root = null;
    await renderWorkspace();
    expect(container.querySelector('select[aria-label="Film draft"]').value).toBe("film_1");
    expect(container.querySelector('input[value="Workshop delivery"]')).not.toBeNull();
  });

  it("authors a screenplay brief and keeps Qwen optional and separate from the video model", async () => {
    const screenplay = draft({ originalScript: "INT. SHOP - NIGHT\nMARA\nPut it down." });
    const parsed = {
      synopsis: "INT. SHOP - NIGHT",
      styleNotes: "",
      targetTotalSeconds: 10,
      beats: [{ id: "B001", summary: "INT. SHOP - NIGHT" }],
      dialogue: [{ id: "D001", beatId: "B001", speaker: "MARA", text: "Put it down." }],
    };
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([screenplay]);
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [
        { provider: "prompt_refiner", modelId: "prompt_refine_anubis_8b", available: true },
        { provider: "native", modelId: "film_planner_qwen3_6_27b", available: false },
      ] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.endsWith("/brief/parse") && options.method === "POST") return Promise.resolve(parsed);
      throw new Error(`Unexpected request ${path}`);
    });
    await renderWorkspace();
    expect(container.querySelector('select[aria-label="Planning provider"]').value).toBe("prompt_refiner");
    expect(container.querySelector('input[aria-label="Planning target video model"]').value).toBe("minimax_h3");
    expect(container.textContent).toContain("Qwen3.6-27B is not required");

    const extract = [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Extract editable"));
    await act(async () => { extract.click(); await Promise.resolve(); });
    expect(container.querySelector('textarea[aria-label="Beat B001"]').value).toBe("INT. SHOP - NIGHT");
    expect(container.querySelector('textarea[aria-label="Dialogue D001 text"]').value).toBe("Put it down.");

    const provider = container.querySelector('select[aria-label="Planning provider"]');
    await act(async () => { changeValue(provider, "native"); });
    expect(container.textContent).toContain("no download starts automatically");
    expect([...container.querySelectorAll("button")].some((button) => button.textContent.includes("Install Qwen3.6-27B"))).toBe(true);
    expect(apiFetchMock.mock.calls.some(([path]) => path.includes("/models/"))).toBe(false);
  });

  it("saves, tests, and selects an OpenAI-compatible planner with explicit disclosure", async () => {
    const external = draft({
      originalScript: "A courier enters a workshop.",
      planning: {
        provider: "openai_compatible",
        connectionId: "fixture",
        modelId: "manual-model",
        thinkingMode: "disabled",
        refinePrompts: false,
        sendReferencePixels: false,
      },
    });
    const connection = {
      schemaVersion: 1,
      id: "fixture",
      label: "LAN planner",
      baseUrl: "http://planner.local:8080/v1",
      credentialHost: "planner.local:8080",
      supportsModelListing: true,
      supportsImageInput: true,
      timeoutSeconds: 60,
      maxOutputTokens: 8192,
    };
    loadCredentialsMock.mockResolvedValue([{ host: "planner.local:8080", present: true }]);
    saveCredentialMock.mockResolvedValue([{ host: "planner.local:8080", present: true }]);
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([external]);
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path === "/api/v1/film-planner-connections") return Promise.resolve([connection]);
      if (path === "/api/v1/film-planner-connections/fixture/test") {
        return Promise.resolve({ ok: true, detail: "Connection succeeded.", models: ["listed-model"] });
      }
      if (path === "/api/v1/film-planner-connections/fixture" && options.method === "PUT") {
        return Promise.resolve({ ...connection, ...JSON.parse(options.body) });
      }
      throw new Error(`Unexpected request ${path}`);
    });

    await renderWorkspace();
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 100));
      await Promise.resolve();
    });
    expect(container.querySelector('select[aria-label="Planning provider"]').value).toBe("openai_compatible");
    expect(container.textContent).toContain("Destination: http://planner.local:8080/v1");
    expect(container.textContent).toContain("script, edited brief, beats and dialogue");
    expect(container.querySelector('input[aria-label="Planning target video model"]').value).toBe("minimax_h3");
    expect(container.querySelector('input[aria-label="External planner model ID"]').value).toBe("manual-model");

    const testButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "Test and list models");
    await act(async () => { testButton.click(); await Promise.resolve(); await Promise.resolve(); });
    const listed = container.querySelector('select[aria-label="Listed planner model"]');
    expect(listed).not.toBeNull();
    await act(async () => { changeValue(listed, "listed-model"); });
    expect(container.querySelector('input[aria-label="External planner model ID"]').value).toBe("listed-model");

    await act(async () => {
      changeValue(container.querySelector('input[aria-label="Planning connection credential"]'), "new-secret");
      const pixelToggle = [...container.querySelectorAll('input[type="checkbox"]')]
        .find((input) => input.parentElement.textContent.includes("Send approved reference"));
      pixelToggle.click();
    });
    const saveConnection = [...container.querySelectorAll("button")].find((button) => button.textContent === "Save connection");
    await act(async () => { saveConnection.click(); await Promise.resolve(); await Promise.resolve(); });
    expect(saveCredentialMock).toHaveBeenCalledWith(expect.objectContaining({ token: "new-secret" }));
    const saveCall = apiFetchMock.mock.calls.find(([path, , options]) => path.endsWith("/fixture") && options?.method === "PUT");
    expect(saveCall).toBeTruthy();
    expect(saveCall[2].body).not.toContain("new-secret");
    expect(container.querySelector('input[aria-label="Planning target video model"]').value).toBe("minimax_h3");
  });

  it("authors generated dialogue and sound-bus controls without starting synthesis or export", async () => {
    const film = draft();
    let savedBody;
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.endsWith("/films/film_1") && options.method === "PUT") {
        savedBody = JSON.parse(options.body);
        return Promise.resolve({ ...savedBody, revision: savedBody.revision + 1 });
      }
      throw new Error(`Unexpected request ${path}`);
    });
    await renderWorkspace();
    const addLine = [...container.querySelectorAll("button")].find((button) => button.textContent === "Add generated dialogue");
    await act(async () => { addLine.click(); await Promise.resolve(); });
    await act(async () => {
      changeValue(container.querySelector('textarea[aria-label="Sound 1 dialogue text"]'), "The parcel is here.");
      changeValue(container.querySelector('input[aria-label="Sound 1 voice"]'), "am_michael");
      changeValue(container.querySelector('select[aria-label="Sound 1 speech model"]'), "chatterbox_tts");
      changeValue(container.querySelector('select[aria-label="Generated picture audio"]'), "include");
      changeValue(container.querySelector('select[aria-label="Shot SH010 generated audio"]'), "mute");
      changeValue(container.querySelector('input[aria-label="Dialogue bus gain"]'), "0.75");
      container.querySelector('input[aria-label="Mute dialogue bus"]').click();
    });
    const save = [...container.querySelectorAll("button")].find((button) => button.textContent === "Save draft");
    await act(async () => { save.click(); await Promise.resolve(); });
    expect(savedBody.referencePack.sound[0]).toMatchObject({ kind: "dialogue", text: "The parcel is here.", voice: "am_michael", model: "chatterbox_tts" });
    expect(savedBody.productionPlan.sound).toMatchObject({ generatedAudio: "include", dialogue: { gain: 0.75, muted: true } });
    expect(savedBody.productionPlan.shots[0].generatedAudio).toBe("mute");
    expect(apiFetchMock.mock.calls.some(([path]) => path.includes("/audio/jobs") || path.endsWith("/export"))).toBe(false);
  });

  it("places a staged SFX role as an editable sequence bed", async () => {
    const film = draft();
    film.referencePack.sound = [{ role: "door_close", kind: "sfx", file: "sound/door.wav", description: "Door close" }];
    let savedBody;
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.endsWith("/films/film_1") && options.method === "PUT") {
        savedBody = JSON.parse(options.body);
        return Promise.resolve({ ...savedBody, revision: 2 });
      }
      throw new Error(`Unexpected request ${path}`);
    });
    await renderWorkspace();
    const add = [...container.querySelectorAll("button")].find((button) => button.textContent === "Add sound effect bed");
    await act(async () => { add.click(); await Promise.resolve(); });
    const save = [...container.querySelectorAll("button")].find((button) => button.textContent === "Save draft");
    await act(async () => { save.click(); await Promise.resolve(); });
    expect(savedBody.productionPlan.sound.sfx).toEqual([{ role: "door_close", gain: 1, muted: false, startSeconds: 0, sourceInSeconds: 0, fadeInSeconds: 0, fadeOutSeconds: 0 }]);
  });
});
