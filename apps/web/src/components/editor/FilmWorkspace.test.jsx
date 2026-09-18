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
vi.mock("../../api.js", () => ({ apiFetch: apiFetchMock, isAbortError: (error) => error?.name === "AbortError" }));
vi.mock("../../credentials.js", async (importOriginal) => ({
  ...(await importOriginal()),
  loadCredentials: loadCredentialsMock,
  saveCredential: saveCredentialMock,
}));

import { describeActiveFilmShots, FilmWorkspace } from "./FilmWorkspace.jsx";

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

function renderOptions() {
  return {
    selectedRegime: "recommended_turbo",
    recommendedTurbo: { available: true, adapterIds: ["minimax_h3_turbo_4step_v01"], effectiveSteps: 4 },
    quality: { adapterIds: [], effectiveSteps: 30 },
    effective: { adapterIds: ["minimax_h3_turbo_4step_v01"], effectiveSteps: 4 },
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
  vi.useRealTimers();
});

async function renderWorkspace(context = {}) {
  root = createRoot(container);
  await act(async () => {
    root.render(
      <AppContext.Provider value={{
        activeProject: { id: "project_1", name: "Project" }, token: "",
        refreshTimelines: vi.fn(), setSelectedTimelineId: vi.fn(),
        ...context,
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
  it("offers both start paths when the project has no film draft, and hides in Timeline mode", async () => {
    apiFetchMock.mockImplementation((path) => {
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([]);
      throw new Error(`Unexpected request ${path}`);
    });
    const onNewTimeline = vi.fn();
    root = createRoot(container);
    const tree = (mode) => (
      <AppContext.Provider value={{ activeProject: { id: "project_1", name: "Project" }, token: "", refreshTimelines: vi.fn(), setSelectedTimelineId: vi.fn() }}>
        <FilmWorkspace mode={mode} onNewTimeline={onNewTimeline} />
      </AppContext.Provider>
    );
    await act(async () => { root.render(tree("film")); await Promise.resolve(); });

    const workspace = container.querySelector('section[aria-label="Film workspace"]');
    expect(workspace.hidden).toBe(false);
    expect(container.querySelector('[role="tablist"]')).toBeNull();
    const buttons = [...container.querySelectorAll(".ve-film-start-card button")].map((button) => button.textContent);
    expect(buttons).toEqual(["New film draft", "New timeline"]);
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "New timeline").click());
    expect(onNewTimeline).toHaveBeenCalledTimes(1);

    await act(async () => { root.render(tree("timeline")); });
    expect(workspace.hidden).toBe(true);
  });

  it("with no timeline, lists existing drafts beside both start paths until the operator picks one", async () => {
    apiFetchMock.mockImplementation((path) => {
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([draft(), draft({ id: "film_2", title: "Second film" })]);
      if (path === "/api/v1/projects/project_1/films/film_2") return Promise.resolve(draft({ id: "film_2", title: "Second film" }));
      if (path === "/api/v1/projects/project_1/film-runs") return Promise.resolve([]);
      if (path.endsWith("/planning")) return Promise.resolve(null);
      return Promise.resolve({ providers: [] });
    });
    const onEngage = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ activeProject: { id: "project_1", name: "Project" }, token: "", refreshTimelines: vi.fn(), setSelectedTimelineId: vi.fn() }}>
          <FilmWorkspace onEngage={onEngage} showStart />
        </AppContext.Provider>,
      );
      await Promise.resolve();
    });

    expect(container.querySelectorAll(".ve-film-start-card")).toHaveLength(2);
    expect(container.querySelector('[role="tablist"]')).toBeNull();
    const continues = [...container.querySelectorAll(".ve-film-start-drafts button")];
    expect(continues.map((button) => button.querySelector("strong").textContent)).toEqual(["First film", "Second film"]);

    await act(async () => { continues[1].click(); await Promise.resolve(); await Promise.resolve(); });
    expect(onEngage).toHaveBeenCalled();
    expect(container.querySelector(".ve-film-start")).toBeNull();
    expect(container.querySelector('[role="tablist"]')).not.toBeNull();
    expect(container.querySelector(".ve-film-bar strong").textContent).toBe("Second film");
  });

  it("opens Review on the run and draft that delivered a timeline clip, not the draft that happens to be selected", async () => {
    const other = draft({ id: "film_2", title: "Second film" });
    const clipRun = { locator: { id: "filmrun_9", draftId: "film_2", selectedShotIds: ["SH010"] }, controllerActive: false, record: { shots: [] } };
    apiFetchMock.mockImplementation((path) => {
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([draft(), other]);
      if (path === "/api/v1/projects/project_1/films/film_2") return Promise.resolve(other);
      if (path === "/api/v1/projects/project_1/film-runs/filmrun_9") return Promise.resolve(clipRun);
      if (path === "/api/v1/projects/project_1/film-runs/filmrun_9/review") return Promise.resolve({ run: clipRun, selections: [], observations: [], assistiveNotice: "" });
      if (path === "/api/v1/projects/project_1/film-runs") return Promise.resolve([]);
      if (path.endsWith("/planning")) return Promise.resolve(null);
      return Promise.resolve({ providers: [] });
    });
    root = createRoot(container);
    const tree = (viewRequest) => (
      <AppContext.Provider value={{ activeProject: { id: "project_1", name: "Project" }, token: "", refreshTimelines: vi.fn(), setSelectedTimelineId: vi.fn() }}>
        <FilmWorkspace viewRequest={viewRequest} />
      </AppContext.Provider>
    );
    await act(async () => { root.render(tree(null)); await Promise.resolve(); });
    expect(container.querySelector(".ve-film-bar strong").textContent).toBe("First film");

    await act(async () => { root.render(tree({ view: "review", runId: "filmrun_9", shotId: "SH010" })); });
    await act(async () => { for (let i = 0; i < 6; i += 1) await Promise.resolve(); });
    expect(container.querySelector(".ve-film-bar strong").textContent).toBe("Second film");
    expect(container.querySelector("#film-view-tab-review").getAttribute("aria-selected")).toBe("true");
    expect(container.querySelector("#film-view-review").hidden).toBe(false);
    expect(apiFetchMock.mock.calls.some(([path]) => String(path).includes("/film-runs/filmrun_9/review"))).toBe(true);
  });

  it("shows every planned shot in the cut strip and routes a shot to the step that can act on it", async () => {
    const film = draft();
    film.productionPlan.shots.push({ ...film.productionPlan.shots[0], id: "SH020", beat: "Second shot" });
    const filmRun = {
      locator: { id: "filmrun_1", draftId: "film_1", selectedShotIds: ["SH010", "SH020"] }, controllerActive: false,
      record: { state: "finished", timeline: { timelineId: "timeline_cut", revision: 1 }, shots: [
        { shotId: "SH010", attempts: [{ attempt: 1, status: "completed", take: { assetId: "asset_1" } }], humanDecision: null },
      ] },
    };
    apiFetchMock.mockImplementation((path) => {
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path === "/api/v1/projects/project_1/film-runs") return Promise.resolve([filmRun]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.includes("/review")) return Promise.reject(new Error("No review"));
      throw new Error(`Unexpected request ${path}`);
    });
    const onOpenTimeline = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ activeProject: { id: "project_1", name: "Project" }, token: "", refreshTimelines: vi.fn().mockResolvedValue({ ok: true }), setSelectedTimelineId: vi.fn() }}>
          <FilmWorkspace onOpenTimeline={onOpenTimeline} />
        </AppContext.Provider>,
      );
      await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
    });

    const shots = [...container.querySelectorAll(".ve-film-cut-shot")];
    expect(shots.map((shot) => shot.getAttribute("aria-label"))).toEqual(["SH010, Needs decision", "SH020, Not rendered"]);
    await act(async () => { shots[0].click(); });
    expect(container.querySelector("#film-view-review").hidden).toBe(false);
    await act(async () => { shots[1].click(); });
    expect(container.querySelector("#film-view-shots").hidden).toBe(false);
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Open in Timeline").click());
    expect(onOpenTimeline).toHaveBeenCalledWith("timeline_cut");
  });

  it("describes selected active shots from attempts without relabeling terminal or unselected outcomes", () => {
    const active = {
      locator: { selectedShotIds: ["SH010", "SH020"] },
      controllerActive: true,
      record: {
        state: "running",
        selectedShotIds: ["SH010", "SH020"],
        shots: [
          { shotId: "SH010", outcome: "not_selected", attempts: [{ status: "running" }] },
          { shotId: "SH020", outcome: "not_selected", attempts: [] },
          { shotId: "SH030", outcome: "not_selected", attempts: [] },
        ],
      },
    };
    expect(describeActiveFilmShots(active)).toBe("SH010: running · SH020: pending · SH030: not_selected");

    const terminal = structuredClone(active);
    terminal.controllerActive = false;
    terminal.record.state = "finished";
    terminal.record.shots[0].attempts[0].status = "completed";
    expect(describeActiveFilmShots(terminal)).toBe("SH010: not_selected · SH020: not_selected · SH030: not_selected");
  });

  it("keeps unsaved work mounted across three keyboard-navigable views and collapses advanced controls", async () => {
    const film = draft({ originalScript: "A courier enters." });
    apiFetchMock.mockImplementation((path) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      throw new Error(`Unexpected request ${path}`);
    });

    await renderWorkspace({
      models: [
        { id: "minimax_h3", name: "MiniMax-H3", type: "video", installState: "installed" },
        { id: "wan_2_2", name: "Wan 2.2", type: "video", installState: "installed" },
      ],
    });

    const tabs = [...container.querySelectorAll('[role="tab"]')];
    expect(tabs.map((tab) => tab.textContent)).toEqual(["Script", "References", "Plan", "Shots", "Review", "Sound"]);
    expect(tabs[0].getAttribute("aria-selected")).toBe("true");
    expect(container.querySelector("#film-view-script").hidden).toBe(false);
    expect(container.querySelector("#film-view-references").hidden).toBe(true);
    expect(container.querySelector("#film-view-plan").hidden).toBe(true);
    expect(container.querySelector("#film-view-shots").hidden).toBe(true);
    expect(container.querySelector("#film-view-review").hidden).toBe(true);
    expect([...container.querySelectorAll("button")].some((button) => button.textContent === "Render selected shots")).toBe(false);
    const planningVideoModel = container.querySelector('select[aria-label="Planning target video model"]');
    expect(planningVideoModel).not.toBeNull();
    expect(planningVideoModel.value).toBe("minimax_h3");
    expect([...container.querySelectorAll("details")]
      .find((item) => item.querySelector(":scope > summary")?.textContent === "Advanced planning settings").open).toBe(false);

    const script = container.querySelector('textarea[aria-label="Original prose or screenplay"]');
    await act(async () => {
      changeValue(script, "Unsaved courier revision.");
      changeValue(planningVideoModel, "wan_2_2");
    });
    await act(async () => { tabs[0].dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true })); });
    expect(document.activeElement).toBe(tabs[1]);
    expect(tabs[1].getAttribute("aria-selected")).toBe("true");
    expect(container.querySelector("#film-view-references").hidden).toBe(false);
    await act(async () => { tabs[3].click(); });
    expect(container.querySelector("#film-view-shots").hidden).toBe(false);
    expect(container.querySelector('select[aria-label="Video model"]').value).toBe("wan_2_2");
    expect([...container.querySelectorAll("button")].some((button) => button.textContent === "Render selected shots")).toBe(true);

    const advanced = [...container.querySelectorAll("details")]
      .find((item) => item.querySelector(":scope > summary")?.textContent.startsWith("Advanced video and budget settings"));
    expect(advanced.open).toBe(false);
    await act(async () => { advanced.querySelector("summary").click(); });
    expect(advanced.open).toBe(true);

    await act(async () => { tabs[4].click(); });
    expect(container.querySelector("#film-view-review").hidden).toBe(false);
    expect([...container.querySelectorAll("button")].some((button) => button.textContent === "Export current cut")).toBe(true);
    await act(async () => { tabs[0].click(); });
    expect(container.querySelector('textarea[aria-label="Original prose or screenplay"]').value).toBe("Unsaved courier revision.");
    expect(advanced.open).toBe(true);
  });

  it("discovers a resumed run timeline once without stealing another timeline or saving user edits", async () => {
    vi.useFakeTimers();
    const film = draft({ originalScript: "A courier enters." });
    let runReads = 0;
    const run = (revision) => ({
      locator: { id: "filmrun_resumed", draftId: "film_1" },
      controllerActive: true,
      controllerOwner: "api-resume:filmrun_resumed",
      record: { state: "running", timeline: { timelineId: "timeline_film", revision }, shots: [] },
    });
    apiFetchMock.mockImplementation((path) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([run(++runReads)]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      throw new Error(`Unexpected request ${path}`);
    });
    const refreshTimelines = vi.fn(async () => ({ ok: true, value: [{ id: "timeline_film" }] }));
    const setSelectedTimelineId = vi.fn();
    const saveTimeline = vi.fn();

    await renderWorkspace({
      activeTimeline: { id: "timeline_user", name: "Unsaved user cut", revision: 4 },
      refreshTimelines,
      saveTimeline,
      setSelectedTimelineId,
    });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });

    expect(refreshTimelines).toHaveBeenCalledTimes(1);
    expect(refreshTimelines).toHaveBeenCalledWith("project_1", expect.objectContaining({ signal: expect.any(AbortSignal) }));
    expect(setSelectedTimelineId).not.toHaveBeenCalled();
    expect(saveTimeline).not.toHaveBeenCalled();

    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(runReads).toBeGreaterThanOrEqual(2);
    expect(refreshTimelines).toHaveBeenCalledTimes(1);
    expect(setSelectedTimelineId).not.toHaveBeenCalled();
    expect(saveTimeline).not.toHaveBeenCalled();
  });

  it("defaults review to the newest run, preserves an explicit older choice, and leaves a dirty timeline selected", async () => {
    vi.useFakeTimers();
    const film = draft({ originalScript: "A courier enters." });
    const makeRun = (id, timelineId, outcome) => ({
      locator: { id, draftId: "film_1" },
      controllerActive: false,
      record: { state: "finished", outcome, timeline: { timelineId, revision: 1 }, shots: [] },
    });
    const newest = makeRun("filmrun_new", "timeline_new", "completed");
    const older = makeRun("filmrun_old", "timeline_old", "failed");
    let runReads = 0;
    const reviewRequests = [];
    apiFetchMock.mockImplementation((path) => {
      if (path.endsWith("/film-runs")) return Promise.resolve(runReads++ === 0 ? [older] : [newest, older]);
      if (path.endsWith("/film-runs/filmrun_new/review")) {
        reviewRequests.push("filmrun_new");
        return Promise.resolve({ run: newest, takeAssets: [], observations: [], selections: [], assistiveNotice: "Newest review" });
      }
      if (path.endsWith("/film-runs/filmrun_old/review")) {
        reviewRequests.push("filmrun_old");
        return Promise.resolve({ run: older, takeAssets: [], observations: [], selections: [], assistiveNotice: "Older review" });
      }
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      throw new Error(`Unexpected request ${path}`);
    });
    const refreshTimelines = vi.fn(async () => ({ ok: true, value: [] }));
    const saveTimeline = vi.fn();
    const setSelectedTimelineId = vi.fn();

    await renderWorkspace({
      activeTimeline: { id: "timeline_user", name: "Unsaved user cut", revision: 7 },
      refreshTimelines,
      saveTimeline,
      setSelectedTimelineId,
    });
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });

    const selector = container.querySelector('select[aria-label="Review run"]');
    expect([...selector.options].map((option) => option.value)).toEqual(["filmrun_new", "filmrun_old"]);
    expect(selector.value).toBe("filmrun_new");
    const reviewTab = [...container.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Review");
    await act(async () => { reviewTab.click(); await Promise.resolve(); await Promise.resolve(); });
    expect(reviewRequests).toContain("filmrun_new");

    await act(async () => {
      changeValue(selector, "filmrun_old");
      await Promise.resolve(); await Promise.resolve();
    });
    expect(selector.value).toBe("filmrun_old");
    expect(reviewRequests.at(-1)).toBe("filmrun_old");
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(selector.value).toBe("filmrun_old");
    expect(reviewRequests.at(-1)).toBe("filmrun_old");
    expect(setSelectedTimelineId).not.toHaveBeenCalled();
    expect(saveTimeline).not.toHaveBeenCalled();

    const openSavedCut = [...container.querySelectorAll("button")].find((button) => button.textContent === "Open saved cut");
    await act(async () => { openSavedCut.click(); await Promise.resolve(); });
    expect(setSelectedTimelineId).toHaveBeenLastCalledWith("timeline_old");
    expect(saveTimeline).not.toHaveBeenCalled();
  });

  it("creates, edits, saves, and reopens a project film draft without JSON authoring", async () => {
    const created = draft();
    const saved = draft({ revision: 2, title: "Workshop delivery" });
    let lists = 0;
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films" && options.method === "POST") return Promise.resolve(created);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve(lists++ === 0 ? [] : [saved]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
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
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
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

  it("tracks an explicit Qwen install to completion and refreshes availability without starting planning", async () => {
    vi.useFakeTimers();
    const film = draft({
      originalScript: "A courier enters.",
      planning: { provider: "native", modelId: "film_planner_qwen3_6_27b", thinkingMode: "enabled", refinePrompts: false },
    });
    let plannerReads = 0;
    let jobReads = 0;
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) {
        plannerReads += 1;
        return Promise.resolve({ providers: [
          { provider: "prompt_refiner", modelId: "prompt_refine_anubis_8b", available: true },
          { provider: "native", modelId: "film_planner_qwen3_6_27b", available: plannerReads > 1 },
        ] });
      }
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.endsWith("/models/film_planner_qwen3_6_27b/download") && options.method === "POST") {
        return Promise.resolve({ id: "job_qwen", type: "model_download", status: "queued", progress: 0 });
      }
      if (path === "/api/v1/jobs/job_qwen") {
        jobReads += 1;
        return Promise.resolve(jobReads === 1
          ? { id: "job_qwen", type: "model_download", status: "running", progress: 0.4 }
          : { id: "job_qwen", type: "model_download", status: "completed", progress: 1 });
      }
      throw new Error(`Unexpected request ${path}`);
    });

    await renderWorkspace();
    const install = [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Install Qwen3.6-27B"));
    await act(async () => { install.click(); await Promise.resolve(); });
    expect(container.querySelector('select[aria-label="Planning provider"]').value).toBe("native");
    expect(container.textContent).toContain("download queued. Planning will not start automatically");

    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(container.textContent).toContain("Qwen3.6-27B download: running");
    expect(container.querySelector('progress[aria-label="Qwen3.6-27B download progress"]').value).toBe(0.4);

    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(container.textContent).toContain("Qwen3.6-27B is installed and available");
    expect(container.textContent).toContain("download completed. Planning did not start automatically");
    expect(container.querySelector('select[aria-label="Planning provider"]').value).toBe("native");
    expect([...container.querySelectorAll("button")].some((button) => button.textContent.includes("Install Qwen3.6-27B"))).toBe(false);
    expect(apiFetchMock.mock.calls.some(([path, , request]) => path.endsWith("/planning") && request?.method === "POST")).toBe(false);
  });

  it("recovers an in-flight Qwen install from the normal queue after remount and stops at completion", async () => {
    vi.useFakeTimers();
    const film = draft({
      originalScript: "A courier enters.",
      planning: { provider: "native", modelId: "film_planner_qwen3_6_27b", thinkingMode: "enabled", refinePrompts: false },
    });
    let plannerReads = 0;
    let jobReads = 0;
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) {
        plannerReads += 1;
        return Promise.resolve({ providers: [
          { provider: "prompt_refiner", modelId: "prompt_refine_anubis_8b", available: true },
          { provider: "native", modelId: "film_planner_qwen3_6_27b", available: plannerReads > 1 },
        ] });
      }
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path === "/api/v1/jobs/job_recovered") {
        jobReads += 1;
        return Promise.resolve({
          id: "job_recovered", type: "model_download", status: "completed", progress: 1,
          payload: { modelId: "film_planner_qwen3_6_27b" },
        });
      }
      throw new Error(`Unexpected request ${path} ${options.method ?? "GET"}`);
    });

    await renderWorkspace({
      jobs: [{
        id: "job_recovered", type: "model_download", status: "running", progress: 0.75,
        payload: { modelId: "film_planner_qwen3_6_27b" },
      }],
    });

    expect(container.textContent).toContain("Qwen3.6-27B download: running");
    expect(container.querySelector('progress[aria-label="Qwen3.6-27B download progress"]').value).toBe(0.75);
    expect([...container.querySelectorAll("button")].find((button) => button.textContent.includes("Install Qwen3.6-27B")).disabled).toBe(true);

    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(container.textContent).toContain("Qwen3.6-27B download: completed");
    expect(container.textContent).toContain("Qwen3.6-27B is installed and available");
    expect(container.textContent).toContain("download completed. Planning did not start automatically");
    expect(container.querySelector('select[aria-label="Planning provider"]').value).toBe("native");
    expect(apiFetchMock.mock.calls.some(([path, , request]) => path.includes("/models/") && request?.method === "POST")).toBe(false);
    expect(apiFetchMock.mock.calls.some(([path, , request]) => path.endsWith("/planning") && request?.method === "POST")).toBe(false);

    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(jobReads).toBe(1);
  });

  it("refreshes externally changed Qwen availability when Native is selected", async () => {
    const film = draft({ originalScript: "A courier enters." });
    let plannerReads = 0;
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) {
        plannerReads += 1;
        return Promise.resolve({ providers: [
          { provider: "prompt_refiner", modelId: "prompt_refine_anubis_8b", available: true },
          { provider: "native", modelId: "film_planner_qwen3_6_27b", available: plannerReads > 1 },
        ] });
      }
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      throw new Error(`Unexpected request ${path} ${options.method ?? "GET"}`);
    });

    await renderWorkspace();
    const provider = container.querySelector('select[aria-label="Planning provider"]');
    await act(async () => {
      changeValue(provider, "native");
      await Promise.resolve();
    });

    expect(plannerReads).toBe(2);
    expect(container.textContent).toContain("Qwen3.6-27B is installed and available");
    expect([...container.querySelectorAll("button")].some((button) => button.textContent.includes("Install Qwen3.6-27B"))).toBe(false);
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Generate candidate plan").disabled).toBe(false);
    expect(apiFetchMock.mock.calls.some(([path, , request]) => path.includes("/models/") && request?.method === "POST")).toBe(false);
  });

  it("shows an explicit Qwen install failure and allows retry without changing provider", async () => {
    vi.useFakeTimers();
    const film = draft({
      originalScript: "A courier enters.",
      planning: { provider: "native", modelId: "film_planner_qwen3_6_27b", thinkingMode: "enabled", refinePrompts: false },
    });
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [{ provider: "native", modelId: "film_planner_qwen3_6_27b", available: false }] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.endsWith("/models/film_planner_qwen3_6_27b/download") && options.method === "POST") {
        return Promise.resolve({ id: "job_failed", type: "model_download", status: "queued" });
      }
      if (path === "/api/v1/jobs/job_failed") return Promise.resolve({ id: "job_failed", type: "model_download", status: "failed", error: "Snapshot did not contain model weights." });
      throw new Error(`Unexpected request ${path}`);
    });

    await renderWorkspace();
    const install = [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Install Qwen3.6-27B"));
    await act(async () => { install.click(); await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });

    expect(container.textContent).toContain("Qwen3.6-27B download: failed");
    expect(container.textContent).toContain("Snapshot did not contain model weights.");
    expect(container.querySelector('select[aria-label="Planning provider"]').value).toBe("native");
    expect([...container.querySelectorAll("button")].find((button) => button.textContent.includes("Install Qwen3.6-27B")).disabled).toBe(false);
  });

  it("stops polling an explicit Qwen install after unmount", async () => {
    vi.useFakeTimers();
    const film = draft({
      originalScript: "A courier enters.",
      planning: { provider: "native", modelId: "film_planner_qwen3_6_27b", thinkingMode: "enabled", refinePrompts: false },
    });
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [{ provider: "native", modelId: "film_planner_qwen3_6_27b", available: false }] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.endsWith("/models/film_planner_qwen3_6_27b/download") && options.method === "POST") {
        return Promise.resolve({ id: "job_unmount", type: "model_download", status: "queued" });
      }
      if (path === "/api/v1/jobs/job_unmount") return Promise.resolve({ id: "job_unmount", type: "model_download", status: "running" });
      throw new Error(`Unexpected request ${path}`);
    });

    await renderWorkspace();
    const install = [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Install Qwen3.6-27B"));
    await act(async () => { install.click(); await Promise.resolve(); });
    act(() => root.unmount());
    root = null;
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(apiFetchMock.mock.calls.some(([path]) => path === "/api/v1/jobs/job_unmount")).toBe(false);
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
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
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
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
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
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
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
  it("persists displayed questions for additions and carries custom questions through rename without review edits", async () => {
    const film = draft();
    film.reviewPlan = { schemaVersion: 1, id: "review", version: 1, shots: { SH010: { questions: [{ id: "custom", topic: "identity", intended: "Courier", ask: "Is it the courier?", expect: ["yes"], contradict: ["no"], frames: "last", mustObserve: true }] } } };
    const original = structuredClone(film.reviewPlan.shots.SH010);
    let savedBody;
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.endsWith("/films/film_1") && options.method === "PUT") {
        savedBody = JSON.parse(options.body);
        return Promise.resolve({ ...savedBody, revision: 2 });
      }
      throw new Error(`Unexpected request ${path}`);
    });
    await renderWorkspace();
    await act(async () => changeValue(container.querySelector('input[aria-label="Shot ID"]'), "OPENING"));
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Add shot").click());
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Save draft").click());
    expect(savedBody.reviewPlan.shots.OPENING).toEqual(original);
    expect(savedBody.reviewPlan.shots.SH010).not.toEqual(original);
    const addedId = savedBody.productionPlan.shots[1].id;
    expect(savedBody.reviewPlan.shots[addedId].questions).toHaveLength(1);
    expect(savedBody.reviewPlan.shots[addedId].questions[0].ask).toBe(container.querySelector(`textarea[aria-label="${addedId} review question 1"]`).value);
  });

  it("carries the saved draft revision through preflight and run creation", async () => {
    const film = draft();
    const posts = [];
    apiFetchMock.mockImplementation((path, _token, options = {}) => {
      if (path.endsWith("/film-runs")) return Promise.resolve([]);
      if (path === "/api/v1/projects/project_1/films") return Promise.resolve([film]);
      if (path.endsWith("/render-options")) return Promise.resolve(renderOptions());
      if (path.endsWith("/planners")) return Promise.resolve({ providers: [] });
      if (path.endsWith("/planning")) return Promise.reject(new Error("No planning operation"));
      if (path.endsWith("/films/film_1") && options.method === "PUT") return Promise.resolve({ ...JSON.parse(options.body), revision: 7 });
      if (options.method === "POST" && (path.endsWith("/preflight") || path.endsWith("/runs"))) {
        posts.push([path, JSON.parse(options.body)]);
        if (path.endsWith("/preflight")) return Promise.resolve({ valid: true, draftRevision: 7, findings: [] });
        return Promise.reject(new Error("Film draft revision conflict: another session saved"));
      }
      throw new Error(`Unexpected request ${path}`);
    });
    await renderWorkspace();
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Shots").click());
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Render selected shots").click());
    expect(posts).toHaveLength(2);
    expect(posts.map(([, body]) => body.expectedDraftRevision)).toEqual([7, 7]);
    expect(container.textContent).toContain("Film draft revision conflict");
    expect(apiFetchMock.mock.calls.some(([path]) => path.endsWith("/start"))).toBe(false);
  });

});
