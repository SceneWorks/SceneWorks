import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../../api.js", () => ({
  API_BASE_URL: "http://localhost:44777",
  apiFetch: apiFetchMock,
  withMediaTicket: (url) => url,
}));

import { FilmReview, normalizedPlan } from "./FilmReview.jsx";

function draft() {
  return {
    id: "film_1",
    productionPlan: { shots: [{ id: "SH010", beat: "Courier enters", endState: "Parcel delivered" }] },
    reviewPlan: { schemaVersion: 1, questions: [] },
  };
}

function reviewView(overrides = {}) {
  return {
    run: {
      locator: { id: "run_1", draftId: "film_1" }, controllerActive: false,
      record: {
        timeline: { timelineId: "timeline_cut" },
        shots: [{
          shotId: "SH010", selectedAttempt: 2,
          attempts: [
            { attempt: 1, status: "completed", jobId: "job_1", take: { assetId: "asset_1", model: "minimax_h3", seed: 3 }, rejection: { at: "then", reason: "Wrong parcel" } },
            { attempt: 2, status: "completed", jobId: "job_2", take: { assetId: "asset_2", model: "minimax_h3", seed: 4 }, humanRequested: true },
          ],
          needsReview: [{ raisedAt: "now", sourceShotId: "SH000", dependency: "continuity", reason: "upstream take changed" }],
          reviews: [], humanDecision: null,
        }],
        decisions: [{ at: "then", action: "reject_take", shotId: "SH010", detail: "Wrong parcel" }],
      },
    },
    takeAssets: [
      { id: "asset_1", projectId: "project_1", type: "video", file: { path: "assets/one.mp4", mimeType: "video/mp4" } },
      { id: "asset_2", projectId: "project_1", type: "video", file: { path: "assets/two.mp4", mimeType: "video/mp4" } },
    ],
    observations: [{ reviewId: "review_1", shotId: "SH010", attempt: 2, mismatches: [{ questionId: "action", severity: "mismatch", topic: "action_completion", intended: "parcel delivered", observed: "parcel retained", confidence: 0.91, detail: "The final frame contradicts intent." }] }],
    reviewTimelineId: "timeline_review",
    selections: [{ shotId: "SH010", recordSelectedAttempt: 2, recordSelectedAssetId: "asset_2", timelineAttempt: 1, timelineAssetId: "asset_1", state: "trim_conflict", trimConflict: { incomingAssetId: "asset_2" } }],
    assistiveNotice: "Review is ASSISTIVE, not quality assurance.",
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
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
  vi.useRealTimers();
});

async function renderReview({ view = reviewView(), onChange = vi.fn(), select = vi.fn(), refresh = vi.fn() } = {}) {
  apiFetchMock.mockImplementation((path, _token, options = {}) => {
    if (path.endsWith("/film-runs")) return Promise.resolve([{ locator: { id: "run_1", draftId: "film_1" }, record: {} }]);
    if (path.endsWith("/film-runs/run_1/review") && !options.method) return Promise.resolve(view);
    if (path.endsWith("/review/swap") && options.method === "POST") return Promise.resolve({ ...view, selections: [{ ...view.selections[0], state: "aligned", timelineAttempt: 2, timelineAssetId: "asset_2", trimConflict: undefined }] });
    if (path.endsWith("/review/decision") && options.method === "POST") return Promise.resolve({ ...view, selections: [{ ...view.selections[0], state: "aligned", timelineAttempt: 2, timelineAssetId: "asset_2", trimConflict: undefined }] });
    if (path.endsWith("/review/repair") && options.method === "POST") return Promise.resolve({ ...view, run: { ...view.run, controllerActive: true } });
    throw new Error(`Unexpected request ${path}`);
  });
  root = createRoot(container);
  await act(async () => {
    root.render(<FilmReview draft={draft()} onChange={onChange} projectId="project_1" refreshTimelines={refresh} setNotice={vi.fn()} setSelectedTimelineId={select} token="token" />);
    await Promise.resolve(); await Promise.resolve();
  });
  return { onChange, select, refresh };
}

async function renderReviewVisibility({ active, runLocatorId = "", setNotice = vi.fn() }) {
  await act(async () => {
    root.render(<FilmReview active={active} draft={draft()} onChange={vi.fn()} projectId="project_1" refreshTimelines={vi.fn()} runLocatorId={runLocatorId} setNotice={setNotice} setSelectedTimelineId={vi.fn()} token="token" />);
    await Promise.resolve(); await Promise.resolve();
  });
}

function button(text) {
  return [...container.querySelectorAll("button")].find((item) => item.textContent === text);
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

describe("FilmReview", () => {
  it("replaces placeholder review intent while preserving concrete authored targets", () => {
    const current = draft();
    current.reviewPlan = normalizedPlan(current);
    current.reviewPlan.shots.SH010.questions[0].intended = "Closing state";
    current.productionPlan.shots[0].endState = "Mara rests her hand on the closed red box.";
    expect(normalizedPlan(current).shots.SH010.questions[0].intended).toBe(current.productionPlan.shots[0].endState);
    current.reviewPlan.shots.SH010.questions[0].intended = "A red box";
    expect(normalizedPlan(current).shots.SH010.questions[0].intended).toBe("A red box");
  });

  it.each(["match", "mismatch", "unobserved"])("explains a %s result with question and frame answers without a confidence percentage", async (verdict) => {
    await renderReview({ view: reviewView({ observations: [{
      reviewId: "r1", shotId: "SH010", attempt: 2, mismatches: [],
      frames: [{ id: "SH010-a2-f3", timestampSeconds: 4.5 }],
      observations: [{ questionId: "q1", question: "Is the box closed?", intended: "A closed box", verdict,
        observed: verdict === "unobserved" ? null : verdict === "match" ? "Yes." : "No.", confidence: 0.9,
        answers: [{ frameId: "SH010-a2-f3", answer: verdict === "unobserved" ? "I cannot tell." : verdict === "match" ? "Yes." : "No." }] }],
    }] }) });
    const result = container.querySelector('[aria-label="SH010 assistive findings"]');
    expect(result.textContent).toContain("Question: Is the box closed?");
    expect(result.textContent).toContain("Expected: A closed box");
    expect(result.textContent).toContain("Frame SH010-a2-f3 at 4.50s");
    expect(result.textContent).not.toContain("90%");
    expect(result.textContent).not.toContain("No advisory mismatches");
  });

  it("shows the saved timeout reason and warns about legacy ungrounded reviews", async () => {
    await renderReview({ view: reviewView({ observations: [{
      reviewId: "r1", shotId: "SH010", attempt: 1, mismatches: [], observations: [{
        questionId: "SH010_action", question: "Does the final frame show the authored action completed? Answer yes or no.",
        intended: "Closing state", verdict: "unobserved", answers: [],
        note: "answer_timeout: limits.maxAnswerSeconds is 30s and frame SH010-a1-f3 got no answer in 60.4s",
      }],
    }] }) });
    const result = container.querySelector('[aria-label="SH010 assistive findings"]');
    expect(result.textContent).toContain("Analysis timed out");
    expect(result.textContent).toContain("SH010-a1-f3 got no answer in 60.4s");
    expect(result.textContent).toContain("did not include the intended state");
    expect(result.textContent).not.toContain("no frames");
    expect(result.textContent).not.toContain("0%");
  });

  it.each([
    ["startup_timeout: limits.maxStartupSeconds is 120s", "Model startup timed out", "120-second limit"],
    ["review_timeout: limits.maxSeconds is 180s", "Review time limit reached", "180-second limit"],
  ])("explains the expired budget: %s", async (note, label, limit) => {
    await renderReview({ view: reviewView({ observations: [{
      reviewId: "r1", shotId: "SH010", attempt: 1, observations: [{
        questionId: "q", question: "Is the box closed?", intended: "Closed box",
        verdict: "unobserved", note, answers: [],
      }],
    }] }) });
    const result = container.querySelector('[aria-label="SH010 assistive findings"]');
    expect(result.textContent).toContain(label);
    expect(result.textContent).toContain(limit);
  });

  it("defaults separate startup and answer budgets while retaining explicit saved limits", () => {
    const current = draft();
    expect(normalizedPlan(current).limits).toMatchObject({ maxSeconds: 180, maxStartupSeconds: 120, maxAnswerSeconds: 30 });
    current.reviewPlan.limits = { maxSeconds: 120, maxStartupSeconds: 90, maxAnswerSeconds: 10 };
    expect(normalizedPlan(current).limits).toMatchObject(current.reviewPlan.limits);
  });

  it("refreshes a run created while hidden and addresses review by locator identity", async () => {
    const locatorId = "filmrun_5c1b068b8ded44e49513775feff1d71e";
    const recordRunId = "run_c54dd33cb3e14cd98735136aba56cbaa";
    const baseView = reviewView();
    const view = reviewView({
      run: {
        ...baseView.run,
        locator: { id: locatorId, draftId: "film_1" },
        record: { ...baseView.run.record, runId: recordRunId },
      },
    });
    apiFetchMock.mockImplementation((path) => {
      if (path.endsWith(`/film-runs/${locatorId}/review`)) return Promise.resolve(view);
      throw new Error(`Unexpected request ${path}`);
    });
    root = createRoot(container);

    await renderReviewVisibility({ active: false, runLocatorId: locatorId });
    expect(apiFetchMock).not.toHaveBeenCalled();
    expect(container.textContent).toContain("Render a shot to review takes.");

    await renderReviewVisibility({ active: true, runLocatorId: locatorId });
    expect(apiFetchMock).toHaveBeenCalledWith(
      `/api/v1/projects/project_1/film-runs/${locatorId}/review`,
      "token",
    );
    expect(apiFetchMock.mock.calls.some(([path]) => path.includes(recordRunId))).toBe(false);
    expect(container.querySelectorAll("video")).toHaveLength(2);
    expect(container.textContent).not.toContain("Render a shot to review takes.");
  });

  it("shows append-only take history, saved-cut truth, advisory findings, and provenance", async () => {
    const { select, refresh } = await renderReview();
    expect(container.querySelectorAll("video")).toHaveLength(2);
    expect(container.textContent).toContain("trim conflict");
    expect(container.textContent).toContain("Current saved asset: asset_1");
    expect(container.textContent).toContain("Generation selection");
    expect(container.textContent).toContain("Dependency stale from SH000");
    expect(container.textContent).toContain("parcel retained");
    expect(container.textContent).toContain("reject_take");

    await act(async () => { button("Open review frames").click(); await Promise.resolve(); });
    expect(refresh).toHaveBeenCalledWith("project_1");
    expect(select).toHaveBeenCalledWith("timeline_review");
  });

  it("swaps the saved cut and sends explicit decisions and bounded repairs", async () => {
    await renderReview();
    const useButtons = [...container.querySelectorAll("button")].filter((item) => item.textContent === "Use in saved cut");
    await act(async () => { useButtons[1].click(); await Promise.resolve(); });
    expect(apiFetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/review\/swap$/), "token", expect.objectContaining({ method: "POST", body: JSON.stringify({ shotId: "SH010", assetId: "asset_2" }) }));
    expect(container.textContent).toContain("aligned");

    const accept = [...container.querySelectorAll("button")].find((item) => item.textContent === "Accept" && !item.disabled);
    await act(async () => { accept.click(); await Promise.resolve(); });
    expect(apiFetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/review\/decision$/), "token", expect.objectContaining({ method: "POST" }));
    await act(async () => { button("Repair from findings").click(); await Promise.resolve(); });
    expect(apiFetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/review\/repair$/), "token", expect.objectContaining({ method: "POST" }));
  });

  it("edits frame selection, finite limits, and per-shot questions in the draft", async () => {
    let current = draft();
    const onChange = vi.fn((mutator) => { const next = structuredClone(current); mutator(next); current = next; });
    await renderReview({ onChange });
    const positions = container.querySelector('input[aria-label="Review frame positions"]');
    const question = container.querySelector('textarea[aria-label="SH010 review question 1"]');
    const frames = container.querySelector('select[aria-label="SH010 question 1 frames"]');
    await act(async () => {
      changeValue(positions, "0.2, 0.8");
      changeValue(question, "Is the parcel on the counter?");
      changeValue(frames, "all");
    });
    expect(onChange).toHaveBeenCalled();
    expect(current.reviewPlan.sampling.positions).toEqual([0.2, 0.8]);
    expect(current.reviewPlan.shots.SH010.questions[0].ask).toBe("Is the parcel on the counter?");
    expect(current.reviewPlan.shots.SH010.questions[0].frames).toBe("all");
    expect(current.reviewPlan.limits.maxQuestionsPerShot).toBe(8);
  });

  it("shows analysis progress after starting and unlocks take actions when polling completes", async () => {
    vi.useFakeTimers();
    const idle = reviewView({ selections: [{ shotId: "SH010", state: "aligned", timelineAssetId: "asset_2", timelineAttempt: 2 }] });
    await renderReview({ view: idle });
    const running = {
      ...idle,
      run: { ...idle.run, controllerActive: true, controllerOwner: "api-review:run_1" },
      reviewOperation: { status: "running" },
      actionDisabledReason: "api-review:run_1 is active. Wait for it to stop or cancel it before changing takes.",
    };
    apiFetchMock.mockResolvedValue(running);
    await act(async () => { button("Analyze selected takes").click(); });
    expect(apiFetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/review$/), "token", expect.objectContaining({ method: "POST" }));
    expect(container.textContent).toContain("Analyzing selected takes.");
    expect(container.textContent).not.toContain("Actions unavailable");
    expect(container.textContent).not.toContain("api-review:");
    expect(button("Analyze selected takes").disabled).toBe(true);
    expect(button("Render one replacement").disabled).toBe(true);
    expect([...container.querySelectorAll("button")].filter((item) => item.textContent === "Accept").every((item) => item.disabled)).toBe(true);

    apiFetchMock.mockImplementation((path) => Promise.resolve(path.endsWith("/film-runs")
      ? [{ locator: { id: "run_1", draftId: "film_1" }, record: {} }]
      : { ...idle, reviewOperation: { status: "completed" } }));
    await act(async () => { await vi.advanceTimersByTimeAsync(800); });
    expect(container.textContent).not.toContain("Analyzing selected takes.");
    expect(button("Analyze selected takes").disabled).toBe(false);
    expect(button("Render one replacement").disabled).toBe(false);
    expect([...container.querySelectorAll("button")].some((item) => item.textContent === "Accept" && !item.disabled)).toBe(true);
  });

  it("disables incompatible actions with the server reason", async () => {
    await renderReview({ view: reviewView({ actionDisabledReason: "api-repair:run_1 is active. Wait for it to stop." }) });
    expect(button("Analyze selected takes").disabled).toBe(true);
    expect(button("Repair from findings").disabled).toBe(true);
    expect(button("Reject").title).toContain("api-repair:run_1 is active");
    expect(container.textContent).toContain("Actions unavailable");
  });

  it("refreshes a cached terminal view on Resume and polls the take delivered on the same timeline", async () => {
    vi.useFakeTimers();
    const base = reviewView();
    const waiting = reviewView({
      run: {
        ...base.run,
        controllerActive: false,
        record: {
          ...base.run.record,
          timeline: null,
          shots: base.run.record.shots.map((shot) => ({ ...shot, selectedAttempt: null, attempts: [] })),
        },
      },
      takeAssets: [],
      observations: [],
      reviewTimelineId: null,
      selections: [{ shotId: "SH010", state: "not_in_saved_cut" }],
    });
    const active = {
      ...waiting,
      run: { ...waiting.run, controllerActive: true, controllerOwner: "api-resume:run_1" },
      actionDisabledReason: "api-resume:run_1 is active. Wait for it to stop or cancel it before changing takes.",
    };
    const delivered = reviewView({
      run: {
        ...base.run,
        controllerActive: false,
        record: { ...base.run.record, timeline: { ...base.run.record.timeline, revision: 2 } },
      },
    });
    let response = waiting;
    let reads = 0;
    apiFetchMock.mockImplementation((path) => {
      if (path.endsWith("/film-runs/run_1/review")) {
        reads += 1;
        return Promise.resolve(response);
      }
      throw new Error(`Unexpected request ${path}`);
    });
    root = createRoot(container);
    const render = async (runControllerActive) => {
      await act(async () => {
        root.render(<FilmReview active draft={draft()} onChange={vi.fn()} projectId="project_1" refreshTimelines={vi.fn()} runControllerActive={runControllerActive} runLocatorId="run_1" setNotice={vi.fn()} setSelectedTimelineId={vi.fn()} token="token" />);
        await Promise.resolve(); await Promise.resolve();
      });
    };

    await render(false);
    expect(button("Analyze selected takes").disabled).toBe(false);
    expect(button("Open saved cut").disabled).toBe(true);

    response = active;
    await render(true);
    expect(button("Analyze selected takes").disabled).toBe(true);
    expect(container.textContent).toContain("Actions unavailable: api-resume:run_1 is active");

    response = delivered;
    await render(false);
    expect(button("Open saved cut").disabled).toBe(false);
    expect(container.querySelectorAll("video")).toHaveLength(2);
    const settledReads = reads;
    await act(async () => { await vi.advanceTimersByTimeAsync(2400); });
    expect(reads).toBe(settledReads);
  });
  it.each(["failed", "rejected"])("shows a durable %s analysis when reopened", async (status) => {
    await renderReview({ view: reviewView({ reviewOperation: { status, detail: "SH010: frame extraction timed out" } }) });
    expect(container.querySelector('[role="alert"]').textContent).toContain("frame extraction timed out");
    expect(container.querySelector('[role="alert"]').textContent).toContain("partial");
  });

});

it.each(["replacement", "repair"])("reloads a failed %s with its take history and explicit retry", async (action) => {
  const view = reviewView();
  view.run.actionOperation = { action, status: "failed", detail: "No video_generate worker is available" };
  view.selections = [{ shotId: "SH010", state: "aligned", timelineAssetId: "asset_2", timelineAttempt: 2 }];
  await renderReview({ view });
  expect(container.querySelector('[role="alert"]').textContent).toContain(`${action} failed: No video_generate worker`);
  expect(container.querySelectorAll("video")).toHaveLength(2);
  const label = action === "repair" ? "Repair from findings" : "Render one replacement";
  expect([...container.querySelectorAll("button")].find((button) => button.textContent === label).disabled).toBe(false);
});
