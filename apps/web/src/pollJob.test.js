import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { apiFetch } from "./api.js";
import { pollJobToCompletion } from "./pollJob.js";

vi.mock("./api.js", () => ({
  apiFetch: vi.fn(),
}));

// pollJobToCompletion (sc-8856) is the shared POST-then-poll-until-terminal runner behind the
// five App.jsx callbacks (refinePrompt / magicPrompt / imageCaption / imageDescribe /
// compareFaceLikeness). These tests pin the mechanism that was previously copy-pasted five times:
// it POSTs the create endpoint, polls GET /jobs/:id on a 1s cadence, and honours each caller's own
// resolve/terminal/timeout semantics. Fake timers advance the inter-poll abortableDelay(1000).

// Drive apiFetch as: first call (the POST) returns `created`; every subsequent call (a GET
// /jobs/:id poll) returns the next status object from `pollResponses` (last one repeats).
function stubApi({ created, pollResponses }) {
  let pollIdx = 0;
  apiFetch.mockImplementation(async (path, _token, options) => {
    if (options?.method === "POST") {
      return created;
    }
    const idx = Math.min(pollIdx, pollResponses.length - 1);
    pollIdx += 1;
    return pollResponses[idx];
  });
}

// Await a promise while flushing the fake-timer queue, so the internal abortableDelay(1000) waits
// resolve and the poll loop advances without real wall-clock time.
async function runWithTimers(promise) {
  const settled = promise.then(
    (value) => ({ ok: true, value }),
    (error) => ({ ok: false, error }),
  );
  // Advance well past several 1s poll intervals; the loop settles as soon as a terminal status
  // (or the deadline) is reached.
  for (let i = 0; i < 200; i += 1) {
    await vi.advanceTimersByTimeAsync(1000);
  }
  return settled;
}

const baseArgs = {
  createPath: "/api/v1/prompts/refine",
  body: { prompt: "dog" },
  deadlineMs: 120000,
  resolveResult: (job) => {
    const refined = job.result?.refinedPrompt;
    if (!refined) {
      throw new Error("Refinement returned an empty prompt.");
    }
    return refined;
  },
  token: "tok",
  startError: "Could not start prompt refinement.",
  failureError: "Prompt refinement failed.",
  timeoutError: "Prompt refinement timed out. Is the refinement runtime running?",
};

describe("pollJobToCompletion (sc-8856)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    apiFetch.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("POSTs the create endpoint with the stringified body and the token", async () => {
    stubApi({
      created: { id: "job-1" },
      pollResponses: [{ status: "completed", result: { refinedPrompt: "rewritten" } }],
    });

    const settled = await runWithTimers(pollJobToCompletion({ ...baseArgs }));

    expect(settled).toEqual({ ok: true, value: "rewritten" });
    const postCall = apiFetch.mock.calls.find(([, , opts]) => opts?.method === "POST");
    expect(postCall[0]).toBe("/api/v1/prompts/refine");
    expect(postCall[1]).toBe("tok");
    expect(postCall[2].body).toBe(JSON.stringify({ prompt: "dog" }));
  });

  it("polls GET /jobs/:id until completed and returns the caller's resolved result", async () => {
    stubApi({
      created: { id: "job-9" },
      pollResponses: [
        { status: "queued" },
        { status: "running" },
        { status: "completed", result: { refinedPrompt: "final prompt" } },
      ],
    });

    const settled = await runWithTimers(pollJobToCompletion({ ...baseArgs }));

    expect(settled).toEqual({ ok: true, value: "final prompt" });
    const getCalls = apiFetch.mock.calls.filter(([, , opts]) => opts?.method !== "POST");
    expect(getCalls.every(([path]) => path === "/api/v1/jobs/job-9")).toBe(true);
    // queued + running + completed = three polls.
    expect(getCalls.length).toBe(3);
  });

  it("throws the caller's startError when the POST returns no job id", async () => {
    stubApi({ created: {}, pollResponses: [] });

    const settled = await runWithTimers(pollJobToCompletion({ ...baseArgs }));

    expect(settled.ok).toBe(false);
    expect(settled.error.message).toBe("Could not start prompt refinement.");
  });

  it("lets the caller's resolveResult throw on an empty completed result", async () => {
    stubApi({
      created: { id: "job-2" },
      pollResponses: [{ status: "completed", result: {} }],
    });

    const settled = await runWithTimers(pollJobToCompletion({ ...baseArgs }));

    expect(settled.ok).toBe(false);
    expect(settled.error.message).toBe("Refinement returned an empty prompt.");
  });

  it.each(["failed", "canceled", "interrupted"])(
    "throws the failureError on a %s terminal status",
    async (status) => {
      stubApi({ created: { id: "job-3" }, pollResponses: [{ status }] });

      const settled = await runWithTimers(pollJobToCompletion({ ...baseArgs }));

      expect(settled.ok).toBe(false);
      expect(settled.error.message).toBe("Prompt refinement failed.");
    },
  );

  it("prefers the job's own message/error over the generic failureError", async () => {
    stubApi({
      created: { id: "job-4" },
      pollResponses: [{ status: "failed", message: "runtime exploded" }],
    });

    const settled = await runWithTimers(pollJobToCompletion({ ...baseArgs }));

    expect(settled.ok).toBe(false);
    expect(settled.error.message).toBe("runtime exploded");
  });

  it("throws the timeoutError once the deadline passes with no terminal status", async () => {
    // deadlineMs 3s => the loop gets ~2 polls before Date.now() crosses the deadline.
    stubApi({ created: { id: "job-5" }, pollResponses: [{ status: "running" }] });

    const settled = await runWithTimers(
      pollJobToCompletion({ ...baseArgs, deadlineMs: 3000 }),
    );

    expect(settled.ok).toBe(false);
    expect(settled.error.message).toBe(
      "Prompt refinement timed out. Is the refinement runtime running?",
    );
  });

  it("aborts the inter-poll delay when the signal fires", async () => {
    stubApi({ created: { id: "job-6" }, pollResponses: [{ status: "running" }] });
    const controller = new AbortController();

    const promise = pollJobToCompletion({ ...baseArgs, signal: controller.signal });
    const settled = promise.then(
      (value) => ({ ok: true, value }),
      (error) => ({ ok: false, error }),
    );
    controller.abort();
    await vi.advanceTimersByTimeAsync(1000);

    const result = await settled;
    expect(result.ok).toBe(false);
    expect(result.error.name).toBe("AbortError");
  });

  it("rejects immediately when the signal is already aborted", async () => {
    stubApi({ created: { id: "job-7" }, pollResponses: [{ status: "running" }] });
    const controller = new AbortController();
    controller.abort();

    const settled = await runWithTimers(
      pollJobToCompletion({ ...baseArgs, signal: controller.signal }),
    );

    expect(settled.ok).toBe(false);
    expect(settled.error.name).toBe("AbortError");
  });
});
