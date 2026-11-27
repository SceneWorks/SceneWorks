import React, { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { useJobActions } from "./useJobActions.js";
import { jobsApi } from "../api/jobsResources.js";

vi.mock("../api/jobsResources.js", () => ({ jobsApi: { cancelMany: vi.fn() } }));

let root, container, actions, jobs, setError;
function Owner() {
  const [items, setJobs] = React.useState([{ id: "a", status: "queued", createdAt: "2026-01-01T00:00:00Z" }, { id: "z", status: "queued", createdAt: "2026-01-02T00:00:00Z" }]);
  jobs = items;
  ({ cancelJobs: actions } = useJobActions({ token: "token", setJobs, setError }));
  return null;
}

beforeEach(async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
  setError = vi.fn();
  vi.mocked(jobsApi.cancelMany).mockReset();
  await act(async () => root.render(<Owner />));
});
afterEach(() => unmountRoot(root, container));

it("cancels a batch in one request per 500 ids and upserts the returned snapshots", async () => {
  vi.mocked(jobsApi.cancelMany).mockImplementation(async (_token, { body }) => ({
    jobs: JSON.parse(body).jobIds.map((id) => ({ id, status: "canceled", createdAt: "2026-01-03T00:00:00Z" })),
  }));
  const ids = ["a", ...Array.from({ length: 600 }, (_, index) => `job-${index}`), "a"];
  let canceled;
  await act(async () => { canceled = await actions(ids); });
  const calls = vi.mocked(jobsApi.cancelMany).mock.calls;
  expect(calls.map(([, options]) => JSON.parse(options.body).jobIds.length)).toEqual([500, 101]);
  expect(calls.every(([token, options]) => token === "token" && options.method === "POST")).toBe(true);
  expect(canceled).toHaveLength(601);
  expect(jobs.find((job) => job.id === "a").status).toBe("canceled");
  expect(jobs.find((job) => job.id === "z").status).toBe("queued");
  expect(setError).toHaveBeenLastCalledWith("");
});

it("reports a failed bulk cancel and sends nothing for an empty list", async () => {
  await act(async () => { await actions([]); });
  expect(jobsApi.cancelMany).not.toHaveBeenCalled();
  vi.mocked(jobsApi.cancelMany).mockRejectedValue(new Error("API down"));
  await act(async () => { await actions(["a"]); });
  expect(setError).toHaveBeenLastCalledWith("API down");
});
