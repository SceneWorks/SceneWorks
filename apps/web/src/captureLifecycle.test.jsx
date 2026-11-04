import { deferred } from "./testUtils/async.js";
import React, { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "./testUtils/dom.js";

vi.mock("./api.js", async (original) => ({ ...(await original()), apiFetch: vi.fn() }));
import { apiFetch } from "./api.js";
import { stagePoseSources } from "./poseLibrary.js";
import { useCaptureLifecycle } from "./useCaptureLifecycle.js";


afterEach(() => vi.restoreAllMocks());

it("keeps staged files associated through reordered and partial responses", async () => {
  const first = deferred();
  const second = deferred();
  apiFetch.mockReset().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise).mockRejectedValueOnce(new Error("upload failed"));
  const files = ["a.png", "b.png", "c.png"].map((name) => new File([name], name, { type: "image/png" }));
  const pending = stagePoseSources("token", files);
  second.resolve({ sources: [{ path: "/b" }] });
  first.resolve({ sources: [{ path: "/a" }] });
  const results = await pending;
  expect(results.map((result) => result.status)).toEqual(["fulfilled", "fulfilled", "rejected"]);
  expect(new Set(results.slice(0, 2).map((result) => result.value.id)).size).toBe(2);
  expect(results.slice(0, 2).map((result) => [result.value.path, result.value.file.name])).toEqual([["/a", "a.png"], ["/b", "b.png"]]);
  expect(apiFetch.mock.calls.map((call) => call[2].body.getAll("file").length)).toEqual([1, 1, 1]);
});

it("releases source URLs and ignores a job response after reset or unmount", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const response = deferred();
  apiFetch.mockReset().mockReturnValue(response.promise);
  const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:source");
  const complete = vi.fn();
  let lifecycle;
  function Harness() {
    lifecycle = useCaptureLifecycle({ token: "token", project: { id: "p", name: "P" }, requestedGpu: "auto", jobs: [], setError: vi.fn(), onComplete: complete, failureMessage: "failed" });
    return <span>{lifecycle.phase}</span>;
  }
  const { root, container } = mountRoot();
  await act(async () => root.render(<Harness />));
  let pending;
  await act(async () => {
    lifecycle.ownUrl(new File(["a"], "a.png"));
    const id = lifecycle.begin();
    pending = lifecycle.startJob(id, "pose_detect", { sources: [] });
    lifecycle.invalidate();
  });
  await act(async () => response.resolve({ id: "late", status: "completed" }));
  await pending;
  expect(container.textContent).toBe("idle");
  expect(complete).not.toHaveBeenCalled();
  expect(revoke).toHaveBeenCalledTimes(1);
  await unmountRoot(root, container);
  expect(revoke).toHaveBeenCalledTimes(1);
});

it("releases a live source on unmount and ignores its late job response", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const response = deferred();
  apiFetch.mockReset().mockReturnValue(response.promise);
  const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:live");
  const complete = vi.fn();
  let lifecycle;
  function Harness() {
    lifecycle = useCaptureLifecycle({ token: "token", project: { id: "p" }, jobs: [], setError: vi.fn(), onComplete: complete, failureMessage: "failed" });
    return null;
  }
  const { root, container } = mountRoot();
  await act(async () => root.render(<Harness />));
  let pending;
  await act(async () => {
    lifecycle.ownUrl(new File(["a"], "a.png"));
    pending = lifecycle.startJob(lifecycle.begin(), "pose_detect", { sources: [] });
  });
  await unmountRoot(root, container);
  await act(async () => response.resolve({ id: "late", status: "completed" }));
  await pending;
  expect(revoke).toHaveBeenCalledOnce();
  expect(complete).not.toHaveBeenCalled();
});
