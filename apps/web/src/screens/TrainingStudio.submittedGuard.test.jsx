import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { withTrainingStudioContext, response, settle, field, changeField, zImageTrainingTarget } from "../main.testSupport.jsx";
import { deferred } from "../testUtils/async.js";

const confirm = vi.hoisted(() => vi.fn(async () => false));
vi.mock("../appConfirm.jsx", () => ({ appConfirm: confirm, useConfirm: () => confirm, ConfirmHost: () => null }));
let root, container, leaveGuard, trainingProps;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  localStorage.clear();
  confirm.mockClear();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  vi.stubGlobal("fetch", vi.fn(async () => response([])));
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
});
async function renderTraining(createTrainingJob, activeProject = { id: "project", name: "Project" }) {
  trainingProps = {
    activeProject,
    datasets: [{ id: "dataset", name: "Portraits", modality: "image", itemCount: 1 }],
    loadDataset: async () => ({ id: "dataset", name: "Portraits", version: 1,
      items: [{ id: "item", assetId: "image", path: "images/portrait.png", caption: { text: "portrait", source: "manual" } }] }),
    trainingTargets: [zImageTrainingTarget], gpuOptions: ["auto"], createTrainingJob,
    registerLeaveGuard: (guard) => { leaveGuard = guard; return () => { leaveGuard = null; }; },
  };
  await act(async () => root.render(withTrainingStudioContext(trainingProps)));
  await settle();
  if (!activeProject) return;
  await changeField(field(container, "Dataset"), "dataset");
  await settle();
  await changeField(field(container, "Steps"), "4321");
}
async function switchProject(id) {
  trainingProps = { ...trainingProps, activeProject: { id, name: id } };
  await act(async () => root.render(withTrainingStudioContext(trainingProps)));
  await settle();
}
const startButton = () => [...container.querySelectorAll("button")].find((node) => node.textContent === "Start training");
// Start training stays disabled while the draft's server options check is pending
// ("Checking options with the server…"). That check resolves through dynamic imports of the
// portable validator, which a fixed microtask flush cannot outlast on a cold module graph, so
// wait for the real signal. Each poll is a whole act() turn, so the feedback update always
// lands inside act and never between turns.
async function waitForStartable(timeoutMs = 5000) {
  const deadline = Date.now() + timeoutMs;
  while (startButton()?.disabled !== false && Date.now() < deadline) {
    await act(async () => new Promise((resolve) => setTimeout(resolve, 5)));
  }
  return startButton();
}
async function submit() {
  const button = await waitForStartable();
  expect(button.disabled).toBe(false);
  await act(async () => button.click());
  await settle();
}

it("allows leaving after queuing unchanged settings, but protects later edits", async () => {
  const createTrainingJob = vi.fn(async () => ({ id: "queued" }));
  await renderTraining(createTrainingJob);
  await submit();
  expect(createTrainingJob).toHaveBeenCalledOnce();
  expect(await leaveGuard()).toBe(true);
  expect(confirm).not.toHaveBeenCalled();
  await changeField(field(container, "Steps"), "4322");
  expect(await leaveGuard()).toBe(false);
  expect(confirm).toHaveBeenCalledOnce();
  await changeField(field(container, "Steps"), "4321");
  expect(await leaveGuard()).toBe(true);
  expect(confirm).toHaveBeenCalledOnce();
});

it.each(["rejected", "missing job"])("keeps the warning when submission is %s", async (outcome) => {
  await renderTraining(vi.fn(async () => {
    if (outcome === "rejected") throw new Error("Could not queue training");
    return null;
  }));
  await submit();
  expect(await leaveGuard()).toBe(false);
  expect(confirm).toHaveBeenCalledOnce();
});

it("keeps edits made while enqueue is pending separate from the saved request", async () => {
  const pending = deferred();
  await renderTraining(vi.fn(() => pending.promise));
  await submit();
  await changeField(field(container, "Steps"), "5432");
  await act(async () => pending.resolve({ id: "queued" }));
  expect(await leaveGuard()).toBe(false);
  await changeField(field(container, "Steps"), "4321");
  expect(await leaveGuard()).toBe(true);
});

it("allows leaving the no-workspace screen before any request has been queued", async () => {
  await renderTraining(vi.fn(), null);
  expect(await leaveGuard()).toBe(true);
  expect(confirm).not.toHaveBeenCalled();
});

it.each([false, true])("clears the saved baseline across workspace A → B → A (submission pending: %s)", async (pendingSubmission) => {
  const pending = deferred();
  await renderTraining(vi.fn(() => pendingSubmission ? pending.promise : Promise.resolve({ id: "queued" })));
  await submit();
  await switchProject("other-project");
  expect(await leaveGuard()).toBe(true);
  await switchProject("project");
  if (pendingSubmission) await act(async () => pending.resolve({ id: "queued" }));
  expect(await leaveGuard()).toBe(true);
  expect(confirm).not.toHaveBeenCalled();
});
