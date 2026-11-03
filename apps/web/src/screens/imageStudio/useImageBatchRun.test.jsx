import React, { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import { useImageBatchRun } from "./useImageBatchRun.js";
let root, container;
beforeEach(() => { globalThis.IS_REACT_ACT_ENVIRONMENT = true; ({ root, container } = mountRoot()); });
afterEach(() => unmountRoot(root, container));

it("continues the committed project's batch while a different project render suspends", async () => {
  let changeProject, resolveFirst, release, running;
  let ready = false;
  const pending = new Promise((resolve) => { release = () => { ready = true; resolve(); }; });
  const createImageJob = vi.fn()
    .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }))
    .mockResolvedValue({ id: "second", status: "queued" });
  const cancelA = vi.fn(), cancelB = vi.fn();
  function Owner({ project }) {
    const [batchRun, setBatchRun] = React.useState(null);
    const { runBatch } = useImageBatchRun({ batchRun, setBatchRun, activeProject: { id: project },
      jobs: [], jobAction: project === "A" ? cancelA : cancelB, BATCH_ACTIVE_JOB_LIMIT: 10,
      preflightError: () => "", setBatchError: () => {}, batchTotal: 2, BATCH_RENDER_CAP: 100,
      setBatchConfirmPending: () => {}, batchPrompts: ["one", "two"], batchVariables: {},
      resolutionOptions: [], composeJobPrompt: ({ promptToSend }) => ({ composedPrompt: promptToSend }),
      stylePreviewActive: false, structuredPromptModel: false, batchJobCount: 2,
      BATCH_PREFLIGHT_YIELD_INTERVAL: 100, selectedModel: null,
      buildBatchJobRequest: (prompt) => ({ prompt, count: 1 }), onMagicExpand: () => {}, createImageJob });
    if (project === "B" && !ready) throw pending;
    return <button onClick={() => { running = runBatch(); }}>Run</button>;
  }
  function Harness() {
    const [project, setProject] = React.useState("A");
    changeProject = setProject;
    return <React.Suspense fallback="loading"><Owner project={project} /></React.Suspense>;
  }
  await act(async () => root.render(<Harness />));
  await act(async () => container.querySelector("button").click());
  expect(createImageJob).toHaveBeenCalledTimes(1);
  await act(async () => React.startTransition(() => changeProject("B")));
  await act(async () => { resolveFirst({ id: "first", status: "queued" }); await running; });
  expect(createImageJob).toHaveBeenCalledTimes(2);
  expect(cancelA).not.toHaveBeenCalled();
  expect(cancelB).not.toHaveBeenCalled();
  await act(async () => release());
  expect(cancelB.mock.calls.map(([job]) => job.id)).toEqual(["first", "second"]);
});
