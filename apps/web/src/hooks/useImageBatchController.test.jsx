import React, { act } from "react";
import { expect, it, vi } from "vitest";
import { useImageBatchController } from "./useImageBatchController.js";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";

vi.mock("../screens/imageStudio/useImageBatchRun.js", () => { throw new Error("batch chunk unavailable"); });

it("reports a failed runtime load without queuing work or retaining page callbacks", async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const { root, container } = mountRoot();
  let current;
  function Owner() {
    current = useImageBatchController({ activeProject: { id: "project" }, jobs: [], jobAction: vi.fn() });
    return null;
  }
  try {
    await act(async () => root.render(<Owner />));
    await act(async () => current.runBatch({ batchJobCount: 260 }));
    expect(current.batchRun).toMatchObject({ submitting: false, submitted: 0, firstError: expect.stringContaining("Could not load batch processing") });
    expect(current.batchRunProgress).toMatchObject({ total: 260, pending: 260, active: 0 });
    await act(async () => current.replaceBatchRun(null));
    expect(current.batchRun).toBeNull();
    expect(current.batchRunProgress).toBeNull();
  } finally { unmountRoot(root, container); }
});
