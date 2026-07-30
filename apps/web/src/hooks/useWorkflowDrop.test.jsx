// "Drop a shared image anywhere → Workflow found → prefill Image Studio" (sc-15951, epic 15945).
//
// The behaviour under test is mostly about what does NOT happen: every foreign PNG in the world
// answers `no_workflow`, and that path has to stay byte-for-byte the swallow it was before this
// hook existed — no panel, no spinner, no error band.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The transport is stubbed at `inspectWorkflowFile` (not at `apiFetch`): the hook's contract is
// the endpoint's `{ status, workflow, resolution }` body and its typed `ApiError`s, and pinning
// it to a FormData round trip would test axum's multipart rather than this hook.
const inspectWorkflowFile = vi.fn();
vi.mock("../api.js", async () => {
  const actual = await vi.importActual("../api.js");
  return { ...actual, inspectWorkflowFile: (...args) => inspectWorkflowFile(...args) };
});

import { ApiError } from "../api.js";
import { inspectFailureMessage, useWorkflowDrop } from "./useWorkflowDrop.js";

const PNG_HEAD = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function pngFile(name = "shared.png") {
  return new File([new Uint8Array([...PNG_HEAD, 1, 2, 3, 4])], name, { type: "image/png" });
}

function workflowBody(overrides = {}) {
  return {
    status: "workflow",
    workflow: {
      sceneworksWorkflow: "image",
      schemaVersion: 1,
      mode: "text_to_image",
      model: "krea_2_turbo",
      prompt: "a lighthouse in heavy fog",
      negativePrompt: "",
      seed: 4242,
      width: 1024,
      height: 1024,
      count: 4,
      advanced: { steps: 28 },
    },
    resolution: {
      model: {
        slug: "krea_2_turbo",
        state: "resolved",
        catalogId: "krea_2_turbo",
        name: "Krea 2 Turbo",
        detail: "Krea 2 Turbo is installed on this machine.",
      },
      loras: [],
      styles: [],
      inputs: [],
      omitted: [],
      runnable: true,
      inputImagesRequired: 0,
    },
    ...overrides,
  };
}

describe("useWorkflowDrop", () => {
  let container;
  let root;
  let hook;

  function Harness(props) {
    hook = useWorkflowDrop(props);
    return null;
  }

  const render = async (props) => {
    await act(async () => root.render(<Harness {...props} />));
  };

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    inspectWorkflowFile.mockReset();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  it("opens the offer when the dropped image carries a workflow", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    await render({ projectId: "project_1", token: "t" });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    expect(hook.offer?.share?.prompt).toBe("a lighthouse in heavy fog");
    expect(hook.offer?.report?.model?.state).toBe("resolved");
    const [sent, options] = inspectWorkflowFile.mock.calls[0];
    expect(sent.name).toBe("shared.png");
    expect(options).toEqual({ projectId: "project_1", token: "t" });
  });

  it("does nothing at all for an image with no workflow — the common case", async () => {
    inspectWorkflowFile.mockResolvedValue({
      status: "no_workflow",
      workflow: null,
      resolution: null,
      detail: "This image carries no SceneWorks workflow…",
    });
    await render({});
    await act(async () => {
      await hook.handleDroppedFile(pngFile("holiday.png"));
    });
    expect(hook.offer).toBeNull();
  });

  it("never asks the server about a file that is not a PNG", async () => {
    await render({});
    await act(async () => {
      await hook.handleDroppedFile(new File(["not an image"], "notes.txt", { type: "text/plain" }));
    });
    expect(inspectWorkflowFile).not.toHaveBeenCalled();
    expect(hook.offer).toBeNull();
  });

  it("does nothing while disabled", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    await render({ enabled: false });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    expect(inspectWorkflowFile).not.toHaveBeenCalled();
    expect(hook.offer).toBeNull();
  });

  it("shows the sentence for a file that claims a workflow it cannot read", async () => {
    inspectWorkflowFile.mockRejectedValue(
      new ApiError("This PNG could not be read far enough to look for a workflow.", {
        status: 422,
        code: "workflow_inspect_unreadable",
      }),
    );
    await render({});
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    expect(hook.offer?.error).toContain("could not be read far enough");
    expect(hook.offer?.share).toBeNull();
  });

  it("stays silent for a failure that says nothing about the file", async () => {
    // A network drop, an axum plain-text multipart rejection (no `code` at all), or a `not_png`
    // that slipped past the prescreen. None of them is worth a modal the user did not ask for.
    for (const error of [
      new ApiError("Failed to fetch", {}),
      new ApiError("Request failed with 400", { status: 400 }),
      new ApiError("Not a PNG", { status: 400, code: "workflow_inspect_not_png" }),
      new ApiError("bad multipart", { status: 400, code: "workflow_inspect_bad_multipart" }),
    ]) {
      inspectWorkflowFile.mockReset();
      inspectWorkflowFile.mockRejectedValue(error);
      await render({});
      await act(async () => {
        await hook.handleDroppedFile(pngFile());
      });
      expect(hook.offer).toBeNull();
    }
  });

  it("lets a second drop supersede a slow first one", async () => {
    let releaseFirst;
    let firstReached;
    const firstInFlight = new Promise((resolve) => {
      firstReached = resolve;
    });
    inspectWorkflowFile
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            releaseFirst = () =>
              resolve(
                workflowBody({ workflow: { ...workflowBody().workflow, prompt: "stale" } }),
              );
            firstReached();
          }),
      )
      .mockResolvedValueOnce(workflowBody());
    await render({});
    let firstDrop;
    await act(async () => {
      firstDrop = hook.handleDroppedFile(pngFile("first.png"));
      // Wait until the first inspect is genuinely in flight, so the race under test is
      // "a response arrives late", not "a drop was superseded before it left".
      await firstInFlight;
    });
    await act(async () => {
      await hook.handleDroppedFile(pngFile("second.png"));
    });
    await act(async () => {
      releaseFirst();
      await firstDrop;
    });
    expect(hook.offer?.share?.prompt).toBe("a lighthouse in heavy fog");
  });

  it("hands the recipe to the shared launch and closes", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    const launchRecipe = vi.fn(async () => true);
    await render({ launchRecipe });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    await act(async () => {
      await hook.useWorkflow();
    });
    expect(launchRecipe).toHaveBeenCalledTimes(1);
    const { recipe, replaySeed } = launchRecipe.mock.calls[0][0];
    expect(recipe.model).toBe("krea_2_turbo");
    expect(recipe.prompt).toBe("a lighthouse in heavy fog");
    // The shared image's own seed, and ONE image — not the batch of four it came from.
    expect(replaySeed).toBe("4242");
    expect(recipe.normalizedSettings.count).toBe(1);
    expect(hook.offer).toBeNull();
  });

  it("imports through the ordinary upload path and keeps the offer open", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    const importAsset = vi.fn(async () => ({ id: "asset_9" }));
    await render({ importAsset });
    const file = pngFile();
    await act(async () => {
      await hook.handleDroppedFile(file);
    });
    await act(async () => {
      await hook.importImage();
    });
    expect(importAsset).toHaveBeenCalledWith(file, { throwOnError: true });
    expect(hook.importState.done).toBe(true);
    // "too", not "instead": the workflow is still there to take into the studio.
    expect(hook.offer?.share).toBeTruthy();
  });

  it("reports an import failure without losing the offer", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    await render({ importAsset: vi.fn(async () => { throw new Error("Create or open a project first."); }) });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    await act(async () => {
      await hook.importImage();
    });
    expect(hook.importState.error).toBe("Create or open a project first.");
    expect(hook.offer?.share).toBeTruthy();
  });

  it("dismisses the offer, and a late inspect cannot reopen it", async () => {
    let release;
    let reached;
    const inFlight = new Promise((resolve) => {
      reached = resolve;
    });
    inspectWorkflowFile.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => resolve(workflowBody());
          reached();
        }),
    );
    await render({});
    let pending;
    await act(async () => {
      pending = hook.handleDroppedFile(pngFile());
      await inFlight;
    });
    await act(async () => hook.dismiss());
    await act(async () => {
      release();
      await pending;
    });
    expect(hook.offer).toBeNull();
  });
});

describe("inspectFailureMessage", () => {
  it("passes through the sentences the API wrote for a person", () => {
    expect(
      inspectFailureMessage(
        new ApiError("Two workflow chunks.", { status: 422, code: "workflow_inspect_unreadable" }),
      ),
    ).toBe("Two workflow chunks.");
  });

  it("is null for everything else, including a body with no code at all", () => {
    expect(inspectFailureMessage(null)).toBeNull();
    expect(inspectFailureMessage(new ApiError("boom", { status: 400 }))).toBeNull();
    expect(
      inspectFailureMessage(new ApiError("", { status: 500, code: "workflow_inspect_read_failed" })),
    ).toBeNull();
  });
});
