import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { usePresets } from "./usePresets.js";
import { usePromptBatches } from "./usePromptBatches.js";
import { useTraining } from "./useTraining.js";

const pendingRequests = [];
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(
      (url, _token, options) =>
        new Promise((resolve, reject) => {
          pendingRequests.push({ url, options, resolve, reject });
        }),
    ),
  };
});

function mountHook(hook, activeProjectRef, extraProps = {}) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latest;

  function Harness() {
    latest = hook({
      token: "token",
      activeProject: activeProjectRef.current,
      activeProjectRef,
      setError: extraProps.setError ?? vi.fn(),
      setJobs: vi.fn(),
    });
    return null;
  }

  act(() => root.render(<Harness />));
  return {
    get: () => latest,
    unmount: () => {
      act(() => root.unmount());
      container.remove();
    },
  };
}

const CATALOG_CASES = [
  {
    name: "presets",
    hook: usePresets,
    refresh: "refreshPresets",
    state: "presets",
    create: "createPreset",
    path: "/api/v1/recipe-presets",
  },
  {
    name: "prompt batches",
    hook: usePromptBatches,
    refresh: "refreshPromptBatches",
    state: "promptBatches",
    create: "createPromptBatch",
    path: "/api/v1/prompt-batches",
  },
];

describe.each(CATALOG_CASES)(
  "$name stale-project guards (sc-13626)",
  ({ hook, refresh, state, create, path }) => {
    let mounted;

    afterEach(() => mounted?.unmount());

    it("drops project A success and error commits after switching to B", async () => {
      const activeProjectRef = { current: { id: "A" } };
      const setError = vi.fn();
      mounted = mountHook(hook, activeProjectRef, { setError });

      let success;
      act(() => {
        success = mounted.get()[refresh]("A");
      });
      activeProjectRef.current = { id: "B" };
      await act(async () => {
        pendingRequests[0].resolve([{ id: "from-A" }]);
        await success;
      });

      expect(mounted.get()[state]).toEqual([]);
      expect(setError).not.toHaveBeenCalled();

      activeProjectRef.current = { id: "A" };
      let failure;
      act(() => {
        failure = mounted.get()[refresh]("A");
      });
      activeProjectRef.current = { id: "B" };
      await act(async () => {
        pendingRequests[1].reject(new Error("stale A failure"));
        await failure;
      });

      expect(mounted.get()[state]).toEqual([]);
      expect(setError).not.toHaveBeenCalled();
    });

    it("does not launch a project A post-mutation refetch after switching to B", async () => {
      const activeProjectRef = { current: { id: "A" } };
      mounted = mountHook(hook, activeProjectRef);

      let mutation;
      act(() => {
        mutation = mounted.get()[create]({ scope: "project", name: "A item" });
      });
      expect(pendingRequests[0].url).toBe(`${path}?scope=project&projectId=A`);

      activeProjectRef.current = { id: "B" };
      await act(async () => {
        pendingRequests[0].resolve({ id: "created-in-A" });
        await mutation;
      });

      expect(pendingRequests).toHaveLength(1);
    });

    it("preserves global catalog refreshes across project changes", async () => {
      const activeProjectRef = { current: { id: "A" } };
      mounted = mountHook(hook, activeProjectRef);

      let refreshGlobal;
      act(() => {
        refreshGlobal = mounted.get()[refresh](null);
      });
      expect(pendingRequests[0].url).toBe(path);

      activeProjectRef.current = { id: "B" };
      await act(async () => {
        pendingRequests[0].resolve([{ id: "global" }]);
        await refreshGlobal;
      });

      expect(mounted.get()[state]).toEqual([{ id: "global" }]);
    });

    it("refreshes the active project view after a global mutation", async () => {
      const activeProjectRef = { current: { id: "A" } };
      mounted = mountHook(hook, activeProjectRef);

      let mutation;
      act(() => {
        mutation = mounted.get()[create]({ scope: "global", name: "global item" });
      });
      pendingRequests[0].resolve({ id: "global-created" });
      await act(async () => {
        await Promise.resolve();
      });

      expect(pendingRequests[1].url).toBe(`${path}?projectId=A`);
      await act(async () => {
        pendingRequests[1].resolve([{ id: "global-created" }]);
        await mutation;
      });
      expect(mounted.get()[state]).toEqual([{ id: "global-created" }]);
    });
  },
);

describe("useTraining stale-project guards (sc-13626)", () => {
  let mounted;

  afterEach(() => mounted?.unmount());

  it("keeps loading true when older same-project success resolves before the newest request", async () => {
    const activeProjectRef = { current: { id: "A" } };
    mounted = mountHook(useTraining, activeProjectRef);

    let refreshA1;
    let refreshA2;
    act(() => {
      refreshA1 = mounted.get().refreshTrainingDatasets("A");
      refreshA2 = mounted.get().refreshTrainingDatasets("A");
    });

    await act(async () => {
      pendingRequests[0].resolve([{ id: "older-A" }]);
      await refreshA1;
    });
    expect(mounted.get().loadingTrainingDatasets).toBe(true);
    expect(mounted.get().trainingDatasets).toEqual([]);
    expect(mounted.get().trainingDatasetsProjectId).toBeNull();

    await act(async () => {
      pendingRequests[1].resolve([{ id: "newest-A" }]);
      await refreshA2;
    });
    expect(mounted.get().loadingTrainingDatasets).toBe(false);
    expect(mounted.get().trainingDatasets).toEqual([{ id: "newest-A" }]);
    expect(mounted.get().trainingDatasetsProjectId).toBe("A");
  });

  it("does not let an older same-project error overwrite a newer success", async () => {
    const activeProjectRef = { current: { id: "A" } };
    mounted = mountHook(useTraining, activeProjectRef);

    let refreshA1;
    let refreshA2;
    act(() => {
      refreshA1 = mounted.get().refreshTrainingDatasets("A");
      refreshA2 = mounted.get().refreshTrainingDatasets("A");
    });

    await act(async () => {
      pendingRequests[1].resolve([{ id: "newest-A" }]);
      await refreshA2;
    });
    expect(mounted.get().loadingTrainingDatasets).toBe(false);
    expect(mounted.get().trainingDatasets).toEqual([{ id: "newest-A" }]);
    expect(mounted.get().trainingDatasetsProjectId).toBe("A");
    expect(mounted.get().trainingDatasetsError).toBe("");

    await act(async () => {
      pendingRequests[0].reject(new Error("older A failed"));
      await refreshA1;
    });
    expect(mounted.get().loadingTrainingDatasets).toBe(false);
    expect(mounted.get().trainingDatasets).toEqual([{ id: "newest-A" }]);
    expect(mounted.get().trainingDatasetsProjectId).toBe("A");
    expect(mounted.get().trainingDatasetsError).toBe("");
  });

  it("keeps project B loading and data authoritative when A resolves first", async () => {
    const activeProjectRef = { current: { id: "A" } };
    mounted = mountHook(useTraining, activeProjectRef);

    let refreshA;
    act(() => {
      refreshA = mounted.get().refreshTrainingDatasets("A");
    });

    activeProjectRef.current = { id: "B" };
    let refreshB;
    act(() => {
      refreshB = mounted.get().refreshTrainingDatasets("B");
    });
    expect(mounted.get().loadingTrainingDatasets).toBe(true);

    await act(async () => {
      pendingRequests[0].resolve([{ id: "from-A" }]);
      await refreshA;
    });
    expect(mounted.get().loadingTrainingDatasets).toBe(true);
    expect(mounted.get().trainingDatasets).toEqual([]);

    await act(async () => {
      pendingRequests[1].resolve([{ id: "from-B" }]);
      await refreshB;
    });
    expect(mounted.get().loadingTrainingDatasets).toBe(false);
    expect(mounted.get().trainingDatasets).toEqual([{ id: "from-B" }]);
    expect(mounted.get().trainingDatasetsProjectId).toBe("B");
  });

  it("keeps the A to null reset authoritative when A's request later rejects", async () => {
    const activeProjectRef = { current: { id: "A" } };
    mounted = mountHook(useTraining, activeProjectRef);

    let staleRefresh;
    act(() => {
      staleRefresh = mounted.get().refreshTrainingDatasets("A");
    });
    expect(mounted.get().loadingTrainingDatasets).toBe(true);

    activeProjectRef.current = null;
    await act(async () => {
      await mounted.get().refreshTrainingDatasets(null);
    });
    expect(mounted.get().loadingTrainingDatasets).toBe(false);
    expect(mounted.get().trainingDatasetsProjectId).toBeNull();
    expect(mounted.get().trainingDatasetsError).toBe("");

    await act(async () => {
      pendingRequests[0].reject(new Error("stale A failure"));
      await staleRefresh;
    });

    expect(mounted.get().loadingTrainingDatasets).toBe(false);
    expect(mounted.get().trainingDatasets).toEqual([]);
    expect(mounted.get().trainingDatasetsProjectId).toBeNull();
    expect(mounted.get().trainingDatasetsError).toBe("");
  });

  it("does not launch a project A post-mutation refetch after switching to B", async () => {
    const activeProjectRef = { current: { id: "A" } };
    mounted = mountHook(useTraining, activeProjectRef);

    let mutation;
    act(() => {
      mutation = mounted.get().updateTrainingDataset("dataset", { name: "updated" }, "A");
    });
    activeProjectRef.current = { id: "B" };
    await act(async () => {
      pendingRequests[0].resolve({ id: "dataset" });
      await mutation;
    });

    expect(pendingRequests).toHaveLength(1);
  });
});

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  pendingRequests.length = 0;
});
