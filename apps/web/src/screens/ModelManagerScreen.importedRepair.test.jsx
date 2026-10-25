import React, { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { useModelsAndLoras } from "../hooks/useModelsAndLoras.js";
import { ModelManagerScreen } from "./ModelManagerScreen.jsx";

const IMPORT = {
  id: "my_h3", name: "My H3", type: "video", family: "minimax-h3",
  catalogScope: "user", installState: "missing", downloadable: false,
  importPlan: { checkpointId: "managed/h3" },
  importedSource: {
    family: "minimax-h3", ownershipMode: "managed", sourceVariant: "ref2va",
    declaredPartition: "ref2va", baseModel: "minimax_h3_ref", componentTier: "q4",
  },
  source: { provider: "imported-checkpoint" },
  componentReadiness: { ready: false, components: [] },
};

let container, root, fetchMock, errors, jobs;
let libraryState;
function Harness({ model, initialJobs = [] }) {
  const [currentJobs, setJobs] = useState(initialJobs);
  jobs = currentJobs;
  const actions = useModelsAndLoras({
    token: "tok", activeProject: null, setError: (message) => errors.push(message),
    setJobs, setActiveView: () => {}, refreshData: async () => {},
    refreshDataWithLoraOverlay: async () => {},
  });
  return <AppContext.Provider value={{ ...actions, models: [model], loras: [], jobs: currentJobs,
    token: "tok", presets: [], jobAction: () => {},
    setActiveView: () => {}, activeProject: null,
  }}><ModelManagerScreen /></AppContext.Provider>;
}
function requests(suffix) {
  return fetchMock.mock.calls.filter(([url, options]) => url.endsWith(suffix) && options?.method === "POST");
}
function button(scope, text) {
  return [...scope.querySelectorAll("button")].find((node) => node.textContent.trim() === text);
}
async function click(node) {
  expect(node, "repair control exists").toBeTruthy();
  await act(async () => node.click());
}
async function input(node, value) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(node, value);
    node.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
async function render(model, initialJobs) {
  await act(async () => root.render(<Harness model={model} initialJobs={initialJobs} />));
  await click([...container.querySelectorAll('[role="tab"]')].find((node) => node.textContent.startsWith("Video")));
}
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  errors = []; jobs = []; libraryState = "needs_rescan";
  window.localStorage.clear();
  fetchMock = vi.fn(async (url, options = {}) => {
    let payload = {};
    if (url.endsWith("/library-roots")) payload = { roots: [{ rootId: "r", path: "/models" }] };
    if (url.endsWith("/r/scan")) payload = { available: libraryState !== "needs_relink", candidates: [], unmatched: [{
      checkpointId: "linked/r/h3", rootId: "r", relativePath: "h3", state: libraryState,
    }] };
    if (options.method === "POST" && url.endsWith("/download")) {
      const body = JSON.parse(options.body);
      return { ok: Boolean(body.componentId), status: body.componentId ? 200 : 400,
        json: async () => body.componentId ? { id: "repair-job", createdAt: "2026-10-24T00:00:00Z", type: "model_download", status: "queued", payload: { modelId: "my_h3" } }
          : { detail: "Imported model is not downloadable" } };
    }
    if (url.endsWith("/import")) payload = { id: "import-job", type: "model_import", status: "queued" };
    if (url.endsWith("/rescan")) payload = { checkpointId: "linked/r/h3", state: "ready" };
    if (url.endsWith("/host-capabilities")) payload = { totalMemoryBytes: 32 * 1024 ** 3 };
    return { ok: true, status: 200, json: async () => payload };
  });
  vi.stubGlobal("fetch", fetchMock);
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount()); container.remove(); vi.unstubAllGlobals();
});

describe("imported H3 card repair through real actions and HTTP boundary", () => {
  it.each([{ initialJobs: [] }, { initialJobs: [{ id: "failed", createdAt: "2026-10-23T00:00:00Z", type: "model_download", status: "failed", payload: { modelId: "my_h3" } }] }])(
    "downloads the named resident component rather than the non-downloadable import ($initialJobs)", async ({ initialJobs }) => {
      await render({ ...IMPORT, repairActions: [{ id: "audio", kind: "download", componentId: "audio_vae", label: "Repair audio" }] }, initialJobs);
      await click(button(container.querySelector(".imported-checkpoint-repairs"), "Repair audio"));
      const posts = requests("/my_h3/download");
      expect(posts).toHaveLength(1);
      expect(JSON.parse(posts[0][1].body)).toEqual({ requestedGpu: "auto", componentId: "audio_vae" });
      expect(jobs.some((job) => job.id === "repair-job")).toBe(true);
      expect(errors).not.toContain("Imported model is not downloadable");
    },
  );

  it("opens managed reimport options without downloading the damaged transformer", async () => {
    await render({ ...IMPORT, repairActions: [{ id: "source", kind: "reimport", label: "Reimport transformer" }] });
    await click(button(container.querySelector(".imported-checkpoint-repairs"), "Reimport transformer"));
    expect(container.querySelector('.checkpoint-import-toggle').getAttribute("aria-expanded")).toBe("true");
    expect(container.querySelector('form[aria-label="Add to SceneWorks"]')).toBeTruthy();
    expect(requests("/my_h3/download")).toHaveLength(0);
    expect(container.querySelector('[role="radio"][aria-checked="true"]').textContent).toContain("Add to SceneWorks");
    const form = container.querySelector('form[aria-label="Add to SceneWorks"]');
    await click(button(form, "URL"));
    const urlInput = [...form.querySelectorAll("label")].find((node) => node.textContent.trim() === "Source URL").querySelector("input");
    await input(urlInput, "https://example.test/h3.safetensors");
    await act(async () => form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
    expect(requests("/import")).toHaveLength(1);
    expect(JSON.parse(requests("/import")[0][1].body)).toEqual({
      ownershipMode: "managed", type: "video", name: "My H3", family: "minimax-h3",
      baseModel: "minimax_h3_ref", minimaxH3Partition: "ref2va",
      source: { kind: "url", url: "https://example.test/h3.safetensors" },
    });
  });

  it("keeps edited import drafts and does not reopen a dismissed repair on catalog rerender", async () => {
    const model = { ...IMPORT, repairActions: [{ id: "source", kind: "reimport", label: "Reimport transformer" }] };
    await render(model);
    await click(button(container.querySelector(".imported-checkpoint-repairs"), "Reimport transformer"));
    let form = container.querySelector('form[aria-label="Add to SceneWorks"]');
    await input([...form.querySelectorAll("label")].find((node) => node.textContent.trim() === "Name").querySelector("input"), "Edited name");
    await click(button(form, "URL"));
    await input(form.querySelector('input[placeholder="https://..."]'), "https://example.test/new.safetensors");
    await click(button(container, "Hide options"));
    await render({ ...model });
    expect(container.querySelector('.checkpoint-import-toggle').getAttribute("aria-expanded")).toBe("false");
    expect(requests("/import")).toHaveLength(0);
    await click(button(container.querySelector(".imported-checkpoint-repairs"), "Reimport transformer"));
    form = container.querySelector('form[aria-label="Add to SceneWorks"]');
    expect([...form.querySelectorAll("label")].find((node) => node.textContent.trim() === "Name").querySelector("input").value).toBe("Edited name");
    expect(form.querySelector('input[placeholder="https://..."]').value).toBe("https://example.test/new.safetensors");
  });

  it("does not offer managed reimport for a linked checkpoint", async () => {
    await render({ ...IMPORT, importPlan: { checkpointId: "linked/r/h3" },
      source: { provider: "linked-library", rootId: "r", relativePath: "h3" },
      importedSource: { ...IMPORT.importedSource, ownershipMode: "linked" },
      repairActions: [{ id: "source", kind: "reimport", label: "Reimport transformer" }],
    });
    await click(button(container.querySelector(".imported-checkpoint-repairs"), "Reimport transformer"));
    expect(container.querySelector('.checkpoint-import-toggle').getAttribute("aria-expanded")).toBe("false");
    expect(requests("/import")).toHaveLength(0);
  });

  it("rescans linked source repairs in place, never downloading or transferring ownership", async () => {
    await render({ ...IMPORT, importPlan: { checkpointId: "linked/r/h3" },
      source: { provider: "linked-library", rootId: "r", relativePath: "h3" },
      importedSource: { ...IMPORT.importedSource, ownershipMode: "linked" },
      repairActions: [{ id: "source", kind: "rescan", label: "Repair linked transformer" }],
    });
    await click(button(container.querySelector(".imported-checkpoint-repairs"), "Repair linked transformer"));
    expect(requests("/r/rescan")).toHaveLength(1);
    expect(JSON.parse(requests("/r/rescan")[0][1].body)).toEqual({ relativePath: "h3" });
    expect(requests("/my_h3/download")).toHaveLength(0);
    expect(requests("/import")).toHaveLength(0);
  });
});
