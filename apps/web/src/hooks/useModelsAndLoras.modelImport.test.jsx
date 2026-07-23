// sc-14020: base-checkpoint import contract. `createModelImportJob` must key the model type as
// `type` — the literal field name the backend's multipart parser reads (models.rs
// `model_import_request_from_multipart`) — because a multipart upload gets none of the
// `#[serde(alias = "type")]` tolerance the JSON path enjoys. Keying it `modelType` (the old bug)
// was silently dropped on file uploads, defaulting every imported checkpoint to `image`. These
// tests drive the real hook with a mocked apiFetch and assert the outgoing request body carries
// the type under `type` on both the multipart (file) and JSON (URL) branches.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../api.js", () => ({
  apiFetch: (...args) => apiFetchMock(...args),
  isAbortError: () => false,
}));

import { useModelsAndLoras } from "./useModelsAndLoras.js";

let container;
let root;
let hookApi;

function Harness() {
  hookApi = useModelsAndLoras({
    token: "tok",
    activeProject: { id: "proj-1" },
    activeProjectRef: { current: { id: "proj-1" } },
    setError: () => {},
    setJobs: () => {},
    setActiveView: () => {},
    refreshData: async () => {},
    refreshDataWithLoraOverlay: async () => {},
  });
  return null;
}

beforeEach(async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  apiFetchMock.mockReset();
  apiFetchMock.mockResolvedValue({ id: "model-import-job-1", type: "model_import", status: "running" });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(<Harness />);
  });
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe("createModelImportJob type-field contract (sc-14020)", () => {
  it("multipart upload keys the type as `type` (the field the backend reads), never `modelType`", async () => {
    const file = new File([new Uint8Array([1, 2, 3])], "krea2-checkpoint.safetensors");

    await act(async () => {
      await hookApi.createModelImportJob({ file, type: "image", name: "krea2" });
    });

    expect(apiFetchMock).toHaveBeenCalledTimes(1);
    const [path, token, options] = apiFetchMock.mock.calls[0];
    expect(path).toBe("/api/v1/models/import");
    expect(token).toBe("tok");
    expect(options.method).toBe("POST");

    const body = options.body;
    expect(body).toBeInstanceOf(FormData);
    // The backend multipart parser reads `type`; `modelType` would be silently dropped.
    expect(body.get("type")).toBe("image");
    expect(body.get("modelType")).toBeNull();
    expect(body.get("name")).toBe("krea2");
    expect(body.get("file")).toBe(file);
  });

  it("URL (JSON) import keeps the type under `type` in the JSON body", async () => {
    await act(async () => {
      await hookApi.createModelImportJob({ sourceUrl: "https://example.com/krea2.safetensors", type: "image", name: "krea2" });
    });

    expect(apiFetchMock).toHaveBeenCalledTimes(1);
    const [, , options] = apiFetchMock.mock.calls[0];
    expect(typeof options.body).toBe("string");
    const payload = JSON.parse(options.body);
    // JSON deserialization accepts `type` via `#[serde(alias = "type")]` on ModelImportRequest.
    expect(payload.type).toBe("image");
    expect(payload.sourceUrl).toBe("https://example.com/krea2.safetensors");
    expect(payload.name).toBe("krea2");
  });
});
