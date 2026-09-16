import React, { useState } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../../api.js", () => ({ apiFetch: apiFetchMock }));

import { FilmReferences } from "./FilmReferences.jsx";

function filmDraft(references = []) {
  return {
    id: "film_1",
    projectId: "project_1",
    revision: 1,
    referencePack: {
      schemaVersion: 1,
      id: "film_1-references",
      version: 1,
      description: "",
      references,
      sound: [],
    },
    productionPlan: {
      shots: [{
        id: "SH010",
        conditioning: { mode: "reference_to_video", referenceRoles: references.map((item) => item.role) },
        continuityRoles: [],
      }],
    },
  };
}

let container;
let root;
let latestDraft;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  apiFetchMock.mockReset();
  latestDraft = null;
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
});

async function renderReferences({ assets = [], draft = filmDraft(), importAsset = vi.fn() } = {}) {
  function Harness() {
    const [current, setCurrent] = useState(draft);
    latestDraft = current;
    const mutate = (mutator) => setCurrent((value) => {
      const next = structuredClone(value);
      mutator(next);
      return next;
    });
    return (
      <FilmReferences
        assets={assets}
        draft={current}
        importAsset={importAsset}
        onDraftChange={mutate}
        onReplaceDraft={setCurrent}
        saveDraft={async () => ({ ...current, revision: current.revision + 1 })}
        setNotice={vi.fn()}
        token="token"
      />
    );
  }
  root = createRoot(container);
  await act(async () => { root.render(<Harness />); });
}

function setControl(control, value) {
  const descriptor = Object.getOwnPropertyDescriptor(
    control.tagName === "SELECT" ? window.HTMLSelectElement.prototype : window.HTMLInputElement.prototype,
    "value",
  );
  descriptor.set.call(control, value);
  control.dispatchEvent(new Event("change", { bubbles: true }));
}

describe("FilmReferences", () => {
  it("adds an existing project asset with its role, kind, approval, and source identity", async () => {
    const asset = {
      id: "asset_courier",
      projectId: "project_1",
      displayName: "Courier Portrait",
      type: "image",
      file: { mimeType: "image/png" },
      status: {},
    };
    const added = filmDraft([{
      role: "courier",
      kind: "character",
      file: "references/asset_courier.png",
      sourceAssetId: asset.id,
      description: "Blue jacket",
      approved: true,
      generated: false,
    }]);
    added.revision = 3;
    apiFetchMock.mockResolvedValueOnce(added);
    await renderReferences({ assets: [asset] });

    await act(async () => {
      setControl(container.querySelector('[aria-label="Reference project image"]'), asset.id);
      setControl(container.querySelector('[aria-label="Reference role name"]'), "courier");
      setControl(container.querySelector('[aria-label="Reference description"]'), "Blue jacket");
      container.querySelector('.ve-film-reference-check input').click();
    });
    const add = [...container.querySelectorAll("button")].find((button) => button.textContent === "Add asset");
    await act(async () => { add.click(); await Promise.resolve(); });

    expect(apiFetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/project_1/films/film_1/references",
      "token",
      expect.objectContaining({ method: "POST" }),
    );
    expect(JSON.parse(apiFetchMock.mock.calls[0][2].body)).toEqual(expect.objectContaining({
      draftRevision: 2,
      assetId: "asset_courier",
      role: "courier",
      kind: "character",
      description: "Blue jacket",
      approved: true,
    }));
    expect(latestDraft.referencePack.references[0].sourceAssetId).toBe("asset_courier");
  });

  it("imports an uploaded image into the project before adding it to the pack", async () => {
    const imported = {
      id: "asset_uploaded",
      projectId: "project_1",
      displayName: "Uploaded plate",
      type: "image",
      file: { mimeType: "image/png" },
      status: {},
    };
    const importAsset = vi.fn().mockResolvedValue(imported);
    const added = filmDraft([{
      role: "uploaded_plate",
      kind: "character",
      file: "references/asset_uploaded.png",
      sourceAssetId: imported.id,
      description: "",
      approved: false,
      generated: false,
    }]);
    added.revision = 3;
    apiFetchMock.mockResolvedValueOnce(added);
    await renderReferences({ importAsset });
    const input = [...container.querySelectorAll('input[type="file"]')]
      .find((element) => element.accept.startsWith("image/"));
    const file = new File(["pixels"], "Uploaded plate.png", { type: "image/png" });
    Object.defineProperty(input, "files", { configurable: true, value: [file] });

    await act(async () => { input.dispatchEvent(new Event("change", { bubbles: true })); await Promise.resolve(); });

    expect(importAsset).toHaveBeenCalledWith(file, { select: false, throwOnError: true });
    expect(JSON.parse(apiFetchMock.mock.calls[0][2].body)).toEqual(expect.objectContaining({
      assetId: "asset_uploaded",
      role: "uploaded_plate",
      kind: "character",
      approved: false,
    }));
    expect(latestDraft.referencePack.references[0].sourceAssetId).toBe("asset_uploaded");
  });

  it("keeps approved per-shot bindings ordered and removes the mode when the last binding leaves", async () => {
    const draft = filmDraft([
      { role: "hero", kind: "character", file: "references/hero.png", approved: true, generated: false },
      { role: "parcel", kind: "prop", file: "references/parcel.png", approved: true, generated: false },
      { role: "look", kind: "style", file: "references/look.png", approved: true, generated: false },
      { role: "pending", kind: "location", file: "references/pending.png", approved: false, generated: false },
    ]);
    draft.productionPlan.shots[0].conditioning.referenceRoles = ["hero", "parcel"];
    await renderReferences({
      draft,
    });
    expect([...container.querySelectorAll('[aria-label="Reference binding role"] option')].map((item) => item.value))
      .toEqual(["", "hero", "parcel"]);

    const moveParcelUp = container.querySelector('[aria-label="Move parcel up"]');
    await act(async () => { moveParcelUp.click(); });
    expect(latestDraft.productionPlan.shots[0].conditioning.referenceRoles.slice(0, 2))
      .toEqual(["parcel", "hero"]);

    await act(async () => { container.querySelector('[aria-label="Remove parcel binding"]').click(); });
    await act(async () => { container.querySelector('[aria-label="Remove hero binding"]').click(); });
    expect(latestDraft.productionPlan.shots[0].conditioning.referenceRoles).toEqual([]);
    expect(latestDraft.productionPlan.shots[0].conditioning.mode).toBe("text_to_video");
  });
});
