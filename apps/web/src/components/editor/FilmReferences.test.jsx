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
    planning: { provider: "prompt_refiner" },
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
let mutateLatestDraft;
let saveDraftMock;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  apiFetchMock.mockReset();
  latestDraft = null;
  mutateLatestDraft = null;
  saveDraftMock = vi.fn();
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
});

async function renderReferences({ assets = [], draft = filmDraft(), findings = [], importAsset = vi.fn() } = {}) {
  function Harness() {
    const [current, setCurrent] = useState(draft);
    latestDraft = current;
    const mutate = (mutator) => setCurrent((value) => {
      const next = structuredClone(value);
      mutator(next);
      return next;
    });
    mutateLatestDraft = mutate;
    return (
      <FilmReferences
        assets={assets}
        draft={current}
        findings={findings}
        importAsset={importAsset}
        onDraftChange={mutate}
        onReplaceDraft={setCurrent}
        saveDraft={async (options = {}) => {
          const saved = { ...current, revision: current.revision + 1 };
          saveDraftMock(options, saved);
          if (options.updateLocal !== false) setCurrent(saved);
          return saved;
        }}
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
      // sc-24024: which subject in the image this role names, for the pair of roles that share one.
      setControl(container.querySelector('[aria-label="Reference locator"]'), "the woman on the left");
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
      locator: "the woman on the left",
      approved: true,
    }));
    expect(latestDraft.referencePack.references[0].sourceAssetId).toBe("asset_courier");
  });

  it("merges a delayed added reference without overwriting newer draft or reference edits", async () => {
    const asset = {
      id: "asset_courier",
      projectId: "project_1",
      displayName: "Courier Portrait",
      type: "image",
      file: { mimeType: "image/png" },
      status: {},
    };
    const initial = filmDraft([{
      role: "hero",
      kind: "character",
      file: "references/hero.png",
      sourceAssetId: "asset_hero",
      description: "Original description",
      approved: true,
      generated: false,
    }]);
    const returned = structuredClone(initial);
    returned.revision = 3;
    returned.updatedAt = "2026-09-17T12:00:00Z";
    returned.productionPlan.version = 3;
    returned.referencePack.version = 2;
    returned.referencePack.references.push({
      role: "courier",
      kind: "character",
      file: "references/asset_courier.png",
      sourceAssetId: asset.id,
      description: "",
      approved: false,
      generated: false,
    });
    let resolveAdd;
    apiFetchMock.mockReturnValue(new Promise((resolve) => { resolveAdd = resolve; }));
    await renderReferences({ assets: [asset], draft: initial });

    await act(async () => {
      setControl(container.querySelector('[aria-label="Reference project image"]'), asset.id);
      setControl(container.querySelector('[aria-label="Reference role name"]'), "courier");
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add asset").click();
      await Promise.resolve();
    });
    expect(saveDraftMock).toHaveBeenCalledWith(
      { updateLocal: false },
      expect.objectContaining({ revision: 2 }),
    );
    expect(apiFetchMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      mutateLatestDraft((current) => {
        current.planning.provider = "openai_compatible";
        current.referencePack.references[0].description = "Edited while the asset copied";
        current.referencePack.references.push({
          role: "local_style",
          kind: "style",
          file: "references/local.png",
          description: "Unsaved reference",
          approved: true,
          generated: false,
        });
      });
    });
    await act(async () => { resolveAdd(returned); await Promise.resolve(); await Promise.resolve(); });

    expect(latestDraft.planning.provider).toBe("openai_compatible");
    expect(latestDraft.referencePack.references.map((reference) => reference.role))
      .toEqual(["hero", "local_style", "courier"]);
    expect(latestDraft.referencePack.references[0].description).toBe("Edited while the asset copied");
    expect(latestDraft.revision).toBe(3);
    expect(latestDraft.referencePack.version).toBe(2);
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

  // sc-24028 / sc-24025. A DESCRIBED-ONLY role is authored through the SAME route with no image.
  // The body shape is asserted exactly, key for key, against `FilmReferenceInput`: that struct is
  // `deny_unknown_fields`, and it reads `assetId`/`draftRevision` in camelCase, so an extra or
  // misnamed key here is a 400 the panel would only discover against a real server.
  it("adds a role with no image by posting no assetId key at all", async () => {
    const added = filmDraft([{
      role: "recipient",
      kind: "character",
      description: "The recipient: grey work apron.",
      approved: true,
      generated: false,
    }]);
    added.revision = 3;
    apiFetchMock.mockResolvedValueOnce(added);
    await renderReferences();

    await act(async () => {
      setControl(container.querySelector('[aria-label="Reference role name"]'), "recipient");
      setControl(container.querySelector('[aria-label="Reference description"]'), "The recipient: grey work apron.");
      container.querySelector('.ve-film-reference-check input').click();
    });
    const add = [...container.querySelectorAll("button")].find((button) => button.textContent === "Add described role");
    await act(async () => { add.click(); await Promise.resolve(); });

    expect(apiFetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/project_1/films/film_1/references",
      "token",
      expect.objectContaining({ method: "POST" }),
    );
    const body = JSON.parse(apiFetchMock.mock.calls[0][2].body);
    expect(body).toEqual({
      draftRevision: 2,
      role: "recipient",
      kind: "character",
      description: "The recipient: grey work apron.",
      approved: true,
    });
    expect(Object.keys(body)).not.toContain("assetId");
    expect(Object.keys(body)).not.toContain("locator");
    // The stored entry is fileless, and it is what the panel now holds.
    expect(latestDraft.referencePack.references[0]).toMatchObject({ role: "recipient" });
    expect(latestDraft.referencePack.references[0].file).toBeUndefined();
    expect(latestDraft.referencePack.references[0].sourceAssetId).toBeUndefined();
  });

  it("shows a described-only role without a locator field and refuses a blank description in the core's words", async () => {
    const draft = filmDraft([
      { role: "recipient", kind: "character", description: "Grey work apron.", approved: true, generated: false },
      { role: "hero", kind: "character", file: "references/hero.png", description: "Blue jacket.", approved: true, generated: false },
    ]);
    await renderReferences({ draft });

    const rows = [...container.querySelectorAll(".ve-film-reference-row")];
    // No image: a plain indication instead of the locator control, because the core refuses a
    // locator on a role that has no picture to pick a subject out of.
    expect(rows[0].querySelector(".ve-film-reference-described").textContent)
      .toBe("no image — described in text");
    expect(rows[0].querySelector('[aria-label="Reference recipient locator"]')).toBeNull();
    // Its description is the whole of the role, so the field says it is required.
    expect(rows[0].querySelector('[aria-label="Reference recipient description"]').getAttribute("aria-required"))
      .toBe("true");
    // An image-backed role is shown differently: it keeps its locator field and names its file.
    expect(rows[1].querySelector(".ve-film-reference-described")).toBeNull();
    expect(rows[1].querySelector('[aria-label="Reference hero locator"]')).not.toBeNull();
    expect(rows[1].textContent).toContain("references/hero.png");

    // Description, kind and approval stay editable on the described-only role.
    await act(async () => {
      setControl(rows[0].querySelector('[aria-label="Reference recipient description"]'), "The recipient: grey work apron, holds the door.");
      setControl(rows[0].querySelector('[aria-label="Reference recipient kind"]'), "prop");
      rows[0].querySelector('input[type="checkbox"]').click();
    });
    expect(latestDraft.referencePack.references[0]).toMatchObject({
      description: "The recipient: grey work apron, holds the door.",
      kind: "prop",
      approved: false,
    });

    // The server's refusal of a blank description reaches the add form, in the core's own words.
    apiFetchMock.mockRejectedValueOnce(new Error(
      "[plan] referencePack.references[2].description: reference \"silent\" has no `file` and no `description`: a role with no image is DESCRIBED-ONLY, so its description is the whole of it — give it one, or give the role an image",
    ));
    await act(async () => { setControl(container.querySelector('[aria-label="Reference role name"]'), "silent"); });
    const add = [...container.querySelectorAll("button")].find((button) => button.textContent === "Add described role");
    await act(async () => { add.click(); await Promise.resolve(); await Promise.resolve(); });
    const refusal = container.querySelector(".ve-film-reference-add .ve-film-findings");
    expect(refusal.textContent).toContain("has no `file` and no `description`");
    // The operator is shown the message, never the diagnostic's internal scope and field path.
    expect(refusal.textContent).not.toContain("[plan]");
    expect(refusal.textContent).not.toContain("referencePack.references[");

    // Editing the pack answers the refusal, so it stops being shown.
    await act(async () => {
      setControl(rows[0].querySelector('[aria-label="Reference recipient description"]'), "Grey work apron, holds the door.");
    });
    expect(container.querySelector(".ve-film-reference-add .ve-film-findings")).toBeNull();
  });

  // sc-24028. The two adjacent buttons take different bodies. "Add described role" reads neither
  // the image nor the locator, so leaving it clickable with either filled in would silently
  // discard the operator's choice and store a fileless role with a success notice.
  it("disables Add described role while an image or a locator is filled in", async () => {
    const asset = {
      id: "asset_courier",
      projectId: "project_1",
      displayName: "Courier",
      type: "image",
      file: { mimeType: "image/png" },
      status: {},
    };
    await renderReferences({ assets: [asset] });
    const describedButton = () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent === "Add described role");

    await act(async () => { setControl(container.querySelector('[aria-label="Reference role name"]'), "recipient"); });
    expect(describedButton().disabled).toBe(false);

    await act(async () => { setControl(container.querySelector('[aria-label="Reference project image"]'), "asset_courier"); });
    expect(describedButton().disabled).toBe(true);

    await act(async () => { setControl(container.querySelector('[aria-label="Reference project image"]'), ""); });
    expect(describedButton().disabled).toBe(false);

    await act(async () => { setControl(container.querySelector('[aria-label="Reference locator"]'), "the woman on the left"); });
    expect(describedButton().disabled).toBe(true);

    // With no image chosen the description is the whole of the role, so the field says so.
    await act(async () => { setControl(container.querySelector('[aria-label="Reference locator"]'), ""); });
    const addDescription = container.querySelector('.ve-film-reference-add [aria-label="Reference description"]');
    expect(addDescription.required).toBe(true);
    expect(addDescription.getAttribute("aria-required")).toBe("true");
    await act(async () => { setControl(container.querySelector('[aria-label="Reference project image"]'), "asset_courier"); });
    expect(container.querySelector('.ve-film-reference-add [aria-label="Reference description"]').required).toBe(false);
  });

  // sc-24028 / sc-24024. Adding one asset under a second role leaves TWO entries on ONE file, and
  // the core requires a distinct locator on each. The panel has to say so at the moment it happens.
  it("marks both locators required when two roles share one image and surfaces the core's refusal", async () => {
    const asset = {
      id: "asset_pair",
      projectId: "project_1",
      displayName: "Pair",
      type: "image",
      file: { mimeType: "image/png" },
      status: {},
    };
    const draft = filmDraft([{
      role: "courier",
      kind: "character",
      file: "references/asset_pair.png",
      sourceAssetId: "asset_pair",
      description: "Blue jacket.",
      approved: true,
      generated: false,
    }]);
    await renderReferences({ assets: [asset], draft });

    // Choosing the image a role already uses marks the add form's locator required and names it.
    await act(async () => {
      setControl(container.querySelector('[aria-label="Reference project image"]'), "asset_pair");
    });
    const addLocator = container.querySelector('[aria-label="Reference locator"]');
    expect(addLocator.required).toBe(true);
    expect(addLocator.getAttribute("aria-required")).toBe("true");
    expect(container.querySelector(".ve-film-reference-add").textContent)
      .toContain("This image already backs courier");

    // The server refuses the unlocated pair; the message lands under the field, not only in a notice.
    apiFetchMock.mockRejectedValueOnce(new Error(
      "[plan] referencePack.references.locator: roles \"courier\", \"guard\" share the file \"references/asset_pair.png\", so each of them needs a `locator` saying which subject in that image it names",
    ));
    await act(async () => { setControl(container.querySelector('[aria-label="Reference role name"]'), "guard"); });
    const add = [...container.querySelectorAll("button")].find((button) => button.textContent === "Add asset");
    await act(async () => { add.click(); await Promise.resolve(); await Promise.resolve(); });
    const sharedRefusal = container.querySelector(".ve-film-reference-add .ve-film-findings");
    expect(sharedRefusal.textContent).toContain("share the file");
    expect(sharedRefusal.textContent).not.toContain("[plan]");
    expect(sharedRefusal.textContent).not.toContain("referencePack.references.locator:");

    // Once both roles are on that one file, BOTH rows mark their locator required and carry the
    // core's message under it.
    await act(async () => {
      mutateLatestDraft((current) => {
        current.referencePack.references.push({
          role: "guard",
          kind: "character",
          file: "references/asset_pair.png",
          sourceAssetId: "asset_pair",
          description: "Grey coat.",
          approved: true,
          generated: false,
        });
      });
    });
    const rows = [...container.querySelectorAll(".ve-film-reference-row")];
    for (const [index, role] of [[0, "courier"], [1, "guard"]]) {
      const field = rows[index].querySelector(`[aria-label="Reference ${role} locator"]`);
      expect(field.required).toBe(true);
      expect(field.getAttribute("aria-required")).toBe("true");
      expect(rows[index].textContent).toContain("Locator required — this image also backs");
    }

    // Each row can author its own locator, and the edits land on the right entries.
    await act(async () => {
      setControl(rows[0].querySelector('[aria-label="Reference courier locator"]'), "the woman on the left");
      setControl(rows[1].querySelector('[aria-label="Reference guard locator"]'), "the man on the right");
    });
    expect(latestDraft.referencePack.references.map((item) => item.locator))
      .toEqual(["the woman on the left", "the man on the right"]);
  });

  // sc-24028. A description is inserted into prompts word for word, so the core refuses `<`, `>`
  // and control characters in one — and that refusal has to appear under the field that holds it.
  it("surfaces pack findings under the row field they name", async () => {
    const draft = filmDraft([
      { role: "courier", kind: "character", file: "references/a.png", description: "Wears <Picture 2>.", locator: "the woman", approved: true, generated: false },
      { role: "guard", kind: "character", file: "references/a.png", description: "Grey coat.", locator: "the woman", approved: true, generated: false },
    ]);
    await renderReferences({
      draft,
      findings: [
        { field: "referencePack.references[0].description", message: "description must not contain '<' or '>'" },
        { field: "referencePack.references.locator", message: "roles \"courier\", \"guard\" share the file \"references/a.png\" and declare the same locator" },
        { shotId: "SH010", field: "audio", message: "a shot finding belongs to the shots panel" },
      ],
    });

    const rows = [...container.querySelectorAll(".ve-film-reference-row")];
    expect(rows[0].textContent).toContain("description must not contain '<' or '>'");
    expect(rows[1].textContent).not.toContain("description must not contain");
    // The shared-file refusal is shown under BOTH rows that share the file.
    for (const row of rows) expect(row.textContent).toContain("declare the same locator");
    // A shot's finding is not this panel's to show.
    expect(container.textContent).not.toContain("a shot finding belongs to the shots panel");
  });

  // sc-24028. A pack finding on a field no row renders beside — a duplicate role comes straight
  // out of the row's own rename input — must still be readable. `FilmWorkspace` counts every
  // finding in the step header, and `FilmShots`' catch-all only matches findings with a `shotId`,
  // so anything unrouted here is displayed nowhere at all.
  it("shows pack findings whose field no row renders, and leaves sound findings to the sound step", async () => {
    const draft = filmDraft([
      { role: "courier", kind: "character", file: "references/a.png", description: "Blue jacket.", approved: true, generated: false },
      { role: "courier", kind: "character", file: "references/b.png", description: "Grey coat.", approved: true, generated: false },
    ]);
    await renderReferences({
      draft,
      findings: [
        { field: "referencePack.references[0].role", message: "duplicate reference role \"courier\"" },
        { field: "referencePack.references[1].kind", message: "unknown reference kind \"costume\"" },
        { field: "referencePack.references[1].sourceAssetId", message: "source asset id must be 1-64 characters" },
        { field: "referencePack.sound[0].role", message: "duplicate sound role \"theme\"" },
        { field: "referencePack.references[0].description", message: "description must not contain '<' or '>'" },
      ],
    });

    const other = container.querySelector('[aria-label="Other reference pack findings"]');
    expect(other.textContent).toContain("duplicate reference role \"courier\"");
    expect(other.textContent).toContain("unknown reference kind \"costume\"");
    expect(other.textContent).toContain("source asset id must be 1-64 characters");
    // Already shown beside its row, so it is not repeated here.
    expect(other.textContent).not.toContain("description must not contain");
    // The sound half of the pack is authored on the Sound step, which renders it there.
    expect(container.textContent).not.toContain("duplicate sound role");
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
