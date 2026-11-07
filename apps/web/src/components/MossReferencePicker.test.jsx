import React, { act, useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot, click, setFileInput } from "../testUtils/dom.js";
import { MossReferencePicker } from "./MossReferencePicker.jsx";

let root, container;
afterEach(async () => { if (root) await unmountRoot(root, container); });
const image = (id, extra = {}) => ({ id, type: "image", projectId: "project", displayName: id, ...extra });
const characters = [{ id: "hero", name: "Hero", approvedReferences: [{ assetId: "portrait" }] }];
const initialAssets = [image("landscape"), image("portrait"), image("other-project", { projectId: "other" })];
const button = (scope, name) => [...scope.querySelectorAll("button")].find((node) => node.textContent === name);
const dialog = () => document.body.querySelector('[role="dialog"]');
const card = (id) => dialog().querySelector(`small[title="${id}"]`).closest('[role="option"]');
const slotIds = () => [...container.querySelectorAll(".asset-picker-field .asset-preview-chip small[title]")].map((node) => node.title);

function Picker({ initial = ["landscape"], max = 4, firstOrdinal = 1, importFile = vi.fn() }) {
  const [ids, setIds] = useState(initial);
  const [assets, setAssets] = useState(initialAssets);
  return <MossReferencePicker assets={assets} assetIds={ids} onChange={setIds} characters={characters}
    projectId="project" max={max} firstOrdinal={firstOrdinal}
    importAsset={async (...args) => { const imported = await importFile(...args); setAssets((items) => [...items, imported]); return imported; }} />;
}
async function render(props) {
  ({ root, container } = mountRoot());
  await act(async () => root.render(<Picker {...props} />));
}

describe("MOSS references use the shared image-source dialog", () => {
  it("selects project and character assets in ordered, repeatable slots and cancels safely", async () => {
    await render({ initial: ["landscape", "landscape"] });
    const slots = () => container.querySelectorAll(".asset-picker-field");
    await click(slots()[1].querySelector('button[aria-haspopup="dialog"]'));
    expect(dialog().textContent).toContain("Choose image 2");
    expect([...dialog().querySelectorAll('[role="tab"]')].map((node) => node.textContent.replace(/\d/g, ""))).toEqual(["Assets", "File Upload", "Character"]);
    expect(dialog().textContent).not.toContain("other-project");
    await click(card("landscape"));
    await click(button(dialog(), "Cancel"));
    expect(slotIds()).toEqual(["landscape", "landscape"]);
    await click(slots()[1].querySelector('button[aria-haspopup="dialog"]'));
    await click([...dialog().querySelectorAll('[role="tab"]')].find((node) => node.textContent.startsWith("Character")));
    await click(card("portrait"));
    await click(button(dialog(), "Use Selection"));
    expect(slotIds()).toEqual(["landscape", "portrait"]);
    await click(button(container, "Repeat image 2"));
    expect(slotIds()).toEqual(["landscape", "portrait", "portrait"]);
    await click(container.querySelector('button[aria-label="Move Image 3 earlier"]'));
    await click(button(container, "Remove image 1"));
    expect(slotIds()).toEqual(["portrait", "portrait"]);
  });

  it("uploads directly into an empty slot and preserves Editor numbering and limits", async () => {
    const importFile = vi.fn(async () => image("uploaded"));
    await render({ initial: [], max: 3, firstOrdinal: 2, importFile });
    await click(button(container, "Add reference image"));
    await click(button(container, "Select image"));
    expect(dialog().textContent).toContain("Choose image 2");
    await click(button(dialog(), "File Upload"));
    const file = new File(["pixels"], "reference.png", { type: "image/png" });
    await act(async () => setFileInput(dialog().querySelector('input[type="file"]'), [file]));
    expect(importFile).toHaveBeenCalledWith(file, { select: false, throwOnError: true });
    expect(dialog()).toBeNull();
    expect(slotIds()).toEqual(["uploaded"]);
    await click(button(container, "Repeat image 2"));
    await click(button(container, "Repeat image 3"));
    expect(slotIds()).toEqual(["uploaded", "uploaded", "uploaded"]);
    expect(button(container, "Add reference image").disabled).toBe(true);
    expect(button(container, "Repeat image 4").disabled).toBe(true);
  });
});
