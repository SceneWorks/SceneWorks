import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ImageEditSourcePickerField } from "./AssetPicker.jsx";

// The picker was previously "Change"/"Select"-only: once an optional source (img2img reference,
// control image, second edit image) was picked there was no way to un-pick it, so it kept driving
// generations until reload. `clearable` adds an opt-in "Remove" control that resets to "".
describe("ImageEditSourcePickerField clear affordance", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(ui) {
    await act(async () => root.render(ui));
  }

  const buttonByLabel = (label) =>
    [...container.querySelectorAll("button")].find((b) => b.textContent.trim() === label);

  it("renders a Remove control that clears the selection when clearable with a value set", async () => {
    const onChange = vi.fn();
    await render(
      <ImageEditSourcePickerField assets={[]} clearable label="Reference image" onChange={onChange} value="a1" />,
    );
    const remove = buttonByLabel("Remove");
    expect(remove).toBeTruthy();
    await act(async () => remove.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onChange).toHaveBeenCalledWith("");
  });

  it("omits Remove when clearable but nothing is selected", async () => {
    await render(
      <ImageEditSourcePickerField assets={[]} clearable label="Reference image" onChange={() => {}} value="" />,
    );
    expect(buttonByLabel("Remove")).toBeFalsy();
  });

  it("omits Remove when not clearable even with a value (required edit source)", async () => {
    await render(
      <ImageEditSourcePickerField assets={[]} label="Source image" onChange={() => {}} value="a1" />,
    );
    expect(buttonByLabel("Remove")).toBeFalsy();
  });
});

// The source picker splits the project library into two disjoint tabs: "Assets"
// (general images) and "Character" (images that already belong to a character).
// The Assets tab must EXCLUDE character-owned images — otherwise every character
// asset shows up on both tabs and the split does nothing (the "not filtering"
// bug: the Assets tab was rendering the whole library).
describe("ImageEditSourcePickerField Assets tab excludes character assets", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(ui) {
    await act(async () => root.render(ui));
  }

  const click = async (el) =>
    act(async () => el.dispatchEvent(new MouseEvent("click", { bubbles: true })));

  // Modal portals to document.body, so grid/tab queries target the document.
  const gridTitles = () =>
    [...document.body.querySelectorAll('.asset-picker-grid [role="option"] strong')].map((el) =>
      el.textContent.trim(),
    );
  const tabBadge = (label) => {
    const tab = [...document.body.querySelectorAll('[role="tab"]')].find((b) =>
      b.textContent.startsWith(label),
    );
    return tab?.querySelector("span")?.textContent ?? null;
  };

  const asset = (id, displayName, extra = {}) => ({
    id,
    type: "image",
    projectId: "p1",
    url: `/${id}.png`,
    displayName,
    ...extra,
  });

  // a1: plain project image (belongs to no character) → Assets tab only.
  // a2: generated FOR character c1 (recipe) → Character tab only.
  // a3: an approved reference of c1 → Character tab only.
  const assets = [
    asset("a1", "Plain One"),
    asset("a2", "Hero Gen", { recipe: { normalizedSettings: { characterId: "c1" } } }),
    asset("a3", "Hero Ref"),
  ];
  const characters = [{ id: "c1", name: "Hero", approvedReferences: [{ assetId: "a3" }], references: [] }];

  it("shows only non-character images on the Assets tab and moves the rest to Character", async () => {
    await render(
      <ImageEditSourcePickerField
        assets={assets}
        buttonLabel="Select reference image"
        characters={characters}
        clearable
        label="Reference image"
        onChange={() => {}}
        projectId="p1"
        value=""
      />,
    );

    const openButton = container.querySelector('button[aria-haspopup="dialog"]');
    await click(openButton);

    // Assets tab is the default. Only the non-character image is listed; the two
    // character-owned images are excluded.
    expect(gridTitles()).toEqual(["Plain One"]);
    expect(tabBadge("Assets")).toBe("1");
    expect(tabBadge("Character")).toBe("2");

    // The Character tab (defaulting to the first character) holds the two excluded images.
    const characterTab = [...document.body.querySelectorAll('[role="tab"]')].find((b) =>
      b.textContent.startsWith("Character"),
    );
    await click(characterTab);
    expect(gridTitles().sort()).toEqual(["Hero Gen", "Hero Ref"]);
  });
});
