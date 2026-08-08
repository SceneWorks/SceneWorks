import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ImageEditSourcePickerField, VideoSourcePickerField } from "./AssetPicker.jsx";

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

describe("VideoSourcePickerField video-only sources", () => {
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

  const click = async (el) =>
    act(async () => el.dispatchEvent(new MouseEvent("click", { bubbles: true })));
  const gridTitles = () =>
    [...document.body.querySelectorAll('.asset-picker-grid [role="option"] strong')].map((el) =>
      el.textContent.trim(),
    );
  const tab = (label) =>
    [...document.body.querySelectorAll('[role="tab"]')].find((button) =>
      button.textContent.startsWith(label),
    );

  const assets = [
    { id: "plain-video", type: "video", projectId: "p1", displayName: "Library Video" },
    {
      id: "hero-video",
      type: "video",
      projectId: "p1",
      displayName: "Hero Video",
      recipe: { normalizedSettings: { characterId: "hero" } },
    },
    { id: "plain-image", type: "image", projectId: "p1", displayName: "Library Image" },
    { id: "other-video", type: "video", projectId: "p2", displayName: "Other Project Video" },
  ];
  const characters = [{ id: "hero", name: "Hero", references: [], approvedReferences: [] }];

  it("shows project videos and selected-character videos in separate tabs", async () => {
    await act(async () => {
      root.render(
        <VideoSourcePickerField
          assets={assets}
          characters={characters}
          label="Source clip"
          onChange={() => {}}
          projectId="p1"
        />,
      );
    });

    await click(container.querySelector('button[aria-haspopup="dialog"]'));
    expect(gridTitles()).toEqual(["Library Video"]);
    expect(tab("Assets").querySelector("span").textContent).toBe("1");
    expect(tab("Character").querySelector("span").textContent).toBe("1");

    await click(tab("Character"));
    expect(gridTitles()).toEqual(["Hero Video"]);
    expect(document.body.textContent).not.toContain("Library Image");
    expect(document.body.textContent).not.toContain("Other Project Video");
  });

  it("uploads a video and selects the imported asset immediately", async () => {
    const onChange = vi.fn();
    const importAsset = vi.fn(async () => ({ id: "uploaded-video", type: "video", projectId: "p1" }));
    await act(async () => {
      root.render(
        <VideoSourcePickerField
          assets={assets}
          characters={characters}
          importAsset={importAsset}
          label="Source clip"
          onChange={onChange}
          projectId="p1"
        />,
      );
    });

    await click(container.querySelector('button[aria-haspopup="dialog"]'));
    await click(tab("File Upload"));
    const input = document.body.querySelector('input[type="file"]');
    expect(input.accept).toBe("video/*");
    const file = new File(["video"], "source.mp4", { type: "video/mp4" });
    Object.defineProperty(input, "files", { configurable: true, value: [file] });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));

    expect(importAsset).toHaveBeenCalledWith(file, { select: false, throwOnError: true });
    expect(onChange).toHaveBeenCalledWith("uploaded-video");
    expect(document.body.querySelector('[role="dialog"]')).toBeFalsy();
  });

  it("rejects a non-video file before import", async () => {
    const importAsset = vi.fn();
    await act(async () => {
      root.render(
        <VideoSourcePickerField
          assets={assets}
          characters={characters}
          importAsset={importAsset}
          label="Source clip"
          onChange={() => {}}
          projectId="p1"
        />,
      );
    });

    await click(container.querySelector('button[aria-haspopup="dialog"]'));
    await click(tab("File Upload"));
    const input = document.body.querySelector('input[type="file"]');
    const file = new File(["image"], "not-a-video.png", { type: "image/png" });
    Object.defineProperty(input, "files", { configurable: true, value: [file] });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));

    expect(importAsset).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain("Choose video files only.");
  });

  it("accepts a backend-supported video extension when the browser reports a generic MIME", async () => {
    const onChange = vi.fn();
    const importAsset = vi.fn(async () => ({ id: "uploaded-wmv", type: "video", projectId: "p1" }));
    await act(async () => {
      root.render(
        <VideoSourcePickerField
          assets={assets}
          importAsset={importAsset}
          onChange={onChange}
          projectId="p1"
        />,
      );
    });

    await click(container.querySelector('button[aria-haspopup="dialog"]'));
    await click(tab("File Upload"));
    const input = document.body.querySelector('input[type="file"]');
    const file = new File(["video"], "legacy.wmv", { type: "application/octet-stream" });
    Object.defineProperty(input, "files", { configurable: true, value: [file] });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));

    expect(importAsset).toHaveBeenCalledWith(file, { select: false, throwOnError: true });
    expect(onChange).toHaveBeenCalledWith("uploaded-wmv");
  });

  it("imports multiple images and returns their ids with the existing selection", async () => {
    const onChange = vi.fn();
    const importAsset = vi
      .fn()
      .mockResolvedValueOnce({ id: "uploaded-image-1", type: "image", projectId: "p1" })
      .mockResolvedValueOnce({ id: "uploaded-image-2", type: "image", projectId: "p1" });
    await act(async () => {
      root.render(
        <ImageEditSourcePickerField
          assets={assets}
          characters={characters}
          importAsset={importAsset}
          label="Reference images"
          multiple
          onChange={onChange}
          projectId="p1"
          values={["plain-image"]}
        />,
      );
    });

    await click(container.querySelector('button[aria-haspopup="dialog"]'));
    await click(tab("File Upload"));
    const input = document.body.querySelector('input[type="file"]');
    expect(input.accept).toBe("image/*");
    expect(input.multiple).toBe(true);
    const files = [
      new File(["one"], "one.png", { type: "image/png" }),
      new File(["two"], "two.jpg", { type: "image/jpeg" }),
    ];
    Object.defineProperty(input, "files", { configurable: true, value: files });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));

    expect(importAsset).toHaveBeenCalledTimes(2);
    expect(onChange).toHaveBeenCalledWith(["plain-image", "uploaded-image-1", "uploaded-image-2"]);
  });

  it("keeps successful multi-file imports selected when a sibling fails", async () => {
    const onChange = vi.fn();
    const importAsset = vi
      .fn()
      .mockResolvedValueOnce({ id: "uploaded-image-1", type: "image", projectId: "p1" })
      .mockRejectedValueOnce(new Error("second upload failed"));
    await act(async () => {
      root.render(
        <ImageEditSourcePickerField
          assets={assets}
          importAsset={importAsset}
          label="Reference images"
          multiple
          onChange={onChange}
          projectId="p1"
          values={["plain-image"]}
        />,
      );
    });

    await click(container.querySelector('button[aria-haspopup="dialog"]'));
    await click(tab("File Upload"));
    const input = document.body.querySelector('input[type="file"]');
    const files = [
      new File(["one"], "one.png", { type: "image/png" }),
      new File(["two"], "two.jpg", { type: "image/jpeg" }),
    ];
    Object.defineProperty(input, "files", { configurable: true, value: files });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));

    expect(onChange).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain("Imported 1 image; 1 failed.");
    expect(document.body.textContent).toContain("2 selected");
    await click([...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use Selection"));
    expect(onChange).toHaveBeenCalledWith(["plain-image", "uploaded-image-1"]);
  });
});
