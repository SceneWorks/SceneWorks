import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AssetPickerField, ImageEditSourcePickerField, VideoSourcePickerField } from "./AssetPicker.jsx";

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

  it("keeps a selection made while a source-image upload is pending", async () => {
    let resolveFirst;
    let rejectSecond;
    const importAsset = vi
      .fn()
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }))
      .mockImplementationOnce(() => new Promise((_, reject) => { rejectSecond = reject; }));
    const onChange = vi.fn();
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
          values={[]}
        />,
      );
    });

    await click(container.querySelector('button[aria-haspopup="dialog"]'));
    await click(tab("File Upload"));
    const input = document.body.querySelector('input[type="file"]');
    const files = [
      new File(["upload"], "upload-1.png", { type: "image/png" }),
      new File(["upload"], "upload-2.png", { type: "image/png" }),
    ];
    Object.defineProperty(input, "files", { configurable: true, value: files });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));

    // Switch back and select a pre-existing card before the deferred importer settles.
    await click(tab("Assets"));
    await click(document.body.querySelector('.asset-picker-grid [role="option"]'));
    await act(async () => {
      // Settle out of order: a partial failure must not overwrite the concurrent pick
      // or masquerade as a complete success.
      rejectSecond(new Error("upload-2 failed"));
      resolveFirst({ id: "uploaded-late", type: "image", projectId: "p1" });
    });

    expect(onChange).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain("Imported 1 image; 1 failed.");
    await click([...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use Selection"));
    expect(onChange).toHaveBeenCalledWith(["plain-image", "uploaded-late"]);
  });
});

// sc-17137 review, item 2. `AssetPickerModal` grew an upload/dropzone import path so a picker
// scoped to a media kind the project has none of is not a dead end — the case that forced it is
// VideoStudio's REFERENCE AUDIO field, which is a plain `AssetPickerField` (categories hidden,
// `mediaKind="audio"`, `importAsset` wired) rather than one of the `MediaSourcePickerModal`
// pickers the tests above drive. That modal is a DIFFERENT component with a different upload
// path, so its coverage says nothing about this one. These drive the real field and mock only the
// transport boundary (`importAsset`), the same seam the video/image import tests mock.
//
// 🔴 sc-18650 pre-merge review. These originally stubbed `importAsset` with an invented three-key
// row (`{ id, type, projectId }`) — a shape no server ever sends — while the server it stood for
// REFUSED audio outright: `ProjectStore::import_asset` accepted `image/` and `video/` only, so the
// dropzone's real success rate was zero and `handleUpload`'s `Promise.allSettled` turned the 400
// into "Could not import the selected audio file." That gate is now open, and these stubs answer
// with the row the store actually writes (`importedAudioAsset` below) so a picker that started
// depending on a field the response does not carry fails here instead of in the app.
describe("AssetPickerField audio import (VideoStudio reference audio)", () => {
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

  // What VideoStudio hands the field: audio-only assets, categories hidden, multi-select.
  const audioAssets = [{ id: "voice-1", type: "audio", projectId: "p1", displayName: "Voice Take 1" }];

  // The row `ProjectStore::import_asset` writes for an audio upload, field for field:
  // `type: "audio"` (`media_type_for_mime`), `origin: "upload"`, and a `file` block whose
  // `mimeType` is ALWAYS `audio/wav` because `normalize_audio_upload` transcodes every accepted
  // container to PCM-16 RIFF/WAVE before storing it — the one encoding `read_wav_pcm16` decodes.
  //
  // `mimeType` being fixed is the load-bearing part, not decoration: it means the picker may not
  // key its acceptance on the mime the BROWSER reported for the chosen file, because the stored
  // asset's mime is frequently a different one. The `.flac` case below is exactly that.
  const importedAudioAsset = (id) => ({
    schemaVersion: 1,
    id,
    projectId: "p1",
    generationSetId: null,
    type: "audio",
    displayName: "take.wav",
    createdAt: "2026-08-20T00:00:00Z",
    origin: "upload",
    file: {
      path: `assets/uploads/take-${id}.wav`,
      mimeType: "audio/wav",
      width: null,
      height: null,
      duration: 1.5,
      fps: null,
      sampleRate: 24000,
      channels: 1,
    },
    status: { favorite: false, rating: 0, rejected: false, trashed: false },
  });

  async function openPicker(props) {
    await act(async () => {
      root.render(
        <AssetPickerField
          assets={audioAssets}
          buttonLabel="Select audio"
          label="Reference audio"
          mediaKind="audio"
          multiple
          showCategories={false}
          values={[]}
          {...props}
        />,
      );
    });
    await click(container.querySelector('button[aria-haspopup="dialog"]'));
    return document.body.querySelector('input[type="file"]');
  }

  async function chooseFiles(input, files) {
    Object.defineProperty(input, "files", { configurable: true, value: files });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));
  }

  it("imports a browsed audio file and confirms it alongside the existing selection", async () => {
    const onChange = vi.fn();
    const importAsset = vi.fn(async () => importedAudioAsset("uploaded-audio"));
    const input = await openPicker({ importAsset, onChange, values: ["voice-1"] });

    // Scoped to the picker's kind, and multi because the field is.
    expect(input.accept).toBe("audio/*");
    expect(input.multiple).toBe(true);
    expect(document.body.textContent).toContain("Drop audio files here");

    const file = new File(["audio"], "take.wav", { type: "audio/wav" });
    await chooseFiles(input, [file]);

    // `select: false` — a field-scoped picker must not hijack the app-wide Library selection.
    expect(importAsset).toHaveBeenCalledWith(file, { select: false, throwOnError: true });
    expect(onChange).toHaveBeenCalledWith(["voice-1", "uploaded-audio"]);
    expect(document.body.querySelector('[role="dialog"]')).toBeFalsy();
  });

  it("keeps a selection made while an asset-picker upload is pending", async () => {
    let resolveImport;
    const importAsset = vi.fn(() => new Promise((resolve) => { resolveImport = resolve; }));
    const onChange = vi.fn();
    const input = await openPicker({ importAsset, onChange, values: [] });
    const file = new File(["audio"], "take.wav", { type: "audio/wav" });

    Object.defineProperty(input, "files", { configurable: true, value: [file] });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));
    await click(document.body.querySelector('.asset-picker-grid [role="option"]'));
    await act(async () => resolveImport(importedAudioAsset("uploaded-late")));

    expect(onChange).toHaveBeenCalledWith(["voice-1", "uploaded-late"]);
  });

  it("imports a DROPPED audio file through the same path", async () => {
    const onChange = vi.fn();
    const importAsset = vi.fn(async () => importedAudioAsset("dropped-audio"));
    await openPicker({ importAsset, onChange });

    const zone = document.body.querySelector(".dataset-add-dropzone");
    expect(zone, "the picker's dropzone").toBeTruthy();
    const file = new File(["audio"], "dropped.mp3", { type: "audio/mpeg" });
    await act(async () => {
      const event = new Event("drop", { bubbles: true, cancelable: true });
      event.dataTransfer = { files: [file] };
      zone.dispatchEvent(event);
    });

    expect(importAsset).toHaveBeenCalledWith(file, { select: false, throwOnError: true });
    expect(onChange).toHaveBeenCalledWith(["dropped-audio"]);
  });

  it("rejects a non-audio file before any import is attempted", async () => {
    const onChange = vi.fn();
    const importAsset = vi.fn();
    const input = await openPicker({ importAsset, onChange });

    await chooseFiles(input, [new File(["image"], "cover.png", { type: "image/png" })]);

    expect(importAsset).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain("Choose audio files only.");
  });

  it("accepts a backend-supported audio extension when the browser reports a generic MIME", async () => {
    const onChange = vi.fn();
    // The server's answer for a `.flac`: `audio/flac` clears `import_asset`'s `audio/` gate (an
    // `application/octet-stream` content type is discarded in favour of the filename guess), and
    // `normalize_audio_upload` then stores it as PCM-16 WAV — so the row comes back `audio/wav`,
    // NOT `audio/flac`.
    const importAsset = vi.fn(async () => importedAudioAsset("uploaded-flac"));
    const input = await openPicker({ importAsset, onChange });

    // Browsers routinely hand .flac/.m4a back as application/octet-stream, so the client's
    // pre-flight has to fall back to the extension or it would refuse a file the server accepts.
    const file = new File(["audio"], "master.flac", { type: "application/octet-stream" });
    await chooseFiles(input, [file]);

    expect(importAsset).toHaveBeenCalledWith(file, { select: false, throwOnError: true });
    // ...and the post-flight has to accept the NORMALIZED row rather than looking for the mime it
    // sent. A picker that compared the response's `file.mimeType` to the chosen file's would drop
    // this import on the floor with "Could not import the selected audio file."
    expect(onChange).toHaveBeenCalledWith(["uploaded-flac"]);
  });

  it("does not select an import that came back as the wrong kind", async () => {
    const onChange = vi.fn();
    // The importer answers for whatever the project made of the bytes; a row that is not audio
    // cannot drive an audio-conditioned render, so it must not silently become the selection.
    // `media_type_for_mime` is what decides that `type`, and it answers from the mime the store
    // resolved — not from what the caller believed they were uploading.
    const importAsset = vi.fn(async () => ({
      ...importedAudioAsset("not-audio"),
      type: "image",
      file: { path: "assets/uploads/take-not-audio.png", mimeType: "image/png" },
    }));
    const input = await openPicker({ importAsset, onChange });

    await chooseFiles(input, [new File(["audio"], "take.wav", { type: "audio/wav" })]);

    expect(importAsset).toHaveBeenCalledTimes(1);
    expect(onChange).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain("Could not import the selected audio file.");
  });

  it("surfaces a failed import instead of confirming an empty selection", async () => {
    const onChange = vi.fn();
    const importAsset = vi.fn().mockRejectedValue(new Error("disk full"));
    const input = await openPicker({ importAsset, onChange });

    await chooseFiles(input, [new File(["audio"], "take.wav", { type: "audio/wav" })]);

    expect(onChange).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain("Could not import the selected audio file.");
    expect(document.body.querySelector('[role="dialog"]')).toBeTruthy();
  });

  it("renders no dropzone without an importer, so every other caller is unchanged", async () => {
    // `canImport` needs BOTH `importAsset` and `mediaKind`; the character/preview pickers pass
    // neither, and must stay byte-identical to before the affordance existed.
    await openPicker({ onChange: vi.fn() });
    expect(document.body.querySelector('input[type="file"]')).toBeFalsy();
    expect(document.body.querySelector(".dataset-add-dropzone")).toBeFalsy();
  });
});
