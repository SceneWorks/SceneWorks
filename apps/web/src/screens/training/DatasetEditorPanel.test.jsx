import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DatasetEditorPanel } from "./DatasetEditorPanel.jsx";

// Build a full-but-inert prop bag; individual tests override only what they exercise.
// Empty `memberAssets` and a null Doctor report keep the caption grid and Dataset Doctor
// band in their empty states, so the panel renders without heavy fixtures.
function makeSessions(overrides = {}) {
  const onDeleteDataset = overrides.onDeleteDataset ?? vi.fn();
  const activeDataset =
    overrides.activeDataset === undefined
      ? { id: "ds-1", name: "Kelsie", version: 1, items: [] }
      : overrides.activeDataset;
  return {
    onDeleteDataset,
    props: {
      datasetSession: {
        loadingDatasets: false,
        onRefreshDatasets: vi.fn(),
        busyDatasetId: "",
        datasetThumbAsset: () => null,
        datasets: [{ id: "ds-1", name: "Kelsie", version: 1 }],
        startNewDataset: vi.fn(),
        openDataset: vi.fn(),
        activeDataset,
        selectedDatasetId: activeDataset?.id ?? "",
        datasetsError: "",
        datasetError: "",
        datasetMessage: "",
        draftName: activeDataset?.name ?? "",
        setDraftName: vi.fn(),
        dirty: false,
        discardDraft: vi.fn(),
        setAddDialogOpen: vi.fn(),
        renamePrefix: "item",
        setRenamePrefix: vi.fn(),
        renaming: false,
        memberAssets: [],
        applyOrderedNames: vi.fn(),
        setCaptionDialog: vi.fn(),
        health: { itemCount: 0, missingCaptions: 0, duplicateFilenames: 0, valid: false },
        canSave: false,
        saveValidity: { surfaced: [] },
        saveDataset: vi.fn(),
        savingDataset: false,
        unavailableAssetIds: [],
        removeUnavailableAsset: vi.fn(),
        onImportParquet: vi.fn(),
        onDeleteDataset,
        deletingDataset: overrides.deletingDataset ?? false,
      },
      captionSession: {
        captionDraftById: {},
        onPreview: vi.fn(),
        updateCaption: vi.fn(),
        captioning: false,
        addDialogOpen: false,
        selectedAssetIds: [],
        addAssets: vi.fn(),
        handleImport: vi.fn(),
        captionDialog: null,
        updateCaptionSetting: vi.fn(),
        runCaptionJob: vi.fn(),
        toggleCaptionExtraOption: vi.fn(),
        displayedCaptionPrompt: "",
        captionSettings: {},
        captionModelMissing: false,
        onDownloadCaptionModel: vi.fn(),
        captionModelSizeLabel: "",
        captionModelName: "JoyCaption",
      },
      doctorSession: {
        datasetDoctor: { report: null, loading: false },
        readinessByKey: new Map(),
        onToggleItemAck: vi.fn(),
      },
      config: {
        imageAssets: [],
        characters: [],
        associatedCharacterId: "",
        setActiveView: vi.fn(),
        importingAssets: false,
        gpuOptions: [],
      },
    },
  };
}

describe("DatasetEditorPanel delete affordance", () => {
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
  });

  function deleteButton() {
    return container.querySelector('button[aria-label="Delete dataset Kelsie"]');
  }

  it("renders a Delete button for a saved dataset and calls onDeleteDataset when clicked", async () => {
    const { onDeleteDataset, props } = makeSessions();
    await act(async () => {
      root.render(<DatasetEditorPanel {...props} />);
    });

    const button = deleteButton();
    expect(button).not.toBeNull();
    expect(button.textContent).toContain("Delete");
    expect(button.disabled).toBe(false);

    await act(async () => {
      button.dispatchEvent(new window.Event("click", { bubbles: true }));
    });
    expect(onDeleteDataset).toHaveBeenCalledTimes(1);
  });

  it("shows a busy, disabled Delete button while a delete is in flight", async () => {
    const { props } = makeSessions({ deletingDataset: true });
    await act(async () => {
      root.render(<DatasetEditorPanel {...props} />);
    });

    const button = deleteButton();
    expect(button).not.toBeNull();
    expect(button.disabled).toBe(true);
    expect(button.textContent).toContain("Deleting");
  });

  it("hides the Delete button when no saved dataset is open", async () => {
    const { props } = makeSessions({ activeDataset: null });
    await act(async () => {
      root.render(<DatasetEditorPanel {...props} />);
    });

    expect(deleteButton()).toBeNull();
    // Nothing dataset-scoped to delete, but the panel still renders its draft shell.
    expect(container.querySelector(".dataset-identity-actions")).not.toBeNull();
  });

  it("keeps Parquet-import failures visible and handled at the parent boundary", async () => {
    const onImportParquet = vi.fn(() => Promise.reject({ message: { detail: "not renderable" } }));
    const { props } = makeSessions();
    props.datasetSession.onImportParquet = onImportParquet;
    const unhandled = [];
    const onUnhandled = (event) => {
      unhandled.push(event.reason);
      event.preventDefault();
    };
    window.addEventListener("unhandledrejection", onUnhandled);

    try {
      await act(async () => root.render(<DatasetEditorPanel {...props} />));
      await act(async () => {
        [...container.querySelectorAll("button")].find((button) => button.textContent === "Import Parquet").click();
      });
      const input = [...document.body.querySelectorAll("label")]
        .find((label) => label.textContent.includes("Parquet file or folder"))
        .querySelector("input");
      await act(async () => {
        const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), "value")?.set;
        setter.call(input, "D:\\broken.parquet");
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      });
      await act(async () => {
        [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Start import").click();
        await Promise.resolve();
      });

      expect(onImportParquet).toHaveBeenCalledTimes(1);
      expect(container.textContent).toContain("Could not start the Parquet import.");
      expect(unhandled).toEqual([]);
    } finally {
      window.removeEventListener("unhandledrejection", onUnhandled);
    }
  });
});

// sc-2126: subject masks — coverage, the generate action, per-image overlay, the empty-mask flag, and
// the replacement upload (with its field error surfaced on the owning card).
describe("DatasetEditorPanel subject masks", () => {
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
  });

  const items = [
    { id: "item_a", assetId: "asset-a", displayName: "a.png", path: "images/item_a.png" },
    { id: "item_b", assetId: "asset-b", displayName: "b.png", path: "images/item_b.png" },
  ];
  const members = items.map((item) => ({
    id: item.assetId,
    projectId: "proj-1",
    type: "image",
    displayName: item.displayName,
    file: { path: `training/datasets/ds-1/${item.path}`, mimeType: "image/png" },
  }));
  const report = {
    datasetId: "ds-1",
    total: 2,
    masked: 2,
    empty: 1,
    uploaded: 0,
    items: [
      { itemId: "item_a", hasMask: true, empty: false, source: "auto", maskPath: "training/datasets/ds-1/masks/aaa.png", updatedAt: "t1" },
      { itemId: "item_b", hasMask: true, empty: true, source: "auto", maskPath: "training/datasets/ds-1/masks/bbb.png", updatedAt: "t1" },
    ],
  };

  function maskProps(maskSession) {
    const { props } = makeSessions({ activeDataset: { id: "ds-1", name: "Kelsie", version: 2, items } });
    props.datasetSession.memberAssets = members;
    props.datasetSession.health = { itemCount: 2, missingCaptions: 2, duplicateFilenames: 0, valid: true };
    props.maskSession = { projectId: "proj-1", report, onGenerateMasks: vi.fn(), onUploadMask: vi.fn(), ...maskSession };
    return props;
  }

  const button = (text) => [...container.querySelectorAll("button")].find((node) => node.textContent.includes(text));
  const overlays = () => [...container.querySelectorAll("img.training-subject-mask-overlay")];

  it("reports coverage, flags the empty mask, and runs Generate subject masks", async () => {
    const props = maskProps();
    await act(async () => root.render(<DatasetEditorPanel {...props} />));

    expect(container.querySelector('[aria-label="Subject mask coverage"]').textContent).toBe("2/2 masked");
    expect(container.textContent).toContain("1 with no subject");
    const cards = [...container.querySelectorAll(".training-caption-card")];
    expect(cards[0].textContent).not.toContain("No subject in mask");
    expect(cards[1].textContent).toContain("No subject in mask");

    await act(async () => button("Generate subject masks").click());
    expect(props.maskSession.onGenerateMasks).toHaveBeenCalledTimes(1);
  });

  it("toggles a mask overlay on every masked image", async () => {
    const props = maskProps();
    await act(async () => root.render(<DatasetEditorPanel {...props} />));
    expect(overlays()).toHaveLength(0);

    await act(async () => button("Show masks").click());
    const shown = overlays();
    expect(shown).toHaveLength(2);
    expect(shown[0].getAttribute("src")).toContain("/api/v1/projects/proj-1/files/training/datasets/ds-1/masks/aaa.png?v=t1");
    expect(shown[1].getAttribute("alt")).toBe("Subject mask for b.png");

    await act(async () => button("Hide masks").click());
    expect(overlays()).toHaveLength(0);
  });

  it("ignores a report that belongs to another dataset", async () => {
    const props = maskProps({ report: { ...report, datasetId: "other" } });
    await act(async () => root.render(<DatasetEditorPanel {...props} />));
    expect(container.querySelector('[aria-label="Subject mask coverage"]')).toBeNull();
    expect(container.textContent).not.toContain("No subject in mask");
  });

  it("uploads a replacement mask and shows the server's field error on that card", async () => {
    const onUploadMask = vi
      .fn()
      .mockResolvedValueOnce({})
      .mockRejectedValueOnce(new Error("Subject mask is 8x8 but the image is 16x8"));
    const props = maskProps({ onUploadMask });
    await act(async () => root.render(<DatasetEditorPanel {...props} />));

    const input = container.querySelector('input[aria-label="Upload subject mask for b.png"]');
    expect(input.getAttribute("accept")).toBe("image/png,image/jpeg,image/webp");
    const file = new File([new Uint8Array([1, 2, 3])], "mask.png", { type: "image/png" });
    const choose = async () => {
      Object.defineProperty(input, "files", { configurable: true, value: [file] });
      await act(async () => {
        input.dispatchEvent(new Event("change", { bubbles: true }));
        await Promise.resolve();
      });
    };

    await choose();
    expect(onUploadMask).toHaveBeenCalledWith(members[1], file);
    expect(overlays()).toHaveLength(2); // a successful upload reveals the overlay
    expect(container.querySelector('[role="alert"]')).toBeNull();

    await choose();
    const cards = [...container.querySelectorAll(".training-caption-card")];
    expect(cards[1].querySelector('[role="alert"]').textContent).toContain("8x8 but the image is 16x8");
    expect(cards[0].querySelector('[role="alert"]')).toBeNull();
  });
});
