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
