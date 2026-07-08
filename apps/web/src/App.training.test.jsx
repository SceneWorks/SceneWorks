import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { withTrainingStudioContext, withTrainingDataSetsLibraryContext, FakeEventSource, response, settle, field, buttonInside, zImageTrainingTarget, zImageTrainingPresets, changeField } from "./main.testSupport.jsx";

describe("SceneWorks app shell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("gates the studios behind workspace creation when no projects exist", async () => {
    const requests = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      requests.push({ path, method: options.method ?? "GET" });
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects") && options.method === "POST") {
        return Promise.resolve(response({ id: "project-new", name: "My First Project" }));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    // With zero workspaces the studio area is replaced by the create gate.
    expect(container.textContent).toContain("Create your first workspace");

    await changeField(document.body.querySelector('[aria-label="Workspace name"]'), "My First Project");
    await act(async () => {
      buttonInside(container, "Create workspace").click();
    });
    await settle();

    // Creating the first workspace clears the gate and lands in a studio.
    expect(requests.some((request) => request.path.endsWith("/projects") && request.method === "POST")).toBe(true);
    expect(container.textContent).not.toContain("Create your first workspace");
  });

  it("opens the Train navigation item without exposing a queue action", async () => {
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-a", name: "Project A" }]));
      }
      if (path.includes("/training/datasets")) {
        return Promise.resolve(response([{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 2 }]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Train").click();
    });
    await settle();

    expect(container.textContent).toContain("Training Studio");
    expect(container.textContent).toContain("Configure Job");
    expect(container.textContent).toContain("Data Sets");
    expect(container.textContent).not.toContain("Rename & Caption");
    expect([...document.body.querySelectorAll("button")].some((button) => /queue training/i.test(button.textContent))).toBe(false);
  });

  it("keeps Training Studio focused on selecting existing datasets", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 2 }],
          onRefreshDatasets: () => {},
        }),
      );
    });

    expect(document.body.querySelector("#training-tab-configure").getAttribute("aria-selected")).toBe("true");
    expect(container.textContent).toContain("A dry run validates the Rust-resolved training plan");
    expect(document.body.querySelector("#training-tab-dataset")).toBeNull();
    expect(document.body.querySelector("#training-tab-rename-caption")).toBeNull();
    expect(container.textContent).not.toContain("Import images & captions");
  });

  it("creates a training dataset from selected image assets", async () => {
    const createDataset = vi.fn(async (payload) => ({
      id: "dataset-new",
      name: payload.name,
      version: 1,
      items: payload.items.map((item) => ({ ...item, caption: { text: "", triggerWords: [] } })),
    }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          createDataset,
          datasets: [],
        }),
      );
    });

    await changeField(field(container, "Dataset name"), "Mira Set");
    // Add the image via the Asset Library tab of the add dialog.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add images").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Asset Library").click();
    });
    await act(async () => {
      document.body.querySelector(".dataset-add-card").click();
    });
    await act(async () => {
      document.body.querySelector(".dataset-add-footer button.primary-action").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Create dataset").click();
    });

    expect(createDataset).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Mira Set",
        modality: "image",
        items: [expect.objectContaining({ assetId: "asset-a", displayName: "Mira.png" })],
      }),
    );
    expect(container.textContent).toContain("Dataset created");
  });

  it("imports caption sidecars alongside images and bakes them into the saved dataset", async () => {
    const createDataset = vi.fn(async (payload) => ({
      id: "dataset-new",
      name: payload.name,
      version: 1,
      items: payload.items,
    }));
    const uploadDatasetItem = vi.fn(async (file) => ({
      id: "dataset-upload-mira",
      datasetOnly: true,
      displayName: file.name,
      file: { path: "training/uploads/mira.png", mimeType: "image/png" },
    }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-mira", type: "image", displayName: "mira.png", file: { path: "assets/images/mira.png", mimeType: "image/png" } }],
          createDataset,
          datasets: [],
          uploadDatasetItem,
        }),
      );
    });

    const imageFile = new File([new Uint8Array([1, 2, 3])], "mira.png", { type: "image/png" });
    const captionFile = new File(["a portrait of mira"], "mira.txt", { type: "text/plain" });
    // Import via the File tab of the add dialog (default tab).
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add images").click();
    });
    const fileInput = document.body.querySelector(".dataset-add-dropzone input[type=file]");
    await act(async () => {
      Object.defineProperty(fileInput, "files", { configurable: true, value: [imageFile, captionFile] });
      fileInput.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    await settle();

    // Only the image is uploaded to dataset-owned staging; the .txt is parsed locally.
    expect(uploadDatasetItem).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain("Imported 1 image with 1 caption");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Close").click();
    });
    await changeField(field(container, "Dataset name"), "Mira Set");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Create dataset").click();
    });

    expect(createDataset).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Mira Set",
        items: [
          expect.objectContaining({
            path: "training/uploads/mira.png",
            caption: expect.objectContaining({ text: "a portrait of mira", source: "imported" }),
          }),
        ],
      }),
    );
  });

  it("does not mutate the API-returned asset record when flagging dataset-only (sc-8939)", async () => {
    // Capture the exact object the upload transport returns so we can assert the import
    // path treats it as immutable (the datasetOnly flag must land on a copy, not here).
    const returnedAsset = {
      id: "dataset-upload-mira",
      displayName: "mira.png",
      file: { path: "training/uploads/mira.png", mimeType: "image/png" },
    };
    const uploadDatasetItem = vi.fn(async () => returnedAsset);

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [],
          datasets: [],
          createDataset: vi.fn(async (payload) => ({ id: "dataset-new", name: payload.name, version: 1, items: payload.items })),
          uploadDatasetItem,
        }),
      );
    });

    const imageFile = new File([new Uint8Array([1, 2, 3])], "mira.png", { type: "image/png" });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add images").click();
    });
    const fileInput = document.body.querySelector(".dataset-add-dropzone input[type=file]");
    await act(async () => {
      Object.defineProperty(fileInput, "files", { configurable: true, value: [imageFile] });
      fileInput.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    await settle();

    expect(uploadDatasetItem).toHaveBeenCalledTimes(1);
    // The returned record is unchanged — no datasetOnly leaked onto the shared instance.
    expect(returnedAsset).not.toHaveProperty("datasetOnly");
    expect(Object.keys(returnedAsset)).toEqual(["id", "displayName", "file"]);
  });

  it("opens and saves an existing training dataset membership", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [{ assetId: "asset-a", displayName: "Mira.png", caption: { text: "mira portrait", triggerWords: [] } }],
    }));
    const updateDataset = vi.fn(async (datasetId, payload) => ({
      id: datasetId,
      name: payload.name,
      version: 4,
      items: payload.items.map((item) => ({ ...item, caption: item.caption ?? { text: "", triggerWords: [] } })),
    }));
    const assets = [
      { id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } },
      { id: "asset-b", type: "image", displayName: "Mira close.png", file: { path: "assets/images/Mira-close.png", mimeType: "image/png" } },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset,
          updateDataset,
        }),
      );
    });

    await act(async () => {
      document.body.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Portrait Set")).click();
    });
    await settle();
    expect(loadDataset).toHaveBeenCalledWith("dataset-a");
    // The editor body shows only the dataset's own member (asset-a), not all assets.
    expect(document.body.querySelectorAll(".training-caption-card")).toHaveLength(1);
    expect(document.body.querySelector(".training-caption-grid").textContent).toContain("Mira.png");

    // Add the second asset through the Asset Library tab (current members excluded).
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add images").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Asset Library").click();
    });
    await act(async () => {
      document.body.querySelector(".dataset-add-card").click();
    });
    await act(async () => {
      document.body.querySelector(".dataset-add-footer button.primary-action").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Save dataset").click();
    });

    expect(updateDataset).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({
        name: "Portrait Set",
        items: [
          expect.objectContaining({ assetId: "asset-a" }),
          expect.objectContaining({ assetId: "asset-b" }),
        ],
      }),
    );
    expect(container.textContent).toContain("Dataset changes saved");
  });

  it("scopes the add dialog: Library excludes Character Studio outputs, Character tab pulls them in (sc-2026)", async () => {
    const createDataset = vi.fn(async (payload) => ({ id: "dataset-new", name: payload.name, version: 1, items: payload.items }));
    const assets = [
      {
        id: "asset-lib",
        type: "image",
        displayName: "Studio render.png",
        origin: "image_studio",
        file: { path: "assets/images/studio.png", mimeType: "image/png" },
      },
      {
        id: "asset-char",
        type: "image",
        displayName: "Kelsie hero.png",
        origin: "character_studio",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        file: { path: "assets/images/kelsie.png", mimeType: "image/png" },
      },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets,
          characters: [{ id: "char-1", name: "Kelsie" }],
          createDataset,
          datasets: [],
        }),
      );
    });

    await changeField(field(container, "Dataset name"), "Kelsie Set");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add images").click();
    });

    // Asset Library tab is scoped: the Character Studio output is hidden.
    await act(async () => {
      [...document.body.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Asset Library").click();
    });
    const libraryCards = [...document.body.querySelectorAll(".dataset-add-card")];
    expect(libraryCards).toHaveLength(1);
    expect(libraryCards[0].textContent).toContain("Studio render.png");

    // Character tab intentionally surfaces the character's image (its character_studio output).
    await act(async () => {
      [...document.body.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Character").click();
    });
    const characterCards = [...document.body.querySelectorAll(".dataset-add-card")];
    expect(characterCards).toHaveLength(1);
    expect(characterCards[0].textContent).toContain("Kelsie hero.png");
    await act(async () => {
      characterCards[0].click();
    });
    await act(async () => {
      document.body.querySelector(".dataset-add-footer button.primary-action").click();
    });

    // The editor body is the member grid only — no all-asset picker remains.
    expect(document.body.querySelector(".training-asset-picker")).toBeNull();
    expect(document.body.querySelector(".training-caption-grid").textContent).toContain("Kelsie hero.png");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Create dataset").click();
    });
    // Importing from the Character tab associates the dataset with that
    // character (sc-2022).
    expect(createDataset).toHaveBeenCalledWith(
      expect.objectContaining({
        characterId: "char-1",
        items: [expect.objectContaining({ assetId: "asset-char" })],
      }),
    );
  });

  it("shows a cover thumbnail per dataset and a New dataset item in the selector (sc-2025)", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [],
          datasets: [
            { id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 3, coverPath: "training/datasets/dataset-a/images/item_0001.png" },
          ],
        }),
      );
    });

    // Pill shows the New-dataset draft placeholder before anything is opened.
    expect(document.body.querySelector(".compact-selector-pill").textContent).toContain("New dataset");

    await act(async () => {
      document.body.querySelector(".compact-selector-pill").click();
    });

    // A "New dataset" create item sits at the top of the dropdown.
    expect(document.body.querySelector(".compact-selector-create").textContent).toContain("New dataset");

    // Every dataset row renders its server-provided cover thumbnail.
    const datasetItem = [...document.body.querySelectorAll(".compact-selector-item")].find((button) =>
      button.textContent.includes("Portrait Set"),
    );
    const cover = datasetItem.querySelector("img");
    expect(cover).not.toBeNull();
    expect(cover.getAttribute("src")).toContain("training/datasets/dataset-a/images/item_0001.png");
  });

  it("lets users remove unavailable dataset assets before saving", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [
        { assetId: "asset-a", displayName: "Mira.png", caption: { text: "mira portrait", triggerWords: [] } },
        { assetId: "asset-missing", displayName: "Missing.png", caption: { text: "missing portrait", triggerWords: [] } },
      ],
    }));
    const updateDataset = vi.fn(async (datasetId, payload) => ({
      id: datasetId,
      name: payload.name,
      version: 4,
      items: payload.items,
    }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 2 }],
          loadDataset,
          updateDataset,
        }),
      );
    });

    await act(async () => {
      document.body.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Portrait Set")).click();
    });
    await settle();

    expect(container.textContent).toContain("Asset is no longer available");
    expect([...document.body.querySelectorAll("button")].find((button) => button.textContent === "Save dataset").disabled).toBe(true);

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Remove").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Save dataset").click();
    });

    expect(updateDataset).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({
        items: [expect.objectContaining({ assetId: "asset-a" })],
      }),
    );
  });

  it("does not save unchanged existing datasets", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [{ assetId: "asset-a", displayName: "Mira.png", caption: { text: "mira portrait", triggerWords: [] } }],
    }));
    const updateDataset = vi.fn();

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset,
          updateDataset,
        }),
      );
    });

    await act(async () => {
      document.body.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Portrait Set")).click();
    });
    await settle();

    const saveButton = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Save dataset");
    expect(saveButton.disabled).toBe(true);
    expect(updateDataset).not.toHaveBeenCalled();
  });

  // Open the lone "Portrait Set" dataset through the compact selector.
  async function openPortraitSet() {
    await act(async () => {
      document.body.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Portrait Set")).click();
    });
    await settle();
  }

  function singleItemDataset() {
    return {
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: [] },
        },
      ],
    };
  }

  it("edits a caption inline and saves it with the dataset (sc-2025)", async () => {
    const updateDataset = vi.fn(async (datasetId, payload) => ({ id: datasetId, name: payload.name, version: 4, items: payload.items }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset: vi.fn(async () => singleItemDataset()),
          updateDataset,
        }),
      );
    });
    await openPortraitSet();

    const caption = document.body.querySelector(".training-caption-card-text");
    expect(caption.value).toBe("mira portrait");
    await changeField(caption, "mira studio portrait");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Save dataset").click();
    });

    expect(updateDataset).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({
        items: [expect.objectContaining({ assetId: "asset-a", caption: expect.objectContaining({ text: "mira studio portrait", source: "manual" }) })],
      }),
    );
  });

  it("queues a caption job for all images via the caption dialog (sc-2025)", async () => {
    const updateDataset = vi.fn(async (datasetId, payload) => ({ id: datasetId, name: payload.name, version: 4, items: singleItemDataset().items }));
    const createCaptionJob = vi.fn(async () => ({ id: "job-caption-1", type: "training_caption" }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          createCaptionJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset: vi.fn(async () => singleItemDataset()),
          updateDataset,
        }),
      );
    });
    await openPortraitSet();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Caption all").click();
    });
    // The dialog prefills the character name from the dataset.
    expect(field(document.body, "Character name").value).toBe("Portrait Set");
    await act(async () => {
      [...document.body.querySelectorAll(".dataset-caption-footer button")].find((button) => button.textContent.startsWith("Caption")).click();
    });
    await settle();

    expect(createCaptionJob).toHaveBeenCalledWith("dataset-a", expect.objectContaining({ captioner: "joy_caption" }));
    expect(createCaptionJob.mock.calls[0][1].itemIds).toBeUndefined();
    expect(container.textContent).toContain("Caption job queued (job-caption-1)");
  });

  it("re-captions a single image with the itemIds filter (sc-2025)", async () => {
    const updateDataset = vi.fn(async (datasetId, payload) => ({ id: datasetId, name: payload.name, version: 4, items: singleItemDataset().items }));
    const createCaptionJob = vi.fn(async () => ({ id: "job-caption-2", type: "training_caption" }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          createCaptionJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset: vi.fn(async () => singleItemDataset()),
          updateDataset,
        }),
      );
    });
    await openPortraitSet();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Re-Caption").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".dataset-caption-footer button")].find((button) => button.textContent.startsWith("Re-caption")).click();
    });
    await settle();

    expect(createCaptionJob).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({ itemIds: ["item_0001"], recaption: true }),
    );
  });

  it("applies ordered names to the dataset from the toolbar (sc-2025)", async () => {
    const updateDataset = vi.fn(async (datasetId, payload) => ({ id: datasetId, name: payload.name, version: 4, items: singleItemDataset().items }));
    const batchRenameDataset = vi.fn(async (datasetId) => ({ ...singleItemDataset(), id: datasetId, version: 5 }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          batchRenameDataset,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset: vi.fn(async () => singleItemDataset()),
          updateDataset,
        }),
      );
    });
    await openPortraitSet();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent.includes("Apply ordered names")).click();
    });
    await settle();

    expect(batchRenameDataset).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({
        items: [expect.objectContaining({ itemId: "item_0001", newItemId: "portrait_set_0001", fileStem: "portrait_set_0001" })],
      }),
    );
  });

  it("defaults the training trigger phrase from the selected dataset name until edited", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "first portrait", source: "manual", triggerWords: ["oldOne"] },
        },
      ],
    }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });

    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    expect(field(container, "Trigger phrase").value).toBe("Portrait Set");

    await changeField(field(container, "Trigger phrase"), "manualTrigger");
    expect(document.body.querySelector("#training-tab-rename-caption")).toBeNull();
    expect(field(container, "Trigger phrase").value).toBe("manualTrigger");
  });

  it("shows active training progress with live sample previews", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          jobs: [
            {
              id: "job-train-1",
              type: "lora_train",
              status: "running",
              stage: "rendering",
              progress: 0.42,
              elapsedSeconds: 31,
              projectId: "project-a",
              requestedGpu: "0",
              payload: { outputName: "Portrait Set LoRA" },
              result: {
                latestTrainingSamples: [
                  { step: 250, prompt: "Portrait Set, studio portrait", relativePath: "loras/lora_1/samples/sample-1.png" },
                  { step: 250, prompt: "Portrait Set, full body", relativePath: "loras/lora_1/samples/sample-2.png" },
                  { step: 250, prompt: "Portrait Set, outdoor", relativePath: "loras/lora_1/samples/sample-3.png" },
                  { step: 250, prompt: "Portrait Set, close-up", relativePath: "loras/lora_1/samples/sample-4.png" },
                ],
              },
            },
          ],
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });

    expect(container.textContent).toContain("Training in progress");
    expect(container.textContent).toContain("Portrait Set LoRA");
    // The unified WorkerProgressCard renders the stage with title-case via
    // defaultChipLabel ("rendering" -> "Rendering"); same content, different style.
    expect(container.textContent).toContain("Rendering");
    expect([...document.body.querySelectorAll(".worker-progress-card__thumb-media")].map((image) => image.src)).toEqual([
      "http://localhost:8000/api/v1/projects/project-a/files/loras/lora_1/samples/sample-1.png",
      "http://localhost:8000/api/v1/projects/project-a/files/loras/lora_1/samples/sample-2.png",
      "http://localhost:8000/api/v1/projects/project-a/files/loras/lora_1/samples/sample-3.png",
      "http://localhost:8000/api/v1/projects/project-a/files/loras/lora_1/samples/sample-4.png",
    ]);
  });

  it("builds a training config snapshot from registry defaults", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_dryrun_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          createTrainingJob,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");

    expect(field(container, "Target").value).toBe("z_image_turbo_lora");
    expect(field(container, "Base model").value).toBe("z_image_turbo");
    expect(field(container, "Guidance scale").value).toBe("1");
    expect(field(container, "Rank").value).toBe("16");
    expect(field(container, "Precision").value).toBe("bf16");
    expect([...field(container, "Optimizer").options].map((option) => option.value)).toEqual(["adamw8bit", "adamw", "adam", "prodigyopt", "rose"]);
    await changeField(field(container, "Optimizer"), "prodigyopt");
    await changeField(field(container, "Guidance scale"), "1.2");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").click();
    });
    await settle();

    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        targetId: "z_image_turbo_lora",
        datasetId: "dataset-a",
        datasetVersion: 5,
        outputName: "Portrait Set LoRA",
        dryRun: true,
        config: expect.objectContaining({
          rank: 16,
          alpha: 16,
          learningRate: 0.0001,
          optimizer: "prodigyopt",
          triggerWord: "miraStyle",
          advanced: expect.objectContaining({
            mixedPrecision: "bf16",
            qualityPreset: "balanced",
            outputScope: "project",
            requestedGpu: "auto",
            sampleSteps: 8,
            sampleGuidanceScale: 1.2,
            samplePrompts: expect.arrayContaining([expect.stringContaining("miraStyle")]),
          }),
        }),
      }),
    );
    expect(container.textContent).toContain("Queued dry-run job");
  });

  it("applies training presets and includes selected preset metadata", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_dryrun_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          createTrainingJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingPresets: zImageTrainingPresets,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();

    expect(field(container, "Preset").value).toBe("z_image_turbo_lora.character.adamw8bit.balanced");
    expect(container.textContent).toContain("Character balanced");
    await changeField(field(container, "Optimizer"), "prodigyopt");
    expect(field(container, "Preset").value).toBe("z_image_turbo_lora.character.prodigyopt.balanced");
    expect(field(container, "Learning rate").value).toBe("1");
    expect(field(container, "Sample cadence").value).toBe("200");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").click();
    });
    await settle();

    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        presetId: "z_image_turbo_lora.character.prodigyopt.balanced",
        presetVersion: 1,
        config: expect.objectContaining({
          learningRate: 1,
          optimizer: "prodigyopt",
          steps: 1600,
          advanced: expect.objectContaining({ sampleEvery: 200 }),
        }),
      }),
    );
  });

  it("submits the selected de-distill adapter version for Z-Image-Turbo", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_dryrun_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          createTrainingJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingPresets: zImageTrainingPresets,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();

    // The selector appears for Z-Image-Turbo and normalizes the preset's legacy
    // "v2-default" value to "v2".
    const adapterSelect = field(container, "De-distill adapter");
    expect(adapterSelect).toBeTruthy();
    expect(adapterSelect.value).toBe("v2");

    await changeField(adapterSelect, "v1");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").click();
    });
    await settle();

    // The repo + chosen version must reach config.advanced — the worker only fuses
    // the de-distill adapter when trainingAdapterRepo is present.
    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({
          advanced: expect.objectContaining({
            trainingAdapterRepo: "ostris/zimage_turbo_training_adapter",
            trainingAdapterVersion: "v1",
          }),
        }),
      }),
    );
  });

  it("marks manual training preset edits as customizations", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          trainingPresets: zImageTrainingPresets,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Rank"), "24");

    expect(container.textContent).toContain("Customized: Rank");
  });

  it("queues a dry-run training job from the configure tab", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_dryrun_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          createTrainingJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").click();
    });
    await settle();

    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        targetId: "z_image_turbo_lora",
        datasetId: "dataset-a",
        datasetVersion: 5,
        outputName: "Portrait Set LoRA",
        dryRun: true,
        config: expect.objectContaining({ rank: 16, triggerWord: "miraStyle" }),
      }),
    );
    expect(container.textContent).toContain("Queued dry-run job job_dryrun_1");
  });

  it("queues a real training job when run mode is set to training", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_train_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          createTrainingJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");
    await changeField(field(container, "Run mode"), "real");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Start training").click();
    });
    await settle();

    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        targetId: "z_image_turbo_lora",
        datasetId: "dataset-a",
        outputName: "Portrait Set LoRA",
        dryRun: false,
        config: expect.objectContaining({ rank: 16, triggerWord: "miraStyle" }),
      }),
    );
    expect(container.textContent).toContain("Queued training job job_train_1");
  });

  it("keeps config edits when GPU options are recomputed", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));

    function render(gpuOptions) {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions,
          loadDataset,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    }

    root = createRoot(container);
    await act(async () => {
      render(["auto", "0"]);
    });
    await settle();
    await act(async () => {
      document.body.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");
    await changeField(field(container, "Rank"), "24");
    await changeField(field(container, "Requested GPU"), "0");

    await act(async () => {
      render(["auto", "0"]);
    });
    await settle();

    expect(field(container, "Trigger phrase").value).toBe("miraStyle");
    expect(field(container, "Rank").value).toBe("24");
    expect(field(container, "Requested GPU").value).toBe("0");

    await act(async () => {
      render(["auto"]);
    });
    await settle();

    expect(field(container, "Trigger phrase").value).toBe("miraStyle");
    expect(field(container, "Rank").value).toBe("24");
    expect(field(container, "Requested GPU").value).toBe("auto");
  });

  it("blocks job submission until required fields are valid", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn();

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset,
          createTrainingJob,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();
    await act(async () => {
      document.body.querySelector("#training-tab-configure").click();
    });

    expect(container.textContent).toContain("Select a saved dataset");
    expect(container.textContent).toContain("Add a trigger phrase");
    expect([...document.body.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").disabled).toBe(true);

    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");
    await changeField(field(container, "Checkpoint cadence"), "");

    const submitButton = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job");
    expect(container.textContent).toContain("Checkpoint cadence must be greater than zero");
    expect(submitButton.disabled).toBe(true);

    await act(async () => {
      submitButton.click();
    });
    await settle();

    expect(createTrainingJob).not.toHaveBeenCalled();
  });

});
