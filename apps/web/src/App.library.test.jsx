import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AssetDetail } from "./components/assetPanels.jsx";
import { dropUpscaledVariants, foldUpscaledAssetVariants, restrictFoldedToScope } from "./assetVariants.js";
import { DocumentStudio } from "./screens/DocumentStudio.jsx";
import { LibraryScreen } from "./screens/LibraryScreen.jsx";
import { ReplacePersonPanel } from "./screens/ReplacePersonPanel.jsx";
import { withAppContext, FakeEventSource, response, settle, changeField, openAdvancedSection } from "./main.testSupport.jsx";

// sc-12068: the Library Trashcan "Empty Trash" purge confirms through the shared desktop-safe
// appConfirm dialog rather than the raw window.confirm, which silently no-ops inside the Tauri
// WebView. Mock it so the test controls the choice and asserts the guard fired.
const { appConfirmMock } = vi.hoisted(() => ({ appConfirmMock: vi.fn(async () => true) }));
vi.mock("./appConfirm.jsx", () => ({
  appConfirm: appConfirmMock,
  useConfirm: () => appConfirmMock,
  ConfirmHost: () => null,
}));

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

  function replacePanelProps(overrides = {}) {
    const track = {
      id: "track_1",
      projectId: "project-1",
      name: "Hero",
      sourceAssetId: "clip-1",
      frames: [
        { timestamp: 0, box: { x: 0.1, y: 0.1, width: 0.2, height: 0.5 }, confidence: 0.92, detected: true, mask: "person-tracks/track_1/masks/frame_000001.png", flags: [] },
        { timestamp: 0.5, box: { x: 0.3, y: 0.1, width: 0.2, height: 0.5 }, confidence: 0.3, detected: true, mask: null, flags: ["low_confidence"] },
      ],
      corrections: [],
      status: { maskState: "active", averageConfidence: 0.6, correctionState: "ready_for_box_corrections" },
    };
    const props = {
      createPersonDetectionJob: () => {},
      createPersonTrackJob: () => {},
      detectionResult: null,
      matchingTracks: [track],
      representativeFrame: null,
      selectedDetection: null,
      selectedTrack: track,
      setPersonTrackId: () => {},
      setReplacementMode: () => {},
      setSelectedDetectionId: () => {},
      setSourceClipAssetId: () => {},
      setTrackName: () => {},
      sourceClipAssetId: "clip-1",
      trackName: "Hero",
      personTrackId: "track_1",
      replacementMode: "full_person_keep_outfit",
      videoAssets: [{ id: "clip-1", type: "video", projectId: "project-1", file: { path: "clip.mp4", mimeType: "video/mp4" } }],
      personReadiness: {},
      ...overrides,
    };
    return { track, props };
  }

  it("scrubs tracked frames and persists a corrected box", async () => {
    const saveTrackCorrections = vi.fn(() => Promise.resolve(null));
    const { props } = replacePanelProps({ saveTrackCorrections });
    root = createRoot(container);
    await act(async () => {
      root.render(<ReplacePersonPanel {...props} />);
    });

    expect(container.textContent).toContain("Review & correct track");
    expect(container.textContent).toContain("Frame 1 / 2");

    // Scrub to the second (low-confidence) frame and confirm the quality flag shows.
    const scrubber = document.body.querySelector('input[type="range"]');
    await changeField(scrubber, "1");
    expect(container.textContent).toContain("Frame 2 / 2");
    expect(container.textContent).toContain("low confidence");

    // Scrub back and nudge the box X, then save the correction set.
    await changeField(scrubber, "0");
    await changeField(document.body.querySelector('input[aria-label="Box x"]'), "0.5");

    const save = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Save corrections");
    expect(save.disabled).toBe(false);
    await act(async () => {
      save.click();
    });

    expect(saveTrackCorrections).toHaveBeenCalledWith("track_1", [
      { frameIndex: 0, rejected: false, author: "ui", source: "manual", box: { x: 0.5, y: 0.1, width: 0.2, height: 0.5 } },
    ]);
  });

  it("rejects a low-quality frame and records it as a correction", async () => {
    const saveTrackCorrections = vi.fn(() => Promise.resolve(null));
    const { props } = replacePanelProps({ saveTrackCorrections });
    root = createRoot(container);
    await act(async () => {
      root.render(<ReplacePersonPanel {...props} />);
    });

    const scrubber = document.body.querySelector('input[type="range"]');
    await changeField(scrubber, "1");

    const reject = document.body.querySelector('.person-correction-reject input[type="checkbox"]');
    await act(async () => {
      reject.click();
    });

    // Rejecting a frame disables its box inputs — replacement borrows a neighbor box.
    expect(document.body.querySelector('input[aria-label="Box x"]').disabled).toBe(true);

    const save = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Save corrections");
    await act(async () => {
      save.click();
    });

    expect(saveTrackCorrections).toHaveBeenCalledWith("track_1", [
      { frameIndex: 1, rejected: true, author: "ui", source: "manual" },
    ]);
  });

  it("shows VQA history and asks a question from the asset detail panel", async () => {
    const createVqaJob = vi.fn();
    const asset = { id: "asset-1", type: "image", displayName: "Frame One", recipe: { prompt: "neon street" } };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [asset],
            jobs: [
              {
                id: "job-vqa-1",
                type: "image_vqa",
                status: "completed",
                payload: { sourceAssetId: "asset-1", question: "What time of day is it?" },
                result: { question: "What time of day is it?", answer: "It appears to be nighttime." },
              },
            ],
            imageModels: [{ id: "sensenova_u1_8b", name: "SenseNova-U1 8B", type: "image", capabilities: ["text_to_image", "vqa"] }],
            createVqaJob,
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: asset,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    // The prior answer is surfaced on the asset.
    expect(container.textContent).toContain("It appears to be nighttime.");

    // Asking a new question dispatches a VQA job for this asset.
    const input = document.body.querySelector('textarea[aria-label="Ask about this image"]');
    expect(input).not.toBeNull();
    await changeField(input, "What is the person wearing?");
    const askButton = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Ask");
    await act(async () => {
      askButton.click();
    });
    // Defaults to the short (256-token) response length.
    expect(createVqaJob).toHaveBeenCalledWith(asset, "What is the person wearing?", 256);

    // Choosing a longer response length is passed through to the job.
    await changeField(document.body.querySelector('select[aria-label="Response length"]'), "512");
    await changeField(input, "Write a detailed critique of this image.");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Ask").click();
    });
    expect(createVqaJob).toHaveBeenLastCalledWith(asset, "Write a detailed critique of this image.", 512);
  });

  it("filters library assets by tag and edits selected asset tags", async () => {
    const updateAssetTags = vi.fn();
    const portrait = {
      id: "asset-portrait",
      projectId: "project-1",
      type: "image",
      displayName: "Portrait One",
      tags: ["portrait"],
      recipe: { prompt: "studio portrait" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const landscape = {
      id: "asset-landscape",
      projectId: "project-1",
      type: "image",
      displayName: "Wide Hill",
      tags: ["landscape"],
      recipe: { prompt: "wide hill" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Tagged" },
            assets: [portrait, landscape],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: portrait,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags,
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    await changeField(document.body.querySelector('select[aria-label="Asset tag"]'), "landscape");
    const filteredTiles = [...document.body.querySelectorAll(".asset-tile")];
    expect(filteredTiles).toHaveLength(1);
    expect(filteredTiles[0].textContent).toContain("Wide Hill");

    await changeField(document.body.querySelector('input[aria-label="Add asset tag"]'), "  Moody  ");
    await act(async () => {
      [...document.body.querySelectorAll(".asset-tag-form button")].find((button) => button.textContent === "Add").click();
    });
    expect(updateAssetTags).toHaveBeenCalledWith(portrait, ["portrait", "moody"]);

    await act(async () => {
      document.body.querySelector('button[aria-label="Remove portrait tag"]').click();
    });
    expect(updateAssetTags).toHaveBeenLastCalledWith(portrait, []);
  });

  it("searches Library assets by prompt or tag and selects all results for batch actions", async () => {
    const neonPortrait = {
      id: "asset-neon",
      projectId: "project-1",
      type: "image",
      displayName: "Neon One",
      tags: ["portrait"],
      recipe: { prompt: "neon street at night" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const neonAlley = {
      id: "asset-neon-alley",
      projectId: "project-1",
      type: "image",
      displayName: "Second Plate",
      tags: ["cyberpunk"],
      recipe: { prompt: "a neon-lit rain-slicked alley" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const forest = {
      id: "asset-forest",
      projectId: "project-1",
      type: "image",
      displayName: "Forest",
      tags: ["landscape"],
      recipe: { prompt: "misty forest" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Searchable" },
            assets: [neonPortrait, neonAlley, forest],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: neonPortrait,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    const searchInput = document.body.querySelector('input[aria-label="Search assets"]');
    expect(searchInput).not.toBeNull();
    // No query: all three tiles show and no "Select all" affordance yet.
    expect([...document.body.querySelectorAll(".asset-tile")]).toHaveLength(3);
    expect([...document.body.querySelectorAll("button")].some((button) => button.textContent.startsWith("Select all"))).toBe(false);

    // Search matches prompt text ("neon" appears in both neon prompts, not the forest).
    await changeField(searchInput, "neon");
    let tiles = [...document.body.querySelectorAll(".asset-tile")];
    expect(tiles).toHaveLength(2);
    expect(tiles.map((tile) => tile.textContent).join(" ")).not.toContain("Forest");

    // Search matches tags too — "landscape" is only a tag on the forest asset.
    await changeField(searchInput, "landscape");
    tiles = [...document.body.querySelectorAll(".asset-tile")];
    expect(tiles).toHaveLength(1);
    expect(tiles[0].textContent).toContain("Forest");

    // Search matches the display name too — "Second Plate" is in neither prompt nor tags.
    await changeField(searchInput, "second plate");
    tiles = [...document.body.querySelectorAll(".asset-tile")];
    expect(tiles).toHaveLength(1);
    expect(tiles[0].textContent).toContain("Second Plate");

    // Back to the neon result set, "Select all" picks every visible result for a batch action.
    await changeField(searchInput, "neon");
    const selectAll = [...document.body.querySelectorAll("button")].find((button) => button.textContent.startsWith("Select all"));
    expect(selectAll).toBeTruthy();
    expect(selectAll.textContent).toContain("(2)");
    await act(async () => {
      selectAll.click();
    });
    const bar = document.body.querySelector(".batch-selection-bar");
    expect(bar).toBeTruthy();
    expect(bar.textContent).toContain("2 selected");
  });

  it("moves a Library asset into a selected character's assets", async () => {
    // sc-10200: a TRUE move — the context action hits the move-to-character
    // endpoint; no character reference (Approved set entry) is created.
    const moveAssetToCharacter = vi.fn(async (asset) => ({ ...asset, origin: "character_studio" }));
    const asset = {
      id: "asset-portrait",
      projectId: "project-1",
      type: "image",
      displayName: "Portrait One",
      recipe: { prompt: "studio portrait" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Characters" },
            assets: [asset],
            characters: [
              { id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] },
              { id: "char-2", name: "Dax", type: "person", references: [], approvedReferences: [], looks: [], loras: [] },
            ],
            moveAssetToCharacter,
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: asset,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    await changeField(document.body.querySelector('select[aria-label="Target character"]'), "char-2");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Move to Character").click();
    });
    await settle();

    expect(moveAssetToCharacter).toHaveBeenCalledWith(asset, "char-2");
    expect(container.textContent).toContain("Moved to Dax's assets.");
  });

  it("bulk-discards every selected Library asset to the Trash", async () => {
    const deleteAsset = vi.fn(async () => {});
    const assetA = {
      id: "asset-a",
      projectId: "project-1",
      type: "image",
      displayName: "Plate A",
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const assetB = {
      id: "asset-b",
      projectId: "project-1",
      type: "image",
      displayName: "Plate B",
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Characters" },
            assets: [assetA, assetB],
            characters: [],
            addCharacterReference: vi.fn(),
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset,
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: null,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector('input[aria-label="Select Plate A"]').click();
    });
    await act(async () => {
      document.body.querySelector('input[aria-label="Select Plate B"]').click();
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Discard").click();
    });
    await settle();

    expect(deleteAsset).toHaveBeenCalledTimes(2);
    expect(deleteAsset).toHaveBeenCalledWith(assetA);
    expect(deleteAsset).toHaveBeenCalledWith(assetB);
    // Selection clears once the fan-out finishes, so the bar disappears.
    expect(document.body.querySelector(".batch-selection-bar")).toBeNull();
  });

  it("bulk-moves selected Library assets into a chosen character's assets", async () => {
    // sc-10200: the batch Move is a TRUE move per asset — no Approved-set links.
    const moveAssetToCharacter = vi.fn(async (asset) => ({ ...asset, origin: "character_studio" }));
    const assetA = {
      id: "asset-a",
      projectId: "project-1",
      type: "image",
      displayName: "Plate A",
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const assetB = {
      id: "asset-b",
      projectId: "project-1",
      type: "image",
      displayName: "Plate B",
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Characters" },
            assets: [assetA, assetB],
            characters: [
              { id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] },
              { id: "char-2", name: "Dax", type: "person", references: [], approvedReferences: [], looks: [], loras: [] },
            ],
            moveAssetToCharacter,
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: null,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector('input[aria-label="Select Plate A"]').click();
    });
    await act(async () => {
      document.body.querySelector('input[aria-label="Select Plate B"]').click();
    });
    await settle();

    // Reveal the inline character picker, target Dax, then confirm.
    await act(async () => {
      [...document.body.querySelectorAll(".batch-selection-bar button")].find((button) => button.textContent === "Move").click();
    });
    await settle();
    await changeField(document.body.querySelector('select[aria-label="Move target"]'), "char-2");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent?.startsWith("Move 2 to assets")).click();
    });
    await settle();

    expect(moveAssetToCharacter).toHaveBeenCalledTimes(2);
    expect(moveAssetToCharacter).toHaveBeenCalledWith(assetA, "char-2");
    expect(moveAssetToCharacter).toHaveBeenCalledWith(assetB, "char-2");
    expect(document.body.querySelector(".batch-selection-bar")).toBeNull();
  });

  it("disables the Library move action for a character that already owns the asset", async () => {
    const moveAssetToCharacter = vi.fn();
    const asset = {
      id: "asset-portrait",
      projectId: "project-1",
      type: "image",
      displayName: "Portrait One",
      recipe: { prompt: "studio portrait" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Characters" },
            assets: [asset],
            characters: [
              {
                id: "char-1",
                name: "Mira",
                type: "person",
                references: [{ assetId: "asset-portrait", role: "asset" }],
                approvedReferences: [],
                looks: [],
                loras: [],
              },
            ],
            moveAssetToCharacter,
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: asset,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    const button = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Already added");
    expect(button).toBeTruthy();
    expect(button.disabled).toBe(true);
    await act(async () => {
      button.click();
    });
    expect(moveAssetToCharacter).not.toHaveBeenCalled();
  });

  it("excludes Character Studio outputs from the Asset Library (sc-2024)", async () => {
    const studioImage = {
      id: "asset-studio",
      projectId: "project-1",
      type: "image",
      displayName: "Studio Render",
      origin: "image_studio",
      recipe: { prompt: "studio render" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const characterImage = {
      id: "asset-character",
      projectId: "project-1",
      type: "image",
      displayName: "Character Test",
      origin: "character_studio",
      recipe: { prompt: "character test" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Scoped" },
            assets: [studioImage, characterImage],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: studioImage,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    const tiles = [...document.body.querySelectorAll(".asset-tile")];
    expect(tiles).toHaveLength(1);
    expect(tiles[0].textContent).toContain("Studio Render");
    expect(container.textContent).not.toContain("Character Test");
  });

  it("Asset Library is a positive origin allow-list, not just a character exclusion (sc-8339)", async () => {
    const mk = (id, origin, displayName) => ({
      id,
      projectId: "project-1",
      type: "image",
      displayName,
      ...(origin === undefined ? {} : { origin }),
      recipe: { prompt: displayName },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    });
    const assets = [
      mk("a-image", "image_studio", "Image Studio"),
      mk("a-video", "video_studio", "Video Studio"),
      mk("a-doc", "document_studio", "Doc Studio"),
      mk("a-upload", "upload", "Uploaded"),
      mk("a-legacy", undefined, "Legacy No Origin"),
      mk("a-character", "character_studio", "Character Out"),
      mk("a-pose", "pose_library", "Pose Out"),
      mk("a-keypoint", "keypoint_library", "Keypoint Out"),
      mk("a-future", "some_future_studio", "Future Out"),
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Scoped" },
            assets,
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: assets[0],
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    const tileText = [...document.body.querySelectorAll(".asset-tile")].map((tile) => tile.textContent).join(" ");
    // Allow-listed origins + legacy (no origin) appear.
    for (const name of ["Image Studio", "Video Studio", "Doc Studio", "Uploaded", "Legacy No Origin"]) {
      expect(tileText).toContain(name);
    }
    // Everything else stays out — character, pose/keypoint library, and unknown future origins.
    for (const name of ["Character Out", "Pose Out", "Keypoint Out", "Future Out"]) {
      expect(container.textContent).not.toContain(name);
    }
  });

  it("Library detail sidebar falls back to a library asset, not the global (character) selection (sc-8339)", async () => {
    const studioImage = {
      id: "asset-studio",
      projectId: "project-1",
      type: "image",
      displayName: "Studio Render",
      origin: "image_studio",
      recipe: { prompt: "studio render" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const characterImage = {
      id: "asset-character",
      projectId: "project-1",
      type: "image",
      displayName: "Character Test",
      origin: "character_studio",
      recipe: { prompt: "character test" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Scoped" },
            assets: [characterImage, studioImage],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            // The global selection is the most-recent (character) asset — the bug case.
            selectedAsset: characterImage,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    // The detail panel shows the most-recent LIBRARY asset, never the character image.
    const detail = document.body.querySelector(".asset-detail");
    expect(detail).toBeTruthy();
    expect(detail.querySelector("h3").textContent).toBe("Studio Render");
    expect(container.textContent).not.toContain("Character Test");
  });

  it("folds original and upscaled library variants into one representative tile", async () => {
    const setPreviewAsset = vi.fn();
    const original = {
      id: "asset-original",
      projectId: "project-1",
      type: "image",
      displayName: "Castle original",
      file: { path: "assets/images/castle-original.png", mimeType: "image/png" },
      recipe: { prompt: "castle" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const upscaled = {
      id: "asset-upscaled",
      projectId: "project-1",
      type: "image",
      displayName: "Castle upscaled",
      file: { path: "assets/images/castle-upscaled.png", mimeType: "image/png" },
      lineage: { sourceAssetId: "asset-original", parents: ["asset-original"] },
      extra: { isUpscaled: true, upscaledFromAssetId: "asset-original", factor: 2, engine: "real-esrgan" },
      recipe: { prompt: "castle" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const other = {
      id: "asset-other",
      projectId: "project-1",
      type: "image",
      displayName: "Other frame",
      file: { path: "assets/images/other.png", mimeType: "image/png" },
      recipe: { prompt: "other" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };

    expect(foldUpscaledAssetVariants([original, upscaled, other]).map((asset) => asset.id)).toEqual([
      "asset-upscaled",
      "asset-other",
    ]);

    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Variants" },
            assets: [original, upscaled, other],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset,
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: upscaled,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });

    const tiles = [...document.body.querySelectorAll(".asset-tile")];
    expect(tiles).toHaveLength(2);
    expect(tiles.map((tile) => tile.textContent).join(" ")).toContain("Castle upscaled");
    expect(tiles.map((tile) => tile.textContent).join(" ")).not.toContain("Castle original");
    expect(tiles[0].querySelector("img").getAttribute("src")).toContain("castle-upscaled.png");

    await act(async () => {
      tiles[0].dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });

    expect(setPreviewAsset).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "asset-upscaled",
        variants: {
          original,
          upscaled,
        },
      }),
      // The library binds the preview to its currently filtered view so
      // navigation can't escape into other collections.
      expect.arrayContaining([expect.objectContaining({ id: "asset-upscaled" })]),
    );
  });

  it("drops upscaled variants from Recent Batches so a generation is not shown twice", () => {
    const original = { id: "asset-original", type: "image" };
    const upscaled = {
      id: "asset-upscaled",
      type: "image",
      extra: { isUpscaled: true, upscaledFromAssetId: "asset-original" },
    };
    const other = { id: "asset-other", type: "image" };

    // Original present -> keep the original tile, hide its upscaled duplicate.
    expect(dropUpscaledVariants([original, upscaled, other]).map((asset) => asset.id)).toEqual([
      "asset-original",
      "asset-other",
    ]);

    // Original gone (e.g. purged) -> the upscale stays so it doesn't vanish entirely.
    expect(dropUpscaledVariants([upscaled, other]).map((asset) => asset.id)).toEqual([
      "asset-upscaled",
      "asset-other",
    ]);
  });

  it("binds fullscreen preview navigation to the launch collection", () => {
    // The full project (what the old global navigation roamed across).
    const folded = [
      { id: "batch-1", type: "image" },
      { id: "batch-2", type: "image" },
      { id: "library-1", type: "image" },
      { id: "character-1", type: "image" },
    ];

    // Launched from a two-image batch -> navigation stays on those two, in order,
    // and never reaches the library/character assets.
    expect(restrictFoldedToScope(folded, ["batch-1", "batch-2"]).map((asset) => asset.id)).toEqual([
      "batch-1",
      "batch-2",
    ]);

    // A scoped id that has since been discarded/purged drops out instead of
    // leaking the neighbouring collection in to fill its slot.
    expect(restrictFoldedToScope(folded, ["batch-1", "gone", "batch-2"]).map((asset) => asset.id)).toEqual([
      "batch-1",
      "batch-2",
    ]);

    // A folded scope id resolves through its variants, and no scope falls back to
    // the full set (legacy callers).
    const variant = { id: "up-1", type: "image", variants: { original: { id: "orig-1" }, upscaled: { id: "up-1" } } };
    expect(restrictFoldedToScope([variant, folded[2]], ["orig-1"]).map((asset) => asset.id)).toEqual(["up-1"]);
    expect(restrictFoldedToScope(folded, null)).toBe(folded);
  });

  it("shows discarded assets in Trashcan and exposes restore and purge actions", async () => {
    const updateAssetStatus = vi.fn();
    const purgeAsset = vi.fn();
    const active = {
      id: "asset-active",
      projectId: "project-1",
      type: "image",
      displayName: "Active Frame",
      recipe: { prompt: "active" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const trashed = {
      id: "asset-trash",
      projectId: "project-1",
      type: "image",
      displayName: "Discarded Frame",
      recipe: { prompt: "discarded" },
      status: { favorite: false, rating: 0, rejected: false, trashed: true },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Trash" },
            assets: [active, trashed],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset,
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: trashed,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus,
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    expect([...document.body.querySelectorAll(".asset-tile")].map((tile) => tile.textContent).join(" ")).toContain("Active Frame");
    expect([...document.body.querySelectorAll(".asset-tile")].map((tile) => tile.textContent).join(" ")).not.toContain("Discarded Frame");

    await act(async () => {
      [...document.body.querySelectorAll('button')].find((button) => button.textContent === "Trashcan").click();
    });
    expect([...document.body.querySelectorAll(".asset-tile")].map((tile) => tile.textContent).join(" ")).toContain("Discarded Frame");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Restore").click();
    });
    expect(updateAssetStatus).toHaveBeenCalledWith(trashed, { trashed: false });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Purge").click();
    });
    expect(purgeAsset).toHaveBeenCalledWith(trashed);
  });

  it("Empty Trash purges every discarded asset in the Library Trashcan view only", async () => {
    appConfirmMock.mockClear();
    appConfirmMock.mockResolvedValue(true);
    const purgeAsset = vi.fn();
    const active = {
      id: "asset-active",
      projectId: "project-1",
      type: "image",
      displayName: "Active Frame",
      status: { trashed: false },
    };
    const trashedA = {
      id: "trash-a",
      projectId: "project-1",
      type: "image",
      displayName: "Trash A",
      status: { trashed: true },
    };
    const trashedB = {
      id: "trash-b",
      projectId: "project-1",
      type: "image",
      displayName: "Trash B",
      status: { trashed: true },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Trash" },
            assets: [active, trashedA, trashedB],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset,
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: null,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    // Empty Trash only appears in the Trashcan view.
    expect([...document.body.querySelectorAll("button")].some((button) => button.textContent.startsWith("Empty Trash"))).toBe(false);
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Trashcan").click();
    });
    const emptyButton = [...document.body.querySelectorAll("button")].find((button) => button.textContent.startsWith("Empty Trash"));
    expect(emptyButton).toBeTruthy();
    expect(emptyButton.textContent).toContain("(2)");

    await act(async () => {
      emptyButton.click();
    });
    await settle();
    expect(appConfirmMock).toHaveBeenCalledWith(expect.objectContaining({ tone: "danger" }));
    expect(purgeAsset).toHaveBeenCalledTimes(2);
    expect(purgeAsset).toHaveBeenCalledWith(trashedA);
    expect(purgeAsset).toHaveBeenCalledWith(trashedB);
    expect(purgeAsset).not.toHaveBeenCalledWith(active);
  });

  it("gates Document Studio behind a model download when no interleave model is present", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [],
            createInterleaveJob: () => {},
            documentLocalJobs: [],
            gpuOptions: ["auto"],
            imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", capabilities: ["text_to_image"] }],
            jobAction: () => {},
            rememberLocalGenerationJob: () => {},
            setActiveView: () => {},
            requestedGpu: "auto",
            setRequestedGpu: () => {},
          },
          <DocumentStudio />,
        ),
      );
    });
    await settle();
    // An image model without interleave doesn't satisfy Document Studio → gate shows.
    expect(container.textContent).toContain("Document Studio needs an interleave-capable model");
    expect(document.body.querySelector(".model-availability-gate")).not.toBeNull();
    expect(document.body.querySelector(".studio-form")).toBeNull();
  });

  it("DocumentStudio renders an interleaved document and submits a compose job", async () => {
    const createInterleaveJob = vi.fn(() =>
      Promise.resolve({ id: "job-il-new", type: "image_interleave", status: "queued" }),
    );
    const setActiveView = vi.fn();
    const rememberLocalGenerationJob = vi.fn();
    const imageAsset = {
      id: "img-1",
      type: "image",
      projectId: "project-1",
      file: { path: "assets/images/a.png" },
      url: "/api/v1/projects/project-1/files/assets/images/a.png",
    };
    const completedJob = {
      id: "job-il-done",
      type: "image_interleave",
      status: "completed",
      payload: { prompt: "tea guide" },
      result: {
        segments: [
          { type: "text", text: "Boil the water." },
          { type: "image", assetId: "img-1", path: "assets/images/a.png" },
          { type: "text", text: "Steep three minutes." },
        ],
      },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [imageAsset],
            createInterleaveJob,
            documentLocalJobs: [completedJob],
            gpuOptions: ["auto"],
            imageModels: [
              { id: "sensenova_u1_8b", name: "SenseNova-U1 8B", type: "image", capabilities: ["text_to_image", "interleave"] },
              { id: "z_image_turbo", name: "Z-Image Turbo", type: "image", capabilities: ["text_to_image"] },
            ],
            jobAction: () => {},
            rememberLocalGenerationJob,
            setActiveView,
            requestedGpu: "auto",
            setRequestedGpu: () => {},
          },
          <DocumentStudio />,
        ),
      );
    });
    await settle();

    // The completed document renders text segments in order + the image segment.
    expect(container.textContent).toContain("Boil the water.");
    expect(container.textContent).toContain("Steep three minutes.");
    const image = document.body.querySelector("img.document-image");
    expect(image).not.toBeNull();
    expect(image.getAttribute("src")).toContain("assets/images/a.png");

    // Only interleave-capable models are offered (Z-Image filtered out).
    const optionValues = [...document.body.querySelectorAll("select option")].map((option) => option.value);
    expect(optionValues).toContain("sensenova_u1_8b");
    expect(optionValues).not.toContain("z_image_turbo");
    // The size control offers the interleave buckets.
    expect(optionValues).toContain("2048x1152");
    // The system prompt now lives in the Advanced disclosure (GPU · system prompt),
    // collapsed by default. Expand it, then confirm it's prefilled with the default.
    await openAdvancedSection();
    const textareas = [...document.body.querySelectorAll("textarea")];
    expect(textareas.some((field) => field.value.includes("multimodal assistant capable of reasoning"))).toBe(true);

    // Submitting composes an interleave job with prompt, model, size, and max images.
    await changeField(document.body.querySelector("textarea"), "An illustrated guide to brewing tea");
    await changeField(document.body.querySelector("input[type='number']"), "999");
    const submit = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Compose document"),
    );
    await act(async () => {
      submit.closest("form").dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(createInterleaveJob).toHaveBeenCalledTimes(1);
    const payload = createInterleaveJob.mock.calls[0][0];
    expect(payload.prompt).toBe("An illustrated guide to brewing tea");
    expect(payload.model).toBe("sensenova_u1_8b");
    expect(payload.maxImages).toBe(10);
    expect(payload.width).toBe(2048);
    expect(payload.height).toBe(1152);
    // Unedited system prompt is not sent (worker uses its own default).
    expect(payload.advanced?.systemMessage).toBeUndefined();
    // No reference images attached, so no reference-strength slider and no image-guidance
    // field (the run stays on the worker's plain-generation defaults).
    expect(container.textContent).not.toContain("Reference strength");
    expect(payload.advanced?.imageGuidanceScale).toBeUndefined();

    // Submitting stacks the run in the studio rather than routing to the Queue.
    await settle();
    expect(rememberLocalGenerationJob).toHaveBeenCalledWith(
      "document",
      expect.objectContaining({ id: "job-il-new" }),
    );
    expect(setActiveView).not.toHaveBeenCalledWith("Queue");
  });

  it("DocumentStudio grounds on a storyboard frame and sends reference guidance in frame order", async () => {
    const createInterleaveJob = vi.fn(() =>
      Promise.resolve({ id: "job-il-ref", type: "image_interleave", status: "queued" }),
    );
    const imageAsset = {
      id: "img-ref-1",
      type: "image",
      displayName: "Hero shot",
      projectId: "project-1",
      file: { path: "assets/images/hero.png" },
      url: "/api/v1/projects/project-1/files/assets/images/hero.png",
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [imageAsset],
            createInterleaveJob,
            documentLocalJobs: [],
            gpuOptions: ["auto"],
            imageModels: [
              { id: "sensenova_u1_8b", name: "SenseNova-U1 8B", type: "image", capabilities: ["text_to_image", "interleave"] },
            ],
            jobAction: () => {},
            rememberLocalGenerationJob: () => {},
            setActiveView: () => {},
            requestedGpu: "auto",
            setRequestedGpu: () => {},
          },
          <DocumentStudio />,
        ),
      );
    });
    await settle();

    // The reference-guidance slider is part of the storyboard block (always visible),
    // defaulting to the neutral 1.0 baseline. The old "Reference strength" label is gone.
    expect(container.textContent).not.toContain("Reference strength");
    expect(container.textContent).toContain("Reference guidance");
    const slider = document.body.querySelector(".doc-refguidance-slider");
    expect(slider).not.toBeNull();
    expect(slider.value).toBe("1");

    // The storyboard starts empty. Add a frame, then attach a reference image to it
    // through the frame's picker: open it, pick the card, confirm.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent.includes("Add frame")).click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".doc-frame-thumb").click();
    });
    await settle();
    await act(async () => {
      document.body.querySelector(".asset-picker-card").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use Selection").click();
    });
    await settle();

    // Dial the reference guidance up toward the character/identity regime and submit.
    await changeField(slider, "1.5");
    await changeField(document.body.querySelector("textarea"), "A brand lookbook grounded in the hero shot");
    await act(async () => {
      [...document.body.querySelectorAll("button")]
        .find((button) => button.textContent.includes("Compose document"))
        .closest("form")
        .dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });

    expect(createInterleaveJob).toHaveBeenCalledTimes(1);
    const payload = createInterleaveJob.mock.calls[0][0];
    // sourceAssetIds are derived from the storyboard frames in order.
    expect(payload.sourceAssetIds).toEqual(["img-ref-1"]);
    expect(payload.advanced.imageGuidanceScale).toBe(1.5);
  });

  it("DocumentStudio stacks a queued compose run beneath the active document", async () => {
    const completedJob = {
      id: "doc-job-done",
      type: "image_interleave",
      status: "completed",
      createdAt: "2026-05-27T10:00:00Z",
      payload: { prompt: "tea guide" },
      result: {
        segments: [
          { type: "text", text: "Boil the water." },
          { type: "text", text: "Steep three minutes." },
        ],
      },
    };
    const queuedJob = {
      id: "doc-job-queued",
      type: "image_interleave",
      status: "queued",
      createdAt: "2026-05-27T10:01:00Z",
      payload: { prompt: "coffee guide" },
      result: {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [],
            createInterleaveJob: () => {},
            documentLocalJobs: [completedJob, queuedJob],
            gpuOptions: ["auto"],
            imageModels: [
              { id: "sensenova_u1_8b", name: "SenseNova-U1 8B", type: "image", capabilities: ["text_to_image", "interleave"] },
            ],
            jobAction: () => {},
            rememberLocalGenerationJob: () => {},
            setActiveView: () => {},
            requestedGpu: "auto",
            setRequestedGpu: () => {},
          },
          <DocumentStudio />,
        ),
      );
    });
    await settle();

    // Both runs stack: the finished document plus the queued run's progress card.
    expect(document.body.querySelectorAll(".local-job-group").length).toBe(2);
    expect(container.textContent).toContain("Boil the water.");
    expect(document.body.querySelector(".document-view")).not.toBeNull();
    expect(document.body.querySelector(".worker-progress-card.queued")).not.toBeNull();
    expect(container.textContent).not.toContain("Your generated document will appear here.");
  });

  it("AssetDetail reopens a saved document from the Library", async () => {
    global.fetch.mockImplementation((url) => {
      if (String(url).includes("assets/documents")) {
        return Promise.resolve(
          response({
            schemaVersion: 1,
            id: "doc_1",
            segments: [
              { type: "text", text: "Boil the water." },
              { type: "image", assetId: "img-1", path: "assets/images/a.png" },
              { type: "text", text: "Steep three minutes." },
            ],
          }),
        );
      }
      return Promise.resolve(response([]));
    });
    const documentAsset = {
      id: "doc_1",
      type: "document",
      projectId: "project-1",
      displayName: "Tea guide",
      file: { path: "assets/documents/doc_1.json", mimeType: "application/json" },
      url: "/api/v1/projects/project-1/files/assets/documents/doc_1.json",
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        <AssetDetail
          asset={documentAsset}
          deleteAsset={() => {}}
          purgeAsset={() => {}}
          onPreview={() => {}}
          onSendImage={() => {}}
          onSendVideo={() => {}}
          onSendEditor={() => {}}
          updateAssetStatus={() => {}}
        />,
      );
    });
    await settle();

    // The saved document's text + image segments render in order.
    expect(container.textContent).toContain("Boil the water.");
    expect(container.textContent).toContain("Steep three minutes.");
    const image = document.body.querySelector("img.document-image");
    expect(image).not.toBeNull();
    expect(image.getAttribute("src")).toContain("assets/images/a.png");
    // The document JSON was fetched from its file path (with an abort signal).
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("assets/documents/doc_1.json"),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });
});
