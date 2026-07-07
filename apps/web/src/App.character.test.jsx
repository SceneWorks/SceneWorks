import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { AssetPickerField, CharacterImportDialog } from "./components/AssetPicker.jsx";
import { CharacterStudio } from "./screens/CharacterStudio.jsx";
import { CharacterAssets, CharacterDatasets } from "./screens/characterPanels.jsx";
import { withAppContext, FakeEventSource, response, errorResponse, settle, field, changeField } from "./main.testSupport.jsx";

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

  it("keeps in-progress picker selection across parent rerenders", async () => {
    const onChange = vi.fn();
    const assets = [
      { id: "image-alpha", type: "image", displayName: "Alpha" },
      { id: "image-beta", type: "image", displayName: "Beta" },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(<AssetPickerField assets={assets} label="Source" onChange={onChange} value="image-alpha" />);
    });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Change").click();
    });
    await act(async () => {
      document.body.querySelectorAll(".asset-picker-card")[1].click();
    });
    await act(async () => {
      root.render(<AssetPickerField assets={[...assets]} label="Source" onChange={onChange} value="image-alpha" />);
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use Selection").click();
    });

    expect(onChange).toHaveBeenCalledWith("image-beta");
  });

  it("toggles and confirms multiple assets through the thumbnail picker", async () => {
    const onChange = vi.fn();
    const assets = [
      { id: "image-alpha", type: "image", displayName: "Alpha" },
      { id: "image-beta", type: "image", displayName: "Beta" },
      { id: "image-gamma", type: "image", displayName: "Gamma" },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(<AssetPickerField assets={assets} label="Reference assets" multiple onChange={onChange} values={[]} />);
    });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Select").click();
    });

    const cards = [...document.body.querySelectorAll(".asset-picker-card")];
    await act(async () => {
      cards[0].click();
      cards[1].click();
      cards[0].click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use Selection").click();
    });

    expect(onChange).toHaveBeenCalledWith(["image-beta"]);

    await act(async () => {
      root.render(<AssetPickerField assets={assets} label="Reference assets" multiple onChange={onChange} values={["image-beta"]} />);
    });

    expect(document.body.querySelectorAll(".asset-preview-chip")).toHaveLength(1);
    expect(container.textContent).toContain("Beta");
  });

  it("drops the category tabs when showCategories is false (sc-6042 reference picker)", async () => {
    const onChange = vi.fn();
    const assets = [
      { id: "image-alpha", type: "image", displayName: "Alpha" },
      { id: "clip-gamma", type: "video", displayName: "Gamma", file: { mimeType: "video/mp4" } },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        <AssetPickerField assets={assets} label="Reference assets" multiple onChange={onChange} showCategories={false} values={[]} />,
      );
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Select").click();
    });

    // No "Asset category" segmented control, but the scoped grid still renders.
    expect(document.body.querySelector('[aria-label="Asset category"]')).toBeNull();
    expect(container.textContent).not.toContain("Renders");
    expect(document.body.querySelectorAll(".asset-picker-card")).toHaveLength(2);
  });

  const characterImportAssets = () => [
    { id: "p-img1", type: "image", projectId: "project-1", displayName: "Project Image 1" },
    { id: "p-img2", type: "image", projectId: "project-1", displayName: "Project Image 2" },
    { id: "p-vid1", type: "video", projectId: "project-1", displayName: "Project Video 1", file: { mimeType: "video/mp4" } },
    // Already in the character library → excluded from the Images tab.
    { id: "c-img", type: "image", projectId: "project-1", displayName: "Already linked", recipe: { normalizedSettings: { characterId: "char-1" } } },
    // Different project → excluded everywhere.
    { id: "x-img", type: "image", projectId: "project-2", displayName: "Other project" },
  ];

  function renderCharacterImport(overrides = {}) {
    const props = {
      assets: characterImportAssets(),
      character: { id: "char-1", name: "Mira", references: [] },
      characterId: "char-1",
      characterName: "Mira",
      importAsset: vi.fn(async () => ({ id: "uploaded-1", type: "image", projectId: "project-1", displayName: "Fresh" })),
      onClose: vi.fn(),
      onImport: vi.fn(async () => {}),
      projectId: "project-1",
      ...overrides,
    };
    root = createRoot(container);
    return { props };
  }

  it("imports the selected project images into the character library (sc-6042)", async () => {
    const { props } = renderCharacterImport();
    await act(async () => {
      root.render(<CharacterImportDialog {...props} />);
    });

    // Three tabs, all multi-select; Images excludes already-linked + other-project.
    expect([...document.body.querySelectorAll('[role="tab"]')].map((tab) => tab.textContent.replace(/\d+/g, ""))).toEqual([
      "Images",
      "Videos",
      "Upload",
    ]);
    const cards = [...document.body.querySelectorAll(".asset-picker-card")];
    expect(cards).toHaveLength(2);
    await act(async () => {
      cards[0].click();
      cards[1].click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent.startsWith("Import")).click();
    });
    expect(props.onImport).toHaveBeenCalledWith(["p-img1", "p-img2"]);
    expect(props.onClose).toHaveBeenCalled();
  });

  it("lists project videos and uploads local files into the character library (sc-6042)", async () => {
    const { props } = renderCharacterImport();
    await act(async () => {
      root.render(<CharacterImportDialog {...props} />);
    });

    // Videos tab lists only project videos.
    await act(async () => {
      [...document.body.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent.startsWith("Videos")).click();
    });
    expect(document.body.querySelectorAll(".asset-picker-card")).toHaveLength(1);
    expect(document.body.querySelector('[title="p-vid1"]')).not.toBeNull();

    // Upload tab accepts local image + video files, then imports + attaches them.
    await act(async () => {
      [...document.body.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Upload").click();
    });
    const fileInput = document.body.querySelector('input[type="file"]');
    expect(fileInput.getAttribute("accept")).toBe("image/*,video/*");
    expect(fileInput.multiple).toBe(true);
    const file = new File(["x"], "ref.png", { type: "image/png" });
    await act(async () => {
      Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
      fileInput.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    expect(props.importAsset).toHaveBeenCalledTimes(1);
    expect(props.onImport).toHaveBeenCalledWith(["uploaded-1"]);
  });

  it("keeps unsaved character reference selections when a multi-add partially fails", async () => {
    const addCharacterReference = vi.fn(async (_characterId, reference) => {
      if (reference.assetId === "image-beta") {
        throw new Error("network hiccup");
      }
      return {};
    });
    // sc-6042: the reference picker is scoped to this character's assets, so the
    // candidates must already belong to char-1 (here: generated for it).
    const assets = [
      { id: "image-alpha", type: "image", displayName: "Alpha", recipe: { normalizedSettings: { characterId: "char-1" } } },
      { id: "image-beta", type: "image", displayName: "Beta", recipe: { normalizedSettings: { characterId: "char-1" } } },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference,
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add image or frame").click();
    });

    const cards = [...document.body.querySelectorAll(".asset-picker-card")];
    await act(async () => {
      cards[0].click();
      cards[1].click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use Selection").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Add").click();
    });

    expect(addCharacterReference).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("Added 1 reference");
    expect(container.textContent).toContain("network hiccup");
    expect(document.body.querySelectorAll(".asset-preview-chip")).toHaveLength(1);
    expect(container.textContent).toContain("Beta");
  });

  it("groups the panels into accessible tabs, preserves working state, and persists the active tab (sc-2294)", async () => {
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    // Five accessible tabs, Character selected by default.
    const tablist = document.body.querySelector('[role="tablist"]');
    expect(tablist.getAttribute("aria-label")).toBe("Character workspace");
    expect([...document.body.querySelectorAll('[role="tab"]')].map((tab) => tab.textContent)).toEqual([
      "Character",
      "Assets",
      "Angles",
      "Poses",
      "Test",
    ]);
    expect(document.body.querySelector("#character-tab-character").getAttribute("aria-selected")).toBe("true");

    // The active panel is shown; the others are mounted but hidden so their state survives.
    expect(document.body.querySelector("#character-panel-character").hidden).toBe(false);
    expect(document.body.querySelector("#character-panel-assets").hidden).toBe(true);
    expect(document.body.querySelector("#character-panel-test").hidden).toBe(true);

    // The identity form carries its own section heading (sc-2295) like its siblings.
    expect([...document.body.querySelectorAll("#character-panel-character .eyebrow")].map((node) => node.textContent)).toContain(
      "Identity",
    );

    // Edit the lifted character draft, switch tabs, and switch back: the draft survives.
    await changeField(field(container, "Name"), "Mira Vex");
    await act(async () => {
      document.body.querySelector("#character-tab-test").click();
    });
    expect(document.body.querySelector("#character-panel-test").hidden).toBe(false);
    expect(document.body.querySelector("#character-panel-character").hidden).toBe(true);
    expect(document.body.querySelector("#character-tab-test").getAttribute("aria-selected")).toBe("true");
    await act(async () => {
      document.body.querySelector("#character-tab-character").click();
    });
    expect(field(container, "Name").value).toBe("Mira Vex");

    // Keyboard: arrows move between tabs (and wrap), Home/End jump to the ends.
    await act(async () => {
      container
        .querySelector("#character-tab-character")
        .dispatchEvent(new window.KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    });
    expect(document.body.querySelector("#character-tab-assets").getAttribute("aria-selected")).toBe("true");
    await act(async () => {
      container
        .querySelector("#character-tab-assets")
        .dispatchEvent(new window.KeyboardEvent("keydown", { key: "End", bubbles: true }));
    });
    expect(document.body.querySelector("#character-tab-test").getAttribute("aria-selected")).toBe("true");
    await act(async () => {
      container
        .querySelector("#character-tab-test")
        .dispatchEvent(new window.KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    });
    expect(document.body.querySelector("#character-tab-character").getAttribute("aria-selected")).toBe("true");

    // The active tab is persisted per workspace and restored on remount.
    await act(async () => {
      document.body.querySelector("#character-tab-poses").click();
    });
    expect(window.localStorage.getItem("sceneworks-studio-character-project-1")).toContain('"activeTab":"poses"');
    await act(async () => {
      root.unmount();
    });
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });
    expect(document.body.querySelector("#character-tab-poses").getAttribute("aria-selected")).toBe("true");
    expect(document.body.querySelector("#character-panel-poses").hidden).toBe(false);
  });

  it("confirms before archiving and lets you view and restore archived characters (sc-6066)", async () => {
    const archiveCharacter = vi.fn();
    const unarchiveCharacter = vi.fn(async (id) => ({
      id,
      name: "Old Hero",
      type: "creature",
      references: [],
      approvedReferences: [],
      looks: [],
      loras: [],
    }));
    const listArchivedCharacters = vi.fn(async () => [
      { id: "char-archived", name: "Old Hero", type: "creature", references: [{ assetId: "a" }] },
    ]);
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter,
      unarchiveCharacter,
      listArchivedCharacters,
      assets: [],
      attachCharacterLora: () => {},
      characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    const archiveButton = () =>
      [...document.body.querySelectorAll("#character-panel-character button")].find((button) => button.textContent === "Archive");

    // Declining the confirm leaves the character untouched.
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    await act(async () => {
      archiveButton().click();
    });
    expect(confirmSpy).toHaveBeenCalled();
    expect(archiveCharacter).not.toHaveBeenCalled();

    // Confirming archives it.
    confirmSpy.mockReturnValue(true);
    await act(async () => {
      archiveButton().click();
    });
    expect(archiveCharacter).toHaveBeenCalledWith("char-1");

    // The archived view is hidden until opened, then lazily fetches archived characters.
    expect(document.body.querySelector(".archived-character-list")).toBeNull();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Show archived characters").click();
    });
    await settle();
    expect(listArchivedCharacters).toHaveBeenCalled();
    expect(document.body.querySelector(".archived-character-list").textContent).toContain("Old Hero");

    // Restore returns it to the active roster.
    await act(async () => {
      [...document.body.querySelectorAll(".archived-character-row button")].find((button) => button.textContent === "Restore").click();
    });
    await settle();
    expect(unarchiveCharacter).toHaveBeenCalledWith("char-archived");
    expect(document.body.querySelector(".archived-character-list")).toBeNull();

    confirmSpy.mockRestore();
  });

  it("surfaces character videos alongside images in the Assets tab (sc-2296)", async () => {
    const assets = [
      { id: "img-1", type: "image", displayName: "Still", projectId: "project-1", recipe: { normalizedSettings: { characterId: "char-1" } } },
      {
        id: "vid-1",
        type: "video",
        displayName: "Clip",
        projectId: "project-1",
        file: { mimeType: "video/mp4", path: "clip.mp4" },
        recipe: { normalizedSettings: { characterId: "char-1" } },
      },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: assets,
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    await act(async () => {
      document.body.querySelector("#character-tab-assets").click();
    });
    const section = [...document.body.querySelectorAll(".character-section")].find(
      (item) => item.querySelector(".eyebrow")?.textContent === "Character assets",
    );
    expect(section).toBeTruthy();
    // Both the still and the clip are listed; the clip renders as a <video>.
    expect(section.querySelectorAll(".character-asset-card").length).toBe(2);
    expect(section.querySelector("video")).toBeTruthy();
    expect([...section.querySelectorAll("button")].some((button) => button.textContent.startsWith("Media (2)"))).toBe(true);
  });

  it("surfaces assets moved from the Library in the Character Assets tab immediately", async () => {
    const movedAsset = {
      id: "asset-moved",
      type: "image",
      displayName: "Library Portrait",
      projectId: "project-1",
      recipe: { prompt: "library portrait" },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets: [movedAsset],
            attachCharacterLora: () => {},
            characters: [
              {
                id: "char-1",
                name: "Mira",
                type: "person",
                references: [{ assetId: "asset-moved", role: "asset", approved: false }],
                approvedReferences: [],
                looks: [],
                loras: [],
              },
            ],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: [movedAsset],
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    await act(async () => {
      document.body.querySelector("#character-tab-assets").click();
    });
    const section = [...document.body.querySelectorAll(".character-section")].find(
      (item) => item.querySelector(".eyebrow")?.textContent === "Character assets",
    );
    expect(section).toBeTruthy();
    expect(section.textContent).toContain("Generated for Mira (1)");
    expect(section.querySelectorAll(".character-asset-card")).toHaveLength(1);
    expect([...section.querySelectorAll("button")].some((button) => button.textContent.startsWith("Media (1)"))).toBe(true);
  });

  it("switches the active character via the compact selector (sc-2025)", async () => {
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [
        { id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] },
        { id: "char-2", name: "Dax", type: "person", references: [], approvedReferences: [], looks: [], loras: [] },
      ],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    // The full-height character list is replaced by a compact selector pill.
    expect(document.body.querySelector(".character-list")).toBeNull();
    const pill = document.body.querySelector(".compact-selector-pill");
    expect(pill.textContent).toContain("Mira");
    expect(field(container, "Name").value).toBe("Mira");

    // Open the dropdown and switch to the second character.
    await act(async () => {
      pill.click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Dax")).click();
    });

    expect(field(container, "Name").value).toBe("Dax");
    expect(document.body.querySelector(".compact-selector-pill").textContent).toContain("Dax");
  });

  it("creates a character from the selector's New item (sc-2025)", async () => {
    const createCharacter = vi.fn(async (payload) => ({
      id: "char-new",
      ...payload,
      references: [],
      approvedReferences: [],
      looks: [],
      loras: [],
    }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
      createCharacter,
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    // No inline header create form anymore — creation lives in the dropdown.
    expect([...document.body.querySelectorAll("input")].some((input) => input.getAttribute("aria-label") === "Character name")).toBe(false);

    await act(async () => {
      document.body.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      document.body.querySelector(".compact-selector-create").click();
    });

    expect(createCharacter).toHaveBeenCalledWith(expect.objectContaining({ name: "New character", type: "person" }));
  });

  it("fires a one-click angle-set batch job from the Character Studio panel", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-angle" }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [
        {
          id: "char-1",
          name: "Mira",
          type: "person",
          references: [],
          approvedReferences: [{ assetId: "ref-1", role: "hero", asset: { id: "ref-1", type: "image", displayName: "Mira ref" } }],
          looks: [],
          loras: [],
        },
      ],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      createImageJob,
      importAsset: vi.fn(),
      imageLocalJobs: [],
      rememberLocalGenerationJob: vi.fn(),
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [
        {
          id: "instantid_realvisxl",
          name: "InstantID (RealVisXL)",
          type: "image",
          ui: { viewAngles: [{ id: "front", label: "Front" }, { id: "left_profile", label: "Left profile" }, { id: "up", label: "Looking up" }] },
        },
      ],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    // The angle-set panel renders with the model's angle count in the button.
    const generateButton = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.startsWith("Generate angle set"),
    );
    expect(generateButton).toBeTruthy();
    expect(generateButton.textContent).toContain("3 views");

    await act(async () => {
      generateButton.click();
    });

    // One batch job with a valid count (worker expands to all pack angles) +
    // advanced.angleSet; the job is tracked for live in-panel progress.
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        model: "instantid_realvisxl",
        characterId: "char-1",
        referenceAssetId: "ref-1",
        count: 1,
        advanced: expect.objectContaining({ angleSet: true, ipAdapterScale: 0.8 }),
      }),
    );
    expect(baseContext.rememberLocalGenerationJob).toHaveBeenCalledWith("image", { id: "job-angle" });
  });

  it("seeds the angle-set Identity-structure lock from angleSetDefault (sc-8354)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-angle-cn" }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [
        {
          id: "char-1",
          name: "Mira",
          type: "person",
          references: [],
          approvedReferences: [{ assetId: "ref-1", role: "hero", asset: { id: "ref-1", type: "image", displayName: "Mira ref" } }],
          looks: [],
          loras: [],
        },
      ],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      createImageJob,
      importAsset: vi.fn(),
      imageLocalJobs: [],
      rememberLocalGenerationJob: vi.fn(),
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [
        {
          id: "instantid_realvisxl",
          name: "InstantID (RealVisXL)",
          type: "image",
          ui: {
            referenceStrengthDefault: 0.8,
            identityStructure: { label: "Identity structure", default: 0.8, angleSetDefault: 0.65, min: 0.3, max: 1.0, step: 0.05 },
            viewAngles: [{ id: "front", label: "Front" }, { id: "left_profile", label: "Left profile" }],
          },
        },
      ],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    // The Identity-structure slider seeds from angleSetDefault, not the 0.80 single-image default.
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Advanced")?.click();
    });
    const lockSlider = [...document.body.querySelectorAll(".reference-strength")].find((label) =>
      label.textContent.includes("Identity structure"),
    );
    expect(lockSlider?.querySelector("span")?.textContent).toBe("0.65");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent.startsWith("Generate angle set")).click();
    });

    // The softer lock rides advanced.controlnetConditioningScale so the worker's angle-set default
    // isn't pinned back to the single-image 0.80.
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        advanced: expect.objectContaining({ angleSet: true, controlnetConditioningScale: 0.65 }),
      }),
    );
  });

  it("fires a pose-library batch job from the Character Studio pose picker", async () => {
    const poseKeypoints = Array.from({ length: 18 }, (_, i) => [0.5, i / 18]);
    global.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        version: 1,
        categories: ["standing"],
        poses: [
          { id: "standing_01", category: "standing", label: "Standing 01", preview: "poses/standing_01.png", keypoints: poseKeypoints },
        ],
      }),
    }));
    const createImageJob = vi.fn(async () => ({ id: "job-pose" }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [
        {
          id: "char-1",
          name: "Mira",
          type: "person",
          references: [],
          approvedReferences: [{ assetId: "ref-1", role: "hero", asset: { id: "ref-1", type: "image", displayName: "Mira ref" } }],
          looks: [],
          loras: [],
        },
      ],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      createImageJob,
      importAsset: vi.fn(),
      imageLocalJobs: [],
      rememberLocalGenerationJob: vi.fn(),
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [
        {
          id: "instantid_realvisxl",
          name: "InstantID (RealVisXL)",
          type: "image",
          ui: { poseLibrary: true },
        },
      ],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });
    // Let the bundled pose library fetch resolve and the picker render.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    // The pose thumbnail loads from the library; select it.
    const poseButton = [...document.body.querySelectorAll("button")].find((button) =>
      (button.getAttribute("aria-label") ?? "").includes("pose Standing 01"),
    );
    expect(poseButton).toBeTruthy();
    await act(async () => {
      poseButton.click();
    });

    const generateButton = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.startsWith("Generate") && button.textContent.includes("pose"),
    );
    expect(generateButton).toBeTruthy();
    await act(async () => {
      generateButton.click();
    });

    // One batch job carrying the selected pose's keypoints in advanced.poses; the worker
    // emits one image per pose. count stays within the API's 1-8 guard.
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        model: "instantid_realvisxl",
        characterId: "char-1",
        referenceAssetId: "ref-1",
        count: 1,
        advanced: expect.objectContaining({
          ipAdapterScale: 0.8,
          poses: [{ id: "standing_01", keypoints: poseKeypoints }],
          faceRestore: false,
        }),
      }),
    );
    expect(baseContext.rememberLocalGenerationJob).toHaveBeenCalledWith("image", { id: "job-pose" });
  });

  it("exposes a pose-lock-strength slider for the strict Z-Image tier and threads controlScale (sc-2257)", async () => {
    const poseKeypoints = Array.from({ length: 18 }, (_, i) => [0.5, i / 18]);
    global.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        version: 1,
        categories: ["standing"],
        poses: [
          { id: "standing_01", category: "standing", label: "Standing 01", preview: "poses/standing_01.png", keypoints: poseKeypoints },
        ],
      }),
    }));
    const createImageJob = vi.fn(async () => ({ id: "job-zpose" }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [
        {
          id: "char-1",
          name: "Mira",
          type: "person",
          references: [],
          approvedReferences: [{ assetId: "ref-1", role: "hero", asset: { id: "ref-1", type: "image", displayName: "Mira ref" } }],
          looks: [],
          loras: [],
        },
      ],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      createImageJob,
      importAsset: vi.fn(),
      imageLocalJobs: [],
      rememberLocalGenerationJob: vi.fn(),
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [
        {
          id: "z_image_turbo",
          name: "Z-Image-Turbo",
          type: "image",
          ui: { poseLibrary: true, poseControlScale: true },
        },
      ],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const poseButton = [...document.body.querySelectorAll("button")].find((button) =>
      (button.getAttribute("aria-label") ?? "").includes("pose Standing 01"),
    );
    expect(poseButton).toBeTruthy();
    await act(async () => {
      poseButton.click();
    });

    // Strict tier exposes the pose-lock-strength slider (best-effort tiers don't).
    const slider = document.body.querySelector('input[aria-label="Pose lock strength"]');
    expect(slider).toBeTruthy();

    // Move it off the 1.0 default; the value must thread into advanced.controlScale.
    const valueSetter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(slider), "value").set;
    await act(async () => {
      valueSetter.call(slider, "0.75");
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const generateButton = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.startsWith("Generate") && button.textContent.includes("pose"),
    );
    expect(generateButton).toBeTruthy();
    await act(async () => {
      generateButton.click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "z_image_turbo",
        advanced: expect.objectContaining({
          poses: [{ id: "standing_01", keypoints: poseKeypoints }],
          controlScale: 0.75,
        }),
      }),
    );
  });

  it("threads a selected LoRA into the angle-set payload (sc-2223)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-angle-lora" }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      assets: [],
      characters: [
        {
          id: "char-1",
          name: "Mira",
          type: "person",
          references: [],
          approvedReferences: [{ assetId: "ref-1", role: "hero", asset: { id: "ref-1", type: "image", displayName: "Mira ref" } }],
          looks: [],
          loras: [],
        },
      ],
      createImageJob,
      importAsset: vi.fn(),
      imageLocalJobs: [],
      rememberLocalGenerationJob: vi.fn(),
      imageModels: [
        {
          id: "instantid_realvisxl",
          name: "InstantID (RealVisXL)",
          type: "image",
          family: "sdxl",
          ui: { viewAngles: [{ id: "front", label: "Front" }] },
        },
      ],
      latestImageAssets: [],
      // A family-matching LoRA the picker should surface + serialize into the payload.
      loras: [{ id: "kelsie-lora", name: "Kelsie", families: ["sdxl"], scope: "project" }],
      setPreviewAsset: () => {},
      removeCharacterReference: () => {},
      updateCharacter: () => {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    const loraCheckbox = document.body.querySelector(".character-lora-picker input[type='checkbox']");
    expect(loraCheckbox).toBeTruthy();
    await act(async () => {
      loraCheckbox.click();
    });

    const generateButton = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.startsWith("Generate angle set"),
    );
    await act(async () => {
      generateButton.click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        model: "instantid_realvisxl",
        loras: [expect.objectContaining({ id: "kelsie-lora", weight: 0.8 })],
      }),
    );
  });

  it("threads a selected LoRA into the pose-library payload (sc-2223)", async () => {
    const poseKeypoints = Array.from({ length: 18 }, (_, i) => [0.5, i / 18]);
    global.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        version: 1,
        categories: ["standing"],
        poses: [{ id: "standing_01", category: "standing", label: "Standing 01", preview: "poses/standing_01.png", keypoints: poseKeypoints }],
      }),
    }));
    const createImageJob = vi.fn(async () => ({ id: "job-pose-lora" }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      assets: [],
      characters: [
        {
          id: "char-1",
          name: "Mira",
          type: "person",
          references: [],
          approvedReferences: [{ assetId: "ref-1", role: "hero", asset: { id: "ref-1", type: "image", displayName: "Mira ref" } }],
          looks: [],
          loras: [],
        },
      ],
      createImageJob,
      importAsset: vi.fn(),
      imageLocalJobs: [],
      rememberLocalGenerationJob: vi.fn(),
      imageModels: [
        { id: "instantid_realvisxl", name: "InstantID (RealVisXL)", type: "image", family: "sdxl", ui: { poseLibrary: true } },
      ],
      latestImageAssets: [],
      loras: [{ id: "kelsie-lora", name: "Kelsie", families: ["sdxl"], scope: "project" }],
      setPreviewAsset: () => {},
      removeCharacterReference: () => {},
      updateCharacter: () => {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const poseButton = [...document.body.querySelectorAll("button")].find((button) =>
      (button.getAttribute("aria-label") ?? "").includes("pose Standing 01"),
    );
    await act(async () => {
      poseButton.click();
    });
    const loraCheckbox = document.body.querySelector(".character-lora-picker input[type='checkbox']");
    expect(loraCheckbox).toBeTruthy();
    await act(async () => {
      loraCheckbox.click();
    });

    const generateButton = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.startsWith("Generate") && button.textContent.includes("pose"),
    );
    await act(async () => {
      generateButton.click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        model: "instantid_realvisxl",
        loras: [expect.objectContaining({ id: "kelsie-lora", weight: 0.8 })],
      }),
    );
  });

  it("collects all character-associated assets in the Character Studio gallery (sc-2076)", async () => {
    const onPreview = vi.fn();
    const selectedCharacter = { id: "char-1", name: "Mira" };
    const assets = [
      { id: "a1", type: "image", displayName: "by recipe", recipe: { normalizedSettings: { characterId: "char-1" } } },
      { id: "a2", type: "image", displayName: "by reference", metadata: { characterReferences: [{ characterId: "char-1" }] } },
      { id: "a3", type: "image", displayName: "other character", recipe: { normalizedSettings: { characterId: "char-2" } } },
      { id: "a4", type: "image", displayName: "unassociated" },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(<CharacterAssets assets={assets} onPreview={onPreview} selectedCharacter={selectedCharacter} />);
    });

    // Counts only assets associated with this character (by recipe characterId or by
    // characterReferences) — not other characters' or unassociated assets.
    expect(container.textContent).toContain("Generated for Mira (2)");
    const previewButtons = [...document.body.querySelectorAll("button")].filter((button) =>
      (button.getAttribute("aria-label") ?? "").startsWith("Preview "),
    );
    expect(previewButtons).toHaveLength(2);
    await act(async () => {
      previewButtons[0].click();
    });
    expect(onPreview).toHaveBeenCalled();
  });

  it("opens the Import dialog from the Character Assets page (sc-6042)", async () => {
    const selectedCharacter = { id: "char-1", name: "Mira", references: [] };
    const assets = [{ id: "p-img1", type: "image", projectId: "project-1", displayName: "Project Image 1" }];
    root = createRoot(container);
    await act(async () => {
      root.render(
        <CharacterAssets
          addCharacterReference={vi.fn(async () => ({}))}
          assets={assets}
          importAsset={vi.fn()}
          onPreview={vi.fn()}
          projectId="project-1"
          selectedCharacter={selectedCharacter}
        />,
      );
    });

    const importButton = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Import");
    expect(importButton).toBeTruthy();
    await act(async () => {
      importButton.click();
    });
    expect(document.body.textContent).toContain("Import to Mira");
    expect(document.body.querySelector('[title="p-img1"]')).not.toBeNull();
  });

  it("bulk-discards selected media from the Character Assets toolbar", async () => {
    const deleteAsset = vi.fn(async () => {});
    const selectedCharacter = { id: "char-1", name: "Mira" };
    const assets = [
      {
        id: "ca-1",
        type: "image",
        projectId: "project-1",
        displayName: "Shot A",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: { trashed: false },
      },
      {
        id: "ca-2",
        type: "image",
        projectId: "project-1",
        displayName: "Shot B",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: { trashed: false },
      },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets,
            characters: [selectedCharacter],
            deleteAsset,
            addCharacterReference: vi.fn(),
            purgeAsset: () => {},
            updateAssetStatus: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            jobs: [],
            imageModels: [],
          },
          <CharacterAssets
            addCharacterReference={vi.fn()}
            assets={assets}
            deleteAsset={deleteAsset}
            onPreview={vi.fn()}
            projectId="project-1"
            purgeAsset={() => {}}
            selectedCharacter={selectedCharacter}
            updateAssetStatus={() => {}}
          />,
        ),
      );
    });
    await settle();

    // Both character images render as full-size selectable cards (matching the Assets page).
    expect(document.body.querySelectorAll(".character-asset-card")).toHaveLength(2);

    await act(async () => {
      document.body.querySelector('input[aria-label="Select Shot A"]').click();
    });
    await act(async () => {
      document.body.querySelector('input[aria-label="Select Shot B"]').click();
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll(".batch-selection-bar button")].find((button) => button.textContent === "Discard").click();
    });
    await settle();

    expect(deleteAsset).toHaveBeenCalledTimes(2);
    expect(deleteAsset).toHaveBeenCalledWith(assets[0]);
    expect(deleteAsset).toHaveBeenCalledWith(assets[1]);
    expect(document.body.querySelector(".batch-selection-bar")).toBeNull();
  });

  it("bulk-moves selected character media to the Main Asset Library (sc-8341)", async () => {
    const moveAssetToLibrary = vi.fn(async (asset) => asset);
    const selectedCharacter = { id: "char-1", name: "Mira" };
    const assets = [
      {
        id: "ca-1",
        type: "image",
        projectId: "project-1",
        displayName: "Shot A",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: { trashed: false },
      },
      {
        id: "ca-2",
        type: "image",
        projectId: "project-1",
        displayName: "Shot B",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: { trashed: false },
      },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets,
            characters: [selectedCharacter],
            deleteAsset: () => {},
            addCharacterReference: vi.fn(),
            moveAssetToLibrary,
            purgeAsset: () => {},
            updateAssetStatus: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            jobs: [],
            imageModels: [],
          },
          <CharacterAssets
            addCharacterReference={vi.fn()}
            assets={assets}
            deleteAsset={() => {}}
            onPreview={vi.fn()}
            projectId="project-1"
            purgeAsset={() => {}}
            selectedCharacter={selectedCharacter}
            updateAssetStatus={() => {}}
          />,
        ),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector('input[aria-label="Select Shot A"]').click();
    });
    await act(async () => {
      document.body.querySelector('input[aria-label="Select Shot B"]').click();
    });
    await settle();

    // Open the Move picker; "Assets Library" is offered as the first target.
    await act(async () => {
      [...document.body.querySelectorAll(".batch-selection-bar button")].find((button) => button.textContent === "Move").click();
    });
    await settle();
    const select = document.body.querySelector('select[aria-label="Move target"]');
    expect([...select.options].map((option) => option.textContent)).toContain("Assets Library");
    await changeField(select, "__sceneworks_library__");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent?.startsWith("Move 2 to library")).click();
    });
    await settle();

    expect(moveAssetToLibrary).toHaveBeenCalledTimes(2);
    expect(moveAssetToLibrary).toHaveBeenCalledWith(assets[0]);
    expect(moveAssetToLibrary).toHaveBeenCalledWith(assets[1]);
    expect(document.body.querySelector(".batch-selection-bar")).toBeNull();
  });

  it("bulk-moves selected character media into another character as a true move (sc-10200)", async () => {
    const moveAssetToCharacter = vi.fn(async (asset) => ({ ...asset, origin: "character_studio" }));
    const selectedCharacter = { id: "char-1", name: "Mira" };
    const otherCharacter = { id: "char-2", name: "Dax" };
    const assets = [
      {
        id: "ca-1",
        type: "image",
        projectId: "project-1",
        displayName: "Shot A",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: { trashed: false },
      },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets,
            characters: [selectedCharacter, otherCharacter],
            deleteAsset: () => {},
            moveAssetToCharacter,
            moveAssetToLibrary: vi.fn(),
            purgeAsset: () => {},
            updateAssetStatus: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            jobs: [],
            imageModels: [],
          },
          <CharacterAssets
            assets={assets}
            deleteAsset={() => {}}
            onPreview={vi.fn()}
            projectId="project-1"
            purgeAsset={() => {}}
            selectedCharacter={selectedCharacter}
            updateAssetStatus={() => {}}
          />,
        ),
      );
    });
    await settle();

    await act(async () => {
      document.body.querySelector('input[aria-label="Select Shot A"]').click();
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll(".batch-selection-bar button")].find((button) => button.textContent === "Move").click();
    });
    await settle();
    await changeField(document.body.querySelector('select[aria-label="Move target"]'), "char-2");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent?.startsWith("Move 1 to assets")).click();
    });
    await settle();

    // The move endpoint is hit per asset; no character reference (Approved-set
    // entry) is created for the target.
    expect(moveAssetToCharacter).toHaveBeenCalledTimes(1);
    expect(moveAssetToCharacter).toHaveBeenCalledWith(assets[0], "char-2");
    expect(document.body.querySelector(".batch-selection-bar")).toBeNull();
  });

  it("lists a character's associated datasets and opens one (sc-2022)", async () => {
    const onOpenDataset = vi.fn();
    const onCreateDataset = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <CharacterDatasets
          datasets={[
            { id: "ds-1", name: "Mira identity set", itemCount: 12, status: "ready", characterId: "char-1" },
          ]}
          imageCount={5}
          onCreateDataset={onCreateDataset}
          onOpenDataset={onOpenDataset}
          projectId="project-1"
          selectedCharacter={{ id: "char-1", name: "Mira" }}
        />,
      );
    });

    expect(container.textContent).toContain("For Mira (1)");
    const row = document.body.querySelector(".character-dataset-row");
    expect(row.textContent).toContain("Mira identity set");
    expect(row.textContent).toContain("12 images · ready");

    await act(async () => {
      [...row.querySelectorAll("button")].find((button) => button.textContent === "Open").click();
    });
    expect(onOpenDataset).toHaveBeenCalledWith("ds-1");

    // The create button reflects how many of the character's images would seed it.
    const createButton = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Create dataset from 5 images"),
    );
    await act(async () => {
      createButton.click();
    });
    expect(onCreateDataset).toHaveBeenCalled();
  });

  it("creates a dataset from a character's images and opens it (sc-2022)", async () => {
    const createTrainingDataset = vi.fn(async () => ({ id: "ds-new" }));
    const openDatasetInLibrary = vi.fn();
    const assets = [
      { id: "img-1", type: "image", displayName: "hero", recipe: { normalizedSettings: { characterId: "char-1" } } },
      { id: "img-2", type: "image", displayName: "ref", metadata: { characterReferences: [{ characterId: "char-1" }] } },
      { id: "img-3", type: "image", displayName: "other", recipe: { normalizedSettings: { characterId: "char-2" } } },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            createTrainingDataset,
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: [],
            loras: [],
            openDatasetInLibrary,
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            trainingDatasets: [],
            trainingDatasetsProjectId: "project-1",
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    // Two of three images belong to this character.
    const createButton = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Create dataset from 2 images"),
    );
    expect(createButton).toBeTruthy();
    await act(async () => {
      createButton.click();
    });

    expect(createTrainingDataset).toHaveBeenCalledWith(
      expect.objectContaining({
        characterId: "char-1",
        name: "Mira dataset",
        items: [{ assetId: "img-1" }, { assetId: "img-2" }],
      }),
    );
    expect(openDatasetInLibrary).toHaveBeenCalledWith("ds-new");
  });

  it("hides discarded character images from the grid and surfaces them in the Trashcan", async () => {
    const assets = [
      { id: "img-active", type: "image", displayName: "keep", recipe: { normalizedSettings: { characterId: "char-1" } } },
      {
        id: "img-trashed",
        type: "image",
        displayName: "discarded",
        status: { trashed: true },
        recipe: { normalizedSettings: { characterId: "char-1" } },
      },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: assets,
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    // The active toggle counts only non-trashed images for this character.
    const showButton = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Show this character's images (1)"),
    );
    expect(showButton).toBeTruthy();
    await act(async () => {
      showButton.click();
    });

    // Active grid shows the kept image, not the discarded one.
    expect(document.body.querySelectorAll(".review-grid .review-card").length).toBe(1);
    expect(document.body.querySelectorAll(".review-grid .review-card.trashed").length).toBe(0);

    // Switch to the Trashcan view and the discarded image becomes reachable.
    // Scope to the Sample-outputs panel, since CharacterAssets also has a toggle.
    const testPanel = document.body.querySelector(".test-character-panel");
    const trashButton = [...testPanel.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Trashcan (1)"),
    );
    expect(trashButton).toBeTruthy();
    await act(async () => {
      trashButton.click();
    });
    expect(testPanel.querySelectorAll(".review-grid .review-card.trashed").length).toBe(1);
    expect([...testPanel.querySelectorAll(".review-grid button")].some((button) => button.textContent === "Purge")).toBe(true);
  });

  it("hides discarded images from the Character assets grid and exposes Trashcan restore/purge", async () => {
    const purgeAsset = vi.fn();
    const updateAssetStatus = vi.fn();
    const assets = [
      { id: "img-active", type: "image", displayName: "keep", recipe: { normalizedSettings: { characterId: "char-1" } } },
      {
        id: "img-trashed",
        type: "image",
        displayName: "discarded",
        status: { trashed: true },
        recipe: { normalizedSettings: { characterId: "char-1" } },
      },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: assets,
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset,
            removeCharacterReference: () => {},
            updateAssetStatus,
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    // The Character assets section renders thumbnails; the discarded one is hidden.
    const findSection = () =>
      [...document.body.querySelectorAll(".character-section")].find((section) =>
        section.querySelector(".eyebrow")?.textContent === "Character assets",
      );
    const section = findSection();
    expect(section).toBeTruthy();
    expect(section.querySelectorAll(".character-asset-card").length).toBe(1);
    // Heading count reflects active images only.
    expect(section.querySelector("h2").textContent).toContain("(1)");

    // Switch to the Trashcan and the discarded image surfaces with restore/purge.
    const trashButton = [...section.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Trashcan (1)"),
    );
    expect(trashButton).toBeTruthy();
    await act(async () => {
      trashButton.click();
    });
    const trashSection = findSection();
    expect(trashSection.querySelectorAll(".character-asset-card").length).toBe(1);
    const restore = [...trashSection.querySelectorAll("button")].find((button) => button.textContent === "Restore");
    const purge = [...trashSection.querySelectorAll("button")].find((button) => button.textContent === "Purge");
    expect(restore).toBeTruthy();
    expect(purge).toBeTruthy();
    await act(async () => {
      purge.click();
    });
    expect(purgeAsset).toHaveBeenCalledWith(expect.objectContaining({ id: "img-trashed" }));
  });

  it("Empty Trash purges all discarded images for the character and only in the Trashcan view", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const purgeAsset = vi.fn();
    const assets = [
      { id: "img-active", type: "image", displayName: "keep", recipe: { normalizedSettings: { characterId: "char-1" } } },
      {
        id: "img-trash-1",
        type: "image",
        status: { trashed: true },
        recipe: { normalizedSettings: { characterId: "char-1" } },
      },
      {
        id: "img-trash-2",
        type: "image",
        status: { trashed: true },
        recipe: { normalizedSettings: { characterId: "char-1" } },
      },
      // Belongs to another character — must never be purged by this view.
      { id: "img-other", type: "image", status: { trashed: true }, recipe: { normalizedSettings: { characterId: "char-2" } } },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: assets,
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset,
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    const findSection = () =>
      [...document.body.querySelectorAll(".character-section")].find((section) =>
        section.querySelector(".eyebrow")?.textContent === "Character assets",
      );
    // No Empty Trash in the active Images view.
    expect([...findSection().querySelectorAll("button")].some((button) => button.textContent.startsWith("Empty Trash"))).toBe(false);

    await act(async () => {
      [...findSection().querySelectorAll("button")].find((button) => button.textContent.includes("Trashcan (2)")).click();
    });
    const emptyButton = [...findSection().querySelectorAll("button")].find((button) => button.textContent.startsWith("Empty Trash"));
    expect(emptyButton).toBeTruthy();
    expect(emptyButton.textContent).toContain("(2)");

    await act(async () => {
      emptyButton.click();
    });
    expect(purgeAsset).toHaveBeenCalledTimes(2);
    expect(purgeAsset).toHaveBeenCalledWith(expect.objectContaining({ id: "img-trash-1" }));
    expect(purgeAsset).toHaveBeenCalledWith(expect.objectContaining({ id: "img-trash-2" }));
    expect(purgeAsset).not.toHaveBeenCalledWith(expect.objectContaining({ id: "img-other" }));
    confirm.mockRestore();
  });

  it("launches reference-based generation from an approved character reference", async () => {
    const sendCharacterToImage = vi.fn();
    const reference = { assetId: "ref-1", approved: true, asset: { id: "ref-1", type: "image", displayName: "Mira ref" } };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets: [],
            attachCharacterLora: () => {},
            characters: [
              { id: "char-1", name: "Mira", type: "person", references: [reference], approvedReferences: [reference], looks: [], loras: [] },
            ],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage,
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate variations").click();
    });

    expect(sendCharacterToImage).toHaveBeenCalledWith(expect.objectContaining({ id: "char-1" }), null, "ref-1");
  });

  it("keeps the shell usable when presets are unavailable", async () => {
    global.fetch.mockImplementation((url) => {
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
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/recipe-presets")) {
        return Promise.resolve(errorResponse(404, "Not Found"));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    expect(container.textContent).toContain("Library");
    expect(container.textContent).toContain("Assets");
    expect(container.textContent).toContain("Project One");
    expect(container.textContent).not.toContain("Not Found");
  });

  it("does not show a stale timeline lookup error after creating a workspace", async () => {
    const requests = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      requests.push({ method: options.method ?? "GET", path });
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects") && options.method === "POST") {
        return Promise.resolve(response({ id: "project-2", name: "Fresh Workspace" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/projects/project-1/timelines/timeline-1")) {
        return Promise.resolve(
          response({
            id: "timeline-1",
            projectId: "project-1",
            name: "Main timeline",
            aspectRatio: "16:9",
            width: 1280,
            height: 720,
            fps: 30,
            duration: 0,
            tracks: [],
            transitions: [],
          }),
        );
      }
      if (path.endsWith("/projects/project-1/timelines")) {
        return Promise.resolve(
          response([
            {
              id: "timeline-1",
              name: "Main timeline",
              filePath: "timelines/main.sceneworks.timeline.json",
              aspectRatio: "16:9",
              width: 1280,
              height: 720,
              fps: 30,
              duration: 0,
              createdAt: "2026-05-19T12:00:00Z",
              updatedAt: "2026-05-19T12:00:00Z",
            },
          ]),
        );
      }
      if (path.endsWith("/projects/project-2/timelines/timeline-1")) {
        return Promise.resolve(errorResponse(404, "Timeline not found"));
      }
      if (path.endsWith("/projects/project-2/timelines")) {
        return Promise.resolve(response([]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      document.body.querySelector(".project-pill").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "New workspace").click();
    });
    await changeField(document.body.querySelector('[aria-label="New workspace name"]'), "Fresh Workspace");
    await act(async () => {
      [...document.body.querySelectorAll(".project-menu-create button")].find((button) => button.textContent === "Create").click();
    });
    await settle();

    expect(requests.some((request) => request.path.endsWith("/projects/project-2/timelines/timeline-1"))).toBe(false);
    expect(container.textContent).toContain("Fresh Workspace");
    expect(container.textContent).not.toContain("Timeline not found");
  });

});
