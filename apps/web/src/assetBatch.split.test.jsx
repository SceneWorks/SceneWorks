import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AssetSelectionBar, useAssetBatch } from "./assetBatch.jsx";
import { AppStaticContext, AppLiveContext, AppContext } from "./context/AppContext.js";

// sc-9751 (F-052 follow-up): the App renders the split AppStaticContext + AppLiveContext
// providers, NOT the legacy combined <AppContext.Provider>. useAssetBatch previously read
// useContext(AppContext) directly, which resolves to null under the split tree and blanked
// the batch toolbar (empty assets/characters, no delete/move actions) in the real app —
// while unit tests that render a single <AppContext.Provider> kept passing. These tests
// pin the fix: the hook reads the merged view across BOTH split providers, and still
// degrades to inert defaults when rendered with no provider at all.

const assets = [
  { id: "a1", kind: "image", file: { path: "a1.png" } },
  { id: "a2", kind: "image", file: { path: "a2.png" } },
];
const characters = [
  { id: "c1", name: "Ada", archived: false },
  { id: "c2", name: "Grace", archived: true },
];

// Probe: run the hook, select an asset, and surface the values a real toolbar would show.
function AssetBatchProbe({ onReady }) {
  const batch = useAssetBatch();
  React.useEffect(() => {
    batch.toggleSelect("a1");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  onReady(batch);
  return null;
}

function render(ui) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(ui);
  });
  return {
    cleanup() {
      act(() => root.unmount());
      container.remove();
    },
  };
}

describe("useAssetBatch under the split context providers (sc-9751)", () => {
  let harness;

  afterEach(() => {
    harness?.cleanup();
    harness = null;
  });

  it("reads assets/characters through the split AppStatic + AppLive providers", () => {
    let latest;
    const staticValue = { assets, characters, imageModels: [] };
    const liveValue = { jobs: [] };
    harness = render(
      <AppStaticContext.Provider value={staticValue}>
        <AppLiveContext.Provider value={liveValue}>
          <AssetBatchProbe onReady={(b) => { latest = b; }} />
        </AppLiveContext.Provider>
      </AppStaticContext.Provider>,
    );
    // Selecting a1 must surface the matching asset — proving the hook read `assets` from
    // the split static context rather than falling back to the empty [] default.
    expect(latest.selectedAssetList.map((a) => a.id)).toEqual(["a1"]);
    // Only the non-archived character is a valid move target — proving `characters` was read.
    expect(latest.availableCharacters.map((c) => c.id)).toEqual(["c1"]);
  });

  it("still degrades to inert defaults when rendered with no provider", () => {
    let latest;
    harness = render(<AssetBatchProbe onReady={(b) => { latest = b; }} />);
    // No provider at all: the toolbar stays inert (no crash), empty selection/derived lists.
    expect(latest.selectedAssetList).toEqual([]);
    expect(latest.availableCharacters).toEqual([]);
  });

  it("remains backward-compatible with a lone legacy <AppContext.Provider> (tests)", () => {
    let latest;
    harness = render(
      <AppContext.Provider value={{ assets, characters, imageModels: [], jobs: [] }}>
        <AssetBatchProbe onReady={(b) => { latest = b; }} />
      </AppContext.Provider>,
    );
    expect(latest.selectedAssetList.map((a) => a.id)).toEqual(["a1"]);
    expect(latest.availableCharacters.map((c) => c.id)).toEqual(["c1"]);
  });
});

// The Assets Library folds an upscale + its source original into one tile keyed by the
// UPSCALED asset id. A bulk Discard on that tile trashes only the upscaled variant and
// orphans the source, so discardSelected confirms first whether to also discard the
// originals. These pin the three outcomes: both / upscaled-only / cancel.
describe("useAssetBatch discard-upscaled confirmation", () => {
  let harness;

  afterEach(() => {
    harness?.cleanup();
    harness = null;
  });

  const original = { id: "orig", kind: "image", projectId: "p", file: { path: "orig.png" }, extra: {} };
  const upscaled = {
    id: "up",
    kind: "image",
    projectId: "p",
    file: { path: "up.png" },
    extra: { isUpscaled: true, upscaledFromAssetId: "orig" },
  };
  const plain = { id: "plain", kind: "image", projectId: "p", file: { path: "plain.png" }, extra: {} };

  // Mount the hook with a deleteAsset spy; drive it directly through the returned API.
  function mount(assetList) {
    const deleted = [];
    let latest;
    const deleteAsset = (asset) => {
      deleted.push(asset.id);
      return Promise.resolve();
    };
    harness = render(
      <AppContext.Provider value={{ assets: assetList, characters: [], imageModels: [], jobs: [], deleteAsset }}>
        <AssetBatchProbe onReady={(b) => { latest = b; }} />
      </AppContext.Provider>,
    );
    return { deleted, batch: () => latest };
  }

  // Overwrite the shared probe's default "select a1" with an explicit selection.
  async function select(batch, id) {
    await act(async () => {
      batch().clearSelection();
      batch().toggleSelect(id);
    });
  }

  const flush = () =>
    act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

  it("prompts (without deleting) when a selected tile is an upscale with a present source", async () => {
    const { deleted, batch } = mount([original, upscaled]);
    await select(batch, "up");
    await act(async () => {
      await batch().discardSelected();
    });
    expect(batch().discardPrompt).toBeTruthy();
    expect(batch().discardPrompt.sources.map((a) => a.id)).toEqual(["orig"]);
    expect(deleted).toEqual([]);
  });

  it("discards both the upscaled tile and its source original on 'Discard both'", async () => {
    const { deleted, batch } = mount([original, upscaled]);
    await select(batch, "up");
    await act(async () => {
      await batch().discardSelected();
    });
    await act(async () => {
      batch().resolveDiscardPrompt(true);
    });
    await flush();
    expect([...deleted].sort()).toEqual(["orig", "up"]);
    expect(batch().discardPrompt).toBeNull();
  });

  it("discards only the upscaled tile on 'Only the upscaled'", async () => {
    const { deleted, batch } = mount([original, upscaled]);
    await select(batch, "up");
    await act(async () => {
      await batch().discardSelected();
    });
    await act(async () => {
      batch().resolveDiscardPrompt(false);
    });
    await flush();
    expect(deleted).toEqual(["up"]);
    expect(batch().discardPrompt).toBeNull();
  });

  it("discards immediately without a prompt when no upscale/source pair is selected", async () => {
    const { deleted, batch } = mount([plain]);
    await select(batch, "plain");
    await act(async () => {
      await batch().discardSelected();
    });
    expect(batch().discardPrompt).toBeNull();
    expect(deleted).toEqual(["plain"]);
  });

  it("does not prompt for a source original that is already trashed (deleteAsset just flags it)", async () => {
    // The source lingers in `assets` with status.trashed=true; there's nothing new to discard.
    const trashedOriginal = { ...original, status: { trashed: true } };
    const { deleted, batch } = mount([trashedOriginal, upscaled]);
    await select(batch, "up");
    await act(async () => {
      await batch().discardSelected();
    });
    expect(batch().discardPrompt).toBeNull();
    expect(deleted).toEqual(["up"]);
  });

  // Presentational check: AssetSelectionBar actually renders the dialog (Modal portals to
  // document.body), with pluralized copy and the three wired buttons.
  it("renders the confirmation dialog with three wired actions and pluralized copy", () => {
    const resolveDiscardPrompt = vi.fn();
    const cancelDiscard = vi.fn();
    const fakeBatch = {
      selectedAssetIds: new Set(["up1", "up2"]),
      eligibleSelected: [],
      movableSelected: [],
      availableCharacters: [],
      setBatchOpen() {},
      bulkAction: null,
      discardSelected() {},
      discardPrompt: { targets: [upscaled], sources: [original, { ...original, id: "orig2" }] },
      resolveDiscardPrompt,
      cancelDiscard,
      moveOpen: false,
      setMoveOpen() {},
      moveCharacterId: "",
      setMoveCharacterId() {},
      moveSelectedToCharacter() {},
      clearSelection() {},
    };
    harness = render(<AssetSelectionBar batch={fakeBatch} />);

    const modal = document.body.querySelector(".discard-confirm-modal");
    expect(modal).toBeTruthy();
    // Two source originals → plural copy.
    expect(modal.querySelector(".discard-confirm-title").textContent).toContain("Discard source images too?");
    expect(modal.querySelector(".discard-confirm-body").textContent).toContain("2 source originals");

    const buttons = [...modal.querySelectorAll(".discard-confirm-actions button")];
    expect(buttons.map((b) => b.textContent)).toEqual(["Cancel", "Only the upscaled", "Discard both"]);

    act(() => buttons[2].dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(resolveDiscardPrompt).toHaveBeenCalledWith(true);
    act(() => buttons[1].dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(resolveDiscardPrompt).toHaveBeenCalledWith(false);
    act(() => buttons[0].dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(cancelDiscard).toHaveBeenCalled();
  });
});
