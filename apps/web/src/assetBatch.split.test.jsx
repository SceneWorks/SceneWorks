import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import { useAssetBatch } from "./assetBatch.jsx";
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
