import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useVideoReferences } from "./useVideoReferences.js";

// Restore-time validation (sc-11964) drops restored asset ids the catalog does not hold. On a
// route entered from the Asset Library the shell first holds only the Library slice and then
// widens it, so a NON-EMPTY catalog is not yet the whole catalog: validation must wait for
// `assetCatalogComplete`, or a restored Character Studio source is discarded before it lands.

let root;
let container;
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

function Probe(props) {
  useVideoReferences(props);
  return null;
}

function stateSetter(initial) {
  const cell = { value: initial };
  cell.set = vi.fn((update) => {
    cell.value = typeof update === "function" ? update(cell.value) : update;
  });
  return cell;
}

function render(source, { assets, assetCatalogComplete }) {
  const noop = () => {};
  return act(async () => root.render(
    <Probe
      hasRestoredSource
      selectedAssetId={null}
      selectedAsset={null}
      setSourceAssetOrigin={noop}
      setSourceAssetId={source.set}
      setSourceClipAssetId={noop}
      launchRequest={null}
      setMode={noop}
      setModel={noop}
      setSelectedPresetId={noop}
      setCharacterId={noop}
      setCharacterLookId={noop}
      assets={assets}
      assetCatalogComplete={assetCatalogComplete}
      setLastFrameAssetId={noop}
      setBridgeRightClipAssetId={noop}
      setReferenceClipAssetId={noop}
      setReferenceAssetIds={noop}
      setSourceClipAssetIds={noop}
      personTracks={[]}
      setPersonTrackId={noop}
      selectedModel={null}
      setDuration={noop}
      setResolution={noop}
      setFps={noop}
      sourceAssetId={source.value}
      lastFrameAssetId=""
      mode="image_to_video"
    />,
  ));
}

const library = { id: "library-1", type: "image", origin: "image_studio" };
const portrait = { id: "portrait-1", type: "image", origin: "character_studio" };

it("keeps a restored Character Studio source while the Library slice is being widened", async () => {
  const source = stateSetter("portrait-1");
  await render(source, { assets: [library], assetCatalogComplete: false });
  expect(source.value).toBe("portrait-1");

  await render(source, { assets: [library, portrait], assetCatalogComplete: true });
  expect(source.value).toBe("portrait-1");
});

it("still drops a restored source that the whole catalog does not hold", async () => {
  const source = stateSetter("purged-1");
  await render(source, { assets: [library], assetCatalogComplete: false });
  expect(source.value).toBe("purged-1");

  await render(source, { assets: [library, portrait], assetCatalogComplete: true });
  expect(source.value).toBe("");
});
