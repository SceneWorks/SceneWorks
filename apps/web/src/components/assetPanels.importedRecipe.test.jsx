// "Use this recipe" on an IMPORTED asset carrying an embedded workflow (sc-15952, epic 15945).
//
// The acceptance criterion is two-sided — "appears on imported assets carrying a workflow, and
// NOWHERE ELSE" — and the second half is the one a UI silently gets wrong, so most of this file is
// negative. A generated asset must keep exactly the one button it has always had; an imported image
// with no chunk in it (every foreign PNG in the world) must gain nothing at all.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../assetActions.js", () => ({ saveAssetAs: vi.fn(), revealAsset: vi.fn() }));
vi.mock("../runtime.js", () => ({
  isDesktop: false,
  tauriInvoke: vi.fn(),
  setViewerFullscreen: vi.fn(() => Promise.resolve()),
  isPageFullscreen: () => false,
}));

import { FullscreenPreview } from "./assetPanels.jsx";

const noop = () => {};

// The envelope shape sc-15949 records under `extra.importedWorkflow`.
const IMPORTED_WORKFLOW = {
  sceneworksWorkflow: "image",
  schemaVersion: 1,
  mode: "text_to_image",
  model: "krea_2_turbo",
  prompt: "a lighthouse in heavy fog",
};

function asset(overrides = {}) {
  return {
    id: "asset-img",
    projectId: "project-1",
    displayName: "Plate",
    type: "image",
    status: {},
    file: { path: "assets/images/plate.png", mimeType: "image/png" },
    ...overrides,
  };
}

describe("FullscreenPreview — the imported-workflow recipe action", () => {
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

  const render = async (props) =>
    act(async () =>
      root.render(
        <FullscreenPreview
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
          {...props}
        />,
      ),
    );

  // The Modal portals to <body>.
  const recipeButtons = () =>
    [...document.querySelectorAll("button")].filter(
      (node) => node.textContent.trim() === "Use this recipe",
    );

  it("offers the recipe an imported image is carrying", async () => {
    const onUseImportedRecipe = vi.fn();
    await render({
      asset: asset({ extra: { importedWorkflow: IMPORTED_WORKFLOW } }),
      onUseImportedRecipe,
      onUseRecipe: vi.fn(),
    });
    expect(recipeButtons()).toHaveLength(1);
    await act(async () =>
      recipeButtons()[0].dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    expect(onUseImportedRecipe).toHaveBeenCalledWith(
      expect.objectContaining({ id: "asset-img" }),
    );
  });

  it("offers nothing for an imported image with no workflow in it", async () => {
    // The common case: every PNG that did not come from SceneWorks. `extra` exists and the key
    // under it does not, so a truthiness check on `extra` would have shipped this button on every
    // imported image in the library.
    await render({
      asset: asset({ extra: { source: "upload" } }),
      onUseImportedRecipe: vi.fn(),
      onUseRecipe: vi.fn(),
    });
    expect(recipeButtons()).toHaveLength(0);
  });

  it("offers nothing for an asset with no extra at all", async () => {
    await render({ asset: asset(), onUseImportedRecipe: vi.fn(), onUseRecipe: vi.fn() });
    expect(recipeButtons()).toHaveLength(0);
  });

  it("leaves a generated asset with exactly the one button it has always had", async () => {
    const onUseRecipe = vi.fn();
    const onUseImportedRecipe = vi.fn();
    await render({
      asset: asset({ recipe: { mode: "text_to_image", prompt: "a lighthouse", seed: 42 } }),
      onUseImportedRecipe,
      onUseRecipe,
    });
    expect(recipeButtons()).toHaveLength(1);
    await act(async () =>
      recipeButtons()[0].dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    expect(onUseRecipe).toHaveBeenCalled();
    expect(onUseImportedRecipe).not.toHaveBeenCalled();
  });

  it("never doubles the button when an asset somehow has both", async () => {
    // Two identical buttons pointing at two different recipes, with nothing on screen saying which
    // is which. The recorded one wins: it was made by THIS install and carries this image's seed.
    const onUseRecipe = vi.fn();
    const onUseImportedRecipe = vi.fn();
    await render({
      asset: asset({
        recipe: { mode: "text_to_image", prompt: "a lighthouse", seed: 42 },
        extra: { importedWorkflow: IMPORTED_WORKFLOW },
      }),
      onUseImportedRecipe,
      onUseRecipe,
    });
    expect(recipeButtons()).toHaveLength(1);
    await act(async () =>
      recipeButtons()[0].dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    expect(onUseRecipe).toHaveBeenCalled();
    expect(onUseImportedRecipe).not.toHaveBeenCalled();
  });

  it("offers nothing when the app passed no handler for it", async () => {
    await render({ asset: asset({ extra: { importedWorkflow: IMPORTED_WORKFLOW } }) });
    expect(recipeButtons()).toHaveLength(0);
  });
});
