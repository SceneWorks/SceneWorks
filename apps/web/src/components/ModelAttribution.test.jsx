// sc-17161 — licence-required attribution on the GENERATION surfaces.
//
// sc-17227 landed `ui.attribution` on the Models card, which discharges MiniMax H3 §IV.2 on
// exactly one screen a user may never revisit after installing.
// `docs/minimax-h3-use-restriction-safeguards.md` S4 names the generation surfaces — Video Studio,
// the Simple video studio, the queue and the asset detail — as the rest of the same obligation.
//
// Two things are being pinned, and they are different claims: that the string RENDERS on those
// surfaces, and that it comes from the manifest field rather than a second hard-coded copy. The
// second is the one that keeps it correct: §IV.2 requires the exact string "MiniMax H3" (a space,
// not the hyphenated product name), so a literal typed into a component is a copy that can drift
// away from the licence it exists to satisfy.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import { createRoot } from "react-dom/client";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { ModelAttribution } from "./ModelAttribution.jsx";
import { AssetDetail } from "./assetPanels.jsx";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../../config/manifests/builtin.models.jsonc");
const manifestModels = (() => {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  return Array.isArray(parsed) ? parsed : parsed.models;
})();
const MINIMAX = manifestModels.find((model) => model.id === "minimax_h3");
const MINIMAX_REF = manifestModels.find((model) => model.id === "minimax_h3_ref");
const LTX = manifestModels.find((model) => model.id === "ltx_2_3");

describe("ModelAttribution (sc-17161)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
  });

  function render(ui) {
    root = createRoot(container);
    act(() => root.render(ui));
  }

  it("renders the manifest string for a resolved model and nothing for one without", () => {
    render(<ModelAttribution model={MINIMAX} />);
    expect(container.textContent).toBe(MINIMAX.ui.attribution);
    // §IV.2 requires this exact substring — a space, not the hyphenated repo name.
    expect(container.textContent).toContain("MiniMax H3");
    expect(container.textContent).not.toBe("MiniMax-H3");

    act(() => root.render(<ModelAttribution model={LTX} />));
    expect(LTX.ui.attribution).toBeUndefined();
    expect(container.textContent).toBe("");
  });

  it("resolves a bare model id against the catalog on the app context", () => {
    // The queue and the asset detail hold only an id (`job.payload.model`, `asset.recipe.model`)
    // and never join it back to the catalog, so this lookup is the whole mechanism there.
    render(
      <AppContext.Provider value={{ models: [MINIMAX, MINIMAX_REF, LTX] }}>
        <ModelAttribution modelId="minimax_h3_ref" />
      </AppContext.Provider>,
    );
    expect(container.textContent).toBe(MINIMAX_REF.ui.attribution);
  });

  it("renders nothing for an unknown id, an empty id, or no provider at all", () => {
    render(
      <AppContext.Provider value={{ models: [MINIMAX] }}>
        <ModelAttribution modelId="a_model_that_was_uninstalled" />
      </AppContext.Provider>,
    );
    expect(container.textContent).toBe("");

    act(() =>
      root.render(
        <AppContext.Provider value={{ models: [MINIMAX] }}>
          <ModelAttribution modelId="" />
        </AppContext.Provider>,
      ),
    );
    expect(container.textContent).toBe("");

    // Rendered outside the providers (a unit test of a host component) must not throw.
    act(() => root.render(<ModelAttribution modelId="minimax_h3" />));
    expect(container.textContent).toBe("");
  });

  it("carries the attribution into the asset detail's Model row", () => {
    const asset = {
      id: "asset_1",
      type: "video",
      displayName: "clip",
      recipe: { model: "minimax_h3" },
      file: { path: "a.mp4", duration: 5.1667 },
    };
    render(
      <AppContext.Provider value={{ models: [MINIMAX, LTX] }}>
        <AssetDetail asset={asset} />
      </AppContext.Provider>,
    );
    const line = container.querySelector(".model-attribution");
    expect(line, "the asset detail is a generation surface").toBeTruthy();
    expect(line.textContent).toBe(MINIMAX.ui.attribution);

    // Per-model, never a global footer: a render from a model that declares none carries none.
    act(() =>
      root.render(
        <AppContext.Provider value={{ models: [MINIMAX, LTX] }}>
          <AssetDetail asset={{ ...asset, recipe: { model: "ltx_2_3" } }} />
        </AppContext.Provider>,
      ),
    );
    expect(container.querySelector(".model-attribution")).toBeNull();
  });

  it("keeps the manifest and the offline mirror agreeing on the string", async () => {
    // `fallbackModels` is a MIRROR, and the generation surfaces render whichever catalog is in
    // hand — the fallback one is what they get before /api/v1/models answers. A licence obligation
    // discharged only once the network responds is not discharged.
    const { fallbackModels } = await import("../constants.js");
    for (const entry of [MINIMAX, MINIMAX_REF]) {
      const mirrored = fallbackModels.find((model) => model.id === entry.id);
      expect(mirrored, `${entry.id} must be mirrored`).toBeTruthy();
      expect(mirrored.ui?.attribution).toBe(entry.ui.attribution);
    }
  });

  it("pins every manifest entry that declares an attribution", () => {
    // A new entry with a licence-required string must be a conscious addition, not a silent one:
    // this list is what the generation surfaces will start displaying.
    const declaring = manifestModels
      .filter((model) => model.ui?.attribution)
      .map((model) => model.id)
      .sort();
    expect(declaring).toEqual(["minimax_h3", "minimax_h3_ref"]);
  });
});
