import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import { useVideoRecipeReplay } from "./useVideoRecipeReplay.js";

let root;
let container;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
});
afterEach(() => unmountRoot(root, container));

// Every setter is a spy; only the data the replay effect reads is real.
function replayParams(launchRequest, setFlashAttn) {
  const fixed = {
    launchRequest,
    setFlashAttn,
    macVideoModels: [],
    videoModels: [],
    model: "m",
    lastI2vAssetIdRef: { current: null },
  };
  return new Proxy(fixed, { get: (target, key) => (key in target ? target[key] : vi.fn()) });
}
function Harness({ params }) {
  useVideoRecipeReplay(params);
  return null;
}
let launchSeq = 0;
function replay(rawAdapterSettings) {
  const setFlashAttn = vi.fn();
  const launchRequest = { id: `r${++launchSeq}`, view: "Video", recipe: { model: "m", rawAdapterSettings } };
  act(() => root.render(<Harness params={replayParams(launchRequest, setFlashAttn)} />));
  return setFlashAttn;
}

describe("video recipe replay: flashAttn (SW-7276)", () => {
  it("restores an explicit opt-out", () => {
    expect(replay({ flashAttn: false })).toHaveBeenCalledWith(false);
  });
  it("resets a manually unchecked box to on for a default recipe (absent means true)", () => {
    expect(replay({})).toHaveBeenCalledWith(true);
    expect(replay({ flashAttn: true })).toHaveBeenCalledWith(true);
  });
});
