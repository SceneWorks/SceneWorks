// A hybrid MiniMax-H3 import serves both partitions, but a render runs one: a `reference_to_video`
// render the reference partition, every other mode the base one — and the API refuses the other
// partition's turbo adapter for that request. These pin the studio half: the turbo hook offers and
// defaults the adapter for the partition the MODE runs, and a mode switch swaps an active turbo
// adapter rather than leaving one the API would refuse (or silently dropping turbo).
//
// The harness reproduces the shared studio wiring the hook sits under: `compatibleLoras` comes from
// `loraMatchesModel` against `loraModelForMode`, and the same prune `useGenerationStudio` runs.
// LoRA fixtures are the SHIPPED `builtin.loras.jsonc` entries, so their `modelIds` are the real ones.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act, useEffect, useMemo, useState } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import { loraMatchesModel, loraModelForMode } from "../../presetUtils.js";
import { useVideoTurboControls } from "./useVideoTurboControls.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const LORAS = JSON5.parse(
  readFileSync(resolve(HERE, "../../../../../config/manifests/builtin.loras.jsonc"), "utf8"),
).loras;
const shippedLora = (id) => {
  const entry = LORAS.find((lora) => lora.id === id);
  if (!entry) throw new Error(`manifest has no lora ${id}`);
  return { ...entry, scope: "builtin", installState: "installed" };
};
const TURBO_768P = shippedLora("minimax_h3_turbo_4step_768p");
const TURBO_REF2V = shippedLora("minimax_h3_ref2v_turbo_4step");
const CATALOG = [TURBO_768P, TURBO_REF2V];

// The API-served row of a verified hybrid import (`adapterPartitionModels`, `supportsMinimaxH3Turbo`).
const HYBRID = {
  id: "my_h3_hybrid",
  family: "minimax-h3",
  loraCompatibility: { families: ["minimax-h3"] },
  supportsMinimaxH3Turbo: true,
  adapterPartitionModels: ["minimax_h3", "minimax_h3_ref"],
};

let root;
let container;
let latest;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
});
afterEach(() => unmountRoot(root, container));

function Harness({ mode, initialSelectedLoraIds = [] }) {
  const [selectedLoraIds, setSelectedLoraIds] = useState(initialSelectedLoraIds);
  const loraModel = useMemo(() => loraModelForMode(HYBRID, mode), [mode]);
  const compatibleLoras = useMemo(() => CATALOG.filter((lora) => loraMatchesModel(lora, loraModel)), [loraModel]);
  // `useGenerationStudio`'s prune: drop selections that fall out of the compatible set.
  useEffect(() => {
    setSelectedLoraIds((ids) => ids.filter((id) => compatibleLoras.some((lora) => lora.id === id)));
  }, [compatibleLoras]);
  const turbo = useVideoTurboControls({
    saved: {},
    selectedModel: HYBRID,
    compatibleLoras,
    selectedLoraIds,
    model: HYBRID.id,
    mode,
    setSelectedLoraIds,
  });
  latest = { selectedLoraIds, setSelectedLoraIds, ...turbo };
  return null;
}
const render = (props) => act(() => root.render(<Harness {...props} />));

describe("useVideoTurboControls on a hybrid MiniMax-H3 import", () => {
  it("defaults a reference render to the ref2v adapter even with both installed", () => {
    render({ mode: "reference_to_video" });
    expect(latest.turboVariants.map((variant) => variant.id)).toEqual([TURBO_REF2V.id]);
    expect(latest.selectedLoraIds).toEqual([TURBO_REF2V.id]);
  });

  it("swaps the active turbo adapter when the mode crosses partitions", () => {
    render({ mode: "text_to_video" });
    expect(latest.selectedLoraIds).toEqual([TURBO_768P.id]);
    render({ mode: "reference_to_video" });
    expect(latest.selectedLoraIds).toEqual([TURBO_REF2V.id]);
    expect(latest.activeTurboVariant?.id).toBe(TURBO_REF2V.id);
    render({ mode: "first_last_frame" });
    expect(latest.selectedLoraIds).toEqual([TURBO_768P.id]);
  });

  it("keeps turbo off across a partition switch once the user turned it off", () => {
    render({ mode: "text_to_video" });
    act(() => latest.selectTurboVariant(null));
    expect(latest.selectedLoraIds).toEqual([]);
    render({ mode: "reference_to_video" });
    expect(latest.selectedLoraIds).toEqual([]);
  });

  it("does not swap within one partition", () => {
    render({ mode: "text_to_video" });
    render({ mode: "image_to_video" });
    expect(latest.selectedLoraIds).toEqual([TURBO_768P.id]);
  });
});
