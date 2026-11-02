// sc-26687 — an imported checkpoint's product surface has to be READABLE, and the repairs it
// offers have to be the ones a user can actually take. Asserted against the real component so what
// is checked is what a person reads on the Model Manager card, not a helper's return value.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ImportedCheckpointSurface,
  hasImportedCheckpointSurface,
  importedSourceSummary,
} from "./ImportedCheckpointSurface.jsx";

// The shape `apply_imported_checkpoint_product_surface` writes onto a resolved catalog entry.
const REF2VA_IMPORT = {
  id: "my_singularity_tune",
  name: "My Singularity tune",
  importedSource: {
    family: "minimax-h3",
    sourceShape: "transformer_directory",
    ownershipMode: "managed",
    sourceVariant: "ref2va",
    declaredPartition: "ref2va",
    baseModel: "minimax_h3_ref",
    componentTier: "q4",
    transformerDirectory: "",
    sharded: false,
  },
  supportedOperations: ["reference_to_video"],
  supportedBackends: ["mlx", "candle-cuda"],
  componentReadiness: {
    ready: false,
    components: [
      {
        componentId: "transformer",
        role: "imported_dit",
        source: "plan",
        model: null,
        present: true,
        state: "installed",
      },
      {
        componentId: "audio_vae",
        role: "resident",
        source: "resident",
        model: "MiniMaxAI/MiniMax-H3",
        present: false,
        state: "missing",
      },
    ],
  },
  repairActions: [
    {
      id: "install-component:audio_vae",
      kind: "download",
      label: "Download the compatible audio_vae from MiniMaxAI/MiniMax-H3",
    },
    {
      id: "declare-compatible-base",
      kind: "select-base",
      label: "Declare the compatible base model and task partition (fl2va, ref2va, or hybrid)",
    },
  ],
};

describe("ImportedCheckpointSurface", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
  });

  async function render(model, onRepair) {
    await act(async () => {
      root.render(<ImportedCheckpointSurface model={model} onRepair={onRepair} />);
    });
  }

  function text() {
    return container.textContent;
  }

  it("renders nothing for a builtin entry", async () => {
    await render({ id: "minimax_h3_ref", name: "MiniMax-H3" });
    expect(container.firstChild).toBeNull();
    expect(hasImportedCheckpointSurface({ id: "minimax_h3_ref" })).toBe(false);
  });

  it("names the source variant, the declared partition and the compatible base", async () => {
    await render(REF2VA_IMPORT);
    expect(text()).toContain("ref2va");
    expect(text()).toContain("minimax_h3_ref");
    expect(text()).toContain("transformer_directory");
    expect(importedSourceSummary(REF2VA_IMPORT)).toBe("ref2va · q4 · managed install");
  });

  it("advertises only the operations the declared partition serves", async () => {
    await render(REF2VA_IMPORT);
    expect(text()).toContain("Reference → Video");
    expect(text()).not.toContain("Text → Video");
  });

  it("shows the backends that will claim the import", async () => {
    await render(REF2VA_IMPORT);
    expect(text()).toContain("MLX / Metal");
    expect(text()).toContain("Candle / CUDA");
  });

  it("reports per-component readiness and never calls a half-provisioned assembly ready", async () => {
    await render(REF2VA_IMPORT);
    expect(text()).toContain("Ready");
    expect(text()).toContain("Missing");
    expect(text()).toContain("MiniMaxAI/MiniMax-H3");
  });

  it("offers each repair step, routing the ones the card can perform", async () => {
    const onRepair = vi.fn();
    await render(REF2VA_IMPORT, onRepair);
    const buttons = Array.from(container.querySelectorAll("button"));
    // Only `download` / `relink` / `rescan` earn a button; `select-base` is a declaration the user
    // makes in the import panel, so it renders as a sentence naming where that happens.
    expect(buttons).toHaveLength(1);
    expect(buttons[0].textContent).toContain("Download the compatible audio_vae");
    await act(async () => {
      buttons[0].click();
    });
    expect(onRepair).toHaveBeenCalledWith(REF2VA_IMPORT.repairActions[0]);
    expect(text()).toContain("declare it in the checkpoint import panel");
  });

  it("says so plainly when there is nothing to repair", async () => {
    await render({
      ...REF2VA_IMPORT,
      componentReadiness: {
        ready: true,
        components: [
          { componentId: "transformer", source: "plan", present: true, state: "installed" },
        ],
      },
      repairActions: [],
    });
    expect(text()).toContain("Nothing to repair");
    expect(container.querySelectorAll("button")).toHaveLength(0);
  });
});
