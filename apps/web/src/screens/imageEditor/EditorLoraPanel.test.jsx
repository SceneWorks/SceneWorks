import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import { EditorLoraPanel } from "./EditorLoraPanel.jsx";

let container;
let root;
beforeEach(() => ({ container, root } = mountRoot()));
afterEach(() => unmountRoot(root, container));

function props(overrides = {}) {
  return {
    selectedModel: { id: "model" },
    selectedLoras: [],
    selectedLoraIds: [],
    toggleLora: vi.fn(),
    weightFor: () => 1,
    setWeight: vi.fn(),
    availableLoras: [{ id: "next", name: "Next", scope: "global" }],
    showIncompatible: false,
    setShowIncompatible: vi.fn(),
    onUpdateLora: vi.fn(),
    ...overrides,
  };
}

describe("EditorLoraPanel", () => {
  it("does not count builtin LoRAs against the user cap", async () => {
    await act(async () => root.render(<EditorLoraPanel {...props({
        selectedLoras: [
          { id: "builtin", name: "Builtin", scope: "builtin" },
          { id: "u1", scope: "global" },
          { id: "u2", scope: "project" },
          { id: "u3", scope: "global" },
        ],
        selectedLoraIds: ["builtin", "u1", "u2", "u3"],
      })} />));
    await act(async () => container.querySelector(".lora-add").click());
    expect(container.querySelector(".lora-pick-row").disabled).toBe(false);
  });

  it("memoizes stable props across an unrelated parent rerender", async () => {
    let renders = 0;
    const stable = props();
    const CountedPanel = React.memo(function CountedPanel(panelProps) {
      renders += 1;
      return <EditorLoraPanel {...panelProps} />;
    });
    function Harness() {
      const [tick, setTick] = React.useState(0);
      return (
        <>
          <button data-action="tick" onClick={() => setTick((value) => value + 1)} type="button">{tick}</button>
          <CountedPanel {...stable} />
        </>
      );
    }
    await act(async () => root.render(<Harness />));
    await act(async () => container.querySelector("[data-action='tick']").click());
    expect(renders).toBe(1);
  });
});
