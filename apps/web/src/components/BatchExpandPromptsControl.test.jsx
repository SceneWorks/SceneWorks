import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";
import { BatchExpandPromptsControl } from "./BatchExpandPromptsControl.jsx";
import { H3_IMAGE_EXPAND_COPY } from "../minimaxH3ImageExpand.js";
import { splitPromptLines } from "../promptBatch.js";

// The batch-mode prompt expander: every prompt of the batch list is expanded in turn, and the
// list changes only when the user applies the result (and can be undone).

let root, container;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
});
afterEach(() => unmountRoot(root, container));

const button = (text) => [...container.querySelectorAll("button")].find((node) => node.textContent.includes(text));

function Harness({ initial, expandPrompt, onText }) {
  const [text, setText] = React.useState(initial);
  onText(text);
  return (
    <BatchExpandPromptsControl
      copy={H3_IMAGE_EXPAND_COPY}
      expandPrompt={expandPrompt}
      modelId="minimax_h3_image"
      onPromptsTextChange={setText}
      projectId="p1"
      promptsText={text}
      suggestedResolutionFor={(suggestion) => (suggestion?.whRatio === "16:9" ? "2752x1536" : "")}
    />
  );
}

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

describe("BatchExpandPromptsControl", () => {
  it("expands each prompt, keeps directives, skips placeholders, and applies only on request", async () => {
    let text = "";
    const expandPrompt = vi.fn(async ({ prompt }) => ({
      refinedPrompt: `Expanded: ${prompt}`,
      rewriteSuggestion: { whRatio: "16:9" },
    }));
    const initial = "a fox\n[832x1216] a heron\n{{name}} in a coat\nbroken";
    expandPrompt.mockImplementation(async ({ prompt }) => {
      if (prompt === "broken") throw new Error("worker exploded");
      return { refinedPrompt: `Expanded: ${prompt}`, rewriteSuggestion: { whRatio: "16:9" } };
    });
    await act(async () => root.render(<Harness initial={initial} expandPrompt={expandPrompt} onText={(value) => { text = value; }} />));

    expect(container.textContent).toContain("1 prompt uses {{…}} placeholders");
    await act(async () => click(button("Expand all 3 prompts")));
    await settle();

    // The directive is not sent to the expander; the placeholder prompt is never expanded.
    expect(expandPrompt.mock.calls.map(([request]) => request.prompt)).toEqual(["a fox", "a heron", "broken"]);
    expect(expandPrompt.mock.calls[0][0]).toMatchObject({ modelId: "minimax_h3_image", projectId: "p1" });
    expect(text).toBe(initial);
    expect(container.textContent).toContain("Expanded 2 of 3 prompts.");
    expect(container.textContent).toContain("Prompt 4: worker exploded");

    // Opt in to the proposed sizes: only the prompt without its own directive takes one.
    const sizes = container.querySelector(".batch-expand-sizes input");
    await act(async () => click(sizes));
    await act(async () => click(button("Apply to batch")));
    expect(splitPromptLines(text)).toEqual([
      "[2752x1536] Expanded: a fox",
      "[832x1216] Expanded: a heron",
      "{{name}} in a coat",
      "broken",
    ]);

    await act(async () => click(button("Undo")));
    expect(text).toBe(initial);
  });

  it("keeps finished expansions when stopped", async () => {
    let text = "";
    let release;
    const expandPrompt = vi.fn(({ prompt, signal }) =>
      prompt === "one"
        ? Promise.resolve({ refinedPrompt: "Expanded one" })
        : new Promise((resolve, reject) => {
            release = resolve;
            signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
          }),
    );
    await act(async () => root.render(<Harness initial={"one\ntwo"} expandPrompt={expandPrompt} onText={(value) => { text = value; }} />));
    await act(async () => click(button("Expand all 2 prompts")));
    await settle();
    expect(container.textContent).toContain("Expanding prompt 2 of 2");
    await act(async () => click(button("Stop")));
    await settle();
    expect(release).toEqual(expect.any(Function));
    expect(container.textContent).toContain("Expanded 1 of 2 prompts.");
    await act(async () => click(button("Apply to batch")));
    expect(splitPromptLines(text)).toEqual(["Expanded one", "two"]);
  });

  it("refuses to apply over a batch list edited during expansion", async () => {
    let text = "";
    let setOuter;
    function Outer() {
      const [value, setValue] = React.useState("a fox");
      setOuter = setValue;
      text = value;
      return (
        <BatchExpandPromptsControl
          copy={H3_IMAGE_EXPAND_COPY}
          expandPrompt={async () => ({ refinedPrompt: "Expanded" })}
          onPromptsTextChange={setValue}
          promptsText={value}
        />
      );
    }
    await act(async () => root.render(<Outer />));
    await act(async () => click(button("Expand the prompt")));
    await settle();
    await act(async () => setOuter("a fox\na wolf"));
    await act(async () => click(button("Apply to batch")));
    expect(text).toBe("a fox\na wolf");
    expect(container.textContent).toContain("The batch prompts changed while expanding.");
  });
});
