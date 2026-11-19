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

function Harness({ initial, expandPrompt, onText, variableValues = {} }) {
  const [text, setText] = React.useState(initial);
  onText(text);
  return (
    <BatchExpandPromptsControl
      copy={H3_IMAGE_EXPAND_COPY}
      variableValues={variableValues}
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
  it("fills in variables first, expands each resolved prompt, keeps directives, and applies only on request", async () => {
    let text = "";
    const expandPrompt = vi.fn(async ({ prompt }) => {
      if (prompt === "broken") throw new Error("worker exploded");
      return { refinedPrompt: `Expanded: ${prompt}`, rewriteSuggestion: { whRatio: "16:9" } };
    });
    const initial = "a fox\n[832x1216] a heron\n{{name}} in a {{red|blue}} coat\nbroken";
    await act(async () =>
      root.render(
        <Harness
          initial={initial}
          expandPrompt={expandPrompt}
          variableValues={{ name: ["Ann", "Bo"] }}
          onText={(value) => { text = value; }}
        />,
      ),
    );

    expect(container.textContent).toContain("Variables are filled in first, so each of the 7 resulting prompts");
    await act(async () => click(button("Expand all 7 prompts")));
    await settle();

    // The directive is not sent to the expander; every variable combination is expanded on its own.
    expect(expandPrompt.mock.calls.map(([request]) => request.prompt)).toEqual([
      "a fox",
      "a heron",
      "Ann in a red coat",
      "Ann in a blue coat",
      "Bo in a red coat",
      "Bo in a blue coat",
      "broken",
    ]);
    expect(expandPrompt.mock.calls[0][0]).toMatchObject({ modelId: "minimax_h3_image", projectId: "p1" });
    expect(text).toBe(initial);
    expect(container.textContent).toContain("Expanded 6 of 7 prompts.");
    expect(container.textContent).toContain("Prompt 7: worker exploded");

    // Opt in to the proposed sizes: only prompts without their own directive take one.
    await act(async () => click(container.querySelector(".batch-expand-sizes input")));
    await act(async () => click(button("Apply to batch")));
    expect(splitPromptLines(text)).toEqual([
      "[2752x1536] Expanded: a fox",
      "[832x1216] Expanded: a heron",
      "[2752x1536] Expanded: Ann in a red coat",
      "[2752x1536] Expanded: Ann in a blue coat",
      "[2752x1536] Expanded: Bo in a red coat",
      "[2752x1536] Expanded: Bo in a blue coat",
      "broken",
    ]);

    await act(async () => click(button("Undo")));
    expect(text).toBe(initial);
  });

  it("asks before a large run, and refuses an unfilled or oversized template", async () => {
    const expandPrompt = vi.fn(async ({ prompt }) => ({ refinedPrompt: `Expanded ${prompt}` }));
    const many = Array.from({ length: 101 }, (_, index) => `v${index}`);
    await act(async () =>
      root.render(<Harness initial="{{x}} portrait" expandPrompt={expandPrompt} variableValues={{ x: many }} onText={() => {}} />),
    );
    await act(async () => click(button("Expand all 101 prompts")));
    expect(expandPrompt).not.toHaveBeenCalled();
    expect(container.textContent).toContain("That runs the expander 101 times");
    await act(async () => click(button("Cancel")));
    expect(expandPrompt).not.toHaveBeenCalled();

    await act(async () =>
      root.render(<Harness key="missing" initial="{{missing}} portrait" expandPrompt={expandPrompt} onText={() => {}} />),
    );
    expect(container.textContent).toContain("Add a value for {{missing}} before expanding.");
    expect(button("Expand").disabled).toBe(true);

    const huge = Array.from({ length: 50 }, (_, index) => `v${index}`);
    await act(async () =>
      root.render(
        <Harness key="huge" initial="{{a}} {{b}}" expandPrompt={expandPrompt} variableValues={{ a: huge, b: huge }} onText={() => {}} />,
      ),
    );
    expect(container.textContent).toContain("fill in to 2,500 prompts — more than the 2,000");
    expect(button("Expand").disabled).toBe(true);
  });

  it("re-checks the limits when a confirmed run starts after the variables changed", async () => {
    const expandPrompt = vi.fn(async ({ prompt }) => ({ refinedPrompt: `Expanded ${prompt}` }));
    const values = (count) => Array.from({ length: count }, (_, index) => `v${index}`);
    const view = (variableValues) => <Harness key="grow" initial="{{a}} {{b}}" expandPrompt={expandPrompt} variableValues={variableValues} onText={() => {}} />;
    await act(async () => root.render(view({ a: values(101), b: values(1) })));
    await act(async () => click(button("Expand all 101 prompts")));
    expect(container.textContent).toContain("That runs the expander 101 times");

    // Edited past the maximum while the confirmation is open.
    await act(async () => root.render(view({ a: values(101), b: values(50) })));
    const confirm = container.querySelector(".batch-confirm button");
    expect(confirm.disabled).toBe(true);
    expect(container.querySelector(".batch-confirm").textContent).toContain("more than the 2,000");
    await act(async () => click(confirm));
    await settle();
    expect(expandPrompt).not.toHaveBeenCalled();
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
