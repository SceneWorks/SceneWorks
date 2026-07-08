import React from "react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { KeywordTagEditor } from "./KeywordTagEditor.jsx";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";

// Controlled wrapper: the editor is presentational, so the harness owns the value
// and feeds it back through onChange exactly like the real consumers do.
function Harness({ initial = [], suggestions = [] }) {
  const [value, setValue] = React.useState(initial);
  return <KeywordTagEditor onChange={setValue} suggestions={suggestions} value={value} />;
}

async function pressKey(input, key) {
  await act(async () => {
    input.dispatchEvent(new window.KeyboardEvent("keydown", { bubbles: true, key }));
  });
}

const chipText = (container) =>
  [...container.querySelectorAll(".kw-chip")].map((chip) => chip.firstChild.textContent.trim());

describe("KeywordTagEditor", () => {
  let root;
  let container;

  beforeEach(() => {
    ({ root, container } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
  });

  it("adds a keyword chip on Enter and clears the input", async () => {
    await act(async () => root.render(<Harness />));
    const input = container.querySelector(".kw-input");
    await act(async () => setInput(input, "sksStyle"));
    await pressKey(input, "Enter");
    expect(chipText(container)).toEqual(["sksStyle"]);
    expect(container.querySelector(".kw-input").value).toBe("");
  });

  it("adds on comma and de-duplicates case-insensitively", async () => {
    await act(async () => root.render(<Harness initial={["neon"]} />));
    const input = container.querySelector(".kw-input");
    await act(async () => setInput(input, "glow"));
    await pressKey(input, ",");
    expect(chipText(container)).toEqual(["neon", "glow"]);
    // Duplicate (different case) is ignored.
    await act(async () => setInput(input, "NEON"));
    await pressKey(input, "Enter");
    expect(chipText(container)).toEqual(["neon", "glow"]);
  });

  it("removes a keyword when its chip button is clicked", async () => {
    await act(async () => root.render(<Harness initial={["a", "b", "c"]} />));
    const removeB = container.querySelectorAll(".kw-chip")[1].querySelector("button");
    await click(removeB);
    expect(chipText(container)).toEqual(["a", "c"]);
  });

  it("offers suggestions and adds one on click, hiding it afterward", async () => {
    await act(async () => root.render(<Harness suggestions={["1girl", "solo"]} />));
    expect(container.querySelectorAll(".kw-suggestion")).toHaveLength(2);
    await click(container.querySelector(".kw-suggestion"));
    expect(chipText(container)).toEqual(["1girl"]);
    // The chosen suggestion is now a chip, so only the other remains offered.
    const remaining = [...container.querySelectorAll(".kw-suggestion")].map((b) => b.textContent.trim());
    expect(remaining).toEqual(["+ solo"]);
  });
});
