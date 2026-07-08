import React from "react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { LoraKeywordSummary } from "./LoraKeywordSummary.jsx";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";

describe("LoraKeywordSummary", () => {
  let root;
  let container;

  beforeEach(() => {
    ({ root, container } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
  });

  it("renders keyword chips and notes", async () => {
    await act(async () =>
      root.render(<LoraKeywordSummary lora={{ triggerWords: ["a", "b"], notes: "use at 0.7" }} />),
    );
    expect([...container.querySelectorAll(".kw-chip")].map((chip) => chip.textContent)).toEqual([
      "a",
      "b",
    ]);
    expect(container.querySelector(".lora-notes").textContent).toBe("use at 0.7");
  });

  it("renders nothing when there are no keywords or notes", async () => {
    await act(async () => root.render(<LoraKeywordSummary lora={{ triggerWords: [], notes: "" }} />));
    expect(container.querySelector(".lora-keyword-summary")).toBeNull();
  });

  it("renders only notes (trimmed) when keywords are absent", async () => {
    await act(async () => root.render(<LoraKeywordSummary lora={{ notes: "  trim me  " }} />));
    expect(container.querySelector(".lora-keywords")).toBeNull();
    expect(container.querySelector(".lora-notes").textContent).toBe("trim me");
  });
});
