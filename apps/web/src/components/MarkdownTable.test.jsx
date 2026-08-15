// sc-17161 — GFM pipe tables in the prompt-guide renderer.
//
// Eight shipped guides already used tables and every one of them rendered as literal-pipe
// paragraphs, delimiter row included. The MiniMax-H3 guide is the reason this landed with that
// story: it opens with two tables, one of which is the canvas-cost table a first-time user needs
// to read BEFORE starting a two-hour render, so the guide was "reachable" while its most
// load-bearing content was the least legible thing on the page.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Markdown } from "./Markdown.jsx";

const HERE = dirname(fileURLToPath(import.meta.url));
const GUIDE = resolve(HERE, "../../public/prompt-guides/minimax-h3.md");

describe("Markdown tables (sc-17161)", () => {
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

  it("renders a header row, a body and per-column alignment", () => {
    render(
      <Markdown
        content={[
          "| Canvas | Clip | Approx. render time |",
          "|---|:-:|---:|",
          "| 576x320 | 5.17 s | ~14 minutes |",
          "| 1344x768 | 5.17 s | **~2 hours** |",
        ].join("\n")}
      />,
    );
    const table = container.querySelector(".markdown-table-wrap table");
    expect(table).toBeTruthy();
    expect([...table.querySelectorAll("thead th")].map((cell) => cell.textContent)).toEqual([
      "Canvas",
      "Clip",
      "Approx. render time",
    ]);
    const rows = [...table.querySelectorAll("tbody tr")];
    expect(rows).toHaveLength(2);
    expect([...rows[0].querySelectorAll("td")].map((cell) => cell.textContent)).toEqual([
      "576x320",
      "5.17 s",
      "~14 minutes",
    ]);
    // The delimiter row is structure, never content — it must not survive as a body row.
    expect(table.textContent).not.toContain("---");
    // Inline rules still run inside cells.
    expect(rows[1].querySelector("strong").textContent).toBe("~2 hours");
    // Alignment comes from the delimiter's colons.
    const headers = table.querySelectorAll("thead th");
    expect(headers[1].style.textAlign).toBe("center");
    expect(headers[2].style.textAlign).toBe("right");
    expect(headers[0].style.textAlign).toBe("");
  });

  it("leaves a pipe paragraph alone when there is no delimiter row", () => {
    // The delimiter is what makes it a table. Without this leg the parser would swallow any
    // paragraph containing pipes — e.g. a prompt example with an alternation in it.
    render(<Markdown content={"| this is just prose | with a pipe |"} />);
    expect(container.querySelector("table")).toBeNull();
    expect(container.querySelector("p").textContent).toContain("just prose");
  });

  it("tolerates a short row and an escaped pipe", () => {
    render(
      <Markdown
        content={["| A | B |", "|---|---|", "| only-one |", "| a \\| b | second |"].join("\n")}
      />,
    );
    const rows = [...container.querySelectorAll("tbody tr")];
    // A short row is padded rather than dropping the table: a guide is content, not a data feed.
    expect([...rows[0].querySelectorAll("td")].map((cell) => cell.textContent)).toEqual([
      "only-one",
      "",
    ]);
    expect(rows[1].querySelectorAll("td")[0].textContent).toBe("a | b");
  });

  it("renders the shipped MiniMax-H3 guide's tables as tables", () => {
    // Against the REAL guide bytes: the manifest points both partitions at this file, and the
    // point of the feature is that THIS content renders, not that the parser passes a fixture.
    render(<Markdown content={readFileSync(GUIDE, "utf8")} />);
    const tables = [...container.querySelectorAll(".markdown-table-wrap table")];
    expect(tables.length).toBeGreaterThanOrEqual(2);
    expect(container.textContent).toContain("~2 hours");
    // No table markup survives as prose anywhere on the page.
    for (const paragraph of container.querySelectorAll("p")) {
      expect(paragraph.textContent).not.toMatch(/^\s*\|/);
      expect(paragraph.textContent).not.toContain("|---|");
    }
  });
});
