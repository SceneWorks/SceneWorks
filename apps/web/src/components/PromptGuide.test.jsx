import { settle } from "../testUtils/dom.js";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Markdown } from "./Markdown.jsx";
import { PromptGuideModal } from "./PromptGuideModal.jsx";



describe("Markdown renderer", () => {
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
    vi.restoreAllMocks();
  });

  function render(ui) {
    root = createRoot(container);
    act(() => root.render(ui));
  }

  it("renders headings, lists, bold, inline code, and safe links", () => {
    render(
      <Markdown
        content={[
          "# Title",
          "## Section",
          "A paragraph with **bold**, `code`, and a [link](https://example.com/guide).",
          "",
          "- first item",
          "- second item",
        ].join("\n")}
      />,
    );

    expect(document.body.querySelector("h1").textContent).toBe("Title");
    expect(document.body.querySelector("h2").textContent).toBe("Section");
    expect(document.body.querySelector("strong").textContent).toBe("bold");
    expect(document.body.querySelector("code").textContent).toBe("code");
    const link = document.body.querySelector("a");
    expect(link.getAttribute("href")).toBe("https://example.com/guide");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noopener noreferrer");
    expect([...document.body.querySelectorAll("li")].map((li) => li.textContent)).toEqual(["first item", "second item"]);
  });

  it("renders ordered lists and fenced code blocks", () => {
    render(<Markdown content={["1. step one", "2. step two", "", "```", "raw code line", "```"].join("\n")} />);
    expect(document.body.querySelector("ol")).not.toBeNull();
    expect([...document.body.querySelectorAll("ol li")].map((li) => li.textContent)).toEqual(["step one", "step two"]);
    expect(document.body.querySelector("pre code").textContent).toBe("raw code line");
  });

  it("drops unsafe link hrefs instead of rendering a javascript: anchor", () => {
    render(<Markdown content={"Click [here](javascript:alert(1)) now."} />);
    expect(document.body.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("here");
  });
});

describe("PromptGuideModal", () => {
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
    vi.restoreAllMocks();
  });

  it("fetches the guide path and renders its markdown with the model name", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, text: async () => "# Z-Image Guide\n\nUse short prompts." }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        <PromptGuideModal
          guide={{ title: "Z-Image-Turbo Prompt Guide", path: "/prompt-guides/z-image-turbo.md" }}
          modelName="Z-Image-Turbo"
          onClose={() => {}}
        />,
      );
    });
    await settle();

    expect(global.fetch).toHaveBeenCalledWith("/prompt-guides/z-image-turbo.md", { signal: expect.any(AbortSignal) });
    expect(document.body.querySelector("#prompt-guide-title").textContent).toBe("Z-Image-Turbo Prompt Guide");
    expect(document.body.textContent).toContain("Z-Image-Turbo · Prompt guide");
    expect(document.body.querySelector(".markdown-body h1").textContent).toBe("Z-Image Guide");
    expect(document.body.textContent).toContain("Use short prompts.");
  });

  it("aborts the previous guide on path change and the current guide on unmount", async () => {
    const pending = [];
    global.fetch = vi.fn((path, { signal }) => new Promise((resolve) => pending.push({ path, signal, resolve })));
    root = createRoot(container);
    const guide = (path) => <PromptGuideModal guide={{ title: "Guide", path }} onClose={() => {}} />;
    await act(async () => root.render(guide("/prompt-guides/first.md")));
    expect(pending[0].signal.aborted).toBe(false);
    await act(async () => root.render(guide("/prompt-guides/second.md")));
    expect(pending[0].signal.aborted).toBe(true);
    expect(pending[1].signal.aborted).toBe(false);
    await act(async () => {
      pending[1].resolve({ ok: true, text: async () => "# Current guide" });
      pending[0].resolve({ ok: true, text: async () => "# Stale guide" });
    });
    await settle();
    expect(document.body.textContent).toContain("Current guide");
    expect(document.body.textContent).not.toContain("Stale guide");
    act(() => root.unmount());
    root = null;
    expect(pending[1].signal.aborted).toBe(true);
  });

  it("renders metadata source links when provided", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, text: async () => "Body." }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        <PromptGuideModal
          guide={{
            title: "Guide",
            path: "/prompt-guides/x.md",
            sources: [{ label: "Model card", url: "https://example.com/card" }],
          }}
          onClose={() => {}}
        />,
      );
    });
    await settle();

    const source = document.body.querySelector(".prompt-guide-sources a");
    expect(source.textContent).toBe("Model card");
    expect(source.getAttribute("href")).toBe("https://example.com/card");
  });

  it("shows an error message when the guide cannot be loaded", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: false, status: 404, text: async () => "" }));
    root = createRoot(container);
    await act(async () => {
      root.render(<PromptGuideModal guide={{ title: "Guide", path: "/prompt-guides/missing.md" }} onClose={() => {}} />);
    });
    await settle();

    expect(document.body.textContent).toContain("could not be loaded");
    expect(document.body.querySelector(".markdown-body")).toBeNull();
  });

  it("closes on Escape", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, text: async () => "Body." }));
    const onClose = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(<PromptGuideModal guide={{ title: "Guide", path: "/prompt-guides/x.md" }} onClose={onClose} />);
    });
    await settle();

    await act(async () => {
      document.body.querySelector("[role=dialog]").dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(onClose).toHaveBeenCalled();
  });
});
