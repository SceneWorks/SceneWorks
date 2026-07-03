import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PromptGuideModal } from "./PromptGuideModal.jsx";

// sc-8881 [F-079]: prompt-guide `sources` come from the model manifest
// (`ui.promptGuide.sources`), rendered into `<a href>`. A `javascript:` source
// URL must never produce a clickable javascript link. The Modal portals to
// document.body, so anchors are queried there (see shared_modal_portals_to_body).
describe("PromptGuideModal hardens manifest-supplied source URLs (sc-8881)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    // The guide body fetch is irrelevant to link hardening; resolve it empty.
    global.fetch = vi.fn(() =>
      Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve("") }),
    );
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  async function renderModal(sources) {
    await act(async () => {
      root.render(
        <PromptGuideModal
          guide={{ title: "Guide", path: "/prompt-guides/x.md", sources }}
          modelName="Test"
          onClose={() => {}}
        />,
      );
    });
  }

  it("drops a javascript: source instead of rendering a clickable javascript href", async () => {
    await renderModal([{ label: "Evil", url: "javascript:alert(1)" }]);
    const anchors = Array.from(document.body.querySelectorAll("a"));
    // No anchor points at the javascript: payload.
    expect(anchors.some((a) => a.getAttribute("href")?.startsWith("javascript:"))).toBe(false);
    // The malicious source is omitted entirely (no dead link with its label).
    expect(anchors.some((a) => a.textContent === "Evil")).toBe(false);
  });

  it("renders valid https sources unchanged", async () => {
    await renderModal([{ label: "Docs", url: "https://example.com/docs" }]);
    const anchor = Array.from(document.body.querySelectorAll("a")).find(
      (a) => a.textContent === "Docs",
    );
    expect(anchor).toBeTruthy();
    expect(anchor.getAttribute("href")).toBe("https://example.com/docs");
  });

  it("keeps safe sources while dropping unsafe ones in a mixed list", async () => {
    await renderModal([
      { label: "Good", url: "https://example.com/a" },
      { label: "Bad", url: "javascript:alert(1)" },
    ]);
    const anchors = Array.from(document.body.querySelectorAll("a"));
    expect(anchors.some((a) => a.textContent === "Good")).toBe(true);
    expect(anchors.some((a) => a.textContent === "Bad")).toBe(false);
  });
});
