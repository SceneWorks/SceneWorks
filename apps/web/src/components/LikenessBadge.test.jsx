import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { LikenessBadge, LIKENESS_METHOD_LABEL, assetFaceLikeness } from "./LikenessBadge.jsx";
import { LIKENESS_METHOD } from "../faceLikeness.js";

// Frontal-identity likeness badge (epic 4406, sc-4413). Proves the three render states the AC
// requires: a colour-banded score badge, the neutral N/A treatment (never a low number), and the
// graceful no-render for legacy/unscored assets — all reading the band from the shared classifier.

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function render(ui) {
  act(() => root.render(ui));
}

function assetWith(faceLikeness) {
  return { id: "asset-1", recipe: { rawAdapterSettings: { faceLikeness } } };
}

describe("LikenessBadge (sc-4413)", () => {
  describe("scored results show a colour-banded percentage", () => {
    it("renders a strong score as a percentage in the strong band", () => {
      render(<LikenessBadge faceLikeness={{ score: 0.92, detected: true, method: LIKENESS_METHOD }} />);
      const badge = container.querySelector(".likeness-badge");
      expect(badge).toBeTruthy();
      expect(badge.dataset.band).toBe("strong");
      expect(badge.classList.contains("likeness-badge--strong")).toBe(true);
      expect(badge.textContent).toContain("92%");
    });

    it("bands a mid score as moderate and a drift score as weak", () => {
      render(<LikenessBadge faceLikeness={{ score: 0.71, detected: true }} />);
      expect(container.querySelector(".likeness-badge").dataset.band).toBe("moderate");
      expect(container.querySelector(".likeness-badge").textContent).toContain("71%");

      render(<LikenessBadge faceLikeness={{ score: 0.2, detected: true }} />);
      expect(container.querySelector(".likeness-badge").dataset.band).toBe("weak");
      expect(container.querySelector(".likeness-badge").textContent).toContain("20%");
    });

    it("tooltip exposes raw cosine, method, and the scored-against source", () => {
      render(
        <LikenessBadge
          faceLikeness={{ score: 0.876, detected: true, sourceAssetId: "ref-42" }}
        />,
      );
      const title = container.querySelector(".likeness-badge").getAttribute("title");
      expect(title).toContain("Cosine: 0.876");
      expect(title).toContain(LIKENESS_METHOD_LABEL);
      expect(title).toContain("ref-42");
      // honest framing, not a quality claim
      expect(title.toLowerCase()).toContain("frontal-identity");
    });

    it("prefers a resolved source label over the raw id", () => {
      render(
        <LikenessBadge
          faceLikeness={{ score: 0.876, detected: true, sourceAssetId: "ref-42" }}
          sourceLabel="Reference portrait"
        />,
      );
      const title = container.querySelector(".likeness-badge").getAttribute("title");
      expect(title).toContain("Reference portrait");
      expect(title).not.toContain("ref-42");
    });

    it("clamps a percentage to [0, 100] and rounds", () => {
      render(<LikenessBadge faceLikeness={{ score: 1.0, detected: true }} />);
      expect(container.querySelector(".likeness-badge").textContent).toContain("100%");
    });
  });

  describe("N/A state (detected:false) is neutral, not a low number", () => {
    it("renders the neutral dash + explanatory copy", () => {
      render(
        <LikenessBadge
          faceLikeness={{ score: null, detected: false, method: LIKENESS_METHOD, reason: "no_face" }}
        />,
      );
      const badge = container.querySelector(".likeness-badge");
      expect(badge).toBeTruthy();
      expect(badge.dataset.band).toBe("na");
      expect(badge.classList.contains("likeness-badge--na")).toBe(true);
      // neutral dash, explanatory copy — and crucially NOT a red low percentage
      expect(badge.textContent).toContain("—");
      expect(badge.textContent).toContain("No frontal face to score");
      expect(badge.textContent).not.toMatch(/\d+%/);
      expect(badge.classList.contains("likeness-badge--weak")).toBe(false);
    });

    it("frames N/A as identity confidence, not quality, in the tooltip", () => {
      render(<LikenessBadge faceLikeness={{ score: null, detected: false }} />);
      const title = container.querySelector(".likeness-badge").getAttribute("title");
      expect(title).toContain("No frontal face");
      expect(title.toLowerCase()).toContain("not overall image quality");
      // never invents a cosine for an unscored block
      expect(title).not.toContain("Cosine:");
    });
  });

  describe("absent / legacy block renders nothing", () => {
    it("renders no badge when faceLikeness is absent on the asset", () => {
      render(<LikenessBadge asset={{ id: "legacy", recipe: { rawAdapterSettings: {} } }} />);
      expect(container.querySelector(".likeness-badge")).toBeNull();
      expect(container.textContent).toBe("");
    });

    it("renders no badge when the recipe itself is absent", () => {
      render(<LikenessBadge asset={{ id: "legacy" }} />);
      expect(container.querySelector(".likeness-badge")).toBeNull();
    });

    it("renders no badge when neither asset nor block is provided", () => {
      render(<LikenessBadge />);
      expect(container.querySelector(".likeness-badge")).toBeNull();
    });
  });

  describe("asset accessor", () => {
    it("reads the block off recipe.rawAdapterSettings.faceLikeness", () => {
      const block = { score: 0.8, detected: true };
      expect(assetFaceLikeness(assetWith(block))).toBe(block);
    });

    it("returns null for a legacy/unscored asset", () => {
      expect(assetFaceLikeness({ id: "x", recipe: { rawAdapterSettings: {} } })).toBeNull();
      expect(assetFaceLikeness(null)).toBeNull();
    });
  });

  it("reads the block from an asset and bands it end-to-end", () => {
    render(<LikenessBadge asset={assetWith({ score: 0.85, detected: true })} />);
    const badge = container.querySelector(".likeness-badge");
    expect(badge.dataset.band).toBe("strong");
    expect(badge.dataset.method).toBe(LIKENESS_METHOD);
    expect(badge.textContent).toContain("85%");
  });
});
