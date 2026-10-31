import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  DatasetDoctorDistributions,
  DatasetDoctorReadout,
  NearDuplicateClusters,
  ReadinessBadge,
  ReadinessFlagDetails,
} from "./DatasetDoctor.jsx";

// Render helper mirroring the project's createRoot + act convention (no testing-library).
let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

function mount(node) {
  act(() => root.render(node));
}

function report(overrides = {}) {
  return {
    gate: "needs_attention",
    subScores: { technical: 0.8 },
    counts: { info: 0, warn: 2, fatal: 0 },
    itemCount: 5,
    items: [],
    datasetFlags: [],
    ...overrides,
  };
}

describe("DatasetDoctorReadout gating states", () => {
  it("renders nothing for an unsaved draft (no report, not loading)", () => {
    mount(<DatasetDoctorReadout report={null} loading={false} />);
    expect(container.querySelector(".dataset-doctor")).toBeNull();
  });

  it("shows a pending state while the first fetch is in flight", () => {
    mount(<DatasetDoctorReadout report={null} loading />);
    expect(container.textContent).toContain("Checking dataset");
  });

  it("renders the gate headline, meter, and summary for a warnable set", () => {
    mount(
      <DatasetDoctorReadout
        report={report({
          itemCount: 18,
          items: [
            { itemId: "a", severity: "warn", flags: [{ check: "blur", severity: "warn" }] },
            { itemId: "b", severity: "warn", flags: [{ check: "blur", severity: "warn" }] },
          ],
        })}
      />,
    );
    const node = container.querySelector(".dataset-doctor");
    expect(node.className).toContain("tone-warn");
    expect(container.textContent).toContain("Trainable");
    expect(container.textContent).toContain("18 photos.");
    expect(container.textContent).toContain("2 look blurry");
    // Technical sub-score drives the meter width.
    expect(container.querySelector(".dataset-doctor-meter-fill").style.width).toBe("80%");
  });

  it("renders the Variety meter only once a diversity sub-score is present (sc-6535)", () => {
    // No diversity sub-score yet (pre-embedding-job) → no Variety meter.
    mount(<DatasetDoctorReadout report={report()} />);
    expect(container.querySelector(".dataset-doctor-meter-variety")).toBeNull();

    // With the Tier-1 diversity sub-score → a second, distinct meter at that width.
    mount(<DatasetDoctorReadout report={report({ subScores: { technical: 0.8, diversity: 0.3 } })} />);
    const variety = container.querySelector(".dataset-doctor-meter-variety");
    expect(variety).not.toBeNull();
    expect(variety.querySelector(".dataset-doctor-meter-fill").style.width).toBe("30%");
  });

  it("renders the Caption match meter only once an alignment sub-score is present (sc-6537)", () => {
    mount(<DatasetDoctorReadout report={report()} />);
    expect(container.querySelector(".dataset-doctor-meter-alignment")).toBeNull();

    // alignment is a raw CLIP cosine (~0.15 is a good match), rescaled for display by alignmentPercent.
    mount(<DatasetDoctorReadout report={report({ subScores: { technical: 0.8, alignment: 0.15 } })} />);
    const alignment = container.querySelector(".dataset-doctor-meter-alignment");
    expect(alignment).not.toBeNull();
    expect(alignment.getAttribute("role")).toBe("meter");
    expect(alignment.getAttribute("aria-label")).toBe("Caption match score");
    expect(alignment.getAttribute("aria-valuenow")).toBe("77");
    expect(alignment.getAttribute("title")).toContain("Caption match score");
    expect(alignment.querySelector(".dataset-doctor-meter-fill").style.width).toBe("77%");
  });

  it("surfaces a re-caption action for caption-alignment-flagged items (sc-6537)", () => {
    const onRecaptionFlagged = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [
            { itemId: "a", flags: [{ check: "caption_alignment", severity: "warn" }] },
            { itemId: "b", flags: [] },
            { itemId: "c", flags: [{ check: "caption_alignment", severity: "warn" }] },
          ],
        })}
        onRecaptionFlagged={onRecaptionFlagged}
      />,
    );
    const button = container.querySelector(".dataset-doctor-actions button");
    expect(button).not.toBeNull();
    expect(button.textContent).toContain("Re-caption 2");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onRecaptionFlagged).toHaveBeenCalledWith(["a", "c"]);
  });

  it("hides the re-caption action with no flagged items or no handler (sc-6537)", () => {
    mount(
      <DatasetDoctorReadout
        report={report({ items: [{ itemId: "a", flags: [{ check: "blur", severity: "warn" }] }] })}
        onRecaptionFlagged={() => {}}
      />,
    );
    expect(container.querySelector(".dataset-doctor-actions")).toBeNull();
    // Even with flagged items, no button without a handler (e.g. the compact readout).
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [{ itemId: "a", flags: [{ check: "caption_alignment", severity: "warn" }] }],
        })}
      />,
    );
    expect(container.querySelector(".dataset-doctor-actions")).toBeNull();
  });

  it("surfaces a drop-duplicates action and removes the planned copies (sc-6539)", () => {
    const onRemoveDuplicates = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          duplicateRemoval: {
            groups: [
              { keep: "a", remove: ["b"] },
              { keep: "c", remove: ["d", "e"] },
            ],
          },
        })}
        onRemoveDuplicates={onRemoveDuplicates}
      />,
    );
    const button = container.querySelector(".dataset-doctor-actions button");
    expect(button).not.toBeNull();
    expect(button.textContent).toContain("Remove 3 duplicates");
    expect(button.textContent).toContain("keeps the sharpest");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onRemoveDuplicates).toHaveBeenCalledWith(["b", "d", "e"]);
  });

  it("hides the drop-duplicates action with no plan or no handler (sc-6539)", () => {
    // A plan present but no handler (e.g. a read-only consumer) → no button.
    mount(
      <DatasetDoctorReadout
        report={report({ duplicateRemoval: { groups: [{ keep: "a", remove: ["b"] }] } })}
      />,
    );
    expect(container.querySelector(".dataset-doctor-actions")).toBeNull();
    // A handler but no removable duplicates → no button.
    mount(
      <DatasetDoctorReadout report={report({})} onRemoveDuplicates={() => {}} />,
    );
    expect(container.querySelector(".dataset-doctor-actions")).toBeNull();
  });

  it("surfaces an upscale action for resolution-flagged items (sc-6539)", () => {
    const onUpscaleLowRes = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [
            { itemId: "a", flags: [{ check: "resolution", severity: "warn" }] },
            { itemId: "b", flags: [] },
            { itemId: "c", flags: [{ check: "resolution", severity: "warn" }] },
          ],
        })}
        onUpscaleLowRes={onUpscaleLowRes}
      />,
    );
    const button = container.querySelector(".dataset-doctor-actions button");
    expect(button).not.toBeNull();
    expect(button.textContent).toContain("Upscale 2 low-res");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onUpscaleLowRes).toHaveBeenCalledWith(["a", "c"]);
  });

  it("surfaces a smart-crop action for crop_loss-flagged items (sc-6539)", () => {
    const onSmartCrop = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [
            { itemId: "a", flags: [{ check: "crop_loss", severity: "warn" }] },
            { itemId: "b", flags: [] },
            { itemId: "c", flags: [{ check: "crop_loss", severity: "warn" }] },
          ],
        })}
        onSmartCrop={onSmartCrop}
      />,
    );
    const button = [...container.querySelectorAll(".dataset-doctor-actions button")].find((node) =>
      node.textContent.includes("Smart-crop"),
    );
    expect(button).toBeTruthy();
    expect(button.textContent).toContain("Smart-crop 2 images");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onSmartCrop).toHaveBeenCalledWith(["a", "c"]);
  });

  it("surfaces a strip-metadata action only for items that still carry metadata (sc-6539)", () => {
    const onStripExif = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [
            { itemId: "a", flags: [], metadataStrippable: true },
            // Already a clean PNG (stripped) — must not be counted, so the action clears once done.
            { itemId: "b", flags: [], metadataStrippable: false },
            { itemId: "c", flags: [], metadataStrippable: true },
          ],
        })}
        onStripExif={onStripExif}
      />,
    );
    const button = [...container.querySelectorAll(".dataset-doctor-actions button")].find((node) =>
      node.textContent.includes("Strip metadata"),
    );
    expect(button).toBeTruthy();
    expect(button.textContent).toContain("Strip metadata from 2 images");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onStripExif).toHaveBeenCalledWith(["a", "c"]);
  });

  it("hides the strip-metadata action when every item is already a clean PNG (sc-6539)", () => {
    const onStripExif = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [
            { itemId: "a", flags: [], metadataStrippable: false },
            { itemId: "b", flags: [], metadataStrippable: false },
          ],
        })}
        onStripExif={onStripExif}
      />,
    );
    const button = [...container.querySelectorAll(".dataset-doctor-actions button")].find((node) =>
      node.textContent.includes("Strip metadata"),
    );
    expect(button).toBeFalsy();
  });

  it("surfaces a Check-faces action for a person dataset, even on a clean set (sc-6538)", () => {
    const onAnalyzeFaces = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          kind: "person",
          gate: "ready",
          counts: { info: 0, warn: 0, fatal: 0 },
          items: [{ itemId: "a", flags: [] }],
        })}
        onAnalyzeFaces={onAnalyzeFaces}
      />,
    );
    const button = [...container.querySelectorAll(".dataset-doctor-actions button")].find((node) =>
      node.textContent.includes("Check faces"),
    );
    expect(button).toBeTruthy();
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onAnalyzeFaces).toHaveBeenCalled();
  });

  it("surfaces an Analyze-photos action for any kind when wired, even on a clean set (sc-6535)", () => {
    const onAnalyzeDataset = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          kind: "style",
          gate: "ready",
          counts: { info: 0, warn: 0, fatal: 0 },
          items: [{ itemId: "a", flags: [] }],
        })}
        onAnalyzeDataset={onAnalyzeDataset}
      />,
    );
    const button = [...container.querySelectorAll(".dataset-doctor-actions button")].find((node) =>
      node.textContent.includes("Analyze photos"),
    );
    expect(button).toBeTruthy();
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onAnalyzeDataset).toHaveBeenCalled();
  });

  it("hides the Check-faces action for a non-person dataset (sc-6538)", () => {
    const onAnalyzeFaces = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({ kind: "style", items: [{ itemId: "a", flags: [] }] })}
        onAnalyzeFaces={onAnalyzeFaces}
      />,
    );
    const button = [...container.querySelectorAll(".dataset-doctor-actions button")].find((node) =>
      node.textContent.includes("Check faces"),
    );
    expect(button).toBeFalsy();
  });

  it("renders the blocked headline when the set is untrainable", () => {
    mount(
      <DatasetDoctorReadout
        report={report({
          gate: "blocked",
          itemCount: 3,
          counts: { info: 0, warn: 0, fatal: 1 },
          datasetFlags: [{ check: "count", severity: "fatal", value: 3, threshold: 15 }],
        })}
      />,
    );
    expect(container.querySelector(".dataset-doctor").className).toContain("tone-fatal");
    expect(container.textContent).toContain("Not ready to train");
    expect(container.textContent).toContain("add 12 more");
  });

  it("renders a ready set as ready", () => {
    mount(<DatasetDoctorReadout report={report({ gate: "ready", itemCount: 20, counts: { info: 0, warn: 0, fatal: 0 } })} />);
    expect(container.querySelector(".dataset-doctor").className).toContain("tone-good");
    expect(container.textContent).toContain("Ready to train");
  });
});

describe("ReadinessBadge gating states", () => {
  it("is pending (never good) for an item the report has not covered", () => {
    mount(<ReadinessBadge entry={undefined} />);
    const badge = container.querySelector(".dataset-doctor-badge");
    expect(badge.className).toContain("tone-pending");
    expect(badge.getAttribute("title")).toBe("Not assessed yet");
  });

  it("is pending while loading", () => {
    mount(<ReadinessBadge entry={undefined} loading />);
    expect(container.querySelector(".dataset-doctor-badge").className).toContain("tone-pending");
  });

  it("is good for a covered, clean item", () => {
    mount(<ReadinessBadge entry={{ severity: undefined, flags: [] }} />);
    expect(container.querySelector(".dataset-doctor-badge").className).toContain("tone-good");
  });

  it("surfaces the worst-flag reason for a warned item", () => {
    mount(<ReadinessBadge entry={{ severity: "warn", flags: [{ check: "blur", severity: "warn" }] }} />);
    const badge = container.querySelector(".dataset-doctor-badge");
    expect(badge.className).toContain("tone-warn");
    expect(badge.getAttribute("title")).toContain("blurry");
  });

  it("is fatal for an untrainable item", () => {
    mount(<ReadinessBadge entry={{ severity: "fatal", flags: [{ check: "decode", severity: "fatal" }] }} />);
    expect(container.querySelector(".dataset-doctor-badge").className).toContain("tone-fatal");
  });
});

describe("ReadinessFlagDetails (Advanced raw numbers)", () => {
  it("renders nothing for a clean item", () => {
    mount(<ReadinessFlagDetails entry={{ flags: [] }} />);
    expect(container.querySelector(".dataset-doctor-flags")).toBeNull();
  });

  it("shows the raw value and threshold behind a flag", () => {
    mount(
      <ReadinessFlagDetails
        entry={{ flags: [{ check: "blur", severity: "warn", value: 42, threshold: 100 }] }}
      />,
    );
    expect(container.textContent).toContain("blurry");
    expect(container.textContent).toContain("sharpness");
    expect(container.textContent).toContain("42");
    expect(container.textContent).toContain("threshold 100");
  });

  it("keeps sub-one fractions readable (e.g. clip/crop)", () => {
    mount(
      <ReadinessFlagDetails
        entry={{ flags: [{ check: "exposure", severity: "warn", value: 0.0734, threshold: 0.05 }] }}
      />,
    );
    expect(container.textContent).toContain("0.073");
  });

  it("offers Dismiss on an active finding and calls back to acknowledge it", () => {
    const onToggle = vi.fn();
    mount(
      <ReadinessFlagDetails
        entry={{ flags: [{ check: "blur", severity: "warn", value: 40, threshold: 100 }] }}
        onToggle={onToggle}
      />,
    );
    const button = container.querySelector(".dataset-doctor-flag-toggle");
    expect(button.textContent).toBe("Dismiss");
    act(() => button.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    expect(onToggle).toHaveBeenCalledWith("blur", true);
  });

  it("shows caption alignment reason, cosine metric, and dismiss callback", () => {
    const onToggle = vi.fn();
    mount(
      <ReadinessFlagDetails
        entry={{
          flags: [
            {
              check: "caption_alignment",
              severity: "warn",
              value: 0.12,
              threshold: 0.2,
            },
          ],
        }}
        onToggle={onToggle}
      />,
    );
    expect(container.textContent).toContain("Caption may not match this image");
    expect(container.textContent).toContain("caption-image cosine similarity");
    expect(container.textContent).toContain("0.12");
    expect(container.textContent).toContain("threshold 0.2");
    const button = container.querySelector(".dataset-doctor-flag-toggle");
    act(() => button.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    expect(onToggle).toHaveBeenCalledWith("caption_alignment", true);
  });

  it("shows a dismissed finding struck-through with an Undo affordance", () => {
    const onToggle = vi.fn();
    mount(
      <ReadinessFlagDetails
        entry={{ flags: [{ check: "blur", severity: "warn", value: 40, threshold: 100, acknowledged: true }] }}
        onToggle={onToggle}
      />,
    );
    const li = container.querySelector(".dataset-doctor-flags li");
    expect(li.className).toContain("dismissed");
    const button = container.querySelector(".dataset-doctor-flag-toggle");
    expect(button.textContent).toBe("Undo");
    act(() => button.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    expect(onToggle).toHaveBeenCalledWith("blur", false);
  });

  it("does not offer to dismiss a non-acknowledgeable finding (decode)", () => {
    mount(
      <ReadinessFlagDetails entry={{ flags: [{ check: "decode", severity: "warn" }] }} onToggle={() => {}} />,
    );
    expect(container.querySelector(".dataset-doctor-flag-toggle")).toBeNull();
  });
});

describe("DatasetDoctorDistributions", () => {
  it("renders nothing without distributions", () => {
    mount(<DatasetDoctorDistributions report={{ gate: "ready" }} />);
    expect(container.querySelector(".dataset-doctor-distributions")).toBeNull();
  });

  it("draws a histogram per metric with the threshold marked", () => {
    mount(
      <DatasetDoctorDistributions
        report={{
          distributions: {
            blurVariance: { values: [10, 200, 400, 50], threshold: 100, higherIsBetter: true },
            shadowClip: { values: [0, 0.01, 0.2], threshold: 0.05, higherIsBetter: false },
            highlightClip: { values: [0, 0, 0], threshold: 0.05, higherIsBetter: false },
          },
        }}
      />,
    );
    expect(container.querySelectorAll(".dataset-doctor-histogram").length).toBe(3);
    expect(container.querySelectorAll(".dataset-doctor-histogram-threshold").length).toBe(3);
    expect(container.textContent).toContain("Sharpness");
    expect(container.textContent).toContain("higher is better");
  });
});

describe("NearDuplicateClusters (sc-8564)", () => {
  const similarReport = () => report({
    items: [
      { itemId: "a", flags: [{ check: "near_duplicate_embedding", severity: "warn", value: 0.98, peers: ["b"] }] },
      { itemId: "c", flags: [{ check: "near_duplicate_embedding", severity: "warn", value: 0.97, peers: ["d"] }] },
    ],
  });

  it("allows manual selection in similarity-only groups and deletes only the selected row's images", async () => {
    let finishDelete;
    const onDeleteSelected = vi.fn(() => new Promise((resolve) => { finishDelete = resolve; }));
    mount(<NearDuplicateClusters report={similarReport()} onDeleteSelected={onDeleteSelected} />);
    const rows = container.querySelectorAll(".dataset-doctor-nearup");
    const deleteButtons = [...rows].map((row) => row.querySelector("button"));
    expect(deleteButtons.every((button) => button.disabled)).toBe(true);
    act(() => rows[0].querySelector('input[aria-label="Select b"]').click());
    act(() => rows[1].querySelector('input[aria-label="Select c"]').click());
    expect(deleteButtons[0].textContent).toBe("Delete selected (1)");
    act(() => deleteButtons[0].click());
    expect(onDeleteSelected).toHaveBeenCalledExactlyOnceWith(["b"]);
    expect(deleteButtons[0].textContent).toBe("Deleting…");
    expect(rows[0].querySelector("input").disabled).toBe(true);
    act(() => deleteButtons[0].click());
    expect(onDeleteSelected).toHaveBeenCalledTimes(1);
    await act(async () => finishDelete(true));
    expect(rows[0].querySelector('input[aria-label="Select b"]').checked).toBe(false);
    expect(rows[1].querySelector('input[aria-label="Select c"]').checked).toBe(true);
  });

  it("keeps selections available for retry after failed deletion and respects the dataset busy state", async () => {
    const onDeleteSelected = vi.fn().mockRejectedValueOnce(new Error("Save failed")).mockResolvedValue(false);
    const props = { report: similarReport(), onDeleteSelected };
    mount(<NearDuplicateClusters {...props} />);
    const input = container.querySelector("input");
    act(() => input.click());
    await act(async () => container.querySelector("button").click());
    expect(container.querySelector('[role="alert"]').textContent).toBe("Save failed");
    expect(input.checked).toBe(true);
    await act(async () => container.querySelector("button").click());
    expect(input.checked).toBe(true);
    mount(<NearDuplicateClusters {...props} busy />);
    expect(input.disabled).toBe(true);
    expect(container.querySelector("button").disabled).toBe(true);
  });

  it("shows a 4x preview on hover or keyboard focus, stays in the viewport, and dismisses it", () => {
    mount(<NearDuplicateClusters report={similarReport()} onDeleteSelected={vi.fn()}
      renderThumbnail={(id) => <img src={`/${id}.png`} alt="" />} />);
    const thumb = container.querySelector(".dataset-doctor-nearup-thumb");
    thumb.getBoundingClientRect = () => ({ left: window.innerWidth - 56, top: window.innerHeight - 56, width: 48, bottom: window.innerHeight - 8 });
    act(() => thumb.dispatchEvent(new MouseEvent("mouseover", { bubbles: true })));
    const preview = document.body.querySelector(".dataset-doctor-nearup-preview");
    expect(preview.querySelector("img").getAttribute("src")).toBe("/a.png");
    expect(preview.style.width).toBe("192px");
    expect(preview.style.height).toBe("192px");
    expect(parseFloat(preview.style.left) + 192).toBeLessThanOrEqual(window.innerWidth - 8);
    expect(parseFloat(preview.style.top) + 192).toBeLessThan(window.innerHeight - 56);
    // The portal escapes the thumbnail's clipping box.
    expect(preview.parentElement).toBe(document.body);
    act(() => thumb.dispatchEvent(new MouseEvent("mouseout", { bubbles: true })));
    expect(document.body.querySelector(".dataset-doctor-nearup-preview")).toBeNull();
    act(() => thumb.querySelector("input").focus());
    expect(document.body.querySelector(".dataset-doctor-nearup-preview")).not.toBeNull();
    act(() => thumb.querySelector("input").dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
    expect(document.body.querySelector(".dataset-doctor-nearup-preview")).toBeNull();
    act(() => thumb.dispatchEvent(new MouseEvent("mouseover", { bubbles: true })));
    act(() => window.dispatchEvent(new Event("scroll")));
    expect(document.body.querySelector(".dataset-doctor-nearup-preview")).toBeNull();
  });

  it("renders each cluster's sibling thumbnails, similarity label, and a scoped remove action", () => {
    const onRemoveDuplicates = vi.fn();
    const r = report({
      items: [
        { itemId: "a", flags: [{ check: "near_duplicate", severity: "warn", value: 2, peers: ["b"] }] },
        { itemId: "b", flags: [{ check: "near_duplicate", severity: "warn", value: 2, peers: ["a"] }] },
        { itemId: "e", flags: [{ check: "near_duplicate_embedding", severity: "warn", value: 0.96, peers: ["f"] }] },
        { itemId: "f", flags: [{ check: "near_duplicate_embedding", severity: "warn", value: 0.96, peers: ["e"] }] },
      ],
      duplicateRemoval: { groups: [{ keep: "a", remove: ["b"] }] },
    });
    mount(
      <NearDuplicateClusters
        report={r}
        renderThumbnail={(itemId) => <span className="test-thumb" data-item={itemId} />}
        onRemoveDuplicates={onRemoveDuplicates}
      />,
    );

    // Two clusters, each labeled by its relationship kind.
    expect(container.querySelectorAll(".dataset-doctor-nearup")).toHaveLength(2);
    expect(container.textContent).toContain("2 nearly identical photos");
    expect(container.textContent).toContain("2 very similar photos (96% similar)");
    // Every member of every cluster gets a sibling thumbnail via the render-prop.
    expect(container.querySelectorAll(".test-thumb")).toHaveLength(4);

    // Only the pHash cluster (server-dedupable) shows a remove button; the CLIP cluster is variety
    // and shows none. The button passes THIS cluster's removable subset, not the global list.
    const removeButtons = container.querySelectorAll("button");
    expect(removeButtons).toHaveLength(1);
    act(() => removeButtons[0].click());
    expect(onRemoveDuplicates).toHaveBeenCalledWith(["b"]);
  });

  it("renders nothing when there are no near-duplicate clusters", () => {
    mount(<NearDuplicateClusters report={report()} renderThumbnail={() => null} />);
    expect(container.querySelector(".dataset-doctor-nearups")).toBeNull();
  });
});
