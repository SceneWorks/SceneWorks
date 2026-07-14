import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ReplacePersonPanel } from "./ReplacePersonPanel.jsx";
import { changeField, field } from "../main.testSupport.jsx";

// sc-11966 — S7: a background track refresh (SSE / track-job update) must not
// clobber the user's unsaved in-progress per-frame drafts. The inner
// PersonTrackCorrections seed effect used to reseed `drafts` on any
// corrections-signature change for the SAME track id, overwriting dirty edits.
// These tests drive PersonTrackCorrections through ReplacePersonPanel (the
// exported surface) so the fix is exercised end-to-end.
describe("ReplacePersonPanel track corrections drafts (sc-11966)", () => {
  let container;
  let root;

  const frame = (index) => ({
    timestamp: index * 0.5,
    confidence: 0.9,
    box: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 },
    flags: [],
  });

  const track = (id, corrections) => ({
    id,
    name: `Track ${id}`,
    projectId: "project-1",
    sourceAssetId: "clip-1",
    status: {},
    frames: [frame(0), frame(1), frame(2)],
    corrections,
  });

  // Only the props PersonTrackCorrections needs to mount and re-render; the rest
  // of ReplacePersonPanel's surface is inert with these safe defaults.
  const renderPanel = (selectedTrack) =>
    act(() => {
      root.render(
        <ReplacePersonPanel
          detectionResult={null}
          matchingTracks={[]}
          personReadiness={{}}
          personTrackId={selectedTrack.id}
          replacementMode="face_only"
          saveTrackCorrections={vi.fn()}
          selectedDetection={null}
          selectedTrack={selectedTrack}
          setPersonTrackId={() => {}}
          setReplacementMode={() => {}}
          setSelectedDetectionId={() => {}}
          setSourceClipAssetId={() => {}}
          setTrackName={() => {}}
          sourceClipAssetId=""
          trackName=""
          videoAssets={[]}
          videoModels={[]}
        />,
      );
    });

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("initial seed populates the box from the persisted sidecar corrections", async () => {
    await renderPanel(track("track-1", [{ frameIndex: 0, box: { x: 0.65, y: 0.15, width: 0.2, height: 0.25 }, rejected: false }]));

    // First render of a track shows its saved corrections (acceptance #3).
    expect(field(container, "Box x").value).toBe("0.65");
    expect(container.textContent).toContain("1 saved");
  });

  it("keeps dirty per-frame drafts when a same-track corrections refresh arrives", async () => {
    await renderPanel(track("track-1", []));

    // User nudges frame 0's box -> a dirty, unsaved draft.
    await changeField(field(container, "Box x"), "0.55");
    expect(field(container, "Box x").value).toBe("0.55");
    expect(container.textContent).toContain("1 unsaved");

    // A background refresh mutates the SAME track id's corrections (e.g. a
    // track-job update landing a correction on a different frame). This bumps
    // correctionsSignature. Before the fix, the seed effect reseeded `drafts`
    // and clobbered the in-progress edit (acceptance #1).
    await renderPanel(track("track-1", [{ frameIndex: 1, box: { x: 0.7, y: 0.7, width: 0.2, height: 0.2 }, rejected: false }]));

    // The user's unsaved edit on frame 0 survives the refresh.
    expect(field(container, "Box x").value).toBe("0.55");
    expect(container.textContent).toContain("1 unsaved");
  });

  it("reseeds from the new track when switching to a different track id", async () => {
    await renderPanel(track("track-1", []));
    await changeField(field(container, "Box x"), "0.55");
    expect(field(container, "Box x").value).toBe("0.55");

    // Switching to a genuinely different track (new id -> key remount) must
    // reset drafts and seed from that track's corrections (acceptance #2).
    await renderPanel(track("track-2", [{ frameIndex: 0, box: { x: 0.33, y: 0.15, width: 0.2, height: 0.25 }, rejected: false }]));

    expect(field(container, "Box x").value).toBe("0.33");
    expect(container.textContent).toContain("1 saved");
  });

  it("still reseeds on a corrections-signature change while drafts are clean", async () => {
    await renderPanel(track("track-1", []));
    // Do not touch anything: drafts are clean.
    expect(field(container, "Box x").value).toBe("0.1");

    // An external corrections change on a clean panel should stay visible.
    await renderPanel(track("track-1", [{ frameIndex: 0, box: { x: 0.8, y: 0.15, width: 0.2, height: 0.25 }, rejected: false }]));

    expect(field(container, "Box x").value).toBe("0.8");
    expect(container.textContent).toContain("1 saved");
  });
});
