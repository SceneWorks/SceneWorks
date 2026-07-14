import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ReplacePersonPanel } from "./ReplacePersonPanel.jsx";
import { buttonInside, changeField, field, settle } from "../main.testSupport.jsx";

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
  const renderPanel = (selectedTrack, saveTrackCorrections = vi.fn()) =>
    act(() => {
      root.render(
        <ReplacePersonPanel
          detectionResult={null}
          matchingTracks={[]}
          personReadiness={{}}
          personTrackId={selectedTrack.id}
          replacementMode="face_only"
          saveTrackCorrections={saveTrackCorrections}
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

  it("after a save lands, a later external correction on another frame is reflected, not clobbered", async () => {
    // saveTrackCorrections resolves with the updated track (mirrors the hook), so
    // a successful save re-baselines the drafts to the just-saved clean state.
    const saveTrackCorrections = vi.fn(() => Promise.resolve(track("track-1", [])));

    await renderPanel(track("track-1", []), saveTrackCorrections);

    // User edits frame 0's box -> a dirty, unsaved draft, then saves it.
    await changeField(field(container, "Box x"), "0.55");
    expect(container.textContent).toContain("1 unsaved");
    await act(async () => {
      buttonInside(container, "Save corrections").click();
    });
    await settle();
    expect(saveTrackCorrections).toHaveBeenCalledWith("track-1", [
      { frameIndex: 0, box: { x: 0.55, y: 0.2, width: 0.3, height: 0.4 }, rejected: false, author: "ui", source: "manual" },
    ]);

    // The save LANDS: the parent refetch replaces the track, so the persisted
    // corrections now equal the user's draft on frame 0 (same track id -> no
    // remount). Drafts must read clean against the just-saved corrections.
    await renderPanel(
      track("track-1", [{ frameIndex: 0, box: { x: 0.55, y: 0.2, width: 0.3, height: 0.4 }, rejected: false }]),
      saveTrackCorrections,
    );
    expect(field(container, "Box x").value).toBe("0.55");
    expect(container.textContent).toContain("1 saved");

    // Now an externally-landed correction arrives on a DIFFERENT frame (frame 1)
    // while drafts are clean post-save. Before the post-save fix this reseed was
    // skipped (drafts still measured "dirty" vs the pre-edit seed): frame 1 was
    // hidden AND the still-enabled Save would have dropped it (data loss).
    await renderPanel(
      track("track-1", [
        { frameIndex: 0, box: { x: 0.55, y: 0.2, width: 0.3, height: 0.4 }, rejected: false },
        { frameIndex: 1, box: { x: 0.7, y: 0.7, width: 0.2, height: 0.2 }, rejected: false },
      ]),
      saveTrackCorrections,
    );

    // The panel converged to both corrections: clean ("2 saved"), so the next Save
    // would NOT drop the external frame-1 correction.
    expect(container.textContent).toContain("2 saved");
    // And the externally-added frame-1 box is visible when scrubbed to.
    await changeField(field(container, "Scrub tracking frames"), "1");
    expect(field(container, "Box x").value).toBe("0.7");
  });

  it("a no-op touched draft (reject on->off) reads clean, so a concurrent external correction is reflected, not hidden or dropped", async () => {
    await renderPanel(track("track-1", []));

    const rejectCheckbox = () => container.querySelector('.person-correction-reject input[type="checkbox"]');

    // Frame 0 starts clean at the tracked box.
    expect(field(container, "Box x").value).toBe("0.1");
    expect(container.textContent).toContain("0 saved");

    // User toggles Reject ON (a meaningful, dirty draft) ...
    await act(async () => {
      rejectCheckbox().click();
    });
    expect(container.textContent).toContain("1 unsaved");

    // ... then changes their mind and toggles it OFF. The draft is now a NO-OP
    // touched entry (box unchanged, not rejected): the display correctly reads
    // clean. The seed effect must agree it is clean — the bug was that it measured
    // dirtiness on the RAW drafts object (a lingering no-op entry != the seed),
    // so it treated this as dirty and skipped reseeding.
    await act(async () => {
      rejectCheckbox().click();
    });
    expect(container.textContent).toContain("0 saved");

    // A concurrent external correction lands on a DIFFERENT frame (frame 1) on the
    // SAME track id (no remount). Because the no-op draft reads clean, the seed
    // effect reseeds and the external correction is shown/converged. Pre-fix it
    // was HIDDEN (seed skipped) and the next Save would have posted only [] —
    // dropping the external frame-1 correction (regression vs. main).
    await renderPanel(track("track-1", [{ frameIndex: 1, box: { x: 0.7, y: 0.7, width: 0.2, height: 0.2 }, rejected: false }]));

    // Converged: clean "1 saved" (not "0 unsaved"), so a subsequent Save posts the
    // external frame-1 correction instead of dropping it.
    expect(container.textContent).toContain("1 saved");
    // And the externally-added frame-1 box is visible when scrubbed to.
    await changeField(field(container, "Scrub tracking frames"), "1");
    expect(field(container, "Box x").value).toBe("0.7");
  });
});
