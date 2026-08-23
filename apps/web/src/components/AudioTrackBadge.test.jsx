// sc-19577 — a user can TELL that a render has a soundtrack, without pressing play.
//
// sc-17161 unmuted every player, which made generated audio audible. It could not finish its own
// acceptance criterion ("the user can tell the audio is generated (not silent)") because no asset
// record carried the fact: the worker measured nothing and no field existed to render. This suite
// covers the last link of that chain — the indicator — and asserts each of the three states against
// a fixture that genuinely has it.
//
// The trap being avoided is explicit in the story: a test that only checks the `true` case passes
// against a component hard-coded to render the badge always, and a test that only checks `false`
// passes against one that renders it never. So all three are pinned, and the negative cases are
// paired with a positive one built from the SAME fixture so the assertion is discriminating on the
// field rather than on some unrelated absence.

import React from "react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { AudioTrackBadge, assetHasAudioTrack } from "./AudioTrackBadge.jsx";
import { AssetCard, AssetDetail } from "./assetPanels.jsx";

// A generated MiniMax-H3 clip. `hasAudio` is spliced in per case so every fixture below differs in
// exactly that one key.
const clip = (hasAudio) => {
  const file = {
    path: "asset_1.mp4",
    mimeType: "video/mp4",
    width: 1344,
    height: 768,
    duration: 5.1667,
    fps: 24,
    frameCount: 124,
  };
  if (hasAudio !== undefined) {
    file.hasAudio = hasAudio;
  }
  return {
    id: "asset_1",
    type: "video",
    projectId: "project_1",
    displayName: "a lighthouse keeper hums",
    generationSetId: "genset_1",
    file,
    status: {},
    recipe: { model: "minimax_h3", prompt: "a lighthouse keeper hums" },
  };
};

const noop = () => {};

describe("assetHasAudioTrack", () => {
  it("reads the recorded measurement and refuses to guess", () => {
    expect(assetHasAudioTrack(clip(true))).toBe(true);
    expect(assetHasAudioTrack(clip(false))).toBe(false);
    // Absent — every asset generated before sc-19577. NOT false: claiming an existing LTX render
    // is silent would be a fresh lie about a file on disk.
    expect(assetHasAudioTrack(clip(undefined))).toBe(null);
    expect(assetHasAudioTrack(null)).toBe(null);
    expect(assetHasAudioTrack({ type: "video" })).toBe(null);
    // A non-boolean is not a measurement. A truthy string must not light the badge.
    expect(assetHasAudioTrack({ file: { hasAudio: "yes" } })).toBe(null);
    expect(assetHasAudioTrack({ file: { hasAudio: 1 } })).toBe(null);
  });
});

describe("AudioTrackBadge (sc-19577)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
  });

  const render = (node) => act(() => root.render(node));

  it("renders for a measured soundtrack and for nothing else", async () => {
    await render(<AudioTrackBadge asset={clip(true)} />);
    expect(container.querySelector(".audio-track-badge")).not.toBeNull();
    expect(container.querySelector(".audio-track-badge").getAttribute("aria-label")).toBe(
      "Has audio",
    );

    // MUTATION: strip the field from the same fixture — the indicator must DISAPPEAR rather than
    // default to claiming audio.
    await render(<AudioTrackBadge asset={clip(undefined)} />);
    expect(container.querySelector(".audio-track-badge")).toBeNull();

    // A measured-silent render is not a soundtrack either.
    await render(<AudioTrackBadge asset={clip(false)} />);
    expect(container.querySelector(".audio-track-badge")).toBeNull();

    // …and back, so the two assertions above discriminate on `hasAudio` and not on the harness.
    await render(<AudioTrackBadge asset={clip(true)} />);
    expect(container.querySelector(".audio-track-badge")).not.toBeNull();
  });

  it("is never inferred from the model id", async () => {
    // The whole point of the field: the SAME MiniMax-H3 checkpoint produces clips with and without
    // a soundtrack, so a family lookup would badge a silent t2va render. `recipe.model` says
    // minimax_h3 in every fixture here; only `file.hasAudio` moves.
    const silent = clip(false);
    expect(silent.recipe.model).toBe("minimax_h3");
    await render(<AudioTrackBadge asset={silent} />);
    expect(container.querySelector(".audio-track-badge")).toBeNull();
  });

  it("shows on the review grid card", async () => {
    await render(
      <AssetCard
        asset={clip(true)}
        deleteAsset={noop}
        purgeAsset={noop}
        onPreview={noop}
        updateAssetStatus={noop}
      />,
    );
    expect(container.querySelector(".review-card .audio-track-badge")).not.toBeNull();

    await render(
      <AssetCard
        asset={clip(undefined)}
        deleteAsset={noop}
        purgeAsset={noop}
        onPreview={noop}
        updateAssetStatus={noop}
      />,
    );
    expect(container.querySelector(".review-card .audio-track-badge")).toBeNull();
  });

  it("shows on the asset detail, and spells out a measured-silent render there", async () => {
    const detail = (asset) => (
      <AssetDetail
        asset={asset}
        deleteAsset={noop}
        purgeAsset={noop}
        onPreview={noop}
        onSendImage={noop}
        onSendVideo={noop}
        onSendEditor={noop}
        updateAssetStatus={noop}
        updateAssetTags={noop}
        availableTags={[]}
      />
    );
    const audioRow = () =>
      Array.from(container.querySelectorAll(".asset-detail dl div")).find(
        (row) => row.querySelector("dt")?.textContent === "Audio",
      );

    await render(detail(clip(true)));
    expect(container.querySelector(".asset-detail .audio-track-badge")).not.toBeNull();
    expect(audioRow()?.querySelector("dd").textContent).toBe("Yes");

    // Measured and silent: no badge, but the detail panel SAYS so — this is the surface with room
    // to distinguish "we know it is silent" from "we never measured".
    await render(detail(clip(false)));
    expect(container.querySelector(".asset-detail .audio-track-badge")).toBeNull();
    expect(audioRow()?.querySelector("dd").textContent).toBe("No");

    // Never measured: the row is absent entirely, so a legacy asset makes no claim either way.
    await render(detail(clip(undefined)));
    expect(container.querySelector(".asset-detail .audio-track-badge")).toBeNull();
    expect(audioRow()).toBeUndefined();
  });
});
