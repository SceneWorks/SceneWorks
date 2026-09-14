// sc-17161 — a generated audio track has to be AUDIBLE.
//
// `AssetMedia` is the single seam every video surface in the app renders through — asset detail,
// the review grid, the fullscreen preview, the queue card's output player, the Simple preview,
// the A/B compare. It hard-set `muted` on the <video> it produced and no call site overrode it,
// so every clip in SceneWorks played silent.
//
// That was invisible while every video model produced a video-only mp4. MiniMax-H3 is a joint
// audio+video family — `text_to_video_audio`, `first_last_frame_video_audio`, Ref2VA — whose mp4
// carries an AAC track (`crates/sceneworks-worker/src/video_jobs/mod.rs` muxes `-c:a aac` from
// the pipeline's audio, the same path LTX already used). A hard-muted player makes a model that
// generated sound indistinguishable from one that did not, which is the whole point of the family.

import React from "react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { AssetMedia } from "./assetMedia.jsx";

const clip = (type = "video") => ({
  id: "asset_1",
  type,
  projectId: "project_1",
  displayName: "a rendered clip",
  file: { path: "asset_1.mp4" },
});

describe("AssetMedia playback audio (sc-17161)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
  });

  const draw = async (element) => {
    await act(async () => root.render(element));
  };

  it("renders a user-playable video UNMUTED", async () => {
    // The user-initiated surfaces — asset detail, preview, the results zone — all render this
    // element with native controls on. Pressing play must produce the sound the model generated.
    await draw(<AssetMedia asset={clip()} />);
    const video = container.querySelector("video");
    expect(video).toBeTruthy();
    expect(video.controls, "the user drives playback here").toBe(true);
    expect(video.muted, "a joint audio+video render must not play silent").toBe(false);
    expect(video.hasAttribute("muted")).toBe(false);
  });

  it("still lets a surface mute deliberately", async () => {
    // `muted` became a PROP rather than being deleted: a surface that drives playback from script
    // instead of from a user gesture (the Replace Person frame scrubber) needs it, because
    // autoplay policy blocks an unmuted programmatic play(). Preserving that is what keeps this a
    // per-surface decision instead of a new blanket in the other direction.
    await draw(<AssetMedia asset={clip()} controls={false} muted />);
    expect(container.querySelector("video").muted).toBe(true);
  });

  it("never mutes the audio branch", async () => {
    // The <audio> element the Audio Studio's transport drives was never muted; the default must
    // not have leaked a mute onto it.
    await draw(<AssetMedia asset={{ ...clip("audio"), file: { path: "take.wav" } }} />);
    const audio = container.querySelector("audio");
    expect(audio).toBeTruthy();
    expect(audio.muted).toBe(false);
  });
});
