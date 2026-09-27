import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { useAudioTakePlayer } from "../components/audioTakeParts.jsx";
import { audioAssetRunGroups } from "../audioTakes.js";
import { SimpleAudioDeck } from "./simpleAudioParts.jsx";

// sc-23000: the Simple UI's play deck for a YuE2 take. A YuE2 song replays only from the Song Lab
// (the generic audio route refuses it), so the deck offers no "Run again"; and its download carries
// the licence-marked name, never the style text.

const POLICY = { nonCommercial: true, experimental: true, license: { notice: "(CC BY-NC 4.0)" } };
const YUE2_ASSET = {
  id: "asset_song",
  type: "audio",
  projectId: "p",
  displayName: "dream pop, airy female vocal",
  file: { path: "a.wav", duration: 12 },
  recipe: { model: "yue2", prompt: "dream pop" },
  extra: { yue2: { kind: "create" }, usagePolicy: POLICY },
};
const KOKORO = { id: "kokoro_82m", name: "Kokoro", audio: { voices: [{ id: "af_heart" }] } };
const SPEECH_ASSET = {
  id: "asset_speech",
  type: "audio",
  projectId: "p",
  displayName: "Take 1",
  file: { path: "b.wav", duration: 4 },
  recipe: { model: "kokoro_82m", prompt: "hello", normalizedSettings: { voice: "af_heart" } },
};
const YUE2_MODEL = { id: "yue2", name: "YuE2", audio: { supportsSymbolicSong: true, sampleRates: [48000] } };

function Deck({ asset, onRunAgain }) {
  const player = useAudioTakePlayer();
  const run = audioAssetRunGroups([asset], [KOKORO, YUE2_MODEL])[0];
  return <SimpleAudioDeck asset={asset} breakpoint="desktop" onRunAgain={onRunAgain} player={player} run={run} takeIndex={0} />;
}

describe("Simple audio deck — YuE2 takes (sc-23000)", () => {
  let container;
  let root;
  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
  });
  afterEach(async () => {
    await unmountRoot(root, container);
  });

  const runAgain = () =>
    [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === "Run again");

  it("offers no Run again for a YuE2 take and downloads it under the licence-marked name", async () => {
    await act(async () => root.render(<Deck asset={YUE2_ASSET} onRunAgain={vi.fn()} />));
    expect(runAgain()).toBeUndefined();
    expect(container.querySelector("a[download]").getAttribute("download")).toBe(
      "yue2-song-asset_song-noncommercial.wav",
    );
  });

  it("still offers Run again for a replayable standard take", async () => {
    await act(async () => root.render(<Deck asset={SPEECH_ASSET} onRunAgain={vi.fn()} />));
    expect(runAgain()).toBeTruthy();
    expect(container.querySelector("a[download]").getAttribute("download")).toBe("Take 1");
  });
});
