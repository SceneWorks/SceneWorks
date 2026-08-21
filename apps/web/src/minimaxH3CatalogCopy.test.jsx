// sc-17162 — the catalog copy for the MiniMax-H3 family must disclose the WITHHELD upstream
// components, and that disclosure has to reach a real screen.
//
// Three upstream pieces of MiniMax's own stack ship with the hosted Hailuo product and NOT with
// these weights — `H3-Context-IR` (prompt understanding), `H3-Regenerate-2K` (2K upsampling) and
// the sparse-attention inference path — plus a fourth surface, the `<d>` dialogue markers, which
// the model card's own examples use and which resolve to untrained embedding rows here. The engine
// side of the story is recorded in inference `docs/reference/minimax-h3-withheld-upstream-components.md`;
// this file is the SceneWorks half: it pins that the product copy says so.
//
// Why it is pinned rather than trusted:
//
//   1. This epic's recurring defect is copy that outruns the code ("declared but undrivable" —
//      `limits.hardMinSteps` shipped with no web reader, `referenceAudioAssetIds` with no control
//      that could set it). The inverse is just as bad and is what this file guards: a description
//      that quietly drops the caveats while the weights keep their limits.
//   2. `ui.description` has exactly ONE reader (`ModelManagerScreen.jsx` model card) and
//      `ui.durationHint` exactly one (`VideoStudio.jsx` helper copy). A single reader is a single
//      point of silent loss, so the last `describe` renders the card and reads the text back out
//      of the DOM rather than trusting the manifest string alone.
//   3. `fallbackModels` (constants.js) is a MIRROR the studios and the Models screen render BEFORE
//      /api/v1/models answers. Disclosure that only arrives once the network does is not
//      disclosure — the same argument sc-17161 made for the licence attribution.
//
// SET-DERIVED, not id-listed: every assertion below is driven off `family === "minimax-h3"` in the
// shipped manifest, and the prompt-guide path is read off `ui.promptGuide.path` rather than typed.
// A hard-coded pair would let a new partition ship with no disclosure while this file stayed green —
// which is precisely the failure it exists to catch.
//
// What that does NOT mean, and what this comment used to claim: a third partition is *not* "covered
// the day it lands". `covers both shipped partitions of the family` asserts the id set EXACTLY
// (`toEqual(["minimax_h3", "minimax_h3_ref"])`), so a third partition REDS that test on arrival even
// if its copy is perfect. That is deliberate — it is the anti-vacuity guard, and a family rename
// would otherwise empty every loop below and turn the whole file green while disclosing nothing —
// but it means a third partition is BLOCKED until someone widens that list by hand. The set-derived
// loops are what make widening it a one-line change instead of an audit.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import { createRoot } from "react-dom/client";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { setInput } from "./testUtils/dom.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");
const PUBLIC_DIR = resolve(HERE, "../public");

const manifestModels = (() => {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  return Array.isArray(parsed) ? parsed : parsed.models;
})();

const FAMILY = "minimax-h3";
const familyEntries = manifestModels.filter((model) => model.family === FAMILY);

// The four claims the disclosure is made of. Each is a SEPARATE assertion so a mutation that drops
// one clause fails on that clause's own name rather than on a single opaque "copy changed".
//
// Every pattern is checkable against something concrete:
//   `contextIr`   — the component is unreleased; the engine does no prompt rewriting at all.
//   `noTwoK`      — `max_size: 1344` per edge is ENFORCED (`resolve_geometry`), not just declared,
//                   and `limits.maxPixels` here is 1,032,192 = 768 x 1344.
//   `denseAttention` + `twoHourCost` — no sparse-attention path, and the measured 50-step cost of
//                   the SHORTEST clip at the shipped default canvas (1344x768 @ 124 f) is ~7 394 s.
//   `fifteenRefused` — `limits.hardMaxDuration` is 14.375 and the next `17n + 5` rung is 15.083 s,
//                   so a flat 15 s request has no lattice point and ERRORS. It is not clamped, and
//                   a user planning around MiniMax's advertised "5–15 s" must learn that here.
//   `noStillImage` — `min_duration = 5.0` is hardcoded upstream, so `T = 1` does not render.
const DESCRIPTION_CLAIMS = [
  { name: "names the withheld prompt front end", re: /H3-Context-IR/ },
  { name: "names the withheld 2K upsampler", re: /H3-Regenerate-2K/ },
  { name: "says 2K is unreachable from these weights", re: /2K is not reachable from these weights/ },
  { name: "says attention runs dense", re: /attention runs dense/ },
  { name: "gives the two-hour default-canvas cost", re: /two-hour render/ },
  { name: "says 15s is refused, not shortened", re: /\b15s\b[^.]{0,120}\brefused\b/ },
  { name: "says there is no still-image mode", re: /no still-image mode/ },
];

// The duration hint is a different field on a different screen (Video Studio), so it carries the
// refusal independently — a user who never opens the Models screen still has to be told.
//
// Deliberately ONE claim: a `/5\.17s/` or `/14\.38s/` literal looked like it pinned the bounds and
// did not. Both hints quote those figures again in the cost sentence ("5.17s at 576x320 is ~14
// minutes", "the whole 14.38s range"), so a mutation that removed them from the RANGE clause left
// the regex matching on the leftovers — an assertion that survives its own mutation. The bounds are
// pinned instead by `hintQuotesTheDeclaredBounds` below, which derives them from `limits` so the
// copy is checked against the numbers the engine enforces rather than against itself.
const DURATION_HINT_CLAIMS = [
  { name: "says 15s is refused, not shortened", re: /\b15s is refused\b[^.]{0,60}\bshortened\b/ },
];

// The drift this actually guards: `limits.hardMinDuration` / `hardMaxDuration` move (a lattice
// correction, a new clamp) and the prose keeps quoting the old window. Derived from the entry, so
// changing a bound without changing the copy is red.
//
// ⚠️ THE TWO `includes` CHECKS ARE NOT SUFFICIENT ON THEIR OWN and were once claimed to be. They are
// independent substring tests with no adjacency, and every hint quotes BOTH figures a second time in
// its cost sentence ("5.17s at 576x320 is ~14 minutes", "the whole 14.38s range"). Delete the range
// clause outright and both substrings survive on the leftovers, so the assertion passes over copy
// that no longer states a range at all — an assertion that cannot fail its own mutation.
//
// `RANGE` is the anchor that closes that: it requires the two bounds ADJACENT, separated only by a
// range connector, which the cost sentence's leftovers can never satisfy. Still derived from
// `limits` rather than typed, so moving a bound without moving the copy is red for the same reason
// the substring checks were meant to be. The connector set is deliberately small but covers both
// shipped spellings — the manifest writes "5.17s to 14.38s", the `fallbackModels` mirror condenses
// it to "5.17s-14.38s".
function hintQuotesTheDeclaredBounds(hint, entry, label) {
  const quoted = {};
  for (const [name, seconds] of [
    ["hardMinDuration", entry.limits?.hardMinDuration],
    ["hardMaxDuration", entry.limits?.hardMaxDuration],
  ]) {
    expect(Number.isFinite(seconds), `${label} ${name} must be declared`).toBe(true);
    quoted[name] = seconds.toFixed(2);
    expect(hint.includes(`${quoted[name]}s`), `${label} must quote ${name} (${quoted[name]}s)`).toBe(true);
  }
  const escape = (value) => value.replace(/\./g, "\\.");
  const CONNECTOR = "\\s*(?:to|-|\u2013|\u2014)\\s*"; // "to", hyphen, en dash, em dash
  const range = new RegExp(`${escape(quoted.hardMinDuration)}s${CONNECTOR}${escape(quoted.hardMaxDuration)}s`);
  expect(
    range.test(hint),
    `${label} must state the two bounds as an adjacent RANGE (${quoted.hardMinDuration}s to ${quoted.hardMaxDuration}s), not merely mention both figures somewhere`,
  ).toBe(true);
}

function assertClaims(text, claims, label) {
  for (const claim of claims) {
    expect(claim.re.test(text), `${label} ${claim.name}`).toBe(true);
  }
}

describe("MiniMax-H3 catalog copy discloses the withheld upstream components (sc-17162)", () => {
  it("covers both shipped partitions of the family", () => {
    // Guards the whole file against a vacuous pass: a manifest rename of `family` would otherwise
    // empty every loop below and turn this suite green while disclosing nothing.
    expect(familyEntries.map((model) => model.id).sort()).toEqual(["minimax_h3", "minimax_h3_ref"]);
  });

  it("states every withheld-component consequence on each entry's description", () => {
    expect(familyEntries.length).toBeGreaterThan(0);
    for (const entry of familyEntries) {
      const description = entry.ui?.description ?? "";
      assertClaims(description, DESCRIPTION_CLAIMS, `${entry.id} description`);
    }
  });

  it("states the 15s refusal on each entry's duration hint", () => {
    expect(familyEntries.length).toBeGreaterThan(0);
    for (const entry of familyEntries) {
      const hint = entry.ui?.durationHint ?? "";
      assertClaims(hint, DURATION_HINT_CLAIMS, `${entry.id} durationHint`);
      hintQuotesTheDeclaredBounds(hint, entry, `${entry.id} durationHint`);
    }
  });

  it("carries the same disclosure into the offline catalog mirror", async () => {
    // `fallbackModels` is what the Models screen and the studios render before /api/v1/models
    // answers, and on a desktop cold start that window is the user's first look at the card.
    const { fallbackModels } = await import("./constants.js");
    for (const entry of familyEntries) {
      const mirrored = fallbackModels.find((model) => model.id === entry.id);
      expect(mirrored, `${entry.id} must be mirrored in fallbackModels`).toBeTruthy();
      assertClaims(mirrored.ui?.description ?? "", DESCRIPTION_CLAIMS, `${entry.id} mirrored description`);
      assertClaims(mirrored.ui?.durationHint ?? "", DURATION_HINT_CLAIMS, `${entry.id} mirrored durationHint`);
      hintQuotesTheDeclaredBounds(mirrored.ui?.durationHint ?? "", entry, `${entry.id} mirrored durationHint`);
    }
  });
});

describe("MiniMax-H3 prompt guide (sc-17162)", () => {
  // Resolved off the manifest rather than typed, so a path change moves the test with it instead of
  // leaving it asserting against a file nothing serves.
  const guidePaths = [...new Set(familyEntries.map((entry) => entry.ui?.promptGuide?.path))];

  it("is declared once for the whole family and exists on disk", () => {
    expect(guidePaths).toEqual(["/prompt-guides/minimax-h3.md"]);
  });

  const guide = readFileSync(resolve(PUBLIC_DIR, guidePaths[0].replace(/^\//, "")), "utf8");

  it("warns that all seven declared special tokens are inert", () => {
    // sc-17143 measured it: these seven are declared only as strings in `tokenizer_config.json`,
    // and the embedding rows `transformers` assigns them (151669-151675) are statistically
    // indistinguishable from the untrained padding tail — the text-encoder shards are byte-identical
    // to `Qwen/Qwen3-VL-32B-Instruct`, which never trained them. The model card's own examples use
    // `<d>`, so a user WILL copy the syntax unless the guide heads them off. All seven, not just
    // `<d>`: the lyrics/caption/cutoff markers fail the same way and appear in the same examples.
    for (const marker of [
      "<d>",
      "</d>",
      "<|cutoff|>",
      "<|lyrics_start|>",
      "<|lyrics_end|>",
      "<|caption_start|>",
      "<|caption_end|>",
    ]) {
      expect(guide.includes(marker), `guide must name ${marker}`).toBe(true);
    }
    expect(/none of them do anything/i.test(guide), "guide must say the markers are inert").toBe(true);
  });

  // The engine's recorded decision is PASS THROUGH UNCHANGED — no strip, no repair, no submit-time
  // warning — and the guide has to say so without overclaiming, because the WEB layer has three
  // prompt-altering paths the engine knows nothing about. The guide has now carried two wrong
  // versions of this paragraph and each was pinned by a test that agreed with it:
  //
  //   v1 "SceneWorks passes your prompt through **unchanged**" — false. `RefinePromptControl`
  //      (`VideoStudio.jsx:1578`, gated only on `promptless`, which `minimax_h3` never sets) rewrites
  //      it, and `composeStyledPrompt` / the general-preset stack rewrap it at submit
  //      (`VideoStudio.jsx:1274-1279`, `:1357`).
  //   v2 "SceneWorks never alters your prompt on its own … the one exception is Refine" — still
  //      false twice over. Refine does NOT replace what you wrote: it renders the rewrite behind
  //      **Apply** / **Keep original** (`RefinePromptControl.jsx:193-212`), so an unapplied
  //      suggestion changes nothing. And Refine is not "the one exception" — the style fold and the
  //      stack fold both alter the outgoing prompt with no review step at all.
  //   v3 "**Nothing strips or repairs the markers.**" + "a rewrite you do apply may well drop the
  //      markers" — true when written, FALSIFIED BY sc-17162's own refiner. The H3 rewrite path now
  //      post-filters the seven markers out of the model's reply
  //      (`prompt_refine_jobs.rs::strip_untrained_markers`, reached from `finalize_refined_output`
  //      whenever the target model is an H3 partition), so an applied refinement can never contain
  //      one. "May well drop" describes a likelihood; what ships is a GUARANTEE, and the unscoped
  //      "nothing strips" sentence contradicted it outright.
  //
  // So this pins the SHAPE of the correct claim rather than one sentence: each false absolute is
  // asserted ABSENT, and each surviving path is asserted named. A future rewrite that reintroduces
  // any of them reds on that absolute's own assertion.
  //
  // ⚠️ The v3 no-strip forbid is anchored on the BOLD MARKUP (`**Nothing strips`), because the true
  // replacement contains "nothing strips or repairs the markers" as a SUBSTRING — it merely scopes
  // it with "At submit time". A forbid regex on the bare phrase would match the correct copy and
  // could never go green, which is the mirror of the survives-its-own-mutation defect this file has
  // hit three times. The scoping is pinned positively below instead.
  it("states the prompt-alteration story without the three absolutes that were false", () => {
    for (const overclaim of [
      { name: "the v1 unqualified pass-through", re: /passes your prompt through \*\*unchanged\*\*/ },
      { name: "the v2 'never alters your prompt' absolute", re: /never alters your prompt/i },
      { name: "the v2 'the one exception' framing", re: /the one exception/i },
      // Refine is a SUGGESTION until Apply. Copy that says it replaces the prompt outright is the
      // same class of error in the opposite direction.
      { name: "the claim that Refine replaces the prompt outright", re: /replaces what you wrote/i },
      // v3, both halves — the unscoped no-strip absolute and the likelihood framing it implied.
      { name: "the v3 unscoped 'nothing strips' absolute", re: /\*\*Nothing strips or repairs the markers/ },
      { name: "the v3 'may well drop' likelihood framing", re: /may well drop the markers/i },
    ]) {
      expect(overclaim.re.test(guide), `guide must not carry ${overclaim.name}`).toBe(false);
    }
    for (const claim of [
      // The no-strip claim survives, SCOPED to the path where it is still true.
      {
        name: "the no-strip claim scoped to the submit path",
        re: /At submit time nothing strips or repairs the markers/,
      },
      { name: "the markers are not stripped or repaired", re: /not removed, not rewritten and not warned about/ },
      // …and the refine path's strip is stated as the GUARANTEE it is, with both halves named:
      // the instruction in the embedded asset, and the worker-side post-filter that does not
      // depend on the model complying with it.
      { name: "the applied-refinement guarantee", re: /guaranteed marker-free/ },
      { name: "the refiner instruction as the first half", re: /instructed never to write them/ },
      { name: "the worker post-filter as the enforcing half", re: /strips any that survive/ },
      { name: "the Refine button as a prompt-altering path", re: /\*\*Refine\*\* button/ },
      { name: "that Refine needs Apply before it changes anything", re: /until you press \*\*Apply\*\*/ },
      { name: "the Style Catalog fold as a second path", re: /\*\*Style Catalog\*\* entry/ },
      { name: "the preset stack fold as a third path", re: /\*\*preset stack\*\*/ },
      { name: "that the folds have no confirmation step", re: /no confirmation step/ },
      // The pass-through claim survives, but CONDITIONED on all three paths being idle.
      { name: "the conditioned pass-through claim", re: /With no style, no stack and no applied refinement/ },
    ]) {
      expect(claim.re.test(guide), `guide must state ${claim.name}`).toBe(true);
    }
  });

  it("names all four withheld surfaces, not three", () => {
    for (const claim of [
      { name: "H3-Context-IR", re: /H3-Context-IR/ },
      { name: "H3-Regenerate-2K", re: /H3-Regenerate-2K/ },
      { name: "sparse attention", re: /Sparse-attention inference/ },
      { name: "the dialogue markers", re: /dialogue markers/ },
    ]) {
      expect(claim.re.test(guide), `guide must name ${claim.name}`).toBe(true);
    }
  });

  it("routes past-the-canvas at the real upscaler, labelled as an upscale rather than 2K (sc-17162)", () => {
    // DECIDED, and this assertion inverted when it was. It previously FORBADE mentioning an
    // upscale, because whether SceneWorks offered one for this family was an open product question
    // and pointing at one while it was open would have quietly re-advertised the capability the
    // paragraph above withdraws. Michael's ruling (story activity 20258) settled it the other way:
    // the SeedVR2 video-upscale card in Video Studio Advanced already accepts an H3 clip — it is
    // rendered with NO family gate (`VideoStudio.jsx` → `<VideoUpscalePanel>`) — so the old copy
    // created a dead end whose answer was three clicks away on the same screen.
    //
    // What the ruling did NOT concede is the label. `H3-Regenerate-2K` upsampled INSIDE the
    // diffusion; SeedVR2 runs over finished pixels. So the guide may route, and must simultaneously
    // say that what it routes to is a different artifact — otherwise the route re-advertises the
    // withheld component under a new name, which is the failure the old ban existed to prevent.
    expect(
      /an upscale of a sub-2K render, not native 2K/.test(guide),
      "the route must be labelled an upscale rather than 2K",
    ).toBe(true);
    // The route has to be findable: naming the surface, not just the concept.
    expect(/Video upscale/.test(guide), "guide must name the Studio card that does it").toBe(true);
    expect(/SeedVR2/.test(guide), "guide must name the engine behind it").toBe(true);

    // The two factors are 2 and 4 (`VIDEO_UPSCALE_ENGINES` in `VideoUpscalePanel.jsx`), so 2x of
    // the full canvas OVERSHOOTS 2K on both axes and there is no rung that lands on it. A guide
    // that said "upscale to 2K" would be promising a target the control cannot produce.
    expect(
      /larger than 2K on both axes/.test(guide),
      "guide must say the 2x result overshoots 2K rather than reaching it",
    ).toBe(true);
    expect(/2688x1536/.test(guide), "guide must show the arithmetic, not assert it").toBe(true);
    expect(/no ~?1\.5x rung/.test(guide), "guide must say there is no intermediate factor").toBe(true);

    // H3's headline is joint audio+video, and this path demuxes and re-encodes the audio
    // (`video_jobs/seedvr2.rs`). A user picking H3 FOR the sound has to be told before they run it.
    // The length-truncation defect on the same path was fixed in sc-19549, so the caveat is the
    // re-encode alone — overstating it would be its own inaccuracy.
    const audioSentence = guide
      .split(/\n\n+/)
      .find((block) => /re-encode/i.test(block) && /audio/i.test(block));
    expect(audioSentence, "guide must caveat the audio re-encode on the upscale path").toBeTruthy();
    expect(
      /keeps its full length/i.test(audioSentence),
      "the -shortest truncation was fixed (sc-19549) — the caveat must not still claim it",
    ).toBe(true);

    // And it must read as INTERIM. MiniMax committed publicly (HF discussion #39, undated) to
    // open-sourcing `H3-Regenerate-2K` "once this set of technologies becomes stable"; without that
    // line the upscale route reads as the product's identity rather than a stand-in.
    expect(
      /once this set of\s+technologies becomes stable/.test(guide),
      "guide must carry the upstream open-sourcing commitment so the route reads as interim",
    ).toBe(true);
    expect(
      /\bno date\b/i.test(guide),
      "the commitment is undated — the guide must not imply a timeline it does not have",
    ).toBe(true);

    // Unchanged by the ruling: the canvas ceiling is ENFORCED, so 1344 is still the number to plan
    // the render against. The upscale is what happens after that, not a way around it.
    expect(/refused rather than refitted/.test(guide), "guide must say an over-size request is refused").toBe(true);
    expect(
      // Tolerates the markdown blockquote continuation (`\n>   `) the claim wraps across.
      /No render from these weights is natively[\s>]+2K\./.test(guide),
      "the withdrawal of native 2K survives the routing",
    ).toBe(true);
  });

  it("keeps the 14.38s cap a LATTICE ceiling, distinct from wall-clock cost (sc-17162)", () => {
    // Three different things on this page could be mistaken for the reason clips stop at 14.38 s —
    // the dense-attention cost table, the Turbo adapter, and the withheld sparse-attention path —
    // and all three are about WALL CLOCK. The ceiling is the `17n + 5` lattice meeting the
    // checkpoint's own 5-15 s clamp. Every one of the three has to disclaim it in its own section,
    // because a reader who only reads the Turbo section never sees the sparse-attention bullet.
    expect(
      /Wall-clock cost and the length cap are separate things/.test(guide),
      "the cost section must separate the two",
    ).toBe(true);
    expect(
      /Turbo changes how long a render takes;\s*\n?\s*it does not change which lengths exist/.test(guide),
      "the cost section must say Turbo moves the clock, not the cap",
    ).toBe(true);
    expect(
      /the fourteen clip lengths and the 14\.38 s ceiling are\s*\n?\s*properties of the frame lattice/.test(guide),
      "the Turbo section must disclaim the cap on its own",
    ).toBe(true);

    // Turbo is presented as the MITIGATION for the dense table, with its measured figure and its
    // caveat — a cost table with no mitigation next to it reads as the whole story.
    expect(/\b12\.6 minutes\b/.test(guide), "guide must give Turbo's measured wall clock").toBe(true);
    expect(
      /different sample/.test(guide),
      "Turbo's softer-detail caveat must ride with the mitigation, not be dropped for it",
    ).toBe(true);

    // The dense table itself is unchanged: it is the reference schedule and the honest worst case.
    for (const row of ["576x320", "1344x768"]) {
      expect(guide.includes(row), `the dense cost table must still carry ${row}`).toBe(true);
    }
    expect(/~2 hours|~ 2 hours|\*\*~2 hours\*\*/.test(guide), "the two-hour dense figure stays").toBe(true);
  });

  it("offers every canvas the entries advertise, and no more", () => {
    // Found while writing the 2K paragraph: the guide's size table listed seven of the nine
    // advertised buckets, silently dropping the 21:9 pair — so a user reading only the guide would
    // have concluded 1344 px was the long-edge ceiling, which is exactly the over-tight claim this
    // story is meant to prevent. Set-derived off `limits.resolutions`, so the table and the menu
    // cannot drift again in either direction: an advertised bucket the guide never mentions is a
    // capability the copy hides, and a bucket the guide names that the menu refuses is a promise
    // nothing keeps.
    const advertised = new Set(familyEntries.flatMap((entry) => entry.limits?.resolutions ?? []));
    expect(advertised.size).toBeGreaterThan(0);
    for (const bucket of advertised) {
      expect(guide.includes(bucket), `guide must name the advertised bucket ${bucket}`).toBe(true);
    }
    // sc-17162 adds ONE dimension that is deliberately not a render bucket: the 2x upscale of the
    // full canvas. It is in the copy precisely to show that the upscale OVERSHOOTS 2K rather than
    // reaching it, so it must not be silently admitted as a size the model can render — it is
    // excepted by name, and derived from the advertised canvas times the panel's factor so a change
    // to either side is red rather than absorbed.
    const FULL_CANVAS = "1344x768";
    const UPSCALE_FACTOR = 2;
    expect(advertised.has(FULL_CANVAS)).toBe(true);
    const [w, h] = FULL_CANVAS.split("x").map(Number);
    const upscaled = `${w * UPSCALE_FACTOR}x${h * UPSCALE_FACTOR}`;
    expect(advertised.has(upscaled), `${upscaled} must NOT be an advertised render bucket`).toBe(false);
    for (const named of guide.match(/\b\d{3,4}x\d{3,4}\b/g) ?? []) {
      if (named === upscaled) continue;
      expect(advertised.has(named), `guide names ${named}, which no entry advertises`).toBe(true);
    }
    // The excepted dimension is only allowed where it is labelled an upscale result.
    const upscaleBlocks = guide
      .split(/\n\n+/)
      .filter((block) => block.includes(upscaled));
    expect(upscaleBlocks.length).toBeGreaterThan(0);
    for (const block of upscaleBlocks) {
      expect(
        /upscal/i.test(block),
        `${upscaled} appears outside the upscale section: ${block.slice(0, 120)}`,
      ).toBe(true);
    }
  });

  it("no longer carries the 21:9 caveat, because the pin that caused it has moved (sc-19721)", () => {
    // HISTORY, because the shape of this assertion inverted and the reason matters.
    //
    // The 21:9 pair (`1536x672`) is advertised in `limits.resolutions` and has always satisfied the
    // AREA budget — it is 1,032,192 px, byte for byte what `1344x768` is. What refused it was the
    // engine's INDEPENDENT per-edge ceiling, which sat below 1536. inference PR #640 raised that
    // ceiling; the caveat survived #640 merging because SceneWorks runs a PINNED revision and the
    // pin predated it, so this file asserted the caveat was cited against the PIN BUMP (sc-18650)
    // rather than against the PR.
    //
    // sc-19721 is that pin bump: `Cargo.toml` now pins `75d66db5`, which carries #640, and the
    // engine's own `max_size` is `MAX_CANVAS_EDGE = 2016` on both lanes. The refusal is gone, so the
    // caveat is a false claim and is deleted rather than re-pointed at a third event.
    //
    // What keeps this from being a vacuous absence check is that the real guard lives where it can
    // read the engine: `pinned_engine_geometry.rs` asserts every advertised MiniMax-H3 resolution's
    // long edge against the PINNED provider's `capabilities.max_size`. If the pin ever moves back
    // below 1536, that goes red in Rust — the copy cannot silently re-acquire a lie. The three
    // assertions here are the three ways the COPY can go wrong.
    expect(
      /not renderable|pending an engine change/i.test(guide),
      "the 21:9 pair renders at the current pin — the guide must not tell the user it does not",
    ).toBe(false);
    expect(
      /\b(?:pending|awaiting|until|once|when|blocked (?:on|by))\b[^.]{0,120}(?:#640|sc-18650)/is.test(
        guide,
      ),
      "neither #640 nor the pin bump is outstanding any more; framing either as awaited is stale",
    ).toBe(false);
    // The bucket itself must still be DOCUMENTED — deleting the caveat by deleting the row would
    // satisfy both assertions above while removing the thing they are about.
    expect(
      /\b1536x672\b/.test(guide),
      "the 21:9 bucket must still appear in the Sizes table; it is reachable now, not absent",
    ).toBe(true);
    // NO NUMERIC PER-EDGE CEILING IN CATALOG COPY. The resolver's widest output is an engine
    // internal a user never sees, and writing it down dates the page the moment the resolver moves.
    expect(
      /per-edge (?:ceiling|cap)[^.]{0,60}\b\d{3,4}\b/i.test(guide),
      "guide must describe the per-edge ceiling relatively, never as a number",
    ).toBe(false);
  });

  it("does not attribute the 14.38s ceiling to the withheld sparse attention", () => {
    // A live trap on this epic: the ceiling is the checkpoint's own 5-15 s clamp meeting the
    // `17n + 5` lattice (`MINIMAX_H3_LEGAL_FRAME_COUNTS`), and the lattice comes from the VAE's
    // chunking, which sparse attention does not touch. Sparse attention would move the COST of the
    // long end, not the bound. The guide has to keep those apart.
    expect(/It is not what caps the clip at 14\.38 s/.test(guide), "guide must separate cost from the ceiling").toBe(true);
  });
});

describe("the disclosure reaches the Models screen (sc-17162)", () => {
  // `ui.description`'s only reader. Pinning the string in the manifest proves nothing on its own —
  // this reads it back out of the rendered card.
  let container;
  let root;
  let ModelManagerScreen;
  let AppContext;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.__TAURI__ = {
      core: {
        invoke: vi.fn(async (command) => {
          switch (command) {
            case "get_gpu_info":
              return { platform: "macos", devices: [] };
            case "list_credentials":
              return [];
            default:
              return null;
          }
        }),
      },
    };
    vi.resetModules();
    ({ AppContext } = await import("./context/AppContext.js"));
    ({ ModelManagerScreen } = await import("./screens/ModelManagerScreen.jsx"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    delete window.__TAURI__;
    vi.restoreAllMocks();
  });

  async function renderCards(models) {
    await act(async () => {
      root.render(
        <AppContext.Provider
          value={{
            activeProject: null,
            jobs: [],
            loras: [],
            models,
            presets: [],
            jobAction: () => {},
            setActiveView: () => {},
            deleteLora: () => {},
            deleteModel: () => {},
            createModelDownloadJob: () => {},
            createModelConvertJob: () => {},
            createLoraImportJob: () => {},
            createModelImportJob: () => {},
          }}
        >
          <ModelManagerScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    // The Models screen opens on the `image` tab, so a video entry is off-screen until something
    // selects it. Searching is the tab-independent route (it switches to the transient Search
    // Results tab), and it also proves the card is reachable rather than merely constructed.
    const searchBox = container.querySelector('input[type="search"]');
    expect(searchBox, "the Models screen must offer a search box").toBeTruthy();
    await act(async () => {
      setInput(searchBox, models[0].name);
    });
    await act(async () => {});
  }

  it("renders each partition's withheld-component disclosure on its card", async () => {
    for (const entry of familyEntries) {
      await renderCards([entry]);
      const rendered = [...container.querySelectorAll(".model-card-description")]
        .map((node) => node.textContent)
        .join("\n");
      expect(rendered, `${entry.id} must render a description`).not.toBe("");
      assertClaims(rendered, DESCRIPTION_CLAIMS, `${entry.id} rendered card`);
    }
  });
});
