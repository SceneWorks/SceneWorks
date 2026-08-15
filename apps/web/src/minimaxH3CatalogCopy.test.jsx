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
// A third partition added to the family is covered the day it lands. A hard-coded pair would let it
// ship with no disclosure while this file stayed green — which is precisely the failure it exists
// to catch.

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
function hintQuotesTheDeclaredBounds(hint, entry, label) {
  for (const [name, seconds] of [
    ["hardMinDuration", entry.limits?.hardMinDuration],
    ["hardMaxDuration", entry.limits?.hardMaxDuration],
  ]) {
    expect(Number.isFinite(seconds), `${label} ${name} must be declared`).toBe(true);
    expect(hint.includes(`${seconds.toFixed(2)}s`), `${label} must quote ${name} (${seconds.toFixed(2)}s)`).toBe(true);
  }
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
  //
  // So this pins the SHAPE of the correct claim rather than one sentence: the two absolutes are
  // asserted ABSENT, and each surviving path is asserted named. A future rewrite that reintroduces
  // either absolute reds on the absolute's own assertion.
  it("states the prompt-alteration story without the two absolutes that were false", () => {
    for (const overclaim of [
      { name: "the v1 unqualified pass-through", re: /passes your prompt through \*\*unchanged\*\*/ },
      { name: "the v2 'never alters your prompt' absolute", re: /never alters your prompt/i },
      { name: "the v2 'the one exception' framing", re: /the one exception/i },
      // Refine is a SUGGESTION until Apply. Copy that says it replaces the prompt outright is the
      // same class of error in the opposite direction.
      { name: "the claim that Refine replaces the prompt outright", re: /replaces what you wrote/i },
    ]) {
      expect(overclaim.re.test(guide), `guide must not carry ${overclaim.name}`).toBe(false);
    }
    for (const claim of [
      { name: "the markers are not stripped or repaired", re: /not removed, not rewritten and not warned about/ },
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

  it("does not offer an upscale as a route to the withheld 2K", () => {
    // The compensating question — whether SceneWorks offers a 2K upscale path for this family — is
    // an open product decision on sc-17162 and is NOT settled. Until it is, the guide must not send
    // a user off to do it: `H3-Regenerate-2K` was an IN-CONTEXT upsample the DiT performed, and a
    // post-hoc upscale of a 1344px render is a different artifact, so telling someone to "run an
    // upscaler afterwards" quietly re-advertises the capability the paragraph just withdrew.
    expect(
      /\bupscal\w*\b[^.]{0,40}\b(afterwards|later|after the render)\b/i.test(guide),
      "guide must not route the withheld 2K through a post-hoc upscale",
    ).toBe(false);
    // What it must say instead: the ceiling is enforced, so 1344 is the number to plan against.
    expect(/refused rather than refitted/.test(guide), "guide must say an over-size request is refused").toBe(true);
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
    for (const named of guide.match(/\b\d{3,4}x\d{3,4}\b/g) ?? []) {
      expect(advertised.has(named), `guide names ${named}, which no entry advertises`).toBe(true);
    }
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
