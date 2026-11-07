

// YuE2 Song Lab (sc-23000, epic 22988) — the Audio Studio's EXPERIMENTAL surface for YuE2.
//
// Why a separate surface rather than a fifth AUDIO_MODES tab: YuE2 serves none of the four standard
// modes (`audioModelServesMode` is false for it by design, sc-22998) and is submitted through its own
// route (`POST /projects/:id/yue2/jobs` — the generic audio route refuses it). Its request is a
// style + lyrics + symbolic score plan with job kinds, score versions and comparisons that none of
// the standard modes builds. Keeping it out of AUDIO_MODES leaves every standard mode, its picker,
// its defaults and YuE1 (Music) exactly as they are; the lab is entered only by an explicit click on
// the separately-marked "Song Lab · Experimental" tab, and it opens behind an experimental opt-in
// plus the server-side licence acknowledgment.
//
// Persistence: the lab's settings live in their own studio snapshot (`yue2lab`), which the
// studio-settings seam mirrors to the SERVER's ui-preferences (desktop localStorage does not
// survive a relaunch). The acknowledgment is the server's own record; runs are server jobs; score
// versions and comparisons are server records. Nothing the lab needs after a relaunch lives only
// in the browser.

export const LAB_TABS = [
  { id: "compose", label: "Compose" },
  { id: "cover", label: "Cover" },
  { id: "scores", label: "Scores" },
  { id: "compare", label: "Compare" },
];

export const TIER_LABELS = { bf16: "BF16 (original)", q8: "Q8 (derived here)", q4: "Q4 (derived here)" };

export const PRESET_LIMIT = 12;

// Preset snapshots carry CONTROLS only — never free text (lyrics, style, scores, notes, drafts).
// Twelve presets each holding a full lyric sheet would blow the durable snapshot budget, and a
// preset is a reusable setup, not a song.
export const PRESET_EXCLUDED = new Set([
  "presets",
  // Run-specific: the saved plan a restore renders, the version a cover follows, and the plan
  // source that points at them belong to one song, not to a reusable setup.
  "planSource",
  "restorePlanJobId",
  "coverVersionId",
  "optIn",
  "tab",
  "lyrics",
  "style",
  "suppliedScore",
  "coverScore",
  "coverTranslatedFrom",
  "transcribeAssetId",
  "transcriptionId",
  "selectedVersionId",
  "compareA",
  "compareB",
  "compareRenderA",
  "compareRenderB",
  "compareNotes",
  "advancedOpen",
  "editDraft",
  "importDraft",
]);
