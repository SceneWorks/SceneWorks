// Reading a dropped image's embedded workflow, and turning it into something the Image Studio
// can be prefilled with (sc-15951, epic 15945).
//
// The envelope contract is `docs/workflow-share-envelope.md`; the resolution report is
// `crates/sceneworks-core/src/workflow_resolution.rs`. Nothing here re-derives either — this
// module only bridges the two onto the shape the studio's existing replay seam already accepts
// (`recipe.normalizedSettings` / `recipe.rawAdapterSettings`, read by the injection effect in
// `screens/ImageStudio.jsx`).
//
// # The one rule
//
// **Import what resolves, name what does not, never silently substitute.** Every requirement the
// report had to look up — the model, each LoRA, each style axis — is prefilled ONLY in
// `resolved` state. A requirement in any other state is left out of the recipe entirely and is
// named by the panel instead, because the alternatives are both lies: prefilling a catalog id the
// studio's own pickers filter out (they drop `installState === "missing"` rows) drops it
// silently, and falling back to a default model would reproduce someone else's image with our
// model.
//
// The same rule has a second half, and it is the one this module got wrong first: a knob that
// travels in the envelope but reaches no Image Studio control is ALSO a silent substitution — the
// studio keeps whatever it was already on while the panel says "PiD decoder — On". So
// [`ADVANCED_PREFILL`] below is the single table both halves read: the recipe carries exactly the
// keys the studio applies, and the panel marks exactly the rest as not restored.

// `status` values of `POST /api/v1/workflows/inspect`. Both keys are always present in the body,
// so a reader branches on `status` rather than on the absence of `workflow`.
export const WORKFLOW_STATUS_WORKFLOW = "workflow";
export const WORKFLOW_STATUS_NO_WORKFLOW = "no_workflow";

// The first eight bytes of every PNG (the signature the endpoint's own `NotPng` check reads).
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

// `MAX_WORKFLOW_INSPECT_BYTES` in `apps/rust-api/src/lib.rs`. Mirrored rather than fetched: it
// exists here to stop an upload that is already known to 413, and a client that is one release
// behind the server simply declines a file the server would have accepted, which is the safe way
// for the two to disagree.
export const MAX_WORKFLOW_INSPECT_BYTES = 512 * 1024 * 1024;

// A `Blob`'s bytes, through whichever reader this runtime has.
//
// `Blob.arrayBuffer` is the direct route and is what every shipping shell uses. jsdom's `Blob`
// does not implement it, and neither did some older WebViews — so `FileReader`, which every one
// of them has, is the fallback rather than the feature being unavailable off the browser.
function readBlobBytes(blob) {
  if (typeof blob.arrayBuffer === "function") {
    return blob.arrayBuffer().then((buffer) => new Uint8Array(buffer));
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the dropped file."));
    reader.onload = () => resolve(new Uint8Array(reader.result));
    reader.readAsArrayBuffer(blob);
  });
}

// Whether a blob's first eight bytes are the PNG signature.
//
// A range read (`Blob.slice`), not a load of the file — so it costs the same on an 8 KB icon and
// on a 700 MB scan. Deliberately says nothing about SIZE: "is this a PNG" and "would the server
// accept it" are different questions with different consumers, and the Image Editor
// (`editorSourceWorkflow.js`) needs the first one on files the second one refuses.
//
// Anything that throws (a permission-revoked drag, a runtime with neither `Blob.arrayBuffer` nor
// `FileReader`) answers `false`: a blob nobody can read the head of is not a PNG we can vouch for.
export async function blobIsPng(file) {
  if (!file || typeof file.slice !== "function") {
    return false;
  }
  const size = Number(file.size);
  if (!Number.isFinite(size) || size < PNG_SIGNATURE.length) {
    return false;
  }
  try {
    const bytes = await readBlobBytes(file.slice(0, PNG_SIGNATURE.length));
    return PNG_SIGNATURE.every((byte, index) => bytes[index] === byte);
  } catch {
    return false;
  }
}

// Whether a dropped file is worth sending to `POST /api/v1/workflows/inspect`.
//
// This is the whole reason the drop path does not just POST whatever it was given. The endpoint
// stages the entire body to disk BEFORE it looks at the first byte, and a scan miss is O(file) —
// so dragging a 400 MB TIFF onto the app would upload 400 MB to be told "not a PNG". Two cheap
// local checks close that:
//
// * **The PNG signature** ([`blobIsPng`]). Only a PNG can carry the chunk at all, so anything else
//   is rejected here and the drop stays exactly today's silent swallow.
// * **The server's own cap**, so a file guaranteed to come back 413 is never sent.
//
// A `false` here is NOT "this file has no workflow" — it is "nobody looked". The drop path can
// treat the two the same because its alternative is today's silent swallow; the Image Editor
// cannot, and keeps them apart (`editorSourceWorkflow.js`).
export async function looksLikeWorkflowCandidate(file) {
  const size = Number(file?.size);
  if (!Number.isFinite(size) || size > MAX_WORKFLOW_INSPECT_BYTES) {
    return false;
  }
  return blobIsPng(file);
}

// ---- The envelope an already-imported asset is carrying -------------------------------------

// `IMPORTED_WORKFLOW_KEY` in `crates/sceneworks-core/src/project_store.rs`, mirrored.
//
// Import clears this key unconditionally before writing its own, so what is under it is either the
// envelope THIS install's reader parsed and sanitized, or nothing — a caller-supplied
// `provenance.importedWorkflow` cannot masquerade as one (`docs/workflow-share-envelope.md`, "The
// trust boundary on import").
export const IMPORTED_WORKFLOW_KEY = "importedWorkflow";

// The envelope an imported asset carries, or null.
//
// The one predicate behind "Use this recipe appears on imported assets carrying a workflow, and
// nowhere else". Deliberately not a truthiness check on `extra`: an asset imported from any of the
// billions of PNGs with no SceneWorks chunk in them has an `extra` and no key under it, and that is
// the common case rather than the edge one.
export function assetImportedWorkflow(asset) {
  const stored = asset?.extra?.[IMPORTED_WORKFLOW_KEY];
  return stored && typeof stored === "object" ? stored : null;
}

// ---- Reading the report -------------------------------------------------------------------

function requirementResolved(requirement) {
  return requirement?.state === "resolved";
}

// The local catalog id to prefill for a looked-up requirement, or null.
//
// `catalogId` is set whenever the report MATCHED a row, including one that is merely installable.
// The `resolved` gate on top of it is deliberate: the studio's model list
// (`generationModelsForType`) and its LoRA picker (`compatibleLoras`) both drop rows that are not
// on disk, so seeding an installable id would set a value with no option behind it — a blank
// control, or a selection that vanishes on the next render. Named in the panel instead.
function resolvedCatalogId(requirement) {
  return requirementResolved(requirement) ? (requirement.catalogId ?? null) : null;
}

// The model catalog id to select, or null when this install cannot resolve the envelope's model.
export function workflowModelId(report) {
  return resolvedCatalogId(report?.model);
}

// The `{ id, weight }` list the studio's LoRA picker can seed from — resolved entries only, in
// envelope order.
export function workflowLoras(report) {
  const requirements = Array.isArray(report?.loras) ? report.loras : [];
  return requirements
    .map((requirement) => {
      const id = resolvedCatalogId(requirement);
      if (!id) {
        return null;
      }
      return Number.isFinite(Number(requirement.weight))
        ? { id, weight: Number(requirement.weight) }
        : { id };
    })
    .filter(Boolean);
}

// Whether the live Style-catalog axis (`styleId`) resolved here.
//
// Gates the style round-trip: the studio re-selects its Style picker only when BOTH
// `rawAdapterSettings.styleId` and a string `rawAdapterSettings.stylePrompt` are present, and it
// then seeds the prompt box with the RAW pre-style prompt so submit recomposes. With the style
// absent from this catalog that recomposition cannot happen, so both keys are dropped and the
// already-composed `prompt` is used verbatim — which is the closest this install can get to the
// image that was shared.
function styleIdResolved(report) {
  const styles = Array.isArray(report?.styles) ? report.styles : [];
  const live = styles.find((style) => style.field === "styleId");
  return live ? requirementResolved(live) : false;
}

// Every requirement that BLOCKS replay, as the sentences the report already wrote. The panel
// renders these; nothing here rewords them.
//
// `inputs` are deliberately absent, mirroring `ResolutionReport::runnable`
// (`crates/sceneworks-core/src/workflow_resolution.rs`), which excludes them for the reason
// written there: nothing on this machine could ever have supplied someone else's source image, so
// an input is a thing the user provides and not a thing this install lacks. Counting them here
// made the verdict read "2 things in this recipe cannot be reproduced on this install" for a
// perfectly runnable edit recipe whose only "problem" was that it starts from an image. The panel
// still renders every input row — the count is what excludes them.
export function workflowUnresolved(report) {
  if (!report) {
    return [];
  }
  const rows = [];
  if (report.model && report.model.state !== "resolved") {
    rows.push({ kind: "model", state: report.model.state, detail: report.model.detail });
  }
  for (const lora of report.loras ?? []) {
    if (lora.state !== "resolved") {
      rows.push({ kind: "lora", state: lora.state, detail: lora.detail });
    }
  }
  for (const style of report.styles ?? []) {
    if (style.state !== "resolved") {
      rows.push({ kind: "style", state: style.state, detail: style.detail });
    }
  }
  for (const entry of report.omitted ?? []) {
    rows.push({ kind: "omitted", state: "missing", detail: entry.detail });
  }
  return rows;
}

// ---- What the studio actually does with each allow-listed knob ------------------------------

// The three fates an `advanced` key can meet on the way into Image Studio.
export const PREFILL_CONTROL = "control"; // the replay effect sets a control from it
export const PREFILL_PROMPT = "prompt"; // applied, but as prompt text — the panel shows it there
export const PREFILL_NONE = "none"; // it travels, and this studio has nowhere to put it

// **The one table both halves of the panel read.**
//
// The envelope's allow-list is `ADVANCED_KEY_RULES` in `crates/sceneworks-core/src/workflow_share.rs`
// and is pinned to the `advanced-shared` table of `docs/workflow-share-envelope.md`. That list
// answers "may this key leave the machine?". It says nothing about whether the RECEIVING studio has
// a control for it — and those are different questions, because an image envelope carries keys from
// four lanes (the Image Studio, the standalone Detail pass, the Character studio's angle set, the
// interleaved-document lane) while this prefill lands in exactly one of them.
//
// Splitting that judgement across two places is what shipped ten knobs — `enhancePrompt`, `usePid`,
// `pidTarget`, `strength`, `textStyleGain`, `faceRestore`, `controlMode`, `controlScale`, `poses`,
// `phases` — displayed as ordinary "Settings" rows beside "Steps — 28" while none of them reached a
// control. The user was told a recipe replayed faithfully when it had not, which is the exact
// failure epic 15945 exists to prevent.
//
// So there is one table, and two readers of it:
//
// * `recipeFromWorkflowShare` copies across exactly the `control` / `prompt` keys, so the recipe
//   the studio receives contains nothing the studio ignores;
// * `workflowSettingRows` marks exactly the `none` keys as not restored, and
//   `workflowNotRestored` folds them into the panel's verdict.
//
// A key that appears in the envelope and NOT in this table is treated as `none`: rendered, marked
// as not restored, and left out of the recipe. The safe default is "the user is told", never "it
// quietly disappears".
export const ADVANCED_PREFILL = {
  resolution: { label: "Resolution", prefill: PREFILL_CONTROL },
  // sc-15952 reclassified this. It was `PREFILL_PROMPT` on the reading that the panel shows its
  // content in the prompt block above the grid — true of `stylePrompt`, which the studio really
  // does read back (`rawSettings.stylePrompt`), and false of this one. The builder rehydrates from
  // `structuredPrompt.caption`, and the envelope reduces `structuredPrompt` to its SCALAR fields
  // and drops the caption object (`docs/workflow-share-envelope.md`, Shared table) — so a shared
  // structured-caption recipe reaches no builder control at all and replays as the composed prose
  // in the plain prompt box.
  //
  // That prose replay is the DECIDED behaviour, not a stopgap: the top-level `prompt` is the exact
  // string the model received, so it reproduces the image. Rehydrating the builder instead would
  // need a classified sub-schema for the caption in the envelope itself, which is a contract change
  // and belongs to its own story (sc-16197). What this row buys is that the panel says so.
  structuredPrompt: {
    label: "Structured prompt",
    prefill: PREFILL_NONE,
    detail:
      "The caption object does not travel — only its scalar fields do — so the structured builder " +
      "cannot be rehydrated. The prompt is restored as prose, exactly as the model received it.",
  },
  sampler: { label: "Sampler", prefill: PREFILL_CONTROL },
  scheduler: { label: "Scheduler", prefill: PREFILL_CONTROL },
  schedulerShift: { label: "Schedule shift", prefill: PREFILL_CONTROL },
  steps: { label: "Steps", prefill: PREFILL_CONTROL },
  guidanceScale: { label: "Guidance", prefill: PREFILL_CONTROL },
  guidanceMethod: { label: "Guidance method", prefill: PREFILL_CONTROL },
  enhancePrompt: { label: "Prompt upsampling", prefill: PREFILL_CONTROL },
  usePid: { label: "PiD decoder", prefill: PREFILL_CONTROL },
  pidTarget: { label: "PiD output", prefill: PREFILL_CONTROL },
  ipAdapterScale: { label: "Reference strength", prefill: PREFILL_CONTROL },
  controlnetConditioningScale: { label: "Identity structure", prefill: PREFILL_CONTROL },
  trueCfgScale: { label: "Variation strength", prefill: PREFILL_CONTROL },
  strength: { label: "img2img strength", prefill: PREFILL_CONTROL },
  viewAngle: { label: "Head angle", prefill: PREFILL_CONTROL },
  textStyleGain: { label: "Text-style gain", prefill: PREFILL_CONTROL },
  faceRestore: { label: "Face restoration", prefill: PREFILL_CONTROL },
  controlMode: { label: "Control type", prefill: PREFILL_CONTROL },
  controlScale: { label: "Control strength", prefill: PREFILL_CONTROL },
  styleId: { label: "Style", prefill: PREFILL_CONTROL },
  stylePrompt: { label: "Pre-style prompt", prefill: PREFILL_PROMPT },
  phases: { label: "Multi-phase denoise", prefill: PREFILL_CONTROL },
  // Shared by Image Studio substitutions and the older LTX video selector. Image workflow replay
  // restores the opaque authored id even when this install no longer lists it; Image Studio keeps
  // that unavailable choice visible and the API then fails closed until it is restored or changed.
  textEncoderModel: { label: "Text encoder", prefill: PREFILL_CONTROL },
  // The five that travel and land nowhere. Each names why, because "not restored" without a
  // reason reads like a bug.
  poses: {
    label: "Poses",
    prefill: PREFILL_CONTROL,
  },
  cnScale: {
    label: "Tile ControlNet",
    prefill: PREFILL_NONE,
    detail: "A knob of the standalone Detail pass, which Image Studio does not run.",
  },
  angleSet: {
    label: "Angle set",
    prefill: PREFILL_NONE,
    detail: "A Character Studio turnaround request. Image Studio has no angle-set control.",
  },
  systemMessage: {
    label: "System prompt",
    prefill: PREFILL_NONE,
    detail: "A Document Studio interleave setting. Image Studio has no system prompt.",
  },
  imageGuidanceScale: {
    label: "Reference guidance",
    prefill: PREFILL_NONE,
    detail: "A Document Studio interleave setting. Image Studio has no control for it.",
  },
  // The VIDEO arm (sc-15956). Every one of these travels in a shared MP4 and lands nowhere HERE,
  // for one reason that covers all eleven: this registry drives Image Studio, and a video recipe
  // replays in Video Studio. That is a different panel and a different story, so the honest thing
  // for this one to say is that the setting arrived and this studio has no control for it —
  // exactly what the four rows above say about the Detail, Character and Document lanes.
  //
  // They are listed individually rather than collapsed because the panel renders a LABEL per key:
  // "videoStgGuidanceScale arrived" is not a sentence, and a shared clip whose settings show as
  // raw camelCase reads as a bug rather than as a boundary.
  motion: {
    label: "Camera motion",
    prefill: PREFILL_NONE,
    detail: "A Video Studio setting. Image Studio has no camera-motion control.",
  },
  ltxPipeline: {
    label: "LTX pipeline",
    prefill: PREFILL_NONE,
    detail: "A Video Studio setting for the LTX video model.",
  },
  distilledVariant: {
    label: "Distilled variant",
    prefill: PREFILL_NONE,
    detail: "A Video Studio setting: which distilled LTX checkpoint the clip was made with.",
  },
  lightning: {
    label: "Lightning steps",
    prefill: PREFILL_NONE,
    detail: "A Video Studio setting: the Wan2.2 fast four-step recipe.",
  },
  videoCfgGuidanceScale: {
    label: "Video guidance",
    prefill: PREFILL_NONE,
    detail: "A Video Studio setting for the LTX video model.",
  },
  videoStgGuidanceScale: {
    label: "Spatiotemporal guidance",
    prefill: PREFILL_NONE,
    detail: "A Video Studio setting for the LTX video model.",
  },
  videoRescaleScale: {
    label: "Guidance rescale",
    prefill: PREFILL_NONE,
    detail: "A Video Studio setting for the LTX video model.",
  },
  videoConditioningStrength: {
    label: "Clip strength",
    prefill: PREFILL_NONE,
    detail: "A Video Studio setting: how strongly the source clip conditioned the result.",
  },
  bridgeRightVideoConditioningStrength: {
    label: "Right-clip strength",
    prefill: PREFILL_NONE,
    detail: "A Video Studio setting: the second clip strength in a bridge.",
  },
  timelineAction: {
    label: "Timeline action",
    prefill: PREFILL_NONE,
    detail:
      "Which timeline operation made the clip. The timeline it belonged to deliberately does " +
      "not travel, so there is nothing here to place it into.",
  },
};

function prefillDisposition(key) {
  return ADVANCED_PREFILL[key]?.prefill ?? PREFILL_NONE;
}

// ---- The adapter --------------------------------------------------------------------------

// The seed of THIS image, as the launch's `replaySeed`.
//
// Passed as a string so a seed of 0 survives the studio's `!= null && !== ""` guard, and so a
// large value is not reformatted by a round trip through Number.
export function workflowReplaySeed(share) {
  const seed = share?.seed;
  return seed === null || seed === undefined || seed === "" ? null : String(seed);
}

// A `WorkflowShare` (plus this install's resolution report) as a studio recipe.
//
// The shape is the one `crates/sceneworks-core/src/project_store.rs` writes for a generated
// asset, because that is what the ImageStudio injection effect reads — this returns the same
// object the "Use this recipe" path hands it, so there is exactly one prefill path.
//
// `report` is not optional in spirit: with it absent nothing that had to be looked up is
// prefilled, since without a report this module cannot tell a resolvable model from one that
// would leave the studio's picker blank.
//
// `substituteModelId` and `referenceAssetIds` are the two things the user can supply BY HAND in
// the missing-inputs panel (sc-15952), and they are parameters rather than defaults for the reason
// this whole module exists: a model chosen from a picker is the user's decision and a model
// reached for automatically is a substitution. There is no fallback value for either — absent
// means absent, and the panel blocks the launch instead.
export function recipeFromWorkflowShare(
  share,
  report = null,
  { substituteModelId = null, referenceAssetIds = [] } = {},
) {
  if (!share) {
    return null;
  }
  // The studio-facing copy of the envelope's settings: exactly the keys `ADVANCED_PREFILL` says
  // the replay effect reads, and nothing else. Filtered rather than copied wholesale so the recipe
  // cannot carry a key the studio ignores while the panel presents it as restored — the two now
  // disagree only if someone edits the table, which is the point of there being a table.
  // (Built fresh, never aliased: the panel still renders the envelope and must not see the
  // studio-facing edits below.)
  const rawSettings = {};
  for (const [key, value] of Object.entries(share.advanced ?? {})) {
    if (value !== null && value !== undefined && prefillDisposition(key) !== PREFILL_NONE) {
      rawSettings[key] = value;
    }
  }
  if (!styleIdResolved(report)) {
    delete rawSettings.styleId;
    delete rawSettings.stylePrompt;
  }
  // The multi-phase schedule, reindexed onto the LoRA stack this install will actually hold.
  const phases = workflowPhases(share, report);
  if (phases) {
    rawSettings.phases = phases;
  } else {
    delete rawSettings.phases;
  }
  if (share.upscale) {
    rawSettings.upscale = { ...share.upscale };
  }
  if (share.fitMode) {
    rawSettings.fitMode = share.fitMode;
  }
  // The reference images the USER picked for an `inputs[].kind === "reference"` requirement. The
  // envelope carries the SHAPE of the requirement and never an id (`docs/workflow-share-envelope.md`
  // — "Input images travel by shape, not by id"), so nothing here is restored; this is the user's
  // own selection travelling from the panel's pickers to the studio's controls.
  const references = (Array.isArray(referenceAssetIds) ? referenceAssetIds : []).filter(Boolean);
  if (references.length) {
    rawSettings.referenceAssetIds = references;
    // The single-reference control the studio has always read, so one picked image lands whether
    // or not the selected model offers the plural multi-reference picker.
    rawSettings.referenceAssetId = references[0];
  }

  const normalizedSettings = {};
  const width = Number(share.width);
  const height = Number(share.height);
  if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
    normalizedSettings.width = width;
    normalizedSettings.height = height;
  }
  // COUNT IS NOT REPLAYED. The envelope's `seed` identifies ONE image while `count` came from the
  // batch the run requested, so replaying both would reproduce the shared image as the first of an
  // N-image batch — N-1 images the user did not ask for, each burning a generation. The recipe
  // therefore always asks for one, and the panel states the file was one of a batch of N so the
  // fact is reported rather than lost.
  normalizedSettings.count = 1;

  const recipe = {
    mode: share.mode ?? "text_to_image",
    prompt: share.prompt ?? "",
    negativePrompt: share.negativePrompt ?? "",
    seed: share.seed ?? null,
    loras: workflowLoras(report),
    normalizedSettings,
    rawAdapterSettings: rawSettings,
  };
  // The model the RECIPE named, when this install resolved it — otherwise the one the user picked
  // in the panel, and otherwise nothing at all. The resolved id wins because it is what the file
  // asked for; the substitute is only ever reached for when there was no resolution, which is the
  // difference between a choice and a fallback.
  const modelId = workflowModelId(report) ?? (substituteModelId || null);
  if (modelId) {
    recipe.model = modelId;
  }
  // `stylePreset` is NOT set here even when it resolved. Nothing in `apps/web/src` reads
  // `recipe.stylePreset` — the studio's preset seam is `launchRequest.presetId`, a different
  // launch shape entirely — so assigning it was a dead write that made the panel's "Style — <name>
  // [Ready]" row a claim about an axis that reaches no control. The row is marked not-restored
  // instead (see `workflowNotRestored`).
  return recipe;
}

// The envelope's multi-phase schedule, reindexed onto the LoRA stack this install will hold — or
// null when there is no schedule to replay.
//
// `advanced.phases[].loras[].index` is a position in the ORIGINAL request's LoRA list. The recipe's
// list is `workflowLoras(report)`, which drops every LoRA this install could not resolve, so those
// positions no longer line up: a 3-LoRA share whose first LoRA is missing would have every phase
// pointing one slot too far right, silently activating the WRONG adapter — worse than not
// replaying at all. So the map is envelope index → position in the resolved subset, and a phase
// reference to a LoRA that did not resolve is dropped rather than repointed. (The dropped LoRA is
// already named by the report, so the loss is visible.)
//
// The output stays in the index shape the recorded-recipe path also uses, so `ImageStudio`'s
// replay effect has ONE thing to deserialize (`imageMultiPhase.deserializePhases`) rather than a
// share-shaped branch beside a recipe-shaped one.
export function workflowPhases(share, report = null) {
  const phases = share?.advanced?.phases;
  if (!Array.isArray(phases) || phases.length === 0) {
    return null;
  }
  const requirements = Array.isArray(report?.loras) ? report.loras : [];
  const compacted = new Map();
  let position = 0;
  requirements.forEach((requirement, index) => {
    if (resolvedCatalogId(requirement)) {
      compacted.set(index, position);
      position += 1;
    }
  });
  return phases.map((phase) => {
    const steps = Math.trunc(Number(phase?.steps));
    const out = { steps: Number.isFinite(steps) ? steps : 0 };
    const guidance = Number(phase?.guidance);
    if (Number.isFinite(guidance)) {
      out.guidance = guidance;
    }
    out.loras = (Array.isArray(phase?.loras) ? phase.loras : []).flatMap((ref) => {
      const index = compacted.get(Number(ref?.index));
      if (index === undefined) {
        return [];
      }
      const entry = { index };
      const weight = Number(ref?.weight);
      if (ref?.weight != null && ref.weight !== "" && Number.isFinite(weight)) {
        entry.weight = weight;
      }
      return [entry];
    });
    return out;
  });
}

// ---- Display ------------------------------------------------------------------------------

const MODE_LABELS = {
  text_to_image: "Text to image",
  edit_image: "Edit image",
  character_image: "Character image",
  reference_image: "Reference image",
  pose_image: "Pose image",
};

// How long a prose value may run inside a one-line settings row before it is elided. The full
// text is in the file either way; this is a row, not a document viewer.
const PROSE_ROW_MAX_CHARS = 90;

// One row's value, as a string. Collections say how many they are rather than dumping coordinates
// (a 12-pose selection is ~800 numbers), and prose is elided rather than allowed to run the row
// off the panel.
function settingValueLabel(key, value) {
  if (typeof value === "boolean") {
    return value ? "On" : "Off";
  }
  if (Array.isArray(value)) {
    const noun = key === "phases" ? "phase" : key === "poses" ? "pose" : "entry";
    const plural = noun === "entry" ? "entries" : `${noun}s`;
    return `${value.length} ${value.length === 1 ? noun : plural}`;
  }
  if (value && typeof value === "object") {
    return "Recorded";
  }
  const text = String(value);
  return text.length > PROSE_ROW_MAX_CHARS ? `${text.slice(0, PROSE_ROW_MAX_CHARS - 1)}…` : text;
}

export function workflowModeLabel(share) {
  const mode = share?.mode;
  return MODE_LABELS[mode] ?? (typeof mode === "string" && mode ? mode : "Unknown mode");
}

// The allow-listed advanced knobs as `{ key, label, value, restored, detail }` rows fit for a
// definition list — every row carrying whether "Use this workflow" will actually apply it.
//
// `restored` is read from `ADVANCED_PREFILL`, the same table `recipeFromWorkflowShare` filters the
// recipe with, so a row cannot claim an application the prefill does not perform. A key this build
// has no entry for is rendered under its own name and marked NOT restored, because an unrecognized
// knob is one this studio certainly has no control for.
//
// Only the `prompt`-fated keys are omitted — today that is `stylePrompt` alone. It IS applied, and
// the panel already shows its content in the prompt block above the grid, so a second row would be
// a duplicate rather than a disclosure. (`structuredPrompt` used to sit here on the same reasoning
// and did not deserve to; see its `ADVANCED_PREFILL` row.)
//
// `report` is what makes `styleId` honest. It is a `control` key, but the recipe carries it ONLY
// when the style resolved here (see `recipeFromWorkflowShare`) — so with the style absent from this
// catalog the row has to say so rather than inherit the table's optimism.
export function workflowSettingRows(share, report = null) {
  const settings = share?.advanced;
  if (!settings || typeof settings !== "object") {
    return [];
  }
  const styleResolvedHere = styleIdResolved(report);
  return Object.keys(settings)
    .filter((key) => prefillDisposition(key) !== PREFILL_PROMPT)
    .map((key) => ({ key, value: settings[key] }))
    .filter(
      (row) =>
        row.value !== null &&
        row.value !== undefined &&
        row.value !== "" &&
        !(Array.isArray(row.value) && row.value.length === 0),
    )
    .map((row) => {
      const styleUnavailable = row.key === "styleId" && !styleResolvedHere;
      return {
        key: row.key,
        label: ADVANCED_PREFILL[row.key]?.label ?? row.key,
        value: settingValueLabel(row.key, row.value),
        restored: prefillDisposition(row.key) === PREFILL_CONTROL && !styleUnavailable,
        detail: styleUnavailable
          ? "This install does not have that style, so the prompt is used exactly as it was already composed."
          : (ADVANCED_PREFILL[row.key]?.detail ?? "This build has no control for this setting."),
      };
    });
}

// Everything the panel promises will NOT survive "Use this workflow", as report-shaped rows.
//
// Two sources, one list, because the user's question is one question — "what will I not get?" —
// and answering it in two places is how one of them goes stale:
//
// * the `advanced` knobs that reach no Image Studio control (`workflowSettingRows`, above); and
// * a RESOLVED `stylePreset`. The report says "Ready" for it because the preset is installed here,
//   which is true and beside the point: the recipe seam this prefill uses carries no preset (the
//   studio takes one through `launchRequest.presetId`, a different launch shape), so an installed
//   preset is as unrestored as a missing one. An UNRESOLVED `stylePreset` is left out — it is
//   already a blocking row in the report, and listing it twice would double-count it.
//
// The panel's verdict counts this list, which is what stops successful transport being punished:
// a pose selection DROPPED over the writer's cap lands in `omitted` and forces `runnable: false`,
// so before this existed the only silent case was the one where the poses arrived intact.
export function workflowNotRestored(share, report = null) {
  const rows = workflowSettingRows(share, report)
    .filter((row) => !row.restored)
    .map((row) => ({
      kind: "setting",
      key: row.key,
      title: `${row.label} — ${row.value}`,
      detail: row.detail,
    }));
  for (const style of report?.styles ?? []) {
    if (style.field === "stylePreset" && style.state === "resolved") {
      rows.push({
        kind: "style",
        key: "stylePreset",
        title: `Style preset — ${style.name ?? style.id}`,
        detail:
          "Installed here, but this replay path carries no preset — the studio keeps the preset " +
          "it is already on.",
      });
    }
  }
  return rows;
}

// ---- What the user can actually DO about an unresolved requirement (sc-15952) ----------------

// The requirements this install could fetch, as install-action rows.
//
// Exactly the `installable` state and nothing else, because that is the only state the report
// publishes an action for (`install_for` in `workflow_resolution.rs` clears it for every other
// one). A `missing` row has no button — offering one that cannot work is the lie the report's own
// docs call out — and a `resolved` row needs none.
export function workflowInstallable(report) {
  const rows = [];
  if (report?.model?.state === "installable" && report.model.catalogId) {
    rows.push({
      kind: "model",
      id: report.model.catalogId,
      name: report.model.name ?? report.model.slug,
      detail: report.model.detail,
    });
  }
  for (const lora of report?.loras ?? []) {
    if (lora.state === "installable" && lora.catalogId) {
      rows.push({
        kind: "lora",
        id: lora.catalogId,
        name: lora.name ?? lora.repo ?? lora.catalogId,
        detail: lora.detail,
      });
    }
  }
  return rows;
}

// Whether this install has nothing at all that matches the model the file names.
//
// The one requirement that gets a SUBSTITUTE picker rather than an install button. Not because a
// substitute is a good outcome — it makes a different image — but because the alternative on offer
// is worse: the studio would otherwise keep whatever model it happened to be on and render the
// prompt with it, which is a silent substitution wearing a successful replay's clothes. Picking one
// is at least the user's decision, made in front of the sentence naming what the file asked for.
export function workflowModelUnknown(report) {
  return Boolean(report?.model) && report.model.state === "missing";
}

// The input-image requirements, expanded into ONE SLOT PER IMAGE.
//
// `InputRequirement` is `{ kind, count }` — one row per kind, `reference` carrying a count rather
// than repeating (`docs/workflow-share-envelope.md`). A panel that renders one picker for a
// "2 reference images" requirement quietly asks for half of what the recipe needs, so the count is
// expanded here and both the pickers and the CTA gate read the same list.
//
// `pickable` is whether a chosen image would reach a control. `source` and `reference` are the two
// the launch seam carries (`launchImageRecipe`'s `sourceAssetId`, and `referenceAssetIds` through
// the recipe); a `mask` is supplied with the Image Editor's inpaint brush and a `control` map in
// the studio's own Control panel, neither of which this launch can seed — so those state the
// requirement and offer no picker, rather than offering one whose value goes nowhere.
const PICKABLE_INPUT_KINDS = new Set(["source", "reference"]);

const INPUT_KIND_LABELS = {
  source: "Source image",
  reference: "Reference image",
  mask: "Mask",
  control: "Control image",
};

// Where an unpickable input is actually supplied, so the row is a direction and not a dead end.
//
// Each of these has to name the OUTCOME of proceeding without it, the way every other advisory
// does — "what you will get instead", not "where the control lives". The two are not the same
// warning, and the mask is the case where the difference bites: a mask does not weaken the edit,
// it CHANGES WHICH PIXELS ARE EDITED. `maskAssetId` reaches the worker only from the Image
// Editor's inpaint brush (`ImageEditor.jsx`); Image Studio has no mask control and its Generate
// gate does not know one was wanted. So a user who reads "painted in the Image Editor" as a note
// about where a knob lives, and generates anyway, gets the model rewriting the whole frame and
// nothing on screen says the replay was not faithful. The sentence says so instead.
const INPUT_KIND_ELSEWHERE = {
  mask:
    "Masks are painted on the image in the Image Editor, not chosen here — so generating from " +
    "this panel edits the WHOLE image rather than the masked region, which is a different edit " +
    "from the one you were sent. To reproduce it, load the recipe and then open the image in the " +
    "Image Editor and paint the mask there.",
  control:
    "A pre-made control map is attached in Image Studio's own Control panel once the recipe is " +
    "loaded.",
};

export function workflowInputSlots(report) {
  const slots = [];
  (report?.inputs ?? []).forEach((input, index) => {
    // The report floors an absent/zero count at 1 for its own total
    // (`ResolutionReport::input_images_required`); the same floor is used here so the panel never
    // renders zero pickers for a requirement the report is simultaneously counting.
    const count = Math.max(1, Number(input.count) || 0);
    const pickable = PICKABLE_INPUT_KINDS.has(input.kind);
    for (let slot = 0; slot < count; slot += 1) {
      slots.push({
        key: `${input.kind}-${index}-${slot}`,
        kind: input.kind,
        controlMode: input.controlMode ?? null,
        slot,
        count,
        pickable,
        label:
          count === 1
            ? (INPUT_KIND_LABELS[input.kind] ?? input.kind)
            : `${INPUT_KIND_LABELS[input.kind] ?? input.kind} ${slot + 1} of ${count}`,
        // The report already wrote the sentence ("Needs 2 reference images."); it is shown once
        // per requirement, on the first slot, rather than repeated under every picker.
        detail: slot === 0 ? input.detail : "",
        elsewhere: pickable ? "" : (INPUT_KIND_ELSEWHERE[input.kind] ?? ""),
      });
    }
  });
  return slots;
}

// **Whether the missing-inputs view has anything to render at all.**
//
// The AC this answers is a negative one: a fully-resolvable workflow must show NO missing-inputs
// UI — no empty container, no zero-state line, no stray heading. So the view's own render gate is
// this predicate rather than a truthy `report`, and the three things it counts are the three the
// view can act on. A dropped collection (`omitted`) deliberately does NOT count: it makes the
// report unrunnable and there is nothing anyone can do about it, so it stays a report row.
export function workflowHasMissingInputs(report) {
  return (
    workflowInstallable(report).length > 0 ||
    workflowModelUnknown(report) ||
    workflowInputSlots(report).length > 0
  );
}

// "1024 × 1024", or null when the envelope carried no geometry.
export function workflowResolutionLabel(share) {
  const width = Number(share?.width);
  const height = Number(share?.height);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return `${width} × ${height}`;
}
