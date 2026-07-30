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

// Whether a dropped file is worth sending to `POST /api/v1/workflows/inspect`.
//
// This is the whole reason the drop path does not just POST whatever it was given. The endpoint
// stages the entire body to disk BEFORE it looks at the first byte, and a scan miss is O(file) —
// so dragging a 400 MB TIFF onto the app would upload 400 MB to be told "not a PNG". Two cheap
// local checks close that:
//
// * **The PNG signature**, read out of the first eight bytes with `Blob.slice` — a range read, not
//   a load of the file. Only a PNG can carry the chunk at all, so anything else is rejected here
//   and the drop stays exactly today's silent swallow.
// * **The server's own cap**, so a file guaranteed to come back 413 is never sent.
//
// Anything that throws (a permission-revoked drag, an environment with no `Blob.arrayBuffer`)
// answers `false`: an unreadable file is not a workflow candidate, and today's behaviour is the
// correct fallback for every uncertainty on this path.
export async function looksLikeWorkflowCandidate(file) {
  if (!file || typeof file.slice !== "function") {
    return false;
  }
  const size = Number(file.size);
  if (!Number.isFinite(size) || size < PNG_SIGNATURE.length || size > MAX_WORKFLOW_INSPECT_BYTES) {
    return false;
  }
  try {
    const bytes = await readBlobBytes(file.slice(0, PNG_SIGNATURE.length));
    return PNG_SIGNATURE.every((byte, index) => bytes[index] === byte);
  } catch {
    return false;
  }
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

// Every requirement that is not resolved, as the sentences the report already wrote. The panel
// renders these; nothing here rewords them.
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
  for (const input of report.inputs ?? []) {
    rows.push({ kind: "input", state: input.state, detail: input.detail });
  }
  for (const entry of report.omitted ?? []) {
    rows.push({ kind: "omitted", state: "missing", detail: entry.detail });
  }
  return rows;
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
export function recipeFromWorkflowShare(share, report = null) {
  if (!share) {
    return null;
  }
  // `share.advanced` is already the allow-listed, re-sanitized map the reader produced, and the
  // studio's replay effect reads exactly these keys out of `rawAdapterSettings` — so it is copied
  // across rather than re-filtered here. (Copied, never aliased: the panel still renders the
  // envelope and must not see the studio-facing edits below.)
  const rawSettings = { ...(share.advanced ?? {}) };
  if (!styleIdResolved(report)) {
    delete rawSettings.styleId;
    delete rawSettings.stylePrompt;
  }
  if (share.upscale) {
    rawSettings.upscale = { ...share.upscale };
  }
  if (share.fitMode) {
    rawSettings.fitMode = share.fitMode;
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
  const modelId = workflowModelId(report);
  if (modelId) {
    recipe.model = modelId;
  }
  const stylePreset = report?.styles?.find(
    (style) => style.field === "stylePreset" && style.state === "resolved",
  );
  if (stylePreset) {
    recipe.stylePreset = stylePreset.id;
  }
  return recipe;
}

// ---- Display ------------------------------------------------------------------------------

const MODE_LABELS = {
  text_to_image: "Text to image",
  edit_image: "Edit image",
  character_image: "Character image",
  reference_image: "Reference image",
  pose_image: "Pose image",
};

// Human labels for the envelope's `advanced` keys. A key with no entry is shown under its own
// name rather than hidden — this panel's job is to show what is in the file, and a knob nobody
// wrote a label for is still a knob that changes the image.
const SETTING_LABELS = {
  resolution: "Resolution",
  sampler: "Sampler",
  scheduler: "Scheduler",
  schedulerShift: "Schedule shift",
  steps: "Steps",
  guidanceScale: "Guidance",
  guidanceMethod: "Guidance method",
  enhancePrompt: "Prompt upsampling",
  usePid: "PiD decoder",
  pidTarget: "PiD output",
  ipAdapterScale: "Reference strength",
  controlnetConditioningScale: "Identity structure",
  trueCfgScale: "Variation strength",
  strength: "img2img strength",
  viewAngle: "Head angle",
  textStyleGain: "Text-style gain",
  faceRestore: "Face restoration",
  controlMode: "Control type",
  controlScale: "Control strength",
  styleId: "Style",
  cnScale: "Tile ControlNet",
  angleSet: "Angle set",
  imageGuidanceScale: "Reference guidance",
};

// Keys whose value is prose or a structure the panel shows elsewhere (or cannot render as a
// one-line scalar). Skipped from the settings grid, never from the file.
const SETTING_SKIP = new Set(["stylePrompt", "systemMessage", "structuredPrompt", "poses", "phases"]);

export function workflowModeLabel(share) {
  const mode = share?.mode;
  return MODE_LABELS[mode] ?? (typeof mode === "string" && mode ? mode : "Unknown mode");
}

// The allow-listed advanced knobs, as `{ key, label, value }` rows fit for a definition list.
export function workflowSettingRows(share) {
  const settings = share?.advanced;
  if (!settings || typeof settings !== "object") {
    return [];
  }
  return Object.keys(settings)
    .filter((key) => !SETTING_SKIP.has(key))
    .map((key) => ({ key, label: SETTING_LABELS[key] ?? key, value: settings[key] }))
    .filter((row) => row.value !== null && row.value !== undefined && row.value !== "")
    .map((row) => ({
      ...row,
      value: typeof row.value === "boolean" ? (row.value ? "On" : "Off") : String(row.value),
    }));
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
