import { editedFilename } from "./provenance.js";
import { blobToImage } from "../../imageDimensions.js";
import { isDesktop } from "../../runtime.js";
import { saveExportFile } from "../../saveExportFile.js";
import { SOURCE_WORKFLOW_ABSENT, SOURCE_WORKFLOW_PENDING, SOURCE_WORKFLOW_PRESENT, SOURCE_WORKFLOW_UNKNOWN } from "../../editorSourceWorkflow.js";
import { SAVE_WITHOUT_WORKFLOW_LABEL } from "../../workflowEmbedPreferences.js";
import { DEFAULT_BLEND_MODE, compositeLayersToCanvas, identityTransform } from "../../imageLayers.js";
import { maskAlphaFromRgba } from "../../maskRefine.js";
import { applyMaskSelectionToRgba, colorKeySelectionMaskRgba } from "./colorKeyMath.js";

// A conservative cross-WebKit ceiling. WebKitGTK builds and GPU drivers vary in
// their exact canvas allocation limit; keeping both the longest side and total
// area bounded prevents the silent transparent-canvas failure seen above these
// values while retaining the largest normal SceneWorks generation size.
export const EDITOR_CANVAS_MAX_SIDE = 4096;

export const EDITOR_CANVAS_MAX_AREA =
  EDITOR_CANVAS_MAX_SIDE * EDITOR_CANVAS_MAX_SIDE;

export function boundedEditorCanvasDimensions(width, height) {
  const sourceWidth = Math.max(1, Math.round(Number(width) || 0));
  const sourceHeight = Math.max(1, Math.round(Number(height) || 0));
  const sideScale = Math.min(
    1,
    EDITOR_CANVAS_MAX_SIDE / sourceWidth,
    EDITOR_CANVAS_MAX_SIDE / sourceHeight,
  );
  const areaScale = Math.min(
    1,
    Math.sqrt(EDITOR_CANVAS_MAX_AREA / (sourceWidth * sourceHeight)),
  );
  const scale = Math.min(sideScale, areaScale);
  return {
    width: Math.max(1, Math.floor(sourceWidth * scale)),
    height: Math.max(1, Math.floor(sourceHeight * scale)),
    scaled: scale < 1,
    sourceWidth,
    sourceHeight,
  };
}

// Flatten the visible layer stack onto a fresh canvas at the document size (sc-6117). The layers'
// images are already decoded, so this is synchronous; callers toBlob it (Save / Download / AI-op
// source) or paint overlays on top first (the box-keyed edit). The shared composite behind every
// editor export.
//
// Module-level and exported since sc-24111 so the alpha round-trip test can drive the REAL export
// rather than a copy of it. It used to be a closure over `working`, which meant every assertion
// about "the exported PNG still has its transparency" was an assertion about the test's own
// reimplementation: switching this to `{ alpha: false }` or the encode below to `image/jpeg` left
// the suite green. `documentRef` is the seam that lets a test supply a canvas that actually holds
// pixels — jsdom's throws.
export function compositeWorkingToCanvas(work, { documentRef = globalThis.document } = {}) {
  const canvas = documentRef.createElement("canvas");
  canvas.width = work.width;
  canvas.height = work.height;
  // No `{ alpha: false }` and no fill before the first `drawImage`: either one composites every
  // transparent pixel onto black before the encode can see it.
  compositeLayersToCanvas(canvas.getContext("2d"), work.layers, { visibleOnly: true });
  return canvas;
}

// The working document as one PNG File. Behind BOTH Save (which imports it as a Library asset) and
// Download (in its re-encoding mode), so this single `image/png` is what keeps an edited cut-out a
// cut-out.
export function workingToPngFile(work, filename, { documentRef = globalThis.document } = {}) {
  return new Promise((resolve, reject) => {
    if (!work) {
      reject(new Error("No working image."));
      return;
    }
    const canvas = compositeWorkingToCanvas(work, { documentRef });
    const base = (work.source.name || "image").replace(/\.[^./\\]+$/, "");
    const name = filename || `${base}.png`;
    canvas.toBlob((blob) => {
      canvas.width = 0;
      canvas.height = 0;
      if (!blob) {
        reject(new Error("Could not encode the working image."));
        return;
      }
      resolve(new File([blob], name, { type: "image/png" }));
    }, "image/png");
  });
}

// The shared desktop-dialog / browser-anchor save (sc-6554), under the editor's historical name.
export const exportEditorFile = saveExportFile;

// ── What Download will actually contain (sc-15954, epic 15945) ───────────────
//
// sc-15948 puts the sanitized recipe INSIDE every generated PNG. Every other way a file leaves
// this app hands those bytes through unchanged, so the chunk survives for free. The editor was
// the exception: `Download` always went through `canvas.toBlob`, which re-encodes the composite
// and keeps nothing but pixels — so a user who opened a generated image, changed nothing, and
// downloaded it got a file with the recipe silently gone.
//
// The decision recorded on sc-15954, in one sentence:
//
//   An untouched PNG downloads the file you opened, byte for byte. A changed document downloads a
//   fresh PNG of what you see, with no recipe in it — and the header says which one you will get.
//
// # Why no `editedAfterGeneration` flag on the envelope
//
// The story's leaning was to re-embed the original envelope marked as provenance. Three reasons
// not to, in order of weight.
//
// 1. The envelope's extension rule is "adding a field does not need a version bump: an older
//    reader drops what it does not recognize" (docs/workflow-share-envelope.md). That rule holds
//    for fields whose absence is benign, and inverts for this one: an older build drops the flag
//    and then presents the envelope as a recipe that reproduces the image — the exact
//    misstatement this epic exists to stop, shipped by construction. Making it safe needs a
//    `schemaVersion` bump (which stops every older build reading every shared file, edited or
//    not) or a new marker kind that older builds refuse as `UnsupportedKind`. Either is a
//    contract change of the same size as the video lane, for a provenance note.
// 2. "The recipe this came from" is not well defined here. The document composites N layers from
//    N sources while `working.source` names one; an AI op replaces the base layer with the output
//    of a DIFFERENT job with its own model, prompt and seed; a crop changes the content and the
//    geometry.
// 3. It would be a new prompt egress, not a preserved one. The prompt is in the file as authored;
//    attaching it to a hand-edited composite the user believes is a fresh rasterization is a
//    surprise in the privacy direction too.
//
// # Why the geometry in the last AC cannot lie
//
// Structurally, not by a check. The only branch that ships an envelope ships the ORIGINAL FILE,
// so its `width`/`height` and its pixels are the same file's by definition. The branch that can
// be a different size — `boundedEditorCanvasDimensions` downscaling on load, a crop, an outpaint
// — is the branch that writes no envelope at all.
//
// # Why the untouched test is blob identity and not `dirty`
//
// `dirty` is not a statement about the bitmap: `runSave` clears it on a thoroughly edited
// document, so "not dirty" would let an edited composite take the passthrough branch. `edits` is
// not one either — layer transforms, opacity and blend changes never append to it. The one thing
// that IS a statement about the bitmap is reference identity between the single layer's blob and
// the blob the layer stack was installed from: every mutation the editor has (crop, colour grade,
// AI write-back, added layer, transform) replaces or supplements that object. Undo restores the
// same blob by reference (`snapshotLayers` shares it), so undoing back to the opened image
// correctly re-enables the passthrough.
//
// # Who decides whether the source carries a recipe
//
// `POST /api/v1/workflows/inspect`, which runs the one reader
// (`sceneworks_core::workflow_png::read_workflow_chunk`). Not a chunk walker in this app — see the
// header of `editorSourceWorkflow.js` for the five files a second implementation got wrong. The
// consequence for the plan is that the verdict is one of FOUR states and arrives late, so both
// `pending` and `unknown` have to survive all the way to the pill rather than being flattened into
// "no recipe" on the way.

/// The desktop shell's export ceiling, mirrored from `MAX_EXPORT_BYTES` in
/// `apps/desktop/src/settings.rs` and pinned to it by
/// `the_desktop_export_ceiling_matches_the_editors_mirror_of_it`.
///
/// It has to be known HERE rather than discovered from the error, because the byte-exact
/// passthrough is what can exceed it: before sc-15954 every editor download was a bounded canvas
/// raster and this limit was unreachable, and afterwards an untouched 300 MB source would hand the
/// shell 300 MB and get a hard "exceeds the 256 MB desktop limit" on a file that used to download
/// fine. So the plan falls back to the raster for those — the pre-sc-15954 behaviour, which still
/// works — and the pill says so BEFORE the click rather than the shell erroring after it. The
/// browser path has no such ceiling (it writes through an object URL) and keeps the passthrough.
export const MAX_DESKTOP_EXPORT_BYTES = 256 * 1024 * 1024;

/// Whether a layer sits 1:1 over the document with nothing done to it.
export function layerIsUntouched(layer, working) {
  const transform = layer.transform ?? identityTransform();
  return (
    layer.visible !== false &&
    layer.opacity === 1 &&
    (layer.blendMode || DEFAULT_BLEND_MODE) === DEFAULT_BLEND_MODE &&
    transform.x === 0 &&
    transform.y === 0 &&
    transform.scaleX === 1 &&
    transform.scaleY === 1 &&
    transform.rotation === 0 &&
    working.width === (layer.image?.naturalWidth ?? -1) &&
    working.height === (layer.image?.naturalHeight ?? -1)
  );
}

/// Why an untouched document is being rasterized anyway. Null when it is not.
export const RASTER_REASON_EDITED = null;

export const RASTER_REASON_DESKTOP_CAP = "desktop-export-cap";

/// What `Download` will produce for this document. Pure, so the claim the header prints and the
/// bytes the export writes are decided in ONE place and cannot drift.
///
/// * `mode: "original"` — hand the opened file through untouched.
/// * `mode: "raster"` — flatten the layer stack to a new PNG, which carries no recipe ever.
///   `reason` says why, since one of the two reasons is not "you edited it".
///
/// `sourceWorkflow` is the reader's verdict about the OPENED file, one of the four
/// `SOURCE_WORKFLOW_*` states. It is a parameter and not a field on `working` because it resolves
/// asynchronously and must not be baked into an undo snapshot: the pill, the export branch and the
/// banner all read this one value, and `pending` / `unknown` reach all three intact.
export function editorDownloadPlan(
  working,
  sourceWorkflow = SOURCE_WORKFLOW_UNKNOWN,
  { desktop = isDesktop } = {},
) {
  const original = working?.source?.originalExport ?? null;
  // With no `originalExport` the source is not a PNG this editor can hand back — a JPEG, a blank
  // canvas, an AI result. None of those can carry an envelope, so the question is ANSWERED rather
  // than open, and `absent` is the honest state instead of `unknown`.
  const workflow = original ? sourceWorkflow : SOURCE_WORKFLOW_ABSENT;
  const present = workflow === SOURCE_WORKFLOW_PRESENT;
  const layers = working?.layers ?? [];
  const untouched =
    Boolean(original) &&
    layers.length === 1 &&
    layers[0].blob === original.installedBlob &&
    layerIsUntouched(layers[0], working);
  const overDesktopCap =
    untouched && desktop && Number(original.blob?.size) > MAX_DESKTOP_EXPORT_BYTES;
  if (untouched && !overDesktopCap) {
    return {
      mode: "original",
      reason: RASTER_REASON_EDITED,
      filename: original.filename,
      workflow,
      carriesWorkflow: present,
      hadWorkflow: present,
    };
  }
  return {
    mode: "raster",
    reason: overDesktopCap ? RASTER_REASON_DESKTOP_CAP : RASTER_REASON_EDITED,
    filename: editedFilename(working?.source),
    workflow,
    carriesWorkflow: false,
    hadWorkflow: present,
  };
}

/// The sentence about RESOLUTION, for a source the editor resampled on load — or "" when it did
/// not resample.
///
/// This exists because sc-15954 made the old banner false. `boundedEditorCanvasDimensions`
/// downscales anything over `EDITOR_CANVAS_MAX_SIDE`, and the banner used to say the scaling was
/// "for reliable WebKit editing and export" — true until the untouched branch started exporting
/// the ORIGINAL file at its original size. One sentence, rendered in both places that talk about
/// it (the canvas banner and the pill beside Download), so the two cannot say different things.
export function downloadSizeSentence(plan, downscaled) {
  if (!plan || !downscaled?.scaled) return "";
  const source = `${downscaled.sourceWidth} × ${downscaled.sourceHeight}`;
  const working = `${downscaled.width} × ${downscaled.height}`;
  if (plan.mode === "original") {
    return `Download saves the original file at ${source}, not the ${working} working copy on the canvas.`;
  }
  if (plan.reason === RASTER_REASON_DESKTOP_CAP) {
    return `Download writes a new PNG at the ${working} working size, not the original ${source}.`;
  }
  return `This image has been edited, so Download writes a new PNG at the ${working} working size rather than the original ${source}.`;
}

/// The desktop-ceiling sentence, or "".
export function desktopCapSentence(plan) {
  if (plan?.reason !== RASTER_REASON_DESKTOP_CAP) return "";
  return (
    `This file is larger than the ${Math.round(MAX_DESKTOP_EXPORT_BYTES / (1024 * 1024))} MB ` +
    "the desktop save dialog accepts, so Download writes a new PNG of the working copy instead of " +
    "handing the original file through."
  );
}

export function joinSentences(...parts) {
  return parts.filter(Boolean).join(" ");
}

/// The line beside the Download button, or null when there is genuinely nothing to say — an
/// ordinary-sized image the reader confirmed carries no recipe, which is most images in the world.
///
/// Every other combination gets a sentence, including the two that a first cut left silent:
///
/// * **`pending` / `unknown`** render rather than reading as "no recipe". A verdict that has not
///   arrived, or one the reader could not produce, is a different fact from a verdict of "no", and
///   collapsing the three is what let an over-ceiling export ship a recipe with no pill at all.
/// * **A resampled source with no recipe** renders too, because sc-15954 changed which RESOLUTION
///   the file downloads at and `hadWorkflow` has nothing to do with that.
export function editorDownloadNote(plan, { downscaled = null } = {}) {
  if (!plan) return null;
  const size = downloadSizeSentence(plan, downscaled);
  const cap = desktopCapSentence(plan);
  if (plan.workflow === SOURCE_WORKFLOW_PRESENT) {
    if (plan.mode === "original") {
      return {
        tone: "included",
        label: "Recipe included",
        detail: joinSentences(
          "Nothing has been changed, so Download saves the file you opened byte for byte — " +
            "including the SceneWorks recipe embedded in it.",
          `To send a copy without the recipe, use “${SAVE_WITHOUT_WORKFLOW_LABEL}” on this image ` +
            "in the Library — saving from the editor rasterizes, so it drops the recipe along " +
            "with everything else the file carries.",
          size,
        ),
      };
    }
    return {
      tone: "dropped",
      label: "Recipe not carried",
      detail: joinSentences(
        cap ||
          "This image has been edited, so Download writes a new PNG of what you see. The " +
            "SceneWorks recipe from the image you opened is not in it — it describes a run that " +
            "did not produce these pixels.",
        cap && "The SceneWorks recipe embedded in the original is not in it.",
        size,
      ),
    };
  }
  if (plan.workflow === SOURCE_WORKFLOW_PENDING) {
    return {
      tone: "pending",
      label: "Checking for a recipe…",
      detail: joinSentences(
        "SceneWorks is reading the file you opened to find out whether it carries an embedded " +
          "recipe. Until that answers, what Download will contain is not yet known.",
        size,
      ),
    };
  }
  if (plan.workflow === SOURCE_WORKFLOW_UNKNOWN) {
    return {
      tone: "unknown",
      label: "Recipe unknown",
      detail: joinSentences(
        "SceneWorks could not read the file you opened, so whether it carries an embedded recipe " +
          "is unknown — not that it has none.",
        plan.mode === "original"
          ? "Nothing has been changed, so Download saves the file byte for byte and anything in " +
              "it travels."
          : joinSentences(
              cap ||
                "This image has been edited, so Download writes a new PNG of what you see and " +
                  "anything the original carried is not in it.",
              cap && "Anything the original carried is not in it.",
            ),
        size,
      ),
    };
  }
  // `absent`: the reader walked the file and found nothing, so there is no recipe to speak about.
  // What is left is what sc-15954 changed about the FILE ITSELF — which resolution it downloads
  // at, and whether it is the original at all. Both are silent otherwise.
  if (!size && !cap) return null;
  return {
    tone: "size",
    label: cap
      ? "Not the original file"
      : plan.mode === "original"
        ? "Original size"
        : "Working size",
    detail: joinSentences(cap, size),
  };
}

// Decode a blob into an HTMLImageElement via a same-origin object: URL. Asset
// files are served cross-origin from the API in local dev, so loading the bytes
// this way (rather than an <img crossOrigin> against the file URL) guarantees the
// Konva canvas is never tainted — later crop/export (sc-2430/sc-2434) need to read
// pixels back. Resolves { image, objectUrl }; caller owns revoking objectUrl.

export function colorKeyMaskCanvasForImage(image, seed, options) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  imageData.data.set(colorKeySelectionMaskRgba(imageData.data, canvas.width, canvas.height, { ...options, ...seed }));
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

export function maskCutoutCanvasForImage(image, maskCanvas, keepSelected) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const mask = maskCanvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
  imageData.data.set(applyMaskSelectionToRgba(imageData.data, maskAlphaFromRgba(mask.data), keepSelected));
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

export function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the cutout."))), "image/png");
  });
}

// Decode an editor source and, only when it exceeds the conservative WebKit
// canvas ceiling, rasterize it directly into a bounded surface before any Konva
// or export canvas is allocated. The caller owns the returned object URL.
export async function blobToEditorImage(blob, options) {
  const decoded = await blobToImage(blob, options);
  const dimensions = boundedEditorCanvasDimensions(
    decoded.image.naturalWidth,
    decoded.image.naturalHeight,
  );
  if (!dimensions.scaled) {
    return { ...decoded, blob, downscaled: null };
  }

  const canvas = document.createElement("canvas");
  canvas.width = dimensions.width;
  canvas.height = dimensions.height;
  let resizedBlob;
  try {
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Could not create a 2D editor canvas.");
    }
    context.drawImage(
      decoded.image,
      0,
      0,
      dimensions.width,
      dimensions.height,
    );
    resizedBlob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/png"),
    );
  } finally {
    URL.revokeObjectURL(decoded.objectUrl);
    canvas.width = 0;
    canvas.height = 0;
  }
  if (!resizedBlob) {
    throw new Error("Could not prepare this large image for the editor.");
  }
  const resized = await blobToImage(resizedBlob, options);
  return {
    ...resized,
    blob: resizedBlob,
    downscaled: dimensions,
  };
}

// Revoke the object URLs of a set of live layers (sc-6117). Undo snapshots hold
// only blobs (no URLs), so the only URLs that ever need revoking are the live
// ones, when their layer is evicted — on delete, on a session replace, and on
// unmount. Tolerant of null/missing URLs so callers don't have to guard.
export function revokeLayerUrls(layers) {
  for (const layer of layers ?? []) {
    if (layer?.objectUrl) URL.revokeObjectURL(layer.objectUrl);
  }
}

// Wait for every layer so successful decodes remain owned even if another fails.
export async function collectCropLayers(layers, decode, isCurrent) {
  const results = await Promise.allSettled(layers.map(decode));
  const cropped = results.filter((result) => result.status === "fulfilled").map((result) => result.value);
  const failure = results.find((result) => result.status === "rejected");
  if (failure || !isCurrent()) {
    revokeLayerUrls(cropped);
    if (failure && isCurrent()) throw failure.reason;
    return null;
  }
  return cropped;
}
