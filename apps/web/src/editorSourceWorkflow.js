// Does the file the Image Editor opened carry a workflow a RECIPIENT can read? (sc-15954, epic 15945)
//
// The Image Editor is the one egress path that does not hand a file through unchanged, so it is
// the one path that has to say what its download will contain. This module answers that question —
// and, more importantly, decides WHO answers it.
//
// # There is one reader, and it is the Rust one
//
// The first cut of this module walked the PNG chunk framing in JavaScript: signature, chunk
// lengths, `iTXt` keyword match, stop at IEND. It was a second implementation of
// `read_workflow_chunk` (`crates/sceneworks-core/src/workflow_png.rs`), and a second
// implementation of a predicate is a second set of answers. Measured against the Rust reader over
// a 17-file corpus it disagreed on five, in both directions:
//
// * **It pooled pre-IDAT and post-IDAT chunks.** A file carrying one of each counted 2 and
//   answered "no recipe"; the reader has a PRECEDENCE rule (pass one is the pre-IDAT metadata, the
//   tail is consulted only when pass one is empty) and returns the recipe.
// * **It gave up over a 64 MiB scan ceiling** and answered "no recipe" for anything larger — which
//   an 8192² SeedVR2 upscale output routinely is.
// * **It said "recipe" for four files the reader refuses**: malformed JSON under the keyword, a
//   `schemaVersion` from a newer build, a marker `kind` that is not `image`, and text over
//   `MAX_WORKFLOW_TEXT_BYTES`. Framing is not readability, and only the reader knows the
//   difference.
//
// Each false negative violated an acceptance criterion in one of two directions, because the same
// field feeds both "will the recipe go out" and "was there something to lose": an untouched export
// shipped a recipe with no pill at all, and an edited export dropped a real recipe with no pill at
// all. So the walker is gone. `POST /api/v1/workflows/inspect` (sc-15950) runs
// `read_workflow_chunk` itself, the sc-15951 drop panel already calls it on every dropped file,
// and this module asks the same endpoint the same way.
//
// # Four states, because "not known" is not "no"
//
// The answer is a round trip now, so there is a window before it arrives and a way for it to
// fail. Neither may collapse into `absent` — that is precisely the silent-loss outcome the ACs
// forbid — so both are their own state and both render a note:
//
// * `pending` — the file went up, the answer has not come back.
// * `present` / `absent` — the reader spoke.
// * `unknown` — nobody looked, or the look failed: a PNG over the endpoint's own cap, an offline
//   app, a 500. The passthrough still happens byte for byte, so whatever the file carries still
//   travels; what is missing is the sentence saying so, and the UI says THAT instead of guessing.

import {
  WORKFLOW_STATUS_NO_WORKFLOW,
  WORKFLOW_STATUS_WORKFLOW,
  blobIsPng,
  looksLikeWorkflowCandidate,
} from "./workflowShare.js";

/// The file went to the reader; the answer has not come back yet.
export const SOURCE_WORKFLOW_PENDING = "pending";
/// `read_workflow_chunk` parsed an envelope out of it.
export const SOURCE_WORKFLOW_PRESENT = "present";
/// `read_workflow_chunk` walked it and found no chunk of ours. The ordinary case for every image
/// in the world that did not come out of a SceneWorks generation.
export const SOURCE_WORKFLOW_ABSENT = "absent";
/// Nobody looked, or the look failed. NEVER folded into [`SOURCE_WORKFLOW_ABSENT`].
export const SOURCE_WORKFLOW_UNKNOWN = "unknown";

/// The name an untouched PNG downloads under.
///
/// Deliberately NOT `editedFilename`'s `-edited` suffix: this branch hands back the file that was
/// opened, byte for byte, and calling it "edited" would be the same category of misstatement the
/// rest of this story is about. Always `.png`, because the branch only runs for a PNG source.
export function originalExportFilename(name) {
  const base = (name || "image").replace(/\.[^./\\]+$/, "").trim() || "image";
  return `${base}.png`;
}

/// Measure an opened blob: is it a PNG the editor can hand back unchanged, and can its recipe
/// question be put to the reader at all?
///
/// Returns null when the answer is "re-rasterize it" — a non-PNG, or a blob that could not be
/// read. Null is the pre-sc-15954 behaviour, so every failure degrades to what the editor did
/// before rather than to a wrong claim. A JPEG cannot carry the envelope, so nothing is lost by
/// re-encoding one.
///
/// `workflow` is the STARTING state only. `pending` means the caller should now ask the endpoint;
/// `unknown` means it must not — a PNG past `MAX_WORKFLOW_INSPECT_BYTES` would be staged to disk
/// server-side and refused with a 413, so uploading it buys a slower way to learn nothing. The
/// passthrough is unaffected either way: the bytes go out whole, chunk and all.
export async function describeOpenedImage(blob, name) {
  if (!(await blobIsPng(blob))) {
    return null;
  }
  return {
    blob,
    filename: originalExportFilename(name),
    workflow: (await looksLikeWorkflowCandidate(blob))
      ? SOURCE_WORKFLOW_PENDING
      : SOURCE_WORKFLOW_UNKNOWN,
  };
}

/// The reader's verdict, from a `POST /api/v1/workflows/inspect` body.
///
/// The endpoint has exactly two success shapes and everything else THROWS (a typed 4xx/5xx), so
/// this maps only what a 200 can say and the caller maps the throw to [`SOURCE_WORKFLOW_UNKNOWN`].
/// The default arm matters: a body from a newer server carrying a third status is not "no recipe",
/// it is a status this build does not understand.
export function workflowStateFromInspect(body) {
  if (body?.status === WORKFLOW_STATUS_WORKFLOW) {
    return SOURCE_WORKFLOW_PRESENT;
  }
  if (body?.status === WORKFLOW_STATUS_NO_WORKFLOW) {
    return SOURCE_WORKFLOW_ABSENT;
  }
  return SOURCE_WORKFLOW_UNKNOWN;
}
