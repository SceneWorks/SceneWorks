import { apiFetch } from "./api.js";

// The client half of the universal checkpoint-import seam (epic 20398, sc-20650).
//
// Two ownerships, ONE flow. "Use existing model library" (linked) references a user's own library
// in place; "Add to SceneWorks" (managed) copies bytes into app-owned storage. They differ only in
// what they name — a `(rootId, relativePath)` pair versus a transfer source — and both end at
// `POST /api/v1/models/import`, which queues the same job the same progress card renders. Nothing
// here re-implements a second import path.
//
// THREE properties this module owns, none of them cosmetic:
//
// 1. **A typed refusal is never swallowed into a default.** Every plan-store rejection arrives as
//    `code: "checkpoint_library_rejected"` with the store's own stable kebab-case code in
//    `context.reason`, and a message that already reads `[checkpoint-plan:<code>] …`. The UI
//    branches on the CODE and shows the store's own sentence; `describeRefusal` never invents a
//    "something went wrong" fallback, because a refusal the user cannot name is a refusal the user
//    cannot act on.
// 2. **The corrective action comes from the seam.** `Needs Relink` and `Needs Rescan` are states the
//    server computed; the client maps state → affordance and never re-derives availability from a
//    path or a missing file.
// 3. **Removal is ownership-discriminated.** Deleting a managed model tears down bytes SceneWorks
//    wrote. Forgetting a linked library drops SceneWorks' own plan documents and NEVER touches the
//    user's files. Those are different promises and they get different words.

export const OWNERSHIP_LINKED = "linked";
export const OWNERSHIP_MANAGED = "managed";

// The two ownership choices, in the order the screen offers them. Linked leads: a user who already
// has a checkpoint collection should not have to copy it to use it.
export const OWNERSHIP_CHOICES = Object.freeze([
  Object.freeze({
    id: OWNERSHIP_LINKED,
    label: "Use existing model library",
    summary: "Point SceneWorks at a folder you already keep checkpoints in. Nothing is copied or moved.",
  }),
  Object.freeze({
    id: OWNERSHIP_MANAGED,
    label: "Add to SceneWorks",
    summary: "Bring a checkpoint in from a file, a folder, a URL, Hugging Face or Civitai. SceneWorks stores and owns the copy.",
  }),
]);

// The five managed inputs. `kind` is the discriminant `ModelImportSourceV1` deserializes
// (`apps/rust-api/src/dto.rs`), so these strings are a wire contract, not labels.
export const MANAGED_SOURCES = Object.freeze([
  Object.freeze({ kind: "upload", label: "Upload", field: "file", hint: "A checkpoint file on this device." }),
  Object.freeze({ kind: "localPath", label: "Local copy", field: "path", hint: "A file or folder already on this machine. The source is only read." }),
  Object.freeze({ kind: "url", label: "URL", field: "url", hint: "A direct download link." }),
  Object.freeze({ kind: "huggingFace", label: "Hugging Face", field: "repo", hint: "An owner/name repo. Gated repos use your stored Hugging Face token." }),
  Object.freeze({ kind: "civitai", label: "Civitai", field: "url", hint: "A Civitai download link. Records the model version and file ids." }),
]);

// Hosts whose stored credential a managed source consumes. Used to tell the user, before they
// queue, that this source will need a token they have not stored yet — rather than letting the
// download fail with a 401 twenty minutes in.
export const MANAGED_SOURCE_CREDENTIAL_HOST = Object.freeze({
  huggingFace: "huggingface.co",
  civitai: "civitai.com",
});

// The two typed codes `checkpoint_library.rs` answers with.
export const CHECKPOINT_LIBRARY_REJECTED_CODE = "checkpoint_library_rejected";
export const CHECKPOINT_LIBRARY_NOT_PERMITTED_CODE = "checkpoint_library_not_permitted";

// The linked states `LinkedCheckpointStateV1` serializes (snake_case on the wire).
export const LINKED_READY = "ready";
export const LINKED_NEEDS_RELINK = "needs_relink";
export const LINKED_NEEDS_RESCAN = "needs_rescan";

// ---------------------------------------------------------------------------------------------
// Transport. One function per route; no client invents an endpoint.
// ---------------------------------------------------------------------------------------------

export function fetchLibraryRoots(token, options) {
  return apiFetch("/api/v1/models/library-roots", token, options);
}

export function approveLibraryRoot(token, { path, label } = {}) {
  return apiFetch("/api/v1/models/library-roots", token, {
    method: "POST",
    body: JSON.stringify(label ? { path, label } : { path }),
  });
}

// Rename and relink are the same route and are independent: the server applies whichever halves are
// present and refuses a body carrying neither, so an empty edit is a refusal rather than a no-op.
export function updateLibraryRoot(token, rootId, { label, path } = {}) {
  const body = {};
  if (label != null) body.label = label;
  if (path != null) body.path = path;
  return apiFetch(`/api/v1/models/library-roots/${encodeURIComponent(rootId)}`, token, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

export function removeLibraryRoot(token, rootId) {
  return apiFetch(`/api/v1/models/library-roots/${encodeURIComponent(rootId)}`, token, {
    method: "DELETE",
  });
}

export function scanLibraryRoot(token, rootId, options) {
  return apiFetch(`/api/v1/models/library-roots/${encodeURIComponent(rootId)}/scan`, token, options);
}

export function rescanLibraryCheckpoint(token, rootId, relativePath) {
  return apiFetch(`/api/v1/models/library-roots/${encodeURIComponent(rootId)}/rescan`, token, {
    method: "POST",
    body: JSON.stringify({ relativePath }),
  });
}

// ---------------------------------------------------------------------------------------------
// Request bodies
// ---------------------------------------------------------------------------------------------

// A linked import names its checkpoint by `(rootId, relativePath)` and NOTHING else. It must not
// carry `ownershipMode` — the served enum has only `managed`, and `"linked"` is deliberately
// refused there (dto.rs `OwnershipModeV1`) — nor a repo/URL/path/discriminated source, which the
// route rejects outright.
export function linkedImportBody({ rootId, relativePath, name, type = "image", family } = {}) {
  const body = { linkedRootId: rootId, linkedRelativePath: relativePath, type };
  if (name) body.name = name;
  if (family) body.family = family;
  return body;
}

// A managed import states its source with the discriminated `source` object rather than leaving the
// server to infer it from whichever flat field happens to be set. `upload` carries no path: the
// staged path is filled in server-side from the multipart `file` part and is never accepted from
// client JSON, so the file rides as `file` and the metadata says only `kind`.
export function managedImportBody(input = {}) {
  const { kind, name, type = "image", family } = input;
  const body = { ownershipMode: OWNERSHIP_MANAGED, type };
  if (name) body.name = name;
  if (family) body.family = family;
  switch (kind) {
    case "upload":
      body.source = { kind: "upload" };
      break;
    case "localPath":
      body.source = { kind: "localPath", path: input.path ?? "" };
      break;
    case "url":
      body.source = { kind: "url", url: input.url ?? "" };
      if (input.expectedSha256) body.source.expectedSha256 = input.expectedSha256;
      break;
    case "huggingFace":
      body.source = { kind: "huggingFace", repo: input.repo ?? "" };
      if (input.revision) body.source.revision = input.revision;
      if (input.files?.length) body.source.files = input.files;
      break;
    case "civitai":
      body.source = { kind: "civitai", url: input.url ?? "" };
      if (input.modelVersionId) body.source.modelVersionId = input.modelVersionId;
      if (input.fileId) body.source.fileId = input.fileId;
      if (input.expectedSha256) body.source.expectedSha256 = input.expectedSha256;
      break;
    default:
      throw new Error(`Unknown managed import source: ${kind}`);
  }
  return body;
}

// What the submit button must not let through, stated per source so the user is told WHICH field is
// missing instead of watching a disabled button and guessing. Returns "" when the form can be sent.
export function managedSourceProblem(input = {}) {
  switch (input.kind) {
    case "upload":
      return input.file ? "" : "Choose a checkpoint file to upload.";
    case "localPath":
      return input.path?.trim() ? "" : "Enter the path of the file or folder to copy.";
    case "url":
      return input.url?.trim() ? "" : "Enter a download URL.";
    case "huggingFace":
      return input.repo?.trim() ? "" : "Enter a Hugging Face repo as owner/name.";
    case "civitai":
      return input.url?.trim() ? "" : "Enter a Civitai download URL.";
    default:
      return "Choose where the checkpoint comes from.";
  }
}

// ---------------------------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------------------------

// Which lifecycle action clears a given store refusal, keyed on the store's own `code()`. Anything
// absent here has no one-click corrective action — which is different from having no explanation,
// and is why `describeRefusal` still surfaces the server's sentence for those.
//
// This map and `CheckpointPlanError::lifecycle_action` on the Rust side are ONE contract, pinned by
// `packages/schemas/checkpoint-refusal-actions.json`, which both sides are tested against for
// equality. It is exported so that test can read it; nothing in the UI should need it directly.
export const REFUSAL_ACTION = Object.freeze({
  "root-unavailable": "relink",
  "unknown-root": "relink",
  "source-missing": "rescan",
  "source-drifted": "rescan",
  "path-escapes-root": "rescan",
  "plan-tampered": "rescan",
  "missing-plan": "rescan",
  "invalid-relative-path": "rescan",
  "unsupported-locator": "rescan",
});

export const REFUSAL_ACTION_LABEL = Object.freeze({
  relink: "Relink library",
  rescan: "Rescan checkpoint",
});

// The typed shape of a checkpoint-library refusal, or `null` for anything that is not one.
//
// Both halves are required: a bare `code` could survive a contract change that dropped the payload,
// and a reason with no code could come from an unrelated route.
export function checkpointLibraryRefusal(error) {
  const code = error?.code;
  if (code !== CHECKPOINT_LIBRARY_REJECTED_CODE && code !== CHECKPOINT_LIBRARY_NOT_PERMITTED_CODE) {
    return null;
  }
  const reason = error?.context?.reason;
  if (typeof reason !== "string" || reason === "") return null;
  // A typed refusal that carried no sentence still has to SAY something: this message is rendered
  // straight into the panel's `role="status"` region, and an empty string there is a refusal the
  // user never learns about. The reason code is the least the store already told us.
  const message = String(error.message ?? "").trim() || `[checkpoint-plan:${reason}]`;
  return { code, reason, message, action: REFUSAL_ACTION[reason] ?? null };
}

// Everything the UI needs to render ANY failure of this seam, typed or not.
//
// There is no "something went wrong" branch on purpose. A typed refusal is shown with the store's
// own `[checkpoint-plan:<code>] …` sentence — the actionable detail for an unrunnable source is the
// inspector's diagnostics inside that sentence, and for drift it is the two digests — and an
// untyped failure is shown with whatever message it carried. Collapsing either into a generic
// string is how a user ends up with a model that will not load and no way to find out why.
export function describeRefusal(error) {
  const typed = checkpointLibraryRefusal(error);
  if (typed) return typed;
  const message = String(error?.message ?? "").trim();
  return {
    code: error?.code ?? null,
    reason: null,
    message: message || "The request could not be completed.",
    action: null,
  };
}

// A local-only refusal: approving and relinking name an absolute host path, so the server refuses
// them from a LAN peer. The copy has to say that, not "forbidden".
export function isLocalOnlyRefusal(error) {
  return error?.code === CHECKPOINT_LIBRARY_NOT_PERMITTED_CODE;
}

// ---------------------------------------------------------------------------------------------
// Linked state → affordance
// ---------------------------------------------------------------------------------------------

// The corrective action for a persisted linked checkpoint's state (AC2). `Needs Relink` and
// `Needs Rescan` are NOT "missing" and must never render as "not installed": the plans are intact,
// the user has one button to press, and saying otherwise invites a re-download of bytes they
// already have.
export function linkedCorrection(status) {
  switch (status?.state) {
    case LINKED_NEEDS_RELINK:
      return {
        action: "relink",
        label: REFUSAL_ACTION_LABEL.relink,
        headline: "Needs relink",
        summary: "The folder holding this library is not available right now — it was moved, renamed, or its drive is not attached. Nothing was lost.",
        detail: status.detail ?? "",
      };
    case LINKED_NEEDS_RESCAN:
      return {
        action: "rescan",
        label: REFUSAL_ACTION_LABEL.rescan,
        headline: "Needs rescan",
        summary: "This checkpoint's file changed since SceneWorks last read it. Rescanning re-reads it in place and keeps the same model.",
        detail: status.detail ?? "",
      };
    case LINKED_READY:
      return null;
    default:
      return null;
  }
}

// Index the linked statuses a set of scans reports, keyed by checkpoint id, so a catalog row can
// find its own state in one lookup. Both `candidates[].status` and `unmatched[]` are folded in —
// an unmatched entry is a persisted checkpoint the scan did NOT see, which is exactly the case that
// must not silently vanish from the catalog.
export function linkedStatusIndex(scans = []) {
  const index = new Map();
  for (const scan of scans) {
    if (!scan) continue;
    for (const candidate of scan.candidates ?? []) {
      if (candidate?.status?.checkpointId) index.set(candidate.status.checkpointId, candidate.status);
    }
    for (const status of scan.unmatched ?? []) {
      if (status?.checkpointId) index.set(status.checkpointId, status);
    }
  }
  return index;
}

// The persisted checkpoint id a catalog row carries, or null for a row that is not plan-backed.
// Stamped by the worker as `importPlan.checkpointId` once a full-content compile succeeded, so its
// presence is what makes a row plan-backed at all.
export function modelCheckpointId(model) {
  const id = model?.importPlan?.checkpointId;
  return typeof id === "string" && id !== "" ? id : null;
}

// Which ownership a catalog row was installed under, read off its persisted checkpoint id — the
// same discriminator the server uses (`managed/<installId>` vs `linked/<rootId>/<relativePath>`).
// `null` for a row that predates plan-backed import; those keep the pre-epic delete behaviour.
export function modelOwnership(model) {
  const id = modelCheckpointId(model);
  if (id === null) return null;
  if (id.startsWith("managed/")) return OWNERSHIP_MANAGED;
  if (id.startsWith("linked/")) return OWNERSHIP_LINKED;
  return null;
}

// The linked state of a catalog row, given the scan index. `null` when the row is not linked or the
// scans have not loaded — the caller falls back to its ordinary install-state rendering.
export function modelLinkedStatus(model, index) {
  const id = modelCheckpointId(model);
  if (id === null || !id.startsWith("linked/")) return null;
  return index?.get?.(id) ?? null;
}

// ---------------------------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------------------------

const PROVIDER_LABEL = Object.freeze({
  "linked-library": "Linked library",
  huggingface: "Hugging Face",
  civitai: "Civitai",
  url: "URL",
  local: "Local copy",
  upload: "Upload",
});

// Where a plan-backed row's bytes came from, in the words the import used. Reads the persisted
// manifest `source` block; returns `null` when the row records none, so the UI omits the line
// rather than printing "unknown".
export function modelProvenance(model) {
  const source = model?.source;
  if (!source || typeof source !== "object") return null;
  const provider = typeof source.provider === "string" ? source.provider : null;
  if (!provider) return null;
  const reference =
    source.rootId && source.relativePath
      ? `${source.relativePath}`
      : source.repo || source.url || source.path || null;
  return {
    provider,
    label: PROVIDER_LABEL[provider] ?? provider,
    reference: reference ? String(reference) : null,
    rootId: source.rootId ? String(source.rootId) : null,
    relativePath: source.relativePath ? String(source.relativePath) : null,
  };
}

// ---------------------------------------------------------------------------------------------
// Duplicates
// ---------------------------------------------------------------------------------------------

// The checkpoint ids an import found already installed with the same content digest. The server
// records them and completes the import anyway — the same bytes under two names is legal — so this
// is a WARNING, never a failure, and it names the existing entries so the user can delete one.
export function duplicateCheckpointIds(job) {
  const ids = job?.result?.duplicateCheckpointIds;
  return Array.isArray(ids) ? ids.filter((id) => typeof id === "string" && id !== "") : [];
}

export function duplicateWarningText(job) {
  const ids = duplicateCheckpointIds(job);
  if (ids.length === 0) return "";
  return ids.length === 1
    ? `These are the same bytes as a checkpoint you already have (${ids[0]}). Both entries were kept.`
    : `These are the same bytes as ${ids.length} checkpoints you already have (${ids.join(", ")}). All entries were kept.`;
}

// ---------------------------------------------------------------------------------------------
// Removal copy — the two ownerships promise different things
// ---------------------------------------------------------------------------------------------

// What deleting this catalog row actually does, discriminated on ownership.
//
// The managed sentence is a deletion warning. The linked sentence is the opposite reassurance, and
// getting them the wrong way round is the single most damaging copy defect in this screen: a user
// told "this deletes the files" about a linked entry keeps a model they no longer want, and a user
// told "your files are untouched" about a managed entry loses bytes they thought were safe.
export function removalCopy(model) {
  const ownership = modelOwnership(model);
  const name = model?.name ?? model?.id ?? "this model";
  if (ownership === OWNERSHIP_LINKED) {
    return {
      ownership,
      title: "Remove from SceneWorks?",
      confirmLabel: "Remove",
      message: `${name} lives in your own model library. Removing it drops SceneWorks' record of it — the file itself is never opened, moved or deleted. You can add it again by rescanning the library.`,
    };
  }
  return {
    ownership: ownership ?? OWNERSHIP_MANAGED,
    title: "Delete model?",
    confirmLabel: "Delete",
    message: `${name} was copied into SceneWorks' own storage. Deleting it removes those files from this machine. Anything you imported it from is untouched.`,
  };
}

// The same discrimination for a whole library root. Forgetting a library is not deleting it, and
// the count of dropped records is the honest thing to name.
export function rootRemovalCopy(root, scan) {
  const label = root?.displayLabel ?? root?.label ?? root?.path ?? "this library";
  const count = (scan?.candidates ?? []).filter((candidate) => candidate?.status).length;
  return {
    title: "Forget this library?",
    confirmLabel: "Forget library",
    message: `SceneWorks will forget ${label}${count ? ` and the ${count} checkpoint${count === 1 ? "" : "s"} it has records for` : ""}. The folder and every file in it are left exactly as they are — SceneWorks never writes to a linked library.`,
  };
}
