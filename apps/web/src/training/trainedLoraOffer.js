// sc-24815 — the return edge of the character identity loop (references → dataset →
// LoRA → generation). A completed `lora_train` run already registers its adapter into
// the LoRA catalog (`register_trained_lora`, apps/rust-api/src/jobs.rs); this decides
// whether that run also deserves an OFFER to attach the adapter to the character that
// owns the dataset it trained from.
//
// All of the decision logic lives here, with no React in it, because the one failure
// mode that matters is data corruption rather than a missed prompt:
// `character_store::attach_lora` mints a fresh `character_lora_<hex>` link id on every
// call and PREPENDS it (crates/sceneworks-core/src/character_store.rs:612) — attaching
// the same adapter twice is a duplicate row, not an upsert. So "ask once per job id,
// across re-render, reload and project switch" is the whole contract, and it is
// testable here without a component.
//
// TrainingStudio.jsx owns only when to run this and what to render; it never attaches
// on its own initiative — the caller attaches only from the accept button.

import { extractFamilies } from "../presetUtils.js";

// Decided job ids are remembered per project: the dataset, the character and the LoRA
// are all project-scoped, and one key per project means a project switch can neither
// leak another project's ids nor have two windows clobber one shared blob.
const DECISIONS_KEY_PREFIX = "sceneworks-trained-lora-offer:";

// Bounded so a long-lived install cannot grow the blob without limit. The ids only
// have to outlive a reload of the run they belong to, so evicting the oldest is safe:
// the worst case is re-asking about a run the freshness window below already rules out.
const MAX_REMEMBERED_DECISIONS = 200;

// Only offer a run that finished recently. Without this the offer is derived purely
// from the job list, so the first launch after upgrading would interrupt the user with
// an offer for the last character run in their history — possibly months old — and the
// Queue's terminal-job retention (jobs_store.rs) is far longer than a useful prompt.
export const TRAINED_LORA_OFFER_WINDOW_MS = 24 * 60 * 60 * 1000;

// The ways a completed `lora_train` job produces NOTHING to attach, each of which must
// surface no offer at all:
//   * a base-checkpoint (full fine-tune) run — its registrar reports under
//     `baseCheckpointRegistered` and leaves `loraRegistered` absent (sc-15036);
//   * a registration that FAILED — `loraRegistered: false` + `loraRegistrationError`,
//     so there is no catalog entry to attach;
//   * a dry run — never registers.
// Requiring strict `loraRegistered === true` plus a non-empty `loraId` covers all
// three, and also excludes a `control_training` overlay run, a different job type.
export function isRegisteredAdapterTrainingJob(job) {
  return (
    job?.type === "lora_train" &&
    job?.status === "completed" &&
    job?.result?.loraRegistered === true &&
    typeof job?.result?.loraId === "string" &&
    job.result.loraId !== ""
  );
}

function completedAtMs(job) {
  const raw = job?.completedAt ?? job?.updatedAt ?? job?.createdAt;
  const parsed = raw ? Date.parse(raw) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : null;
}

// The attach payload, field for field identical to Character Studio's manual
// "Attach imported LoRA" submit (`submitLora`, apps/web/src/screens/CharacterStudio.jsx:572),
// so an offered attach and a hand-made one cannot drift into two link shapes.
//
// `sourcePath` is deliberately catalog-only. Character Studio can only ever attach a
// catalog entry, whose `installedPath` the API resolved to an absolute path
// (apps/rust-api/src/loras.rs:858). A run's staged `manifestEntry` carries
// `source.path` RELATIVE by design — "loras/<id>", kept that way on purpose at
// apps/rust-api/src/jobs.rs:1851 so `normalize_lora_entry` can resolve it under the
// scope root — and the store rejects a relative attach source: `copy_lora_into_project`
// does `PathBuf::from(source_path)` then requires `.exists()` plus containment in
// `data/loras` or `project/loras` (crates/sceneworks-core/src/character_store.rs:1351),
// so sending it answers with 400 "LoRA source path not found: loras/<id>" on every
// retry. `null` is the supported "no file to copy" answer — the same function returns
// `Ok((None, false))` for it (character_store.rs:1362) and the link still resolves to
// the catalog adapter by id. That matters because the Train route does NOT hydrate the
// `loras` domain (apps/web/src/appHydration.js:38) and nothing refreshes the catalog on
// training completion, so the manifest entry is frequently the only source available.
export function characterLoraAttachPayload(source) {
  const loraId = typeof source?.id === "string" ? source.id : "";
  if (!loraId) {
    return null;
  }
  const name = String(source?.name ?? "").trim() || loraId;
  return {
    loraId,
    name,
    sourcePath: source?.installedPath ?? null,
    triggerWords: source?.triggerWords ?? [],
    defaultWeight: source?.defaultWeight ?? 1.0,
    compatibility: { families: extractFamilies(source) },
    scope: source?.scope ?? "global",
  };
}

// Where a completed run records the dataset it trained from.
//
// NOT `payload.datasetId`. A `lora_train` payload is assembled from scratch at
// apps/rust-api/src/training.rs:1872 and carries only `dryRun`, `outputName`, `plan`,
// `manifestEntry` and `baseModel`; the top-level `datasetId` belongs to the caption,
// parquet-import, analysis, face-analysis and upscale jobs (training.rs:565, 706, 904,
// 1004, 1188), so reading it here would match nothing and the offer would never appear.
// The worker writes `datasetId` into the RESULT
// (crates/sceneworks-worker/src/training_jobs.rs:2625), which is also the object the
// registrar status is folded into (`result.extend(status)`, apps/rust-api/src/jobs.rs:1417).
// The submit-time copy survives at `manifestEntry.provenance.datasetId`
// (training.rs:1765, asserted by apps/rust-api/src/tests/training.rs:1988) and covers a
// run whose result predates the result field.
export function trainedLoraDatasetId(job) {
  const fromResult = job?.result?.datasetId;
  if (typeof fromResult === "string" && fromResult !== "") {
    return fromResult;
  }
  const fromProvenance = job?.payload?.manifestEntry?.provenance?.datasetId;
  return typeof fromProvenance === "string" && fromProvenance !== "" ? fromProvenance : "";
}

// The single candidate to offer right now, or null. `jobs` is newest-first
// (`upsertJobNewest`), so scanning in order offers the most recent undecided run and
// leaves an older undecided one to surface once this one is answered.
//
// Every lookup is a fail-closed filter: no project match, no dataset, no character
// association, an archived character, or an unresolvable LoRA id each yield no offer.
export function trainedLoraOfferCandidate({
  jobs,
  datasets,
  characters,
  loras,
  projectId,
  decidedJobIds,
  now,
  windowMs = TRAINED_LORA_OFFER_WINDOW_MS,
} = {}) {
  if (!projectId) {
    return null;
  }
  const decided = decidedJobIds instanceof Set ? decidedJobIds : new Set(decidedJobIds ?? []);
  const reference = Number.isFinite(now) ? now : Date.now();
  const datasetsById = new Map((datasets ?? []).filter((item) => item?.id).map((item) => [item.id, item]));
  const lorasById = new Map((loras ?? []).filter((item) => item?.id).map((item) => [item.id, item]));

  for (const job of jobs ?? []) {
    if (!isRegisteredAdapterTrainingJob(job) || decided.has(job.id)) {
      continue;
    }
    if (job.projectId !== projectId) {
      continue;
    }
    const finishedAt = completedAtMs(job);
    if (finishedAt === null || reference - finishedAt > windowMs) {
      continue;
    }
    const datasetId = trainedLoraDatasetId(job);
    if (!datasetId) {
      continue;
    }
    const characterId = datasetsById.get(datasetId)?.characterId;
    if (typeof characterId !== "string" || !characterId) {
      continue;
    }
    const character = (characters ?? []).find((item) => item?.id === characterId);
    if (!character || character.archived) {
      continue;
    }
    const loraId = job.result.loraId;
    const manifestEntry = job.payload?.manifestEntry;
    const source = lorasById.get(loraId) ?? (manifestEntry?.id === loraId ? manifestEntry : null);
    const payload = characterLoraAttachPayload(source ?? { id: loraId });
    if (!payload) {
      continue;
    }
    return {
      jobId: job.id,
      projectId,
      loraId,
      loraName: payload.name,
      characterId,
      characterName: character.name ?? characterId,
      payload,
    };
  }
  return null;
}

// Exported so the caller can listen for the storage event on exactly this key: another
// window answering the same run must retire the offer here too.
export function trainedLoraDecisionsKey(projectId) {
  return DECISIONS_KEY_PREFIX + projectId;
}

// Job ids the user has already answered (accepted or declined) for this project. Read
// fresh every call so a reload — the case an in-memory guard cannot cover — sees them.
// A missing/blocked store yields empty, which re-asks; re-asking is safe because the
// attach itself only ever happens from the accept button.
export function readTrainedLoraOfferDecisions(projectId) {
  if (!projectId) {
    return [];
  }
  try {
    const raw = globalThis.localStorage?.getItem(trainedLoraDecisionsKey(projectId));
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((id) => typeof id === "string" && id !== "") : [];
  } catch {
    return [];
  }
}

export function rememberTrainedLoraOfferDecision(projectId, jobId) {
  if (!projectId || !jobId) {
    return;
  }
  try {
    const remembered = readTrainedLoraOfferDecisions(projectId).filter((id) => id !== jobId);
    globalThis.localStorage?.setItem(
      trainedLoraDecisionsKey(projectId),
      JSON.stringify([jobId, ...remembered].slice(0, MAX_REMEMBERED_DECISIONS)),
    );
  } catch {
    // localStorage blocked (private mode, quota). The caller's in-memory per-project
    // record still holds it — mergeTrainedLoraOfferDecisions unions the two — so the
    // answer survives a project round trip in this window; only a reload can re-ask.
  }
}

// The decided set the offer must consult: what this window answered in memory, keyed by
// project, UNION what storage recorded. Neither half is sufficient alone — storage does
// not survive a blocked/quota-limited store, and the in-memory map does not survive a
// reload. Both are read on every check, so an answer recorded by another window or a
// project switch cannot leave a stale offer standing.
export function mergeTrainedLoraOfferDecisions(projectId, sessionDecisions) {
  const merged = new Set(readTrainedLoraOfferDecisions(projectId));
  const session =
    sessionDecisions instanceof Set ? sessionDecisions : new Set(sessionDecisions ?? []);
  for (const jobId of session) {
    merged.add(jobId);
  }
  return merged;
}
