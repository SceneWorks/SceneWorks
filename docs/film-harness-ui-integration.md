# How a UI calls the film harness

An implementation note for the Film workspace now present in the Video Editor. It describes the
shared library/CLI/route layering and the durable state documents. Operator behavior belongs in
[film-editor.md](film-editor.md); the harness architecture and CLI manual remain in
[film-harness-overview.md](film-harness-overview.md) and [film-harness.md](film-harness.md).

Some line references below describe the original harness commit and are retained as historical
navigation. Use the current source and router as the authority for the product routes.

## The layering today

```text
apps/rust-api/src/bin/film-harness.rs      arg parsing, signal handling, printing (a thin shell)
        │  calls
apps/rust-api/src/{film_harness,film_planner}.rs   the ORCHESTRATION, a rust-api library module
        │  drives, through ApiTransport
project-scoped film routes                 drafts, planning, runs, review, explicit export
        │  call
existing SceneWorks HTTP routes              projects, assets, timelines, jobs
```

- The orchestration lives in the **library**: `pub mod film_harness; pub mod film_planner;`
  (`apps/rust-api/src/lib.rs:283`–`:284`).
- The CLI is a thin shell over it. Every verb it offers is a library call:
  `film_harness::validate` (`apps/rust-api/src/film_harness.rs:1709`), `run` (`:4823`),
  `run_with_control` (`:4837`), `resume` (`:5200`), `replace_take` (`:5260`), `edit_timeline`
  (`:7129`), `request_cancel` (`:322`), `read_run_record` (`:378`);
  `film_harness::review::{review, decide_take, request_repair, review_eval}`
  (`apps/rust-api/src/film_harness/review.rs:769`, `:1604`, `:1757`, `:1924`);
  `film_planner::{generate, compile_existing}` (`apps/rust-api/src/film_planner.rs:667`, `:783`);
  `film_harness::references::make_references` (`apps/rust-api/src/film_harness/references.rs:149`).
- Everything the harness does to SceneWorks goes through `ApiTransport`
  (`apps/rust-api/src/film_harness.rs:176`), a two-method trait (`call` and `get_bytes`, `:190`),
  whose shipped implementation is `HttpTransport` (`:5897`). There is no privileged in-process
  backdoor; a UI-hosted run would use the same trait.
- Project-scoped routes expose drafts, references, sound, preflight, planning, run lifecycle, and
  explicit export. The review routes call the same decision, swap, replacement, repair, and bounded
  analysis functions as the CLI. Their router registrations are in `apps/rust-api/src/lib.rs` and
  their adapters are in `films.rs`, `film_planning.rs`, `film_lifecycle.rs`, and `film_review.rs`.
- **There is no in-memory state beyond the files.** The record on disk is the run
  (`crates/sceneworks-core/src/film_plan.rs:3726`).

### The routes it drives

Every one already exists and is used by other clients:

| area | routes |
| --- | --- |
| projects | `POST /api/v1/projects`, `GET /api/v1/projects/:id/files/*path` |
| assets | asset upload (multipart), `POST /api/v1/projects/:id/assets/:assetId/tags`, `GET/PUT /api/v1/projects/:id/assets/:assetId` |
| timelines | `POST /api/v1/projects/:id/timelines`, `GET/PUT /api/v1/projects/:id/timelines/:tid`, `POST /api/v1/projects/:id/timelines/:tid/exports`, `POST /api/v1/projects/:p/timelines/:t/items/:i/frames` |
| jobs | `POST /api/v1/video/jobs`, `POST /api/v1/audio/jobs`, `POST /api/v1/image/jobs`, `POST /api/v1/image/vqa/jobs`, `GET /api/v1/jobs`, `GET /api/v1/jobs/:id`, `GET /api/v1/jobs/:id/metrics`, job cancel |
| capability | `GET /api/v1/host-capabilities`, `GET /api/v1/models`, `GET /api/v1/loras`, `GET /api/v1/workers` |
| LLM | `POST /api/v1/prompts/refine` |
| film drafts | `/api/v1/projects/:project_id/films/...`: draft, render options, reference pack, sound, brief parse, planner availability, planning, preflight, run creation |
| film runs | `/api/v1/projects/:project_id/film-runs/...`: read/list/progress, start/resume/cancel, review decisions and bounded take mutations, explicit export |
| external planners | `/api/v1/film-planner-connections/...`: non-secret connection settings, connection test, optional model listing |

Route strings are in `apps/rust-api/src/film_harness.rs`,
`apps/rust-api/src/film_harness/{references,review}.rs` and `apps/rust-api/src/film_planner.rs`.

### Render regime contract

`FilmDraft.renderRegime` persists `recommended_turbo`, `quality`, or `custom`. A missing field is a
legacy draft and preserves its current `productionPlan.model.loras` and
`productionPlan.model.advanced.steps` without migration-time rewrites.

`GET /api/v1/projects/:project_id/films/:draft_id/render-options` resolves the saved document.
Editors preview unsaved model, resolution, conditioning, or reference changes with `POST` to the
same route:

```json
{
  "draftRevision": 3,
  "productionPlan": { "...": "the edited production plan" },
  "referencePack": { "...": "the edited reference pack" },
  "renderRegime": "recommended_turbo"
}
```

The response carries `selectedRegime`, `recommendedTurbo`, `quality`, and `effective`.
`recommendedTurbo` includes `available`, concrete `adapterIds`, `effectiveSteps`, and, when it is
unavailable, one stable reason: `model_unavailable`, `no_installed_compatible_adapter`,
`incomplete_partition_coverage`, or `incompatible_resolution`. POST is read-only and rejects a
stale `draftRevision`; older clients may omit `renderRegime`, which previews the legacy Custom
semantics rather than opting the draft into Turbo. PUT of the draft remains the persistence
boundary. Saving `recommended_turbo` materializes the live compatible recipe, saving `quality`
clears adapters and step overrides, and saving `custom` leaves both fields exactly as authored.

## The three operation shapes

A front end has to treat these differently, because they differ in duration, in failure mode and in
who owns the result.

### 1. Durable planning and synchronous preflight

Planning and compilation use the shared planner library, but the Film workspace wraps them in a
durable operation. The latest operation records its stage, progress, findings, candidate, provider
identity, thinking mode and any underlying job id. Leaving the workspace or restarting the API does
not silently post a second planner request: startup reconciles the recorded operation and job.

- Preflight reads the saved plan, pack, host, catalog and worker list and returns findings naming the
  shot and field. It also exposes the effective model, partition, adapters, steps, reference
  encoding and budget for each selected shot. It creates no render job.
- `compile_existing` (`apps/rust-api/src/film_planner.rs:783`) rebuilds `compiled.json` from an
  edited plan. One `prompt_refine` call per shot unless `--no-refine`, so tens of seconds to minutes.
- `generate` (`apps/rust-api/src/film_planner.rs:667`) is the long one of the three: one full decode
  plus one per repair round plus one rewrite per shot. Local planner jobs default to the
  `DEFAULT_LLM_JOB_TIMEOUT` bound of 1200 s per job (`:57`); the Film workspace exposes that same
  positive-seconds setting and persists its effective value with the durable operation. Minutes on
  the dev Mac.

The workspace starts new drafts with the built-in `prompt_refine_anubis_8b` planner. Qwen3.6-27B is
an optional local planner that must be explicitly installed and selected; it is never downloaded or
selected automatically. A saved OpenAI-compatible connection is a third, explicit provider. Its
disclosure is shown before dispatch, its secret stays in the backend, and reference pixels are sent
only after a separate opt-in. The planner selection is independent of the plan's target video model,
and provider failure never falls back to another provider. See [film-editor.md](film-editor.md) for
the operator flow.

The workspace sends the saved `expectedDraftRevision` with preflight and run creation.
Preflight returns `draftRevision`; creation checks the expected revision again under the project
lock that pins the plan, references, questions, compiled requests and shot selection. A concurrent
save returns HTTP 409 and creates no run or job. Older clients may omit the expected revision;
the route still pins only the exact snapshot it validated, and CLI store callers validate any
supplied compiled document against the snapshot held under that same lock.

### 2. Long-running and resumable: `run`, `resume`

Hours. The CLI `run` (`apps/rust-api/src/film_harness.rs:4823`) can create a project, import
assets, speak dialogue, dispatch one video job per shot attempt, assemble a timeline, and optionally
export. The Film workspace creates its run inside the active project, delivers each completed shot
into the editable saved timeline, and leaves export as a separate operator action.

The server-hosted run is a task the API owns, with the record file as the durable truth:

- the record is rewritten atomically at **every** transition, by temp file, `sync_all` and rename
  (`apps/rust-api/src/film_harness.rs:1684`–`:1703`), and mirrored into the project at
  `film-harness/<run_id>/run.json` (`:1636`). A reader never sees a half-written record, and a UI
  polling it is never more than one transition behind the API;
- every attempt carries an idempotency key written **before** the job is created
  (`apps/rust-api/src/film_harness.rs:1422`, `crates/sceneworks-core/src/film_plan.rs:3420`), so a
  process that dies between the POST and the record write finds its own job on replay;
- `resume` (`:5200`) reconciles a record against the API without re-dispatching anything.

So the task does not need its own state store and must not have one. A UI that renders from
`run.json` and a CLI that renders from `run.json` cannot disagree.

`RunControl` (`apps/rust-api/src/film_harness.rs:282`) is the in-process cancel handle
(`cancel` at `:301`, `is_canceled` at `:305`) that `run_with_control` (`:4837`) takes; the
out-of-process equivalent is `request_cancel` (`:322`), which writes a cancel request into the run
directory, and `clear_cancel_request` (`:338`). A hosted task would use the former for a UI cancel
button and keep the latter working for a shell.

An interrupted replacement or repair carries `activeTakeOperation` in `run.json`: its kind,
target shot, attempt/idempotency identity, original shot bound, export choice and prior run verdict.
The linked attempt owns the job ID. API startup and CLI resume recover that one operation outside
the automatic run budget, adopt its existing job, and preserve the prior whole-run stop. They do
not authorize another take or resume unrelated pending shots.

### 3. Human decisions and bounded edits

Short, and all but one of them touch the API very little.

Draft saving and generated-plan application materialize missing review questions. Shot renames in
the editor carry existing questions; custom questions for detached shots remain editable in the
draft and are excluded from a run whose plan no longer contains those shots. Only the current
plan's questions are pinned. Review validates the requested shots and questions before HTTP 202.
`review-operation.json`, exposed as `reviewOperation` in the review response, retains running,
completed, rejected and failed status with actionable detail. Later transport errors and partial
reviews stopped by a limit remain visible after reload; existing observations remain advisory.


| verb | library entry | API cost |
| --- | --- | --- |
| `review` | `film_harness::review::review` (`review.rs:769`) | frame extraction + one `image_vqa` per question; minutes, bounded by the review plan's own `limits` |
| `accept-take` / `reject-take` | `decide_take` (`review.rs:1604`) | **none; it never reaches the API** |
| `request-repair` | `request_repair` (`review.rs:1757`) | one bounded render |
| `replace-take` | `replace_take` (`film_harness.rs:5260`) | one bounded render |
| `trim` / `reorder` / `swap-take` | `edit_timeline` (`film_harness.rs:7129`), `TimelineEdit` (`:7088`), `EditOptions` (`:7114`) | a timeline PUT; a render only if `--export` |
| export | re-dispatched by `edit_timeline` / `replace_take` with `--export` | one `timeline_export` job |

`accept-take` and `reject-take` needing no API call at all is worth designing around: in a hosted
model they are the cheapest possible writes, and a UI can make them feel instant honestly.

## The run record is the state document

`RunRecord` (`crates/sceneworks-core/src/film_plan.rs:3726`), schema version 2 (`:58`). Written at
`<out>/run.json` and mirrored to `<project>/film-harness/<run_id>/run.json`. A UI reads this and
nothing else.

| field | line | what a UI does with it |
| --- | --- | --- |
| `schemaVersion`, `runId`, `createdAt`, `finishedAt` | `:3727`–`:3731` | identity and age |
| `state` | `:3733` | `running` or `finished`, the field a resume reads first; drives "is anyone holding this?" |
| `outcome` | `:3734` | `completed` / `rejected` / `stopped_run_budget` / `stopped_memory_limit` / `canceled` / `failed` (`:3057`) |
| `stop` | `:3737` | `{ reason, detail, resumable }` (`:3095`). **`resumable` is what enables or disables a Resume button** |
| `plan`, `referencePack`, `compiled` | `:3738`, `:3739`, `:3743` | id, version, path and sha256 of each pinned document |
| `projectId`, `projectPath` | `:3745`, `:3747` | where the assets live |
| `model` | `:3749` | requested tier, fps, lane, observed backend, weights rows per partition, and the render host's platform/memory/GPU (`ModelRecord`, `HardwareRecord`) |
| `limits` | `:3750` | the declared budgets, for showing progress against them |
| `selectedShotIds` | `:3751` | which shots this run is about |
| `references`, `sound` | `:3753`, `:3758` | imported assets, with role, kind, sha256, asset id and approval |
| `synthesizedSound` | `:3762` | lines this run spoke: model, voice, text, `textSha256`, job, asset, pack file |
| `shots` | `:3764` | the body of the record, detailed below |
| `timeline`, `export`, `exportPending`, `supersededExportJobIds` | `:3766`, `:3768`, `:3771`, `:3780` | the sequence and the MP4 |
| `diagnostics` | `:3782` | findings and transport errors |
| `decisions` | `:3785` | the edit log |
| `elapsedSeconds`, `humanRequestedElapsedSeconds` | `:3794`, `:3799` | the two clocks; total cost is their sum |

**Per shot** (`ShotRunRecord`, `crates/sceneworks-core/src/film_plan.rs:3504`): `shotId`, `outcome`
(`rendered` / `failed` / `timed_out` / `canceled` / `not_dispatched` / `not_selected`, `:3186`),
`intended`, `conditioningAssets`, `attempts[]`, `selectedAttempt`, `needsReview[]`, `reviews[]`,
`humanDecision`. Nothing is ever removed: a rejected take stays beside the one that replaced it.

**Per attempt** (`AttemptRecord`, `:3414`): `attempt`, `idempotencyKey`, `resolvedModelId`,
`partitionReason`, `referenceImageShortEdge`, `loras`, `effectiveSteps`, `turboSchedulerShift`,
`jobId`, `status`, timestamps, `elapsedSeconds`, `peakMemoryGb` + `peakMemorySource`, `error`,
`take`, `rejection`, `humanRequested`. `AttemptRecord::has_live_take` (`:3497`) is the "is this still
a candidate" predicate a take strip wants.

**Decisions** (`ProductionDecision`, `:3173`): `at`, `action` (one of `resume`, `cancel`,
`replace_take`, `review`, `accept_take`, `reject_take`, `request_repair`), optional `shotId`,
`detail`. Timeline edits append here as well as to `timeline.edits`.

**Timeline** (`TimelineRecord`, `:3629`): `timelineId`, `name`, `aspectRatio` (what the timeline was
created at) alongside `sourceAspectRatio` / `sourceWidth` / `sourceHeight` (what the takes actually
are), `fps`, `durationSeconds`, `items[]` in cut order, `tracks[]` with each bus's `gain` and
`muted`, `generatedAudioDefault`, and `edits[]` (`:3662`; `TimelineEditRecord` at `:3668`). **Items are read
back off the saved timeline, never off the harness's intent**, so the record cannot claim items the
project does not hold.

**Export** (`ExportRecord`, `:3680`): `jobId`, `status`, `stale`, `assetId`, `renderPath`
(project-relative), `error`, and `droppedAudioLayers[]` (`:3698`), each `{ assetId, trackId, role,
generated, reason }`, so a UI can say *why* a bus is missing from the file rather than leaving the
user to guess.

`RunRecord::is_resumable` (`:3820`) already encodes the rule a Resume button needs.

## What is already a project asset

Everything the harness makes is an ordinary SceneWorks asset in an ordinary project, which is most of
what makes a front end cheap:

| thing | how it got there |
| --- | --- |
| reference images | imported and tagged `film-harness-reference` (or `film-harness-reference-unapproved`), `apps/rust-api/src/film_harness.rs:3206`, `:3306` |
| dialogue clips and beds | imported on the sound buses, `:3133`; a synthesized clip's asset carries `extra.filmHarness.synthesized: true` |
| takes | the video job's own output asset, recorded in `AttemptRecord::take` |
| extracted review frames | persisted as project assets by the `frame_extract` job |
| the export | the `timeline_export` job's render asset |

The timeline routes are used exactly as the editor uses them: `POST /timelines` to create (adopted
afterwards by the `<title> (<runId>)` name it was created under, because the route always creates),
`GET`/`PUT /timelines/:id` to read and re-lay, `POST /timelines/:id/exports` to render. A review's
frame extraction rides a **separate one-item timeline** named `film-harness review (<runId>)`, never
the export timeline (runbook § *The two seams it drives*). A UI must keep that separation, because
reviewing must not rewrite the thing the run is for.

## Historical route proposal

The table below records the original route sketch and is not the current HTTP contract. The shipped
project-scoped routes are registered in `apps/rust-api/src/lib.rs`; use those routes or the API
helpers in `apps/web/src/api/films.js` and `apps/web/src/api/filmReview.js`. The governing rule still
applies: routes call the same library functions the CLI calls.

| route | library call | shape |
| --- | --- | --- |
| `POST /film-harness/validate` | `film_harness::validate` | synchronous, findings |
| `POST /film-harness/plan` | `film_planner::generate` | long, poll the `prompt_refine` job ids it reports |
| `POST /film-harness/compile` | `film_planner::compile_existing` | long-ish |
| `POST /film-harness/runs` | `film_harness::run_with_control` | starts a task, returns the run id |
| `POST /film-harness/runs/:id/resume` | `film_harness::resume` | starts a task |
| `POST /film-harness/runs/:id/cancel` | `RunControl::cancel` / `request_cancel` | instant |
| `GET  /film-harness/runs/:id` | `read_run_record` | the record, verbatim |
| `POST /film-harness/runs/:id/shots/:shot/decision` | `review::decide_take` | instant, no downstream API call |
| `POST /film-harness/runs/:id/shots/:shot/repair` | `review::request_repair` | one bounded render |
| `POST /film-harness/runs/:id/shots/:shot/replace` | `film_harness::replace_take` | one bounded render |
| `POST /film-harness/runs/:id/review` | `review::review` | long, bounded by the review plan |
| `POST /film-harness/runs/:id/timeline/edit` | `film_harness::edit_timeline` | instant, or long with export |

**The rule: CLI and routes never fork the orchestration.** The reason is not tidiness. Correctness
here lives in the record-write ordering, the idempotency keys and the resume reconciliation, and a
second implementation of any of those is a second set of the bugs sc-22711 and sc-22715 already
fixed. A route that "just does the simple case" of a run is how a UI-started run stops being
resumable from a shell. If a route needs behaviour the library does not have, the change goes in the
library.

One consequence worth stating: `read_run_record` (`:378`) is a plain file read, so
`GET /film-harness/runs/:id` costs nothing and can be polled freely.

## Concurrency and safety

**One controller per run directory.** `ControllerLease` holds an advisory lock for every mutating
controller. A second controller is refused before it can rewrite `run.json`. The lease file also
records the owner while the lock is held: clean release clears the marker, while an abrupt process
exit releases the OS lock but leaves the marker for crash-only startup adoption. Idempotency keys
remain the protection for the separate crash window between posting a job and recording its id.

What the record *does* imply:

- **`run` refuses a directory that already holds a record** (runbook § *Durable run state*). A second
  run over the same `--out` would mint a new run id over the previous run's takes while the document
  copies beside it stayed from the old run. So a UI's "new run" is a new directory, always.
- **Multiple runs per film are supported.** The Operations panel lists the draft's runs and exposes
  a run selector. It follows the newest run until the operator explicitly chooses another, then
  keeps that selection stable while polling. Runs remain independently addressable by run id and
  retain their own records, assets and timelines.
- **`replace-take` refuses outright while the shot still has an unsettled attempt** and points at
  `resume` (runbook § *Durable run state*). A UI can surface that as a disabled button rather than a
  failed call.
- **Cancel semantics**: stop dispatching, cancel the in-flight job through the existing cancel route,
  leave every finished take in the project, skip the timeline and export, record `outcome: canceled`
  with a **resumable** stop. Attempts already spent are not re-spent: a cancel is not a retry. A
  cancel the worker does not honour within the grace (30 s, capped at `maxShotSeconds`) stops the run
  rather than dispatching a second render beside one still on the GPU (runbook § *Validation before
  dispatch*; the constant is `CANCEL_GRACE`, `apps/rust-api/src/film_harness.rs:76`, and the cap is
  applied at `:659`–`:661`).
- **Resume after a server restart is crash-only.** Startup adopts only a `running` record whose
  unlocked controller lease still names the process that died. A cleanly released controller is not
  restarted automatically, even when the record remains resumable; the Operations panel leaves an
  explicit Resume action for that case. A recorded job id is read back by its exact job route, even
  if it no longer appears in the queue listing. The idempotency-key listing is only needed when the
  process died after posting a job but before recording its id.

## Render controls the front end exposes

The Shots view exposes the plan-level model, tier, resolution, reference short edge, adapters and
steps, plus each shot's conditioning, intent, dialogue placement and dependencies. Capability menus
come from the selected model and partition; preflight resolves the effective values per shot before
rendering. The production plan and compiled request remain the shared contract, so the UI does not
invent a second job-body shape.

## Editor concepts that map naturally

| harness | editor |
| --- | --- |
| reference pack | the cast and props: approved images with roles, plus the voices and beds |
| plan | the shot list, with beats, framings, durations and intended start/end state |
| compiled plan | what will actually be dispatched, per shot: a preflight sheet |
| run | the shoot; each attempt is a take, and nothing is ever deleted |
| `selectedAttempt` | the circled take |
| review | the script supervisor's notes, advisory only |
| `needsReview` flags | continuity notes raised when an upstream take changed |
| decisions | the edit log: who accepted what, when, and why |
| timeline + export | the assembly and the cut |

The mapping is close enough that most of the UI work is presentation, not new concepts. The one place
it is *not* a clean mapping is review: a script supervisor's note is a person's judgment, and this
one is a local vision model that both misses real faults and flags correct takes. The UI must not let
a flag read as a verdict.

## Current workspace behavior

The Film workspace stores its original script, structured brief, optional reference pack, editable
production plan and compiled plan in a project-scoped draft. Replacing the plan is an explicit
operator action; the source text remains preserved. Planning, runs, review and export expose durable
status in the Operations panel. Per-step progress comes from the job named by the operation or run,
while the project record remains the recovery authority.

Rendering delivers completed shots into an editable saved timeline and does not export. A timeline
edit marks any older export stale. **Export current cut** is the explicit boundary that saves and
renders the current timeline, reports failures and lists any audio layers that could not be mixed.
