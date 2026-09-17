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

### 1. Synchronous document transforms: `plan`, `validate`, `compile`

Input documents in, documents or findings out. Nothing is created in the project.

- `validate` (`apps/rust-api/src/film_harness.rs:1709`) reads the plan, the pack, the host, the
  catalog and the worker list and returns findings naming the shot and the field. Seconds.
- `compile_existing` (`apps/rust-api/src/film_planner.rs:783`) rebuilds `compiled.json` from an
  edited plan. One `prompt_refine` call per shot unless `--no-refine`, so tens of seconds to minutes.
- `generate` (`apps/rust-api/src/film_planner.rs:667`) is the long one of the three: one full decode
  plus one per repair round plus one rewrite per shot, bounded by `DEFAULT_LLM_JOB_TIMEOUT` of 1200 s
  per job (`:57`). Minutes on the dev Mac.

For a UI these are request/response, but `generate` is long enough that it wants progress. Its cost
is already persisted into `compiled.json`'s `planner` block: every `prompt_refine` job id, their
summed wall clock, the highest peak memory their metrics reported, the rounds taken and the budget
they ran under (runbook § *Planning from a brief*). So a UI can poll those job ids through the
ordinary jobs routes rather than needing a new progress channel.

### 2. Long-running and resumable: `run`, `resume`

Hours. The CLI `run` (`apps/rust-api/src/film_harness.rs:4823`) can create a project, import
assets, speak dialogue, dispatch one video job per shot attempt, assemble a timeline, and optionally
export. The Film workspace creates its run inside the active project, delivers each completed shot
into the editable saved timeline, and leaves export as a separate operator action.

**Recommendation: a server-hosted run should be a task the API owns, with the record file as the
only truth.** Three properties of the current design push that way and none pushes against it:

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

### 3. Human decisions and bounded edits

Short, and all but one of them touch the API very little.

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

**One controller per run directory.** Nothing locks `run.json`; whichever controller holds the run
rewrites it, so two live controllers against one directory interleave their writes. The idempotency
keys stop a *sequential* replay from duplicating work; they are not a lock
(`apps/rust-api/src/bin/film-harness.rs:54`, runbook § *Durable run state*). A hosted implementation
therefore needs a real single-holder guarantee per run directory; the record cannot provide one.

What the record *does* imply:

- **`run` refuses a directory that already holds a record** (runbook § *Durable run state*). A second
  run over the same `--out` would mint a new run id over the previous run's takes while the document
  copies beside it stayed from the old run. So a UI's "new run" is a new directory, always.
- **One run per project is not enforced anywhere.** A project is adopted by the
  `<title> (<runId>)` name, and the run id is in the name, so two runs in one project are
  addressable and do not collide on assets or timelines. Nothing checks for it either way. The epic
  should decide whether it wants that and, if not, enforce it in the route layer rather than
  assuming the record does.
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
- **Resume after a server restart** is the crash path, and it already works: the record is left
  `running` with the job named in it, and `resume` adopts that job at whatever state it reached. On
  the sc-22715 evaluation the resumed controller found the render 32 % through and simply polled it
  to completion, enqueuing nothing. A hosted API should run that reconciliation at startup for every
  `running` record it owns, rather than waiting for a user to press Resume.
- **What adoption cannot see: a cleared job.** Every lookup goes through `GET /api/v1/jobs`, which
  filters `cleared_at is null`, so a job the operator cleared out of the queue is invisible and a
  replay will enqueue that attempt again (runbook § *Durable run state*). A UI that offers a "clear
  queue" action near a live run should say so.

## Two controls this front end should expose

**`advanced.referenceImageShortEdge`.** It appears nowhere in `apps/web/src` (grep returns nothing),
and it is the single biggest measured cost lever on the reference partition: 30–32 s/step at 1024
against 137–140 at 2048, with no rubric loss on the two shots measured
(`docs/film-harness-evaluation-phase-2.md`, §4.4). A film front end that cannot set it is asking its
user to accept a 4.4x bill silently.

**`model.loras` and `advanced.steps`, as plan-level declarations.** The Studio already has a LoRA
selector (`apps/web/src/components/generationStudio.jsx:508`–`:522`); the `{ id, weight }` shape is
what its **preset-save** payload carries, not a generation job body
(`apps/web/src/components/generationStudio.jsx:797` feeding `buildStudioPresetPayload`,
`apps/web/src/presetUtils.js:777`–`:786`). The
editor rail has a steps override (`apps/web/src/components/editor/GenerationRail.jsx:292`), so these
are not missing from the product. What is missing is the *shape a film needs*: declared **once on the
family** and resolved **per shot** against the partition that shot resolved to
(`crates/sceneworks-core/src/film_compile.rs:351`), with the one-recipe-per-partition rule enforced
before a weight is read. Per-generation controls cannot express that, and hand-editing the plan JSON
is the only way to get it today. This is the 14 h versus 1 h 19 m difference on the six-shot film
(`docs/film-harness-evaluation-phase-2.md`, §4.6).

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

## Open questions for the epic

1. **Who owns a run task?** If the API hosts it, what happens to a run when the API restarts mid-shot:
   auto-reconcile every `running` record it owns, or wait for a person? What happens to a run
   started from a shell while the API is also hosting one against the same directory? The record
   cannot arbitrate; something has to.
2. **Multi-user.** Nothing in the record has an author. `ProductionDecision` records `at`, `action`,
   `shotId` and `detail` (`crates/sceneworks-core/src/film_plan.rs:3173`) and no identity. If more
   than one person can accept takes on one run, the decision log needs a field and the schema needs a
   version bump.
3. **Where is the brief edited?** The planner's `ProductionBrief` is the only document with no
   editing surface in this note. It is also the document whose defaults determine whether a user
   lands on the turbo path, and in phase 2 the shipped brief produced no valid plan in 2 or 5 repair
   rounds (`docs/film-harness-evaluation-phase-2.md`, §5.2). The brief editor is a first-class
   surface, not a text box.
4. **Streaming progress.** Today a UI would poll `run.json` plus the job routes. The record is at
   most one transition behind and `read_run_record` is a file read, so polling is honest and cheap,
   but per-step render progress lives on the job, not the record, and a 137 s/step shot needs
   something between transitions. Decide whether that is job polling, an event stream, or nothing.
5. **Where the documents live.** The CLI takes filesystem paths. A UI has a project. Whether plans
   and packs become project-scoped documents, or stay files the API reads, changes what every route
   above takes as its body.
