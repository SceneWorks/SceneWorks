# Local filmmaking harness (`film-harness`)

Epic 22708 / sc-22710, sc-22711. Renders a hand-authored production plan into a SceneWorks sequence
through the existing API seams — projects, asset import, `POST /api/v1/video/jobs`, job polling,
timelines and the `timeline_export` job — and leaves a versioned run record behind. No new UI, no
parallel renderer: every take is produced by whatever GPU worker claims the job.

## Documents

| Document | Schema | Fixture |
| --- | --- | --- |
| Production plan | `sceneworks_core::film_plan::ProductionPlan` | `config/film-harness/courier-workshop/plan.jsonc` |
| Reference pack | `sceneworks_core::film_plan::ReferencePack` | `config/film-harness/courier-workshop/references.jsonc` |
| Run record | `sceneworks_core::film_plan::RunRecord` | written to `--out/run.json` and `<project>/film-harness/<run_id>/run.json` |

A plan carries stable shot ids, narrative beat, framing, prompt, target duration, intended start/end
state, dialogue/sound intent and the conditioning each shot wants, expressed as **reference roles**
(`firstFrameRole`, `lastFrameRole`, `referenceRoles`). The reference pack maps roles to approved
image files; it is a separate versioned document, so approved references stay addressable
independently of any generated take. Both files tolerate JSONC comments and refuse unknown fields.

A pack entry with `"approved": false` is still imported (so a human can review it), but it is tagged
`film-harness-reference-unapproved` instead of `film-harness-reference`, recorded with
`approved: false`, and never resolved into a shot's conditioning slots. A reference `file` must have
a `[A-Za-z0-9._-]` basename: it is sent as a multipart filename, so a CR/LF in it would inject
headers.

Each conditioning mode fixes which slots it takes — `text_to_video` none, `image_to_video` a first
frame, `first_last_frame` first and last, `reference_to_video` references only — which is how a
plan stays honest about MiniMax-H3's rule that keyframes and references are different tasks on
different checkpoints.

### Dependencies (plan schema 2)

A shot may declare `dependsOn: [{ shotId, kind, note }]`, with `kind` either:

| kind | meaning | what a changed take upstream means |
| --- | --- | --- |
| `conditioning` | this shot's conditioning came out of that shot's **selected take** (last-frame chaining, a plate cut from a take) | the input this shot was built on is gone |
| `continuity` | this shot's `startState` is that shot's `endState` | the take is still valid input-wise, but the story state it continues from may not match |

Edges must name another shot in the same plan and may not form a cycle. They are declarations, not
wiring: nothing here reaches the model. Their one job is to tell the harness who to **flag** when a
selected take changes — see *Replacing a take* below. A schema 1 plan reads unchanged and declares
no edges.

## Durable run state, resume and take replacement

`run.json` is the run's state, not a report: it is rewritten (atomically) at every transition —
before a job is created, once its id is known, once a take is adopted, after the timeline, after the
export. `state` is `running` while a controller holds it, so a record left by a killed controller
says so.

Every attempt carries an **idempotency key** (`<runId>:<shotId>:a<attempt>`), written into the
record *before* the job is created and stamped into the job payload
(`advanced.filmHarness.idempotencyKey`). That closes the one window a record alone cannot: a
controller that died between creating the job and recording its id finds its own job by the key
instead of enqueuing a second one. References are adopted the same way, by the `filmHarness`
provenance the import stamps; the project by the `<title> (<runId>)` name it was created under; the
export job by the timeline it renders.

`film-harness resume --out DIR` picks a run up:

- every recorded take is reused; no shot with a selected take is re-dispatched;
- every job the record names is read back from the API and adopted at whatever state it reached —
  completed jobs become takes, failed ones count as spent attempts, running ones are polled again
  under what is left of their budget;
- the plan and reference pack are re-read and must still hash to what the run recorded. **An edited
  plan is refused**: that is a new run, not a resume;
- the wall-clock budget is cumulative across controllers (`elapsedSeconds` accumulates) and the
  per-shot attempt cap counts every automatic attempt ever made, so restarting cannot turn a bounded
  run into an unbounded one.

Only a resumable stop can be resumed. A cancel or a crash is resumable; an exhausted wall-clock
budget, an over-budget memory peak or an exhausted attempt cap is **terminal**, and `stop.detail`
says which plan value to change. `film-harness status --out DIR` prints all of this without touching
the API.

**One controller per run directory.** The record is rewritten by whichever controller holds it, and
nothing locks it: running `run` and `resume` against the same `--out` at the same time interleaves
their writes. The idempotency keys stop a *sequential* replay from duplicating work; they are not a
substitute for a lock between two live controllers. `replace-take` refuses outright while the shot
still has an unsettled attempt, and points at `resume`. Cancel the run, or let it finish, before
starting another command against its directory.

### Cancellation

`film-harness cancel --out DIR` (or Ctrl-C in the running shell) stops new dispatch, cancels the
in-flight job through the existing cancel route, leaves every finished take in the project, skips
the timeline and export, and records `outcome: canceled` with a resumable stop. Attempts already
spent are not re-spent: a cancel is not a retry.

### Replacing a take

A shot keeps **all** its takes with their provenance; at most one is `selectedAttempt`. The first
usable take of a shot is selected, and nothing but an explicit replacement ever moves it — replay
and resume never do.

```text
film-harness replace-take --out DIR --shot SH030 --reason "the parcel is the wrong red" [--export]
```

marks the take the shot is carrying `rejection: { at, reason }` (it stays in the record, and its
asset stays in the project), and dispatches **exactly one** new attempt for that shot. The
replacement is `humanRequested`, so it neither spends nor respects the plan's automatic attempt cap;
it does not loop, and a failed replacement stops with `replacement_failed` rather than trying again.
Every other shot's takes, jobs and assets are untouched.

When the replacement lands, shots that **declared a dependency** on it are flagged `needsReview`
with the reason and the dependency kind, and the timeline item for that shot is rewritten in place
(a PUT — no job, and no other item moves). Flagged shots are never re-rendered: that is a decision
for the person who read the flag. Without `--export` the existing export is marked `stale`; with it,
one re-export runs.

## Validation before dispatch

`film-harness` creates nothing until every check passes; findings name the shot and the field:

1. plan and pack structure (schema version, ids, limits, per-shot fields, slot shape per mode);
2. cross-references (every role exists in the pack and is approved) and files on disk;
3. the host: `GET /api/v1/host-capabilities` reports the **API host's** platform and memory. That
   platform — not the machine `film-harness` runs on, which `--api` may make a different one —
   decides the lane (`mlx` on macOS, `candle` elsewhere) whose `minMemoryGb` the plan is checked
   against, and it is what the run record's `model.hardware.platform` states;
4. the model's catalog entry from `GET /api/v1/models`: capability per mode, target duration on the
   declared menu and inside the hard bounds, fps and resolution on the declared menus / under
   `maxPixels`, reference counts against `limits.maxReferenceAssets`, negative prompts against
   `video.supportsNegativePrompt`, the plan's `limits.maxMemoryGb` against the lane's
   `minMemoryGb`, and the route's own gates — platform reachability
   (`ensure_video_model_available_on_platform`) and the reference-payload check
   (`validate_video_reference_asset_ids_payload`) — run here rather than discovered as a 400 at
   enqueue; install state of the requested tier unless `--skip-install-check`;
5. a registered worker advertising `video_generate` (and `timeline_export` unless `--no-export`),
   and the host memory it reports against the plan's memory budget.

Limits are declared in the plan (`limits.maxRunSeconds`, `maxShotSeconds`, `maxAttemptsPerShot`,
`maxMemoryGb`). A shot budget cancels the in-flight job through the API and counts the attempt; the
attempt cap bounds retries; the run budget or an observed memory peak over budget stops new
dispatch and marks the remaining shots `not_dispatched`. A cancel the worker does not honour within
the grace (30s, capped at `maxShotSeconds`) stops the run rather than dispatching a second render
beside one that is still on the GPU.

The observed peak is read off the job's metrics block (`GET /api/v1/jobs/:id/metrics`
`peakMemoryBytes`, which the worker POSTs after its terminal progress), compared in GiB against
`limits.maxMemoryGb`; `peakMemoryPct × hostMemoryGb` and finally the job snapshot's
`peakGpuMemoryPct` are fallbacks, and each attempt records which one it used
(`peakMemorySource`).

The run record is written at **every state transition** past document validation, not once at the
end: refusal (`outcome: rejected` with the findings), a limit that stopped the run, an operator
cancel (`outcome: canceled` with a resumable `stop`), and a transport/API failure partway through
(the error in `diagnostics`, the record left `running` so a resume can reconcile it) — a run that
created a project and dispatched jobs never ends with no record of it, and never ends with a record
a resume cannot pick up.

## Running

```text
cargo build -p sceneworks-rust-api --bin film-harness
target/debug/film-harness validate     --plan PLAN --references REFS [--api URL] [--shots A,B]
target/debug/film-harness run          --plan PLAN --references REFS [--api URL] [--shots A,B]
                                       [--project-id ID] [--out DIR] [--poll-seconds N]
                                       [--no-export] [--skip-install-check]
target/debug/film-harness resume       --out DIR [--api URL] [--poll-seconds N] [--no-export]
target/debug/film-harness replace-take --out DIR --shot SHxxx [--reason TEXT] [--export]
target/debug/film-harness cancel       --out DIR
target/debug/film-harness status       --out DIR
target/debug/film-harness fixture-images --out DIR
```

`run` needs a SceneWorks API (default `http://127.0.0.1:8000`, or `$SCENEWORKS_API_URL`; token from
`$SCENEWORKS_ACCESS_TOKEN`) with a registered GPU worker and a utility worker for the ffmpeg export
(`SCENEWORKS_RUN_UTILITY_INPROCESS=1` on the API). Exit codes: 0 completed, 2 refused before
dispatch or the action does not apply to the record, 3 stopped on a limit / a cancel / a failed shot
(and from `status`, a run that can still be resumed), 1 transport/API/io error.

Ctrl-C during a `run` (or `film-harness cancel --out DIR` from another shell) cancels the in-flight
job through the API, stops dispatching and writes the record — it does not leave a render on the GPU
with nothing to say it happened. A second Ctrl-C exits immediately without a record. The cancelled
run is **resumable**: `film-harness resume --out DIR` picks it back up.

The timeline is created at the nearest aspect ratio the timeline route admits (`16:9` / `9:16` /
`1:1`). The fixture's 576x320 takes are 9:5, so the MP4 export is a **letterboxed** 16:9 render of
them; the record states both — `aspectRatio` is what the timeline was created at, and
`sourceAspectRatio` / `sourceWidth` / `sourceHeight` are what the takes actually are. Timeline items
in the record are read back off the saved timeline, not off the harness's intent.

`scripts/film-harness-smoke.sh` builds this checkout, starts the API and the native GPU worker
against a scratch data dir, waits for both to register, renders `SH010,SH020` of the fixture on
MiniMax-H3 q4 (MLX), and tears both down. The fixture's placeholder plates are deterministic
(`fixture-images`), so the checked-in PNGs are reproducible byte for byte.

It builds the **release** profile, so export the release prebuilt libmlx before running it on
macOS — the fetch script defaults to Debug, and a Debug directory is the wrong key for a release
build (it fails rather than falling back):

```sh
eval "$(scripts/fetch-prebuilt-mlx.sh --build-type Release)"
export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
scripts/film-harness-smoke.sh
```

The script sets `SCENEWORKS_GPU_ID` for the render worker (`mlx` on macOS): the worker binary
defaults that to `cpu`, and a cpu worker spawns the utility pool and advertises no
`video_generate`, so the harness would refuse for want of a GPU worker that is in fact running.
Override it (`SCENEWORKS_GPU_ID=0`) to smoke a CUDA host.
