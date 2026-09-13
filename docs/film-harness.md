# Local filmmaking harness (`film-harness`)

Epic 22708 / sc-22710. Renders a hand-authored production plan into a SceneWorks sequence through
the existing API seams — projects, asset import, `POST /api/v1/video/jobs`, job polling, timelines
and the `timeline_export` job — and leaves a versioned run record behind. No new UI, no parallel
renderer: every take is produced by whatever GPU worker claims the job.

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

The run record is written on **every** path past document validation: refusal
(`outcome: rejected` with the findings), a limit that stopped the run, an operator Ctrl-C
(`outcome: failed` with a "canceled by operator" diagnostic), and a transport/API failure partway
through (`outcome: failed`, the error in `diagnostics`) — a run that created a project and
dispatched jobs never ends with no record of it.

## Running

```text
cargo build -p sceneworks-rust-api --bin film-harness
target/debug/film-harness validate --plan PLAN --references REFS [--api URL] [--shots A,B]
target/debug/film-harness run      --plan PLAN --references REFS [--api URL] [--shots A,B]
                                   [--project-id ID] [--out DIR] [--poll-seconds N]
                                   [--no-export] [--skip-install-check]
target/debug/film-harness fixture-images --out DIR
```

`run` needs a SceneWorks API (default `http://127.0.0.1:8000`, or `$SCENEWORKS_API_URL`; token from
`$SCENEWORKS_ACCESS_TOKEN`) with a registered GPU worker and a utility worker for the ffmpeg export
(`SCENEWORKS_RUN_UTILITY_INPROCESS=1` on the API). Exit codes: 0 completed, 2 refused before
dispatch, 3 stopped on a limit or a shot failed, 1 transport/API/io error.

Ctrl-C during a `run` cancels the in-flight job through the API, stops dispatching and writes the
record — it does not leave a render on the GPU with nothing to say it happened.

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
