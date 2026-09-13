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

Each conditioning mode fixes which slots it takes — `text_to_video` none, `image_to_video` a first
frame, `first_last_frame` first and last, `reference_to_video` references only — which is how a
plan stays honest about MiniMax-H3's rule that keyframes and references are different tasks on
different checkpoints.

## Validation before dispatch

`film-harness` creates nothing until every check passes; findings name the shot and the field:

1. plan and pack structure (schema version, ids, limits, per-shot fields, slot shape per mode);
2. cross-references (every role exists in the pack and is approved) and files on disk;
3. the model's catalog entry from `GET /api/v1/models`: capability per mode, target duration on the
   declared menu and inside the hard bounds, fps and resolution on the declared menus / under
   `maxPixels`, reference counts against `limits.maxReferenceAssets`, negative prompts against
   `video.supportsNegativePrompt`, and the plan's `limits.maxMemoryGb` against the lane's
   `minMemoryGb`; install state of the requested tier unless `--skip-install-check`;
4. a registered worker advertising `video_generate` (and `timeline_export` unless `--no-export`),
   and the host memory it reports against the plan's memory budget.

Limits are declared in the plan (`limits.maxRunSeconds`, `maxShotSeconds`, `maxAttemptsPerShot`,
`maxMemoryGb`). A shot budget cancels the in-flight job through the API and counts the attempt; the
attempt cap bounds retries; the run budget or an observed memory peak over budget stops new
dispatch and marks the remaining shots `not_dispatched`. The run record is written on every path,
including refusal (`outcome: rejected` with the findings).

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

`scripts/film-harness-smoke.sh` builds this checkout, starts the API and the native GPU worker
against a scratch data dir, waits for both to register, renders `SH010,SH020` of the fixture on
MiniMax-H3 q4 (MLX), and tears both down. The fixture's placeholder plates are deterministic
(`fixture-images`), so the checked-in PNGs are reproducible byte for byte.
