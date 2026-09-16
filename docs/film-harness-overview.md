# What the film harness does

For someone who has never run it. The operating manual is
[film-harness.md](film-harness.md); this page is the shape of the thing, why each piece exists, and
what it cost when it was measured. The seams a front end would call are in
[film-harness-ui-integration.md](film-harness-ui-integration.md).

Citation convention below: code is cited as `path:line` against this commit; the runbook is cited by
section, because its line numbers move and its sections do not.

## The problem it solves

SceneWorks can render one clip at a time from the Video Studio. A film is not one clip. It is a shot
list, a cast of approved reference images, a set of spoken lines, a sequence, and a person deciding
which takes are good enough, over hours of GPU time on a machine that may be interrupted.

The harness is the bounded, resumable, human-controlled way to get from a shot list plus a reference
pack to a reviewed multi-shot film through the workers SceneWorks already has, and to measure what it
cost. It adds no renderer, no UI and no second LLM stack: "No new UI, no parallel renderer and no
second LLM stack: every take is produced by whatever GPU worker claims the job, and every token by
the `prompt_refine` worker that already ships" (`docs/film-harness.md`, opening paragraph).

Four properties are the reason it exists at all:

| property | where it lives |
| --- | --- |
| **Bounded** | every run declares `limits.maxRunSeconds`, `maxShotSeconds`, `maxAttemptsPerShot`, `maxMemoryGb` before dispatch (`crates/sceneworks-core/src/film_plan.rs`, `PlanLimits`; runbook § *Validation before dispatch*), and nothing is created until they and the plan validate |
| **Resumable** | `run.json` is rewritten atomically at every state transition, so a killed controller leaves a record `resume` can reconcile (`crates/sceneworks-core/src/film_plan.rs:3726`, `apps/rust-api/src/film_harness.rs:1684`) |
| **Human-controlled** | only `accept-take`, `reject-take`, `request-repair`, `replace-take` and the edit verbs change anything (`apps/rust-api/src/bin/film-harness.rs:38`) |
| **Measurable** | every attempt records the checkpoint, the adapters, the step count, the reference short edge and where its memory number came from (`crates/sceneworks-core/src/film_plan.rs:3414`) |

## The three documents

A run is driven by three versioned documents. They are separate on purpose: what the film is, what it
may be conditioned on, and how to interrogate the result are three different authorities with three
different lifetimes. All three tolerate JSONC comments and refuse unknown fields (runbook
§ *Documents*).

### 1. The plan: what the film is

`ProductionPlan` (`crates/sceneworks-core/src/film_plan.rs:175`), schema version 2
(`:43`, with `:45` still accepting version 1). It declares the model once, then a list of shots with
stable ids.

```jsonc
// config/film-harness/courier-workshop/plan.v2.jsonc
"schemaVersion": 2,
"id": "courier-workshop-v2",
"model": { "id": "minimax_h3", "tier": "q4", "fps": 24, "resolution": "576x320" },
"limits": { "maxRunSeconds": 64800, "maxShotSeconds": 10800,
            "maxAttemptsPerShot": 1, "maxMemoryGb": 96 },
"shots": [
  {
    "id": "SH010",
    "beatId": "arrival",                       // the brief beat this shot covers
    "beat": "Establish the empty workshop; the door opens.",
    "framing": "wide static, eye level, door camera-left, workbench centre",
    "prompt": "A quiet, cluttered woodworking workshop in warm late-afternoon light...",
    "targetDurationSeconds": 5.1667,
    "startState": "Empty workshop, door closed, workbench clear.",
    "endState":   "Door open, courier standing in the doorway holding the red parcel...",
    "conditioning": {
      "mode": "reference_to_video",
      "referenceRoles": ["courier", "red_parcel", "workshop_location", "workbench_table"]
    },
    "seed": 23405,
    "continuityRoles": ["workshop_location", "workbench_table", "courier",
                        "red_parcel", "house_style"]
  }
]
```

Two things about that shot are load-bearing. It names **roles**, never files, so the same plan runs
against any pack that declares them (runbook § *The reference-conditioned courier plan*). And
`continuityRoles` is the continuity claim: every shot must bind at least one approved role from the
pack, and a shot that binds none is refused (runbook § *Editing the assembled sequence*, the
"Continuity is canonical" paragraph).

`conditioning.referenceRoles` order is the order the compiled request's `referenceAssetIds` keeps, so
the shipped plan writes it subject-first (`config/film-harness/courier-workshop/plan.v2.jsonc`,
header comment).

### 2. The reference pack: what it may be conditioned on

`ReferencePack` (`crates/sceneworks-core/src/film_plan.rs:484`), schema version 1 (`:47`). A separate
document with its own version, so approved references stay addressable independently of any generated
take.

```jsonc
// config/film-harness/courier-workshop/references.jsonc
"references": [
  { "role": "courier",           "kind": "character", "file": "references/courier.png",
    "description": "The courier: blue jacket, carries the parcel." },
  { "role": "red_parcel",        "kind": "prop",      "file": "references/red_parcel.png" },
  { "role": "workshop_location", "kind": "location",  "file": "references/workshop_location.png" },
  { "role": "house_style",       "kind": "style",     "file": "references/house_style.png" },
  { "role": "workshop_plate",    "kind": "plate",     "file": "references/workshop_plate.png" }
],
"sound": [
  { "role": "courier_line", "kind": "dialogue", "voice": "am_michael",
    "text": "Delivery. I'll leave it on the bench." },              // spoken by the run
  { "role": "workshop_room_tone", "kind": "ambience", "file": "sound/workshop_room_tone.wav" }
]
```

`referenceRoles` may bind only the **subject** kinds (`character`, `prop`, `location`) because
Ref2VA treats every bound image as a subject to depict; a `style` or a `plate` bound there is refused
naming the shot, the role and the kind (runbook § *Documents*). An entry with `"approved": false` is
still imported so a human can look at it, but it is tagged `film-harness-reference-unapproved` and
never resolved into conditioning.

References are **user-provided input** in the product. `film-harness make-references` exists only to
build this repository's own courier fixtures on this machine, and its section opens by saying so
(runbook § *Generating the reference plates*).

### 3. The review plan: how to interrogate a take

`ReviewPlan` (`crates/sceneworks-core/src/film_review.rs:38`). Per shot, closed questions with the
allowed answers named in the question itself.

```jsonc
// config/film-harness/courier-workshop/review.jsonc
{
  "id": "sh010_parcel_custody", "topic": "parcel_custody",
  "frames": "last", "mustObserve": true,
  "intended": "The parcel is in the courier's hands, not on the workbench.",
  "ask": "Where is the parcel in this frame? Answer with exactly one word: hands, surface, nobody, or unclear.",
  "expect": ["hands"],
  "contradict": ["surface", "workbench", "bench", "table", "floor", "ground", "nobody"]
}
```

Closed and presupposition-free is a finding, not a style choice: on real weights the model answered
closed short questions stably and open ones inconsistently. An open "is there a parcel, what
colour?" gave a two-paragraph essay on one frame of a take and "no small parcel or box" on the next
(runbook § *The review plan*).

## One family, two checkpoints, resolved per shot

MiniMax-H3 ships its reference conditioning as a **separate catalog entry**: `minimax_h3` serves
`text_to_video | image_to_video | first_last_frame` with `limits.maxReferenceAssets: 0`, while
`minimax_h3_ref` serves `reference_to_video` only (runbook § *Partition resolution*). A plan declares
the **family once** and the compiler resolves the partition per shot
(`crates/sceneworks-core/src/film_compile.rs:295`):

| the shot's `conditioning.referenceRoles` | it compiles to |
| --- | --- |
| non-empty | `minimax_h3_ref`, with `referenceAssetIds` in the plan's role order |
| empty | `minimax_h3`, its declared mode, and no reference field at all |

**What is optional is image conditioning, not the pack.** A reference pack document is always
required: `--references` is a required argument (`apps/rust-api/src/bin/film-harness.rs:467`), and
every shot must bind at least one **approved** pack role, through `continuityRoles` or through a
conditioning slot, or it is refused (`crates/sceneworks-core/src/film_plan.rs:1741`–`:1758`; the
anchor rule stated above, runbook § *Editing the assembled sequence*). What is optional is
`conditioning.referenceRoles`, and it is optional **per shot**: a shot that names none compiles to
the base `minimax_h3` with its declared mode and no reference field at all. A shot that declares
`reference_to_video` and binds nothing is the contradiction, and is refused by name (runbook
§ *Partition resolution*). The compiled request's
`model`, the dispatched body's `model` and the attempt record's `resolvedModelId` are one string read
three times, so they cannot disagree about which checkpoint produced a take
(`crates/sceneworks-core/src/film_plan.rs:3426`).

The **reference short edge** is a plan-level knob:
`model.advanced.referenceImageShortEdge` sets the pixel short edge an image reference is *encoded*
at, admitted over 1024..=2048 and defaulting to the engine's 2048. It sizes the reference, never the
render, and a value outside the range is refused naming the field and the range rather than clamped,
because a silent clamp would change the token budget the author measured (runbook § *Partition
resolution*). It reaches only shots that resolve to the reference partition
(`crates/sceneworks-core/src/film_compile.rs:342`).

## Dialogue is spoken through the audio route

A `dialogue` pack entry may carry `text` instead of `file`. Before any render, `ensure_sound`
(`apps/rust-api/src/film_harness.rs:2538`) speaks each **placed** line through the ordinary
`POST /api/v1/audio/jobs`, writes the WAV into the pack directory and imports it on the dialogue bus
exactly as it imports a pre-recorded clip (runbook § *Spoken dialogue*). Models are `kokoro_82m`
(default), `chatterbox_tts`, `moss_tts_realtime`, `moss_ttsd_v05`. Only what the run places is
spoken, so a `--shots` selection never pays for a line it left out. No live `audio_generate` worker
is a resumable stop *before the job exists*, scoped to the lines still owed.

Re-casting a line is a **new run, not a resume**: `resume` and `replace-take` hash the pack's bytes
and refuse a document that no longer matches what the run started from.

## The planner: brief to plan

`film-harness plan` turns a brief plus the pack into a plan in the **same schema a hand-authored plan
uses** (runbook § *Planning from a brief*). The loop:

1. **Refuse before the first token.** The hosted-credential and remote-API ban
   (`apps/rust-api/src/film_planner.rs:65`, `:551`), the brief's structure, the pack and its files,
   the model's catalog entry, a beat requiring a role the pack does not approve, and the brief's own
   model block against the installed menus. The brief must declare `limits.plannerMaxMemoryGb`,
   checked against the host's reported memory before the first token.
2. **One draft** through `POST /api/v1/prompts/refine` with `task: "film_plan"`
   (`apps/rust-api/src/film_planner.rs:60`), decoded under a valid-JSON constraint. Object shape is
   *not* enforced by the decoder; the plan schema is enforced after the decode by
   `parse_planner_output`.
3. **Validate**, then **bounded repair rounds**, default 2, ceiling 5
   (`apps/rust-api/src/film_planner.rs:49`, `:53`). Each round hands the validator's findings back
   verbatim and asks for the whole plan again. No round drops a beat, shortens the film or rounds a
   duration to make a finding go away. On exhaustion the refused answer is written to
   `planner-rejected.txt`.

**The capability envelope** is what the planner is held to. Whether it may write reference shots is
decided from exactly two facts, and install state is not one of them: the catalog must serve the
family's reference partition, and the pack must approve at least one reference. With both, the
default mode inverts to `reference_to_video` for every shot showing an approved subject; with either
missing the plan stays on the base checkpoint. The same brief and pack therefore produce the same
film on two machines (runbook § *Partition resolution*, "Planning a reference film").

**Turbo is the default; `preferQuality` is the opt-out.** The envelope lists the step-distill
accelerators this host has **installed** (at most one per partition, paired by the catalog's
`modelIds`), and the output contract shows the exact `loras` array to copy. Install state *is* a
filter here, unlike the reference partition's, because an adapter whose weights are not on the render
host's disk is a 400 at enqueue. A brief setting `"preferQuality": true` keeps the full step path:
nothing is offered, and any selection a draft writes anyway is stripped rather than argued with
(`crates/sceneworks-core/src/film_planner.rs:80`, `:938`).

**The role-array copying rule, and why it exists.** Told in prose that a beat "MUST show courier,
red_parcel, workbench_table", the real local planner (Anubis-Mini-8B) twice wrote one character, one
prop and one place (the location standing in for the second prop) in both role lists, and never
repaired it inside the round budget, *while reproducing the literal `loras` array byte for byte in
every run*. So every place the planner reads a required-role list now states it as the JSON array to
write and where, and a `continuityRoles` finding hands back the corrected array built against the
pack, filtered to bindable subject kinds wherever the message names `referenceRoles`. Validation did
not change; what changed is that a repair became something a copy-only model can do (runbook
§ *Planning from a brief*).

## Compile: the model-specific half

`film-harness compile` writes `compiled.json`: one request per shot with mode, prompt, duration,
fps, geometry, seed and reference bindings exactly as they will be dispatched.
`COMPILED_PLAN_SCHEMA_VERSION` is 3 (`crates/sceneworks-core/src/film_compile.rs:42`); a document at
any earlier schema version — v1 or v2 — is refused by version rather than read
(`crates/sceneworks-core/src/film_compile.rs:626`), because a stale document read under this build
would have its derived fields defaulted and then be blamed as hand-edited (`:37`–`:41`). The remedy
either way is to recompile.

Per shot it carries the resolved partition and its `partitionReason`
(`crates/sceneworks-core/src/film_compile.rs:381`, `:387`); `referenceAssetIds` in the plan's role
order, written only when the list is non-empty (`:602`); the resolved geometry; the LoRA ids resolved
against **that shot's** partition (`:351`); `effectiveSteps` (`:375`); and
`referenceImageShortEdge`, written only for a reference-partition request (`:342`).

Two properties earn it its own file. It is **what the engine sees**: each prompt is run through the
model's own `prompt_refine` rewrite with `modelId` set to the plan's model, and the authored text is
kept beside it as `authoredPrompt` (`crates/sceneworks-core/src/film_compile.rs:134`). And it is
**the only place a job body is built**: `CompiledRequest::to_job_body`
(`crates/sceneworks-core/src/film_compile.rs:492`) produces the `POST /api/v1/video/jobs` payload for
the generated and hand-authored paths alike, so what a reviewer reads in `compiled.json` and what the
API receives cannot drift. It records the SHA-256 of the plan it came from, and `validate` / `run`
refuse a compiled document whose plan has changed (runbook § *Compiled requests*).

## The run loop

`film-harness run` (`apps/rust-api/src/film_harness.rs:4823`) creates nothing until the plan, the
pack, the model's catalog entry and the host all validate
(`apps/rust-api/src/film_harness.rs:1709`). Then, in order:

- **Project**, created as `<title> (<runId>)`, or reused with `--project-id`
  (`apps/rust-api/src/film_harness.rs:2381`, `:2385`). That name is how a resume adopts it.
- **Assets**: references imported and tagged (`:3206`, `:3306`); sound imported or synthesized
  (`:2538`, `:3133`).
- **Idempotency keys**, `<runId>:<shotId>:a<attempt>`
  (`apps/rust-api/src/film_harness.rs:1422`), written into the record **before** the job is created
  and stamped into `advanced.filmHarness.idempotencyKey`. That closes the one window a record alone
  cannot: a controller that died between creating the job and recording its id finds its own job
  instead of enqueuing a second (`crates/sceneworks-core/src/film_plan.rs:3420`).
- **Attempt caps**: `limits.maxAttemptsPerShot` counts **automatic** attempts only; a
  human-requested replacement is not a retry
  (`crates/sceneworks-core/src/film_plan.rs:3492`, `:3545`).
- **Memory preflight**: the budget must clear the **largest** declared `minMemoryGb` among the
  partitions the plan uses, not their sum: shots dispatch one job at a time, so both checkpoints are
  never resident together and each must fit on its own
  (`crates/sceneworks-core/src/film_plan.rs:2824`–`:2849`).
- **Two clocks**: `elapsedSeconds` is automatic work, cumulative across every controller;
  `humanRequestedElapsedSeconds` holds replacements and the exports they re-run, so one replacement
  can never exhaust the budget the run's own `resume` needs
  (`crates/sceneworks-core/src/film_plan.rs:3794`, `:3799`).
- **Cancel**: Ctrl-C, SIGTERM, or `film-harness cancel --out DIR` from another shell cancels the
  in-flight job through the API, stops dispatching and writes the record
  (`apps/rust-api/src/film_harness.rs:322`). A directory with no `run.json` is refused, not created,
  because a mistyped `--out` that prints "cancel requested" while the render keeps going is the one
  thing a cancel must never do (runbook § *Cancellation*).
- **Resume reconciliation**: `film-harness resume` (`apps/rust-api/src/film_harness.rs:5200`) reuses
  every recorded take, reads back every job the record names and adopts it at whatever state it
  reached, and refuses an edited plan, pack or compiled document: that is a new run, not a resume
  (runbook § *Durable run state, resume and take replacement*).

Only a **resumable** stop can be resumed. A cancel or a crash is resumable; an exhausted wall-clock
budget, an over-budget memory peak or an exhausted attempt cap is terminal, and `stop.detail` says
which plan value to change (`crates/sceneworks-core/src/film_plan.rs:3095`, `RunStop`).

**One controller per run directory.** Nothing locks `run.json`; the idempotency keys stop a
*sequential* replay from duplicating work but are not a lock between two live controllers (runbook
§ *Durable run state*, and `apps/rust-api/src/bin/film-harness.rs:54`).

## Review: assistive, never deciding

`film-harness review` (`apps/rust-api/src/film_harness/review.rs:769`) drives two routes the app
already serves and adds no model and no job type (runbook § *The two seams it drives*):

1. `POST /api/v1/projects/:p/timelines/:t/items/:i/frames`, the `frame_extract` job, samples the take
   at the review plan's declared positions and persists each frame as a project asset;
2. `POST /api/v1/image/vqa/jobs`, the `image_vqa` job, (SenseNova-U1-8B) is asked **one** declared
   question per frame.

Frame extraction rides a **separate one-item review timeline**, never the export timeline: reviewing
must not rewrite the thing the run is for. The vision half sits behind `ReviewVision`, so the whole
flow also runs against a scripted backend with no weights
(`apps/rust-api/src/film_harness/review.rs:406`); a scripted summary is evidence of a rehearsal, not
of a review (`crates/sceneworks-core/src/film_plan.rs:3134`).

Each review writes one `ObservedState` document under `<out>/reviews/`. It **references** the
intended state by JSON pointer and never copies it, so the two cannot drift; `unobserved` carries no
value at all, and an action the reviewer did not see is recorded `unobserved`, never `completed`
(runbook § *Observed state is not intended state*). Nothing there is an input to generation: the run
record gains only a `reviews[]` index entry, a path and some counts
(`crates/sceneworks-core/src/film_plan.rs:3134`).

**Flags never auto-decide.** Nothing the reviewer reports approves, rejects, conditions or re-renders
anything; only a human decision recorded through the controller does (runbook § *Reviewing a take*).

## Human decisions

| command | what it changes | what it never does |
| --- | --- | --- |
| `accept-take` | records `humanDecision: accepted`, clears **that shot's** `needsReview` flags | touch another shot; reach the API at all; accept a take carrying a `rejection` |
| `reject-take` | marks the take `rejection` (take, job and asset all stay), clears the selection, flags dependents, marks the export stale | re-render anything |
| `request-repair` | **one** bounded attempt through `replace-take`, with the review's actionable flags folded into the recorded reason | loop, retry a failed repair, or put an observation into the prompt |
| `replace-take` | rejects the carried take and dispatches **exactly one** more, outside the automatic budgets | respect or spend `maxAttemptsPerShot` |

Sources: runbook § *The human loop* and § *Replacing a take*;
`apps/rust-api/src/film_harness/review.rs:1604` (`decide_take`), `:1757` (`request_repair`),
`apps/rust-api/src/film_harness.rs:5260` (`replace_take`),
`crates/sceneworks-core/src/film_plan.rs:3492` (`human_requested`).

Attempt *n* renders at the plan's seed plus `(n − 1) × 1000`
(`film_compile::ATTEMPT_SEED_STRIDE`). The MLX render is deterministic for a seed (two runs of
SH010 at seed 22710, four hours apart, were pixel-identical frame for frame), so a replacement that
kept the plan's seed would re-render the take it had just rejected. The stride is 1000 rather than 1
because a plan's own per-shot seeds are usually spaced by one (runbook § *Replacing a take*).

Every decision is appended to `decisions[]` in the order it was made; replay adds nothing there
(`crates/sceneworks-core/src/film_plan.rs:3173`, `ProductionDecision`; the list at `:3785`).

## Timeline and export

The assembled timeline has four tracks: `track_main` (picture), `track_dialogue`, `track_ambience`
and `track_music`, each audio track a bus with its own `gain` and `muted`, and **the beds placed once
for the whole sequence** so they play straight through the cuts rather than restarting at each one
(runbook § *Sound*).

The export mixes every non-muted audio track: gain is `track.gain * item.volume`, clips are delayed
to where they land in the *exported picture*, per-item fades become `afade`, the summed mix passes a
limiter, and sound never extends the export. Timeline seconds are not picture seconds when a shot
carries a crossfade, and the mix is placed against the picture that was actually built.

**A layer the export could not mix is reported, not merely logged.** A placed clip whose asset is
missing, whose media file is gone, or which carries no decodable audio stream is dropped from the mix
and named in the job result, in the render asset's recipe, and in the run record's `export` entry
(`crates/sceneworks-core/src/film_plan.rs:3698`). "Why is the music missing" is answerable from
`run.json` alone.

The export lands as an ordinary project asset. The timeline is created at the nearest aspect ratio
the route admits (`16:9` / `9:16` / `1:1`), so the fixture's 9:5 takes export letterboxed, and the
record states both what the timeline was created at and what the takes actually are
(`crates/sceneworks-core/src/film_plan.rs:3635`, `:3638`).

`trim`, `reorder` and `swap-take` change a saved sequence without re-rendering
(`apps/rust-api/src/film_harness.rs:7088`, `:7129`). Every later assembly **merges into the saved
document** rather than rebuilding it: an existing picture item keeps its order, source range, version
history and `generatedAudio`, and an item a person pointed at a foreign asset with `swap-take` is
left exactly as they left it (runbook § *Editing the assembled sequence*). Each edit is appended to
`timeline.edits` **and** to the decision log, and leaves the MP4 flagged `export.stale` unless
`--export` re-runs it.

## Provenance recorded per attempt

`AttemptRecord` (`crates/sceneworks-core/src/film_plan.rs:3414`) is what makes a run answerable
afterwards:

| field | line | what it settles |
| --- | --- | --- |
| `idempotencyKey` | `:3420` | which job this attempt is, written before the POST |
| `resolvedModelId` | `:3426` | the checkpoint that rendered it, not the family the plan named |
| `partitionReason` | `:3429` | why that checkpoint and not the other |
| `referenceImageShortEdge` | `:3439` | the **effective** encode edge (the plan's, or 2048); absent on a base-partition attempt, which encodes no reference |
| `loras` | `:3446` | the adapter ids actually sent, in payload order; empty is a recorded state, not an absence |
| `effectiveSteps` | `:3455` | the count that ran: `advanced.steps`, else the recipe's, else the partition's `defaults.steps` (50) |
| `turboSchedulerShift` | `:3460` | the recipe's video sigma shift; absent in the base regime |
| `peakMemoryGb` | `:3476` | the number compared against `limits.maxMemoryGb` |
| `peakMemorySource` | `:3480` | which of `metrics.peakMemoryBytes`, `metrics.peakMemoryPct`, `job.peakGpuMemoryPct` supplied it |

`run.json`'s `model.partitionWeights` records the manifest download row behind **each** partition a
mixed run dispatched on, keyed by catalog model id, because a split family's reference `transformer_ref`
files are a second 18.78 GB download that `model.weights` never named
(`crates/sceneworks-core/src/film_plan.rs:3248`).

## Evaluation discipline

The harness is evaluated, not merely exercised. Two reports ship, with the same rubric, the same six
shots and the same evaluator method, so the numbers are comparable column for column
(`docs/film-harness-evaluation-phase-2.md:8`):

- an advance-declared **rubric** (identity, costume, location, parcel continuity, action completion,
  cut, 0/1/2 each; sound once at sequence level) with the decision rule stated before any frame was
  viewed (`docs/film-harness-evaluation-2026-09-14.md:229`);
- **budgets declared before dispatch**, and overruns reported against them
  (`docs/film-harness-evaluation-phase-2.md:44`);
- **`review-eval` labeled sets** that count detections, misses, false alarms, abstentions and
  **overclaims** separately, because an uncertain observation becoming a fact must not hide inside a
  detection count (runbook § *Labeled evaluation*);
- **offline proof**: the pass is run with the network observed
  (`docs/film-harness-evaluation-phase-2.md:139`);
- **cost metrics** per cell: wall clock, s/step, peak memory, and wall per accepted second.

Headline numbers, both passes:

| pass | configuration | picture /72 | render wall | s/step | peak | wall per accepted second |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (2026-09-14) | no references, `minimax_h3`, 50 steps | **58** | 6 332 s over 9 attempts | ~13 | 14.2 GB | 255 s |
| 2 (2026-09-15/16) | references on every shot, `minimax_h3_ref`, 50 steps, edge 2048 | **66** | 50 669 s | 137–140 | 24.3 GB | 1 634 s |
| 2 | six shots, turbo 4-step, edge 2048 | **64** | 4 722 s | 135–167 | 24.3 GB | **152 s** |
| 2 | SH010 alone at 1344x768, 50 steps | 12/12 | 23 961 s | 457 | 28.8 GB | 4 638 s |
| 2 | SH010 + SH050, reference edge 1024, 50 steps | 24/24 (vs 22/24 at 2048) | 3 376 s | 30–32 | 16.3 GB | 327 s |

Source: `docs/film-harness-evaluation-phase-2.md:360`–`:367`. Phase 1 recommended **revise**; phase 2
recommended **continue** (`docs/film-harness-evaluation-phase-2.md:461`). Neither report authorizes
the next phase, and both say so at the top.

## Subcommands

Header: `apps/rust-api/src/bin/film-harness.rs:5`–`:27`; worker requirements and exit codes: runbook
§ *Running*.

| command | what it does | shape | reads | writes |
| --- | --- | --- | --- | --- |
| `plan` | brief → LLM draft → validate → repair rounds | blocking, minutes | brief, pack, catalog | `plan.json`, `compiled.json`, `brief.json`, `planner-rejected.txt` on exhaustion |
| `compile` | rebuild the per-shot requests from an edited plan | blocking | plan, pack, sibling brief | `compiled.json` |
| `validate` | check plan, pack, host, catalog and workers; create nothing | blocking, seconds | plan, pack, `compiled.json` | nothing |
| `run` | render the selected shots, assemble, export | **long-running, resumable** | plan, pack, compiled | `run.json` (+ project mirror), project assets |
| `resume` | pick a run back up, adopting live jobs | **long-running, resumable** | `run.json` + the pinned documents | `run.json` |
| `replace-take` | reject the carried take, render exactly one more | long-running, one attempt | `run.json` | `run.json` |
| `request-repair` | `replace-take` with the review's flags in the reason | long-running, one attempt | `run.json`, `reviews/*.json` | `run.json` |
| `review` | frames plus one VQA question each, per declared question | long-running, bounded by the review plan's `limits` | `run.json`, `review.jsonc` | `reviews/<shot>-a<n>-r<n>.json`, `run.json` index |
| `accept-take` / `reject-take` | record the human decision | instant, **no API at all** | `run.json` | `run.json` |
| `trim` / `reorder` / `swap-take` | edit the saved sequence, render nothing | blocking; long only with `--export` | `run.json` | timeline, `run.json` |
| `cancel` | cancel the in-flight job from another shell | instant | `run.json` | cancel request file |
| `status` | print the record without touching the API | instant | `run.json` | nothing |
| `review-eval` / `review-fixtures` | measure the reviewer on a labeled set / write its frames | long-running / instant | labeled set | `review-eval.json`, `review-eval.txt`, observed states |
| `make-references` | render a pack's plates, **test fixtures only** | long-running | spec | a pack directory, published by rename |
| `fixture-images` / `fixture-sound` | deterministic placeholder plates and tone beds | instant | (none) | PNGs / WAVs |

`plan` and `compile` need a `prompt_refine` worker; `run` additionally needs `video_generate`,
`audio_generate` when the pack carries a `dialogue` entry with `text`, and `timeline_export` unless
`--no-export`; `review` and `review-eval` need `image_vqa`, and `review` also `frame_extract`. Exit
codes: 0 completed, 2 refused before dispatch (or the action does not apply to the record), 3 stopped
on a limit or a failed shot, 1 transport/API/io error (runbook § *Running*).

## Known costs on this Mac

Apple M5 Max, 128 GB unified, macOS 26.6.2, one native MLX worker, MiniMax-H3 q4, lane `mlx`
(`docs/film-harness-evaluation-phase-2.md`, §1 *Tested configuration*).

| configuration | s/step | peak |
| --- | --- | --- |
| base `minimax_h3`, 576x320, 50 steps | ~13–17 | 13.5–14.2 GB |
| `minimax_h3_ref`, 576x320, edge 2048, 50 steps | 137–140 | 24.3 GB |
| `minimax_h3_ref`, 576x320, edge **1024**, 50 steps | 30–32 | 16.3 GB |
| `minimax_h3_ref`, **1344x768**, edge 2048, 50 steps | 457 | 28.8 GB |
| `minimax_h3_ref`, 576x320, edge 2048, **turbo 4-step** | 135–167 | 24.3 GB |

Add a per-shot cold load on top: roughly 5–10 min for the reference DiT in cell (a), about 45 min
spread over the six shots because the checkpoint is re-selected per job
(`docs/film-harness-evaluation-phase-2.md:64`, with the per-job re-selection and the 5–10 min load
figure at `:103`–`:105`).

**The caveats the phase-2 report states, and they are not small**
(`docs/film-harness-evaluation-phase-2.md:430`–`:459`):

- one film, one brief, one seed set, one generated pack, one Mac; the 1344x768 cells are one shot
  each and the edge-1024 cell is two;
- the evaluator was an AI agent scoring three stills per take. Motion, texture and how a face reads
  in a moving shot (the very things turbo and the short edge change) were judged from stills or
  noted outside the rubric;
- the turbo per-step figures are medians over ~5 progress gaps per attempt, against ~56 for the
  50-step cells;
- the host was **shared** during parts of cell (a); two shots are flagged contaminated and excluded
  from the per-step figures (load average 16–20 pushed SH050 to 197 s/step against the clean
  137–140);
- cell (c)'s 28.8 GB peak against a 70–110 GB expectation is **unexplained**, and no conclusion about
  memory at full resolution should rest on it;
- the edge-1024 takes are different draws from the edge-2048 takes at the same seed, so their
  per-shot deltas mix sampling with fidelity;
- the turbo default was **not reached through the planner** in that pass: the plans were
  hand-authored, so cell (e) is about the recipe, not about a user getting to it;
- the no-reference baseline was not re-rendered; its comparability rests on a code reading and on the
  determinism phase 1 measured.

## What the harness deliberately does not do

- **No new UI, no parallel renderer, no second LLM stack** (runbook, opening paragraph). Every take is
  produced by whatever GPU worker claims the job.
- **No HTTP route.** The orchestration is a rust-api library module
  (`apps/rust-api/src/lib.rs:283`–`:284`) with a CLI in front of it; nothing in the running API
  exposes it today. See [film-harness-ui-integration.md](film-harness-ui-integration.md).
- **The reviewer decides nothing.** Only `accept-take`, `reject-take`, `request-repair`,
  `replace-take` and the edit verbs change anything.
- **No automatic regeneration of a flagged shot.** A `needsReview` flag is raised once per
  `(sourceShotId, dependency)` (`crates/sceneworks-core/src/film_plan.rs:3114`, `ReviewFlag`).
  Nothing automatic clears it; `accept-take` retires that shot's flags
  (`apps/rust-api/src/film_harness/review.rs:1659`–`:1660`, matching the `accept-take` row above),
  and the next upstream change raises the flag again
  (`apps/rust-api/src/film_harness.rs:5553`–`:5556`) — which is the point of resolving it.
- **The reference pack is required at the document level.** The epic's E1 wording, "the reference
  pack is never required", holds at the **shot** level and for **image conditioning**: a shot may
  name no `referenceRoles` and still compile. It does not hold at the document level — the harness
  today always requires a pack file with approved roles, and every shot must anchor to one of them.
- **Nothing chains a shot on the previous shot's last frame.** `chainFromShotId` records that a shot
  continues an earlier one and rides into provenance; it is never an anchor by itself, and a chained
  shot naming no canonical role is refused (runbook § *Editing the assembled sequence*).
- **No hosted LLM path.** The planner refuses before its first token if the environment carries a
  hosted credential or endpoint, or if `--api` is not loopback, a private-network address, a `.local`
  name or a bare hostname, and that rule guards every command that reaches an API, from the one
  place the binary builds its transport (`apps/rust-api/src/film_planner.rs:65`, `:551`).
- **No per-shot model overrides across families.** The only id resolution can produce is the declared
  model's own reference partition (runbook § *Partition resolution*).
- **`make-references` is not a product feature.** In the product the user supplies the reference
  images; it exists so the repository can build its own fixtures.
- **No editor UI for audio.** SC-12807 still owns the media bin, placing and trimming audio items
  from the timeline UI, and the fader controls. The backend will mix whatever is on a track; nothing
  in the UI puts a clip there (runbook § *What this did and did not settle for SC-12807*).
