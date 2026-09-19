# What the film harness does

For someone who has never run it. The operating manual is
[film-harness.md](film-harness.md); this page is the shape of the thing, why each piece exists, and
what it cost when it was measured. The current editor workflow is in
[film-editor.md](film-editor.md), and its shared API/library seams are in
[film-harness-ui-integration.md](film-harness-ui-integration.md).

Citation convention below: code is cited as `path:line` against this commit; the runbook is cited by
section, because its line numbers move and its sections do not.

## The problem it solves

SceneWorks can render one clip at a time from the Video Studio. A film is not one clip. It is an
ordered shot list, optional approved references, optional dialogue and sound, an editable sequence,
and a person deciding which takes are good enough over GPU work that may be interrupted.

The harness is the bounded, resumable, human-controlled orchestration behind that work. The Film
workspace now exposes it through project-scoped routes while the CLI remains available for document
and automation workflows. Both use the existing workers, jobs, assets, timelines, and durable run
record; neither introduces a second renderer.

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
`continuityRoles` is the continuity claim for a reference-backed shot: every named role must exist
in the approved pack. A script-only shot may leave both role arrays empty; explicitly requesting
reference conditioning without an approved bindable role is refused (runbook § *Editing the
assembled sequence*, the "Continuity is canonical" paragraph).

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
(`crates/sceneworks-core/src/film_compile.rs`, `compile_shot`):

| the shot's `conditioning.referenceRoles` | it compiles to |
| --- | --- |
| non-empty | `minimax_h3_ref`, with `referenceAssetIds` in the plan's role order |
| empty | `minimax_h3`, its declared mode, and no reference field at all |

**References and image conditioning are optional.** The CLI keeps `--references` as the path to a
reference-pack document, while the Film workspace creates and stores that document with the draft.
Its `references` array may be empty, and a script-only shot needs no approved role. A shot that names
no `conditioning.referenceRoles` compiles to the base `minimax_h3` with its declared mode and no
reference field. A shot that explicitly declares `reference_to_video` still must bind approved
character, prop, or location roles and is refused by name when it does not (runbook
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
(`crates/sceneworks-core/src/film_compile.rs`, `CompiledRequest::reference_image_short_edge`).

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

1. **Refuse before the first token.** The CLI's hosted-credential and remote-API ban, the brief's
   structure, the pack and its files, the model's catalog entry, a beat requiring a role the pack
   does not approve, and the brief's own model block against the installed menus. The brief must
   declare `limits.plannerMaxMemoryGb`, checked against the host's reported memory before the first
   token.
2. **One draft** through `POST /api/v1/prompts/refine` with `task: "film_plan"`
   (`apps/rust-api/src/film_planner.rs:60`), decoded under a valid-JSON constraint. Object shape is
   *not* enforced by the decoder; the plan schema is enforced after the decode by
   `parse_planner_output`.
3. **Validate**, then **bounded repair rounds**, default 2, ceiling 5
   (`apps/rust-api/src/film_planner.rs:49`, `:53`). Each round hands the validator's findings back
   verbatim and asks for the whole plan again. A repair may change the number of shots or choose
   another legal duration to meet the brief's running-time window, but it must retain every required
   beat and pass the same full validation. On exhaustion the refused answer is written to
   `planner-rejected.txt`.

The Film workspace uses the same generation, validation, repair and compile functions through a
durable planning operation. New drafts select the built-in `prompt_refine_anubis_8b`; the optional
local Qwen3.6-27B planner is installed and selected only on request. A saved OpenAI-compatible
connection is also an explicit choice, with disclosure, backend-held secret and separate reference
pixel opt-in. Planner identity and thinking mode are recorded independently of the target video
model. There is no provider fallback, and an unavailable, canceled or exhausted operation leaves
the draft available for manual editing. The full operator flow is in
[film-editor.md](film-editor.md).

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
`COMPILED_PLAN_SCHEMA_VERSION` is 4 (`crates/sceneworks-core/src/film_compile.rs`,
`COMPILED_PLAN_SCHEMA_VERSION`); a document at any earlier schema version — v1, v2 or v3 — is
refused by version rather than read (`CompiledPlan::staleness_findings`), because a stale document
read under this build would have its derived fields defaulted and then be blamed as hand-edited.
The remedy either way is to recompile.

Per shot it carries the resolved partition and its `partitionReason`
(`crates/sceneworks-core/src/film_compile.rs`, `CompiledRequest::model` and
`CompiledRequest::partition_reason`); `referenceAssetIds` in the plan's role order, written only when
the list is non-empty (`CompiledRequest::to_job_body_with`); the resolved geometry; the LoRA ids
resolved against **that shot's** partition (`CompiledRequest::loras`); `effectiveSteps`
(`CompiledRequest::effective_steps`); and `referenceImageShortEdge`, written only for a
reference-partition request (`CompiledRequest::reference_image_short_edge`).

A reference shot's prompt also carries text the **compiler** wrote. MiniMax-H3 labels each supplied
reference `<Picture 1>`, `<Picture 2>`, … ahead of the prompt, in supply order, and the model's own
prompt guide is explicit that a reference needs a job in the text. So the compile leads such a
prompt with one plain binding sentence per bound role — "The courier is the person shown in
`<Picture 1>`." — built from the pack entry's kind and its own description. Two rules make it
trustworthy: the sentences are written **after** the `prompt_refine` rewrite, so no language model
can paraphrase a label the engine applies positionally; and the `<Picture N>` and the position of
that role's asset in `referenceAssetIds` both come from `shot_reference_pictures`, the one function
that owns the reference order, called by the compiler and by the dispatcher alike. The inserted text
is recorded per kind in the request's `insertedText`, separately from `authoredPrompt`, and is a
derived field — a hand-edited one is refused by `conformance_findings` like any other.

Two properties earn it its own file. It is **what the engine sees**: each prompt is run through the
model's own `prompt_refine` rewrite with `modelId` set to the plan's model, and the authored text is
kept beside it as `authoredPrompt` (`crates/sceneworks-core/src/film_compile.rs`,
`CompiledRequest::authored_prompt`). And it is **the only place a job body is built**:
`CompiledRequest::to_job_body_with` (`crates/sceneworks-core/src/film_compile.rs`) produces the
`POST /api/v1/video/jobs` payload for the generated and hand-authored paths alike, over conditioning
`CompiledRequest::resolve_conditioning` resolved — the one resolver, so the `<Picture N>` in the
prompt and the position of that role's asset in `referenceAssetIds` are one decision — and so what a
reviewer reads in `compiled.json` and what the API receives cannot drift. It records the SHA-256 of the plan it came from, and `validate` / `run`
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

**One controller per run directory.** Every mutating controller holds `ControllerLease`, an
advisory lock whose owner marker is cleared on clean release and retained after a crash. A second
controller is refused. Idempotency keys separately prevent duplicate work when a process dies
between posting a job and recording its id (runbook § *Durable run state*).

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

The assembled timeline has a picture track and ordered audio lanes for dialogue, ambience, music
and each placed effect. Each audio track is an editable bus with its own `gain` and `muted` state.
Sequence beds are placed once so they play through cuts instead of restarting at every shot. When a
later incremental delivery lengthens the cut, an untouched full-sequence bed extends with it;
gain, mute, fades and item volume remain intact. An explicit placement, source-range or duration
trim is preserved and is never stretched automatically (runbook § *Sound*).

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

The Film workspace always starts runs without export. It delivers each completed shot into the
saved timeline, preserves concurrent edits with revision checks and tombstones, and exports only
when the operator chooses **Export current cut**. The Operations panel can select among the draft's
durable runs; choosing an older run does not move or recreate its jobs.

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
| `run` | render selected shots and assemble; CLI exports unless `--no-export` | **long-running, resumable** | plan, pack, compiled | `run.json` (+ project mirror), project assets |
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

- **No parallel renderer.** The Film workspace and CLI both dispatch through the existing worker
  and job paths. Project-scoped film routes call the shared orchestration rather than reimplementing
  it. See [film-editor.md](film-editor.md) and
  [film-harness-ui-integration.md](film-harness-ui-integration.md).
- **The reviewer decides nothing.** Only `accept-take`, `reject-take`, `request-repair`,
  `replace-take` and the edit verbs change anything.
- **No automatic regeneration of a flagged shot.** A `needsReview` flag is raised once per
  `(sourceShotId, dependency)` (`crates/sceneworks-core/src/film_plan.rs:3114`, `ReviewFlag`).
  Nothing automatic clears it; `accept-take` retires that shot's flags
  (`apps/rust-api/src/film_harness/review.rs:1659`–`:1660`, matching the `accept-take` row above),
  and the next upstream change raises the flag again
  (`apps/rust-api/src/film_harness.rs:5553`–`:5556`) — which is the point of resolving it.
- **References are optional.** The editor supports script-only films with an empty pack and
  text-only shots. The CLI still accepts a reference-pack document path, but that document need not
  contain an approved image unless a shot actually requests reference conditioning.
- **Nothing chains a shot on the previous shot's last frame.** `chainFromShotId` records that a shot
  continues an earlier one and rides into provenance; it is never an anchor by itself, and a chained
  shot naming no canonical role is refused (runbook § *Editing the assembled sequence*).
- **External planning is explicit and has no fallback.** The CLI planner keeps its local/private
  transport policy. The Film workspace can instead use a saved OpenAI-compatible connection when an
  operator selects it, with bounded time/output, a separate host secret, a data disclosure, and
  reference pixels disabled by default. Provider failure or restart never switches to a local model.
- **No per-shot model overrides across families.** The only id resolution can produce is the declared
  model's own reference partition (runbook § *Partition resolution*).
- **`make-references` is not a product feature.** In the product the user supplies the reference
  images; it exists so the repository can build its own fixtures.
- **Sound remains an explicit edit.** The Film workspace can stage prerecorded or synthesized
  dialogue, ambience, music, and effects and set placement, trims, fades, gain, and mute. The saved
  timeline remains authoritative, and export reports any audio layer it could not include.
