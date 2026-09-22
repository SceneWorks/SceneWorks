# What the film harness does

For someone who has never run it. The operating manual is
[film-harness.md](film-harness.md); this page is the shape of the thing, why each piece exists, and
what it cost when it was measured. The current editor workflow is in
[film-editor.md](film-editor.md), and its shared API/library seams are in
[film-harness-ui-integration.md](film-harness-ui-integration.md).

Citation convention below: code is cited by **file and symbol** — the function, struct or constant
that owns the behaviour — never by line number, because a line number is stale the next time anyone
edits above it and a reader then checks the claim against the wrong code. The runbook is cited by
section for the same reason. Reports under `docs/` keep their line citations: they are frozen
records, not code.

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
| **Resumable** | `run.json` (`RunRecord`) is rewritten atomically at every state transition through `film_harness::write_atomically`, so a killed controller leaves a record `resume` can reconcile |
| **Human-controlled** | only `accept-take`, `reject-take`, `request-repair`, `replace-take` and the `TimelineEdit` verbs change anything (`apps/rust-api/src/bin/film-harness.rs`, module header) |
| **Measurable** | every attempt records the checkpoint, the adapters, the step count, the reference short edge and where its memory number came from (`crates/sceneworks-core/src/film_plan.rs`, `AttemptRecord`) |

## The three documents

A run is driven by three versioned documents. They are separate on purpose: what the film is, what it
may be conditioned on, and how to interrogate the result are three different authorities with three
different lifetimes. All three tolerate JSONC comments and refuse unknown fields (runbook
§ *Documents*).

### 1. The plan: what the film is

`ProductionPlan` (`crates/sceneworks-core/src/film_plan.rs`), schema version `PLAN_SCHEMA_VERSION`
= 3; `SUPPORTED_PLAN_SCHEMA_VERSIONS` is the accepted set, and it holds that one version only, so a
version 1 or 2 document is refused **by version** — they predate the required `shots[].audio`
sentence, and the refusal names the edit (`unsupported_plan_schema_message`). The refusal reaches
both paths: `parse_plan_document` refuses the document before serde decodes it, and
`validate_plan_structure` reports the same sentence as a finding. A plan pinned inside a run and a
plan imported into the workspace are read through those same functions, so neither is a way in for
an older document. The plan declares the model once, then a list of shots with stable ids.

```jsonc
// config/film-harness/courier-workshop/plan.v2.jsonc
"schemaVersion": 3,
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
    "audio": "Room tone, distant birds, a door latch clicking and the door creaking open. No music.",
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

`audio` is **required on every shot** (sc-24026) and is never pattern-matched: MiniMax-H3 generates
its soundtrack from the same prompt it renders the picture from, so whatever the prompt leaves
unsaid the model invents. A silent shot says so outright — `"No audio. Silence."` is a complete,
accepted value; only saying *nothing* is refused, and the refusal names the shot. The compiler
appends it to the dispatched prompt as the final sentence, `Audio: <the shot's text>`, after the
prompt-refine rewrite and recorded in `insertedText` — see § *What the compiler writes into a
prompt* below.

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

`ReferencePack` (`crates/sceneworks-core/src/film_plan.rs`), schema version
`REFERENCE_PACK_SCHEMA_VERSION` = 2. A separate document with its own version, so approved
references stay addressable independently of any generated take. A version 1 pack is refused by
`validate_reference_pack`, naming the edit that fixes it: version 2 only **adds** the optional
`references[].locator` and makes `references[].file` itself optional, so a version 1 document needs
no other edit unless two of its roles share one file.

```jsonc
// config/film-harness/courier-workshop/references.jsonc
"references": [
  { "role": "courier",           "kind": "character", "file": "references/courier.png",
    "description": "The courier: blue jacket, carries the parcel." },
  { "role": "red_parcel",        "kind": "prop",      "file": "references/red_parcel.png" },
  { "role": "workshop_location", "kind": "location",  "file": "references/workshop_location.png" },
  { "role": "house_style",       "kind": "style",     "file": "references/house_style.png" },
  { "role": "workshop_plate",    "kind": "plate",     "file": "references/workshop_plate.png" },
  // DESCRIBED-ONLY (sc-24025): a subject the pack describes but has no picture of. `file` is
  // optional; an entry with neither a file nor a description is refused.
  { "role": "recipient",         "kind": "character",
    "description": "The recipient: grey work apron, rolled sleeves." }
],
"sound": [
  { "role": "courier_line", "kind": "dialogue", "voice": "am_michael",
    "text": "Delivery. I'll leave it on the bench." },              // spoken by the run
  { "role": "workshop_room_tone", "kind": "ambience", "file": "sound/workshop_room_tone.wav" }
]
```

**A described-only role is never bindable** — it supplies no image, so naming it in
`conditioning.referenceRoles` or a keyframe slot is refused, and the refusal points at
`continuityRoles` as the place it belongs. Having no image is the whole of what it is, so a fileless
entry may carry nothing that presumes one: a `locator` picks a subject out of a picture, and a
`sourceAssetId` or generation provenance describes one, so each of the three is refused on a role
with no `file`. Its `description` is correspondingly not optional — an entry with neither a file nor
a description is a role that is nothing at all. It belongs in a shot's
`continuityRoles`, and that is where it earns its keep: for every shot, the compiler writes the
pack's `description` of each continuity role the shot does **not** bind to an image into the prompt
**word for word**, identically every time (the **text identity lock**). With no picture anywhere,
that repetition is the only thing holding a subject together across cuts. One exception, and it is
the shared-file case below: if a continuity role's `file` is the file of a picture the shot *is*
binding, its image is already being supplied, so it gets that picture's **binding** sentence with
its locator instead of a description of its own. See
[film-harness.md](film-harness.md) for the full rule, and
`config/film-harness/courier-workshop/plan.described.jsonc` (with
`references.described.jsonc` beside it) for a six-shot no-reference turbo film built this way.

**What the identity lock actually holds, and what it does not.** It is words, so it holds what words
specify: wardrobe, props and setting — a blue jacket stays a blue jacket, a red parcel stays a red
parcel, the same workshop stays the same workshop, because every shot states them in the same
sentence rather than in whatever paraphrase that shot's author reached for. It is **not** expected
to hold a face. No description distinguishes one courier's features from another's closely enough
for a text-to-video model to re-draw the same person, and nothing here claims otherwise: the lock
removes the drift that comes from re-wording a subject, not the drift that comes from having no
pixels of them. A film that needs a recognisable face needs an image-backed role and a shot that
binds it.

**Several roles may name the same `file`** (sc-24024) — one photograph holding two people is one
image with two subjects in it. Then each sharing role must carry a `locator`, the phrase that picks
its subject out of that image, and the pack is refused naming the roles and the file if any does not:

```jsonc
{ "role": "courier",   "kind": "character", "file": "references/pair.png",
  "locator": "the woman on the left" },
{ "role": "recipient", "kind": "character", "file": "references/pair.png",
  "locator": "the man on the right" }
```

A `locator` is a **noun phrase including its article**: it completes the sentence "The courier is …",
which the compiler writes verbatim and adds nothing to. Write `"the woman on the left"`, not
`"woman on the left"` — the latter is accepted (a locator is free prose; nothing can check it) and
reads "The courier is woman on the left in `<Picture 1>`."

"The same file" is the `file` string compared **literally** — nothing canonicalizes through the
filesystem, so two entries meaning one image must spell its path one way. To make that enforceable
rather than a convention, a `file` is refused unless it is already in canonical form: no leading
`./`, no `.` component, no doubled or trailing `/`, no `\`. Two entries whose paths differ only by
ASCII case are refused too, because a case-insensitive volume holds one file where the pack declares
two. A shared file is imported as
**one** project asset that every sharing role resolves to, supplied to the engine **once** under one
`<Picture N>`, and each role's binding sentence carries its own locator ("The courier is the woman on
the left in `<Picture 1>`."). Because the file is supplied once, `limits.maxReferenceAssets` counts
**distinct files**: a shot binding ten roles across nine files fits MiniMax-H3's cap of nine. Sharing
roles must also agree on `approved` and on their generation provenance — one image carries one of
each.

`referenceRoles` may bind only the **subject** kinds (`character`, `prop`, `location`) because
Ref2VA treats every bound image as a subject to depict; a `style` or a `plate` bound there is refused
naming the shot, the role and the kind (runbook § *Documents*). An entry with `"approved": false` is
still imported so a human can look at it, but it is tagged `film-harness-reference-unapproved` and
never resolved into conditioning.

References are **user-provided input** in the product. `film-harness make-references` exists only to
build this repository's own courier fixtures on this machine, and its section opens by saying so
(runbook § *Generating the reference plates*).

### 3. The review plan: how to interrogate a take

`ReviewPlan` (`crates/sceneworks-core/src/film_review.rs`, schema version
`REVIEW_PLAN_SCHEMA_VERSION` = 1). Per shot, closed questions with the
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
(`crates/sceneworks-core/src/film_plan.rs`, `AttemptRecord::resolved_model_id`).

The **reference short edge** is a plan-level knob:
`model.advanced.referenceImageShortEdge` sets the pixel short edge an image reference is *encoded*
at, admitted over 1024..=2048 and defaulting to the engine's 2048. It sizes the reference, never the
render, and a value outside the range is refused naming the field and the range rather than clamped,
because a silent clamp would change the token budget the author measured (runbook § *Partition
resolution*). It reaches only shots that resolve to the reference partition
(`crates/sceneworks-core/src/film_compile.rs`, `CompiledRequest::reference_image_short_edge`).

## Dialogue is spoken through the audio route

A `dialogue` pack entry may carry `text` instead of `file`. Before any render, `Session::ensure_sound`
(`apps/rust-api/src/film_harness.rs`, which speaks a line through `synthesize_dialogue`) speaks each **placed** line through the ordinary
`POST /api/v1/audio/jobs`, writes the WAV into the pack directory and imports it on the dialogue bus
exactly as it imports a pre-recorded clip (runbook § *Spoken dialogue*). Models are `kokoro_82m`
(default), `chatterbox_tts`, `moss_tts_realtime`, `moss_ttsd_v05`. Only what the run places is
spoken, so a `--shots` selection never pays for a line it left out. No live `audio_generate` worker
is a resumable stop *before the job exists*, scoped to the lines still owed.

Because the harness speaks a **placed** line itself, the words of that line belong in
`dialogueClip` and stay out of the shot's `audio`: the compiler appends `NO_SPEECH_SENTENCE` to any
shot carrying a `dialogueClip`, so H3 does not score a second voice over ours. A shot with a spoken
line and no clip is the other case, and there the speaker, the words and the delivery belong in
`audio` — it is the only text H3 scores a voice from. A generated plan is always that second case:
`film_planner::draft_to_plan` writes `dialogue_clip: None` on every shot, so planner-written films
carry their spoken lines in `audio` and only a hand edit ever places a clip.

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
   (`apps/rust-api/src/film_planner.rs`, `FILM_PLAN_TASK`, sent by `generate_with_refiner`), decoded
   under a valid-JSON constraint. Object shape is *not* enforced by the decoder; the plan schema is
   enforced after the decode by `parse_planner_output`.
3. **Validate**, then **bounded repair rounds**, default 2, ceiling 5
   (`apps/rust-api/src/film_planner.rs`, `DEFAULT_MAX_REPAIR_ROUNDS` and
   `MAX_REPAIR_ROUNDS_CEILING`). Each round hands the validator's findings back
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
family's reference partition, and the pack must approve at least one **image-backed** role of a
bindable kind — `ReferenceEntry::is_bindable_image`, which is `approved`, a `file`, and a kind in
`BINDABLE_REFERENCE_KINDS`. With no such entry, `PlannerCapabilities::narrowed_to_pack` drops
`reference_to_video` from the offered modes and sets the image budget to zero, so the contract the
planner is shown never mentions references at all. A described-only role is an approval of
words, and `reference_to_video` conditions on pixels — counting one would invert the default mode
to a binding `validate_plan_against_pack` then refuses for the whole round budget, and would offer
the reference partition's Turbo adapter for a partition the film never dispatches. With both, the
default mode inverts to `reference_to_video` for every shot showing an approved subject; with either
missing the plan stays on the base checkpoint. The same brief and pack therefore produce the same
film on two machines (runbook § *Partition resolution*, "Planning a reference film").

**Turbo is the default; `preferQuality` is the opt-out.** The envelope lists the step-distill
accelerators this host has **installed** (at most one per partition, paired by the catalog's
`modelIds`), and the output contract shows the exact `loras` array to copy. Install state *is* a
filter here, unlike the reference partition's, because an adapter whose weights are not on the render
host's disk is a 400 at enqueue. A brief setting `"preferQuality": true` keeps the full step path:
nothing is offered, and any selection a draft writes anyway is stripped rather than argued with
(`crates/sceneworks-core/src/film_planner.rs`, `ProductionBrief::prefer_quality`, applied in
`draft_to_plan`).

**The output contract carries the sound, and disowns the labels.** Every shot shape the planner is
shown — the template it fills and the one worked example it copies — carries `audio`, with the
wording MiniMax-H3's own prompt guide asks for
([minimax-h3.md](../apps/web/public/prompt-guides/minimax-h3.md), *Prompt the audio explicitly*):
the diegetic sound the action makes, then the ambience, then `"No music."` unless music is wanted;
an outright statement of silence is a complete answer; and the value never opens with the
compiler's own `Audio: ` label.

A **spoken** shot is the one case where the two paths differ, and the planner is told only about
its own. `audio` is the only text H3 scores a voice from, so the contract has a planner-written
shot carry the line itself — the speaker, the words in quotes and the delivery, the guide's *Voice*
bullet — beside the diegetic sound and the ambience, shown as a literal value to copy, with a
reminder to keep the line short enough to say inside the shot's duration. A generated plan can
never contradict that, because it places no sound: `draft_to_plan` writes `dialogueClip: None` and
nothing but a hand edit ever sets one. In a HAND-AUTHORED plan that does place a clip, the words
belong in `dialogueClip` and out of `audio`: the harness speaks that line itself, and the compiler
appends `NO_SPEECH_SENTENCE` so H3 does not lay a second voice over ours. `Shot::dialogue` is the
plan's own record of the line either way — `film_compile` never reads it — and is what the run
record's `IntendedState` carries. A shot whose
`audio` is missing or blank is a finding naming the shot, and the repair round restates the
requirement as the key and a value to adapt rather than as prose about it
(`film_planner::AUDIO_REQUIREMENT_RESTATEMENT`).

The contract also tells the planner what it must NOT write: no `<Picture N>`, `<Audio N>` or
`<Video N>` label anywhere, no restatement of a role's pack description in `prompt`, and nothing
about which picture shows whom — all three are the compiler's, written after the answer. What it
asks for instead is the positive half of the same rule: **name the role and show it doing
something**. A beat's required roles are handed over as the JSON arrays to write, with the
instruction to show each of them on screen — "name the role and say what it does, never restate the
pack's description of it" — and the contract's own rule tells the planner to spend `prompt` on what
HAPPENS, because the compiler writes one identity sentence for every role a shot names and one
binding sentence for every role it conditions on, word for word from the pack. A draft
that writes one anyway is a finding naming the shot, quoting the label and asking for its deletion
(`film_planner::anchoring_findings`), and the repair round restates that in copyable form too. This
is a **planner** finding, deliberately outside `validate_all`: a person who types `<Picture 1>` into
a hand-authored plan means it, and `validate`/`compile` leave them alone. The one exception is
`audio`, which is inserted prose on every path and where a `<` is already refused by the document
rules — so a label there is reported once, by them.

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
`COMPILED_PLAN_SCHEMA_VERSION` is 7 (`crates/sceneworks-core/src/film_compile.rs`,
`COMPILED_PLAN_SCHEMA_VERSION`, which is the value to read rather than this sentence); a document at
any *earlier* schema version is refused by version rather than read
(`CompiledPlan::staleness_findings`), because a stale document read under this build would have its
derived fields defaulted and then be blamed as hand-edited. The remedy either way is to recompile.

A compiled document is also tied to the **reference pack** it was compiled against, by the
`referencePackSha256` of the parsed pack (sc-24029). The pack decides what the compiler writes into
a prompt, so editing a pack description — a first-class action in the Film workspace — invalidates
every compiled document made before it. `staleness_findings` says so directly: *the reference pack
changed since these requests were compiled; recompile, or use authored prompts*. In the workspace
the remedy is the **Use authored prompts** button or a re-plan; from the CLI it is `film-harness
compile`. Hashing the *parsed* pack rather than the document's bytes is what lets the CLI's JSONC
file and the workspace's typed draft agree on one identity, and is why a comment- or
whitespace-only edit of the document does not stale a compile.

Per shot it carries the resolved partition and its `partitionReason`
(`crates/sceneworks-core/src/film_compile.rs`, `CompiledRequest::model` and
`CompiledRequest::partition_reason`); `referenceAssetIds` in the plan's role order, written only when
the list is non-empty (`CompiledRequest::to_job_body_with`); the resolved geometry; the LoRA ids
resolved against **that shot's** partition (`CompiledRequest::loras`); `effectiveSteps`
(`CompiledRequest::effective_steps`); and `referenceImageShortEdge`, written only for a
reference-partition request (`CompiledRequest::reference_image_short_edge`).

### What the compiler writes into a prompt

A shot's prompt also carries text the **compiler** wrote. MiniMax-H3 labels each supplied
reference `<Picture 1>`, `<Picture 2>`, … ahead of the prompt, in supply order, and the model's own
prompt guide is explicit that a reference needs a job in the text. So the compile leads such a
prompt with one plain binding sentence per bound role — "The courier is the person shown in
`<Picture 1>`." — built from the pack entry's kind and its own description. Two rules make it
trustworthy: the sentences are written **after** the `prompt_refine` rewrite, so no language model
can paraphrase a label the engine applies positionally; and the `<Picture N>` and the position of
that role's asset in `referenceAssetIds` both come from `shot_reference_pictures`, the one function
that owns the reference order, called by the compiler and by the dispatcher alike. The inserted text
is recorded per kind in the request's `insertedText`, separately from `authoredPrompt`, and is a
derived field — a hand-edited one is refused by `CompiledPlan::conformance_findings` like any other.

Because the labels are the compiler's, **a refined prompt that contains one is refused**, naming the
shot and quoting the label (`compile_shot`, over `film_plan::engine_label_at`). The rewrite comes
back from a language model that was handed the model's own prompt guide, and that guide teaches
`<Picture N>` as the way to give a reference a job — so this is exactly the text most likely to
carry one, and a label written there names a picture the request never supplies. It is refused
rather than stripped, because a rewrite that named a picture is a rewrite built around one. Three
remedies are named in the message: re-run the refinement, pass `--no-refine` from the CLI, or untick
*Run model-specific prompt refinement when compiling shots*
(`film_compile::REFINE_PROMPTS_CONTROL_LABEL`) in the Film workspace. A **hand-authored** prompt is
deliberately not scanned: a person who types `<Picture 1>` into a plan means it. A planner draft that
writes one is the third case, and it is a finding rather than a refusal
(`film_planner::anchoring_findings`).

`conformance_findings` covers the dispatched **`prompt`** and the recorded **`authoredPrompt`** as
well as the derived fields. It recompiles the shot from the plan and compares: an authored request
must reproduce the plan's prompt exactly; a refined one must still have the compiler's own leading
and trailing sentences around it where the compiler wrote them, and the refined text recovered from
between them must itself be label-free.

Every shot's prompt also **trails** with its audio sentence, `Audio: <the shot's `audio` text>` —
base partition and reference partition alike, since H3 scores a soundtrack from the same text it
renders the picture from. The author's words are repeated with their whitespace normalized and
nothing else changed; the compiler never reads the prose, so a shot that states silence gets exactly
that sentence. When the shot also **places** a dialogue line — a `dialogueClip`, whether
`ensure_sound` speaks it through TTS or imports a recording, since both put our own voice on the
dialogue bus — one further fixed sentence follows it, `film_compile::NO_SPEECH_SENTENCE` ("No
spoken dialogue in the generated audio; no voices on the soundtrack."), so H3 does not lay a second
voice over ours. It constrains the **soundtrack** and says nothing about the picture: these are
exactly the shots where someone *is* speaking on camera, so a sentence phrased as a statement about
what is shown would ask for closed mouths under our own dialogue track. A shot with no placed clip
gets nothing: it has no voice to double, and `dialogue` beside it is intent prose the run never
plays. Both are written after the refine rewrite and recorded as their own `insertedText` kinds
(`audio`, `no_speech`).

Each trailing sentence is joined to what precedes it by a **sentence boundary**, not a bare space:
neither the authored prompt, the refiner's rewrite nor the author's `audio` text is guaranteed to
end in terminal punctuation, so the compiler supplies the missing `.` rather than dispatching
`...a courier enters Audio: Room tone`. The `Audio: ` label is compiler-owned, and a plan whose
`audio` value starts with it is refused naming the shot rather than dispatching it twice.

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

`film-harness run` (`apps/rust-api/src/film_harness.rs`, `run` into `run_with_control_inner`)
creates nothing until the plan, the pack, the model's catalog entry and the host all validate
(`preflight_documents`, with `host_findings` for the host half). Then, in order:

- **Project**, created as `<title> (<runId>)`, or reused with `--project-id`
  (`Session::ensure_project`). That name is how a resume adopts it.
- **Assets**: references imported and tagged (`Session::ensure_references`, which uploads **one
  asset per distinct `file`**, so roles sharing a photograph resolve to one id); sound imported or
  synthesized (`Session::ensure_sound`, `synthesize_dialogue`).
- **Idempotency keys**, `<runId>:<shotId>:a<attempt>`
  (`film_harness::idempotency_key`), written into the record **before** the job is created
  and stamped into `advanced.filmHarness.idempotencyKey`. That closes the one window a record alone
  cannot: a controller that died between creating the job and recording its id finds its own job
  instead of enqueuing a second (`AttemptRecord::idempotency_key`).
- **Attempt caps**: `limits.maxAttemptsPerShot` counts **automatic** attempts only; a
  human-requested replacement is not a retry (`AttemptRecord::human_requested`, which the attempt
  accounting filters on).
- **Memory preflight**: the budget must clear the **largest** declared `minMemoryGb` among the
  partitions the plan uses, not their sum: shots dispatch one job at a time, so both checkpoints are
  never resident together and each must fit on its own (`validate_plan_against_model`, over
  `model_min_memory_gb` for each partition `ModelEntries::partitions_used` reports).
- **Two clocks**: `elapsedSeconds` is automatic work, cumulative across every controller;
  `humanRequestedElapsedSeconds` holds replacements and the exports they re-run, so one replacement
  can never exhaust the budget the run's own `resume` needs (`RunRecord::elapsed_seconds` and
  `RunRecord::human_requested_elapsed_seconds`).
- **Cancel**: Ctrl-C, SIGTERM, or `film-harness cancel --out DIR` from another shell cancels the
  in-flight job through the API, stops dispatching and writes the record
  (`film_harness::request_cancel`). A directory with no `run.json` is refused, not created,
  because a mistyped `--out` that prints "cancel requested" while the render keeps going is the one
  thing a cancel must never do (runbook § *Cancellation*).
- **Resume reconciliation**: `film-harness resume` (`film_harness::resume`, into `continue_run`) reuses
  every recorded take, reads back every job the record names and adopts it at whatever state it
  reached, and refuses an edited plan, pack or compiled document: that is a new run, not a resume
  (runbook § *Durable run state, resume and take replacement*).

Only a **resumable** stop can be resumed. A cancel or a crash is resumable; an exhausted wall-clock
budget, an over-budget memory peak or an exhausted attempt cap is terminal, and `stop.detail` says
which plan value to change (`crates/sceneworks-core/src/film_plan.rs`, `RunStop`).

**One controller per run directory.** Every mutating controller holds `ControllerLease`, an
advisory lock whose owner marker is cleared on clean release and retained after a crash. A second
controller is refused. Idempotency keys separately prevent duplicate work when a process dies
between posting a job and recording its id (runbook § *Durable run state*).

## Review: assistive, never deciding

`film-harness review` (`apps/rust-api/src/film_harness/review.rs`, `review`) drives two routes the app
already serves and adds no model and no job type (runbook § *The two seams it drives*):

1. `POST /api/v1/projects/:p/timelines/:t/items/:i/frames`, the `frame_extract` job, samples the take
   at the review plan's declared positions and persists each frame as a project asset;
2. `POST /api/v1/image/vqa/jobs`, the `image_vqa` job, (SenseNova-U1-8B) is asked **one** declared
   question per frame.

Frame extraction rides a **separate one-item review timeline**, never the export timeline: reviewing
must not rewrite the thing the run is for. The vision half sits behind the `ReviewVision` trait — `VqaVision` for the API seam, `ScriptedVision`
for a fake — so the whole flow also runs against a scripted backend with no weights; a scripted
summary is evidence of a rehearsal, not of a review, and `TakeReviewSummary::backend` is where it
says which it was.

Each review writes one `ObservedState` document under `<out>/reviews/`. It **references** the
intended state by JSON pointer and never copies it, so the two cannot drift; `unobserved` carries no
value at all, and an action the reviewer did not see is recorded `unobserved`, never `completed`
(runbook § *Observed state is not intended state*). Nothing there is an input to generation: the run
record gains only a `reviews[]` index entry, a path and some counts (`TakeReviewSummary`).

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
`apps/rust-api/src/film_harness/review.rs` (`decide_take`, `request_repair`),
`apps/rust-api/src/film_harness.rs` (`replace_take`),
`crates/sceneworks-core/src/film_plan.rs` (`AttemptRecord::human_requested`).

Attempt *n* renders at the plan's seed plus `(n − 1) × 1000`
(`film_compile::ATTEMPT_SEED_STRIDE`). The MLX render is deterministic for a seed (two runs of
SH010 at seed 22710, four hours apart, were pixel-identical frame for frame), so a replacement that
kept the plan's seed would re-render the take it had just rejected. The stride is 1000 rather than 1
because a plan's own per-shot seeds are usually spaced by one (runbook § *Replacing a take*).

Every decision is appended to `decisions[]` in the order it was made; replay adds nothing there
(`crates/sceneworks-core/src/film_plan.rs`, `ProductionDecision`, held in `RunRecord::decisions`).

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
(`crates/sceneworks-core/src/film_plan.rs`, `ExportRecord::dropped_audio_layers`). "Why is the music
missing" is answerable from
`run.json` alone.

The export lands as an ordinary project asset. The timeline is created at the nearest aspect ratio
the route admits (`16:9` / `9:16` / `1:1`), so the fixture's 9:5 takes export letterboxed, and the
record states both what the timeline was created at and what the takes actually are
(`TimelineRecord::aspect_ratio` and `TimelineRecord::source_aspect_ratio`).

`trim`, `reorder` and `swap-take` change a saved sequence without re-rendering
(`apps/rust-api/src/film_harness.rs`, `TimelineEdit` applied by `edit_timeline`). Every later
assembly **merges into the saved
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

`AttemptRecord` (`crates/sceneworks-core/src/film_plan.rs`) is what makes a run answerable
afterwards:

| field (`AttemptRecord`) | what it settles |
| --- | --- |
| `idempotencyKey` | which job this attempt is, written before the POST |
| `resolvedModelId` | the checkpoint that rendered it, not the family the plan named |
| `partitionReason` | why that checkpoint and not the other |
| `referenceImageShortEdge` | the **effective** encode edge (the plan's, or 2048); absent on a base-partition attempt, which encodes no reference |
| `loras` | the adapter ids actually sent, in payload order; empty is a recorded state, not an absence |
| `effectiveSteps` | the count that ran: `advanced.steps`, else the recipe's, else the partition's `defaults.steps` (50) |
| `turboSchedulerShift` | the recipe's video sigma shift; absent in the base regime |
| `peakMemoryGb` | the number compared against `limits.maxMemoryGb` |
| `peakMemorySource` | which of `metrics.peakMemoryBytes`, `metrics.peakMemoryPct`, `job.peakGpuMemoryPct` supplied it |

`run.json`'s `model.partitionWeights` records the manifest download row behind **each** partition a
mixed run dispatched on, keyed by catalog model id, because a split family's reference `transformer_ref`
files are a second 18.78 GB download that `model.weights` never named
(`crates/sceneworks-core/src/film_plan.rs`, `RunRecord`'s `model.partitionWeights`, written by
`film_harness::partition_weights`).

## Evaluation discipline

The harness is evaluated, not merely exercised. Three reports ship, with the same rubric, the same
six shots and the same evaluator method, so the numbers are comparable column for column
(`docs/film-harness-evaluation-phase-2.md:8`, `docs/film-harness-evaluation-phase-3.md`):

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

Headline numbers, all three passes:

| pass | configuration | picture /72 | render wall | s/step | peak | wall per accepted second |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (2026-09-14) | no references, `minimax_h3`, 50 steps | **58** | 6 332 s over 9 attempts | ~13 | 14.2 GB | 255 s |
| 2 (2026-09-15/16) | references on every shot, `minimax_h3_ref`, 50 steps, edge 2048 | **66** | 50 669 s | 137–140 | 24.3 GB | 1 634 s |
| 2 | six shots, turbo 4-step, edge 2048 | **64** | 4 722 s | 135–167 | 24.3 GB | **152 s** |
| 2 | SH010 alone at 1344x768, 50 steps | 12/12 | 23 961 s | 457 | 28.8 GB | 4 638 s |
| 2 | SH010 + SH050, reference edge 1024, 50 steps | 24/24 (vs 22/24 at 2048) | 3 376 s | 30–32 | 16.3 GB | 327 s |
| 3 (2026-09-22) | prompt anchoring on, references + turbo 4-step, edge 2048 | **65** (+ 2/2 sound) | 4 876 s | — | 33.7 GB † | 157 s |
| 3 | described-only, **no references**, base `minimax_h3`, turbo 4-step | **49** | 570 s | — | 33.7 GB † | 55 s (570 s / 10.33 s accepted — SH010 + SH020; the other four shots were rejections) |

† one guard-sampled peak for the whole A6a + A6b + locator stack, not a per-cell figure, and a
transient at the refiner→H3 handover; steady-state rendering sat at 15–29 GB
(`docs/film-harness-evaluation-phase-3.md`, § *Memory*).

Source: `docs/film-harness-evaluation-phase-2.md:360`–`:367` and
`docs/film-harness-evaluation-phase-3.md` §4–§5. Phase 1 recommended **revise**; phase 2 recommended
**continue**  (`docs/film-harness-evaluation-phase-2.md:461`); phase 3 records measurements and makes
no recommendation. No report authorizes the next phase, and all three say so at the top.

Phase 3's two picture scores are **not** like-for-like with phase 2's. The 65/72 cell re-renders
phase 2's turbo film with the anchoring sentences added, so every take is a different draw at the
same seed — "no regression, and the plates still dominate", not "+1 from anchoring". The 49/72 cell
has no baseline at all; it characterizes the described-only path.

## Subcommands

Header: the `apps/rust-api/src/bin/film-harness.rs` module header; worker requirements and exit
codes: runbook
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
  hand-authored, so cell (e) is about the recipe, not about a user getting to it. **Closed in phase
  3**, where the unmodified brief planned in one repair round
  (`docs/film-harness-evaluation-phase-3.md` §2);
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
  `(sourceShotId, dependency)` (`crates/sceneworks-core/src/film_plan.rs`, `ReviewFlag`).
  Nothing automatic clears it; `accept-take` retires that shot's flags
  (`decide_take_with_lease`, matching the `accept-take` row above), and the next upstream change
  raises the flag again (`film_harness::flag_dependents`) — which is the point of resolving it.
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
