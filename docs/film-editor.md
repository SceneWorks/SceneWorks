# Film Editor operator guide

The Film workspace turns a script or prose brief into editable shots, renders selected shots into a
saved timeline, and keeps planning, takes, review, sound, and export under explicit operator control.
It uses the same project assets, workers, jobs, and timelines as the rest of SceneWorks.

For the CLI and document schemas, see [Film harness](film-harness.md). For implementation and route
details, see [Film harness UI integration](film-harness-ui-integration.md). This page describes the
operator workflow in the editor.

New to writing a script or testing Film? Start with [Writing your first film script](film-script-writing.md),
which includes copy-and-paste examples, parser limitations, and a staged first-test checklist.

You can also choose **Guides** beside the editor’s **Film | Timeline** switch to read both guides
in a dialog without leaving the project. They ship with the app and are available offline.

## Before you begin

Open or create a project, then open **Editor**. The editor toolbar has a **Film | Timeline** switch.
**Film** is the script-to-shots workspace described here; **Timeline** is the ordinary timeline
editor. Both act on the same project timelines, and a project with no timeline opens on Film with
two start paths: **Start from a script** or **Start with a blank timeline**. Existing film
drafts are listed under them as **Continue a film draft**.

Film mode walks six steps down its left side: **Script**, **References**, **Plan**, **Shots**,
**Review**, and **Sound**. Each step shows its current status, and unsaved edits survive moving
between them. The **Cut** strip along the bottom shows every planned shot in order with its delivery
state, and **Open in Timeline** switches to the film's timeline. In Timeline mode, clips delivered by
a film run carry their shot ID; selecting one shows the shot's beat, framing, planned duration, run, and
attempts in the right rail, and offers **Review this shot**, which opens the Review step on the
film and run that delivered that clip. The ordinary clip actions stay available.

A film draft belongs to the current project. Its script, editable brief, reference pack, shot plan,
sound plan, review plan, and compiled preflight are saved together and versioned.

Check these prerequisites before a real run:

- The target video model and the needed worker lane must show as available. The live Models and
  Workers views are authoritative; a catalog entry by itself does not mean its weights are installed
  or a compatible worker is running.
- Planning needs a planner. New drafts select **Built-in prompt refiner (default)**, catalog ID
  `prompt_refine_anubis_8b` (`TheDrummer/Anubis-Mini-8B-v1`). It is a local model with
  `autoDownload: false`, so install it through the normal Models flow if it is missing.
- **Native Qwen3.6-27B (optional)** is catalog ID `film_planner_qwen3_6_27b`
  (`Qwen/Qwen3.6-27B`). It is never downloaded or selected automatically. Its catalog estimate is
  57 GB and its declared minimum memory is 96 GB. Selecting it exposes an explicit
  **Install Qwen3.6-27B (large download)** action when it is absent.
- MiniMax-H3 rendering may require accepting the model's license notice. Use the install state,
  platform eligibility, memory admission, and limits shown by the current catalog rather than
  assuming another machine matches a previous evaluation.

The optional Qwen planner changes only how the shot plan is drafted. The **Video model** remains a
separate choice, and the planning operation records both identities.

## Create and edit a film draft

1. Choose **New film draft**. On the **Script** step, give it a title and paste prose or
   screenplay text into **Original prose or screenplay**.
2. Choose **Extract editable beats and dialogue**. SceneWorks performs a deterministic parse into a
   synopsis, visual direction, beats, and dialogue lines. Review and edit every field. Extraction is
   a starting point, not an approval step.
3. On the **Shots** step, add, remove, reorder, or edit shots under **Shots and render controls**. Each shot has a stable
   ID, beat, framing, prompt, **Audio**, intended start and end state, duration, conditioning mode,
   continuity roles, optional dependencies, and optional dialogue placement.

   **Audio is required on every shot.** MiniMax-H3 generates its soundtrack from the same prompt it
   renders the picture from, so anything the prompt leaves unsaid the model invents. Say what the
   shot sounds like — diegetic sound, ambience, music or "no music". Stating silence is a valid
   answer: write "No audio. Silence." and that is exactly what the compiler dispatches; the text is
   never pattern-matched. Leaving it blank is not, and surfaces as a finding naming the shot until
   you fill it in. Do not begin the value with "Audio:" — the compiler writes that label itself.

   **Speech.** When the shot places a dialogue clip, the harness speaks that line itself, so leave
   the words out of Audio. Otherwise a spoken line — who speaks, the words, and the delivery —
   belongs in Audio, because the model invents any speech the prompt does not describe.
4. Choose a render regime. **Turbo (recommended)** resolves the installed adapter recipe for the
   current model, output canvas, and reference-conditioned partitions and shows its concrete adapter
   IDs and effective step count. **Full quality** explicitly clears accelerator and step overrides
   and uses the model's declared default steps. **Custom** preserves the adapters and steps you enter
   under advanced controls. When Turbo is unavailable, the editor states whether the model or a
   compatible adapter is missing, partition coverage is incomplete, or the edited shot resolutions
   require conflicting recipes; a new draft then starts in Full quality instead of naming an
   adapter the host cannot run.
5. Set the video model, tier, frame rate, resolution, reference image short edge, and finite run,
   shot, attempt, memory, and planner budgets. Model-specific menus and preflight findings take
   precedence over values copied from another project.
6. Use **Save draft** whenever you want the current document state to become durable.

The render regime is saved on the draft. Planning cannot silently replace recommended Turbo with a
full-step plan when a planner omits adapters: the server pins the saved recipe outside the planner's
answer. Drafts created before render regimes existed keep their exact adapter and step fields as a
legacy custom selection until the operator chooses Turbo or Full quality.

You can author a film without a planner by editing the initial manual shot. You can also import or
export production plans, reference packs, and compiled plans. Imported documents are validated at
preflight; importing a file does not make it trusted or runnable.

## References are optional

A script-only film can leave the reference pack empty and use `text_to_video` shots. There is no
requirement to create placeholder references.

To use references:

1. On the **References** step, choose an existing project image or upload an image.
2. Give it a stable role name, choose its kind, describe it, and mark it **Approved** only after a
   person has checked it.
3. Bind approved character, prop, or location roles to a shot. Bound role order is preserved in the
   compiled request. Style and plate entries can remain descriptive references but are not
   `reference_to_video` subject bindings.

A role's **description** is not a note to yourself. The compiler repeats it word for word into
every shot that names the role, so write it as a complete sentence about the subject, naming the
subject itself: "The courier: blue jacket, carries the parcel." works; "blue jacket" alone reads as
a fragment in the dispatched prompt. Descriptions are editable on every row. `<`, `>`, and control
characters are refused, because text that forges a `<Picture 3>` marker would reach the engine
looking like one.

### Roles with no image

A role does not need a picture. Fill in **Role name**, **Kind**, and **Description**, clear the
selected project image and leave the locator empty, then choose **Add described role**: the button
stays disabled until the form describes a role with no image, and the role is then stored with no
image at all, its description the whole of it. This is how you hold a subject steady across a film you have no photograph of — the
compiler inserts the same words in every shot that lists the role under **Continuity roles**,
rather than letting each shot reword it.

A described-only role is listed with *no image — described in text* in place of a locator, and its
description, kind, approval, and removal work like any other row's. Leave the description blank and
the role is refused by name: a role with no image and no words says nothing and shows nothing.

Because it supplies no picture, a described-only role can never be a first-frame, last-frame, or
ordered reference binding — those controls and the reference-role suggestions offer image-backed
roles only. **Continuity roles** offers every approved role, described or not.

### Two roles on one image

One photograph can hold two subjects. Add the same image a second time under a second role name and
the pack ends up with two roles on one file, which SceneWorks supplies to the engine once, under a
single `<Picture N>` that both roles are bound to, and imports as one project asset.

Roles that share a file each need a **Locator** saying which subject in that image they name. The
panel marks the field required on the add form and on both rows as soon as an image is reused, and
names the other roles using it. A locator is a noun phrase, with its article, that completes "The
courier is …" — write "the woman on the left", not "woman on the left", because the sentence is
assembled verbatim and no article is supplied for you. Two roles sharing a file may not declare the
same locator, and a role with no image may not declare one at all.

If either locator is missing or the two are identical, the add is refused and the reason appears
under the field. Fill in the locator on the existing row first, then add the second role.

A continuity role whose photograph a shot is **already** supplying for some other role is named by
that picture rather than described on its own: it gets the binding sentence with its locator ("The
recipient is the man on the right in `<Picture 1>`.") and no separate identity sentence, since the
image is being sent either way and its description is already in that sentence. Nothing about the
picture order or the images dispatched changes.

SceneWorks copies selected image bytes into the film draft's project-owned reference pack. A run
pins its own staged copy, so later project changes do not silently change an in-progress planning or
render operation. Removing or renaming a reference updates shot bindings; a reference-conditioned
shot with no remaining roles returns to `text_to_video`.

Reference metadata and image pixels are separate for external planning. See the disclosure below.

## Choose a planner

### Built-in prompt refiner

This is the default for every new draft. It uses local Anubis weights through the existing
`prompt_refine` worker path. Qwen is not required. If the required local weights or worker are
missing, planning stops with an actionable error; SceneWorks does not switch providers.
Advanced planning settings expose the same per-call local planner job timeout as the film harness
`--llm-timeout-seconds` option (1200 seconds by default), along with the bounded repair count.

### Native Qwen3.6-27B

Select **Native Qwen3.6-27B (optional)** only when you intend to use and, if necessary, install the
large local model. Thinking is a separate planner setting and is recorded separately from the
candidate text. Selecting native Qwen does not change the target video model and does not create a
remote request. The local planner job timeout also applies to Qwen and is retained by a recovered
planning operation.

### Saved OpenAI-compatible connection

External planning is opt-in per draft. There is no automatic external fallback.

When **Run model-specific prompt refinement when compiling shots** is enabled, film planning
still uses the selected external endpoint, while each shot is refined locally with the target
video model’s guide. The local refiner must be available; its per-call timeout is separate from
the connection timeout, and its execution identity is retained separately. A local-refiner failure
does not switch the film-planning provider.

1. Select **Saved OpenAI-compatible connection**.
2. Create or choose a connection. Enter a label and an HTTP(S) base URL ending at the provider's
   OpenAI-compatible API root, such as `https://provider.example/v1`.
3. Add an optional bearer credential. Connection metadata is saved as a non-secret setting. In the
   desktop app the credential stays in the operating-system secret facility and reaches the API
   sidecar through credential IPC. With the standalone API it stays in the configured server
   credential store or explicitly supplied credential environment. It is not written into the film
   draft, project files, exported media, provenance, logs, or browser storage. Saving, rotating, or
   removing a desktop credential takes effect for Test, model discovery, and planning without an
   app or API restart.
4. If the endpoint implements `GET /models`, leave **Endpoint supports model listing** enabled and
   choose **Test and list models**. This proves model listing only; the first planning run validates
   Chat Completions. If listing is unavailable, turn it off and type the exact model ID in
   **Planner model ID**.
5. Set the timeout and maximum output tokens. Both bounds are enforced. Provider errors and
   cancellation end the operation; SceneWorks does not silently retry through a native planner.

The connection credential is sent only as authentication to the configured endpoint. Planning
prompt data includes the original script, edited brief, beats and dialogue, reference role names and
descriptions, and the target video model capability envelope. Local paths and unrelated project
assets are not prompt data.

Approved reference image bytes remain local by default. They are included only when all three are
true:

- the operator enables **Send approved reference image pixels to this image-capable endpoint**;
- the saved connection is marked **Endpoint supports image input**; and
- the image is an approved reference staged for this operation.

Changing the draft field alone cannot bypass the connection capability check. Treat a connection's
image-capable flag as an operator assertion about that endpoint, not automatic capability detection.

## Generate and apply a candidate plan

On the **Plan** step, choose **Generate candidate plan** after saving the script and brief. Planning runs as a durable,
bounded operation and never starts video rendering.

The operation shows status, stage, progress, planner identity, target video model, execution
identity, and validation findings. **Cancel planning** requests a bounded stop. When a candidate is
ready, inspect every shot before choosing **Replace current edited plan with this candidate**. This
explicit action replaces the current edited shot plan; generating or regenerating alone never does.

If the API restarts during local planning, SceneWorks reconciles the durable local job IDs and can
adopt the known work. An OpenAI-compatible Chat Completions request has no durable remote job ID that
SceneWorks can recover. After a restart it is marked interrupted and retryable with its draft,
findings, and known artifacts preserved. SceneWorks does not resend a potentially paid request or
fall back to a local planner. Retry is an operator action.

## Preflight before rendering

Choose the shots to render, then choose **Run preflight**. Preflight creates no video job. It checks
the saved draft, selected shots, reference bindings, model and worker availability, install state,
license acknowledgement, geometry and duration menus, adapters, and declared budgets.

Read both parts of the result:

- findings identify the shot and field that must change, in plain words — the internal field path is
  used to route a finding to the panel that owns it, never shown to you. A finding about the pack
  appears under **References**, except a sound-and-dialogue finding, which appears under **Sound and
  dialogue**; every pack finding has one of those two owners, so none is reported in a place you
  cannot act on it. And
- **Effective requests** show the resolved model partition, output geometry, conditioning roles,
  reference edge, adapters, effective steps, seed, prompt source, and any prompt-refinement
  provenance.

**Added by the compiler** lists, per request, the sentences SceneWorks wrote into the dispatched
prompt around your authored text, labelled by what each one is: reference bindings, continuity
descriptions, the audio sentence, and the no-speech sentence. They are written after any refinement
rewrite, so a refiner cannot paraphrase a `<Picture N>` the engine labels or soften a stated
silence. You change them by editing their sources — a reference's description or locator, or the
shot's Audio field — not by editing the prompt.

**What this view does not show.** It does not show the fully composed prompt that is dispatched, and
it does not show the refiner's rewritten text. The **Prompt** row says only where the prompt came
from — `authored` or `refined` — and **Added by the compiler** says what was added around it. The
composed prompt itself is in the compiled document: export it with **Export compiled plan**, or read
`compiled.json` from a CLI run, where each request carries its `prompt`, its `authoredPrompt` and
its `insertedText`.

**Editing the references invalidates a compiled plan.** The reference pack decides the sentences
above, so changing a description or a locator changes what every shot repeating it would dispatch. A
compiled document is tied to the pack it was compiled against, and after a pack edit preflight
reports that the reference pack changed since those requests were compiled. Clear it with **Use
authored prompts**, which drops the stored compiled plan and returns the shots to the prompts you
wrote, or re-plan to compile fresh requests against the edited pack. A comment- or whitespace-only
difference does not do this: what the compiled document is keyed on is the pack's content, not its
formatting.

Fix every blocking finding and run preflight again. **Render selected shots** also runs preflight and
does not dispatch when the saved selection is invalid.

## Render into the editable timeline

Rendering creates a durable run and dispatches one bounded attempt at a time. A completed shot is
delivered into the run's ordinary project timeline as soon as it is available; later shots can keep
rendering while you inspect and edit the cut. Export does not start automatically.

Each delivered picture item carries run, shot, attempt, job, model, seed, and asset provenance. New
deliveries are merged into the current saved timeline instead of rebuilding it from the original
plan:

- user trims, ordering, speed, fit, volume, and unrelated timeline items are preserved;
- film-owned audio is delivered once, after which its placement, trim, gain, mute state, fades, and
  deletion belong to the editor;
- deleting a delivered film item records a durable tombstone, so replay or a later take does not
  resurrect it; restoring it, including with Undo, clears that tombstone; and
- concurrent saves use timeline revisions. A stale save is rejected with a revision conflict so
  the operator can reload or deliberately reapply the edit instead of overwriting newer work.

When a replacement take is long enough, it inherits the existing item's timing and edit state. If a
replacement is too short for the current trim, SceneWorks keeps the saved cut unchanged and records
a trim conflict. Resolve it explicitly by clamping the trim to the new take, resetting the new take
to its beginning, or keeping the current take. No replacement choice alters unrelated shots or
audio.

The **Operations** area sits under the step list and remains visible on every step while work is
active. **Cancel** stops further dispatch,
requests cancellation of the in-flight job, and preserves completed takes and spent attempts.
**Resume** reconciles the saved record and adopts known jobs instead of starting the run over. Only
one controller may own a run at a time; a competing UI, API, or CLI mutation is refused.

## Review and take history

Review is assistive, not quality assurance. A local vision model can miss real faults and flag
correct takes. Its observations never approve, reject, condition, or regenerate a shot. A human
decision recorded through the controller is required.

On the **Review** step:

1. Expand **Review questions, frames, and limits** to edit the future-run review plan. Questions are
   scoped per shot and declare intended state, expected and contradicting answers, sampled frames,
   whether an unobserved answer is actionable, and whether to compare across a cut. The plan also
   bounds seconds, frames, questions, answer time, answer tokens, memory, and the uncertainty
   threshold.
2. Choose **Analyze selected takes** to start bounded assistive review. Review frames use a separate
   review timeline; they do not rewrite the saved cut.
3. Inspect the video, asset/job/model/seed provenance, findings, confidence, and any unobserved or
   uncertain result. Use **Open review frames** when you need to inspect what the reviewer saw.
4. Use **Use in saved cut** to select a retained take. Accept or reject is enabled only when the
   generation selection and saved-cut selection are aligned and no trim conflict is pending.
5. **Reject** keeps the take, asset, job, findings, and decision history. It flags declared dependent
   shots for human review and does not render anything.
6. **Render one replacement** authorizes one new attempt. **Repair from findings** also authorizes
   one new attempt and folds the latest actionable findings into its reason. Neither action loops,
   auto-accepts, or automatically regenerates dependent shots.

The take strip keeps prior attempts, rejections, the current saved-cut selection, and decision
provenance. A finding about an older attempt does not become a finding about its replacement.

## Sound and dialogue

Open the **Sound** step to configure the film before or during shot planning:

- generated picture audio defaults to **Mute** so it does not double with placed dialogue;
- generated dialogue supports text, voice, and the listed speech models;
- prerecorded project audio can be copied into the draft as dialogue, ambience, music, or sound
  effect roles;
- dialogue placement supports timeline offset, source in, duration, gain, and fades; and
- ambience, music, sound-effect beds, and the dialogue bus expose placement, gain, and mute controls.

Prerecorded source assets can be previewed in the sound panel. Generated dialogue is previewable on
the timeline after synthesis. The ordinary timeline remains the authority for the saved cut, so
use its transport and meters to hear the assembled result before export.

## Export the current saved cut

Rendering shots does not export a movie. Save the timeline, review the current cut, then choose
**Export current cut**. SceneWorks saves the active run timeline first when needed and starts one
explicit `timeline_export` job from that saved revision.

The export status records its timeline revision, output asset, error, and any dropped audio layers
with reasons. A later timeline change marks the previous export **stale**; it does not start a new
export. Export again only after you decide the new saved cut is ready.

## Limits and evidence

- SceneWorks preserves provenance and applies declared limits, but it does not guarantee visual
  identity, action completion, or continuity. Reference conditioning and assistive review reduce
  uncertainty; they do not replace watching the cut.
- Runtime and memory vary by model, tier, geometry, duration, reference resolution, worker backend,
  host, and cold-load state. The figures in
  [the phase-2 evaluation](film-harness-evaluation-phase-2.md) are measurements from the named Mac
  and test cells only. Do not present them as cross-platform requirements or expected performance.
- The live catalog, preflight, and worker admission response decide whether a request can run on the
  current host. Do not infer support from a model family name.
- OpenAI-compatible behavior varies by endpoint. Model listing and image input are separately
  declared capabilities; successful model listing does not prove Chat Completions or multimodal
  support.
- A run record, its project assets, and its locator's immutable authoring snapshot are the durable
  audit trail. New runs retain the complete script, brief, structured authoring, planning choices,
  and draft revision they were created from. Legacy run locators report that this snapshot is
  absent rather than reconstructing one from render inputs.
- Queue visibility is not the same as run ownership: clearing a completed job from the queue does
  not remove its run provenance.
