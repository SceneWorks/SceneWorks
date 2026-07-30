# The workflow share envelope

A PNG that SceneWorks generates carries a small block of JSON describing the recipe that made it,
so the recipe travels with the image. The local sidecar (`<media>.sceneworks.json`) does not survive
a copy-paste into a chat window; this does, because it is inside the file.

Two things read it back today: importing such an image records the envelope on the asset, and
`POST /api/v1/workflows/inspect` reads one out of an uploaded file without importing anything. The
studio surfaces that offer it as "use this recipe" are still being built (sc-15951 / sc-15952).

This document is the contract. It exists for two people:

- **Someone about to share an image**, who needs an exact list of what is in the file rather than a
  reassuring paragraph. That is [What travels](#what-travels) and [What does not
  travel](#what-does-not-travel).
- **A contributor adding an advanced setting**, who has just been failed by a coverage lint and
  needs to know what it wants. That is [Adding a new advanced
  setting](#adding-a-new-advanced-setting).

> ## Your prompt is in the file, as you wrote it
>
> These six fields are recorded as authored, and are deliberately **exempt from the filesystem-path
> guard** that drops every other field that looks like a location. Silently mangling someone's
> prompt because it mentions a directory would be worse than the leak it prevents — you wrote it,
> and you can read it back before you share. The only things removed from them are invisible
> formatting characters and anything past the 16 KiB prose ceiling.
>
> <!-- PINNED: prose-fields -->
>
> | Field | What it is |
> | --- | --- |
> | `prompt` | The prompt. |
> | `negativePrompt` | The negative prompt. |
> | `advanced.stylePrompt` | The pre-style prompt, before a catalog style was composed onto it. |
> | `advanced.systemMessage` | The interleaved-document system prompt, when it was edited away from the default. |
> | `advanced.structuredPrompt.intent` | The authored intent line of a structured-caption recipe. |
> | `advanced.structuredPrompt.runtimePrompt` | The serialized structured prompt the model actually received. |
>
> <!-- END PINNED: prose-fields -->
>
> So a prompt reading `in the style of D:\Clients\Acme\brief-final.png` puts that path — and the
> client's name — into every copy of the image. Nothing sanitizes it, on purpose. If you would not
> paste the prompt into the same chat window, do not share the image with the workflow in it.
>
> **Embedding is ON by default.** The switch is the `embedWorkflowInImages` UI preference; see
> [The setting](#the-setting).

## The envelope

The block is a single JSON object written into the PNG as a compressed `iTXt` text chunk. It is
identified by a marker key rather than by position, so a reader can tell our block from
Automatic1111's `parameters` or ComfyUI's `prompt` / `workflow` without sniffing the body.

<!-- PINNED: identity -->

| Constant | Value | Meaning |
| --- | --- | --- |
| `WORKFLOW_CHUNK_KEYWORD` | `sceneworks:workflow` | The PNG `iTXt` keyword the JSON lives under. |
| `WORKFLOW_SHARE_MARKER_KEY` | `sceneworksWorkflow` | The JSON key whose presence makes the blob ours. |
| `WORKFLOW_KIND_IMAGE` | `image` | The marker's value for the image lane. |
| `WORKFLOW_SHARE_SCHEMA_VERSION` | `1` | The contract version this build writes and reads. |
| `PRODUCER_NAME` | `SceneWorks` | Names the software. Never the installation. |
| `PRODUCER_URL` | `https://github.com/SceneWorks/SceneWorks` | So a file that reaches a stranger is self-identifying. |

<!-- END PINNED: identity -->

`producer.version` is the released `MAJOR.MINOR.PATCH` of the build that wrote the file, taken from
the workspace version at compile time. It is deliberately not derived from git, the environment, the
build host or the build path — `0.8.1-dirty-alice` would walk straight into every shared image — and
a test asserts it is strict semver.

### Two versions, and only one of them is parsed

`schemaVersion` is the contract version and is **the only field the parser branches on**.
`producer.version` is recorded so a bug report is actionable and is never interpreted.

| The file says | This build does |
| --- | --- |
| `schemaVersion` newer than this build's | Refuses the whole envelope and names both versions, so the user is told to update rather than shown a parse error. |
| `schemaVersion` equal to or older than this build's | Reads it with today's field set. Fields this build does not know are dropped; fields it knows and the file lacks take their defaults. |
| `schemaVersion` absent or `null` | Refuses the envelope. A version-less blob is not readable safely. |
| `schemaVersion` present but not a whole number | Refuses the envelope as malformed, naming the field. |

Adding a field does not need a version bump: an older reader drops what it does not recognize.
Bump only when a field changes meaning or disappears — and a bump means every older build stops
reading the file at all, rather than reading it partially.

## What travels

Every field the envelope can carry, and nothing else. There is no passthrough bucket for unknown
keys in either direction: a key this table does not name is dropped on write **and** on read.

<!-- PINNED: envelope-fields -->

| Field | What it is |
| --- | --- |
| `sceneworksWorkflow` | The marker. Always `image` for this lane. |
| `schemaVersion` | The contract version. |
| `producer.name` | `SceneWorks`. |
| `producer.url` | The project's repository URL. |
| `producer.version` | The released version of the build that wrote the file. |
| `mode` | The generation mode (`text_to_image`, `edit_image`, `character_image`, …). |
| `model` | The model catalog **slug** (`z_image_turbo`), never a weights location. |
| `prompt` | The prompt, verbatim. See the callout above. |
| `negativePrompt` | The negative prompt, verbatim. |
| `seed` | The seed of **this** image. The rest of the batch's seeds do not travel. |
| `width` | Requested output width. |
| `height` | Requested output height. |
| `count` | How many images the run requested — so the file says it was one of a batch of N. |
| `stylePreset` | The style preset label the run used. |
| `styleId` | The catalog style id the user picked. |
| `fitMode` | How the source image was fitted (`crop`, `contain`, …). |
| `upscale.enabled` | Present only when the run enabled an upscale pass. |
| `upscale.factor` | The upscale factor. |
| `upscale.engine` | The upscale engine label. |
| `upscale.softness` | The upscale softness. |
| `loras[].name` | A LoRA's display name, as the user sees it in the catalog. |
| `loras[].weight` | Its weight. |
| `loras[].repo` | Its Hugging Face `owner/name`, when the catalog entry resolved to one. Never a path, never a local id. |
| `inputs[].kind` | The kind of input image the recipe needs. See [Input images](#input-images-travel-by-shape-not-by-id). |
| `inputs[].count` | How many of that kind. |
| `inputs[].controlMode` | For a control input, the conditioning it feeds (`canny`, `depth`, …). |
| `advanced` | The allow-listed subset of the request's advanced settings. See [Advanced settings](#advanced-settings). |
| `omitted` | Which collections were declared but not recorded. See [The `omitted` marker](#the-omitted-marker). |

<!-- END PINNED: envelope-fields -->

Optional fields are omitted entirely when they have nothing to say, so a small recipe is a small
envelope.

## What does not travel

Not a promise about categories — a consequence of the field list above being closed. Anything not
in that table is not in the file. Named explicitly because these are the ones people ask about:

- **Identity of the machine or the person.** No user name, no host name, no account, no
  installation id. `producer` names the *software*, never the install.
- **Filesystem paths, anywhere.** Every non-prose string is dropped if it looks like a location —
  absolute or relative, Windows or POSIX, `~` expansions, UNC shares, `file://`, percent-encoded
  forms and `..` traversals. A value that trips the check is dropped rather than trimmed. Two
  honest limits on that: it is a **shape** test and not knowledge of your disk, so a bare name with
  no separators in it (`acme-brief`) is not a location and travels; and it does **not** apply to
  the six prose fields, which is the callout at the top of this document.
- **Project, job and asset identity.** `projectId`, `projectName`, `jobId`, `assetId`,
  `generationSetId`, `characterId`, `characterLookId`.
- **Timestamps.** The envelope has no time field.
- **The rest of the batch.** `seeds` does not travel; `seed` is the one that rendered this file.
- **This machine's hardware budget.** Quant tier, INT8-ConvRot selection, flash-attention, the
  requested GPU. The receiving install picks its own. See the withheld table below.
- **Local ids that resolve to nothing elsewhere.** Recipe preset ids, control-image asset ids,
  trained-overlay ids and their resolved weights paths, Key Point Library collection ids.
- **Pose library ids.** A pose selection travels as coordinate arrays (`keypoints`, `hands`,
  `face`), because those are what the worker renders. The library ids that named them do not.

The image pixels are the image pixels. Nothing here redacts, watermarks, or alters them.

## Advanced settings

`advanced` is an untyped map that grows whenever someone adds a knob to a job builder. A deny-list
would leak every future field by default, so the contract is an **allow-list**: every key a
registered builder can emit is classified, and anything unclassified is dropped.

The line between the two lists is *what to make* versus *what this machine can afford to make it
with*. Sampler, steps, guidance and decoder choice describe the intended output and travel; quant
tier, attention kernel and the requested GPU describe this install's budget and do not.

### Shared

These reach the file. `Shape` is how the value is reduced — anything that does not match its shape
is dropped rather than passed through, which is what stops an object (and a path inside it) being
smuggled under a scalar key.

<!-- PINNED: advanced-shared -->

| Key | Shape | What it is |
| --- | --- | --- |
| `resolution` | `Scalar` | The output geometry label the studio control was set to. |
| `structuredPrompt` | `StructuredPrompt` | A structured-caption recipe, reduced to its scalar fields. The free-form `caption` object is dropped. |
| `sampler` | `Scalar` | Sampler choice. |
| `scheduler` | `Scalar` | Scheduler choice. |
| `schedulerShift` | `Scalar` | Time-shift (mu) for the curated schedule. |
| `steps` | `Scalar` | Step-count override. |
| `guidanceScale` | `Scalar` | Guidance override. |
| `guidanceMethod` | `Scalar` | Guidance method (CFG / CFG++). |
| `enhancePrompt` | `Scalar` | Caption-upsampling opt-in — it changes the prompt the model sees. |
| `usePid` | `Scalar` | PiD decoder opt-in. Changes the produced image, and is its non-commercial marker. |
| `pidTarget` | `Scalar` | PiD output tier (2k / 4k). |
| `ipAdapterScale` | `Scalar` | Reference strength. |
| `controlnetConditioningScale` | `Scalar` | Identity-structure strength (InstantID). |
| `trueCfgScale` | `Scalar` | Variation strength. |
| `strength` | `Scalar` | img2img strength. |
| `viewAngle` | `Scalar` | Head-angle label. |
| `textStyleGain` | `Scalar` | Krea text-style tap-reweight gain. |
| `poses` | `Poses` | Pose selection, reduced to the `keypoints` / `hands` / `face` coordinate arrays. |
| `faceRestore` | `Scalar` | Face-restoration opt-in. |
| `controlMode` | `Scalar` | Control type (canny / depth / …). The control *image* rides as an input shape instead. |
| `controlScale` | `Scalar` | Control-lock strength. |
| `styleId` | `Scalar` | The catalog style picked. |
| `stylePrompt` | `Scalar` | **Prose.** The raw pre-style prompt. Path-exempt — see the callout. |
| `cnScale` | `Scalar` | Tile-ControlNet strength for the Detail pass. |
| `angleSet` | `Scalar` | Turnaround request. It makes the worker emit one image per view angle, so it decides what is made. |
| `systemMessage` | `Scalar` | **Prose.** The interleave system prompt. Path-exempt — see the callout. |
| `imageGuidanceScale` | `Scalar` | Reference-guidance strength for an interleaved document. |
| `phases` | `Phases` | Multi-phase denoise schedule, reduced to `{ steps, guidance, loras: [{ index, weight }] }`. LoRA references are indices into this request's own list, not ids. |

<!-- END PINNED: advanced-shared -->

### Withheld

Classified and deliberately dropped. A key here is a decision someone wrote down, not an oversight.

<!-- PINNED: advanced-withheld -->

| Key | Why it is withheld |
| --- | --- |
| `keypointCollectionId` | A local Key Point Library collection id. Resolves to nothing on another install. |
| `flashAttn` | A backend kernel toggle — what this install's attention path can do. |
| `mlxQuantize` | Quant tier: a memory accommodation for this machine. |
| `mlxQuantizeExplicit` | Marks a deliberate tier pick on this install. |
| `convRot` | The INT8-ConvRot tier selector. |
| `quantTier` | Install-specific and fingerprinting. |
| `controlImage` | A local asset id. The *need* for a control image rides in `inputs` instead. |
| `controlWeights` | A trained-overlay id plus the resolved weights path stamped onto it. |
| `recipePresetId` | A local preset id stamped by the API. |
| `presetMissingLoras` | Local LoRA ids the API could not resolve on this install. |

<!-- END PINNED: advanced-withheld -->

## Input images travel by shape, not by id

A recipe that started from an image records **that it needs one, and of what kind** — never the
local asset id, and never the image itself as base64. An id from someone else's library resolves to
nothing here, and would be a leak for no benefit; the bytes would make every shared image several
times larger.

<!-- PINNED: input-kinds -->

| `inputs[].kind` | The image it stands for |
| --- | --- |
| `source` | The image an edit starts from. |
| `reference` | Identity or style reference image(s). One entry with a `count`, not N entries. |
| `mask` | An inpaint mask. |
| `control` | A pre-made control map, with the conditioning it feeds in `controlMode`. |

<!-- END PINNED: input-kinds -->

The consequence is deliberate and worth stating plainly: a shared edit or character image **cannot
replay on its own**. The receiving user has to supply the input images. The resolution report calls
that out per input rather than pretending the recipe is runnable.

## The `omitted` marker

A collection that could not be recorded whole is **dropped whole**, never truncated: "the first 5 of
these 8,000 LoRAs" is not the recipe that made the image, and a reader cannot tell a plausible
subset from the real thing.

But a reader cannot tell an absence either — an envelope whose LoRAs were dropped and one that
genuinely had none serialize identically. So a drop is recorded. `omitted` is a closed vocabulary
of field names, sorted, with unknown entries stripped on read:

<!-- PINNED: omitted-fields -->

| `omitted[]` entry | What was dropped |
| --- | --- |
| `loras` | The LoRA list. |
| `inputs` | The input-image list. |
| `advanced.poses` | The pose selection. |
| `advanced.phases` | The multi-phase schedule. |
| `advanced.phases[].loras` | A phase's own LoRA schedule. |

<!-- END PINNED: omitted-fields -->

The marker is emitted in **both** directions. On the write side it is the difference between a
silently lost 70-pose selection and a visible one, which is the point: our own writer can hit these
caps, and a recipe that says "no LoRAs" when it had five is exactly the silent loss this contract
exists to prevent.

## Ceilings

Two kinds of bound. Per-collection caps, each inherited from the validator that already limits the
thing — where one exists; `MAX_SHARE_POSES` is the one that has no upstream validator to inherit
and is derived from the size of the shipped pose library instead. And one ceiling on the serialized
envelope, checked after every per-field rule has run. The second is what actually composes:
per-field bounds did not, and each new measurement found a new way to spend what they left.

<!-- PINNED: ceilings -->

| Constant | Value | What it bounds | What happens at the limit |
| --- | --- | --- | --- |
| `WORKFLOW_SHARE_MAX_BYTES` | 163,840 | The whole serialized envelope, in bytes. | The **entire** envelope is refused — no chunk is written, and a file over it reads as an error rather than as a partial recipe. |
| `PROSE_MAX_BYTES` | 16,384 | Each authored prose field, in bytes. | Truncated at a whole character. Prose still means what it said after its tail is cut. |
| `LABEL_MAX_CHARS` | 200 | Each non-prose label (model slug, style id, LoRA name, producer block), in characters. | **Dropped**, not truncated — a slug's spelling is its identity. |
| `MAX_SHARE_LORAS` | 5 | Entries in `loras`. | The list is dropped whole and `omitted` gains `loras`. |
| `MAX_SHARE_INPUTS` | 4 | Entries in `inputs` — one per kind, and the kinds are closed. | The list is dropped whole and `omitted` gains `inputs`. |
| `MAX_SHARE_PHASES` | 8 | Entries in `advanced.phases`. | The key is dropped and `omitted` gains `advanced.phases`. |
| `MAX_SHARE_POSES` | 64 | Entries in `advanced.poses`. | The key is dropped and `omitted` gains `advanced.poses`. |
| `MAX_SHARE_POSE_SLOTS` | 6,144 | Coordinate slots across the whole `advanced.poses` array — a number, or a `null` standing in for one. | The key is dropped and `omitted` gains `advanced.poses`. |
| `MAX_WORKFLOW_TEXT_BYTES` | 1,048,576 | The **decompressed** chunk text, on read. | Decompression stops and the file is refused, so a zip bomb costs a megabyte rather than its claimed size. |
| `MAX_METADATA_BYTES` | 8,388,608 | What the PNG decoder may buffer from an untrusted file's ancillary chunks, cumulatively. | The read is refused, whatever length a chunk header *claims*. |

<!-- END PINNED: ceilings -->

No request this app accepted can exceed a per-collection cap, because each cap **is** the limit the
generation path already enforces: five LoRAs is the hard per-job total, eight phases is the
multi-phase validator's own number. A run that declared more could not have happened here. The pose
cap is the exception — nothing upstream clamps how many poses a user may select, so this one can
fire on our own write side, which is why `omitted` is emitted in that direction too.

## The trust boundary on import

An image being read arrived from a stranger. Extending the reader means keeping these true.

- **One parse path.** Everything goes through the same parser, which runs the same reducer the
  writer runs. There is deliberately no second, laxer path; a structural test fails the build if one
  appears.
- **Reduction is at value granularity, not key granularity.** Dropping the keys we do not declare
  says nothing about the strings under the keys we do. On the way in, every label is re-checked for
  paths and control characters, `loras[].repo` is re-validated as `owner/name` (it is joined into a
  cache directory name, so a traversal string there is the sharpest edge in the contract),
  `inputs[].kind` is checked against the closed vocabulary, `omitted` is filtered to the closed
  vocabulary, and the producer block is bounded — a URL that is not `http(s)` or a version that is
  not strict semver is reduced to empty rather than echoed back as provenance.
- **Prose is bounded and stripped, not trusted.** Incoming prose is truncated to the prose ceiling,
  and control characters plus Unicode `Cf` format characters (bidi overrides, zero-width joiners,
  tag characters) are removed so a rendered prompt cannot claim to say something other than what is
  stored. Newlines and tabs survive. This is narrower than "prose is safe": characters that render
  blank but are not `Cf` — the Hangul and Braille fillers — are not currently stripped.
- **Every failure degrades to "no workflow".** On import: a hostile or malformed chunk costs the
  user the recipe field and nothing else, and their image still imports. A PNG with no chunk is the
  normal case for every image in the world and is never an error. The read-only inspect endpoint is
  the one surface that reports a typed error instead, because nothing is being imported there and
  the user asked specifically what the file contains.
- **`extra.importedWorkflow` is ours or absent.** Import clears that key unconditionally before
  writing its own, so a caller-supplied `provenance.importedWorkflow` cannot masquerade as an
  envelope this reader sanitized.
- **The resolution report is computed, never stored.** What a machine can run changes the moment the
  user installs a model; the envelope is a fact about the file and does not.

## Adding a new advanced setting

If you added a knob to a job builder and a Rust test failed naming your key, this is what it wants.

The failure is one of the coverage lints in `crates/sceneworks-core/tests/workflow_share.rs`:

<!-- PINNED: lints -->

| Test | Fails when |
| --- | --- |
| `every_registered_builder_has_its_advanced_keys_classified` | A registered builder can emit a key `ADVANCED_KEY_RULES` does not classify — or classifies a key that builder no longer emits. |
| `every_advanced_builder_in_the_web_app_is_accounted_for` | A new `advanced`-map builder appears anywhere in `apps/web/src` and is in neither registry. |
| `every_source_tag_names_exactly_one_registered_builder` | A rule is tagged to a builder that is not registered, or two builders share a tag. |
| `every_deferred_builder_names_the_story_that_owns_it` | A deferred builder's reason does not name the story that will classify it, or does not justify a permanent exemption. |

<!-- END PINNED: lints -->

**It is not an obstacle. It is the guardrail.** An unclassified key is dropped silently from every
shared image the lane writes, and in practice that shows up as silent loss more often than as a
leak. Two real ones, both found when the lint was generalized past its first builder: `cnScale` was
already vanishing from every shared Detail-pass image, and `angleSet` — the knob that makes the
worker emit one image per view angle — was vanishing, so a shared angle-set image replayed as a
single image.

### Classify the key

Add a row to `ADVANCED_KEY_RULES` in `crates/sceneworks-core/src/workflow_share.rs`. The question is
whether the key describes **what to make** or **what this machine can afford to make it with**.

- `allow(key, shape, reason)` — it describes the intended output. It travels, and a stranger opening
  your image sees it. Pick the `shape` that matches: `Scalar` for a string, number or bool (an
  object or array under a `Scalar` key is dropped, which is what closes the smuggling channel); one
  of the structured shapes otherwise.
- `deny(key, reason)` — it describes this install (a tier, a kernel, a GPU) or names something local
  (an id, a path, a preset). It is dropped, and a shared image will not reproduce whatever it did.

Either way write the reason; the test requires one. If the value is authored text the user typed
rather than a slug, say so in the reason and add it to `PROSE_KEYS` — but understand that you are
adding a field to the callout at the top of this document, and update that table too.

### Registering a builder

If the lint says a *builder* is unaccounted for, decide whether its lane embeds.

<!-- PINNED: builders -->

| File | Function | Lane |
| --- | --- | --- |
| `apps/web/src/imageJobAdvanced.js` | `buildImageJobAdvanced` | The Image Studio's ~30-knob builder. |
| `apps/web/src/imageJobs.js` | `buildEditJobBody` | The Image Editor's prompt-edit body. |
| `apps/web/src/imageJobs.js` | `buildDetailJobBody` | The standalone Detail pass. |
| `apps/web/src/components/CharacterAdvancedOptions.jsx` | `buildAdvanced` | The character lane's shared tuning block. |
| `apps/web/src/screens/characterPanels.jsx` | `useAngleController` | The Angle Set form's extras. |
| `apps/web/src/screens/characterPanels.jsx` | `usePoseController` | The Pose Library form's extras. |
| `apps/web/src/screens/DocumentStudio.jsx` | `submit` | The interleaved-document lane. |
| `apps/web/src/imageJobs.js` | `buildUpscaleJobBody` | The standalone upscale job. Registered as emitting **no** `advanced` map, so the day it grows a knob the lint demands a classification. |

<!-- END PINNED: builders -->

Adding an entry to `ADVANCED_BUILDERS` is what turns the lint on for that builder — at which point
every key it emits needs an `allow`/`deny` decision.

If the lane does **not** embed, the entry goes in `DEFERRED_ADVANCED_BUILDERS` with a reason naming
the story that will classify it, or a `PERMANENT EXEMPTION:` reason saying why no story ever will
and what would have to change. That list is what makes "unaccounted for" and "accounted for as out
of scope" different states:

<!-- PINNED: deferred-builders -->

| File | Function | Why it is deferred |
| --- | --- | --- |
| `apps/web/src/screens/VideoStudio.jsx` | `submit` | Video lane. No video write seam embeds yet. |
| `apps/web/src/components/editor/useEditorGeneration.js` | `buildBasePayload` | Video lane — the timeline editor's shared payload. |
| `apps/web/src/screens/EditorScreen.jsx` | `extendSelectedClip` | Video lane — a timeline action. |
| `apps/web/src/screens/EditorScreen.jsx` | `replaceSelectedItem` | Video lane — a timeline action. |
| `apps/web/src/screens/EditorScreen.jsx` | `bridgeGap` | Video lane — a timeline action. |
| `apps/web/src/simple/simpleJobs.js` | `buildSimpleVideoRequest` | Video lane. Its image sibling delegates to the studio builder and is already covered. |
| `apps/web/src/training/trainingConfig.js` | `trainingConfigSnapshot` | Permanently exempt: trainer hyperparameters are a different namespace, and no training write seam embeds. |

<!-- END PINNED: deferred-builders -->

Be clear about what that list is: a decision record, not an enforced gate. Nothing in the build
stops a write seam on a deferred lane from calling the embedder while its builder is still parked
here — the lint would stay green and every one of that builder's keys would be dropped silently,
which is exactly the failure the registry exists to make visible. Moving the entry up into
`ADVANCED_BUILDERS` is the step that turns the lint on, and it is the step whoever makes that lane
embed has to remember.

## The setting

`embedWorkflowInImages`, a UI preference stored in `ui-preferences.json` and read by the worker at
the PNG write seam.

- **Default: on.** A feature whose point is that a shared image reloads its recipe is worthless off
  by default. Someone who does not want it should have to find the switch once rather than opt in
  forever.
- **Read live, per job.** Flipping it takes effect on the next job, not the next launch.
- **Fails closed on an unreadable file.** If reading the preference file fails with anything other
  than "not found" — a sharing violation, an ACL error — embedding is off. "Absent" and "unreadable"
  are different states, and collapsing them is how a deliberate opt-out silently inverts itself. A
  file that is present and readable but not parseable falls back to the default, which is on.
- Turning it off changes nothing about images already written. The chunk is in those files.

The settings UI and its first-run disclosure are sc-15953's, not this document's.

## How this document is kept honest

`crates/sceneworks-core/tests/workflow_share_doc.rs` parses the `<!-- PINNED: … -->` blocks above out
of this file and asserts them against the shipped code, in both directions — a row here that the
code does not have fails, and a thing the code has that is not a row here fails too. Specifically:

| Block | Pinned against |
| --- | --- |
| `prose-fields` | Observed behaviour: a path-shaped value is seeded into every string-bearing field of a real request, and the set of fields that carry it through must be exactly this table. |
| `identity` | The constants themselves. |
| `envelope-fields` | The serialized field paths of a fully-populated envelope. |
| `advanced-shared` / `advanced-withheld` | `ADVANCED_KEY_RULES`, key and disposition, plus the `Shape` column. |
| `input-kinds` | `INPUT_KINDS`. |
| `omitted-fields` | `OMITTED_FIELDS`. |
| `ceilings` | Observed behaviour for the collection and string bounds — the largest input that survives and the smallest that does not — and the constants for the envelope and PNG ceilings. |
| `builders` / `deferred-builders` | `ADVANCED_BUILDERS` and `DEFERRED_ADVANCED_BUILDERS`, file path and function name. |
| `lints` | Each named test exists in `crates/sceneworks-core/tests/workflow_share.rs`. |

What is **not** machine-checked is the prose in the right-hand "what it is" columns and the narrative
sections. Those are read against the source, not derived from it. The version-behaviour table, the
trust-boundary list and the setting's semantics are prose; the modules they describe are
`workflow_share.rs`, `workflow_png.rs`, `project_store.rs` and `app_paths.rs`.
