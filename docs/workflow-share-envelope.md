# The workflow share envelope

A PNG that SceneWorks generates carries a small block of JSON describing the recipe that made it,
so the recipe travels with the image. The local sidecar (`<media>.sceneworks.json`) does not survive
a copy-paste into a chat window; this does, because it is inside the file.

Three things read it back today: importing such an image records the envelope on the asset,
`POST /api/v1/workflows/inspect` reads one out of an uploaded file without importing anything, and
`GET /api/v1/projects/:project_id/assets/:asset_id/workflow` re-reads the recorded one and resolves
it against this install's catalogs. Both studio surfaces — dropping a shared image anywhere, and
"Use this recipe" on an imported one — go through the same offer panel.

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
> and you can read it back before you share. The only things removed from them are control
> characters other than newline and tab, Unicode `Cf` format characters, the whitespace around the
> whole value, and anything past the 16 KiB prose ceiling. Nothing in the middle of the text is
> rewritten — but a prompt written with Windows line endings arrives with its carriage returns
> gone, and one typed with a leading or trailing blank line arrives trimmed.
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

A second, smaller block rides beside it in the A1111 convention, purely so third-party galleries can
*display* something — see [The `parameters` chunk](#the-parameters-chunk-an-a1111-readable-trailer).

<!-- PINNED: identity -->

| Constant | Value | Meaning |
| --- | --- | --- |
| `WORKFLOW_CHUNK_KEYWORD` | `sceneworks:workflow` | The PNG `iTXt` keyword the JSON lives under. |
| `PARAMETERS_CHUNK_KEYWORD` | `parameters` | The PNG text keyword the A1111 trailer lives under. |
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

`schemaVersion` is the contract version and is **the only version the parser branches on**.
`producer.version` is recorded so a bug report is actionable and is never interpreted. (It is not
the only *field* the parser branches on: the `sceneworksWorkflow` marker is checked too, and a blob
whose marker names a kind this build has no reader for at all — `hologram`, say — is refused as an
unsupported kind rather than parsed as an image one. That check runs BEFORE the version check, so a
kind we do not understand is never reported as a version problem.)

A kind this build *does* know is a different question, and the shared parser is deliberately not
where it is answered — see [Every container asserts its own
kind](#every-container-asserts-its-own-kind).

### Every container asserts its own kind

The parser accepts **every** kind in the contract, because an envelope is an envelope. A *reader* is
narrower: it hands its result to one lane, and a lane can only act on the kind it was built for.
Offering a clip's recipe to Image Studio is the same failure as parsing an unknown marker as an
image — it presents a file as a kind it is not — so it is refused in the same way, with the same
error and the same `UnsupportedKind` shape.

So the rule is per-reader, and there are three readers:

<!-- PINNED: container-kinds -->

| What is being read | File | Reader | Kind it accepts |
| --- | --- | --- | --- |
| A PNG's `sceneworks:workflow` chunk | `crates/sceneworks-core/src/workflow_png.rs` | `read_workflow_chunk_from` | `image` |
| An MP4's `comment` tag | `crates/sceneworks-core/src/workflow_mp4.rs` | `read_workflow_metadata_from` | `video` |
| The envelope recorded on an imported asset | `apps/rust-api/src/workflows.rs` | `get_asset_workflow` | `image` |

<!-- END PINNED: container-kinds -->

The third row is the one that was missed. Until sc-15956 the marker had exactly one legal value, so
the shared parser's refusal of `"video"` did every reader's container check for it, for free. Widening
the marker to `image` | `video` removed that inheritance — silently, from a route in a crate the
change did not touch. `GET …/assets/:asset_id/workflow` went on calling `parse_workflow_share` alone
and started answering `200 { status: "workflow" }` for a video envelope, whose response feeds
`recipeFromWorkflowShare` and an **Image Studio prefill**. The asset panel's `asset.type === "image"`
button guard is not a substitute: a video envelope recorded on an image asset passes it unchanged.

`the_asset_route_500s_when_the_stored_envelope_names_another_container` in
`apps/rust-api/src/tests/workflows.rs` pins that reader against a complete, valid video envelope —
one the parser accepts — so the test fails if the container assert is removed rather than passing on
a deserialize error. The table above is pinned by `the_doc_lists_exactly_the_container_kind_asserts`,
which reads each named file and fails if the assert it claims is not there.

**Adding a container, or a reader, means adding a row here and an assert there.** The parser is not
the place to put it: an envelope that is legal to parse and wrong to act on is exactly the case this
rule exists for.

| The file says | This build does |
| --- | --- |
| `schemaVersion` newer than this build's | Refuses the whole envelope and names both versions, so the user is told to update rather than shown a parse error. |
| `schemaVersion` equal to or older than this build's | Reads it with today's field set. Fields this build does not know are dropped; fields it knows and the file lacks take their defaults. |
| `schemaVersion` absent or `null` | Refuses the envelope. A version-less blob is not readable safely. |
| `schemaVersion` present but not a whole number | Refuses the envelope as malformed, naming the field. |

Adding a field does not need a version bump: an older reader drops what it does not recognize.
Bump only when a field changes meaning or disappears — and a bump means every older build stops
reading the file at all, rather than reading it partially.

## The `parameters` chunk: an A1111-readable trailer

Every embedded PNG carries a **second** text chunk, under the unprefixed keyword `parameters`, in
the layout AUTOMATIC1111's WebUI popularised (sc-15957):

```
<prompt> <lora:readable_file_key:weight>
Negative prompt: <negative>
Steps: N, Sampler: X, CFG scale: N, Seed: N, Size: WxH, Model: <slug>, Model hash: <sha256>, Lora hashes: "readable_file_key: <sha256>", Version: <producer.version>, software: SceneWorks
```

Civitai and most galleries and viewers parse that block. They **display** generation settings; they
do not execute them — which is what makes it worth writing for models those tools have never heard
of. A Krea or FLUX.2 image posted to a gallery shows its prompt and seed instead of arriving as
opaque pixels. Judge it on that: it is a display-legibility feature, not an execution-interop one,
and `sceneworks:workflow` remains the only block a SceneWorks install reads back.

**It is not ComfyUI's format.** ComfyUI embeds `workflow` / `prompt` chunks holding a serialized
node graph. SceneWorks has no node graph and emitting a fake one would misrepresent the product.

### It is a second rendering, not a second channel

The trailer is written from the **already-sanitized envelope** above, never from the raw job payload.
Everything on this page therefore applies to it unchanged: a key the allow-list withholds cannot
appear in it, because the renderer never sees one. `Version:` is fed from `producer.version` off that
same envelope, so the two blocks in one file cannot disagree about which build wrote it, and
[the setting](#the-setting) governs both together — off means neither is written.

For generated base images, the worker adds trusted post-resolution facts before the two blocks are
written. When a generation lane chose the actual denoise count after request parsing, its trusted
`numInferenceSteps` result becomes `advanced.steps`. The imported-Krea lane also records and exports
the actual `euler` sampler it passes to the runtime. These are execution facts, not model defaults
guessed later. Client-supplied internal telemetry is never promoted, and multi-phase schedules
remain uncollapsed.

The worker also resolves every selected LoRA through the same exact-file function inference uses,
hashes those bytes, and replaces request-derived LoRA hints with that proven stack before writing.
The safe filename stem is only the readable association key shared by the prompt tag and hash map;
the SHA-256 is the identity Civitai resolves. Renaming identical bytes changes the key but not the
resource match, while changing the bytes changes the digest. No path or user-entered Civitai id is
recorded. Hashing failure removes attribution only and never fails an otherwise successful image.

### Omit rather than approximate

A guessed mapping ships misleading data to a public gallery, and nobody downstream can tell a guess
from a fact. So every field below is an exact restatement of something the envelope holds, and
anything without a clean equivalent is **left out rather than approximated**.

<!-- PINNED: a1111-fields -->

| Field | What it is |
| --- | --- |
| `Steps` | `advanced.steps` for a single-phase run, or the exact sum of `advanced.phases[].steps` for a recorded multi-phase run. This includes the worker-resolved count above when the request omitted its own value. |
| `Sampler` | `advanced.sampler`, verbatim — except the literal `default`, which names no sampler and is omitted. For imported Krea, the worker overwrites this with the actual `euler` sampler it executes. |
| `CFG scale` | `advanced.guidanceScale`, only when `advanced.guidanceMethod` is absent. The studio emits that key only for a method that is not plain CFG, so its presence means the number is not a CFG scale. |
| `Seed` | `seed`, verbatim — the seed of this image, which is the only one the envelope carries. |
| `Size` | `width` x `height`, only when they are the dimensions of the file being written. |
| `Model` | `model`, the safe readable catalog slug, verbatim. |
| `Model hash` | `modelHash`, only when the worker retained the exact SHA-256 of the imported checkpoint that the resolved route executed. Civitai uses it with `Model` to link the precise model version and author. |
| `Lora hashes` | A quoted `readable_file_key: SHA-256` map for every exact adapter file the worker resolved and hashed. The same key appears in `<lora:readable_file_key:weight>` for fixed weights, allowing Civitai to associate the hash with the prompt resource. |
| `Version` | `producer.version` off the envelope's own producer block. |
| `software` | `producer.name` from the trusted producer block. Civitai displays this canonical lowercase field as the generating-software badge. |

<!-- END PINNED: a1111-fields -->

The prompt is always the first line, even when empty, because the layout is positional. The
`Negative prompt:` line is omitted entirely when there is no negative prompt, which is what A1111
itself does. Values containing a comma, a colon or a newline are JSON-quoted, mirroring A1111's own
`quote()` — without that, a comma inside a model slug would split one field into two for every
reader in the wild.

**The settings line is withheld whole below three pairs.** A1111's parser puts a trailing line of
fewer than three `Key: value` pairs back into the *prompt*, and every gallery modelled on it does the
same — so a two-pair line is not a thin settings block, it is a wrong prompt. `SETTINGS_PAIR_FLOOR`
in `workflow_parameters.rs` drops the line rather than emitting one below the threshold. Every
shipping lane clears it unaided; the thinnest, a standalone upscale, emits
`Seed` + `Model` + `Version` + `software`, one field above the floor. The floor remains a guard
rather than an observation because malformed foreign producer/model labels can still be reduced.

What is deliberately **not** in the trailer, and why: the multi-phase Krea guidance schedule (it
has no single-number form, while `Steps` is safely emitted as the exact sum of contiguous phase
step counts);
`textStyleGain`; `scheduler` / `schedulerShift` (A1111's `Schedule type` is a noise schedule, ours
is closer to its *sampler* — two plausible mappings and no correct one); `strength` (the sense is
lane-dependent, and the candle fork lane inverts it); `upscale.*` (A1111's hires fix is a specific
latent second pass, ours is a post-pass on the decoded image); unresolved or unhashed `loras[]`
(a readable name alone is not resource identity), and a single `<lora:...:weight>` tag for any
adapter whose multi-phase schedule actually varies that weight (its exact hash still travels);
`count`; the control-overlay fields; `faceRestore` (A1111's field names an engine and ours is a
boolean); and the tier / kernel / GPU keys, which the allow-list withholds so this rendering could
not see them if it wanted to.

### `tEXt` when the text is ASCII, `iTXt` when it is not

Deliberately different from the envelope chunk's compressed `iTXt`, because a different audience
reads it. `PIL.PngImagePlugin.PngInfo.add_text` — what A1111 writes through, and what third-party
parsers are tested against — writes an uncompressed `tEXt` when the value encodes as Latin-1 and
falls back to an uncompressed `iTXt` when it does not. So **PIL's boundary is Latin-1, not ASCII**;
ours is narrower on purpose, and differs only in the U+00A0..U+00FF band, where a `tEXt` chunk's raw
0xE9 byte is `é` to a Latin-1 reader and a replacement character to the very common reader that
assumes UTF-8. `iTXt` is UTF-8 by specification, so both classes of reader agree. Uncompressed in
both arms, matching PIL's default: being findable is the entire value of the chunk.

Uncompressed is not free, and the size is worth stating rather than waving at. On the representative
recipe the trailer is **186 bytes** framed, beside a 565-byte compressed envelope chunk. With both
prose fields at the sanitizer's 16 KiB `PROSE_MAX_BYTES` cap it is **32,913 bytes** — against a
467-byte envelope chunk carrying the same prose deflated. That is the ceiling, not an estimate: no
other field on the line is prose, so 2 x 16 KiB plus a settings line is the whole of it.

It stays uncompressed at that size, and the ratio is the reason rather than the absolute number. The
file already holds a compact copy of every byte in the trailer — that is what the envelope chunk
beside it *is* — so compressing this one saves bytes the file has already spent and risks the single
property it exists for. A gallery that cannot find the block gets nothing; one that finds a 33 KB
block displays the prompt. `the_parameters_chunk_at_the_prose_cap_is_the_worst_case` in
`crates/sceneworks-core/tests/workflow_png.rs` measures both rows.

`pil_agrees_with_our_encoding_choice` in `crates/sceneworks-core/tests/workflow_parameters.rs` runs
the real library, asserts it reads both of our chunks back verbatim, and asserts PIL's own boundary
so a future Pillow that moved it fails here rather than leaving this paragraph describing a library
that changed. It skips loudly when Python or Pillow is not installed.

### Reading A1111 and ComfyUI PNGs is a different story

This is a write-only convention here. The reader resolves `sceneworks:workflow` and nothing else, and
a foreign `parameters` chunk is an absence. Parsing one — mapping unknown samplers and model names
onto this install's catalog, and degrading well when the mapping fails — is genuinely useful and
much larger, and is deliberately not part of this contract.

## What travels

Every field the envelope can carry, and nothing else. There is no passthrough bucket for unknown
keys in either direction: a key this table does not name is dropped on write **and** on read.

<!-- PINNED: envelope-fields -->

| Field | What it is |
| --- | --- |
| `sceneworksWorkflow` | The marker, naming the workflow **kind**: `image` or `video`. A build that does not know a kind refuses the file rather than presenting it as one it does understand. |
| `schemaVersion` | The contract version. |
| `producer.name` | `SceneWorks`. |
| `producer.url` | The project's repository URL. |
| `producer.version` | The released version of the build that wrote the file. |
| `mode` | The generation mode (`text_to_image`, `edit_image`, `character_image`, …). |
| `model` | The model catalog **slug** (`z_image_turbo`), never a weights location. |
| `modelHash` | SHA-256 of the exact imported checkpoint bytes, when worker-proven. It is content identity for gallery attribution, never a local path or user-entered Civitai id. |
| `prompt` | The prompt, verbatim. See the callout above. |
| `negativePrompt` | The negative prompt, verbatim. |
| `seed` | The seed of **this** image. The rest of the batch's seeds do not travel. |
| `width` | Requested output width. |
| `height` | Requested output height. |
| `count` | How many images the run requested — so the file says it was one of a batch of N. |
| `durationSeconds` | **Video.** The clip length the run asked for. The ask, not the measurement — a 6.0 s ask that rendered 5.96 s replays as 6.0. |
| `fps` | **Video.** The frame rate the run asked for, for the same reason. |
| `quality` | **Video.** The quality preset the run was submitted at — `fast`, `balanced` or `best`. A named tier off a menu the receiving install also has (it shows them as "Draft" / "Balanced" / "Final"; the value is what travels). |
| `stylePreset` | The style preset label the run used. |
| `styleId` | The catalog style id the user picked. |
| `fitMode` | How the source image was fitted (`crop`, `contain`, …). |
| `upscale.enabled` | Present only when the run enabled an upscale pass. |
| `upscale.factor` | The upscale factor. |
| `upscale.engine` | The upscale engine label. |
| `upscale.softness` | The upscale softness. |
| `loras[].name` | For a generated image, the safe readable stem of the exact adapter filename. It associates the prompt tag with the hash map but is not resource identity. A foreign or unresolved recipe may instead carry its portable display hint. |
| `loras[].weight` | Its weight. |
| `loras[].repo` | Its Hugging Face `owner/name`, when the catalog entry resolved to one. Never a path, never a local id. |
| `loras[].hash` | SHA-256 of the exact adapter bytes inference resolved, when worker-proven. Client payload hashes are ignored; malformed foreign hashes are dropped without breaking import. |
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
  forms and `..` traversals. A value that trips the check is dropped rather than trimmed. Three
  honest limits on that. **One:** it is a *shape* test and not knowledge of your disk, so a bare
  name with no separators in it (`acme-brief`) is not a location and travels. **Two:** a relative
  POSIX path of only **two** segments is deliberately not treated as a location — two segments is
  the shape of a Hugging Face repo id (`acme/mira`), which `loras[].repo` carries for real, and a
  slash turns up in free-text labels people write by hand (`Ghibli / soft light`). So a
  `loras[].name` or a `styleId` reading `Clients/Acme` is not caught and reaches the file; three or
  more segments (`Clients/Acme/brief`) is. **Three:** it does not apply at all to the six prose
  fields, which is the callout at the top of this document.
- **Project, job and asset identity.** `projectId`, `projectName`, `jobId`, `assetId`,
  `generationSetId`, `characterId`, `characterLookId`.
- **Timestamps.** The envelope has no time field.
- **The rest of the batch.** `seeds` does not travel; `seed` is the one that rendered this file.
- **This machine's hardware budget.** Quant tier, INT8-ConvRot selection and flash-attention are
  classified and deliberately dropped — see the withheld table below. The requested GPU is not in
  that table because it never enters `advanced` at all: it is a top-level request field, and the
  envelope has no slot for it, so it is left behind by the field list being closed rather than by a
  decision written down against its name. Either way the receiving install picks its own.
- **Local ids that resolve to nothing elsewhere.** Recipe preset ids, control-image asset ids,
  trained-overlay ids and their resolved weights paths, Key Point Library collection ids.
- **Where the adapter file lives.** `loras[].repo` is the repo id and there is no `loras[].file` or
  path beside it. A generated record may use the filename's safe stem as its readable `name`, but
  that contains no directory and is not identity; `loras[].hash` is the exact content identity.
  The consequence is on the reading side: a receiving install that has two adapters from one repo
  cannot tell from the repo id alone which the sender used. `loras[].name` is then read as a
  tie-break *among those rows only* — both parties installing the same multi-adapter pack is the
  usual way this happens, and it is the case where the two installs' display names agree. A name
  matching none of the tied rows, or two of them, leaves the entry unresolved and named rather than
  picked. A repo of which the receiver registered only one adapter is not ambiguous to it and
  resolves on the repo alone; a `name` that disagrees with that row is not consulted, because a
  display name is whatever the sender's install called the file and the repo id means the same
  thing on both machines.
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
| `decoder` | `Scalar` | Experimental alternate terminal decoder id. Native is omitted. |
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
| `motion` | `Scalar` | **Video.** Camera-motion preset off a closed menu (`static`, `slow push-in`, `handheld`). It conditions the generation. |
| `ltxPipeline` | `Scalar` | **Video.** LTX pipeline selector. It picks which denoise path runs, so two values give two different clips from one prompt. |
| `distilledVariant` | `Scalar` | **Video.** LTX distilled-checkpoint variant. A different checkpoint is a different model for replay purposes. |
| `textEncoderModel` | `Scalar` | **Video.** The text-encoder pick — it changes what the model *sees* of the prompt. A catalog-global slug, not a local id. |
| `lightning` | `Scalar` | **Video.** Wan2.2 A14B fast-4-step toggle. It swaps in a distilled recipe and overrides the step count. |
| `videoCfgGuidanceScale` | `Scalar` | **Video.** LTX native CFG scale — the video lane's `guidanceScale`. |
| `videoStgGuidanceScale` | `Scalar` | **Video.** LTX spatiotemporal-guidance scale. |
| `videoRescaleScale` | `Scalar` | **Video.** LTX guidance-rescale factor. |
| `videoConditioningStrength` | `Scalar` | **Video.** Source-clip conditioning strength for extend / bridge / v2v — the video lane's `strength`. |
| `bridgeRightVideoConditioningStrength` | `Scalar` | **Video.** The right-clip strength for a bridge. Without it a shared bridge replays lopsided. |
| `timelineAction` | `Scalar` | **Video.** Which timeline operation made the clip (`extend` / `replace` / `bridge`). A closed vocabulary with no id in it; its companion `timelineContext` is withheld. |

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
| `durationHint` | **Video.** Model-catalog prose the studio echoes back ("Recommended: 5s or less."), not a knob anybody set. The receiving install renders its own. |
| `precision` | **Video.** LTX weight precision (`fp8` / `bf16`): what this machine can afford to hold the weights in. |
| `quantization` | **Video.** The torch lane's quant tier — `mlxQuantize` for a different backend. |
| `selectedPersonTrack` | **Video, and the sharpest object in any payload.** The whole person-track record: a user-typed `name` that is routinely a real person's name, a `sourceDisplayName` that is the original imported filename, local asset ids, and a `frames[].mask` array of filesystem paths. Withheld on privacy first and shape second. |
| `replacementModeLabel` | **Video.** The display label for a person-replacement mode ("Face Only"). Neither an id nor a path, and withheld anyway — see [Person replacement](#person-replacement-is-withheld-by-default). |
| `timelineContext` | **Video.** Where in the *local* timeline to write the result: timeline / item / track / asset ids, plus the user's own typed timeline name. `timelineAction` carries the part that travels. |

<!-- END PINNED: advanced-withheld -->

## Person replacement is withheld by default

Four fields carry a person-replacement run's specifics, and none of them travel: `personTrackId` and
`replacementMode` at the top level of the request, `advanced.selectedPersonTrack` and
`advanced.replacementModeLabel` inside the map. The first is an install-local id, which alone would
put it in the same class as `controlImage`. The rest are withheld for a stronger reason.

`selectedPersonTrack` is the sharpest object any payload carries: a `name` the user typed, which is
routinely a real person's name; a `sourceDisplayName` that is the original imported filename; local
asset ids; and a `frames[].mask` array of filesystem paths. `replacementModeLabel` is neither an id
nor a path — it is the string "Face Only" — and it is withheld anyway, because **which variant of a
replacement ran is a fact about a real person who did not choose to share it**. There is no replay
value to weigh against that: a recipient has no access to the track, so the field could only ever
inform, never reproduce.

### What does travel, and why the copy has to say so

`mode` is `replace_person`, verbatim, and `model` names a replacement engine
(`wan_2_2_vace_fun_14b`). Both are ordinary shared fields, and neither is special-cased. **So the
file does disclose the technique.** What it withholds is the identity and the variant.

That distinction is not a technicality, because it is the difference between a true sentence and a
false one on the Settings surface. The "Not in the file" copy read *"anything about a person
replacement"* until the sc-15956 review measured it against real bytes; the enumerated tail was
true, the leading clause was not, and the key-level pin joining that copy to the tables above could
not catch it, because `mode` is not a key in either list. It is asserted directly instead, by
`the_settings_copy_does_not_overclaim_about_person_replacement` in
`crates/sceneworks-core/tests/workflow_share_doc.rs`, which builds a real `replace_person` envelope,
proves the four fields are absent from the serialized bytes, proves `mode` is present, and then
reads the sentence.

Narrowing the withholding to close that gap was available and was refused: the disclosure that
matters is *who*, and `mode` is a field every video envelope carries for every mode. Dropping it for
one mode would be a hole in the field list that a reader could detect — an absent `mode` on a video
envelope would itself say "this was a replacement".

### Why it is not in `omitted`

This is the one place the contract's "a stated absence beats an invisible one" rule is deliberately
inverted, and it is worth reading before anyone "fixes" the inconsistency.

`omitted` exists for collections too large to record — `loras`, `inputs`, `advanced.poses` — where
naming the gap costs nothing and tells the receiving user why their replay is incomplete. Writing
`omitted: ["personTrackId"]` would cost something: it would **announce the replacement to every
reader** while withholding only the id. Naming a withheld thing in a field designed to be read is
worse than an absence, when the absence is the point.

The honest weakness of that argument, recorded here rather than left for someone to rediscover: it
is weaker than it looks, because `mode` announces the replacement anyway (above). What `omitted`
would add is not the disclosure — that is already there — but a second, redundant one, in a field
whose whole purpose is to be surfaced to a reader, naming the private half by key. The inversion
stands on the narrower ground: the technique travels once, in the field that always carries it, and
nothing else points at what was withheld.

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
| `sourceClip` | **Video.** A clip the run continues, re-times or bridges from. Separate from `source` because "needs a still to start from" and "needs a clip to continue" are different asks of whoever replays it. |
| `referenceClip` | **Video.** A reference clip the run conditions on — the moving counterpart of `reference`. |
| `referenceAudio` | **Video.** A reference audio clip the run conditions on — the audible counterpart of `reference`. |

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

There is **one deliberate exception**, and it is the only place this rule is inverted: the
person-replacement fields are withheld without an `omitted` entry. The argument, and its honest
weakness, are in [Person replacement](#person-replacement-is-withheld-by-default). Read it before
adding them here for consistency — the inconsistency is the decision.

## Ceilings

Two kinds of bound. Per-collection caps, each inherited from the validator that already limits the
thing. `MAX_SHARE_INPUTS` is a **shape** rather than a validator's number (it is
`INPUT_KINDS.len()` — one entry per kind, and the kinds are closed). `MAX_SHARE_POSES` derives from
the API/UI `MAX_JOB_POSES` output-count contract: every pose renders one image, so new jobs are
stopped at selection or request validation rather than first losing poses in a shared file. And one
ceiling on the serialized envelope, checked after every
per-field rule has run. The second is what actually composes: per-field bounds did not, and each new
measurement found a new way to spend what they left.

<!-- PINNED: ceilings -->

| Constant | Value | What it bounds | What happens at the limit |
| --- | --- | --- | --- |
| `WORKFLOW_SHARE_MAX_BYTES` | 163,840 | The whole serialized envelope, in bytes. | The **entire** envelope is refused — no chunk is written, and a file over it reads as an error rather than as a partial recipe. |
| `PROSE_MAX_BYTES` | 16,384 | Each authored prose field, in bytes. | Truncated at a whole character. Prose still means what it said after its tail is cut. |
| `LABEL_MAX_CHARS` | 200 | Each non-prose label (model slug, style id, LoRA name, producer block), in characters. | **Dropped**, not truncated — a slug's spelling is its identity. |
| `MAX_SHARE_LORAS` | 5 | Entries in `loras`. | The list is dropped whole and `omitted` gains `loras`. |
| `MAX_SHARE_INPUTS` | 7 | Entries in `inputs` — one per kind, and the kinds are closed. | The list is dropped whole and `omitted` gains `inputs`. |
| `MAX_SHARE_PHASES` | 8 | Entries in `advanced.phases`. | The key is dropped and `omitted` gains `advanced.phases`. |
| `MAX_SHARE_POSES` | 64 | Entries in `advanced.poses`. | The key is dropped and `omitted` gains `advanced.poses`. |
| `MAX_SHARE_POSE_SLOTS` | 6,144 | Coordinate slots across the whole `advanced.poses` array — a number, or a `null` standing in for one. | The key is dropped and `omitted` gains `advanced.poses`. |
| `MAX_WORKFLOW_TEXT_BYTES` | 1,048,576 | The **decompressed** chunk text, on read. | Decompression stops and the file is refused, so a zip bomb costs a megabyte rather than its claimed size. |
| `MAX_METADATA_BYTES` | 8,388,608 | **Approximately** what the PNG decoder may buffer from an untrusted file's ancillary chunks, cumulatively. `png` grows the buffer and only then finds it over budget, so a measured refusal peaks at 8,408,413 bytes live rather than at this number. | The read is refused, whatever length a chunk header *claims* — in the walk before the image data. A budget already spent by the time the post-IDAT tail pass runs is reported as an absence instead, so a fat-but-foreign PNG that read as "no workflow" before that pass existed still does. |

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
| `every_worker_write_seam_declares_the_lane_it_embeds_for` | A worker function that can put an envelope in a file has no `WORKFLOW_WRITE_SEAMS` entry, or its entry embeds for a builder that is deferred, or its declared disposition disagrees with what the code does. |
| `the_core_workflow_surface_is_classified` | A new public entry point in `workflow_share.rs` / `workflow_png.rs` handles a `WorkflowShare` and is on neither the read nor the write surface — so the seam scan would not look for its call sites. |

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

Either way write the reason; the test requires one.

#### Tag the rule with the builder that emits it

Every rule carries an `AdvancedKeySource`, and `allow` / `deny` are shorthands that hard-code
`AdvancedKeySource::StudioBuilder`. Reaching for them out of habit is the most likely way to end up
red. A `buildDetailJobBody` key written as `allow(…)` is classified correctly and tagged wrongly,
and **three** tests in `workflow_share.rs` fail on it: the per-builder coverage test for the builder
that really emits it, the one for the studio builder that is now credited with a key it never
emits, and `every_registered_builder_has_its_advanced_keys_classified`, whose "is every classified
key still emitted?" half runs per tag.

<!-- PINNED: rule-helpers -->

| Helper | The source tag it sets | Use it for |
| --- | --- | --- |
| `allow(key, shape, reason)` | `StudioBuilder` | A key that travels, emitted by `buildImageJobAdvanced`. |
| `deny(key, reason)` | `StudioBuilder` | A key that is withheld, emitted by `buildImageJobAdvanced`. |
| `allow_from(key, shape, source, reason)` | Whichever variant you pass | A key that travels, emitted by any other registered builder. |
| `deny_from(key, source, reason)` | Whichever variant you pass | A key that is withheld, emitted by any other registered builder. |
| `deny_server(key, reason)` | `Server` | A key the API stamps onto `advanced` after the request arrives. No web builder emits it, and the lint does not look for it in the JS. |

<!-- END PINNED: rule-helpers -->

The variant to pass is the one on the builder's row in [Registering a
builder](#registering-a-builder) — `DetailBuilder` for `buildDetailJobBody`, `InterleaveBuilder` for
`DocumentStudio.submit`, and so on. A key more than one builder emits is tagged with its primary
builder; the "is every emitted key classified?" half of the lint runs for every registered builder
regardless of tags, and only the "is every classified key still emitted?" half is per-tag.

#### Then add the row to this document

Classifying the key is half of it. `ADVANCED_KEY_RULES` and the two tables above are pinned to each
other **in both directions** by `crates/sceneworks-core/tests/workflow_share_doc.rs`, so a key in the
code that this document does not list fails just as loudly as a row here for a key that does not
exist:

- an `allow` needs a row in [Shared](#shared) — the key, its `Shape` spelled exactly as the enum
  variant is, and what it is — or `the_doc_lists_exactly_the_shared_advanced_keys` fails;
- a `deny` needs a row in [Withheld](#withheld) — the key and why — or
  `the_doc_lists_exactly_the_withheld_advanced_keys` fails.

Both of those are in `workflow_share_doc.rs`, not in the lint file above. A green
`workflow_share.rs` is not the finish line.

#### And say what the receiving studio does with it

An `allow` decides that the key may leave the machine. It says nothing about whether the install
that opens the file has a control for it — and those are different questions, because an image
envelope carries keys from four lanes (the Image Studio, the standalone Detail pass, the Character
studio's angle set, the interleaved-document lane) while the "use this workflow" prefill lands in
exactly one of them.

So a shared key also needs a row in `ADVANCED_PREFILL` (`apps/web/src/workflowShare.js`), which is
read by BOTH halves of the offer panel: the recipe carries exactly the keys marked as reaching a
control, and the panel marks exactly the rest as "not restored", with the reason you write there. A
key with no row is treated as not restored — rendered and marked, never silently dropped — and
`workflowShare.test.js` fails if this document's [Shared](#shared) table and that map disagree in
either direction.

That guardrail exists because the first cut of the panel displayed ten knobs — `enhancePrompt`,
`usePid`, `decoder`, `pidTarget`, `strength`, `textStyleGain`, `faceRestore`, `controlMode`, `controlScale`,
`poses`, `phases` — as ordinary settings rows while none of them reached a control. Being told a
recipe replayed faithfully when it did not is the failure this whole contract exists to prevent.

An `allow` whose *shape* is reduced past the point of replay belongs in the same category, and
`structuredPrompt` is the standing example. It travels, as its scalar fields; the studio's
structured builder rehydrates from the `caption` object, which does not. So a shared
structured-caption recipe reaches no builder control at all and replays as the composed prose in the
plain prompt box — which reproduces the image faithfully, because the top-level `prompt` is the
exact string the model received, and is still not the builder being restored. Its `ADVANCED_PREFILL`
row therefore marks it not restored and says why. Rehydrating the builder instead would mean giving
the caption a classified sub-schema of its own, which is a change to this contract and not to the
panel.

If the value is authored text the user typed rather than a slug, say so in the reason and add it to
`PROSE_KEYS` — which makes it path-exempt, so you are also adding a field to the privacy callout at
the top of this document and must add that row too.
`the_doc_lists_exactly_the_path_exempt_prose_fields` seeds a filesystem path into every field of a
real request and fails if the callout and the sanitizer disagree about which ones carry it through.

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
| `apps/web/src/screens/VideoStudio.jsx` | `submit` | The Video Studio's ~20-knob builder. |
| `apps/web/src/components/editor/useEditorGeneration.js` | `buildBasePayload` | The timeline editor's shared video payload. |
| `apps/web/src/screens/EditorScreen.jsx` | `extendSelectedClip` | Timeline extend. |
| `apps/web/src/screens/EditorScreen.jsx` | `replaceSelectedItem` | Timeline replace. |
| `apps/web/src/screens/EditorScreen.jsx` | `bridgeGap` | Timeline bridge. |
| `apps/web/src/simple/simpleJobs.js` | `buildSimpleVideoRequest` | The Simple shell's video request. |
| `apps/web/src/screens/VideoUpscalePanel.jsx` | `onUpscale` | The standalone video upscale. Registered as emitting **no** `advanced` map, for the same reason `buildUpscaleJobBody` is. |

<!-- END PINNED: builders -->

Adding an entry to `ADVANCED_BUILDERS` is what turns the lint on for that builder — at which point
every key it emits needs an `allow`/`deny` decision, and a row in the table above, which
`the_doc_lists_exactly_the_registered_builders` pins in both directions.

An entry is more than a path and a function: it is what tells the lint how to read that file, and
how to know it still can.

<!-- PINNED: builder-fields -->

| Field | What to put in it |
| --- | --- |
| `source` | A **new** `AdvancedKeySource` variant for this builder, added to `AdvancedKeySource::ALL` as well as declared. One variant per registered builder, checked both ways, or `every_source_tag_names_exactly_one_registered_builder` fails. |
| `path` | Repo-relative path of the file that defines it. Read out of the repo, so a move fails loudly. |
| `function` | The JS function whose body the extractor reads. Read out of the repo, so a rename fails loudly. |
| `shape` | Which `AdvancedBuilderShape` the JS is written in, and therefore which extractor can read it: `ReturnedObject`, `FlatAdvancedLiteral`, `SpreadAdvancedLiteral`, `AssignedObject`, `ExtrasLiteral`, or `NoAdvancedMap` for a builder that posts no `advanced` map at all. |
| `lane` | Prose: the embedding lane this builder's payload ends up written by, so a reader can see why the keys matter. |
| `anchors` | Keys the extractor **must** still find. The floor against an extractor that has quietly stopped understanding the file — a lint that reads zero keys and passes is the failure mode the whole registry exists to prevent. |
| `minimum_keys` | The smallest key count the extractor may report before the lint calls itself broken. Set it well under the real count; it is a floor, not a census. |
| `spread_of` | Identifiers spread into an `AssignedObject` or `SpreadAdvancedLiteral` initializer whose keys another registry entry already accounts for. A dotted member path (`base.advanced`) counts as one name; a call expression is refused, because its keys have no declarable name. Empty for most builders, and declared so a **new** spread of something nobody classified fails the lint instead of vanishing into it. |

<!-- END PINNED: builder-fields -->

A new variant on `AdvancedKeySource` is the step most easily missed, because nothing about writing
`allow_from(…, AdvancedKeySource::MyBuilder, …)` reminds you that the variant also has to appear in
`::ALL`.

If the lane does **not** embed, the entry goes in `DEFERRED_ADVANCED_BUILDERS` with a reason naming
the story that will classify it, or a `PERMANENT EXEMPTION:` reason saying why no story ever will
and what would have to change. That list is what makes "unaccounted for" and "accounted for as out
of scope" different states:

<!-- PINNED: deferred-builders -->

| File | Function | Why it is deferred |
| --- | --- | --- |
| `apps/web/src/training/trainingConfig.js` | `trainingConfigSnapshot` | Permanently exempt: trainer hyperparameters are a different namespace, and no training write seam embeds. |

<!-- END PINNED: deferred-builders -->

Until sc-16113 that list was a decision record and nothing more. It said, in its own doc comment,
that no video lane could start embedding while its keys sat in it — and nothing enforced that.
Adding a `write_workflow_chunk` + `embeddable_workflow_share` call to a real video-lane PNG write in
`crates/sceneworks-worker/src/video_jobs/seedvr2.rs` left every lint green while all of
`VideoStudio.jsx`'s unclassified keys were dropped from the written file, with nothing anywhere to
say so.

**sc-15956 is the story the gate was built for, and it did its job.** All six video builders moved
into the table above, which is what forced every key behind them to be classified before a video
seam could embed. Two findings are worth keeping, because both argue for the enforcement being a
discovery scan rather than a list:

- the deferral said "~15 keys" and the real count was **17**. `timelineAction` and `timelineContext`
  are emitted only by the three `EditorScreen.jsx` actions, and nobody had read those builders. A
  deferral records that a decision is owed; it is a poor estimate of what the decision costs;
- two of the 17 were **objects**, not scalars — `selectedPersonTrack` carries a person's name, an
  original imported filename and an array of mask file paths, and `timelineContext` carries local
  ids plus the user's own timeline name. Under the pre-sc-16113 arrangement those would have
  travelled, silently, in every shared clip.

## The write seams, and what each one embeds for

`WORKFLOW_WRITE_SEAMS` in `crates/sceneworks-core/src/workflow_share.rs` is what closes that. It
pairs every place in `sceneworks-worker` an envelope can reach a written file with the **web
builders that feed it** — the same `(file, function)` key the two builder registries are joined on —
and `every_worker_write_seam_declares_the_lane_it_embeds_for` refuses a seam whose builder is
deferred.

<!-- PINNED: write-seams -->

| File | Function | What it does about the chunk |
| --- | --- | --- |
| `crates/sceneworks-worker/src/image_jobs.rs` | `write_image_asset` | Embeds — the one funnel every generated image is written through, for both `/image/jobs` and `/image/interleave/jobs`. |
| `crates/sceneworks-worker/src/image_jobs.rs` | `upscaled_workflow_share` | Embeds — the inline-upscale sub-step's envelope: the generation payload with the applied pass overlaid. |
| `crates/sceneworks-worker/src/image_jobs.rs` | `write_upscaled_asset` | Embeds — the inline-upscaled variant's own PNG. |
| `crates/sceneworks-worker/src/image_jobs.rs` | `detail_workflow_share` | Embeds — the standalone detail pass's envelope. |
| `crates/sceneworks-worker/src/image_jobs.rs` | `standalone_upscale_workflow_share` | Embeds — the standalone upscale job's envelope. |
| `crates/sceneworks-worker/src/image_jobs/detail.rs` | `run_image_detail_job` | Embeds — writes the refined PNG. macOS-gated; the scan is textual, so it is read on every platform. |
| `crates/sceneworks-worker/src/upscale_jobs.rs` | `run_image_upscale_job` | Embeds — hands the standalone upscale's envelope to the shared single-child writer. |
| `crates/sceneworks-worker/src/single_child_asset.rs` | `write_single_child_asset` | Conduit — writes the envelope its caller decided on, and builds none itself. |
| `crates/sceneworks-worker/src/video_jobs/mod.rs` | `video_workflow_metadata` | Embeds — the funnel every *generated* clip is encoded through, whatever engine or studio produced it. |
| `crates/sceneworks-worker/src/video_jobs/seedvr2.rs` | `seedvr2_workflow_metadata` | Embeds — the SeedVR2 video upscale's own clip. macOS + candle-gated; the scan is textual, so it is read on every platform. |
| `crates/sceneworks-worker/src/segment_jobs.rs` | `run_image_segment_job` | Declines — a smart-select mask has no generation recipe to replay, and is grayscale. |

<!-- END PINNED: write-seams -->

**The seams are discovered, not listed.** The lint walks every `.rs` file under
`crates/sceneworks-worker/src` and treats a function as a seam if its body NAMES the core write
surface (`build_workflow_share`, `build_workflow_share_from`, `embeddable_workflow_share`,
`write_workflow_chunk`) — called or taken as a value — or carries a `WorkflowShare` in its own
signature, directly or inside a struct that holds one, or names something that does. That last
closure is what reaches `upscale_jobs.rs` and `segment_jobs.rs` through `write_single_child_asset`,
neither of which mentions `sceneworks_core` at all. `use … as …` renames are resolved before the
walk, so importing the writer under another name does not switch the lint off for that file. A
brand-new worker file with an embedding call is caught the moment it appears, and there is no file
list anywhere to forget to update.

So the five states are distinct, and every one of them is checked against the source:

- **Embeds** — must name at least one builder, and every one must be in `ADVANCED_BUILDERS`. A
  reference into `DEFERRED_ADVANCED_BUILDERS` fails the build, printing the seam, the lane and the
  deferral's own reason. This is the failure sc-15956 will hit, and moving the video builders up —
  which forces every key to be classified — is what clears it.
- **Conduit** — writes an envelope its caller built. Must accept one through its signature, must
  reach a writer, and must obtain none of its own — no builder, no parse, no mention of the type.
  Its callers are seams in their own right, so no lane escapes through it.
- **Declines** — writes no envelope at all. Checked positively rather than by the absence of a
  builder call: it may not pass anything but a literal `None` to `write_workflow_chunk`, may not
  name `WorkflowShare` in its body, may not accept one, and must fill every share-carrying field
  with `None` — by initializer, by later assignment, or in shorthand. **Declining is not a way to
  embed without classifying a lane, and where the envelope came from makes no difference**: a
  declining seam that starts writing one — built here, parsed back out of another file, or cloned
  from somewhere else — flips to a failure rather than staying quiet.
- **Inert** — a share reaches its signature and goes nowhere. A logging or validation helper that
  takes a spec is this. Must reach no writer at all; the moment it calls one it owes a lane. It
  exists so that "writes an envelope its caller built" is never written about a function that
  writes nothing.
- **Undeclared** — fails, naming the file and the function. An unmapped seam does not pass, and
  two same-named functions in one file fail too, because one entry cannot describe both.

### What this does not prove

The list below was re-derived after the sc-16113 review, which found three working evasions this
section did not mention — two of them worse than anything it did. Those three are closed; these are
what is left.

- **That a seam's declared builder list is the whole truth.** This is the biggest hole and it is
  deliberate. The mapping is an explicit declaration because a Rust write seam has no inherent
  knowledge of which web builder feeds it, and every inference available (the job type, the file's
  directory) fails open. An author who declared a video seam as embedding for
  `buildImageJobAdvanced` would have written a false statement into a public registry, beside prose
  describing the lane; the lint cannot tell, and review is what catches it. What it does guarantee
  is that the statement had to be written at all, and that the honest version of it fails the
  build.
- **That an envelope reaching a seam through a Rust `type` alias is seen.** `use … as …` is
  resolved; `type Ws = WorkflowShare;` is not. Nor is a share reached only through a generic
  parameter or a trait object, nor a writer whose name a macro pastes together.
- **That an `Inert` or `Conduit` claim survives an indirection the scan cannot follow** — a `dyn`
  call or a trait method that reaches a writer is invisible to it.
- **Anything about seams outside `crates/sceneworks-worker/src`.** `apps/rust-api` writes chunks in
  its own tests; the product write path is the worker's.

### Wiring a new lane, and the two loud surprises waiting for it

The back-check that every `ADVANCED_BUILDERS` entry is named by some embedding seam means the two
halves of a new lane cannot be split across two PRs: classifying the ~15 video keys on their own
would leave `VideoStudio.jsx::submit` registered with no seam behind it, which fails. Move the
builder up and add the `WORKFLOW_WRITE_SEAMS` entry in the same change.

And `a_seam_that_embeds_for_a_deferred_builder_fails_the_build` hard-codes `VideoStudio.jsx::submit`
as its deferred example, so promoting that builder turns that mutation proof into a "did not panic
as expected" failure. Re-point it at whatever is still deferred; the proof is that *some* deferred
builder is refused, not that one particular one is. Both surprises are loud rather than silent,
which is why they are documented rather than designed away.

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
- **It governs both chunks.** The envelope and the A1111 `parameters` trailer hang off the same
  decision at the same seam, so off means neither is written. There is no arrangement in which the
  prompt rides out in the trailer after the switch was turned off.
- Turning it off changes nothing about images already written. The chunks are in those files. To
  share one of those without them, use **Save a copy without the workflow** on the asset — which
  excises **both** from the copied bytes and leaves the asset alone. The browser and LAN download
  does the same server-side, through `?stripWorkflow=true` on the file route.
- **MCP image egress strips by default.** Both `generate_image`'s inline base64 response and its
  oversize `resource_link` fallback request that same hardened server-side strip representation;
  `get_job_result` follows the same link policy. This default is intentionally different from a
  human Save As: MCP results commonly travel to model-provider infrastructure, so forwarding the
  prompt, model settings, LoRA repositories and pose/face coordinates without a deliberate choice
  is the riskier default. An agent that genuinely needs recipe inspection can opt in per call with
  `includeWorkflow: true`. Each image result reports the requested policy as
  `strip-requested` or `preserve-if-present`; it does not claim metadata existed when the source was
  already clean or was not a PNG.

### The strip has to take the `parameters` chunk too, and one rule decides whose it is

A trailer left behind by "Save a copy without the workflow" would put the prompt in a file the user
deliberately stripped, with no way for them to know. So it comes out.

But `parameters` is *the* generic keyword — an A1111 or ComfyUI export the user imported carries one
too, and stripping ours is not a licence to scrub someone else's metadata out of their image. The two
are separated by **co-presence**: a `parameters` chunk is treated as ours only when the file also
carries a `sceneworks:workflow` chunk. That is exact rather than heuristic in both directions,
because the writer emits the pair or neither, and nothing writes our keyword into a file we did not
encode. It also makes the operation idempotent — stripping an already-stripped copy is a clean no-op.

The honest limit, recorded rather than left to be rediscovered: a SceneWorks image whose envelope
chunk some *other* tool removed, leaving the trailer, reads to us as a foreign A1111 file and keeps
it. Nothing in the bytes distinguishes that case, our own strip never produces it, and treating every
`parameters` chunk as ours would trade a hypothetical leak for a certain one.

That control is named by exactly one constant, `SAVE_WITHOUT_WORKFLOW_LABEL`, and it is reachable
from three places rather than one: a button beside **Save As…** in the advanced preview, the
right-click context menu there, and a second download button in the Simple shell. All three, because
the copy above tells a user to go and find it — and the Simple shell is the default on a phone,
where a right-click does not exist.

The switch lives in **Settings → Settings → Sharing** and, in the Simple shell, under **Sharing** on
its Settings screen. The copy under it is the summary this document is the long form of. What keeps
the two from drifting is that the summary links here and that its three lists — the path-exempt prose
fields, what is in the file, and what is not — are rendered in the UI from declared lists that
[How this document is kept honest](#how-this-document-is-kept-honest) pins against the tables above,
in both directions.

### The Image Editor is the one export that re-encodes

Every other way a file leaves this app copies bytes, so the chunk rides along without anyone
arranging it. The editor cannot: `Download` there flattens a layer stack onto a canvas, and a
canvas holds pixels and nothing else — no text chunks, no colour profile, no EXIF. Before sc-15954
that quietly cost the recipe on a round trip in which the user changed nothing.

So the editor's `Download` has two behaviours, and its header says which one the next click gets:

- **Nothing has been changed** → the file you opened is written out **byte for byte**, under its
  own name. Not a re-encode with the recipe put back — the same bytes, so the chunk, the colour
  profile and the pixels are all the original's. The header reads *Recipe included*.
- **Anything has been changed** → a fresh PNG of what you see, with **no** recipe in it. The header
  reads *Recipe not carried*. An edited image is not the output of that recipe, and there is no
  flagged "provenance, not reproduction" variant of the envelope: the additive-field rule above
  means an older reader would drop such a flag and go on presenting the recipe as one that
  reproduces the image, which is the failure this whole contract exists to prevent. sc-15954
  records the decision in full.

"Changed" is measured against the bitmap, not against the unsaved-edits pill — a crop, a colour
grade, an AI op, a second layer, a moved or faded or blended layer all fall out of the first case,
and undoing back to the opened image falls back into it. Note the consequence for a source over
the editor's canvas ceiling: it is resampled on load, so the canvas is a proxy, and the untouched
download hands back the **original** file — at its original resolution — rather than the smaller
one on screen. That is what stops a recipe claiming a resolution the file it travels in does not
have, and it is why the over-ceiling banner on the canvas states which size the next click writes.

The first case is a genuine egress the editor did not use to have — an authored prompt now leaves
in a file that previously lost it — which is why the header names it rather than only naming the
loss. To send that image without the recipe, use **Save a copy without the workflow** on it in the
Library; saving from the editor rasterizes, so a copy saved there has already lost the recipe along
with everything else the file carried.

**The editor does not decide whether a file carries a recipe — the reader does.** There is one
implementation of that question, `read_workflow_chunk`, and the editor reaches it over
`POST /api/v1/workflows/inspect` rather than walking chunk framing itself. That answer is a round
trip, so the header has two more states than the two above, and neither may be shown as *no
recipe*: while the answer is outstanding it reads *Checking for a recipe…*, and when the reader
could not produce one — the app is offline, the file is past the endpoint's own size cap, the file
claims a workflow this build refuses to read — it reads *Recipe unknown*. Not knowing and knowing
there is nothing are different facts, and only one of them is safe to stay quiet about.

The strip itself refuses rather than guesses. A PNG whose chunk framing cannot be walked to an IEND,
or whose unwalkable trailing bytes carry `sceneworks:workflow`, is an error and not a copy: "here is
your file without the workflow" answered with the original, or with a truncated file that does not
decode, are both worse than saying no. The server route bounds what one request may cost — it serves
the kept spans as views into the single buffer it read, behind a small semaphore and a size cap — and
revalidates the stripped representation on its own ETag alone, because it shares a modification date
with the full file and a date cannot tell the two apart.

A one-time disclosure is shown the first time a generation is submitted while embedding is on. It
is dismissible and never repeated; the "already told them" flag is `workflowEmbedNoticeSeen` in the
same preference file, durable rather than browser-side because the desktop shell serves the UI from
an origin that changes each launch. An absent flag means "not told", so an install upgrading into a
build that embeds gets the notice once rather than never. The durable write retries and then reports
its failure rather than swallowing it — a dropped PUT does not degrade to "saved locally" when the
mirror is wiped every launch — and a dismissal in one browser tab carries to the others through the
mirror's `storage` event, so "once" is per user rather than per tab.

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
| `ceilings` | Observed behaviour for the collection and string bounds — the largest input that survives, the smallest that does not, and whether the overflow **truncates** or **drops**, which is read out of the last column rather than assumed — and the constants for the envelope and PNG ceilings. |
| `builders` / `deferred-builders` | `ADVANCED_BUILDERS` and `DEFERRED_ADVANCED_BUILDERS`, file path and function name. |
| `write-seams` | `WORKFLOW_WRITE_SEAMS`, file path and function name, plus the disposition word (`Embeds` / `Conduit` / `Declines` / `Inert`) read out of the third column — so a seam that switches from declining to embedding cannot leave this table describing the old behaviour. A seam embedding for a builder this document lists as **deferred** fails separately. |
| `builder-fields` | The fields of the `AdvancedBuilder` struct, so a registry entry that grows a field a contributor is never told to fill fails. |
| `rule-helpers` | The `const fn … -> AdvancedKeyRule` constructors in `workflow_share.rs`, so a helper this section does not mention fails. |
| `lints` | Each named test exists in `crates/sceneworks-core/tests/workflow_share.rs`. |

The web copy is pinned against the same blocks, so the summary a user reads before sharing cannot
outrun the contract:

| Claim in `apps/web/src/workflowEmbed.js` | Pinned against |
| --- | --- |
| `EMBEDDED_PROSE_FIELDS` — the fields that travel exactly as typed | `prose-fields`, both directions. |
| `WORKFLOW_FIELDS_IN_FILE` — the "Also in the file" list | `envelope-fields` plus `advanced-shared` (prefixed `advanced.`), both directions. A new allow-listed setting cannot reach a shared image until someone decides, in the copy, what to say about it. |
| `WORKFLOW_FIELDS_NOT_IN_FILE` — the "Not in the file" list | Every `advanced-withheld` key must appear, and **no** key it names may appear in either table above. That second half is the direction that is a leak rather than an omission: it fires the day a withheld key becomes shared and the paragraph goes on promising it stays home. |
| `SAVE_WITHOUT_WORKFLOW_LABEL` | This document must name the control with the same string the UI does. |
| `PRODUCER_URL` / `WORKFLOW_SHARE_DOC_URL` | The Rust `PRODUCER_URL`, whole; the doc link is derived from it rather than written out beside it. |

Three claims outside a pinned block are checked too, because each had a way to drift silently:

- the **16 KiB** the privacy callout quotes is asserted to be the `PROSE_MAX_BYTES` row of the
  `ceilings` table, which is itself measured — so the callout cannot restate a ceiling the code
  stopped having;
- the pose **coordinate arrays** the callout-adjacent bullet and the `poses` row name are asserted
  to be exactly `POSE_FIELDS`;
- the number of prose fields the callout says it lists ("These six fields") is asserted against the
  row count of the table under it.

What is **not** machine-checked is the prose in the right-hand "what it is" columns and the narrative
sections. Those are read against the source, not derived from it. The version-behaviour table, the
trust-boundary list and the setting's semantics are prose; the modules they describe are
`workflow_share.rs`, `workflow_png.rs`, `project_store.rs` and `app_paths.rs`.
