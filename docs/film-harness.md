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

## Sound (sc-22712)

The pack's `sound` array maps roles to approved audio files (`dialogue` / `ambience` / `music` /
`sfx`), in the same role namespace as the references but placed through different slots — sound is
never conditioning. The plan's `sound` block then places them:

```jsonc
"sound": {
  "generatedAudio": "mute",                 // run-level policy for the takes' OWN audio
  "dialogue": { "gain": 1.0, "muted": false },
  "ambience": { "role": "workshop_room_tone", "gain": 0.35, "fadeInSeconds": 1.0 },
  "music":    { "role": "main_theme", "gain": 0.2, "startSeconds": 0.0 }
}
```

and a shot places its own line with `"dialogueClip": { "role": ..., "offsetSeconds": ... }`, where
the offset is measured **from the start of that shot** rather than from the head of the sequence.
A role placed on the wrong bus (an `ambience` slot pointing at a `dialogue` entry) is a finding
before any job exists, because a plan like that renders, exports, and simply sounds wrong.

The assembled timeline therefore has four tracks: `track_main` (picture), `track_dialogue`,
`track_ambience` and `track_music`. Each audio track is a bus with its own `gain` and `muted`;
**the beds are placed once for the whole sequence**, so they play straight through the cuts instead
of restarting at each one. The export mixes every non-muted audio track — gain is
`track.gain * item.volume`, clips are delayed to where they land in the exported picture, per-item
`fadeInSeconds`/`fadeOutSeconds` become `afade`, the summed mix passes through a limiter so the
timeline's gains cannot add up past the encoder's ceiling and clip, and the mix is padded so the
file ends on the last picture frame. Sound never extends the export: a clip that would overrun the
picture is shortened.

**Timeline seconds are not picture seconds when a shot carries a crossfade.** The picture pass
overlaps a crossfaded shot with the one before it, so the exported file is shorter than the timeline
by the crossfade duration once per transition and every shot after a transition begins that much
earlier in the file. The mix is placed against the picture that was actually built: the ceiling is
the real exported length, and a clip keyed to a shot is delayed to where that shot IS, not to where
the timeline says it starts. Two 2 s shots with a 1 s crossfade export as a 3.000 s file with the
second shot's line at 1.0 s. Only the picture track's items can contribute their own generated
audio; an overlay item is never rendered into the picture, so its take's audio is not mixed in.

**Generated clip audio** (`generatedAudio`, `include` | `mute`, default `mute`) decides whether a
take's own audio joins the mix. The default is the doubling guard: a take whose model spoke the
line and a placed dialogue clip for the same beat would otherwise both be heard, with nothing in
the documents saying which was meant. A shot may override the run-level setting. The resolved value
is written into the timeline item and into `run.json`, so the export obeys the saved document and
the record says what it obeyed.

The fixture's clips are deterministic placeholder tones (`film-harness fixture-sound`) — one
frequency per role, so the three buses are distinguishable by ear when checking an export. Replace
a file with real audio and bump the pack's `version`.

### What this did and did not settle for SC-12807

[SC-12807](https://app.shortcut.com/trefry/story/12807) owns full audio-track support for the video
editor. This story took the two engine-side items it listed — **multiple audio tracks** and
**mixing every non-muted audio track into the export**, including honouring the per-item `volume`
that had been validated and read by nothing — and added per-track `gain`/`role` and per-item audio
fades to the timeline schema on the way. Those are done for every timeline, not only a harness one.

What SC-12807 still owns is the **editor** half, untouched here: audio assets in the media bin,
placing and trimming audio items from the timeline UI, and controls for a track's gain and mute.
The editor can create an audio track today and the backend will mix whatever is on it, but nothing
in the UI puts a clip there or moves a fader. This story is not that audit.

## Editing the assembled sequence

Three subcommands change a saved sequence without re-rendering anything. Each reads the run record,
edits the timeline through the same API the editor uses, re-lays the sequence, and rewrites the
record; `--export` also re-runs the MP4 export.

```text
film-harness trim         --run RUN.json --shot SH010 --source-in 1.0 --source-out 3.0
film-harness reorder      --run RUN.json --order SH020,SH010
film-harness replace-take --run RUN.json --shot SH010 --asset asset_...
```

All three change one input to a single re-layout pass: picture items are laid end to end in cut
order at whatever span their own source range implies (a trim ripples; there are no holes), each
dialogue clip returns to its shot's new start plus the offset it has always had, and both beds
re-span the new duration. Shot → asset links are never rebuilt from the plan — the picture item
carries its own shot id and version history, so `replace-take` appends to that history rather than
overwriting it, and the take that was there stays addressable. A reorder that does not name every
shot exactly once is refused rather than silently dropping one.

`trim` measures the take before it cuts, the same way `replace-take` measures its replacement: an
out point past the end of the media is clamped to the take's real length, and a range that starts
past the end is refused with the length named. An unbounded out point used to be accepted in
silence and produced a sequence longer than the file it points at — picture shorter than the saved
timeline, sound drifting against it, and a duration in the render sidecar that no file has.

A re-layout only re-places what the harness placed. A dialogue, ambience or music item the harness
put there is re-derived from the plan (and dropped when the re-layout leaves it no room, because the
plan can put it back); an audio clip the harness did not place — the editor's own — is clamped into
the new duration and **kept**, and dropped only when it starts past the end of the re-laid sequence.

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
target/debug/film-harness trim         --run RUN.json --shot ID [--source-in S] [--source-out S]
                                       [--export] [--api URL] [--poll-seconds N]
target/debug/film-harness reorder      --run RUN.json --order A,B,C     [--export] [--api URL]
target/debug/film-harness replace-take --run RUN.json --shot ID --asset ASSET_ID [--export]
target/debug/film-harness fixture-images --out DIR
target/debug/film-harness fixture-sound  --out DIR
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
MiniMax-H3 q4 (MLX), and tears both down. The fixture's placeholder plates and clips are
deterministic (`fixture-images`, `fixture-sound`), so the checked-in PNGs and WAVs are reproducible
byte for byte — the clips are integer triangle waves with no floating point anywhere, because a
sine's last ULP differs between platforms and that is enough to break a byte-for-byte check.

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
