# FFmpeg Timeline Export — Minimum Viable Feature Set (sc-1175)

> **Story:** [sc-1175 — Define minimum viable FFmpeg timeline export features](https://app.shortcut.com/trefry/story/1175)
> **Epic:** [1093 — SceneWorks: Research Tracks](https://app.shortcut.com/trefry/epic/1093)
> **Last updated:** 2026-06-18
> **Status:** Validated — the v1 export path is exercised against the real renderer code and ffmpeg 8.1.2.

**Provenance legend:** ⚙️ = empirically spiked on this machine (ffmpeg 8.1.2, macOS 26.5) · 📄 = grounded in shipped renderer code (`crates/sceneworks-worker/src/media_jobs.rs`, `crates/sceneworks-core/src/contracts.rs`).

## Decision

v1 timeline export is a **backend FFmpeg render of a SceneWorks-owned timeline model** — no
browser/canvas recording, no third-party editor SDK. Each main-track item is rendered to a
normalized intermediate segment, then the segments are muxed (stream-copy concat, or `xfade`
when a crossfade is present). The exact filter chains the renderer emits were replicated
standalone and all succeed on ffmpeg 8.1.2.

## Empirical validation ⚙️

Replicated the renderer's exact ffmpeg invocations (below) against synthetic assets at a
1280×720@24 target. All five primitives produced valid output:

| Export primitive | Renderer source | Result |
|---|---|---|
| Still → fixed-duration clip (`-loop 1` + scale/pad/fps + fades) | `media_jobs.rs:2232-2253` | ✅ 1280×720@24, 4.000 s |
| Video trim + speed (`-ss`/`-t` + `setpts=(1/speed)*PTS`) | `media_jobs.rs:2256-2278` | ✅ trims + retimes |
| Concat mux (`-f concat -safe 0 -c copy`) | `media_jobs.rs:2296-2314` | ✅ 7.667 s join, stream-copy |
| Crossfade mux (`xfade=transition=fade`) | `media_jobs.rs:2361-2392` | ✅ 7.167 s (= Σ − 0.5 s overlap) |
| Frames → mp4 (`libx264 yuv420p`) + faststart + poster | `video_jobs.rs:743-895` | ✅ encode + `+faststart` + poster.jpg |

ffmpeg 8.1.2 has every required component: `libx264` encoder; `xfade`, `fade`, `setpts`,
`concat`, `scale`, `pad`, `acrossfade` filters.

## Required v1 export handling (acceptance criteria)

### Trims 📄⚙️
Per-item source trim via input seek + duration: `-ss {sourceIn} -i SRC -t {sourceDuration}`,
where `sourceDuration = max(sourceOut − sourceIn, 0.1)` (`media_jobs.rs:2182-2186, 2265-2270`).

### Still-image durations 📄⚙️
Stills are looped to the item's timeline duration: `-loop 1 -framerate {fps} -i IMG -t {duration}`
(`media_jobs.rs:2232-2249`). Source type is detected by asset `type`/`mimeType`
(`media_jobs.rs:2225-2231`); `duration` resolves from `timelineEnd − timelineStart`, falling back
to `sourceDuration / speed` (`media_jobs.rs:2188-2192`).

### Speed changes 📄⚙️
Video items prepend `setpts={1/speed:.6}*PTS` to the filter chain; `speed` is floored at 0.1
(`media_jobs.rs:2187, 2256-2260`). (Audio is dropped per-segment — see "not in v1".)

### Aspect ratio & resolution presets 📄⚙️
The timeline carries `aspectRatio`, `width`, `height`, `fps` (`contracts.rs:1190-1193`). Every
segment is normalized to the timeline frame with
`scale=W:H:force_original_aspect_ratio=decrease, pad=W:H:(ow-iw)/2:(oh-ih)/2:color=black, fps=N, format=yuv420p`
(`media_jobs.rs:2193-2204`) — letterbox/pillarbox to fit, never crop or distort. Frame-extract
(thumbnails) snaps to a `resolution` in `[240, 2160]` (default 720) and `fps` in `[1, 60]`
(default 30) (`media_jobs.rs:2037-2038`); the generation export path also snaps `target_width/height`
to a multiple per `contracts.rs:640-657`.

### Initial transitions 📄⚙️
Closed set — `TimelineTransitionType` is `Crossfade | FadeFromBlack | FadeToBlack`
(`contracts.rs:499-503`):
- **Fade from/to black** — applied in the per-segment chain: `fade=t=in:st=0:d={d}` /
  `fade=t=out:st={duration−d}:d={d}`, default 0.5 s, clamped to ≤ item duration
  (`media_jobs.rs:2205-2223`).
- **Crossfade** — applied at mux time: `xfade=transition=fade:duration={d}:offset={prev−d}`, with
  `d` clamped to **[0.1, 1.5] s** (`crossfade_duration`, `media_jobs.rs:2376-2395`). Any crossfade in
  the timeline switches the whole mux from stream-copy concat to the `xfade` filtergraph
  (`mux_segments`, `media_jobs.rs:2282-2294`).

## Explicitly NOT in v1

- **Advanced audio mixing** 📄 — segments are rendered with `-an` (`media_jobs.rs:2247, 2273`); there
  is no multi-track audio bed, ducking, level automation, or per-item gain in the export. (The
  generation encode path can mux a single AAC track via `-shortest`, `video_jobs.rs:769-783`, but
  that is single-source, not a timeline mix.)
- **Browser / canvas recording** 📄 — export is a backend FFmpeg render, not `MediaRecorder`/canvas
  capture. The browser editor is preview-only; the authoritative pixels come from the worker.
- **Deferred (post-v1):** rich multi-track compositing/overlays, transition types beyond
  fade/crossfade, GPU-accelerated encode, persistent undo across sessions, and generation-aware
  timeline hooks (bridge clips, frame extraction → extend, nondestructive replace).

## Renderer command reference (exact invocations) 📄

```
# still item            (media_jobs.rs:2232-2249)
ffmpeg -y -loop 1 -framerate {fps} -i STILL -t {duration} \
  -vf "scale=W:H:force_original_aspect_ratio=decrease,pad=W:H:(ow-iw)/2:(oh-ih)/2:color=black,fps={fps},format=yuv420p[,fade...]" -an SEG.mp4

# video item            (media_jobs.rs:2256-2278)
ffmpeg -y -ss {sourceIn} -i SRC -t {sourceDuration} \
  -vf "setpts={1/speed}*PTS,scale=...,pad=...,fps=...,format=yuv420p[,fade...]" -an SEG.mp4

# concat mux            (media_jobs.rs:2296-2314)   — non-crossfade joins, stream copy
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy OUT.mp4

# crossfade mux         (media_jobs.rs:2361-2392)
ffmpeg -y -i SEG0 -i SEG1 ... -filter_complex \
  "[i:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[vi]; \
   [cur][vi]xfade=transition=fade:duration={d}:offset={off},format=yuv420p[mixN]" -map "[mixN]" OUT.mp4
```

`run_ffmpeg` resolves the binary from `SCENEWORKS_FFMPEG` else `ffmpeg` on PATH
(`media_jobs.rs:2488-2504`).

## Rust backend target

- **Export job** 📄 — `timeline_export` is an enumerated `JobType` handled by the
  `timeline-exporter` worker role (`contracts.rs:148, 169, 256, 380`). FFmpeg execution stays behind
  the worker/job boundary (`media_jobs.rs`), never in the API or web layer.
- **Timeline serialization** 📄 — `Timeline` / `TimelineTrack` / `TimelineItem` / `TimelineTransition`
  are versioned contracts (`schema_version`, `contracts.rs:1185-1264`). The export payload field
  names are the camelCase serde forms the renderer reads directly: `sourceIn`, `sourceOut`,
  `timelineStart`, `timelineEnd`, `speed`, `transitionIn`, `transitionOut` (confirmed against
  `media_jobs.rs:2182-2206`). **No field-name divergence** between the contract and the renderer.
- **Asset lookup** 📄 — items reference `assetId`; the renderer resolves the on-disk media via the
  project path with `safe_project_path` and fails the job if the source is missing
  (`media_jobs.rs:2170-2180`).
- **FFmpeg invocation boundary** 📄 — all spawning funnels through `run_ffmpeg(args, context)`
  (`media_jobs.rs:2488`), so timeouts/progress/logging are centralized; the override env is
  `SCENEWORKS_FFMPEG`.
- **Progress events / output asset** 📄 — segments render to a temp dir, mux to the final mp4, then
  the render is registered as a normal video asset with lineage under the project (the per-item
  `render_item_segment` returns the realized duration for timeline accounting, `media_jobs.rs:2169`).
- **No new versioned contract change required** for the v1 feature set above — it is fully expressible
  on the existing `Timeline*` contracts and the `timeline_export` job. New transition types or an
  audio-mix model *would* require additive contract fields (the `#[serde(flatten)] extra` passthrough
  absorbs experiments without a breaking change).

## Sources

Empirical: standalone replication of the renderer chains on ffmpeg 8.1.2 (macOS 26.5, this machine).
Code: `crates/sceneworks-worker/src/media_jobs.rs` (timeline render/mux), `crates/sceneworks-worker/src/video_jobs.rs`
(generation encode), `crates/sceneworks-core/src/contracts.rs` (Timeline contracts, job/role enums).
Prior research: `documents/TIMELINE_LIBRARY_RESEARCH.md`.
