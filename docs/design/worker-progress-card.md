# `WorkerProgressCard` — Design Spec

> Epic [sc-2080](https://app.shortcut.com/trefry/epic/2080) — Worker Progress Redesign
> Drives slices sc-2083 (skeleton), sc-2084 (thumbnail variants), sc-2087 (Job Title), sc-2088–sc-2092 (migrations).
> Living document — updated as design decisions land. Last update: 2026-05-28.

## Purpose

Single shared component that renders worker/job progress everywhere in the SceneWorks UI. Replaces today's `JobProgressCard` (`apps/web/src/components/JobProgress.jsx`), `JobRow` (`apps/web/src/screens/QueueScreen.jsx`), and `TrainingLiveJobCard` (`apps/web/src/screens/TrainingStudio.jsx`).

## Component layout

The card is a vertical stack of rows. **Every consumer renders the same skeleton.** Only the thumbnails region varies by variant.

```
┌─────────────────────────────────────────────────────────────────────┐
│ [JobType chip]                                       [Status badge] │  Row 1 — Header
│ Job Title                                            Job ID (mono)  │  Row 2 — Title
│ [CPU|GPU] [NVIDIA|Apple] [cuda|mps|mlx]  [Mem ====] [Load ======]  │  Row 3 — Hardware
│ Stage              Elapsed              Attempt                     │  Row 4 — Status
│ [Progress bar ████████████░░░░░░░░░░░░░░░░░░░░░░]                  │  Row 5 — Progress
│ [Cancel] [Retry] [View in Queue] [Duplicate]                        │  Row 6 — Actions
│ [Thumbnails region — variant-specific]                              │  Row 7 — Thumbnails
└─────────────────────────────────────────────────────────────────────┘
```

### Row 1 — Header

- **Job Type chip** (left) — colored chip with one-word label per job type
- **Status badge** (right) — colored badge per status

**Job Type chip — palette**

| Job type (internal) | Chip label | Color (semantic) |
|---|---|---|
| `image_generate`, `image_edit`, `image_vqa`, `image_interleave` | `Generate Image` | `--accent-image` (cyan/blue) |
| `video_generate`, `video_extend`, `video_bridge` | `Generate Video` | `--accent-video` (purple) |
| `lora_train` | `Training Run` | `--accent-training` (orange) |
| `training_caption` | `Dataset Captioning` | `--accent-caption` (yellow) |
| `model_download`, `model_import`, `model_convert` | `Model Import` | `--accent-import` (grey) |
| `lora_import` | `LoRA Import` | `--accent-import` (grey) |
| `prompt_refine` | `Prompt Refine` | `--accent-utility` (slate) |
| `person_replace` | `Person Replace` | `--accent-image` (cyan/blue) |

**Status badge — variants**

| Status (internal) | Badge label | Color |
|---|---|---|
| `queued` | `Queued` | grey |
| `running` | `Running` | green (pulsing) |
| `completed` | `Complete` | green (solid) |
| `canceled` | `Cancelled` | grey-dim |
| `failed` | `Failed` | red |
| `interrupted` | `Interrupted` | red-dim |

Terminal statuses (`completed`, `canceled`, `failed`, `interrupted`) are non-pulsing.

### Row 2 — Title

- **Job Title** (left) — human-readable, derived per type (see [Job Title rules](#job-title-rules))
- **Job ID** (right, monospace) — copyable on click; truncated middle-ellipsis if needed

### Row 3 — Hardware

Three sub-clusters left-to-right:

1. **Device** — pill: `CPU` or `GPU`
2. **Vendor** — pill: `NVIDIA` / `Apple` / blank for CPU
3. **Architecture** — pill: `cuda` / `mps` / `mlx` / blank for CPU
4. **GPU Mem** — bar with `XX%` label; tooltip shows `used / total`
5. **GPU Load** — bar with `XX%` label

**Live vs static meters:**

| Job status | Mem / Load source |
|---|---|
| `queued`, `running` | **Live** values from the global worker heartbeat (subscribed at page level; see sc-2082) |
| `completed`, `canceled`, `failed`, `interrupted` | **Static** `peakGpuMemoryPct` / `peakGpuLoadPct` from the job record (worker captures peaks during the run; see sc-2086) |

For CPU-only jobs (`prompt_refine`, etc.), the hardware row hides the GPU pills + meters and shows just `CPU` + worker name.

### Row 4 — Status

Three label/value pairs:

- **Stage** — short label, e.g. `Generating`, `Encoding`, `Sampling step 14/30`
- **Elapsed** — `mm:ss` or `Xh Ym`
- **Attempt** — `1/5` (only shown when attempt > 1 or status is failed/retry-eligible)

### Row 5 — Progress bar

Single 0–100 bar.

- **Determinate** — width = `job.progress`
- **Indeterminate** — when `job.progress` is null/undefined but status is `running`, render a moving sweep animation
- **Terminal states** — bar shows final value, no animation

### Row 6 — Action buttons

Visibility matrix (button shown only when checked):

| Button | `queued` | `running` | `completed` | `canceled` | `failed` | `interrupted` |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Cancel | ✓ | ✓ | — | — | — | — |
| Retry | — | — | ✓¹ | ✓ | ✓ | ✓ |
| Duplicate | — | — | ✓ | ✓ | ✓ | ✓ |
| View in Queue | ✓² | ✓² | ✓² | ✓² | ✓² | ✓² |

¹ Retry on completed = "re-run with same payload" (alias of Duplicate behavior; keep button visible for symmetry).
² Hidden when the component is already on the Queue screen.

**Additional rules:**

- Retry/Duplicate disabled when `attempts >= maxAttempts` (today: 5)
- Cancel disabled when `job.cancelRequested === true`; label switches to `Canceling…`
- Buttons are right-aligned on wide cards, full-width stacked on narrow (≤480px)

### Row 7 — Thumbnails region (variants)

Variant chosen by the consumer via prop `thumbnailsVariant`:

| Variant | Layout | Used by |
|---|---|---|
| `image-grid` | Larger thumbnails in a responsive grid (`minmax(120px, 1fr)`); shows interim + final | Image Studio, Character Turnaround, Training Studio |
| `video-player` | Single embedded `<video>` element with poster while encoding; controls visible on completion | Video Studio |
| `small-row` | Compact horizontal scroller of 48px square thumbnails | Queue Screen, batch contexts |
| `hidden` | Region not rendered at all (no empty box) | Caption + Model/LoRA Import + Prompt Refine |

**Thumbnail sourcing:**

- Final assets: `job.result.assets` (existing)
- Interim previews while running: `job.interimThumbnails[]` streamed via heartbeat (see sc-2085)
- All thumbnails clickable → existing full-asset modal/route

**Empty states:**

- `image-grid`, `small-row`: while job is `queued` show a placeholder skeleton row matching the expected output count if known (`payload.batchSize`); otherwise show a single shimmering tile
- `video-player`: poster placeholder with a small overlay icon

## Job Title rules

Server-side enrichment writes `job.title` onto the job record. The component reads it directly — no per-type logic in the UI. Implemented in sc-2087.

| Type pattern | Title format | Subject source |
|---|---|---|
| `lora_train` | `Training Run — <lora name>` | `payload.loraName` or LoRA lookup by `payload.loraId` |
| `training_caption` | `Dataset Captioning — <dataset name>` | `payload.datasetName` or dataset lookup by `payload.datasetId` |
| `image_generate`, `image_edit`, `image_vqa`, `image_interleave` | `Generate Image — <prompt>` | `payload.prompt`, truncated to 80 chars + ellipsis |
| `video_generate`, `video_extend`, `video_bridge` | `Generate Video — <prompt>` | `payload.prompt`, truncated to 80 chars |
| `person_replace` | `Person Replace — <prompt>` | `payload.prompt`, truncated to 80 chars |
| `model_download`, `model_import`, `model_convert` | `Model Import — <model name>` | `payload.modelName` or `payload.filename` |
| `lora_import` | `LoRA Import — <lora name>` | `payload.loraName` or `payload.filename` |
| `prompt_refine` | `Prompt Refine — <prompt>` | `payload.prompt`, truncated to 60 chars |

**Character Turnaround** is `image_generate` with a `payload.characterId` — title becomes `Character Turnaround — <character name>` (override applied when characterId is present).

**Fallbacks** (when the referenced entity is missing/orphaned):

- LoRA name missing → `Training Run — (deleted LoRA)` + tooltip with `loraId`
- Dataset name missing → `Dataset Captioning — (deleted dataset)`
- Prompt missing → fall back to `<JobType> — <job.id short>` (e.g. `Generate Image — a3f8e2…`)

**Prompt truncation:** find a word boundary near the limit; append `…`. Implementation in `apps/api` job enrichment middleware.

## Component API (target)

```tsx
type WorkerProgressCardProps = {
  job: Job;                                  // job record, see jobTypes
  thumbnailsVariant?: "image-grid" | "video-player" | "small-row" | "hidden";
  onCancel?: (job: Job) => void;
  onRetry?: (job: Job) => void;
  onDuplicate?: (job: Job) => void;
  onOpenQueue?: (job: Job) => void;          // omitted on the queue screen itself
  onThumbnailClick?: (asset: Asset) => void; // opens full asset view
  className?: string;
};
```

Live GPU stats come from a global heartbeat context (sc-2082), not props.

## CSS class naming

Adopt a single `worker-progress-card` prefix; namespace every sub-element. Migrate or delete:

- `.local-job-card`, `.local-job-main`, `.local-job-actions`, `.local-job-group`, `.local-job-stack` → `worker-progress-card[...]`
- `.job-row`, `.job-main`, `.job-meta`, `.job-actions` → `worker-progress-card[...]`
- `.training-live-card`, `.training-sample-grid` → `worker-progress-card[...]` + `worker-progress-card__thumbnails--image-grid`
- `.lora-import-progress`, `.lora-import-model-progress` → `worker-progress-card[...]` + `worker-progress-card__thumbnails--hidden`

Keep `.progress-track` and `.worker-meter` as low-level primitives (used inside the card).

## States — at-a-glance

| State | Header chip | Status badge | Progress bar | Meters | Buttons | Thumbnails |
|---|---|---|---|---|---|---|
| Queued, no worker | Job type | `Queued` grey | empty | live (worker not yet assigned → blank) | Cancel, View in Queue | skeleton |
| Queued, worker assigned | Job type | `Queued` grey | empty | live | Cancel, View in Queue | skeleton |
| Running | Job type | `Running` green pulse | determinate or indeterminate | live | Cancel, View in Queue | interim + final mix |
| Completed | Job type | `Complete` green | full | static (peak) | Retry, Duplicate, View in Queue | final |
| Cancelled | Job type | `Cancelled` grey-dim | last value | static (peak) | Retry, Duplicate, View in Queue | partial |
| Failed | Job type | `Failed` red | last value | static (peak) | Retry, Duplicate, View in Queue | partial + error message |
| Interrupted | Job type | `Interrupted` red-dim | last value | static (peak) | Retry, Duplicate, View in Queue | partial |

## Responsive behavior

- ≥ 768px: rows render as drawn above
- 480–768px: Hardware row wraps so meters drop below the device/vendor/arch pills
- < 480px: Title and Job ID stack vertically; action buttons become a full-width row

## Out of scope

- Mobile-specific gestures
- Drag-to-reorder in the queue
- Per-card customization (theming) beyond CSS variables
- New job types — this spec covers the existing enum from `apps/web/src/jobTypes.js`

## References

- Epic [sc-2080](https://app.shortcut.com/trefry/epic/2080)
- Today's components:
  - [apps/web/src/components/JobProgress.jsx](../../apps/web/src/components/JobProgress.jsx)
  - [apps/web/src/screens/QueueScreen.jsx](../../apps/web/src/screens/QueueScreen.jsx) — `JobRow`, `WorkerCard`
  - [apps/web/src/screens/TrainingStudio.jsx](../../apps/web/src/screens/TrainingStudio.jsx) — `TrainingLiveJobCard`
- Enums: [apps/web/src/jobTypes.js](../../apps/web/src/jobTypes.js)
