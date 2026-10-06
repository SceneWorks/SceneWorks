# Epic 2123 real-weight A/B harness

`scripts/epic-2123-ab.mjs` measures what each epic 2123 training technique does to a character
LoRA. It trains one adapter per technique row from the app's own character preset, on the same
dataset, seed, step count and resolution. It then renders a fixed prompt grid with every adapter
(and, on request, every intermediate checkpoint) and with the bare base model, scores the grids,
and writes a report. Story: sc-2124.

The harness drives the real `sceneworks-rust-api` and the native worker. It is the only path that
accepts the epic's `config.advanced` fields. The `lora_train_driver` test builds a default
`TrainingConfig` and cannot set them.

## Isolation

Everything the harness writes goes under `--root`, which defaults to
`~/.cache/sceneworks-epic-2123-ab`:

| Path | Contents |
| --- | --- |
| `data/`, `config/` | The API's `SCENEWORKS_DATA_DIR` and `SCENEWORKS_CONFIG_DIR` |
| `logs/` | One API log and one worker log per invocation |
| `samples/<model>/<row>/` | `<prompt>_s<seed>.png` and `prompts.json`; checkpoint sub-rows use `<row>@<step>/` |
| `eval/<model>/<row>.json` | The full `lora_eval_harness` report (`<row>@<step>.json` for a checkpoint) |
| `report/` | `results.json`, `report.md`, `grid-<model>.png` |
| `state.json` | Resume state: ids, job ids, finished rows, timings, metrics |

The desktop app's data directory is never used. Model weights come from the shared Hugging Face
cache. `HF_HOME` defaults to `~/.cache/huggingface`, which is how the app itself resolves models,
so a snapshot that is already cached shows as `installed` and is read in place. Nothing is copied,
linked or downloaded again. Use `--hf-home` to point at a different cache.

## GPU discipline

These phases run MLX work: `masks`, `baseline-timing`, `train`, `samples`, and the scoring step of
`eval`. Each one refuses to start unless you name it with `--phase` **and** pass `--confirm-gpu`.

- The native worker starts only for the phase that needs it, and runs one job at a time.
- GPU phases need the release binary. `--build` builds it and fetches the matching prebuilt
  libmlx. `--allow-debug-build` overrides the check.
- `--max-worker-rss-gb` cancels the job and stops the worker if its RSS grows past the cap. The
  default is 85% of RAM. RSS does not count every Metal allocation, so treat this as a backstop.
- On exit, Ctrl-C or SIGTERM the harness tears down in a fixed order:
  1. Cancel the active job.
  2. Send the worker SIGTERM, and only SIGTERM, so it finishes its current command buffer.
  3. Stop the API.

  The children run in their own process group, so a terminal Ctrl-C reaches only the driver.

## Phases

All phases resume. A finished row, sample or score is skipped on rerun unless you pass `--force`.
A job left running by an interrupted run is canceled before its row is resubmitted.

1. **`setup`** (CPU). Creates the project and imports the dataset (default `~/Datasets/Kelsie`)
   for the run's caption mode. Each `<stem>.png` that has a `<stem>.txt` is uploaded through
   `POST /projects/:pid/training/uploads`, and every item carries the trigger word (`--trigger`,
   default `kelsie`). The dataset is created with `POST /projects/:pid/training/datasets`. See
   [Captions](#captions) for what each item's caption text is. Setup then
   resolves every base and auxiliary model by catalog id through `GET /api/v1/models`, and prints
   which are already in the cache, which would be downloaded, and their sizes. Missing models are
   fetched with `POST /api/v1/models/:id/download` only when you pass `--allow-downloads`.
2. **`masks`** (GPU). Runs the SAM3 subject-mask job for the current caption mode's dataset. The
   `subject_mask` row needs it. Each caption mode's dataset has its own masks, so run it once per
   mode you train.
3. **`baseline-timing`** (GPU). Runs one Z-Image baseline with every technique off, at the A/B
   settings. It records wall time (job start to finish) and s/step, measured from the worker's
   `Training step N of M` progress clock. The Z-Image `baseline` row reuses this run when the
   configs are identical.
4. **`train --model zimage|sdxl`** (GPU). Trains the model's grid one row at a time. If the target
   does not advertise a row's `limits.supports*` flags, the row is recorded as skipped.
5. **`samples --model …`** (GPU). Renders 8 prompts × 2 seeds through `POST /api/v1/image/jobs`
   for the base model and for every trained adapter (`loras: [{id, weight}]`, weight
   `--lora-weight`, default 0.8). Every row uses the same settings. With `--checkpoints` it also
   renders every intermediate checkpoint of each row; see [Checkpoints](#intermediate-checkpoints).
   A row whose adapter, LoRA weight or render size changed since its samples were rendered is
   re-rendered from scratch, so one grid row never mixes renders.
6. **`eval --model …`**. Computes the stable rank of each adapter and each intermediate
   checkpoint on the CPU, every run. With
   `--confirm-gpu` it also runs `lora_eval_harness::harness::eval_lora_outputs` on each row's
   samples (checkpoint sub-rows included), using the dataset images as the reference pool. A row
   whose samples were re-rendered since its last score is re-scored. That test is `#[ignore]` and loads
   CLIP ViT-L/14 and SCRFD/ArcFace on MLX. It reports ArcFace likeness, CLIP prompt adherence,
   same-prompt and overall spread, sharpness, and face-detect rate.
7. **`report`** (CPU). Writes `results.json`, a markdown report and one sample grid PNG per model.
   Each model section opens with a header table giving every trained row's caption mode, preset
   id, resolution (or bucket ladder), steps, rank/alpha, learning rate and training-adapter
   version. A table of rows × metrics follows, with deltas against the baseline row's final
   adapter. Checkpoint sub-rows (`<row> @ step N`) sit directly above their row's final adapter in
   both the table and the grid. The grid has one row per technique (or checkpoint) and one column
   per prompt at the first seed, and is stitched with Python PIL.

   `results.json` keeps each row's `finalLoss`. It is the **last logged step loss (may be an
   aux-only step)**: the loss in the last `Training step N of M (loss …)` progress message the
   poller saw. On a row that alternates auxiliary-loss steps with diffusion steps, that step can be
   an auxiliary-only step, so it is not a convergence figure to compare across rows.

`--phase status` prints where the run stands.

### Stable rank

Stable rank is ‖ΔW‖_F² / ‖ΔW‖_2², where ΔW = B·A for each LoRA module. The reported value is the
mean over modules, with median, min and max alongside. It is computed in Node, inside the
harness, without materializing ΔW:

- ‖ΔW‖_F² is the exact trace of (BᵀB)(AAᵀ), which needs only the r×r Gram matrices.
- σ_max² comes from power iteration on ΔWᵀΔW, applied implicitly.

The harness reads the PEFT layout the native trainers write (`.lora_A.weight` [r, in] and
`.lora_B.weight` [out, r]), and kohya's `lora_down`/`lora_up`. Stable rank does not depend on
scale, so the stored alpha does not matter. `--stable-rank <file>` prints the value for any LoRA
file.

The dry run checks the math against synthetic adapters with known answers: rank 1 gives 1,
isotropic rank 4 gives 4, and singular values (3, 1) give 10/9. On a real 400-module LoRA it
matched a numpy QR+SVD reference to about 1e-10.

## The grid

Every row starts from the same config: the target's default **character preset**, fetched from
`GET /api/v1/training/presets` (the route the Studio reads). The default is the target's preset
with `recommendedFor: character` and `ui.default: true`:

| Model | Preset | Resolution | Steps | Rank / alpha | LR | Optimizer | Training adapter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Z-Image | `z_image_turbo_lora.character.adamw8bit.balanced` ("Character balanced") | 1024 | 3000 | 16 / 16 | 1e-4 | adamw8bit | `ostris/zimage_turbo_training_adapter` `v2-default` |
| SDXL | `sdxl_lora.character.adamw8bit.balanced` ("Character balanced") | 1024 | 1500 | 16 / 16 | 1e-4 | adamw8bit | none |

`--preset <id>[,<id>]` replaces the default for the model whose target the preset belongs to; an
id that is not in the API's list, or belongs to another target, is refused.

Every preset key is kept as the API returns it, including `advanced.trainingAdapterRepo` /
`trainingAdapterVersion`, the timestep type and bias, precision, caching, weight decay and LoRA
target modules. Only these are overridden:

| Knob | Value |
| --- | --- |
| `seed` | 7 (`--seed`) |
| `triggerWord` | the trigger (`--trigger`) |
| `advanced.sampleEvery` | 0 (no in-training samples) |
| `advanced.requestedGpu` | `auto` |
| `saveEvery` | `--save-every`, default the step count (final adapter only) |
| `steps`, `rank`, `learningRate`, `batchSize`, `resolution` | only when `--steps`, `--rank`, `--lr`, `--batch` or `--resolution` is passed; otherwise the preset's value |

`--rank` keeps the preset's alpha/rank ratio (both presets use alpha = rank). The dry run prints
the resolved baseline config for each model. Each row overlays its technique keys on top of that
config:

| Row | `advanced` overlay |
| --- | --- |
| `baseline` | none |
| `weight_noise_0p005` / `0p0125` / `0p02` | `weightNoiseSigma` 0.005 / 0.0125 / 0.02 |
| `resolution_buckets` | the bucket ladder (below) |
| `subject_mask` | `subjectMaskLoss`, background 0.1, subject 1.0 |
| `depth_anchoring` | `depthAnchoringWeight` 0.1, DA2 `--depth-model` (default `small`) |
| `identity` | `identityLossWeight` 0.05 + `faceLandmarkLossWeight` 0.05 |
| `body` | `bodyProportionWeight` 0.1 + `bodyShapeWeight` 0.05 + `normalWeight` 0.05 |
| `latent_perceptual` | `vaeAnchorWeight` 0.5 + `latentLpipsWeight` 0.5 |

The weights start from the sceneworks-core suggested values, which are themselves the upstream
ai-toolkit-perceptual starting points. Where upstream gives a range, the harness uses the modest
end. Noise windows, alternation periods and gates keep the shared parser's upstream defaults.

### Bucket ladder

Bucket rows must be resolutions the target trains at; submit-time validation refuses 384 on
Z-Image and 576 on SDXL. The ladder is every edge in the target's `limits.resolutions` up to and
including the run's base resolution, smallest first. Repeats follow the upstream 2^k skew, so the
smallest edge is visited most. At the presets' 1024 base:

- Z-Image: 512 / 768 / 1024 at 4:2:1.
- SDXL: 768 / 1024 at 2:1.

When fewer than two allowed edges sit at or below the base, the ladder climbs from the base
instead. For example, `--resolution 512` on Z-Image gives 512 / 768 / 1024 at 4:2:1, and
`--resolution 768` on SDXL gives 768 / 1024 at 2:1. The bucket row runs the same number of steps
as the others, so most of its steps are at the smaller edges.

## Captions

`--captions trigger|full` (default `trigger`) picks how each dataset item is captioned:

- `trigger`: the caption text is the trigger word alone. The dataset's `.txt` captions describe
  the subject's appearance ("Kelsie, a blonde woman with wavy hair…"), so with them the identity
  binds to those caption words instead of the trigger, and a prompt that names only the trigger
  does not recall the subject.
- `full`: the `.txt` text, as before. The trigger is already in that text, so the plan does not
  prepend it again.

Each mode is its own dataset, stored in `state.json` under `datasets.<mode>` with its own subject
masks, so both can exist side by side. A stored dataset is reused only when its mode, source
directory and trigger all match. Every training row records its `captionMode` and `datasetId`,
and both are part of its resume key, along with the preset id and version. Switching mode, or
moving from the old 512 px / 400-step configs, therefore retrains a row rather than reusing it.

State written before caption modes existed held one dataset imported with the full captions. It
is read as the `full` dataset.

## Intermediate checkpoints

`--save-every N` sets the trainer's `saveEvery`. The default is the step count, which writes only
the final adapter.

Every native trainer writes `<stem>-step<NNNNNN>.safetensors` into the row's LoRA output dir
(`<project>/loras/<loraId>/`, beside the final `<stem>.safetensors`) every N steps, but not at
the final step. Beside each one it writes a `.resume.safetensors` (Candle) or `.optim.safetensors`
(MLX) snapshot, which the harness ignores. The job result lists the steps under `checkpoints`.

Only the final adapter is registered as a LoRA: `register_trained_lora` registers the file the
plan declared, never a step checkpoint. An image job can only reference a registered LoRA id. So
`samples --checkpoints` imports each checkpoint through the app's own route,
`POST /api/v1/loras/import`, with a local `sourcePath`, `scope: project`, and the trained row's
family and base model. The API's in-process CPU utility worker runs the import, which copies the
file into `<project>/loras/imports/<id>/` and registers it. The harness then renders the
checkpoint as sub-row `<row>@<step>` through `POST /api/v1/image/jobs`, like any other adapter.
The imported id carries the training job id, so a retrained row never resolves to an older run's
checkpoint. The dry run runs this whole path on a synthetic checkpoint.

## Time estimate

Seconds per step scale roughly with the pixel count of the training resolution. The measured
Z-Image baseline on this Mac was **1.63 s/step at 512 px**. At the preset's 1024 px, expect about
4× that, roughly 6.5 s/step. The Z-Image preset's 3000 steps then take about 5.4 h per row, so
about 54 h for the ten-row grid. Technique rows that add auxiliary models run slower than the
baseline. The bucket row's cost depends on its mix of edges. Use `--steps` and `--rows` to size a
run, and run `baseline-timing` first for a measured figure at the settings you chose.

## Commands

```sh
# CPU only. Sets up an isolated tree under <root>/dry-run with both caption datasets, prints each
# model's resolved baseline config, and submits every row of both models with dryRun:true, so the
# API runs full submit-time validation (target limits, every technique parser, plan build). It
# checks that every submitted key reaches the plan unchanged and that the plan's captions match
# the caption mode. It also checks the mask and sample routes and the checkpoint import + sample
# path on a synthetic checkpoint, self-tests the stable-rank math, and renders the report.
node scripts/epic-2123-ab.mjs --dry-run

# Checkpoint curve: save every 500 steps, then render each checkpoint.
node scripts/epic-2123-ab.mjs --phase train --model zimage --save-every 500 --confirm-gpu
node scripts/epic-2123-ab.mjs --phase samples --model zimage --checkpoints --confirm-gpu

# One-time setup. Add --allow-downloads to fetch the missing auxiliary models (about 2.2 GB).
node scripts/epic-2123-ab.mjs --phase setup --build --allow-downloads

# GPU phases, one at a time.
node scripts/epic-2123-ab.mjs --phase masks --confirm-gpu
node scripts/epic-2123-ab.mjs --phase baseline-timing --confirm-gpu
node scripts/epic-2123-ab.mjs --phase train --model zimage --confirm-gpu
node scripts/epic-2123-ab.mjs --phase samples --model zimage --confirm-gpu
node scripts/epic-2123-ab.mjs --phase eval --model zimage --confirm-gpu
node scripts/epic-2123-ab.mjs --phase train --model sdxl --confirm-gpu
node scripts/epic-2123-ab.mjs --phase samples --model sdxl --confirm-gpu
node scripts/epic-2123-ab.mjs --phase eval --model sdxl --confirm-gpu
node scripts/epic-2123-ab.mjs --phase report
```

Useful flags:

- `--rows a,b` limits `train` and `samples` to some rows.
- `--preset <id>[,<id>]` replaces a model's default character preset.
- `--captions trigger|full` picks the caption mode (default `trigger`).
- `--steps`, `--rank`, `--lr`, `--batch` and `--resolution` override the preset's values.
  `--seed` sets the training seed (default 7).
- `--save-every N` writes a checkpoint every N steps (default: the step count, final only).
- `--checkpoints` makes `samples` render every intermediate checkpoint as a `<row>@<step>`
  sub-row.
- `--seeds 1001,2002`, `--lora-weight` (default 0.8) and `--gen-size <px>` control the sample
  render. `--gen-size` sets a square width and height; the default is 1024 for both models.
- `--port` sets the API port (default 8766).
- `--eval-release` builds the eval test in release.

## Caveats

- `train` and `baseline-timing` must get the same config flags (`--preset`, `--captions`,
  `--steps`, `--rank`, `--lr`, `--batch`, `--resolution`, `--save-every`, `--seed`) on every
  invocation. Each changes a row's resume key, so a later run with different flags retrains the
  rows it touches instead of skipping them.

- The dry run validates every row at API submit time. The worker's own preflight is stricter: it
  checks that the subject masks exist and that the auxiliary models are installed. That preflight
  first runs when the GPU phases run.
- The eval reference pool is the training dataset itself, not a held-out set. ArcFace likeness
  therefore measures fidelity to the training identity, not generalization.
