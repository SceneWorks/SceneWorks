# Epic 2123 real-weight A/B harness

`scripts/epic-2123-ab.mjs` measures what each epic 2123 training technique does to a character
LoRA. It trains one adapter per technique row on the same dataset, seed, step count and resolution.
It then renders a fixed prompt grid with every adapter and with the bare base model, scores the
grids, and writes a report. Story: sc-2124.

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
| `samples/<model>/<row>/` | `<prompt>_s<seed>.png` and `prompts.json` |
| `eval/<model>/<row>.json` | The full `lora_eval_harness` report |
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

1. **`setup`** (CPU). Creates the project and imports the dataset (default `~/Datasets/Kelsie`).
   Each `<stem>.png` is uploaded through `POST /projects/:pid/training/uploads`. The captions come
   from the matching `<stem>.txt`, and every item carries the trigger word (`--trigger`, default
   `kelsie`). The dataset is created with `POST /projects/:pid/training/datasets`. Setup then
   resolves every base and auxiliary model by catalog id through `GET /api/v1/models`, and prints
   which are already in the cache, which would be downloaded, and their sizes. Missing models are
   fetched with `POST /api/v1/models/:id/download` only when you pass `--allow-downloads`.
2. **`masks`** (GPU). Runs the dataset's SAM3 subject-mask job. The `subject_mask` row needs it.
3. **`baseline-timing`** (GPU). Runs one Z-Image baseline with every technique off, at the A/B
   settings. It records wall time (job start to finish) and s/step, measured from the worker's
   `Training step N of M` progress clock. The Z-Image `baseline` row reuses this run when the
   configs are identical.
4. **`train --model zimage|sdxl`** (GPU). Trains the model's grid one row at a time. If the target
   does not advertise a row's `limits.supports*` flags, the row is recorded as skipped.
5. **`samples --model …`** (GPU). Renders 8 prompts × 2 seeds through `POST /api/v1/image/jobs`
   for the base model and for every trained adapter (`loras: [{id, weight}]`). Every row uses the
   same settings.
6. **`eval --model …`**. Computes each adapter's stable rank on the CPU, every run. With
   `--confirm-gpu` it also runs `lora_eval_harness::harness::eval_lora_outputs` on each row's
   samples, using the dataset images as the reference pool. That test is `#[ignore]` and loads
   CLIP ViT-L/14 and SCRFD/ArcFace on MLX. It reports ArcFace likeness, CLIP prompt adherence,
   same-prompt and overall spread, sharpness, and face-detect rate.
7. **`report`** (CPU). Writes `results.json`, a markdown table of rows × metrics with deltas against
   the baseline row, and one sample grid PNG per model. The grid has one row per technique and one
   column per prompt at the first seed, and is stitched with Python PIL.

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

Every row starts from the same config: the target's own defaults, with these knobs pinned.

| Knob | Value |
| --- | --- |
| steps | 400 |
| rank / alpha | 16 / 16 |
| learning rate | 1e-4 |
| seed | 7 |
| batch | 1 |
| `saveEvery` | the step count |
| `sampleEvery` | 0 (no in-training samples) |

The defaults kept from the target include the optimizer, timestep schedule, precision and caching.
Z-Image trains at 512 px. SDXL trains at 768 px, because its target only allows 768 and 1024. Each
row overlays its technique keys on top of that config:

| Row | `advanced` overlay |
| --- | --- |
| `baseline` | none |
| `weight_noise_0p005` / `0p0125` / `0p02` | `weightNoiseSigma` 0.005 / 0.0125 / 0.02 |
| `resolution_buckets` | Z-Image 512/768/1024 × 4:2:1; SDXL 768/1024 × 2:1 |
| `subject_mask` | `subjectMaskLoss`, background 0.1, subject 1.0 |
| `depth_anchoring` | `depthAnchoringWeight` 0.1, DA2 `--depth-model` (default `small`) |
| `identity` | `identityLossWeight` 0.05 + `faceLandmarkLossWeight` 0.05 |
| `body` | `bodyProportionWeight` 0.1 + `bodyShapeWeight` 0.05 + `normalWeight` 0.05 |
| `latent_perceptual` | `vaeAnchorWeight` 0.5 + `latentLpipsWeight` 0.5 |

The weights start from the sceneworks-core suggested values, which are themselves the upstream
ai-toolkit-perceptual starting points. Where upstream gives a range, the harness uses the modest
end. Noise windows, alternation periods and gates keep the shared parser's upstream defaults.

Bucket rows must be resolutions the target trains at; submit-time validation refuses 384 on
Z-Image and 576 on SDXL. So the bucket ladder starts at the A/B resolution and climbs through the
target's allowed edges.

## Commands

```sh
# CPU only. Sets up an isolated tree under <root>/dry-run and submits every row of both
# models with dryRun:true, so the API runs full submit-time validation (target limits, every
# technique parser, plan build). It also checks the mask and sample routes, self-tests the
# stable-rank math, and renders the report.
node scripts/epic-2123-ab.mjs --dry-run

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
- `--steps`, `--rank`, `--lr`, `--seed`, `--batch` and `--resolution` change the A/B settings.
- `--seeds 1001,2002` and `--lora-weight` control the sample render.
- `--port` sets the API port (default 8766).
- `--eval-release` builds the eval test in release.

## Caveats

- The dry run validates every row at API submit time. The worker's own preflight is stricter: it
  checks that the subject masks exist and that the auxiliary models are installed. That preflight
  first runs when the GPU phases run.
- The eval reference pool is the training dataset itself, not a held-out set. ArcFace likeness
  therefore measures fidelity to the training identity, not generalization.
