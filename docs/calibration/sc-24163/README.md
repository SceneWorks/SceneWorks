# sc-24163 — Qwen Image 2.1 LoRA/LoKr: CUDA real-weight evidence (epic sc-24107 Phase 2)

The CUDA half of the epic's terminal-story acceptance (A1–A5). The raw numbers behind every table
are in [`cuda-measurements.json`](cuda-measurements.json). Renders, adapters, logs and VRAM samples
were written outside the repository (`E:\sc24163\` on the capture host) and are not committed.
[`scratch/`](scratch/) holds the probe sources, stored as `.txt` so nothing
builds or lints them: the inference example binaries used for A5 (built in a throwaway inference
worktree at the pin and never committed there), the dataset generator, and the pixel-diff metric.

## Run identity

| | |
| --- | --- |
| SceneWorks | `190e0c4c7884bfa7898e0991728e91f75b049e18` (`feature/sc-24107-qwen-image-2-1-lora`), plus `599233825` (the edit-route fix below) for every edit render |
| inference pin | `457f1a37161d80ee91e138c80e9dac63a93d0868` |
| GPU / driver | 2× NVIDIA RTX PRO 6000 Blackwell Max-Q (97,887 MiB), driver 596.36, CUDA 12.9, `CUDA_COMPUTE_CAP=120`, MSVC 14.44 |
| bf16 weights | `Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758` |
| q8 / q4 weights | `SceneWorks/qwen-image-2-1-mlx@1691de01c24a070131e0a28bf4c065fd027f4fe9` |
| stack | release `sceneworks-rust-api` + `sceneworks-rust-worker` (`backend-candle`), worker pinned to one GPU (`SCENEWORKS_GPU_ID`/`CUDA_VISIBLE_DEVICES`), HF cache staged as hardlinks |
| date | 2026-10-03 |

## A1 — training through SceneWorks (Training Studio API → worker → library)

Both runs went through the full SceneWorks stack: `POST /api/v1/projects/:id/training/jobs` with
`licenseAcknowledged: true`, claimed by the candle worker, adapter registered in the project library.
Config: rank 16 / alpha 16, lr 4e-4, adamw8bit, 200 steps (the target's minimum), 768², save every
50, previews every 100 at 20 steps, gradient checkpointing on.

| | T2I LoRA (`qwen_image_2_1_lora`) | Edit LoKr (`qwen_image_2_1_edit_lora`) |
| --- | --- | --- |
| dataset | 8 captioned images (synthetic "zxq" ring style) | 6 instruction pairs, 1 reference each (invert + posterize) |
| wall | 503 s (steps 1–112) + 436 s (resumed 101–200) | 3,840 s (≈18.5 s/step) |
| loss | 0.023 at step 2, 0.008–0.04 mid-run, 0.0146 final (per-step flow-matching loss is noisy) | mean 0.0151 (steps 1–50) → 0.0149 → 0.0132 → 0.0133 (151–200), final 0.0065 |
| previews | step 100 and step 200, rendered from the live adapter; the style is visible at step 100 | step 100 and step 200, rendered as edits |
| checkpoint / resume | cancelled at step 112, then retried with `plan.config.advanced.resume=true`: same output dir, resumed at step 101 from `step000100.resume.safetensors`, completed | not repeated (resume was shown on the T2I run) |
| worker peak (`/jobs/:id/metrics`) | 33.3 GB (nvidia-smi, pool reserved) | 75.7 GB |
| library entry | family `qwen-image-2-1`, baseModel `qwen_image_2_1`, licence `Qwen RESEARCH LICENSE AGREEMENT` + URL, networkType `lora` | the same, networkType `lokr`, **`trainingMode: edit`** |
| safetensors `__metadata__` | `family`, `baseModel`, `ss_base_model_version=qwen_image_2_1`, research licence + notice | the same + `trainingMode=edit`, `decomposeFactor=-1` |

**The two-reference edit dataset is refused by the preflight, and that refusal is correct.** With 2
references per item at 768² the engine predicted ~95.5 GiB against an ~80.1 GiB budget and failed the
job before step 1. Running the same request directly with the budget disabled filled the card
(97,295 MiB) and took 516 s for step 1, which is a WDDM spill. Every 2.1 condition image is fitted to
1024 px, so each reference adds ~4k joint tokens to every training step. With one reference
(predicted ~51.1 GiB) the run fit and was used.

## A2 — trained adapters on CUDA at bf16 / q8 / q4 (SceneWorks image jobs)

768², 20 steps, seed 24163. The tier came from `advanced.mlxQuantize` (16/8/4), and the worker's
`image_memory_strategy_selected` event confirmed `actual_tier`. The edit renders use **two ordered
references** (the held-out source and the key image).

| pair (same seed) | mean \|Δ\| | pixels with \|Δ\|>16 | PSNR |
| --- | --- | --- | --- |
| T2I bf16 none vs LoRA 1.0 | 49.9 | 99.6 % | 11.1 dB |
| T2I q8 none vs LoRA 1.0 | 58.1 | 99.2 % | 9.7 dB |
| T2I q4 none vs LoRA 1.0 | 60.4 | 97.4 % | 9.1 dB |
| Edit (2 refs) bf16 none vs LoKr 1.0 | 40.6 | 99.99 % | 12.1 dB |
| Edit (2 refs) q8 none vs LoKr 1.0 | 38.1 | 98.4 % | 12.1 dB |
| Edit (2 refs) q4 none vs LoKr 1.0 | 36.8 | 89.4 % | 13.5 dB |
| T2I stack: LoRA 1.0 + 3rd-party 1.0 vs LoRA 1.0 + 3rd-party **0.3** | 15.6 | 19.0 % | 15.4 dB |
| T2I stack: LoRA 1.0 + 3rd-party 1.0 vs LoRA **0.5** + 3rd-party 1.0 | 48.0 | 88.3 % | 9.8 dB |
| Edit stack: LoKr 1.0 + 3rd-party 1.0 vs LoKr 1.0 + 3rd-party **0.3** | 40.2 | 99.97 % | 13.1 dB |

The trained effect is visible by eye. The T2I LoRA turns the base model's soft 3D render into the
trained flat graphic (thick black outlines, flat teal/orange rings) at all three tiers. The edit LoKr
darkens the shapes and inverts the source's black outlines to light ones at bf16 and q8, and more
weakly at q4. In each stack only the named weight was varied, and each change alone moved the output,
so the two adapters apply with independent weights. bf16 vs q8 with the LoKr: 3.2 mean |Δ|, 28.9 dB.

**Failure found and fixed in place: a LoRA/LoKr on a 2.1 edit was refused on candle.** Before
`599233825`, every 2.1 edit job that carried an adapter failed with *"qwen_image_2_1 cannot apply the
selected LoRA/LoKr stack through the resolved QwenImage21Edit image route"*: the trained edit LoKr and
the third-party LoRA, at all three tiers. Root cause: the fail-closed allow-list
`CandleImageRoute::applies_request_loras` (`crates/sceneworks-worker/src/image_jobs/base.rs`) omitted
`QwenImage21Edit`. That route renders through the same `generate_candle_stream` as 2.1 T2I, and that
stream already threads `resolve_adapters` into the LoadSpec. The fix adds the arm with the T2I
predicate. A unit test was added, and reverting the arm makes it fail (mutation-checked). All edit
rows above ran on the fixed worker. macOS has no dedicated 2.1 edit route: it falls through to
`ImageRoute::Mlx`, which is allow-listed. The Mac session's A2 confirms this on hardware.

## A3 — every E9 format on CUDA

* The fixture-based adapter suite (`cargo test --release -p candle-gen-qwen-image-2-1 --features cuda
  --test integration -- adapters`, inference `457f1a37`) passed **24/24**. It was built with
  `--features cuda`, but the fixtures run on `Device::Cpu` by construction (`tests/common/mod.rs`), so
  this is a CUDA-build run of the format matrix, not a CUDA-device run.
* Formats exercised **on the CUDA device with real 2.1 weights**: the SceneWorks/candle-trained LoRA
  (dotted diffusers keys), the candle-trained PEFT LoKr, and a public third-party LoRA in PEFT
  `lora_A/lora_B.default` spelling, each at bf16, q8 and q4.
* **A third-party Qwen-Image-2.1 LoRA exists.** The HF filter `base_model:adapter:Qwen/Qwen-Image-2.1`
  lists dozens. The one used was `prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA@382d066d079854a86c9513f3c5a4026ada21ddbe`
  (whole repo downloaded, 5 × 84 MB; `Qwen-Image-2.1-Natural-Exposure-LoRA-4000.safetensors`, bf16,
  448 tensors, licence `qwen-research`). Library import auto-detected family `qwen-image-2-1`. As a
  1-reference edit ("Transform the image with balanced neutral exposure") it applied: none vs 1.0 gave
  12.5 mean |Δ| / 20.6 dB at bf16, 12.4 / 20.6 dB at q8 and 32.0 / 17.1 dB at q4. It also stacks with
  both trained adapters (A2).

## A4 — family refusals and the unresolved adapter (live rust-api)

All requests were `POST /api/v1/image/jobs` against the running API.

| adapter | detected family at import | on `qwen_image_2_1` | on `qwen_image` (2512) |
| --- | --- | --- | --- |
| probe stamped `ss_base_model_version=Qwen-Image-2512` | `qwen-image` | 400 *"appears to be a qwen-image LoRA, which is not compatible with model qwen_image_2_1 (qwen-image-2-1)"* | — |
| third-party 2.1 LoRA | `qwen-image-2-1` | renders (A2/A3) | 400 *"appears to be a qwen-image-2-1 LoRA, which is not compatible with model qwen_image (qwen-image)"* |
| attention-only, **no metadata**, 32 blocks | **none (left unresolved)** | 400 *"has no declared family; cannot verify compatibility"* | the same |
| … after the user declares `qwen-image-2-1` | `qwen-image-2-1` (declared) | 201 accepted | 400 *"the LoRA is a qwen-image-2-1 adapter, and qwen_image loads qwen-image adapters"* |

Both cross-version refusals name both families. The unit tests behind this table also pass:
`loras::base_model_gating_tests` (3) and `lora_family::tests::qwen*` (4).

## A5 — measurement campaign (once)

**What the repo harness can and cannot capture.** The calibration plan declares exactly one
`qwen_image_2_1` anchor per (tier, lane), all text-to-image `overlay: none` at 2048² (epic 22505;
calibration runbook, qwen_image_2_1 block). The candle memory adapter's 2.1 arm is base-only
(`QWEN_IMAGE_2_1_PLAIN_EXECUTION_PATH`; `validate_plain_overlay_target` refuses a `lora` overlay).
There is no LoRA-overlay plan entry and no adapter arm, which is runbook §2d's "declared, not planned,
not implementable" row: new plan entries plus an adapter arm, landed together or not at all. There
is no training-footprint harness. **No anchor was added or re-captured, and nothing in
`config/memory-anchors.json`, `docs/generated/memory-matrix.*` or the census changed.** The campaign
below used the engine's own instruments instead: the CUDA driver's `CU_MEMPOOL_ATTR_USED_MEM_HIGH`
(true live high-water, GiB = 2³⁰) per process, one process per cell. Runners were checked before and
between cells.

### Training footprint vs the trainer's preflight (bf16 — the only trainable tier)

Q8/Q4 training is a typed refusal by design ("install the BF16 tier"). Each cell is the A1 request
run through the engine trainer directly for 10 steps with the budget disabled. The prediction is the
engine's own derived peak, taken from its refusal text under a forced 1 KiB budget.

| cell | predicted peak | live high-water | live / predicted |
| --- | --- | --- | --- |
| T2I LoRA 768² ckpt | 22.4 GiB | 28.28 GiB | 1.26 |
| T2I LoKr 768² ckpt | 21.8 GiB | 27.80 GiB | 1.28 |
| T2I LoRA 1024² ckpt | 35.1 GiB | 51.03 GiB | **1.45** |
| Edit LoKr 768², 1 ref | 51.1 GiB | 64.60 GiB | 1.26 |
| Edit LoRA 768², 1 ref | 51.4 GiB | 64.86 GiB | 1.26 |
| Edit LoKr 768², 2 refs | 95.5 GiB | > card (spilled, see A1) | — |
| T2I LoRA 768² no ckpt | 199.2 GiB | not run (refused; the worker forces checkpointing on) | — |

Per-phase (1024² T2I): captions 14.11 GiB (predicted 14.1, exact); latents 3.66 GiB (predicted 2.9);
**training step 51.02 GiB (predicted 35.1)**.

**Failure (inference, not fixed here): the training preflight under-prices the train step by
26–45 %, so it admits runs that cannot fit.** The miss is in `training_footprint`'s step term
(`candle-gen-qwen-image-2-1/src/training.rs:515-541`) and its structural constants
(`training.rs:223-244`: `SCORE_COMPUTE_TENSORS=3`, `SCORE_F32_TENSORS=4`, `BACKWARD_SCORE_GRADS=2`,
the `[S, inner]` counts). Those constants are self-described as "not a measurement". Fitting the two
T2I geometries attributes the gap to ≈25 B per attention-score element (32 heads · S²) plus ≈0.8 MB per
joint token, so the miss grows with resolution (1.26× at 768², 1.45× at 1024²).

A balloon demonstrated it end to end. 51 GiB was held on GPU 1, leaving 44,316 MiB free (budget
≈ 0.85 × free ≈ 36.8 GiB). A SceneWorks 1024² T2I job **passed the preflight** (35.1 ≤ 36.8) and
trained three steps while its live need (51 GiB) exceeded the physical free memory: the card sat at
97,312 MiB, a spill. On a 48 GB-class card this run is admitted and then OOMs. The fix belongs in
inference (re-derive the step constants from a measurement, or price the step at the measured live
high-water) with a pin bump. It is reported to the coordinator, not started here.

### LoRA inference overlay vs `memory_strategy::adapter_overlay` (768², 8 steps)

| tier | base live high-water T2I / edit (2 refs) | + trained LoRA (f32 file 167.9 MB) | + trained edit LoKr (6.8 MB) | + third-party LoRA (bf16 file 83.9 MB) |
| --- | --- | --- | --- | --- |
| bf16 | 33.99 / 35.25 GiB | Δ 251.7 MB (pred 167.9) | Δ 14.5 MB (pred 14.5) | Δ 251.7 MB (pred 83.9) |
| q8 | 21.72 / 22.97 GiB | Δ 251.7 MB | Δ 14.5 MB | Δ 251.7 MB |
| q4 | 15.99 / 17.25 GiB | Δ 251.7 MB | Δ 14.5 MB | Δ 251.7 MB |

The overlay is tier-independent and additive: a stack equals the sum of its members. LoKr is priced
exactly. A LoRA's **resident** delta right after load equals the f32 size of its factors (167.8 MB
for both LoRAs), and its peak adds a further ≈84 MB transient. Fix-scale finding (inference, small
in absolute terms): `gen_core::adapter_stack_resident_bytes` prices the safetensors **file** bytes
(`gen-core/src/memory_strategy.rs:816-818`), while `candle_gen::quant::install_dotted_adapters`
upcasts every factor to f32 (`candle-gen/src/quant/adapters.rs:375-376`). A bf16/fp16 LoRA is
therefore under-priced 2× resident (3× at peak), and an f32 one 1.5× at peak. In production the
worker's admission absorbs this: the bf16 prediction sits on the 45 GiB measured floor (live 34.2
GiB), and q8/q4 add the file bytes to predictions well above the live peaks (q8 30.2 GiB predicted vs
21.95 GiB live). No admitted render came near the card.

The q8 tier loads in 100–140 s against 14–20 s for bf16/q4. That is noted, but it is not a memory
finding.

## MLX half — pending Apple Silicon

Nothing here was measured on MLX. The Mac session must run, against the same SceneWorks feature head
and inference pin:

1. **A1 (MLX)**: through the Mac desktop/API stack, train `qwen_image_2_1_lora` (T2I, LoRA) and
   `qwen_image_2_1_edit_lora` (edit, LoKr) with the same datasets/config as above. The generator is
   [`scratch/gen_datasets.py.txt`](scratch/gen_datasets.py.txt): Pillow, deterministic. Record steps, wall, loss,
   previews, interrupt + `advanced.resume=true` retry, library metadata (`family`, licence,
   `trainingMode=edit`). Record whether the 2-reference 768² edit set fits unified memory or is
   refused by the MLX preflight, and compare with the MLX trainer's own prediction.
2. **A2 (MLX)**: bf16 / q8 / q4 same-seed with-vs-without renders for the T2I LoRA and the edit LoKr
   with **2 references**, plus the two stacked-weight pairs. Use the per-pair metrics in this
   document's format, and include **cross-backend loads**: the CUDA-trained adapters on MLX and the
   MLX-trained ones on CUDA. Keys are shared by design, so that check is the point.
3. **A3 (MLX)**: `cargo test -p mlx-gen-qwen-image-2-1 --test integration -- adapters` (or the
   crate's equivalent adapter suite) and the third-party LoRA above at bf16 + one packed tier.
4. **A4**: backend-independent and done above. Re-run only if the Mac rust-api differs.
5. **A5 (MLX)**: `node scripts/measure-memory-catalog.mjs --backend mlx --model qwen_image_2_1` for
   the planned `overlay: none` anchors, but only if a re-capture is wanted: the six
   `qwen_image_2_1` anchors exist from sc-24114. The **LoRA overlay** has no plan entry or MLX adapter
   arm either (`mlx.rs` 2.1 arm is base-only), so measure it the way the CUDA half did: per-process
   peak (MLX `peak_memory`) with vs without each adapter per tier, compared against the MLX
   provider's `adapter_overlay` prediction. Measure the **training footprint** per cell (T2I LoRA
   768²/1024², LoKr 768², edit 1-ref/2-ref) against the MLX trainer's preflight prediction. Check
   whether the MLX twin shares the CUDA step-term under-count above.
