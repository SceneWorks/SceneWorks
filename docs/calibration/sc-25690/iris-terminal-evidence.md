# Epic sc-25678 (Iris-3B) — terminal evidence campaign (sc-25690)

Updated 2026-10-13, at the **final inference pin 6e402df8f** (inference `main`, merge of #1243; the
pin carries inference #1238–#1243 on top of the epic's #1232). SceneWorks code under test:
`story/sc-25690-epic-25678-iris-3b` — the smokes ran at 6b98e044c (the pin commit),
ff2342e71 and 83c191067; after 6b98e044c the branch changed only a test-only assertion, the campaign's
own evidence commits, the catalog listing script, two CI fixes (a lint dependency, a pinned census
fixture), one workflow flag and this document.

Machines.
- **Mac2** = nax-macos-2, Apple M5 Max, 128 GiB unified memory. Runner name enforced before any build.
- **CUDA box** = two RTX PRO 6000 Blackwell cards, each `memory.total` 97,887 MiB = **95.59 GiB**
  (102.64 GB decimal; sold as "96 GB"). Card 0 has been held by another tenant throughout
  (95,262 MiB of 97,887 MiB in use at 2026-10-13 18:47 UTC, read by the two-card preflight of run [38446768292](https://github.com/SceneWorks/SceneWorks/actions/runs/38446768292)); every run here used card 1 alone. The limit on two-card evidence is **tenancy,
  not hardware**.
- The dev Mac GPU was not used (CPU-only compile and unit tests locally).

**Units.** Everything compared with a floor is **GiB** (2^30 bytes): the floors, the capacities the
gates read (`memory.total` MiB ÷ 1024; `hw.memsize` ÷ 2^30) and the guard's caps and peaks. Probe
summaries print decimal GB; they are quoted as printed, labelled "GB", and converted
(÷ 1.073741824) before a reserve is added and the sum rounded up
(`crates/sceneworks-core/src/training/iris/floors.rs`).

**Evidence types** (E9) are kept apart: *fixture parity* (native vs the frozen upstream on a
miniature), *unit / job-path tests* (the production job path with the trainer or estimator
stubbed), *real-weight job-path runs* (a test binary driving the production job path on real
weights), *trainer-level runs* (inference's own real-weight tests), *memory-catalog captures* and
*installed-product acceptance* (a packaged build driven by hand).

## Epic acceptance — status of every test

`closed` only where evidence of the kind the test asks for exists and is cited. `open` names the
exact blocker.

| AT | Asks for | Status | Evidence / blocker |
|---|---|---|---|
| AT1 | Packaged Apple Silicon build + NVIDIA worker: install each task via Model Manager; run generation, depth, restoration through the product/job path; depth/restoration with no Qwen installed; save/reopen jobs, settings, assets, raw-depth exports | **open** | Real-weight job-path runs exist on both backends (rows J1–J3 at the final pin; J10; depth and restoration ran against a hub that holds no Qwen repository). **Blocker:** nothing ran on a *packaged* build, installed through Model Manager, with save/reopen — see "Installed packaged-build acceptance". |
| AT2 | Native generation conditioning/model/solver tensors and fixed-noise outputs vs the frozen upstream: square, portrait, landscape, prompt batches, negative conditioning, CFG intervals, solver orders, changed shift/steps; declared tolerances | **closed (fixture parity)** | Inference `mlx-gen-iris` / `candle-gen-iris` `tests/{text,dit,solver,e2e,controls}_parity.rs` against goldens produced from the frozen upstream (speridlabs/iris-3b @a8d15239; producers `crates/media/mlx-gen/tools/dump_iris_*.py`). The controls oracle covers solver order 1, shift 2, CFG interval (0.3, 0.8), a negative prompt, CFG off, a portrait canvas (12×8), a two-prompt batch, the `prediction: x` reading and — added by inference #1241 (afc88d24f, in the pin) — a **square** canvas (8×8), a second **landscape** canvas (8×16) and a **changed step count** (8 vs 5). `controls_parity::every_control_matches_upstream_on_the_miniature` pins the case list by name on both backends. Declared tolerances: MLX 6e-2 of peak (FP32 backbone / solver, bf16 text tower; observed 2.2e-2 – 4.0e-2 on the new cases), Candle 2e-4 (fp32 tower; observed ≤ 5.3e-5). Three mutations of the shared `gen-core` plan each red exactly the case they should (#1241). CI: inference `CI gate` on `main` (inference run [38429443167](https://github.com/SceneWorks/inference/actions/runs/38429443167), success at 6e402df8f). Tensor-level on a miniature snapshot, not on the 3B weights — what the test asks for ("controlled fixtures"). |
| AT3 | Depth and restoration outputs vs the reference across EXIF orientation, odd/small/non-square inputs, capped/native depth, restoration scales, budgeted/original inputs, colour correction on/off, tile boundaries; NPY round-trip, lineage, depth-control adaptation without raw-data loss; a saved depth asset usable after restart through a separate consumer | **open (one item)** | Fixture parity, in the pin: inference `tests/depth_parity.rs` (odd / small / non-square sources, capped vs native max side, the size law, no-resize sources) and `tests/restoration_parity.rs` (torch bicubic at fractional / non-dyadic scales, antialiased resize, Gaussian fusion window and tile grid, wavelet colour fix at the boundary, PIL-Lanczos `fit_budget` bit-exact, the whole restorer in FP32 at several scales incl. a single tile, a portrait 3×, the budgeted input path and 1.3×, colour fix on and off). **EXIF orientation:** `depth_jobs/tests.rs` `an_exif_rotated_source_yields_the_reference_depth_of_its_upright_pixels` — for every orientation 1..=8 the job's raw `.npy` is bit for bit the reference estimate of the upright pixels, the upright reference itself pinned to Pillow 12.3.0's `exif_transpose` (`the_upright_reference_is_pils_exif_transpose`). NPY round-trip: `sceneworks-core` `depth_map.rs` `npy_round_trips_shape_dtype_byte_order_and_bits`. Real weights: an `image_depth` job on an EXIF-6 photo (row J2). **Blocker:** "usable after restart through a separate consumer" has not been run on an installed build (part of packaged acceptance). |
| AT4 | Both backends: short real-weight runs of generation full fine-tuning, LoRA, LoKr, depth training, restoration training; cancellation/resume; export/registration; inference with the result; random-init generation and all curriculum stages on controlled fixtures | **closed (real-weight job-path runs, single device per backend)** | At the final pin, through the production job path on both backends: generation LoRA (T1; Candle in J10), LoKr with cancel → resume and the continuation assertion (T2, T3), generation **full fine-tune** with cancel → resume → export → render (T9, T9c), depth full fine-tune → export → depth job (T4, T4c) and under gradient checkpointing with cancel → resume (T4r, T4rc), restoration LoRA (T6; Candle in J10) and restoration **full fine-tune** → export → restore job (T10, T10c). Export / registration and inference with the result are part of every row. Every leg also ran on Candle at the final head in one run (J10). Not claimed here: more than one device (AT5). Curriculum stages / random init on fixtures: inference `tests/curricula_{parity,lifecycle}.rs`, `train_{parity,lifecycle,lineage}.rs` (CI gate at the pin). |
| AT5 | Distributed replication and sharding, global-batch accounting, checkpoint save/resume, rank failure, with real multi-rank execution on supported MLX and CUDA topologies; missing hardware evidence leaves it open | **open** | Mechanism on one device: FSDP full / grad_op vs DDP, sharded checkpoint resume + reshard, rank exit / stall → typed failure, multi-process on ONE card (`share_device`) — inference run 38233658960 (CUDA); MLX ring/TCP multi-rank mechanism tests (inference CI) and a real-weight two-rank DDP run on one Mac at the final pin (row T5). **Blockers:** (1) **a real two-card CUDA run** — the smoke exists and is dispatchable (`iris_two_card_distributed_smoke`: DDP cancel → resume with the continuation assertion, a killed rank failing the job typed with no rank left holding a card, FSDP to completion, global batch 2, each rank's card read from the run itself). Dispatched at the final pin (run [38446768292](https://github.com/SceneWorks/SceneWorks/actions/runs/38446768292), `iris_smoke_cards=0,1`): the preflight refused before any build — card 0 held 95,262 MiB of 97,887 MiB for another tenant (over the 2,048 MiB free-card threshold), card 1 was free at 449 MiB. No rank was started; there is no two-card result. (2) **multi-host MLX** needs a second Mac free at the same time. |
| AT6 | Memory-cap admission, task switching, source unavailability, incomplete/wrong-task artifacts, invalid training targets, no-valid-depth pixels, canceled tiles/updates, failed exports; ordinary CI and regression behaviour for existing paths | **closed (unit / job-path tests)** | On this branch; `npm run rust:check` (macOS/MLX), `rust:check:candle` and `rust:check:neither` green locally at the final pin, and the Candle worker's own test run green on the CUDA box in every dispatch below (job `candle-worker`); the PR's required CI is reported on the PR. Memory admission: `iris_training.rs` `tests` + `job_tests` (under-floor, unreadable host, no row, per-row provenance, the checkpointing hint and per-lane gate). Missing backbone / encoder / teacher / training-resource refusals before load; cancel before and after the first step. Task switching and eviction: `image_jobs/iris.rs` `run_iris_task_switch`, `an_iris_training_run_evicts_a_resident_generator_or_transform_before_loading`. Wrong-task artifacts: `iris_restore/tests.rs` `resolution_refuses_a_wrong_task_export`; inference `restoration_parity::wrong_task_artifacts_are_refused`. Canceled tiles: inference `cancel_between_tiles_is_typed_and_yields_no_image`. **No valid depth pixels:** inference — `depth_jobs/tests.rs` `an_estimate_with_no_valid_depth_pixels_is_refused_and_publishes_nothing` (a product defect fixed with it: an all-NaN estimate used to be published as an asset; it is now refused `[iris_depth_no_valid_pixels]` before anything is staged); training — `iris_training.rs` `a_depth_dataset_with_no_valid_depth_pixels_is_refused_and_publishes_nothing`. **Failed exports**, every objective: `a_failed_export_reports_the_failure_and_registers_nothing_for_every_objective` (worker), `a_failed_task_export_registers_no_model_and_leaves_its_sources_untouched` and `a_failed_generation_export_registers_nothing_and_leaves_its_sources_untouched` (API). Not claimed: any of this on an installed build. |
| AT7 | After inference lands and the single final pin is stable: one terminal capability/memory/calibration campaign over the new registered routes; final refs and installed-product outcomes; revisions, hashes, settings, devices, outputs, URLs | **campaign done; installed-product outcomes open** | The terminal memory-catalog campaign ran once at the final pin over all six new cells (MLX [38435700778](https://github.com/SceneWorks/SceneWorks/actions/runs/38435700778), Candle [38449877331](https://github.com/SceneWorks/SceneWorks/actions/runs/38449877331)) — see "Memory-catalog campaign". Capability facts re-dumped at the pin on both lanes. Training-floor campaign: "Training-floor measurements". Revisions, devices, settings and outputs are in each evidence record (`docs/calibration/sc-25690/iris-3b-*-evidence.json`) and the run URLs here. **Open:** "installed-product outcomes" — nothing in the campaign ran on an installed packaged build (see the packaged-build row). |
| — | Installed packaged-build acceptance | **open** | Not run. **Blocker:** needs a packaged Apple Silicon build and an installed NVIDIA worker driven by hand: install the three tasks through Model Manager, run each, save/reopen jobs, settings, assets and raw-depth exports, restart and read a saved depth asset through a separate consumer (AT3's last item). Every run in this ledger is a test binary or a capture harness on the job path — a different evidence type. |
| — | 16 GB card | **open** | No 16 GB card on any runner, and nothing emulates a small card's real allocation (only admission caps). The adapter floor rows that would admit one are declared, not measured. **Blocker:** hardware. |
| — | Real two-card CUDA run | **open** | The smoke and its dispatch input exist and were dispatched twice (38422258699 at pin c1f594203; [38446768292](https://github.com/SceneWorks/SceneWorks/actions/runs/38446768292) at the final pin); both stopped at the preflight because card 0 is occupied by another tenant (95,262 MiB used at the second attempt). **Blocker:** tenancy of card 0. To close: `gh workflow run windows-candle.yml -f run_iris_worker_smoke=true -f iris_smoke_cards=0,1 -f iris_worker_smoke_filter=iris_two_card_distributed_smoke` once card 0 is free. |
| — | Multi-host MLX | **open** | Mechanism tests only. **Blocker:** a second Mac free at the same time; the scheduler also has no multi-host job concept (sc-25689 closeout). |

## Runs at the final pin

**Peak (Mac2) = process-tree phys_footprint in GiB**, read in bytes from `proc_pid_rusage` by
`scripts/ci/iris-mlx-real-weights/tree_guard.py`: `kernel max` is the root process's
`ri_lifetime_max_phys_footprint` (exact, no sampling); `sampled` is the largest tree sum seen at a
0.25 s poll. Footprint includes MLX's buffer cache, so it can exceed the run's need; it is what the
process held, not a floor. CUDA smoke runs record no peak (the floors come from the probe runs).

Mac2 jobs ran one at a time, each far under the 60-minute ceiling, at wip 6b98e044c:
[38432985359](https://github.com/SceneWorks/SceneWorks/actions/runs/38432985359) 7 min 15 s (all nine smokes, incl. the build), [38434050471](https://github.com/SceneWorks/SceneWorks/actions/runs/38434050471) 1 min 55 s,
[38434402891](https://github.com/SceneWorks/SceneWorks/actions/runs/38434402891) 1 min 41 s. Host before each: 128 GiB unified memory.

| # | AT | What ran | Machine | Run | Result | Peak (GiB) |
|---|---|---|---|---|---|---|
| J1 | AT1 | Worker `image_generate`, native prompt batch 2×768², 20 steps | Mac2 (MLX) | [38432985359](https://github.com/SceneWorks/SceneWorks/actions/runs/38432985359) | pass | 20.704 kernel max |
| J2 | AT1, AT3 | Worker `image_depth`, EXIF-6 photo 1024×768 → upright 768×1024, raw NPY + preview; hub holds only speridlabs/iris-3b (no Qwen) | Mac2 (MLX) | 38432985359 | pass; range [-1.023, 0.996] | 9.133 kernel max |
| J3 | AT1 | Worker `image_upscale` engine=iris, 4× (6 tiles, 2048×1536) + 1× (2 tiles); no Qwen in hub | Mac2 (MLX) | 38432985359 | pass | 10.987 kernel max |
| J10 | AT1, AT4 | The whole default NVIDIA dispatch in one run at the final head: worker smokes 4/4 (generation + controls + LoRA use; native batch 2×768²; depth on an EXIF-6 photo; restore 4× / 1×) and every one-card training smoke 8/8 (generation LoRA, LoKr cancel → resume, generation full fine-tune cancel → resume, depth full fine-tune, depth checkpointed cancel → resume, restoration LoRA, restoration full fine-tune, two-rank DDP on one card) | CUDA card 1 | [38455663460](https://github.com/SceneWorks/SceneWorks/actions/runs/38455663460) @83c191067 | 12/12 pass. The run's conclusion is `failure` for a dispatch defect, not a smoke: the default `iris_training_gpu_smoke` filter also matched the two-card test's module path, and on one card that test refuses (it needs two cards and saw one). Fixed: the one-card step now passes `--skip iris_two_card_distributed_smoke` (verified by listing the selected tests; not re-dispatched) | not recorded |
| T1 | AT4 | Training job: generation LoRA (8 steps, 256²) → adapter → generate | Mac2 (MLX) | 38432985359 | pass; adapter Δ 84.7 | 17.113 kernel max |
| T2 | AT4 | Generation LoKr, `save_every` 2: first durable checkpoint at step 2; cancel requested once it existed and landed at step 4; resumed (`resumeFromJobId` translation) → generate. Continuation asserted from the resumed job's own progress | Mac2 (MLX) | 38432985359 | pass; `canceled_at_step=4 first_checkpoint_step=2 resumed_first_step=5 resumed_steps_executed=4 resumed_to=8`; Δ 12.1 | 42.584 kernel max (40.985 sampled) |
| T3 | AT4 | Same LoKr cancel → resume → generate, with the continuation assertion | CUDA card 1 | [38432985395](https://github.com/SceneWorks/SceneWorks/actions/runs/38432985395) | pass; gate read the 24 GiB declared row against 95.59 GiB; `canceled_at_step=4 resumed_first_step=5 resumed_steps_executed=4 resumed_to=8`; Δ 20.7 | not recorded |
| T4 | AT4 | Depth full fine-tune job (128², 4 steps) → registered export → depth job selects it; no Qwen in hub. Plus the gradient-checkpointing capability: the lane's own trainer accepts it, and a second short job WITH the toggle on runs at the lane's `full_ac` row | Mac2 (MLX) | 38432985359 | pass; `checkpointing_lane=mlx engine_accepts=true checkpointed_run=pass checkpointed_steps=2`, admitted at the 52 GiB `full_ac` row (extrapolated) against 128 GiB | 58.897 kernel max |
| T4c | AT4 | T4 on Candle | CUDA card 1 | [38441826442](https://github.com/SceneWorks/SceneWorks/actions/runs/38441826442) | pass; `checkpointing_lane=candle engine_accepts=true checkpointed_run=pass`; gate 68 GiB (declared row at 128²) against 95.59 GiB | not recorded |
| T4r | AT4 | **Depth full fine-tune UNDER gradient checkpointing, canceled after its first durable checkpoint and resumed** (128², 8 steps, `save_every` 2) → export → depth job selects it. The lane's trainer accepts the toggle; the resumed job's recorded plan still carries it; continuation asserted. Resume materializes the lazily-loaded trainables inside the first resumed step's checkpointed region — proven on a miniature in the engine, and here on the 3B weights | Mac2 (MLX) | 38432985359 | pass; `engine_accepts=true recorded_plan_checkpointing=true canceled_at_step=4 first_checkpoint_step=2 resumed_first_step=5 resumed_steps_executed=4 resumed_to=8`; gate 52 GiB (extrapolated) | 54.549 kernel max |
| T4rc | AT4 | T4r on Candle | CUDA card 1 | 38441826442 | pass; `canceled_at_step=4 first_checkpoint_step=2 resumed_first_step=5 resumed_steps_executed=4 resumed_to=8`; gate 68 GiB (declared) | not recorded |
| T5 | AT4, AT5 | Two-rank DDP generation LoRA (`share_device`, one Mac) → generate | Mac2 (MLX) | 38432985359 | pass; Δ 10.5 | 43.724 sampled tree sum over 3 processes (upper bound 60.966 = sum of per-process kernel maxima) |
| T6 | AT4 | Restoration LoRA job (256², full recipe incl. critic / LPIPS) → export → restore job selects it; no Qwen | Mac2 (MLX) | 38432985359 | pass; gate read the 28 GiB declared row | 30.246 kernel max |
| T9 | AT4 | **Generation FULL fine-tune through the job path:** the engine's official `stage1_256` preset (REPA against DINOv2-B/14, Muon), bounded to batch 1 / 8 steps (LR 1e-3, no warmup, EMA off). Admitted at the full row: 64 GiB + 4 GiB teacher = 68 GiB (extrapolated) against 128 GiB. Canceled after the step-4 checkpoint, resumed with the continuation assertion, exported, the exported checkpoint rendered by a generation job as a registered trained model | Mac2 (MLX) | [38434050471](https://github.com/SceneWorks/SceneWorks/actions/runs/38434050471) | pass; `canceled_at_step=5 first_checkpoint_step=4 resumed_first_step=6 resumed_steps_executed=3 resumed_to=8`; `irisPlan.stage = stage1_256` with REPA; render Δ 101.6 vs the release weights, std 62.5 | 60.994 kernel max (60.945 sampled) |
| T9c | AT4 | T9 on Candle | CUDA card 1 | [38436748957](https://github.com/SceneWorks/SceneWorks/actions/runs/38436748957) | pass; admitted at 79 + 4 = 83 GiB (extrapolated) against 95.59 GiB; `canceled_at_step=6 first_checkpoint_step=4 resumed_first_step=7 resumed_steps_executed=2 resumed_to=8`; render Δ 114.9, std 25.3 | not recorded |
| T10 | AT4 | **Restoration FULL fine-tune through the job path** (256² center crop, the paper recipe incl. critic / LPIPS, 2 optimizer steps) → export → restore job selects it by id; no Qwen in hub. Admitted at the full row: 69 GiB (extrapolated) against 128 GiB. Plus the gradient-checkpointing capability as T4: trainer accepts, a second short job WITH the toggle on runs at the `full_ac` row | Mac2 (MLX) | [38434402891](https://github.com/SceneWorks/SceneWorks/actions/runs/38434402891) | pass; `checkpointing_lane=mlx engine_accepts=true checkpointed_run=pass checkpointed_steps=2`, admitted at the 61 GiB `full_ac` row (extrapolated) | 81.503 kernel max — above the 69 GiB floor: the footprint includes MLX's cache and the follow-up jobs in the same process; the floor is the cache-off active peak + 2 (see the emulated-limit evidence below) |
| T10c | AT4 | T10 on Candle | CUDA card 1 | 38436748957 | pass; admitted at the 73 GiB measured row; `engine_accepts=true checkpointed_run=pass`, the checkpointed job admitted at the 73 GiB `full_ac` row (extrapolated) | not recorded |
| D1 | AT5 | FSDP full / grad_op vs DDP, sharded checkpoint resume + reshard, rank exit / stall → typed failure, multi-process on ONE card (trainer level) | CUDA card 1 | inference 38233658960 | pass | one device only |
| D2 | AT5 | Two-card distributed smoke through the job path (`iris_two_card_distributed_smoke`) | CUDA cards 0 + 1 | [38446768292](https://github.com/SceneWorks/SceneWorks/actions/runs/38446768292) | **not run** — preflight refusal (run conclusion `failure` by design): card 0 occupied, 95,262 of 97,887 MiB; nothing built, no rank started | — |

Superseded trainer-level rows (inference 38179054198 / 38233658960 for generation full fine-tune,
38145886212 / 38149768393 for restoration) and the earlier-pin job-path runs (pins 25231d15b and
c1f594203: Mac2 38320809891, 38321534896, 38321764171, 38411944085, 38412625724; CUDA 38406592757,
38412227879, 38417641518, and the first two-card attempt 38422258699) are replaced by the rows
above. Two Mac2 dispatches at the c1f594203 pin (38410831839, 38411624596) failed in a defect of
the then-new checkpointing probe (it loaded a task trainer with the generation path's load inputs),
not of the product; fixed in bc6419be9. No smoke needed a re-dispatch at the final pin; the one red run there is J10's dispatch defect, described in its row.

## Memory-catalog campaign (AT7)

The Iris memory-strategy contracts (inference #1239 MLX, #1240 Candle) are absorbed and the three
task routes are wired into the catalog: one plan anchor per (route, lane), bf16 the only tier —
six cells. Captured once at the final pin with `measure-memory-catalog.mjs`
(`memory-catalog-campaign.yml`, profile `catalog`, campaign `sc-25690`), eager materialization.
Record status follows each lane's convention: MLX records are `runtime_complete`; every Candle
record in the store is `gated` (the capture measures exact per-phase device memory and the strategy
identity but does not repeat the promotion-quality scenario suite).

| Cell | Machine | Run | Geometry | Rung | Recorded peak | Phases (active peak) |
|---|---|---|---|---|---|---|
| `iris_3b:bf16:mlx` (generation) | Mac2 | [38435700778](https://github.com/SceneWorks/SceneWorks/actions/runs/38435700778) | 1024×1024 | resident | allocator 23.19 GB = **21.60 GiB**; active 16.38 GB = 15.26 GiB | conditioning 16.36 GB · denoise 16.38 GB · decode 14.23 GB |
| `iris_3b_depth:bf16:mlx` | Mac2 | 38435700778 | 1024×1024 | resident | allocator 7.87 GB = **7.33 GiB**; active 7.86 GB | conditioning 6.02 GB · denoise 7.86 GB · decode 6.09 GB |
| `iris_3b_restore:bf16:mlx` | Mac2 | 38435700778 | 2048×1536 output | resident | allocator 12.39 GB = **11.53 GiB**; active 7.86 GB | conditioning 6.02 GB · denoise 7.86 GB · decode 6.08 GB |
| `iris_3b:bf16:candle` (generation) | CUDA card 1 | [38449877331](https://github.com/SceneWorks/SceneWorks/actions/runs/38449877331) | 1024×1024 | staged_residency (engaged: resident, staged_residency) | device peak 12.36 GB = **11.51 GiB** | conditioning 8.84 GB · denoise 12.36 GB · decode 12.36 GB |
| `iris_3b_depth:bf16:candle` | CUDA card 1 | 38449877331 | 1024×1024 | resident | device peak 10.15 GB = **9.45 GiB** | conditioning 7.66 GB · denoise 10.15 GB · decode 10.15 GB |
| `iris_3b_restore:bf16:candle` | CUDA card 1 | 38449877331 | 2048×1536 output | resident | device peak 10.11 GB = **9.42 GiB** | conditioning 7.66 GB · denoise 10.11 GB · decode 10.11 GB |

**Listing.** The runner's own unfiltered `node scripts/measure-memory-catalog.mjs --list --backend mlx|candle` reports all six cells `current` ("anchor is current at the pinned inference revision"), and `--campaign sc-25690` reports all six `already_captured`. The campaign surfaced one tooling defect here: depth and restoration are task routes outside the matrix modalities, so the anchor store carries no row for them and the listing — which read currency only from store anchors — called both `runnable` right after their capture. Fixed in c39c558b4 (`readTaskRouteCurrency`: the record's own provider closure digest against the pinned provider closure table). The fix is in the listing script only; it is not one of the sources a record or the matrix is fingerprinted on, so no cell needed re-capture (the extractor and the matrix generator are byte-idempotent after it, and all six records' closure digests equal the pinned table's). Only `iris_3b` (generation) has a store anchor that admission reads; the depth and restoration records are packaged evidence.

The MLX numbers match the rehearsals taken before the final pin (runs 38396033589 depth 7.33 GiB,
38398047045 restoration 11.53 GiB, 38399806219 generation 21.60 GiB). Capability facts were re-dumped
at the final pin on both lanes (MLX natively; Candle in a Linux container with stubbed CUDA
libraries — the dump is weights-free): content identical to the checked-in files on all five,
revision labels only. 

## Training-floor measurements

Probe: `train_memory_probe` — full fine-tune, batch 1, whole-process peak; summaries print decimal
GB, converted here to GiB (÷ 1.073741824). Floor = ⌈peak GiB + 2⌉ on both lanes.

**MLX.** Mac2; the quantity is the **active peak with MLX's buffer cache disabled** — what the run
needs — steady by optimizer step 2 and flat through step 6; the process footprint with the cache
off agrees within 0.5 GB. Runs: inference PR #1238 @b1c04bd0d, [38312190794](https://github.com/SceneWorks/inference/actions/runs/38312190794) (two steps +
export) and [38318086154](https://github.com/SceneWorks/inference/actions/runs/38318086154) (six steps); the checkpointed depth / restoration shapes (MLX since
inference #1243) [38405154244](https://github.com/SceneWorks/inference/actions/runs/38405154244). Production runs keep MLX's cache, so process footprint can
rise to the allocator GC ceiling (0.95 × recommended working set: 109.68 GB = 102.2 GiB on Mac2)
without that being need; the first campaign's MLX numbers (run 38292137908: 109.96 / 110.08 GB at
1024²) were that ceiling, not peaks.

*Smaller-Mac evidence* (inference run [38322999659](https://github.com/SceneWorks/inference/actions/runs/38322999659), Mac2 with MLX's memory / cache / GC limits
set to what a 64 GiB and a 96 GiB Mac would get, default cache, six steps): every probed shape
completed with the footprint held at the emulated limit and the active peak unchanged — on the
64 GiB limits (48.96 / 54.84 GB) depth 256² active 49.12–49.35 GB, footprint ≤ 55.77 GB, and depth
512² active 52.78–53.60 GB, footprint ≤ 55.30 GB; on the 96 GiB limits (73.44 / 82.26 GB)
generation 512² active 65.57 GB, restoration 512² 71.57 GB, generation 1024² + AC 74.20–74.88 GB,
depth 1024² 91.59 GB (footprint 92.14 GB = 85.8 GiB). This is allocator-limit emulation on a
128 GiB machine, not a run on a smaller Mac: it shows the cache yields to the limit; it does not
show behaviour under real memory pressure from other processes.

**Candle.** RTX PRO 6000 card 1 (95.59 GiB), inference run [38393679427](https://github.com/SceneWorks/inference/actions/runs/38393679427) @1449045c8: the
**live pool peak** over six optimizer steps, steady from step 2 — after the training-tape memory
fix of inference #1242, which removed a 7–39 GB excess (live pool peaks before → after, GB:
generation 512² 95.40 → 82.41; generation 1024² + AC 89.54 → 82.58; depth 512² 83.38 → 70.41;
restoration 512² 126.28 → 87.36, which did not fit the card before and does now; "before" from
CUDA run 38312194527). Three shapes still need more than the card and completed only by spilling
to system memory (115–157 s/step for generation): their rows carry the measured need under the
provenance `measured_spill` and refuse the card; they are not capped to it.

| Shape | MLX active peak, cache off | MLX floor (GiB) | Candle live pool peak | Candle floor (GiB) |
|---|---|---|---|---|
| Generation full 512² | 65.57 GB = 61.07 GiB | 64 measured | 82.41 GB = 76.75 GiB | 79 measured |
| Generation full 512² + AC | — | 64 extrapolated | 69.19 GB = 64.44 GiB | 67 measured |
| Generation full 1024² | 104.49 GB = 97.31 GiB | 100 measured | 128.75 GB = 119.91 GiB, spilling 29.2 GB | 122 measured_spill (does not fit the card) |
| Generation full 1024² + AC | 74.88 GB = 69.74 GiB | 72 measured | 82.58 GB = 76.91 GiB | 79 measured |
| Depth full 512² | 53.60 GB = 49.92 GiB | 52 measured | 70.41 GB = 65.57 GiB | 68 measured |
| Depth full 1024² | 92.58 GB = 86.22 GiB | 89 measured | 116.77 GB = 108.75 GiB, spilling 17.2 GB | 111 measured_spill (does not fit the card) |
| Depth full 1024² + AC | 63.01 GB = 58.68 GiB | 61 measured | 70.56 GB = 65.71 GiB | 68 measured |
| Restoration full 256² crop | — | 69 extrapolated | 76.15 GB = 70.92 GiB | 73 measured |
| Restoration full 512² crop | 71.57 GB = 66.65 GiB | 69 measured | 87.36 GB = 81.36 GiB | 84 measured |
| Restoration full 512² crop + AC | 63.21 GB = 58.87 GiB | 61 measured | — | 83 extrapolated |
| Restoration full 1024² crop | lower bound only: footprint 116.47 GB = 108.47 GiB at the guard stop with the cache off (live need above 115 GB; the active peak had reached ≥ 87.45 GB by step 2 of 4) | 128 lower_bound | 142.96 GB = 133.14 GiB, spilling 42.5 GB | 136 measured_spill (does not fit the card) |
| Restoration full 1024² crop + AC | 77.25 GB = 71.94 GiB — completes | 74 measured | 86.47 GB = 80.53 GiB | 83 measured |

## Floors (`crates/sceneworks-core/src/training/iris{,/depth,/restore}.rs`)

- Floors are GiB and each row carries its own provenance — `measured`, `measured_spill`,
  `extrapolated`, `lower_bound` or `declared` — published per row (`memoryFloors.provenance`) and
  named in every refusal, the curriculum check and the Studio hint. There is no table-wide flag.
- A measured row is the probe's peak in the unit it printed (the `*_PEAK` constants), converted to
  GiB, plus a 2 GiB reserve, rounded up. A Candle row that ran without spilling never exceeds the
  card that held its run; a spilled measurement is never held to the card (that does not
  compile). The MLX reserve covers the ≤ 0.5 GB the footprint sat above the active peak plus
  allocator granularity; the macOS/app reserve (4 GiB) is separate. Extrapolated rows are computed
  from the same constants: a smaller run is bounded by the measured larger one; a checkpointed
  run by the lower of its plain row and the next measured `full_ac` row; depth between its two
  measured sizes by the lane's own chord.
- Full tables, GiB (m = measured, s = measured_spill, e = extrapolated, lb = lower_bound,
  d = declared):

  | Table | Rows | MLX | Candle |
  |---|---|---|---|
  | Generation `full` | 256 / 512 / 1024 | 64 e / 64 m / 100 m | 79 e / 79 m / 122 s |
  | Generation `full_ac` | 256 / 512 / 1024 | 64 e / 64 e / 72 m | 67 e / 67 m / 79 m |
  | Generation `adapter` | 256 / 512 / 1024 | 16 / 24 / 40 d | 24 / 32 / 48 d |
  | Depth | 16 384 / 262 144 / 307 200 / 589 824 / 1 048 576 px | 52 e / 52 m / 54 e / 68 e / 89 m | 68 d / 68 m / 71 e / 86 e / 111 s |
  | Depth `full_ac` | same sizes | 52 e / 52 e / 54 e / 61 e / 61 m | 68 d / 68 e / 68 e / 68 e / 68 m |
  | Restoration `full` | 65 536 / 262 144 / 1 048 576 px | 69 e / 69 m / 128 lb | 73 m / 84 m / 136 s |
  | Restoration `full_ac` | same sizes | 61 e / 61 m / 74 m | 73 e / 83 e / 83 m |
  | Restoration `adapter` | same sizes | 28 / 36 / 56 d | 32 / 40 / 60 d |

  Before the campaign (the PR's first commit, decimal-GB footprints compared against GiB
  capacity) the full rows were: generation MLX 80 / 85 / 110, Candle 80 / 100 / 125; depth MLX
  64 / 80 / 96 / 112, Candle 68 / 90 / 104 / 126; restoration MLX 92 / 102 / 128, Candle
  94 / 104 / 140. No row is declared below a measured peak.
- **Gradient (activation) checkpointing for depth and restoration** is the shared
  `advanced.gradientCheckpointing` toggle, implemented on both native lanes at the final pin
  (Candle since inference #1242, MLX since #1243). The engine publishes no declaration of it (the
  trainer descriptor has no such flag; a lane that lacks it refuses in `Trainer::validate`, which
  needs the weights), so `iris_gradient_checkpointing_lanes` is an explicit table, published on
  the target (`limits.gradientCheckpointingLanes`) and read by the Studio, the submit and the
  worker. The real-weight depth and restoration smokes ask the lane's own trainer and fail if its
  answer differs from the table (rows T4, T4c, T10, T10c: accepted on both lanes). On MLX it turns
  the restoration 1024² crop from "needs more than 115 GB" into a measured 74 GiB floor. A run
  refused at its plain row is told when the toggle admits it at the `full_ac` row. Generation
  keeps its richer `advanced.iris.activation_checkpointing` policy option.
