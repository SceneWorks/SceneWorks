# SW-7033 · Tier 2 base-vs-head comparison for APP-1 (SW-F348)

## Purpose and rule

`REQUIREMENTS.md` line 662 sets Tier 2 for a refactor-only feature: at feature end, A/B only the
affected proof-set routes on the same runner and window, combining fidelity, time and memory
observations. DEC-15 (line 822) approves that scope. DEC-7 (line 814) bounds the work: bit-exact
is the preferred result, and further runs need a concrete unresolved question. Line 664 records
that Tier 2 is dispatched separately and its evidence recorded. `IMPLEMENTATION_PLAN.md` §8.1
(line 996) defines the proof set and line 1022 restates the DEC-14/15/18 scope.

APP-1 changed no engine code: base and head both pin inference
`4a36a7f8c9645110df4df656ca9d492e55ab6c7c`. It changed the API and worker side: the typed
registry, the claim gate, portable model identity with worker-local rebinding, the trusted
pinned-file ledger and the external-library lease. The same request and seed should therefore
give the same decoded output on both sides.

## Method

The driver is `scripts/tier2-ab.mjs`, run by `.github/workflows/tier2-ab.yml`, which is called
only by the `tier2-ab` profile of `.github/workflows/memory-catalog-campaign.yml`
(`workflow_dispatch`). No push, pull-request or merge event reaches it.

- One job per backend checks out the driver commit, the base SHA and the head SHA into separate
  directories, verifies each checkout's HEAD and clean tree, and builds each side's release
  API/worker into one shared scratch target.
- Each run starts a fresh API and a separate worker from one side's binary with fresh state,
  creates a project, POSTs the route's fixed request with a fixed seed, waits for the claim and
  completion, and reads the saved asset back through the product file route. Both sides receive
  the same request JSON and point at the same read-only model hub.
- Order and stop rule (DEC-7): base#1, then head. If the decoded content of head equals base#1
  the route is `equal`. Otherwise base#2 runs once: base#1 ≠ base#2 is
  `nondeterministic_on_runner`; base#1 = base#2 is `head_differs`.
- Fidelity is the SHA-256 of decoded content: PNG pixels as RGBA8, WAV PCM samples, video frames
  decoded by ffmpeg to RGBA plus the s16le audio track. File hashes are recorded too; saved PNGs
  carry a workflow metadata chunk, so file hashes differ between jobs with equal pixels.
- Time is enqueue to completion, with a load/denoise split read from the job's stages at a
  500 ms polling interval.
- Memory on MLX (nax-macos-2): the worker process's physical footprint from `top` MEM, which
  includes Metal allocations, sampled about every 2 s.
- Memory on CUDA: `nvidia-smi` `memory.used` on physical GPU1 every ~500 ms, reported as the
  peak above the largest of five pre-request samples. It is device-wide, not per-process.
- CUDA lease: physical GPU1 (nvidia-smi index 1, PCI `C1:00.0`) is the host's display GPU; the
  worker sees it as logical CUDA 0 (`CUDA_VISIBLE_DEVICES=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
  It is never strictly idle, so this profile uses the `shared-display` admission policy of
  `scripts/ci/windows_cuda_execution.py`: after the exact-UUID lock, a 10-sample window must show
  free memory ≥ 75,776 MiB, stable `memory.used` and utilization, and no foreign compute footprint
  above 8 GiB. Every other caller keeps strict-idle. **CUDA times and memory were taken alongside
  display work** on that GPU, and each run's baseline is recorded with its peak.

## Routes and why

| Route | Request | Why it is in the subset |
| --- | --- | --- |
| `mlx-flux1-dev-t2i` | flux_dev, 768², 4 steps, seed 33111, no tier named | MLX image on a standard quant-tier model; exercises tier freezing in the portable binding |
| `mlx-flux3-klein-t2i` | flux3_klein bf16, 512², 4 steps | alternative to FLUX.1 (first present runs) |
| `mlx-moss-image-t2i` | moss_image_sft, 256², 2 steps | multi-file diffusers closure |
| `mlx-ltx25-distilled-t2v` | ltx_2_5 q4, 768×512, 4 s, 24 fps, distilled | largest closure (two variants in one tier row); video |
| `mlx-chatterbox-tts` | chatterbox_tts with an uploaded synthetic reference WAV | audio route with an uploaded project asset |
| `cuda-sdxl-q4-t2i` | sdxl, 1024², q4, seed 16402 | CUDA image with coRequisite components |
| `cuda-moss-image-t2i` | as MLX | CUDA diffusers closure |
| `cuda-ltx23-distilled-t2v` / `cuda-wan22-ti2v-5b-t2v` | 768×512, 4 s, 24 fps, 8 steps / 832×480, 4 s, 16 fps, 8 steps | CUDA video group, first present runs; replaces LTX-2.5 on CUDA (see SW-7124 below) |
| `cuda-ltx25-distilled-t2v` | as MLX | ran in run 43441772001; now selectable only |

## Runs

Run URL: `https://github.com/SceneWorks/SceneWorks/actions/runs/<run>`. Every run below pins
inference `4a36a7f8` on both sides. Base is `d3f7b05bf2` (main before the feature) in every run.
Times are enqueue → complete in seconds. MLX memory is the worker's peak footprint; CUDA memory is
the peak above the pre-request baseline (baseline in parentheses). Values are copied from each
run's `report.json`.

| Run | Driver | Head | Runner | Route | Verdict | base#1 s | head s | base#2 s | Memory base#1 / head |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 43287955401 | `c558340dea` | `d2b0c76c9b` | nax-macos-2 | mlx-flux1-dev-t2i | head_differs | 68.2 | 229.8 | 71.8 | 21.0 / 13.0 GiB |
| | | | | mlx-ltx25-distilled-t2v | error (head timed out) | 436.1 | — | — | 65.0 / — GiB |
| | | | | mlx-chatterbox-tts | equal | 174.2 | 22.6 | — | 4.13 / 4.20 GiB |
| 43424868140 | `935821c3f8` | `f1de2e5997` | nax-macos-2 | mlx-flux1-dev-t2i | equal | 67.7 | 24.5 | — | 37.0 / 42.0 GiB |
| | | | | mlx-ltx25-distilled-t2v | equal | 341.3 | 1602.0 | — | 65.0 / 65.0 GiB |
| | | | | mlx-chatterbox-tts | equal | 30.1 | 11.2 | — | 3.88 / 3.91 GiB |
| 43427303870 | `935821c3f8` | `f1de2e5997` | cuda-windows-2 | all three CUDA routes | invalid (driver defect, no model ran) | — | — | — | — |
| 43441772001 | `8ffc3fc87b` | `f1de2e5997` | cuda-windows-2 | cuda-sdxl-q4-t2i | equal | 36.9 | 37.9 | — | 8.35 (8.13) / 9.32 (8.13) GiB |
| | | | | cuda-moss-image-t2i | equal | 61.2 | 86.2 | — | 24.50 (8.00) / 24.49 (8.00) GiB |
| | | | | cuda-ltx25-distilled-t2v | error on both sides (SW-7124) | — | — | — | — |
| 43482734077 | `511d70f1d9` | `f1de2e5997` | cuda-windows | cuda-ltx23-distilled-t2v | equal | 87.4 | 128.5 | — | 55.14 (2.93) / 55.04 (2.97) GiB |

Skips: MOSS-Image on MLX in both MLX runs (weights absent from `/Volumes/Models/huggingface/hub`
on nax-macos-2); FLUX.3 Klein in both MLX runs (alternative of FLUX.1, which ran); Wan2.2 TI2V-5B
in run 43482734077 (alternative of LTX-2.3, which ran).

## Regressions found and fixed

Run 43287955401 compared the first feature merge (`d2b0c76c9b`) and found two.

1. **Untiered image request ran a different tier.** FLUX.1 dev with no tier named rendered q8 on
   base and q4 on head (`head_differs`; head peak 13 GiB against base 21 GiB; payload diff on the
   frozen manifest entry). Admission froze the catalog's default download row (q4) into the
   binding, while base ran the worker's implicit standard tier. PR #3379 (`6b837b8aa5`) freezes the
   tier the worker would run and makes the worker honour the frozen tier; PR #3381 (`f1de2e5997`)
   makes every tier selector exact for a frozen tier.
2. **Copy-and-hash of the closure before the first step.** The head LTX-2.5 job stayed unclaimed
   for the whole execution bound while the worker answered source challenges. PR #3379 records
   the cause: the portable guard copied every pinned member into
   `data/models/portable-attempts/<route>/bundle` and re-hashed the closure on every attempt
   (115 GB, over 75 minutes). #3379 verifies pinned members in place with a per-host SHA-256
   cache. Its interim design served each route through a private folder of links to the verified
   files; #3381 withdrew that link folder because loaders that confine shard paths refused link
   targets in `blobs/`, and serves the verified hub snapshot in place again, with a load-boundary
   check that refuses unverified paths.

Run 43424868140 (head `f1de2e5997`, after both PRs) shows FLUX.1, LTX-2.5 and Chatterbox
`equal` on MLX.

## Observations (not failures)

- First-job verification cost: the head LTX-2.5 MLX run in 43424868140 spent 1,456 s in load,
  verifying the closure read from `/Volumes/Models`. On CUDA (43441772001) head verified 90.9 GB
  for a distilled LTX-2.5 request: all of `distilled/q4`, all of `dev/q4` and the distilled LoRA.
  The catalog's q4 row lists both variant directories, so the bound closure includes `dev/q4`,
  which a distilled request does not load.
- FLUX.1 on MLX in 43424868140: head peak 42 GiB against base 37 GiB with equal pixels. The cause
  was not investigated.
- Each head run follows its base run on the same host, so the operating system's file cache is
  warm for head. Head times are not speed claims.

## Families with no Tier 2 evidence here

FLUX.2, SD3.5, Qwen, Z-Image, Krea, Wan (no run reached it), H3, Boogu, Iris, Mage-Flow,
SDXL with InstantID, LTX-2.5 on CUDA (SW-7124), MOSS-Image on MLX (weights absent on the runner)
and FLUX.3 Klein (alternative not exercised).

## Residual limits

- Bytes changed between verification and load are not re-checked.
- The load-boundary check added in #3381 (`ab170ed660`) covers `LoadSpec` sources in the engine
  entry points it lists. It does not cover direct ONNX sessions, `TextLoadSpec` text LLMs, or the
  pid, identity and adapter slots.
- The API host and a remote worker can have different installed tiers; the frozen tier comes from
  the API host's install probe.
- **LTX-2.5 on Candle (SW-7124):** at pin `4a36a7f8` the Candle `ltx_2_5_distilled` provider is
  registered without a component footprint (inference `candle-gen-ltx/src/lib.rs`, `REGISTRATION_25`
  next to `REGISTRATION … footprint = component_footprint`). The registry then returns no footprint,
  and the worker's LTX admission (`crates/sceneworks-worker/src/video_jobs/candle.rs`
  `ltx_resident_weight_bytes`) refuses every LTX-2.5 request on base and head alike, whatever the
  tier, variant, length or size.
- **Invalid run 43427303870:** every worker registered and immediately shut down. On Windows the
  worker treats end-of-input on stdin as a graceful-shutdown request when
  `SCENEWORKS_WORKER_CHILD=1`, and the driver had started it with stdin closed. Driver `8ffc3fc87b`
  holds stdin open, as the existing Windows admitted-replay launcher does.
- **Model hub changed during BASE runs on Windows.** In 43482734077, during base#1, entries under
  `E:\huggingface\hub\models--SceneWorks--ltx-2.3-mlx\snapshots\01df27d3…\gemma\` changed from
  links to regular files (159 field changes; the head run changed nothing). In 43441772001 the
  mtimes of `bundles`, `distilled_lora`, `enhancer` and `licenses` under the LTX-2.5 snapshot
  changed during base#1 with unchanged listings. The cause is pre-existing worker code present at
  base and head: on Windows `huggingface_snapshot_dir` and `huggingface_pinned_snapshot_dir`
  (`crates/sceneworks-worker/src/model_jobs/hf_snapshot.rs`, lines 14–22 and 91–104, identical at base and head) call
  `materialize_snapshot_hardlinks` (`model_jobs/receipts.rs`, line 1699 at base, 1812 at head),
  which replaces every symlink in the resolved snapshot with a hard link to its blob and falls back
  to a full copy only if the hard link fails (`relink_to_blob`, line 1724 at base, 1837 at head).
  The Candle video path reaches it through `candle_video_snapshot_dir`, the LTX Gemma resolution
  and the pinned LTX-2.5 snapshot (`video_jobs/candle.rs` lines 222, 465 and 1224 at base; 222,
  478 and 1237 at head). A hard link adds no disk use; a copy would add each file's size. The
  artifacts do not record link counts, so they cannot show which happened; the snapshot and its
  blobs are on the same volume, where hard links normally succeed. Head would do the same on a
  snapshot whose entries are still links; it changed nothing in these runs because base had
  already converted them. Nothing was removed. The driver now records each entry's link count and
  treats a change made only during base runs as a report note.
