# D — Writing each model ONCE instead of once per backend: options and evidence

Researcher D of 8. Read-only study of the `inference` repository at **`origin/main` = `a43724b82`**
(the working tree is stale and was not used), and of the pinned upstream sources in the local cargo
cache (`~/.cargo/git/checkouts/mlx-rs-aaca89f470ab15b6/11d0ff7`, `~/.cargo/registry/.../candle-core-0.9.2`),
plus public web sources for burn, CubeCL, MLX-CUDA and candle-Metal on iOS. Every `path:line` below is
on `origin/main` unless it is a URL. I ran no builds, tests or models.

---

## 0. TL;DR

* **The code base is ~2.03 M lines of Rust in `crates/{media,audio,llm}`** (1.43 M src, 0.54 M tests,
  45 K vendored candle, 14 K examples). The "~1.5 M" figure in the brief matches **src only**.
  The two media engines alone are 1.59 M lines: `mlx-gen` 930 K (640 K src / 286 K tests) and
  `candle-gen` 659 K (523 K src / 127 K tests).
* **39 model families exist in both backends.** Their src totals are 596 K (MLX) and 473 K (Candle).
  The sum of the smaller side of each pair is **437 K lines**. That is the most a "write once" design
  could delete from media src, before keeping any backend-specific code.
* **"Host math once" works but stays small.** The tensor-free contract crates hold 40 % of FLUX 3,
  25 % of Iris, 13 % of H3, 12 % of MOSS-Image, 5 % of LTX and 5 % of Wan. Older families share
  about 0 %.
* **The twins are already mirror-written.** I normalised type and API names (`Array`/`Tensor`,
  `Dtype`/`DType`, `i32`/`usize`, crate prefixes) and diffed same-named files. After that:
  - The **newer twins are 35–60 % line-identical**: H3 37 %, Bernini 43 %, FLUX 3 47 % of MLX lines,
    SAM3 58 %, Face 65 %, Body 66 %.
  - For example, FLUX 3's `dit.rs` is 57 % identical and its `pipeline.rs` 68 %.
  - **The older ports are 2–8 % identical**: FLUX.1, Qwen-Image, Wan, Z-Image, SDXL, Krea, Kolors.
  - **The divergence clusters in the per-backend layer module.** FLUX 3's `nn.rs` is 9 % identical,
    against 57 % for its model graph. That module holds the quantised `Linear`, attention dispatch
    and dtype promotion. This is the strongest single piece of evidence: **a backend trait at the
    layer level, not the raw-op level, fits how the code already divides.**
* **The op surface is small and closed.** About 115–120 distinct tensor ops cover essentially every
  call site on both backends: ~24 K MLX op calls and ~26.5 K Candle calls matched against a
  160-name op list. Add ~10 fused ops (SDPA, RMS/Layer/Group norm, RoPE, GELU variants), ~6 quant
  ops, ~6 execution ops (eval, async-eval, compile, checkpoint, synchronise, memory limits) and
  ~15 custom-kernel sites per backend.
  - Candle's own `BackendStorage` + `BackendDevice` traits are 31 + 12 methods, and they back a
    ~141-method `Tensor` API. A SceneWorks trait in that style would be **~50–60 primitive
    methods, ~80 derived ops with default implementations, and a ~15-method layer/exec/autodiff
    surface**.
* **Recommendation.** Do (b) now, as a no-regret step. In parallel, run a **pilot of (c)**: our own
  thin two-level backend trait (generic tensor glue plus backend-provided layers, exec hooks and
  autodiff), with model graphs written once.
  - Pilot first on a small high-similarity module (Face/Body/SAM3), then on FLUX 3 end to end,
    including quantisation and LoRA training.
  - **Confidence: high (≈85 %)** that (b) pays off; **medium (≈60 %)** that (c) is the right end
    state at acceptable risk. The 40 % uncertainty is the training and autodiff abstraction and
    the cost of migrating the old families.
  - **Do not adopt burn now (≈75 % confidence).** It means a full re-port, burn removed its
    `burn-candle` backend in 0.22, its quantisation is symmetric-centric where ours is MLX affine,
    and it has no proven MLX-class Metal performance or iOS path for this workload.
  - **Do not consolidate onto candle-Metal on macOS yet (≈65 %).** No MLX-vs-candle-Metal media
    benchmark exists anywhere in the repo or memory, and the active iOS lane is MLX-only. A single
    benchmark is cheap and would settle it. If the gap were small, (f) would delete more code
    than any other option.

---

## 1. Method and caveats

**LOC.** I counted with `git grep -c '' origin/main -- <paths>`, which is every line including
comments and blank lines.
- **Test:** a file under `/tests/` or named `*_tests.rs` / `tests.rs`.
- **Vendor:** a file under `/vendor/`.
- Inline `#[cfg(test)] mod tests` blocks remain in "src", so **src is overstated by an unknown
  amount** (these crates carry heavy inline tests).

**Twin similarity.** Script at
`/private/tmp/claude-501/-Users-michael-Repos-SceneWorks--claude-worktrees-multi-resolution-inference-848ec4/8763ea89-d38f-425e-befa-5d78750d51bf/scratchpad/tools/sim.py`.
1. It drops blank lines and `//` lines, then applies a normalisation map:
   - type names: `Array|Tensor → T`, `Dtype|DType → DT`;
   - index types: `i32|usize|i64 → I`;
   - crate prefixes: `mlx_gen*|candle_gen* → G/X`, and `mlx_rs::…|candle_core::…|candle_nn::… → B`;
   - small cleanups: `.expect("…") → .unwrap()`, and `'_` lifetimes removed.
2. It pairs files by **same relative path** inside `mlx-gen-<fam>` and `candle-gen-<fam>`, skipping
   `tests/` and `examples/`.
3. It counts matched lines with `difflib.SequenceMatcher` (autojunk off).

The metric has known biases:
- **Under-counts** twins whose files were renamed (FLUX.1: MLX `transformer.rs` vs Candle `packed_dit.rs`).
- **Over-counts** trivial lines (`}`, `Ok(())`). Re-running with lines of fewer than 4 alphanumeric
  characters removed lowered the shares by only 2–4 points: FLUX 3 47→44 %, H3 37→35 %, SAM3 58→54 %,
  Face 65→60 %, Wan 8→6 %. So the signal holds up.
- It measures **textual mirroring, not semantic duplication.** Two independent ports of the same
  math with different structure score low but are still duplicated work.

---

## 2. Size baseline (`origin/main` a43724b82)

### 2.1 Totals

| Area | All | src | tests | vendor | examples |
|---|---:|---:|---:|---:|---:|
| `crates/media/**` | 1,642,863 | 1,163,180 | 422,499 | 44,608 | 12,576 |
| `crates/llm/**` | 233,788 | 159,037 | 73,655 | – | 1,096 |
| `crates/audio/**` | 151,058 | 107,118 | 43,455 | – | 485 |
| **Total** | **2,027,709** | **1,429,335** | **539,609** | 44,608 | 14,157 |

| Engine | src | tests | total |
|---|---:|---:|---:|
| `crates/media/mlx-gen` (root + 46 families) | 639,621 | 286,215 | 929,647 |
| `crates/media/candle-gen` (root + 44 families, vendor excluded) | 523,433 | 127,089 | 659,033 |
| `crates/llm/mlx-llm` | 95,125 | 43,149 | 139,317 |
| `crates/llm/candle-llm` | 62,989 | 30,506 | 93,548 |
| Contracts (tensor-free): `gen-core` 87,362; `core-llm` 29,515; `gen-core-iris` 22,709; `gen-core-flux3` 18,550; `gen-core-testkit` 15,404; `gen-core-h3` 11,292; `gen-core-train` 6,261; `gen-core-ltx` 6,042; `gen-core-family` 5,826; `gen-core-memory-wan` 5,418; `gen-core-moss-image` 4,015 | | | ~212 K |
| `crates/kernels/candle-quant-kernels` (CUDA `.cu` + Rust) | | | 12,654 |

Candle is pinned to upstream rev `1e6aa85e` (`Cargo.toml:126-128`). On top of that pin,
`candle-core`, `candle-kernels` and `candle-metal-kernels` are **vendored and `[patch]`ed** in
`Cargo.toml:177-180`:
- CUDA-graph htod cache: `crates/media/candle-gen/vendor/candle-core/VENDORED.md:1-30`.
- Rank > 4 strided-reduce fix: `vendor/candle-metal-kernels/VENDORED.md`.
- `candle-flash-attn` is an in-tree fork (`Cargo.toml:129-131`).

MLX is the `pmetal-mlx-rs` fork at `11d0ff7c` (`Cargo.toml:95-122`). It carries SceneWorks patches
that are load-bearing for the GPU watchdog and for error propagation (§6).

### 2.2 Per-family twin sizes (src, excluding `tests/`; inline tests included)

The sum of the smaller side over all twin pairs is 436,697; the sum of the larger side is 632,994.

| family | MLX | Candle | | family | MLX | Candle |
|---|---:|---:|---|---|---:|---:|
| sdxl | 68,832 | 25,482 | | sana | 23,742 | 11,266 |
| z-image | 54,210 | 21,303 | | qwen-image-2-1 | 21,223 | 17,265 |
| ltx | 47,136 | 43,920 | | flux2 | 21,199 | 20,093 |
| iris | 46,839 | 17,314 | | mage | 20,621 | 10,291 |
| minimax-h3 | 38,143 | 37,243 | | anima | 15,298 | 8,510 |
| wan | 35,450 | 28,439 | | moss-image | 15,132 | 13,237 |
| krea | 25,722 | 35,806 | | lens | 13,881 | 12,777 |
| bernini | 12,997 | 13,839 | | qwen-image | 13,246 | 15,108 |
| sensenova | 12,615 | 10,826 | | flux | 11,704 | 16,150 |
| sd3 | 11,448 | 13,635 | | kolors | 10,179 | 11,347 |
| flux3 | 7,901 | 11,638 | | boogu | 7,577 | 11,117 |
| scail2 | 7,080 | 8,259 | | chroma | 6,991 | 6,394 |
| sam3 | 6,598 | 6,684 | | pid | 6,124 | 4,222 |
| svd | 4,227 | 5,541 | | seedvr2 | 4,561 | 4,494 |
| ideogram | 4,643 | 8,493 | | mochi | 4,306 | 4,024 |
| face | 3,461 | 3,803 | | pulid | 3,516 | 3,193 |
| instantid | 2,758 | 3,379 | | body | 2,223 | 2,210 |
| depth | 1,625 | 2,018 | | perceptual | 1,302 | 1,246 |
| supra2 | 1,040 | 1,175 | | clip/joycaption | 332/410 | 744/914 |

- **MLX-only:** `ios-catalog` 20,963, `krea-realtime` 13,234, `sam2` 4,103.
- **Candle-only:** all of audio (107 K src), `candle-gen` root shared code (49,735) and `vendor/` (44.6 K).

### 2.3 Cross-cutting code implemented per model *and* per backend

These were measured with path globs, so they are approximate and overlap each other.

| Concern | MLX src | Candle src | Notes |
|---|---:|---:|---|
| Memory strategy / admission / probes (`*memory_strategy*`, `*admission*`, `*footprint*`, `*memory_probe*`) | 76,552 | 59,120 | One `memory_strategy.rs` per family per backend. Mostly host declarations (e.g. `mlx-gen-flux3/src/memory_strategy.rs:1-30`, the rung table). |
| Training (`train/`, `training*`) | 65,355 | 68,186 | Per family per backend; autodiff differs (§7). |
| iOS pricing/quantum (`ios_*`, `*quantum*`) | 102,999 | – | MLX-only. Z-Image alone carries ~31 K of `ios_pricing*`; SDXL has `quantum_tests_pricing.rs` 8,717. |

These ~270 K lines are host logic or per-backend realisations of contracts. They are exactly the
"hooks at every stage" material the re-architecture targets. Other researchers own the memory and
iOS questions; here they matter because **every option below either keeps or removes their
per-backend duplication.**

---

## 3. Q1 — What the "host math once" contract crates share

### 3.1 The pattern

Every family contract crate declares itself *tensor-free*. Its `Cargo.toml` depends only on
`sceneworks-gen-core`, sometimes `gen-core-train`, `image`, `serde`, `sha2`, `safetensors` and
`tokenizers`. Examples: `crates/contracts/gen-core-flux3/Cargo.toml` (deps block) and
`gen-core-h3/Cargo.toml`. Each carries the same comment: *"The backend-neutral contract this crate
builds on… Tensor-free, like gen-core."* The FLUX 3 header states the intent
(`crates/contracts/gen-core-flux3/src/lib.rs:1-8`): *"Everything here is host-side (no tensors), so
the MLX provider, a Candle twin and the SceneWorks worker read one definition."*

What qualifies as "host math", from the module lists:

| Crate | src LOC | What is shared (modules) |
|---|---:|---|
| `gen-core-flux3` | 12,964 (+2,518 test files) | Frozen upstream identity and pins, package layout and `config.json` (`lib.rs`). Qwen3-VL text window. Flow-map schedule. Packed-token **ids and RoPE tables computed on host** (`rope_cos_sin` `lib.rs:933`, `timestep_freqs` `lib.rs:953`, returning `Vec<f32>`). Checkpoint conversion (`conversion.rs`, `native_convert.rs`, `safetensors_stream.rs`). **Quantisation oracles and packers**: `affine.rs` is an MLX-affine W4/8 packer in pure Rust (`affine.rs:1-10`); `packed_gemm.rs` does row-chunk planning plus exact per-backend transient accounting for GGML Q4_1/Q8_0, FP8R, NVFP4 and MLX affine (`packed_gemm.rs:1-60`). Prompt upsampler (`pup.rs` 2,261 lines incl. JPEG and Python-JSON compat). Reference planning (`references.rs`, `reference_workspace.rs`, `image_input.rs`). Layout compile (`layouts.rs`). Phase resources. **Training controller** (`training/{controller,options,policy,preprocess,rng,sampler,identity,checkpoint}.rs`, ~2.6 K). |
| `gen-core-iris` | 21,077 | Degradation pipelines (1,575), restoration (1,921), depth (880), downstream, tables. **Training controllers**: `train.rs` 3,267, `train/{depth,dist,lineage,repa,shapes,upstream,holdout,pipeline}` ~9 K, `restoration_train.rs` 2,380. Upstream YAML configs. |
| `gen-core-h3` | 11,292 | Adapter/conditioning/total memory models (`h3_*memory.rs` ~3.4 K), audio config, chunking, keyframe resize, reference planning (1,614), schedule (621), pruned AdaLN plan, and the **Veda sparse-attention host contract** (`veda/contract.rs` 1,392, `veda/workspace.rs` 1,213, `topk.rs`, `frozen-plans.json`). |
| `gen-core-ltx` | 5,022 | `ltx_checkpoint.rs` 2,956 (key maps and checkpoint facts), Gemma assets, DFR, duration head. |
| `gen-core-moss-image` | 4,015 | Request and edit contract, training admission and recovery. |
| `gen-core-memory-wan` | 3,266 (+2,152 tests) | Wan memory fingerprint, inventory, load, prepared, receipts, request, routes, selection. |
| `gen-core-family` | 5,826 | Small per-family host pieces: face align, FX program, Ideogram scheduler, InstantID keypoints, Krea multiphase schedule, Lens resolution, Qwen-Image-2.1 config/reference/scheduler, SD3 encoder artifacts, SDXL LDM, SeedVR2 video. |
| `gen-core` (shared by all) | ~87 K | Generator/registry/runtime contracts, sampling, tiling planner, `attention_budget` (rung-3 planner), `block_window`, residency, memory-strategy vocabulary (8.8 K), checkpoint codec, licence, comfy-quant, numeric helpers. |

**Why it works.** The upstream reference math was written in Python/PyTorch and **is partly host
math**: schedules, ids, RoPE frequencies, key maps, package identity, preprocessing, training
cadence. Pulling it into plain-`Vec<f32>` Rust:
1. needs no tensor abstraction;
2. is testable on any CI lane with golden fixtures (`tests/fixtures/flux3-*`);
3. gives byte-identical numbers to both backends, because the backend only uploads the host table.

The backend crates `use gen_core_flux3::{rope_cos_sin, timestep_freqs, DitConfig, MODES, …}`
(`mlx-gen-flux3/src/dit.rs:20-22`; `candle-gen-flux3/src/dit.rs:26`). Import counts:
`memory_strategy.rs` imports the contract 26× (MLX) and 23× (Candle); `model.rs` 25×/17×.

### 3.2 Shared fraction per family (src)

Formula: contract / (contract + MLX + Candle).

| Family | contract | MLX | Candle | shared |
|---|---:|---:|---:|---:|
| FLUX 3 | 12,964 | 7,901 | 11,638 | **39.9 %** |
| Iris | 21,077 | 46,839 | 17,314 | 24.7 % |
| H3 | 11,292 | 38,143 | 37,243 | 13.0 % |
| MOSS-Image | 4,015 | 15,132 | 13,237 | 12.4 % |
| LTX | 5,022 | 47,136 | 43,920 | 5.2 % |
| Wan (memory only) | 3,266 | 35,450 | 28,439 | 4.9 % |
| FLUX.1 / Qwen-Image / SDXL / Z-Image / Krea … | ~0–1 K via `gen-core-family` | | | ~0–3 % |

These match the prior audit (flux3 ~40 %, iris ~27 %).

### 3.3 What remains duplicated after "host math once"

Normalised twin similarity (§1). Shares are of non-blank, non-comment lines in same-named src files.

| Family | MLX lines | Candle lines | identical | % of MLX | % of Candle |
|---|---:|---:|---:|---:|---:|
| body | 1,799 | 1,853 | 1,179 | 66 % | 64 % |
| face | 2,661 | 3,020 | 1,718 | 65 % | 57 % |
| instantid | 2,138 | 2,582 | 1,297 | 61 % | 50 % |
| sam3 | 5,087 | 5,199 | 2,945 | 58 % | 57 % |
| svd | 3,397 | 4,564 | 1,659 | 49 % | 36 % |
| **flux3** | 6,820 | 10,120 | 3,230 | **47 %** | 32 % |
| pid | 4,283 | 3,109 | 1,972 | 46 % | 63 % |
| bernini | 10,147 | 11,182 | 4,348 | 43 % | 39 % |
| seedvr2 | 3,640 | 3,581 | 1,519 | 42 % | 42 % |
| scail2 | 5,172 | 6,651 | 2,094 | 40 % | 31 % |
| **minimax-h3** | 26,317 | 25,690 | 9,860 | **37 %** | 38 % |
| mochi | 3,088 | 2,969 | 1,079 | 35 % | 36 % |
| qwen-image-2-1 | 16,493 | 13,121 | 4,483 | 27 % | 34 % |
| sensenova | 9,856 | 8,341 | 2,461 | 25 % | 30 % |
| moss-image | 14,569 | 12,775 | 3,664 | 25 % | 29 % |
| boogu | 5,642 | 8,818 | 1,196 | 21 % | 14 % |
| pulid | 2,721 | 2,433 | 575 | 21 % | 24 % |
| ltx | 37,066 | 36,954 | 6,930 | 19 % | 19 % |
| ideogram | 3,296 | 6,929 | 634 | 19 % | 9 % |
| iris | 40,901 | 14,698 | 6,363 | 16 % | 43 % |
| anima | 11,573 | 6,794 | 1,232 | 11 % | 18 % |
| chroma | 5,137 | 5,279 | 583 | 11 % | 11 % |
| wan | 26,989 | 22,623 | 2,141 | 8 % | 9 % |
| sd3 | 8,533 | 10,343 | 647 | 8 % | 6 % |
| flux2 | 17,350 | 16,353 | 1,294 | 7 % | 8 % |
| krea | 20,155 | 26,981 | 1,508 | 7 % | 6 % |
| sana | 19,904 | 8,651 | 1,383 | 7 % | 16 % |
| qwen-image | 10,280 | 11,952 | 641 | 6 % | 5 % |
| kolors | 7,114 | 9,052 | 450 | 6 % | 5 % |
| lens | 10,520 | 10,160 | 461 | 4 % | 5 % |
| mage | 15,289 | 9,136 | 652 | 4 % | 7 % |
| flux | 9,404 | 12,730 | 303 | 3 % | 2 % |
| z-image | 46,890 | 16,790 | 890 | 2 % | 5 % |
| sdxl | 58,412 | 19,602 | 1,096 | 2 % | 6 % |
| **Σ 34 pairs** | **472,643** | **371,035** | **≈72,500** | **15 %** | **20 %** |
| LLM `models/` | 10,709 | 12,718 | 2,579 | 24 % | 20 % |
| LLM `primitives/` | 28,803 | 14,119 | 2,282 | 8 % | 16 % |

**What this shows.**

1. **The newer families are written as mirrors.** Candle files say so in their headers: 313 Candle
   files and 74 MLX files declare themselves "twin" of the other side. Example:
   `candle-gen-flux3/src/dit.rs:1-2`: *"the Candle twin of `mlx_gen_flux3::dit`"*.
   - The two `Flux3Dit` structs list the same fields in the same order, with the same doc comments
     and the same `push_mode` packing.
   - Candle adds a `device: Device` field and a `freqs: Tensor` cache. Its bounds are `usize`
     instead of `i32`, and its errors use `CandleError` instead of `mlx_gen::Error`.
     (Compare `mlx-gen-flux3/src/dit.rs:289-305` with `candle-gen-flux3/src/dit.rs:245-268`.)
   - Mechanical renaming covers half of these files.
2. **Divergence concentrates in the backend's layer module and backend-specific concerns.**

   | flux3 file | MLX | Candle | identical |
   |---|---:|---:|---:|
   | `lib.rs` | | | 73 % |
   | `pipeline.rs` | | | 68 % |
   | `pup.rs` | | | 59 % |
   | `dit.rs` | | | 57 % |
   | `model.rs` | | | 49 % |
   | `text_encoder.rs` | | | 48 % |
   | `training/checkpoint.rs` | | | 45 % |
   | `memory_strategy.rs` | | | 45 % |
   | `autoencoder.rs` | | | 39 % |
   | `training/trainer.rs` | | | 18 % |
   | **`nn.rs`** | **308 lines** | **1,157 lines** | **9 %** |

   - Candle's `nn.rs` (`candle-gen-flux3/src/nn.rs` items at lines 39–903) holds:
     - dtype promotion (`common`, `promotion_follows_torch`);
     - `PackedSource`/`PublisherSource`/`PublisherKernel` (FP8R / NVFP4 / GGML packed GEMM with
       row chunking);
     - the `Linear` enum over dense/packed/LoRA bases;
     - an explicit `Rope` struct with a device;
     - `packed_attention_impl`.
   - MLX's `nn.rs` is a thin `Linear` over `quantized_matmul` plus `layer_norm_f32` and `silu`.
   - Candle-only files: `convert.rs`, `autoencoder/bounded.rs` (756), `kernel_oracle_tests.rs`,
     `gpu_tests.rs`, and `training/{runner,profile}.rs`, which are the tape-based autodiff driver.
3. The same holds elsewhere:
   - **H3:** `denoise/packing.rs` 74 % and `spatial_tiling.rs` 73 % identical, but the text encoder
     only 18 %.
   - **MOSS-Image:** `training_memory.rs` 86 % identical, but `transformer.rs` only 8 %.
   - **LTX:** `diff_vae.rs` 67 % and `dfr.rs` 64 %, but `pipeline.rs` 7 %.
   - **Iris:** `train/mod.rs` 73 %, `train/depth.rs` 62 %, but `dit.rs` 17 %.

   **In every case the identical code is host orchestration, packing and tiling, and those are
   candidates for (b) today.**
4. **The older twins (FLUX.1, Qwen-Image, SDXL, Z-Image, Wan, Krea) are independent ports.** Their
   structure differs: different file decomposition, MLX `block_stream.rs`/`model_inference_device.rs`
   against Candle `packed_dit.rs`/`quant.rs`/`comfyui.rs`. **Neither (b) nor a generic trait helps
   them without a rewrite.** They are also the largest: SDXL, Z-Image, Wan, Krea and LTX total
   ~255 K MLX + ~155 K Candle src.
5. **There is also duplication within each backend.** `crates/media/candle-gen/CODE_REVIEW_2026-07-12.md:394`
   records that `Rotary`, `repeat_kv`, `mrope_cos_sin`, `causal_mask` and others are "near-byte-identical"
   between candle-gen-krea and candle-gen-boogu, one "ported verbatim". A write-once design needs a
   shared component library; a backend trait alone does not provide it.

---

## 4. Q2 — The tensor-op surface the model code actually uses

### 4.1 Raw op inventory

Method: count `[.:]<op>(` occurrences across `crates/media/mlx-gen`, `crates/llm/mlx-llm` and the
Candle equivalents (plus `crates/audio`; vendor excluded) against a 160-name candidate list. Output
files are `ops_mlx.txt` and `ops_candle.txt` in the scratchpad. **Caveat:** `max`, `min`, `abs`,
`var`, `sqrt`, `sum` also match `f32`/iterator/`std` methods, so those rows are inflated. Rows below
are grouped and rounded.

| Group | MLX (top call counts) | Candle (top call counts) | Notes |
|---|---|---|---|
| dtype/device | `as_dtype` 2,877 | `to_dtype` 2,742; `to_device` 569 | Candle carries explicit device moves. MLX has unified memory and per-stream CPU/GPU placement (`StreamOrDevice` 511 mentions). |
| shape/view | `reshape` 2,471, `transpose_axes` 918, `transpose` 352, `flatten` 307, `expand_dims` 238, `permute` 79, `squeeze` 45, `contiguous` 497 | `reshape` 1,963, `transpose` 848, `permute` 411, `unsqueeze` 400, `squeeze` 227, `flatten` 153, `contiguous` 1,558, `narrow` 1,382 | Candle needs `contiguous()` before many kernels: 1,591 calls vs 497. |
| slice/gather | `take` 373, `split` 428, `gather` 18, `take_along_axis` 4, `scatter` 3, indexing ops | `narrow` 1,382, `index_select` 111, `gather` 16, `where_cond` 15, `scatter` 2, `index_add` 2 | MLX slicing is mostly through `index`/`try_index` macros (not counted). |
| concat/split | `cat` 133, `concatenate(_axis)` 143, `stack` 77, `chunk` 55, `split` 428 | `cat` 827, `stack` 41, `chunk` 113, `split` 176 | |
| elementwise | `multiply` 305, `add` 263, `subtract` 170, `divide` 48, `exp` 101, `sin`/`cos` ~511, `square` 83, `powf` 208, `clamp` 185, `round` 113, `floor`/`ceil` 110, `sigmoid` 15, `tanh` 5, `relu` 18, `silu` 53, `gelu` 15 | `broadcast_mul` 666, `broadcast_add` 412, `broadcast_div` 183, `broadcast_sub` 121, `affine` 336, `sqr` 249, `powf` 206, `silu` 210, `gelu_erf` 63, `gelu` 51, `tanh` 51, `relu` 56, `neg` 63, `recip` 27 | **Broadcasting is implicit in MLX and explicit in Candle** (`broadcast_*` 1,382 calls). A trait must pick one. |
| reductions | `sum`, `mean` 224, `max`/`min`, `argmax` 14, `cumsum` 9, `softmax` 15, `std` 72, `sort` 151, `topk` 3 | `sum`, `sum_keepdim` 86, `mean_keepdim` 125, `softmax_last_dim` 32, `argmax` 10, `sort` 141 | MLX `keepdims` is a flag; Candle has separate `_keepdim` methods. |
| matmul | `matmul` 107, `matmul2d` 9, `addmm` (in `nn.rs`) | `matmul` 377 | |
| quant | `quantize` (mostly MLX `ops::quantize`; count inflated by name) 876, `quantized_matmul` 9, `qmm` 7, `dequantize` 9 | `quantize` 235, `QTensor`/`QMatMul` 363, `dequantize` 32 | Formats differ (§5). |
| conv | `conv2d` 39, `conv1d` 2, `conv3d` 1 (+ `nn::conv3d`), `conv_transpose1d/2d` 3, `conv_general` (nn) | `conv2d` 70, `conv1d` 16, `conv_transpose1d/2d` 15, `upsample_nearest2d` 28, `avg_pool2d` 11, `max_pool2d` 6 | **MLX convs are NHWC; Candle's are NCHW.** NHWC/channels-last is mentioned 1,189× in mlx-gen vs 317× in candle-gen. Candle-Wan has its own `conv3d.rs`. |
| fused "fast" | `fast::scaled_dot_product_attention` 36/`sdpa` 19, `fast::rms_norm` 51, `fast::layer_norm` 25, `fast::rope` 6, `group_norm` 12, `gelu_tanh_exact`, `silu_exact`, `exp2` | `sdpa` 11 (Metal, `candle_nn::ops::sdpa` e.g. `candle-gen-z-image/src/packed_dit.rs:318`), `flash_attn` 2 (CUDA, vendored FA2), `rms_norm` 52, `layer_norm` 20, `rope`/`rope_i` 33, `group_norm` 6 | Candle CUDA attention is materialised and query-chunked (`candle-gen/src/attention.rs:1-30`) or FA2 (`fused_attention.rs:1-60`). |
| random | `random::normal` 465, `key` 373, `uniform` 25, `split` 4 | `randn` 832, `rand` 81 | Different RNG streams, so **seeds are not cross-backend reproducible**. |
| execution | `transforms::eval` 650 (+ method `eval` ~860), `async_eval` 7, `compile` 25+, `checkpoint`, `eval_pending_loads` 35, memory limits 591 mentions | `synchronize` 390, `Device` 7,854 mentions | Lazy graph vs eager (§5). |
| autodiff | `keyed_value_and_grad` 66, `value_and_grad` 8, `grad` 2, `stop_gradient`/custom VJP 26 | `backward()` 168, `Var::from_tensor` 114, `GradStore` 163, `detach` 171 | Functional vs tape (§7). |

**Distinct ops matched: 118 (MLX) and 115 (Candle)** out of the 160 candidates. Total matched call
sites: 23,964 and 26,525.

### 4.2 Custom kernels (code that lives below any tensor API)

| Backend | Where | Size | What |
|---|---|---|---|
| CUDA | `crates/kernels/candle-quant-kernels/src/*.cu` | 12 `.cu` (decode_attention 561, moe_gemv 1,039, nvfp4_gemv 300, nvfp4_quant 158, fused_decode 255, flux3_quant 162, veda_int8 149, veda_score 52, training_attention 103, w4a8_dequant 56, kitchen_int8 46, yue2_stable_conv 71) + Rust wrappers incl. `cublaslt.rs` 2,584, `nvrtc.rs` 632 | NVFP4/FP8 GEMM via cuBLASLt, MoE GEMV, Veda sparse attention, training attention fwd/bwd, a YuE2 stable conv |
| CUDA | `candle-llm/src/primitives/{prism_cuda,sampler_cuda}.cu`, `candle-gen-minimax-h3/src/veda_pointer_v1.cu` | – | LLM prism/sampler, H3 Veda pointer |
| CUDA (vendored) | `vendor/candle-flash-attn` (1,209 Rust + CUTLASS) | – | FA2 with a multi-arch fatbin ladder |
| Candle `CustomOp` | 16 impls: `candle-llm/primitives/prism.rs` (MlxMatmul, GgufMatmul, MlxEmbedding, GgufEmbedding), `sampler.rs`, `candle-gen/src/{training_attention,reference_training_attention}.rs` (Forward/Backward), `train/{norm,krea_ops,projection}.rs`, `candle-gen-iris/src/tape.rs` (Fold1/Fold3), `decode_attention.rs` InplaceOp3 | – | Custom forward/backward ops |
| Metal (MLX `MetalKernel`) | 15 construction sites: `mlx-gen/src/{int8_gemm(540),w4a8_int8(526),convrot_int8(303),ios_quantum_kernels(1,312)}.rs`, `mlx-gen-minimax-h3/src/veda/metal.rs` 767, `mlx-gen-krea-realtime/src/compressed_kv.rs` 1,562, `mlx-llm/src/primitives/{packed_metal(3,452),gated_delta(2,250),prism(884),kv_candidates/*}` | MSL source inline in Rust | INT8/W4A8 GEMM, iOS quantum kernels, Veda sparse attention, packed KV attention |
| Metal (Candle) | vendored `candle-metal-kernels` ships **`mlx_gemm.metal` and `scaled_dot_product_attention.metal`** (MLX's "steel" GEMM and attention ported); head dims {32,64,72,80,96,128,256,512} (`vendor/candle-metal-kernels/src/kernels/sdpa.rs:77`) | – | Relevant to (f) |

**Veda shows that "kernel per backend, plan once" already works.** The host plan and contract live
in `gen-core-h3/src/veda/{contract,workspace,topk}.rs` (~2.8 K). The kernels are `veda_int8.cu` +
`veda_score.cu` + `veda_pointer_v1.cu` on CUDA and `veda/metal.rs` on MLX. A write-once model would
call `B::veda_sparse_attention(plan, q, k, v)` through an **extension trait** that each backend
implements, and the plan stays where it is.

### 4.3 How large a backend-neutral trait would need to be

From the inventory and two reference points:
- candle-core's `BackendStorage` has 31 methods and `BackendDevice` 12 (in
  `~/.cargo/registry/src/*/candle-core-0.9.2/src/backend.rs`). They back a `Tensor` API of ~141
  public methods (`tensor.rs`, `tensor_cat.rs`, `tensor_indexing.rs`), with unary and binary
  families as op enums.
- mlx-rs `ops/` exposes ~326 public functions (`mlx-rs/src/ops`, fork 11d0ff7), but the repo uses
  only ~118 names.

A trait that expresses **≥95 % of the model-graph code** (not kernels, not loaders) would need:

| Layer | Methods | Contents |
|---|---:|---|
| `Tensor` primitives (backend must implement) | ~50–60 | create (zeros/ones/full/arange/from_host/rand), dtype cast, reshape/permute/broadcast/contiguous, slice/narrow/index_select/gather/scatter/where, cat/stack/split, unary-op enum (~20 ops), binary-op enum with **implicit broadcasting** (Candle impl inserts `broadcast_as`), reduce-op enum with `keepdim` flag, matmul, cmp ops, to_host. |
| Derived ops (default methods) | ~80 | silu/gelu variants, softmax, chunk, repeat_kv, modulate, `layer_norm_f32`, `rotate_half`, upsample-nearest, pad, … overridable by a backend. |
| `Layers` (backend-provided modules) | ~12–15 | `Linear` (dense / MLX-affine / GGML / FP8R / NVFP4 / W4A8, + additive LoRA/LoKr residuals), `Attention::sdpa(q,k,v,mask,plan)` with budget/flash/streaming dispatch, `RmsNorm`, `LayerNorm`, `GroupNorm`, `Rope` (table-in), `Conv{1,2,3}d` with a **declared layout**, `ConvTranspose`, `Embedding` (dense/packed). |
| `Exec` hooks | ~6–8 | `materialize(&[T])` (MLX `eval`, Candle no-op or sync), `async_materialize`, `fuse(region)` (MLX `compile`, Candle identity), `memory_scope`, `cancel_point`, `stream_scope(cpu|gpu)`. |
| `Autodiff` | ~4–6 | `loss_and_grads(params, f)`, `checkpoint_segment(f)`, `stop_gradient`, `custom_vjp` registration. |
| Extension traits (opt-in per kernel) | n | Veda, packed KV, INT8 GEMM, MoE GEMV, … |

**Total ~150 methods, of which ~70–80 must be implemented per backend.** The rest are defaults.
That is the same order of magnitude as burn's `FloatTensorOps`/`ModuleOps` family and candle's own
storage trait. **It is not a multi-thousand-method surface.** The hard part is the *layer* and
*exec* semantics (§5–§7), not the op count. Confidence ≈70 % on these method counts (±30 %); I
derived them from call-site names, not from writing the trait.

---

## 5. Semantic differences a shared model definition must absorb

| Concern | MLX (`pmetal-mlx-rs` @11d0ff7, MLX 0.32) | Candle (1e6aa85 + vendored patches) | Evidence | Trait consequence |
|---|---|---|---|---|
| Execution model | Lazy graph; `eval`/`async_eval` materialise; `compile` fuses elementwise glue | Eager; each op launches | `mlx-gen/src/attention.rs:1-60` measures that per-chunk `eval` cuts the graph (−1.7 % denoise peak) | `materialize()` and `fuse()` are **hooks, not ops**. Candle implements them as no-ops. Model code must place materialize points deliberately (MLX has 1,514 `eval` sites today). |
| Memory | Unified memory; allocator cache with `set_memory_limit`/`set_cache_limit`/`get_active_memory` (591 mentions); CPU vs GPU streams | Discrete device memory on CUDA (explicit `to_device`, 599 sites); CUDA allocator; Metal shared buffers | `runtime-ios/src/lib.rs:925-945` (`bound_mlx_to_platform_limits`) | Memory policy stays per backend behind an `Exec`/residency hook. Model code must not encode device moves; the backend places tensors. |
| Device identity | One default stream/device | Every `Device::new_metal(0)`/`new_cuda(0)` is a **distinct, non-equal** device, so cross-op "device mismatch" | `crates/audio/candle-audio/src/lib.rs:168-182` (process-cached device); `candle-gen/CODE_REVIEW_2026-07-12.md:39` (CPU-built masks vs CUDA backbone) | A backend context object owns the device; tensor constructors take `&Ctx`. This removes a whole Candle bug class. |
| Broadcasting | Implicit (NumPy) | Explicit `broadcast_*` (1,382 calls) | §4.1 | The trait's binary ops broadcast implicitly; the Candle impl calls `broadcast_as`. |
| Dtype promotion | Automatic promotion | Ops require equal dtypes; FLUX 3 Candle reimplements torch promotion (`candle-gen-flux3/src/nn.rs:39-48`, test `promotion_follows_torch` `:1251`) | §3.3 | The trait specifies a promotion rule (torch-compatible) implemented once in the generic layer. |
| Shapes | `i32` dims | `usize` dims | dit twins | Generic code uses `usize`; the MLX impl converts. |
| Layout | Conv NHWC; weights `[out,kH,kW,in]` (`mlx-gen/src/nn.rs:1123`) | NCHW | §4.1 | `Conv` layer declares its layout, and the weight loader transposes once at load. Graph code that permutes around convs must go through a layout-aware helper. **Largest source of mechanical diffs in VAEs/UNets.** |
| Contiguity | Mostly transparent | Many kernels need `contiguous()` (1,591 calls) | §4.1 | `contiguous()` is a no-op on MLX. Over-calling is cheap on Candle only when the tensor is already contiguous. |
| Quant formats | Group-wise **affine** W4/W8 (`scale·q+bias`, groups 32/64), `quantized_matmul`; W4A16, so no activation quant (`gen-core-flux3/src/packed_gemm.rs:30-32`) | GGML `Q4_1` (lossless repack of MLX Q4) and `Q8_0` (re-quantised, 0.56 % rel RMS); **dequantise-then-dense matmul** to avoid `q8_1` activation quant producing NaN on outliers; NVFP4/FP8R via cuBLASLt; W4A8 | `candle-gen/src/quant/mod.rs:1-35` | Quantised `Linear` must be a backend-provided layer, not an op. The host packers and oracles (`gen-core-flux3/src/{affine,quantization,packed_gemm}.rs`) are already shared. |
| Attention | Fused Metal SDPA streams scores, so no `[Sq,Sk]` tensor; chunking only helps through graph cuts | CPU/CUDA `attention_basic` **materialises** scores, i32-indexing hazard above 2^31 elements, query-row chunking −32 % on CUDA; FA2 when eligible (bf16/f16, sm≥80, no additive mask) | `mlx-gen/src/attention.rs:1-60`; `candle-gen/src/attention.rs:1-30`; `fused_attention.rs:1-60` | **One `Attention::sdpa` layer whose backend dispatch differs.** The shared `gen_core::attention_budget` planner already exists. |
| Numerics | Different GEMM tilings; MLX f64 is CPU-only | Chunking changes the GEMM `M`, so outputs differ by 1–2 ULP (sc-15943); candle-Metal has no f32→f64 cast | `candle-gen/src/attention.rs:21-27`; `docs/reference/sheetsage2-utility-assessment.md:439` | Parity stays **per backend against the PyTorch reference** with tolerances. It is never cross-backend bit identity. |
| RNG | `random::key`/`normal` | `randn` | §4.1 | Noise is generated on host (or via a shared counter-based RNG) if cross-backend seed reproducibility is wanted. Today it is not. |
| Error/abort | C++ exceptions bridged; fork makes GPU watchdog errors surface as `Err` (sticky command-buffer error, CPU-stream wait error) | `Result` everywhere | `Cargo.toml:95-118` lineage comment | The trait's error type wraps both. Backend fork patches remain necessary. |

---

## 6. Q4 — Performance-critical constraints visible in the code

1. **MLX `compile` of elementwise glue is a measured win.**
   - Wan reports **~14 % faster per step**, bit-exact (`mlx-gen-wan/src/pipeline/denoise.rs:236`,
     `:401`).
   - There is a `compile_micro`/`perf.rs` harness per family (`mlx-gen-{flux,flux2,qwen-image,sdxl,ltx,wan,z-image}/tests/compile_micro.rs`).
   - The shared mechanism is `mlx-gen/src/nn.rs` `RetainedUnary/Binary/Ternary/Slice`,
     `DitCompileGlue`, `set_compile_glue` (`nn.rs:199-290`, `:747`).
   - **A write-once model must keep "fusable region" boundaries explicit** (`B::fuse(|x| gated(x, …))`).
     Otherwise MLX loses this.
2. **The GPU watchdog and lazy weight loads.** `mlx-gen/src/weights.rs:122-140`: evaluating a GPU
   cast over an unread lazy safetensors `Load` makes the Metal command buffer wait on disk. On a slow
   drive this passes the watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`).
   - The fix is `eval_pending_loads` first, a fork API (mlx-rs#32).
   - This lives in the **loader**, not in model code, so it survives any model-once design, provided
     the backend owns weight materialisation.
   - Any option that replaces MLX (d, f) must re-establish an equivalent guarantee.
3. **Per-chunk eval as a graph cut.** In MLX, the memory win of bounded attention comes from
   `eval_per_chunk` graph cuts (`mlx-gen/src/attention.rs:40-60`). In Candle it comes from not
   materialising scores. The same planner, `gen_core::attention_budget`, drives different mechanisms,
   which supports **shared planner, per-backend layer**.
4. **Candle CUDA kernels.**
   - i32 element indexing in elementwise/softmax/matmul (`candle-gen/src/attention.rs:1-10`).
   - Packed GEMM launches are chunked to stay within `i32::MAX` (`gen-core-flux3/src/packed_gemm.rs:15-16`).
   - CUDA-graph capture needed vendored candle patches (`vendor/candle-core/VENDORED.md`).
   - FA2 eligibility rules (`fused_attention.rs:14-45`).

   These are backend-layer concerns, so they stay below the trait.
5. **Quantised kernels.**
   - MLX affine `quantized_matmul` reads the compute-dtype input directly (W4/8A16).
   - Candle dequantises to dense or uses cuBLASLt NVFP4/FP8 with dynamic activation quantisation.
   - Each format's transient accounting differs, and `packed_gemm.rs::transient_bytes_per_row` already
     encodes it per backend, on host.
6. **iOS memory limits.**
   - `bound_mlx_to_platform_limits` (`runtime-ios/src/lib.rs:925`) caps MLX's memory and cache
     limits to `os_proc_available` memory.
   - The iOS lane is the narrow `mlx-gen-ios-catalog` (20,963 src).
   - About 103 K lines of `ios_*`/`quantum*` pricing and proof code are MLX-only.
   - Candle has iOS `cfg`s in the vendored Metal backend (shared storage instead of private:
     `vendor/candle-core/src/metal_backend/device.rs:79-82`). **This repo has no evidence that
     candle-Metal runs SceneWorks models on iOS.** Older public reports say candle-Metal did not run
     on iOS ([candle#1841](https://github.com/huggingface/candle/issues/1841),
     [Strathweb 2024](https://strathweb.com/2024/05/running-microsoft-phi-3-model-in-an-ios-app-with-rust)).
7. **Dispatch overhead is not a concern for a trait.** Ops are kernel launches (µs) or graph nodes,
   so static (generic) or dynamic dispatch costs ns. **A trait has no inherent performance cost.**
   Performance comes only from *which* kernels and *where* eval/fuse boundaries sit, and the
   two-level design keeps both under backend control.

---

## 7. Q5 — Training per backend and whether one model definition could serve it

**Today.**
- **MLX is functional.** A closure takes a `LoraParams` map, writes the factors into the model
  (`replace_training_lora`), runs the forward and returns the loss.
  `keyed_value_and_grad(loss_fn)(params, 0)` returns `(loss, grads)`
  (`mlx-gen-flux3/src/training.rs:87-111`). It has 66 `keyed_value_and_grad` sites, 8
  `value_and_grad`, and MLX `checkpoint` for activation checkpointing.
  - Shared loop: `mlx-gen/src/train/trainer_loop.rs:1-12` (`TrainLoopFamily` trait: the family
    supplies one micro-step's loss and grads; the loop owns accumulation, LR schedule, checkpoint
    cadence, previews).
- **Candle is a tape.** Parameters are `Var`s, and `(&loss * scale)?.backward()?` returns a
  `GradStore` (`candle-gen-flux3/src/training/runner.rs:292`, `:408`).
  - 168 `backward()` sites, 114 `Var::from_tensor`, 163 `GradStore`.
  - Candle **has no activation-checkpoint primitive**, so `candle-gen/src/train/gradient_checkpoint.rs:1-22`
    implements a manual segmented VJP. `CustomOp::bwd` cannot surface gradients for `Var`s inside a
    recomputed block.
  - Memory-efficient training attention is a CUDA kernel with fwd/bwd `CustomOp3`
    (`candle-gen/src/training_attention.rs:156`, `:266`).
- Training LOC: MLX 65 K src, Candle 68 K src. The host-side controllers have partly moved to
  contracts (`gen-core-train` 6.3 K, `gen-core-flux3/src/training/*` 2.6 K, `gen-core-iris` train ~13 K).
- **Twin similarity of training code** is 18 % (`flux3/training/trainer.rs`), 45 %
  (`training/checkpoint.rs`), 73 % (`iris/train/mod.rs`) and 86 % (`moss-image/training_memory.rs`).
  The differences sit in the step driver; the controllers are near-identical.

**Can one model definition serve both?** Yes, for forward graphs, with a small autodiff surface.

```rust
trait Autodiff: Backend {
    type Grads;
    /// Evaluate `f` with `params` as differentiable leaves; return loss and per-key grads.
    fn loss_and_grads(&self, params: &ParamMap<Self>, f: &mut dyn FnMut(&ParamMap<Self>) -> Result<Self::T>)
        -> Result<(Self::T, Self::Grads)>;
    /// Recompute `f` in the backward pass (MLX: `mx.checkpoint`; Candle: segmented VJP helper).
    fn checkpoint<F: Fn(&[Self::T]) -> Result<Vec<Self::T>>>(&self, f: F, xs: &[Self::T]) -> Result<Vec<Self::T>>;
    fn stop_gradient(&self, x: &Self::T) -> Self::T;   // MLX stop_gradient / Candle detach
}
```

- **MLX:** `loss_and_grads` = `keyed_value_and_grad` with the model reading params from the map.
  That is what FLUX 3 already does via `replace_training_lora`.
- **Candle:** wrap params as `Var`s, run `f`, `backward()`, extract by key. `checkpoint` =
  the existing `gradient_checkpoint.rs` machinery.
- **The model graph is identical in both.** The only requirement is that **trainable params are
  read through a parameter map or adapter seam** rather than baked into fields. Both FLUX 3 twins
  already do this.

Risks:
1. Custom backward kernels (training attention, norm/rope `CustomOp`s, Iris `Fold`) need an
   extension-trait implementation on each backend, which is what exists today anyway.
2. MLX graph-transform semantics (closures must be pure functions of params) are stricter than the
   Candle tape. A design that satisfies MLX also satisfies Candle, not the reverse. **Write to the
   MLX discipline.**
3. Gradient numerics differ per backend, so training parity stays per backend.

**Confidence ≈65 %** that a shared training forward plus this autodiff seam covers all current
trainers (LoRA/LoKr, full-adapter, Iris restoration/depth, MOSS-Image). Not verified: Iris's
tape-specific memory tricks (`candle-gen-iris/src/tape.rs`) may need their own hook.

---

## 8. Q3 — Options

LOC is reported against **2,027,709 (all)** and **1,429,335 (src)**. "Removable" means lines that
would no longer need to exist or be hand-maintained once the option is fully applied. These are
**estimates with explicit assumptions**, not measurements.

### (a) Status quo — per-backend models, share only orchestration and component contracts

- **Evidence.** This is the decided architecture. `docs/architecture/inference-rearchitecture.md:66`
  ("Preserve backend-neutral contracts and independent MLX/Candle implementations") and the
  **non-goal** at `:81` ("merge MLX and Candle implementations behind a lowest-common-denominator
  engine"). Also `crates/media/mlx-gen/docs/MODEL_ARCHITECTURE.md:134-139` ("each family crate owns
  its blocks") and `mlx-gen/src/nn.rs:1-8` ("we do not lift a block to a shared abstraction off a
  single model").
- **Pros:** zero migration; each port is free to use backend idioms; parity evidence is untouched.
- **Cons:**
  - Every model and every cross-cutting feature is written about twice (437 K duplicated-min media
    src).
  - Twin drift: 313 "twin" files must be kept in sync by hand. There are review findings of one
    side diverging, e.g. the `PR #1288 review, the MLX mirror of the Candle runner case` comment in
    `mlx-gen-flux3/src/training/trainer.rs:1073`.
  - Audio stays Candle-only on Mac.
- **Migration cost:** 0. **Risk:** low technical, high ongoing cost. **Performance:** best possible
  per backend.
- **Removable:** ~0 %.

### (b) Expand "host math once" — a tensor-free contract crate per family; tensor code per backend

- **Evidence that it works:** FLUX 3 at 40 % shared, Iris 25 %, H3 13 % (§3.2).
- **Evidence of headroom:**
  - ≈72.5 K normalised lines are textually identical across twins (§3.3). In raw LOC including the
    heavy doc comments that is roughly ×1.3–1.5, so ~95–110 K per side.
  - Most of it is orchestration, packing, tiling, memory declarations, training checkpoint and
    controller code, configs and tokenisers: host logic that can move to contracts without any
    tensor abstraction.
  - Memory-strategy files (76.5 K MLX + 59 K Candle src) are largely host declarations. FLUX 3's
    `memory_strategy.rs` is 45 % identical and Wan's `i2v_memory_strategy.rs` 61 %.
- **Pros:**
  - No new abstraction and no performance risk.
  - Fits the accepted architecture (contracts stay tensor-free).
  - Can be done family by family.
  - Shrinks the parity test surface (host goldens run once on any lane).
  - Gives the cross-cutting "hooks" a single home (memory contract, training controller, reference
    planning).
- **Cons:**
  - Tops out around 40 % shared even for the best family, because the forward graphs stay
    duplicated.
  - Does nothing for old independent ports unless they are restructured.
  - Moving host code needs a typed handoff (host tables to backend upload), which adds some
    boilerplate per family.
- **Migration cost:** moderate. Per family, mostly moving code and dedup review. The new families
  (H3, Bernini, MOSS, SAM3, Face, Body, SeedVR2, SVD, Mochi, PiD, InstantID, Qwen-Image-2.1) are
  cheapest because they are already mirrors.
- **Risk:** low. Host code is deterministic and golden-testable on the CPU lane.
- **Performance:** neutral.
- **Removable:** ~100–150 K src (≈7–10 % of src, ≈5–7 % of all), plus ~20–40 K of tests.
  **Confidence medium (≈60 %)** on the range.

### (c) Our own thin backend-neutral tensor trait; model graphs written once, generic over the backend

**Shape.** A two-level design (§4.3):
- generic `Tensor` glue of ~50–60 primitives plus ~80 defaults;
- backend-provided `Layers` (quantised Linear, Attention, Norms, Rope, Conv with layout, Embedding);
- `Exec` hooks (materialize, fuse/compile, memory scope, cancel, stream);
- `Autodiff` (§7);
- per-kernel extension traits.

Family code becomes a generic crate (`gen-model-flux3<B: Backend>`), and the existing
`mlx-gen-flux3`/`candle-gen-flux3` shrink to registration, loaders and backend-only extras.

**Evidence that it fits:**
- In mirrored families the model graph is 45–70 % textually identical after renaming types alone.
  The residue sits in `nn.rs`-type modules, which become the `Layers` implementations (§3.3).
- The op surface is small and closed (§4.1–4.3).
- Prior art exists in-repo for "plan once, kernel per backend": Veda, `attention_budget`,
  `block_window`, tiling, `packed_gemm` (§4.2).
- The Candle `quant` module is already described as *"the candle twin of `mlx_gen::quant::{lin, embedding}`"*
  (`candle-gen/src/quant/mod.rs:1-4`), so a `QuantLinear` layer contract exists informally.

**Pros:**
- Removes the dominant duplication.
- New models are written once and parity-tested against the PyTorch reference on each backend.
- Cross-cutting hooks (LoRA/LoKr adapters, memory scopes, cancel points, block streaming, load
  progress, telemetry, compile regions) live in **one** generic layer or in backend `Exec`
  implementations, and every model inherits them.
- **Audio could get an MLX path at almost no model cost**, removing the cross-backend audio seam
  (`docs/architecture/audio-backend-strategy.md`).
- **It is not lowest-common-denominator.** Backends keep their own layer implementations and kernels.

**Cons:**
- Contradicts the recorded non-goal (`inference-rearchitecture.md:81`). That needs an explicit
  decision reversal with the "not LCD" argument above.
- Generic Rust (trait bounds, associated types) is harder to read than concrete code.
- Error messages get worse. Compile times are roughly unchanged (two monomorphisations, as today's
  two crates).
- Old independent ports need a real rewrite, not a rename.
- A backend-specific optimisation that changes graph structure (MLX `block_stream.rs`, Candle
  `autoencoder/bounded.rs`) has to become a hook or a backend override point. Some families have
  several of these.

**Migration cost:**
- Trait plus two backend implementations plus test harness: an estimated ~8–15 K lines and a few
  weeks of one engineer, with the pilot included (low confidence on the time).
- Per family, roughly proportional to the smaller twin for mirrored families (rename and lift), and
  to the larger twin for independent ports (rewrite).
- Every migrated family needs a parity re-run on both backends. That falls under the existing
  "campaign once at epic end" rule (FEATURE_DEVELOPMENT.md gate teardown), not per change.

**Risk:**
- Medium.
- The main technical risks are training/autodiff (§7), `fuse`/`materialize` placement silently
  changing MLX memory and performance, and NHWC/NCHW layout bugs in VAEs and UNets.
- The main organisational risk is a long dual-state period (some families generic, some twinned).

**Performance:**
- Neutral by construction if the generic code calls the same layer implementations and keeps the
  same eval and compile boundaries.
- **This must be proven per pilot** with s/step and peak-memory A/B, not assumed.

**Removable:**
- Media: sum of min twin sides 437 K × (0.65–0.80) after keeping backend residue (FLUX 3 residue:
  ~35 % of the Candle side — `nn.rs`, `convert.rs`, bounded AE, tape runner — and ~19 % of the MLX
  side) gives **~285–350 K src**.
- LLM: models plus primitives glue ~15–30 K.
- Tests: duplicated parity and contract tests become one generic test run per backend, ~60–120 K.
- **Total ~360–500 K ≈ 18–25 % of all lines (25–33 % of src).** Confidence low–medium (≈45 %) on
  the range. It depends on how many old ports are rewritten rather than left twinned.

#### (c2) Variant: make MLX a *candle backend* — candle's `Tensor` as the shared API

The repo already vendors `candle-core` (`Cargo.toml:177-180`), so a fourth `Storage` variant backed
by `mlx_rs::Array` is conceivable. Then all model code would be written against candle's API, which
already has autodiff and `CustomOp`.

- **Against:**
  - candle's `Storage`/`Device` enums are closed. Adding a variant is a permanent fork of
    candle-core internals far beyond today's two cherry-picks.
  - candle tracks `Layout`/strides itself, while MLX owns its own.
  - candle's quantised path is GGML `QTensor`, not MLX affine.
  - candle is eager by API: no `eval`/`compile` concept, so MLX's fusion and graph cuts would need
    side channels.
  - MLX's functional autodiff would be replaced by candle's tape running over lazy arrays. Possible,
    but every op becomes a graph node that the tape also records, with unclear memory behaviour.
- **Verdict:** higher risk than (c) for the same benefit. **Not recommended** (≈70 % confidence).
  It is the only variant that gets "model once" without writing our own trait, so it is listed for
  completeness.

### (d) Adopt an existing multi-backend framework (burn with CubeCL)

**Facts:**
- burn is MIT/Apache-2.0
  ([Phoronix on 0.20](https://phoronix.com/news/burn-0.20-released)).
- The latest release is **v0.22.1 (20 Oct)**, with v0.23.0-pre.1 on 29 Oct
  ([GitHub releases](https://github.com/tracel-ai/burn/releases)).
- In 0.22, "Candle has been removed… NdArray and LibTorch are being deprecated", so burn is
  consolidating on CubeCL (CUDA, ROCm, Metal via a wgpu MSL compiler, Vulkan, WebGPU).
- Quantisation: `QuantValue` variants Q8F/Q8S/Q4F/Q4S/Q2F/Q2S/E4M3/E5M2/E2M1 with block scales and
  `PackedU32` stores
  ([burn-std quantization.rs](https://docs.rs/burn-std/latest/src/burn_std/tensor/quantization.rs.html)).
  The documented mode is **symmetric**
  ([QuantScheme](https://burn.dev/docs/burn/tensor/quantization/struct.QuantScheme.html)).
  Asymmetric or affine is not confirmed. **Our shipped MLX tiers are affine** (scale + bias), and
  GGML Q4_1 is asymmetric.
- Only post-training quantisation; no QAT (docs.rs 0.20.1).
- CubeCL flash attention was "near-complete" at 0.8, and a downstream project reports a hardcoded
  `causal: true` issue ([burn_attention / tribev2 search results](https://docs.rs/crate/burn_attention/latest/source/README.md)).
- Fusion and autotune exist.
- Prior SceneWorks assessment (memory `epic_10386_amd_rocm_support.md`, 2026-07-08):
  - burn/CubeCL was "Route B… a CONSOLIDATION play: one impl could replace mlx-gen+candle-gen+add
    AMD… Cost=re-port whole zoo+quant+parity. Sharp risk=Mac perf/mem regression vs hand-tuned MLX.
    De-risk via SDXL 3-way benchmark spike", folded into story 10387.
  - **No such benchmark result exists** in the repo docs or memory.

**Pros:**
- Model once, autodiff included.
- Adds ROCm/Vulkan/WebGPU (AMD epic 10386).
- One upstream instead of two forks.

**Cons:**
- **Total re-port:** ~1.16 M media src plus LLM and audio, and all parity evidence regenerated.
- Loses MLX's tuned Metal kernels (steel GEMM and attention), `mx.compile`, unified-memory
  allocator controls and the fork's watchdog/error patches. These are the reasons MLX was chosen
  for the Mac.
- No affine-quant parity with our published MLX tiers; re-quantisation would change shipped artefacts.
- Unproven on iOS (active lane, ~124 K MLX-only lines).
- Rapid churn: backends removed and deprecated within one minor version.
- Our 27+ custom kernels would need CubeCL rewrites.
- A third-party abstraction stands between us and kernel-level fixes; today we patch MLX and candle
  directly.

**Migration cost:** very high, a whole-zoo rewrite.

**Risk:** high, both for performance on Mac and iOS and for maturity on quantised DiTs at our
sequence lengths (~141 K tokens in H3: `fused_attention.rs:8-12`).

**Performance:** unknown for this workload. Must be benchmarked (SDXL or FLUX 3 DiT, three-way)
before any commitment.

**Removable (steady state):** similar to (c), ~20–30 %, and possibly more, since both engine roots'
shared code (`mlx-gen` root ~60 K, `candle-gen` root ~50 K) could largely be replaced. But it
**requires rewriting ~100 % of model code** to get there.

**Verdict:** not now (≈75 % confidence). Re-evaluate only on a benchmark showing Mac s/step and peak
memory within ~10 % of mlx-gen, with affine-quant support.

### (e) Single-source codegen, or macro-generated twins

- **Idea:** write each model in a DSL or annotated Rust once, then generate `mlx-gen-*` and
  `candle-gen-*` source (build.rs or macro_rules/proc-macro over a token-substitution map like §1's
  normaliser).
- **Evidence it is mechanically feasible:**
  - For mirrored families, the type and API rename map in `sim.py` (about 20 rules) already makes
    35–66 % of lines identical.
  - The remaining differences are API shape (implicit vs explicit broadcast, `_keepdim` methods,
    `to_device`, `contiguous`, `i32`/`usize`, error types) and real backend layers.
- **Pros:** zero runtime abstraction; generated code reads like today's twins.
- **Cons:**
  - A substitution macro breaks on any real semantic difference, which becomes `#[cfg]` islands.
  - Generated code is hard to debug and review, and IDE support is poor.
  - Two copies still compile and still need tests.
  - Everything (e) can express, (c) expresses with ordinary generics, type-checked once.
  - A DSL-to-Rust compiler is a project in itself.
- **Migration cost:** similar to (c), plus tooling. **Risk:** medium–high (tooling debt).
  **Performance:** neutral.
- **Removable:** ~the same hand-maintained lines as (c), but generated lines still exist in the
  build (and the repo, if checked in).
- **Verdict:** inferior to (c) (≈80 % confidence). Useful at most as a *migration aid*: a one-shot
  rename script to lift a mirrored twin into the generic crate.

### (f) Consolidate onto one backend per platform differently

**f1 — candle on every platform (candle-Metal on macOS), retire MLX:**
- Candle-Metal already ships to users for audio (`runtime-macos/Cargo.toml:36-45`): **"~38x
  steady"** over CPU on the diffusion audio models (moss-sfx, acestep — sc-13039).
- Vendored candle-Metal includes MLX's steel GEMM (`mlx_gemm.metal`) and steel attention
  (`scaled_dot_product_attention.metal`). The kernel quality gap for dense matmul and attention may
  therefore be small.
- **Gaps recorded in-repo:**
  - candle-Metal lacked `upsample_nearest1d` (worked around).
  - It has no f32→f64 cast.
  - Rank > 4 strided reduce read out of bounds (vendored fix #3873).
  - Device-identity bugs (sc-13922).
  - Reference renders are non-reproducible on Metal in Qwen-Image-2.1 tests (memory
    `epic_25899…`: "Candle on METAL: reference renders not reproducible… four S3 request_surface
    tests fail on Metal").
- Candle has no lazy graph, `compile` or allocator limits like MLX's.
- Its Metal quantised kernels are GGML formats; MLX affine tiers would be repacked as Q4_1 losslessly
  and Q8_0 lossily (0.56 % rel RMS).
- **No MLX-vs-candle-Metal media benchmark exists anywhere** in repo docs or memory.
- iOS is unproven and is the active MLX lane.

**Removable:** all of `mlx-gen` (929,647) and `mlx-gen-ios-catalog`, plus `mlx-llm` (139,317):
**~1.07 M lines ≈ 53 % of all**. That is the largest of any option. But MLX-only capability would
have to be ported to candle:
- the iOS lane and its ~103 K pricing/quantum code;
- krea-realtime and SAM2;
- 15 Metal kernel sites;
- mlx-llm Metal primitives (~34.7 K);
- per-family MLX memory realisations;
- the fork's watchdog and error semantics.

The net is therefore maybe 35–45 %. **Risk:** high, mostly from Mac performance and memory, plus
iOS. **Verdict:** not now (≈65 %), but **benchmark it first.** Cost is one DiT forward A/B on the
dev Mac: FLUX 3 DiT dense bf16 and q4 at 1024², candle-Metal vs MLX, s/step and peak footprint.
Under the Mac-GPU-loan rule that is under an hour of GPU time. If candle-Metal is within ~15 % on
s/step and memory, this becomes the cheapest route to "write once" for macOS desktop; iOS would
still decide.

**f2 — MLX everywhere (MLX-CUDA on Linux/Windows), retire candle:**
- Upstream MLX has a Linux CUDA backend (`pip install mlx[cuda]`) and has been adding Windows CUDA CI
  (#3775, #3809, mid-2026) ([MLX README mirror](https://upd.dev/ml-explore/mlx)).
- Our fork only has "**throwing CUDA and no-GPU backend stubs**" (`crates/media/mlx-gen/Cargo.toml:39-42`).
- No NVFP4/FP8 cuBLASLt path, FA2 or CUDA-graph equivalents are known in mlx-rs.
- Audio (151 K) and candle-llm are candle-only.

**Removable:** `candle-gen` (659 K) plus `candle-llm` (94 K) ≈ 37 %, minus large ports.
**Verdict:** not viable today (≈85 %). Revisit when MLX-CUDA on Windows is GA and the fork can build
it.

### 8.1 Comparison matrix

| Option | Removable (all lines) | Migration cost | Technical risk | Mac perf | CUDA perf | iOS | Fits recorded decisions |
|---|---|---|---|---|---|---|---|
| (a) status quo | ~0 % | 0 | low | best | best | ✓ | ✓ |
| (b) expand host-math | ~5–7 % (+tests) | low–moderate, incremental | low | = | = | ✓ | ✓ |
| (c) own two-level trait | ~18–25 % | moderate (trait) + per-family | medium | = if hooks kept (must prove) | = | ✓ (MLX impl) | needs reversal of non-goal `:81` |
| (c2) MLX as candle backend | ~18–25 % | high (candle-core fork) | high | risk (no compile/eval) | = | ? | ✗ |
| (d) burn/CubeCL | ~20–30 % (steady) | very high (re-port all) | high | unknown | unknown | unproven | ✗ |
| (e) codegen/macros | ~same as (c) hand-maintained | moderate + tooling | medium–high | = | = | ✓ | partial |
| (f1) candle everywhere | ~35–45 % net (53 % gross) | high | high | **unmeasured** | = | unproven | ✗ (audio doc kept MLX for media) |
| (f2) MLX everywhere | ~25–35 % net | very high | very high | = | unknown/immature | ✓ | ✗ |

---

## 9. Q6 — Recommendation and staged path

**Recommendation: (b) now as the base layer, with (c) as the target, reached through a gated pilot.
Run the cheap (f1) benchmark in parallel so the decision is made with data.**

- **Confidence (b):** high (≈85 %). It is no-regret: it reduces duplication and gives cross-cutting
  hooks a home regardless of what follows.
- **Confidence (c) as end state:** medium (≈60 %).
- **Confidence in rejecting (d) and (e) now:** ≈75–80 %.

### Stage 0 — no-regret consolidation, done as part of the re-architecture

1. For mirrored families (H3, Bernini, MOSS-Image, Qwen-Image-2.1, SAM3, Face, Body, InstantID,
   PiD, SeedVR2, SVD, Mochi, Scail2), move host-identical code into their contract crates or
   `gen-core-family`. Use the per-file similarity tables (`sim.py` output) as the worklist. Target
   modules: `denoise/packing`, `spatial_tiling`, `training_memory`, `training/checkpoint`, config,
   tokeniser and text helpers.
2. Lift the per-family `memory_strategy.rs` host declarations into contract crates, as
   `gen-core-memory-wan` already does for Wan. Coordinate with the memory researcher.
3. Fix the within-backend duplication (krea/boogu Qwen-VL pieces, CODE_REVIEW:394) by creating shared
   *component* crates per backend.

### Stage 1 — trait walking skeleton (pilot A, inference-only, small)

- **Scope:** one small, highly mirrored module. The candidates are **Face** (65 %/57 %),
  **Body** (66 %/64 %) or **SAM3** (58 %/57 %, an attention-heavy ViT).
- **Work:**
  - Define `Backend` (primitives with implicit broadcast and torch promotion), `Layers` (Linear,
    Attention, LayerNorm, Conv2d with layout), `Exec` (materialize, fuse).
  - Implement `MlxBackend` and `CandleBackend` as thin wrappers in `mlx-gen` and `candle-gen`.
  - Write `gen-model-<fam><B>` once.
- **Acceptance evidence:**
  1. Existing parity tests pass on both backends at their current tolerances, with no new tolerance
     loosened.
  2. Outputs are bit-identical to the current twin on each backend for the same weights and inputs.
     The generic code calls the same kernels, so anything else indicates a semantic mismatch.
  3. s/step and peak memory are within noise of the current twin, MLX measured on the dev Mac under
     the loan rule.
  4. The removed LOC is reported.

### Stage 2 — pilot B: FLUX 3 end to end (DiT + AE + Qwen3-VL text encoder + PUP + LoRA training)

- **Why FLUX 3:** it is the most "host math once" family (40 %) with the highest graph similarity
  among large models (`dit.rs` 57 %). It exercises every hard seam:
  - quantised Linear in five storages (MLX affine, GGML Q4_1/Q8_0, FP8R, NVFP4: `packed_gemm.rs`);
  - packed attention;
  - RoPE from host tables;
  - the bounded autoencoder (a Candle-only `autoencoder/bounded.rs`, which becomes a backend override
    or a generic bounded-decode hook);
  - **training** with `keyed_value_and_grad` vs `GradStore`.
- **Acceptance evidence:**
  - Stage 1's three checks.
  - Training parity: the existing `training_parity.rs`/`training_checkpoint.rs` on both backends, and
    the `each_workflow_steps_exactly_the_factors_its_graph_reaches` invariant
    (`mlx-gen-flux3/src/training/trainer.rs:1073-1090`).
  - The MLX compile-glue A/B shows no regression.
  - The CUDA quantised path stays packed (`no_dmmv_switch_or_dequantizing_qmatmul_constructor_is_reachable`,
    `candle-gen-flux3/src/nn.rs:1196`).
- **Go/no-go:** go if all hold and the generic FLUX 3 is ≤ the larger of today's twins in LOC (target:
  ≤ MLX 7.9 K + Candle residue ~4 K). **No-go → stay at (b)** and record the decision, with the
  failing seam as the reason.

### Stage 3 — rollout

- Migrate mirrored families first (cheap), then rewrite old independent ports **only when they are
  touched for a feature** (FLUX.1, SDXL, Z-Image, Wan, Krea, Qwen-Image), not as a big bang.
- Put new models in generic crates by default.
- **Audio:** once the trait is stable, a generic audio model gains an MLX path, so the macOS audio
  lane can run on MLX and the cross-backend audio seam can retire. That reverses
  `audio-backend-strategy.md` §4's "MLX escape hatch" from per-model to wholesale.

### Parallel de-risking measurements (each is one run, no campaign)

1. **(f1) benchmark:** FLUX 3 or Z-Image DiT forward, dense bf16 and q4, 1024², candle-Metal vs MLX,
   s/step and peak footprint, on the dev Mac (loan the GPU, state duration, hand back).
2. **(d) benchmark**, only if (f1) and (c) both disappoint: the long-planned SDXL three-way (burn-Metal
   vs mlx-gen, burn-CUDA vs candle-gen), per story 10387 in memory.
3. **Trait overhead micro-benchmark** in the pilot: generic vs concrete call path at production
   shapes. Expected to be a no-op; prove it once.

---

## 10. Design implications for the re-architecture

1. **Put the backend seam at the layer level, not the op level.**
   - Model graphs (blocks, packing, modulation, residuals) are generic.
   - Quantised Linear, Attention, Norm, Conv (with layout), Embedding and Rope are backend-provided.
   - The evidence: FLUX 3's `nn.rs` is 9 % similar while its `dit.rs` is 57 %.
   - This is also what makes it "not lowest-common-denominator", which answers the recorded non-goal.
2. **Make execution policy a hook set, not code inside models.** `materialize` (MLX eval / graph
   cut), `fuse` (mx.compile region), `memory_scope`/residency, `cancel_point`, `stream_scope`,
   `load_progress`. Every cross-cutting feature in the brief attaches here once:
   - memory admission and bounded rungs;
   - block streaming;
   - cancellation;
   - telemetry;
   - load progress ("generic for every model", per memory);
   - compile glue.
3. **Keep "plan once, kernel per backend" as the extension pattern.** Host plan in a tensor-free
   contract (Veda, `attention_budget`, `packed_gemm`); kernels behind opt-in extension traits per
   backend. This is the only pattern already proven across 3+ kernels here.
4. **Weight materialisation belongs to the backend loader.** The GPU-watchdog guarantee
   (`eval_pending_loads` before any GPU-derived view, `mlx-gen/src/weights.rs:122-140`) and Candle's
   device and dtype placement then stay correct for every model automatically.
5. **Trainable parameters are read through a parameter map or adapter seam** (the existing
   `replace_training_lora`/`AdaptLinear` pattern). That is the precondition for a single forward to
   serve MLX functional autodiff and the Candle tape. Write models to MLX's purity discipline.
6. **Contracts stay tensor-free** (no change to the existing rule). The generic model crates are a
   *new tier* between contracts and backends. The workspace rule "an mlx provider never depends on a
   candle crate" holds: generic crates depend on neither, and backends depend on them.
7. **Parity remains per backend against the PyTorch reference**, never cross-backend bit identity.
   GEMM tiling, chunking (1–2 ULP) and RNG differ. One generic parity test, instantiated per backend,
   replaces today's two copies.
8. **Normalise the API differences once in the trait:**
   - implicit broadcasting;
   - torch-compatible dtype promotion;
   - `usize` shapes;
   - `keepdim` flags;
   - backend-owned device context. This kills the `Device::new_*` non-equality bug class.
9. **Layout must be explicit for convs** (NHWC on MLX vs NCHW on Candle). Convert weights at load,
   and keep graph code layout-agnostic through layout-aware helpers. This is the most likely source
   of bugs when lifting VAEs and UNets.
10. **Do not delete mlx-rs or candle fork patches** in any option. They encode correctness fixes
    (watchdog errors, CUDA graph capture, Metal reduce OOB) that a trait does not replace.
11. **The iOS lane is the constraint that most limits consolidation onto one backend.** Any design
    must keep MLX first-class unless candle-Metal or another backend is proven on iOS.

---

## 11. Uncertain or not verified

- **No performance numbers** compare MLX with candle-Metal, burn-Metal or a generic trait on any
  media model. Every performance statement for (c), (d) and (f) is reasoning, not measurement.
- **Trait size** (~150 methods, ~70–80 required per backend) comes from call-site names, not from
  writing the trait. Hidden needs (custom indexing macros in MLX via `index`/`try_index`, Candle
  `IndexOp`) were not inventoried and could add ~10–20 methods.
- **The twin-similarity metric** pairs files by name only. It under-counts renamed twins and does
  not measure semantic duplication in independent ports. The ≈72.5 K figure covers non-blank,
  non-comment lines; the raw-LOC conversion (×1.3–1.5) is an assumption.
- **Removable-LOC ranges** for (b) and (c) assume backend residue of 20–35 % per family, taken from
  FLUX 3 only. Older families may carry more (MLX `block_stream.rs`, Candle `packed_dit.rs`/`comfyui.rs`).
- **The src/test split** keeps inline `#[cfg(test)]` modules in "src". The true src share is lower.
- **burn facts** come from web sources and were not verified against the crate source:
  - latest version 0.22.1 and Candle backend removal (GitHub releases page);
  - quant variant names (docs.rs);
  - whether asymmetric/affine quant exists;
  - flash-attention maturity and iOS viability.
- **Candle-Metal on iOS:** the code has iOS `cfg`s; public reports from 2024 say it did not work;
  current status unknown.
- **The MLX-CUDA Windows status** comes from upstream CI commit titles, not release notes. The fork
  only has stubs.
- **The autodiff seam (§7)** has not been checked against the Iris tape tricks
  (`candle-gen-iris/src/tape.rs` Fold ops), YuE2/MOSS audio trainers, or LLM training, if any.
- **Unexplained gaps in the measurements:**
  - why MLX Iris and Z-Image are 2–3× their Candle twins: partly iOS pricing (~13.5 K and ~31 K)
    and streaming; the rest was not classified;
  - whether Candle "missing" features are real gaps;
  - LLM decode-path similarity beyond `models/` and `primitives/`.
- **The recorded non-goal** (`inference-rearchitecture.md:81`) and the policy at
  `MODEL_ARCHITECTURE.md:134-139` are design decisions; reversing them is the user's call. I did not
  find the original rationale beyond those lines.
- **I did not query CodeGraph or Shortcut** for prior decisions on a backend trait. The only prior
  record found is the burn/CubeCL "Route B" note in memory `epic_10386_amd_rocm_support.md`
  (story 10387).
