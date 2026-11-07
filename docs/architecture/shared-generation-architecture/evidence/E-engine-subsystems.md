# E — Non-denoise engine subsystems: weights, quant, adapters, catalog, training, audio/LLM, iOS, testing

Research input for the SceneWorks + inference re-architecture. Scope: everything around the
denoise loop that every model family re-implements — weight lifecycle, quantization, adapters,
catalog/capabilities, training, the audio/LLM lanes, iOS constraints, and the test harness.

**Sources.** Inference is read ONLY at `origin/main = a43724b82` (cited `inf:<path>:<line>`).
SceneWorks is read at this worktree's HEAD `a1d95e2c4` (cited `sw:<path>:<line>`). Line numbers are
from `git show origin/main:<path> | cat -n` / `git grep -n`. Byte sizes are blob sizes from
`git ls-tree -l` (bytes, not lines). Nothing was built or executed.

**Confidence legend.** [H] read directly in source; [M] inferred from headers/docs plus partial
reading; [L] extrapolation — flagged again in the last section.

---

## 0. Executive summary

1. **The weight lifecycle has a good backend-neutral spine that almost nobody routes through.**
   `gen-core` already owns `WeightsSource` (local paths only), file pins/seals, source inventory,
   header reading (`weightsmeta`), the codec registry + `LogicalKeyMapping` → `LogicalWeightPlan`
   compiler, tier integrity, residency (`Residency<Text,Heavy,R>`), and the rung-4 block window
   (`BlockWindowBackend`). But the plan compiler is driven by only **5 production mappings in 4
   crates** (candle flux2/krea/wan + MLX krea ×2) [H], and the SceneWorks-native **MLX affine
   packed tier (u32 codes + scales + biases) is not a registered codec at all** [H] — it is
   packed-detected per family by sniffing `{base}.scales`. Every other family keeps a hand-written
   loader (`convert.rs`/`loader.rs`/`weights.rs`/`quant.rs`), ~137 key-remap/sanitize fns [H], and
   **15 separate `struct Weights` wrappers** [H].
2. **Quantization is two-stage and asymmetric.** SceneWorks ships pre-quantized MLX-affine tiers
   produced offline by per-family `convert.rs::prequantize_*`; MLX loads them packed (or quantizes
   bf16 at load via `AdaptableLinear::quantize`); **Candle re-encodes the MLX tier at load** into
   GGML `Q4_1`/`Q8_0` (Q8 is a lossy double-quant) and caches the result as content-addressed
   sidecars [H]. ComfyUI imports (scalar fp8, MXFP8, NVFP4, int8-per-row) go through the codec
   plan; GGUF k-quants have a Wan-only path; W4A8/ConvRot are descriptor-driven Linear bases; the
   LLM crates have their own `QuantizedLinear`. There are **68 `*Linear*` struct/enum definitions**
   outside vendor code [H]; the causes are enumerable (§2.4) and mostly removable.
3. **Adapters: MLX has one runtime spine; Candle has four strategies.** MLX:
   `AdaptableLinear{base, Vec<Adapter>}` + `AdaptableHost` path routing, ~102 host impls in 53
   files [H]. Candle per-family: fold-at-load (SDXL, Z-Image, Wan, SD3, …), forward residual via
   `AdaptLinear` (Qwen-Image-2.1, FLUX.2, Anima, …), residual via training `LoraLinear` (LTX),
   residual via family `LinearNoBias` (MiniMax-H3) [H]. Per-phase adapter weights (Krea
   multi-phase) force candle to **load a second, job-local DiT** to toggle adapters [H]; LTX
   per-pass scales forced a **LTX-local `Linear`** on MLX [H]. Stable Audio 3 has 8 adapter types
   including non-commutative DoRA/BoRA that need the base weight [H].
4. **Capabilities are hand-declared (177 literal sites, 60+ fields) and then policed by source
   scanners**: a 4,516-line Rust-source parser re-derives `supports_preview` from sampler call
   sites [H], an additivity scanner polices literal syntax [H], conformance tests check
   `plan_driven_backends` both ways [H]. The SceneWorks dump consumes only a thin slice
   (`supportsPreview`, latent space, decoder options, imports, memory contracts, route witnesses)
   [H] while worker `MODEL_TABLE` re-declares defaults [H].
5. **Training reuses inference modules on MLX (same `AdaptableLinear`, functional autograd) but
   forks on Candle** (separate `LoraLinear`; vendored trainable DiTs because fused candle
   `CustomOp`s have no backward) [H]. Per-family trainer code is ~4.6 MB across both backends
   (§5.3) [H].
6. **Audio and LLM lanes are parallel universes.** Audio implements `gen_core::Generator` but uses
   none of `Residency`, memory-strategy registration, or `candle_gen::quant` (0 hits each) [H].
   LLM has its own contract crate (`core-llm`: own `CancelFlag`, progress, registry, resource
   admission) [H]. Text encoders (Qwen3/Gemma/T5/CLIP/ChatGLM) are re-implemented in ~15 media/audio
   files even though `mlx-llm`/`candle-llm` hold the same architectures; `mlx-gen-ltx → mlx-llm`
   (Gemma 4) is the first inversion edge [H].
7. **iOS imposes crate-granular feature gating** (`default-features=false` strips `training`),
   a narrow catalog, per-quantum OS-headroom checks, and evidence-table admission [H]. Any shared
   layer must keep training, video, audio and CUDA-only codecs **compile-time separable**.
8. **Testing**: 236 Python `dump_*` producers in `mlx-gen/tools`, 475 fixture blobs, ~3,900
   `#[ignore]` model-execution tests run only by explicit workflows, a weights-free conformance
   testkit [H]. A migration is protectable family-by-family with **weights-free plan receipts**
   plus existing tiny-fixture goldens; real-weight parity belongs in one terminal campaign.

---

## 1. Weight lifecycle

### 1.1 What exists in the shared contract (gen-core, tensor-free)

| Concern | Location | Notes |
|---|---|---|
| Source of weights | `inf:crates/contracts/gen-core/src/runtime/source.rs:44-50` | `WeightsSource::{Dir, File}` only; "**no** hub-fetch variant: inference never self-fetches" (`:37-43`). Downloads are SceneWorks' job. [H] |
| Component slots | `inf:…/runtime/source.rs:5-35` | string ids `base_snapshot`, `vae`, `spatial_upscaler`, `comfyui_text_encoder`, `comfyui_vae`, `krea_convrot_dit` — provider-specific names on a shared map. [H] |
| Load request | `inf:…/runtime/load_spec.rs:9-171` | `weights`, `resolved_route`, decode-geometry policies, `quantize: Option<Quant>` (`:54`), `precision` (`:55`), `control`, `extra_controls`, `ip_adapter`, `adapters` (`:83`), `pid`, `identity`, `text_encoder` (`:107`), `offload_policy`, `load_shape`, `load_shape_declaration_result`, `components: BTreeMap` (`:171`), plus file-pin APIs (`:420-1210`). Typed slots for some overlays and a string map for others = two mechanisms for the same thing. [H] |
| File identity | `inf:…/runtime/source.rs:52-202`, `runtime/seal.rs` (1,354 lines), `runtime/load_spec.rs:431-1210` | stat fingerprints (dev/inode/ctime; symlink entry+target split because HF snapshots link `x.safetensors` → extensionless blob), `ArtifactSeal` content digests, `read_files_unchanged` brackets so a read is proven to have seen an unchanged file. [H] |
| Source inventory | `inf:…/source_inventory.rs` (999), `snapshot_inventory.rs` (479) | snapshot walker consolidating 7 drifted candle walkers (symlink/`.incomplete`/`.partial` handling) — `snapshot_inventory.rs:3-11`. [H] |
| Header reading + adapter-format predicates | `inf:…/weightsmeta.rs:1-12` (2,794 lines) | `CheckpointMeta` reads keys/dtypes/shapes/`__metadata__`/byte views without tensors; LoRA/LoKr/LoHa/kohya predicates live here. [H] |
| Sharded index resolution | `inf:…/safetensors_shards.rs:1-5` | index-selected shards, recognizes HF `blobs/` layout. [H] |
| Codec registry + logical plan | `inf:…/checkpoint_codec.rs:1-25`; `spec.rs`, `plan.rs`, `compile.rs`, `transform.rs` | see §1.3. [H] |
| Checkpoint facts | `inf:…/checkpoint_facts.rs:1-12` (2,053) | three separate facts: source binding, stored codecs, resident encoding — prevents "dense bf16 fallback labelled NVFP4". [H] |
| Tier integrity | `inf:…/tier_integrity.rs:1-11` | "No component is resident above the user's selected quant tier unless a declared, measured exception says otherwise"; Krea control branch floor q8 under q4 is the declared exception (`:33-40`). [H] |
| Encoder substitution | `inf:…/encoder_contract.rs:1-8` | header + `config.json` validated against descriptor's `EncoderContract` **before** any backend opens tensors. [H] |
| Load progress | `inf:…/load_activity.rs` (39), `inf:crates/contracts/load-progress/src/lib.rs:1-2` | request-local I/O observations; "Captured contexts travel with lazy reads, never through a process-global current request." [H] |
| Component residency driver | `inf:…/residency.rs:1-7,18-22,45-55,58` | `ResidencyRuntime` (one hook: `after_component_drop`), `StagedHeavy::shed_dit`, `Residency<Text,Heavy,R>`; 40 media files reference `Residency<`. Image-shaped: exactly two phases (text, heavy). [H] |
| Rung 4 block window | `inf:…/block_window.rs:1-40,152-176` | `BlockWindowBackend { type View; open_view(); release() }` — view must be FRESH per window, reading must be "a transfer of already-device-format bytes, never a per-window format conversion" (`:159-162`, the ~26× failure). Adopted by 9 candle + 19 MLX crate dirs. [H] |

### 1.2 What each backend adds

**MLX** (`inf:crates/media/mlx-gen/src/weights.rs`, 44 KB):
- `Weights` = `HashMap<String, Array>` of **lazy** file-backed arrays + metadata + an `accessed`
  set (`:22-55`). Loads on the CPU stream because `Load` has no Metal implementation (`:12-16`).
- `materialize()` evaluates in 512 MiB batches in sorted-key order, **pending loads first, derived
  arrays second**, then GPU-view coherence verification (`:91-145`). The comment at `:122-127` is
  the GPU-watchdog lesson: evaluating a GPU cast/transpose over an unread `Load` makes the Metal
  command buffer wait on disk (`kIOGPUCommandBufferCallbackErrorTimeout`, sd3/ideogram/seedvr2 in
  sc-24245). Adapters repeat the rule: `apply_adapter_specs` calls `w.materialize_accessed()`
  after installing each file (`inf:crates/media/mlx-gen/src/adapters/loader.rs:247-250`); offline
  `quantize_map` has the same note (`inf:crates/media/mlx-gen/src/quant.rs:475-496`). [H]
- `materialize_accessed()` + `remove_accessed()` exist for block-window loaders so only the current
  window's bytes are consumed under the file guard (`weights.rs:147-160`). [H]
- `block_residency.rs:1-40`: one block = 97.3 MiB vs 97.1 header-computed; **dropping before
  forcing evaluation frees nothing** (8.0 MiB with guard vs 238.4 without) — silent. [H]
- `logical_weights.rs:1-30` (97 KB): MLX reader for the codec plan; fp8 stored as lazy `U8`
  decoded on device; `DenseResidencyPolicy` (no packed fp8/int8 matmul on MLX). [H]

**Candle** (`inf:crates/media/candle-gen/candle-gen/src/`):
- `loader.rs:1-27`: the one audited `unsafe` mmap seam, de-duplicating ~34 copies; a CI script
  forbids other `MmapedSafetensors::new` sites (sc-27279). **Shard-order invariant**: lexical sort,
  last shard wins on duplicate keys — "Do NOT change the sort key" (`:22-27`). [H]
- `weights.rs:1-10`: a raw key→`Tensor` map for non-`VarBuilder` loads (IP-Adapter, ControlNet
  zero-convs, EVA-CLIP). Plus **per-family `Weights` wrappers** in krea, boogu, ideogram, seedvr2,
  depth, face, sam3 (all `struct Weights`). [H]
- `logical_weights.rs` (172 KB): candle reader for the codec plan, including packed-native NVFP4.
- `quant/sidecar.rs:1-30`: converts MLX-packed triples to GGML once per component, writes
  `.candle-device-format-v1/` sidecars (or an external cache dir, `SCENEWORKS_CANDLE_DEVICE_CACHE_DIR`),
  mmaps them per window — this exists precisely to satisfy the block-window "device-format bytes"
  rule. [H]
- `residency.rs:1-6`: candle's `ResidencyRuntime` is a no-op drop hook; device sync is a
  request-local closure. A separate three-component **video** driver stays in candle. [H]

**Audio** (`inf:crates/audio/candle-audio/src/weights.rs:1-15`): a *third* audited mmap surface,
separate from candle-gen's loader (sc-27259). [H]

**LLM**: `mlx-llm/src/primitives/weights.rs` (31 KB), `candle-llm/src/primitives/weights.rs`
(9 KB), `mlx-llm/src/load_memory.rs` (98 KB), GGUF readers/converters (`candle-llm/src/gguf.rs`,
`mlx-llm/src/gguf/convert.rs`). [H]

### 1.3 The declarative plan that already exists — and why it covers so little

`LogicalKeyMapping` (`inf:crates/contracts/gen-core/src/checkpoint_codec/transform.rs:11-46`):
`mapping_id()`, `logical_key(physical) -> Option<String>` (None ⇒ refuse, never skip),
`logical_shape(logical)` (needed to unpad MXFP8/NVFP4 storage), `logical_transform(physical)` →
`LogicalTransformDeclaration { outputs: [rename | row_slice | half_swap], source_logical_shape }`
(`:73-160`). The compiler (`compile.rs:25-66`) takes headers + `.comfy_quant` descriptor bytes +
optional file-level `_quantization_metadata` + mapping + `CheckpointCodecRegistry` +
`CodecResidencyPolicy` and fails closed before any tensor exists. Per tensor it records
`physical_key`, `logical_key`, encoding, shape, `codec_id`, `resident_encoding`,
`TensorCodecSpec`, `PlannedResidency{Dense|Packed, bytes}`, transform (`plan.rs:8-33`). [H]

Registered codec ids: `dense-bf16-v1`, `dense-f16-v1`, `dense-f32-v1`, `fp8-e4m3-scalar-v1`,
`fp8-e5m2-scalar-v1`, `mxfp8-v1`, `int8-per-row-v1`, `nvfp4-v1`, `gguf-container-v1` [H]. The
registry is engine-level, registered once per catalog
(`inf:crates/media/mlx-gen/mlx-gen-catalog/src/lib.rs:88-94`). The registry row
`CheckpointCanonicalMappingRegistration.plan_driven_backends`
(`inf:crates/contracts/gen-core/src/registry/registration.rs:179-203`) explicitly records that "for
most shipped families the answer is **no**: their loaders read the checkpoint on their own native
path". [H]

Production `impl LogicalKeyMapping` (excluding tests) [H]:
`candle-gen-flux2/src/single_file.rs:225`, `candle-gen-krea/src/native_mapping.rs:136`,
`candle-gen-wan/src/gguf.rs:171`, `mlx-gen-krea/src/native_remap.rs:176,231`. (Prior audit said 3
candle crates; MLX Krea is a 4th adopter.)

**Why coverage is low** (the gap list a weight-plan design must close):
1. The plan describes **imports** (ComfyUI/BFL single files), not SceneWorks' own tiers. The MLX
   affine u32 triple is outside the codec registry; it is detected by `quant::lin`/`packed_quant_bits`
   (`inf:crates/media/mlx-gen/src/quant.rs:50-310`) and candle's `QLinear::linear_detect`
   (`inf:…/candle-gen/src/quant/mod.rs:1-34`). [H]
2. It is a **rename/slice** language: no conv layout transpose, no dtype policy, no selective
   quantization predicate, no multi-file component topology, no `.pth` containers, no
   weight-norm folding, no tied weights.
3. It is per **file**, not per **model**: component topology (TE/DiT/VAE/upsampler/control branch),
   co-requisites and conditional components live elsewhere (`LoadSpec.components`,
   `CheckpointAdapterRegistration.component_topology`, `PerComponentBytes`).
4. It ends at "resident arrays by logical key"; **binding those arrays to modules** (the
   `lin(w, "blocks.3.attn.to_q", bias, gs)` calls) is still per-family code.

### 1.4 Per-family loader/convert inventory (sizes in bytes) [H]

Largest offenders: `candle-gen-krea/src/loader.rs` 216,817; `candle-gen-wan/src/gguf.rs` 81,985;
`mlx-gen-krea/src/loader.rs` 76,771; `candle-gen-bernini/src/convert.rs` 67,486;
`mlx-gen-wan/src/convert.rs` 60,606; `candle-gen-flux2/src/single_file.rs` 57,051;
`mlx-gen-krea/src/native_remap.rs` 56,076; `mlx-gen-ltx/src/convert.rs` 55,148;
`candle-gen-moss-image/src/loader.rs` 54,091; `mlx-gen-moss-image/src/loader.rs` 50,008;
`mlx-gen-qwen-image-2-1/src/convert.rs` 45,964; `mlx-gen-flux2/src/convert.rs` 43,199;
`mlx-gen-krea-realtime/src/load.rs` 42,080; `mlx-gen-minimax-h3/src/convert.rs` 40,637.

What the `convert.rs` files actually do (three distinct jobs mixed under one name) [H]:
- **Offline pre-quantization** to a turnkey tier: `mlx-gen-z-image/src/convert.rs:1-12,72-106`
  ("Group-B per-crate converter template", same `mlx_rs::ops::quantize` as load-time; differs only
  in key layout and three target predicates — DiT, Qwen3 TE, **VAE mid-block attention**).
- **Native-format → MLX layout conversion**: `mlx-gen-wan/src/convert.rs:1-23` (native safetensors
  + `.pth` T5/VAE, sanitizers ported from Python, f32 VAE, dual A14B experts, VACE assembly);
  `mlx-gen-ltx/src/convert.rs:1-25` (single flat file split by prefix, key sanitize,
  **channels-last conv transposes** Conv3d `[O,I,D,H,W]→[O,D,H,W,I]`, Conv1d, ConvTranspose1d,
  selective Q4 via `_quantize_ltx_predicate`, cast to bf16).
- **Architecture validation**: `candle-gen-krea/src/convert.rs:1-8,54-186`
  (`expected_transformer_keys(cfg)` vs on-disk set).

Key-remap/sanitize functions: 137 matches of `fn *(remap|sanitize|map_key|rename_key|…)` outside
tests [H] (prior audit counted 101 with a different pattern; same order of magnitude). Top: z-image
MLX 15, LTX MLX 13, LTX candle 13, Wan candle 9, Wan MLX 8.

### 1.5 Lazy vs eager, the watchdog, and re-openable sources

- **MLX hazard (load-bearing)**: arrays are lazy; reading must be split into "evaluate pending
  loads" then "evaluate derived ops" in bounded batches (`weights.rs:101-145`). Any shared loader
  that composes `load → cast → transpose → quantize` as one lazy graph and evaluates it inside a
  forward will re-introduce sc-24245. Memory note "NEVER eager-read inside a block stream" is the
  same rule seen from the block-window side: per-window reads must be device-format bytes only.
- **Candle**: eager per tensor but mmap-backed; the hazard there is per-window **conversion** cost
  (sidecars solve it) and host transients (Q8 repack done projection-at-a-time,
  `quant/sidecar.rs:19-20`).
- **Re-openable source for rung 4**: `BlockWindowBackend::open_view` must produce a fresh view per
  window (`block_window.rs:165-169`); MLX re-opens `Weights` (cheap, lazy handles), candle maps
  sidecars. File pins (`LoadSpec::prepared_file_pins`, `read_prepared_files_unchanged*`,
  `load_spec.rs:420-1210`) guarantee each reopen sees the same bytes. A shared weight plan must
  therefore emit, per block, a **re-openable, already-device-format tensor set** and must know
  block boundaries.
- `LoadShape::{EagerMaterialization, DeferredMaterialization}` (`runtime/quant.rs:183-192`) is
  orthogonal to `OffloadPolicy::{Resident, Sequential}` (`:160-167`); plus an authority marker
  `LoadShapeDeclarationResult` (`:203-217`). [H]

### 1.6 What a declarative per-family **weight plan** must express

Proposed shape (strawman; names illustrative). One plan per *route family*, data not code:

```text
WeightPlan {
  components: [ComponentPlan],          // text_encoder[], denoiser (maybe 2 experts), vae, upsampler,
                                        // control_branch, ip_adapter, … with role, optionality,
                                        // co-requisite source (base_snapshot), residency class
}
ComponentPlan {
  role, phase (text|denoise|decode|aux), required: bool | conditional_on(source_form),
  sources: [SourceDialect]              // diffusers dir, sceneworks-tier dir, single-file(prefix),
                                        // comfyui single-file, gguf, .pth, sharded index
  key_map:  KeyMapSpec                  // ordered rules: strip prefix, regex rename, table rename,
                                        // row-slice/half-swap (existing LogicalTransform), ignore-set
                                        // (with reason), alias/tie, block index capture
  layout:   [LayoutRule]                // conv OIHW→OHWI (MLX only), transpose, reshape, weight-norm fold
  dtype:    DtypePolicy                 // compute dtype, f32 islands (VAE, norms), Precision sentinel
  quant:    QuantPolicy                 // predicate (which Linear/embedding), group size, alignment
                                        // fallback, tier→resident-tier floors (tier integrity)
  blocks:   BlockLayout                 // prefix + count => block-window boundaries + grad-ckpt boundaries
  expect:   ArchitectureExpectation     // expected logical key set/shapes from config (validation)
  adapter_surface: AdapterSurface       // canonical target names + alias tables (kohya/BFL/native)
}
```

Execution pipeline the plan drives (backend supplies only the leaf ops):
`ArtifactSource(pins) → inventory/seal → header plan (fail-closed, weights-free) → per-tensor
codec decode → layout/dtype → quantize-if-requested → residency (eager | deferred | windowed) →
module binding by logical key → adapter install → receipt`.

**Hard cases the plan must cover (each observed in source)** [H unless noted]:

1. **Fused↔split projections and AdaLN order** — already in `LogicalTransformDeclaration`; must
   also feed the adapter layer (BFL LoRA row-slices, `inf:crates/media/mlx-gen/src/adapters.rs:1367-1384`).
2. **MLX affine packed triple** with group size not inferable from shapes (Boogu 32 vs default 64,
   `inf:…/candle-gen/src/quant/mod.rs:25-34`) → must read `config.json` `quantization` block.
3. **Cross-backend re-encoding**: MLX Q4 → GGML `Q4_1` lossless; MLX Q8 → `Q8_0` lossy
   double-quant (0.56% mean rel. RMS) (`quant/mod.rs:30-34`) + sidecar caching. The plan must
   record that a candle "q8" resident tier is not bit-identical to MLX q8.
4. **Packed-native vs dense-fallback per layer** (NVFP4 on `sm_120`; `full_precision_matrix_mult`
   forces dense) — `checkpoint_codec/spec.rs:90-117`.
5. **Padded storage**: NVFP4 16-padded, MXFP8 32-padded; packed-native requires logical ==
   stored shape (`spec.rs:50-58`) → plan needs declared logical shapes.
6. **GGUF k-quant containers** have no per-element width (`plan.rs:351-374`
   `ResidentTensorHeadersError`); Wan keeps `Q4_K_M` resident and dequantizes per matmul
   (`inf:…/candle-gen-wan/src/gguf.rs:1-11`).
7. **Non-safetensors containers**: `.pth` zip-of-pickle (Wan T5/VAE, `mlx-gen-wan/src/convert.rs:3-5`;
   Supra2 via `inf:crates/contracts/gen-core-checkpoint/src/pth.rs`, `supra2.rs`).
8. **Single all-in-one files** split by prefix (LTX `model.diffusion_model.*`, `vae.*`,
   `audio_vae.*`, `vocoder.*`, `text_embedding_projection.*`, `mlx-gen-ltx/src/convert.rs:3-10`)
   and **single-file DiT + `base_snapshot` companion** for TE/VAE (`runtime/source.rs:5-12`).
9. **Channels-last conv layout** is MLX-only; candle keeps OIHW. Layout rules must be
   backend-conditional.
10. **Selective quantization predicates** differ per family: LTX quantizes attention+FF only;
    Wan packs per-block attention/FFN, norms stay dense (relied on by diff-patch folds,
    `adapters.rs:1387-1392`); Z-Image packs DiT, TE, **and** VAE mid-block attention.
11. **Alignment fallback**: `quantize_if_aligned` packs only when `in_features % group == 0`
    (`adapters.rs:1011`, test at `:1460-1470`); a plan must allow "dense if unaligned".
12. **Dtype islands**: Wan VAE f32; SDXL/Kolors run fp16 under the `Precision::Bf16` "provider
    default" sentinel (`runtime/quant.rs:130-148`); Iris autocast policy keeps norms f32
    (`mlx-gen-iris/src/nn.rs:1-14`).
13. **Dual-expert MoE** (Wan2.2 A14B high/low) — two denoiser components switched at a noise
    boundary; adapters target one or both (`runtime/adapters.rs:19-25,80-88`).
14. **Duplicate keys across shards** (last lexical shard wins, `candle-gen/src/loader.rs:22-27`).
15. **HF symlink farms**: dispatch on the extension of the *link*, not the blob; never
    `canonicalize` before dispatch (`runtime/source.rs:52-57`; memory note "mlx-rs dispatches
    safetensors on the path extension").
16. **Variant files sharing a directory** (Anima base/aesthetic/turbo DiTs in one dir;
    `PerComponentBytes` over-counts, `registration.rs:27-33`) and **shared co-requisites across
    tiers** (Mage-Flow tiers share TE/VAE, `sw:crates/sceneworks-worker/src/engines.rs:38-39`).
17. **Encoder substitution** must be validated header-only before load (`encoder_contract.rs:1-8`).
18. **Adapter overlay vs packed base**: candle Krea's merge overlay takes priority over the mmap and
    dequantizes the packed base for targeted projections only (`candle-gen-krea/src/loader.rs:17-23`).
19. **Block streaming**: per-window re-open, device-format bytes, eval-before-drop on MLX
    (§1.5) — plan must expose block layout.
20. **Pruned/reduced modules** (`mlx-gen-minimax-h3/src/dit/pruned_adaln.rs` `PrunedReducedLinear`)
    and **embeddings** quantized separately (`QEmbedding`, `quant::embedding`) [M].
21. **Format-specific compute seams**: NVFP4 forward is wired per family by porting a whole DiT
    trunk seam (`candle-gen-krea/src/nvfp4_dit.rs:1-13`, "direct port of candle-gen-sana's
    nvfp4_dit"). With a single Linear spine (§2.5) this becomes a resident-encoding choice, not a
    second DiT.
22. **Iris / LTX / H3 receipts and seals**: provenance must be preserved (`checkpoint_facts.rs`),
    including "what the load actually read".

### 1.7 SceneWorks side of the lifecycle

- Download tiers are manifest data: per model `downloads[]` with `repo`, `revision`, `variant`
  (`q4`/`q8`/`bf16`), `files` globs, `estimatedSizeBytes`, `footprint{disk,resident,peak}`
  (`sw:config/manifests/builtin.models.jsonc`, `z_image_turbo` entry) plus an `mlx.calibrations[]`
  block keyed by inference revision / closure digest. [H]
- The worker resolves and stages every path; inference never fetches (`runtime/source.rs:37-43`). [H]
- `MODEL_TABLE` maps SceneWorks id → engine id plus `default_repo`, `default_steps`,
  `default_guidance`, `adapter_label` (`sw:crates/sceneworks-worker/src/engines.rs:4-35`) — these
  defaults duplicate the manifest `defaults` block. [H]

---

## 2. Quantization formats and where quantization happens

### 2.1 Tier vocabulary (contract)

`Quant::{Q4, Q8, Nvfp4, Fp8Rowwise}` (`inf:crates/contracts/gen-core/src/runtime/quant.rs:15-32`);
`key()` = `q4|q8|nvfp4|fp8r`, `tier_label(None)="bf16"` (`:76-102`); `affine_bits()` refuses
packed float formats so NVFP4 (bits()=4) can never reach the MLX affine quantizer (`:116-127`).
"A quant tier is a creative choice — a distinct, additive tier, never a silent numerics swap"
(`:7-13`). The MLX catalog rejects `Nvfp4` at the registry (`mlx-gen-catalog/src/lib.rs:131-154`).
Extra non-tier knobs: `Yue2ArMode`, `Yue2ComputePolicy` (`quant.rs:34-58`) — family-specific
enums on the shared runtime module. [H]

### 2.2 Stored/resident formats in use [H]

| Format | Produced | Consumed | Code |
|---|---|---|---|
| MLX affine Q4/Q8 (u32 + scales + biases, gs 64 or 32) | **offline** per-family `prequantize_*` (`mlx-gen/src/quant.rs:437-660` `load_dir_map/quantize_map/save_map/write_quantized_config`) or **load-time** `AdaptableLinear::quantize` from bf16 | MLX `quantized_matmul`; candle repacks to GGML | `mlx-gen/src/quant.rs`, `candle-gen/src/quant/{mod,repack,sidecar}.rs` |
| GGML Q4_0/Q8_0 (load-time from dense), Q4_1/Q8_0 (from MLX tier) | candle at load | `QLinear::Quantized` dequant-to-dense per forward (activation stays full precision — `quant/mod.rs:11-18`, the outlier-NaN lesson) | `candle-gen/src/quant/mod.rs:520-535` |
| GGUF k-quants (Q4_K_M) | third-party | candle Wan DiT only, resident `QTensor` | `candle-gen-wan/src/gguf.rs` |
| ComfyUI scalar fp8 (E4M3/E5M2), MXFP8, int8 per-row, NVFP4 | third-party import | codec plan; dense on MLX; packed-native NVFP4 on candle sm_120 | `gen-core/src/comfy_quant.rs` (2,154), `checkpoint_codec/*` |
| NVFP4 tier | — | candle `Nvfp4Linear` via `candle-quant-kernels` | `inf:crates/kernels/candle-quant-kernels/src/nvfp4_linear.rs` |
| ConvRot INT8, W4A8 (asym_w4a8_int8) | descriptor-driven | MLX `LinearBase::{ConvRot, W4A8}` (`mlx-gen/src/adapters.rs:547-555`); candle `QLinear::W4A8` | `gen-core/src/convrot_int8.rs`, `w4a8_int8.rs` |
| FP8 rowwise (W8A8) | FLUX3 source | FLUX3 only | `quant.rs:27-31` |
| LLM affine Q4/Q8 + GGUF | `mlx-llm`, `candle-llm` | own `QuantizedLinear` | `inf:crates/llm/mlx-llm/src/primitives/quant.rs:1-7` ("greenfield for the engine") |
| YuE2 experimental FP8 AR | runtime swap | YuE2 only | `quant.rs:34-44` |

### 2.3 Where quantization happens — decision points

1. **Offline (SceneWorks-published tiers)**: per-family converter writes `q4/`,`q8/` folders
   (`sw:…/builtin.models.jsonc` `files: ["q4/*"]`). Each converter re-states its target
   predicate.
2. **Load-time on MLX**: `needs_load_time_quant(root, component, bits, model_id)`
   (`mlx-gen/src/quant.rs:259-289`) — packed turnkey must match request exactly (quantize is a no-op
   on packed weights; a mismatch would silently serve the wrong tier). Note: "Direct provider
   struct loaders that bypass the registry/load seam also bypass this guard by design" (`:262-263`).
3. **Load-time on candle**: dense → `QLinear::quantize`; MLX-packed → repack + sidecar.
4. **Plan time** for imports: `CodecResidencyPolicy` decides packed vs dense per layer.
5. **Tier integrity**: component floors (`ComponentPrecisionFloor` on `Capabilities`,
   `generator/descriptor.rs:193-196`; Krea control-branch floor) — currently provider logic.

### 2.4 Why ~30–68 per-family Linear types exist (causes, with evidence)

68 struct/enum definitions with `Linear` in the name outside vendor code [H]. Causes:

| Cause | Examples | Removable by a shared spine? |
|---|---|---|
| Packed-detect + group-size wrappers per family | `candle-gen-{boogu,chroma,krea,ltx,minimax-h3}/src/quant.rs` (`QLinear`/`TieredLinear`) — headers say "candle twin of …, built on the shared `candle_gen::quant`" | Yes — plan supplies gs; one detect path |
| Per-pass adapter scales | `mlx-gen-ltx/src/transformer/linear.rs:1-2,103-149` (`LtxAdapter{pass_scale: Vec<f32>}`, `Cell<usize>` pass set by `set_lora_pass`, `attention.rs:259,310`) | Yes — scale context in the spine |
| Precision/autocast policy | `mlx-gen-iris/src/nn.rs:1-14`, `mlx-gen-flux3/src/nn.rs`, `candle-gen-{iris,flux3}/src/nn.rs` | Yes — dtype policy on the projection |
| Format-specific forward seams | `candle-gen-krea/src/nvfp4_dit.rs`, sana `nvfp4_dit`, `ConvRotLinear` in `candle-gen-ltx/src/quant.rs` | Yes — base variant |
| Bias/no-bias, special heads | `candle/mlx-gen-minimax-h3/src/dit/{layers,heads}.rs` `LinearNoBias`/`LinearBias`, `PrunedReducedLinear` | Mostly |
| Training differentiability (candle) | `candle-gen/src/train/lora.rs` `LoraLinear`; `candle-gen-krea/src/train_dit.rs` `ComposableLinear` + vendored blocks because fused `CustomOp`s have no backward (`train_dit.rs:1-5`) | Partially (needs differentiable op set) |
| iOS accounting | `mlx-gen-sdxl/src/quant.rs` `HeaderLinear`, `quantum.rs` `LinearWork/LinearEffects` | Yes if pricing reads the plan |
| Text-encoder stacks | `mlx-gen-ltx/src/gemma.rs` `GemmaLinear`, `mlx-gen-kolors/src/chatglm3.rs` `ChatGlmLinear`, `mlx-gen-lens/src/text_encoder/gpt_oss.rs` | Yes via shared LLM blocks |
| LLM crates independent | `mlx-llm` `QuantizedLinear`, `PrismLinear`, `SwitchLinear`; `candle-llm` `QuantizedLinear` | Yes (shared spine below both) |
| Ported vendor models | `candle-gen-sdxl` stock candle-transformers UNet (`QLinear` dropped into `#[derive(Debug)]` vendored modules, `quant/mod.rs:532-540`) | Partly |

### 2.5 Backend split of the spine today

- MLX: **one** type `AdaptableLinear { base: LinearBase::{Dense, Quantized, ConvRot, W4A8}, adapters }`
  serves inference, quantization, adapters and training (`adapters.rs:547-1170`). [H]
- Candle: **three** types — `QLinear` (quant: Dense/W4A8/Quantized{weight, bias, fwd}),
  `AdaptLinear` (forward-time adapters over dense or packed base, `quant/adapt.rs:1-25,713`),
  `LoraLinear` (trainable Var-backed residual, `train/lora.rs:1-22`) — plus `Nvfp4Linear`,
  `Fp8Linear`, `Int8Linear`, `W4A8Linear`, `ConvRotLinear` bases. [H]

---

## 3. Adapters

### 3.1 Formats in the wild (all seen in source)

| Format | Detection | Notes / where |
|---|---|---|
| PEFT/diffusers LoRA (`lora_A/B.weight`, optional `.alpha`, prefixes `transformer.`/`diffusion_model.`/`base_model.model.`) | key suffix + `COMMON_LORA_PREFIXES` | `mlx-gen/src/adapters/loader.rs:9-16`, `weightsmeta` |
| kohya/sd-scripts (`lora_unet_<flattened>.lora_down/up`) | `keys_are_kohya` | needs host path enumeration to un-flatten (`adaptable_paths`, `adapters.rs:1340-1355`) |
| BFL/ComfyUI fused (`double_blocks_*.img_attn.qkv`, `linear1`) | `bfl_targets` | row-sliced into split targets (FLUX.1/FLUX.2 only) |
| LoKr, `networkType=lokr` stamped | metadata | dense delta or `LokrStructured` vec-trick (no `[out,in]` delta) |
| Third-party LyCORIS LoKr (unstamped, per-module alpha, tucker `lokr_t2`) | keys | `apply_lokr_thirdparty` |
| LoHa (`hada_w1_a/b`, `hada_w2_a/b`) | keys | no low-rank residual form ⇒ dense delta; candle Qwen21 refuses on packed tiers |
| Diff-patch (`.diff`, `.diff_b`, incl. norm params) | keys | lightx2v Wan step-distill; norm folds via `diff_patch_param_mut` (`adapters.rs:1386-1398`); **cannot be toggled per phase** |
| Conv LoRA | SDXL only | `adaptable_conv_mut` (`adapters.rs:1333-1338`) |
| SA3 eight types: LoRA, DoRA rows/cols, BoRA, LoRA-XS (+DoRA/BoRA-XS) | file sniff | **non-commutative**, needs base weight norms and an SVD of the base (`inf:crates/audio/candle-audio-stable-audio-3/src/adapters.rs:7-33`) |
| Wan MoE high/low pairs | `AdapterSpec.moe_expert` | `runtime/adapters.rs:19-25` |
| H3 lightx2v turbo (alpha resolution rule, ComfyUI key-space conversion) | family | `candle-gen-minimax-h3/src/adapters.rs:1-12` |
| Training adapter (frozen de-distill LoRA applied only during training) | `TrainingTechniques.training_adapter` | `gen-core/src/train/techniques.rs:45-47` |

Contract surface: `AdapterSpec{path, scale, kind: AdapterKind::{Lora, Lokr}, pass_scales:
Option<Vec<f32>>, moe_expert}` (`runtime/adapters.rs:8-26,75-78`). LoHa/diff-patch/kohya/BFL are
**not** in the kind enum — they are detected from keys at load; `kind` is effectively a hint that
can contradict the file (`loader.rs:224-232` errors on `Lora` + `networkType=lokr`). Install outcome:
`AdapterApplyReport{applied, skipped}` (`runtime/adapters.rs:34-45`). [H]

### 3.2 MLX: runtime residual spine [H]

- `Adapter::{Lora{a,b,scale}, Lokr{delta,scale}, LokrStructured{factors}}` (`adapters.rs:415-427`);
  forward `base(x) + Σ residual(x)`; the base is never mutated, so quantization and adapters
  compose (`adapters.rs:1-12`).
- Routing: `AdaptableHost` (`adapters.rs:1307-1399`) — `adaptable_mut(path)`, `adaptable_facts`
  (non-unfusing probe for fused QKV hosts; every intermediate node must forward it),
  `lycoris_dtype` (bf16 default for fork parity; f32 for Iris fp32), `adaptable_conv_mut`,
  `adaptable_paths` (kohya), `bfl_targets`, `diff_patch_param_mut`. ~102 `impl AdaptableHost`
  across 53 files: each family hand-writes a path-routing tree.
- Dispatch: `apply_adapter_specs` / `_autoprefix` / `apply_adapters_strict*`
  (`loader.rs:201-262,512-708`); header-only `classify_unfused_adapter_format`
  (`loader.rs:342-375`) mirrors the loader for pre-load facts (`adapters/facts.rs`,
  `loader/source_facts.rs` 124 KB).
- Merge paths still exist: `merge_dense_delta`, `merge_bias_delta`, conv merge
  (`adapters.rs:777-850,1203`) for diff-patch and conv.

### 3.3 Candle: four strategies [H]

| Strategy | Families | Rationale cited |
|---|---|---|
| **Fold at load into dense safetensors map, then build** | SDXL, Z-Image, Wan, SD3, Krea (single-phase), SCAIL2, Lens (`train/merge.rs:3-10,40-48`) | "chaos-sensitive samplers make `(W+δ)·x` and `W·x + δ·x` diverge by ~1 ULP into a visibly different image" (`candle-gen-sdxl/src/adapters.rs:6-11`) |
| **Forward residual via `AdaptLinear` + `install_dotted_adapters`** | Qwen-Image-2.1, FLUX.2, Anima, Qwen-Image-Edit Lightning, … (AdaptLinear referenced in 18 crate dirs) | packed q4/q8 bases cannot be folded (`quant/adapt.rs:17-25`); LoHa folds on bf16 only (`candle-gen-qwen-image-2-1/src/adapters.rs:1-14`) |
| **Residual via training `LoraLinear`** | LTX | avoid materializing 22B dense to fold; LoHa/diff-patch rejected (`candle-gen-ltx/src/adapters.rs:1-10`) |
| **Residual via family layer** | MiniMax-H3 (`LinearNoBias`) | parity with MLX lane: "a LoRA that folded at a different strength per backend would make the quant tier … decide the picture" (`candle-gen-minimax-h3/src/adapters.rs:5-12`) |

Shared candle pieces: `train/merge.rs` (MergeReport, kohya table, diff-patch fold, third-party
LyCORIS engine) with the explicit statement that **key→module resolution "MUST stay in each
crate — unifying it would change merge output"** (`merge.rs:38-48`) [H]. Per-family `adapters.rs`
sizes: krea 119 KB, sdxl 101 KB, minimax-h3 66 KB, sd3 66 KB, qwen-image 55 KB, wan 52 KB,
anima 43 KB, z-image 44 KB, qwen21 42 KB, lens 40 KB, ltx 35 KB, iris 31 KB, scail2 30 KB,
moss-image 20 KB, ideogram 16 KB, flux3 5 KB [H].

### 3.4 Per-phase / per-pass adapter scales

- **Contract**: `GenerationPhase{steps, guidance, adapters: Vec<PhaseAdapter{adapter: usize,
  weight: Option<f32>}>}` (`inf:crates/contracts/gen-core/src/generator/params.rs:352-397`) — phases
  index the load-time adapter stack. Host math shared in
  `inf:crates/contracts/gen-core-family/src/krea_multiphase.rs` (`mlx-gen-krea/src/multiphase.rs:1-4`). [H]
- **MLX Krea**: `prepare_multiphase` builds per-phase plans once; renders toggle residuals
  (`mlx-gen-krea/src/model.rs:1542-1584`). Multi-phase refused when a diff-patch was folded at load
  (`model.rs:1881`, test `:3949-3967`). [H]
- **Candle Krea**: single-phase bakes adapters into the resident DiT; multi-phase loads a
  **job-local base DiT** so the shared resident DiT is never mutated, and re-installs each phase's
  low-rank residuals on it (`candle-gen-krea/src/lib.rs:372-392,1371-1389`). This is a second
  DiT's worth of memory per multi-phase job. [H]
- **LTX `pass_scales`** (2-stage distilled): `AdapterSpec.pass_scales` (`runtime/adapters.rs:12-18`),
  MLX `LtxAdapter{pass_scale: Vec<f32>}` + `Cell<usize>` active pass (`mlx-gen-ltx/src/transformer/linear.rs:103-149`);
  candle `candle-gen-ltx/src/adapters.rs` (26 `pass_scales` refs). [H]
- **Wan MoE expert targeting**: shared spec, only Wan reads it; `Some(_)` elsewhere is "surfaced,
  not silently honored" (`runtime/adapters.rs:19-25`). [H]

These three are the same concept — **an adapter's effective scale is a function of a request-time
context (phase index / pass index / expert id)** — implemented three ways.

### 3.5 What a backend-neutral adapter layer looks like

Split into three layers, mirroring the weight plan:

1. **Header-only parse → `AdapterPlan`** (tensor-free, gen-core): format classification (exists
   in `weightsmeta` + `classify_unfused_adapter_format`), per-target factor groups, alpha/rank
   resolution rules (family override hook for H3/lightx2v), key → **canonical target id**
   through the family's declared `AdapterSurface` (canonical names + alias tables: kohya
   flattening, BFL fused row-slices, native→diffusers renames — all already exist as tables
   inside hosts). Output also feeds facts/capabilities (`adapter_apply_reports` pre-load).
2. **Install policy per target** chosen from `(format, base residency, parity mode)`:
   `Residual(LoRA | LoKr-structured)`, `DenseDeltaResidual(LoHa, small)`, `Fold(diff-patch,
   DoRA/BoRA, conv)` — Fold allowed only on dense bases and makes the adapter non-toggleable
   (phases refuse, as both Krea backends already do). Ordered folds for non-commutative types (SA3).
3. **Scale context**: each residual carries `scale × ctx_weight(adapter_idx, ctx)` where
   `ctx = {phase, pass, expert}` is set per step by the shared sampler driver. Replaces
   `LtxAdapter.pass_scale`, Krea phase re-install, and Wan `moe_expert` filtering with one
   mechanism. Host routing is generated from the module tree's canonical names (§8 Design
   implications) instead of 102 hand-written `adaptable_mut` trees.

### 3.6 Can candle move to runtime adapters? Tradeoffs

- **Compute**: a rank-r residual on a `[out,in]` Linear adds ≈ `r·(in+out)` MACs per token vs
  `in·out` for the base: r=16, 3072×3072 ⇒ ~1.0% per adapted Linear; r=64 ⇒ ~4.2%; 4 stacked
  adapters scale linearly [L — arithmetic, not measured]. On CUDA the bigger cost is **extra
  small GEMM launches** (two per adapter per Linear per forward); fusing A/B of all adapters into one
  concatenated pair per Linear (stack ranks) recovers most of it [L]. LoKr structured costs two
  small matmuls over reshaped activations (`quant/adapt.rs:12-15`).
- **Memory**: fold-at-load on candle builds a CPU `HashMap<String,Tensor>` of the dense base,
  adds δ, then `VarBuilder::from_tensors` (`candle-gen-wan/src/adapters.rs:7-11`) — a transient
  host copy of every targeted weight and **impossible on packed tiers**; residual keeps only factors.
  Multi-phase today costs a second DiT (§3.4); runtime scale context removes it.
- **Parity**: residual vs fold differs by ~1 ULP per layer; the SDXL/Z-Image/Wan goldens encode the
  fold. Options: (a) keep `Fold` as an explicit policy on dense bases for parity-pinned families
  and re-baseline later; (b) re-baseline goldens to residual (MLX already runs residual, so this
  also gives cross-backend parity for free — H3 adopted residual for exactly this reason).
  Recommendation [M]: default residual everywhere; `Fold` only for non-low-rank formats; accept a
  one-time golden re-baseline per family recorded in the migration campaign.
- **Training round-trip**: candle trainers reconstruct δ with the same f32 math the merge uses
  (`candle-gen-z-image/src/adapters.rs:8-12`); with residual inference, the trainer's `LoraLinear`
  and inference `AdaptLinear` become the same type with a `trainable` flag.

---

## 4. Catalog, registration, capabilities

### 4.1 Registration mechanics [H]

- Builder: `ProviderRegistryBuilder` (`inf:crates/contracts/gen-core/src/registry/builder.rs`, 863
  lines; macro-generated `register_*` at `:36`; `register_checkpoint_codec` `:491`;
  `register_composed_memory_strategy` `:543`; `reject_quant` `:593`; `build` `:599`).
- Registration kinds (`registry/registration.rs`): `ModelRegistration{descriptor, load, footprint}`
  (`:98-107`), `PerComponentBytes{text_encoder, dit, vae}` (`:35-38`),
  `CheckpointAdapterRegistration` (signatures, dialects, component topology, base compatibility,
  canonical mappings, config recovery, eligible backends, backend bindings, operations,
  capabilities — `:252-266`), `ImportedModelRegistration` (derived view), `EncoderContractRoute`,
  `MemoryRegistration`, memory contract fixtures, `TrainerRegistration{descriptor, load}` (`:529-532`),
  `CaptionerRegistration`, …
- Call-site counts across crates (incl. tests): `register_generator` 258,
  `register_memory_contract_fixture` 139, `register_memory_strategy` 128,
  `register_memory_behavior` 125, `register_trainer` 50, `register_checkpoint_adapter` 50,
  `register_composed_memory_strategy` 26, `register_activation_memory` 15,
  `register_encoder_contract_route` 10.
- Each family crate exports `register_providers(builder)`; a route = 4–5 chained calls with
  near-identical bodies (generator + memory strategy + weights-free contract fixture closure +
  behavior), e.g. Z-Image's four routes `inf:crates/media/mlx-gen/mlx-gen-z-image/src/lib.rs:337-437`.
  Trainers are `#[cfg(feature = "training")]` (`:346-349`).
- Platform composition: `mlx-gen-catalog::register_providers` registers codecs then 33 families
  in a fixed order (`mlx-gen-catalog/src/lib.rs:88-129`); `provider_registry()` rejects NVFP4
  (`:150-154`). Non-registry crates are listed (`BESPOKE_UTILITY_CRATES` = depth, face, instantid,
  pid, sam2, sam3, `:62`). Candle: `candle-gen-catalog`; audio: `candle-audio-catalog` (71 KB);
  bundles: `runtime-catalog` (80 KB), `runtime-macos`, `runtime-ios`, `runtime-cuda`, `runtime-cpu`.
- **Routes outside the registry**: name-driven edit/control/IP-Adapter providers with their own
  request types (Qwen-Image edit/Fun-control, SDXL edit/IP-Adapter, Kolors pose/IP-Adapter,
  FLUX.2 edit/pose, FLUX.1 Fun-Controlnet/XLabs IP-Adapter, PuLID registers nothing) —
  enumerated in `candle-gen-catalog/src/preview_advertising.rs:14-70` and `BESPOKE_PROVIDER_CRATES`
  (`:1175`). These routes have no `ModelDescriptor`, so no capability record. [H]
- `Generator` trait carries family-specific methods (`generate_moss_image_edit`,
  `moss_image_edit_request_peak`, `veda_execution_report`, `moss_image_owned_weight_bytes`,
  `inf:crates/contracts/gen-core/src/generator/traits.rs:13-60`) — contract leakage. [H]

### 4.2 Capabilities [H]

`Capabilities` (`generator/capabilities.rs:49-444`, 1,436-line file) has 60+ fields spanning
image (guidance, true CFG, conditioning kinds, samplers, schedulers, guidance methods, sizes,
steps, preview, HDR, CFG interval, prompt batch, alpha output), audio (sample rates, voices,
languages, edit modes, multi-speaker, lyrics, song plan/cover, limiter), video (generated
keyframes, temporal upsample rounds, auto duration), LLM-ish (conversation history/session, KV
cache), memory (`request_memory_strategies`, `supports_sequential_offload`,
`unconditionally_engages_staged_residency`), quant (`supported_quants`,
`component_precision_floors`), and `execution`/`approximation` surfaces. 177 `Capabilities {`
literal sites (prior audit: 160 provider literals) are policed by a **source scan** that every
literal ends in `..base` (`gen-core/tests/capabilities_are_constructed_additively.rs:1-18`, which
documents why `#[non_exhaustive]` and private fields both fail cross-crate).

### 4.3 How SceneWorks consumes engine facts [H]

- **Stage 1 dump** `sw:crates/sceneworks-worker/src/engine_capability_facts.rs:1-50` (2,308 lines)
  walks `registry.generators()` weights-free and writes `config/engine-capabilities/
  capabilities.{mlx,candle}.json` (one per backend; verified on the matching CI lane by re-dump +
  diff) and `audio/capabilities.candle.json`.
- **Content actually exported**: MLX file has top keys `backend, generatedFrom, engines(75),
  imports(7), memoryContracts(65), memoryRouteWitnesses(783)`; engine rows only carry
  `id, modality, supportsPreview, denoiserOutputLatentSpace, decoderOptions`; candle (63 engines)
  adds `bespokeMemoryRouteWaivers`; audio rows `id, modality, supportsPreview`. Import rows carry
  `family, source, operation, providerId, conditioning, supportsLora, supportsLokr, supportedQuants,
  supportsKvCache, supportsSequentialOffload, registryCached`. Memory contracts carry per-selector
  `implementedRungs` etc.
- **Runtime artifacts** `config/engine-capabilities/runtime/capabilities.{mlx,candle}.json`:
  `schemaVersion, modelMappings, videoModelMappings, trainerMappings, workerCapabilities, snapshot`.
- **Backend capability matrix** `sw:config/backend-capabilities/README.md:1-40`: replays canonical
  requests through the production worker claim predicate, joined to manifests + runtime snapshots;
  sha256s of predicates; exceptions file; MLX-only cells must name a story.
- Stage 2 (`apps/web/scripts/generate-preview-support.mjs`) derives the served preview catalog.

### 4.4 The preview advertising scanner (the cost of hand declaration) [H]

`candle-gen-catalog/src/preview_advertising.rs` (4,516 lines) has a **declared half** — exact
route ids wired for previews (`PREVIEW_ROUTE_IDS :187`, inert `:449`, deferred `:514`) — and a
**derived half** that parses provider Rust source: strips comments/strings, evaluates `cfg`
predicates to drop test code (`:1343-1830`), resolves the module tree with `#[path]`
(`:1867-2114`), finds sampler-driver call sites and direct `emit_preview` calls (`:1264-1297,
2144-2290`), and asserts bidirectional equality (`:2432`). Its doc enumerates four kinds of
id↔lane non-correspondence (one id many lanes; many ids one lane; lanes that are not sampler
sites; shared latent space across crates with different packing). This is a compiler written
because the capability is not a structural property of the code.

### 4.5 How capabilities could be DERIVED from the composed pipeline

If a route is a **typed composition** (stages declared as data/types, see §8), most fields fold out:

| Capability | Derivation source |
|---|---|
| `supports_preview` | denoiser stage uses the shared sampler driver (always hooked) or declares a `PreviewEmitter`; latent space from the decoder/latent stage |
| `supports_lora`, `supports_lokr` (+ LoHa/diff-patch/etc.) | presence of an `AdapterSurface` on the denoiser + the install policies legal for each tier (e.g. LoHa only where base is dense) |
| `supported_quants`, `component_precision_floors` | intersection over components of the weight plan's quant policies × backend codec registry × tier-integrity exceptions |
| `supports_sequential_offload`, `unconditionally_engages_staged_residency`, `request_memory_strategies` | whether the route uses `Residency` / block window / bounded decode stages (already in memory contracts) |
| `conditioning`, `control_kinds` | union of conditioning stages (reference encoder, control branch, IP-adapter) |
| `samplers`, `schedulers`, `supports_true_cfg`, `supports_cfg_interval`, guidance methods | the sampler driver + guidance stage in use |
| `max_prompt_tokens`, encoder substitution | `EncoderContract` of the text stage |
| `supports_negative_prompt`, `supports_guidance` | guidance stage kind |
| size floor/multiple, `max_temporal_upsample_rounds` | latent/patch geometry + upsampler stage |
| imports (`CheckpointAdapterRegistration`) | weight plan source dialects |

Remaining hand-declared (product/quality facts, not structure): `max_size`, `max_count`,
`supported_steps` defaults, `mac_only`, quality-based exceptions. These can be a small
`RouteProductLimits` record — and SceneWorks' `MODEL_TABLE` defaults should move there so the
manifest/worker read one source.

### 4.6 Memory contract fixtures

`register_memory_contract_fixture` ×139 call sites; each closure points at a
`weights_free_memory_strategy_contract(MODEL_ID, spec)` (e.g. `mlx-gen-z-image/src/lib.rs:365-371`),
consumed by the testkit oracle (`gen-core-testkit/src/memory_strategy/{contract,registration,registry}.rs`).
With a derived route record, the builder can register memory strategy + fixture + behavior in one
call (they are 1:1 per route today) [M].

---

## 5. Training

### 5.1 Contract layers [H]

- `gen-core::train` (`train.rs:1-43`): `Trainer` (`descriptor`, `validate`, `train`,
  `train_with_report`, `train/request.rs:242-270`), `TrainerDescriptor` (`id, family, backend,
  modality, supports_lora, supports_lokr, supports_control, supports_full_finetune,
  max_reference_images, techniques`, `request.rs:196-234`), `TrainingConfig` (834 lines),
  `buckets.rs`, `schedule.rs` (LR), `aspect_ratio.rs`, `body.rs`.
- `TrainingTechniques` flags: weight noise, depth anchoring, gradient noise, resolution buckets,
  subject-mask loss, identity loss, face landmark, body proportion/shape, normal loss, VAE anchor,
  latent LPIPS, training adapter, builtin x0 decoder (`train/techniques.rs:13-56`).
- `gen-core-train` crate: shared validation floors and helpers (`aspect_ratio, aux_schedule, body,
  face_loss, resume, subject_mask, validate`) — sc-27316.
- Registry: `TrainerRegistration{descriptor, load(&LoadSpec)}` (`registration.rs:529-532`).

### 5.2 Backend harnesses [H]

- **MLX** `mlx-gen/src/train/` (lora 59 KB, optim 40 KB, trainer_loop 35 KB, perceptual 53 KB,
  taehv 49 KB, tae 46 KB, vae_anchor 40 KB, latent_lpips 28 KB, checkpoint 18 KB, lazy, loss,
  vocab, dataset, formula). **Functional autograd**: trainable factors live outside the model in
  `LoraParams` and are re-injected each step into the same `AdaptableLinear`s via
  `set_adapters` — "mirrors the inference reload op-for-op … so the trained adapter round-trips"
  (`mlx-gen/src/train/lora.rs:1-25`). Same module tree serves inference and training.
- **Candle** `candle-gen/src/train/` (22 files: lora 148 KB, flow_match 134 KB, optim 50 KB,
  merge 49 KB, tae/taehv 43 KB each, perceptual 38 KB, vae_anchor 37 KB, projection 36 KB,
  policy 35 KB, profile 33 KB, latent_lpips 30 KB, norm 24 KB, gradient_checkpoint 21 KB, …).
  `LoraLinear` wraps a frozen base with Var-backed f32 factors (`train/lora.rs:1-22`);
  `gradient_checkpoint` is a manual segmented VJP (no candle primitive) (`train/mod.rs:22-26`).
  Per-family trainers "intentionally NOT consolidated" for `compute_loss_grads` (sc-7787,
  `train/merge.rs:49-50`).
- **Candle forks the model for training**: fused `softmax_last_dim`/`rms_norm` `CustomOp`s have
  no backward, so trainers vendor composable copies of the blocks (`candle-gen-krea/src/train_dit.rs:1-14`,
  "for the same reason the Z-Image trainer vendors its DiT"); files with this rationale also in
  `candle-gen-iris/src/train/{model,teacher}.rs`.

### 5.3 Per-family trainer code (bytes of `training*`/`train*` under family `src/`) [H]

| Family | MLX | Candle |
|---|---:|---:|
| Iris (full training, REPA, DDP/FSDP, EMA, Muon) | 308,706 | 339,682 |
| LTX (2.3 + 2.5) | 295,413 | 249,336 |
| Qwen-Image 2.1 | 265,398 | 291,146 |
| Z-Image | 175,124 | 105,013 |
| MOSS-Image | 144,241 | 157,565 |
| Anima | 145,722 | 57,261 |
| SDXL | 138,107 | 104,974 |
| Wan | 136,003 | 126,595 |
| Mage | 130,093 | 93,294 |
| Krea (incl. control training) | 91,048 | 130,035 (+ `control_train.rs` 36 KB) |
| Lens | 128,414 | 78,052 |
| SD3 | 125,239 | 61,861 |
| FLUX3 | 72,736 | 95,543 |
| Kolors | 55,759 | 61,345 |
| Face | 66,681 | 57,381 |

≈ 2.48 MB MLX + 2.06 MB Candle of family trainer code, on top of ~0.6 MB + ~1.1 MB of shared
harness [H for sizes; sum is arithmetic]. Iris documents its own curriculum (positional draws,
logit-normal timesteps, REPA/iREPA vs DINOv2, DDP/FSDP, EMA from pre-step weights,
atomic checkpoints with optimizer/scheduler/data-cursor, resharding) (`mlx-gen-iris/src/train/mod.rs:1-26`).

### 5.4 What the shared architecture must expose to trainers

1. **One module tree with a differentiable mode** — forward with grads through the same
   blocks; backends that have fused no-backward kernels need an op-set switch
   (`fused | composable`) at the primitive level rather than a vendored DiT per family.
2. **Adapter injection** = the inference adapter layer with `trainable: true` (same target ids, same
   reconstruct math ⇒ round-trip guaranteed by construction; PEFT save keys derived from the same
   canonical target names — today the SDXL prefix differs, `mlx-gen/src/train/lora.rs:20-22`).
3. **Frozen-stage caching**: text embeddings / VAE latents computed once by the inference stages
   (text encoder, VAE encode) with residency control (trainers currently re-implement loads).
4. **Gradient-checkpoint boundaries = block-window boundaries** from the weight plan's block layout.
5. **x0 decode hooks** for aux losses (perceptual, latent LPIPS, VAE anchor, face/body), selected
   by `TrainingTechniques`; `builtin_x0_decoder` already exists as a flag.
6. **Data pipeline**: buckets/aspect ratio/subject masks (already in gen-core/gen-core-train).
7. **Preview samples** through the real `Generator` path (trainers render previews; report
   warnings via `train_with_report`).
8. **Checkpoint/resume identity** (`gen-core-train/src/resume.rs`) and output format = adapter
   file the inference loader classifies.
9. **Distributed** (Iris `dist.rs`) stays family-specific [M].

---

## 6. Audio and LLM crates

### 6.1 Audio [H]

- 16 provider crates + `candle-audio` commons (`dsp` 71 KB, `neural_codec` 31 KB, `harness`,
  `qwen3` 19 KB, `vocos`, `sampling`, `ops`, `mel`, `prepare`, `wav`, `weights`). Candle-native on
  every platform; implements the **existing** `gen_core::Generator` with `Modality::Audio`,
  registered through `candle-audio-catalog` (`inf:crates/audio/candle-audio/src/lib.rs:1-16`).
- Overlap with media concepts, measured by `git grep -l` over `crates/audio`: `Quant::` 20 files,
  `OffloadPolicy` 3, `AdapterSpec` 3, `Residency<` **0**, `register_memory_strategy` **0**,
  `candle_gen::quant` **0**, `QLinear` **0**. Audio has its own quant handling, own mmap seam,
  own adapter system (SA3), no shared residency or memory admission.
- Duplicates LLM pieces: `candle-audio/src/qwen3.rs` ("the Qwen3 decoder block … MOSS-TTS
  backbones share", sc-27270) and `sampling.rs` (token sampler) while `candle-llm` has Qwen3.5/
  Llama/Gemma4 and its own samplers (`inf:crates/llm/candle-llm/src/models/*`).
- testkit covers audio contracts (`gen-core-testkit/src/{audio_generator,audio_transform,
  audio_embedder,voice_embedder,transcriber}.rs`).

### 6.2 LLM [H]

- `core-llm` (contracts, ~1 MB of source: request, registry 48 KB, resource 35 KB, speculative 69 KB,
  kv_compression 64 KB, template 62 KB, …) with its **own** `CancelFlag`
  (`inf:crates/contracts/core-llm/src/cancel.rs:15`) alongside gen-core's
  (`inf:crates/contracts/gen-core/src/runtime/sinks.rs:8`); gen-core depends on core-llm and
  re-exports it (`mlx-gen/Cargo.toml:95-106` comments).
- `mlx-llm` own `QuantizedLinear` ("greenfield for the engine — the mlx-gen LLM stacks … reject
  quantization at load", `mlx-llm/src/primitives/quant.rs:1-7`), own `Weights`, GGUF convert,
  per-request memory admission (`load_memory.rs` 98 KB; memory note: pin ≥ b120a876f).
- First media→LLM edge: `mlx-gen-ltx → mlx-llm` for LTX-2.5's Gemma 4 encoder
  (`mlx-gen-ltx/Cargo.toml:14-25`, "the documented direction of the inversion").
- Text-encoder re-implementations in media/audio (struct names): `candle-gen-flux/src/packed_te.rs`
  (CLIP + T5 layers), `candle-gen-kolors/src/chatglm3.rs` (`ChatGlmModel`),
  `candle-gen-ltx/src/gemma.rs` (`GemmaEncoder`), `candle-gen-minimax-h3/src/text_encoder/encoder.rs`
  (`Qwen3DecoderLayer`), `candle-gen-pid/src/gemma2.rs`, `candle-gen-bernini/src/qwen2_5_vl.rs`,
  `candle-gen-qwen-image{,-2-1}/src/text_encoder.rs`, `candle-audio-stable-audio-3/src/t5gemma.rs`,
  `mlx-gen-ltx/src/gemma.rs`, `mlx-gen-kolors/src/chatglm3.rs`, `mlx-gen-lens/src/text_encoder/gpt_oss.rs`,
  `mlx-gen-qwen-image/src/text_encoder/*` (14 files), `mlx-gen-z-image/src/text_encoder/*`,
  `mlx-gen-krea/src/text_encoder/*`, `mlx-gen-mage/src/text_encoder/*`, … (iOS reuses SANA's
  Gemma-2 from `mlx-gen-pid` and Z-Image's Qwen3 block for Anima — reuse exists but is crate-to-crate).

### 6.3 Should the shared layers cover them?

| Concept | Media | Audio | LLM | Share? |
|---|---|---|---|---|
| Cancel / progress / load activity | gen-core | gen-core | core-llm (own) | **Yes** — one tensor-free crate |
| `WeightsSource` + pins/seals + inventory | gen-core | via gen-core `LoadSpec` | own | **Yes** |
| Weight plan (keys/layout/dtype/quant) | per family | per crate | per arch + GGUF | **Yes**, same plan language |
| Linear spine (dense/affine/GGML/fp8/nvfp4 + adapters) | 2–3 spines | own | own `QuantizedLinear` | **Yes**, one per backend |
| Transformer blocks (attention, RoPE, RMSNorm, SwiGLU, GQA) | duplicated | duplicated (qwen3) | canonical | **Yes**: LLM crates own decoder blocks; media/audio consume as "encoder mode" |
| Staged residency / memory admission | `Residency`, memory_strategy | none | `load_memory`/`resource` | Vocabulary yes; drivers differ (two-phase image vs AR KV) [M] |
| Sampler (denoise) / latent / VAE | media | audio diffusion (SA3, AceStep, MMAudio) | n/a | media+audio diffusion share; LLM no |
| Token sampler | — | `candle-audio/src/sampling.rs` | core-llm/llm | **Yes** (LLM owns) |
| Adapters | yes | SA3 8-type | LLM LoRA? [not checked] | adapter layer yes |

---

## 7. iOS constraints

[H] from `inf:crates/media/mlx-gen/mlx-gen-ios-catalog/{src/lib.rs:1-49, Cargo.toml}`,
`inf:crates/bundles/runtime-ios/{src/lib.rs:1-40, Cargo.toml}`, `mlx-gen/src/ios_quantum.rs:1-7`.

1. **Crate-granular composition**: the iOS catalog depends on exactly sana, anima, iris, sdxl,
   z-image (+ pid transitively), **not** `mlx-gen-catalog` ("would compile a graph the platform
   cannot run"). Anima pulls `mlx-gen-qwen-image` and `mlx-gen-wan` as compiled code only (Qwen VAE
   reuse). ⇒ shared code must not force video/audio/CUDA code into the image graph; cross-family
   reuse should live in shared crates, not family→family edges.
2. **Feature gating**: every included family is `default-features = false` to drop `training`
   (trainer, perceptual, depth/face/body perceptors). The `training` feature must remain a clean
   compile-time cut in any new layout (trainers, aux-loss models, dataset decode).
3. **No audio feature** on `runtime-ios` by design; LLM via `mlx-llm` whole; LLM loads are
   unadmitted (no footprint estimate) — `runtime-ios/src/lib.rs:30-33`.
4. **Admission**: one evidence-table seam (`ios_admission.rs` 185 KB, `evidence.rs` 85 KB), typed
   `RefusalKind`, request-bound wrappers that recheck live capacity and **source pins** before every
   provider call; `device-capture` feature for evidence collection only.
5. **Per-quantum OS-headroom checks** before each graph construction (`ios_quantum.rs:1-7`), adapter
   loads included (`ADAPTER_LOAD_QUANTUM`, `adapters/loader.rs:198-208`). Per-family pricing modules
   (`mlx-gen-sana/src/ios_pricing/` 15 files, `mlx-gen-z-image/src/ios_pricing/` 9, sdxl 3, iris 3)
   plus `mlx-gen/src/ios_{pricing,quantum*,anchor_law}.rs` (~250 KB). A shared weight plan that
   knows per-component/per-block bytes could price quanta generically [M].
6. **Binary size**: no measured numbers found in source [not verified]. The structural constraint is
   that generic code must not be monomorphized per family unnecessarily and that codec
   implementations for CUDA-only formats stay out of the MLX/iOS graph (MLX catalog already rejects
   NVFP4 at registry level).
7. **Metallib packaging** and sandbox paths (`runtime-ios/src/lib.rs:10-16`) are bundle concerns,
   unaffected by the shared layers.

---

## 8. Testing infrastructure and migration protection

### 8.1 What exists [H]

- **Weights-free conformance**: `gen-core-testkit` drives any boxed provider through the public
  contract (cancellation, progress monotonicity, seed determinism, capability honesty) for
  generators, trainers, captioners, audio contracts; zero tensor deps
  (`gen-core-testkit/src/lib.rs:1-19`); memory-strategy oracle over registrations
  (`memory_strategy/{contract,registration,registry}.rs`); `weights_free_trainer_conformance.rs`
  per family (anima, kolors, krea, lens, ltx, mage, sd3, sdxl, wan, z-image).
- **Parity producers**: 236 Python scripts in `inf:crates/media/mlx-gen/tools/` (by family: ltx 23,
  qwen 18, flux2 17, z 15, sensenova 14, sdxl 12, iris 12, flux3 12, bernini 11, minimax 10, lens 10, …),
  plus adapter-parity tools (`sc3642_lokr_reference.py`, `sc3643_loha_reference.py`,
  `verify_adapter_parity_artifacts.py`); producers are hash-pinned (memory note).
- **Fixtures**: 475 `fixtures/*.{safetensors,json,npz,npy,bin}` blobs; heaviest dirs
  `mlx-gen-iris/tests/fixtures` 46, `mlx-gen-flux3` 35, `mlx-gen-ltx` 33, `mlx-gen-wan` 30,
  `mlx-gen-qwen-image-2-1` 28; cross-backend portable adapter fixtures
  (`gen-core/tests/fixtures/moss_image/portable_adapters/{mlx,candle}-*.safetensors`).
- **Gated model execution**: `#[ignore = "model inference; explicit model-proof workflow …"]` 1,364,
  `"model inference; explicit model workflow only"` 1,322, bare `#[ignore` 949, synthetic block-stack
  ignores ~105 (≈ 6,000 `#[ignore` total). Workflows: `model-inference-proofs.yml`,
  `model-test-suites.yml`, `real-weights.yml`, family campaigns (`ltx25-quant-*`, `veda-ab-cuda`,
  `yue2-*`); `scripts/ci/real-weights/` (332 files). Feature `model-inference-proof` threads through
  gen-core/core-llm/families (`gen-core/Cargo.toml:15-17`).
- **Source-scanning guards**: preview scanner, additive-capabilities scanner, mmap-site check
  (`scripts/check-workspace.py`, per `candle-gen/src/loader.rs:11-12`).

### 8.2 How to protect a migration (per family, cheapest first)

1. **Plan receipts (weights-free, every PR)**: for each family and each source dialect, compile the
   new `WeightPlan` against a **header-only fixture** (safetensors headers + config.json, no
   payloads) and snapshot `(logical_key, dtype, shape, codec, residency, layout ops, block id)`.
   Headers for real checkpoints can be captured once (`docs/reference/sc-18756-headers` already
   holds header captures [M]).
2. **Load equivalence (tiny weights, every PR where cheap)**: on existing tiny fixtures, old loader
   vs new plan must produce byte-identical `(key → hash(bytes), dtype, shape)` maps after decode.
   This is not model execution (no forward), so it fits the "no real model execution in ordinary CI"
   rule [M — check the rule's scope with Michael].
3. **Forward goldens (explicit workflow)**: existing tiny-fixture parity tests per family
   (`tests/fixtures` + `dump_*.py`) run unchanged against the new loader.
4. **Adapter parity**: portable adapter fixtures + `AdapterApplyReport` snapshot (applied/skipped
   per file) before/after; for candle families moving fold→residual, record the expected ULP-level
   divergence and re-baseline explicitly.
5. **Real-weight parity**: once, at the end of the migration epic (memory: campaigns once at epic
   end), per family × tier, comparing against pre-migration outputs from the same pin.

---

## Design implications

1. **Make the weight plan the single entry for every load — including SceneWorks' own tiers.**
   Register the MLX affine triple (`affine-q{4,8}-g{32,64}-v1`), GGML Q4_1/Q8_0 and GGUF k-quants as
   codecs in the existing `CheckpointCodecRegistry`; extend `LogicalKeyMapping` into a declarative
   `WeightPlan` (components, source dialects, key rules, layout rules, dtype policy, quant
   predicate, block layout, expectation). Per-family code shrinks to data + rare hooks
   (`.pth` readers, custom sanitizers). Keep fail-closed, header-only compilation.
2. **Split the lifecycle into hookable stages**: `stage → pin/seal → plan → decode → layout →
   quantize → residency(eager|deferred|windowed) → bind → adapters → receipt`. Each stage is a
   hook point for cross-cutting features (progress, pins, iOS quanta, tier integrity, memory
   pricing, telemetry). Enforce the MLX ordering rule (pending loads first, bounded batches, coherence
   verify) **inside** the decode stage so no family can re-introduce the watchdog bug.
3. **One Linear spine per backend** with `base ∈ {Dense, Affine, Ggml, GgufK, Nvfp4, Fp8Row,
   Int8Row, W4A8, ConvRot}`, `dtype_policy`, `adapters: Vec<Residual>`, `scale_ctx` (phase/pass/
   expert), and a `trainable` slot. Candle merges `QLinear` + `AdaptLinear` + `LoraLinear`; MLX's
   `AdaptableLinear` is already close. Removes most of the 68 Linear types and the per-family
   `nvfp4_dit`/`quant.rs` seams.
4. **Generate adapter host routing from canonical module names** (the bind stage knows every
   logical key → module) instead of 102 hand-written `AdaptableHost` trees; alias tables (kohya,
   BFL, native) become plan data. Adapter install policy `Residual | DenseDeltaResidual | Fold`
   chosen from (format, base, parity mode). Default Candle to residual; budget one golden
   re-baseline per fold family.
5. **Request-time adapter scale context** replaces LTX-local `Linear`, Krea's job-local second DiT,
   and Wan MoE filtering. The shared sampler driver sets `ctx` per step/phase.
6. **Routes as typed compositions; capabilities derived**. A route record lists its stages (text
   encoders, conditioning, denoiser + sampler driver, decoder, adapter surface, residency driver,
   weight plan) and a small `RouteProductLimits`. Derive `Capabilities`, memory fixtures,
   preview support, imports and adapter facts from it; register generator + memory strategy +
   fixture + behavior in one call. Retire the 4.5k-line preview scanner and the additivity scanner
   once descriptors stop being hand literals. Bring name-driven edit/control providers into the
   registry as routes. Move `MODEL_TABLE` defaults into the route record and dump them.
7. **Remove family-specific methods from the `Generator` trait** (moss/veda reports) — route them
   through a typed report/extension map.
8. **Training = inference modules in differentiable mode**. Provide a backend op-set switch
   (fused vs composable) so candle stops vendoring DiTs; trainers get adapter injection, frozen-stage
   caching, x0 decode hooks and checkpoint boundaries from the shared layers. Keep `training` a
   compile-time feature.
9. **Unify tensor-free contracts with core-llm** (cancel, progress, load activity, weights source,
   pins) and make LLM crates the owner of transformer decoder blocks consumed by media/audio text
   encoders; extend the weight plan and Linear spine underneath LLM crates too.
10. **Audio joins the shared weight plan, Linear spine and adapter layer**; its non-commutative
    adapters (DoRA/BoRA/-XS) are the `Fold`-ordered case. Residency/memory admission for audio can
    follow later but should use the same vocabulary.
11. **iOS**: keep catalogs crate-granular and shared layers feature-partitioned (no CUDA codecs,
    no training, no video in the iOS graph); feed iOS quantum pricing from plan bytes.
12. **Migration safety net**: plan-receipt snapshots (weights-free) + load-equivalence on tiny
    fixtures per PR; forward goldens on explicit workflows; one real-weight campaign at the end.

## Uncertain / not verified

- **Counts** come from regex greps and may include/exclude edge cases: 177 `Capabilities {`
  literals (prior audit: 160), 137 remap-like fns (prior audit: 101 with a different pattern),
  68 Linear-named types, ~102 `AdaptableHost` impls (prior audit: 113 — likely includes nested
  `impl<…>` forms my pattern missed), `register_*` counts include tests.
- **Which candle families fold vs residual** was read from 6 headers (z-image, wan, sdxl, qwen21,
  minimax-h3, ltx) plus `train/merge.rs`'s list; SD3/Lens/Iris/Anima/Scail2/Ideogram/MOSS-image
  strategies were not individually confirmed. `AdaptLinear` appears in 18 candle crate dirs, so some
  "fold" families may already be hybrid.
- **Residual-vs-fold perf numbers** in §3.6 are arithmetic estimates, not measurements.
- **Binary size impact on iOS** — no size budget or measurement found in source.
- **LLM-side adapter (LoRA) support** in mlx-llm/candle-llm was not checked.
- Whether audio providers' 20 `Quant::` files implement real quantization or only refuse tiers was
  not read.
- `docs/reference/sc-18756-headers` contents (assumed header captures) not opened.
- SceneWorks `runtime/capabilities.*.json` is empty for `engines` but rich in other keys; I did not
  trace `trainerMappings` / `workerCapabilities` fully.
- Whether "load equivalence on tiny fixtures" counts as forbidden "real model execution in ordinary
  CI" is a policy judgment for Michael; it executes no forward pass.
- The inference working tree is stale; all inference facts are `origin/main@a43724b82` only.
